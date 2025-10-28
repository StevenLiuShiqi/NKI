import argparse
import copy
import gc
import math
import re
import time
from typing import List, Tuple, Union, Optional, Callable, Any

import torch
import torch.nn.functional as F
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.experts import Experts
from neuronx_distributed.modules.moe.model_utils import DEFAULT_BLOCK_SIZE, DEFAULT_LNC_SIZE
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig, BlockwiseMatmulConfig
from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedLinear,
    ExpertFusedRowParallelLinear,
    ExpertFusedColumnParallelLinear,
    ExpertFusedLinearWithAsyncCommunication,
)
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.rms_norm import RMSNorm
from neuronx_distributed.parallel_layers import layers, mappings, parallel_state, utils
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
)
from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter
from neuronxcc.nki._private_kernels.blockwise_mm import BlockShardStrategy
from torch import Tensor
from torch import nn
from torch.distributed import ProcessGroup
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

_CONSTANT_INIT_VALUE = 0.5


def inspect(self, tensor):
    tensor_list = tensor.tolist()
    with open(str(time.time()) + '.json', 'w') as f:
        json.dump({'tensor': tensor_list, 'dtype': str(tensor.device), 'class': self.__class__.__name__}, f)


def _fill_module_parameters(module: torch.nn.Module, value: float = _CONSTANT_INIT_VALUE) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def neuron_gptoss_swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    # gate, up = torch.chunk(x, chunks=2, dim=-1)
    # return F.silu(gate) * (up + 1)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class NeuronGPTOSSExpertFusedColumnParallelLinear(layers.ColumnParallelLinear, ExpertFusedLinear):
    autograd_func_class = ExpertFusedLinearWithAsyncCommunication

    def __init__(
            self,
            num_experts: int,
            input_size: int,
            output_size: int,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            stride: int = 1,
            init_method: Optional[Callable[..., Any]] = None,
            keep_master_weight: bool = False,
            tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.num_experts = num_experts
        self._n_local_experts = utils.divide(num_experts, parallel_state.get_expert_model_parallel_size())

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=True,
            gather_output=False,
            dtype=dtype,
            device=device,
            stride=stride,
            init_method=init_method,
            sequence_parallel_enabled=False,
            keep_master_weight=keep_master_weight,
            skip_bias_add=False,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )
        self._mark_expert_parallel_weights()

    def set_weight_and_bias_config(self):
        # Define 3D weight tensor, one linear layer per expert
        self.weight_shape = (
            self._n_local_experts,
            self.input_size,
            self.output_size_per_partition,
        )
        # Column parallel partitioning for each expert
        self.weight_partition_dim = 2
        self.bias_shape = (
            self._n_local_experts,
            self.output_size_per_partition,
        )
        # Bias follows the column-partitioned output dimension
        self.bias_partition_dim = 1

    def _init_weight(self, weight):
        # Initialize the linear layer of each expert separately
        assert len(weight.shape) == 3
        for e in range(weight.shape[0]):
            if self.arg_init_method is None:
                torch.nn.init.kaiming_uniform_(weight[e], a=math.sqrt(5))
            else:
                self.arg_init_method(weight[e])

    def _init_bias(self) -> None:
        bound = 1 / math.sqrt(self.input_size) if self.input_size > 0 else 0
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_: torch.Tensor, expert_indices: Optional[torch.Tensor] = None, *_: Any) -> torch.Tensor
        inspect(self, input_)
        """If expert_indices is provided, then the computations are performed only on the specified experts.
        Otherwise, the input is passed through all experts in the layer."""

        if self.async_tensor_model_parallel_allreduce or self.sequence_parallel_enabled:
            input_parallel = input_
        else:
            input_parallel = mappings.copy_to_tensor_model_parallel_region(
                input_,
                process_group=self.tensor_parallel_group,
            )

        # Matrix multiply.
        weight = self.weight[expert_indices, :, :] if expert_indices is not None else self.weight
        output = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=None,
            async_grad_allreduce=self.async_tensor_model_parallel_allreduce,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
        )
        if self.bias is not None:
            bias = self.bias[expert_indices, :] if expert_indices is not None else self.bias
            # Reshape bias to broadcast across expert capacity dimensions
            while bias.dim() < output.dim():
                bias = bias.unsqueeze(1)
            output = output + bias
        return output


class NeuronGPTOSSExpertFusedRowParallelLinear(layers.RowParallelLinear, ExpertFusedLinear):
    """Row-parallel expert linear with per-expert bias support."""

    autograd_func_class = ExpertFusedLinearWithAsyncCommunication

    def __init__(
            self,
            num_experts: int,
            input_size: int,
            output_size: int,
            reduce_output: bool = True,
            dtype: torch.dtype = torch.float32,
            device: Optional[torch.device] = None,
            stride: int = 1,
            init_method: Optional[Callable[..., Any]] = None,
            keep_master_weight: bool = False,
            tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ) -> None:
        self.num_experts = num_experts
        self._n_local_experts = utils.divide(num_experts, parallel_state.get_expert_model_parallel_size())

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=True,
            input_is_parallel=True,
            dtype=dtype,
            device=device,
            stride=stride,
            init_method=init_method,
            sequence_parallel_enabled=False,
            keep_master_weight=keep_master_weight,
            skip_bias_add=False,
            reduce_output=reduce_output,
            tensor_model_parallel_group=tensor_model_parallel_group,
        )
        self._mark_expert_parallel_weights()

    def set_weight_and_bias_config(self):
        # Define 3D weight tensor, one linear layer per expert
        self.weight_shape = (
            self._n_local_experts,
            self.input_size_per_partition,
            self.output_size,
        )
        # Row parallel partitioning for each expert
        self.weight_partition_dim = 1
        self.bias_shape = (
            self._n_local_experts,
            self.output_size,
        )

    def _init_weight(self, weight):
        # Initialize the linear layer of each expert separately
        assert len(weight.shape) == 3
        for e in range(weight.shape[0]):
            if self.arg_init_method is None:
                torch.nn.init.kaiming_uniform_(weight[e], a=math.sqrt(5))
            else:
                self.arg_init_method(weight[e])

    def forward(self, input_: torch.Tensor, expert_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """If expert_indices is provided, only compute the specified experts."""
        inspect(self, input_)

        weight = self.weight[expert_indices, :, :] if expert_indices is not None else self.weight
        output_parallel = self._forward_impl(
            input=input_,
            weight=weight,
            bias=None,
            async_grad_allreduce=False,
            sequence_parallel_enabled=False,
            autograd_func_class=self.autograd_func_class,
            process_group=self.tensor_parallel_group,
            reduce_dtype=self.reduce_dtype,
        )

        output = self._rpl_maybe_reduce_output(output_parallel)

        if self.bias is not None:
            bias = self.bias[expert_indices, :] if expert_indices is not None else self.bias
            while bias.dim() < output.dim():
                bias = bias.unsqueeze(1)
            output = output + bias

        return output


class NeuronGPTOSSExperts(Experts):
    def __init__(
            self,
            num_experts: int,
            hidden_size: int,
            intermediate_size: int,
            glu: bool,
            activation_fn: Callable[[Tensor], Tensor],
            dtype: torch.dtype,
            device: torch.device,
            input_layer_init_method=None,
            output_layer_init_method=None,
            tensor_model_parallel_group: Optional[ProcessGroup] = None,
    ) -> None:
        super().__init__(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            glu=glu,
            activation_fn=activation_fn,
            dtype=dtype,
            device=device,
            input_layer_init_method=input_layer_init_method,
            output_layer_init_method=output_layer_init_method,
            tensor_model_parallel_group=tensor_model_parallel_group
        )

        self._glu = glu
        self._activation_fn = activation_fn
        self.num_experts = num_experts
        # todo: we can also generalize expert-parallel group
        self.tensor_parallel_group = tensor_model_parallel_group if \
            tensor_model_parallel_group is not None else get_tensor_model_parallel_group()

        if self._glu:
            self.gate_up_proj = NeuronGPTOSSExpertFusedColumnParallelLinear(
                # we pass the global number of experts. the linear layer will itself
                # decide to initialize a subset of them if EP is applied.
                num_experts=num_experts,
                input_size=hidden_size,
                # we fuse up and gate projections to a single matmul. Later on in code
                # we'll split the resulting output to yield up and gate matrices.
                output_size=intermediate_size * 2,
                dtype=dtype,
                device=device,
                stride=2,
                init_method=input_layer_init_method,
                tensor_model_parallel_group=self.tensor_parallel_group,
            )
        else:
            self.up_proj = NeuronGPTOSSExpertFusedColumnParallelLinear(
                # we pass the global number of experts. the linear layer will itself
                # decide to initialize a subset of them if EP is applied.
                num_experts=num_experts,
                input_size=hidden_size,
                output_size=intermediate_size,
                dtype=dtype,
                device=device,
                init_method=input_layer_init_method,
                tensor_model_parallel_group=self.tensor_parallel_group,
            )

        self.down_proj = NeuronGPTOSSExpertFusedRowParallelLinear(
            # we pass the global number of experts. the linear layer will itself
            # decide to initialize a subset of them if EP is applied.
            num_experts=num_experts,
            input_size=intermediate_size,
            output_size=hidden_size,
            reduce_output=get_tensor_model_parallel_size() > 1,
            dtype=dtype,
            device=device,
            init_method=output_layer_init_method,
            tensor_model_parallel_group=self.tensor_parallel_group,
        )

    def _activation(self, x: Tensor) -> Tensor:
        return neuron_gptoss_swiglu(x)


class NeuronGPTOSSExpertMLPsV2(ExpertMLPsV2):
    def __init__(
            self,
            routed_experts_mlp_config: RoutedExpertsMLPOpsConfig,
            blockwise_matmul_config: BlockwiseMatmulConfig = BlockwiseMatmulConfig.default(),
            return_bias: bool = False,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu"),
            tensor_model_parallel_group: Optional[ProcessGroup] = None,
            expert_model_parallel_group: Optional[ProcessGroup] = None,
            # spmd_rank will be removed once we support ReplicaID (P87857655)
    ):
        super().__init__(
            routed_experts_mlp_config=routed_experts_mlp_config,
            blockwise_matmul_config=blockwise_matmul_config,
            return_bias=return_bias,
            dtype=dtype,
            device=device,
            tensor_model_parallel_group=tensor_model_parallel_group,
            expert_model_parallel_group=expert_model_parallel_group
        )

        self.mlp_op = NeuronGPTOSSExperts(
            num_experts=routed_experts_mlp_config.num_experts,
            hidden_size=routed_experts_mlp_config.hidden_size,
            intermediate_size=routed_experts_mlp_config.intermediate_size,
            glu=routed_experts_mlp_config.glu_mlp,
            activation_fn=F.silu,
            dtype=dtype,
            device=device,
            input_layer_init_method=routed_experts_mlp_config.input_layer_init_method,
            output_layer_init_method=routed_experts_mlp_config.output_layer_init_method,
            tensor_model_parallel_group=self.tensor_parallel_group,
        )


class NeuronGPTOSSExpertMLPs(NeuronGPTOSSExpertMLPsV2):
    def __init__(
            self,
            num_experts: int,
            top_k: int,
            hidden_size: int,
            intermediate_size: int,
            hidden_act: str,
            glu_mlp: bool,
            capacity_factor: Union[None, float],
            block_size: Union[None, int] = None,
            normalize_top_k_affinities: bool = False,
            return_bias: bool = False,
            init_method: Optional[Callable[..., Any]] = torch.nn.init.kaiming_uniform_,
            output_layer_init_method: Optional[Callable[..., Any]] = torch.nn.init.kaiming_uniform_,
            dtype: torch.dtype = torch.float32,
            device: torch.device = torch.device("cpu"),
            tensor_model_parallel_group: Optional[ProcessGroup] = None,
            expert_model_parallel_group: Optional[ProcessGroup] = None,
            enable_spmd_rank: bool = False,  # spmd_rank will be removed once we support ReplicaID (P87857655)
            blockwise_nki_autograd_cls=None,
            use_torch_block_wise: bool = False,
            logical_nc_config=DEFAULT_LNC_SIZE,  # uses lnc1 blockwise kernel by default
            parallelize_token_to_block_mapping: bool = True,
            early_expert_affinity_modulation: bool = False,
            optimized_block_to_token_mapping: bool = True,
            use_block_parallel: bool = False,
            always_augment_inputs_for_blockwise_matmul: bool = False,
            block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.HI_LO,
            skip_dma_token: bool = False,
            skip_dma_weight: bool = False
    ):
        routed_experts_mlp_config = RoutedExpertsMLPOpsConfig(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            glu_mlp=glu_mlp,
            normalize_top_k_affinities=normalize_top_k_affinities,
            early_expert_affinity_modulation=early_expert_affinity_modulation,
            input_layer_init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            capacity_factor=capacity_factor,
            top_k=top_k,
            hidden_act=hidden_act,
            enable_spmd_rank=enable_spmd_rank,
        )

        blockwise_matmul_config = BlockwiseMatmulConfig.from_kwargs(
            block_size=block_size if block_size else DEFAULT_BLOCK_SIZE,
            logical_nc_config=logical_nc_config,
            use_torch_block_wise=use_torch_block_wise,
            blockwise_nki_autograd_cls=blockwise_nki_autograd_cls,
            parallelize_token_to_block_mapping=parallelize_token_to_block_mapping,
            early_expert_affinity_modulation=early_expert_affinity_modulation,
            optimized_block_to_token_mapping=optimized_block_to_token_mapping,
            use_block_parallel=use_block_parallel,
            always_augment_inputs_for_blockwise_matmul=always_augment_inputs_for_blockwise_matmul,
            block_sharding_strategy=block_sharding_strategy,
            skip_dma_token=skip_dma_token,
            skip_dma_weight=skip_dma_weight
        )

        super().__init__(
            routed_experts_mlp_config=routed_experts_mlp_config,
            blockwise_matmul_config=blockwise_matmul_config,
            dtype=dtype,
            device=device,
            return_bias=return_bias,
            tensor_model_parallel_group=tensor_model_parallel_group,
            expert_model_parallel_group=expert_model_parallel_group,
        )


# Working MoE Block
class NeuronMLPBlock(torch.nn.Module):
    def __init__(
            self,
            config,
            device: torch.device | None = None,
            weight_init_value: float | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        # self.swiglu_limit = config.swiglu_limit
        self.world_size = 1

        # RMSNorm (kept as is)
        self.norm = RMSNorm(config.hidden_size, device=device)

        # Create Router (replaces self.gate)
        self.router = RouterTopK(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            act_fn="softmax",  # matches your softmax
            dtype=torch.bfloat16,
            device=device,
            # bias=False,
            sequence_parallel_enabled=False,  # adjust based on your setup
        )

        # Create ExpertMLPs (replaces manual mlp1/mlp2 weights)
        self.expert_mlps = NeuronGPTOSSExpertMLPs(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act="silu",  # SwiGLU uses SiLU internally
            glu_mlp=True,  # SwiGLU is a GLU variant
            # glu_type=GLUType.SWIGLU,  # specify SwiGLU
            capacity_factor=None,  # full capacity (no dropping)
            normalize_top_k_affinities=True,  # your softmax normalizes
            # bias=False,  # as you mentioned
            dtype=torch.bfloat16,
            device=device,
            tensor_model_parallel_group=None,  # set if using TP
            # sequence_parallel_enabled=False,
        )

        # Create complete MoE layer
        # self.moe = MoE(
        #     router=self.router,
        #     expert_mlps=self.expert_mlps,
        #     # rmsnorm=self.norm,  # can pass norm to MoE
        #     sequence_parallel_enabled=False,
        #     return_router_logits=False,  # set True if you need them
        #     return_expert_index=False,   # set True if you need them
        # )
        if weight_init_value is not None:
            _fill_module_parameters(self, weight_init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inspect(self, x)
        # Original: x → norm → gate → topk → expert_mlps → weighted_sum → x + residual
        # With MoE blocks: x → MoE (does all of the above) → x + residual

        # Option 1: Use MoE layer directly (it handles norm, routing, experts)
        # moe_output = self.moe(x)
        # return moe_output

        # Option 2: Manual control (closer to your original)
        # If you want to separate norm from MoE:
        t = self.norm(x)
        router_logits, expert_affinities, expert_index = self.router(t)
        t_flat = t.view(-1, t.shape[-1])  # (B*S, H)
        seq_len = x.shape[1]
        moe_output = self.expert_mlps(
            hidden_states=t,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=seq_len
        )
        moe_output = moe_output.view_as(x)
        return moe_output


def convert_gptoss_to_neuron_state_dict(
        gptoss_sd: dict,
        config,
        *,
        target_dtype=None,  # e.g., torch.bfloat16
        target_device="cpu",  # keep CPU for loader
        gc_every=2,
        inplace_pop=True
):
    """
    Convert GPT-OSS checkpoint to NXD (trace) expected keys.

    Emits keys:
      layers.{L}.self_attn.qkv_proj.{q_proj,k_proj,v_proj}.{weight,bias}
      layers.{L}.self_attn.o_proj.o_proj.{weight,bias}
      layers.{L}.ffn.router.linear_router.{weight,bias}
      layers.{L}.ffn.expert_mlps.mlp_op.{gate_up_proj,down_proj}.{weight,bias}
      layers.{L}.{input_layernorm,post_attention_layernorm}.weight
      embed_tokens.weight, norm.weight, lm_head.weight

    Silently drops non-parameter buffers like self_attn.sinks.
    """

    def take(key, default=None, *, pop=inplace_pop):
        if pop:
            return gptoss_sd.pop(key, default)
        return gptoss_sd.get(key, default)

    def first_present(*keys):
        for k in keys:
            if k in gptoss_sd and gptoss_sd[k] is not None:
                return take(k)
        return None

    def to_target(x):
        if x is None:
            return None
        if target_dtype is not None and x.dtype != target_dtype:
            x = x.to(dtype=target_dtype, copy=False)
        if target_device is not None and (getattr(x, "device", None) is None or x.device.type != target_device):
            x = x.to(device=target_device, copy=False)
        return x

    with torch.no_grad():
        nsd = {}

        # ---- Top-level ----
        etw = first_present("embed_tokens.weight", "model.embed_tokens.weight")
        assert etw is not None, "Missing embed_tokens.weight"
        nsd["embed_tokens.weight"] = to_target(etw)

        nw = first_present("norm.weight", "model.norm.weight")
        assert nw is not None, "Missing norm.weight"
        nsd["norm.weight"] = to_target(nw)

        lmh = first_present("lm_head.weight", "model.lm_head.weight")
        assert lmh is not None, "Missing lm_head.weight"
        nsd["lm_head.weight"] = to_target(lmh)

        # ---- Infer dims from config / tensors ----
        H = int(getattr(config, "hidden_size", nsd["embed_tokens.weight"].shape[1]))
        # Find one gate_up to infer (E, H, 2I)
        gate_key = None
        for k in list(gptoss_sd.keys()):
            if k.endswith(".mlp.experts.gate_up_proj"):
                gate_key = k;
                break
        assert gate_key is not None, "Cannot infer expert dims; gate_up_proj not found."
        E = gptoss_sd[gate_key].shape[0]
        H_chk = gptoss_sd[gate_key].shape[1]
        I = gptoss_sd[gate_key].shape[2] // 2
        assert H == H_chk, f"H mismatch: config={H} vs gate_up={H_chk}"

        # ---- Layer ids present in the GPT-OSS dict ----
        layer_ids = sorted({
            int(m.group(1))
            for k in gptoss_sd.keys()
            if k.startswith("model.layers.") and (m := re.match(r"model.layers\.(\d+)\.", k))
        })

        for idx, L in enumerate(layer_ids):
            # ===== Attention: q/k/v -> qkv_proj.{q_proj,k_proj,v_proj}.{weight,bias}
            for proj in ("q", "k", "v"):

                w = take(f"model.layers.{L}.self_attn.{proj}_proj.weight", None)
                if w is not None:
                    nsd[f"layers.{L}.self_attn.qkv_proj.{proj}_proj.weight"] = to_target(w)
                b = take(f"model.layers.{L}.self_attn.{proj}_proj.bias", None)
                if b is not None:
                    nsd[f"layers.{L}.self_attn.qkv_proj.{proj}_proj.bias"] = to_target(b)

            # ===== Attention: out-proj -> o_proj.o_proj.{weight,bias}
            ow = take(f"model.layers.{L}.self_attn.o_proj.weight", None)
            if ow is not None:
                nsd[f"layers.{L}.self_attn.o_proj.o_proj.weight"] = to_target(ow)
            ob = take(f"model.layers.{L}.self_attn.o_proj.bias", None)
            if ob is not None:
                nsd[f"layers.{L}.self_attn.o_proj.o_proj.bias"] = to_target(ob)

            # Drop sinks buffer if present
            _buf = take(f"model.layers.{L}.self_attn.sinks", None)
            _buf = None

            # ===== Router -> ffn.router.linear_router.{weight,bias}
            rw = take(f"model.layers.{L}.mlp.router.weight", None)
            if rw is not None:
                # Some GPT-OSS exports as [H,E]; NXD expects [E,H]
                if rw.shape == (H, E):
                    rw = rw.t().contiguous()
                elif rw.shape != (E, H):
                    raise ValueError(f"router.weight[{L}] has shape {rw.shape}, expected (E,H) or (H,E)")
                nsd[f"layers.{L}.ffn.router.linear_router.weight"] = to_target(rw)

            rb = take(f"model.layers.{L}.mlp.router.bias", None)
            if rb is not None:
                # Expect [E]
                if rb.dim() != 1 or rb.numel() != E:
                    raise ValueError(f"router.bias[{L}] has shape {tuple(rb.shape)}, expected ({E},)")
                nsd[f"layers.{L}.ffn.router.linear_router.bias"] = to_target(rb)

            # ===== Experts -> ffn.expert_mlps.mlp_op.{gate_up_proj,down_proj}.{weight,bias}
            gu = take(f"model.layers.{L}.mlp.experts.gate_up_proj", None)
            if gu is not None:
                if gu.shape != (E, H, 2 * I):
                    raise ValueError(f"gate_up_proj[{L}] {gu.shape} != (E,H,2I)=({E},{H},{2 * I})")
                nsd[f"layers.{L}.ffn.expert_mlps.mlp_op.gate_up_proj.weight"] = to_target(gu)

            gub = take(f"model.layers.{L}.mlp.experts.gate_up_proj_bias", None)
            if gub is not None:
                if gub.shape != (E, 2 * I):
                    raise ValueError(f"gate_up_proj_bias[{L}] {gub.shape} != (E,2I)=({E},{2 * I})")
                nsd[f"layers.{L}.ffn.expert_mlps.mlp_op.gate_up_proj.bias"] = to_target(gub)

            dp = take(f"model.layers.{L}.mlp.experts.down_proj", None)
            if dp is not None:
                if dp.shape == (E, H, I):
                    dp = dp.transpose(1, 2).contiguous()  # -> [E, I, H]
                if dp.shape != (E, I, H):
                    raise ValueError(f"down_proj[{L}] {dp.shape} != (E,I,H)=({E},{I},{H})")
                nsd[f"layers.{L}.ffn.expert_mlps.mlp_op.down_proj.weight"] = to_target(dp)

            dpb = take(f"model.layers.{L}.mlp.experts.down_proj_bias", None)
            if dpb is not None:
                if dpb.shape != (E, H):
                    raise ValueError(f"down_proj_bias[{L}] {dpb.shape} != (E,H)=({E},{H})")
                nsd[f"layers.{L}.ffn.expert_mlps.mlp_op.down_proj.bias"] = to_target(dpb)

            # ===== Norms
            iln = take(f"model.layers.{L}.input_layernorm.weight", None)
            if iln is not None:
                nsd[f"layers.{L}.input_layernorm.weight"] = to_target(iln)

            paln = take(f"model.layers.{L}.post_attention_layernorm.weight", None)
            if paln is not None:
                nsd[f"layers.{L}.post_attention_layernorm.weight"] = to_target(paln)

            if gc_every and (idx % gc_every == 0):
                gc.collect()

        # Clean up source and return
        gptoss_sd.clear()
        gc.collect()
        return nsd


class NeuronGPTOSSConfig(MoENeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fused_qkv = False


class GPTOSSInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
            "hidden_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "vocab_size", "max_position_embeddings",
            "num_hidden_layers", "rms_norm_eps", "pad_token_id",
            # MoE
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronGPTOSSConfig


class NeuronGPTOSSMLPBlock(torch.nn.Module):
    def __init__(
            self,
            config,
            device: torch.device | None = None,
            weight_init_value: float | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        # self.swiglu_limit = config.swiglu_limit
        self.world_size = 1

        # RMSNorm
        # self.norm = RMSNorm(config.hidden_size, device=device)

        # Create Router (replaces self.gate)
        self.router = RouterTopK(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            act_fn="softmax",  # matches your softmax
            dtype=torch.bfloat16,
            device=device,
            # bias=False,
            sequence_parallel_enabled=False,  # adjust based on your setup
        )

        # Create ExpertMLPs (replaces manual mlp1/mlp2 weights)
        self.expert_mlps = NeuronGPTOSSExpertMLPs(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act="silu",  # SwiGLU uses SiLU internally
            glu_mlp=True,  # SwiGLU is a GLU variant
            # glu_type=GLUType.SWIGLU,  # specify SwiGLU
            capacity_factor=None,  # full capacity (no dropping)
            normalize_top_k_affinities=True,  # your softmax normalizes
            # bias=False,  # as you mentioned
            dtype=torch.bfloat16,
            device=device,
            tensor_model_parallel_group=None,  # set if using TP
            # sequence_parallel_enabled=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original: x → norm → gate → topk → expert_mlps → weighted_sum → x + residual
        # With MoE blocks: x → MoE (does all of the above) → x + residual
        inspect(self, x)

        t = x
        _, expert_affinities, expert_index = self.router(t)
        t_flat = t.view(-1, t.shape[-1])  # (B*S, H)
        seq_len = x.shape[1]
        moe_output = self.expert_mlps(
            hidden_states=t_flat,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=seq_len
        )
        moe_output = moe_output.view_as(x)
        return moe_output


class NeuronGPTOSSAttentionBlock(NeuronAttentionBase):
    def __init__(self, config: InferenceConfig):
        rotary_emb = RotaryEmbedding(
            dim=config.head_dim,  # ← 64 (matches Q/K last dim)
            max_position_embeddings=config.max_position_embeddings,  # 131072 from config
            base=config.rope_theta,  # 150000 from config
        )

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rotary_emb=rotary_emb,
            qkv_bias=True,  # <-- set to True if your checkpoint has q/k/v biases
            o_bias=True,
        )


class NeuronGPTOSSBlock(nn.Module):
    def __init__(self, config: GPTOSSInferenceConfig, block_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.block_idx = block_idx

        self.self_attn = NeuronGPTOSSAttentionBlock(config=config)
        self.ffn = NeuronGPTOSSMLPBlock(config=config)

        # RMS Norm
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Final linear

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            **kwargs,
    ):
        inspect(self, hidden_states)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states).to(dtype=hidden_states.dtype)

        # Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(dtype=hidden_states.dtype)

        # MoE
        hidden_states = self.ffn(hidden_states)[0]  # not sure why indexing
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)

        return outputs


class NeuronGPTOSSModel(NeuronBaseModel):
    """
    The neuron version of the GPT OSS
    """

    def setup_attr_for_model(self, config: GPTOSSInferenceConfig):
        # self.emb_pdrop = config.emb_pdrop

        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    # TODO
    def init_model(self, config: InferenceConfig):
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )  # FIX

        self.layers = nn.ModuleList(
            [NeuronGPTOSSBlock(config, block_idx) for block_idx in range(config.num_hidden_layers)]
        )

        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
            pad=True,
            # tensor_model_parallel_group=get_tp_group(config),
        )


class NeuronGPTOSSForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as GPTOSSForCausalLM
    """
    _STATE_DICT_MODEL_PREFIX = "transformer."
    _model_cls = NeuronGPTOSSModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        print(f"STATE DICT: {state_dict.keys()}")
        return convert_gptoss_to_neuron_state_dict(state_dict, config)

    @classmethod
    def get_config_cls(cls):
        return GPTOSSInferenceConfig

    def get_compiler_args(self):
        return


if __name__ == "__main__":
    model_path = "/home/ubuntu/models/gpt-oss-20b/"
    compiled_model_path = "/home/ubuntu/traced_model/"
    top_k = 1
    top_p = 1.0
    temperature = 1.0
    do_sample = True
    dynamic = False
    # COMMENTED
    # (Pdb) generation_config = GenerationConfig.from_pretrained(model_path)
    # (Pdb) generation_config
    # GenerationConfig {
    #   "bos_token_id": 199998,
    #   "do_sample": true,
    #   "eos_token_id": [
    #     200002,
    #     199999
    #   ],
    #   "pad_token_id": 199999
    # }
    # Potentially dangerous
    # pad_token_id = 2
    tp_degree = 8
    prompts = ["I believe the meaning of life is"]

    neuron_config = NeuronGPTOSSConfig(
        tp_degree=tp_degree
    )

    config = GPTOSSInferenceConfig(
        neuron_config, load_config=load_pretrained_config(model_path)
    )

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        # EDITED
        # Changed with tokenizer working on CPU
        # "openai/gpt-oss-20b"
        model_path,
        padding_side=neuron_config.padding_side
    )
    # COMMENTED
    # (Pdb) p tokenizer.eos_token
    # '<|return|>'
    # (Pdb) p tokenizer.pad_token
    # '<|endoftext|>'
    # Potentially dangerous
    # tokenizer.pad_token = tokenizer.eos_token
    neuron_config.pad_token_id = tokenizer.pad_token_id

    # Configure generation config.
    generation_config = GenerationConfig.from_pretrained(model_path)

    generation_config.update(
        do_sample=do_sample,
        top_k=top_k,
        # pad_token_id=pad_token_id,
        dynamic=dynamic,
        top_p=top_p,
        temperature=temperature
    )

    # Load model
    model = NeuronGPTOSSForCausalLM(model_path, config)

    # Compile and save model.
    # ~5-6 minutes
    compiling_start_time = time.monotonic()
    print("\nCompiling and saving model...")
    model.compile(compiled_model_path, debug=False)
    compiling_end_time = time.monotonic()
    total_compiling_time = compiling_end_time - compiling_start_time
    print(f"Compiling and tracing time: {total_compiling_time} seconds")

    tokenizer.save_pretrained(compiled_model_path)

    import pdb; pdb.set_trace()

    # Load compiled model to Neuron.
    print("\nLoading model to Neuron...")
    model.load(compiled_model_path)
    loading_end_time = time.monotonic()
    model_loading_time = loading_end_time - compiling_end_time
    print(f"Total model loading time: {model_loading_time} seconds")

    print("\nGenerating outputs...")

    _, output_tokens = get_generate_outputs(
        model,
        prompts,
        tokenizer,
        is_hf=False,
        generation_config=generation_config,
        max_length=model.neuron_config.max_length,
    )

    print("Generated outputs:")
    for i, output_token in enumerate(output_tokens):
        print(f"Output {i}: {output_token}")
