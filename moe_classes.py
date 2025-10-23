from typing import Union, Optional, Callable, Any
import math

import torch 
from torch import Tensor
from torch.distributed import ProcessGroup
import torch.nn.functional as F


from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.rms_norm import RMSNorm
from neuronxcc.nki._private_kernels.blockwise_mm import BlockShardStrategy

from neuronx_distributed.modules.moe.model_utils import DEFAULT_BLOCK_SIZE, DEFAULT_LNC_SIZE
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig, BlockwiseMatmulConfig

from neuronx_distributed.modules.moe.experts import Experts
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2

from neuronx_distributed.modules.moe.moe_parallel_layers import (
    ExpertFusedLinear,
    ExpertFusedLinearWithAsyncCommunication,
)

from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_size,
)

from neuronx_distributed.parallel_layers import layers, mappings, parallel_state, utils


_CONSTANT_INIT_VALUE = 0.5

def _fill_module_parameters(module: torch.nn.Module, value: float = _CONSTANT_INIT_VALUE) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)

def neuron_gptoss_swiglu(x,  alpha: float = 1.702, limit: float = 7.0):
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

    def forward(self, input_: torch.Tensor, expert_indices: Optional[torch.Tensor] = None, *_: Any) -> torch.Tensor:
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
        logical_nc_config=DEFAULT_LNC_SIZE, #uses lnc1 blockwise kernel by default
        parallelize_token_to_block_mapping: bool = True,
        early_expert_affinity_modulation: bool = False,
        optimized_block_to_token_mapping: bool = True,
        use_block_parallel: bool = False,
        always_augment_inputs_for_blockwise_matmul: bool = False,
        block_sharding_strategy:BlockShardStrategy = BlockShardStrategy.HI_LO,
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
