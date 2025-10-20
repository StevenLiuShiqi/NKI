import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import (
    GPTOSSInferenceConfig,
    NeuronGPTOSSConfig,
    # NeuronMLPBlock,
)
from gpt_oss import MLPBlock, ModelConfig

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed.modules.moe.expert_mlps import ExpertMLPs
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.rms_norm import RMSNorm

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_PATH = _ARTIFACTS_DIR / "neuron_mlp_checkpoint.pt"
_CONSTANT_INIT_VALUE = 0.1


def _fill_module_parameters(module: torch.nn.Module, value: float = _CONSTANT_INIT_VALUE) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _make_tiny_inference_config():
    neuron_config = NeuronGPTOSSConfig(
        batch_size=2,
        seq_len=6,
        tp_degree=1,
        torch_dtype="bfloat16",
        # glu_mlp=True,
        capacity_factor=None,
    )
    return GPTOSSInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=8,
        intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        max_position_embeddings=32,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        rope_theta=10000.0,
        num_experts=4,
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
        self.expert_mlps = ExpertMLPs(
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
        # t_flat = t.view(-1, t.shape[-1])  # (B*S, H)
        seq_len = x.shape[1]
        moe_output = self.expert_mlps(
            hidden_states=t,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=seq_len
        )
        moe_output = moe_output.view_as(x)
        return moe_output

def test_validate_accuracy_basic_module():
    config = _make_tiny_inference_config()

    sample = torch.randn(12, config.hidden_size, dtype=config.neuron_config.torch_dtype)
    inputs = [(sample,)]
    example_inputs = [(torch.zeros_like(sample),)]

    reference_config = ModelConfig(
        num_hidden_layers=config.num_hidden_layers,
        num_experts=config.num_local_experts,
        experts_per_token=config.num_experts_per_tok,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        sliding_window=getattr(config, "sliding_window", ModelConfig.sliding_window),
        initial_context_length=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rope_scaling_factor=getattr(
            config, "rope_scaling_factor", ModelConfig.rope_scaling_factor
        ),
        rope_ntk_alpha=getattr(config, "rope_ntk_alpha", ModelConfig.rope_ntk_alpha),
        rope_ntk_beta=getattr(config, "rope_ntk_beta", ModelConfig.rope_ntk_beta),
    )

    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()

    neuron_model = build_module(
        NeuronMLPBlock,
        example_inputs,
        module_init_kwargs={
            "config": config,
            "weight_init_value": _CONSTANT_INIT_VALUE,
        },
        checkpoint_path=str(_CHECKPOINT_PATH),
    )

    module_cpu = MLPBlock(
        config=reference_config,
        weight_init_value=_CONSTANT_INIT_VALUE,
    )

    _fill_module_parameters(module_cpu)
    
    def cpu_forward(x):
        return module_cpu(x)
    
    with torch.no_grad():
        expected_output = cpu_forward(*inputs[0])


    validate_accuracy(neuron_model, inputs, expected_outputs=[expected_output])


test_validate_accuracy_basic_module()
