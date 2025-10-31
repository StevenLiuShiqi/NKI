import sys
import os

import torch 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import (
    GPTOSSInferenceConfig,
    NeuronGPTOSSConfig,
)
from gpt_oss import ModelConfig


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

def _make_original_inference_config():
    # Match the released GPT-OSS 20B configuration.
    neuron_config = NeuronGPTOSSConfig(
        batch_size=1,
        seq_len=4096,
        tp_degree=8,
        torch_dtype=torch.bfloat16,
        capacity_factor=None,
    )
    return GPTOSSInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=2880,
        intermediate_size=2880,
        num_local_experts=32,
        num_experts_per_tok=4,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        vocab_size=201088,
        max_position_embeddings=131072,
        num_hidden_layers=24,
        rms_norm_eps=1e-5,
        pad_token_id=199999,
        rope_theta=150000.0,
        sliding_window=128,
        num_experts=32,
    )

def _fill_module_parameters(module: torch.nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)

def _get_ref_config(config):
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
        rope_scaling_factor=1.0,
        rope_ntk_alpha=getattr(config, "rope_ntk_alpha", ModelConfig.rope_ntk_alpha),
        rope_ntk_beta=getattr(config, "rope_ntk_beta", ModelConfig.rope_ntk_beta),
    )
    
    return reference_config