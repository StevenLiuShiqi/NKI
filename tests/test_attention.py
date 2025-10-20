# Incomplete
# TODO: Test attention implementation

import argparse
import copy
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import NeuronGPTOSSAttentionBlock, GPTOSSInferenceConfig, NeuronGPTOSSConfig
from gpt_oss import AttentionBlock, ModelConfig

import torch

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from dataclasses import dataclass

_CONSTANT_INIT_VALUE = 0.01


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
    

class _NeuronAttentionWrapper(torch.nn.Module):
    """Wrap Neuron attention block to expose a tensor-only output for tracing."""

    def __init__(self, config):
        super().__init__()
        self.inner = NeuronGPTOSSAttentionBlock(config=config)

    def forward(self, hidden_states, attention_mask, position_ids):
        output = self.inner(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return output.hidden_states


class _ReferenceAttentionWrapper(torch.nn.Module):
    """Adapters the PyTorch GPT-OSS attention block to the Neuron interface."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block = AttentionBlock(config=config)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        bsz, seq, hidden = hidden_states.shape
        tokens = hidden_states.reshape(bsz * seq, hidden)
        output = self.block(tokens)
        return output.reshape(bsz, seq, hidden)


def test_validate_accuracy_basic_module():
    
    batch_size = 1
    seq_len = 16

    def generate_position_ids(batch, seq, device):
        base = torch.arange(seq, dtype=torch.long, device=device)
        return base.unsqueeze(0).expand(batch, -1)

    def generate_attention_mask(batch, seq, device):
        return torch.ones((batch, 1, seq, seq), dtype=torch.bool, device=device)

    example_hidden_states = torch.randn(batch_size, seq_len, 2880, dtype=torch.bfloat16)
    example_position_ids = generate_position_ids(batch_size, seq_len, example_hidden_states.device)
    example_attention_mask = generate_attention_mask(batch_size, seq_len, example_hidden_states.device)

    x = [(example_hidden_states, example_attention_mask, example_position_ids)]

    run_hidden_states = torch.randn(batch_size, seq_len, 2880, dtype=torch.bfloat16)
    run_position_ids = generate_position_ids(batch_size, seq_len, run_hidden_states.device)
    run_attention_mask = generate_attention_mask(batch_size, seq_len, run_hidden_states.device)

    config = _make_tiny_inference_config()
    
    reference_config = ModelConfig(
        num_hidden_layers=config.num_hidden_layers,
        num_experts=getattr(config, "num_experts", ModelConfig.num_experts),
        experts_per_token=getattr(config, "experts_per_token", ModelConfig.experts_per_token),
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=getattr(config, "intermediate_size", config.hidden_size),
        swiglu_limit=getattr(config, "swiglu_limit", ModelConfig.swiglu_limit),
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        sliding_window=getattr(config, "sliding_window", ModelConfig.sliding_window),
        initial_context_length=getattr(
            config, "max_position_embeddings", ModelConfig.initial_context_length
        ),
        rope_theta=config.rope_theta,
        rope_scaling_factor=getattr(config, "rope_scaling_factor", ModelConfig.rope_scaling_factor),
        rope_ntk_alpha=getattr(config, "rope_ntk_alpha", ModelConfig.rope_ntk_alpha),
        rope_ntk_beta=getattr(config, "rope_ntk_beta", ModelConfig.rope_ntk_beta),
    )

    reference_module = _ReferenceAttentionWrapper(reference_config)
    _fill_module_parameters(reference_module)
    reference_module.eval()

    neuron_module = _NeuronAttentionWrapper(config=config)
    _fill_module_parameters(neuron_module)
    neuron_module.eval()


    checkpoint_dir = Path(tempfile.mkdtemp(prefix="nxdi_test_cpu_"))
    checkpoint_path = checkpoint_dir / "cpu_checkpoint.pt"
    torch.save(neuron_module.state_dict(), checkpoint_path)
    neuron_module.load_state_dict(torch.load(checkpoint_path))
    _fill_module_parameters(neuron_module)

    neuron_model = build_module(
        _NeuronAttentionWrapper,
        x,
        module_init_kwargs={"config": config},
        checkpoint_path=str(checkpoint_path),
    )

    neuron_x = [(run_hidden_states, run_attention_mask, run_position_ids)]

    with torch.no_grad():
        expected_output = reference_module(*neuron_x[0])

    validate_accuracy(neuron_model, neuron_x, expected_outputs=[expected_output])

    shutil.rmtree(checkpoint_dir, ignore_errors=True)

test_validate_accuracy_basic_module()
