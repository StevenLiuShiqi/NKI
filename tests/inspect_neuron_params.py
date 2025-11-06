"""
Quick script to inspect Neuron model parameters after compilation.
"""

import os
import sys
from pathlib import Path
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import NeuronGPTOSSAttentionBlock, GPTOSSInferenceConfig, NeuronGPTOSSConfig
from neuronx_distributed_inference.utils.testing import build_module

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Simple config
neuron_config = NeuronGPTOSSConfig(
    batch_size=1,
    seq_len=128,
    tp_degree=1,
    torch_dtype="bfloat16",
    capacity_factor=None,
)

config = GPTOSSInferenceConfig(
    neuron_config=neuron_config,
    hidden_size=512,
    intermediate_size=2048,
    num_local_experts=4,
    num_experts_per_tok=4,
    num_attention_heads=8,
    num_key_value_heads=8,
    head_dim=64,
    vocab_size=1024,
    max_position_embeddings=128,
    num_hidden_layers=2,
    rms_norm_eps=1e-5,
    pad_token_id=0,
    rope_theta=10000.0,
    num_experts=4,
)

checkpoint_path = _ARTIFACTS_DIR / "inspect_params.pt"
if checkpoint_path.exists():
    checkpoint_path.unlink()

example_inputs = [
    (
        torch.zeros(1, 128, 512, dtype=torch.bfloat16),
        torch.zeros(1, 128, dtype=torch.long),
    )
]

print("Building Neuron module...")
neuron_block = build_module(
    NeuronGPTOSSAttentionBlock,
    example_inputs,
    tp_degree=1,
    module_init_kwargs={
        "config": config,
        "layer_idx": 1,
        "weight_init_value": 0.5,
    },
    checkpoint_path=str(checkpoint_path),
)

print("\n" + "="*80)
print("NEURON MODULE PARAMETERS")
print("="*80)

for name, param in neuron_block.named_parameters():
    print(f"{name:60s} {str(param.shape):20s} {param.dtype}")

print("\n" + "="*80)
print("NEURON MODULE BUFFERS")
print("="*80)

for name, buffer in neuron_block.named_buffers():
    print(f"{name:60s} {str(buffer.shape):20s} {buffer.dtype}")
