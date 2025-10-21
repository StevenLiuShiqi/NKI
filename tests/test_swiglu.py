import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from neuronx_distributed_inference.utils.testing import build_function, validate_accuracy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model import (
    GPTOSSInferenceConfig,
    NeuronGPTOSSConfig,
)

from gpt_oss import swiglu

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
    
def neuron_swiglu(x,  alpha: float = 1.702, limit: float = 7.0):
    # gate, up = torch.chunk(x, chunks=2, dim=-1)
    # return F.silu(gate) * (up + 1)
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)



def test_validate_accuracy_basic_module():
    config = _make_tiny_inference_config()
    
    sample = torch.randn(12, config.hidden_size, dtype=config.neuron_config.torch_dtype)
    inputs = [(sample,)]
    example_inputs = [(torch.zeros_like(sample),)]
    
    neuron_model = build_function(neuron_swiglu, example_inputs)
    validate_accuracy(neuron_model, inputs, cpu_callable=swiglu)

test_validate_accuracy_basic_module()