import os
import sys
from pathlib import Path

import torch
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import NeuronGPTOSSBlock  # noqa: E402
from src.gpt_oss import AttentionBlock, MLPBlock  # noqa: E402

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from test_utils import (  # noqa: E402
    _fill_module_parameters,
    _get_ref_config,
    _make_tiny_inference_config,
)

_CONSTANT_INIT_VALUE = 0.5

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_PATH = _ARTIFACTS_DIR / "neuron_decoder_checkpoint.pt"
_CONSTANT_INIT_VALUE = 0.5

class _ReferenceDecoderBlock(torch.nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.attention = AttentionBlock(config=config, layer_idx=layer_idx)
        self.mlp = MLPBlock(config=config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        tokens = hidden_states.view(-1, original_shape[-1])
        attn_output = self.attention(tokens)
        mlp_input = attn_output
        mlp_output = self.mlp(mlp_input)
        combined = mlp_input + mlp_output
        return combined.view(original_shape)


def test_decoder_block_forward_matches_reference():
    config = _make_tiny_inference_config()
    reference_config = _get_ref_config(config=config)

    
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(0)
    sample = torch.arange(batch_size * seq_len * hidden_size, dtype=config.neuron_config.torch_dtype)
    sample = sample.reshape(batch_size, seq_len, hidden_size).to(dtype=config.neuron_config.torch_dtype)
    # Update inputs to include position_ids as a keyword argument
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    inputs = [(sample, position_ids)]
    example_inputs = [(torch.zeros_like(sample), torch.zeros(batch_size, seq_len, dtype=torch.long))]
    
    
    neuron_block = build_module(
        NeuronGPTOSSBlock,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "block_idx": 0,
        },
        checkpoint_path=str(_CHECKPOINT_PATH),
    )

    reference_block = _ReferenceDecoderBlock(reference_config, layer_idx=0)

    _fill_module_parameters(neuron_block, _CONSTANT_INIT_VALUE)
    _fill_module_parameters(reference_block, _CONSTANT_INIT_VALUE)


    with torch.no_grad():
        reference_output = reference_block(sample)

    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])


test_decoder_block_forward_matches_reference()
