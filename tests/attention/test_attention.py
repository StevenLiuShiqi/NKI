import os
import sys
from pathlib import Path

import torch
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import NeuronGPTOSSAttentionBlock  # noqa: E402
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
_CHECKPOINT_PATH = _ARTIFACTS_DIR / "neuron_attention_checkpoint.pt"
_CONSTANT_INIT_VALUE = 0.5

def test_attention_block_forward_matches_reference():
    config = _make_tiny_inference_config()
    reference_config = _get_ref_config(config=config)

    
    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()
        
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(0)
    inp = torch.rand(batch_size, seq_len, hidden_size, dtype=config.neuron_config.torch_dtype)
    sample = torch.arange(batch_size * seq_len * hidden_size, dtype=config.neuron_config.torch_dtype)
    sample = sample.reshape(batch_size, seq_len, hidden_size).to(dtype=config.neuron_config.torch_dtype)
    # Update inputs to include position_ids as a keyword argument
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    inputs = [(inp, position_ids)]
    example_inputs = [(torch.zeros_like(sample), torch.zeros(batch_size, seq_len, dtype=torch.long))]
    
    # Use layer_idx=1 to disable sliding window (odd layer indices have no sliding window)
    # This helps avoid potential issues with the sliding window attention kernel on short sequences
    neuron_block = build_module(
        NeuronGPTOSSAttentionBlock,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "layer_idx": 1,
            "weight_init_value": _CONSTANT_INIT_VALUE,
        },
        checkpoint_path=str(_CHECKPOINT_PATH),
    )

    reference_block = AttentionBlock(reference_config, layer_idx=1)

    _fill_module_parameters(reference_block, _CONSTANT_INIT_VALUE)

    with torch.no_grad():
        flat_sample = inp.view(-1, hidden_size)
        ref_tokens = reference_block(flat_sample)
        reference_output = ref_tokens.view(batch_size, seq_len, hidden_size)

    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

test_attention_block_forward_matches_reference()
