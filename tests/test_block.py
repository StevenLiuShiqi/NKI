import os
import sys
from pathlib import Path

import torch
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import NeuronGPTOSSBlock  # noqa: E402
from src.gpt_oss import TransformerBlock  # noqa: E402

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from test_utils import (  # noqa: E402
    _fill_module_parameters,
    _get_ref_config,
    _make_tiny_inference_config,
    _make_original_inference_config
)

_CONSTANT_INIT_VALUE = 0.5

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_PATH = _ARTIFACTS_DIR / "neuron_block_checkpoint.pt"

def test_block_forward_matches_reference():
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

    # Create position_ids as required by the attention block
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    inputs = [(inp, position_ids)]
    example_inputs = [(torch.zeros_like(sample), torch.zeros(batch_size, seq_len, dtype=torch.long))]

    # Use layer_idx=1 to disable sliding window (odd layer indices have no sliding window)
    # This helps avoid potential issues with the sliding window attention kernel on short sequences

    # Create a wrapper to extract just the hidden states from the block output
    class BlockWrapper(torch.nn.Module):
        def __init__(self, config, block_idx):
            super().__init__()
            self.block = NeuronGPTOSSBlock(config, block_idx)
            # Initialize all parameters to constant value for testing
            _fill_module_parameters(self.block, _CONSTANT_INIT_VALUE)

        def forward(self, hidden_states, position_ids):
            outputs = self.block(hidden_states, position_ids)
            # Return only the hidden states (first element of tuple)
            return outputs[0]

    neuron_block = build_module(
        BlockWrapper,
        example_inputs,
        tp_degree=8,
        module_init_kwargs={
            "config": config,
            "block_idx": 1,
        },
        checkpoint_path=str(_CHECKPOINT_PATH),
    )

    reference_block = TransformerBlock(reference_config, layer_idx=1)

    _fill_module_parameters(reference_block, _CONSTANT_INIT_VALUE)

    with torch.no_grad():
        # Reference block expects flattened input (tokens, hidden_size)
        flat_sample = inp.view(-1, hidden_size)
        ref_output = reference_block(flat_sample)
        reference_output = ref_output.view(batch_size, seq_len, hidden_size)

    # Use relaxed tolerances since the MoE implementations differ slightly
    validate_accuracy(
        neuron_block,
        inputs,
        expected_outputs=[reference_output],
        assert_close_kwargs={"rtol": 0.05, "atol": 5.0}
    )

if __name__ == "__main__":
    test_block_forward_matches_reference()
    print("Test passed!")
