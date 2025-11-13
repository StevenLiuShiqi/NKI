import os
import sys
import torch

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.model import NeuronGPTOSSBlock
from src.gpt_oss import TransformerBlock
from tests.test_utils import _fill_module_parameters, _get_ref_config, _make_tiny_inference_config

_CONSTANT_INIT_VALUE = 0.5

def test_simple():
    config = _make_tiny_inference_config()
    reference_config = _get_ref_config(config=config)

    batch_size = 1
    seq_len = 2
    hidden_size = config.hidden_size

    # Simple input for debugging
    inp = torch.ones(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Create blocks
    neuron_block = NeuronGPTOSSBlock(config, block_idx=1, weight_init_value=_CONSTANT_INIT_VALUE)
    reference_block = TransformerBlock(reference_config, layer_idx=1)
    _fill_module_parameters(reference_block, _CONSTANT_INIT_VALUE)

    with torch.no_grad():
        # Neuron output
        neuron_out = neuron_block(inp, position_ids)
        print(f"Neuron output shape: {neuron_out[0].shape}")
        print(f"Neuron output: {neuron_out[0]}")

        # Reference output
        flat_inp = inp.view(-1, hidden_size)
        ref_out = reference_block(flat_inp)
        ref_out_reshaped = ref_out.view(batch_size, seq_len, hidden_size)
        print(f"\nReference output shape: {ref_out_reshaped.shape}")
        print(f"Reference output: {ref_out_reshaped}")

        print(f"\nDifference: {neuron_out[0] - ref_out_reshaped}")
        print(f"Max difference: {(neuron_out[0] - ref_out_reshaped).abs().max()}")

if __name__ == "__main__":
    test_simple()
