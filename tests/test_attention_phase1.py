"""
Phase 1: Isolated Component Testing for Attention

Tests the core SDPA implementation from gpt_oss.py against the NeuronAttentionBase
implementation in model.py, focusing on isolated features:
- Test 1.1: Basic attention without special features
- Test 1.2: Causal masking
- Test 1.3: Sliding window attention
- Test 1.4: Learned sinks
"""

import os
import sys
from pathlib import Path

import torch
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import NeuronGPTOSSAttentionBlock, GPTOSSInferenceConfig, NeuronGPTOSSConfig
from src.gpt_oss import AttentionBlock, ModelConfig

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from test_utils import (
    _fill_module_parameters,
    _get_ref_config,
    _make_tiny_inference_config,
)

_CONSTANT_INIT_VALUE = 0.5
_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def _make_test_config(
    batch_size=2,
    seq_len=128,
    hidden_size=8,
    num_attention_heads=2,
    num_key_value_heads=2,
    head_dim=4,
    sliding_window=None,
    max_position_embeddings=None,
):
    """Create a test configuration with specified parameters."""
    if max_position_embeddings is None:
        max_position_embeddings = seq_len

    neuron_config = NeuronGPTOSSConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        tp_degree=1,
        torch_dtype="bfloat16",
        capacity_factor=None,
    )

    config = GPTOSSInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=hidden_size,
        intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=4,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=64,
        max_position_embeddings=max_position_embeddings,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        rope_theta=10000.0,
        num_experts=4,
    )

    if sliding_window is not None:
        config.sliding_window = sliding_window

    return config


def _build_neuron_and_reference_blocks(
    config,
    layer_idx=0,
    checkpoint_name="test_checkpoint.pt",
):
    """Build both Neuron and reference attention blocks with matching weights."""
    checkpoint_path = _ARTIFACTS_DIR / checkpoint_name
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    # Create example inputs for build_module
    example_inputs = [
        (
            torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.bfloat16),
            torch.zeros(batch_size, seq_len, dtype=torch.long),
        )
    ]

    # Build Neuron attention block
    neuron_block = build_module(
        NeuronGPTOSSAttentionBlock,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "layer_idx": layer_idx,
            "weight_init_value": _CONSTANT_INIT_VALUE,
        },
        checkpoint_path=str(checkpoint_path),
    )

    # Build reference block
    reference_config = _get_ref_config(config)
    reference_block = AttentionBlock(reference_config, layer_idx=layer_idx)
    _fill_module_parameters(reference_block, _CONSTANT_INIT_VALUE)

    return neuron_block, reference_block


def test_1_1_basic_attention_no_special_features():
    """
    Test 1.1: Basic Attention Without Special Features

    Setup: Single token sequence, no sliding window, no causal mask
    Goal: Validate core QKV computation and basic attention mechanics
    """
    print("\n" + "="*80)
    print("Test 1.1: Basic Attention Without Special Features")
    print("="*80)

    # Use small dimensions for testing: 4 tokens, 2 heads, 4 head_dim
    config = _make_test_config(
        batch_size=1,
        seq_len=4,
        hidden_size=8,  # 2 heads * 4 head_dim
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        sliding_window=None,
        max_position_embeddings=4,
    )

    # Use layer_idx=1 (odd) to disable sliding window
    neuron_block, reference_block = _build_neuron_and_reference_blocks(
        config, layer_idx=1, checkpoint_name="test_1_1.pt"
    )

    # Create test inputs
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(42)
    inp = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Reference forward pass (CPU)
    with torch.no_grad():
        flat_sample = inp.view(-1, hidden_size)
        ref_tokens = reference_block(flat_sample)
        reference_output = ref_tokens.view(batch_size, seq_len, hidden_size)

    print(f"Input shape: {inp.shape}")
    print(f"Reference output shape: {reference_output.shape}")
    print(f"Reference output sample: {reference_output[0, 0, :4]}")

    # Validate against Neuron implementation
    inputs = [(inp, position_ids)]
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test 1.1 PASSED: Basic attention matches!")


def test_1_2_causal_masking():
    """
    Test 1.2: Causal Masking

    Setup: Enable causal mask in both implementations
    Goal: Verify mask application produces identical results
    """
    print("\n" + "="*80)
    print("Test 1.2: Causal Masking")
    print("="*80)

    # 8 tokens, 4 heads
    config = _make_test_config(
        batch_size=1,
        seq_len=8,
        hidden_size=16,  # 4 heads * 4 head_dim
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        sliding_window=None,
        max_position_embeddings=8,
    )

    # Use layer_idx=1 (odd) to disable sliding window but keep causal mask
    neuron_block, reference_block = _build_neuron_and_reference_blocks(
        config, layer_idx=1, checkpoint_name="test_1_2.pt"
    )

    # Create test inputs
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(42)
    inp = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Reference forward pass (CPU)
    with torch.no_grad():
        flat_sample = inp.view(-1, hidden_size)
        ref_tokens = reference_block(flat_sample)
        reference_output = ref_tokens.view(batch_size, seq_len, hidden_size)

    print(f"Input shape: {inp.shape}")
    print(f"Reference output shape: {reference_output.shape}")
    print(f"Reference output sample: {reference_output[0, 0, :4]}")

    # Validate against Neuron implementation
    inputs = [(inp, position_ids)]
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test 1.2 PASSED: Causal masking matches!")


def test_1_3_sliding_window_attention():
    """
    Test 1.3: Sliding Window Attention

    Setup: Enable sliding window (e.g., window_size=4)
    Goal: Verify both implementations correctly limit attention context
    """
    print("\n" + "="*80)
    print("Test 1.3: Sliding Window Attention")
    print("="*80)

    # 16 tokens, 4 heads, sliding window of 4
    config = _make_test_config(
        batch_size=1,
        seq_len=16,
        hidden_size=16,  # 4 heads * 4 head_dim
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=4,
        sliding_window=4,
        max_position_embeddings=16,
    )

    # Use layer_idx=0 (even) to enable sliding window
    neuron_block, reference_block = _build_neuron_and_reference_blocks(
        config, layer_idx=0, checkpoint_name="test_1_3.pt"
    )

    # Create test inputs
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(42)
    inp = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Reference forward pass (CPU)
    with torch.no_grad():
        flat_sample = inp.view(-1, hidden_size)
        ref_tokens = reference_block(flat_sample)
        reference_output = ref_tokens.view(batch_size, seq_len, hidden_size)

    print(f"Input shape: {inp.shape}")
    print(f"Sliding window: {config.sliding_window}")
    print(f"Reference output shape: {reference_output.shape}")
    print(f"Reference output sample: {reference_output[0, 0, :4]}")

    # Validate against Neuron implementation
    inputs = [(inp, position_ids)]
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test 1.3 PASSED: Sliding window attention matches!")


def test_1_4_learned_sinks():
    """
    Test 1.4: Learned Sinks

    Setup: Add single learned sink parameter per head
    Goal: Verify sink tokens are correctly integrated into softmax
    """
    print("\n" + "="*80)
    print("Test 1.4: Learned Sinks")
    print("="*80)

    # 8 tokens, 2 heads, sink values initialized to constants
    config = _make_test_config(
        batch_size=1,
        seq_len=8,
        hidden_size=8,  # 2 heads * 4 head_dim
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        sliding_window=None,
        max_position_embeddings=8,
    )

    # Use layer_idx=1 (odd) to disable sliding window
    neuron_block, reference_block = _build_neuron_and_reference_blocks(
        config, layer_idx=1, checkpoint_name="test_1_4.pt"
    )

    # Create test inputs
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(42)
    inp = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Reference forward pass (CPU)
    with torch.no_grad():
        flat_sample = inp.view(-1, hidden_size)
        ref_tokens = reference_block(flat_sample)
        reference_output = ref_tokens.view(batch_size, seq_len, hidden_size)

    print(f"Input shape: {inp.shape}")
    print(f"Sink values (reference): {reference_block.sinks.data}")
    print(f"Reference output shape: {reference_output.shape}")
    print(f"Reference output sample: {reference_output[0, 0, :4]}")

    # Validate against Neuron implementation
    inputs = [(inp, position_ids)]
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test 1.4 PASSED: Learned sinks matches!")


def run_all_phase1_tests():
    """Run all Phase 1 tests sequentially."""
    print("\n" + "="*80)
    print("PHASE 1: ISOLATED COMPONENT TESTING")
    print("="*80)

    try:
        test_1_1_basic_attention_no_special_features()
    except Exception as e:
        print(f"✗ Test 1.1 FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_1_2_causal_masking()
    except Exception as e:
        print(f"✗ Test 1.2 FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_1_3_sliding_window_attention()
    except Exception as e:
        print(f"✗ Test 1.3 FAILED: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_1_4_learned_sinks()
    except Exception as e:
        print(f"✗ Test 1.4 FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("PHASE 1 TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run_all_phase1_tests()
