"""
Phase 1: Isolated Component Testing for Attention

Tests the core SDPA implementation from gpt_oss.py against the NeuronAttentionBase
implementation in model.py, focusing on isolated features:
- Test 1.1: Basic attention without special features
- Test 1.2: Basic attention with RMS normalized inputs
- Test 1.3: Sliding window attention
- Test 1.4: Learned sinks
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model import NeuronGPTOSSAttentionBlock, NeuronGPTOSSAttentionBlockCompiled, GPTOSSInferenceConfig, NeuronGPTOSSConfig
from src.gpt_oss import AttentionBlock, ModelConfig

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from tests.test_utils import (
    _fill_module_parameters,
    _get_ref_config,
    _make_original_inference_config,
    _make_tiny_inference_config,
    _sync_reference_weights_to_neuron,
)

import logging

# Enable debug logging for neuronx modules
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Specifically enable debug logging for the "Neuron" logger used by attention_base
logging.getLogger("Neuron").setLevel(logging.DEBUG)
_CONSTANT_INIT_VALUE = 0.1
_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# NOTE: We do NOT initialize the process group here.
# The neuronx_distributed library will initialize it automatically when needed
# via initialize_fallback_parallel_state(). Since we can't create multiple
# uncompiled neuron blocks (each tries to init), we'll use the uniform weight
# approach for tests instead of random weights.


def _make_test_config(
    batch_size=1,
    sliding_window=128,
):
    """Create a test configuration with specified parameters."""

    neuron_config = NeuronGPTOSSConfig(
        batch_size=batch_size,
        seq_len=4096,
        tp_degree=1,
        torch_dtype=torch.bfloat16,
        capacity_factor=None,
    )
    config = GPTOSSInferenceConfig(
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
        sliding_window=sliding_window,
        num_experts=32,
    )
    

    if sliding_window is not None:
        config.sliding_window = sliding_window

    return config


def _build_neuron_and_reference_blocks(
    config,
    layer_idx=0,
    checkpoint_name="test_checkpoint.pt",
):
    """Build both Neuron and reference attention blocks with matching UNIFORM weights."""
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
        NeuronGPTOSSAttentionBlockCompiled,
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

def _build_neuron_and_reference_blocks_random_weights(
    config,
    layer_idx=0,
    checkpoint_name="test_checkpoint.pt",
    seed=42,
):
    """
    Build both Neuron and reference attention blocks with matching RANDOM weights.

    This function works around the process group initialization issue by:
    1. First compiling with uniform weights to generate a checkpoint
    2. Loading and modifying the checkpoint with random weights
    3. Recompiling with the modified checkpoint

    NOTE: We cannot create uncompiled neuron blocks multiple times in the same
    process because each one tries to initialize the distributed process group.

    Args:
        config: GPTOSSInferenceConfig
        layer_idx: Layer index (affects sliding window behavior)
        checkpoint_name: Name for the checkpoint file
        seed: Random seed for reproducibility

    Returns:
        (neuron_block, reference_block) with matching random weights
    """
    checkpoint_path = _ARTIFACTS_DIR / checkpoint_name
    temp_checkpoint_path = _ARTIFACTS_DIR / f"temp_{checkpoint_name}"

    # Clean up old checkpoints
    for path in [checkpoint_path, temp_checkpoint_path]:
        if path.exists():
            path.unlink()

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

    print(f"\n{'='*80}")
    print(f"Building blocks with RANDOM weights (seed={seed})")
    print(f"{'='*80}")

    # Step 1: Build with uniform weights to generate initial checkpoint
    print(f"\nStep 1: Creating initial neuron block with uniform weights...")
    _temp_neuron_block = build_module(
        NeuronGPTOSSAttentionBlockCompiled,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "layer_idx": layer_idx,
            "weight_init_value": _CONSTANT_INIT_VALUE,
        },
        checkpoint_path=str(temp_checkpoint_path),
    )
    print(f"  Generated checkpoint at: {temp_checkpoint_path}")
    del _temp_neuron_block  # Free memory

    # Step 2: Create reference block with random weights
    torch.manual_seed(seed)
    print(f"\nStep 2: Creating reference block with random weights...")
    reference_config = _get_ref_config(config)
    reference_block = AttentionBlock(reference_config, layer_idx=layer_idx)

    # Step 3: Load checkpoint and sync weights
    print(f"\nStep 3: Loading checkpoint and syncing weights...")
    neuron_state_dict = torch.load(temp_checkpoint_path)
    ref_state_dict = reference_block.state_dict()

    # Get dimensions from reference block
    num_heads = reference_block.num_attention_heads
    num_kv_heads = reference_block.num_key_value_heads
    head_dim = reference_block.head_dim

    q_out = num_heads * head_dim
    kv_out = num_kv_heads * head_dim

    print(f"  Unfusing QKV weights:")
    print(f"    num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"    Q output dim: {q_out}, K/V output dim: {kv_out}")

    updated_count = 0

    # Handle QKV weights - unfuse from reference to separate Q, K, V
    if 'qkv.weight' in ref_state_dict:
        qkv_weight = ref_state_dict['qkv.weight']  # [qkv_dim, hidden]
        print(f"    Fused QKV weight shape: {qkv_weight.shape}")

        # Split into Q, K, V (matching gpt_oss.py:220-232)
        q_weight = qkv_weight[:q_out, :]
        k_weight = qkv_weight[q_out:q_out+kv_out, :]
        v_weight = qkv_weight[q_out+kv_out:q_out+2*kv_out, :]

        # Copy to neuron state
        for key, tensor in [
            ('qkv_proj.q_proj.weight', q_weight),
            ('qkv_proj.k_proj.weight', k_weight),
            ('qkv_proj.v_proj.weight', v_weight)
        ]:
            if key in neuron_state_dict:
                if tensor.dtype != neuron_state_dict[key].dtype:
                    tensor = tensor.to(dtype=neuron_state_dict[key].dtype)
                neuron_state_dict[key] = tensor
                updated_count += 1
                print(f"  [COPY] qkv.weight -> {key}")

    # Handle QKV biases - unfuse from reference to separate Q, K, V
    if 'qkv.bias' in ref_state_dict:
        qkv_bias = ref_state_dict['qkv.bias']  # [qkv_dim]

        # Split into Q, K, V
        q_bias = qkv_bias[:q_out]
        k_bias = qkv_bias[q_out:q_out+kv_out]
        v_bias = qkv_bias[q_out+kv_out:q_out+2*kv_out]

        # Copy to neuron state
        for key, tensor in [
            ('qkv_proj.q_proj.bias', q_bias),
            ('qkv_proj.k_proj.bias', k_bias),
            ('qkv_proj.v_proj.bias', v_bias)
        ]:
            if key in neuron_state_dict:
                if tensor.dtype != neuron_state_dict[key].dtype:
                    tensor = tensor.to(dtype=neuron_state_dict[key].dtype)
                neuron_state_dict[key] = tensor
                updated_count += 1
                print(f"  [COPY] qkv.bias -> {key}")

    # Handle output projection
    if 'out.weight' in ref_state_dict and 'o_proj.o_proj.weight' in neuron_state_dict:
        ref_tensor = ref_state_dict['out.weight']
        if ref_tensor.dtype != neuron_state_dict['o_proj.o_proj.weight'].dtype:
            ref_tensor = ref_tensor.to(dtype=neuron_state_dict['o_proj.o_proj.weight'].dtype)
        neuron_state_dict['o_proj.o_proj.weight'] = ref_tensor
        updated_count += 1
        print(f"  [COPY] out.weight -> o_proj.o_proj.weight")

    if 'out.bias' in ref_state_dict and 'o_proj.o_proj.bias' in neuron_state_dict:
        ref_tensor = ref_state_dict['out.bias']
        if ref_tensor.dtype != neuron_state_dict['o_proj.o_proj.bias'].dtype:
            ref_tensor = ref_tensor.to(dtype=neuron_state_dict['o_proj.o_proj.bias'].dtype)
        neuron_state_dict['o_proj.o_proj.bias'] = ref_tensor
        updated_count += 1
        print(f"  [COPY] out.bias -> o_proj.o_proj.bias")

    # Handle learned sinks
    if 'sinks' in ref_state_dict:
        ref_sinks = ref_state_dict['sinks']
        for nk in ['learned_sinks.sink', 'tkg_learned_sinks.sink']:
            if nk in neuron_state_dict:
                ref_converted = ref_sinks.clone()
                if ref_converted.dtype != neuron_state_dict[nk].dtype:
                    ref_converted = ref_converted.to(dtype=neuron_state_dict[nk].dtype)
                neuron_state_dict[nk] = ref_converted
                updated_count += 1
                print(f"  [COPY] sinks -> {nk}")

    # Step 4: Save modified checkpoint
    print(f"\nStep 4: Saving modified checkpoint with random weights...")
    torch.save(neuron_state_dict, checkpoint_path)
    print(f"  Saved {updated_count} weight tensors to: {checkpoint_path}")

    # Step 5: Compile with modified checkpoint
    print(f"\nStep 5: Compiling neuron block with modified checkpoint...")
    neuron_block = build_module(
        NeuronGPTOSSAttentionBlockCompiled,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "layer_idx": layer_idx,
        },
        checkpoint_path=str(checkpoint_path),
    )

    # Clean up temp checkpoint
    # if temp_checkpoint_path.exists():
    #     temp_checkpoint_path.unlink()

    print(f"\n{'='*80}")
    print(f"Blocks ready with matching random weights")
    print(f"{'='*80}\n")

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

    config = _make_test_config(
        batch_size=1,
        sliding_window=None,
    )
    
    config = _make_original_inference_config()

    # Use layer_idx=1 (odd) to disable sliding window
    neuron_block, reference_block = _build_neuron_and_reference_blocks_random_weights(
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



def test_1_3_sliding_window_attention():
    """
    Test 1.3: Sliding Window Attention

    Setup: Enable sliding window
    Goal: Verify both implementations correctly limit attention context
    """
    print("\n" + "="*80)
    print("Test 1.3: Sliding Window Attention")
    print("="*80)

    config = _make_test_config(
        batch_size=1,
        sliding_window=128
    )

    # Use layer_idx=0 (even) to enable sliding window
    neuron_block, reference_block = _build_neuron_and_reference_blocks_random_weights(
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


def test_1_2_normalized_inputs():
    """
    Test 1.2: Basic Attention With RMS Normalized Inputs

    Setup: Single token sequence with RMS normalization applied to inputs
    Goal: Validate attention behavior with normalized inputs (as used in actual transformer layers)
    """
    print("\n" + "="*80)
    print("Test 1.2: Basic Attention With RMS Normalized Inputs")
    print("="*80)

    config = _make_test_config(
        batch_size=1,
        sliding_window=None,
    )

    config = _make_original_inference_config()

    # Use layer_idx=1 (odd) to disable sliding window
    neuron_block, reference_block = _build_neuron_and_reference_blocks_random_weights(
        config, layer_idx=0, checkpoint_name="test_1_2.pt"
    )

    # Create test inputs
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    # Create RMS norm layer matching the config
    from src.gpt_oss import RMSNorm as ReferenceRMSNorm
    rms_norm = ReferenceRMSNorm(hidden_size, eps=config.rms_norm_eps)

    # Initialize RMS norm scale to constant for reproducibility
    torch.nn.init.constant_(rms_norm.scale, _CONSTANT_INIT_VALUE)

    torch.manual_seed(42)
    inp_raw = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Apply RMS normalization to inputs
    with torch.no_grad():
        inp = rms_norm(inp_raw)

    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Reference forward pass (CPU)
    with torch.no_grad():
        flat_sample = inp.view(-1, hidden_size)
        ref_tokens = reference_block(flat_sample)
        reference_output = ref_tokens.view(batch_size, seq_len, hidden_size)

    print(f"Raw input shape: {inp_raw.shape}")
    print(f"Raw input sample (before norm): {inp_raw[0, 0, :4]}")
    print(f"Normalized input sample (after norm): {inp[0, 0, :4]}")
    print(f"Reference output shape: {reference_output.shape}")
    print(f"Reference output sample: {reference_output[0, 0, :4]}")

    # Validate against Neuron implementation
    inputs = [(inp, position_ids)]
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test 1.2 PASSED: Attention with normalized inputs matches!")


def test_1_4_learned_sinks():
    """
    Test 1.4: Learned Sinks

    Setup: Add single learned sink parameter per head
    Goal: Verify sink tokens are correctly integrated into softmax

    NOTE: This test is currently disabled due to NKI kernel compilation issues
    with learned sinks. The error occurs during neuronx-cc compilation:
    "Access pattern out of bound" when loading sink values.
    """
    print("\n" + "="*80)
    print("Test 1.4: Learned Sinks (SKIPPED)")
    print("="*80)
    print("⚠ Test 1.4 SKIPPED: Learned sinks cause NKI compilation errors")
    return

    # 8 tokens, 2 heads, sink values initialized to constants
    config = _make_test_config(
        batch_size=2,
        sliding_window=None,
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


def test_1_5_basic_attention_with_random_weights():
    """
    Test 1.5: Basic Attention With Random Weights

    Setup: Single token sequence with random weight initialization
    Goal: Validate that the weight sync mechanism works and both implementations
          produce identical outputs with realistic random weights
    """
    print("\n" + "="*80)
    print("Test 1.5: Basic Attention With Random Weights")
    print("="*80)

    config = _make_original_inference_config()

    # Use layer_idx=1 (odd) to disable sliding window
    neuron_block, reference_block = _build_neuron_and_reference_blocks_random_weights(
        config, layer_idx=1, checkpoint_name="test_1_5.pt", seed=12345
    )

    # Create test inputs
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(67890)  # Different seed for inputs
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
    print(f"Reference output mean: {reference_output.mean():.6f}, std: {reference_output.std():.6f}")

    # Validate against Neuron implementation
    inputs = [(inp, position_ids)]
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test 1.5 PASSED: Basic attention with random weights matches!")

def test_1_6_compiled_attention_with_random_weights():
    """

    """
    print("\n" + "="*80)
    print("Test 1.6: Compiled Attention With Random Weights")
    print("="*80)

    config = _make_tiny_inference_config()

    # Use layer_idx=1 (odd) to disable sliding window
    neuron_block, reference_block = _build_neuron_and_reference_blocks_random_weights(
        config, layer_idx=0, checkpoint_name="test_1_6.pt", seed=12345
    )

    # Create test inputs
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(67890)  # Different seed for inputs
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
    print(f"Reference output mean: {reference_output.mean():.6f}, std: {reference_output.std():.6f}")

    # Validate against Neuron implementation
    inputs = [(inp, position_ids)]
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test 1.6 PASSED: Compiled attention with random weights matches!")
    
def run_all_phase1_tests():
    """Run all Phase 1 tests sequentially."""
    print("\n" + "="*80)
    print("PHASE 1: ISOLATED COMPONENT TESTING")
    print("="*80)

    # try:
    #     test_1_1_basic_attention_no_special_features()
    # except Exception as e:
    #     print(f"✗ Test 1.1 FAILED: {e}")
    #     import traceback
    #     traceback.print_exc()

    # try:
    #     test_1_2_normalized_inputs()
    # except Exception as e:
    #     print(f"✗ Test 1.2 FAILED: {e}")
    #     import traceback
    #     traceback.print_exc()

    # try:
    #     test_1_3_sliding_window_attention()
    # except Exception as e:
    #     print(f"✗ Test 1.3 FAILED: {e}")
    #     import traceback
    #     traceback.print_exc()

    # try:
    #     test_1_4_learned_sinks()
    # except Exception as e:
    #     print(f"✗ Test 1.4 FAILED: {e}")
    #     import traceback
    #     traceback.print_exc()

    # try:
    #     test_1_5_basic_attention_with_random_weights()
    # except Exception as e:
    #     print(f"✗ Test 1.5 FAILED: {e}")
    #     import traceback
    #     traceback.print_exc()
        
    try:
        test_1_6_compiled_attention_with_random_weights()
    except Exception as e:
        print(f"✗ Test 1.6 FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("PHASE 1 TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run_all_phase1_tests()
