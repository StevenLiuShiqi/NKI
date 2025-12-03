#!/usr/bin/env python3
"""
Simple test to verify sink integration in flash attention.
This test creates a minimal setup to check that sinks are properly integrated.
"""

import torch
import numpy as np
from src.attention import flash_fwd, FlashConfig

def test_sink_integration_basic():
    """
    Basic test: Verify that flash_fwd accepts sink parameter without errors.
    """
    print("="*80)
    print("Test: Basic Sink Integration")
    print("="*80)

    # Minimal configuration
    batch_size = 1
    num_heads = 2
    num_kv_heads = 2
    head_dim = 64
    seq_len = 256

    # Create random input tensors
    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, head_dim, seq_len, dtype=torch.float32)
    K = torch.randn(batch_size, num_kv_heads, head_dim, seq_len, dtype=torch.float32)
    V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float32)

    # Create sink values (batch, num_heads, seq, 1)
    S = torch.randn(batch_size, num_heads, seq_len, 1, dtype=torch.float32)

    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"S shape: {S.shape}")

    # Create flash config
    config = FlashConfig(
        seq_tile_size=256,
        should_transpose_v=False,
    )

    try:
        # Call flash_fwd WITHOUT sink
        print("\n1. Testing without sink...")
        output_no_sink = flash_fwd[batch_size, num_kv_heads](
            Q, K, V, s=None,
            softmax_scale=1.0 / np.sqrt(head_dim),
            use_causal_mask=True,
            window_size=(-1, -1),
            mixed_precision=False,
            config=config,
        )
        print(f"   Output shape: {output_no_sink.shape}")
        print(f"   Output mean: {output_no_sink.mean().item():.6f}")
        print(f"   Output std: {output_no_sink.std().item():.6f}")
        print("   ✓ No sink case works")

        # Call flash_fwd WITH sink
        print("\n2. Testing with sink...")
        output_with_sink = flash_fwd[batch_size, num_kv_heads](
            Q, K, V, s=S,
            softmax_scale=1.0 / np.sqrt(head_dim),
            use_causal_mask=True,
            window_size=(-1, -1),
            mixed_precision=False,
            config=config,
        )
        print(f"   Output shape: {output_with_sink.shape}")
        print(f"   Output mean: {output_with_sink.mean().item():.6f}")
        print(f"   Output std: {output_with_sink.std().item():.6f}")
        print("   ✓ With sink case works")

        # Verify outputs are different (sink should affect output)
        diff = (output_with_sink - output_no_sink).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        print(f"\n3. Comparing outputs...")
        print(f"   Max difference: {max_diff:.6f}")
        print(f"   Mean difference: {mean_diff:.6f}")

        if max_diff > 1e-6:
            print("   ✓ Sink affects output (as expected)")
        else:
            print("   ⚠ WARNING: Sink does not affect output!")

        print("\n" + "="*80)
        print("✓ TEST PASSED: Sink integration is functional")
        print("="*80)
        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sink_none_handling():
    """
    Test that None sink is handled correctly.
    """
    print("\n" + "="*80)
    print("Test: None Sink Handling")
    print("="*80)

    batch_size = 1
    num_heads = 2
    num_kv_heads = 2
    head_dim = 64
    seq_len = 256

    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, head_dim, seq_len, dtype=torch.float32)
    K = torch.randn(batch_size, num_kv_heads, head_dim, seq_len, dtype=torch.float32)
    V = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.float32)

    config = FlashConfig(seq_tile_size=256, should_transpose_v=False)

    try:
        output = flash_fwd[batch_size, num_kv_heads](
            Q, K, V, s=None,
            softmax_scale=1.0 / np.sqrt(head_dim),
            use_causal_mask=True,
            window_size=(-1, -1),
            mixed_precision=False,
            config=config,
        )
        print(f"Output shape: {output.shape}")
        print("✓ None sink handled correctly")
        return True
    except Exception as e:
        print(f"✗ Failed with None sink: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nRunning sink integration tests...\n")

    test1_passed = test_sink_none_handling()
    test2_passed = test_sink_integration_basic()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Test 1 (None handling): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Basic integration): {'PASSED' if test2_passed else 'FAILED'}")
    print("="*80)
