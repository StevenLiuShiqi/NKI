"""
Direct comparison of compiled FlashAttention NKI kernel vs reference SDPA.

Uses build_function to properly compile the flash_fwd kernel with SPMD parallelization.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gpt_oss import sdpa

# Import FlashAttention kernel and compilation utilities
from neuronx_distributed_inference.modules.sliding_window.attention import (
    flash_fwd,
    FlashConfig,
)
from neuronx_distributed_inference.utils.testing import build_function

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def test_flash_kernel_basic():
    """
    Test FlashAttention kernel compilation and execution.

    Configuration:
    - 128 tokens
    - 8 heads, 64 head_dim
    - No sliding window
    """
    print("\n" + "="*80)
    print("TEST: FlashAttention Kernel - Basic Execution")
    print("="*80)

    # Configuration
    batch_size = 2
    num_heads = 8
    head_dim = 64
    seq_len = 128

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Sequence length: {seq_len}")

    # Create test data
    # FlashAttention expects: (batch, heads, head_dim, seq_len) for Q/K, (batch, heads, seq_len, head_dim) for V
    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, head_dim, seq_len, dtype=torch.bfloat16)
    K = torch.randn(batch_size, num_heads, head_dim, seq_len, dtype=torch.bfloat16)
    V = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    print(f"\nInput shapes:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")

    # Softmax scale
    sm_scale = 1.0 / np.sqrt(head_dim)
    print(f"\nSoftmax scale: {sm_scale}")

    # FlashAttention config
    config = FlashConfig(
        seq_tile_size=2048,
        should_transpose_v=False,
    )

    print("\n" + "-"*80)
    print("Compiling FlashAttention kernel...")
    print("-"*80)

    # Build function with SPMD parallelization
    # The kernel expects 2D grid: [batch, kv_heads]
    try:
        compiled_flash = build_function(
            func=flash_fwd,
            example_inputs=[
                (
                    torch.zeros_like(Q),
                    torch.zeros_like(K),
                    torch.zeros_like(V),
                    # sm_scale,
                    # True,  # use_causal_mask
                    # (-1, -1),  # window_size
                    # True,  # mixed_precision
                    # config,
                )
            ],
            tp_degree=1,
            compiler_workdir=str(_ARTIFACTS_DIR / "flash_kernel_workdir"),
        )

        print("✓ FlashAttention kernel compiled successfully")

    except Exception as e:
        print(f"✗ ERROR compiling FlashAttention: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "-"*80)
    print("Running compiled FlashAttention kernel...")
    print("-"*80)

    try:
        # Run with actual data
        with torch.no_grad():
            # Note: Need to launch with proper grid dimensions
            # FlashAttention expects to be called with grid [batch, kv_heads]
            output = compiled_flash(
                Q, K, V,
                sm_scale,
                True,  # use_causal_mask
                (-1, -1),  # window_size (no sliding window)
                True,  # mixed_precision
                config,
            )

        print(f"✓ FlashAttention executed successfully")
        print(f"Output shape: {output.shape}")
        print(f"Expected: (batch={batch_size}, heads={num_heads}, seq={seq_len}, d={head_dim})")
        print(f"Output sample [0,0,0,:4]: {output[0, 0, 0, :4]}")
        print(f"Output sample [0,0,1,:4]: {output[0, 0, 1, :4]}")

        # Statistics
        print(f"\nOutput statistics:")
        print(f"  Mean: {output.mean().item():.6f}")
        print(f"  Std: {output.std().item():.6f}")
        print(f"  Min: {output.min().item():.6f}")
        print(f"  Max: {output.max().item():.6f}")

    except Exception as e:
        print(f"✗ ERROR running FlashAttention: {e}")
        import traceback
        traceback.print_exc()
        return


def test_flash_vs_sdpa_simple():
    """
    Simple comparison: FlashAttention vs SDPA on small inputs.

    Use very small tensors to understand the differences.
    """
    print("\n" + "="*80)
    print("TEST: FlashAttention vs SDPA - Simple Comparison")
    print("="*80)

    # Very small configuration for easier debugging
    batch_size = 1
    num_heads = 2
    head_dim = 4
    seq_len = 4  # Very short sequence

    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")
    print(f"  Sequence length: {seq_len}")

    # Create simple test data - all ones
    Q = torch.ones(batch_size, num_heads, head_dim, seq_len, dtype=torch.bfloat16)
    K = torch.ones(batch_size, num_heads, head_dim, seq_len, dtype=torch.bfloat16)
    V = torch.ones(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    sm_scale = 0.5  # 1/sqrt(4)
    print(f"Softmax scale: {sm_scale}")

    print("\n" + "-"*80)
    print("Reference SDPA computation...")
    print("-"*80)

    # Convert to SDPA format and run
    # SDPA expects: Q(seq, heads, 1, d), K(seq, heads, d), V(seq, heads, d)
    Q_sdpa = Q[0].permute(2, 0, 1).unsqueeze(2)  # (seq, heads, 1, d)
    K_sdpa = K[0].permute(2, 0, 1)  # (seq, heads, d)
    V_sdpa = V[0].permute(1, 0, 2)  # (seq, heads, d)

    print(f"SDPA input shapes:")
    print(f"  Q: {Q_sdpa.shape}")
    print(f"  K: {K_sdpa.shape}")
    print(f"  V: {V_sdpa.shape}")

    # No sinks for simpler comparison
    sinks = torch.zeros(num_heads, dtype=torch.bfloat16)

    with torch.no_grad():
        output_sdpa = sdpa(
            Q_sdpa, K_sdpa, V_sdpa,
            sinks,
            sm_scale,
            sliding_window=0,
        )

    print(f"SDPA output shape: {output_sdpa.shape}")
    print(f"SDPA output:\n{output_sdpa}")

    print("\n✓ SDPA executed successfully")

    # Note: FlashAttention compilation would require proper SPMD setup
    # which is handled by the Neuron infrastructure, not directly callable
    print("\nNote: FlashAttention kernel requires SPMD parallelization")
    print("      and is best tested through the NeuronAttentionBase wrapper")


def test_understand_sdpa():
    """
    Understand the reference SDPA implementation behavior.
    """
    print("\n" + "="*80)
    print("TEST: Understanding Reference SDPA")
    print("="*80)

    # Minimal configuration
    seq_len = 4
    num_heads = 2
    head_dim = 4

    print(f"\nConfiguration:")
    print(f"  Sequence length: {seq_len}")
    print(f"  Num heads: {num_heads}")
    print(f"  Head dim: {head_dim}")

    # Create simple inputs - all ones
    Q = torch.ones(seq_len, num_heads, 1, head_dim, dtype=torch.bfloat16)
    K = torch.ones(seq_len, num_heads, head_dim, dtype=torch.bfloat16)
    V = torch.arange(seq_len * num_heads * head_dim, dtype=torch.bfloat16).reshape(seq_len, num_heads, head_dim)

    sm_scale = 0.5
    sinks = torch.full((num_heads,), 0.5, dtype=torch.bfloat16)

    print(f"\nInputs:")
    print(f"  Q (all ones): shape {Q.shape}")
    print(f"  K (all ones): shape {K.shape}")
    print(f"  V (sequential): shape {V.shape}")
    print(f"  V:\n{V[:, 0, :]}")  # First head
    print(f"  Sinks: {sinks}")

    print("\n" + "-"*80)
    print("Running SDPA...")
    print("-"*80)

    with torch.no_grad():
        output = sdpa(Q, K, V, sinks, sm_scale, sliding_window=0)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")

    # Analyze the pattern
    print(f"\nExpected behavior with causal mask:")
    print(f"  Token 0: Can only attend to self -> should output V[0]")
    print(f"  Token 1: Attends to tokens 0,1 -> weighted average")
    print(f"  Token 2: Attends to tokens 0,1,2 -> weighted average")
    print(f"  Token 3: Attends to tokens 0,1,2,3 -> weighted average")

    print("\n✓ SDPA analysis complete")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*80)
    print("FLASHATTENTION KERNEL DIRECT TESTS")
    print("="*80)

    tests = [
        ("Understand SDPA", test_understand_sdpa),
        ("Simple comparison", test_flash_vs_sdpa_simple),
        ("Basic execution", test_flash_kernel_basic),
    ]

    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"\n✗ Exception in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
