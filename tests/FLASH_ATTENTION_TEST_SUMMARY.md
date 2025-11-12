# FlashAttention Kernel Test Summary

## Problem
The original test was failing with the error:
```
Expect spmd grid with 2 dimensions, got 0 instead!
```

## Root Cause
The `flash_fwd` NKI kernel requires SPMD (Single Program Multiple Data) grid dimensions to be specified using bracket notation when calling the kernel. The `build_function` utility wraps kernels in a module that doesn't preserve this SPMD launch syntax.

## Solution
Instead of using `build_function`, directly call the NKI kernel with proper SPMD grid syntax:

```python
output = flash_fwd[batch_size, num_heads](Q, K, V, ...)
```

### Key Requirements

1. **SPMD Grid Dimensions**: The kernel expects a 2D grid `[batch_size, kv_heads]`
   - For MHA (Multi-Head Attention): `kv_heads = num_heads`
   - For GQA (Grouped Query Attention): `kv_heads = num_key_value_heads`

2. **XLA Tensors**: All input tensors must be XLA tensors (on Neuron device)
   ```python
   device = xm.xla_device()
   Q = torch.randn(..., device=device)
   ```

3. **Sequence Length**: Must be divisible by `seq_tile_size` (default 2048)
   - Minimum supported: 512 (using `FlashConfig(seq_tile_size=512)`)
   - Recommended: 2048 for better performance

4. **Tensor Shapes**:
   - Q: `(batch, heads, head_dim, seq_len)`
   - K: `(batch, kv_heads, head_dim, seq_len)`
   - V: `(batch, kv_heads, seq_len, head_dim)` when `should_transpose_v=False`
   - Output: `(batch, heads, seq_len, head_dim)`

## Known Issues

### Library Bug
There's a typo in the FlashAttention library at line 420 of `attention.py`:
```python
casual_mask = True  # Should be: causal_mask = True
```

This causes an `UnboundLocalError` when:
- `use_causal_mask=False` AND
- `sliding_window <= 0`

**Workaround**: Use `sliding_window > 0` (e.g., `window_size=(seq_len, -1)`) to avoid this code path.

## Test Results

The test successfully executes FlashAttention with:
- ✓ SPMD grid properly specified
- ✓ XLA tensors on Neuron device
- ✓ Sequence length 2048 (divisible by tile size)
- ✓ Output shape matches expectations
- ✓ No NaN or Inf values in output
- ✓ Compilation and execution complete successfully

## Example Usage

```python
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed_inference.modules.sliding_window.attention import (
    flash_fwd,
    FlashConfig,
)

# Configuration
batch_size = 2
num_heads = 8
head_dim = 64
seq_len = 2048

# Get XLA device
device = xm.xla_device()

# Create input tensors
Q = torch.randn(batch_size, num_heads, head_dim, seq_len,
                dtype=torch.bfloat16, device=device)
K = torch.randn(batch_size, num_heads, head_dim, seq_len,
                dtype=torch.bfloat16, device=device)
V = torch.randn(batch_size, num_heads, seq_len, head_dim,
                dtype=torch.bfloat16, device=device)

# Configure kernel
config = FlashConfig(seq_tile_size=2048, should_transpose_v=False)
sm_scale = 1.0 / (head_dim ** 0.5)

# Call kernel with SPMD grid [batch_size, num_heads]
output = flash_fwd[batch_size, num_heads](
    Q, K, V,
    softmax_scale=sm_scale,
    use_causal_mask=True,
    window_size=(seq_len, -1),  # Large window to effectively disable sliding window
    mixed_precision=True,
    config=config,
)

# Output shape: (batch_size, num_heads, seq_len, head_dim)
```

## Files Modified
- [tests/test_flash_kernel_compiled.py](tests/test_flash_kernel_compiled.py) - Updated to use SPMD grid syntax and XLA tensors

## References
- NKI Documentation: SPMD parallelization requires bracket notation for grid dimensions
- FlashAttention kernel: [neuronx_distributed_inference/modules/sliding_window/attention.py](https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/modules/sliding_window/attention.py)
- Example usage in production: See `NeuronAttentionBase` in `attention_base.py`
