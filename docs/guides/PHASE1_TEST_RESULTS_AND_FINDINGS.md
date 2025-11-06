# Phase 1 Test Results and Findings

## Executive Summary

Phase 1 isolated component testing has been implemented and executed for the attention mechanism. Tests successfully compile and run on AWS Trainium hardware, but numerical outputs do not match between the reference PyTorch implementation (`gpt_oss.py`) and the Neuron FlashAttention implementation (`model.py`).

**Status**: ❌ Tests FAILING - Numerical mismatch detected
**Root Cause**: Under investigation - likely architectural differences in attention computation

---

## Test Implementation

### Tests Created

**File**: [`tests/test_attention_phase1.py`](tests/test_attention_phase1.py)

1. **Test 1.1**: Basic Attention Without Special Features
   - Configuration: 4 tokens, 2 heads, 4 head_dim, no sliding window
   - Status: ❌ FAIL
   - Error: 50% elements mismatched, max error 1.625 (14.4% relative)

2. **Test 1.2**: Causal Masking
   - Configuration: 8 tokens, 4 heads, causal mask enabled
   - Status: ❌ FAIL
   - Error: 37.5% elements mismatched, max error 11.750 (25.6% relative)

3. **Test 1.3**: Sliding Window Attention
   - Configuration: 16 tokens, 4 heads, window=4
   - Status: ❌ FAIL
   - Error: 62.5% elements mismatched, max error 11.750 (25.6% relative)

4. **Test 1.4**: Learned Sinks
   - Configuration: 8 tokens, 2 heads, sink values initialized
   - Status: ❌ FAIL
   - Error: 37.5% elements mismatched, max error 3.125 (23.8% relative)

---

## Key Findings

### 1. SDPA Implementation is Correct

**Verification**: [`test_sdpa_simple.py`](test_sdpa_simple.py)

The standalone `sdpa()` function from `gpt_oss.py` has been verified to work correctly:
- ✅ Correct Q@K^T computation
- ✅ Proper scaling by `1/sqrt(head_dim)`
- ✅ Correct causal masking
- ✅ Learned sinks properly integrated into softmax
- ✅ Output matches manual computation exactly

```
Manual == SDPA: True
Max difference: 0.0
```

### 2. Reference Implementation Produces Expected Outputs

**File**: [`debug_attention_mismatch.py`](debug_attention_mismatch.py)

The reference `AttentionBlock` from `gpt_oss.py` processes inputs correctly:
- Input: `torch.Size([1, 4, 8])` (batch=1, seq=4, hidden=8)
- QKV projection works: `torch.Size([4, 24])` → Q/K/V split
- RoPE applied: Max diff ~5.7 (expected for rotary embeddings)
- SDPA output: Varies per token based on causal attention
- Final output after O-projection: `torch.Size([1, 4, 8])`

**Sample output**:
```
tensor([[ 8.5000,  8.5000,  8.5000,  8.5000,  8.5000,  8.5000,  8.5000,  8.5000],
        [11.3125, 11.3125, 11.3125, 11.3125, 11.3125, 11.3125, 11.3125, 11.3125],
        [10.0625, 10.0625, 10.0625, 10.0625, 10.0625, 10.0625, 10.0625, 10.0625],
        [11.2500, 11.2500, 11.2500, 11.2500, 11.2500, 11.2500, 11.2500, 11.2500]])
```

### 3. Neuron Implementation Differences

**Architecture**: `NeuronGPTOSSAttentionBlock` extends `NeuronAttentionBase`

The Neuron implementation uses:
- **FlashAttention kernels** via NKI (Neuron Kernel Interface)
- **Tiled computation**: 128-token query tiles, 2048-token KV tiles
- **Online softmax**: Incrementally updates max/sum statistics
- **Mixed precision**: BF16 matmul with FP32 accumulation
- **Hardware-optimized**: Custom kernels for AWS Trainium/Inferentia

### 4. Potential Sources of Mismatch

#### A. Input/Output Format Differences
- **Reference**: Expects flat tokens `(batch*seq, hidden)`
- **Neuron**: Expects 3D tensors `(batch, seq, hidden)` + position_ids
- Tests handle reshaping, but may introduce subtle issues

#### B. Weight Initialization
Both implementations initialize with constant value `0.5`:
- Reference: Direct `param.fill_(0.5)`
- Neuron: `_initialize_weights()` method fills all params

However, Neuron has additional parameters:
- `learned_sinks` for CTE (Context Time Encoding)
- `tkg_learned_sinks` for TKG (Token Generation)
- Both are initialized from same source `sinks` tensor

####  C. Attention Computation Strategy
- **Reference SDPA**: Computes full `QK^T` matrix, applies mask, softmax, then `@V`
- **Neuron FlashAttention**: Tiled computation with online softmax
  - Computes attention in blocks to save memory
  - Uses `_flash_attention_core()` from `sliding_window/attention.py`
  - Different numerical precision characteristics

#### D. RoPE Implementation
Both use rotary embeddings, but:
- **Reference**: Custom `RotaryEmbedding` class in `gpt_oss.py`
- **Neuron**: Uses `neuronx_distributed_inference.modules.attention.utils.RotaryEmbedding`

May have slight numerical differences in frequency computation or application.

#### E. Sink Token Integration
- **Reference**: Sinks concatenated to attention scores before softmax, then removed from weights
- **Neuron**: Sinks passed to Flash kernel via `sink` parameter
  - Kernel handles sink integration internally
  - May use different numerical approach

---

## Test Infrastructure

### Successful Components

1. **Test Framework**: Using `neuronx_distributed_inference.utils.testing`
   - `build_module()`: Successfully compiles models for Trainium
   - `validate_accuracy()`: Compares outputs with configurable tolerances

2. **Compilation**: All models compile successfully
   - Compiler status: PASS for all test configurations
   - NEFFs (Neuron Executable File Format) generated correctly

3. **Execution**: Models run on Trainium without errors
   - No runtime crashes
   - Outputs have correct shapes
   - Values are reasonable (not NaN/Inf)

### Test Configuration Helper

```python
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
```

---

## Recommendations

### Immediate Next Steps

1. **Isolate the SDPA vs FlashAttention Difference**
   - Create a minimal test that calls Neuron's FlashAttention kernel directly
   - Compare with reference SDPA on identical Q/K/V inputs
   - Determine if difference is in:
     - The attention computation itself
     - Weight projections (QKV, O)
     - RoPE application

2. **Check Weight Loading**
   - Verify that `_initialize_weights(0.5)` actually sets all parameters to 0.5
   - Print Neuron model parameters after initialization
   - Compare parameter names and shapes between reference and Neuron

3. **Debug FlashAttention Kernel**
   - **File**: `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/modules/sliding_window/attention.py`
   - Add logging to `flash_fwd()` and `_flash_attention_core()`
   - Compare intermediate QK, softmax, and output values

4. **Verify RoPE Equivalence**
   - Test RoPE implementations separately
   - Ensure identical frequency generation
   - Check cos/sin cache computation

### Testing Strategy Adjustments

1. **Increase Tolerance** (short-term workaround)
   - FlashAttention may have acceptable numerical differences due to tiling
   - Typical tolerance for BF16: `rtol=1e-2, atol=1e-4`
   - Current: `rtol=0.016, atol=1e-5` (too strict?)

2. **Unit Test Each Component**
   Before testing full attention:
   - QKV projection alone
   - RoPE alone
   - Attention scores (QK^T) alone
   - Softmax alone
   - Output projection alone

3. **Reference Implementation on Neuron**
   - Port `sdpa()` to run on Trainium (without FlashAttention)
   - Compare Neuron FlashAttention vs Neuron SDPA
   - Isolates hardware effects from algorithm differences

### Long-term Improvements

1. **Visualization Tools**
   - Plot attention patterns from both implementations
   - Visualize where differences occur (which tokens, which heads)

2. **Numerical Analysis**
   - Analyze error distribution (uniform vs concentrated)
   - Check if errors accumulate or are bounded

3. **Performance Benchmarking**
   - Once accuracy is validated, measure throughput
   - Compare latency between reference and Neuron

---

## Code Artifacts

### Test Files
- [`tests/test_attention_phase1.py`](tests/test_attention_phase1.py) - Main test suite
- [`test_sdpa_simple.py`](test_sdpa_simple.py) - SDPA verification
- [`debug_attention_mismatch.py`](debug_attention_mismatch.py) - Detailed debugging

### Model Files
- [`gpt_oss.py`](gpt_oss.py) - Reference PyTorch implementation
- [`model.py`](model.py) - Neuron implementation
- [`moe_classes.py`](moe_classes.py) - MoE expert layers

### Test Utilities
- [`tests/test_utils.py`](tests/test_utils.py) - Configuration helpers
- [`tests/test_attention.py`](tests/test_attention.py) - Original integration test

---

## Environment

- **Hardware**: AWS Trainium (trn1 instance)
- **Software**:
  - `neuronx-cc`: 2.21.33363.0
  - PyTorch: 2.8 with Neuron extensions
  - Python: 3.10

---

## Conclusion

Phase 1 testing infrastructure is complete and functional. The tests successfully compile models for Trainium and execute them, but reveal numerical differences between the reference and Neuron implementations. These differences are likely due to:

1. **Algorithmic differences**: FlashAttention's tiled computation vs standard SDPA
2. **Precision**: Mixed-precision (BF16/FP32) in Neuron vs BF16-only in reference
3. **Implementation details**: Sink integration, RoPE application, or weight handling

**Next Priority**: Isolate which component(s) cause the mismatch by testing each sub-module independently.

---

## Plan Document Reference

This document implements **Phase 1** of the testing plan described in the main planning document. See the initial conversation for the complete 4-phase testing strategy.

**Phase 1**: ✅ Implemented, ❌ Tests Failing
**Phase 2**: Pending (Combined features testing)
**Phase 3**: Pending (Edge cases & numerical precision)
**Phase 4**: Pending (Integration testing)
