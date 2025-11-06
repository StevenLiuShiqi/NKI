# Attention Implementation Comparison - Executive Summary

## Quick Overview

| Aspect | PyTorch Reference | Neuron Implementation |
|--------|-------------------|----------------------|
| **Input Shape** | `(n_tokens, 2880)` - unbatched | `(batch_size, seq_len, 2880)` - batched |
| **QKV Projection** | Single combined projection | Separate q, k, v projections |
| **Q Shape** | `(n_tokens, 8, 8, 64)` | Unknown (delegated to parent) |
| **K Shape** | `(n_tokens, 8, 64)` | Unknown (delegated to parent) |
| **V Shape** | `(n_tokens, 8, 64)` | Unknown (delegated to parent) |
| **Attention Function** | Custom `sdpa()` | `NeuronAttentionBase.forward()` |
| **Sliding Window** | Alternates per layer (even=128, odd=0) | Not visible |
| **Sink Tokens** | Explicitly concatenated/removed in softmax | Parameterized as `learned_sinks_size=1` |
| **Output Projection** | `self.out` linear layer | Delegated to parent |
| **Return Type** | Single tensor | Tuple from parent class |
| **Residual Connection** | Not applied in attention block | Applied in wrapper block |

---

## Critical Findings

### 1. Sink Token Mechanism

**PyTorch:**
```python
S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)  # (8, 8, n_tokens, 1)
QK = torch.cat([QK, S], dim=-1)  # Add sink to attention scores
W = torch.softmax(QK, dim=-1)    # Softmax includes sink
W = W[..., :-1]                  # Remove sink weights before applying to V
```

**Result:** Sinks affect softmax normalization but don't contribute to output values.

**Neuron:** Uses `learned_sinks_size=1` but exact mechanism unknown.

**Question:** Does NeuronAttentionBase implement the same behavior?

### 2. Sliding Window Attention

**PyTorch:**
```python
self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
# Even layers: sliding_window = 128
# Odd layers: sliding_window = 0 (full context)

if sliding_window > 0:
    mask += torch.tril(
        mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
    )
```

**Neuron:** No per-layer sliding window configuration visible in code.

**Question:** How does NeuronAttentionBase handle sliding windows? Is it configurable per layer?

### 3. Multi-Query Attention (MQA)

**PyTorch:**
```python
Q: (n_tokens, 8, 8, 64)  # 64 heads total (8 key_value heads * 8 q_mult)
K: (n_tokens, 8, 64)     # 8 key_value heads
V: (n_tokens, 8, 64)     # 8 key_value heads

# Expand K and V to match Q's q_mult dimension
K = K[:, :, None, :].expand(-1, -1, q_mult, -1)  # (n_tokens, 8, 8, 64)
V = V[:, :, None, :].expand(-1, -1, q_mult, -1)  # (n_tokens, 8, 8, 64)

# Now all three have same shape, 8 query heads share each key_value head
```

**Neuron:** Likely handles this in NeuronAttentionBase, but expansion logic not visible.

---

## Input/Output Shape Differences

### PyTorch Reference Flow

```
Input:  (n_tokens, 2880)
          ↓
        qkv projection
          ↓
        extract q, k, v
          ↓
        reshape to multi-head
          ↓
        apply rotary embeddings
          ↓
        sdpa (attention computation)
          ↓
        reshape attention output
          ↓
        output projection
          ↓
Output: (n_tokens, 2880)
```

### Neuron Implementation Flow

```
Input:  (batch_size, seq_len, 2880)
          ↓
        NeuronAttentionBase.forward()
          ├─ qkv projection (separate q, k, v)
          ├─ reshape to multi-head
          ├─ apply rotary embeddings
          ├─ attention computation
          ├─ reshape attention output
          └─ output projection
          ↓
Output: tuple from parent class
          ↓
        extract [0] from tuple
          ↓
Output: (batch_size, seq_len, 2880)
```

**Key Difference:** Neuron is batched, PyTorch is unbatched.

---

## Code Location Reference

### PyTorch Reference (gpt_oss.py)

| Component | Lines | Content |
|-----------|-------|---------|
| RMSNorm | 32-47 | Layer normalization implementation |
| RotaryEmbedding | 63-150 | Rotary position embeddings with YaRN scaling |
| sdpa() | 153-173 | Scaled dot-product attention with sinks and sliding window |
| AttentionBlock.__init__ | 176-215 | Initialization: sliding_window, sinks, qkv, output projections |
| AttentionBlock.forward | 217-247 | Forward pass: qkv extraction, rope, sdpa, output projection |

### Neuron Implementation (model.py)

| Component | Lines | Content |
|-----------|-------|---------|
| NeuronGPTOSSConfig | 210-213 | Configuration with fused_qkv=False |
| GPTOSSInferenceConfig | 215-226 | Inference configuration class |
| NeuronGPTOSSAttentionBlock.__init__ | 292-310 | Initialization: delegates to NeuronAttentionBase |
| NeuronGPTOSSAttentionBlock.forward | 313-386 | **DUPLICATE DEFINITIONS - NEEDS FIX** |
| convert_gptoss_to_neuron_state_dict | 41-207 | Checkpoint conversion function |

---

## Key Configuration Parameters

Both implementations use the same base configuration:

```python
ModelConfig / InferenceConfig:
  hidden_size: 2880
  num_attention_heads: 64
  num_key_value_heads: 8
  head_dim: 64
  sliding_window: 128
  rope_theta: 150000.0
  rope_scaling_factor: 32.0
  rope_ntk_alpha: 1.0
  rope_ntk_beta: 32.0
```

**Calculations:**
- Q dimension: 64 * 64 = 4096
- K dimension: 8 * 64 = 512
- V dimension: 8 * 64 = 512
- QKV combined: 4096 + 512 + 512 = 5120
- Q_mult: 64 / 8 = 8 (8 query heads per key_value head)

---

## Verification Checklist

Before modifying Neuron implementation, verify:

- [ ] NeuronAttentionBase handles batched input correctly
- [ ] Separate q/k/v projections produce correct shapes
- [ ] K and V expansion for MQA is implemented
- [ ] Sliding window attention is supported
  - [ ] Can be configured per layer
  - [ ] Produces same mask as PyTorch reference
- [ ] Sink tokens mechanism matches PyTorch
  - [ ] Sinks reshape to (n_heads, q_mult, n_tokens, 1)
  - [ ] Sinks are concatenated along KEY dimension
  - [ ] Softmax includes sinks, weights are discarded
- [ ] Output shape matches expected (batch_size, seq_len, hidden_size)
- [ ] Checkpoint conversion correctly maps weights

---

## Issues to Fix

### Issue 1: Duplicate Forward Method (CRITICAL)
**File:** `/home/ubuntu/NKI/model.py`
**Lines:** 313-347 and 349-386
**Problem:** Two identical forward() definitions; second overrides first
**Fix:** Remove one of the definitions or merge them

### Issue 2: Missing Sliding Window Per-Layer Configuration
**File:** `/home/ubuntu/NKI/model.py`
**Issue:** NeuronGPTOSSAttentionBlock doesn't take `layer_idx` parameter
**Required:** Must pass layer_idx to control sliding_window per layer
**Fix:** Add layer_idx parameter and configure sliding_window based on it

### Issue 3: Input/Output Shape Expectations
**File:** `/home/ubuntu/NKI/test_attention_simple.py`
**Problem:** Test expects different input shapes for reference vs Neuron
**Required:** Clarify whether Neuron should be batched or unbatched
**Related:** Line 42-43 flatten reference input but not Neuron input

### Issue 4: Return Value Handling
**File:** `/home/ubuntu/NKI/model.py`
**Lines:** 349-386
**Problem:** Extracts first element from tuple; unclear what parent returns
**Required:** Document NeuronAttentionBase return format

---

## Detailed Attention Score Computation

### PyTorch Reference (Step-by-Step)

```python
def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    n_tokens, n_heads, q_mult, d_head = Q.shape
    # n_tokens: sequence length
    # n_heads: 8 (num_key_value_heads)
    # q_mult: 8 (num_attention_heads // num_key_value_heads)
    # d_head: 64
    
    # Step 1: Expand K and V to match Q's q_mult dimension
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    # K: (n_tokens, 8, 64) -> (n_tokens, 8, 8, 64)
    # V: (n_tokens, 8, 64) -> (n_tokens, 8, 8, 64)
    
    # Step 2: Reshape sinks
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    # S: (64,) -> (8, 8, n_tokens, 1)
    
    # Step 3: Create causal mask
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    # Upper triangle = -inf (can't attend to future)
    # Lower triangle = 0 (can attend to past)
    
    # Step 4: Add sliding window mask
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
        # Adds -inf below diagonal-sliding_window
        # Result: only attend to [current-sliding_window, current]
    
    # Step 5: Compute attention scores
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    # Q: (n_tokens, 8, 8, 64)
    # K: (n_tokens, 8, 8, 64)
    # QK: (8, 8, n_tokens, n_tokens)
    # QK[h, m, i, j] = sum over d of Q[i, h, m, d] * K[j, h, m, d]
    
    # Step 6: Scale scores
    QK *= sm_scale  # 0.125 = 1/sqrt(64)
    
    # Step 7: Apply mask
    QK += mask[None, None, :, :]
    # Broadcasts mask to (1, 1, n_tokens, n_tokens)
    
    # Step 8: CRITICAL - Concatenate sink tokens
    QK = torch.cat([QK, S], dim=-1)
    # QK: (8, 8, n_tokens, n_tokens) + (8, 8, n_tokens, 1)
    #   = (8, 8, n_tokens, n_tokens+1)
    
    # Step 9: Compute attention weights
    W = torch.softmax(QK, dim=-1)
    # W: (8, 8, n_tokens, n_tokens+1)
    # Each row sums to 1.0 (includes sink)
    
    # Step 10: Remove sink weights
    W = W[..., :-1]
    # W: (8, 8, n_tokens, n_tokens)
    # Sink affected softmax normalization, but weights discarded
    
    # Step 11: Apply weights to values
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    # W: (8, 8, n_tokens, n_tokens)
    # V: (n_tokens, 8, 8, 64)
    # attn: (n_tokens, 8, 8, 64)
    
    # Step 12: Reshape
    return attn.reshape(n_tokens, -1)
    # Output: (n_tokens, 4096) = (n_tokens, 8*8*64)
```

### Neuron Implementation
All of the above happens inside `NeuronAttentionBase.forward()` - implementation not visible.

---

## Testing Strategy

1. **Create minimal test case** comparing reference and Neuron outputs
   - Use same initialization values
   - Use same input
   - Compare output values and shapes

2. **Test individual components**
   - Q, K, V shapes after projection
   - Attention score shapes
   - Softmax output shapes
   - Final output shape

3. **Test sliding window behavior**
   - Verify masking is correct
   - Test per-layer configuration

4. **Test sink token behavior**
   - Verify sinks affect softmax normalization
   - Verify sink weights are discarded
   - Check final output doesn't change

---

## Next Steps

1. **Remove duplicate forward() method** in NeuronGPTOSSAttentionBlock
2. **Add layer_idx parameter** to NeuronGPTOSSAttentionBlock
3. **Configure sliding_window** based on layer_idx
4. **Verify NeuronAttentionBase implementation** against PyTorch reference
5. **Run test_attention_simple.py** to compare outputs
6. **Fix any differences** in:
   - Mask application
   - Sink token handling
   - Output shapes
   - Attention computation order

