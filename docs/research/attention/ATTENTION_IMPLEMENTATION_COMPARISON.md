# PyTorch vs Neuron Attention Implementation Comparison

## Executive Summary

The reference PyTorch implementation (`gpt_oss.py`) and the Neuron implementation (`model.py`) have significant differences in how they handle the attention mechanism. The PyTorch version uses a custom `sdpa()` function with explicit sliding window and sink token handling, while the Neuron version delegates to `NeuronAttentionBase` from the neuronx-distributed library.

---

## 1. INPUT/OUTPUT SHAPES AND TRANSFORMATIONS

### PyTorch Reference (gpt_oss.py)

**AttentionBlock.forward() Input:**
- `x: torch.Tensor` shape: `(n_tokens, hidden_size)` or `(seq_len, hidden_size)`
  - Single token stream (not batched)

**AttentionBlock.forward() Output:**
- Returns: `torch.Tensor` shape: `(n_tokens, hidden_size)`
- Also passes through `self.out` Linear layer: `(n_heads * head_dim) -> hidden_size`

**Key intermediate shapes in sdpa():**
- `Q`: `(n_tokens, n_heads, q_mult, d_head)` where:
  - `n_tokens` = sequence length
  - `n_heads` = num_key_value_heads (8 in config)
  - `q_mult` = num_attention_heads // num_key_value_heads (64 // 8 = 8)
  - `d_head` = head_dim (64)
  
- `K`: `(n_tokens, n_heads, d_head)` → expanded to `(n_tokens, n_heads, q_mult, d_head)`
- `V`: `(n_tokens, n_heads, d_head)` → expanded to `(n_tokens, n_heads, q_mult, d_head)`

**Output of sdpa():**
- Shape: `(n_tokens, n_heads * q_mult * d_head)` which is `(n_tokens, n_heads * 8 * 64)` = `(n_tokens, 4096)`
  - But actually reshapes from `(n_tokens, n_heads, q_mult, d_head)` → `(n_tokens, n_heads * q_mult * d_head)`

### Neuron Implementation (model.py)

**NeuronGPTOSSAttentionBlock.forward() Input:**
- `hidden_states: torch.Tensor` shape: `(batch_size, seq_len, hidden_size)`
  - **NOTE**: Batched input (different from reference!)
- `position_ids: torch.LongTensor` shape: `(batch_size, seq_len)`

**NeuronGPTOSSAttentionBlock.forward() Output:**
- Returns output from `super().forward()` (NeuronAttentionBase)
- Shape: Expected to be `(batch_size, seq_len, hidden_size)` based on test file

**Key Implementation Detail:**
- Delegates to `NeuronAttentionBase` which handles:
  - QKV projection internally
  - RMSNorm (via rmsnorm parameter)
  - Rotary embeddings
  - Attention computation
  - Output projection
- The actual attention computation is NOT visible in the Neuron code (it's in NeuronAttentionBase)

---

## 2. HOW Q, K, V ARE COMPUTED AND RESHAPED

### PyTorch Reference (gpt_oss.py)

**QKV Projection:**
```python
qkv = self.qkv(t)  # Linear layer: hidden_size -> head_dim * (num_attention_heads + 2 * num_key_value_heads)
# qkv shape: (n_tokens, head_dim * (64 + 2*8)) = (n_tokens, 64 * 80) = (n_tokens, 5120)
```

**Q Extraction and Reshaping:**
```python
q = qkv[:, : self.num_attention_heads * self.head_dim]
# shape: (n_tokens, 64 * 64) = (n_tokens, 4096)

q = q.view(
    -1,
    self.num_key_value_heads,                              # 8
    self.num_attention_heads // self.num_key_value_heads,  # 64 // 8 = 8
    self.head_dim,                                         # 64
)
# Final q shape: (n_tokens, 8, 8, 64)
```

**K Extraction and Reshaping:**
```python
k = qkv[
    :,
    self.num_attention_heads * self.head_dim : 
    (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
]
# shape: (n_tokens, 8 * 64) = (n_tokens, 512)

k = k.view(-1, self.num_key_value_heads, self.head_dim)
# Final k shape: (n_tokens, 8, 64)
```

**V Extraction and Reshaping:**
```python
v = qkv[
    :,
    (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : 
    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
]
# shape: (n_tokens, 8 * 64) = (n_tokens, 512)

v = v.view(-1, self.num_key_value_heads, self.head_dim)
# Final v shape: (n_tokens, 8, 64)
```

**Rotary Embedding Application:**
```python
q, k = self.rope(q, k)
# Both q and k are reshaped to apply RoPE, then restored to their original shapes
# After rope: q shape: (n_tokens, 8, 8, 64), k shape: (n_tokens, 8, 64)
```

### Neuron Implementation (model.py)

**QKV Projection:**
```python
# Handled internally by NeuronAttentionBase
# Not visible in NeuronGPTOSSAttentionBlock code
```

**Key Differences:**
1. The Neuron implementation uses `NeuronAttentionBase.forward()` which:
   - Takes batched inputs `(batch_size, seq_len, hidden_size)`
   - Handles QKV projection internally
   - Applies rotary embeddings internally via the provided `rotary_emb`
   
2. The actual Q, K, V computation and reshaping are NOT visible in the provided code
   - They happen inside the parent class `NeuronAttentionBase`
   - Likely uses standard multi-head and multi-query attention patterns

3. **Critical Issue**: The test file shows:
```python
# Reference expects (n_tokens, hidden_size) - FLAT, NO BATCH
flat_sample = sample.view(-1, hidden_size)
ref_tokens = reference_block(flat_sample)

# Neuron expects (batch_size, seq_len, hidden_size) - BATCHED
direct_output = neuron_block_direct(sample, position_ids)
```

---

## 3. HOW THE ATTENTION MECHANISM WORKS

### PyTorch Reference (gpt_oss.py)

**The sdpa() Function (Scaled Dot-Product Attention):**

```python
def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # Q: (n_tokens, n_heads, q_mult, d_head)
    # K: (n_tokens, n_heads, d_head)
    # V: (n_tokens, n_heads, d_head)
    # S: sink token scores (learned parameters)
    # sm_scale: 1/sqrt(d_head)
    # sliding_window: 128 (for even layers) or 0 (for odd layers)
    
    n_tokens, n_heads, q_mult, d_head = Q.shape
    
    # Expand K and V to match Q's q_mult dimension
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)  # (n_tokens, n_heads, q_mult, d_head)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)  # (n_tokens, n_heads, q_mult, d_head)
    
    # Reshape sinks from (n_heads,) to (n_heads, q_mult, 1, 1) for broadcasting
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    # Final S shape: (n_heads, q_mult, n_tokens, 1)
    
    # Create causal mask (lower triangular)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    # Result: upper triangle is -inf (future tokens), lower triangle is 0 (past tokens)
    
    # Add sliding window mask (if enabled)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
        # Adds -inf to positions older than sliding_window
        # So final mask allows: [current - sliding_window, current]
    
    # Compute attention scores
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    # Q: (n_tokens, n_heads, q_mult, d_head)
    # K: (n_tokens, n_heads, q_mult, d_head)
    # Output: (n_heads, q_mult, n_tokens_q, n_tokens_k)
    
    QK *= sm_scale  # Scale by 1/sqrt(d_head)
    QK += mask[None, None, :, :]  # Add causal/sliding window mask
    
    # CRITICAL: Concatenate sink tokens along last dimension
    QK = torch.cat([QK, S], dim=-1)
    # QK shape becomes: (n_heads, q_mult, n_tokens, n_tokens + 1)
    # The +1 is from the sink token dimension
    
    # Softmax over the last dimension (all keys + sink)
    W = torch.softmax(QK, dim=-1)
    # W shape: (n_heads, q_mult, n_tokens, n_tokens + 1)
    
    # Remove sink attention weights
    W = W[..., :-1]
    # W shape back to: (n_heads, q_mult, n_tokens, n_tokens)
    
    # Apply attention weights to values
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    # W: (n_heads, q_mult, n_tokens, n_tokens)
    # V: (n_tokens, n_heads, q_mult, d_head)
    # Output: (n_tokens, n_heads, q_mult, d_head)
    
    # Reshape back to (n_tokens, n_heads * q_mult * d_head)
    return attn.reshape(n_tokens, -1)
```

**Key Characteristics of PyTorch sdpa():**

1. **Sliding Window Attention:**
   - Only applies to every other layer (`if layer_idx % 2 == 0`)
   - Tokens can attend to: `[current_position - sliding_window, current_position]`
   - Default sliding_window = 128
   - Non-sliding layers (layer_idx odd) have full context

2. **Sink Tokens:**
   - `self.sinks` is a learned parameter: shape `(num_attention_heads,)` = `(64,)`
   - Sinks are concatenated to each attention score matrix AFTER masking
   - They are included in softmax but their weights are discarded
   - This is essentially a technique to improve training stability by having a "sink" position that absorbs unwanted attention

3. **Multi-Query Attention (MQA):**
   - Q has shape `(n_tokens, num_key_value_heads, q_mult, d_head)`
   - K and V have shape `(n_tokens, num_key_value_heads, d_head)`
   - K and V are expanded to match Q's q_mult dimension
   - q_mult = num_attention_heads // num_key_value_heads = 8
   - This means 8 query heads share a single key/value head

4. **Output Shape Transformation:**
   - Attention output: `(n_tokens, n_heads, q_mult, d_head)` = `(n_tokens, 8, 8, 64)`
   - Reshaped to: `(n_tokens, 4096)` for the output projection

### Neuron Implementation (model.py)

**Attention Computation:**
```python
# NeuronGPTOSSAttentionBlock delegates to NeuronAttentionBase
output = super().forward(
    hidden_states=hidden_states,
    position_ids=position_ids,
    attention_mask=attention_mask,
    past_key_value=past_key_value,
    cos_cache=cos_cache,
    sin_cache=sin_cache,
    rotary_position_ids=rotary_position_ids,
    kv_mgr=kv_mgr,
    ...
)
```

**What NeuronAttentionBase does (from initialization):**
1. Uses the provided `rotary_emb` (RotaryEmbedding)
2. Handles QKV projection (via `qkv_proj` and separate `q_proj`, `k_proj`, `v_proj`)
3. Applies rotary embeddings to Q and K
4. Performs attention computation (likely standard scaled dot-product)
5. Handles output projection (via `o_proj`)
6. Includes learned sinks: `learned_sinks_size=1`

**Key Differences from PyTorch:**

1. **No visible sliding window implementation**
   - NeuronAttentionBase may have sliding window support, but it's not parameterized in NeuronGPTOSSAttentionBlock
   - The reference applies sliding_window alternately per layer; Neuron version doesn't show this

2. **Sink token handling:**
   - Neuron creates `learned_sinks_size=1` in NeuronAttentionBase initialization
   - But the exact mechanism is hidden (not in provided code)
   - The conversion function shows it maps sinks to both `learned_sinks.sink` and `tkg_learned_sinks.sink`

3. **Batching:**
   - PyTorch reference: processes single unbatched token sequence `(n_tokens, hidden_size)`
   - Neuron version: processes batched sequences `(batch_size, seq_len, hidden_size)`
   - This is a fundamental architectural difference

4. **KV Cache Management:**
   - Neuron has explicit `kv_mgr: Optional[KVCacheManager]` parameter
   - Reference has no KV cache (seems to be for training/full inference)

---

## 4. KEY DIFFERENCES IN THE FLOW

### Side-by-Side Comparison

| Aspect | PyTorch Reference | Neuron Implementation |
|--------|-------------------|----------------------|
| **Input Shape** | `(n_tokens, hidden_size)` - unbatched | `(batch_size, seq_len, hidden_size)` - batched |
| **RMSNorm** | Commented out in forward | Applied before attention in NeuronGPTOSSBlock |
| **QKV Projection** | Single combined projection + manual split | Delegated to NeuronAttentionBase (sep. q/k/v projs) |
| **Q Shape After Projection** | `(n_tokens, num_attention_heads, q_mult, d_head)` | Unknown (in parent class) |
| **K, V Shape After Projection** | `(n_tokens, num_key_value_heads, d_head)` | Unknown (in parent class) |
| **Rotary Embeddings** | Explicit RotaryEmbedding class, applied by reference | Provided to NeuronAttentionBase, applied internally |
| **Attention Computation** | Custom sdpa() function with sliding window + sinks | NeuronAttentionBase.forward() - internal implementation |
| **Sliding Window** | Alternates per layer (layer_idx % 2 == 0) | Not visible/parameterized in code |
| **Sink Tokens** | Explicit concatenation in attention scores | Parameterized in NeuronAttentionBase initialization |
| **Output Projection** | Self.out linear: `(n_heads*d_head) -> hidden_size` | Delegated to NeuronAttentionBase (o_proj) |
| **Residual Connection** | Not applied in AttentionBlock | Applied in NeuronGPTOSSBlock wrapper |
| **Return Format** | Single tensor | Tuple (from NeuronAttentionBase) |

---

## 5. CRITICAL ISSUES TO FIX

### Issue 1: Input Shape Mismatch
**Problem:** Reference expects `(n_tokens, hidden_size)`, Neuron provides `(batch_size, seq_len, hidden_size)`
- **Line in PyTorch:** `sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)` expects unbatched Q of shape `(n_tokens, ...)`
- **Line in Neuron:** `super().forward(hidden_states=hidden_states, ...)` provides batched input

**Solution:** 
- Either modify Neuron to flatten/reshape to match reference, OR
- Understand if NeuronAttentionBase is designed for batched inputs and verify that's correct

### Issue 2: Missing Sliding Window Implementation
**Problem:** Reference applies sliding_window only to even layers, but Neuron code doesn't show this logic
- **Line in PyTorch:** `self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0`
- **Line in Neuron:** No per-layer sliding window configuration

**Solution:**
- Check if NeuronAttentionBase supports sliding window parameter
- If not, may need to implement sliding window attention in Neuron version

### Issue 3: Sink Token Handling
**Problem:** PyTorch explicitly concatenates and removes sink tokens in softmax; Neuron uses NeuronAttentionBase's internal implementation
- **Line in PyTorch:** `QK = torch.cat([QK, S], dim=-1)` and `W = W[..., :-1]`
- **Line in Neuron:** `learned_sinks_size=1` passed to parent class

**Solution:**
- Verify NeuronAttentionBase's sink implementation matches the PyTorch reference
- If not, may need custom implementation

### Issue 4: QKV Projection Architecture
**Problem:** PyTorch uses single combined projection + manual split; Neuron likely uses separate projections
- **Line in PyTorch:** `self.qkv = torch.nn.Linear(...)`  single projection
- **Line in Neuron:** Delegated to parent class with `qkv_bias=True, o_bias=True`

**Solution:**
- This might be handled correctly by NeuronAttentionBase, but verify the checkpoint conversion handles it properly

### Issue 5: Return Value Type Mismatch
**Problem:** Reference returns tensor, Neuron returns tuple from parent class
- **Line in PyTorch:** `return t` where t is shape `(n_tokens, hidden_size)`
- **Line in Neuron:** Lines 349-386 have duplicate forward definitions, and final one returns `tuple(output)[0]`

**Solution:**
- Remove the duplicate forward definitions (lines 349-386)
- Ensure return value matches expected output shape and type

### Issue 6: Duplicate Forward Definition
**Problem:** NeuronGPTOSSAttentionBlock has two forward() method definitions (lines 313-347 and 349-386)
- This is a Python syntax error - the second one will override the first

**Solution:**
- Remove one of the duplicate definitions
- Keep only the one with proper return handling

---

## 6. DETAILED ATTENTION FLOW COMPARISON

### PyTorch Reference Flow

```
Input: x (n_tokens, hidden_size=2880)
  ↓
qkv = self.qkv(x)  → (n_tokens, 5120)
  ↓
q = extract q from qkv → (n_tokens, 4096)
k = extract k from qkv → (n_tokens, 512)
v = extract v from qkv → (n_tokens, 512)
  ↓
q.view(...) → (n_tokens, 8, 8, 64)
k.view(...) → (n_tokens, 8, 64)
v.view(...) → (n_tokens, 8, 64)
  ↓
q, k = self.rope(q, k)  # Apply rotary embeddings
  ↓
sdpa(q, k, v, sinks, sm_scale, sliding_window):
  ├─ Expand K, V to match Q's q_mult dimension
  ├─ Create causal mask (lower triangular matrix)
  ├─ If sliding_window > 0: Add sliding window constraint
  ├─ QK = einsum("qhmd,khmd->hmqk", Q, K)  → (8, 8, n_tokens, n_tokens)
  ├─ QK *= sm_scale
  ├─ QK += mask
  ├─ QK = cat([QK, sinks], dim=-1)  → (8, 8, n_tokens, n_tokens+1)
  ├─ W = softmax(QK, dim=-1)
  ├─ W = W[..., :-1]  → (8, 8, n_tokens, n_tokens)
  ├─ attn = einsum("hmqk,khmd->qhmd", W, V)  → (n_tokens, 8, 8, 64)
  └─ return reshape to (n_tokens, 4096)
  ↓
t = self.out(t)  → (n_tokens, 2880)
  ↓
return t
```

### Neuron Reference Flow (Conceptual - based on what we can infer)

```
Input: hidden_states (batch_size, seq_len, hidden_size=2880)
       position_ids (batch_size, seq_len)
  ↓
NeuronAttentionBase.forward():
  ├─ Apply RMSNorm (if rmsnorm provided)
  ├─ QKV Projection:
  │  ├─ q_proj(hidden_states) → (batch_size, seq_len, hidden_size)
  │  ├─ k_proj(hidden_states) → (batch_size, seq_len, hidden_size)
  │  └─ v_proj(hidden_states) → (batch_size, seq_len, hidden_size)
  ├─ Reshape to multi-head format
  ├─ Apply rotary embeddings to Q and K
  ├─ Attention computation:
  │  ├─ QK = scaled_dot_product(Q, K)
  │  ├─ Apply attention mask
  │  ├─ Apply sliding window (if supported)
  │  ├─ Apply sink tokens (if supported)
  │  ├─ W = softmax(QK)
  │  └─ attn = W @ V
  ├─ Reshape attention output
  └─ Output projection (o_proj)
  ↓
return (output, present_key_value, cos_cache, sin_cache, ...)
  ↓
Extract output[0] and return
```

---

## 7. WHAT NEEDS TO BE CHANGED IN NEURON VERSION

Based on this comparison, here are the required changes:

### Priority 1: Fix Immediate Errors

1. **Remove Duplicate Forward Method** (Lines 349-386)
   - Keep only one forward() definition
   - File: `/home/ubuntu/NKI/model.py`

2. **Verify Input Shape Handling**
   - Confirm whether NeuronAttentionBase expects batched or unbatched input
   - Update test expectations accordingly

### Priority 2: Match Reference Implementation Details

3. **Sliding Window Attention**
   - Check if NeuronAttentionBase has sliding window parameter
   - If yes: configure it properly in NeuronGPTOSSAttentionBlock.__init__()
   - If no: may need to implement custom attention with sliding window

4. **Sink Token Implementation**
   - Verify `learned_sinks_size=1` in NeuronAttentionBase matches PyTorch behavior
   - Compare the sink concatenation and removal logic
   - May need to adjust or implement custom logic

5. **QKV Projection Architecture**
   - Verify that NeuronAttentionBase's separate q/k/v projections match the reference when loading weights
   - Check the checkpoint conversion is correct (it appears to be in lines 122-129)

### Priority 3: Integration & Testing

6. **Test Output Shapes and Values**
   - Run tests with both reference and Neuron implementations
   - Verify outputs match within floating-point tolerance
   - The test file `/home/ubuntu/NKI/test_attention_simple.py` is set up for this

7. **Handle Batching Correctly**
   - If Neuron is batched and reference is unbatched, ensure proper reshaping
   - Line 42 in test_attention_simple.py: `flat_sample = sample.view(-1, hidden_size)` for reference
   - Ensure Neuron test doesn't do this flattening

---

## 8. CODE REFERENCE LOCATIONS

### PyTorch Reference Implementation
- **File:** `/home/ubuntu/NKI/gpt_oss.py`
- **AttentionBlock class:** Lines 176-247
- **sdpa() function:** Lines 153-173
- **RotaryEmbedding class:** Lines 63-150

### Neuron Implementation
- **File:** `/home/ubuntu/NKI/model.py`
- **NeuronGPTOSSAttentionBlock class:** Lines 291-386 (**DUPLICATE FORWARD METHOD - NEEDS FIX**)
- **NeuronGPTOSSBlock class:** Lines 388-433
- **Config conversion function:** Lines 41-207

### Test Files
- `/home/ubuntu/NKI/test_attention_simple.py` - Main test comparing reference vs Neuron

