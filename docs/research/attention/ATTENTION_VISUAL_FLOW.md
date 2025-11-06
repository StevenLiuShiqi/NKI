# Visual Flow Diagrams: PyTorch vs Neuron Attention

## PyTorch Reference Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│ INPUT: x (n_tokens=10, hidden_size=2880)                           │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ QKV PROJECTION                                                      │
│ self.qkv: Linear(2880 -> 5120)                                      │
│ 5120 = head_dim * (num_attention_heads + 2*num_key_value_heads)    │
│       = 64 * (64 + 16)                                              │
└──────────────────────┬──────────────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────┴──────────────┬──────────────┐
        │                             │              │
        ▼                             ▼              ▼
┌──────────────────┐        ┌──────────────────┐  ┌──────────────────┐
│ EXTRACT Q        │        │ EXTRACT K        │  │ EXTRACT V        │
│ (10, 4096)       │        │ (10, 512)        │  │ (10, 512)        │
└────────┬─────────┘        └────────┬─────────┘  └────────┬─────────┘
         │                           │                     │
         ▼                           ▼                     ▼
┌──────────────────┐        ┌──────────────────┐  ┌──────────────────┐
│ RESHAPE Q        │        │ RESHAPE K        │  │ RESHAPE V        │
│ (10, 8, 8, 64)   │        │ (10, 8, 64)      │  │ (10, 8, 64)      │
│ q_mult=8         │        │ num_kv_heads=8   │  │ num_kv_heads=8   │
└────────┬─────────┘        └────────┬─────────┘  └────────┬─────────┘
         │                           │                     │
         └──────────────┬────────────┴─────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ APPLY ROTARY EMBEDDINGS                                             │
│ q, k = self.rope(q, k)                                              │
│ Q: (10, 8, 8, 64) ─┐                                                │
│ K: (10, 8, 64)    ├─> RotaryEmbedding with YaRN scaling             │
└─────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SCALED DOT-PRODUCT ATTENTION (sdpa)                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Expand K and V for MQA                                           │
│    K: (10, 8, 64) ──expand──> (10, 8, 8, 64)                       │
│    V: (10, 8, 64) ──expand──> (10, 8, 8, 64)                       │
│                                                                      │
│ 2. Reshape sinks for broadcasting                                   │
│    S: (64,) ──reshape──> (8, 8, 10, 1)                             │
│                                                                      │
│ 3. Create attention mask                                            │
│    mask = triu(-inf, diagonal=1) + tril(-inf, diagonal=-128)       │
│    Result: causal mask with sliding_window=128                     │
│                                                                      │
│ 4. Compute attention scores                                         │
│    QK = einsum("qhmd,khmd->hmqk", Q, K)                            │
│    QK: (8, 8, 10, 10)                                               │
│                                                                      │
│ 5. Scale by 1/sqrt(head_dim)                                        │
│    QK *= 0.125                                                      │
│                                                                      │
│ 6. Apply causal/sliding window mask                                 │
│    QK += mask[None, None, :, :]                                     │
│                                                                      │
│ 7. CRITICAL: Concatenate sink tokens                                │
│    QK = cat([QK, S], dim=-1)  ──> (8, 8, 10, 11)                   │
│    Sink affects softmax but not output!                             │
│                                                                      │
│ 8. Softmax over keys (including sink)                               │
│    W = softmax(QK, dim=-1)  ──> (8, 8, 10, 11)                     │
│                                                                      │
│ 9. Remove sink attention weights                                    │
│    W = W[..., :-1]  ──> (8, 8, 10, 10)                             │
│                                                                      │
│ 10. Apply weights to values                                         │
│     attn = einsum("hmqk,khmd->qhmd", W, V)                         │
│     attn: (10, 8, 8, 64)                                            │
│                                                                      │
│ 11. Reshape for output projection                                   │
│     attn.reshape(10, -1) ──> (10, 4096)                            │
└─────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ OUTPUT PROJECTION                                                   │
│ self.out: Linear(4096 -> 2880)                                      │
│ Output: (10, 2880)                                                  │
└─────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ RETURN: (10, 2880)                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Neuron Implementation Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│ INPUT: hidden_states (batch_size=2, seq_len=10, hidden_size=2880)   │
│        position_ids (batch_size=2, seq_len=10)                      │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ NeuronAttentionBase.forward()                                        │
│ (Implementation details hidden)                                      │
├──────────────────────────────────────────────────────────────────────┤
│ Likely does:                                                         │
│ 1. Apply RMSNorm (if provided)                                       │
│ 2. QKV Projection (separate q, k, v)                                │
│ 3. Reshape to multi-head format                                      │
│ 4. Apply rotary embeddings                                           │
│ 5. Attention computation (mechanism unknown)                         │
│ 6. Apply attention mask                                              │
│ 7. Reshape attention output                                          │
│ 8. Output projection                                                 │
│                                                                      │
│ Unknowns:                                                            │
│ - Does it expand K, V for MQA?                                       │
│ - How are sink tokens handled?                                       │
│ - Is sliding window supported?                                       │
│ - What is the exact return format?                                   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ RETURN: output (tuple from parent class)                            │
│ Likely: (attention_output, present_kv, cos_cache, sin_cache, ...)  │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ EXTRACT FIRST ELEMENT                                                │
│ return tuple(output)[0]                                              │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│ RETURN: attention_output (2, 10, 2880)                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Sink Token Mechanism - Detailed

### PyTorch Reference

```
┌─────────────────────────────────────────────────────────────────┐
│ SINK TOKEN PARAMETER                                            │
│ self.sinks: Parameter(shape=(num_attention_heads,)) = (64,)   │
│ self.sinks.shape = (64,)                                        │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ RESHAPE FOR MQA BROADCASTING                                    │
│ S = S.reshape(n_heads, q_mult, 1, 1)                           │
│        .expand(-1, -1, n_tokens, -1)                           │
│                                                                 │
│ S: (64,) ──reshape──> (8, 8, 1, 1)                             │
│           ──expand──> (8, 8, 10, 1)                            │
│                                                                 │
│ Dimensions:                                                     │
│ - 8: num_key_value_heads                                        │
│ - 8: q_mult (num_attention_heads // num_key_value_heads)       │
│ - 10: n_tokens                                                  │
│ - 1: sink dimension (one sink per position)                    │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ CONCATENATE TO ATTENTION SCORES                                 │
│ QK: (8, 8, 10, 10)  [attention between tokens]                │
│ S:  (8, 8, 10, 1)   [sink tokens]                              │
│                                                                 │
│ QK = torch.cat([QK, S], dim=-1)                                │
│                                                                 │
│ Result: QK (8, 8, 10, 11)                                       │
│ ┌────────────────────────────────────────────────────────┐     │
│ │ Each query can attend to:                              │     │
│ │ - 10 regular key positions [i=0..9]                    │     │
│ │ - 1 sink position [i=10]                               │     │
│ └────────────────────────────────────────────────────────┘     │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ SOFTMAX OVER ALL POSITIONS (INCLUDING SINK)                     │
│ W = torch.softmax(QK, dim=-1)                                   │
│                                                                 │
│ For each query position:                                        │
│ sum(W[h, m, q, :]) = 1.0  (includes sink weight)               │
│                                                                 │
│ Softmax normalizes over [0..10] = 11 positions                 │
│                                                                 │
│ Result: W (8, 8, 10, 11)                                        │
│ with sum over last dim = 1.0                                    │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ REMOVE SINK ATTENTION WEIGHTS                                   │
│ W = W[..., :-1]                                                 │
│                                                                 │
│ W: (8, 8, 10, 11) ──slice off last dim──> (8, 8, 10, 10)       │
│                                                                 │
│ Keeps regular attention weights, discards sink weights         │
│ BUT sink affected the softmax normalization!                   │
└────────────┬────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│ APPLY TO VALUES (NO SINK DIMENSION)                             │
│ V: (10, 8, 64)  [no sink in values]                            │
│ W: (8, 8, 10, 10)  [sink already removed]                      │
│                                                                 │
│ attn = einsum("hmqk,khmd->qhmd", W, V)                         │
│ Result: (10, 8, 8, 64)                                          │
│                                                                 │
│ The sink influenced softmax normalization but:                  │
│ - No sink values in output                                      │
│ - Attention weights are only to real tokens                     │
│ - But normalization was affected by sink                        │
└─────────────────────────────────────────────────────────────────┘
```

### Effect of Sinks

```
WITHOUT SINKS:
Softmax normalizes over 10 positions (regular tokens)
W_without_sink[h, m, q, :] sums to 1.0 over 10 positions

WITH SINKS:
Softmax normalizes over 11 positions (10 tokens + 1 sink)
W_with_sink[h, m, q, 0:10] will be smaller
W_with_sink[h, m, q, 0:10] sums to < 1.0 (remainder is on sink)
Then we discard the sink: W[..., :-1]
But the effect remains: attention is more distributed!

INTUITION:
- Sinks act as a "absorber" of attention probability
- Some attention that would go to tokens now goes to sink
- This can stabilize training and improve attention patterns
```

---

## Sliding Window Mask Comparison

### Full Context (sliding_window=0)

```
           Attend to:
Query 0: [0]
Query 1: [0, 1]
Query 2: [0, 1, 2]
Query 3: [0, 1, 2, 3]
...
Query 9: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Mask Pattern (10 tokens):
┌─────────────────────┐
│ 0  -∞ -∞ -∞ ... -∞  │  Query 0
│ 0  0  -∞ -∞ ... -∞  │  Query 1
│ 0  0  0  -∞ ... -∞  │  Query 2
│ 0  0  0  0  ... -∞  │  Query 3
│ ...                  │
│ 0  0  0  0  ... 0   │  Query 9
└─────────────────────┘
  (Causal: can attend to current and all past)
```

### Sliding Window (sliding_window=128, but seq_len=10)

```
With window size 128 on 10 tokens:
- All tokens are within the window
- Same as full context in this case

But with longer sequence (seq_len=256, window=128):
Query 128: [0..127, 128]  (128 tokens in window)
Query 129: [1..128, 129]  (sliding window slides)
Query 255: [127..255]     (128 tokens in window)

Mask Pattern for window=128:
┌─────────────────────────────────┐
│ 0  -∞ -∞ ... -∞                │  Query 0
│ 0  0  -∞ ... -∞                │  Query 1
│ ...                             │
│ 0  0  0  ... 0                 │  Query 127
│ 0  0  0  ... 0   -∞            │  Query 128
│ -∞ 0  0  ... 0   -∞            │  Query 129
│                                │
│ -∞ -∞ ... 0  0   0   -∞        │  Query 255
└─────────────────────────────────┘
  (Sliding window: can attend to [pos-128, pos])
```

### Per-Layer Configuration in PyTorch

```
Layer 0 (even): sliding_window = 128
┌─────────────┐
│ Token  Attn │
│ 0      [0]  │
│ 1      [0:2]│
│ 2      [0:3]│ <- Limited window
│ ...         │
└─────────────┘

Layer 1 (odd): sliding_window = 0
┌─────────────┐
│ Token  Attn │
│ 0      [0]  │
│ 1      [0:2]│
│ 2      [0:3]│ <- Full context
│ ...         │
│ N      [0:N]│
└─────────────┘

Configuration:
self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
```

---

## Shape Transformations Summary

### PyTorch Reference

```
INPUT:
x: (n_tokens=10, hidden_size=2880)

QKV PROJECTION:
qkv: (10, 5120)
  Q: (10, 4096) = (10, 64*64) = (10, num_attention_heads*head_dim)
  K: (10, 512)  = (10, 8*64)  = (10, num_key_value_heads*head_dim)
  V: (10, 512)  = (10, 8*64)  = (10, num_key_value_heads*head_dim)

RESHAPE TO MULTI-HEAD:
  Q: (10, 8, 8, 64) = (tokens, n_heads, q_mult, d_head)
  K: (10, 8, 64)    = (tokens, n_heads, d_head)
  V: (10, 8, 64)    = (tokens, n_heads, d_head)

AFTER ROPE:
  Q: (10, 8, 8, 64)  [shape unchanged]
  K: (10, 8, 64)     [shape unchanged]

EXPAND K, V FOR MQA:
  K: (10, 8, 64) → (10, 8, 8, 64)
  V: (10, 8, 64) → (10, 8, 8, 64)

SINK RESHAPE:
  S: (64,) → (8, 8, 10, 1)

ATTENTION COMPUTATION:
  QK = Q @ K^T: (8, 8, 10, 10)
  S concatenated: (8, 8, 10, 11)
  After softmax: (8, 8, 10, 11)
  After removing sink: (8, 8, 10, 10)
  attn = W @ V: (10, 8, 8, 64)

RESHAPE FOR OUTPUT PROJECTION:
  attn: (10, 8, 8, 64) → (10, 4096)

OUTPUT PROJECTION:
  out: (10, 2880)

OUTPUT:
  (10, 2880)
```

### Neuron Implementation (Expected)

```
INPUT:
hidden_states: (batch=2, seq_len=10, hidden_size=2880)
position_ids: (batch=2, seq_len=10)

QKV PROJECTION (In NeuronAttentionBase):
[Not visible, but likely produces:]
  Q: (2, 10, hidden_size)
  K: (2, 10, hidden_size)
  V: (2, 10, hidden_size)

RESHAPE TO MULTI-HEAD (likely):
  Q: (2, 10, num_attention_heads, head_dim)
  K: (2, 10, num_key_value_heads, head_dim)
  V: (2, 10, num_key_value_heads, head_dim)
  OR
  Q: (batch*seq, num_attention_heads, head_dim)
  K: (batch*seq, num_key_value_heads, head_dim)
  V: (batch*seq, num_key_value_heads, head_dim)

ATTENTION COMPUTATION (In NeuronAttentionBase):
[Not visible, but produces attention output]

OUTPUT PROJECTION:
[Produces attention output]

RETURN:
Tuple: (attention_output, present_kv, cos_cache, sin_cache, ...)

EXTRACT & RETURN:
attention_output: (2, 10, 2880)
```

