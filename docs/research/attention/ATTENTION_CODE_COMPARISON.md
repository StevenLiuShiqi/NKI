# Detailed Code Comparison: PyTorch vs Neuron Attention

## Section 1: QKV Extraction and Reshaping

### PyTorch Reference (gpt_oss.py, lines 220-242)

```python
qkv = self.qkv(t)  # t shape: (n_tokens, 2880)
                   # qkv shape: (n_tokens, 5120)
                   # 5120 = 64 * (64 + 8 + 8)
                   #      = head_dim * (num_attention_heads + 2*num_key_value_heads)

# Extract Q: first 64*64 elements
q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
# q shape: (n_tokens, 4096)

# Extract K: next 8*64 elements
k = qkv[
    :,
    self.num_attention_heads * self.head_dim : 
    (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
].contiguous()
# k shape: (n_tokens, 512)

# Extract V: last 8*64 elements
v = qkv[
    :,
    (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : 
    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
].contiguous()
# v shape: (n_tokens, 512)

# Reshape Q to: (n_tokens, num_key_value_heads, q_mult, head_dim)
q = q.view(
    -1,
    self.num_key_value_heads,                              # 8
    self.num_attention_heads // self.num_key_value_heads,  # 8
    self.head_dim,                                         # 64
)
# q shape: (n_tokens, 8, 8, 64)

# Reshape K to: (n_tokens, num_key_value_heads, head_dim)
k = k.view(-1, self.num_key_value_heads, self.head_dim)
# k shape: (n_tokens, 8, 64)

# Reshape V to: (n_tokens, num_key_value_heads, head_dim)
v = v.view(-1, self.num_key_value_heads, self.head_dim)
# v shape: (n_tokens, 8, 64)
```

### Neuron Implementation (model.py)

The Neuron implementation does NOT show Q, K, V extraction because it's delegated to `NeuronAttentionBase`:

```python
class NeuronGPTOSSAttentionBlock(NeuronAttentionBase):
    def __init__(self, config: InferenceConfig):
        rotary_emb = RotaryEmbedding(...)
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rotary_emb=rotary_emb,
            qkv_bias=True,      # <-- QKV projection config
            o_bias=True,        # <-- Output projection config
            learned_sinks_size=1,
        )
    
    def forward(self, hidden_states, position_ids, ...):
        # NeuronAttentionBase handles Q, K, V projection internally
        output = super().forward(
            hidden_states=hidden_states,
            position_ids=position_ids,
            ...
        )
```

**What we don't see:**
- How NeuronAttentionBase extracts Q, K, V from the QKV projection
- Whether it reshapes them identically to the reference
- The exact shapes at each intermediate step

---

## Section 2: Attention Computation (sdpa)

### PyTorch Reference (gpt_oss.py, lines 153-173)

```python
def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    """
    Q: (n_tokens, 8, 8, 64)         - Multi-head with q_mult
    K: (n_tokens, 8, 64)            - Multi-key-value
    V: (n_tokens, 8, 64)            - Multi-key-value
    S: (64,)                         - Sink token parameters (one per num_attention_heads)
    sm_scale: 1/sqrt(64) = 0.125    - Softmax scaling factor
    sliding_window: 128 or 0         - Sliding window size
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    
    # Step 1: Expand K and V to match Q's q_mult dimension
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    # K becomes: (n_tokens, 8, 8, 64)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    # V becomes: (n_tokens, 8, 8, 64)
    
    # Step 2: Reshape sinks for broadcasting
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    # S becomes: (8, 8, n_tokens, 1)
    
    # Step 3: Create causal mask (lower triangular)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    # Upper triangle (future tokens) = -inf
    # Lower triangle (past tokens) = 0
    
    # Step 4: Add sliding window constraint (if enabled)
    if sliding_window > 0:
        # Add -inf to all positions more than sliding_window steps in the past
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    
    # Step 5: Compute attention scores Q @ K^T
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    # Q: (n_tokens, 8, 8, 64)
    # K: (n_tokens, 8, 8, 64)
    # Output: (8, 8, n_tokens, n_tokens)
    
    # Step 6: Scale by sm_scale
    QK *= sm_scale
    # QK shape: (8, 8, n_tokens, n_tokens)
    
    # Step 7: Apply causal/sliding window mask
    QK += mask[None, None, :, :]
    # mask[None, None, :, :] broadcasts from (n_tokens, n_tokens) 
    # to (1, 1, n_tokens, n_tokens)
    
    # Step 8: CRITICAL - Concatenate sink tokens
    QK = torch.cat([QK, S], dim=-1)
    # QK becomes: (8, 8, n_tokens, n_tokens + 1)
    # The sink dimension is concatenated along the KEY dimension
    # This effectively creates an extra "sink" key position
    
    # Step 9: Softmax across all keys (including sink)
    W = torch.softmax(QK, dim=-1)
    # W shape: (8, 8, n_tokens, n_tokens + 1)
    # Sum over last dimension = 1.0 (includes sink)
    
    # Step 10: Remove sink attention weights
    W = W[..., :-1]
    # W becomes: (8, 8, n_tokens, n_tokens)
    # Drop the last dimension (sink weights)
    
    # Step 11: Apply attention weights to values
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    # W: (8, 8, n_tokens, n_tokens)
    # V: (n_tokens, 8, 8, 64)
    # Output: (n_tokens, 8, 8, 64)
    
    # Step 12: Reshape to flat hidden state
    return attn.reshape(n_tokens, -1)
    # Output: (n_tokens, 4096)
    # 4096 = 8 * 8 * 64 = n_heads * q_mult * d_head
```

### Neuron Implementation (model.py)

The Neuron implementation delegates to `NeuronAttentionBase`:

```python
def forward(
    self,
    hidden_states: torch.Tensor,           # (batch_size, seq_len, hidden_size)
    position_ids: torch.LongTensor,        # (batch_size, seq_len)
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    active_mask: Optional[torch.LongTensor] = None,
    adapter_ids=None,
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    rmsnorm=None,
    rotary_position_ids: Optional[torch.LongTensor] = None,
    kv_mgr: Optional[KVCacheManager] = None,
    get_kv_per_layer: bool = False,
    update_kv_per_layer: bool = False,
    residual: Optional[torch.Tensor] = None,
    **kwargs,
):
    output = super().forward(
        hidden_states=hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        active_mask=active_mask,
        adapter_ids=adapter_ids,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        rmsnorm=rmsnorm,
        rotary_position_ids=rotary_position_ids,
        kv_mgr=kv_mgr,
        get_kv_per_layer=get_kv_per_layer,
        update_kv_per_layer=update_kv_per_layer,
        residual=residual,
        **kwargs,
    )
    # Calls NeuronAttentionBase.forward()
    # Returns: tuple or tensor (unclear from code)
```

**What's NOT visible:**
- How NeuronAttentionBase expands K and V
- Whether sinks are concatenated to attention scores
- Whether sliding window masking is applied
- The actual attention computation and softmax
- How attention weights are applied to values

---

## Section 3: Initialization and Configuration

### PyTorch Reference (gpt_oss.py, lines 176-215)

```python
class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim  # 64
        self.num_attention_heads = config.num_attention_heads  # 64
        self.num_key_value_heads = config.num_key_value_heads  # 8
        
        # SLIDING WINDOW LOGIC: Only apply to even layers
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        # Even layers (0, 2, 4, ...): sliding_window = 128
        # Odd layers (1, 3, 5, ...): sliding_window = 0 (full context)
        
        # SINKS: Learned parameters for each attention head
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
            # Shape: (64,) - one sink value per num_attention_heads
        )
        
        self.norm = RMSNorm(config.hidden_size, device=device)
        
        # QKV Projection: Single combined projection
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        # qkv_dim = 64 * (64 + 16) = 64 * 80 = 5120
        self.qkv = torch.nn.Linear(
            config.hidden_size,  # 2880
            qkv_dim,             # 5120
            device=device,
            dtype=torch.bfloat16
        )
        
        # Output Projection
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,  # 64 * 64 = 4096
            config.hidden_size,                             # 2880
            device=device,
            dtype=torch.bfloat16,
        )
        
        # Softmax scaling factor
        self.sm_scale = 1 / math.sqrt(config.head_dim)  # 1 / sqrt(64) = 0.125
        
        # Rotary Embeddings
        self.rope = RotaryEmbedding(
            config.head_dim,                           # 64
            config.rope_theta,                         # 150000.0
            torch.float32,
            initial_context_length=config.initial_context_length,  # 4096
            scaling_factor=config.rope_scaling_factor,              # 32.0
            ntk_alpha=config.rope_ntk_alpha,                        # 1.0
            ntk_beta=config.rope_ntk_beta,                          # 32.0
            device=device,
        )
```

### Neuron Implementation (model.py, lines 291-310)

```python
class NeuronGPTOSSAttentionBlock(NeuronAttentionBase):
    def __init__(self, config: InferenceConfig):
        rotary_emb = RotaryEmbedding(
            dim=config.head_dim,                                      # 64
            max_position_embeddings=config.max_position_embeddings,   # 131072
            base=config.rope_theta,                                   # 150000
        )
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,                # 2880
            head_dim=config.head_dim,                      # 64
            num_attention_heads=config.num_attention_heads,  # 64
            num_key_value_heads=config.num_key_value_heads,  # 8
            rms_norm_eps=config.rms_norm_eps,
            rotary_emb=rotary_emb,
            qkv_bias=True,           # <-- QKV projection has bias
            o_bias=True,             # <-- Output projection has bias
            learned_sinks_size=1,    # <-- Sink tokens enabled
        )
        # Note: No per-layer sliding_window configuration visible
        # Note: No sliding_window = 0 for odd layers
```

**Key Differences:**

| Feature | PyTorch | Neuron |
|---------|---------|--------|
| **Sliding Window** | `layer_idx % 2 == 0 ? 128 : 0` | Not visible in __init__ |
| **QKV Projection** | Single combined linear layer | Delegated to parent class |
| **Separate Projections** | No, one combined projection | Yes, separate q/k/v projections |
| **Output Projection** | `self.out` linear layer | Delegated to parent class |
| **Sink Token Initialization** | `torch.nn.Parameter(torch.empty(...))` | `learned_sinks_size=1` param |
| **Sink Token Shape** | `(num_attention_heads,)` = `(64,)` | Not visible |
| **RotaryEmbedding** | Full implementation with YaRN | Just shape info passed |

---

## Section 4: Attention Block Forward Pass

### PyTorch Reference (gpt_oss.py, lines 217-247)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x shape: (n_tokens, 2880)
    
    # Step 1: Optional RMSNorm (currently disabled)
    # t = self.norm(x)
    t = x  # Skip norm
    
    # Step 2: QKV projection
    qkv = self.qkv(t)  # (n_tokens, 5120)
    
    # Step 3: Extract Q, K, V
    q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
    # q: (n_tokens, 4096)
    
    k = qkv[
        :,
        self.num_attention_heads * self.head_dim : 
        (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
    ].contiguous()
    # k: (n_tokens, 512)
    
    v = qkv[
        :,
        (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : 
        (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
    ].contiguous()
    # v: (n_tokens, 512)
    
    # Step 4: Reshape to multi-head format
    q = q.view(
        -1,
        self.num_key_value_heads,
        self.num_attention_heads // self.num_key_value_heads,
        self.head_dim,
    )
    # q: (n_tokens, 8, 8, 64)
    
    k = k.view(-1, self.num_key_value_heads, self.head_dim)
    # k: (n_tokens, 8, 64)
    
    v = v.view(-1, self.num_key_value_heads, self.head_dim)
    # v: (n_tokens, 8, 64)
    
    # Step 5: Apply Rotary Embeddings
    q, k = self.rope(q, k)
    # q: (n_tokens, 8, 8, 64)
    # k: (n_tokens, 8, 64)
    
    # Step 6: Scaled Dot-Product Attention with sinks and sliding window
    t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
    # t: (n_tokens, 4096)
    
    # Step 7: Output projection
    t = self.out(t)  # (n_tokens, 2880)
    
    # Step 8: Optional residual (currently disabled)
    # t = x + t
    
    # Step 9: Return
    return t
```

### Neuron Implementation (model.py, lines 313-347 and 349-386)

**NOTE: Has duplicate forward definitions - PROBLEMATIC!**

First definition (lines 313-347):
```python
def forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    active_mask: Optional[torch.LongTensor] = None,
    adapter_ids=None,
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    rmsnorm=None,
    rotary_position_ids: Optional[torch.LongTensor] = None,
    kv_mgr: Optional[KVCacheManager] = None,
    get_kv_per_layer: bool = False,
    update_kv_per_layer: bool = False,
    residual: Optional[torch.Tensor] = None,
    **kwargs,
):
    # hidden_states: (batch_size, seq_len, 2880)
    # position_ids: (batch_size, seq_len)
    
    output = super().forward(
        hidden_states=hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        active_mask=active_mask,
        adapter_ids=adapter_ids,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        rmsnorm=rmsnorm,
        rotary_position_ids=rotary_position_ids,
        kv_mgr=kv_mgr,
        get_kv_per_layer=get_kv_per_layer,
        update_kv_per_layer=update_kv_per_layer,
        residual=residual,
        **kwargs,
    )
    # output: tuple from NeuronAttentionBase
```

Second definition (lines 349-386) - **DUPLICATE, overrides first**:
```python
def forward(
    self,
    hidden_states: torch.Tensor,
    position_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    active_mask: Optional[torch.LongTensor] = None,
    adapter_ids=None,
    cos_cache: Optional[torch.Tensor] = None,
    sin_cache: Optional[torch.Tensor] = None,
    rmsnorm=None,
    rotary_position_ids: Optional[torch.LongTensor] = None,
    kv_mgr: Optional[KVCacheManager] = None,
    get_kv_per_layer: bool = False,
    update_kv_per_layer: bool = False,
    residual: Optional[torch.Tensor] = None,
    **kwargs,
):
    output = super().forward(
        hidden_states=hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        active_mask=active_mask,
        adapter_ids=adapter_ids,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        rmsnorm=rmsnorm,
        rotary_position_ids=rotary_position_ids,
        kv_mgr=kv_mgr,
        get_kv_per_layer=get_kv_per_layer,
        update_kv_per_layer=update_kv_per_layer,
        residual=residual,
        **kwargs,
    )
    
    # Extract first element of tuple
    return tuple(output)[0]
```

---

## Section 5: Return Value Handling

### PyTorch Reference

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    ...
    t = self.out(t)  # (n_tokens, 2880)
    return t         # Single tensor
```

Returns: `torch.Tensor` with shape `(n_tokens, hidden_size)`

### Neuron Implementation

```python
def forward(self, ...) -> ???:
    output = super().forward(...)  # Returns tuple from NeuronAttentionBase
    return tuple(output)[0]        # Extract first element
```

Returns: First element of tuple from `NeuronAttentionBase.forward()`

**Question:** What does NeuronAttentionBase return? Is it a tuple like:
- `(attention_output, present_key_value, cos_cache, sin_cache, ...)`?
- Or something else?

---

## Section 6: Sink Token Mechanism - Deep Dive

### PyTorch Reference: How Sinks Work

```python
# In sdpa() function

# Before: sinks parameter
# S shape: (64,) - one scalar per attention head

# Reshape for broadcasting
S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
# S becomes: (8, 8, n_tokens, 1)
# Each position gets the same sink value, broadcast across sequence

# After computing attention scores QK
# QK shape: (8, 8, n_tokens, n_tokens)

# Concatenate sink along KEY dimension
QK = torch.cat([QK, S], dim=-1)
# QK becomes: (8, 8, n_tokens, n_tokens + 1)
# Now each query can attend to regular keys AND a sink key

# Apply softmax including sink
W = torch.softmax(QK, dim=-1)
# W shape: (8, 8, n_tokens, n_tokens + 1)
# Softmax normalizes all n_tokens + 1 positions

# Remove sink attention weights
W = W[..., :-1]
# W becomes: (8, 8, n_tokens, n_tokens) 
# But the sink contributed to the softmax normalization!

# Apply to values (which don't have sink)
attn = torch.einsum("hmqk,khmd->qhmd", W, V)
# Result uses regular V values, but weights were normalized with sink
```

**Purpose of sinks:**
- Sinks act as a "sink" position that can absorb attention probability mass
- They affect the softmax normalization but don't contribute to the output
- This can help training stability and attention distribution

### Neuron Implementation: Sinks

```python
super().__init__(
    ...
    learned_sinks_size=1,  # <-- Parameter passed to NeuronAttentionBase
)
```

**Questions:**
1. Does NeuronAttentionBase handle sinks the same way?
2. Is the sink reshape: `(num_attention_heads,)` -> `(num_heads, q_mult, n_tokens, 1)`?
3. Are sinks concatenated to QK scores?
4. Are sink weights discarded after softmax?

---

## Section 7: Missing Implementation Details in Neuron

Based on code inspection, here are the areas where Neuron is missing visible implementation:

### Missing in Neuron Code:

1. **Sliding Window per Layer**
   - PyTorch: `self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0`
   - Neuron: No such logic visible
   - Must check if NeuronAttentionBase supports this

2. **K, V Expansion for MQA**
   - PyTorch: `K = K[:, :, None, :].expand(-1, -1, q_mult, -1)`
   - Neuron: Hidden in NeuronAttentionBase

3. **Attention Score Computation**
   - PyTorch: `torch.einsum("qhmd,khmd->hmqk", Q, K)`
   - Neuron: Hidden in NeuronAttentionBase

4. **Mask Application**
   - PyTorch: `QK += mask[None, None, :, :]`
   - Neuron: Hidden in NeuronAttentionBase

5. **Sink Token Concatenation**
   - PyTorch: `QK = torch.cat([QK, S], dim=-1)`
   - Neuron: Hidden in NeuronAttentionBase

6. **Attention Weight Application**
   - PyTorch: `torch.einsum("hmqk,khmd->qhmd", W, V)`
   - Neuron: Hidden in NeuronAttentionBase

