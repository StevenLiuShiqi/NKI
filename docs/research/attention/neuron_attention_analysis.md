# Neuron Distributed Inference Attention Module - API & Implementation Analysis

## Overview
The Neuron Distributed Inference attention module provides specialized implementations for deploying attention mechanisms on AWS Neuron accelerators with support for tensor model parallelism, context parallelism, and various optimization kernels.

Location: `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/modules/attention/`

---

## 1. NeuronAttentionBase Class Structure

### Class Definition
```python
class NeuronAttentionBase(nn.Module):
    """
    Core Neuron-optimized attention implementation with:
    1. Column parallel Q, K, V projections
    2. Row parallel output projection  
    3. Head sharding across TP degree
    4. Custom position embeddings and kernels
    """
```

### Key Components

#### 1.1 QKV Projection Infrastructure
The module uses `GroupQueryAttention_QKV` for fused or separated Q, K, V projections:

**Fused QKV Path:**
```
self.Wqkv: ColumnParallelLinear(
    hidden_size → (num_attention_heads + 2*num_key_value_heads) * head_dim
    bias=qkv_bias  # Boolean flag controls bias
)
```

**Separated QKV Path (when not fused):**
```
self.q_proj: ColumnParallelLinear(hidden_size → num_attention_heads * head_dim)
self.k_proj: ColumnParallelLinear(hidden_size → num_key_value_heads * head_dim)
self.v_proj: ColumnParallelLinear(hidden_size → num_key_value_heads * head_dim)
```

#### 1.2 Output Projection
```python
self.o_proj: GroupQueryAttention_O (RowParallelLinear)
    # Reduces: num_heads * head_dim → hidden_size
```

#### 1.3 Parallel Training/Inference Groups
- **Context Parallel (CP)**: For prefill stage when cp_degree > 1
- **Data Parallel (DP)**: For decode stage when attention_dp_degree > 1  
- **Tensor Parallel (TP)**: Default tensor model parallelism across all stages

---

## 2. Learned Sinks and TKG Learned Sinks

### Purpose
Learned sinks are trainable parameters that improve attention distribution in token generation (TKG) scenarios. They provide "sink" tokens that capture excess attention probability mass.

### Data Structure

**LearnedSink Class:**
```python
class LearnedSink(BaseParallelLinear):
    def __init__(
        self,
        learned_sinks_size: int,        # Must be 1
        num_attention_heads: int,
        torch_dtype: torch.dtype,
        tensor_model_parallel_size: int,
        rank_ordering: List[int] = None,
    ):
        # Sink parameter shape: (sink_size_per_partition,)
        # where sink_size_per_partition = num_attention_heads / tensor_model_parallel_size
        self.sink = torch.nn.Parameter(
            torch.zeros(sink_size_per_partition, dtype=torch_dtype),
            requires_grad=False  # Note: No gradients by default
        )
```

**Constraint**: `learned_sinks_size == 1` (only single sink token supported)

### Usage Pattern in NeuronAttentionBase

#### Initialization:
```python
# In init_gqa_properties():
if self.learned_sinks_size is not None:
    self.learned_sinks = LearnedSink(
        learned_sinks_size,
        num_attention_heads,
        torch_dtype,
        tp_degree,
        cte_rank_ordering
    )

# For Token Generation with special CP/DP handling:
if self.learned_sinks_size is not None:
    self.tkg_learned_sinks = LearnedSink(
        learned_sinks_size,
        num_attention_heads,
        torch_dtype,
        process_group.size(),
        rank_ordering
    )
```

#### Retrieval:
```python
def get_learned_sinks(self):
    if self.learned_sinks_size is None:
        return None
    # Select based on stage and parallelism config
    if self.neuron_config.is_prefill_stage and self.cp_degree != self.dp_degree:
        return self.learned_sinks.sink  # Shape: (num_heads,)
    elif not self.neuron_config.is_prefill_stage and self.cp_degree != self.dp_degree:
        return self.tkg_learned_sinks.sink
    else:
        return self.learned_sinks.sink
```

#### Application in Attention:
```python
learned_sinks = self.get_learned_sinks()
if learned_sinks is not None:
    # learned_sinks shape: (num_heads,)
    assert learned_sinks.ndim == 1 and learned_sinks.shape[0] == self.num_heads
    
    # Expand to batch dimension
    bsz, _, seqlen, _ = active_scores.shape
    sinks = learned_sinks.reshape(1, self.num_heads, 1, 1)
    sinks = sinks.expand(bsz, -1, seqlen, -1)  # (bsz, num_heads, seqlen, 1)
    
    # Concatenate with attention scores
    active_scores = torch.cat((active_scores, sinks), dim=-1)
    prior_scores = torch.cat((prior_scores, sinks), dim=-1)

# After softmax, remove sink contributions
if learned_sinks is not None:
    softmax_prior = softmax_prior[..., :-1]  # Remove sink column
```

---

## 3. Forward Signature and Return Values

### Main Forward Method

```python
def forward(
    self,
    hidden_states: torch.Tensor,                    # (batch, seq_len, hidden_size)
    attention_mask: Optional[torch.Tensor] = None,  # (batch, 1, seq_len, seq_len) or similar
    position_ids: Optional[torch.LongTensor] = None, # (batch, seq_len)
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # Cached KV for token gen
    active_mask: Optional[torch.LongTensor] = None,  # Mask for current tokens
    adapter_ids=None,                               # For LoRA adapters
    cos_cache: Optional[torch.Tensor] = None,       # Precomputed RoPE cos
    sin_cache: Optional[torch.Tensor] = None,       # Precomputed RoPE sin
    rmsnorm=None,                                   # RMSNorm layer reference
    rotary_position_ids: Optional[torch.LongTensor] = None,  # For RoPE application
    kv_mgr: Optional[KVCacheManager] = None,        # KV cache manager
    get_kv_per_layer: bool = False,                 # Fetch KV from manager
    update_kv_per_layer: bool = False,              # Update KV in manager
    residual: Optional[torch.Tensor] = None,        # Residual for residual connections
    **kwargs,
) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
```

### Return Type: NeuronAttentionBaseOutput

```python
@dataclass(frozen=True)
class NeuronAttentionBaseOutput:
    hidden_states: torch.Tensor                      # Output attention (batch, seq_len, hidden_size)
    present_key_value: torch.Tensor                  # Updated KV cache tuple (K, V)
    cos_cache: Optional[torch.Tensor] = None         # RoPE cos (for reuse)
    sin_cache: Optional[torch.Tensor] = None         # RoPE sin (for reuse)
    residual: Optional[torch.Tensor] = None          # Residual tensor (for attn blocks)
    
    # Backward compatibility - tuple unpacking support
    def __iter__(self):
        return iter([self.hidden_states, self.present_key_value, self.cos_cache, self.sin_cache])
    
    def __getitem__(self, i):
        return getattr(self, fields(self)[i].name)
```

### Return Usage Example:
```python
# Standard unpacking (backward compatible)
attn_output, kv, cos_cache, sin_cache = attention_output

# Or via dataclass fields
output = attention_output
attn_output = output.hidden_states
kv = output.present_key_value
```

### Forward Flow Routing
```python
def forward(...) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
    if self.attention_chunk_size and self.cp_degree == 1:
        return self.chunked_attention_forward(...)
    elif self.attention_chunk_size and self.cp_degree > 1:
        return self.chunked_attention_with_context_parallel_forward(...)
    elif self.sliding_window:
        return self.windowed_attention_forward(...)
    else:
        return self.standard_causal_attention_forward(...)
```

---

## 4. QKV Projection with qkv_bias

### Bias Parameter Control

**Initialization Parameter:**
```python
def __init__(
    self,
    ...
    qkv_bias: bool = False,         # Enable/disable bias in QKV projections
    o_bias: bool = False,           # Enable/disable bias in output projection
    ...
):
    self.qkv_bias = qkv_bias
    self.o_bias = o_bias
```

### Bias Application in GroupQueryAttention_QKV

#### Fused QKV Case:
```python
if self.fused_qkv:
    self.Wqkv = ColumnParallelLinear(
        self.hidden_size,
        (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
        bias=self.bias,  # Uses qkv_bias parameter
        gather_output=self.gather_output,
        dtype=dtype,
        ...
    )
    
    if self.bias:
        # Bias attributes for weight sharding
        setattr(self.Wqkv.bias, "fused_qkv", True)
        setattr(self.Wqkv.bias, "num_attention_heads", self.num_attention_heads)
        setattr(self.Wqkv.bias, "num_key_value_heads", self.num_key_value_heads)
        setattr(self.Wqkv.bias, "head_dim", self.head_dim)
```

#### Separated QKV Case:
```python
self.q_proj = ColumnParallelLinear(
    self.hidden_size,
    self.num_attention_heads * self.head_dim,
    bias=self.bias,  # Each projection has independent bias
    ...
)
self.k_proj = ColumnParallelLinear(...)
self.v_proj = ColumnParallelLinear(...)
```

### Bias Usage in Forward Pass

```python
def forward(self, hidden_states: torch.Tensor, rmsnorm=None, adapter_ids=None, residual=None):
    if self.qkv_kernel_enabled:
        return self._kernel_qkv_forward(hidden_states, rmsnorm, residual)
    else:
        Q, K, V = self._native_qkv_forward(hidden_states, adapter_ids)
    return Q, K, V, residual

def _native_qkv_forward(self, hidden_states: torch.Tensor, adapter_ids=None):
    if self.fused_qkv:
        # Fused projection with bias (if enabled)
        QKV = self.Wqkv(hidden_states)  # Bias applied inside ColumnParallelLinear
        return self._split_fused_qkv(QKV)
    else:
        # Individual projections with biases (if enabled)
        Q = self.q_proj(hidden_states)  # Bias inside
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        return Q, K, V
```

### Bias in Token Generation Kernel

```python
# TKG kernel includes bias parameter
W_qkv_bias = self.get_qkv_proj().Wqkv.bias.unsqueeze(0) if self.qkv_bias else None
if self.qkv_bias:
    tkg_kernel_kwargs["bias_qkv"] = W_qkv_bias  # Pass to kernel
```

### Output Projection Bias

```python
self.o_proj = GroupQueryAttention_O(
    ...
    bias=self.o_bias,  # Controls o_proj bias
)
```

---

## 5. Implementation Details

### QKV Tensor Processing

**Input to Output Projection:**
```
hidden_states: (batch, seq_len, hidden_size)
    ↓
QKV Projection (with optional bias)
    ↓
Q: (batch, seq_len, num_heads * head_dim)
K: (batch, seq_len, num_key_value_heads * head_dim)
V: (batch, seq_len, num_key_value_heads * head_dim)
    ↓
Layout Change: BSHD → BHSD
    ↓
Q: (batch, num_heads, seq_len, head_dim)
K: (batch, num_kv_heads, seq_len, head_dim)
V: (batch, num_kv_heads, seq_len, head_dim)
    ↓
RoPE Application (rotary embeddings)
    ↓
QK Norm (optional)
    ↓
Attention Computation
```

### Sharding Strategy

**Two strategies via GQA enum:**
1. `REPLICATE_TO_TP_DEGREE`: KV heads replicated to TP degree (default)
2. `CONVERT_TO_MHA`: Convert GQA to full MHA with padding

### Attention Computation Paths

1. **Flash Attention Kernel** (optimized for long sequences)
   - Unsharded: seq_len ≥ 4096
   - Sharded: seq_len ≥ 1024 (LNC2 specific)
   - Sliding window: For models with local attention

2. **Token Generation (TKG) Kernels**
   - Builtin TKG kernel (with fused RoPE)
   - NKI TKG kernel 
   - Manual computation fallback

3. **Chunked Prefill** 
   - For memory-efficient processing
   - Compatible with context parallelism

4. **Context Encoding**
   - Full prefill stage
   - Prefix caching support

---

## 6. Configuration Parameters

### Key Configuration Attributes

```python
self.tp_degree                          # Tensor parallelism degree
self.cp_degree                          # Context parallelism degree
self.dp_degree                          # Data parallelism degree (decode)
self.torch_dtype                        # Computation dtype
self.fused_qkv                          # Use fused QKV projection
self.qkv_bias                           # QKV bias flag
self.o_bias                             # Output bias flag
self.use_qk_norm                        # Apply QK normalization
self.qk_norm_placement                  # PRE_ROPE or POST_ROPE
self.clip_qkv                           # QKV clipping value
self.learned_sinks_size                 # Learned sink size (1 if enabled)
self.attn_kernel_enabled                # Flash attention kernel
self.attn_tkg_builtin_kernel_enabled    # TKG builtin kernel
self.attn_tkg_nki_kernel_enabled        # TKG NKI kernel
self.sequence_parallel_enabled          # Sequence parallelism flag
self.sliding_window                     # SWA window size
self.k_cache_transposed                 # KV cache layout format
```

---

## 7. Key Files

| File | Purpose |
|------|---------|
| `attention_base.py` | Core attention implementation (3000+ lines) |
| `gqa.py` | GroupQueryAttention for QKV/Output projections |
| `sink.py` | LearnedSink implementation |
| `utils.py` | Attention utilities (RoPE, softmax, etc.) |
| `attention_process_groups.py` | Parallel process group management |

---

## 8. Summary of API Patterns

### Pattern 1: Basic Attention Call
```python
output = attention_layer(
    hidden_states,
    attention_mask=mask,
    position_ids=pos_ids,
)
# Returns: NeuronAttentionBaseOutput with hidden_states, present_key_value, etc.
```

### Pattern 2: Token Generation with Cached KV
```python
output = attention_layer(
    hidden_states,  # (batch, 1, hidden_size) for single token
    past_key_value=kv_cache,  # From previous step
    position_ids=current_pos,
    active_mask=mask_for_current_token,
)
```

### Pattern 3: Learned Sinks Integration
```python
# Sinks automatically applied if learned_sinks_size > 0
# No explicit API needed - handled internally
# Sink shape: (num_heads,) per rank
```

### Pattern 4: KV Cache Management
```python
output = attention_layer(
    ...,
    kv_mgr=kv_manager,
    get_kv_per_layer=True,
    update_kv_per_layer=True,
    idx=layer_id,
)
```

---

## Critical Implementation Notes

1. **Learned Sinks Constraint**: Only `learned_sinks_size=1` is supported (single sink token)
2. **Bias Handling**: Separate bias parameters for each projection when not fused
3. **RoPE Integration**: Supports both standard and polar-compatible RoPE variants
4. **Backward Compatibility**: NeuronAttentionBaseOutput maintains tuple unpacking behavior
5. **Kernel Selection**: Automatic based on sequence length and configuration
6. **Parallelism**: Sophisticated handling of CP, DP, TP with stage-aware logic

