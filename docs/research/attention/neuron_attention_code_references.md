# Neuron Attention Module - Code References and Line Numbers

## File Locations

**Base Directory:**
```
/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/modules/attention/
```

## attention_base.py (3000+ lines)

### Class and Data Structure Definitions

| Item | Lines | Description |
|------|-------|-------------|
| Imports and Constants | 1-116 | Module imports, kernel loading, enum definitions |
| FlashAttentionStrategy enum | 119-125 | Attention computation strategies |
| QKNormPlacement enum | 128-130 | QK normalization placement options |
| NeuronAttentionBaseOutput dataclass | 133-147 | Return value structure with backward compatibility |
| NeuronAttentionBase class definition | 150-158 | Class documentation and purpose |

### NeuronAttentionBase.__init__ Method

| Section | Lines | Description |
|---------|-------|-------------|
| Signature | 160-184 | All initialization parameters |
| Config setup | 186-213 | Config and dtype initialization |
| Sequence parallel and chunking setup | 214-228 | SP and attention chunk configuration |
| Window and kernel setup | 228-242 | Sliding window and kernel enablement flags |
| Head dimension and embedding setup | 244-254 | Head counts, rotary embeddings, inv_freqs |
| Bias and norm setup | 256-267 | Bias flags, RMSNorm, QK norm setup |
| GQA initialization | 269-273 | Call to init_gqa_properties() |

### Key Methods

| Method | Lines | Purpose |
|--------|-------|---------|
| init_tkg_cp_qkv_o_proj() | 275-317 | Initialize token gen KV-parallel projections with learned sinks |
| init_gqa_properties() | 319-420 | Initialize GroupQueryAttention projections for all parallelism configs |
| init_qk_norm() | 422-429 | Initialize QK normalization layer |
| get_learned_sinks() | 431-439 | Retrieve learned sinks based on stage/parallelism |
| scaled_qk() | 441-445 | Compute scaled QK scores |
| get_qkv_proj() | 447-453 | Get appropriate QKV projection for stage |
| get_o_proj() | 455-461 | Get appropriate output projection for stage |
| apply_rotary_embedding() | 463-478 | Apply RoPE to Q and K |
| prep_qkv_tensors() | 480-545 | Prepare Q, K, V with layout changes, RoPE, normalization |
| context_parallel_flash_attention_kernel() | 547-649 | CP-specific flash attention |
| get_flash_attention_strategy() | 998-1054 | Select appropriate flash attention kernel based on seq_len |
| compute_for_flash_decoding() | 1056-1086 | Flash decoding computation |
| attention_tokengen_kernel_shared() | 1088-1150 | Shared TKG kernel setup |
| attention_tokengen_kernel_nki() | 1152-1223 | NKI-based TKG kernel execution |
| compute_for_token_gen() | 1572-1660 | Token generation attention with learned sinks |
| perform_contexted_prefill() | 1662-1712 | Chunked prefill attention |
| attention_context_encode() | 1714-1732 | Full context encoding phase |
| attention_context_encode_chunked_attention() | 1753-1762 | Chunked prefill encoding |
| attention_context_encode_windowed_attention() | 1764-1770 | Sliding window attention encoding |
| attention_tokengen() | 1772-1844 | Dispatch token generation computation |
| **forward()** | **1846-1938** | **Main entry point - routes to appropriate forward path** |
| standard_causal_attention_forward() | 1940-2143 | Standard attention with all kernel options |
| chunked_attention_forward() | 2168-2256 | Chunked attention path (CP=1) |
| chunked_attention_with_context_parallel_forward() | 2258-2400+ | Chunked attention with CP > 1 |
| windowed_attention_forward() | Later | Sliding window attention path |

### Forward Method Details (Lines 1846-1938)

```python
def forward(
    self,
    hidden_states: torch.Tensor,                    # Line 1848
    attention_mask: Optional[torch.Tensor] = None,  # Line 1849
    position_ids: Optional[torch.LongTensor] = None, # Line 1850
    past_key_value: Optional[Tuple[torch.Tensor]] = None,  # Line 1851
    active_mask: Optional[torch.LongTensor] = None,  # Line 1852
    adapter_ids=None,                               # Line 1853
    cos_cache: Optional[torch.Tensor] = None,       # Line 1854
    sin_cache: Optional[torch.Tensor] = None,       # Line 1855
    rmsnorm=None,                                   # Line 1856
    rotary_position_ids: Optional[torch.LongTensor] = None,  # Line 1857
    kv_mgr: Optional[KVCacheManager] = None,        # Line 1859
    get_kv_per_layer: bool = False,                 # Line 1860
    update_kv_per_layer: bool = False,              # Line 1861
    residual: Optional[torch.Tensor] = None,        # Line 1862
    **kwargs,
) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:  # Line 1864

# Routing logic
if self.attention_chunk_size and self.cp_degree == 1:  # Line 1865
    return self.chunked_attention_forward(...)
elif self.attention_chunk_size and self.cp_degree > 1:  # Line 1883
    return self.chunked_attention_with_context_parallel_forward(...)
elif self.sliding_window:  # Line 1902
    return self.windowed_attention_forward(...)
else:  # Line 1920
    return self.standard_causal_attention_forward(...)
```

### Learned Sinks Related Code

| Reference | Lines | Code Snippet |
|-----------|-------|--------------|
| Parameter in __init__ | 184 | `learned_sinks_size: Optional[int] = None` |
| Storage in __init__ | 268 | `self.learned_sinks_size = learned_sinks_size` |
| Init TKG learned sinks | 316-317 | Conditional LearnedSink creation for TKG |
| Init learned sinks (CTE) | 372-374 | Conditional LearnedSink creation for CTE |
| Getter method | 431-439 | Returns appropriate sink based on stage |
| Usage in compute_for_token_gen | 1637-1645 | Expand and concatenate with scores |
| Cleanup in softmax | 1652-1653 | Remove sink contributions from softmax |

## gqa.py (1000+ lines)

### Class Definitions

| Class | Lines | Purpose |
|-------|-------|---------|
| GQA enum | 41-68 | Sharding strategies for GQA |
| BaseGroupQueryAttention | 255-329 | Base class for QKV/O projections |
| GroupQueryAttention_QKV | 331-520+ | QKV projection with fused/separated variants |
| GroupQueryAttention_O | Later | Output projection |

### GroupQueryAttention_QKV.__init__

| Item | Lines | Details |
|------|-------|---------|
| Signature | 332-357 | All parameters including bias |
| Validation | 369-372 | Check fused_qkv and gather_output compatibility |
| Fused QKV path | 390-416 | ColumnParallelLinear with bias attributes |
| Separated QKV path | 418-448 | Three separate ColumnParallelLinear layers |
| Non-parallel path | 449-465 | Regular nn.Linear fallback |

### QKV Forward Methods

| Method | Lines | Purpose |
|--------|-------|---------|
| forward() | 467-481 | Main entry point, dispatches to kernel or native |
| _native_qkv_forward() | 483-508 | CPU-side computation |
| _kernel_qkv_forward() | Later | NKI kernel-based computation |
| _split_fused_qkv() | Later | Split fused QKV output into Q, K, V |

## sink.py (45 lines)

### LearnedSink Class

| Item | Lines | Details |
|------|-------|---------|
| Import | 1 | BaseParallelLinear inheritance |
| Class definition | 10 | LearnedSink(BaseParallelLinear) |
| __init__ signature | 12-19 | All parameters with constraint validation |
| Assertion | 21-23 | `learned_sinks_size == 1` constraint |
| Sink parameter | 30-32 | torch.nn.Parameter initialization with zeros |
| get_sink() method | 43-44 | Simple getter returning self.sink |

## Key Configuration Locations

### __init__ Parameters (attention_base.py, lines 160-184)

```python
def __init__(self,
             config: InferenceConfig,                              # Line 161
             *,
             hidden_size: int,                                    # Line 163
             num_attention_heads: int,                            # Line 164
             num_key_value_heads: int,                            # Line 165
             head_dim: int = None,                                # Line 166
             rotary_emb=None,                                     # Line 167
             rope_theta: float = None,                            # Line 168
             use_scaled_rope: bool = False,                       # Line 169
             rms_norm_eps: float = None,                          # Line 170
             use_qk_norm: bool = False,                           # Line 171
             qk_norm_placement: QKNormPlacement = QKNormPlacement.PRE_ROPE,  # Line 172
             q_layernorm: Callable = None,                        # Line 173
             k_layernorm: Callable = None,                        # Line 174
             clip_qkv: float = None,                              # Line 175
             qkv_bias: bool = False,                              # Line 176
             o_bias: bool = False,                                # Line 177
             num_cores_per_group: int = 1,                        # Line 178
             sequence_parallel_enabled: bool = None,              # Line 179
             attention_chunk_size: int = None,                    # Line 180
             sliding_window: int = None,                          # Line 181
             tensor_model_parallel_group: Optional[ProcessGroup] = None,  # Line 182
             o_proj_layer_name: str = "o_proj",                   # Line 183
             learned_sinks_size: Optional[int] = None):           # Line 184
```

## Bias Application Pattern

### Fused QKV Bias (gqa.py, lines 392-416)

```python
self.Wqkv = ColumnParallelLinear(                  # Line 392
    self.hidden_size,                              # Line 393
    (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,  # Line 394
    bias=self.bias,                                # Line 395 - Uses qkv_bias
    ...
)

if self.bias:                                      # Line 412
    # Metadata for weight sharding
    setattr(self.Wqkv.bias, "fused_qkv", True)    # Line 413
    setattr(self.Wqkv.bias, "num_attention_heads", self.num_attention_heads)  # Line 414
    ...
```

### Separated QKV Bias (gqa.py, lines 419-448)

```python
self.q_proj = ColumnParallelLinear(                # Line 419
    self.hidden_size,                              # Line 420
    self.num_attention_heads * self.head_dim,      # Line 421
    bias=self.bias,                                # Line 422 - Each has own bias
    ...
)
self.k_proj = ColumnParallelLinear(...)            # Line 429
self.v_proj = ColumnParallelLinear(...)            # Line 439
```

### TKG Kernel Bias (attention_base.py, line 2290+)

```python
W_qkv_bias = self.get_qkv_proj().Wqkv.bias.unsqueeze(0) if self.qkv_bias else None
if self.qkv_bias:
    tkg_kernel_kwargs["bias_qkv"] = W_qkv_bias
```

## Learned Sinks Integration Points

| Location | Lines | Operation |
|----------|-------|-----------|
| Stored as attribute | 268 | `self.learned_sinks_size = learned_sinks_size` |
| CTE init | 372-374 | Creates `self.learned_sinks` |
| TKG init | 316-317 | Creates `self.tkg_learned_sinks` |
| Getter | 431-439 | Returns appropriate sink tensor |
| Prefill use | 915-922 | Concatenated with active scores |
| Token gen use | 1637-1645 | Concatenated with prior and active scores |
| Cleanup | 1652-1653 | Removed from softmax output |

## Return Value Structure

### NeuronAttentionBaseOutput (lines 133-147)

```python
@dataclass(frozen=True)
class NeuronAttentionBaseOutput:
    hidden_states: torch.tensor                     # Line 135 - Main output
    present_key_value: torch.tensor                 # Line 136 - Updated KV
    cos_cache: Optional[torch.tensor] = None        # Line 137
    sin_cache: Optional[torch.tensor] = None        # Line 138
    residual: Optional[torch.tensor] = None         # Line 139
    
    def __iter__(self):                             # Line 142 - Backward compat
        return iter([self.hidden_states, self.present_key_value, self.cos_cache, self.sin_cache])
    
    def __getitem__(self, i):                       # Line 146 - Index access
        return getattr(self, fields(self)[i].name)
```

## Key Tensor Shapes Throughout Pipeline

```
Input hidden_states:  (batch, seq_len, hidden_size)
After QKV proj:       Q: (batch, seq_len, num_heads * head_dim)
                      K: (batch, seq_len, num_kv_heads * head_dim)
                      V: (batch, seq_len, num_kv_heads * head_dim)
After layout change:  Q: (batch, num_heads, seq_len, head_dim)
                      K: (batch, num_kv_heads, seq_len, head_dim)
                      V: (batch, num_kv_heads, seq_len, head_dim)
Attention scores:     (batch, num_heads, seq_len, seq_len)
With learned sinks:   (batch, num_heads, seq_len, seq_len + 1)
After softmax:        (batch, num_heads, seq_len, seq_len)
Attention output:     (batch, num_heads, seq_len, head_dim)
After merge:          (batch, seq_len, num_heads * head_dim)
After o_proj:         (batch, seq_len, hidden_size)
```

---

## Quick Reference: Finding Specific Features

### To find learned sinks usage:
- Definition: `sink.py` lines 10-44
- Initialization in NeuronAttentionBase: `attention_base.py` lines 316-317, 372-374
- Getter: `attention_base.py` lines 431-439
- Application in attention: Search for "learned_sinks" in attention_base.py

### To find QKV bias handling:
- Parameter in __init__: `attention_base.py` line 176
- GQA initialization: `gqa.py` lines 392-416 (fused) and 419-448 (separated)
- TKG kernel usage: Search for "bias_qkv" in attention_base.py
- Forward pass: `gqa.py` lines 467-481

### To find forward flow:
- Entry point: `attention_base.py` lines 1846-1938
- Chunked path: `attention_base.py` lines 2168-2256
- Standard path: `attention_base.py` lines 1940-2143
- Token gen dispatch: `attention_base.py` lines 1772-1844

### To find return types:
- Output dataclass: `attention_base.py` lines 133-147
- Return statements: Search "return NeuronAttentionBaseOutput" in attention_base.py
