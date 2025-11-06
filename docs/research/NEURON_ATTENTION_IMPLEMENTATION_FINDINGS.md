# NeuronX Distributed Inference Attention Implementation Analysis

## Overview
This document details the underlying attention implementation used by `NeuronAttentionBase` in the AWS Neuron SDK for inference on Trainium/Inferentia hardware.

## Core Architecture

### 1. Main Attention Class
**File**: `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/modules/attention/attention_base.py`

The `NeuronAttentionBase` class is the core attention module that implements:
- Tensor parallel attention across Neuron cores
- KV cache management for autoregressive generation
- Multiple attention computation strategies

### 2. Attention Mechanism Types

NeuronAttentionBase uses **different attention kernels** depending on the scenario:

#### A. FlashAttention-based NKI Kernels (Prefill/Context Encoding)
**Primary Kernel**: `attention_isa_kernel` from `neuronxcc.nki._private_kernels.attention`

**Import Location**:
```python
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel
_flash_fwd_call = nki_jit()(attention_isa_kernel)
```

**Strategy Enum**:
```python
class FlashAttentionStrategy(Enum):
    NONE = 0
    UNSHARDED_KERNEL = 1
    SHARDED_KERNEL = 2
    CONTEXT_PARALLEL_KERNEL = 3
    STRIDED_CONTEXT_PARALLEL_KERNEL = 4
    SLIDING_WINDOW_KERNEL = 5  # uses flash_fwd NKI kernel for SWA
```

#### B. Custom NKI Flash Attention Kernels

**1. Sliding Window Attention**
- **File**: `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/modules/sliding_window/attention.py`
- **Function**: `flash_fwd` - Full custom NKI implementation
- **Usage**: For sliding window attention patterns

**2. Paged Attention with Scheduling**
- **File**: `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/modules/chunked_prefill/flash_pa_with_schedule.py`
- **Function**: `flash_paged_attention_with_schedule`
- **Usage**: For continuous batching with paged KV cache

**3. Flash Attention Core**
- **File**: `/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/modules/chunked_prefill/flash_attn_core.py`
- **Function**: `_flash_attention_core`
- **Usage**: Shared core computation logic for tiled flash attention

#### C. Prefix Caching Attention
**Kernel**: `prefix_caching_attention_fwd_isa_kernel`
```python
from neuronxcc.nki._private_kernels.prefix_caching_attention import prefix_caching_attention_fwd_isa_kernel
_flash_fwd_pc_call = nki_jit()(prefix_caching_attention_fwd_isa_kernel)
```

#### D. Token Generation Attention
**Kernel**: `attention_tkg_fwd_isa_kernel` (optional, newer compiler versions)
```python
from neuronxcc.nki._private_kernels.attention import attention_tkg_fwd_isa_kernel
_attn_builtin_token_gen_call = nki_jit()(attention_tkg_fwd_isa_kernel)
```

#### E. Fallback: Native Compiler Attention
Uses standard PyTorch operations when flash attention is not applicable:
```python
def scaled_qk(self, Q, K, attention_mask):
    QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        QK = torch.where(attention_mask.to(torch.bool), QK, torch.finfo(QK.dtype).min)
    return QK
```

## Flash Attention Implementation Details

### Custom NKI Flash Attention (`flash_fwd`)

**Core Algorithm** (from `sliding_window/attention.py`):

1. **Tiling Strategy**:
   - Query tiles: `B_P_SIZE = 128` (partition dimension)
   - KV tiles: Configurable `seq_tile_size` (default 2048, min 512)
   - Head dimension: Up to 128

2. **Online Softmax Computation**:
   ```python
   # Per-tile computation:
   - Load Q tile (d, B_P_SIZE)
   - Load K tile (d, LARGE_TILE_SZ)
   - Load V tile (LARGE_TILE_SZ // B_P_SIZE, B_P_SIZE, d)
   
   # QK^T computation
   qk_psum = nl.matmul(q_tile, k_tile, transpose_x=True)
   
   # Apply causal + sliding window masks
   if use_causal_mask:
       pred_causal = q_pos >= k_pos
       pred_sliding = k_pos > q_pos - sliding_window
       qk_masked = apply_masks(qk_psum, pred_causal, pred_sliding)
   
   # Online softmax accumulators
   max_ = tensor_reduce(np.max, qk_masked)
   m_buffer = maximum(m_previous, max_)
   
   # Compute exp(QK - max)
   p_local = activation_reduce(np.exp, qk_masked, bias=-m_current)
   
   # Update output
   alpha = exp(m_previous - m_current)
   o_buffer = alpha * o_previous + matmul(p_local, v_tile)
   l_buffer = m_current + log(exp(l_previous - m_current) + ps)
   ```

3. **Key Features**:
   - **Mixed precision**: BF16 for matmul, FP32 for accumulation
   - **Causal masking**: Triangular mask for autoregressive
   - **Sliding window**: Local attention with window size parameter
   - **GQA support**: Queries use Q heads, K/V use KV heads
   - **Tiled transpose**: Optimized P^T computation for PV matmul

### Paged Flash Attention (`flash_paged_attention_with_schedule`)

**Special features**:
- **Continuous batching**: Batch size always 1, requests concatenated
- **Paged KV cache**: Block-based cache with indirect addressing
- **Two-phase attention**:
  1. Prior part: Attend to cached KV blocks
  2. Active part: Attend to current input K/V
- **Tile scheduling**: Pre-computed masks and block tables
- **Block tables**: Map query tiles to KV cache blocks

**Core computation** (from `flash_pa_with_schedule.py`):
```python
# Phase 1: Prior KV cache
for large_tile_idx in range(NUM_LARGE_TILE):
    load_kv_from_paged_cache(block_tables, large_tile_idx)
    _flash_attention_core(q_tile, k_cached, v_cached, 
                          o_buffer, l_buffer, m_buffer,
                          tile_mask, use_causal_mask=False)

# Phase 2: Active K/V
if key is not None and value is not None:
    _flash_attention_core(q_tile, k_active, v_active,
                          o_buffer, l_buffer, m_buffer,
                          active_mask, use_causal_mask=True)

# Finalize
out = o_buffer * exp(m_buffer - l_buffer)
```

## Query, Key, Value Processing

### 1. QKV Projections

**Standard Path**:
```python
# Column parallel linear layers (sharded across TP)
self.q_proj = ColumnParallelLinear(hidden_size, num_heads * head_dim, ...)
self.k_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, ...)
self.v_proj = ColumnParallelLinear(hidden_size, num_kv_heads * head_dim, ...)

# Fused QKV option (single kernel)
self.qkv_proj = ColumnParallelLinear(hidden_size, 
                                     (num_heads + 2*num_kv_heads) * head_dim, ...)
```

**With RMSNorm Fusion** (NKI kernel):
```python
_traced_qkv_kernel = nki_jit()(rmsnorm_qkv_isa_kernel)
# Fuses: RMSNorm + Q/K/V projections in single kernel
```

### 2. Rotary Embeddings (RoPE)

**Two implementations**:

**A. Standard RoPE** (first-half/second-half):
```python
def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed

def _rotate_half(x):
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)
```

**B. Polar-compatible RoPE** (even-odd, for Llama 4):
```python
from neuronx_distributed.modules.attention.utils import apply_rotary_polar_compatible
rotary_freqs = precompute_freqs_cis(head_dim, max_len, rope_theta)
Q, K = apply_rotary_polar_compatible(Q, K, rotary_freqs)
```

**RoPE can be fused into attention kernel** when using builtin token-gen kernels.

### 3. Attention Mask Application

**Mask handling varies by kernel**:

**Flash Attention Kernels**:
- Built-in causal masking (no explicit mask tensor)
- Sliding window masks via predicate selection
- Custom tile masks for paged attention

**Native Compiler**:
```python
QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(head_dim)
if attention_mask is not None:
    QK = torch.where(attention_mask.to(torch.bool), QK, 
                     torch.finfo(QK.dtype).min)
attn_scores = F.softmax(QK, dim=-1, dtype=torch.float32)
```

### 4. Sliding Window Attention

**Two paths**:

**A. FlashAttention NKI kernel** (`SLIDING_WINDOW_KERNEL`):
```python
config = FlashConfig(seq_tile_size=2048 or 512)
attn_output = flash_fwd[batch_size, n_head](
    Q, K, V, 
    window_size=(window_size - 1, -1),  # left window, right window
    config=config
)
```

**B. Native compiler** (fallback):
- Manual masking in `scaled_qk`
- Standard softmax and matmul

### 5. Learned Sinks (Sink Tokens)

**Implementation**:
```python
class NeuronAttentionBase:
    def __init__(self, ..., learned_sinks_size=None):
        if learned_sinks_size:
            self.learned_sinks = LearnedSinks(
                num_heads=num_heads,
                learned_sinks_size=learned_sinks_size
            )
    
    def apply_learned_sinks(self, scores):
        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            # Shape: (1, num_heads, 1, 1) -> (bsz, num_heads, q_len, 1)
            learned_sinks = learned_sinks.reshape(1, num_heads, 1, 1)
            learned_sinks = learned_sinks.expand(bsz, -1, q_len, -1)
            scores = torch.cat((scores, learned_sinks), dim=-1)
        return scores
```

**Usage**: Only with native compiler path (not flash attention).

## NKI (Neuron Kernel Interface) Details

### What is NKI?

NKI is AWS's low-level kernel programming interface for Neuron hardware (Trainium/Inferentia). It provides:

- **Direct hardware access**: Program Tensor Engine, Vector Engine, Scalar Engine
- **Custom kernels**: Write optimized kernels in Python-like syntax
- **ISA-level control**: Fine-grained control over instruction scheduling
- **Buffer management**: Explicit control of SBUF, PSUM, HBM

### NKI Kernel Architecture

**Hardware abstractions**:
```python
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa

# Buffers
nl.sbuf  # State buffer (on-chip SRAM)
nl.psum  # Partial sum buffer (for matmul accumulation)
nl.hbm   # High-bandwidth memory (off-chip DRAM)

# Parallel dimensions
par_dim(128)  # Partition dimension (parallelized across cores)

# Operations
nl.matmul(A, B)           # Matrix multiply
nl.load(tensor)           # DMA load
nl.store(dst, src)        # DMA store
nisa.tensor_reduce(...)   # Reduction operations
nisa.activation(...)      # Element-wise activations
```

### Key NKI Kernels Used

**1. attention_isa_kernel** (builtin, compiled):
- **Location**: `neuronxcc.nki._private_kernels.attention`
- **Type**: Compiled .so file (not Python source)
- **Function**: Optimized flash attention for prefill
- **Features**: Causal masking, GQA, mixed precision

**2. flash_fwd** (custom, source available):
- **Location**: `neuronx_distributed_inference.modules.sliding_window.attention`
- **Type**: Python NKI kernel (source code)
- **Function**: Sliding window flash attention
- **Algorithm**: Online softmax with tiling

**3. _flash_attention_core** (reusable component):
- **Location**: `neuronx_distributed_inference.modules.chunked_prefill.flash_attn_core`
- **Type**: Python NKI kernel
- **Function**: Core attention computation for paged attention
- **Features**: Tile masking, online softmax, GQA

## Differences from Standard SDPA

### Standard PyTorch SDPA:
```python
def sdpa(Q, K, V, mask=None):
    scores = (Q @ K.T) / sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -inf)
    attn = softmax(scores, dim=-1)
    output = attn @ V
    return output
```

### NeuronAttentionBase Differences:

1. **Memory Efficiency**:
   - SDPA: Materializes full attention matrix (O(n²) memory)
   - Neuron: Flash attention tiling (O(n) memory)

2. **Computation Order**:
   - SDPA: QK^T → softmax → matmul(attn, V)
   - Neuron: Online softmax with incremental updates

3. **Hardware Mapping**:
   - SDPA: GPU-optimized (CUDA cores, tensor cores)
   - Neuron: Trainium/Inferentia-optimized (NeuronCore, Tensor Engine)

4. **Parallelism**:
   - SDPA: Tensor parallel across GPUs
   - Neuron: Tensor parallel + context parallel across NeuronCores

5. **KV Cache**:
   - SDPA: Simple concatenation
   - Neuron: Paged cache with block tables, flash decoding

6. **Masking**:
   - SDPA: Explicit mask tensor
   - Neuron: Predicate-based masking in-kernel (no mask materialization)

7. **Precision**:
   - SDPA: Typically FP16/BF16 throughout
   - Neuron: Mixed precision (BF16 matmul, FP32 softmax accumulation)

8. **Rotary Embeddings**:
   - SDPA: Separate preprocessing step
   - Neuron: Can fuse RoPE into attention kernel

9. **Sliding Window**:
   - SDPA: Manual masking
   - Neuron: Dedicated sliding window kernel with efficient local attention

10. **Learned Sinks**:
    - SDPA: Not standard
    - Neuron: Built-in support for sink tokens

## Performance Optimizations

### 1. Kernel Selection Logic
```python
def get_flash_attention_strategy(self, q_len, has_attention_mask):
    # Context parallel
    if self.cp_degree > 1:
        if q_len // cp_degree > head_dim:
            return FlashAttentionStrategy.CONTEXT_PARALLEL_KERNEL
    
    # Sliding window
    if self.sliding_window and q_len >= 512:
        return FlashAttentionStrategy.SLIDING_WINDOW_KERNEL
    
    # Standard flash attention
    if q_len >= 4096:  # Long sequences
        return FlashAttentionStrategy.UNSHARDED_KERNEL
    elif q_len >= 1024:  # Medium sequences
        return FlashAttentionStrategy.SHARDED_KERNEL
    
    # Fallback to native compiler
    return FlashAttentionStrategy.NONE
```

### 2. Tiling Configuration
- **Query tile**: 128 tokens (fits in partition dimension)
- **KV tile**: 2048 tokens (default), 512 (minimum for sliding window)
- **Head dim tile**: Full head dimension (up to 128)

### 3. DMA Optimizations
- Transpose operations use hardware DMA transpose (Gen3+)
- Efficient block loading for paged cache
- Indirect loads with block tables

### 4. Mixed Precision
- Matmul: BF16 (Tensor Engine native)
- Softmax: FP32 (better numerical stability)
- Final output: BF16

## Summary

**NeuronAttentionBase uses a sophisticated multi-kernel attention system**:

- **Primary mechanism**: FlashAttention-style tiled attention via NKI kernels
- **Hardware target**: AWS Trainium/Inferentia NeuronCores
- **Memory efficiency**: Online softmax, O(n) memory vs O(n²)
- **Flexibility**: Multiple kernel variants for different scenarios
- **Optimization**: Hardware-specific ISA kernels for maximum performance
- **Feature-rich**: Supports GQA, sliding windows, paged cache, learned sinks, RoPE fusion

The implementation is fundamentally different from standard GPU SDPA, optimized specifically for Neuron hardware architecture with explicit tiling, paging, and hardware-level optimizations.
