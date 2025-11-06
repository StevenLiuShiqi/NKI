# Neuron Distributed Inference Attention Module - Documentation Index

## Overview
This directory contains comprehensive documentation of the Neuron Distributed Inference attention module, including API documentation, implementation details, code references, and usage patterns.

## Module Location
```
/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/modules/attention/
```

## Documentation Files

### 1. NEURON_ATTENTION_SUMMARY.txt (Quick Reference)
**Best for**: Quick lookup, executive overview, immediate answers
- Executive summary of all 4 key topics
- Critical implementation notes
- Quick API reference with code examples
- Tensor shape transformation pipeline
- ~300 lines of condensed information

**Key Sections**:
- Class structure overview
- Learned sinks constraints and usage
- Forward signature with full parameter list
- QKV bias handling explanation
- Tensor transformations

### 2. neuron_attention_analysis.md (Detailed Analysis)
**Best for**: Deep understanding, implementation details, complete API documentation
- 8 comprehensive sections
- Full class structure with code examples
- Detailed learned sinks mechanics
- Complete forward signature documentation
- Return type structure with examples
- QKV projection patterns (fused/separated)
- Configuration parameters reference
- Summary of API patterns and critical notes

**Key Sections**:
1. NeuronAttentionBase Class Structure
2. Learned Sinks and TKG Learned Sinks (full mechanics)
3. Forward Signature and Return Values
4. QKV Projection with qkv_bias
5. Implementation Details (tensor processing)
6. Configuration Parameters
7. Key Files Reference
8. Summary of API Patterns

### 3. neuron_attention_code_references.md (Developer Reference)
**Best for**: Code navigation, finding specific features, line numbers
- Specific line numbers for all features
- File-by-file breakdown
- Quick reference tables
- Code location index
- Tensor shape reference
- Bias application patterns with line numbers
- Fast lookup tables for finding code

**Key Sections**:
- attention_base.py breakdown (class, methods, line numbers)
- gqa.py breakdown (projections)
- sink.py implementation details
- Configuration parameter locations
- Bias application patterns with code snippets
- Learned sinks integration points
- Return value structure details
- Quick reference for finding specific features

## How to Use These Documents

### If you need to:

**Understand the overall architecture quickly**
→ Read: `NEURON_ATTENTION_SUMMARY.txt`
→ Time: 10-15 minutes

**Get complete API documentation**
→ Read: `neuron_attention_analysis.md` Section 3
→ Time: 20-30 minutes

**Find where specific code is located**
→ Use: `neuron_attention_code_references.md`
→ Time: 2-5 minutes per feature

**Understand learned sinks mechanism**
→ Read: `NEURON_ATTENTION_SUMMARY.txt` Section 2
→ Also read: `neuron_attention_analysis.md` Section 2
→ Time: 15-20 minutes

**Understand QKV bias handling**
→ Read: `NEURON_ATTENTION_SUMMARY.txt` Section 4
→ Also read: `neuron_attention_analysis.md` Section 4
→ Cross-reference: `neuron_attention_code_references.md` "Bias Application Pattern"
→ Time: 15 minutes

**Implement custom attention code**
→ Read: `neuron_attention_analysis.md` Sections 1, 3, 8
→ Reference: `neuron_attention_code_references.md` for exact APIs
→ Time: 30-45 minutes

## Key Findings Summary

### 1. NeuronAttentionBase Structure
- Uses GroupQueryAttention_QKV for QKV projections
- Supports both fused (single layer) and separated (three layers) variants
- Uses ColumnParallelLinear for tensor parallelism
- Output projection via GroupQueryAttention_O (row parallel)
- Sophisticated parallel group management (TP, CP, DP)

### 2. Learned Sinks
- Single trainable parameter per head per TP rank
- Constraints: learned_sinks_size must equal 1
- Separate instances for CTE (context encoding) and TKG (token generation)
- Automatically expanded and concatenated with attention scores
- Removed post-softmax to preserve output shape

### 3. Forward API
- Main entry point: `forward()` at lines 1846-1938
- Takes hidden_states + optional masks, embeddings, caches
- Returns NeuronAttentionBaseOutput dataclass
- Supports backward-compatible tuple unpacking
- Routes to 4 different forward paths based on configuration

### 4. QKV Bias
- Boolean parameter: `qkv_bias` controls inclusion
- Fused path: Single bias with metadata attributes
- Separated path: Three independent biases
- Applied automatically in linear layers
- Passed explicitly to TKG kernels

## Code Organization

### Main Files
- `attention_base.py` (3000+ lines): Core attention implementation
- `gqa.py` (1000+ lines): QKV and output projections
- `sink.py` (45 lines): Learned sink implementation
- `utils.py`: Utility functions (RoPE, softmax, etc.)
- `attention_process_groups.py`: Parallel group management

### Import Hierarchy
```
NeuronAttentionBase (attention_base.py)
    ├── GroupQueryAttention_QKV (gqa.py)
    │   └── ColumnParallelLinear
    ├── GroupQueryAttention_O (gqa.py)
    │   └── RowParallelLinear
    └── LearnedSink (sink.py)
        └── BaseParallelLinear
```

## Common Tasks

### Task: Enable Learned Sinks
```python
attention = NeuronAttentionBase(
    config,
    ...,
    learned_sinks_size=1  # Enable sinks
)
# Usage: No changes needed - applied automatically
```

### Task: Use QKV Bias
```python
attention = NeuronAttentionBase(
    config,
    ...,
    qkv_bias=True,  # Enable QKV bias
    o_bias=True      # Enable output bias (optional)
)
# Bias automatically applied in forward pass
```

### Task: Access Forward Results
```python
output = attention(hidden_states, attention_mask, position_ids)

# Option 1: Tuple unpacking (backward compatible)
attn_out, kv, cos_cache, sin_cache = output

# Option 2: Attribute access
attn_out = output.hidden_states
kv = output.present_key_value

# Option 3: Index access
attn_out = output[0]
kv = output[1]
```

### Task: Cache KV During Token Generation
```python
# First pass (prefill)
output = attention(hidden_states, attention_mask, position_ids)
kv_cache = output.present_key_value

# Subsequent passes (token generation)
output = attention(
    new_token,
    past_key_value=kv_cache,
    position_ids=new_position,
    active_mask=new_mask
)
kv_cache = output.present_key_value  # Update for next step
```

## Important Constraints & Limitations

1. **Learned Sinks Size**: Must be exactly 1 (single sink token)
2. **Learned Sinks Trainability**: Initialized as non-trainable by default
3. **Fused QKV + Gather Output**: Cannot be used together (assertion in gqa.py)
4. **Speculative Decoding**: Not yet compatible with chunked attention
5. **Flash Attention**: Performance varies by sequence length (see thresholds in code)

## Cross-References

### To understand parallel processing flow:
See `attention_base.py` lines 194-210 (initialization) and lines 1964-1985 (forward)

### To understand attention computation dispatch:
See `attention_base.py` lines 1846-1938 (forward) and lines 998-1054 (strategy selection)

### To understand QKV projection options:
See `gqa.py` lines 331-520 (GroupQueryAttention_QKV class)

### To understand learned sinks integration:
See `attention_base.py` lines 316-317, 372-374 (initialization)
See `attention_base.py` lines 1637-1660 (usage in token generation)

## Glossary

- **TP (Tensor Parallelism)**: Sharding model across multiple devices
- **CP (Context Parallelism)**: Sharding sequence dimension during prefill
- **DP (Data Parallelism)**: Sharding batch dimension during decode
- **TKG (Token Generation)**: Single token decoding step
- **CTE (Context Encoding)**: Full prefill stage with entire sequence
- **QKV**: Query, Key, Value projections
- **RoPE**: Rotary Position Embedding
- **GQA**: Grouped Query Attention (shares KV heads)
- **Flash Attention**: Optimized attention kernel
- **Fused Operators**: Combined Q, K, V computation
- **Learned Sinks**: Trainable sink tokens for attention distribution

## Version Information

- Neuron Framework: PyTorch 2.8 NXD (inference)
- Compiler: neuronxcc
- Python: 3.10
- Documentation Date: November 5, 2025

---

**For quick reference**: Start with `NEURON_ATTENTION_SUMMARY.txt`

**For complete details**: Read `neuron_attention_analysis.md`

**For code navigation**: Use `neuron_attention_code_references.md`
