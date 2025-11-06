# Testing Infrastructure Guide for Trn1 Model Implementation

## Overview

This guide provides instructions for developing an automated testing infrastructure for Amazon Trn1 accelerator model inference. The infrastructure validates that Neuron-optimized implementations match the reference PyTorch implementations in `gpt_oss.py`.

## Testing Goals

Create a comprehensive test suite that:
1. Tests individual components (AttentionBlock, MLPBlock, RMSNorm, RoPE, etc.)
2. Tests integrated blocks (full transformer layers)
3. Validates functional correctness against reference implementations
4. Follows the established testing pattern in `tests/test_attention.py`

## Test Structure Pattern

Each test should follow this structure:

### 1. Configuration Setup
```python
config = _make_tiny_inference_config()
reference_config = _get_ref_config(config=config)
```

### 2. Input Preparation
```python
batch_size = config.neuron_config.batch_size
seq_len = config.neuron_config.seq_len
hidden_size = config.hidden_size

torch.manual_seed(0)
inp = torch.rand(batch_size, seq_len, hidden_size, dtype=config.neuron_config.torch_dtype)
sample = torch.arange(batch_size * seq_len * hidden_size, dtype=config.neuron_config.torch_dtype)
sample = sample.reshape(batch_size, seq_len, hidden_size).to(dtype=config.neuron_config.torch_dtype)
```

### 3. Neuron Module Build
```python
neuron_module = build_module(
    NeuronModuleClass,
    example_inputs,
    tp_degree=1,
    module_init_kwargs={
        "config": config,
        "weight_init_value": _CONSTANT_INIT_VALUE,
    },
    checkpoint_path=str(_CHECKPOINT_PATH),
)
```

### 4. Reference Module Setup
```python
reference_module = ReferenceModuleClass(reference_config)
_fill_module_parameters(reference_module, _CONSTANT_INIT_VALUE)
```

### 5. Validation
```python
with torch.no_grad():
    # Run reference implementation
    ref_output = reference_module(ref_input)

# Validate Neuron implementation matches reference
validate_accuracy(neuron_module, inputs, expected_outputs=[ref_output])
```

## Components to Test

### High Priority Components
1. **AttentionBlock** - ✅ Already implemented in `tests/test_attention.py`
2. **MLPBlock** - Expert-based MLP with SwiGLU activation
3. **RMSNorm** - Root Mean Square normalization
4. **RotaryEmbedding** - RoPE with YaRN scaling

### Integrated Components
1. **TransformerBlock** - Combined attention + MLP layer
2. **Full Transformer** - End-to-end model testing (if applicable)

## Test Organization

### File Structure
```
tests/
├── test_attention.py      # AttentionBlock tests (existing)
├── test_mlp.py           # MLPBlock tests
├── test_normalization.py # RMSNorm tests
├── test_rope.py          # RotaryEmbedding tests
├── test_transformer.py   # Integrated transformer block tests
└── test_utils.py         # Shared utilities (existing)
```

### Naming Conventions
- Test functions: `test_{component}_forward_matches_reference()`
- Neuron implementations: `Neuron{ComponentName}` (e.g., `NeuronGPTOSSAttentionBlock`)
- Reference implementations: Use classes from `gpt_oss.py`

## Key Implementation Details

### Constant Weight Initialization
- All tests use constant weight initialization with `_CONSTANT_INIT_VALUE = 0.5`
- This ensures deterministic behavior and easier debugging
- Use `_fill_module_parameters(module, value)` from `test_utils.py`

### Configuration
- Use `_make_tiny_inference_config()` for test configurations
- This provides a small, fast configuration suitable for unit tests
- Convert to reference config using `_get_ref_config(config)`

### Input Handling
- Most components expect flattened inputs: `(n_tokens, hidden_size)`
- Attention components may need `position_ids` as additional input
- Reshape batch-oriented tensors: `inp.view(-1, hidden_size)`

### Layer Index Considerations
- Some components behave differently based on `layer_idx`
- AttentionBlock: sliding window applies only to even-indexed layers
- Use `layer_idx=1` to disable sliding window for simpler testing

## Trn1-Specific Optimizations

### Tensor Parallelism
- Current tests use `tp_degree=1` (no tensor parallelism)
- Future tests may need to validate TP functionality
- MLPBlock has sharding logic for distributed training

### Data Types
- Use `config.neuron_config.torch_dtype` for consistency
- Reference implementation typically uses `torch.bfloat16`
- Ensure dtype compatibility between Neuron and reference implementations

### Checkpoint Management
- Each test should specify a unique checkpoint path
- Example: `_CHECKPOINT_PATH = _ARTIFACTS_DIR / "neuron_{component}_checkpoint.pt"`
- Checkpoints stored in `tests/artifacts/` directory

### Build Process
- `build_module()` compiles the Neuron kernel
- This is a one-time overhead per test
- Provides the compiled module ready for inference

## Common Pitfalls and Debugging Tips

### 1. Shape Mismatches
**Problem**: Reference implementation expects `(n_tokens, hidden_size)` but Neuron expects `(batch_size, seq_len, hidden_size)`

**Solution**:
```python
# Flatten for reference
flat_input = inp.view(-1, hidden_size)
ref_output = reference_module(flat_input)

# Reshape reference output to match Neuron output format
reference_output = ref_output.view(batch_size, seq_len, hidden_size)
```

### 2. Missing position_ids
**Problem**: Attention modules need position information for RoPE

**Solution**:
```python
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
inputs = [(inp, position_ids)]
example_inputs = [(torch.zeros_like(sample), torch.zeros(batch_size, seq_len, dtype=torch.long))]
```

### 3. Sliding Window Issues
**Problem**: Sliding window attention can cause issues with short sequences

**Solution**: Use `layer_idx=1` (odd index) to disable sliding window:
```python
module_init_kwargs={
    "config": config,
    "layer_idx": 1,  # Disables sliding window
    "weight_init_value": _CONSTANT_INIT_VALUE,
}
```

### 4. Normalization Not Applied
**Problem**: Reference implementation may have normalization commented out

**Solution**: Ensure consistent normalization between reference and Neuron:
```python
# In gpt_oss.py AttentionBlock.forward():
# t = self.norm(x)  # May be commented out
t = x  # Using unnormalized input
```
Check if Neuron implementation matches this behavior.

### 5. Checkpoint Path Conflicts
**Problem**: Multiple tests using same checkpoint path

**Solution**: Use unique checkpoint paths per component:
```python
_CHECKPOINT_PATH = _ARTIFACTS_DIR / f"neuron_{component_name}_checkpoint.pt"
```

### 6. Device Mismatches
**Problem**: Reference on CPU, Neuron on accelerator

**Solution**: Keep reference on CPU, ensure Neuron tensors match dtype:
```python
reference_block = AttentionBlock(reference_config)  # CPU by default
# Neuron build_module handles device placement
```

### 7. Random Seed Inconsistency
**Problem**: Non-deterministic test failures

**Solution**: Always set seed before generating inputs:
```python
torch.manual_seed(0)
inp = torch.rand(...)
```

### 8. GQA (Grouped Query Attention) Head Expansion
**Problem**: Query/key/value head dimensions don't match

**Solution**: Reference implementation handles GQA with head expansion:
```python
# In AttentionBlock forward:
q = q.view(-1, num_key_value_heads, num_attention_heads // num_key_value_heads, head_dim)
k = k.view(-1, num_key_value_heads, head_dim)
```
Ensure Neuron implementation maintains this structure.

## Testing Workflow

### Step 1: Identify Component
Choose a component from `gpt_oss.py` that needs Neuron implementation.

### Step 2: Create Neuron Implementation
Implement `Neuron{ComponentName}` in `model.py` using NKI (Neuron Kernel Interface).

### Step 3: Create Test File
Create `tests/test_{component}.py` following the pattern in `test_attention.py`.

### Step 4: Implement Test Function
Follow the 5-step test structure pattern (see above).

### Step 5: Run and Debug
```bash
python tests/test_{component}.py
```

### Step 6: Validate Accuracy
The `validate_accuracy()` function will check:
- Output shapes match
- Numerical values are within tolerance
- No NaN or Inf values

## Quick Reference: Component-Specific Notes

### AttentionBlock
- Needs `position_ids` input
- Has sliding window behavior on even layers
- Uses GQA with head expansion
- RoPE applied to Q and K

### MLPBlock
- Expert-based architecture with gating
- Uses SwiGLU activation with clamping
- Has tensor parallel sharding logic
- Top-k expert selection per token

### RMSNorm
- Simple component, good starting point
- Numerically stable implementation needed
- Learnable scale parameter

### RotaryEmbedding
- Complex YaRN scaling logic
- NTK-by-parts interpolation
- Concentration factor calculation
- Applied to both Q and K tensors

## Next Steps

1. Review existing `test_attention.py` to understand the pattern
2. Identify the next component to implement (recommend starting with RMSNorm)
3. Create corresponding test file using this guide
4. Iterate on implementation until `validate_accuracy()` passes
5. Repeat for remaining components

## References

- Reference PyTorch implementation: `gpt_oss.py`
- Existing test pattern: `tests/test_attention.py`
- Test utilities: `tests/test_utils.py`
- Trn1 technical specs: `agent context/specs.md`
- Neuron implementation: `model.py`
