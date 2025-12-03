# Neuron Module Testing Strategy

## Overview

This document describes the testing strategy for validating custom Neuron modules against reference PyTorch implementations. The strategy is designed to ensure that modules compiled for AWS Trainium hardware produce outputs that match CPU-based reference implementations within acceptable numerical tolerances.

## Table of Contents

- [Core Principles](#core-principles)
- [Testing Framework](#testing-framework)
- [Weight Initialization Strategies](#weight-initialization-strategies)
- [Testing Workflow](#testing-workflow)
- [Key APIs and Utilities](#key-apis-and-utilities)
- [Example Test Structure](#example-test-structure)
- [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
- [Best Practices](#best-practices)

---

## Core Principles

### 1. **Isolated Component Testing**
Test individual components (attention blocks, MLP layers, etc.) in isolation before testing the full model. This enables:
- Faster debugging cycles
- Easier identification of accuracy issues
- Incremental validation of features

### 2. **Numerical Parity Validation**
The Neuron implementation must produce outputs that match the reference PyTorch implementation within tolerance:
- Typical tolerance: `rtol=1e-2, atol=1e-2` for bfloat16
- Use `validate_accuracy()` from `neuronx_distributed_inference.utils.testing`

### 3. **Weight Synchronization**
Both implementations must use **identical weights** to ensure fair comparison:
- Constant weights for debugging basic functionality
- Random weights for realistic validation

### 4. **Progressive Feature Testing**
Test features incrementally:
1. Basic functionality (no special features)
2. Normalized inputs
3. Sliding window attention
4. Learned parameters (e.g., sinks)
5. Full integration

---

## Testing Framework

### Required Dependencies

```python
import torch
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed_inference.models.config import InferenceConfig
```

### Key Components

1. **Reference Module**: Standard PyTorch implementation (runs on CPU)
2. **Neuron Module**: NKI-optimized implementation (compiled for Trainium)
3. **Configuration**: Shared config defining model architecture
4. **Test Inputs**: Identical inputs for both implementations

---

## Weight Initialization Strategies

### Strategy 1: Constant Weights (Simple Debugging)

**Use Case**: Initial development, debugging basic functionality

**Pros**:
- Reproducible
- Easy to reason about
- Fast to set up

**Cons**:
- May not expose numerical issues with realistic weight distributions
- Unrealistic activation patterns

**Implementation**:

```python
# Initialize with constant value
CONSTANT_INIT_VALUE = 0.1

def _fill_module_parameters(module: torch.nn.Module, value: float):
    """Fill all module parameters with a constant value."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)

# Build modules
neuron_block = build_module(
    NeuronAttentionBlock,
    example_inputs,
    tp_degree=1,
    module_init_kwargs={
        "config": config,
        "layer_idx": 0,
        "weight_init_value": CONSTANT_INIT_VALUE,  # Constant init
    },
    checkpoint_path="checkpoint.pt",
)

reference_block = ReferenceAttentionBlock(config)
_fill_module_parameters(reference_block, CONSTANT_INIT_VALUE)
```

### Strategy 2: Random Weights (Realistic Validation)

**Use Case**: Final validation, catching edge cases

**Pros**:
- Realistic weight distributions
- Exposes numerical stability issues
- Tests activation patterns similar to real models

**Cons**:
- More complex setup due to compilation constraints
- Requires checkpoint manipulation

**Implementation Flow**:

```python
def _build_with_random_weights(config, layer_idx=0, seed=42):
    """
    Build Neuron and reference blocks with matching random weights.

    Workflow:
    1. Compile Neuron module with uniform weights → generates checkpoint
    2. Create reference module with random weights (using seed)
    3. Load Neuron checkpoint and replace weights with reference weights
    4. Save modified checkpoint
    5. Recompile Neuron module with modified checkpoint
    """

    checkpoint_path = Path("checkpoint.pt")
    temp_checkpoint_path = Path("temp_checkpoint.pt")

    # Step 1: Initial compilation with uniform weights
    temp_neuron = build_module(
        NeuronAttentionBlock,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "layer_idx": layer_idx,
            "weight_init_value": 0.1,
        },
        checkpoint_path=str(temp_checkpoint_path),
    )
    del temp_neuron  # Free memory

    # Step 2: Create reference block with random weights
    torch.manual_seed(seed)
    reference_block = ReferenceAttentionBlock(config, layer_idx=layer_idx)

    # Step 3: Load checkpoint and sync weights
    neuron_state_dict = torch.load(temp_checkpoint_path)
    ref_state_dict = reference_block.state_dict()

    # Map reference keys to Neuron keys
    weight_mapping = {
        'qkv.weight': 'qkv_proj.Wqkv.weight',
        'qkv.bias': 'qkv_proj.Wqkv.bias',
        'out.weight': 'o_proj.o_proj.weight',
        'out.bias': 'o_proj.o_proj.bias',
    }

    for ref_key, neuron_key in weight_mapping.items():
        if ref_key in ref_state_dict and neuron_key in neuron_state_dict:
            ref_tensor = ref_state_dict[ref_key]
            # Convert dtype if necessary
            if ref_tensor.dtype != neuron_state_dict[neuron_key].dtype:
                ref_tensor = ref_tensor.to(dtype=neuron_state_dict[neuron_key].dtype)
            neuron_state_dict[neuron_key] = ref_tensor

    # Step 4: Save modified checkpoint
    torch.save(neuron_state_dict, checkpoint_path)

    # Step 5: Recompile with modified checkpoint
    neuron_block = build_module(
        NeuronAttentionBlock,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "layer_idx": layer_idx,
        },
        checkpoint_path=str(checkpoint_path),
    )

    return neuron_block, reference_block
```

**Why This Complex Flow?**

The Neuron compilation process has a critical constraint: you **cannot create multiple uncompiled Neuron blocks** in the same process because each attempts to initialize the distributed process group. The checkpoint-based approach works around this limitation.

---

## Testing Workflow

### Step 1: Create Configuration

```python
def _make_test_config(batch_size=2, seq_len=4096):
    """Create test configuration matching your model architecture."""
    neuron_config = NeuronConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        tp_degree=1,
        torch_dtype=torch.bfloat16,
    )

    return InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=2880,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        # ... other model hyperparameters
    )
```

### Step 2: Build Modules

```python
config = _make_test_config()

# Build with random weights
neuron_block, reference_block = _build_with_random_weights(
    config,
    layer_idx=0,
    checkpoint_name="test.pt",
    seed=42
)
```

### Step 3: Prepare Inputs

```python
batch_size = config.neuron_config.batch_size
seq_len = config.neuron_config.seq_len
hidden_size = config.hidden_size

# Create test inputs
torch.manual_seed(42)
hidden_states = torch.rand(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

inputs = [(hidden_states, position_ids)]
```

### Step 4: Generate Reference Output

```python
with torch.no_grad():
    # Reference expects flattened input: (batch_size * seq_len, hidden_size)
    flat_input = hidden_states.view(-1, hidden_size)
    ref_output = reference_block(flat_input)
    reference_output = ref_output.view(batch_size, seq_len, hidden_size)

print(f"Reference output shape: {reference_output.shape}")
print(f"Reference output sample: {reference_output[0, 0, :4]}")
```

### Step 5: Validate Against Neuron Implementation

```python
# validate_accuracy runs neuron_block(inputs) and compares against expected outputs
validate_accuracy(
    neuron_block,
    inputs=inputs,
    expected_outputs=[reference_output]
)

print("✓ Test PASSED: Outputs match within tolerance!")
```

---

## Key APIs and Utilities

### 1. `build_module` (neuronx_distributed_inference.utils.testing)

Compiles a Neuron module for Trainium hardware.

```python
neuron_module = build_module(
    module_class,                    # Your NeuronModule class
    example_inputs,                  # List of example input tuples
    tp_degree=1,                     # Tensor parallelism degree
    module_init_kwargs={},           # Constructor arguments
    checkpoint_path="checkpoint.pt", # Path to save/load weights
)
```

**Key Parameters**:
- `example_inputs`: List of input tuples matching your forward signature
  - Example: `[(hidden_states, position_ids)]`
- `checkpoint_path`: If exists, loads weights; otherwise creates new checkpoint
- `module_init_kwargs`: Passed to module constructor

**Returns**: Compiled module ready for inference

### 2. `validate_accuracy` (neuronx_distributed_inference.utils.testing)

Validates Neuron module outputs against expected outputs.

```python
validate_accuracy(
    neuron_module,
    inputs=[(input1, input2)],           # List of input tuples
    expected_outputs=[expected_output1], # List of expected tensors
    rtol=1e-2,                           # Relative tolerance (default for bfloat16)
    atol=1e-2,                           # Absolute tolerance
)
```

**Behavior**:
- Runs `neuron_module(*inputs[i])` for each input tuple
- Compares outputs against `expected_outputs[i]`
- Raises `AssertionError` if tolerance exceeded
- Prints detailed comparison metrics

### 3. Weight Synchronization Utilities

#### `_fill_module_parameters` (tests/test_utils.py)

```python
def _fill_module_parameters(module: torch.nn.Module, value: float):
    """Fill all parameters with constant value."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)
```

#### `_sync_reference_weights_to_neuron` (tests/test_utils.py)

```python
def _sync_reference_weights_to_neuron(reference_block, neuron_block, layer_idx=0):
    """
    Copy weights from reference to Neuron module.

    Handles:
    - Name mapping between implementations
    - Dtype conversion (e.g., float32 → bfloat16)
    - Shape verification
    - Special cases (e.g., one-to-many mappings)
    """
    # ... (see test_utils.py for full implementation)
```

**Weight Mapping Example**:

```python
weight_mapping = {
    # Reference key → Neuron key
    'qkv.weight': 'qkv_proj.Wqkv.weight',
    'qkv.bias': 'qkv_proj.Wqkv.bias',
    'out.weight': 'o_proj.o_proj.weight',
    'out.bias': 'o_proj.o_proj.bias',

    # One-to-many mapping (sinks replicated for CTE and TKG)
    'sinks': ['learned_sinks.sink', 'tkg_learned_sinks.sink'],
}
```

### 4. Configuration Conversion

#### `_get_ref_config` (tests/test_utils.py)

```python
def _get_ref_config(neuron_inference_config):
    """Convert Neuron config to reference model config."""
    return ReferenceModelConfig(
        num_hidden_layers=neuron_inference_config.num_hidden_layers,
        hidden_size=neuron_inference_config.hidden_size,
        num_attention_heads=neuron_inference_config.num_attention_heads,
        # ... map all required fields
    )
```

---

## Example Test Structure

### Minimal Test Template

```python
import torch
from pathlib import Path
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

# Test constants
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def test_my_neuron_module():
    """Test MyNeuronModule against reference implementation."""

    # 1. Create configuration
    config = create_test_config()

    # 2. Build modules with matching weights
    neuron_module, reference_module = build_modules_with_random_weights(
        config,
        checkpoint_name="test_my_module.pt"
    )

    # 3. Prepare inputs
    inputs = create_test_inputs(config)

    # 4. Generate reference output
    with torch.no_grad():
        reference_output = reference_module(*inputs[0])

    # 5. Validate
    validate_accuracy(
        neuron_module,
        inputs=inputs,
        expected_outputs=[reference_output]
    )

    print("✓ Test PASSED!")

if __name__ == "__main__":
    test_my_neuron_module()
```

### Progressive Test Suite

```python
def run_test_suite():
    """Run tests with increasing complexity."""

    tests = [
        ("Basic functionality", test_basic),
        ("With normalized inputs", test_normalized_inputs),
        ("With special feature X", test_feature_x),
        ("Full integration", test_integration),
    ]

    for name, test_fn in tests:
        try:
            print(f"\n{'='*80}")
            print(f"Running: {name}")
            print('='*80)
            test_fn()
            print(f"✓ {name} PASSED")
        except Exception as e:
            print(f"✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
```

---

## Common Pitfalls and Solutions

### 1. Process Group Initialization Error

**Problem**: Creating multiple uncompiled Neuron modules raises:
```
RuntimeError: Process group already initialized
```

**Solution**: Use the checkpoint-based random weight strategy (see [Strategy 2](#strategy-2-random-weights-realistic-validation))

### 2. Shape Mismatches

**Problem**: Reference module expects different input shapes than Neuron module.

**Solution**:
- Reference attention often expects flattened inputs: `(batch * seq, hidden)`
- Neuron attention expects: `(batch, seq, hidden)`
- Add reshaping logic in your test:

```python
# For reference
flat_input = hidden_states.view(-1, hidden_size)
ref_output = reference_block(flat_input)
reference_output = ref_output.view(batch_size, seq_len, hidden_size)

# For Neuron
neuron_output = neuron_block(hidden_states, position_ids)
```

### 3. Weight Name Mismatches

**Problem**: `KeyError` when syncing weights between reference and Neuron modules.

**Solution**: Print state dict keys and create explicit mapping:

```python
print("Reference keys:", reference_module.state_dict().keys())
print("Neuron keys:", neuron_module.state_dict().keys())

# Create mapping
weight_mapping = {
    'ref_key': 'neuron_key',
    # ...
}
```

### 4. Dtype Mismatches

**Problem**: Reference uses `float32`, Neuron uses `bfloat16`.

**Solution**: Convert during weight sync:

```python
if ref_tensor.dtype != neuron_state_dict[neuron_key].dtype:
    ref_tensor = ref_tensor.to(dtype=neuron_state_dict[neuron_key].dtype)
neuron_state_dict[neuron_key] = ref_tensor
```

### 5. Numerical Tolerance Issues

**Problem**: `validate_accuracy` fails with small differences.

**Diagnosis**:
```python
diff = (neuron_output - reference_output).abs()
print(f"Max abs diff: {diff.max()}")
print(f"Mean abs diff: {diff.mean()}")
print(f"Relative error: {(diff / reference_output.abs().clamp(min=1e-8)).max()}")
```

**Solutions**:
- Adjust tolerance: `rtol=2e-2, atol=2e-2`
- Check for weight sync issues
- Verify both modules use same dtype
- Check for accumulation errors (e.g., in long sequences)

### 6. Missing Position IDs

**Problem**: Neuron attention requires `position_ids` but reference doesn't.

**Solution**: Wrap reference module:

```python
class ReferenceWrapper(nn.Module):
    def __init__(self, reference_block):
        super().__init__()
        self.block = reference_block

    def forward(self, hidden_states, position_ids):
        # Reference doesn't use position_ids directly (uses internal counter)
        return self.block(hidden_states)
```

---

## Best Practices

### 1. **Start Simple**
- Begin with constant weights and basic inputs
- Add complexity incrementally
- Validate each feature in isolation

### 2. **Use Meaningful Seeds**
```python
torch.manual_seed(42)  # For weights
torch.manual_seed(67890)  # Different seed for inputs
```

### 3. **Print Diagnostic Information**
```python
print(f"Input shape: {inp.shape}")
print(f"Input sample: {inp[0, 0, :4]}")
print(f"Reference output sample: {ref_output[0, 0, :4]}")
print(f"Reference output stats: mean={ref_output.mean():.6f}, std={ref_output.std():.6f}")
```

### 4. **Clean Up Artifacts**
```python
checkpoint_path = ARTIFACTS_DIR / "test.pt"
if checkpoint_path.exists():
    checkpoint_path.unlink()  # Remove old checkpoint
```

### 5. **Test Layer-Specific Behavior**
Some features (e.g., sliding window) may be layer-dependent:

```python
# Test with even layer (sliding window enabled)
test_with_layer_idx(layer_idx=0)

# Test with odd layer (sliding window disabled)
test_with_layer_idx(layer_idx=1)
```

### 6. **Use Small Configs for Fast Iteration**
```python
def _make_tiny_config():
    """Tiny config for fast testing during development."""
    neuron_config = NeuronConfig(
        batch_size=2,
        seq_len=128,  # Much smaller than production
        tp_degree=1,
    )
    return InferenceConfig(
        neuron_config=neuron_config,
        hidden_size=64,   # Smaller dimensions
        num_attention_heads=4,
        # ...
    )
```

### 7. **Document Test Coverage**
```python
"""
Test Coverage:
- [x] Basic attention without special features
- [x] Attention with normalized inputs
- [x] Sliding window attention
- [ ] Learned sinks (blocked by compiler issue #123)
- [x] Random weight initialization
"""
```

### 8. **Handle Known Failures Gracefully**
```python
def test_feature_with_known_issue():
    """Test feature X (currently fails due to compiler bug)."""
    print("⚠ Test SKIPPED: Compiler bug #456")
    return

    # Test code here...
```

---

## Advanced Topics

### Testing with Distributed Training

For multi-device testing (tp_degree > 1):

```python
# Initialize process group
import torch.distributed as dist
if not dist.is_initialized():
    dist.init_process_group(backend='xla', init_method='pjrt://')

# Build with tensor parallelism
neuron_module = build_module(
    NeuronAttentionBlock,
    example_inputs,
    tp_degree=8,  # 8-way tensor parallelism
    module_init_kwargs={...},
)
```

### Testing Flash Attention Kernels

When testing custom NKI kernels:

```python
# Enable debug logging for NKI compilation
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("Neuron").setLevel(logging.DEBUG)

# Validate kernel outputs
validate_accuracy(
    neuron_block,
    inputs=inputs,
    expected_outputs=[reference_output],
    rtol=5e-2,  # Flash attention may need higher tolerance
    atol=5e-2,
)
```

### Performance Profiling

```python
import time

# Warmup
for _ in range(5):
    _ = neuron_module(*inputs[0])

# Benchmark
start = time.perf_counter()
for _ in range(100):
    _ = neuron_module(*inputs[0])
elapsed = time.perf_counter() - start

print(f"Average latency: {elapsed/100*1000:.2f} ms")
```

---

## Summary

This testing strategy enables reliable development of Neuron modules by:

1. **Ensuring numerical parity** with reference PyTorch implementations
2. **Supporting both constant and random weight initialization** for different testing phases
3. **Providing clear workflows** for compilation, weight synchronization, and validation
4. **Handling platform-specific constraints** (e.g., process group initialization)

Use this strategy to confidently develop and validate custom modules for AWS Trainium, catching issues early in the development cycle before full model integration.

---

## References

- [neuronx_distributed_inference documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NKI Kernel Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/index.html)
- Example test: [tests/attention/test_attention_phase1.py](../tests/attention/test_attention_phase1.py)
- Utility functions: [tests/test_utils.py](../tests/test_utils.py)
