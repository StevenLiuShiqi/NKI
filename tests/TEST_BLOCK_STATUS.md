# Block Test Status

## Summary
Created a test file `tests/test_block.py` that compares the outputs of `NeuronGPTOSSBlock` and `TransformerBlock` (reference implementation).

## Test Structure
- Uses `build_module` and `validate_accuracy` from `neuronx_distributed_inference.utils.testing`
- Creates a `BlockWrapper` class to extract just the hidden states from the block output
- Initializes all parameters to a constant value (0.5) for deterministic testing
- Uses layer_idx=1 to disable sliding window attention

## Current Issue
The test cannot run due to an incompatibility in `src/model.py`:

### Problem
In `NeuronGPTOSSBlock.forward()` (line 407 in model.py):
```python
hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(...)
```

The code expects `self_attn` to return 4 values, but `NeuronGPTOSSAttentionBlock.forward()` (line 377 in model.py) only returns 1 value:
```python
return tuple(output)[0]  # Returns only the first element
```

### Solution Needed
The `NeuronGPTOSSAttentionBlock.forward()` method needs to return the full tuple instead of just the first element:
```python
# Change line 377 from:
return tuple(output)[0]

# To:
return output
```

## Additional Observations
1. The reference `MLPBlock` includes a residual connection (`return x + t`)
2. The reference `AttentionBlock` does NOT include a residual connection
3. The `NeuronGPTOSSBlock` currently adds residuals for both attention and MLP externally
4. For better matching with the reference, the MLP should include its own residual and attention should not

## Test File Location
`tests/test_block.py`

## How to Run (once model.py is fixed)
```bash
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
python tests/test_block.py
```
