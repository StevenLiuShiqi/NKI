import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import (
    GPTOSSInferenceConfig,
    NeuronGPTOSSConfig,
    NeuronGPTOSSMLPBlock,
)
from src.gpt_oss import MLPBlock
from src.moe_classes import NeuronMLPBlock

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from test_utils import _make_tiny_inference_config, _get_ref_config, _fill_module_parameters, _make_original_inference_config

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_PATH = _ARTIFACTS_DIR / "neuron_mlp_checkpoint.pt"
_CONSTANT_INIT_VALUE = 0.5


def _sync_moe_weights_reference_to_neuron(reference_block, checkpoint_path, config):
    """
    Sync weights from reference MLPBlock to Neuron checkpoint.

    This uses the weight mapping from model.py's weight conversion:
    - router.weight -> ffn.router.linear_router.weight
    - router.bias -> ffn.router.linear_router.bias
    - mlp1_weight (gate_up_proj) -> ffn.expert_mlps.mlp_op.gate_up_proj.weight
    - mlp1_bias -> ffn.expert_mlps.mlp_op.gate_up_proj.bias
    - mlp2_weight (down_proj) -> ffn.expert_mlps.mlp_op.down_proj.weight
    - mlp2_bias -> ffn.expert_mlps.mlp_op.down_proj.bias

    Args:
        reference_block: MLPBlock from gpt_oss.py with random weights
        checkpoint_path: Path to neuron checkpoint to modify
        config: GPTOSSInferenceConfig

    Returns:
        Number of weight tensors copied
    """
    # Load the neuron checkpoint
    neuron_state_dict = torch.load(checkpoint_path)
    ref_state_dict = reference_block.state_dict()

    print(f"\n{'='*80}")
    print(f"Syncing MOE weights from reference to neuron checkpoint")
    print(f"{'='*80}")

    print(f"\nReference block keys:")
    for k in sorted(ref_state_dict.keys()):
        print(f"  - {k}: {ref_state_dict[k].shape}")

    print(f"\nNeuron checkpoint keys:")
    for k in sorted(neuron_state_dict.keys()):
        print(f"  - {k}: {neuron_state_dict[k].shape}")

    # Weight mapping based on model.py conversion logic
    # Note: The checkpoint keys don't have the "ffn." prefix
    weight_mapping = {
        'norm.scale': 'norm.weight',  # RMSNorm parameter
        'gate.weight': 'router.linear_router.weight',
        'gate.bias': 'router.linear_router.bias',
        'mlp1_weight': 'expert_mlps.mlp_op.gate_up_proj.weight',
        'mlp1_bias': 'expert_mlps.mlp_op.gate_up_proj.bias',
        'mlp2_weight': 'expert_mlps.mlp_op.down_proj.weight',
        'mlp2_bias': 'expert_mlps.mlp_op.down_proj.bias',
    }

    updated_count = 0
    for ref_key, neuron_key in weight_mapping.items():
        if ref_key not in ref_state_dict:
            print(f"  [SKIP] {ref_key} not found in reference state")
            continue

        if neuron_key not in neuron_state_dict:
            print(f"  [SKIP] {neuron_key} not found in neuron state")
            continue

        ref_tensor = ref_state_dict[ref_key]

        # Convert dtype if needed
        ref_converted = ref_tensor
        if ref_converted.dtype != neuron_state_dict[neuron_key].dtype:
            ref_converted = ref_converted.to(dtype=neuron_state_dict[neuron_key].dtype)

        # Handle weight dimension transformations
        # Reference MLPBlock uses: (E, I, H) for mlp1 and (E, H, I) for mlp2
        # Neuron expects: (E, H, I) for gate_up and (E, I, H) for down
        if 'mlp1_weight' in ref_key:
            # mlp1_weight: (E, I*2, H) -> gate_up_proj: (E, H, I*2)
            ref_converted = ref_converted.transpose(1, 2).contiguous()
            print(f"  [TRANSPOSE] {ref_key}: {ref_tensor.shape} -> {ref_converted.shape}")
        elif 'mlp2_weight' in ref_key:
            # mlp2_weight: (E, H, I) -> down_proj: (E, I, H)
            ref_converted = ref_converted.transpose(1, 2).contiguous()
            print(f"  [TRANSPOSE] {ref_key}: {ref_tensor.shape} -> {ref_converted.shape}")

        # Verify shapes match after transformation
        if ref_converted.shape != neuron_state_dict[neuron_key].shape:
            print(f"  [WARN] Shape mismatch for {ref_key} -> {neuron_key}: "
                  f"{ref_converted.shape} vs {neuron_state_dict[neuron_key].shape}")
            continue

        neuron_state_dict[neuron_key] = ref_converted
        updated_count += 1
        print(f"  [COPY] {ref_key} -> {neuron_key} (shape: {ref_converted.shape})")

    # Save modified checkpoint
    torch.save(neuron_state_dict, checkpoint_path)
    print(f"\n{'='*80}")
    print(f"Saved {updated_count} weight tensors to: {checkpoint_path}")
    print(f"{'='*80}\n")

    return updated_count


def _build_moe_blocks_with_random_weights(config, checkpoint_name="moe_random.pt", seed=42):
    """
    Build both Neuron and reference MOE blocks with matching RANDOM weights.

    Similar to attention tests, this:
    1. First compiles with uniform weights to generate a checkpoint
    2. Creates reference block with random weights
    3. Loads checkpoint and syncs weights from reference
    4. Recompiles with the modified checkpoint

    Args:
        config: GPTOSSInferenceConfig
        checkpoint_name: Name for the checkpoint file
        seed: Random seed for reproducibility

    Returns:
        (neuron_block, reference_block) with matching random weights
    """
    checkpoint_path = _ARTIFACTS_DIR / checkpoint_name
    temp_checkpoint_path = _ARTIFACTS_DIR / f"temp_{checkpoint_name}"

    # Clean up old checkpoints
    for path in [checkpoint_path, temp_checkpoint_path]:
        if path.exists():
            path.unlink()

    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    # Create example inputs for build_module
    example_inputs = [
        (torch.zeros(batch_size * seq_len, hidden_size, dtype=torch.bfloat16),)
    ]

    print(f"\n{'='*80}")
    print(f"Building MOE blocks with RANDOM weights (seed={seed})")
    print(f"{'='*80}")

    # Step 1: Build with uniform weights to generate initial checkpoint
    print(f"\nStep 1: Creating initial neuron block with uniform weights...")
    _temp_neuron_block = build_module(
        NeuronGPTOSSMLPBlock,
        example_inputs,
        tp_degree=config.neuron_config.tp_degree,
        module_init_kwargs={
            "config": config,
            "weight_init_value": _CONSTANT_INIT_VALUE,
        },
        checkpoint_path=str(temp_checkpoint_path),
    )
    print(f"  Generated checkpoint at: {temp_checkpoint_path}")
    del _temp_neuron_block  # Free memory

    # Step 2: Create reference block with random weights
    torch.manual_seed(seed)
    print(f"\nStep 2: Creating reference block with random weights...")
    reference_config = _get_ref_config(config)
    reference_block = MLPBlock(reference_config)

    # Initialize parameters that are created with torch.empty() but not initialized
    # MLPBlock creates mlp1_weight, mlp1_bias, mlp2_weight, mlp2_bias with torch.empty()
    # We need to initialize them to avoid NaN and uninitialized memory
    torch.manual_seed(seed)  # Reset seed for consistent initialization
    torch.nn.init.normal_(reference_block.mlp1_weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(reference_block.mlp1_bias)
    torch.nn.init.normal_(reference_block.mlp2_weight, mean=0.0, std=0.02)
    torch.nn.init.zeros_(reference_block.mlp2_bias)
    torch.nn.init.ones_(reference_block.norm.scale)

    # gate.weight and gate.bias are initialized by torch.nn.Linear automatically

    print(f"  Reference block created with random initialization")

    # Step 3: Load checkpoint and sync weights
    print(f"\nStep 3: Loading checkpoint and syncing weights...")
    updated_count = _sync_moe_weights_reference_to_neuron(
        reference_block, temp_checkpoint_path, config
    )

    # Step 4: Save as final checkpoint
    print(f"\nStep 4: Saving final checkpoint...")
    torch.save(torch.load(temp_checkpoint_path), checkpoint_path)
    print(f"  Saved to: {checkpoint_path}")

    # Step 5: Compile with modified checkpoint
    print(f"\nStep 5: Compiling neuron block with modified checkpoint...")
    neuron_block = build_module(
        NeuronGPTOSSMLPBlock,
        example_inputs,
        tp_degree=config.neuron_config.tp_degree,
        module_init_kwargs={
            "config": config,
        },
        checkpoint_path=str(checkpoint_path),
    )

    # Clean up temp checkpoint
    if temp_checkpoint_path.exists():
        temp_checkpoint_path.unlink()

    print(f"\n{'='*80}")
    print(f"MOE blocks ready with matching random weights ({updated_count} tensors synced)")
    print(f"{'='*80}\n")

    return neuron_block, reference_block


def test_validate_accuracy_with_random_weights():
    """
    Test MOE block with randomized weights.

    This validates that:
    1. Weight syncing mechanism works correctly for MOE
    2. Both implementations produce identical outputs with realistic random weights
    3. The mapping from reference to neuron weights is correct
    """
    print("\n" + "="*80)
    print("Test: MOE Block with Random Weights")
    print("="*80)

    config = _make_original_inference_config()

    # Build blocks with random weights
    neuron_model, module_cpu = _build_moe_blocks_with_random_weights(
        config, checkpoint_name="moe_random_weights.pt", seed=12345
    )

    # Create test inputs with different seed
    torch.manual_seed(67890)
    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    sample = torch.randn(
        batch_size * seq_len, hidden_size, dtype=config.neuron_config.torch_dtype
    )
    inputs = [(sample,)]

    print(f"\nInput shape: {sample.shape}")
    print(f"Input dtype: {sample.dtype}")
    print(f"Input sample: {sample[0, :4]}")

    # Reference forward pass (CPU)
    with torch.no_grad():
        # Debug: Check if weights have reasonable values
        print(f"\nReference weights check:")
        print(f"  gate.weight range: [{module_cpu.gate.weight.min():.4f}, {module_cpu.gate.weight.max():.4f}]")
        print(f"  mlp1_weight range: [{module_cpu.mlp1_weight.min():.4f}, {module_cpu.mlp1_weight.max():.4f}]")
        print(f"  mlp2_weight range: [{module_cpu.mlp2_weight.min():.4f}, {module_cpu.mlp2_weight.max():.4f}]")
        print(f"  norm.scale: {module_cpu.norm.scale[:4]}")

        expected_output = module_cpu(sample)

    print(f"\nReference output shape: {expected_output.shape}")
    print(f"Reference output sample: {expected_output[0, :4]}")
    if not torch.isnan(expected_output).any():
        print(f"Reference output mean: {expected_output.mean():.6f}, std: {expected_output.std():.6f}")
    else:
        print(f"Reference output contains NaNs!")
        nan_count = torch.isnan(expected_output).sum().item()
        print(f"  NaN count: {nan_count} / {expected_output.numel()}")

    # Validate against Neuron implementation
    validate_accuracy(neuron_model, inputs, expected_outputs=[expected_output])

    print("✓ Test PASSED: MOE with random weights matches!")


def test_validate_accuracy_basic_module():
    """Original test with uniform weights."""
    print("\n" + "="*80)
    print("Test: MOE Block with Uniform Weights")
    print("="*80)

    config = _make_original_inference_config()

    sample = torch.randn(12, config.hidden_size, dtype=config.neuron_config.torch_dtype)
    inputs = [(sample,)]
    example_inputs = [(torch.zeros_like(sample),)]

    reference_config = _get_ref_config(config=config)

    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()

    neuron_model = build_module(
        NeuronGPTOSSMLPBlock,
        example_inputs,
        tp_degree=8,
        module_init_kwargs={
            "config": config,
            "weight_init_value": _CONSTANT_INIT_VALUE,
        },
        checkpoint_path=str(_CHECKPOINT_PATH),
    )

    module_cpu = MLPBlock(
        config=reference_config,
    )

    _fill_module_parameters(module_cpu, _CONSTANT_INIT_VALUE)

    def cpu_forward(x):
        return module_cpu(x)

    with torch.no_grad():
        expected_output = cpu_forward(*inputs[0])


    validate_accuracy(neuron_model, inputs, expected_outputs=[expected_output])

    print("✓ Test PASSED: MOE with uniform weights matches!")


if __name__ == "__main__":
    # Run the random weights test (uses tiny config with tp_degree=1)
    # Note: test_validate_accuracy_basic_module uses original config with tp_degree=8
    # which requires 8 Neuron cores - uncomment if running on a larger instance

    # try:
    #     test_validate_accuracy_basic_module()
    # except Exception as e:
    #     print(f"✗ Uniform weights test FAILED: {e}")
    #     import traceback
    #     traceback.print_exc()

    try:
        test_validate_accuracy_with_random_weights()
    except Exception as e:
        print(f"✗ Random weights test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("ALL MOE TESTS COMPLETED")
    print("="*80)
