import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import (
    GPTOSSInferenceConfig,
    NeuronGPTOSSConfig,
)
from src.openai.gpt_oss import ModelConfig

from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from test_utils import _make_tiny_inference_config, _make_original_inference_config, _get_ref_config, _fill_module_parameters

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class NeuronEmbeddingWrapper(torch.nn.Module):
    """Wrapper for testing ParallelEmbedding in isolation."""

    def __init__(self, config: GPTOSSInferenceConfig, weight_init_value=None):
        super().__init__()
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )

        # Initialize with constant value if provided
        if weight_init_value is not None:
            with torch.no_grad():
                self.embed_tokens.weight.fill_(weight_init_value)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class NeuronLMHeadWrapper(torch.nn.Module):
    """Wrapper for testing ColumnParallelLinear (lm_head) in isolation."""

    def __init__(self, config: GPTOSSInferenceConfig, weight_init_value=None):
        super().__init__()
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=True,  # Always gather for testing
            bias=False,
        )

        # Initialize with constant value if provided
        if weight_init_value is not None:
            with torch.no_grad():
                self.lm_head.weight.fill_(weight_init_value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class ReferenceEmbeddingWrapper(torch.nn.Module):
    """Reference embedding using standard PyTorch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=torch.bfloat16
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Ensure output is in bfloat16 to match Neuron implementation
        output = self.embedding(input_ids)
        if output.dtype != torch.bfloat16:
            output = output.to(torch.bfloat16)
        return output


class ReferenceLMHeadWrapper(torch.nn.Module):
    """Reference lm_head using standard PyTorch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=torch.bfloat16,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Ensure output is in bfloat16 to match Neuron implementation
        output = self.unembedding(hidden_states)
        if output.dtype != torch.bfloat16:
            output = output.to(torch.bfloat16)
        return output


def _build_embedding_with_random_weights(
    config,
    checkpoint_name="embedding_checkpoint.pt",
    seed=42,
):
    """
    Build both Neuron and reference embedding blocks with matching random weights.

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

    vocab_size = config.vocab_size
    hidden_size = config.hidden_size

    # Create example inputs: token IDs
    example_inputs = [
        (torch.zeros(12, dtype=torch.int32),)
    ]

    print(f"\n{'='*80}")
    print(f"Building Embedding with RANDOM weights (seed={seed})")
    print(f"{'='*80}")

    # Step 1: Build with uniform weights to generate initial checkpoint
    print(f"\nStep 1: Creating initial neuron embedding with uniform weights...")
    _temp_neuron_block = build_module(
        NeuronEmbeddingWrapper,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "weight_init_value": 0.5,
        },
        checkpoint_path=str(temp_checkpoint_path),
    )
    print(f"  Generated checkpoint at: {temp_checkpoint_path}")
    del _temp_neuron_block

    # Step 2: Create reference block with random weights
    torch.manual_seed(seed)
    print(f"\nStep 2: Creating reference embedding with random weights...")
    reference_config = _get_ref_config(config=config)
    reference_block = ReferenceEmbeddingWrapper(config=reference_config)

    # Initialize with random values
    with torch.no_grad():
        reference_block.embedding.weight.data = torch.randn_like(
            reference_block.embedding.weight,
            dtype=torch.float32
        ).to(torch.bfloat16)
    print(f"  Initialized embedding.weight with shape {reference_block.embedding.weight.shape}")

    # Step 3: Load checkpoint and sync weights
    print(f"\nStep 3: Loading checkpoint and syncing weights...")
    neuron_state_dict = torch.load(temp_checkpoint_path)
    ref_state_dict = reference_block.state_dict()

    print(f"\n  Reference state dict:")
    for k, v in ref_state_dict.items():
        print(f"    {k}: {v.shape}")
    print(f"\n  Neuron state dict:")
    for k, v in neuron_state_dict.items():
        print(f"    {k}: {v.shape}")

    # Weight mapping
    weight_mapping = {
        'embedding.weight': 'embed_tokens.weight',
    }

    updated_count = 0
    for ref_key, neuron_key in weight_mapping.items():
        if ref_key not in ref_state_dict:
            print(f"  [SKIP] {ref_key} not found in reference state")
            continue

        ref_tensor = ref_state_dict[ref_key]

        if neuron_key in neuron_state_dict:
            # Convert dtype if needed
            ref_converted = ref_tensor.clone()
            if ref_converted.dtype != neuron_state_dict[neuron_key].dtype:
                ref_converted = ref_converted.to(dtype=neuron_state_dict[neuron_key].dtype)

            # Check shape match
            if ref_converted.shape != neuron_state_dict[neuron_key].shape:
                print(f"  [ERROR] Shape mismatch for {ref_key} -> {neuron_key}:")
                print(f"    Reference: {ref_converted.shape}")
                print(f"    Neuron: {neuron_state_dict[neuron_key].shape}")
                continue

            neuron_state_dict[neuron_key] = ref_converted
            updated_count += 1
            print(f"  [COPY] {ref_key} -> {neuron_key} (shape: {ref_converted.shape})")
        else:
            print(f"  [WARN] {neuron_key} not found in neuron state")

    # Step 4: Save modified checkpoint
    print(f"\nStep 4: Saving modified checkpoint with random weights...")
    torch.save(neuron_state_dict, checkpoint_path)
    print(f"  Saved {updated_count} weight tensors to: {checkpoint_path}")

    # Step 5: Compile with modified checkpoint
    print(f"\nStep 5: Compiling neuron embedding with modified checkpoint...")
    neuron_block = build_module(
        NeuronEmbeddingWrapper,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
        },
        checkpoint_path=str(checkpoint_path),
    )

    print(f"\n{'='*80}")
    print(f"Embedding ready with matching random weights")
    print(f"{'='*80}\n")

    return neuron_block, reference_block


def _build_lmhead_with_random_weights(
    config,
    checkpoint_name="lmhead_checkpoint.pt",
    seed=42,
):
    """
    Build both Neuron and reference lm_head blocks with matching random weights.

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

    vocab_size = config.vocab_size
    hidden_size = config.hidden_size

    # Create example inputs: hidden states
    example_inputs = [
        (torch.zeros(12, hidden_size, dtype=torch.bfloat16),)
    ]

    print(f"\n{'='*80}")
    print(f"Building LM Head with RANDOM weights (seed={seed})")
    print(f"{'='*80}")

    # Step 1: Build with uniform weights to generate initial checkpoint
    print(f"\nStep 1: Creating initial neuron lm_head with uniform weights...")
    _temp_neuron_block = build_module(
        NeuronLMHeadWrapper,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "weight_init_value": 0.5,
        },
        checkpoint_path=str(temp_checkpoint_path),
    )
    print(f"  Generated checkpoint at: {temp_checkpoint_path}")
    del _temp_neuron_block

    # Step 2: Create reference block with random weights
    torch.manual_seed(seed)
    print(f"\nStep 2: Creating reference lm_head with random weights...")
    reference_config = _get_ref_config(config=config)
    reference_block = ReferenceLMHeadWrapper(config=reference_config)

    # Initialize with random values
    with torch.no_grad():
        reference_block.unembedding.weight.data = torch.randn_like(
            reference_block.unembedding.weight,
            dtype=torch.float32
        ).to(torch.bfloat16)
    print(f"  Initialized unembedding.weight with shape {reference_block.unembedding.weight.shape}")

    # Step 3: Load checkpoint and sync weights
    print(f"\nStep 3: Loading checkpoint and syncing weights...")
    neuron_state_dict = torch.load(temp_checkpoint_path)
    ref_state_dict = reference_block.state_dict()

    print(f"\n  Reference state dict:")
    for k, v in ref_state_dict.items():
        print(f"    {k}: {v.shape}")
    print(f"\n  Neuron state dict:")
    for k, v in neuron_state_dict.items():
        print(f"    {k}: {v.shape}")

    # Weight mapping
    weight_mapping = {
        'unembedding.weight': 'lm_head.weight',
    }

    updated_count = 0
    for ref_key, neuron_key in weight_mapping.items():
        if ref_key not in ref_state_dict:
            print(f"  [SKIP] {ref_key} not found in reference state")
            continue

        ref_tensor = ref_state_dict[ref_key]

        if neuron_key in neuron_state_dict:
            # Convert dtype if needed
            ref_converted = ref_tensor.clone()
            if ref_converted.dtype != neuron_state_dict[neuron_key].dtype:
                ref_converted = ref_converted.to(dtype=neuron_state_dict[neuron_key].dtype)

            # Check shape match
            if ref_converted.shape != neuron_state_dict[neuron_key].shape:
                print(f"  [ERROR] Shape mismatch for {ref_key} -> {neuron_key}:")
                print(f"    Reference: {ref_converted.shape}")
                print(f"    Neuron: {neuron_state_dict[neuron_key].shape}")
                continue

            neuron_state_dict[neuron_key] = ref_converted
            updated_count += 1
            print(f"  [COPY] {ref_key} -> {neuron_key} (shape: {ref_converted.shape})")
        else:
            print(f"  [WARN] {neuron_key} not found in neuron state")

    # Step 4: Save modified checkpoint
    print(f"\nStep 4: Saving modified checkpoint with random weights...")
    torch.save(neuron_state_dict, checkpoint_path)
    print(f"  Saved {updated_count} weight tensors to: {checkpoint_path}")

    # Step 5: Compile with modified checkpoint
    print(f"\nStep 5: Compiling neuron lm_head with modified checkpoint...")
    neuron_block = build_module(
        NeuronLMHeadWrapper,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
        },
        checkpoint_path=str(checkpoint_path),
    )

    print(f"\n{'='*80}")
    print(f"LM Head ready with matching random weights")
    print(f"{'='*80}\n")

    return neuron_block, reference_block


def test_embedding_constant_weights():
    """Test embedding layer with constant weight initialization."""
    print("\n" + "="*80)
    print("Test: Embedding with Constant Weights")
    print("="*80)

    config = _make_original_inference_config()
    reference_config = _get_ref_config(config=config)

    # Create sample token IDs
    batch_size = 12
    sample = torch.randint(0, config.vocab_size, (batch_size,), dtype=torch.int32)
    inputs = [(sample,)]
    example_inputs = [(torch.zeros_like(sample),)]

    checkpoint_path = _ARTIFACTS_DIR / "embedding_constant.pt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Build neuron model
    neuron_model = build_module(
        NeuronEmbeddingWrapper,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "weight_init_value": 0.5,
        },
        checkpoint_path=str(checkpoint_path),
    )

    # Build reference model
    reference_model = ReferenceEmbeddingWrapper(config=reference_config)
    _fill_module_parameters(reference_model, 0.5)

    # Get expected output
    with torch.no_grad():
        expected_output = reference_model(sample)

    # Validate
    validate_accuracy(neuron_model, inputs, expected_outputs=[expected_output])

    print("✓ Test PASSED: Embedding with constant weights matches!")


def test_lmhead_constant_weights():
    """Test lm_head layer with constant weight initialization."""
    print("\n" + "="*80)
    print("Test: LM Head with Constant Weights")
    print("="*80)

    config = _make_original_inference_config()
    reference_config = _get_ref_config(config=config)

    # Create sample hidden states
    batch_size = 12
    sample = torch.randn(batch_size, config.hidden_size, dtype=config.neuron_config.torch_dtype)
    inputs = [(sample,)]
    example_inputs = [(torch.zeros_like(sample),)]

    checkpoint_path = _ARTIFACTS_DIR / "lmhead_constant.pt"
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    # Build neuron model
    neuron_model = build_module(
        NeuronLMHeadWrapper,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "weight_init_value": 0.5,
        },
        checkpoint_path=str(checkpoint_path),
    )

    # Build reference model
    reference_model = ReferenceLMHeadWrapper(config=reference_config)
    _fill_module_parameters(reference_model, 0.5)

    # Get expected output
    with torch.no_grad():
        expected_output = reference_model(sample)

    # Validate
    validate_accuracy(neuron_model, inputs, expected_outputs=[expected_output])

    print("✓ Test PASSED: LM Head with constant weights matches!")


def test_embedding_random_weights():
    """Test embedding layer with random weight initialization."""
    print("\n" + "="*80)
    print("Test: Embedding with Random Weights")
    print("="*80)

    config = _make_original_inference_config()

    # Build blocks with random weights
    neuron_block, reference_block = _build_embedding_with_random_weights(
        config,
        checkpoint_name="embedding_random.pt",
        seed=12345,
    )

    # Create test inputs
    torch.manual_seed(67890)
    batch_size = 12
    sample = torch.randint(0, config.vocab_size, (batch_size,), dtype=torch.int32)
    inputs = [(sample,)]

    # Reference forward pass
    with torch.no_grad():
        reference_output = reference_block(sample)

    # Validate against Neuron implementation
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test PASSED: Embedding with random weights matches!")


def test_lmhead_random_weights():
    """Test lm_head layer with random weight initialization."""
    print("\n" + "="*80)
    print("Test: LM Head with Random Weights")
    print("="*80)

    config = _make_original_inference_config()

    # Build blocks with random weights
    neuron_block, reference_block = _build_lmhead_with_random_weights(
        config,
        checkpoint_name="lmhead_random.pt",
        seed=12345,
    )

    # Create test inputs
    torch.manual_seed(67890)
    batch_size = 12
    sample = torch.randn(batch_size, config.hidden_size, dtype=config.neuron_config.torch_dtype)
    inputs = [(sample,)]

    # Reference forward pass
    with torch.no_grad():
        reference_output = reference_block(sample)

    # Validate against Neuron implementation
    validate_accuracy(neuron_block, inputs, expected_outputs=[reference_output])

    print("✓ Test PASSED: LM Head with random weights matches!")


def run_all_embedding_lmhead_tests():
    """Run all embedding and lm_head tests sequentially."""
    print("\n" + "="*80)
    print("EMBEDDING AND LM_HEAD TESTS")
    print("="*80)

    tests = [
        ("Embedding (constant weights)", test_embedding_constant_weights),
        ("Embedding (random weights)", test_embedding_random_weights),
        ("LM Head (constant weights)", test_lmhead_constant_weights),
        ("LM Head (random weights)", test_lmhead_random_weights),
    ]

    for test_name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"✗ Test ({test_name}) FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("EMBEDDING AND LM_HEAD TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    run_all_embedding_lmhead_tests()
