import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gpt_oss import RotaryEmbedding as GPTOSSRotaryEmbedding, _apply_rotary_emb  # noqa: E402

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from test_utils import (  # noqa: E402
    _fill_module_parameters,
    _get_ref_config,
    _make_tiny_inference_config,
)

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_PATH = _ARTIFACTS_DIR / "neuron_rope_checkpoint.pt"


class NeuronRotaryEmbeddingWrapper(torch.nn.Module):
    """
    Wrapper for Neuron RotaryEmbedding to make it testable with build_module and validate_accuracy.
    This wrapper computes cos and sin values for rotary embeddings.
    """
    def __init__(self, head_dim: int, max_position_embeddings: int, base: float):
        super().__init__()
        from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding

        self.rope = RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
        )
        self.head_dim = head_dim

    def forward(self, query: torch.Tensor, position_ids: torch.Tensor):
        """
        Compute cos and sin values for rotary embeddings.

        Args:
            query: shape (batch_size, seq_len, num_heads, head_dim) - dummy input for shape inference
            position_ids: shape (batch_size, seq_len)

        Returns:
            Tuple of (cos, sin) each with shape (batch_size, seq_len, head_dim)
        """
        # RotaryEmbedding.forward returns cos, sin
        cos, sin = self.rope(query, position_ids)
        return cos, sin


class GPTOSSRotaryEmbeddingWrapper(torch.nn.Module):
    """
    Wrapper for GPT-OSS RotaryEmbedding to generate cos/sin for comparison.
    """
    def __init__(
        self,
        head_dim: int,
        base: int,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
    ):
        super().__init__()
        self.rope = GPTOSSRotaryEmbedding(
            head_dim=head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=initial_context_length,
            scaling_factor=scaling_factor,
            ntk_alpha=ntk_alpha,
            ntk_beta=ntk_beta,
            device=None,
        )
        self.head_dim = head_dim

    def forward(self, query: torch.Tensor, position_ids: torch.Tensor):
        """
        Compute cos and sin values for rotary embeddings.

        Args:
            query: shape (batch_size, seq_len, num_heads, head_dim) - dummy input for shape inference
            position_ids: shape (batch_size, seq_len) - not used, GPT-OSS uses seq_len directly

        Returns:
            Tuple of (cos, sin) with shapes matching Neuron format
        """
        batch_size, seq_len, num_heads, head_dim = query.shape

        # GPT-OSS computes cos/sin based on seq_len
        cos_raw, sin_raw = self.rope._compute_cos_sin(seq_len)
        # cos_raw, sin_raw have shape (seq_len, head_dim // 2)

        # Expand to match Neuron format: (batch, seq_len, head_dim)
        # Neuron repeats the values across the head_dim
        cos_expanded = torch.cat([cos_raw, cos_raw], dim=-1)  # (seq_len, head_dim)
        sin_expanded = torch.cat([sin_raw, sin_raw], dim=-1)

        # Add batch dimension
        cos_expanded = cos_expanded.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, seq_len, head_dim)
        sin_expanded = sin_expanded.unsqueeze(0).expand(batch_size, -1, -1)

        # Convert to bfloat16 to match Neuron
        return cos_expanded.to(query.dtype), sin_expanded.to(query.dtype)


def test_rope_forward_matches_reference():
    """
    Test that the Neuron RotaryEmbedding forward pass matches the GPT-OSS reference implementation.
    Uses build_module and validate_accuracy to test on Neuron hardware.
    """
    config = _make_tiny_inference_config()
    reference_config = _get_ref_config(config=config)

    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()

    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    head_dim = config.head_dim
    num_attention_heads = config.num_attention_heads

    torch.manual_seed(0)

    # Create input tensors - query is just a dummy for shape inference
    query = torch.rand(
        batch_size, seq_len, num_attention_heads, head_dim,
        dtype=config.neuron_config.torch_dtype
    )
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    inputs = [(query, position_ids)]
    example_inputs = [
        (
            torch.zeros_like(query),
            torch.zeros(batch_size, seq_len, dtype=torch.long)
        )
    ]

    # Build Neuron module using build_module for Neuron hardware compilation
    neuron_rope = build_module(
        NeuronRotaryEmbeddingWrapper,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "head_dim": head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
        },
        checkpoint_path=str(_CHECKPOINT_PATH),
    )

    # Create reference module
    reference_rope = GPTOSSRotaryEmbeddingWrapper(
        head_dim=head_dim,
        base=int(reference_config.rope_theta),
        initial_context_length=reference_config.initial_context_length,
        scaling_factor=reference_config.rope_scaling_factor,
        ntk_alpha=reference_config.rope_ntk_alpha,
        ntk_beta=reference_config.rope_ntk_beta,
    )

    # Run reference forward pass
    with torch.no_grad():
        ref_cos, ref_sin = reference_rope(query.clone(), position_ids)

    # Validate accuracy against reference using validate_accuracy
    # validate_accuracy expects expected_outputs as a list of outputs for each input batch
    # Since the model returns a tuple (cos, sin), we pass it as a tuple
    validate_accuracy(neuron_rope, inputs, expected_outputs=[(ref_cos, ref_sin)])


test_rope_forward_matches_reference()
