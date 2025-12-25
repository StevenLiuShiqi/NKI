import os
import torch
import torch.nn as nn


# ============================================================
# Reference RMSNorm (matches simplified_layers.ipynb semantics)
# ============================================================

class GptOssRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


# ============================================================
# Test
# ============================================================

def test_rmsnorm_reference():
    base_dir = os.path.dirname(__file__)
    vector_path = os.path.join(
        base_dir,
        "test_vectors",
        "model.layers.23.post_attention_layernorm.pt",
    )
    assert os.path.exists(vector_path), f"Missing test vector: {vector_path}"

    data = torch.load(vector_path, map_location="cpu")
    hidden_states = data["hidden_states"]
    expected = data["return"]

    model = GptOssRMSNorm(hidden_size=hidden_states.shape[-1])
    weight_path = os.path.join(
        base_dir,
        "test_vectors",
        "model.layers.23.post_attention_layernorm.ckpt",
    )
    assert os.path.exists(weight_path), f"Missing weight checkpoint: {weight_path}"
    weights = torch.load(weight_path, map_location="cpu")
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    if isinstance(weights, torch.Tensor):
        model.load_state_dict({"weight": weights})
    elif isinstance(weights, dict):
        for key in (
            "post_attention_layernorm.weight",
            "model.layers.23.post_attention_layernorm.weight",
            "layers.23.post_attention_layernorm.weight",
            "weight",
        ):
            if key in weights:
                model.load_state_dict({"weight": weights[key]})
                break
        else:
            raise KeyError("Could not find post_attention_layernorm weights in checkpoint")
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(weights)}")
    model.eval()

    with torch.no_grad():
        actual = model(hidden_states)

    diff = actual - expected
    abs_diff = diff.abs()
    tolerance = 1e-3 + expected.abs() * 1e-3
    max_abs_diff = abs_diff.max().item()
    max_rel_diff = (abs_diff / expected.abs().clamp_min(1e-12)).max().item()
    within_tol = (abs_diff <= tolerance).float().mean().item()

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-3,
        atol=1e-3,
        msg=(
            "RMSNorm reference output mismatch. "
            f"max abs diff={max_abs_diff:.3e}, "
            f"max rel diff={max_rel_diff:.3e}, "
            f"accuracy={within_tol:.3%}"
        ),
    )
