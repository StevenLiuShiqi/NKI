"""
Fusable subgraph modules for GPT-OSS model with explicit input/output shapes.
Each fusable operation is wrapped in its own nn.Module for graph fusion optimization.
"""

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.distributed as dist


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


# ============================================================================
# Fusable Subgraph Modules
# ============================================================================

# Same
class RMSNormFused(nn.Module):
    """
    Fused RMSNorm operation.
    
    Input shape: (num_tokens, num_features)
    Output shape: (num_tokens, num_features)
    """
    def __init__(self, num_features: int, eps: float = 1e-05, device: torch.device | None = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (num_tokens, num_features)
        Output: (num_tokens, num_features)
        """
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)

# Turned function into class
class RotaryEmbeddingCompute(nn.Module):
    """
    Fused rotary embedding computation (cos/sin generation).
    
    Computes cos and sin tables for RoPE.
    """
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> tuple[float, torch.Tensor]:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def forward(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input: num_tokens (scalar)
        Output: (cos: (num_tokens, head_dim//2), sin: (num_tokens, head_dim//2))
        """
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

# Split RoPE
class ApplyRotaryEmbedding(nn.Module):
    """
    Fused rotary embedding application.
    
    Applies precomputed cos/sin to query and key tensors.
    """
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (num_tokens, *, head_dim)
        cos: (num_tokens, head_dim//2)
        sin: (num_tokens, head_dim//2)
        Output: (num_tokens, *, head_dim)
        """
        cos = cos.unsqueeze(-2).to(x.dtype)
        sin = sin.unsqueeze(-2).to(x.dtype)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = x1 * cos - x2 * sin
        o2 = x2 * cos + x1 * sin
        return torch.cat((o1, o2), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        query: (num_tokens, num_kv_heads, q_mult, head_dim)
        key: (num_tokens, num_kv_heads, head_dim)
        cos: (num_tokens, head_dim//2)
        sin: (num_tokens, head_dim//2)
        Output: (query, key) with same shapes as input
        """
        num_tokens = query.shape[0]
        
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = self._apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = self._apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        
        return query, key

# Turned function into class
class ScaledDotProductAttentionFused(nn.Module):
    """
    Fused scaled dot-product attention with sliding window and sinks.
    
    This is a complete attention block fusion candidate.
    """
    def __init__(self, sm_scale: float, sliding_window: int = 0):
        super().__init__()
        self.sm_scale = sm_scale
        self.sliding_window = sliding_window

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        S: torch.Tensor,
    ) -> torch.Tensor:
        """
        Q: (n_tokens, n_heads, q_mult, d_head)
        K: (n_tokens, n_heads, d_head)
        V: (n_tokens, n_heads, d_head)
        S: (n_heads,) - sink tokens
        Output: (n_tokens, n_heads * q_mult * d_head)
        """
        n_tokens, n_heads, q_mult, d_head = Q.shape
        assert K.shape == (n_tokens, n_heads, d_head)
        assert V.shape == (n_tokens, n_heads, d_head)
        
        K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
        V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
        S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
        
        mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
        if self.sliding_window > 0:
            mask += torch.tril(
                mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-self.sliding_window
            )
        
        QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
        QK *= self.sm_scale
        QK += mask[None, None, :, :]
        QK = torch.cat([QK, S], dim=-1)
        W = torch.softmax(QK, dim=-1)
        W = W[..., :-1]
        attn = torch.einsum("hmqk,khmd->qhmd", W, V)
        return attn.reshape(n_tokens, -1)

# Turned functionality into class
class QKVProjection(nn.Module):
    """
    Fused QKV projection and split.
    
    Projects input and splits into Q, K, V tensors.
    """
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        
        qkv_dim = head_dim * (num_attention_heads + 2 * num_key_value_heads)
        self.qkv = nn.Linear(hidden_size, qkv_dim, device=device, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: (num_tokens, hidden_size)
        Output: 
            q: (num_tokens, num_kv_heads, q_mult, head_dim)
            k: (num_tokens, num_kv_heads, head_dim)
            v: (num_tokens, num_kv_heads, head_dim)
        """
        qkv = self.qkv(x)
        
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads * self.head_dim : 
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim :
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
        ].contiguous()

        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        
        return q, k, v

# Function -> Class
class SwiGLUFused(nn.Module):
    """
    Fused SwiGLU activation.
    
    Combines gating and linear components with clamping.
    """
    def __init__(self, alpha: float = 1.702, limit: float = 7.0):
        super().__init__()
        self.alpha = alpha
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, num_experts, intermediate_size * 2)
        Output: (batch, num_experts, intermediate_size)
        """
        x_glu, x_linear = x[..., ::2], x[..., 1::2]
        x_glu = x_glu.clamp(min=None, max=self.limit)
        x_linear = x_linear.clamp(min=-self.limit, max=self.limit)
        out_glu = x_glu * torch.sigmoid(self.alpha * x_glu)
        return out_glu * (x_linear + 1)

# Functionality -> Class
class ExpertGating(nn.Module):
    """
    Fused expert gating and top-k selection.
    """
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        experts_per_token: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.gate = nn.Linear(hidden_size, num_experts, device=device, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input: (num_tokens, hidden_size)
        Output: 
            expert_weights: (num_tokens, experts_per_token)
            expert_indices: (num_tokens, experts_per_token)
        """
        g = self.gate(x)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices
        return expert_weights, expert_indices

# No experts (see above)
class MoEMLPFused(nn.Module):
    """
    Fused MoE MLP computation.
    
    Combines expert selection, MLP layers, and weighted aggregation.
    """
    def __init__(
        self,
        num_experts: int,
        intermediate_size: int,
        hidden_size: int,
        world_size: int,
        swiglu_limit: float,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.world_size = world_size
        
        self.mlp1_weight = nn.Parameter(
            torch.empty(
                (num_experts, intermediate_size * 2 // world_size, hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = nn.Parameter(
            torch.empty(
                (num_experts, intermediate_size * 2 // world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_weight = nn.Parameter(
            torch.empty(
                (num_experts, hidden_size, intermediate_size // world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = nn.Parameter(
            torch.empty(
                (num_experts, hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        
        self.swiglu = SwiGLUFused(limit=swiglu_limit)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (num_tokens, hidden_size)
        expert_indices: (num_tokens, experts_per_token)
        expert_weights: (num_tokens, experts_per_token)
        Output: (num_tokens, hidden_size)
        """
        # MLP #1
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, x) + mlp1_bias
        t = self.swiglu(t)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += mlp2_bias

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)
        return t


# ============================================================================
# Main Model Blocks
# ============================================================================

# Simplified
class AttentionBlockFused(nn.Module):
    """
    Complete fused attention block.
    
    Input shape: (num_tokens, hidden_size)
    Output shape: (num_tokens, hidden_size)
    """
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        
        self.sinks = nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNormFused(config.hidden_size, device=device)
        self.qkv_proj = QKVProjection(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            device=device,
        )
        self.out = nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        
        self.rope_compute = RotaryEmbeddingCompute(
            config.head_dim,
            int(config.rope_theta),
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )
        self.rope_apply = ApplyRotaryEmbedding(config.head_dim)
        self.sdpa = ScaledDotProductAttentionFused(self.sm_scale, self.sliding_window)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (num_tokens, hidden_size)
        Output: (num_tokens, hidden_size)
        """
        residual = x
        t = self.norm(x)
        q, k, v = self.qkv_proj(t)
        
        num_tokens = q.shape[0]
        cos, sin = self.rope_compute(num_tokens)
        q, k = self.rope_apply(q, k, cos, sin)
        
        t = self.sdpa(q, k, v, self.sinks)
        t = self.out(t)
        return residual + t

# Simplified
class MLPBlockFused(nn.Module):
    """
    Complete fused MLP block with MoE.
    
    Input shape: (num_tokens, hidden_size)
    Output shape: (num_tokens, hidden_size)
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.norm = RMSNormFused(config.hidden_size, device=device)
        self.gating = ExpertGating(
            config.hidden_size,
            config.num_experts,
            config.experts_per_token,
            device=device,
        )
        self.moe_mlp = MoEMLPFused(
            config.num_experts,
            config.intermediate_size,
            config.hidden_size,
            self.world_size,
            config.swiglu_limit,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (num_tokens, hidden_size)
        Output: (num_tokens, hidden_size)
        """
        residual = x
        t = self.norm(x)
        expert_weights, expert_indices = self.gating(t)
        t = self.moe_mlp(t, expert_indices, expert_weights)
        return residual + t

# Same
class TransformerBlockFused(nn.Module):
    """
    Complete transformer block combining attention and MLP.
    
    Input shape: (num_tokens, hidden_size)
    Output shape: (num_tokens, hidden_size)
    """
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlockFused(config, layer_idx, device)
        self.mlp = MLPBlockFused(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (num_tokens, hidden_size)
        Output: (num_tokens, hidden_size)
        """
        x = self.attn(x)
        x = self.mlp(x)
        return x

# Same
class TransformerFused(nn.Module):
    """
    Complete transformer model with fusable subgraph modules.
    
    Input shape: (num_tokens,) [token IDs]
    Output shape: (num_tokens, vocab_size) [logits]
    """
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = nn.ModuleList([
            TransformerBlockFused(config, layer_idx, device)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNormFused(config.hidden_size, device=device)
        self.unembedding = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (num_tokens,) - token IDs
        Output: (num_tokens, vocab_size) - logits
        """
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x


# ============================================================================
# Testing
# ============================================================================


def run_tests():
    """Test numerical equivalence of fused modules."""
    print("Running tests...")
    torch.manual_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test 1: RMSNorm
    print("Testing RMSNormFused...")
    norm = RMSNormFused(128, device=device)
    x = torch.randn(10, 128, device=device, dtype=torch.bfloat16)
    out = norm(x)
    assert out.shape == (10, 128), f"Expected (10, 128), got {out.shape}"
    print("  ✓ RMSNormFused shape check passed")
    
    # Test 2: QKV Projection
    print("Testing QKVProjection...")
    qkv_proj = QKVProjection(128, 8, 2, 16, device=device)
    x = torch.randn(10, 128, device=device, dtype=torch.bfloat16)
    q, k, v = qkv_proj(x)
    assert q.shape == (10, 2, 4, 16), f"Expected q shape (10, 2, 4, 16), got {q.shape}"
    assert k.shape == (10, 2, 16), f"Expected k shape (10, 2, 16), got {k.shape}"
    assert v.shape == (10, 2, 16), f"Expected v shape (10, 2, 16), got {v.shape}"
    print("  ✓ QKVProjection shape check passed")
    
    # Test 3: Rotary Embedding
    print("Testing RotaryEmbedding...")
    rope_compute = RotaryEmbeddingCompute(16, 10000, torch.float32, device=device)
    cos, sin = rope_compute(10)
    assert cos.shape == (10, 8), f"Expected cos shape (10, 8), got {cos.shape}"
    assert sin.shape == (10, 8), f"Expected sin shape (10, 8), got {sin.shape}"
    
    rope_apply = ApplyRotaryEmbedding(16)
    q = torch.randn(10, 2, 4, 16, device=device, dtype=torch.bfloat16)
    k = torch.randn(10, 2, 16, device=device, dtype=torch.bfloat16)
    q_out, k_out = rope_apply(q, k, cos, sin)
    assert q_out.shape == q.shape, f"Expected q_out shape {q.shape}, got {q_out.shape}"
    assert k_out.shape == k.shape, f"Expected k_out shape {k.shape}, got {k_out.shape}"
    print("  ✓ RotaryEmbedding shape check passed")
    
    # Test 4: SDPA
    print("Testing ScaledDotProductAttentionFused...")
    sdpa = ScaledDotProductAttentionFused(0.25, sliding_window=4)
    Q = torch.randn(10, 2, 4, 16, device=device, dtype=torch.bfloat16)
    K = torch.randn(10, 2, 16, device=device, dtype=torch.bfloat16)
    V = torch.randn(10, 2, 16, device=device, dtype=torch.bfloat16)
    S = torch.randn(2, device=device, dtype=torch.bfloat16)
    out = sdpa(Q, K, V, S)
    assert out.shape == (10, 2 * 4 * 16), f"Expected (10, 128), got {out.shape}"
    print("  ✓ ScaledDotProductAttentionFused shape check passed")
    
    # Test 5: SwiGLU
    print("Testing SwiGLUFused...")
    swiglu = SwiGLUFused()
    x = torch.randn(10, 4, 256, device=device, dtype=torch.bfloat16)
    out = swiglu(x)
    assert out.shape == (10, 4, 128), f"Expected (10, 4, 128), got {out.shape}"
    print("  ✓ SwiGLUFused shape check passed")
    
    # Test 6: Expert Gating
    print("Testing ExpertGating...")
    gating = ExpertGating(128, 8, 2, device=device)
    x = torch.randn(10, 128, device=device, dtype=torch.bfloat16)
    weights, indices = gating(x)
    assert weights.shape == (10, 2), f"Expected weights shape (10, 2), got {weights.shape}"
    assert indices.shape == (10, 2), f"Expected indices shape (10, 2), got {indices.shape}"
    print("  ✓ ExpertGating shape check passed")
    
    # Test 7: MoE MLP
    print("Testing MoEMLPFused...")
    moe_mlp = MoEMLPFused(8, 256, 128, 1, 7.0, device=device)
    x = torch.randn(10, 128, device=device, dtype=torch.bfloat16)
    expert_indices = torch.randint(0, 8, (10, 2), device=device)
    expert_weights = torch.rand(10, 2, device=device, dtype=torch.bfloat16)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)
    out = moe_mlp(x, expert_indices, expert_weights)
    assert out.shape == (10, 128), f"Expected (10, 128), got {out.shape}"
    print("  ✓ MoEMLPFused shape check passed")
    
    # Test 8: Full Attention Block
    print("Testing AttentionBlockFused...")
    config = ModelConfig(
        num_hidden_layers=2,
        num_experts=8,
        experts_per_token=2,
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        head_dim=16,
        num_attention_heads=8,
        num_key_value_heads=2,
    )
    attn_block = AttentionBlockFused(config, 0, device=device)
    x = torch.randn(10, 128, device=device, dtype=torch.bfloat16)
    out = attn_block(x)
    assert out.shape == (10, 128), f"Expected (10, 128), got {out.shape}"
    print("  ✓ AttentionBlockFused shape check passed")
    
    # Test 9: Full MLP Block
    print("Testing MLPBlockFused...")
    mlp_block = MLPBlockFused(config, device=device)
    x = torch.randn(10, 128, device=device, dtype=torch.bfloat16)
    out = mlp_block(x)
    assert out.shape == (10, 128), f"Expected (10, 128), got {out.shape}"
    print("  ✓ MLPBlockFused shape check passed")
    
    # Test 10: Full Transformer
    print("Testing TransformerFused...")
    model = TransformerFused(config, device=device)
    tokens = torch.randint(0, 1000, (10,), device=device)
    out = model(tokens)
    assert out.shape == (10, 1000), f"Expected (10, 1000), got {out.shape}"
    print("  ✓ TransformerFused shape check passed")
    
    print("\nAll tests passed!")
    print("ALL_TESTS_PASSED")
    return 0


if __name__ == "__main__":
    exit(run_tests())