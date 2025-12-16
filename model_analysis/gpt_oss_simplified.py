import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex
from typing import Optional, Tuple


MAX_POSITION_EMBEDDINGS = 131072
MAX_LENGTH = 20
TOP_K = 50
EOS_TOKEN_ID = [200002, 199999]
PAD_TOKEN_ID = 199999


class GptOssTopKRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(32, 2880, dtype=torch.bfloat16))
        self.bias = nn.Parameter(torch.empty(32, dtype=torch.bfloat16))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, 2880)
        router_logits = F.linear(hidden_states, self.weight, self.bias)  # (seq_len, num_experts)
        router_top_value, router_indices = torch.topk(router_logits, 4, dim=-1)  # (seq_len, top_k)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)
        return router_scores, router_indices


class GptOssRMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2880, dtype=torch.bfloat16))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-05)
        return (self.weight * hidden_states).to(input_dtype)

def get_mscale(scale, mscale=1):
    return 0.1 * mscale * math.log(scale) + 1.0


def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))


def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
    low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
    return (max(low, 0), min(high, dim - 1))


def linear_ramp_factor(min, max, dim):
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class GptOssRotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_scaling = get_mscale(32.0)
        
        low, high = find_correction_range(32.0, 1.0, 64, 150000, 4096)
        inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, 64 // 2)
        pos_freqs = 150000 ** (torch.arange(0, 64, 2).to(dtype=torch.float) / 64)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (32.0 * pos_freqs)
        inv_freq = inv_freq_interpolation * (1 - inv_freq_extrapolation_factor) + inv_freq_extrapolation * inv_freq_extrapolation_factor
        
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = freqs
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return (cos.to(x.dtype), sin.to(x.dtype))


class GptOssExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(32, 2880, 2 * 2880, dtype=torch.bfloat16))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(32, 2 * 2880, dtype=torch.bfloat16))
        self.down_proj = nn.Parameter(torch.empty((32, 2880, 2880), dtype=torch.bfloat16))
        self.down_proj_bias = nn.Parameter(torch.empty(32, 2880, dtype=torch.bfloat16))
        self.alpha = 1.702
        self.limit = 7.0

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, 2880)
        num_experts = routing_weights.shape[1]

        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit[:]:
            expert_idx = expert_idx[0]
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
            gate, up = (gate_up[..., ::2], gate_up[..., 1::2])
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            gated_output = (up + 1) * glu
            out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, expert_idx, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        next_states = next_states.view(batch_size, -1, 2880)

        return next_states


CONFIG_LAYER_TYPES = (
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention',
    'sliding_attention',
    'full_attention'
)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float=0.0,
    # **kwargs
):
    key_states = repeat_kv(key, 8)
    value_states = repeat_kv(value, 8)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    attn_weights = nn.functional.dropout(scores, p=dropout, training=False)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return (attn_output, attn_weights)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


class GptOssAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(2880, 64 * 64, bias=True, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(2880, 8 * 64, bias=True, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(2880, 8 * 64, bias=True, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(64 * 64, 2880, bias=True, dtype=torch.bfloat16)
        self.sinks = nn.Parameter(torch.empty(64, dtype=torch.bfloat16))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 64)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=0.125
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return (attn_output, attn_weights)


class GptOssMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = GptOssTopKRouter()
        self.experts = GptOssExperts()

    def forward(self, hidden_states):
        router_scores, router_indices = self.router(hidden_states)
        routed_out = self.experts(hidden_states, router_indices=router_indices, routing_weights=router_scores)
        return (routed_out, router_scores)


class GptOssDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = GptOssAttention()
        self.mlp = GptOssMLP()
        self.input_layernorm = GptOssRMSNorm()
        self.post_attention_layernorm = GptOssRMSNorm()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]]=None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, _ = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def _vmap_for_bhqkv(mask_function):
    dimensions = [(None, None, None, 0), (None, None, 0, None), (None, 0, None, None), (0, None, None, None)]
    for dims in dimensions:
        mask_function = torch.vmap(mask_function, in_dims=dims, out_dims=0)
    return mask_function


def create_causal_mask(
    attention_mask,
    dtype,
    device,
):
    batch_size = attention_mask.shape[0]
    cur_len = attention_mask.shape[1]
    
    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    q_arange = torch.arange(cur_len, device=device)
    kv_arange = torch.arange(cur_len, device=device)

    def mask_function(batch_idx, head_idx, q_idx, kv_idx):
        return q_idx.new_ones((), dtype=torch.bool) & (kv_idx <= q_idx).to(device) & (attention_mask[batch_idx, kv_idx]).to(device)
    
    with TransformGetItemToIndex():
        mask = _vmap_for_bhqkv(mask_function)(batch_arange, head_arange, q_arange, kv_arange)
    
    mask = torch.where(
        mask,
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.finfo(dtype).min
    )

    return mask


def create_sliding_window_causal_mask(
    attention_mask,
    dtype,
    device,
):
    batch_size = attention_mask.shape[0]
    cur_len = attention_mask.shape[1]
    
    batch_arange = torch.arange(batch_size, device=device)
    head_arange = torch.arange(1, device=device)
    q_arange = torch.arange(cur_len, device=device)
    kv_arange = torch.arange(cur_len, device=device)

    def mask_function(batch_idx, head_idx, q_idx, kv_idx):
        return q_idx.new_ones((), dtype=torch.bool) & (kv_idx > q_idx - 128).to(device) & (kv_idx <= q_idx).to(device) & (attention_mask[batch_idx, kv_idx]).to(device)
    
    with TransformGetItemToIndex():
        mask = _vmap_for_bhqkv(mask_function)(batch_arange, head_arange, q_arange, kv_arange)
    
    mask = torch.where(
        mask,
        torch.tensor(0.0, device=device, dtype=dtype),
        torch.finfo(dtype).min
    )

    return mask


class GptOssModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(201088, 2880, 199999, dtype=torch.bfloat16)
        self.layers = nn.ModuleList([GptOssDecoderLayer() for _ in range(24)])
        self.norm = GptOssRMSNorm()
        self.rotary_emb = GptOssRotaryEmbedding()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        input_embeddings = self.embed_tokens(input_ids)
        causal_mask_mapping = {
            'full_attention': create_causal_mask(
                attention_mask=attention_mask,
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            ),
            'sliding_attention': create_sliding_window_causal_mask(
                attention_mask=attention_mask,
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
        }
        hidden_states = input_embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer, layer_type in zip(self.layers, CONFIG_LAYER_TYPES):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer_type],
                position_embeddings=position_embeddings,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GptOssForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = GptOssModel()
        self.lm_head = nn.Linear(2880, 201088, bias=False, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ):
        hidden_states = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids)
        logits = self.lm_head(hidden_states)
        return logits


@torch.no_grad()
def generate(
    model,
    input_ids,
    attention_mask,
):  
    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[1]

    max_length = min(MAX_LENGTH, MAX_POSITION_EMBEDDINGS)
    pad_token_tensor = torch.tensor(PAD_TOKEN_ID, device=input_ids.device, dtype=torch.long)
    eos_token_tensor = torch.tensor(EOS_TOKEN_ID, device=input_ids.device, dtype=torch.long)

    all_sequences_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

    while not all_sequences_finished:
        # Fully recompute position_ids for new length
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        # Stateless: only pass input_ids, attention_mask, position_ids
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # Get probs for next token in sequence
        next_token_logits = logits[:, -1, :]
        top_k = min(max(TOP_K, 1), next_token_logits.size(-1))
        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
        next_token_scores = next_token_logits.masked_fill(indices_to_remove, -float('Inf'))
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        
        next_tokens = (
            torch.multinomial(probs, num_samples=1).squeeze(1) * unfinished_sequences
            + pad_token_tensor * (1 - unfinished_sequences)
        )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        is_max_length = torch.full((input_ids.shape[0],), input_ids.shape[1] >= max_length, device=input_ids.device, dtype=torch.bool)
        is_eos_token_generated = torch.isin(input_ids[:, -1], eos_token_tensor)
        is_stopping = is_max_length | is_eos_token_generated
        
        unfinished_sequences = unfinished_sequences & ~is_stopping
        all_sequences_finished = unfinished_sequences.max() == 0
        cur_len += 1

        del logits

    return input_ids