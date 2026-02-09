"""PyTorch GPT-OSS model for NXD inference.

    Based on: https://github.com/aws-neuron/nki-llama/blob/main/llama.py
              https://github.com/aws-neuron/neuronx-distributed-inference/blob/e07f0567ad8b77969b0f6eec650234ecb7359419/src/neuronx_distributed_inference/models/dbrx/modeling_dbrx.py
"""

import copy
import gc
import logging
import math
from typing import List, Optional, Tuple, Type

import torch
from torch import Tensor, nn
from dataclasses import dataclass
from transformers import AutoModelForCausalLM

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.moe_classes import NeuronGPTOSSExpertMLPs

from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.rms_norm import RMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase, NeuronAttentionBaseOutput
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)
from neuronx_distributed_inference.modules.moe_v2 import initialize_moe_module


from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager

import gc
import re
import torch
import gc
import re
import torch

def convert_gate_up_proj(tensor: torch.Tensor, is_bias: bool = False) -> torch.Tensor:
    """
    Convert the gate_up_proj tensor from GptOss reference format to NxDI format.

    Reference format: E, 2xI, H with interleaved gate and up projection
    NxDI format: E, H, 2xI with chunked gate and up project

    Args:
        tensor (torch.Tensor): the parameter to convert
        is_bias (bool): flag indicating if parameter is bias

    Returns:
        torch.Tensor: in format needed for NxDI MoE modules
    """
    gate, up_proj = tensor[:, ::2, ...], tensor[:, 1::2, ...]
    gate_up_proj = torch.cat((gate, up_proj), dim=1)
    result = gate_up_proj if is_bias else gate_up_proj
    return result.contiguous()


def convert_gptoss_to_neuron_state_dict(
    gptoss_sd: dict,
    config,
    *,
    target_dtype=None,      # e.g., torch.bfloat16
    target_device="cpu",    # keep CPU for loader
    gc_every=2,
    inplace_pop=True
):
    """
    Convert GPT-OSS checkpoint to NXD (trace) expected keys.

    Emits keys:
      layers.{L}.self_attn.Wqkv.{weight,bias}  (fused Q/K/V projection)
      layers.{L}.self_attn.o_proj.o_proj.{weight,bias}
      layers.{L}.self_attn.learned_sinks.sink (if present)
      layers.{L}.self_attn.tkg_learned_sinks.sink (if present)
      layers.{L}.ffn.router.linear_router.{weight,bias}
      layers.{L}.ffn.expert_mlps.mlp_op.{gate_up_proj,down_proj}.{weight,bias}
      layers.{L}.{input_layernorm,post_attention_layernorm}.weight
      embed_tokens.weight, norm.weight, lm_head.weight
    """

    def take(key, default=None, *, pop=inplace_pop):
        if pop:
            return gptoss_sd.pop(key, default)
        return gptoss_sd.get(key, default)

    def first_present(*keys):
        for k in keys:
            if k in gptoss_sd and gptoss_sd[k] is not None:
                return take(k)
        return None

    def to_target(x):
        if x is None:
            return None
        if target_dtype is not None and x.dtype != target_dtype:
            x = x.to(dtype=target_dtype, copy=False)
        if target_device is not None and (getattr(x, "device", None) is None or x.device.type != target_device):
            x = x.to(device=target_device, copy=False)
        return x

    with torch.no_grad():
        nsd = {}

        # ---- Top-level ----
        etw = first_present("embed_tokens.weight", "model.embed_tokens.weight")
        assert etw is not None, "Missing embed_tokens.weight"
        nsd["embed_tokens.weight"] = to_target(etw)

        nw = first_present("norm.weight", "model.norm.weight")
        assert nw is not None, "Missing norm.weight"
        nsd["norm.weight"] = to_target(nw)

        lmh = first_present("lm_head.weight", "model.lm_head.weight")
        assert lmh is not None, "Missing lm_head.weight"
        nsd["lm_head.weight"] = to_target(lmh)

        # ---- Infer dims from config / tensors ----
        H = int(getattr(config, "hidden_size", nsd["embed_tokens.weight"].shape[1]))
        num_heads = int(getattr(config, "num_attention_heads", 64))
        num_kv_heads = int(getattr(config, "num_key_value_heads", 8))
        head_dim = int(getattr(config, "head_dim", 64))

        # Find one gate_up to infer (E, H, 2I)
        gate_key = None
        for k in list(gptoss_sd.keys()):
            if k.endswith(".mlp.experts.gate_up_proj"):
                gate_key = k; break
        assert gate_key is not None, "Cannot infer expert dims; gate_up_proj not found."
        E = gptoss_sd[gate_key].shape[0]
        H_chk = gptoss_sd[gate_key].shape[1]
        I = gptoss_sd[gate_key].shape[2] // 2
        assert H == H_chk, f"H mismatch: config={H} vs gate_up={H_chk}"

        # ---- Layer ids present in the GPT-OSS dict ----
        layer_ids = sorted({
            int(m.group(1))
            for k in gptoss_sd.keys()
            if k.startswith("model.layers.") and (m := re.match(r"model.layers\.(\d+)\.", k))
        })

        for idx, L in enumerate(layer_ids):
            # ===== Attention: q/k/v projections - NO TRANSPOSE NEEDED
            # Both GPT-OSS and Neuron use standard PyTorch Linear format: (out_features, in_features)
            q_out = num_heads * head_dim
            kv_out = num_kv_heads * head_dim

            # Q projection 
            q_w = take(f"model.layers.{L}.self_attn.q_proj.weight", None)
            nsd[f"layers.{L}.self_attn.qkv_proj.q_proj.weight"] = to_target(q_w)

            q_b = take(f"model.layers.{L}.self_attn.q_proj.bias", None)
            nsd[f"layers.{L}.self_attn.qkv_proj.q_proj.bias"] = to_target(q_b)

            # K projection
            k_w = take(f"model.layers.{L}.self_attn.k_proj.weight", None)
            nsd[f"layers.{L}.self_attn.qkv_proj.k_proj.weight"] = to_target(k_w)

            k_b = take(f"model.layers.{L}.self_attn.k_proj.bias", None)
            nsd[f"layers.{L}.self_attn.qkv_proj.k_proj.bias"] = to_target(k_b)

            # V projection
            v_w = take(f"model.layers.{L}.self_attn.v_proj.weight", None)
            nsd[f"layers.{L}.self_attn.qkv_proj.v_proj.weight"] = to_target(v_w)

            v_b = take(f"model.layers.{L}.self_attn.v_proj.bias", None)
            nsd[f"layers.{L}.self_attn.qkv_proj.v_proj.bias"] = to_target(v_b)

            # ===== Attention: out-proj
            # Keeps PyTorch Linear format: (out, in) = (H, q_out)
            ow = take(f"model.layers.{L}.self_attn.o_proj.weight", None)
            nsd[f"layers.{L}.self_attn.o_proj.o_proj.weight"] = to_target(ow)

            ob = take(f"model.layers.{L}.self_attn.o_proj.bias", None)
            nsd[f"layers.{L}.self_attn.o_proj.o_proj.bias"] = to_target(ob)

            # ===== Attention sinks (learned sink tokens for sliding window attention)
            # Only add learned_sinks (not tkg_learned_sinks unless hybrid parallelism is used)
            sinks = take(f"model.layers.{L}.self_attn.sinks", None)
            nsd[f"layers.{L}.self_attn.learned_sinks.sink"] = to_target(sinks)

            # ===== Router -> ffn.router.linear_router.{weight,bias}
            rw = take(f"model.layers.{L}.mlp.router.weight", None)
            nsd[f"layers.{L}.ffn.moe.router.linear_router.weight"] = to_target(rw)

            rb = take(f"model.layers.{L}.mlp.router.bias", None)
            nsd[f"layers.{L}.ffn.moe.router.linear_router.bias"] = to_target(rb)

            # ===== Experts -> ffn.expert_mlps.mlp_op.{gate_up_proj,down_proj}.{weight,bias}
            gu = take(f"model.layers.{L}.mlp.experts.gate_up_proj", None)
            nsd[f"layers.{L}.ffn.moe.expert_mlps.mlp_op.gate_up_proj.weight"] = to_target(
                convert_gate_up_proj(gu, is_bias=False)  
            )

            gub = take(f"model.layers.{L}.mlp.experts.gate_up_proj_bias", None)
            nsd[f"layers.{L}.ffn.moe.expert_mlps.mlp_op.gate_up_proj.bias"] = to_target(
                convert_gate_up_proj(gub, is_bias=True) 
            )

            dp = take(f"model.layers.{L}.mlp.experts.down_proj", None)
            # Neuron expects: (E, I, H) for einsum("e...i,eih->e...h")
            # The checkpoint has: (E, I, H) 
            dp = dp.transpose(1, 2) # -> (E, I, H)
            nsd[f"layers.{L}.ffn.moe.expert_mlps.mlp_op.down_proj.weight"] = to_target(dp)

            dpb = take(f"model.layers.{L}.mlp.experts.down_proj_bias", None)
            nsd[f"layers.{L}.ffn.moe.expert_mlps.mlp_op.down_proj.bias"] = to_target(dpb)

            # ===== Norms
            iln = take(f"model.layers.{L}.input_layernorm.weight", None)
            nsd[f"layers.{L}.input_layernorm.weight"] = to_target(iln)

            paln = take(f"model.layers.{L}.post_attention_layernorm.weight", None)
            nsd[f"layers.{L}.post_attention_layernorm.weight"] = to_target(paln)

            if gc_every and (idx % gc_every == 0):
                gc.collect()

        # Clean up source and return
        gptoss_sd.clear()
        gc.collect()
        return nsd

class NeuronGPTOSSRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        base: float = 150000,
        scaling_factor: float = 32.0,
        original_max_position_embeddings: int = 4096,
    ):
        
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.attention_scaling = self._get_mscale(scaling_factor)

        low, high = self._find_correction_range(
            low_rot=scaling_factor,
            high_rot=1.0,
            dim=dim,
            base=base,
            max_position_embeddings=original_max_position_embeddings
        )

        inv_freq_extrapolation_factor = 1 - self._linear_ramp_factor(low, high, dim // 2)
        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_extrapolation_factor) +
            inv_freq_extrapolation * inv_freq_extrapolation_factor
        )

        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

# may not need inside the class if exists outside
    def _get_mscale(self, scale: float, mscale: float = 1.0) -> float:
        return 0.1 * mscale * math.log(scale) + 1.0

    def _find_correction_dim(self, num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
        return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def _find_correction_range(self, low_rot: float, high_rot: float, dim: int, base: float, max_position_embeddings: int) -> tuple[float, float]:
        low = self._find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = self._find_correction_dim(high_rot, dim, base, max_position_embeddings)
        return (max(low, 0), min(high, dim - 1))

    def _linear_ramp_factor(self, min_val: float, max_val: float, dim: int) -> torch.Tensor:
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != 'mps' else 'cpu'

        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = freqs
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return (cos.to(x.dtype), sin.to(x.dtype))

class NeuronGPTOSSConfig(MoENeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fused_qkv = False

class GPTOSSInferenceConfig(InferenceConfig):
    # def get_required_attributes(self) -> List[str]:
    #     return [
    #     "hidden_size", "num_attention_heads", "num_key_value_heads",
    #     "head_dim", "vocab_size", "max_position_embeddings",
    #     "num_hidden_layers", "rms_norm_eps", "pad_token_id", "rope_theta",
    #     # MoE
    #     "num_local_experts", "num_experts_per_tok", "intermediate_size",
    #     ]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rms_norm_eps = 1e-05
        self.hidden_act = "sigmoid"
        self.n_shared_experts = 0
        if not hasattr(self, 'original_hidden_size'):
            self.original_hidden_size = self.hidden_size
        if not hasattr(self, 'original_intermediate_size'):
            self.original_intermediate_size = self.intermediate_size
        self.hidden_size = self.hidden_size

        self.intermediate_size = self.intermediate_size
    def add_derived_config(self):
        self.num_cores_per_group = 1
        if self.neuron_config.flash_decoding_enabled:
            num_attn_heads, num_kv_heads = self.num_attention_heads, self.num_key_value_heads
            self.num_cores_per_group = calculate_num_cores_per_group(
                num_attn_heads, num_kv_heads, self.neuron_config.tp_degree
            )

        if not hasattr(self, "num_local_experts") and hasattr(self, "num_experts"):
            self.num_local_experts = getattr(self, "num_experts")

        if not hasattr(self, "num_experts_per_tok") and hasattr(self, "experts_per_token"):
            self.num_experts_per_tok = getattr(self, "experts_per_token")

        if not hasattr(self, "rope_scaling_factor") and hasattr(self, "rope_scaling"):
            self.rope_scaling_factor = self.rope_scaling.get("factor")

        if not hasattr(self, "rope_ntk_alpha") and hasattr(self, "rope_scaling"):
            self.rope_ntk_alpha = self.rope_scaling.get("beta_slow")

        if not hasattr(self, "rope_ntk_beta") and hasattr(self, "rope_scaling"):
            self.rope_ntk_beta = self.rope_scaling.get("beta_fast")

    def get_required_attributes(self) -> List[str]:
        return [
            "num_hidden_layers",
            "num_local_experts",
            "num_experts_per_tok",
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
            "sliding_window",
            "initial_context_length",
            "rope_theta",
            "rope_scaling_factor",
            "rope_ntk_alpha",
            "rope_ntk_beta",
            "pad_token_id",
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronGPTOSSConfig

class NeuronGPTOSSMLPBlock(torch.nn.Module):
    def __init__(self, config: InferenceConfig, rmsnorm: Optional[nn.Module] = None):
        super().__init__()

        # TODO: Handle architecture related configuration separately from NeuronConfig
        config.neuron_config.router_config.dtype = torch.float32
        config.neuron_config.router_config.act_fn = "softmax"
        config.neuron_config.transpose_shared_experts_weights = False
        config.neuron_config.early_expert_affinity_modulation = False
        config.neuron_config.normalize_top_k_affinities = False
        config.neuron_config.glu_type = "swiglu"
        config.neuron_config.hidden_act_scaling_factor = 1.702
        config.neuron_config.hidden_act_bias = 1
        config.neuron_config.gate_clamp_upper_limit = 7.0
        config.neuron_config.gate_clamp_lower_limit = None
        config.neuron_config.up_clamp_upper_limit = 7.0
        config.neuron_config.up_clamp_lower_limit = -7.0

        self.moe = initialize_moe_module(config=config,
                                         rmsnorm=rmsnorm,
                                         init_tkg_module=False,
                                         router_bias=True,
                                         experts_bias=True,
                                         apply_act_fn_over_topk=True)

    def forward(self, hidden_states, is_speculative_decoding=False):
        """Forward pass for the MOE module"""
        # return router_logit and expert_index for testing
        result = self.moe(hidden_states, is_speculative_decoding=is_speculative_decoding)
        hidden_states = result[0]
        router_logits = result[1] if self.moe.return_router_logits else None
        expert_index = result[-1] if self.moe.return_expert_index else None

        return tuple(x for x in (hidden_states, router_logits, expert_index) if x is not None)

    
def _compute_yarn_parameters():
    base = 150000
    partial_rotary_factor = 1.0
    head_dim = 64
    dim = int(head_dim * partial_rotary_factor)
    factor = 32.0
    attention_factor = None
    mscale = None
    mscale_all_dim = None
    original_max_position_embeddings = 4096

    def get_mscale(scale, mscale=1):
        return 0.1 * mscale * math.log(scale) + 1.0

    if attention_factor is None:
        attention_factor = get_mscale(factor)
            
    beta_fast = 32.0
    beta_slow = 1.0

    def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
        return dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate):
        low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
        high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
        return (max(low, 0), min(high, dim - 1))

    def linear_ramp_factor(min, max, dim):
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    pos_freqs = base ** (torch.arange(0, dim, 2).to(dtype=torch.float) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    truncate = False
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings, truncate)
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high, dim // 2).to(dtype=torch.float)
    inv_freq = inv_freq_interpolation * (1 - inv_freq_extrapolation_factor) + inv_freq_extrapolation * inv_freq_extrapolation_factor
    return (inv_freq, attention_factor)


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

class NeuronGPTOSSRotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

        inv_freq, self.attention_scaling = _compute_yarn_parameters()
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

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Specifically enable debug logging for the "Neuron" logger used by attention_base
logger = logging.getLogger("Neuron")
logger.setLevel(logging.DEBUG)

class NeuronGPTOSSAttentionBlockCompiled(NeuronAttentionBase):
    """
    Attention block using NeuronAttentionBase with native compiler path.
    Implements the SDPA logic from gpt_oss.py using PyTorch operations.
    """
    def __init__(self, config: InferenceConfig, layer_idx: int = 0, weight_init_value: float = None):
        rotary_emb = NeuronGPTOSSRotaryEmbedding(
        )
        sliding_window = getattr(config, 'sliding_window', None) if layer_idx % 2 == 0 else None

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rotary_emb=rotary_emb,
            qkv_bias=True,
            o_bias=True,
            learned_sinks_size=1,
            sliding_window=sliding_window,
        )

        self.layer_idx = layer_idx
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

        if weight_init_value is not None:
            self._initialize_weights(weight_init_value)

    def _initialize_weights(self, value: float) -> None:
        """Initialize all parameters to a constant value for deterministic testing."""
        with torch.no_grad():
            for param in self.parameters():
                param.fill_(value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        # args for kv cache usage
        kv_mgr: Optional[KVCacheManager] = None,
        get_kv_per_layer: bool = False,
        update_kv_per_layer: bool = False,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Native compiler attention path matching gpt_oss.py sdpa().

        Uses prep_qkv_tensors from NeuronAttentionBase to get Q, K, V with RoPE applied,
        then implements the SDPA logic with:
        - GQA key/value expansion
        - Learned sinks (single value per head)
        - Sliding window masking (on even layers)
        - Causal masking
        """
        bsz, q_len, _ = hidden_states.shape

        # Use rotary_position_ids if provided, otherwise use position_ids
        rope_position_ids = rotary_position_ids if rotary_position_ids is not None else position_ids
        
        is_token_gen = past_key_value is not None
        original_dtype = hidden_states.dtype
        # Use base class to prepare QKV tensors with RoPE applied
        # This handles: QKV projection, reshaping, and RoPE application
        Q, K, V, cos_cache, sin_cache, residual = self.prep_qkv_tensors(
            rope_position_ids,
            hidden_states,
            past_key_value,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            skip_rope=True, 
            residual=residual,
        )
        
        if cos_cache is None or sin_cache is None:
            # KV cache mode: compute if cache is missing
            cos_cache, sin_cache = self.rotary_emb(hidden_states, rope_position_ids)
            
        Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)
        
        K_cache = K
        V_cache = V
        
        if is_token_gen:
            logger.debug(f"Token gen: attention_mask: {attention_mask}, active_mask: {active_mask}")
            attn_output = self.attention_tokengen(
                Q, K, V, attention_mask, position_ids, past_key_value, active_mask, **kwargs
            )
            
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()
            # merge multi head hidden
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

            # Z = Z.Wo
            attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)

            if self.k_cache_transposed:
                # Output K in BNSd if not transposed, otherwise BNdS
                K = K.permute(0, 1, 3, 2)

            kv: Tuple[Tensor, Tensor] = (K, V)

            if update_kv_per_layer:
                assert kv_mgr is not None
                kv = kv_mgr.update_kv_by_layer_id(
                    kv_per_layer=kv,
                    position_ids=position_ids,
                    **kwargs,
                )
            attn_output = attn_output.to(original_dtype)
            
            return NeuronAttentionBaseOutput(attn_output, kv, cos_cache, sin_cache, residual)

        # Q: [B, num_heads, S, D]
        # K: [B, num_kv_heads, S, D]
        # V: [B, num_kv_heads, S, D]

        # For gpt_oss.py compatibility, we need:
        # Q: [S, num_kv_heads, q_mult, D]
        # K: [S, num_kv_heads, D]
        # V: [S, num_kv_heads, D]

        # Reshape Q to separate GQA groups
        Q = Q.view(bsz, self.num_key_value_heads, self.num_key_value_groups, q_len, self.head_dim)
        Q = Q.permute(0, 3, 1, 2, 4).squeeze(0)  # [S, num_kv_heads, q_mult, D]

        K = K.permute(0, 2, 1, 3).squeeze(0)  # [S, num_kv_heads, D]
        V = V.permute(0, 2, 1, 3).squeeze(0)  # [S, num_kv_heads, D]

        # Prefill mode: use new K, V only
        K_full = K
        V_full = V
        total_seq_len = q_len

        # Get learned sinks (single value per head)
        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            # sinks: [num_heads] -> reshape to [num_kv_heads, q_mult, 1, 1]
            S = learned_sinks.view(self.num_key_value_heads, self.num_key_value_groups, 1, 1)
            S = S.expand(-1, -1, q_len, -1)

        # SDPA implementation from gpt_oss.py:152-172
        n_tokens, n_heads, q_mult, d_head = Q.shape

        # Expand K, V for GQA (using K_full, V_full which includes prior cache if in decode mode)
        K_expanded = K_full[:, :, None, :].expand(-1, -1, q_mult, -1)  # [total_seq_len, num_kv_heads, q_mult, D]
        V_expanded = V_full[:, :, None, :].expand(-1, -1, q_mult, -1)  # [total_seq_len, num_kv_heads, q_mult, D]

        # Original mask logic for prefill
        mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
        if self.sliding_window is not None and self.sliding_window > 0:
            mask += torch.tril(
                mask.new_full((n_tokens, n_tokens), -float("inf")),
                diagonal=-self.sliding_window
            )
        
        # Combine with attention_mask if provided
        # if attention_mask is not None:
        #     # Extract and reshape attention_mask to (n_tokens, n_tokens)
        #     attn_mask = attention_mask
        #     # Handle various input shapes
        #     if attn_mask.dim() == 4:  # (B, 1, S, S)
        #         attn_mask = attn_mask[0, 0, :, :]  # Take first batch, squeeze head dim
        #     elif attn_mask.dim() == 3:  # (B, S, S)
        #         attn_mask = attn_mask[0, :, :]  # Take first batch
        #     elif attn_mask.dim() == 2:  # (S, S)
        #         pass  # Already correct shape
        #     else:
        #         raise ValueError(f"Unexpected attention_mask shape: {attn_mask.shape}")
            
        #     # Ensure shape matches
        #     if attn_mask.shape != (n_tokens, n_tokens):
        #         # Handle padding or truncation if needed
        #         attn_mask = attn_mask[:n_tokens, :n_tokens]
            
        #     # Convert boolean to additive format if needed
        #     if attn_mask.dtype == torch.bool:
        #         # Boolean: True = allow (0.0), False = block (-inf)
        #         attn_mask = torch.where(attn_mask, 0.0, -float("inf"))
        #     else:
        #         # Additive format: convert to standard format (-inf to block, 0.0 to allow)
        #         # If value is negative or -inf, treat as block; otherwise allow
        #         attn_mask = torch.where(attn_mask < 0.0, -float("inf"), 0.0)
            
        #     # Combine masks: both must allow for position to be allowed
        #     mask = torch.maximum(mask, attn_mask)
        #     logger.debug(f"Mask: {mask}")
            
        # Compute attention scores: Q @ K^T
        QK = torch.einsum("qhmd,khmd->hmqk", Q, K_expanded)
        QK *= self.sm_scale
        QK += mask[None, None, :, :]
        
        # Concatenate learned sinks to scores
        if learned_sinks is not None:
            QK = torch.cat([QK, S], dim=-1)

        # Softmax over keys (including sink)
        W = torch.softmax(QK, dim=-1)
        
        # Remove sink weights (we don't have sink values in V)
        if learned_sinks is not None:
            W = W[..., :-1]

        # Compute attention output: W @ V
        attn_output = torch.einsum("hmqk,khmd->qhmd", W, V_expanded)

        # Reshape to [B, S, num_heads * D]
        attn_output = attn_output.reshape(n_tokens, self.num_heads * self.head_dim)
        attn_output = attn_output.unsqueeze(0)  # Add batch dimension

        # Output projection using base class
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)
        
        if self.k_cache_transposed:
            K_cache = K_cache.permute(0, 1, 3, 2)

        kv = (K_cache, V_cache)

        if update_kv_per_layer:
            assert kv_mgr is not None
            kv = kv_mgr.update_kv_by_layer_id(
                kv_per_layer=kv,
                position_ids=position_ids,
                **kwargs,
            )

        # Return just the hidden states to match the test expectations
        # The full output would be NeuronAttentionBaseOutput(attn_output, kv, cos_cache, sin_cache)
        return NeuronAttentionBaseOutput(attn_output, kv, cos_cache, sin_cache)
        
class NeuronGPTOSSAttentionBlock(NeuronAttentionBase):
    def __init__(self, config: InferenceConfig, layer_idx: int = 0, weight_init_value: float = None):
        rotary_emb = NeuronGPTOSSRotaryEmbedding()

        # Sliding window is applied to every other layer (even layer indices)
        sliding_window = getattr(config, 'sliding_window', None) if layer_idx % 2 == 0 else None

        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            # rms_norm_eps=config.rms_norm_eps,
            rotary_emb=rotary_emb,
            qkv_bias=True,
            o_bias=True,
            learned_sinks_size=1,
            sliding_window=sliding_window,
        )

        self.layer_idx = layer_idx

        # Initialize weights to constant value if specified (for testing)
        if weight_init_value is not None:
            self._initialize_weights(weight_init_value)

    def _initialize_weights(self, value: float) -> None:
        """Initialize all parameters to a constant value for deterministic testing."""
        with torch.no_grad():
            for param in self.parameters():
                param.fill_(value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        active_mask: Optional[torch.LongTensor] = None,
        adapter_ids=None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        rmsnorm=None,
        rotary_position_ids: Optional[torch.LongTensor] = None,
        # args for kv cache usage
        kv_mgr: Optional[KVCacheManager] = None,
        get_kv_per_layer: bool = False,
        update_kv_per_layer: bool = False,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        output = super().forward(
            hidden_states=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            active_mask=active_mask,
            adapter_ids=adapter_ids,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
            rotary_position_ids=rotary_position_ids,
            kv_mgr=kv_mgr,
            get_kv_per_layer=get_kv_per_layer,
            update_kv_per_layer=update_kv_per_layer,
            residual=residual,
            **kwargs,
        )

        # Return just the hidden states to match reference implementation
        return output[0]

class GptOssRotaryEmbedding(nn.Module):
    def __init__(self,
                 dim: int,
                 base: int = 10000,
                 initial_context_length: int = 4096,
                 scaling_factor: float = 1.0,
                 ntk_alpha: float = 1.0,
                 ntk_beta: float = 32.0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.register_buffer("inv_freq", None, persistent=False)
        self.concentration = None

    def get_inv_freqs_and_concentration(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
            / self.dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.dim / 2
            # NTK by parts
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

        return inv_freq, concentration

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None or self.concentration is None:
            self.inv_freq, self.concentration = self.get_inv_freqs_and_concentration(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.concentration
        sin = emb.sin() * self.concentration
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronGptOssAttention(NeuronAttentionBase):
    def __init__(
        self,
        config: InferenceConfig,
        layer_idx: int,
    ):
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rotary_emb=self.get_rope(config=config),
            num_cores_per_group=config.num_cores_per_group,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            use_scaled_rope=None,
            qkv_bias=getattr(config, "attention_bias", False),
            o_bias=getattr(config, "attention_bias", False),
            learned_sinks_size=1,
            sliding_window=config.sliding_window if layer_idx % 2 == 0 else 0
        )

    def get_rope(self, config: GPTOSSInferenceConfig):
        rotary_emb = GptOssRotaryEmbedding(dim=config.head_dim,
                                           base=config.rope_theta,
                                           initial_context_length=config.initial_context_length,
                                           scaling_factor=config.rope_scaling_factor,
                                           ntk_alpha=config.rope_ntk_alpha,
                                           ntk_beta=config.rope_ntk_beta)
        return rotary_emb

class NeuronGPTOSSBlock(nn.Module):
    def __init__(self, config: GPTOSSInferenceConfig, block_idx: int):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.block_idx = block_idx
        
        self.self_attn = NeuronGptOssAttention(config=config, layer_idx=self.block_idx)
        self.ffn = NeuronGPTOSSMLPBlock(config=config)
        
        # RMS Norm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Final linear
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        logger.debug(f"Input: {hidden_states.shape}")
        residual = hidden_states
        # hidden_states = self.input_layernorm(hidden_states).to(dtype=hidden_states.dtype)
        
        # Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            rmsnorm=self.input_layernorm,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(dtype=hidden_states.dtype)
        
        # MoE
        hidden_states = self.ffn(hidden_states)[0]
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        return outputs
    
class NeuronGPTOSSModel(NeuronBaseModel):
    """
    The neuron version of the GPT OSS
    """

    def setup_attr_for_model(self, config: GPTOSSInferenceConfig):
        # self.emb_pdrop = config.emb_pdrop

        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    def init_model(self, config: InferenceConfig):
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=config.neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        
        self.layers = nn.ModuleList(
            [NeuronGPTOSSBlock(config, block_idx) for block_idx in range(config.num_hidden_layers)]
        )
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
            # pad=True,
            # tensor_model_parallel_group=get_tp_group(config),
        )


class NeuronGPTOSSForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as GPTOSSForCausalLM
    """
    _STATE_DICT_MODEL_PREFIX = "transformer."
    _model_cls = NeuronGPTOSSModel

    @staticmethod
    def load_hf_model(model_path, **kwargs):
        return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        print(f"STATE DICT: {state_dict.keys()}")
        return convert_gptoss_to_neuron_state_dict(state_dict, config)

    @classmethod
    def get_config_cls(cls):
        return GPTOSSInferenceConfig
    
    def get_compiler_args(self):
        return
