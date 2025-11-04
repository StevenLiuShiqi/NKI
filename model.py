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
from torch import nn
from dataclasses import dataclass
from transformers import AutoModelForCausalLM

from moe_classes import NeuronGPTOSSExpertMLPs

from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.modules.rms_norm import RMSNorm

from neuronx_distributed_inference.models.config import InferenceConfig, MoENeuronConfig
from neuronx_distributed_inference.modules.attention.attention_base import NeuronAttentionBase
from neuronx_distributed_inference.modules.attention.utils import RotaryEmbedding
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
from neuronx_distributed_inference.models.model_base import (
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)

from neuronx_distributed_inference.modules.kvcache.kv_cache_manager import KVCacheManager

import gc
import re
import torch
import gc
import re
import torch

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
      layers.{L}.self_attn.qkv_proj.{q_proj,k_proj,v_proj}.{weight,bias}
      layers.{L}.self_attn.o_proj.o_proj.{weight,bias}
      layers.{L}.ffn.router.linear_router.{weight,bias}
      layers.{L}.ffn.expert_mlps.mlp_op.{gate_up_proj,down_proj}.{weight,bias}
      layers.{L}.{input_layernorm,post_attention_layernorm}.weight
      embed_tokens.weight, norm.weight, lm_head.weight

    Silently drops non-parameter buffers like self_attn.sinks.
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
            # ===== Attention: q/k/v -> qkv_proj.{q_proj,k_proj,v_proj}.{weight,bias}
            for proj in ("q", "k", "v"):

                w = take(f"model.layers.{L}.self_attn.{proj}_proj.weight", None)
                if w is not None:
                    nsd[f"layers.{L}.self_attn.qkv_proj.{proj}_proj.weight"] = to_target(w)
                b = take(f"model.layers.{L}.self_attn.{proj}_proj.bias", None)
                if b is not None:
                    nsd[f"layers.{L}.self_attn.qkv_proj.{proj}_proj.bias"] = to_target(b)

            # ===== Attention: out-proj -> o_proj.o_proj.{weight,bias}
            ow = take(f"model.layers.{L}.self_attn.o_proj.weight", None)
            if ow is not None:
                nsd[f"layers.{L}.self_attn.o_proj.o_proj.weight"] = to_target(ow)
            ob = take(f"model.layers.{L}.self_attn.o_proj.bias", None)
            if ob is not None:
                nsd[f"layers.{L}.self_attn.o_proj.o_proj.bias"] = to_target(ob)

            # Drop sinks buffer if present
            _buf = take(f"model.layers.{L}.self_attn.sinks", None)
            _buf = None

            # ===== Router -> ffn.router.linear_router.{weight,bias}
            rw = take(f"model.layers.{L}.mlp.router.weight", None)
            if rw is not None:
                # Some GPT-OSS exports as [H,E]; NXD expects [E,H]
                if rw.shape == (H, E):
                    rw = rw.t().contiguous()
                elif rw.shape != (E, H):
                    raise ValueError(f"router.weight[{L}] has shape {rw.shape}, expected (E,H) or (H,E)")
                nsd[f"layers.{L}.ffn.router.linear_router.weight"] = to_target(rw)

            rb = take(f"model.layers.{L}.mlp.router.bias", None)
            if rb is not None:
                # Expect [E]
                if rb.dim() != 1 or rb.numel() != E:
                    raise ValueError(f"router.bias[{L}] has shape {tuple(rb.shape)}, expected ({E},)")
                nsd[f"layers.{L}.ffn.router.linear_router.bias"] = to_target(rb)

            # ===== Experts -> ffn.expert_mlps.mlp_op.{gate_up_proj,down_proj}.{weight,bias}
            gu = take(f"model.layers.{L}.mlp.experts.gate_up_proj", None)
            if gu is not None:
                if gu.shape != (E, H, 2*I):
                    raise ValueError(f"gate_up_proj[{L}] {gu.shape} != (E,H,2I)=({E},{H},{2*I})")
                nsd[f"layers.{L}.ffn.expert_mlps.mlp_op.gate_up_proj.weight"] = to_target(gu)

            gub = take(f"model.layers.{L}.mlp.experts.gate_up_proj_bias", None)
            if gub is not None:
                if gub.shape != (E, 2*I):
                    raise ValueError(f"gate_up_proj_bias[{L}] {gub.shape} != (E,2I)=({E},{2*I})")
                nsd[f"layers.{L}.ffn.expert_mlps.mlp_op.gate_up_proj.bias"] = to_target(gub)

            dp = take(f"model.layers.{L}.mlp.experts.down_proj", None)
            if dp is not None:
                if dp.shape == (E, H, I):
                    dp = dp.transpose(1, 2).contiguous()  # -> [E, I, H]
                if dp.shape != (E, I, H):
                    raise ValueError(f"down_proj[{L}] {dp.shape} != (E,I,H)=({E},{I},{H})")
                nsd[f"layers.{L}.ffn.expert_mlps.mlp_op.down_proj.weight"] = to_target(dp)

            dpb = take(f"model.layers.{L}.mlp.experts.down_proj_bias", None)
            if dpb is not None:
                if dpb.shape != (E, H):
                    raise ValueError(f"down_proj_bias[{L}] {dpb.shape} != (E,H)=({E},{H})")
                nsd[f"layers.{L}.ffn.expert_mlps.mlp_op.down_proj.bias"] = to_target(dpb)

            # ===== Norms 
            iln = take(f"model.layers.{L}.input_layernorm.weight", None)
            if iln is not None:
                nsd[f"layers.{L}.input_layernorm.weight"] = to_target(iln)

            paln = take(f"model.layers.{L}.post_attention_layernorm.weight", None)
            if paln is not None:
                nsd[f"layers.{L}.post_attention_layernorm.weight"] = to_target(paln)

            if gc_every and (idx % gc_every == 0):
                gc.collect()

        # Clean up source and return
        gptoss_sd.clear()
        gc.collect()
        return nsd

    
class NeuronGPTOSSConfig(MoENeuronConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fused_qkv = False

class GPTOSSInferenceConfig(InferenceConfig):
    def get_required_attributes(self) -> List[str]:
        return [
        "hidden_size", "num_attention_heads", "num_key_value_heads",
        "head_dim", "vocab_size", "max_position_embeddings",
        "num_hidden_layers", "rms_norm_eps", "pad_token_id",
        # MoE
        ]

    @classmethod
    def get_neuron_config_cls(cls):
        return NeuronGPTOSSConfig
    
class NeuronGPTOSSMLPBlock(torch.nn.Module):
    def __init__(
        self,
        config,
        device: torch.device | None = None,
        weight_init_value: float | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.experts_per_token = config.num_experts_per_tok
        # self.swiglu_limit = config.swiglu_limit
        self.world_size = 1
        
        # RMSNorm
        # self.norm = RMSNorm(config.hidden_size, device=device)
        
        # Create Router (replaces self.gate)
        self.router = RouterTopK(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            act_fn="softmax",  # matches your softmax
            dtype=torch.bfloat16,
            device=device,
            # bias=False, 
            sequence_parallel_enabled=False,  # adjust based on your setup
        )
        
        # Create ExpertMLPs (replaces manual mlp1/mlp2 weights)
        self.expert_mlps = NeuronGPTOSSExpertMLPs(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act="silu",  # SwiGLU uses SiLU internally
            glu_mlp=True,  # SwiGLU is a GLU variant
            # glu_type=GLUType.SWIGLU,  # specify SwiGLU
            capacity_factor=None,  # full capacity (no dropping)
            normalize_top_k_affinities=True,  # your softmax normalizes
            # bias=False,  # as you mentioned
            dtype=torch.bfloat16,
            device=device,
            tensor_model_parallel_group=None,  # set if using TP
            # sequence_parallel_enabled=False,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original: x → norm → gate → topk → expert_mlps → weighted_sum → x + residual
        # With MoE blocks: x → MoE (does all of the above) → x + residual

        t = x
        _, expert_affinities, expert_index = self.router(t)
        t_flat = t.view(-1, t.shape[-1])  # (B*S, H)
        seq_len = x.shape[1]
        moe_output = self.expert_mlps(
            hidden_states=t_flat,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            seq_len=seq_len
        )
        moe_output = moe_output.view_as(x)
        return moe_output

class NeuronGPTOSSAttentionBlock(NeuronAttentionBase):
    def __init__(self, config: InferenceConfig):
        rotary_emb = RotaryEmbedding(
            dim=config.head_dim,                                     # ← 64 (matches Q/K last dim)
            max_position_embeddings=config.max_position_embeddings,  # 131072 from config
            base=config.rope_theta,                                  # 150000 from config
        )
        
        super().__init__(
            config=config,
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            rotary_emb=rotary_emb,
            qkv_bias=True,     # <-- set to True if your checkpoint has q/k/v biases
            o_bias=True,
        )
    
    # enable for testing
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
        
        # return tuple(output)
        return output[0]

class NeuronGPTOSSBlock(nn.Module):
    def __init__(self, config: GPTOSSInferenceConfig, block_idx: int):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.block_idx = block_idx
        
        self.self_attn = NeuronGPTOSSAttentionBlock(config=config)
        self.ffn = NeuronGPTOSSMLPBlock(config=config)
        
        # RMS Norm
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Final linear
        
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states).to(dtype=hidden_states.dtype)
        
        # Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
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
        hidden_states = self.ffn(hidden_states)[0] # not sure why indexing
        hidden_states = residual + hidden_states
        
        # outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        
        outputs = hidden_states
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

    # TODO
    def init_model(self, config: InferenceConfig):
        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        ) # FIX
        
        self.layers = nn.ModuleList(
            [NeuronGPTOSSBlock(config, block_idx) for block_idx in range(config.num_hidden_layers)]
        )
        
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not self.on_device_sampling,
            bias=False,
            pad=True,
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
