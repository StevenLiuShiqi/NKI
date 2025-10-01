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

from neuronx_distributed_inference.modules.moe import initialize_moe_module

from neuronx_distributed_inference.models.config import InferenceConfig, NeuronConfig  # noqa: E402
from neuronx_distributed_inference.models.model_base import (  # noqa: E402
    NeuronBaseForCausalLM,
    NeuronBaseModel,
)

# From https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py
# class ModelConfig:
#     num_hidden_layers: int = 36
#     num_experts: int = 128
#     experts_per_token: int = 4
#     vocab_size: int = 201088
#     hidden_size: int = 2880
#     intermediate_size: int = 2880
#     swiglu_limit: float = 7.0
#     head_dim: int = 64
#     num_attention_heads: int = 64
#     num_key_value_heads: int = 8
#     sliding_window: int = 128
#     initial_context_length: int = 4096
#     rope_theta: float = 150000.0
#     rope_scaling_factor: float = 32.0
#     rope_ntk_alpha: float = 1.0
#     rope_ntk_beta: float = 32.0

def convert_gptoss_to_neuron_state_dict(gptoss_sd: dict, config):
    """
    Convert GPT-OSS state_dict (keys like 'model.layers.X...') into Neuron MoE format
    (keys like 'layers.X...') using the measured shapes you provided.

    Measured (layer 0):
      H=2880, L=24
      qW=(4096,2880), kW=(512,2880), vW=(512,2880)  -> packed Wqkv=(5120,2880), biases present
      o_proj=(2880,4096) with bias
      router.weight=(32,2880), router.bias=(32,)
      gate_up_proj=(32,2880,5760)  [E,H,2I], bias=(32,5760)
      down_proj   =(32,2880,2880)  [E,H,I],  bias=(32,2880)
    """

    nsd = {}

    # ---- Top level ----
    nsd["embed_tokens.weight"] = gptoss_sd["model.embed_tokens.weight"].clone().detach()
    nsd["norm.weight"]         = gptoss_sd["model.norm.weight"].clone().detach()
    nsd["lm_head.weight"]      = gptoss_sd["lm_head.weight"].clone().detach()

    # ---- Sizes (trust tensors; assert against config) ----
    H = nsd["embed_tokens.weight"].shape[1]                      # 2880
    L = len([k for k in gptoss_sd.keys() if k.startswith("model.layers.") and k.endswith(".input_layernorm.weight")])
    E = gptoss_sd["model.layers.0.mlp.experts.gate_up_proj"].shape[0]        # 32
    I = gptoss_sd["model.layers.0.mlp.experts.gate_up_proj"].shape[2] // 2   # 2880

    # Optional sanity checks (won't break if configs differ, just informative)
    try:
        assert H == config.d_model
        assert L == config.n_layers
        assert E == config.ffn_config.moe_num_experts
        assert I == config.ffn_config.ffn_hidden_size
    except Exception:
        pass

    def pack_qkv(q_w, k_w, v_w):
        # shapes: (4096,2880), (512,2880), (512,2880) -> (5120,2880)
        assert q_w.dim() == k_w.dim() == v_w.dim() == 2
        assert q_w.shape[1] == k_w.shape[1] == v_w.shape[1] == H
        return torch.cat([q_w, k_w, v_w], dim=0)

    def cat_biases(q_b, k_b, v_b):
        if q_b is None or k_b is None or v_b is None:
            return None
        return torch.cat([q_b.view(-1), k_b.view(-1), v_b.view(-1)], dim=0)

    for l in range(L):
        # --- Attention ---
        q_w = gptoss_sd[f"model.layers.{l}.self_attn.q_proj.weight"].clone().detach()
        k_w = gptoss_sd[f"model.layers.{l}.self_attn.k_proj.weight"].clone().detach()
        v_w = gptoss_sd[f"model.layers.{l}.self_attn.v_proj.weight"].clone().detach()
        nsd[f"layers.{l}.self_attn.Wqkv.weight"] = pack_qkv(q_w, k_w, v_w)

        q_b = gptoss_sd.get(f"model.layers.{l}.self_attn.q_proj.bias")
        k_b = gptoss_sd.get(f"model.layers.{l}.self_attn.k_proj.bias")
        v_b = gptoss_sd.get(f"model.layers.{l}.self_attn.v_proj.bias")
        wqkv_b = cat_biases(q_b, k_b, v_b)
        if wqkv_b is not None:
            nsd[f"layers.{l}.self_attn.Wqkv.bias"] = wqkv_b.clone().detach()

        nsd[f"layers.{l}.self_attn.o_proj.weight"] = (
            gptoss_sd[f"model.layers.{l}.self_attn.o_proj.weight"].clone().detach()
        )
        o_b = gptoss_sd.get(f"model.layers.{l}.self_attn.o_proj.bias")
        if o_b is not None:
            nsd[f"layers.{l}.self_attn.o_proj.bias"] = o_b.clone().detach()

        # --- Router ---
        r_w = gptoss_sd[f"model.layers.{l}.mlp.router.weight"].clone().detach()  # (E,H)
        # If ever stored as (H,E), transpose -> (E,H)
        if r_w.shape == (H, E):
            r_w = r_w.t()
        nsd[f"layers.{l}.ffn.router.linear_router.weight"] = r_w
        r_b = gptoss_sd.get(f"model.layers.{l}.mlp.router.bias")
        if r_b is not None:
            nsd[f"layers.{l}.ffn.router.linear_router.bias"] = r_b.clone().detach()

        # --- Experts ---
        gate_up = gptoss_sd[f"model.layers.{l}.mlp.experts.gate_up_proj"].clone().detach()  # expect [E,H,2I]
        assert gate_up.shape == (E, H, 2 * I), f"gate_up shape {gate_up.shape} != (E,H,2I)"
        nsd[f"layers.{l}.ffn.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up

        down = gptoss_sd[f"model.layers.{l}.mlp.experts.down_proj"].clone().detach()        # given [E,H,I]
        if down.shape == (E, H, I):
            down = down.transpose(1, 2)   # -> [E,I,H]
        elif down.shape != (E, I, H):
            raise ValueError(f"Unexpected down_proj shape {down.shape}")
        nsd[f"layers.{l}.ffn.expert_mlps.mlp_op.down_proj.weight"] = down

        gub = gptoss_sd.get(f"model.layers.{l}.mlp.experts.gate_up_proj_bias")  # (E,2I)
        if gub is not None:
            nsd[f"layers.{l}.ffn.expert_mlps.mlp_op.gate_up_proj.bias"] = gub.clone().detach()
        dpb = gptoss_sd.get(f"model.layers.{l}.mlp.experts.down_proj_bias")     # (E,I)
        if dpb is not None:
            nsd[f"layers.{l}.ffn.expert_mlps.mlp_op.down_proj.bias"] = dpb.clone().detach()

        # --- Norms ---
        nsd[f"layers.{l}.input_layernorm.weight"] = (
            gptoss_sd[f"model.layers.{l}.input_layernorm.weight"].clone().detach()
        )
        nsd[f"layers.{l}.post_attention_layernorm.weight"] = (
            gptoss_sd[f"model.layers.{l}.post_attention_layernorm.weight"].clone().detach()
        )

        # --- TP metadata (same spirit as your DBRX script) ---
        nsd[f"layers.{l}.self_attn.rank_util.rank"] = torch.arange(
            0, config.neuron_config.tp_degree, dtype=torch.int32
        )

        # NOTE: 'model.layers.{l}.self_attn.sinks' is a buffer for sliding attention; intentionally skipped.

    # Cleanup
    gptoss_sd.clear()
    gc.collect()
    return nsd
    
class GPTOSSInferenceConfig(InferenceConfig):
    def add_derived_config(self):
        ...

    def get_required_attributes(self) -> List[str]:


    @classmethod
    def get_neuron_config_cls(cls) -> Type[NeuronConfig]:

    
class NeuronMLPBlock(torch.nn.Module):
    def __init__(self, config: InferenceConfig):
        self.ffn = initialize_moe_module(
            config=config,
            num_experts=config.ffn_config.moe_num_experts,
            top_k=config.ffn_config.moe_top_k,
            hidden_size=config.d_model,
            intermediate_size=config.ffn_config.ffn_hidden_size,
            hidden_act=config.ffn_config.ffn_act_fn["name"],
        )
    
    def forward(self, x):
        return self.ffn(x)

class NeuronGPTOSSAttentionBlock():
    def __init__(self, config: InferenceConfig):
        
    def forward(self, x):
        

class NeuronGPTOSSBlock(nn.Module):
    def __init__(self, config: GPTOSSInferenceConfig, block_idx: int):
        super().__init__()
        self.hidden_size = config.d_model
        self.resid_pdrop = config.resid_pdrop
        self.block_idx = block_idx
        
        self.self_attn = NeuronGPTOSSAttentionBlock(config=config)
        self.ffn = NeuronMLPBlock(config=config)
        
        # RMS Norm
        self.input_rms_norm = nn.RMSNorm(config.d_model, bias=False)
        self.post_attention_rms_norm = nn.RMSNorm(config.d_model, bias=False)
        # Final linear
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_rms_norm(hidden_states).to(device=hidden_states.dtype)
        
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
        hidden_states = self.post_attention_rms_norm(hidden_states).to(device=hidden_states.dtype)
        
        # MoE
        hidden_states = self.ffn(hidden_states)[0] # not sure why indexing
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states, present_key_value, cos_cache, sin_cache, None)
        
        return outputs
    
class NeuronGPTOSSModel(NeuronBaseModel):
    """
    The neuron version of the GPT OSS
    """

    def setup_attr_for_model(self, config: GPTOSSInferenceConfig):
        self.emb_pdrop = config.emb_pdrop

        # Needed for init_inference_optimization()
        self.on_device_sampling = config.neuron_config.on_device_sampling_config is not None
        self.tp_degree = config.neuron_config.tp_degree
        self.hidden_size = config.d_model
        self.num_attention_heads = config.n_heads
        self.num_key_value_heads = config.attn_config.kv_n_heads
        self.max_batch_size = config.neuron_config.max_batch_size
        self.buckets = config.neuron_config.buckets

    # TODO
    def init_model(self, config: InferenceConfig):
        self.layers = nn.ModuleList(
            [NeuronGPTOSSBlock(config, block_idx) for block_idx in range(config.n_layers)]
        )


class NeuronGPTOSSForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as GPTOSSForCausalLM
    """

    _model_cls = NeuronGPTOSSModel

    @staticmethod
    def load_hf_model(model_path):
        return NeuronGPTOSSForCausalLM.from_pretrained(model_path, **kwargs)
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:
        return convert_gptoss_to_neuron_state_dict(state_dict, config)

    @classmethod
    def get_config_cls(cls):
        return GPTOSSInferenceConfig
    
    def get_compiler_args(self):
        return