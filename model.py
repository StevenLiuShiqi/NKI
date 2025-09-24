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

class NeuronAttentionBlock():
    def __init__(self, config: InferenceConfig):
        
    def forward(self, x):
        
        
class NeuronGPTOSSModel(NeuronBaseModel):
    """
    The neuron version of the GPT OSS
    """

    def setup_attr_for_model(self, config: InferenceConfig):


    def init_model(self, config: InferenceConfig):
        


class NeuronGPTOSSForCausalLM(NeuronBaseForCausalLM):
    """
    This class can be used as GPTOSSForCausalLM
    """

    _model_cls = NeuronGPTOSSModel

    @staticmethod
    def load_hf_model(model_path):

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: InferenceConfig) -> dict:


    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):

    @classmethod
    def get_config_cls(cls):
        return GPTOSSInferenceConfig
    
    def get_compiler_args(self):