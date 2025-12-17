import copy
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union
import torch
from torch import nn
from transformers import AutoConfig, GenerationConfig, PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.generation import SampleDecoderOnlyOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    to_dict,
    to_torch_dtype,
)


def load_pretrained_config(
    model_path_or_name: Optional[Union[str, os.PathLike]] = None,
    hf_config: Optional[PretrainedConfig] = None,
):
    """Return a load_config hook for InferenceConfig that loads the config from a PretrainedConfig."""

    def load_config(self: InferenceConfig):
        if (model_path_or_name is None and hf_config is None) or (
            model_path_or_name is not None and hf_config is not None
        ):
            raise ValueError('Please provide only one of "model_path_or_name" or "hf_config"')

        if model_path_or_name is not None:
            config: PretrainedConfig = AutoConfig.from_pretrained(model_path_or_name)
        else:
            config: PretrainedConfig = hf_config
        config_dict = config.to_dict()

        # Fix transformers_version (config.to_dict() sets it to current transformers version).
        if config.transformers_version is not None:
            config_dict["transformers_version"] = config.transformers_version

        # Set torch_dtype in NeuronConfig.
        if "torch_dtype" in config_dict:
            if self.neuron_config is not None and not self.neuron_config.overrides_torch_dtype:
                # Update neuron_config's torch_dtype if not overriden by the user.
                self.neuron_config.torch_dtype = config_dict["torch_dtype"]
                if isinstance(self.neuron_config.torch_dtype, str):
                    self.neuron_config.torch_dtype = to_torch_dtype(self.neuron_config.torch_dtype)
            del config_dict["torch_dtype"]

        # Convert nested configs to namespaces.
        for k, v in config_dict.items():
            if isinstance(getattr(config, k), PretrainedConfig):
                config_dict[k] = SimpleNamespace(**v)
                if config.transformers_version is not None:
                    config_dict[k].transformers_version = config.transformers_version

        self.__dict__.update(config_dict)
        if hasattr(config, "attribute_map"):
            self.attribute_map = config.attribute_map

    return load_config


def _convert_modality_config_to_pretrained_config(config_dict: Dict, modality: str):
    if modality in config_dict:
        modality_config = config_dict[modality]
        modality_config.pop("neuron_config", None)
        config_dict[modality] = PretrainedConfig(**modality_config)
    return config_dict


def to_pretrained_config(config: InferenceConfig):
    """Convert an InferenceConfig into a PretrainedConfig."""
    config_dict = copy.deepcopy(to_dict(config))
    config_dict["torch_dtype"] = config.neuron_config.torch_dtype
    del config_dict["neuron_config"]

    # handle nested configs for multi-modal models
    config_dict = _convert_modality_config_to_pretrained_config(config_dict, "text_config")
    config_dict = _convert_modality_config_to_pretrained_config(config_dict, "vision_config")

    return PretrainedConfig(**config_dict)


class HuggingFaceGenerationAdapter(PreTrainedModel, GenerationMixin):
    # (optional) HF sometimes probes this; harmless to set
    _is_stateful = True

    def __init__(self, model: NeuronApplicationBase, input_start_offsets=None):
        hf_config = to_pretrained_config(model.config)
        if not hasattr(hf_config, "main_input_name"):
            hf_config.main_input_name = "input_ids"

        PreTrainedModel.__init__(self, hf_config)  # be explicit; avoids MRO surprises

        # Pin transformers version onto generation_config if present
        if self.generation_config is not None and getattr(hf_config, "transformers_version", None):
            self.generation_config.transformers_version = hf_config.transformers_version

        self.neuron_model = model
        self.neuron_config = model.config.neuron_config
        self.padding_side = self.neuron_config.padding_side
        self.input_start_offsets = input_start_offsets

    # IMPORTANT: don’t rebind self.forward; adapt outputs here
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = True,
        **kwargs,
    ) -> ModelOutput:
        # Always use stateless mode (no KV cache)
        out = self.neuron_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        )

        if isinstance(out, ModelOutput) and hasattr(out, "logits"):
            return out
        if isinstance(out, dict) and "logits" in out:
            return CausalLMOutputWithPast(
                logits=out["logits"],
                past_key_values=None,
            )
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            logits = out[0]
            return CausalLMOutputWithPast(logits=logits, past_key_values=None)

        raise TypeError("Underlying Neuron model must return logits (dict/ModelOutput or (logits, pkv)).")

    # Keep Neuron stateless, but call the mixin directly (not super)
    def generate(self, *args, **kwargs):
        self.neuron_model.reset()
        return GenerationMixin.generate(self, *args, **kwargs)

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> Union[SampleDecoderOnlyOutput, torch.LongTensor]:
        r"""
        Bare minimum stateless generation matching simplified_layers.ipynb.
        Always processes full sequences, no KV cache.
        """
        # Get max_length from stopping criteria
        max_length = stopping_criteria[0].max_length if len(stopping_criteria) > 0 else input_ids.shape[1] + 50
        
        # Init values
        pad_token_id = generation_config._pad_token_tensor
        eos_token_id = generation_config.eos_token_id
        if eos_token_id is not None and not isinstance(eos_token_id, (list, tuple)):
            eos_token_id = [eos_token_id]
        eos_token_tensor = torch.tensor(eos_token_id, device=input_ids.device, dtype=torch.long) if eos_token_id is not None else None
        
        top_k = generation_config.top_k if generation_config.top_k is not None else 50
        do_sample = generation_config.do_sample
        temperature = generation_config.temperature if generation_config.temperature is not None else 1.0
        
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init scores / logits tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        all_sequences_finished = False
        
        # Get initial attention_mask from model_kwargs
        attention_mask = model_kwargs.get("attention_mask")
        
        # auto-regressive generation loop (stateless - always process full sequence)
        while not all_sequences_finished:
            # Compute position_ids directly from attention_mask (matching notebook)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            # Forward pass with full input_ids (stateless, no bucket padding)
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True
            )

            if outputs.logits is not None:
                # Get logits for last token (always use last position)
                next_token_logits = outputs.logits[:, -1, :].clone()
                
                # Apply logits processor (e.g., for repetition penalty, etc.)
                next_token_scores = logits_processor(input_ids, next_token_logits)
                
                # Apply top-k filtering (matching simplified_layers.ipynb)
                top_k_actual = min(max(top_k, 1), next_token_scores.size(-1))
                indices_to_remove = next_token_scores < torch.topk(next_token_scores, top_k_actual)[0][..., -1, None]
                next_token_scores = next_token_scores.masked_fill(indices_to_remove, -float('Inf'))
                
                # Apply temperature and softmax
                if temperature != 1.0:
                    next_token_scores = next_token_scores / temperature
                
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                
                # Sample next token
                if do_sample:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(probs, dim=-1)
                
                # Store scores/logits if requested
                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)
            else:
                raise ValueError("Model outputs must contain logits")

            # Finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Update generated ids
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # Update attention mask directly (matching notebook)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)
            model_kwargs["attention_mask"] = attention_mask

            # Check stopping criteria (matching simplified_layers.ipynb)
            is_max_length = torch.full((input_ids.shape[0],), input_ids.shape[1] >= max_length, 
                                       device=input_ids.device, dtype=torch.bool)
            is_eos_token_generated = torch.isin(input_ids[:, -1], eos_token_tensor) if eos_token_tensor is not None else torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            
            # Apply stopping criteria (stopping_criteria returns True when we should stop)
            stop_from_criteria = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            if len(stopping_criteria) > 0:
                # Only check stopping_criteria if it's not empty
                stop_from_criteria = stopping_criteria(input_ids, None)
            
            is_stopping = is_max_length | is_eos_token_generated | stop_from_criteria
            
            unfinished_sequences = unfinished_sequences & ~is_stopping
            all_sequences_finished = unfinished_sequences.max() == 0

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
            )
        else:
            return input_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        sampling_params=None,
        adapter_ids=None,
        **kwargs,
    ):
        # Compute position_ids directly from attention_mask (matching notebook)
        # No bucket padding - stateless generation with variable sequence lengths
        position_ids = None
        if attention_mask is not None:
            # Match notebook exactly: position_ids = attention_mask.long().cumsum(-1) - 1
            # then position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": None,  # Always stateless
                "use_cache": False,  # Always stateless
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_for_token_generation: Optional[bool] = None,
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        # Update attention mask by appending ones for the new token (matching notebook)
        # Direct concatenation - assume right padding (standard for generation)
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )
            model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")
