import copy
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union
import torch

logger = logging.getLogger(__name__)
from neuronx_distributed.utils.medusa_utils import (
    evaluate_posterior,
    generate_candidates,
    generate_medusa_buffers,
    update_inference_inputs,
)
from transformers import AutoConfig, GenerationConfig, PretrainedConfig, PreTrainedModel, GenerationMixin
from transformers.generation import GenerateDecoderOnlyOutput, SampleDecoderOnlyOutput
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.modeling_outputs import ModelOutput

from neuronx_distributed_inference.models.application_base import NeuronApplicationBase
from neuronx_distributed_inference.models.config import (
    InferenceConfig,
    NeuronConfig,
    OnDeviceSamplingConfig,
    to_dict,
    to_torch_dtype,
)
from neuronx_distributed_inference.modules.generation.sampling import (
    Sampler,
    prepare_sampling_params,
)
from neuronx_distributed_inference.modules.lora_serving import LoraCheckpoint


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
        self.on_device_sampling = self.neuron_config.on_device_sampling_config is not None
        self.padding_side = self.neuron_config.padding_side
        self.sampler = None
        self.prev_kv_cache_populated = False
        self.lora_checkpoint = LoraCheckpoint(self.neuron_config.lora_config)
        self.input_start_offsets = input_start_offsets
        # Track the last absolute position for decode mode (when past_key_values is not accessible)
        self.last_absolute_position = None

    # IMPORTANT: don't rebind self.forward; adapt outputs here
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache: bool = True,
        **kwargs,
    ) -> ModelOutput:
        # STATELESS MODE: Force past_key_values=None and use_cache=False
        # to decouple from KV cache mechanism. Every token is appended to 
        # every preceding token and fed back into the model.
        past_key_values = None
        use_cache = False
        
        out = self.neuron_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        if isinstance(out, ModelOutput) and hasattr(out, "logits"):
            return out
        if isinstance(out, dict) and "logits" in out:
            return CausalLMOutputWithPast(
                logits=out["logits"],
                past_key_values=out.get("past_key_values"),
            )
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            logits = out[0]
            pkv = out[1] if len(out) > 1 else None
            return CausalLMOutputWithPast(logits=logits, past_key_values=pkv)

        raise TypeError("Underlying Neuron model must return logits (dict/ModelOutput or (logits, pkv)).")

    # Keep Neuron stateless, but call the mixin directly (not super)
    def generate(self, *args, **kwargs):
        self.neuron_model.reset()
        return GenerationMixin.generate(self, *args, **kwargs)

    # TODO: Remove _sample and define separate flow for on-device sampling that doesn't use HF.
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ) -> Union[SampleDecoderOnlyOutput, torch.LongTensor]:
        r"""
        We override the GenerationMixin sample function (_sample for transformers>=4.39.0) to add support for right side padding.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(
            hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
        )
        do_sample = generation_config.do_sample

        batch_size = model_kwargs["attention_mask"].shape[0]
        sampling_params = prepare_sampling_params(
            batch_size=batch_size,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
        )
        model_kwargs["sampling_params"] = sampling_params

        # init scores / logits tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=input_ids.device
        )
        # convert adapter_ids from strings to indices
        model_kwargs["adapter_ids"] = self.lora_checkpoint.convert_adapter_ids_to_indices(
            model_kwargs.get("adapter_ids"), unfinished_sequences.numel()
        )
        this_peer_finished = False
        # auto-regressive generation
        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_kwargs["attention_mask"] = model_inputs.get("attention_mask")

            # forward pass to get next token
            outputs = self(**model_inputs, return_dict=True)

            if outputs.logits is not None:
                next_token_logits = outputs.logits[:, -1, :].clone()
                
                # Log raw logits
                top_k_logits, top_k_indices = torch.topk(next_token_logits[0], k=5)
                logger.debug(f"GENERATION - Step {input_ids.shape[1]}, top 5 logits: {top_k_logits.tolist()}, tokens: {top_k_indices.tolist()}")
                logger.debug(f"GENERATION - Logits stats: min={next_token_logits.min():.2f}, max={next_token_logits.max():.2f}, mean={next_token_logits.mean():.2f}")

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                
                top_k_scores, top_k_score_indices = torch.topk(next_token_scores[0], k=5)
                logger.debug(f"GENERATION - After logits_processor, top 5 scores: {top_k_scores.tolist()}, tokens: {top_k_score_indices.tolist()}")

                if return_dict_in_generate:
                    if output_scores:
                        scores += (next_token_scores,)
                    if output_logits:
                        raw_logits += (next_token_logits,)

            if not self.on_device_sampling:
                if self.sampler is None:
                    # Temporary placeholder to support CPU sampling with static batching
                    neuron_kwargs = {}

                    config_kwargs = {"top_k": generation_config.top_k}
                    config = OnDeviceSamplingConfig(**config_kwargs)

                    neuron_kwargs["on_device_sampling_config"] = config
                    sampler_config = NeuronConfig(**neuron_kwargs)

                    sampler_config.on_cpu = True
                    self.sampler = Sampler(sampler_config, do_sample=do_sample)

                next_tokens = self.sampler(next_token_scores, sampling_params)
            else:
                next_tokens = outputs.tokens
            
            logger.debug(f"GENERATION - Sampled token: {next_tokens[0].item()}, input_ids shape before cat: {input_ids.shape}")

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                    1 - unfinished_sequences
                )

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            logger.debug(f"GENERATION - input_ids after cat: {input_ids[0, -10:].tolist()}, shape: {input_ids.shape}")

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, None)
            this_peer_finished = unfinished_sequences.max() == 0

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
        # Store KV cache flag before forward pass.
        self.prev_kv_cache_populated = self.neuron_model.kv_cache_populated
        
        # Reset position tracking when starting a new prefill (cache was cleared)
        if not self.neuron_model.kv_cache_populated and self.last_absolute_position is not None:
            self.last_absolute_position = None
            
        # Truncate input_ids to last token during decode mode
        # Check both kv_cache_populated and past_key_values since flag might not be set yet
        # if self.neuron_model.kv_cache_populated or (past_key_values is not None and len(past_key_values) > 0):
        #     input_ids = input_ids[:, -1:] KEVIN

        accepted_indices = kwargs.get("accepted_indices", None)
        current_length = kwargs.get("current_length", None)
        medusa_mask = kwargs.get("medusa_mask", None)
        scatter_index = kwargs.get("scatter_index", None)
        position_ids = kwargs.get("position_ids", None)
        input_capture_hook = kwargs.get("input_capture_hook", None)
        tensor_capture_hook = kwargs.get("tensor_capture_hook", None)
        
        # past_key_values might be in kwargs instead of parameter
        if past_key_values is None:
            past_key_values = kwargs.get("past_key_values", None)

        # In stateless mode, always recompute position_ids from sequence length
        # This is simpler and more direct than deriving from attention_mask
        is_stateless_mode = not self.neuron_model.kv_cache_populated

        if is_stateless_mode:
            # In stateless mode, position_ids are simply [0, 1, 2, ..., seq_len-1]
            # Compute directly from input_ids sequence length
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            
            if self.input_start_offsets:
                # Handle input_start_offsets if present
                if len(self.input_start_offsets) > 1:
                    position_ids += torch.tensor(self.input_start_offsets, dtype=position_ids.dtype, device=position_ids.device)[:, None]
                else:
                    position_ids += self.input_start_offsets[0]
                for i, offset in enumerate(self.input_start_offsets):
                    position_ids[i, 0:offset] = torch.arange(offset, dtype=torch.long, device=position_ids.device)
        elif attention_mask is not None and position_ids is None:
            # Decode mode with KV cache: derive per-token positions from attention_mask.
            #
            # We deliberately ensure that padding tokens never share a valid position id
            # with real tokens; otherwise RoPE would treat padding as a real position
            # (e.g., position 1), which corrupts positional information and can lead
            # to pathological repetition.
            #
            # Base positions from non‑padded tokens.
            position_ids = attention_mask.long().cumsum(-1) - 1  # [0, 1, 2, ...] on tokens

            if self.input_start_offsets:
                # Shift positions by per‑sequence offsets when present.
                if len(self.input_start_offsets) > 1:
                    position_ids += torch.tensor(
                        self.input_start_offsets,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )[:, None]
                else:
                    position_ids += self.input_start_offsets[0]
                for i, offset in enumerate(self.input_start_offsets):
                    # For any explicit prefix region, assign consecutive absolute positions.
                    position_ids[i, 0:offset] = torch.arange(
                        offset, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                # For padding, use a sentinel that does not collide with any real token position.
                # We pick -1 here and rely on attention_mask to prevent pads from participating
                # in attention; RoPE will see a distinct position for pads.
                position_ids = position_ids.masked_fill(attention_mask == 0, -1)

            # Track the max (non‑padding) position during prefill for use in decode
            if not self.neuron_model.kv_cache_populated:
                # During prefill, update last_absolute_position to the max position
                max_pos = torch.amax(position_ids).item()
                self.last_absolute_position = max_pos

        # CRITICAL FIX: During decode, ALWAYS recompute position_ids to absolute position
        # This must run even if position_ids was provided, to ensure correct absolute positions
        # Problem: After a 128-token prefill (positions 0-127), the first decode should be at position 128,
        # but the old code gave position 1. This caused RoPE misalignment and repeated tokens.
        #
        # Solution: Compute absolute position by:
        # 1. If past_key_values is provided, extract cached sequence length from it
        # 2. Otherwise, track position incrementally using self.last_absolute_position
        # 3. Set position_ids to the absolute position of the new token
        #
        # Check for decode mode: either kv_cache_populated is True OR past_key_values exists
        # (kv_cache_populated might not be set yet on first decode step, but past_key_values indicates decode)
        # is_decode_mode = self.neuron_model.kv_cache_populated or (past_key_values is not None and len(past_key_values) > 0)
        
        # KEVIN: Only use decode-mode position_ids when KV cache is actually populated
        # When KV cache is disabled, position_ids should always be computed from full attention_mask
        is_decode_mode = self.neuron_model.kv_cache_populated
        if is_decode_mode:
            cached_seq_len = 0
            
            # Try to get cached sequence length from past_key_values if available
            if past_key_values is not None and len(past_key_values) > 0:
                try:
                    # past_key_values is a tuple of (K, V) for each layer
                    # K shape: [B, num_kv_heads, S_prior, D] or [B, num_kv_heads, D, S_prior]
                    # V shape: [B, num_kv_heads, S_prior, D]
                    # Use V to get sequence length (more reliable than K which might be transposed)
                    first_layer_kv = past_key_values[0]
                    if isinstance(first_layer_kv, (tuple, list)) and len(first_layer_kv) >= 2:
                        V_cache = first_layer_kv[1]  # First layer's V cache
                        if len(V_cache.shape) >= 3:
                            # V shape is [B, num_kv_heads, S_prior, D] - sequence dim is at index 2
                            cached_seq_len = V_cache.shape[2]
                except (IndexError, AttributeError, TypeError) as e:
                    # If we can't extract from past_key_values, fall through to other methods
                    pass
            
            # If we couldn't get it from past_key_values (common with kv_mgr), 
            # try to compute from attention_mask or use incremental tracking
            if cached_seq_len == 0 and attention_mask is not None:
                # attention_mask during decode might be just [1] for the new token
                # or it might be the full sequence [1, 1, 1, ..., 1] including cached tokens
                # If it's the full sequence, the cached length is attention_mask.sum() - 1
                total_seq_len = attention_mask.sum(dim=1, keepdim=True).long()
                # During decode, we're adding one new token, so cached length is total - 1
                if total_seq_len.item() > 1:
                    cached_seq_len = (total_seq_len - 1).item()
            
            # If we still don't have cached_seq_len, use incremental tracking
            if cached_seq_len == 0:
                if self.last_absolute_position is not None:
                    # Increment from last known absolute position
                    cached_seq_len = self.last_absolute_position + 1
                else:
                    # Fallback: this shouldn't happen if tracking works correctly
                    # Use the old logic as last resort (may cause issues)
                    if position_ids is not None:
                        max_pos = torch.amax(position_ids, 1, keepdim=True)
                        cached_seq_len = max_pos.item() + 1
                    else:
                        # Last resort: assume we're at position 1 (shouldn't happen)
                        cached_seq_len = 1
            
            # Set position_ids to absolute position: cached_seq_len (0-indexed)
            # The new token is at position cached_seq_len (after tokens at 0, 1, ..., cached_seq_len-1)
            # Use input_ids.shape[0] to get batch size in case position_ids wasn't set yet
            batch_size = input_ids.shape[0] if position_ids is None else position_ids.shape[0]
            position_ids = torch.full((batch_size, 1), cached_seq_len, 
                                     dtype=torch.long, device=input_ids.device)
            
            # Update tracking for next decode step
            self.last_absolute_position = cached_seq_len

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                # STATELESS MODE: Always force past_key_values=None and use_cache=False
                # to decouple from KV cache mechanism. Every token is appended to every
                # preceding token and fed back into the model.
                "past_key_values": None,
                "use_cache": False,
                "attention_mask": attention_mask,
                "medusa_args": (accepted_indices, current_length, medusa_mask, scatter_index),
                "sampling_params": sampling_params,
                "input_capture_hook": input_capture_hook,
                "tensor_capture_hook": tensor_capture_hook,
                "adapter_ids": adapter_ids,
            }
        )

        # WARNING: This is needed for propagating additional kwargs to the neuron model
        additional_kwargs = self.neuron_model.get_required_kwargs()
        for arg in additional_kwargs:
            model_inputs.update({arg: kwargs.get(arg, None)})

        return model_inputs

    def prepare_medusa_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if self.neuron_model.kv_cache_populated:
            input_ids = input_ids[:, -self.neuron_config.medusa_speculation_length :]
        position_ids = kwargs.get("position_ids")

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "adapter_ids": kwargs.get("adapter_ids", None),
                "sampling_params": kwargs.get("sampling_params", None),
                "medusa_args": (
                    kwargs.get("accepted_indices"),
                    kwargs.get("current_length"),
                    kwargs.get("medusa_mask"),
                    kwargs.get("scatter_index"),
                ),
            }
        )
        return model_inputs

    # We override this function because we want to change the way attention_mask
    # is updated each iteration.
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_for_token_generation: Optional[bool] = None,
        is_encoder_decoder: bool = False,
    ) -> Dict[str, Any]:
        if is_for_token_generation is None:
            is_for_token_generation = self.prev_kv_cache_populated

        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1
            )

        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            # In stateless mode (KV cache disabled), always update attention_mask
            # In decode mode (KV cache enabled), use token generation logic
            is_stateless_mode = not self.neuron_model.kv_cache_populated
            
            if is_stateless_mode:
                # Stateless mode: simply append a new 1 to attention_mask
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                    dim=-1,
                )
            elif is_for_token_generation:
                # Decode mode with KV cache: use existing logic
                if self.padding_side == "left":
                    attention_mask = torch.cat(
                        [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                        dim=-1,
                    )
                    attention_mask = attention_mask[:, 1:]
                else:
                    attention_mask = torch.cat(
                        [attention_mask.new_ones((attention_mask.shape[0], 1)), attention_mask],
                        dim=-1,
                    )
            model_kwargs["attention_mask"] = attention_mask
        return model_kwargs

    def _update_model_kwargs_for_fused_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        accepted_len: int = 0,
    ):
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            if self.padding_side == "left":
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        attention_mask.new_ones((attention_mask.shape[0], accepted_len)),
                    ],
                    dim=-1,
                )
                attention_mask = attention_mask[:, 1:]
            else:
                attention_mask = torch.cat(
                    [
                        attention_mask.new_ones((attention_mask.shape[0], accepted_len)),
                        attention_mask,
                    ],
                    dim=-1,
                )
            model_kwargs["attention_mask"] = attention_mask

        return model_kwargs

    def _assisted_decoding(
        self,
        input_ids: torch.LongTensor,
        candidate_generator: "CandidateGenerator",  # noqa
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        **model_kwargs,
    ):
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        if not self.neuron_config.enable_fused_speculation:
            assistant_model = candidate_generator.assistant_model
        if self.neuron_config.is_medusa:
            # TODO: move this to sampling
            return self._medusa_assisted_decoding(
                input_ids,
                assistant_model,
                stopping_criteria,
                pad_token_id,
                eos_token_id,
                **model_kwargs,
            )
        elif self.neuron_config.enable_fused_speculation:
            return self._fused_assisted_decoding(
                input_ids,
                stopping_criteria,
                pad_token_id,
                eos_token_id,
                generation_config,
                **model_kwargs,
            )
        else:
            return self._standard_assisted_decoding(
                input_ids,
                assistant_model,
                stopping_criteria,
                pad_token_id,
                eos_token_id,
                **model_kwargs,
            )

    def _fused_assisted_decoding(
        self,
        input_ids,
        stopping_criteria,
        pad_token_id,
        eos_token_id,
        generation_config,
        **model_kwargs,
    ):
        # Init values
        if eos_token_id is not None and pad_token_id is None:
            raise ValueError(
                "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
            )
        if not isinstance(eos_token_id, list):
            eos_token_id_list = list([eos_token_id])
        else:
            eos_token_id_list = eos_token_id
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate

        # init scores / logits tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None

        fused_assistant_kwargs = copy.deepcopy(model_kwargs)
        sampling_params = prepare_sampling_params(
            batch_size=input_ids.shape[0],
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            temperature=generation_config.temperature,
        )
        if "sampling_params" not in fused_assistant_kwargs:
            fused_assistant_kwargs["sampling_params"] = sampling_params
        model_inputs = self.prepare_inputs_for_generation(input_ids, **fused_assistant_kwargs)

        # Other auxiliary variables
        bs = input_ids.shape[0]
        max_len = stopping_criteria[0].max_length
        cur_len = input_ids.shape[-1]
        spec_len = self.neuron_config.speculation_length

        # Prompt encoding
        outputs = self(**model_inputs)
        new_token = outputs.fused_outputs[0][:, 0].view(
            bs, 1
        )  # The target generate token after ctx encoding

        returned_ids = new_token
        incremental_len = 0
        end_for_all = False

        # NOTE: fused_outputs looks like the following:
        # [
        #     padded_accepted_tokens,
        #     next_input_ids,
        #     next_attn_mask,
        #     next_pos_ids,
        #     <draft_logits>,
        #     <target_logits>
        # ] <> means that it may or may not be there.
        if return_dict_in_generate:
            if output_scores:
                # TODO: Process raw logits with logits processor when needed
                scores += (outputs.fused_outputs[-1][:, -1, :],)
            if output_logits:
                raw_logits += (outputs.fused_outputs[-1],)

        while True:
            # 1. update the kwargs for fused generation
            fused_assistant_kwargs = self._update_model_kwargs_for_fused_generation(
                outputs, fused_assistant_kwargs, incremental_len
            )
            model_inputs = self.prepare_inputs_for_generation(
                returned_ids, **fused_assistant_kwargs
            )
            # 2. get the generated draft tokens and target tokens
            outputs = self(**model_inputs)

            accepted_tokens_with_padding = outputs.fused_outputs[0]
            next_pos_ids = outputs.fused_outputs[3]
            n_matches = next_pos_ids - model_inputs["position_ids"]
            n_matches = torch.ops.aten.Int(n_matches)
            incremental_len = n_matches

            # 3. retrieve accepted tokens using n_matches
            if len(accepted_tokens_with_padding.shape) == 1:
                accepted_tokens_with_padding.reshape(
                    self.neuron_config.batch_size, self.neuron_config.speculation_length
                )
            accepted_tokens = accepted_tokens_with_padding[:, :n_matches]

            eos_pos = accepted_tokens.shape[1]
            for eos_token_id in eos_token_id_list:
                if eos_token_id in accepted_tokens:
                    # get column indices
                    eos_pos_cur = (accepted_tokens == eos_token_id).nonzero(as_tuple=True)[1]
                    eos_pos = min(torch.min(eos_pos_cur), eos_pos)
            if eos_pos < accepted_tokens.shape[1]:
                end_for_all = True
                accepted_tokens = accepted_tokens[:, : eos_pos + 1]

            returned_ids = torch.cat((returned_ids, accepted_tokens), dim=1)

            if return_dict_in_generate:
                if output_scores:
                    # TODO: Process raw logits with logits processor when needed
                    scores += tuple(outputs.fused_outputs[-1][:, i, :] for i in range(n_matches))
                if output_logits:
                    raw_logits += (outputs.fused_outputs[-1],)

            # 5. Update with the generated token length and check for stopping condition.
            if end_for_all:
                break
            if returned_ids[:, -1:][0] in torch.tensor(eos_token_id_list):
                break
            cur_len = cur_len + n_matches
            if cur_len >= max_len:
                break
            if max_len - cur_len <= spec_len:
                break

        output_ids = torch.cat((input_ids, returned_ids), dim=1)

        if return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=output_ids,
                scores=scores,
                logits=raw_logits,
            )
        else:
            return output_ids

    def _standard_assisted_decoding(
        self,
        input_ids,
        assistant_model,
        stopping_criteria,
        pad_token_id,
        eos_token_id,
        **model_kwargs,
    ):
        # Implementation of standard assisted decoding

        # Initialize the num_assistant_tokens used for speculation.
        if hasattr(assistant_model, "num_assistant_tokens"):
            num_assistant_tokens = assistant_model.num_assistant_tokens
        else:
            num_assistant_tokens = assistant_model.generation_config.num_assistant_tokens

        # Init values
        if eos_token_id is not None and pad_token_id is None:
            raise ValueError(
                "If `eos_token_id` is defined, make sure that `pad_token_id` is defined."
            )
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        )

        # Prepare assistant model's keys of inputs
        assistant_kwargs = copy.deepcopy(model_kwargs)

        # Other auxiliary variables
        max_len = stopping_criteria[0].max_length
        cur_len = input_ids.shape[-1]
        spec_len = self.neuron_config.speculation_length

        # Run the target model once and get the first generated token
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = self(**model_inputs)

        curr_pos = model_inputs["position_ids"][0].argmax(dim=-1)
        new_token = outputs.logits[:, 0].argmax(dim=-1, keepdim=True)

        # Prepare the input ids and attention mask for the draft model
        candidate_input_ids = input_ids

        # This is the finally return outputs; append the first generated token
        returned_ids = torch.cat((input_ids[:, : curr_pos + 1], new_token), dim=1)

        # Speculation loop
        while True:
            # 1 Token generation using draft model
            for _ in range(int(num_assistant_tokens)):
                # 1.1 Prepare assistant model inputs
                assistant_inputs = assistant_model.prepare_inputs_for_generation(
                    candidate_input_ids,
                    **assistant_kwargs,
                )
                is_for_token_generation = assistant_model.neuron_model.kv_cache_populated

                # 1.2 Use the assistant model to obtain the next candidate logits
                assistant_model_outputs = assistant_model(**assistant_inputs)
                assistant_new_token = assistant_model_outputs.logits[:, 0, :].argmax(dim=-1)

                # 1.3 Update inputs and args for next iteration
                candidate_input_ids = torch.cat(
                    (candidate_input_ids, assistant_new_token[:, None]), dim=-1
                )
                assistant_kwargs = assistant_model._update_model_kwargs_for_generation(
                    assistant_model_outputs,
                    assistant_kwargs,
                    is_for_token_generation,
                    is_encoder_decoder=assistant_model.config.is_encoder_decoder,
                )

                # 1.4 Stop assistant generation on EOS
                if eos_token_id_tensor is not None:
                    last_assistant_token_is_eos = assistant_new_token.tile(
                        eos_token_id_tensor.shape[0], 1
                    )
                    last_assistant_token_is_eos = (
                        ~last_assistant_token_is_eos.ne(eos_token_id_tensor.unsqueeze(1))
                        .prod(dim=0)
                        .bool()
                    )
                    if last_assistant_token_is_eos:
                        break
                else:
                    last_assistant_token_is_eos = False

            # 2 Validation of draft model output using the original model
            #   The length could be shorter if the draft loop ends earlier
            candidate_length = candidate_input_ids.shape[1] - input_ids.shape[1]

            # 2.1 Prepare the input arguments
            input_ids = torch.cat((new_token, candidate_input_ids[:, -candidate_length:-1]), dim=-1)
            attention_mask = model_inputs["attention_mask"]
            pos = curr_pos + 1
            position_ids = torch.arange(pos, pos + spec_len).expand(1, spec_len)
            # Pad the input_ids if needed
            if input_ids.shape[-1] < spec_len:
                input_ids = torch.cat(
                    (input_ids, torch.full((1, spec_len - input_ids.shape[-1]), pad_token_id)),
                    dim=-1,
                )

            # 2.2. Run a forward pass on the candidate sequence
            outputs = self(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # 2.3. Process the new logits
            new_tokens = outputs.logits.argmax(dim=-1)
            selected_tokens = outputs.logits[:, : candidate_length - 1].argmax(dim=-1)

            # 3. Compare the argmax from the original model logits with the assistant forecasted tokens. We can keep
            # the assistant forecasted tokens until the first mismatch, or until the max length is reached.
            candidate_new_tokens = candidate_input_ids[:, -candidate_length:-1]
            n_matches = ((~(candidate_new_tokens == selected_tokens)).cumsum(dim=-1) < 1).sum()

            # 4. Ensure we don't generate beyond max_len or an EOS token
            if last_assistant_token_is_eos and n_matches == candidate_length:
                n_matches -= 1
            n_matches = min(n_matches, max_len - cur_len - 1)
            # n_matches = 4

            # 5. Get the valid continuation, after the matching tokens. We also consider the extra token
            # generated by the original model. Update the return ids accordingly
            valid_tokens = new_tokens[:, : n_matches + 1]
            returned_ids = torch.cat((returned_ids, valid_tokens), dim=1)
            # if last_assistant_token_is_eos and n_matches == candidate_length-1:
            #    break;

            # 6. Update the args for the next iteration.
            #    Feed the last correct token to the next loop
            new_token = valid_tokens[:, -1:]
            if new_token[0].item() in torch.tensor(eos_token_id):
                break
            input_ids = valid_tokens[:, -1:]
            candidate_input_ids = valid_tokens[:, -1:]
            model_inputs_attn_mask = model_inputs["attention_mask"]
            n_matches_concat_tensor = torch.zeros(
                1, n_matches + 1, dtype=model_inputs_attn_mask.dtype
            )
            model_inputs_attn_mask = torch.cat(
                [model_inputs_attn_mask, n_matches_concat_tensor], dim=-1
            )
            model_inputs["attention_mask"] = model_inputs_attn_mask.index_fill(
                1, torch.arange(curr_pos + 1, curr_pos + 1 + n_matches + 1), 1
            )

            curr_pos = curr_pos + n_matches + 1
            assistant_kwargs["attention_mask"] = copy.deepcopy(model_inputs["attention_mask"])

            # 7. Update with the generated token length and check for stopping condition.
            cur_len = cur_len + n_matches + 1
            if cur_len >= max_len:
                break
            # 8. If the rest length is smaller than speculation length, we directly run the target model to finish
            if max_len - cur_len < spec_len:
                # @yihsian: TODO: complete with using target tokengen model
                break

        return returned_ids

    def _medusa_assisted_decoding(
        self,
        input_ids,
        assistant_model,
        stopping_criteria,
        pad_token_id,
        eos_token_id,
        **model_kwargs,
    ):
        medusa_kwargs = copy.deepcopy(model_kwargs)

        mc_sim_7b_63 = self.neuron_config.medusa_tree

        medusa_buffers = generate_medusa_buffers(mc_sim_7b_63)

        model_inputs = self.prepare_inputs_for_generation(input_ids, **medusa_kwargs)

        outputs = self(**model_inputs)

        non_zero_input_ids = input_ids.nonzero()
        cur_len = torch.tensor([non_zero_input_ids.size(0)], dtype=torch.int32)

        logits, medusa_logits = self._extract_logits(outputs)

        medusa_logits = medusa_logits[:, :, None, :]

        accept_length = 0
        final_accept_length = 0
        new_token = 0
        accept_lengths_tree = []
        cur_length = cur_len[0].item() + 1
        accept_lengths_tree.append(1)
        count = 0
        select_indices = torch.arange(
            cur_len[0].item(),
            cur_len[0].item() + self.neuron_config.num_medusa_heads + 1,
            dtype=torch.int32,
        )

        for i in range(self.neuron_config.max_new_tokens):
            count = count + 1
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
            )
            position_ids = medusa_buffers["medusa_position_ids"] + input_ids.nonzero().shape[0]

            medusa_kwargs = self._prepare_medusa_kwargs(
                position_ids, cur_len, medusa_buffers, select_indices, medusa_kwargs
            )

            tree_candidates = tree_candidates.long()

            model_inputs = self.prepare_medusa_inputs_for_generation(
                tree_candidates, **medusa_kwargs
            )

            outputs = self(**model_inputs)

            tree_logits, tree_medusa_logits = self._extract_logits(outputs)
            logits = tree_logits[0, 0, medusa_buffers["retrieve_indices"]]
            medusa_logits = tree_medusa_logits[:, 0, 0, medusa_buffers["retrieve_indices"]]

            best_candidate, accept_length = evaluate_posterior(logits, candidates)
            cur_len = torch.tensor([input_ids.nonzero().size(0) - 1], dtype=torch.int32)

            input_ids, logits, medusa_logits, new_token, select_indices = update_inference_inputs(
                input_ids[:, : (int(cur_len[0] + 1))],
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
                new_token,
            )

            medusa_kwargs["attention_mask"] = self._update_attention_mask(
                model_inputs, accept_length, cur_len, medusa_kwargs
            )
            cur_len = 1 + cur_len
            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_lengths_tree.append(accept_length_tree)
            final_accept_length += accept_length + 1
            if eos_token_id in new_token or final_accept_length > self.neuron_config.max_new_tokens:
                break
        return input_ids

    def _prepare_medusa_kwargs(
        self, position_ids, cur_len, medusa_buffers, select_indices, medusa_kwargs
    ):
        medusa_kwargs["position_ids"] = position_ids.unsqueeze(0)
        medusa_kwargs["accepted_indices"] = torch.arange(
            cur_len[0].item(),
            cur_len[0].item() + self.neuron_config.num_medusa_heads + 1,
            dtype=torch.int32,
        )
        for index, value in enumerate(select_indices):
            medusa_kwargs["accepted_indices"][index] = value
        medusa_kwargs["accepted_indices"] = medusa_kwargs["accepted_indices"].unsqueeze(0)
        medusa_kwargs["current_length"] = torch.arange(
            cur_len[0].item(),
            cur_len[0].item() + self.neuron_config.num_medusa_heads + 1,
            dtype=torch.int32,
        ).unsqueeze(0)
        medusa_mask = medusa_buffers["medusa_attn_mask"].unsqueeze(0)
        medusa_kwargs["medusa_mask"] = medusa_mask.type_as(torch.LongTensor())
        medusa_kwargs["scatter_index"] = torch.arange(
            position_ids[0],
            position_ids[0] + self.neuron_config.medusa_speculation_length,
            dtype=torch.int32,
        ).unsqueeze(0)
        return medusa_kwargs

    def _update_attention_mask(self, model_inputs, accept_length, cur_len, medusa_kwargs):
        accept_length_concat_tensor = torch.zeros(
            1, accept_length + 1, dtype=model_inputs["attention_mask"].dtype
        )
        attn_mask = torch.cat([model_inputs["attention_mask"], accept_length_concat_tensor], dim=-1)

        medusa_kwargs["attention_mask"] = attn_mask.index_fill(
            1, torch.arange(int(cur_len[0]) + 1, int(cur_len[0]) + 1 + accept_length + 1), 1
        )
        return medusa_kwargs["attention_mask"]

    def _extract_logits(self, outputs):
        logits = outputs["hidden_states"][:1, :, :]
        medusa_logits = outputs["hidden_states"][1:, :, :].unsqueeze(1)
        return logits, medusa_logits

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")
