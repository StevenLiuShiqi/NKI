import sys
import os

import torch 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import (
    GPTOSSInferenceConfig,
    NeuronGPTOSSConfig,
)
from src.gpt_oss import ModelConfig


def _make_tiny_inference_config():
    neuron_config = NeuronGPTOSSConfig(
        batch_size=2,
        seq_len=1024,
        tp_degree=1,
        torch_dtype="bfloat16",
        # glu_mlp=True,
        capacity_factor=None,
        fused_qkv=True
    )
    return GPTOSSInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=8,
        intermediate_size=16,
        num_local_experts=4,
        num_experts_per_tok=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=64,
        max_position_embeddings=32,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
        pad_token_id=0,
        rope_theta=10000.0,
        num_experts=4,
    )

def _make_original_inference_config():
    # Match the released GPT-OSS 20B configuration.
    neuron_config = NeuronGPTOSSConfig(
        batch_size=1,
        seq_len=4096,
        tp_degree=1,
        torch_dtype=torch.bfloat16,
        capacity_factor=None,
    )
    return GPTOSSInferenceConfig(
        neuron_config=neuron_config,
        hidden_size=2880,
        intermediate_size=2880,
        num_local_experts=32,
        num_experts_per_tok=4,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=64,
        vocab_size=201088,
        max_position_embeddings=131072,
        num_hidden_layers=24,
        rms_norm_eps=1e-5,
        pad_token_id=199999,
        rope_theta=150000.0,
        sliding_window=128,
        num_experts=32,
    )

def _fill_module_parameters(module: torch.nn.Module, value: float) -> None:
    """Fill all module parameters with a constant value (for uniform initialization testing)."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)

def _sync_reference_weights_to_neuron(reference_block, neuron_block, layer_idx: int = 0):
    """
    Copy weights from reference AttentionBlock to NeuronGPTOSSAttentionBlock.

    This ensures both models use identical random weights for testing.
    Handles the name mapping between reference and neuron implementations.

    Args:
        reference_block: Instance of src.gpt_oss.AttentionBlock (with random weights)
        neuron_block: Instance of src.model.NeuronGPTOSSAttentionBlock (to be populated)
        layer_idx: Layer index (used for debugging/logging)
    """
    with torch.no_grad():
        ref_state = reference_block.state_dict()
         # Unwrap neuron_block if it's wrapped by build_module
        # build_module wraps the module, so we need to access .module or ._module
        actual_neuron_block = neuron_block
        if hasattr(neuron_block, 'module'):
            actual_neuron_block = neuron_block.module
            print(f"[Layer {layer_idx}] Detected wrapped module, using .module")
        elif hasattr(neuron_block, '_module'):
            actual_neuron_block = neuron_block._module
            print(f"[Layer {layer_idx}] Detected wrapped module, using ._module")

        neuron_state = actual_neuron_block.state_dict()

        # Debug: Print available keys
        print(f"\n[Layer {layer_idx}] Reference block keys:")
        for k in sorted(ref_state.keys()):
            print(f"  - {k}: {ref_state[k].shape}")

        print(f"\n[Layer {layer_idx}] Neuron block keys:")
        for k in sorted(neuron_state.keys()):
            print(f"  - {k}: {neuron_state[k].shape}")

        # Map reference keys to neuron keys
        weight_mapping = {
            # QKV projection (reference has fused qkv, neuron expects qkv_proj.Wqkv)
            'qkv.weight': 'qkv_proj.Wqkv.weight',
            'qkv.bias': 'qkv_proj.Wqkv.bias',

            # Output projection
            'out.weight': 'o_proj.o_proj.weight',
            'out.bias': 'o_proj.o_proj.bias',

            # Learned sinks - reference has one 'sinks' param
            # Neuron has two: learned_sinks.sink (CTE) and tkg_learned_sinks.sink (TKG)
            'sinks': ['learned_sinks.sink', 'tkg_learned_sinks.sink'],
        }

        # Copy weights with name transformation
        updated_keys = []
        for ref_key, neuron_key in weight_mapping.items():
            if ref_key not in ref_state:
                print(f"  [SKIP] {ref_key} not found in reference state")
                continue

            ref_tensor = ref_state[ref_key]

            # Handle special case: sinks maps to multiple neuron params
            if isinstance(neuron_key, list):
                for nk in neuron_key:
                    if nk in neuron_state:
                        # Clone to avoid sharing the same tensor
                        neuron_state[nk].copy_(ref_tensor.clone())
                        updated_keys.append(nk)
                        print(f"  [COPY] {ref_key} -> {nk}")
                    else:
                        print(f"  [WARN] {nk} not found in neuron state")
            else:
                if neuron_key in neuron_state:
                    # Verify shapes match
                    if ref_tensor.shape != neuron_state[neuron_key].shape:
                        raise ValueError(
                            f"Shape mismatch for {ref_key} -> {neuron_key}: "
                            f"{ref_tensor.shape} vs {neuron_state[neuron_key].shape}"
                        )
                    neuron_state[neuron_key].copy_(ref_tensor)
                    updated_keys.append(neuron_key)
                    print(f"  [COPY] {ref_key} -> {neuron_key}")
                else:
                    print(f"  [WARN] {neuron_key} not found in neuron state")

        # Load the updated state dict back into the actual neuron block
        actual_neuron_block.load_state_dict(neuron_state, strict=False)

        print(f"\n[Layer {layer_idx}] Successfully copied {len(updated_keys)} weight tensors")
        print(f"  Updated keys: {updated_keys}")

        # Verify a sample weight to ensure copy worked
        if 'qkv.weight' in ref_state and 'qkv_proj.Wqkv.weight' in neuron_state:
            ref_sample = ref_state['qkv.weight'][0, :5]
            neuron_sample = actual_neuron_block.state_dict()['qkv_proj.Wqkv.weight'][0, :5]
            print(f"\n  Verification - QKV weight[0, :5]:")
            print(f"    Reference: {ref_sample}")
            print(f"    Neuron:    {neuron_sample}")
            if torch.allclose(ref_sample, neuron_sample, rtol=1e-5, atol=1e-8):
                print(f"    ✓ Weights match!")
            else:
                print(f"    ✗ WARNING: Weights don't match!")

        return updated_keys

def _get_ref_config(config):
    reference_config = ModelConfig(
        num_hidden_layers=config.num_hidden_layers,
        num_experts=config.num_local_experts,
        experts_per_token=config.num_experts_per_tok,
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        head_dim=config.head_dim,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        sliding_window=getattr(config, "sliding_window", ModelConfig.sliding_window),
        initial_context_length=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rope_scaling_factor=1.0,
        rope_ntk_alpha=getattr(config, "rope_ntk_alpha", ModelConfig.rope_ntk_alpha),
        rope_ntk_beta=getattr(config, "rope_ntk_beta", ModelConfig.rope_ntk_beta),
    )
    
    return reference_config