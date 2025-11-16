import os
import sys
from pathlib import Path
import torch
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import NeuronGPTOSSAttentionBlock  # noqa: E402
from src.gpt_oss import AttentionBlock  # noqa: E402

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from test_utils_v1 import (  # noqa: E402
    _fill_module_parameters,
    _get_ref_config,
    _make_tiny_inference_config,
)

# import warnings, logging

# os.environ["NEURON_CC_FLAGS"] = "--verbose=0"
# os.environ["NEURON_LOG_LEVEL"] = "ERROR"
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# os.environ["XLA_FLAGS"] = "--xla_cpu_use_thunk_schedule=false"

# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# logging.getLogger("Neuron").setLevel(logging.ERROR)
# logging.getLogger("neuronx_distributed").setLevel(logging.ERROR)
# logging.getLogger("torch_neuronx").setLevel(logging.ERROR)

def test_attention_block_forward_matches_reference():
    config = _make_tiny_inference_config()
    reference_config = _get_ref_config(config=config)

    batch_size = config.neuron_config.batch_size
    seq_len = config.neuron_config.seq_len
    hidden_size = config.hidden_size

    torch.manual_seed(0)
    sample = torch.arange(batch_size * seq_len * hidden_size, dtype=config.neuron_config.torch_dtype)
    sample = sample.reshape(batch_size, seq_len, hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # =============================
    # Step 1: Initialize both blocks
    # =============================
    neuron_block_raw = NeuronGPTOSSAttentionBlock(config)
    reference_block = AttentionBlock(reference_config)

    # =============================
    # Step 2: Print RotaryEmbedding configs
    # =============================
    print("\n==================== ROTARY CONFIG COMPARISON ====================")
    print("Neuron RotaryEmbedding:")
    neuron_rope = neuron_block_raw.rotary_emb
    print(" - base:", getattr(neuron_rope, "base", None))
    print(" - dim:", getattr(neuron_rope, "dim", None))
    print(" - max_position_embeddings:", getattr(neuron_rope, "max_position_embeddings", None))

    print("\nCPU RotaryEmbedding:")
    cpu_rope = reference_block.rope
    print(" - base:", getattr(cpu_rope, "base", None))
    print(" - head_dim:", getattr(cpu_rope, "head_dim", None))
    print(" - scaling_factor:", getattr(cpu_rope, "scaling_factor", None))
    print(" - ntk_alpha:", getattr(cpu_rope, "ntk_alpha", None))
    print(" - ntk_beta:", getattr(cpu_rope, "ntk_beta", None))
    print(" - initial_context_length:", getattr(cpu_rope, "initial_context_length", None))

    # =============================
    # Step 3: Print inv_freq & cos/sin for first few tokens
    # =============================

    try:
        # Neuron version — some implementations store inv_freq as a tensor directly
        inv_freq_neuron = getattr(neuron_rope, "inv_freq", None)
        if inv_freq_neuron is not None:
            print("\nNeuron inv_freq[:8]:", inv_freq_neuron[:8].detach().cpu().numpy())
        else:
            print("\nNeuron RotaryEmbedding has no direct inv_freq attr, skipping.")

        # CPU version — must call helper
        concentration, inv_freq_cpu = cpu_rope._compute_concentration_and_inv_freq()
        print("CPU inv_freq[:8]:", inv_freq_cpu[:8].detach().cpu().numpy())

        # Compute cos/sin for first few tokens (e.g., 6 tokens)
        cos_cpu, sin_cpu = cpu_rope._compute_cos_sin(num_tokens=6)
        print("\nCPU cos[0,:8]:", cos_cpu[0, :8].detach().cpu().numpy())
        print("CPU sin[0,:8]:", sin_cpu[0, :8].detach().cpu().numpy())

        neuron_dummy_hidden = torch.zeros(
            batch_size,
            config.num_attention_heads,
            seq_len,
            config.head_dim,
            dtype=sample.dtype,
        )
        cos_neuron, sin_neuron = neuron_rope(neuron_dummy_hidden, position_ids)
        print("\nNeuron cos[0,0,:8]:", cos_neuron[0, 0, :8].detach().cpu().numpy())
        print("Neuron sin[0,0,:8]:", sin_neuron[0, 0, :8].detach().cpu().numpy())

    except Exception as e:
        print("Error while inspecting rotary values:", e)

    print("\n==================================================================\n")
    

    print("\nTesting Neuron RotaryEmbedding output behavior:")
    try:
        dummy_hidden = torch.ones(1, 6, 4)  # batch=1, seq_len=6, head_dim=4
        dummy_pos = torch.arange(6).unsqueeze(0)  # position_ids shape [1, 6]

        rope_out = neuron_rope(dummy_hidden, dummy_pos)
        if isinstance(rope_out, tuple):
            print("Neuron rope returned a tuple of length:", len(rope_out))
            for i, t in enumerate(rope_out):
                if isinstance(t, torch.Tensor):
                    print(f"  Tensor {i} shape:", t.shape)
                    print("   first few values:", t.flatten()[:8].detach().cpu().numpy())
                else:
                    print(f"  Tensor {i}:", type(t))
        else:
            print("Neuron rope output type:", type(rope_out))
            print("Neuron rope output shape:", rope_out.shape)
            print("Neuron rope output:", rope_out[0, 0, :4])
    except Exception as e:
        print("Neuron rope forward() failed:", e)




    # =============================
    # Step 4: Build neuron module (optional)
    # =============================
    example_inputs = [(torch.zeros_like(sample), torch.zeros(batch_size, seq_len, dtype=torch.long))]
    neuron_block = build_module(
        NeuronGPTOSSAttentionBlock,
        example_inputs=example_inputs,
        module_init_kwargs={"config": config},
        tp_degree=1,
        checkpoint_path="/home/ubuntu/NKI/tests/artifacts/neuron_attention_checkpoint.pt",
    )

    _fill_module_parameters(neuron_block, 0.5)
    _fill_module_parameters(reference_block, 0.5)

    def cpu_callable(x, pos):
        out = reference_block(x.view(-1, hidden_size))
        return out.view(batch_size, seq_len, hidden_size)

    validate_accuracy(neuron_block, [(sample, position_ids)], cpu_callable=cpu_callable)


if __name__ == "__main__":
    test_attention_block_forward_matches_reference()
