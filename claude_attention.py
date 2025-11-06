import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from model import NeuronGPTOSSAttentionBlock, GPTOSSInferenceConfig, NeuronGPTOSSConfig

# Create tiny config
neuron_config = NeuronGPTOSSConfig(
    batch_size=2,
    seq_len=6,
    tp_degree=1,
    torch_dtype="bfloat16",
    capacity_factor=None,
)
config = GPTOSSInferenceConfig(
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
    sliding_window=128,
)

# Create attention block
attn = NeuronGPTOSSAttentionBlock(config, layer_idx=0)

print("Parameters in Neuron block:")
print("=" * 60)
for name, param in attn.named_parameters():
    print(f"{name}: shape={param.shape}, dtype={param.dtype}")

print("\n" + "=" * 60)
print("Total parameters:", sum(p.numel() for p in attn.parameters()))

# Fill with constant
with torch.no_grad():
    for param in attn.parameters():
        param.fill_(0.5)

print("\nAfter filling with 0.5:")
for name, param in attn.named_parameters():
    print(f"{name}: {param.flatten()[:5]}")

