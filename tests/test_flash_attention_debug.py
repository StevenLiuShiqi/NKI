"""
Debug test to check tensor shapes and basic functionality.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gpt_oss import sdpa

# Create simple test with MHA (no GQA)
print("Testing tensor shapes and basic SDPA functionality")
print("="*80)

# Simple configuration
n_tokens = 4
num_heads = 2
head_dim = 4
q_mult = 1  # For MHA

# Create simple inputs
Q = torch.ones(n_tokens, num_heads, q_mult, head_dim, dtype=torch.bfloat16)
K = torch.ones(n_tokens, num_heads, head_dim, dtype=torch.bfloat16)
V = torch.arange(n_tokens * num_heads * head_dim, dtype=torch.bfloat16).reshape(n_tokens, num_heads, head_dim)
S = torch.zeros(num_heads, dtype=torch.bfloat16)

sm_scale = 1.0 / np.sqrt(head_dim)

print(f"Input shapes:")
print(f"  Q: {Q.shape} (n_tokens, num_heads, q_mult, head_dim)")
print(f"  K: {K.shape} (n_tokens, num_heads, head_dim)")
print(f"  V: {V.shape} (n_tokens, num_heads, head_dim)")
print(f"  S: {S.shape} (num_heads,)")

with torch.no_grad():
    output = sdpa(Q, K, V, S, sm_scale, sliding_window=0)

print(f"\nOutput shape: {output.shape}")
print(f"Expected: ({n_tokens}, {num_heads * q_mult * head_dim})")
print(f"Output:\n{output}")

# Now test what our module would do
print("\n" + "="*80)
print("Testing FlashAttention tensor transformations")
print("="*80)

# Simulate input to our module (batch, seq, hidden_dim)
batch_size = 1
seq_len = 4
hidden_dim = num_heads * head_dim  # 2 * 4 = 8

Q_in = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
K_in = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)
V_in = torch.randn(batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)

print(f"Module inputs:")
print(f"  Q_in: {Q_in.shape} (batch, seq, hidden)")
print(f"  K_in: {K_in.shape}")
print(f"  V_in: {V_in.shape}")

# What SDPA expects
n_tokens_mod = batch_size * seq_len
Q_sdpa = Q_in.view(n_tokens_mod, num_heads, q_mult, head_dim)
K_sdpa = K_in.view(n_tokens_mod, num_heads, head_dim)
V_sdpa = V_in.view(n_tokens_mod, num_heads, head_dim)

print(f"\nSDPA format:")
print(f"  Q_sdpa: {Q_sdpa.shape}")
print(f"  K_sdpa: {K_sdpa.shape}")
print(f"  V_sdpa: {V_sdpa.shape}")

with torch.no_grad():
    output_sdpa = sdpa(Q_sdpa, K_sdpa, V_sdpa, S, sm_scale, sliding_window=0)

print(f"\nSDPA output: {output_sdpa.shape}")

# Reshape back
output_reshaped = output_sdpa.view(batch_size, seq_len, hidden_dim)
print(f"Reshaped output: {output_reshaped.shape}")

# What FlashAttention expects
print("\n" + "="*80)
print("FlashAttention format transformations")
print("="*80)

Q_flash = Q_in.view(batch_size, seq_len, num_heads, head_dim)
Q_flash = Q_flash.permute(0, 2, 3, 1)  # (batch, heads, head_dim, seq)

K_flash = K_in.view(batch_size, seq_len, num_heads, head_dim)
K_flash = K_flash.permute(0, 2, 3, 1)  # (batch, heads, head_dim, seq)

V_flash = V_in.view(batch_size, seq_len, num_heads, head_dim)
V_flash = V_flash.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

print(f"FlashAttention format:")
print(f"  Q_flash: {Q_flash.shape} (batch, heads, head_dim, seq)")
print(f"  K_flash: {K_flash.shape}")
print(f"  V_flash: {V_flash.shape} (batch, heads, seq, head_dim)")

# FlashAttention output would be (batch, heads, seq, head_dim)
# Need to reshape to (batch, seq, hidden_dim)
# output_flash_expected = (batch, heads, seq, head_dim)
# Step 1: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
# Step 2: (batch, seq, heads, head_dim) -> (batch, seq, hidden_dim)

print("\nExpected output transformation:")
print("  flash_out: (batch, heads, seq, head_dim)")
print("  -> permute(0, 2, 1, 3): (batch, seq, heads, head_dim)")
print("  -> reshape: (batch, seq, hidden_dim)")
