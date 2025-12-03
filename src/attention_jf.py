
class NeuronGPTOSSAttentionBlockCompiled(NeuronAttentionBase):
    """
    Attention block using NeuronAttentionBase with native compiler path.
    Implements the SDPA logic from gpt_oss.py using PyTorch operations.
    """
    def __init__(self, config: InferenceConfig, layer_idx: int = 0, weight_init_value: float = None):
        rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
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
        
        bsz, q_len, _ = hidden_states.shape

        # Use rotary_position_ids if provided, otherwise use position_ids
        rope_position_ids = rotary_position_ids if rotary_position_ids is not None else position_ids

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
            skip_rope=False,  # Always apply RoPE
            residual=residual,
        )
        
        # Q, K, V shapes from prep_qkv_tensors:
        # Q: [B, num_heads, S, D]
        # K: [B, num_kv_heads, S, D]
        # V: [B, num_kv_heads, S, D]
        
        from neuronx_distributed_inference.modules.attention.utils import repeat_kv

        # Expand K, V for GQA
        n_rep = self.num_heads // self.num_key_value_heads
        K_expanded = repeat_kv(K, n_rep)
        V_expanded = repeat_kv(V, n_rep)
        
        # Compute attention: Q @ K^T
        attn_weights = torch.matmul(Q, K_expanded.transpose(2, 3))
        attn_weights = attn_weights * self.sm_scale

        # Build causal mask (add batch dimension)
        mask = torch.triu(Q.new_full((q_len, q_len), -float("inf")), diagonal=1)
        if self.sliding_window is not None and self.sliding_window > 0:
            mask += torch.tril(mask.new_full((q_len, q_len), -float("inf")),
                            diagonal=-self.sliding_window)
        attn_weights = attn_weights + mask[None, None, :, :]

        # Learned sinks
        learned_sinks = self.get_learned_sinks()
        if learned_sinks is not None:
            # sinks: [num_heads] -> [1, num_heads, 1, 1] -> [B, num_heads, S, 1]
            sinks = learned_sinks.view(1, -1, 1, 1).expand(bsz, -1, q_len, -1)
            attn_weights = torch.cat([attn_weights, sinks], dim=-1)

        # Numerical stability
        attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values

        # Softmax
        attn_probs = torch.softmax(attn_weights, dim=-1)

        # Remove sink weights
        if learned_sinks is not None:
            attn_probs = attn_probs[..., :-1]

        # Compute output
        attn_output = torch.matmul(attn_probs, V_expanded)
        
        # Reshape: [B, num_heads, S, D] -> [B, S, num_heads * D]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        
        attn_output = self.get_o_proj()(attn_output, adapter_ids=adapter_ids)
        K_cache = K # [B, num_kv_heads, S, D]
        V_cache = V # [B, num_kv_heads, S, D]

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
        # The full output would be NeuronAttentionBaseOutput(attn_output, kv, cos_cache, sin_cache, residual)
        return attn_output, kv, cos_cache, sin_cache