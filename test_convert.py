import types
import copy
import torch
import pytest
import gc

# Adjust this import to wherever your function lives:

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

def make_config(d_model=6, n_layers=2, E=3, I=6, tp_degree=2):
    cfg = types.SimpleNamespace()
    cfg.d_model = d_model
    cfg.n_layers = n_layers

    ffn_cfg = types.SimpleNamespace()
    ffn_cfg.moe_num_experts = E
    ffn_cfg.ffn_hidden_size = I
    cfg.ffn_config = ffn_cfg

    neuron_cfg = types.SimpleNamespace()
    neuron_cfg.tp_degree = tp_degree
    cfg.neuron_config = neuron_cfg
    return cfg

def make_gptoss_sd(H=6, L=2, E=3, I=6, *,
                   router_as_EH=True,
                   down_as_EHI=True,
                   include_biases=True):
    """
    Build a tiny, shape‑compatible GPT‑OSS state_dict.
    We don't care about absolute dims for q,k,v as long as second dim==H.
    """
    vocab = 10
    gsd = {}

    # top-level
    gsd["model.embed_tokens.weight"] = torch.randn(vocab, H)
    gsd["model.norm.weight"] = torch.randn(H)
    gsd["lm_head.weight"] = torch.randn(vocab, H)

    for l in range(L):
        # norms (also used to count L in the function)
        gsd[f"model.layers.{l}.input_layernorm.weight"] = torch.randn(H)
        gsd[f"model.layers.{l}.post_attention_layernorm.weight"] = torch.randn(H)

        # attention q,k,v (2D, second dim = H)
        q_out, kv_out = 8, 1
        gsd[f"model.layers.{l}.self_attn.q_proj.weight"] = torch.randn(q_out, H)
        gsd[f"model.layers.{l}.self_attn.k_proj.weight"] = torch.randn(kv_out, H)
        gsd[f"model.layers.{l}.self_attn.v_proj.weight"] = torch.randn(kv_out, H)
        if include_biases:
            gsd[f"model.layers.{l}.self_attn.q_proj.bias"] = torch.randn(q_out)
            gsd[f"model.layers.{l}.self_attn.k_proj.bias"] = torch.randn(kv_out)
            gsd[f"model.layers.{l}.self_attn.v_proj.bias"] = torch.randn(kv_out)

        # o_proj (any (H, something))
        gsd[f"model.layers.{l}.self_attn.o_proj.weight"] = torch.randn(H, q_out)
        if include_biases:
            gsd[f"model.layers.{l}.self_attn.o_proj.bias"] = torch.randn(H)

        # router
        if router_as_EH:
            rW = torch.randn(E, H)
        else:
            rW = torch.randn(H, E)  # transposed case
        gsd[f"model.layers.{l}.mlp.router.weight"] = rW
        if include_biases:
            gsd[f"model.layers.{l}.mlp.router.bias"] = torch.randn(E)

        # experts
        gsd[f"model.layers.{l}.mlp.experts.gate_up_proj"] = torch.randn(E, H, 2*I)
        if down_as_EHI:
            gsd[f"model.layers.{l}.mlp.experts.down_proj"] = torch.randn(E, H, I)
        else:
            gsd[f"model.layers.{l}.mlp.experts.down_proj"] = torch.randn(E, I, H)

        if include_biases:
            gsd[f"model.layers.{l}.mlp.experts.gate_up_proj_bias"] = torch.randn(E, 2*I)
            gsd[f"model.layers.{l}.mlp.experts.down_proj_bias"] = torch.randn(E, I)

    return gsd

def test_happy_path_transpose_down_and_biases():
    H, L, E, I, tp = 6, 2, 3, 6, 2
    cfg = make_config(H, L, E, I, tp)
    gsd = make_gptoss_sd(H, L, E, I, router_as_EH=True, down_as_EHI=True, include_biases=True)

    gsd_copy = copy.deepcopy(gsd)
    nsd = convert_gptoss_to_neuron_state_dict(gsd_copy, cfg)

    # top-level exist
    assert "embed_tokens.weight" in nsd
    assert "norm.weight" in nsd
    assert "lm_head.weight" in nsd

    for l in range(L):
        # Wqkv packed (dim0 = q_out + kv_out + kv_out = 8+1+1=10, dim1 = H)
        Wqkv = nsd[f"layers.{l}.self_attn.Wqkv.weight"]
        assert Wqkv.shape[1] == H and Wqkv.shape[0] == 10

        # biases present and concatenated
        Wqkv_b = nsd[f"layers.{l}.self_attn.Wqkv.bias"]
        assert Wqkv_b.shape[0] == 10

        # o_proj bias copied
        assert nsd[f"layers.{l}.self_attn.o_proj.weight"].shape == (H, 8)
        assert nsd[f"layers.{l}.self_attn.o_proj.bias"].shape == (H,)

        # router EH retained
        assert nsd[f"layers.{l}.ffn.router.linear_router.weight"].shape == (E, H)
        assert nsd[f"layers.{l}.ffn.router.linear_router.bias"].shape == (E,)

        # experts: gate_up [E,H,2I]; down transposed to [E,I,H]
        assert nsd[f"layers.{l}.ffn.expert_mlps.mlp_op.gate_up_proj.weight"].shape == (E, H, 2*I)
        assert nsd[f"layers.{l}.ffn.expert_mlps.mlp_op.down_proj.weight"].shape == (E, I, H)
        assert nsd[f"layers.{l}.ffn.expert_mlps.mlp_op.gate_up_proj.bias"].shape == (E, 2*I)
        assert nsd[f"layers.{l}.ffn.expert_mlps.mlp_op.down_proj.bias"].shape == (E, I)

        # norms
        assert nsd[f"layers.{l}.input_layernorm.weight"].shape == (H,)
        assert nsd[f"layers.{l}.post_attention_layernorm.weight"].shape == (H,)

        # TP metadata
        rank = nsd[f"layers.{l}.self_attn.rank_util.rank"]
        assert rank.dtype == torch.int32
        assert torch.equal(rank, torch.arange(0, tp, dtype=torch.int32))

def test_router_transposed_and_down_already_EIH_no_biases():
    H, L, E, I, tp = 6, 1, 4, 6, 3
    cfg = make_config(H, L, E, I, tp)
    gsd = make_gptoss_sd(H, L, E, I, router_as_EH=False, down_as_EHI=False, include_biases=False)

    nsd = convert_gptoss_to_neuron_state_dict(copy.deepcopy(gsd), cfg)

    # router weight should be transposed to (E,H)
    rW = nsd["layers.0.ffn.router.linear_router.weight"]
    assert rW.shape == (E, H)

    # no biases exist
    assert "layers.0.self_attn.Wqkv.bias" not in nsd
    assert "layers.0.self_attn.o_proj.bias" not in nsd
    assert "layers.0.ffn.router.linear_router.bias" not in nsd
    assert "layers.0.ffn.expert_mlps.mlp_op.gate_up_proj.bias" not in nsd
    assert "layers.0.ffn.expert_mlps.mlp_op.down_proj.bias" not in nsd

    # down already [E,I,H], should be kept
    down = nsd["layers.0.ffn.expert_mlps.mlp_op.down_proj.weight"]
    assert down.shape == (E, I, H)

def test_bad_down_shape_raises():
    H, L, E, I = 5, 1, 2, 4
    cfg = make_config(H, L, E, I, tp_degree=2)
    gsd = make_gptoss_sd(H, L, E, I)

    # Corrupt down_proj to an unexpected shape
    gsd[f"model.layers.0.mlp.experts.down_proj"] = torch.randn(E, I+1, H+1)

    with pytest.raises(ValueError):
        convert_gptoss_to_neuron_state_dict(copy.deepcopy(gsd), cfg)

def test_input_dict_is_cleared():
    cfg = make_config()
    gsd = make_gptoss_sd()
    assert len(gsd) > 0
    convert_gptoss_to_neuron_state_dict(gsd, cfg)
    # Your function clears the dict in-place
    assert len(gsd) == 0
