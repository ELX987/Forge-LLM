"""Pruning tests: shapes, exactness on structurally-dead components, GQA group handling, depth, experts, sparsity."""
from __future__ import annotations

import pytest
import torch

from forge.compress.calibrate import RouterStats, collect_stats
from forge.compress.prune import PruneConfig, SparsityConfig, estimate_importance, prune_model, sparsify_model
from forge.core.errors import ConfigError
from tests.compress_helpers import logits_of, make_lm, make_moe_lm, rand_tokens, rel_err

TOK = rand_tokens(8, 24, seed=5)


def _zero_mlp_channels(model, idx):
    for blk in model.layers:
        with torch.no_grad():
            blk.mlp.gate_proj.weight[idx] = 0
            blk.mlp.up_proj.weight[idx] = 0
            blk.mlp.down_proj.weight[:, idx] = 0


def test_noop_config_rejected():
    with pytest.raises(ConfigError):
        prune_model(make_lm(), PruneConfig(), TOK)


def test_mlp_width_pruning_is_exact_when_dropping_dead_channels():
    m = make_lm()
    dead = list(range(0, 128, 2))  # half of the intermediate channels
    _zero_mlp_channels(m, dead)
    ref = logits_of(m, TOK)
    for method in ("magnitude", "activation", "fisher", "taylor"):
        res = prune_model(m, PruneConfig(mlp_keep=0.5, importance=method, n_seqs=8), TOK)
        assert res.config.intermediate_size == 64
        assert res.model.layers[0].mlp.gate_proj.weight.shape == (64, 64)
        assert res.model.layers[1].mlp.down_proj.weight.shape == (64, 64)
        assert rel_err(logits_of(res.model, TOK), ref) < 1e-10, method
        assert res.report["needs_recovery"] and res.report["params_after"] < res.report["params_before"]
    # the original model is untouched
    assert m.layers[0].mlp.gate_proj.weight.shape == (128, 64)


def test_gqa_group_pruning_exact_on_dead_group_and_structure():
    m = make_lm()  # 4 q heads, 2 kv heads, head_dim 16 -> groups {0,1} and {2,3}
    d = 16
    for blk in m.layers:
        with torch.no_grad():
            blk.self_attn.v_proj.weight[:d] = 0  # kv group 0 produces zero values
            blk.self_attn.o_proj.weight[:, : 2 * d] = 0
    ref = logits_of(m, TOK)
    res = prune_model(m, PruneConfig(head_mode="group", head_keep=0.5, importance="magnitude"), TOK)
    assert (res.config.num_heads, res.config.num_kv_heads) == (2, 1)
    a = res.model.layers[0].self_attn
    assert a.q_proj.weight.shape == (2 * d, 64) and a.k_proj.weight.shape == (d, 64) and a.o_proj.weight.shape == (64, 2 * d)
    # kept group is original group 1: its k rows are rows d:2d of the original
    assert torch.equal(a.k_proj.weight, m.layers[0].self_attn.k_proj.weight[d : 2 * d])
    assert torch.equal(a.q_proj.weight, m.layers[0].self_attn.q_proj.weight[2 * d :])
    assert rel_err(logits_of(res.model, TOK), ref) < 1e-10


def test_per_group_head_pruning_keeps_group_structure():
    m = make_lm()
    res = prune_model(m, PruneConfig(head_mode="per_group", head_keep=0.5, importance="activation"), TOK)
    assert (res.config.num_heads, res.config.num_kv_heads) == (2, 2)
    kept = res.report["kept"]["q_heads"]["0"]
    assert len(kept) == 2 and kept[0] in (0, 1) and kept[1] in (2, 3)
    out = logits_of(res.model, TOK)
    assert torch.isfinite(out).all() and out.shape == (8, 24, 128)


def test_depth_pruning_removes_identity_block_exactly():
    m = make_lm(num_layers=4)
    with torch.no_grad():  # block 2 becomes an identity map (adds nothing to the residual stream)
        m.layers[2].self_attn.o_proj.weight.zero_()
        m.layers[2].mlp.down_proj.weight.zero_()
    ref = logits_of(m, TOK)
    for mode in ("block_influence", "contiguous"):
        res = prune_model(m, PruneConfig(depth_remove=1, depth_mode=mode, importance="magnitude"), TOK)
        assert res.report["layers_removed"] == [2], mode
        assert res.config.num_layers == 3 and len(res.model.layers) == 3
        assert rel_err(logits_of(res.model, TOK), ref) < 1e-10
    res = prune_model(m, PruneConfig(depth_remove=1, depth_mode="explicit", layers_to_remove=[1]), TOK)
    assert res.report["layers_removed"] == [1]
    with pytest.raises(ConfigError):
        prune_model(m, PruneConfig(depth_remove=4), TOK)


def test_hidden_pruning_shapes():
    m = make_lm()
    res = prune_model(m, PruneConfig(hidden_keep=0.75, hidden_round_to=8, importance="activation"), TOK)
    assert res.config.hidden_size == 48 and res.config.head_dim == 16
    assert res.model.embed_tokens.weight.shape == (128, 48)
    assert res.model.layers[0].self_attn.q_proj.weight.shape == (64, 48)
    assert res.model.layers[0].self_attn.o_proj.weight.shape == (48, 64)
    assert res.model.lm_head.weight.shape == (128, 48)
    assert torch.isfinite(logits_of(res.model, TOK)).all()
    kept = res.report["kept"]["hidden"]
    assert len(kept) == 48 and kept == sorted(kept)


def test_combined_prune_and_tied_embeddings():
    m = make_lm(num_layers=3, tie_word_embeddings=True)
    res = prune_model(m, PruneConfig(mlp_keep=0.5, head_mode="group", head_keep=0.5, depth_remove=1, hidden_keep=0.75), TOK)
    c = res.config
    assert (c.intermediate_size, c.num_heads, c.num_kv_heads, c.num_layers, c.hidden_size) == (64, 2, 1, 2, 48)
    assert res.model.lm_head.weight is res.model.embed_tokens.weight
    assert res.report["ratio"] < 0.6
    assert torch.isfinite(logits_of(res.model, TOK)).all()


def test_importance_estimators_shapes():
    m = make_lm()
    for method in ("activation", "fisher", "taylor", "magnitude"):
        imp = estimate_importance(m, TOK, method)
        assert imp.mlp[(0, None)].shape == (128,) and imp.heads[1].shape == (4,)
        assert imp.hidden.shape == (64,) and imp.block_influence.shape == (2,)
        assert all(torch.isfinite(v).all() and (v >= 0).all() for v in imp.mlp.values())
    # parameters' requires_grad flags are restored
    assert all(p.requires_grad for p in m.parameters())


def _router_stats(moe_layer_name, freq, E=4, top_k=2, with_norm=True):
    tokens = 1000
    counts = torch.tensor(freq) * tokens
    return RouterStats(moe_layer_name, E, top_k, tokens, counts, counts * 0.5, counts / tokens, counts * 1.0 if with_norm else None)  # unit mean expert-output norm


def test_expert_pruning_keeps_salient_and_fixes_router():
    m = make_moe_lm()
    stats = {f"layers.{i}.mlp": _router_stats(f"layers.{i}.mlp", [0.05, 0.9, 0.9, 0.15]) for i in range(2)}
    res = prune_model(m, PruneConfig(expert_keep=0.5, importance="magnitude"), TOK, router_stats=stats)
    assert res.config.moe.num_experts == 2 and res.config.moe.top_k == 2
    assert res.report["kept"]["experts"]["0"] == [1, 2]
    new_moe, old_moe = res.model.layers[0].mlp, m.layers[0].mlp
    assert len(new_moe.experts) == 2 and new_moe.gate.weight.shape == (2, 64)
    assert torch.equal(new_moe.gate.weight, old_moe.gate.weight[[1, 2]])
    assert torch.equal(new_moe.experts[0].down_proj.weight, old_moe.experts[1].down_proj.weight)
    assert torch.isfinite(logits_of(res.model, TOK)).all()
    # per-layer floor: never below top_k
    res2 = prune_model(m, PruneConfig(expert_keep=0.1, importance="magnitude"), TOK, router_stats=stats)
    assert res2.config.moe.num_experts >= 2


def test_expert_merge_is_frequency_weighted_average():
    m = make_moe_lm()
    stats = {f"layers.{i}.mlp": _router_stats(f"layers.{i}.mlp", [0.8, 0.8, 0.1, 0.1]) for i in range(2)}
    res = prune_model(m, PruneConfig(expert_keep=0.5, expert_mode="merge", importance="magnitude"), TOK, router_stats=stats)
    assert res.report["kept"]["experts"]["0"] == [0, 1]
    # layer 1 clustering (deterministic for this seed): expert 3 -> kept 0, expert 2 -> kept 1; weights are frequency-averaged
    old = m.layers[1].mlp
    f = torch.tensor([0.8, 0.8, 0.1, 0.1])
    for kept, member in ((0, 3), (1, 2)):
        want = (f[kept] * old.experts[kept].gate_proj.weight + f[member] * old.experts[member].gate_proj.weight) / (f[kept] + f[member])
        assert torch.allclose(res.model.layers[1].mlp.experts[kept].gate_proj.weight, want, atol=1e-6)
    # router row of the kept expert is the frequency-weighted mean of the merged rows
    want_r = (f[0] * old.gate.weight[0] + f[3] * old.gate.weight[3]) / (f[0] + f[3])
    assert torch.allclose(res.model.layers[1].mlp.gate.weight[0], want_r, atol=1e-6)
    assert torch.isfinite(logits_of(res.model, TOK)).all()


def test_moe_expert_width_pruning_and_calibrated_router_stats():
    m = make_moe_lm()
    cache = collect_stats(m, None, "synthetic", stats=("router",), n_seqs=8, seq_len=16, batch_size=4, track_coactivation=True)
    st = cache.get(cache.model_hash, cache.set_hash, "layers.0.mlp", "router")
    assert isinstance(st, RouterStats) and st.counts.sum() == st.tokens * st.top_k
    assert st.mean_out_norm is not None and st.coact.shape == (4, 4)
    stats = {f"layers.{i}.mlp": cache.get(cache.model_hash, cache.set_hash, f"layers.{i}.mlp", "router") for i in range(2)}
    res = prune_model(m, PruneConfig(mlp_keep=0.5, expert_keep=0.75), TOK, router_stats=stats)
    assert res.config.moe.moe_intermediate_size == 16 and res.config.moe.num_experts == 3
    assert res.model.layers[0].mlp.experts[0].gate_proj.weight.shape == (16, 64)
    assert torch.isfinite(logits_of(res.model, TOK)).all()


def test_sparsity_patterns_and_warning():
    for pattern in ("2:4", "4:8"):
        m = make_lm()
        rep = sparsify_model(m, SparsityConfig(method="magnitude", pattern=pattern))
        n, mm = (int(x) for x in pattern.split(":"))
        w = m.layers[0].mlp.down_proj.weight
        groups = (w.reshape(w.shape[0], -1, mm) != 0).sum(-1)
        assert int(groups.max()) <= n
        assert abs(rep["achieved_sparsity"] - 0.5) < 0.02
        assert "NO GPU speed-up" in rep["warnings"][0]
    m = make_lm()
    rep = sparsify_model(m, SparsityConfig(method="wanda", sparsity=0.6), TOK)
    assert abs(rep["achieved_sparsity"] - 0.6) < 0.02
    assert type(m.lm_head).__name__ == "DenseLinear" and (m.lm_head.weight != 0).all()
    with pytest.raises(ConfigError):
        sparsify_model(make_lm(), SparsityConfig(method="wanda"))
