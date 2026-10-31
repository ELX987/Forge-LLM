"""ForgeLM core: config accounting, rope, attention backends, KV-cache decode equivalence, MoE, checkpointing."""
from __future__ import annotations

import math

import pytest
import torch

from forge.core.errors import ConfigError
from forge.models import ForgeLM, ModelConfig, MoEConfig, build_tiny_model, make_tiny_config
from forge.models.attention import AttnMeta
from forge.models.layers import apply_rotary, compute_inv_freq, rotate_half
from forge.models.moe import load_balancing_loss, route_topk

torch.manual_seed(0)


# ------------------------------------------------------------------ config / accounting
def test_kv_bytes_per_token_matches_plan():
    llama8 = ModelConfig(arch="llama", num_layers=32, num_heads=32, num_kv_heads=8, hidden_size=4096, head_dim=128)
    llama70 = ModelConfig(arch="llama", num_layers=80, num_heads=64, num_kv_heads=8, hidden_size=8192, head_dim=128)
    qwen8 = ModelConfig(arch="qwen3", num_layers=36, num_heads=32, num_kv_heads=8, hidden_size=4096, head_dim=128, qk_norm=True)
    assert llama8.kv_bytes_per_token() == 128 * 1024
    assert llama70.kv_bytes_per_token() == 320 * 1024
    assert qwen8.kv_bytes_per_token() == 144 * 1024
    assert llama8.kv_bytes_per_token(8) == 64 * 1024
    assert llama8.kv_bytes_per_token(4) == 32 * 1024
    # group scales add bytes
    assert llama8.kv_bytes_per_token(8, group_size=128) == 64 * 1024 + 2 * 32 * 8 * 2


def test_llama3_8b_param_count():
    c = ModelConfig(
        arch="llama", vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_layers=32, num_heads=32, num_kv_heads=8,
        rope_theta=500000.0, rms_norm_eps=1e-5, max_position_embeddings=8192,
    )
    assert c.param_count() == 8_030_261_248


@pytest.mark.parametrize(
    "ov",
    [{}, {"moe": True}, {"tie_word_embeddings": True, "qk_norm": True}, {"attention_bias": True}, {"arch": "qwen2", "attention_bias": True}],
)
def test_param_count_matches_instance(ov):
    m = build_tiny_model(**ov)
    assert m.num_parameters() == m.config.param_count()


def test_active_params_and_flops():
    c = make_tiny_config(moe=True)
    assert c.active_param_count() < c.param_count()
    assert c.flops_per_token(128, "train") == pytest.approx(3 * c.flops_per_token(128, "prefill"))
    assert c.flops_per_token(1024, "decode") > c.flops_per_token(0, "decode")


def test_config_validation():
    with pytest.raises(Exception):
        ModelConfig(num_heads=6, num_kv_heads=4)
    with pytest.raises(Exception):
        ModelConfig(arch="mixtral")  # needs moe
    with pytest.raises(Exception):
        ModelConfig(hidden_size=64, num_heads=4, head_dim=15)


def test_hf_config_roundtrip():
    cases = [
        {},
        {"qk_norm": True, "arch": "qwen3"},
        {"arch": "qwen2", "attention_bias": True},
        {"arch": "mistral", "sliding_window": 16},
        {"moe": True},
        {"arch": "mixtral", "moe": MoEConfig(num_experts=4, top_k=2, moe_intermediate_size=128)},
        {"rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 64}},
    ]
    for ov in cases:
        c = make_tiny_config(**ov)
        c2 = ModelConfig.from_hf_config(c.to_hf_config())
        assert c2.param_count() == c.param_count()
        assert c2.kv_bytes_per_token() == c.kv_bytes_per_token()
        assert c2.rope_scaling == c.rope_scaling and c2.sliding_window == c.sliding_window and c2.arch == c.arch
    with pytest.raises(ConfigError):
        ModelConfig.from_hf_config({"model_type": "bert"})


def test_json_schema_has_descriptions():
    from forge.core.jsonschema import schema_for

    props = schema_for(ModelConfig)["properties"]
    missing = [k for k, v in props.items() if not v.get("description")]
    assert not missing, missing


# ------------------------------------------------------------------ rope
def test_rope_plain_and_rotation_property():
    inv, att = compute_inv_freq(16, 10000.0, None, 128)
    assert att == 1.0 and inv.shape == (8,)
    assert inv[0] == 1.0 and inv[-1] == pytest.approx(10000.0 ** (-14 / 16))
    q = torch.randn(1, 5, 2, 16)
    ang = torch.arange(5.0)[:, None] * torch.cat([inv, inv])[None]
    qr, _ = apply_rotary(q, q, torch.cos(ang).unsqueeze(0), torch.sin(ang).unsqueeze(0))
    assert torch.allclose(qr.norm(dim=-1), q.norm(dim=-1), atol=1e-5)  # rotations preserve norm
    assert torch.allclose(rotate_half(rotate_half(q)), -q)


def test_rope_scaling_variants():
    base, _ = compute_inv_freq(64, 10000.0, None, 4096)
    lin, _ = compute_inv_freq(64, 10000.0, {"rope_type": "linear", "factor": 4.0}, 4096)
    assert torch.allclose(lin, base / 4)
    cfg3 = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
    l3, _ = compute_inv_freq(64, 500000.0, cfg3, 131072)
    b3, _ = compute_inv_freq(64, 500000.0, None, 131072)
    assert torch.allclose(l3[:2], b3[:2])  # highest frequencies untouched
    assert torch.allclose(l3[-2:], b3[-2:] / 8)  # lowest frequencies scaled by factor
    y, att = compute_inv_freq(64, 10000.0, {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}, 32768)
    assert att == pytest.approx(0.1 * math.log(8) + 1)
    assert y[0] == pytest.approx(base[0].item()) and y[-1] == pytest.approx(base[-1].item() / 8)
    with pytest.raises(ConfigError):
        compute_inv_freq(64, 10000.0, {"rope_type": "dynamic", "factor": 2.0}, 4096)


# ------------------------------------------------------------------ forward / loss
def test_forward_shapes_and_loss():
    m = build_tiny_model()
    ids = torch.randint(0, 256, (3, 12))
    out = m(ids, labels=ids, return_hidden=True)
    assert out.logits.shape == (3, 12, 256) and out.hidden_states.shape == (3, 12, 64)
    assert out.aux_loss is None
    ref = torch.nn.functional.cross_entropy(out.logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1))
    assert torch.allclose(out.loss, ref, atol=1e-5)
    assert out.loss.item() == pytest.approx(math.log(256), abs=0.5)
    lab = ids.clone()
    lab[:, :6] = -100
    l2 = m(ids, labels=lab).loss
    ref2 = torch.nn.functional.cross_entropy(out.logits[:, :-1].reshape(-1, 256), lab[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(l2, ref2, atol=1e-5)
    k = m(ids, logits_to_keep=2).logits
    assert k.shape == (3, 2, 256) and torch.allclose(k, out.logits[:, -2:], atol=1e-5)


def test_chunked_loss_matches_and_grads():
    m = build_tiny_model()
    ids = torch.randint(0, 256, (2, 17))
    full = m(ids, labels=ids).loss
    full.backward()
    g1 = m.lm_head.weight.grad.clone()
    m.zero_grad()
    m.loss_chunk_size = 5
    out = m(ids, labels=ids, return_logits=False)
    assert out.logits is None and torch.allclose(out.loss, full, atol=1e-5)
    out.loss.backward()
    assert torch.allclose(m.lm_head.weight.grad, g1, atol=1e-5)


def test_tied_embeddings():
    m = build_tiny_model(tie_word_embeddings=True)
    assert m.lm_head.weight is m.embed_tokens.weight


def test_gradient_checkpointing_same_grads():
    m = build_tiny_model(moe=True)
    m.train()
    ids = torch.randint(0, 256, (2, 9))
    m(ids, labels=ids).loss.backward()
    g = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad()
    m.gradient_checkpointing_enable()
    m(ids, labels=ids).loss.backward()
    for n, p in m.named_parameters():
        if n in g:
            assert torch.allclose(p.grad, g[n], atol=1e-5, rtol=1e-4), n


def test_sdpa_matches_eager():
    for ov in [{}, {"sliding_window": 4}]:
        m = build_tiny_model(**ov).eval()
        ids = torch.randint(0, 256, (2, 14))
        a = m(ids).logits
        m.set_attn_impl("eager")
        b = m(ids).logits
        assert torch.allclose(a, b, atol=1e-4), ov
    with pytest.raises(ConfigError):
        m.set_attn_impl("nope")


def test_causality():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (1, 10))
    a = m(ids).logits
    ids2 = ids.clone()
    ids2[0, 7:] = (ids2[0, 7:] + 1) % 256
    b = m(ids2).logits
    assert torch.allclose(a[:, :7], b[:, :7], atol=1e-5) and not torch.allclose(a[:, 8:], b[:, 8:], atol=1e-3)


def test_padding_mask_without_cache():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (1, 8))
    pad = torch.cat([ids, torch.zeros(1, 4, dtype=torch.long)], 1)
    mask = torch.cat([torch.ones(1, 8), torch.zeros(1, 4)], 1)
    a = m(ids).logits
    b = m(pad, attn_meta=AttnMeta(attention_mask=mask)).logits[:, :8]
    assert torch.allclose(a, b, atol=1e-4)


# ------------------------------------------------------------------ kv-cache decode == full forward
@pytest.mark.parametrize(
    "ov",
    [{}, {"sliding_window": 5}, {"moe": True}, {"qk_norm": True, "attention_bias": True}, {"rope_scaling": {"rope_type": "linear", "factor": 2.0}}],
)
def test_cache_decode_equals_full(ov):
    m = build_tiny_model(**ov).eval()
    ids = torch.randint(0, 256, (2, 20))
    full = m(ids).logits
    cache = m.new_kv_cache(2, 32)
    pre = m(ids[:, :12], kv_cache=cache).logits
    assert torch.allclose(pre, full[:, :12], atol=1e-4)
    assert cache.seq_lens.tolist() == [12, 12]
    steps = [m(ids[:, i : i + 1], kv_cache=cache).logits for i in range(12, 20)]
    dec = torch.cat(steps, 1)
    assert torch.allclose(dec, full[:, 12:], atol=1e-4), (dec - full[:, 12:]).abs().max()
    cache2 = m.new_kv_cache(2, 32)  # chunked prefill
    outs = [m(ids[:, i : i + 5], kv_cache=cache2).logits for i in range(0, 20, 5)]
    assert torch.allclose(torch.cat(outs, 1), full, atol=1e-4)


def test_cache_ragged_batch_right_padded():
    m = build_tiny_model().eval()
    a = torch.randint(0, 256, (1, 9))
    b = torch.randint(0, 256, (1, 5))
    ref_a, ref_b = m(a).logits, m(b).logits
    batch = torch.cat([a, torch.cat([b, torch.zeros(1, 4, dtype=torch.long)], 1)], 0)
    mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4])
    cache = m.new_kv_cache(2, 24)
    out = m(batch, kv_cache=cache, attn_meta=AttnMeta(attention_mask=mask)).logits
    assert torch.allclose(out[0], ref_a[0], atol=1e-4) and torch.allclose(out[1, :5], ref_b[0], atol=1e-4)
    assert cache.seq_lens.tolist() == [9, 5]
    nxt = torch.randint(0, 256, (2, 1))  # decode one token each; seq 2 overwrites its padding slots
    d = m(nxt, kv_cache=cache).logits
    ra = m(torch.cat([a, nxt[:1]], 1)).logits[:, -1]
    rb = m(torch.cat([b, nxt[1:]], 1)).logits[:, -1]
    assert torch.allclose(d[0, 0], ra[0], atol=1e-4) and torch.allclose(d[1, 0], rb[0], atol=1e-4)


def test_generate_greedy_matches_manual():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (2, 6))
    out = m.generate(ids, max_new_tokens=5)
    cur = ids
    for _ in range(5):
        nxt = m(cur).logits[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], 1)
    assert torch.equal(out, cur)
    s1 = m.generate(ids, max_new_tokens=4, temperature=0.8, top_k=20, top_p=0.9, seed=3)
    s2 = m.generate(ids, max_new_tokens=4, temperature=0.8, top_k=20, top_p=0.9, seed=3)
    assert torch.equal(s1, s2)


# ------------------------------------------------------------------ MoE
def test_route_topk_and_aux():
    logits = torch.randn(50, 8)
    w, ids, _ = route_topk(logits, 2, True)
    assert w.shape == (50, 2) and torch.allclose(w.sum(-1), torch.ones(50))
    assert torch.equal(ids, torch.topk(logits, 2, -1).indices)
    uni = torch.full((16, 4), 0.25)
    uid = torch.arange(16).unsqueeze(1) % 4
    assert load_balancing_loss(uni, uid, 4).item() == pytest.approx(1.0)
    col_p = torch.zeros(16, 4)
    col_p[:, 0] = 1
    assert load_balancing_loss(col_p, torch.zeros(16, 1, dtype=torch.long), 4).item() == pytest.approx(4.0)
    w2, _, _ = route_topk(logits, 2, False)
    assert (w2.sum(-1) < 1).all()


def test_moe_block_matches_dense_reference():
    m = build_tiny_model(moe=True).eval()
    blk = m.layers[0].mlp
    x = torch.randn(2, 7, 64)
    y, aux = blk(x, want_aux=True)
    x2 = x.reshape(-1, 64)
    w, ids, _ = route_topk(blk.gate(x2), blk.top_k, blk.norm_topk)
    ref = torch.zeros_like(x2)
    for t in range(x2.shape[0]):
        for j in range(blk.top_k):
            ref[t] += w[t, j] * blk.experts[ids[t, j]](x2[t : t + 1])[0]
    assert torch.allclose(y.reshape(-1, 64), ref, atol=1e-5)
    assert aux.item() >= 1.0 - 1e-6
    assert blk.last_expert_counts.sum().item() == 14 * blk.top_k


def test_moe_shared_expert_and_sparse_step():
    cfg = make_tiny_config(
        arch="qwen2_moe",
        num_layers=4,
        moe=MoEConfig(num_experts=4, top_k=2, moe_intermediate_size=32, shared_expert_intermediate_size=48, shared_expert_gate=True, norm_topk_prob=False, decoder_sparse_step=2),
    )
    assert [cfg.is_moe_layer(i) for i in range(4)] == [False, True, False, True]
    m = ForgeLM(cfg, dtype=torch.float32)
    assert m.num_parameters() == cfg.param_count()
    assert m.layers[1].mlp.shared_expert is not None and m.layers[0].is_moe is False
    ids = torch.randint(0, 256, (2, 6))
    out = m(ids, labels=ids)
    assert torch.isfinite(out.loss) and out.aux_loss is not None
    out.loss.backward()
    assert m.layers[1].mlp.gate.weight.grad is not None


def test_moe_aux_loss_added_in_training_only():
    m = build_tiny_model(moe=True)
    ids = torch.randint(0, 256, (2, 8))
    m.train()
    a = m(ids, labels=ids)
    coef = m.config.moe.router_aux_loss_coef
    ce = torch.nn.functional.cross_entropy(a.logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1))
    assert torch.allclose(a.loss, ce + coef * a.aux_loss, atol=1e-5)


def test_grouped_gemm_path_with_fake_ops(monkeypatch):
    """The ops_moe wiring is exercised with an injected torch implementation (CPU) to verify equivalence with the loop."""
    import sys
    import types

    calls = []

    def grouped_gemm(x_sorted, w, offsets):
        calls.append(1)
        out = x_sorted.new_zeros(x_sorted.shape[0], w.shape[1])
        for e in range(w.shape[0]):
            s, t = int(offsets[e]), int(offsets[e + 1])
            out[s:t] = x_sorted[s:t] @ w[e].T
        return out

    fake = types.ModuleType("forge.kernels.ops_moe")
    fake.grouped_gemm = grouped_gemm
    monkeypatch.setitem(sys.modules, "forge.kernels.ops_moe", fake)
    import forge.kernels as fk

    monkeypatch.setattr(fk, "ops_moe", fake, raising=False)
    m = build_tiny_model(moe=True).eval()
    blk = m.layers[0].mlp
    x = torch.randn(2, 5, 64)
    ref, _ = blk(x)
    xs = x.reshape(-1, 64)
    w, topi, _ = route_topk(blk.gate(xs), blk.top_k, blk.norm_topk)
    out = blk._grouped(xs, topi, w.to(xs.dtype))
    assert out is not None and calls
    assert torch.allclose(out.reshape(2, 5, 64), ref, atol=1e-5)


def test_meta_init_has_no_storage_and_materialise():
    from forge.models.loader import materialize

    m = ForgeLM(make_tiny_config(), dtype=torch.bfloat16, device="meta", init_weights=False)
    assert all(p.is_meta for p in m.parameters())
    materialize(m, "cpu")
    assert all(not p.is_meta for p in m.parameters()) and m.dtype == torch.bfloat16
    out = m(torch.randint(0, 256, (1, 4)))
    assert torch.isfinite(out.logits.float()).all()
