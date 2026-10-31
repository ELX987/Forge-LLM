"""KV caches: quantised storage, residual window, rollback, paged view vs dense, paged decode kernel wiring."""
from __future__ import annotations

import sys
import types

import pytest
import torch

from forge.core.errors import ConfigError
from forge.models import DenseKVCache, PagedKVView, SimplePagedPool, build_tiny_model, make_tiny_config
from forge.models.attention import AttnMeta
from forge.models.kvcache import dequantize_kv, kv_dtype_name, quantize_kv


def test_kv_dtype_names():
    assert kv_dtype_name(None) == "bf16" and kv_dtype_name(torch.float16) == "fp16" and kv_dtype_name("fp8") == "fp8_e4m3"
    assert kv_dtype_name(torch.float32) == "fp32" and kv_dtype_name("INT8") == "int8"
    with pytest.raises(ConfigError):
        kv_dtype_name("int3")


@pytest.mark.parametrize("dt,tol", [("fp8_e4m3", 0.07), ("int8", 0.01), ("int4", 0.16)])
def test_quant_dequant_error(dt, tol):
    x = torch.randn(3, 5, 2, 32)
    q, s = quantize_kv(x, dt)
    y = dequantize_kv(q, s, dt, torch.float32)
    assert y.shape == x.shape
    rel = ((y - x).abs().amax(-1) / x.abs().amax(-1)).max()
    assert rel < tol, rel
    q2, s2 = quantize_kv(x, dt, group_size=16)
    assert s2.shape[-1] == 2
    y2 = dequantize_kv(q2, s2, dt, torch.float32)
    assert (y2 - x).abs().mean() <= (y - x).abs().mean() * 1.01
    if dt == "int4":
        assert q.dtype == torch.uint8 and q.shape[-1] == 16


@pytest.mark.parametrize("dt,tol", [("fp16", 5e-3), ("fp8_e4m3", 0.25), ("int8", 0.03), ("int4", 0.6)])
def test_quantised_cache_decode_close_to_full(dt, tol):
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (2, 16))
    full = m(ids).logits
    cache = m.new_kv_cache(2, 24, kv_dtype=dt)
    m(ids[:, :10], kv_cache=cache)
    dec = torch.cat([m(ids[:, i : i + 1], kv_cache=cache).logits for i in range(10, 16)], 1)
    err = (dec - full[:, 10:]).abs().max() / full.abs().max()
    assert err < tol, err
    # memory shrinks with bits
    bf = m.new_kv_cache(2, 24, kv_dtype="bf16").nbytes()
    assert cache.nbytes() <= bf * (1.0 if dt == "fp16" else 0.7)


def test_residual_window_keeps_recent_tokens_exact():
    cfg = make_tiny_config()
    b, h, d = 2, cfg.kv_heads, cfg.head_dim
    k = torch.randn(b, 12, h, d)
    v = torch.randn(b, 12, h, d)
    c = DenseKVCache(b, 16, cfg, dtype=torch.float32, kv_dtype="int4", residual_window=4)
    kf, vf = c.update(0, k, v)
    c.commit(12)
    # last 4 tokens exact, older ones quantised (nonzero error)
    assert torch.equal(kf[:, 8:12], k[:, 8:12]) and torch.equal(vf[:, 8:12], v[:, 8:12])
    assert (kf[:, :8] - k[:, :8]).abs().max() > 1e-3
    # decode a token: window slides, token 8 becomes quantised
    k1, v1 = torch.randn(b, 1, h, d), torch.randn(b, 1, h, d)
    kf2, _ = c.update(0, k1, v1)
    c.commit(1)
    assert torch.equal(kf2[:, 12:13], k1) and torch.equal(kf2[:, 9:12], k[:, 9:12])
    assert (kf2[:, 8] - k[:, 8]).abs().max() > 1e-3
    # without the window, nothing is exact
    c0 = DenseKVCache(b, 16, cfg, dtype=torch.float32, kv_dtype="int4")
    k0, _ = c0.update(0, k, v)
    assert (k0[:, 8:12] - k[:, 8:12]).abs().max() > 1e-3


def test_residual_window_with_ragged_lengths():
    cfg = make_tiny_config()
    b, h, d = 2, cfg.kv_heads, cfg.head_dim
    k = torch.randn(b, 10, h, d)
    v = torch.randn(b, 10, h, d)
    c = DenseKVCache(b, 16, cfg, dtype=torch.float32, kv_dtype="int8", residual_window=3)
    n_new = torch.tensor([10, 6])
    kf, _ = c.update(0, k, v, n_new=n_new)
    assert torch.equal(kf[0, 7:10], k[0, 7:10]) and torch.equal(kf[1, 3:6], k[1, 3:6])


def test_truncate_rollback_and_reset():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (1, 14))
    cache = m.new_kv_cache(1, 20)
    m(ids[:, :10], kv_cache=cache)
    m(ids[:, 10:14], kv_cache=cache)  # speculative tokens
    assert cache.seq_lens.tolist() == [14]
    cache.truncate([10])
    redo = m(ids[:, 10:14], kv_cache=cache).logits
    assert torch.allclose(redo, m(ids).logits[:, 10:], atol=1e-4)
    cache.reset()
    assert cache.seq_lens.tolist() == [0]


def test_bytes_per_token_matches_formula():
    m = build_tiny_model()
    c = m.new_kv_cache(2, 8, kv_dtype="fp16")
    assert c.bytes_per_token() == m.config.kv_bytes_per_token(16)
    c8 = m.new_kv_cache(2, 8, kv_dtype="int8")
    assert c8.bytes_per_token() == m.config.kv_bytes_per_token(8, group_size=m.config.head_dim, scale_bits=16)


# ------------------------------------------------------------------ paged
def _paged_setup(model, num_seqs, lens, block_size=4, kv_dtype=None):
    cfg = model.config
    nb_per = max((l + 8) // block_size + 1 for l in lens)
    pool = SimplePagedPool(cfg, num_blocks=num_seqs * nb_per + 3, block_size=block_size, dtype=torch.float32, kv_dtype=kv_dtype)
    # non-contiguous, shuffled block assignment to exercise the gather
    perm = torch.randperm(pool.num_blocks - 1)[: num_seqs * nb_per] + 1
    tables = perm.reshape(num_seqs, nb_per).to(torch.int32)
    return pool, tables


def test_paged_matches_dense_prefill_and_decode():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (2, 13))
    full = m(ids).logits
    pool, tables = _paged_setup(m, 2, [13])
    view = PagedKVView(tables, torch.zeros(2, dtype=torch.long), pool)
    pre = m(ids[:, :9], kv_cache=view).logits
    assert torch.allclose(pre, full[:, :9], atol=1e-4)
    assert view.seq_lens.tolist() == [9, 9]
    for i in range(9, 13):
        out = m(ids[:, i : i + 1], kv_cache=view).logits
        assert torch.allclose(out[:, 0], full[:, i], atol=1e-4), i


def test_paged_ragged_decode_two_sequences():
    m = build_tiny_model().eval()
    a, b = torch.randint(0, 256, (1, 11)), torch.randint(0, 256, (1, 6))
    pool, tables = _paged_setup(m, 2, [11, 6])
    for i, (x, n) in enumerate([(a, 11), (b, 6)]):  # prefill each sequence separately
        v = PagedKVView(tables[i : i + 1], torch.zeros(1, dtype=torch.long), pool)
        m(x, kv_cache=v)
    nxt = torch.randint(0, 256, (2, 1))
    view = PagedKVView(tables, torch.tensor([11, 6]), pool)
    out = m(nxt, kv_cache=view).logits
    ra = m(torch.cat([a, nxt[:1]], 1)).logits[:, -1]
    rb = m(torch.cat([b, nxt[1:]], 1)).logits[:, -1]
    assert torch.allclose(out[0, 0], ra[0], atol=1e-4) and torch.allclose(out[1, 0], rb[0], atol=1e-4)


def test_paged_padding_does_not_write_unallocated_blocks():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (2, 8))
    mask = torch.tensor([[1] * 8, [1] * 3 + [0] * 5])
    pool, tables = _paged_setup(m, 2, [8])
    tables[1, 1:] = 0  # sequence 1 only owns its first block; others point at block 0 (shared garbage block)
    tables[1, 0] = pool.num_blocks - 1
    before = pool.k(0)[0].clone()
    view = PagedKVView(tables, torch.zeros(2, dtype=torch.long), pool)
    m(ids, kv_cache=view, attn_meta=AttnMeta(attention_mask=mask))
    assert torch.equal(pool.k(0)[0], before)  # block 0 untouched by padded tokens
    assert view.seq_lens.tolist() == [8, 3]


def test_paged_quantised_pool_close():
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (1, 12))
    full = m(ids).logits
    pool, tables = _paged_setup(m, 1, [12], kv_dtype="int8")
    view = PagedKVView(tables, torch.zeros(1, dtype=torch.long), pool)
    m(ids[:, :8], kv_cache=view)
    out = torch.cat([m(ids[:, i : i + 1], kv_cache=view).logits for i in range(8, 12)], 1)
    assert (out - full[:, 8:]).abs().max() / full.abs().max() < 0.05
    with pytest.raises(ConfigError):
        SimplePagedPool(m.config, 4, kv_dtype="int4")


def test_paged_decode_kernel_wiring_with_fake_ops(monkeypatch):
    """Inject a torch implementation of ops_attn.paged_decode_attention (real contract: head-major pools) and check it is used + correct."""
    calls = {}

    def paged_decode_attention(q, k_pool, v_pool, block_tables, seq_lens, *, softmax_scale, kv_dtype, k_scale=None, v_scale=None, window=None):
        calls["n"] = calls.get("n", 0) + 1
        calls["seq_lens"] = seq_lens.clone()
        b, h, d = q.shape
        assert k_pool.shape[1] == h // 2 and block_tables.dtype == torch.int32 and seq_lens.dtype == torch.int32  # [P, Hkv, page, D]
        bs = k_pool.shape[2]
        out = torch.zeros_like(q)
        for i in range(b):
            n = int(seq_lens[i])
            nb = (n + bs - 1) // bs
            ks = k_pool[block_tables[i, :nb].long()].permute(0, 2, 1, 3).reshape(-1, k_pool.shape[1], d)[:n]
            vs = v_pool[block_tables[i, :nb].long()].permute(0, 2, 1, 3).reshape(-1, v_pool.shape[1], d)[:n]
            rep = h // ks.shape[1]
            ks, vs = ks.repeat_interleave(rep, 1), vs.repeat_interleave(rep, 1)
            sc = torch.einsum("hd,nhd->hn", q[i], ks) * softmax_scale
            out[i] = torch.einsum("hn,nhd->hd", torch.softmax(sc, -1), vs)
        return out

    fake = types.ModuleType("forge.kernels.ops_attn")
    fake.paged_decode_attention = paged_decode_attention
    monkeypatch.setitem(sys.modules, "forge.kernels.ops_attn", fake)
    import forge.kernels as fk

    monkeypatch.setattr(fk, "ops_attn", fake, raising=False)
    m = build_tiny_model().eval()
    ids = torch.randint(0, 256, (2, 10))
    full = m(ids).logits
    pool, tables = _paged_setup(m, 2, [10])
    view = PagedKVView(tables, torch.zeros(2, dtype=torch.long), pool)
    m(ids[:, :8], kv_cache=view)
    out = m(ids[:, 8:9], kv_cache=view, attn_meta=AttnMeta(use_paged_kernel=True)).logits
    assert calls["n"] == m.config.num_layers and calls["seq_lens"].tolist() == [9, 9]
    assert torch.allclose(out[:, 0], full[:, 8], atol=1e-4)
    # a failing kernel falls back to the reference path (and stays on it)
    def boom(*a, **k):
        raise RuntimeError("no kernel")

    fake.paged_decode_attention = boom
    for layer in m.layers:
        layer.self_attn._paged_ok = True
    out2 = m(ids[:, 9:10], kv_cache=view, attn_meta=AttnMeta(use_paged_kernel=True)).logits
    assert torch.allclose(out2[:, 0], full[:, 9], atol=1e-4)
    assert all(not layer.self_attn._paged_ok for layer in m.layers)
