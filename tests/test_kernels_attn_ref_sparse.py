"""CPU tests of sparse-attention oracles and the selector-byte accounting (Plan K5, P6 numbers)."""
from __future__ import annotations

import math

import pytest
import torch

from forge.kernels.ref.attn_kv import build_paged_pools
from forge.kernels.ref.attn_decode import paged_decode_attention_ref
from forge.kernels.ref.attn_sparse import (
    SharedSelector,
    dense_attention_bytes,
    expand_block_indices_ref,
    indexer_crossover_tokens,
    indexer_scores_dense_ref,
    indexer_scores_ref,
    index_key_bytes,
    pool_block_scores_ref,
    prefill_indexer_flop_ratio,
    sparse_attention_ref,
    sparse_decode_layer_bytes,
    topk_select_ref,
)

T128K = 131072


def test_p6_byte_accounting_numbers():
    """Reproduces P6 Table 3.2/3.3: 151.0 MB dense, 17.3 MB indexer, 2.36 MB sparse, 88 % index share,
    7.7x -> 22.6x -> 57x reduction for 1 / 4 / 61 layers sharing one indexer pass."""
    assert index_key_bytes(128, 1, 4) == 132
    dense = dense_attention_bytes(T128K)
    assert dense / 1e6 == pytest.approx(151.0, abs=0.1)
    r1 = sparse_decode_layer_bytes(T128K, top_k=2048, index_share_layers=1)
    assert r1.index_bytes / 1e6 == pytest.approx(17.3, abs=0.05)
    assert r1.sparse_bytes / 1e6 == pytest.approx(2.36, abs=0.01)
    assert r1.total_bytes / 1e6 == pytest.approx(19.7, abs=0.05)
    assert r1.index_share == pytest.approx(0.88, abs=0.005)
    assert round(r1.reduction, 1) == 7.7
    r4 = sparse_decode_layer_bytes(T128K, top_k=2048, index_share_layers=4)
    assert round(r4.reduction, 1) == 22.6
    r61 = sparse_decode_layer_bytes(T128K, top_k=2048, index_share_layers=61)
    assert round(r61.reduction) == 57
    assert r4.reduction > r1.reduction and r61.reduction > r4.reduction
    # other rows of Table 3.2
    for t, dense_mb, tot_mb, idx_mb in [(8192, 9.4, 3.4, 1.1), (32768, 37.7, 6.7, 4.3), (524288, 604.0, 71.6, 69.2)]:
        rep = sparse_decode_layer_bytes(t)
        assert rep.dense_bytes / 1e6 == pytest.approx(dense_mb, abs=0.06)
        assert rep.total_bytes / 1e6 == pytest.approx(tot_mb, abs=0.06)
        assert rep.index_bytes / 1e6 == pytest.approx(idx_mb, abs=0.06)


def test_crossover_and_prefill_ratio():
    assert indexer_crossover_tokens() == pytest.approx(17873, abs=1)
    assert prefill_indexer_flop_ratio(64, 128) == pytest.approx(0.0588, abs=1e-3)  # 8192/139264 = 5.9 %
    # short contexts: sparse bytes are min(k, T)
    rep = sparse_decode_layer_bytes(1000)
    assert rep.sparse_bytes == 576 * 2 * 1000
    with pytest.raises(ValueError):
        sparse_decode_layer_bytes(1000, index_share_layers=0)


def test_topk_select_ref_rules():
    s = torch.tensor([[1.0, 5.0, 5.0, 2.0, 5.0, 0.0], [float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0]])
    idx = topk_select_ref(s, 2, torch.tensor([6, 4]))
    # row 0: three 5.0 ties -> lowest positions first (1, 2); output ascending
    assert idx[0].tolist() == [1, 2]
    assert idx[1].tolist() == [2, 3]  # only first 4 valid; NaN counts as -inf
    short = topk_select_ref(s, 8, torch.tensor([3, 0]))
    assert short[0].tolist() == [0, 1, 2, -1, -1, -1, -1, -1]
    assert (short[1] == -1).all()
    # matches torch.topk set on random data
    x = torch.randn(4, 200)
    sel = topk_select_ref(x, 17)
    for r in range(4):
        assert set(sel[r].tolist()) == set(torch.topk(x[r], 17).indices.tolist())
        assert sel[r].tolist() == sorted(sel[r].tolist())


def test_indexer_scores_paged_matches_dense_and_fp8():
    g = torch.Generator().manual_seed(1)
    b, hi, di, ps = 2, 4, 32, 16
    lens = [37, 16]
    ik = torch.randn(b, 1, 48, di, generator=g)
    pools = build_paged_pools(ik, ik, lens, ps, "fp32")
    ik_pool = pools["k_pool"][:, 0]
    iq = torch.randn(b, hi, di, generator=g)
    iw = torch.rand(b, hi, generator=g)
    sc = indexer_scores_ref(iq, iw, ik_pool, None, pools["block_tables"], pools["seq_lens"])
    for bi, n in enumerate(lens):
        exp = (iw[bi, :, None] * torch.relu(iq[bi] @ ik[bi, 0, :n].T)).sum(0)
        torch.testing.assert_close(sc[bi, :n], exp, atol=1e-5, rtol=1e-5)
        assert torch.isinf(sc[bi, n:]).all()
    # fp8 index keys with per-token scale
    scale = ik_pool.abs().amax(-1) / 448.0
    q8 = (ik_pool / scale.unsqueeze(-1).clamp_min(1e-12)).to(torch.float8_e4m3fn)
    sc8 = indexer_scores_ref(iq, iw, q8, scale, pools["block_tables"], pools["seq_lens"])
    ok = torch.isfinite(sc)
    assert (sc8[ok] - sc[ok]).abs().max() / sc[ok].abs().max() < 0.1
    # dense (prefill) form: causal
    d = indexer_scores_dense_ref(iq[0:1].expand(5, -1, -1), iw[0:1].expand(5, -1), ik[0, 0, :5], causal=True)
    assert torch.isinf(d[0, 1:]).all() and torch.isfinite(d[4]).all()


def test_block_scores_and_expansion():
    s = torch.arange(10, dtype=torch.float32).view(1, 10)
    mx = pool_block_scores_ref(s, 4, torch.tensor([9]), "max")
    assert mx.tolist() == [[3.0, 7.0, 8.0]]
    mean = pool_block_scores_ref(s, 4, torch.tensor([9]), "mean")
    assert mean[0].tolist() == pytest.approx([1.5, 5.5, 8.0])
    tok = expand_block_indices_ref(torch.tensor([[[2, 0, -1]]], dtype=torch.int32), 4, torch.tensor([9]))
    assert tok[0, 0].tolist() == [8, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1]


def test_block_selection_cannot_beat_token_selection_in_mass():
    """P6: block selection captures no more attention mass than token selection at equal token budget."""
    torch.manual_seed(5)
    n, bs, k = 256, 16, 64
    logits = torch.randn(n) * 3
    p = torch.softmax(logits, 0)
    tok_mass = p[torch.topk(logits, k).indices].sum()
    blk_scores = pool_block_scores_ref(logits.view(1, n), bs, mode="max")[0]
    blocks = torch.topk(blk_scores, k // bs).indices
    blk_mass = sum(p[b * bs:(b + 1) * bs].sum() for b in blocks.tolist())
    assert float(blk_mass) <= float(tok_mass) + 1e-6


def test_sparse_attention_ref_end_to_end_selects_top_tokens():
    g = torch.Generator().manual_seed(2)
    lens = [64]
    q = torch.randn(1, 4, 16, generator=g)
    k = torch.randn(1, 2, 64, 16, generator=g)
    v = torch.randn(1, 2, 64, 16, generator=g)
    p = build_paged_pools(k, v, lens, 16, "fp32")
    scores = torch.randn(1, 64, generator=g)
    idx = topk_select_ref(scores, 64, torch.tensor([64])).unsqueeze(1)  # all tokens
    out = sparse_attention_ref(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], idx, kv_dtype="fp32")
    dense = paged_decode_attention_ref(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], kv_dtype="fp32")
    torch.testing.assert_close(out, dense, atol=1e-5, rtol=1e-5)


def test_shared_selector_amortises_passes():
    calls = []
    sel = SharedSelector(share_every=4)
    res = []
    for layer in range(12):
        res.append(sel.get(layer, lambda layer=layer: (calls.append(layer), torch.tensor([layer]))[1]))
    assert calls == [0, 4, 8]
    assert [int(r) for r in res] == [0] * 4 + [4] * 4 + [8] * 4
    assert sel.indexer_passes == 3 and len(sel.reused_layers) == 9
    sel.reset()
    assert sel.indexer_passes == 0
    custom = SharedSelector(anchor_layers=[0, 3])
    got = [int(custom.get(i, lambda i=i: torch.tensor([i]))) for i in range(6)]
    assert got == [0, 0, 0, 3, 3, 3]
