"""CPU tests of the MoE oracles (Plan K7): routing, permutation, grouped GEMM with M=0 experts, combine, indirection."""
from __future__ import annotations

import pytest
import torch

from forge.kernels.ref.moe_ref import (
    apply_expert_map_ref,
    build_block_schedule_ref,
    dequant_w4_experts,
    max_blocks_bound,
    moe_experts_forward_ref,
    moe_grouped_gemm_ref,
    moe_permute_ref,
    moe_route_ref,
    moe_unpermute_combine_ref,
    pack_w4_experts,
)


def test_route_softmax_topk_ties_and_norm():
    logits = torch.tensor([[1.0, 1.0, 0.0, 1.0], [0.0, 3.0, 2.0, -1.0]])
    w, ids = moe_route_ref(logits, 2)
    assert ids[0].tolist() == [0, 1]  # ties -> lower expert id
    assert ids[1].tolist() == [1, 2]
    torch.testing.assert_close(w.sum(-1), torch.ones(2))
    w2, _ = moe_route_ref(logits, 2, norm_topk_prob=False)
    p = torch.softmax(logits[1], 0)
    torch.testing.assert_close(w2[1], p[[1, 2]])
    ws, ids_s = moe_route_ref(logits, 2, scoring="sigmoid", score_bias=torch.tensor([0.0, 0.0, 5.0, 0.0]), routed_scaling=2.0)
    assert 2 in ids_s[0].tolist() and 2 in ids_s[1].tolist()  # bias steers selection, not weights
    torch.testing.assert_close(ws.sum(-1), torch.full((2,), 2.0))


@pytest.mark.parametrize("t,e,k", [(1, 4, 1), (7, 8, 2), (33, 16, 4)])
def test_permute_is_stable_grouping_with_empty_experts(t, e, k):
    torch.manual_seed(t)
    ids = torch.randint(0, e, (t, k), dtype=torch.int32)
    ids[ids == 3] = 2  # make expert 3 empty (when e > 3)
    pr = moe_permute_ref(ids, e)
    assert pr.expert_offsets[0] == 0 and pr.expert_offsets[-1] == t * k
    flat = ids.reshape(-1).long()
    sorted_ids = flat[pr.slot_idx.long()]
    assert (sorted_ids[1:] >= sorted_ids[:-1]).all()
    # within an expert, flat assignment index ascending (deterministic)
    for ex in range(e):
        lo, hi = int(pr.expert_offsets[ex]), int(pr.expert_offsets[ex + 1])
        seg = pr.slot_idx[lo:hi].tolist()
        assert seg == sorted(seg) and (sorted_ids[lo:hi] == ex).all()
    # inverse permutation
    assert torch.equal(pr.slot_idx.long()[pr.inv_perm.long()], torch.arange(t * k))
    assert torch.equal(pr.token_idx.long(), pr.slot_idx.long() // k)
    if e > 3:
        assert pr.expert_offsets[4] == pr.expert_offsets[3]  # M = 0 expert


def test_permute_with_dropped_assignments_and_expert_map():
    ids = torch.tensor([[0, 3], [2, 1], [3, 3]], dtype=torch.int32)
    w = torch.full((3, 2), 0.5)
    emap = torch.tensor([0, 1, -1, 1])  # expert 2 pruned, expert 3 merged into slot 1
    mids, mw = apply_expert_map_ref(ids, w, emap)
    assert mids.tolist() == [[0, 1], [-1, 1], [1, 1]]
    assert mw[1, 0] == 0 and mw[1, 1] == 0.5
    mids2, mw2 = apply_expert_map_ref(ids, w, emap, renormalize=True)
    assert mw2[1].tolist() == [0.0, 1.0]
    pr = moe_permute_ref(mids, 2)
    assert pr.num_valid == 5 and pr.expert_offsets.tolist() == [0, 1, 5]
    assert pr.inv_perm[2] == -1  # dropped assignment (token 1, slot 0)


def test_block_schedule_covers_every_row_exactly_once_no_host_sync():
    torch.manual_seed(0)
    for bm in (16, 32):
        counts = torch.tensor([0, 5, 0, 40, 17, 0, 1, 64])
        off = torch.zeros(9, dtype=torch.int32)
        off[1:] = torch.cumsum(counts, 0)
        mb = max_blocks_bound(int(off[-1]), 8, bm)
        be, rs, nr = build_block_schedule_ref(off, bm, mb)
        covered = torch.zeros(int(off[-1]), dtype=torch.int32)
        for i in range(mb):
            if be[i] < 0:
                assert nr[i] == 0
                continue
            e = int(be[i])
            assert off[e] <= rs[i] and rs[i] + nr[i] <= off[e + 1] and nr[i] > 0
            covered[rs[i] : rs[i] + nr[i]] += 1
        assert (covered == 1).all()
        # no tile belongs to an empty expert
        assert not any(int(be[i]) in (0, 2, 5) for i in range(mb))


@pytest.mark.parametrize("use_gather", [False, True])
def test_grouped_gemm_ref_matches_per_expert_matmul(use_gather):
    torch.manual_seed(1)
    e, k, n, t = 5, 24, 16, 20
    ids = torch.randint(0, e, (t, 1), dtype=torch.int32)
    ids[ids == 1] = 0  # expert 1 empty
    pr = moe_permute_ref(ids, e)
    x = torch.randn(t, k)
    w = torch.randn(e, n, k)
    if use_gather:
        y = moe_grouped_gemm_ref(x, w, pr.expert_offsets, gather_idx=pr.token_idx)
        xs = x[pr.token_idx.long()]
    else:
        xs = x[pr.token_idx.long()]
        y = moe_grouped_gemm_ref(xs, w, pr.expert_offsets)
    for ex in range(e):
        lo, hi = int(pr.expert_offsets[ex]), int(pr.expert_offsets[ex + 1])
        torch.testing.assert_close(y[lo:hi], xs[lo:hi] @ w[ex].T, atol=1e-5, rtol=1e-5)


def test_grouped_gemm_scatter_accumulate_alpha():
    torch.manual_seed(2)
    x = torch.randn(6, 8)
    w = torch.randn(2, 4, 8)
    off = torch.tensor([0, 2, 6], dtype=torch.int32)
    out_idx = torch.tensor([5, 1, 0, 2, 3, 4], dtype=torch.int32)
    base = torch.randn(6, 4)
    alpha = torch.tensor([2.0, 0.5])
    y = moe_grouped_gemm_ref(x, w, off, out_idx=out_idx, out=base.clone(), accumulate=True, alpha=alpha)
    exp = base.clone()
    for r in range(6):
        ex = 0 if r < 2 else 1
        exp[int(out_idx[r])] += alpha[ex] * (x[r] @ w[ex].T)
    torch.testing.assert_close(y, exp, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("sym", [True, False])
def test_w4_expert_roundtrip_and_gemm(sym):
    torch.manual_seed(3)
    e, n, k, g = 3, 8, 64, 32
    w = torch.randn(e, n, k)
    q, s, z = pack_w4_experts(w, g, sym)
    assert q.dtype == torch.uint8 and q.shape == (e, n, k // 2) and (z is None) == sym
    wd = dequant_w4_experts(q, s, z, g)
    assert (wd - w).abs().max() < (0.35 if sym else 0.3)
    x = torch.randn(6, k)
    off = torch.tensor([0, 2, 2, 6], dtype=torch.int32)
    y = moe_grouped_gemm_ref(x, q, off, w_scales=s, w_zeros=z, group_size=g)
    y_ref = moe_grouped_gemm_ref(x, wd, off)
    torch.testing.assert_close(y, y_ref, atol=1e-4, rtol=1e-4)


def test_full_moe_pipeline_matches_dense_per_token_reference():
    torch.manual_seed(4)
    t, h, i, e, k = 11, 16, 24, 6, 2
    x = torch.randn(t, h)
    wgu = torch.randn(e, 2 * i, h) * 0.2
    wd = torch.randn(e, h, i) * 0.2
    logits = torch.randn(t, e)
    w, ids = moe_route_ref(logits, k)
    pr = moe_permute_ref(ids, e)
    gu = moe_grouped_gemm_ref(x, wgu, pr.expert_offsets, gather_idx=pr.token_idx)
    act = torch.nn.functional.silu(gu[:, :i]) * gu[:, i:]
    y = moe_grouped_gemm_ref(act, wd, pr.expert_offsets)
    out = moe_unpermute_combine_ref(y, pr.inv_perm, w)
    torch.testing.assert_close(out, moe_experts_forward_ref(x, w, ids, wgu, wd), atol=1e-4, rtol=1e-4)
    # pruned expert: pipeline with indirection == dense with the slot dropped
    emap = torch.tensor([0, 1, -1, 2, 3, -1])
    mids, mw = apply_expert_map_ref(ids, w, emap)
    keep = torch.tensor([0, 1, 3, 4])
    pr2 = moe_permute_ref(mids, 4)
    gu2 = moe_grouped_gemm_ref(x, wgu[keep], pr2.expert_offsets, gather_idx=pr2.token_idx)
    act2 = torch.nn.functional.silu(gu2[:, :i]) * gu2[:, i:]
    y2 = moe_grouped_gemm_ref(act2, wd[keep], pr2.expert_offsets)
    out2 = moe_unpermute_combine_ref(y2, pr2.inv_perm, mw)
    torch.testing.assert_close(out2, moe_experts_forward_ref(x, mw, ids.masked_fill(mids < 0, -1), wgu, wd), atol=1e-4, rtol=1e-4)
