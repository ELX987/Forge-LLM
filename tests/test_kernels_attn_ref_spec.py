"""CPU tests of speculative-decoding oracles: tree attention, exact rejection sampling, KV compaction, Philox RNG."""
from __future__ import annotations

import math

import pytest
import torch

from forge.kernels.ref.attn_kv import build_paged_pools
from forge.kernels.ref.attn_prefill import prefill_attention_ref
from forge.kernels.ref.attn_spec import (
    build_tree_mask,
    chain_parents,
    greedy_verify_ref,
    kv_compact_ref,
    rejection_verify_ref,
    tree_attention_ref,
    tree_depths,
    tree_greedy_verify_ref,
    tree_position_ids,
)
from forge.kernels.ref.util_rng import derive_seed, philox4x32, philox_u32, philox_uniform


def test_philox_known_answer_and_reproducibility():
    r = philox4x32(0, 0, 0, 0, 0, 0)
    assert [int(x) for x in r] == [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8]
    # published KAT 2: counter = key = 0xffffffff
    r2 = philox4x32(*([0xFFFFFFFF] * 4), 0xFFFFFFFF, 0xFFFFFFFF)
    assert [int(x) for x in r2] == [0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD]
    off = torch.arange(1000, dtype=torch.int64)
    a = philox_uniform(1234, off)
    b = philox_uniform(1234, off)
    c = philox_uniform(1235, off)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert float(a.min()) >= 0 and float(a.max()) < 1
    assert abs(float(a.mean()) - 0.5) < 0.03
    # element value independent of chunking / launch geometry
    whole = philox_uniform(7, torch.arange(100))
    parts = torch.cat([philox_uniform(7, torch.arange(0, 37)), philox_uniform(7, torch.arange(37, 100))])
    assert torch.equal(whole, parts)
    assert derive_seed(1, 2, 3) == derive_seed(1, 2, 3) != derive_seed(1, 3, 2)
    assert philox_u32(5, torch.tensor([3])).dtype == torch.int64


def test_tree_helpers():
    parents = torch.tensor([-1, 0, 0, 1, 1, 2], dtype=torch.int32)
    assert tree_depths(parents).tolist() == [0, 1, 1, 2, 2, 2]
    m = build_tree_mask(parents)
    assert m[3].tolist() == [True, True, False, True, False, False]
    assert m[5].tolist() == [True, False, True, False, False, True]
    assert tree_position_ids(parents, 10).tolist() == [10, 11, 11, 12, 12, 12]
    assert tree_position_ids(parents, torch.tensor([3, 4])).shape == (2, 6)


def test_chain_tree_attention_equals_causal_prefill():
    """A chain tree over appended tokens is exactly chunked-prefill causal attention."""
    g = torch.Generator().manual_seed(0)
    hq, hkv, d, ctx, n = 4, 2, 16, 21, 5
    k_all = torch.randn(1, hkv, ctx + n, d, generator=g)
    v_all = torch.randn(1, hkv, ctx + n, d, generator=g)
    q = torch.randn(1, n, hq, d, generator=g)
    pools = build_paged_pools(k_all[:, :, :ctx], v_all[:, :, :ctx], [ctx], 16, "fp32")
    mask = build_tree_mask(chain_parents(n))
    out = tree_attention_ref(q, k_all[:, :, ctx:].permute(0, 2, 1, 3), v_all[:, :, ctx:].permute(0, 2, 1, 3),
                             pools["k_pool"], pools["v_pool"], pools["block_tables"], torch.tensor([ctx]), mask,
                             kv_dtype="fp32")
    exp = prefill_attention_ref(q, k_all.permute(0, 2, 1, 3), v_all.permute(0, 2, 1, 3), causal=True)
    torch.testing.assert_close(out, exp, atol=1e-5, rtol=1e-5)


def test_branching_tree_attention_matches_per_path_attention():
    g = torch.Generator().manual_seed(4)
    hq, hkv, d, ctx = 4, 2, 16, 18
    parents = torch.tensor([-1, 0, 0, 1, 2, 2], dtype=torch.int32)
    n = parents.numel()
    k_ctx = torch.randn(1, hkv, ctx, d, generator=g)
    v_ctx = torch.randn(1, hkv, ctx, d, generator=g)
    kt = torch.randn(1, n, hkv, d, generator=g)
    vt = torch.randn(1, n, hkv, d, generator=g)
    q = torch.randn(1, n, hq, d, generator=g)
    pools = build_paged_pools(k_ctx, v_ctx, [ctx], 16, "fp32")
    out = tree_attention_ref(q, kt, vt, pools["k_pool"], pools["v_pool"], pools["block_tables"], torch.tensor([ctx]),
                             build_tree_mask(parents), kv_dtype="fp32")
    mask = build_tree_mask(parents)
    for i in range(n):
        path = [j for j in range(n) if mask[i, j]]
        k_seq = torch.cat([k_ctx[0].permute(1, 0, 2), kt[0, path]], 0)  # [ctx+len, Hkv, D]
        v_seq = torch.cat([v_ctx[0].permute(1, 0, 2), vt[0, path]], 0)
        qi = q[:, i : i + 1]
        # the node is the last token of its root-to-node path: full (non-causal-limited) attention of 1 query
        exp = prefill_attention_ref(qi, k_seq[None], v_seq[None], causal=True)
        torch.testing.assert_close(out[0, i], exp[0, 0], atol=1e-5, rtol=1e-5)


def test_tree_attention_window_limits_context():
    g = torch.Generator().manual_seed(8)
    hq, hkv, d, ctx = 2, 1, 8, 30
    kc = torch.randn(1, hkv, ctx, d, generator=g)
    vc = torch.randn(1, hkv, ctx, d, generator=g)
    kt = torch.randn(1, 2, hkv, d, generator=g)
    vt = torch.randn(1, 2, hkv, d, generator=g)
    q = torch.randn(1, 2, hq, d, generator=g)
    pools = build_paged_pools(kc, vc, [ctx], 16, "fp32")
    mask = build_tree_mask(chain_parents(2))
    out = tree_attention_ref(q, kt, vt, pools["k_pool"], pools["v_pool"], pools["block_tables"], torch.tensor([ctx]), mask,
                             kv_dtype="fp32", window=4)
    # node 0 at position 30 sees positions 27..30 -> ctx 27,28,29 + itself
    k_seq = torch.cat([kc[0, 0, 27:30], kt[0, 0:1, 0]], 0)
    v_seq = torch.cat([vc[0, 0, 27:30], vt[0, 0:1, 0]], 0)
    p = torch.softmax((q[0, 0] @ k_seq.T) / math.sqrt(d), -1)
    torch.testing.assert_close(out[0, 0], p @ v_seq, atol=1e-5, rtol=1e-5)


def _normalise(x):
    return x / x.sum(-1, keepdim=True)


def test_rejection_sampling_is_distribution_exact_two_step():
    """Statistical exactness (Leviathan et al. Thm 1): emitted tokens follow the target law irrespective of the draft."""
    torch.manual_seed(1)
    v, gamma, n = 5, 2, 120_000
    p0 = _normalise(torch.rand(v) + 0.05)
    q0 = _normalise(torch.rand(v) + 0.05)
    p1 = _normalise(torch.rand(v, v) + 0.05)   # target law of token 1 given token 0
    q1 = _normalise(torch.rand(v, v) + 0.05)
    p2 = _normalise(torch.rand(v, v, v) + 0.05)  # bonus law given (x0, x1)
    gen = torch.Generator().manual_seed(11)
    x0 = torch.multinomial(q0.expand(n, v), 1, generator=gen).squeeze(1)
    x1 = torch.multinomial(q1[x0], 1, generator=gen).squeeze(1)
    draft = torch.stack([x0, x1], 1)
    q_probs = torch.stack([q0.expand(n, v), q1[x0]], 1)
    p_probs = torch.stack([p0.expand(n, v), p1[x0], p2[x0, x1]], 1)
    toks, n_acc = rejection_verify_ref(draft, q_probs, p_probs, seed=99)
    t0 = toks[:, 0].long()
    freq0 = torch.bincount(t0, minlength=v).float() / n
    assert (freq0 - p0).abs().max() < 0.006, (freq0, p0)
    # conditional law of token 1 given token 0, on runs that emitted >= 2 tokens
    two = n_acc >= 1
    t1 = toks[:, 1].long()
    for a in range(v):
        sel = two & (t0 == a)
        f = torch.bincount(t1[sel], minlength=v).float() / sel.sum()
        assert (f - p1[a]).abs().max() < 0.02, (a, f, p1[a])
    # bookkeeping invariants
    assert ((toks >= 0).sum(1) == n_acc + 1).all()
    assert (toks[:, 2][n_acc < 2] == -1).all()
    # acceptance rate of position 0 = sum min(p, q)
    exp_acc0 = torch.minimum(p0, q0).sum()
    assert abs(float((n_acc >= 1).float().mean()) - float(exp_acc0)) < 0.006


def test_rejection_sampling_identical_draft_always_accepts_and_is_reproducible():
    torch.manual_seed(2)
    v, g, n = 7, 3, 2000
    p = _normalise(torch.rand(n, g + 1, v) + 0.1)
    q = p[:, :g].clone()
    draft = torch.multinomial(q.reshape(-1, v), 1).reshape(n, g)
    toks, n_acc = rejection_verify_ref(draft, q, p, seed=5)
    assert (n_acc == g).all()
    assert torch.equal(toks[:, :g], draft.to(torch.int32))
    toks2, _ = rejection_verify_ref(draft, q, p, seed=5)
    toks3, _ = rejection_verify_ref(draft, q, p, seed=6)
    assert torch.equal(toks, toks2) and not torch.equal(toks, toks3)


def test_rejection_edge_cases_zero_q_and_disjoint_support():
    v = 4
    # draft token impossible under target: always rejected, resample from p
    p = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.25] * 4]).unsqueeze(0).expand(1000, 2, v)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).unsqueeze(0).expand(1000, 1, v)
    draft = torch.zeros(1000, 1, dtype=torch.long)
    toks, n_acc = rejection_verify_ref(draft, q, p, seed=1)
    assert (n_acc == 0).all() and set(toks[:, 0].tolist()) == {2, 3}
    # q == 0 at the drafted token but p > 0: accepted (u * 0 < p)
    q2 = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).unsqueeze(0).expand(1000, 1, v)
    p2 = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.25] * 4]).unsqueeze(0).expand(1000, 2, v)
    toks2, n2 = rejection_verify_ref(torch.zeros(1000, 1, dtype=torch.long), q2, p2, seed=3)
    assert (n2 == 1).all() and (toks2[:, 0] == 0).all()


def test_greedy_verify_and_tree_greedy_verify():
    draft = torch.tensor([[3, 4, 5], [3, 9, 5]])
    tgt = torch.tensor([[3, 4, 5, 8], [3, 4, 5, 8]])
    toks, n = greedy_verify_ref(draft, tgt)
    assert n.tolist() == [3, 1]
    assert toks[0].tolist() == [3, 4, 5, 8] and toks[1].tolist() == [3, 4, -1, -1]
    parents = torch.tensor([-1, 0, 0, 1, 2], dtype=torch.int32)
    tokens = torch.tensor([[7, 8, 9, 10, 11]])
    # root_pred 7; node0 predicts 9 (-> node 2 matches); node2 predicts 11 (-> node 4 matches)
    target = torch.tensor([[9, 99, 11, 99, 42]])
    acc, ln, bonus = tree_greedy_verify_ref(tokens, parents, target, torch.tensor([7]))
    assert acc[0].tolist()[:3] == [0, 2, 4] and ln.item() == 3 and bonus.item() == 42
    acc2, ln2, bonus2 = tree_greedy_verify_ref(tokens, parents, target, torch.tensor([1]))
    assert ln2.item() == 0 and bonus2.item() == 1


def test_kv_compact_moves_accepted_path_with_overlaps():
    torch.manual_seed(0)
    ps, h, d = 4, 2, 3
    b, ctx, n = 2, [5, 8], 6
    pool_k = torch.randn(12, h, ps, d)
    pool_v = torch.randn(12, h, ps, d)
    bt = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)
    k0, v0 = pool_k.clone(), pool_v.clone()

    def pos_get(pool, bi, pos):
        return pool[int(bt[bi, pos // ps]), :, pos % ps]

    acc = torch.tensor([[0, 2, 3, 5, -1, -1], [1, 2, -1, -1, -1, -1]], dtype=torch.int32)
    lens = torch.tensor([4, 2], dtype=torch.int32)
    kv_compact_ref(pool_k, pool_v, bt, torch.tensor(ctx), acc, lens)
    for bi in range(b):
        for t in range(int(lens[bi])):
            src = ctx[bi] + int(acc[bi, t])
            dst = ctx[bi] + t
            torch.testing.assert_close(pos_get(pool_k, bi, dst), pos_get(k0, bi, src))
            torch.testing.assert_close(pos_get(pool_v, bi, dst), pos_get(v0, bi, src))
    # untouched region beyond the accepted run stays intact for seq 1 (positions ctx+2..)
    torch.testing.assert_close(pos_get(pool_k, 1, ctx[1] + 4), pos_get(k0, 1, ctx[1] + 4))
