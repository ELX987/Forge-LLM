"""MoE + utility Triton kernels executed in the Triton **CPU interpreter** and compared with the oracles (no GPU involved).

Not collected directly (no ``test_`` prefix): run by ``test_kernels_attn_triton_interp.py`` in a subprocess with
``TRITON_INTERPRET=1`` (the interpreter must be enabled before ``triton`` is first imported).  Covers the edge-case list of Plan
6.7 that applies to these kernels: M=0 experts, ragged expert loads, counter-based-RNG reproducibility, zero rows, etc.
"""
from __future__ import annotations

import os

os.environ["TRITON_INTERPRET"] = "1"  # must precede the first `import triton`

import importlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton")

from forge.kernels.ref.attn_kv import dequantize_rows  # noqa: E402
from forge.kernels.ref.moe_ref import (  # noqa: E402
    apply_expert_map_ref,
    build_block_schedule_ref,
    max_blocks_bound,
    moe_experts_forward_ref,
    moe_grouped_gemm_ref,
    moe_permute_ref,
    moe_route_ref,
    moe_unpermute_combine_ref,
    pack_w4_experts,
)
from forge.kernels.ref.util_adapters import segmented_lora_ref, teal_gemv_ref, teal_threshold  # noqa: E402
from forge.kernels.ref.util_hadamard import hadamard_ref, random_signs  # noqa: E402
from forge.kernels.ref.util_quant import quantize_rowwise_ref, rmsnorm_quant_ref, stochastic_round_ref  # noqa: E402
from forge.kernels.ref.util_rope import make_cos_sin_cache, rope_and_kv_write_ref  # noqa: E402


@pytest.fixture(scope="module")
def tk():
    names = ["moe_kernels", "util_hadamard", "util_quant", "util_rope", "util_gemv"]
    return SimpleNamespace(**{n: importlib.import_module("forge.kernels.triton." + n) for n in names})


# ------------------------------------------------------------------------------------------------ MoE
@pytest.mark.parametrize("e,k", [(8, 2), (64, 8), (5, 1), (33, 4)])
def test_route_matches_reference_including_ties(tk, e, k):
    torch.manual_seed(e)
    logits = torch.randn(40, e)
    logits[3] = 1.0  # exact ties across all experts -> lowest ids
    logits[4, : min(e, 4)] = 2.0
    for norm in (True, False):
        rw, ri = moe_route_ref(logits, k, norm_topk_prob=norm, routed_scaling=1.5)
        tw, ti = tk.moe_kernels.moe_route_triton(logits, k, norm_topk_prob=norm, routed_scaling=1.5)
        assert torch.equal(ri, ti)
        torch.testing.assert_close(tw, rw, atol=1e-6, rtol=1e-5)
    bias = torch.randn(e)
    rw, ri = moe_route_ref(logits, k, scoring="sigmoid", score_bias=bias)
    tw, ti = tk.moe_kernels.moe_route_triton(logits, k, scoring="sigmoid", score_bias=bias)
    assert torch.equal(ri, ti)
    torch.testing.assert_close(tw, rw, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("bm", [16, 32])
def test_grouped_gemm_variable_m_with_empty_experts(tk, bm):
    torch.manual_seed(0)
    e, k, n, t = 7, 40, 24, 30
    ids = torch.randint(0, e, (t, 2), dtype=torch.int32)
    ids[(ids == 2) | (ids == 5)] = 0  # experts 2 and 5 receive nothing (M = 0)
    ids[0] = torch.tensor([6, 6])  # expert 6 gets a couple of rows
    pr = moe_permute_ref(ids, e)
    x = torch.randn(t, k)
    w = torch.randn(e, n, k)
    m = t * 2
    ref = moe_grouped_gemm_ref(x, w, pr.expert_offsets, gather_idx=pr.token_idx)
    out = tk.moe_kernels.moe_grouped_gemm_triton(x, w, pr.expert_offsets, gather_idx=pr.token_idx, total_rows=m,
                                                 block_m=bm, block_n=16, block_k=16)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    # un-gathered variant + block schedule reuse
    xs = x[pr.token_idx.long()]
    sched = build_block_schedule_ref(pr.expert_offsets, bm, max_blocks_bound(m, e, bm))
    out2 = tk.moe_kernels.moe_grouped_gemm_triton(xs, w, pr.expert_offsets, total_rows=m, block_m=bm, schedule=sched)
    torch.testing.assert_close(out2, moe_grouped_gemm_ref(xs, w, pr.expert_offsets), atol=1e-4, rtol=1e-4)


def test_grouped_gemm_all_tokens_one_expert_and_single_row(tk):
    torch.manual_seed(1)
    e, k, n = 4, 32, 16
    w = torch.randn(e, n, k)
    for counts in ([0, 37, 0, 0], [0, 0, 1, 0], [3, 0, 0, 5]):
        off = torch.zeros(e + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(torch.tensor(counts), 0)
        m = int(off[-1])
        x = torch.randn(m, k)
        out = tk.moe_kernels.moe_grouped_gemm_triton(x, w, off, total_rows=m, block_m=16)
        torch.testing.assert_close(out, moe_grouped_gemm_ref(x, w, off), atol=1e-4, rtol=1e-4)


def test_grouped_gemm_scatter_accumulate_alpha(tk):
    torch.manual_seed(2)
    x = torch.randn(10, 16)
    w = torch.randn(3, 8, 16)
    off = torch.tensor([0, 4, 4, 10], dtype=torch.int32)
    out_idx = torch.randperm(10).to(torch.int32)
    base = torch.randn(10, 8)
    alpha = torch.tensor([2.0, 7.0, 0.5])
    ref = moe_grouped_gemm_ref(x, w, off, out_idx=out_idx, out=base.clone(), accumulate=True, alpha=alpha)
    out = tk.moe_kernels.moe_grouped_gemm_triton(x, w, off, out_idx=out_idx, out=base.clone(), accumulate=True,
                                                 alpha=alpha, total_rows=10, block_m=16)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("sym", [True, False])
def test_grouped_gemm_w4a16_hook(tk, sym):
    torch.manual_seed(3)
    e, n, k, g = 3, 16, 64, 32
    w = torch.randn(e, n, k)
    q, s, z = pack_w4_experts(w, g, sym)
    x = torch.randn(9, k)
    off = torch.tensor([0, 5, 5, 9], dtype=torch.int32)
    ref = moe_grouped_gemm_ref(x, q, off, w_scales=s, w_zeros=z, group_size=g)
    out = tk.moe_kernels.moe_grouped_gemm_triton(x, q, off, w_scales=s, w_zeros=z, group_size=g, total_rows=9,
                                                 block_m=16, block_k=32)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)


def test_combine_and_full_moe_pipeline_via_ops_dispatch(tk):
    from forge.kernels import ops_moe

    torch.manual_seed(4)
    t, h, i, e, k = 13, 32, 16, 6, 2
    x = torch.randn(t, h)
    wgu = torch.randn(e, 2 * i, h) * 0.2
    wd = torch.randn(e, h, i) * 0.2
    w, ids = moe_route_ref(torch.randn(t, e), k)
    pr = moe_permute_ref(ids, e)
    y = torch.randn(t * k, h)
    torch.testing.assert_close(tk.moe_kernels.moe_combine_triton(y, pr.inv_perm, w),
                               moe_unpermute_combine_ref(y, pr.inv_perm, w), atol=1e-5, rtol=1e-5)
    # ops-level dispatch uses Triton here (interpreter mode) - compare to the dense per-token reference
    out = ops_moe.moe_experts_forward(x, w, ids, wgu, wd)
    torch.testing.assert_close(out, moe_experts_forward_ref(x, w, ids, wgu, wd), atol=2e-4, rtol=2e-4)
    # pruned experts via the indirection table
    emap = torch.tensor([0, -1, 1, 2, -1, 3])
    keep = torch.tensor([0, 2, 3, 5])
    out_p = ops_moe.moe_experts_forward(x, w, ids, wgu[keep], wd[keep], expert_map=emap)
    mids, mw = apply_expert_map_ref(ids, w, emap)
    torch.testing.assert_close(out_p, moe_experts_forward_ref(x, mw, mids, wgu[keep], wd[keep]), atol=2e-4, rtol=2e-4)


# ------------------------------------------------------------------------------------------------ util
@pytest.mark.parametrize("block", [16, 32, 64, 128])
def test_hadamard_kernel(tk, block):
    torch.manual_seed(block)
    x = torch.randn(7, 256)
    signs = random_signs(256, 5)
    for sg in (None, signs):
        torch.testing.assert_close(tk.util_hadamard.hadamard_triton(x, block, sg), hadamard_ref(x, block, sg),
                                   atol=2e-5, rtol=2e-5)
    out = tk.util_hadamard.hadamard_triton(x, block, None, normalize=False)
    torch.testing.assert_close(out, hadamard_ref(x, block, None, normalize=False), atol=1e-4, rtol=1e-5)


def test_quantize_rowwise_int8_matches_reference(tk):
    torch.manual_seed(1)
    x = torch.randn(6, 96) * 4
    x[2] = 0.0
    for group in (None, 32):
        q, s = tk.util_quant.quantize_rowwise_triton(x, "int8", group)
        qr, sr = quantize_rowwise_ref(x, "int8", group)
        torch.testing.assert_close(s, sr, atol=1e-7, rtol=1e-6)
        assert (q.int() - qr.int()).abs().max() <= 1  # division/ulp differences can flip an exact .5 tie
        assert (q == qr).float().mean() > 0.995
    assert (q[2] == 0).all()


def test_rmsnorm_quant_int8_matches_reference(tk):
    torch.manual_seed(2)
    x = torch.randn(5, 80)
    r = torch.randn(5, 80)
    w = torch.rand(80) + 0.5
    for res in (None, r):
        qt, st, rt = tk.util_quant.rmsnorm_quant_triton(x, w, 1e-6, "int8", res)
        qr, sr, rr = rmsnorm_quant_ref(x, w, 1e-6, "int8", res)
        torch.testing.assert_close(st, sr, atol=1e-6, rtol=1e-5)
        assert (qt.int() - qr.int()).abs().max() <= 1
        if res is not None:
            torch.testing.assert_close(rt, rr)


@pytest.mark.parametrize("fmt,group", [("int8", 32), ("int4", 16), ("fp4", 16)])
def test_stochastic_rounding_kernel_matches_reference_and_is_reproducible(tk, fmt, group):
    torch.manual_seed(3)
    x = torch.randn(8, 64) * 2
    ref_q, scale = stochastic_round_ref(x, fmt, group_size=group, seed=1234, offset=0)
    q = tk.util_quant.stochastic_round_triton(x, scale, group, fmt, 1234, 0)
    assert (q == ref_q).float().mean() > 0.999  # identical uniforms; only division ulp can flip a decision
    q2 = tk.util_quant.stochastic_round_triton(x, scale, group, fmt, 1234, 0)
    assert torch.equal(q, q2)
    q3 = tk.util_quant.stochastic_round_triton(x, scale, group, fmt, 1235, 0)
    assert not torch.equal(q, q3)
    # chunked launches with the right offset reproduce the whole-tensor result (launch-geometry independence)
    lo = tk.util_quant.stochastic_round_triton(x[:3], scale[:3], group, fmt, 1234, 0)
    hi = tk.util_quant.stochastic_round_triton(x[3:], scale[3:], group, fmt, 1234, 3 * 64)
    assert torch.equal(torch.cat([lo, hi]), q)


@pytest.mark.parametrize("kvd,interleaved,rd", [("bf16", False, None), ("bf16", True, 8), ("int8", False, None), ("int8", False, 8)])
def test_rope_kv_write_kernel(tk, kvd, interleaved, rd):
    torch.manual_seed(5)
    t, hq, hkv, d, ps = 6, 4, 2, 16, 8
    q, k, v = torch.randn(t, hq, d), torch.randn(t, hkv, d), torch.randn(t, hkv, d)
    pos = torch.tensor([0, 3, 4, 9, 10, 31])
    cache = make_cos_sin_cache(64, rd or d)
    slots = torch.tensor([17, 18, -1, 9, 23, 0], dtype=torch.int32)
    pdt = torch.float32 if kvd == "bf16" else torch.int8
    mk = lambda: (torch.zeros(4, hkv, ps, d, dtype=pdt), torch.zeros(4, hkv, ps, d, dtype=pdt), torch.zeros(4, hkv, ps))  # noqa: E731
    kp1, vp1, vs1 = mk()
    kp2, vp2, vs2 = mk()
    ks = torch.full((hkv, d), 0.07)
    kw = dict(kv_dtype="fp32" if kvd == "bf16" else "int8", k_scale=ks if kvd == "int8" else None, rotary_dim=rd,
              interleaved=interleaved)
    qr = rope_and_kv_write_ref(q, k, v, pos, cache, slots, kp1, vp1, v_scale_pool=vs1, **kw)
    qt = tk.util_rope.rope_and_kv_write_triton(q, k, v, pos, cache, slots, kp2, vp2, v_scale_pool=vs2, **kw)
    torch.testing.assert_close(qt, qr, atol=1e-5, rtol=1e-5)
    if kvd == "bf16":
        torch.testing.assert_close(kp2, kp1, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(vp2, vp1, atol=1e-5, rtol=1e-5)
    else:
        assert (kp2.int() - kp1.int()).abs().max() <= 1 and (vp2.int() - vp1.int()).abs().max() <= 1
        torch.testing.assert_close(vs2, vs1, atol=1e-6, rtol=1e-5)
    assert torch.equal(kp2 != 0, kp1 != 0)  # the skipped slot (-1) wrote nothing in either implementation


def test_teal_gemv_kernel(tk):
    torch.manual_seed(6)
    x = torch.randn(3, 96)
    w = torch.randn(96, 40)
    thr = teal_threshold(x, 0.5)
    bias = torch.randn(40)
    torch.testing.assert_close(tk.util_gemv.teal_gemv_triton(x, w, thr), teal_gemv_ref(x, w, thr), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tk.util_gemv.teal_gemv_triton(x, w, thr, bias), teal_gemv_ref(x, w, thr, bias), atol=1e-4,
                               rtol=1e-4)
    torch.testing.assert_close(tk.util_gemv.teal_gemv_triton(x, w, 0.0), x @ w, atol=1e-4, rtol=1e-4)


def test_segmented_lora_mixed_batch_via_ops(tk):
    from forge.kernels import ops_util

    torch.manual_seed(7)
    t, k, n = 17, 32, 24
    ranks = [4, 8, 16]
    a_list = [torch.randn(r, k) * 0.3 for r in ranks]
    b_list = [torch.randn(n, r) * 0.3 for r in ranks]
    a_st, b_st = ops_util.pad_adapters(a_list, b_list, 16)
    x, y = torch.randn(t, k), torch.randn(t, n)
    ids = torch.randint(-1, 3, (t,))
    ids[0], ids[1] = 1, -1
    sc = torch.tensor([0.5, 2.0, 1.0])
    ref = segmented_lora_ref(x, y, ids, a_st, b_st, sc)
    out = ops_util.segmented_lora(x, y, ids, a_st, b_st, sc)
    torch.testing.assert_close(out, ref, atol=2e-4, rtol=2e-4)
    # unpadded rank (4) is padded internally on the Triton path
    only = torch.where(ids == 1, torch.zeros_like(ids), torch.full_like(ids, -1))
    out2 = ops_util.segmented_lora(x, y, only, a_list[0][None], b_list[0][None], 3.0)
    ref2 = segmented_lora_ref(x, y, only, a_list[0][None], b_list[0][None], 3.0)
    torch.testing.assert_close(out2, ref2, atol=2e-4, rtol=2e-4)
    # inplace variant mutates y
    y2 = y.clone()
    ops_util.segmented_lora(x, y2, ids, a_st, b_st, sc, inplace=True)
    torch.testing.assert_close(y2, ref, atol=2e-4, rtol=2e-4)


def test_ops_dispatch_triton_vs_forced_reference(tk, monkeypatch):
    """With the interpreter enabled the dispatchers pick Triton for CPU tensors; FORGE_FORCE_REF=1 must force the oracle."""
    from forge.kernels import ops_attn
    from forge.kernels.ref.attn_decode import paged_decode_attention_ref
    from forge.kernels.ref.attn_kv import build_paged_pools
    from forge.kernels.triton import util_dispatch as ud

    q = torch.randn(2, 4, 32)
    k, v = torch.randn(2, 2, 25, 32), torch.randn(2, 2, 25, 32)
    p = build_paged_pools(k, v, [25, 9], 16, "fp32")
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    assert ud.use_triton(q, min_sm=70) is True
    out_tri = ops_attn.paged_decode_attention(*args, kv_dtype="fp32")
    monkeypatch.setenv("FORGE_FORCE_REF", "1")
    assert ud.use_triton(q, min_sm=70) is False
    out_ref = ops_attn.paged_decode_attention(*args, kv_dtype="fp32")
    torch.testing.assert_close(out_ref, paged_decode_attention_ref(*args, kv_dtype="fp32"))
    torch.testing.assert_close(out_tri, out_ref, atol=2e-5, rtol=2e-5)


def test_quant_kernels_flush_underflowing_and_zero_groups(tk):
    x = torch.zeros(4, 32)
    x[1] = 1e-44
    x[2] = 1e-30
    x[3, :2] = torch.tensor([3.0, -3.0])
    q, s = tk.util_quant.quantize_rowwise_triton(x, "int8")
    qr, sr = quantize_rowwise_ref(x, "int8")
    assert torch.isfinite(s).all() and (s > 0).all()
    assert torch.equal(q, qr) and torch.equal(s, sr)
    qn, sn, _ = tk.util_quant.rmsnorm_quant_triton(x, torch.ones(32), 1e-6, "int8")
    assert torch.isfinite(sn).all() and (sn > 0).all()
