"""CPU tests of the ops_* dispatchers (reference path), bench descriptors and catalog metadata (Plan §5.5)."""
from __future__ import annotations

import math

import pytest
import torch

from forge.kernels import ops_attn, ops_moe, ops_util
from forge.kernels.ref.attn_decode import paged_decode_attention_ref
from forge.kernels.ref.attn_kv import build_paged_pools
from forge.kernels.ref.attn_prefill import prefill_attention_ref
from forge.kernels.ref.attn_sparse import SharedSelector, topk_select_ref
from forge.kernels.ref.moe_ref import moe_experts_forward_ref, moe_route_ref
from forge.kernels.ref.util_adapters import pad_adapters, segmented_lora_ref
from forge.kernels.triton import util_dispatch as ud


def test_dispatch_helpers_cpu_defaults(monkeypatch):
    x = torch.zeros(2)
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    assert ud.use_triton(x) is False  # CPU tensors never use Triton outside the interpreter
    assert ud.device_sm(torch.device("cpu")) is None
    assert ud.next_pow2(1) == 1 and ud.next_pow2(17) == 32 and ud.cdiv(5, 2) == 3
    monkeypatch.setenv("FORGE_FORCE_REF", "1")
    assert ud.force_ref() is True
    monkeypatch.setenv("FORGE_FORCE_REF", "0")
    assert ud.force_ref() is False
    assert ud.fp8_native(torch.device("cpu")) is False


def test_importing_triton_modules_without_triton_is_safe():
    import importlib

    for name in ("attn_common", "attn_decode", "attn_mla", "attn_prefill", "attn_sparse", "attn_spec", "moe_kernels",
                 "util_hadamard", "util_quant", "util_rope", "util_gemv", "util_dispatch"):
        importlib.import_module("forge.kernels.triton." + name)


def test_kernels_metadata_tables():
    for table in (ops_attn.KERNELS, ops_moe.KERNELS, ops_util.KERNELS):
        for name, meta in table.items():
            assert {"family", "min_sm", "triton", "ref"} <= set(meta), name
    assert "paged_decode_attention" in ops_attn.KERNELS and "moe_grouped_gemm" in ops_moe.KERNELS


@pytest.mark.parametrize("kvd", ["bf16", "fp8_e4m3", "int8", "int4"])
def test_paged_decode_dispatch_matches_ref_and_force_ref(kvd, monkeypatch):
    lens = [19, 40]
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 4, 32, generator=g)
    k = torch.randn(2, 2, 40, 32, generator=g)
    v = torch.randn(2, 2, 40, 32, generator=g)
    p = build_paged_pools(k.bfloat16() if kvd == "bf16" else k, v.bfloat16() if kvd == "bf16" else v, lens, 16, kvd)
    kw = dict(kv_dtype=kvd, k_scale=p["k_scale"], v_scale=p["v_scale"])
    out = ops_attn.paged_decode_attention(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], **kw)
    monkeypatch.setenv("FORGE_FORCE_REF", "1")
    out2 = ops_attn.paged_decode_attention(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], **kw)
    ref = paged_decode_attention_ref(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], **kw)
    assert torch.equal(out, ref) and torch.equal(out2, ref)
    o, lse = ops_attn.paged_decode_attention(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], **kw,
                                             return_lse=True)
    assert lse.shape == (2, 4) and torch.isfinite(lse).all()


def test_prefill_layouts_varlen_and_chunked():
    g = torch.Generator().manual_seed(1)
    b, sq, sk, hq, hkv, d = 2, 6, 14, 4, 2, 16
    q = torch.randn(b, sq, hq, d, generator=g)
    k = torch.randn(b, sk, hkv, d, generator=g)
    v = torch.randn(b, sk, hkv, d, generator=g)
    ref = prefill_attention_ref(q, k, v)
    torch.testing.assert_close(ops_attn.prefill_attention(q, k, v), ref)
    bh = ops_attn.prefill_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), layout="bhsd")
    torch.testing.assert_close(bh.transpose(1, 2), ref)
    with pytest.raises(ValueError):
        ops_attn.prefill_attention(q, k, v, layout="nope")
    # varlen
    cuq = torch.tensor([0, 6, 12], dtype=torch.int32)
    cuk = torch.tensor([0, 14, 28], dtype=torch.int32)
    vl = ops_attn.prefill_attention(q.reshape(12, hq, d), k.reshape(28, hkv, d), v.reshape(28, hkv, d),
                                    cu_seqlens_q=cuq, cu_seqlens_k=cuk)
    torch.testing.assert_close(vl.reshape(b, sq, hq, d), ref)
    # chunked prefill over a paged prefix == full attention over [prefix | chunk]
    ctx = [9, 3]
    kc = torch.randn(b, hkv, 9, d, generator=g)
    vc = torch.randn(b, hkv, 9, d, generator=g)
    p = build_paged_pools(kc, vc, ctx, 8, "fp32")
    kn = torch.randn(b * sq, hkv, d, generator=g)
    vn = torch.randn(b * sq, hkv, d, generator=g)
    qn = torch.randn(b * sq, hq, d, generator=g)
    cuq2 = torch.tensor([0, sq, 2 * sq], dtype=torch.int32)
    out = ops_attn.chunked_prefill_attention(qn, kn, vn, p["k_pool"], p["v_pool"], p["block_tables"], torch.tensor(ctx),
                                             cuq2, kv_dtype="fp32")
    for bi in range(b):
        kk = torch.cat([kc[bi, :, : ctx[bi]].permute(1, 0, 2), kn[bi * sq:(bi + 1) * sq]], 0)[None]
        vv = torch.cat([vc[bi, :, : ctx[bi]].permute(1, 0, 2), vn[bi * sq:(bi + 1) * sq]], 0)[None]
        exp = prefill_attention_ref(qn[bi * sq:(bi + 1) * sq][None], kk, vv)
        torch.testing.assert_close(out[bi * sq:(bi + 1) * sq], exp[0], atol=1e-5, rtol=1e-5)


def test_paged_kv_write():
    kp = torch.zeros(3, 2, 4, 8)
    vp = torch.zeros(3, 2, 4, 8)
    k = torch.randn(3, 2, 8)
    v = torch.randn(3, 2, 8)
    ops_attn.paged_kv_write(k, v, kp, vp, torch.tensor([5, -1, 11]))
    torch.testing.assert_close(kp[1, :, 1], k[0])
    torch.testing.assert_close(vp[2, :, 3], v[2])
    assert kp[1].abs().sum() == k[0].abs().sum()  # nothing else written


def test_topk_select_torch_equals_reference_with_ties_and_validity():
    torch.manual_seed(0)
    s = torch.randn(5, 400)
    s[0, ::5] = 2.0
    s[1] = 0.0
    s[2, 10] = float("nan")
    s[3, :3] = float("-inf")
    s[4, 7] = -0.0
    lens = torch.tensor([400, 400, 390, 2, 50])
    for k in (1, 8, 64, 500):
        ref = topk_select_ref(s, k, lens)
        assert torch.equal(ops_util.topk_select(s, k, lens), ref), k
        assert torch.equal(ops_util.topk_select(s, k, lens, method="torch"), ref)
    assert torch.equal(ops_util.topk_select(s, 16), topk_select_ref(s, 16))


def test_sparse_pipeline_token_and_block_granular_and_sharing():
    g = torch.Generator().manual_seed(4)
    b, hq, hkv, d, ps, hi, di = 2, 4, 2, 16, 16, 4, 16
    lens = [90, 41]
    k = torch.randn(b, hkv, 90, d, generator=g)
    v = torch.randn(b, hkv, 90, d, generator=g)
    p = build_paged_pools(k, v, lens, ps, "fp32")
    ik = torch.randn(b, 1, 96, di, generator=g)
    ip = build_paged_pools(ik, ik, lens, ps, "fp32")
    ik_pool = ip["k_pool"][:, 0]
    iq = torch.randn(b, hi, di, generator=g)
    iw = torch.rand(b, hi, generator=g)
    q = torch.randn(b, hq, d, generator=g)
    tok_idx = ops_attn.select_sparse_indices(iq, iw, ik_pool, None, ip["block_tables"], p["seq_lens"], 32)
    assert tok_idx.shape == (b, 1, 32)
    # selection = exact top-32 of the scan
    sc = ops_attn.lightning_indexer_scores(iq, iw, ik_pool, None, ip["block_tables"], p["seq_lens"])
    for bi in range(b):
        want = set(torch.topk(sc[bi, : lens[bi]], min(32, lens[bi])).indices.tolist())
        assert set(tok_idx[bi, 0][tok_idx[bi, 0] >= 0].tolist()) == want
    blk_idx = ops_attn.select_sparse_indices(iq, iw, ik_pool, None, ip["block_tables"], p["seq_lens"], 32, block_size=8)
    assert blk_idx.shape == (b, 1, 32)
    for bi in range(b):
        row = blk_idx[bi, 0]
        assert (row[row >= 0] // 8).unique().numel() <= 4  # 32-token budget = 4 blocks of 8
    sel = SharedSelector(share_every=2)
    outs = [ops_attn.sparse_decode_step(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], iq, iw, ik_pool, None,
                                        top_k=32, selector=sel, layer=l, kv_dtype="fp32",
                                        index_block_tables=ip["block_tables"]) for l in range(4)]
    assert sel.indexer_passes == 2
    ref = paged_decode_attention_ref(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], kv_dtype="fp32",
                                     topk_idx=tok_idx)
    torch.testing.assert_close(outs[0], ref)
    rep = ops_attn.selector_bytes(131072, index_share_layers=4)
    assert round(rep.reduction, 1) == 22.6


def test_tree_attention_dispatch_reference():
    g = torch.Generator().manual_seed(2)
    parents = torch.tensor([-1, 0, 0, 1], dtype=torch.int32)
    n, hq, hkv, d, ctx = 4, 4, 2, 16, 11
    kc = torch.randn(1, hkv, ctx, d, generator=g)
    p = build_paged_pools(kc, kc, [ctx], 8, "fp32")
    q = torch.randn(1, n, hq, d, generator=g)
    kt = torch.randn(1, n, hkv, d, generator=g)
    out = ops_attn.tree_attention(q, kt, kt, p["k_pool"], p["v_pool"], p["block_tables"], torch.tensor([ctx]),
                                  ops_attn.build_tree_mask(parents), kv_dtype="fp32")
    assert out.shape == (1, n, hq, d) and torch.isfinite(out).all()
    # residual window on the reference path: last 4 of the 11 context tokens live in the window; must equal the all-in-pool result
    r = 4
    kp_small = build_paged_pools(kc[:, :, : ctx - r], kc[:, :, : ctx - r], [ctx - r], 8, "fp32")
    out_res = ops_attn.tree_attention(q, kt, kt, kp_small["k_pool"], kp_small["v_pool"], kp_small["block_tables"],
                                      torch.tensor([ctx]), ops_attn.build_tree_mask(parents), kv_dtype="fp32",
                                      k_res=kc[:, :, ctx - r:], v_res=kc[:, :, ctx - r:], res_lens=torch.tensor([r]))
    torch.testing.assert_close(out_res, out, atol=1e-5, rtol=1e-5)


def test_mla_dispatch():
    g = torch.Generator().manual_seed(5)
    h, dc, dr, ps = 4, 16, 8, 8
    c = torch.randn(3, ps, dc, generator=g)
    r = torch.randn(3, ps, dr, generator=g)
    bt = torch.tensor([[0, 1]], dtype=torch.int32)
    out = ops_attn.mla_decode_attention(torch.randn(1, h, dc, generator=g), torch.randn(1, h, dr, generator=g), c, r, bt,
                                        torch.tensor([13]), softmax_scale=0.25)
    assert out.shape == (1, h, dc)


def test_moe_ops_pipeline_and_expert_map():
    torch.manual_seed(6)
    t, h, i, e, k = 9, 16, 8, 5, 2
    x = torch.randn(t, h)
    wgu = torch.randn(e, 2 * i, h) * 0.3
    wd = torch.randn(e, h, i) * 0.3
    w, ids = ops_moe.moe_route(torch.randn(t, e), k)
    out = ops_moe.moe_experts_forward(x, w, ids, wgu, wd)
    torch.testing.assert_close(out, moe_experts_forward_ref(x, w, ids, wgu, wd), atol=1e-4, rtol=1e-4)
    emap = torch.tensor([0, -1, 1, 2, -1])
    keep = torch.tensor([0, 2, 3])
    pruned = ops_moe.moe_experts_forward(x, w, ids, wgu[keep], wd[keep], expert_map=emap)
    mids, mw = ops_moe.apply_expert_map(ids, w, emap)
    torch.testing.assert_close(pruned, moe_experts_forward_ref(x, mw, mids, wgu[keep], wd[keep]), atol=1e-4, rtol=1e-4)
    # W4A16 experts through the hook
    q1, s1, _ = ops_moe.pack_w4_experts(wgu, 8)
    q2, s2, _ = ops_moe.pack_w4_experts(wd, 8)
    out4 = ops_moe.moe_experts_forward(x, w, ids, q1, q2, gate_up_scales=s1, down_scales=s2, group_size=8)
    ref4 = moe_experts_forward_ref(x, w, ids, ops_moe.dequant_w4_experts(q1, s1, None, 8),
                                   ops_moe.dequant_w4_experts(q2, s2, None, 8))
    torch.testing.assert_close(out4, ref4, atol=1e-4, rtol=1e-4)
    # empty batch and all-experts-empty edge cases
    assert ops_moe.moe_experts_forward(x[:0], w[:0], ids[:0], wgu, wd).shape == (0, h)


def test_util_ops_shapes_and_oracle_agreement():
    x = torch.randn(3, 5, 64)
    assert torch.equal(ops_util.hadamard_transform(x, 16), ops_util.hadamard_transform(x, 16))
    y = ops_util.hadamard_transform(x, 64, signs=ops_util.random_signs(64, 1))
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-4)
    q, s = ops_util.quantize_fp8_rowwise(x)
    assert q.dtype == torch.float8_e4m3fn and s.shape == (3, 5, 1)
    xd = ops_util.dequantize_rowwise(q, s)
    assert (xd - x).abs().max() / x.abs().max() < 0.08
    qi, si = ops_util.quantize_rowwise(x, "int8", 16)
    assert qi.dtype == torch.int8 and si.shape == (3, 5, 4)
    codes, sc = ops_util.stochastic_round(x, "fp4", group_size=16, seed=3)
    assert codes.dtype == torch.uint8 and int(codes.max()) <= 15
    with pytest.raises(ValueError):
        ops_util.stochastic_round(x, "int8", seed=-1)
    w = torch.rand(64) + 0.5
    qn, sn, res = ops_util.rmsnorm_quant(x, w, 1e-6, "int8", residual=torch.randn_like(x))
    assert qn.shape == x.shape and sn.shape == (3, 5) and res.shape == x.shape
    assert ops_util.absmax(x).shape == (3, 5, 1)


def test_rope_write_rejection_and_compaction_ops():
    t, hq, hkv, d, ps = 4, 2, 2, 16, 8
    q, k, v = torch.randn(t, hq, d), torch.randn(t, hkv, d), torch.randn(t, hkv, d)
    kp = torch.zeros(3, hkv, ps, d)
    vp = torch.zeros(3, hkv, ps, d)
    cache = ops_util.make_cos_sin_cache(32, d)
    qr = ops_util.rope_and_kv_write(q, k, v, torch.arange(4), cache, torch.tensor([0, 1, 2, 9]), kp, vp, kv_dtype="fp32")
    assert qr.shape == q.shape and kp[1, :, 1].abs().sum() > 0
    probs = torch.softmax(torch.randn(6, 3, 11), -1)
    draft = torch.randint(0, 11, (6, 2))
    toks, n = ops_util.rejection_verify(draft, probs[:, :2], probs, seed=5)
    assert toks.shape == (6, 3) and ((toks >= 0).sum(1) == n + 1).all()
    gt, gn = ops_util.greedy_verify(draft, torch.cat([draft, draft[:, :1]], 1))
    assert (gn == 2).all()
    bt = torch.tensor([[0, 1]], dtype=torch.int32)
    kpool = torch.randn(2, 1, 4, 3)
    vpool = torch.randn(2, 1, 4, 3)
    k0 = kpool.clone()
    ops_util.kv_compact(kpool, vpool, bt, torch.tensor([2]), torch.tensor([[1, 3]], dtype=torch.int32), torch.tensor([2]))
    torch.testing.assert_close(kpool[0, 0, 2], k0[0, 0, 3])  # position 2+1=3 -> 2
    torch.testing.assert_close(kpool[0, 0, 3], k0[1, 0, 1])  # position 2+3=5 (page 1, slot 1) -> position 3 (page 0, slot 3)


def test_teal_and_segmented_lora_ops_cpu():
    x = torch.randn(2, 64)
    wt = torch.randn(64, 32)
    thr = ops_util.teal_threshold(x, 0.5)
    y = ops_util.teal_gemv(x, wt, thr)
    assert y.shape == (2, 32)
    a_list = [torch.randn(4, 64) * 0.2, torch.randn(8, 64) * 0.2]
    b_list = [torch.randn(32, 4) * 0.2, torch.randn(32, 8) * 0.2]
    a_st, b_st = pad_adapters(a_list, b_list, 8)
    xx, yy = torch.randn(10, 64), torch.randn(10, 32)
    ids = torch.tensor([0, 1, -1, 1, 0, 0, -1, 1, 5, 1])  # 5 = unknown adapter -> treated as none
    sc = torch.tensor([1.0, 0.5])
    out = ops_util.segmented_lora(xx, yy, ids, a_st, b_st, sc)
    ref = segmented_lora_ref(xx, yy, ids, a_st, b_st, sc)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    assert not torch.equal(out, yy) and torch.equal(out[2], yy[2]) and torch.equal(out[8], yy[8])
    assert torch.equal(yy, yy.clone())  # original untouched when inplace=False


def test_bench_cases_are_well_formed_and_runnable_on_cpu():
    cases = ops_attn.bench_cases() + ops_moe.bench_cases() + ops_util.bench_cases()
    names = {c["name"] for c in cases}
    assert {"paged_decode_attention", "prefill_attention", "lightning_indexer", "moe_grouped_gemm", "moe_route", "hadamard",
            "teal_gemv", "rmsnorm_quant", "quantize_fp8_rowwise", "topk_select"} <= names
    small = {
        "paged_decode_attention": {"batch": 2, "hq": 4, "hkv": 2, "d": 32, "ctx": 40, "kv_dtype": "fp8_e4m3", "device": "cpu"},
        "prefill_attention": {"batch": 1, "seq": 24, "hq": 4, "hkv": 2, "d": 16, "device": "cpu", "dtype": "float32"},
        "lightning_indexer": {"batch": 2, "ctx": 70, "hi": 8, "di": 16, "device": "cpu", "dtype": "float32"},
        "moe_route": {"tokens": 8, "experts": 8, "top_k": 2, "device": "cpu", "dtype": "float32"},
        "moe_grouped_gemm": {"tokens": 8, "experts": 4, "top_k": 2, "hidden": 16, "inter": 8, "device": "cpu", "dtype": "float32"},
        "hadamard": {"rows": 4, "dim": 64, "block": 16, "device": "cpu", "dtype": "float32"},
        "quantize_fp8_rowwise": {"rows": 4, "dim": 64, "device": "cpu", "dtype": "float32"},
        "rmsnorm_quant": {"rows": 4, "dim": 64, "device": "cpu", "dtype": "float32"},
        "teal_gemv": {"batch": 1, "k": 64, "n": 32, "device": "cpu", "dtype": "float32"},
        "topk_select": {"batch": 2, "t": 100, "k": 8, "device": "cpu"},
    }
    for c in cases:
        assert callable(c["make_inputs"]) and callable(c["run"]) and callable(c["ref"])
        assert c["family"] in {"attention", "sparse", "moe", "util"} and c["shapes"]
        cfg = small[c["name"]]
        inp = c["make_inputs"](cfg)
        out, ref = c["run"](inp), c["ref"](inp)
        assert out is not None and ref is not None
        assert c["bytes_moved"](cfg) > 0 and c["flops"](cfg) >= 0
