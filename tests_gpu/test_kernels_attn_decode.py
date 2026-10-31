"""GPU tests (skipped without CUDA) of the decode / MLA / prefill / tree-attention Triton kernels vs the torch oracles.

Edge-case list of Plan 6.7 covered here: GQA group sizes 1, 2, 4, 8, 16; sequence length not a multiple of the page size;
ragged batches (including empty sequences); every KV dtype (bf16 / fp16 / fp8 / int8 / int4, symmetric + asymmetric);
sliding window; split-KV merge (1..16 splits); residual window; deterministic reduction order (bit-identical repeats).
Run on the target GPU:  ``pytest tests_gpu/test_kernels_attn_decode.py -q``.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from forge.kernels import ops_attn  # noqa: E402
from forge.kernels.ref.attn_decode import mla_decode_attention_ref, paged_decode_attention_ref  # noqa: E402
from forge.kernels.ref.attn_kv import build_paged_pools  # noqa: E402
from forge.kernels.ref.attn_prefill import prefill_attention_ref  # noqa: E402
from forge.kernels.ref.attn_spec import build_tree_mask, tree_attention_ref  # noqa: E402
from forge.kernels.triton.util_dispatch import device_sm  # noqa: E402

DEV = torch.device("cuda")
SM = device_sm(DEV) or 0
DTYPES = [torch.float16] + ([torch.bfloat16] if SM >= 80 else [])
TOL = {torch.float16: 4e-3, torch.bfloat16: 3e-2}


def _data(b, hq, hkv, d, lens, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hq, d, generator=g)
    k = torch.randn(b, hkv, max(lens), d, generator=g)
    v = torch.randn(b, hkv, max(lens), d, generator=g)
    return q, k, v


def _to_dev(p, dtype):
    out = dict(p)
    for n in ("k_pool", "v_pool"):
        out[n] = p[n].to(DEV)
        if p["kv_dtype"] in ("bf16", "fp16", "fp32"):
            out[n] = out[n].to(dtype)
    for n in ("block_tables", "seq_lens", "k_scale", "v_scale", "k_zero", "v_zero"):
        out[n] = p[n].to(DEV) if p[n] is not None else None
    return out


def _compare(out, ref, dtype, scale=1.0):
    err = (out.float().cpu() - ref.float()).abs().max().item()
    assert err < TOL[dtype] * max(1.0, float(ref.abs().max())) * scale, err


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gsize", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("d,page", [(64, 16), (128, 32), (128, 64)])
def test_decode_native_kv_gqa_groups_ragged_lengths(dtype, gsize, d, page):
    hkv = 4
    lens = [1, page - 1, page, page + 1, 3 * page + 5, 700, 0]
    q, k, v = _data(len(lens), hkv * gsize, hkv, d, lens, dtype)
    kvd = "bf16" if dtype == torch.bfloat16 else "fp16"
    p = _to_dev(build_paged_pools(k.to(dtype), v.to(dtype), lens, page, kvd), dtype)
    out = ops_attn.paged_decode_attention(q.to(DEV, dtype), p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"],
                                          kv_dtype=kvd)
    ref = paged_decode_attention_ref(q.to(dtype).float(), p["k_pool"].cpu().float(), p["v_pool"].cpu().float(),
                                     p["block_tables"].cpu(), p["seq_lens"].cpu(), kv_dtype="fp32")
    _compare(out, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kvd,asym", [("fp8_e4m3", False), ("int8", False), ("int8", True), ("int4", False), ("int4", True)])
@pytest.mark.parametrize("gsize", [1, 4, 8])
def test_decode_quantised_kv_vs_oracle(dtype, kvd, asym, gsize):
    lens = [33, 130, 17, 512]
    q, k, v = _data(4, 4 * gsize, 4, 128, lens, dtype, seed=3)
    p = _to_dev(build_paged_pools(k, v, lens, 32, kvd, asymmetric=asym), dtype)
    kw = dict(kv_dtype=kvd, k_scale=p["k_scale"], v_scale=p["v_scale"], k_zero=p["k_zero"], v_zero=p["v_zero"])
    out = ops_attn.paged_decode_attention(q.to(DEV, dtype), p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], **kw)
    kwc = {n: (t.cpu() if torch.is_tensor(t) else t) for n, t in kw.items()}
    ref = paged_decode_attention_ref(q.to(dtype).float(), p["k_pool"].cpu(), p["v_pool"].cpu(), p["block_tables"].cpu(),
                                     p["seq_lens"].cpu(), **kwc)
    _compare(out, ref, dtype)


@pytest.mark.parametrize("splits", [1, 2, 7, 16])
@pytest.mark.parametrize("window", [None, 1, 100, 4096])
def test_decode_split_kv_and_sliding_window(splits, window):
    dtype = DTYPES[-1]
    lens = [3000, 129, 5000]
    q, k, v = _data(3, 32, 8, 128, lens, dtype, seed=5)
    kvd = "bf16" if dtype == torch.bfloat16 else "fp16"
    p = _to_dev(build_paged_pools(k.to(dtype), v.to(dtype), lens, 64, kvd), dtype)
    args = (q.to(DEV, dtype), p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    out = ops_attn.paged_decode_attention(*args, kv_dtype=kvd, window=window, num_splits=splits)
    ref = paged_decode_attention_ref(q.to(dtype).float(), p["k_pool"].cpu().float(), p["v_pool"].cpu().float(),
                                     p["block_tables"].cpu(), p["seq_lens"].cpu(), kv_dtype="fp32", window=window)
    _compare(out, ref, dtype)
    again = ops_attn.paged_decode_attention(*args, kv_dtype=kvd, window=window, num_splits=splits)
    assert torch.equal(out, again), "decode must be bit-reproducible for a fixed split count"


def test_decode_residual_window_and_lse():
    dtype = DTYPES[-1]
    lens = [200, 77]
    res = [13, 0]
    q, k, v = _data(2, 16, 4, 128, lens, dtype, seed=8)
    p = _to_dev(build_paged_pools(k, v, lens, 32, "int8"), dtype)
    kr = torch.zeros(2, 4, 16, 128)
    vr = torch.zeros(2, 4, 16, 128)
    kr[0, :, :13] = k[0, :, 187:200]
    vr[0, :, :13] = v[0, :, 187:200]
    kw = dict(kv_dtype="int8", k_scale=p["k_scale"], v_scale=p["v_scale"])
    out, lse = ops_attn.paged_decode_attention(q.to(DEV, dtype), p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"],
                                               k_res=kr.to(DEV, dtype), v_res=vr.to(DEV, dtype),
                                               res_lens=torch.tensor(res, dtype=torch.int32, device=DEV), return_lse=True, **kw)
    ref, lref = paged_decode_attention_ref(q.to(dtype).float(), p["k_pool"].cpu(), p["v_pool"].cpu(), p["block_tables"].cpu(),
                                           p["seq_lens"].cpu(), kv_dtype="int8", k_scale=p["k_scale"].cpu(),
                                           v_scale=p["v_scale"].cpu(), k_res=kr.to(dtype).float(), v_res=vr.to(dtype).float(),
                                           res_lens=torch.tensor(res), return_lse=True)
    _compare(out, ref, dtype)
    assert (lse.cpu() - lref).abs().max() < 5e-2


def test_decode_long_context_32k_fp8():
    dtype = DTYPES[-1]
    lens = [32768 + 5]
    q, k, v = _data(1, 32, 8, 128, lens, dtype, seed=9)
    p = _to_dev(build_paged_pools(k, v, lens, 64, "fp8_e4m3"), dtype)
    kw = dict(kv_dtype="fp8_e4m3", k_scale=p["k_scale"], v_scale=p["v_scale"])
    out = ops_attn.paged_decode_attention(q.to(DEV, dtype), p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], **kw)
    ref = paged_decode_attention_ref(q.to(dtype).float(), p["k_pool"].cpu(), p["v_pool"].cpu(), p["block_tables"].cpu(),
                                     p["seq_lens"].cpu(), kv_dtype="fp8_e4m3", k_scale=p["k_scale"].cpu(),
                                     v_scale=p["v_scale"].cpu())
    _compare(out, ref, dtype)


@pytest.mark.parametrize("per_head", [False, True])
def test_sparse_gather_attention_vs_oracle(per_head):
    dtype = DTYPES[-1]
    lens = [4000, 300]
    q, k, v = _data(2, 16, 4, 128, lens, dtype, seed=11)
    kvd = "bf16" if dtype == torch.bfloat16 else "fp16"
    p = _to_dev(build_paged_pools(k.to(dtype), v.to(dtype), lens, 64, kvd), dtype)
    idx = torch.full((2, 4 if per_head else 1, 2048), -1, dtype=torch.int32)
    for bi, n in enumerate(lens):
        for h in range(idx.shape[1]):
            sel = torch.randperm(n)[: min(n, 2000)].sort().values
            idx[bi, h, : sel.numel()] = sel.int()
    out = ops_attn.sparse_topk_attention(q.to(DEV, dtype), p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"],
                                         idx.to(DEV), kv_dtype=kvd)
    ref = paged_decode_attention_ref(q.to(dtype).float(), p["k_pool"].cpu().float(), p["v_pool"].cpu().float(),
                                     p["block_tables"].cpu(), p["seq_lens"].cpu(), kv_dtype="fp32", topk_idx=idx)
    _compare(out, ref, dtype)


@pytest.mark.parametrize("kvd", ["bf16", "fp8_e4m3"])
@pytest.mark.parametrize("h", [16, 128])
def test_mla_decode(kvd, h):
    dtype = DTYPES[-1]
    dc, dr, ps, lens = 512, 64, 64, [1000, 77]
    g = torch.Generator().manual_seed(2)
    b = len(lens)
    c = torch.randn(b, max(lens), dc, generator=g)
    r = torch.randn(b, max(lens), dr, generator=g)
    n_pages = [(n + ps - 1) // ps for n in lens]
    c_pool = torch.zeros(sum(n_pages) + 3, ps, dc)
    r_pool = torch.zeros(sum(n_pages) + 3, ps, dr)
    bt = torch.zeros(b, max(n_pages), dtype=torch.int32)
    nxt = 1
    for bi in range(b):
        for pi in range(n_pages[bi]):
            bt[bi, pi] = nxt
            lo, hi = pi * ps, min(lens[bi], (pi + 1) * ps)
            c_pool[nxt, : hi - lo] = c[bi, lo:hi]
            r_pool[nxt, : hi - lo] = r[bi, lo:hi]
            nxt += 1
    cs = None
    if kvd == "fp8_e4m3":
        cs = c_pool.abs().amax(-1).clamp_min(1e-6) / 448.0
        c_store = (c_pool / cs.unsqueeze(-1)).to(torch.float8_e4m3fn)
    else:
        c_store = c_pool.to(dtype)
    qn = torch.randn(b, h, dc, generator=g)
    qr = torch.randn(b, h, dr, generator=g)
    sl = torch.tensor(lens, dtype=torch.int32)
    scale = 1 / (192 ** 0.5)
    out = ops_attn.mla_decode_attention(qn.to(DEV, dtype), qr.to(DEV, dtype), c_store.to(DEV), r_pool.to(DEV, dtype),
                                        bt.to(DEV), sl.to(DEV), softmax_scale=scale, kv_dtype=kvd,
                                        c_scale=cs.to(DEV) if cs is not None else None)
    ref = mla_decode_attention_ref(qn.to(dtype).float(), qr.to(dtype).float(), c_store.cpu() if kvd != "bf16" else c_store.float().cpu(),
                                   r_pool.to(dtype).float(), bt, sl, softmax_scale=scale, kv_dtype=kvd, c_scale=cs)
    _compare(out, ref, dtype, scale=2.0)


@pytest.mark.parametrize("hq,hkv,d", [(32, 8, 128), (8, 8, 64), (16, 1, 128), (32, 8, 256)])
@pytest.mark.parametrize("window", [None, 200])
def test_prefill_chunked_prefix_varlen(hq, hkv, d, window):
    dtype = DTYPES[-1]
    q_lens, k_lens = [300, 17, 1], [300, 500, 90]
    cuq = torch.tensor([0, 300, 317, 318], dtype=torch.int32)
    cuk = torch.tensor([0, 300, 800, 890], dtype=torch.int32)
    g = torch.Generator().manual_seed(4)
    q = torch.randn(318, hq, d, generator=g)
    k = torch.randn(890, hkv, d, generator=g)
    v = torch.randn(890, hkv, d, generator=g)
    out = ops_attn.prefill_attention(q.to(DEV, dtype), k.to(DEV, dtype), v.to(DEV, dtype), cu_seqlens_q=cuq.to(DEV),
                                     cu_seqlens_k=cuk.to(DEV), window=window)
    ref = prefill_attention_ref(q.to(dtype).float(), k.to(dtype).float(), v.to(dtype).float(), window=window,
                                cu_seqlens_q=cuq, cu_seqlens_k=cuk)
    _compare(out, ref, dtype)


def test_prefill_batched_layouts_match_sdpa_fallback():
    dtype = DTYPES[-1]
    q = torch.randn(2, 257, 16, 64, device=DEV, dtype=dtype)
    k = torch.randn(2, 257, 4, 64, device=DEV, dtype=dtype)
    v = torch.randn(2, 257, 4, 64, device=DEV, dtype=dtype)
    out = ops_attn.prefill_attention(q, k, v)
    ref = prefill_attention_ref(q.float().cpu(), k.float().cpu(), v.float().cpu())
    _compare(out, ref, dtype)
    bh = ops_attn.prefill_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), layout="bhsd")
    assert (bh.transpose(1, 2).float() - out.float()).abs().max() < 1e-2


def test_tree_attention_vs_oracle():
    dtype = DTYPES[-1]
    parents = torch.tensor([-1, 0, 0, 1, 1, 2, 2, 3, 4], dtype=torch.int32)
    n, ctx = parents.numel(), [300, 41]
    g = torch.Generator().manual_seed(6)
    kc = torch.randn(2, 8, 300, 128, generator=g)
    vc = torch.randn(2, 8, 300, 128, generator=g)
    p = _to_dev(build_paged_pools(kc.to(dtype), vc.to(dtype), ctx, 32, "bf16" if dtype == torch.bfloat16 else "fp16"), dtype)
    q = torch.randn(2, n, 32, 128, generator=g)
    kt = torch.randn(2, n, 8, 128, generator=g)
    vt = torch.randn(2, n, 8, 128, generator=g)
    mask = build_tree_mask(parents)
    kvd = "bf16" if dtype == torch.bfloat16 else "fp16"
    out = ops_attn.tree_attention(q.to(DEV, dtype), kt.to(DEV, dtype), vt.to(DEV, dtype), p["k_pool"], p["v_pool"],
                                  p["block_tables"], torch.tensor(ctx, dtype=torch.int32, device=DEV), mask.to(DEV),
                                  kv_dtype=kvd)
    ref = tree_attention_ref(q.to(dtype).float(), kt.to(dtype).float(), vt.to(dtype).float(), p["k_pool"].cpu().float(),
                             p["v_pool"].cpu().float(), p["block_tables"].cpu(), torch.tensor(ctx), mask, kv_dtype="fp32")
    _compare(out, ref, dtype)
