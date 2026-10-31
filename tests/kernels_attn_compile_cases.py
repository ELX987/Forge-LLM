"""Launch scenarios used by the offline-compile test: each function calls our launchers with small CPU tensors while launches are
being recorded (see ``kernels_attn_compile_harness``).  Nothing here executes a kernel."""
from __future__ import annotations

import contextlib

import torch

from forge.kernels.ref.attn_kv import build_paged_pools
from forge.kernels.ref.moe_ref import moe_permute_ref, pack_w4_experts


@contextlib.contextmanager
def native_fp8(*modules):
    """Pretend the device converts FP8 natively (sm_89+) so that the native code paths are compiled too."""
    saved = [(m, m.fp8_native) for m in modules]
    for m in modules:
        m.fp8_native = lambda device: True
    try:
        yield
    finally:
        for m, f in saved:
            m.fp8_native = f


def decode_cases(dtype: torch.dtype):
    from forge.kernels.triton import attn_decode as ad
    from forge.kernels.triton import attn_spec as asp

    b, hkv, d, ps = 2, 2, 64, 16
    lens = [40, 21]
    g = torch.Generator().manual_seed(0)
    k = torch.randn(b, hkv, 40, d, generator=g)
    v = torch.randn(b, hkv, 40, d, generator=g)
    for gs in (1, 4):
        q = torch.randn(b, hkv * gs, d, generator=g).to(dtype)
        for kvd in ("bf16", "fp8_e4m3", "int8", "int4"):
            p = build_paged_pools(k, v, lens, ps, kvd, asymmetric=(kvd in ("int8", "int4")))
            pk = p["k_pool"].to(dtype) if kvd == "bf16" else p["k_pool"]
            pv = p["v_pool"].to(dtype) if kvd == "bf16" else p["v_pool"]
            ad.paged_decode_attention_triton(q, pk, pv, p["block_tables"], p["seq_lens"], kv_dtype=kvd, k_scale=p["k_scale"],
                                             v_scale=p["v_scale"], k_zero=p["k_zero"], v_zero=p["v_zero"], num_splits=2)
    # production-sized heads: D = 128 / 256, GQA 8 / 16, large pages
    for d2, gs2, ps2, kvd2 in ((128, 8, 64, "fp8_e4m3"), (128, 16, 128, "int4"), (256, 4, 16, "bf16"), (256, 8, 32, "int8")):
        k2 = torch.randn(1, 2, 70, d2)
        p2 = build_paged_pools(k2.to(dtype) if kvd2 == "bf16" else k2, k2.to(dtype) if kvd2 == "bf16" else k2, [70], ps2, kvd2)
        ad.paged_decode_attention_triton(torch.randn(1, 2 * gs2, d2).to(dtype), p2["k_pool"], p2["v_pool"], p2["block_tables"],
                                         p2["seq_lens"], kv_dtype=kvd2, k_scale=p2["k_scale"], v_scale=p2["v_scale"],
                                         num_splits=4)
    # window + residual + lse
    p = build_paged_pools(k, v, lens, ps, "int8")
    kr = torch.zeros(b, hkv, 16, d).to(dtype)
    ad.paged_decode_attention_triton(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"], kv_dtype="int8",
                                     k_scale=p["k_scale"], v_scale=p["v_scale"], window=8, k_res=kr, v_res=kr,
                                     res_lens=torch.tensor([3, 0]), return_lse=True, num_splits=4)
    # sparse gather (shared + per-head selection)
    pb = build_paged_pools(k.to(dtype), v.to(dtype), lens, ps, "bf16")
    for ih in (1, hkv):
        idx = torch.zeros(b, ih, 16, dtype=torch.int32)
        ad.paged_decode_attention_triton(q, pb["k_pool"], pb["v_pool"], pb["block_tables"], pb["seq_lens"], kv_dtype="bf16",
                                         topk_idx=idx, num_splits=2)
    # tree attention (NQ > 1) with residual window
    parents = torch.tensor([-1, 0, 0, 1, 1, 2], dtype=torch.int32)
    from forge.kernels.ref.attn_spec import build_tree_mask

    n = parents.numel()
    qt = torch.randn(b, n, hkv * 4, d).to(dtype)
    kt = torch.randn(b, n, hkv, d).to(dtype)
    asp.tree_attention_triton(qt, kt, kt, pb["k_pool"], pb["v_pool"], pb["block_tables"], pb["seq_lens"],
                              build_tree_mask(parents), kv_dtype="bf16", window=16, k_res=kr, v_res=kr,
                              res_lens=torch.tensor([3, 0]), num_splits=2)


def mla_cases(dtype: torch.dtype):
    from forge.kernels.triton import attn_mla as am

    b, h, dc, dr, ps = 2, 16, 128, 64, 16
    c = torch.randn(6, ps, dc).to(dtype)
    r = torch.randn(6, ps, dr).to(dtype)
    bt = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    sl = torch.tensor([40, 20], dtype=torch.int32)
    qn, qr = torch.randn(b, h, dc).to(dtype), torch.randn(b, h, dr).to(dtype)
    am.mla_decode_attention_triton(qn, qr, c, r, bt, sl, softmax_scale=0.1, window=9, num_splits=2)
    am.mla_decode_attention_triton(qn, qr, c, r, bt, sl, softmax_scale=0.1, topk_idx=torch.zeros(b, 8, dtype=torch.int32),
                                   num_splits=2)
    c8 = torch.randn(6, ps, dc).to(torch.float8_e4m3fn)
    am.mla_decode_attention_triton(qn, qr, c8, r, bt, sl, softmax_scale=0.1, c_scale=torch.ones(6, ps), kv_dtype="fp8_e4m3")
    am.mla_decode_attention_triton(torch.randn(1, 128, 512).to(dtype), torch.randn(1, 128, 64).to(dtype),
                                   torch.randn(2, 64, 512).to(torch.float8_e4m3fn), torch.randn(2, 64, 64).to(dtype),
                                   torch.tensor([[0, 1]], dtype=torch.int32), torch.tensor([100], dtype=torch.int32),
                                   softmax_scale=0.1, c_scale=torch.ones(2, 64), kv_dtype="fp8_e4m3")
    am.mla_decode_attention_triton(torch.randn(1, 64, 512).to(dtype), torch.randn(1, 64, 64).to(dtype),
                                   torch.randn(2, 64, 512).to(dtype), torch.randn(2, 64, 64).to(dtype),
                                   torch.tensor([[0, 1]], dtype=torch.int32), torch.tensor([100], dtype=torch.int32),
                                   softmax_scale=0.1)


def prefill_cases(dtype: torch.dtype):
    from forge.kernels.triton import attn_prefill as ap

    for hq, hkv, d in ((8, 2, 64), (4, 4, 128), (4, 1, 256)):
        q = torch.randn(10, hq, d).to(dtype)
        k = torch.randn(20, hkv, d).to(dtype)
        cuq = torch.tensor([0, 4, 10], dtype=torch.int32)
        cuk = torch.tensor([0, 9, 20], dtype=torch.int32)
        ap.prefill_attention_varlen_triton(q, k, k, cuq, cuk, max_seqlen_q=6)
    ap.prefill_attention_varlen_triton(q, k, k, cuq, cuk, max_seqlen_q=6, window=5)
    ap.prefill_attention_varlen_triton(q, k, k, cuq, cuk, max_seqlen_q=6, causal=False)


def sparse_spec_cases(dtype: torch.dtype):
    from forge.kernels.triton import attn_sparse as sp
    from forge.kernels.triton import attn_spec as asp

    iq, iw = torch.randn(2, 64, 128).to(dtype), torch.rand(2, 64)
    bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    sl = torch.tensor([100, 50], dtype=torch.int32)
    sp.indexer_scores_triton(iq, iw, torch.randn(4, 64, 128).to(dtype), None, bt, sl)
    sp.indexer_scores_triton(iq, iw, torch.randn(4, 64, 128).to(torch.float8_e4m3fn), torch.ones(4, 64), bt, sl)
    sp.indexer_scores_triton(iq, iw, torch.randn(4, 64, 128).to(torch.float8_e4m3fn).view(torch.uint8), torch.ones(4, 64), bt, sl)
    sp.topk_select_triton(torch.randn(2, 5000), 64, torch.tensor([5000, 100], dtype=torch.int32))
    sp.topk_select_triton(torch.randn(1, 20000), 2048)
    draft = torch.randint(0, 100, (4, 3))
    for pdt in (torch.float32, dtype):
        asp.rejection_verify_triton(draft, torch.rand(4, 3, 100).to(pdt), torch.rand(4, 4, 100).to(pdt), seed=5, offset=2)
    asp.rejection_verify_triton(draft, torch.rand(4, 3, 50000), torch.rand(4, 4, 50000), seed=2 ** 40, offset=0)
    kp = torch.randn(8, 2, 16, 64).to(dtype)
    asp.kv_compact_triton(kp, kp.clone(), torch.zeros(2, 4, dtype=torch.int32), torch.tensor([5, 6]),
                          torch.zeros(2, 8, dtype=torch.int32), torch.tensor([3, 4]))


def moe_cases(dtype: torch.dtype):
    from forge.kernels.triton import moe_kernels as mk

    for e, k in ((8, 2), (64, 8), (256, 8)):
        mk.moe_route_triton(torch.randn(10, e).to(dtype), k)
    mk.moe_route_triton(torch.randn(10, 16), 4, scoring="sigmoid", score_bias=torch.randn(16), norm_topk_prob=False,
                        routed_scaling=2.5)
    t, e, k, h, n = 24, 6, 2, 64, 48
    ids = torch.randint(0, e, (t, k), dtype=torch.int32)
    pr = moe_permute_ref(ids, e)
    x = torch.randn(t, h).to(dtype)
    w = torch.randn(e, n, h).to(dtype)
    mk.moe_grouped_gemm_triton(x, w, pr.expert_offsets, gather_idx=pr.token_idx, total_rows=t * k)
    mk.moe_grouped_gemm_triton(x[: t * k // 2].repeat(2, 1), w, pr.expert_offsets, total_rows=t * k, block_m=64)
    big = torch.randn(e, 256, 512).to(dtype)
    mk.moe_grouped_gemm_triton(torch.randn(t * k, 512).to(dtype), big, pr.expert_offsets, total_rows=t * k, block_m=128,
                               block_n=128, block_k=64)
    mk.moe_grouped_gemm_triton(x, w, pr.expert_offsets, gather_idx=pr.token_idx, out_idx=pr.token_idx,
                               out=torch.zeros(t * k, n).to(dtype), accumulate=True, alpha=torch.ones(e), total_rows=t * k)
    q, s, z = pack_w4_experts(torch.randn(e, n, h), 32, sym=False)
    mk.moe_grouped_gemm_triton(x, q, pr.expert_offsets, gather_idx=pr.token_idx, w_scales=s.to(dtype), w_zeros=z,
                               group_size=32, total_rows=t * k)
    q, s, _ = pack_w4_experts(torch.randn(e, n, h), 32, sym=True)
    mk.moe_grouped_gemm_triton(x, q, pr.expert_offsets, w_scales=s.to(dtype), group_size=32, total_rows=t * k)
    mk.moe_combine_triton(torch.randn(t * k, h).to(dtype), pr.inv_perm, torch.rand(t, k))


def util_cases(dtype: torch.dtype):
    from forge.kernels.triton import util_gemv as ug
    from forge.kernels.triton import util_hadamard as uh
    from forge.kernels.triton import util_quant as uq
    from forge.kernels.triton import util_rope as ur

    for blk in (16, 32, 64, 128):
        uh.hadamard_triton(torch.randn(5, 256).to(dtype), blk)
    uh.hadamard_triton(torch.randn(5, 256).to(dtype), 64, signs=torch.ones(256))
    x = torch.randn(4, 256).to(dtype)
    uq.quantize_rowwise_triton(x, "int8")
    uq.quantize_rowwise_triton(x, "int8", 64)
    uq.quantize_rowwise_triton(x, "fp8")
    w = torch.rand(256)
    uq.rmsnorm_quant_triton(x, w, 1e-6, "int8")
    uq.rmsnorm_quant_triton(x, w, 1e-6, "fp8", residual=x, return_norm=True)
    uq.rmsnorm_quant_triton(torch.randn(2, 5120).to(dtype), torch.rand(5120), 1e-6, "int8")
    for fmt in ("int8", "int4", "fp4"):
        uq.stochastic_round_triton(torch.randn(4, 64), torch.ones(4, 4), 16, fmt, 7, 3)
    t, hq, hkv, d, ps = 5, 4, 2, 64, 16
    q, k, v = (torch.randn(t, hq, d).to(dtype), torch.randn(t, hkv, d).to(dtype), torch.randn(t, hkv, d).to(dtype))
    pos = torch.arange(t)
    cache = torch.randn(32, d)
    slots = torch.arange(t, dtype=torch.int32)
    kp = torch.zeros(2, hkv, ps, d, dtype=dtype)
    ur.rope_and_kv_write_triton(q, k, v, pos, cache, slots, kp, kp.clone(), kv_dtype="bf16")
    ur.rope_and_kv_write_triton(q, k, v, pos, cache[:, :32], slots, kp, kp.clone(), kv_dtype="bf16", rotary_dim=32,
                                interleaved=True)
    vs = torch.zeros(2, hkv, ps)
    ur.rope_and_kv_write_triton(q, k, v, pos, cache, slots, torch.zeros(2, hkv, ps, d, dtype=torch.int8),
                                torch.zeros(2, hkv, ps, d, dtype=torch.int8), kv_dtype="int8", k_scale=torch.ones(hkv, d),
                                v_scale_pool=vs)
    ur.rope_and_kv_write_triton(q, k, v, pos, cache[:, :32], slots, torch.zeros(2, hkv, ps, d, dtype=torch.float8_e4m3fn),
                                torch.zeros(2, hkv, ps, d, dtype=torch.float8_e4m3fn), kv_dtype="fp8_e4m3", k_scale=0.1,
                                v_scale_pool=vs, rotary_dim=32)
    ug.teal_gemv_triton(torch.randn(2, 128).to(dtype), torch.randn(128, 64).to(dtype), 0.3)
    ug.teal_gemv_triton(torch.randn(1, 128).to(dtype), torch.randn(128, 64).to(dtype), 0.3, bias=torch.randn(64).to(dtype))


def all_groups():
    """name -> callable(dtype) ; ``native_fp8`` is applied by the test for the groups that have FP8 decode paths."""
    return {
        "decode": decode_cases, "mla": mla_cases, "prefill": prefill_cases, "sparse_spec": sparse_spec_cases,
        "moe": moe_cases, "util": util_cases,
    }
