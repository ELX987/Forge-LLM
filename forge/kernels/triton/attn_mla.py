"""Triton absorbed-MLA decode attention (Plan K4 "MLA variant"; DeepSeek-style latent cache).

**Status: implemented, not yet validated on NVIDIA hardware** (runs in the Triton CPU interpreter in the unit tests).
Min Triton 2.1 / sm_70 (fp16) or sm_80 (bf16).

All ``H`` query heads share one latent cache ``c_pool [P, page_size, Dc]`` (values) with a separate rotary-key cache
``r_pool [P, page_size, Dr]``; scores are ``q_nope . c + q_rope . r`` (absorbed ``W_UK``) and the output stays in latent
space ``[B, H, Dc]`` (the ``W_UV`` up-projection is a tiny batched matmul applied afterwards by the dispatcher).  The
head index plays the role of the M dimension of the ``tl.dot`` tiles, so the latent tile is read once per ``BLOCK_H``
heads.  Latent rows may be stored in FP8-E4M3 with a per-token scale ``c_scale [P, page_size]``.
Split-KV + LSE merge as in :mod:`forge.kernels.triton.attn_decode`; ``topk_idx [B, K]`` enables DSA-style sparse decoding.
"""
from __future__ import annotations

import os

import torch

from forge.kernels.ref.attn_kv import normalize_kv_dtype
from forge.kernels.triton.attn_common import HAVE_TRITON, merge_partials_triton
from forge.kernels.triton.util_dispatch import cdiv, fp8_native, next_pow2, smem_limit_bytes

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from forge.kernels.triton.attn_common import decode_e4m3, online_softmax_update

    @triton.jit
    def _mla_stage1_kernel(
        qc_ptr, qr_ptr, c_ptr, r_ptr, cs_ptr, bt_ptr, seqlen_ptr, idx_ptr, o_part_ptr, lse_part_ptr,
        sm_scale,
        sqc_b, sqc_h, sqr_b, sqr_h,
        sc_pg, sc_t, sr_pg, sr_t, scs_pg, scs_t,
        sbt_b, si_b,
        so_b, so_h, so_s, sl_b, sl_h,
        H: tl.constexpr, DC: tl.constexpr, DR: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr,
        BLOCK_C: tl.constexpr, BLOCK_R: tl.constexpr, PAGE_SIZE: tl.constexpr, NS: tl.constexpr,
        WINDOW: tl.constexpr, FP8: tl.constexpr, FP8_NATIVE: tl.constexpr, SPARSE: tl.constexpr,
        K_SEL: tl.constexpr, LIST_CHUNK: tl.constexpr,
    ):
        b = tl.program_id(0)
        hb = tl.program_id(1)
        pid_s = tl.program_id(2)
        rows = hb * BLOCK_H + tl.arange(0, BLOCK_H)
        rv = rows < H
        offs_c = tl.arange(0, BLOCK_C)
        cm = offs_c < DC
        offs_r = tl.arange(0, BLOCK_R)
        rm = offs_r < DR
        qc = tl.load(qc_ptr + b * sqc_b + rows[:, None] * sqc_h + offs_c[None, :],
                     mask=rv[:, None] & cm[None, :], other=0.0)
        qr = tl.load(qr_ptr + b * sqr_b + rows[:, None] * sqr_h + offs_r[None, :],
                     mask=rv[:, None] & rm[None, :], other=0.0)
        seq_len = tl.load(seqlen_ptr + b)
        lo = seq_len * 0
        if WINDOW > 0:
            lo = tl.maximum(seq_len - WINDOW, 0)
        if SPARSE:
            loop_lo = pid_s * LIST_CHUNK
            loop_hi = tl.minimum(loop_lo + LIST_CHUNK, K_SEL)
        else:
            span = tl.maximum(seq_len - lo, 0)
            chunk = (((span + NS - 1) // NS + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
            loop_lo = lo + pid_s * chunk
            loop_hi = tl.minimum(loop_lo + chunk, seq_len)
        m_i = tl.full([BLOCK_H], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_H], tl.float32)
        acc = tl.zeros([BLOCK_H, BLOCK_C], tl.float32)
        qk_scale = sm_scale * 1.4426950408889634
        for start in range(loop_lo, loop_hi, BLOCK_N):
            offs = start + tl.arange(0, BLOCK_N)
            if SPARSE:
                jv = offs < loop_hi
                pos = tl.load(idx_ptr + b * si_b + offs, mask=jv, other=-1)
                tok_valid = jv & (pos >= 0) & (pos < seq_len)
                pos = tl.where(tok_valid, pos, 0)
            else:
                pos = offs
                tok_valid = offs < loop_hi
            page = tl.load(bt_ptr + b * sbt_b + pos // PAGE_SIZE, mask=tok_valid, other=0)
            slot = pos % PAGE_SIZE
            row_off = page.to(tl.int64) * sc_pg + slot.to(tl.int64) * sc_t
            craw = tl.load(c_ptr + row_off[:, None] + offs_c[None, :], mask=tok_valid[:, None] & cm[None, :], other=0)
            if FP8:
                if FP8_NATIVE:
                    c = craw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                else:
                    c = decode_e4m3(craw)
                sc = tl.load(cs_ptr + page.to(tl.int64) * scs_pg + slot.to(tl.int64) * scs_t, mask=tok_valid, other=1.0)
                c = c * sc[:, None]
            else:
                c = craw.to(tl.float32)
            c = c.to(qc.dtype)
            r_off = page.to(tl.int64) * sr_pg + slot.to(tl.int64) * sr_t
            r = tl.load(r_ptr + r_off[:, None] + offs_r[None, :], mask=tok_valid[:, None] & rm[None, :], other=0.0)
            qk = tl.dot(qc, tl.trans(c)) + tl.dot(qr, tl.trans(r.to(qr.dtype)))
            qk = qk * qk_scale
            valid = tok_valid[None, :] & rv[:, None]
            if WINDOW > 0:
                valid = valid & (pos[None, :] >= lo)
            acc, m_i, l_i = online_softmax_update(acc, m_i, l_i, qk, c, valid)
        l_safe = tl.where(l_i == 0.0, 1.0, l_i)
        o = acc / l_safe[:, None]
        lse = tl.where(l_i == 0.0, float("-inf"), (m_i + tl.log2(l_safe)) * 0.6931471805599453)
        tl.store(o_part_ptr + b * so_b + rows[:, None] * so_h + pid_s * so_s + offs_c[None, :], o,
                 mask=rv[:, None] & cm[None, :])
        tl.store(lse_part_ptr + b * sl_b + rows * sl_h + pid_s, lse, mask=rv)


def mla_decode_attention_triton(q_nope, q_rope, c_pool, r_pool, block_tables, seq_lens, *, softmax_scale: float,
                                w_uv=None, c_scale=None, kv_dtype: str = "bf16", window: int | None = None,
                                topk_idx=None, max_seq_len: int | None = None, num_splits: int | None = None):
    """Launch the MLA decode kernel; returns ``[B, H, Dc]`` (or ``[B, H, Dv]`` when ``w_uv [H, Dc, Dv]`` is given)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    kv = normalize_kv_dtype(kv_dtype)
    if kv not in ("bf16", "fp16", "fp32", "fp8_e4m3"):
        raise ValueError("MLA latent cache supports bf16/fp16/fp8_e4m3 only")
    b, h, dc = q_nope.shape
    dr = q_rope.shape[-1]
    ps = c_pool.shape[1]
    dev = q_nope.device
    q_nope = q_nope.contiguous()
    q_rope = q_rope.contiguous()
    c_view = c_pool.view(torch.uint8) if (kv == "fp8_e4m3" and c_pool.dtype != torch.uint8) else c_pool
    seq_lens = seq_lens.to(torch.int32).contiguous()
    block_tables = block_tables.to(torch.int32).contiguous()
    fp8 = kv == "fp8_e4m3"
    if fp8:
        cs = torch.as_tensor(c_scale if c_scale is not None else 1.0, dtype=torch.float32, device=dev)
        if cs.numel() == 1:
            cs = cs.reshape(1, 1).expand(c_pool.shape[0], ps)
    else:
        cs = seq_lens
    sparse = topk_idx is not None
    idx = topk_idx.to(torch.int32).contiguous() if sparse else seq_lens
    k_sel = idx.shape[1] if sparse else 0
    block_h = 32 if h >= 32 else 16
    block_c = max(16, next_pow2(dc))
    block_r = max(16, next_pow2(dr))
    block_n = 32 if dc > 256 else 64
    stages = 2
    if dc > 256 and smem_limit_bytes(dev) < 150 * 1024:  # sm_86/89/120 class (99 KiB): halve the latent tile
        block_n, stages = 16, 1
    nhb = cdiv(h, block_h)
    max_len = int(max_seq_len) if max_seq_len is not None else block_tables.shape[1] * ps
    from forge.kernels.triton.attn_decode import _pick_splits

    ns = _pick_splits(b * nhb, cdiv(k_sel, block_n) if sparse else cdiv(max_len, block_n), dev, num_splits)
    list_chunk = cdiv(cdiv(k_sel, ns), block_n) * block_n if sparse else 0
    o_part = torch.empty(b, h, ns, dc, dtype=torch.float32, device=dev)
    lse_part = torch.empty(b, h, ns, dtype=torch.float32, device=dev)
    _mla_stage1_kernel[(b, nhb, ns)](
        q_nope, q_rope, c_view, r_pool, cs, block_tables, seq_lens, idx, o_part, lse_part,
        float(softmax_scale),
        q_nope.stride(0), q_nope.stride(1), q_rope.stride(0), q_rope.stride(1),
        c_view.stride(0), c_view.stride(1), r_pool.stride(0), r_pool.stride(1),
        cs.stride(0) if fp8 else 0, cs.stride(1) if fp8 else 0,
        block_tables.stride(0), idx.stride(0) if sparse else 0,
        o_part.stride(0), o_part.stride(1), o_part.stride(2), lse_part.stride(0), lse_part.stride(1),
        H=h, DC=dc, DR=dr, BLOCK_H=block_h, BLOCK_N=block_n, BLOCK_C=block_c, BLOCK_R=block_r, PAGE_SIZE=ps, NS=ns,
        WINDOW=int(window) if window else 0, FP8=fp8, FP8_NATIVE=bool(fp8 and fp8_native(dev)), SPARSE=sparse,
        K_SEL=k_sel, LIST_CHUNK=list_chunk, num_warps=8, num_stages=stages,
    )
    out = torch.empty(b, h, dc, dtype=q_nope.dtype, device=dev)
    merge_partials_triton(o_part.view(b * h, ns, dc), lse_part.view(b * h, ns), out.view(b * h, dc))
    if w_uv is not None:
        out = torch.einsum("bhc,hcv->bhv", out.float(), w_uv.float()).to(q_nope.dtype)
    return out
