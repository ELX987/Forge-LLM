"""Shared Triton building blocks for the attention kernels (decode, MLA, sparse, tree).

Requires Triton >= 2.1 (``tl.cumsum`` / ``tl.randint4x`` / ``tl.dot(a, b, acc)``); developed against the 3.x API.
Everything here is plain-Triton (no TMA / warp-specialisation / inline PTX) so that it compiles on sm_70 ... sm_121.
This module imports without Triton installed (``HAVE_TRITON`` is False and no kernels are defined).

KV tile loaders
---------------
``load_k_tile`` / ``load_v_tile`` gather ``BLOCK_N`` token rows of one KV head from the paged pool (page id and slot are
given per token, so the tile need not be page aligned) and dequantise in registers:

* ``KV_KIND`` 0 native float, 1 FP8-E4M3, 2 INT8, 3 INT4 (two offset-binary nibbles / byte).
* FP8 is read as raw ``uint8`` and decoded either natively (``FP8_NATIVE``: sm_89+, bitcast to ``float8e4nv``) or with
  the portable integer formula of ``ref.attn_kv.decode_e4m3_bits`` (any sm).
* key scales are per (page, kv head, channel), value scales per (page, kv head, token); optional additive zero points.
"""
from __future__ import annotations

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453

try:  # pragma: no cover - exercised only where triton is installed
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAVE_TRITON = False

if HAVE_TRITON:

    @triton.jit
    def decode_e4m3(b):
        """Portable E4M3FN byte -> float32 (see ``ref.attn_kv.decode_e4m3_bits``)."""
        b32 = b.to(tl.int32)
        sign = (b32 >> 7) & 1
        e = (b32 >> 3) & 15
        m = b32 & 7
        mf = m.to(tl.float32)
        sub = mf * 0.001953125  # 2^-9
        nrm = (1.0 + mf * 0.125) * tl.exp2((e - 7).to(tl.float32))
        val = tl.where(e == 0, sub, nrm)
        val = tl.where((e == 15) & (m == 7), float("nan"), val)
        return tl.where(sign == 1, -val, val)

    @triton.jit
    def load_kv_rows(base_ptr, row_off, offs_d, mask2d, stride_d, KV_KIND: tl.constexpr, FP8_NATIVE: tl.constexpr):
        """Load ``[N, BLOCK_D]`` raw KV rows (``row_off`` int64 element offsets of the row starts) as float32 values
        *before* scale application."""
        if KV_KIND == 3:
            ptrs = base_ptr + row_off[:, None] + (offs_d // 2)[None, :] * stride_d
            byte = tl.load(ptrs, mask=mask2d, other=0).to(tl.int32)
            shift = ((offs_d % 2) * 4)[None, :]
            nib = (byte >> shift) & 15
            val = (nib - 8).to(tl.float32)
        else:
            ptrs = base_ptr + row_off[:, None] + offs_d[None, :] * stride_d
            raw = tl.load(ptrs, mask=mask2d, other=0)
            if KV_KIND == 1:
                if FP8_NATIVE:
                    val = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
                else:
                    val = decode_e4m3(raw)
            else:
                val = raw.to(tl.float32)
        return val

    @triton.jit
    def load_k_tile(k_ptr, ks_ptr, kz_ptr, page, slot, row_mask, hkv, offs_d, d_mask,
                    sk_pg, sk_h, sk_t, sk_d, ss_pg, ss_h, ss_d,
                    KV_KIND: tl.constexpr, HAS_ZERO: tl.constexpr, FP8_NATIVE: tl.constexpr):
        """Dequantised key tile ``[N, BLOCK_D]`` (float32)."""
        mask2d = row_mask[:, None] & d_mask[None, :]
        row_off = page.to(tl.int64) * sk_pg + hkv * sk_h + slot.to(tl.int64) * sk_t
        x = load_kv_rows(k_ptr, row_off, offs_d, mask2d, sk_d, KV_KIND, FP8_NATIVE)
        if KV_KIND != 0:
            s_off = page[:, None].to(tl.int64) * ss_pg + hkv * ss_h + offs_d[None, :] * ss_d
            s = tl.load(ks_ptr + s_off, mask=mask2d, other=1.0)
            x = x * s
            if HAS_ZERO:
                z = tl.load(kz_ptr + s_off, mask=mask2d, other=0.0)
                x = x + z
        return x

    @triton.jit
    def load_v_tile(v_ptr, vs_ptr, vz_ptr, page, slot, row_mask, hkv, offs_d, d_mask,
                    sv_pg, sv_h, sv_t, sv_d, ss_pg, ss_h, ss_t,
                    KV_KIND: tl.constexpr, HAS_ZERO: tl.constexpr, FP8_NATIVE: tl.constexpr):
        """Dequantised value tile ``[N, BLOCK_D]`` (float32)."""
        mask2d = row_mask[:, None] & d_mask[None, :]
        row_off = page.to(tl.int64) * sv_pg + hkv * sv_h + slot.to(tl.int64) * sv_t
        x = load_kv_rows(v_ptr, row_off, offs_d, mask2d, sv_d, KV_KIND, FP8_NATIVE)
        if KV_KIND != 0:
            s_off = page.to(tl.int64) * ss_pg + hkv * ss_h + slot.to(tl.int64) * ss_t
            s = tl.load(vs_ptr + s_off, mask=row_mask, other=1.0)
            x = x * s[:, None]
            if HAS_ZERO:
                z = tl.load(vz_ptr + s_off, mask=row_mask, other=0.0)
                x = x + z[:, None]
        return x

    @triton.jit
    def online_softmax_update(acc, m_i, l_i, qk, v, valid):
        """One flash-attention step in the log2 domain.

        ``qk`` is already multiplied by ``sm_scale * log2(e)``; ``valid [M, N]`` masks keys.  Rows that have not seen a
        valid key yet keep ``m_i = -inf`` / ``l_i = 0`` / ``acc = 0`` (no NaNs from ``-inf - -inf``).
        """
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp2(qk - m_safe[:, None])
        alpha = tl.exp2(m_i - m_safe)
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        return acc, m_new, l_new

    @triton.jit
    def philox_uniform(seed, ctr):
        """Philox-4x32-10 uniform in [0, 1) for a ``uint32`` counter block (same formula as ``ref.util_rng``)."""
        r0, r1, r2, r3 = tl.randint4x(seed, ctr)
        r = r0.to(tl.uint32, bitcast=True)
        return (r >> 8).to(tl.float32) * (1.0 / 16777216.0)

    @triton.jit
    def _merge_partials_kernel(o_part_ptr, lse_part_ptr, out_ptr, lse_out_ptr,
                               NSP: tl.constexpr, NSP_PAD: tl.constexpr, D: tl.constexpr, BLOCK_D: tl.constexpr,
                               STORE_LSE: tl.constexpr):
        """Split-KV log-sum-exp merge. ``o_part [F, NSP, D]`` (each split already normalised), ``lse_part [F, NSP]``
        (natural log, ``-inf`` for empty splits) -> ``out [F, D]``; one program per flattened (batch, node, head)."""
        f = tl.program_id(0)
        offs_s = tl.arange(0, NSP_PAD)
        sv = offs_s < NSP
        offs_d = tl.arange(0, BLOCK_D)
        dm = offs_d < D
        lse = tl.load(lse_part_ptr + f * NSP + offs_s, mask=sv, other=float("-inf"))
        m = tl.max(lse, axis=0)
        m_safe = tl.where(m == float("-inf"), 0.0, m)
        w = tl.exp2((lse - m_safe) * 1.4426950408889634)
        z = tl.sum(w, axis=0)
        o = tl.load(o_part_ptr + f * NSP * D + offs_s[:, None] * D + offs_d[None, :],
                    mask=sv[:, None] & dm[None, :], other=0.0)
        out = tl.sum(o * w[:, None], axis=0) / tl.where(z == 0.0, 1.0, z)
        tl.store(out_ptr + f * D + offs_d, out.to(out_ptr.dtype.element_ty), mask=dm)
        if STORE_LSE:
            lse_tot = tl.where(z == 0.0, float("-inf"), m_safe + tl.log(tl.where(z == 0.0, 1.0, z)))
            tl.store(lse_out_ptr + f, lse_tot)


def merge_partials_triton(o_part, lse_part, out, lse_out=None):
    """Launch the merge kernel. ``o_part [F, NSP, D]`` fp32 contiguous, ``lse_part [F, NSP]`` fp32 contiguous,
    ``out [F, D]`` (any float dtype, contiguous)."""
    from forge.kernels.triton.util_dispatch import next_pow2

    f, nsp, d = o_part.shape
    _merge_partials_kernel[(f,)](
        o_part, lse_part, out, lse_out if lse_out is not None else lse_part,
        NSP=nsp, NSP_PAD=max(2, next_pow2(nsp)), D=d, BLOCK_D=max(16, next_pow2(d)),
        STORE_LSE=lse_out is not None, num_warps=4)
    return out
