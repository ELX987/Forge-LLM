"""Block-scaled FP4 (NVFP4 / MXFP4) Triton kernels (Plan 6.4 K2) -- EXPERIMENTAL / GATED.

1. ``_fp4_dequant_gemm_kernel`` -- weight-only (and emulated W4A4) GEMM for *any* sm: FP4 payload + block scales
   are decoded in registers to the activation dtype (``val * block_scale`` is exact in bf16/fp16 for both formats),
   ``tl.dot`` runs in 16-bit, the fp32 tensor scale is applied in the epilogue.  Reads scales either logical
   ``[N, K/B]`` or in the padded 128x4 swizzled layout (index arithmetic, no unswizzle pass).
2. ``_fp4_quant_kernel`` -- fused quantiser (activations -> payload + block scales, optionally written directly in
   the swizzled layout).  Round-to-nearest-even, saturating; NVFP4 needs the fp32 tensor scale precomputed.  The E4M3
   / E8M0 scale conversions are done with integer + bit-cast arithmetic (no ``float8`` dtype, no ``exp2``), so the
   kernels run on any sm and are exactly reproducible.
   NaN / Inf inputs poison their block exactly like the oracle (scale code 0x7F / 0xFF, codes 0).
3. ``_mxfp4_dot_scaled_kernel`` -- native block-scaled MMA through ``tl.dot_scaled`` (MXFP4 operands, E8M0 scales).
   Only attempted when ``tl.dot_scaled`` exists AND the device reports block-scaled FP4 support (sm_100/120).
   The Triton API for this has changed between releases; this kernel follows the Triton 3.3-3.5 signature
   and is NOT validated on hardware.  NVFP4 (E4M3 scales, block 16) has no ``tl.dot_scaled`` mapping in released
   Triton, so NVFP4 uses the dequant kernel (documented gap).

Status: implemented; kernels 1-2 checked in the Triton CPU interpreter, kernel 3 only if the interpreter supports it;
nothing has been compiled or run on a GPU.
"""
from __future__ import annotations

from math import prod as math_prod
from typing import Any

import torch

from forge.kernels.layouts import PackedFP4
from forge.kernels.layouts.bits import roundup
from forge.kernels.layouts.fp4 import ATOM_COLS, ATOM_ROWS, padded_scale_shape

from .gemm_common import HAS_TRITON, cdiv, default_mm_config, has_dot_scaled, launch_fitting, require_triton, tl, triton

if HAS_TRITON:

    @triton.jit
    def _e4m3_decode(bits):
        """E4M3FN bit pattern (int32 0..255) -> fp32, exact; NaN pattern 0x7F/0xFF is NOT mapped to NaN."""
        e = (bits >> 3) & 0xF
        m = bits & 0x7
        normal = ((e + 120) << 23) | (m << 20)  # fp32 exponent field = e - 7 + 127
        v = tl.where(e == 0, m.to(tl.float32) * 0.001953125, normal.to(tl.float32, bitcast=True))
        return tl.where((bits & 0x80) != 0, -v, v)

    @triton.jit
    def _e4m3_encode(v):
        """fp32 v >= 0 -> E4M3FN bit pattern (int32), round-to-nearest-even, saturating at 448 (0x7E)."""
        v = tl.minimum(v, 448.0)
        bits = v.to(tl.int32, bitcast=True)
        e32 = ((bits >> 23) & 0xFF) - 127  # unbiased exponent (normal fp32)
        man = bits & 0x7FFFFF
        m3 = man >> 20
        rem = man & 0xFFFFF
        up = (rem > 0x80000) | ((rem == 0x80000) & ((m3 & 1) == 1))
        m3 = m3 + up.to(tl.int32)
        carry = m3 == 8
        m3 = tl.where(carry, 0, m3)
        e32 = e32 + carry.to(tl.int32)
        normal = tl.minimum(((e32 + 7) << 3) | m3, 0x7E)
        # subnormal range (v < 2**-6): code = round_half_even(v * 512) via the 1.5*2**23 trick
        sub = ((v * 512.0 + 12582912.0) - 12582912.0).to(tl.int32)
        code = tl.where(e32 < -6, sub, normal)
        return tl.where(v > 0.0, code, 0)

    @triton.jit
    def _e8m0_decode(bits):
        """E8M0 code (int32 0..254) -> 2**(code-127), exact."""
        return tl.where(bits == 0, 5.877471754111438e-39, (bits << 23).to(tl.float32, bitcast=True))

    @triton.jit
    def _fp4_dequant_gemm_kernel(
        x_ptr, p_ptr, s_ptr, ts_ptr, bias_ptr, y_ptr,
        M, N, K,
        stride_xm, stride_xk, stride_pn, stride_sn, N_COL_BLOCKS, stride_ym, stride_yn,
        FMT: tl.constexpr, BS: tl.constexpr, SWIZZLED: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kt in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            w_mask = k_mask[:, None] & n_mask[None, :]
            x = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :] * stride_xk,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            byte = tl.load(p_ptr + offs_n64[None, :] * stride_pn + (offs_k // 2)[:, None], mask=w_mask, other=0).to(tl.int32)
            code = (byte >> ((offs_k % 2) * 4)[:, None]) & 0xF
            mag = code & 7
            e = mag >> 1
            m1 = (mag & 1).to(tl.float32)
            val = tl.where(e == 0, m1 * 0.5, (((2 + (mag & 1)) << e).to(tl.float32)) * 0.25)  # (2+m) * 2**e / 4, exact
            val = tl.where((code >> 3) == 1, -val, val)
            kb = offs_k // BS
            if SWIZZLED:
                rb = offs_n // 128
                ri = offs_n % 128
                cb = kb // 4
                ci = kb % 4
                sidx = (rb[None, :] * N_COL_BLOCKS + cb[:, None]) * 512 + (ri % 32)[None, :] * 16 + (ri // 32)[None, :] * 4 + ci[:, None]
            else:
                sidx = offs_n64[None, :] * stride_sn + kb[:, None]
            sbits = tl.load(s_ptr + sidx, mask=w_mask, other=0).to(tl.int32)
            if FMT == 0:
                sc = _e4m3_decode(sbits)
                sc = tl.where((sbits & 0x7F) == 0x7F, float("nan"), sc)  # E4M3FN NaN code poisons the block (oracle convention)
            else:
                sc = _e8m0_decode(sbits)
                sc = tl.where(sbits == 0xFF, float("nan"), sc)  # E8M0 NaN code
            acc += tl.dot(x, (val * sc).to(x.dtype))
        acc = acc * tl.load(ts_ptr)
        if HAS_BIAS:
            acc = acc + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(y_ptr + offs_m[:, None].to(tl.int64) * stride_ym + offs_n[None, :] * stride_yn, acc,
                 mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _fp4_quant_kernel(
        x_ptr, p_ptr, s_ptr, ts_ptr, M, K, stride_xm, stride_pm, stride_sm, N_COL_BLOCKS,
        FMT: tl.constexpr, BS: tl.constexpr, SWIZZLED: tl.constexpr,
        BLOCK_M: tl.constexpr, NB: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_b = pid_k * NB + tl.arange(0, NB)  # global block index along K
        offs_j = tl.arange(0, BS // 2)
        mm = offs_m[:, None, None]
        kb = offs_b[None, :, None]
        jj = offs_j[None, None, :]
        k_even = kb * BS + 2 * jj
        mask = (mm < M) & (k_even < K)  # K is padded to a multiple of BS by the launcher
        xe = tl.load(x_ptr + mm.to(tl.int64) * stride_xm + k_even, mask=mask, other=0.0).to(tl.float32)
        xo = tl.load(x_ptr + mm.to(tl.int64) * stride_xm + k_even + 1, mask=mask, other=0.0).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(xe), axis=2), tl.max(tl.abs(xo), axis=2))  # [BLOCK_M, NB]
        if FMT == 0:
            ts = tl.load(ts_ptr)
            scode = _e4m3_encode((amax / 6.0) / ts)
            sdec = _e4m3_decode(scode) * ts
            sbits = scode.to(tl.uint8)
        else:
            ebits = (amax.to(tl.int32, bitcast=True) >> 23) & 0xFF
            scode = tl.where(amax > 0.0, tl.minimum(tl.maximum(ebits - 2, 0), 254), 127)
            sdec = _e8m0_decode(scode)
            sbits = scode.to(tl.uint8)
        # NaN / Inf poison the whole block (oracle convention): x*0 is NaN iff x is NaN or +-Inf
        poison = tl.sum(xe * 0.0, axis=2) + tl.sum(xo * 0.0, axis=2)
        bad = poison != poison
        if FMT == 0:
            sbits = tl.where(bad, 0x7F, scode).to(tl.uint8)
        else:
            sbits = tl.where(bad, 0xFF, scode).to(tl.uint8)
        valid = (sdec > 0.0) & (~bad)
        sdiv = tl.where(valid, sdec, 1.0)[:, :, None]
        ye = tl.where(valid[:, :, None], xe / sdiv, 0.0)
        yo = tl.where(valid[:, :, None], xo / sdiv, 0.0)
        ae = tl.abs(ye)
        ao = tl.abs(yo)
        ce = ((ae > 0.25).to(tl.int32) + (ae >= 0.75).to(tl.int32) + (ae > 1.25).to(tl.int32) + (ae >= 1.75).to(tl.int32)
              + (ae > 2.5).to(tl.int32) + (ae >= 3.5).to(tl.int32) + (ae > 5.0).to(tl.int32)) + ((ye < 0.0).to(tl.int32) << 3)
        co = ((ao > 0.25).to(tl.int32) + (ao >= 0.75).to(tl.int32) + (ao > 1.25).to(tl.int32) + (ao >= 1.75).to(tl.int32)
              + (ao > 2.5).to(tl.int32) + (ao >= 3.5).to(tl.int32) + (ao > 5.0).to(tl.int32)) + ((yo < 0.0).to(tl.int32) << 3)
        byte = (ce | (co << 4)).to(tl.uint8)
        tl.store(p_ptr + mm.to(tl.int64) * stride_pm + kb * (BS // 2) + jj, byte, mask=mask)
        # block scales
        sm_ = offs_m[:, None]
        sb_ = offs_b[None, :]
        smask = (sm_ < M) & (sb_ * BS < K)
        if SWIZZLED:
            sidx = ((sm_ // 128) * N_COL_BLOCKS + sb_ // 4) * 512 + ((sm_ % 128) % 32) * 16 + ((sm_ % 128) // 32) * 4 + sb_ % 4
        else:
            sidx = sm_.to(tl.int64) * stride_sm + sb_
        tl.store(s_ptr + sidx, sbits, mask=smask)


def _tensor_scale(x2: torch.Tensor, tensor_scale: float | torch.Tensor | None) -> torch.Tensor:
    if tensor_scale is None:
        xa = x2.abs().float()
        ts = torch.where(torch.isfinite(xa), xa, torch.zeros_like(xa)).amax() / (448.0 * 6.0)  # finite elements only (oracle)
    else:
        ts = torch.as_tensor(tensor_scale, dtype=torch.float32, device=x2.device).reshape(())
    return torch.where(ts > 0, ts, torch.ones_like(ts)).reshape(1).contiguous()


def quantize_act_fp4_triton(
    x: torch.Tensor,
    fmt: str = "nvfp4",
    *,
    scale_layout: str = "logical",
    tensor_scale: float | torch.Tensor | None = None,
    block_m: int = 16,
    num_warps: int = 4,
) -> PackedFP4:
    """Fused FP4 quantiser kernel -> :class:`PackedFP4` (RTN).  ``x [..., K]`` finite values."""
    require_triton()
    bs = 16 if fmt == "nvfp4" else 32
    x2 = x.reshape(-1, x.shape[-1])
    rows, k = x2.shape
    kp = roundup(k, bs)
    if kp != k:
        x2 = torch.nn.functional.pad(x2, (0, kp - k))
    x2 = x2.contiguous()
    nblk = kp // bs
    payload = torch.zeros((rows, kp // 2), dtype=torch.uint8, device=x.device)
    if scale_layout == "swizzled":
        rp, cp = padded_scale_shape(rows, nblk)
        scales = torch.zeros(rp * cp, dtype=torch.uint8, device=x.device)
        ncb, sstride = cp // ATOM_COLS, 0
    else:
        scales = torch.zeros((rows, nblk), dtype=torch.uint8, device=x.device)
        ncb, sstride = 0, scales.stride(0)
    ts = _tensor_scale(x2, tensor_scale) if fmt == "nvfp4" else torch.ones(1, dtype=torch.float32, device=x.device)
    nb = 8 if bs == 16 else 4
    if rows:
        _fp4_quant_kernel[(cdiv(rows, block_m), cdiv(nblk, nb))](
            x2, payload, scales, ts, rows, kp, x2.stride(0), payload.stride(0), sstride, ncb,
            FMT=0 if fmt == "nvfp4" else 1, BS=bs, SWIZZLED=scale_layout == "swizzled", BLOCK_M=block_m, NB=nb,
            num_warps=num_warps,
        )
    return PackedFP4(fmt, payload, scales, ts.reshape(()).clone(), rows, k, scale_layout)


def fp4_dequant_gemm_triton(
    x: torch.Tensor,
    w: PackedFP4,
    bias: torch.Tensor | None = None,
    *,
    config: dict[str, Any] | None = None,
    sm: int | None = None,
) -> torch.Tensor:
    """``Y = X @ dq(W)^T + bias`` with a 16-bit activation ``x [..., K]`` and FP4 weights (any sm)."""
    require_triton()
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    m, k = x2.shape
    if k != w.k:
        raise ValueError(f"x has K={k}, weight has K={w.k}")
    n = w.rows
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return y.reshape(*lead, n)
    cfg = dict(default_mm_config(m, n, k, sm=sm))
    cfg.update({"tile_k": min(cfg["tile_k"], 128)})
    if config:
        cfg.update({kk: v for kk, v in config.items() if v is not None})
    swz = w.scale_layout == "swizzled"
    ncb = padded_scale_shape(w.rows, w.n_blocks)[1] // ATOM_COLS if swz else 0
    exp_scales = math_prod(padded_scale_shape(w.rows, w.n_blocks)) if swz else w.rows * w.n_blocks
    if tuple(w.payload.shape) != (w.rows, w.k_pad // 2) or w.scales.numel() != exp_scales or not w.payload.is_contiguous()             or not w.scales.is_contiguous() or w.payload.dtype != torch.uint8 or w.scales.dtype != torch.uint8:
        raise ValueError("malformed PackedFP4 (payload/scales shape, dtype or contiguity); refusing to launch (would read out of bounds)")

    def _launch(c: dict[str, Any]) -> torch.Tensor:
        bm, bn, bk = int(c["tile_m"]), int(c["tile_n"]), int(c["tile_k"])
        _fp4_dequant_gemm_kernel[(cdiv(m, bm) * cdiv(n, bn),)](
            x2, w.payload, w.scales, w.tensor_scale.reshape(1), bias if bias is not None else w.tensor_scale.reshape(1), y,
            m, n, k, x2.stride(0), x2.stride(1), w.payload.stride(0), 0 if swz else w.scales.stride(0), ncb,
            y.stride(0), y.stride(1),
            FMT=0 if w.fmt == "nvfp4" else 1, BS=w.block, SWIZZLED=swz, HAS_BIAS=bias is not None,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=int(c.get("warps", 4)), num_stages=int(c.get("stages", 3)),
        )
        return y

    launch_fitting(("fp4", w.fmt, swz, str(x.dtype), sm, tuple(sorted((kk, v) for kk, v in cfg.items() if isinstance(v, (int, str))))),
                   cfg, _launch, min_tile_k=32)
    return y.reshape(*lead, n)


# ---------------------------------------------------------------------------------------------------------------
# native block-scaled MMA (guarded)
# ---------------------------------------------------------------------------------------------------------------
if HAS_TRITON and has_dot_scaled():

    @triton.jit
    def _mxfp4_dot_scaled_kernel(
        a_ptr, as_ptr, b_ptr, bs_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_asm, stride_bn, stride_bsn, stride_cm, stride_cn,
        HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kt in range(0, tl.cdiv(K, BLOCK_K)):  # K is padded to a multiple of BLOCK_K by the launcher
            offs_kp = kt * (BLOCK_K // 2) + tl.arange(0, BLOCK_K // 2)  # packed bytes along K
            offs_ks = kt * (BLOCK_K // 32) + tl.arange(0, BLOCK_K // 32)  # E8M0 scale columns
            a = tl.load(a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_kp[None, :], mask=m_mask[:, None], other=0)
            a_s = tl.load(as_ptr + offs_m[:, None].to(tl.int64) * stride_asm + offs_ks[None, :], mask=m_mask[:, None], other=127)
            b = tl.load(b_ptr + offs_n[None, :].to(tl.int64) * stride_bn + offs_kp[:, None], mask=n_mask[None, :], other=0)
            b_s = tl.load(bs_ptr + offs_n[:, None].to(tl.int64) * stride_bsn + offs_ks[None, :], mask=n_mask[:, None], other=127)
            acc = tl.dot_scaled(a, a_s, "e2m1", b, b_s, "e2m1", acc)
        if HAS_BIAS:
            acc = acc + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn, acc,
                 mask=m_mask[:, None] & n_mask[None, :])


def mxfp4_dot_scaled_available() -> bool:
    return bool(HAS_TRITON and has_dot_scaled())


def mxfp4_gemm_native_triton(
    xq: PackedFP4,
    w: PackedFP4,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    tile: tuple[int, int, int] = (128, 128, 128),
) -> torch.Tensor:
    """MXFP4 x MXFP4 GEMM through ``tl.dot_scaled`` (experimental, version guarded).  Both operands need
    ``fmt == 'mxfp4'`` and logical scales; tensor scales are 1."""
    require_triton()
    if not has_dot_scaled():
        raise RuntimeError("this Triton build has no tl.dot_scaled")
    if xq.fmt != "mxfp4" or w.fmt != "mxfp4":
        raise ValueError("native block-scaled path supports MXFP4 operands only")
    bm, bn, bk = tile
    kp = roundup(max(xq.k_pad, w.k_pad), bk)
    def pad(p: PackedFP4) -> tuple[torch.Tensor, torch.Tensor]:
        pay = torch.nn.functional.pad(p.payload, (0, kp // 2 - p.payload.shape[1]))
        sc = torch.nn.functional.pad(p.logical_scales(), (0, kp // 32 - p.n_blocks), value=127)
        return pay.contiguous(), sc.contiguous()
    a, a_s = pad(xq)
    b, b_s = pad(w)
    m, n = xq.rows, w.rows
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    zero = torch.zeros(1, dtype=torch.float32, device=a.device)
    _mxfp4_dot_scaled_kernel[(cdiv(m, bm) * cdiv(n, bn),)](
        a, a_s, b, b_s, bias if bias is not None else zero, out, m, n, kp,
        a.stride(0), a_s.stride(0), b.stride(0), b_s.stride(0), out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=8, num_stages=3,
    )
    return out


__all__ = [
    "fp4_dequant_gemm_triton", "quantize_act_fp4_triton", "mxfp4_gemm_native_triton", "mxfp4_dot_scaled_available",
    "HAS_TRITON", "ATOM_ROWS",
]
