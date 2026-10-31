"""FP8 (E4M3) GEMM and fused activation-quantisation kernels in Triton (Plan 6.4 K3, C-044).

GEMM
    ``Y[M, N] = (Xq[M, K] * sx) @ (Wq[N, K] * sw)^T + bias`` with fp32 accumulation and ``tl.dot`` on fp8 operands
    (native FP8 tensor cores: sm_89 Ada, sm_90 Hopper, sm_100/120 Blackwell; Triton >= 3.0).
    Scale modes:  0 = per-tensor (scalar, scalar)   1 = per-token x per-channel ([M], [N])
                  2 = block: activations 1xB ``[M, K/B]``, weights BxB ``[N/B, K/B]`` (B = 128), applied per K tile
    Modes 0/1 apply the scales once in the epilogue; mode 2 multiplies every K-tile partial product.  ``tile_k``
    must divide ``B`` in mode 2 so a tile never straddles two scale blocks.
    Reduction order is fixed (no split-K) => bitwise reproducible run to run for a given config.
    Hopper (sm_90): Triton's default for fp8 ``tl.dot`` is *imprecise* accumulation over the whole K loop (the wgmma
    accumulator keeps ~14 mantissa bits).  Modes 0/1 therefore pass ``max_num_imprecise_acc=min(128, BLOCK_K)`` so the
    partial sum is promoted into the fp32 accumulator every <=128 K elements (DeepSeek-V3 style); mode 2 already
    starts every K tile from a fresh accumulator.

Fused activation quantisation (``rmsnorm + absmax -> fp8``)
    ``act_quant_rowwise``  one program per row: (optional RMSNorm) pass 1 sum of squares, pass 2 absmax of the
    normalised row, pass 3 write ``q = clamp(y / scale)`` and ``scale = amax / 448``.  The normalised activation never
    round-trips HBM in bf16 (P5 section 11.3).   ``act_quant_blockwise`` does the 1xB block scales,
    ``act_quant_static`` a given scalar scale.

Status: implemented; logic verified with the Triton CPU interpreter where it supports fp8; NOT compiled or
benchmarked on a GPU.  fp32 division uses Triton's default ``/`` (not IEEE-rounded), so quantised codes may differ
from the torch oracle by one code at exact rounding boundaries (documented tolerance in the GPU tests).
"""
from __future__ import annotations

from typing import Any

import torch

from .gemm_common import HAS_TRITON, cdiv, default_mm_config, launch_fitting, next_pow2, require_triton, tl, triton

FP8_E4M3_MAX = 448.0

if HAS_TRITON:

    @triton.jit
    def _fp8_gemm_kernel(
        a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        stride_sam, stride_sak, stride_sbn, stride_sbk,
        SCALE_MODE: tl.constexpr, SCALE_BLOCK: tl.constexpr, SB_DIV: tl.constexpr, IMPRECISE: tl.constexpr, HAS_BIAS: tl.constexpr,
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

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for kt in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak,
                        mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptr + offs_n[None, :].to(tl.int64) * stride_bn + offs_k[:, None] * stride_bk,
                        mask=n_mask[None, :] & k_mask[:, None], other=0.0)
            if SCALE_MODE == 2:
                kb = (kt * BLOCK_K) // SCALE_BLOCK
                sa = tl.load(sa_ptr + offs_m * stride_sam + kb * stride_sak, mask=m_mask, other=0.0)
                sb = tl.load(sb_ptr + (offs_n // SB_DIV) * stride_sbn + kb * stride_sbk, mask=n_mask, other=0.0)
                acc += tl.dot(a, b) * sa[:, None] * sb[None, :]
            elif IMPRECISE > 0:
                acc = tl.dot(a, b, acc, max_num_imprecise_acc=IMPRECISE)
            else:
                acc = tl.dot(a, b, acc)

        if SCALE_MODE == 0:
            acc = acc * (tl.load(sa_ptr) * tl.load(sb_ptr))
        elif SCALE_MODE == 1:
            sa = tl.load(sa_ptr + offs_m * stride_sam, mask=m_mask, other=0.0)
            sb = tl.load(sb_ptr + offs_n * stride_sbn, mask=n_mask, other=0.0)
            acc = acc * sa[:, None] * sb[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]
        tl.store(c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn, acc,
                 mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _act_quant_rowwise_kernel(
        x_ptr, w_ptr, q_ptr, s_ptr, K, eps, stride_xm, stride_qm,
        FMAX: tl.constexpr, RMS: tl.constexpr, HAS_W: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        xrow = x_ptr + row * stride_xm
        offs0 = tl.arange(0, BLOCK)
        rstd = 1.0
        if RMS:
            ssv = tl.zeros((BLOCK,), dtype=tl.float32)
            for start in range(0, K, BLOCK):
                offs = start + offs0
                xv = tl.load(xrow + offs, mask=offs < K, other=0.0).to(tl.float32)
                ssv += xv * xv
            rstd = tl.rsqrt(tl.sum(ssv, axis=0) / K + eps)
        amaxv = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(0, K, BLOCK):
            offs = start + offs0
            xv = tl.load(xrow + offs, mask=offs < K, other=0.0).to(tl.float32)
            if RMS:
                xv = xv * rstd
            if HAS_W:
                xv = xv * tl.load(w_ptr + offs, mask=offs < K, other=0.0).to(tl.float32)
            amaxv = tl.maximum(amaxv, tl.abs(xv))
        amax = tl.max(amaxv, axis=0)
        scale = tl.where(amax > 0.0, amax / FMAX, 1.0)
        tl.store(s_ptr + row, scale)
        for start in range(0, K, BLOCK):
            offs = start + offs0
            xv = tl.load(xrow + offs, mask=offs < K, other=0.0).to(tl.float32)
            if RMS:
                xv = xv * rstd
            if HAS_W:
                xv = xv * tl.load(w_ptr + offs, mask=offs < K, other=0.0).to(tl.float32)
            qv = tl.minimum(tl.maximum(xv / scale, -FMAX), FMAX)
            tl.store(q_ptr + row * stride_qm + offs, qv.to(tl.float8e4nv), mask=offs < K)

    @triton.jit
    def _act_quant_blockwise_kernel(
        x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, stride_sm,
        FMAX: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        kb = tl.program_id(1)
        offs = kb * BLOCK + tl.arange(0, BLOCK)
        mask = offs < K
        xv = tl.load(x_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
        amax = tl.max(tl.abs(xv), axis=0)
        scale = tl.where(amax > 0.0, amax / FMAX, 1.0)
        tl.store(s_ptr + row * stride_sm + kb, scale)
        qv = tl.minimum(tl.maximum(xv / scale, -FMAX), FMAX)
        tl.store(q_ptr + row * stride_qm + offs, qv.to(tl.float8e4nv), mask=mask)

    @triton.jit
    def _act_quant_static_kernel(
        x_ptr, q_ptr, s_ptr, numel, FMAX: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        xv = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        qv = tl.minimum(tl.maximum(xv / tl.load(s_ptr), -FMAX), FMAX)
        tl.store(q_ptr + offs, qv.to(tl.float8e4nv), mask=mask)


def prepare_fp8_scales(sx: torch.Tensor, sw: torch.Tensor, m: int, n: int, k: int, block: int):
    """Normalise the scale tensors of any granularity combination for :func:`_fp8_gemm_kernel`.

    Returns ``(mode, sa, sb, strides, sb_div)`` where ``strides = (sa_m, sa_k, sb_n, sb_k)`` are *element* strides
    (0 = broadcast) so that every combination -- per-tensor / per-token / per-channel / 1xB blocks / BxB blocks, mixed
    freely between activations and weights -- is addressed in bounds.  ``sb_div`` is the divisor applied to the
    weight row index in block mode (``block`` for BxB weight scales, 1 for per-row weight scales).
    Mismatched sizes raise ``ValueError`` (the dispatcher then falls back) instead of reading out of bounds.
    """
    sxf, swf = sx.float().contiguous(), sw.float().contiguous()
    kb = -(-k // block)
    if sxf.dim() <= 1 and swf.dim() <= 1:
        if sxf.numel() == 1 and swf.numel() == 1:
            return 0, sxf.reshape(1), swf.reshape(1), (0, 0, 0, 0), block
        if sxf.numel() not in (1, m) or swf.numel() not in (1, n):
            raise ValueError(f"scale sizes {sxf.numel()}/{swf.numel()} do not match M={m}/N={n}")
        return 1, sxf.reshape(-1), swf.reshape(-1), (0 if sxf.numel() == 1 else 1, 0, 0 if swf.numel() == 1 else 1, 0), block
    # block mode
    if sxf.dim() == 2:
        if sxf.shape[0] != m or sxf.shape[1] < kb:
            raise ValueError(f"2-D activation scale must be [M={m}, >= {kb}], got {tuple(sxf.shape)}")
        sa, sam, sak = sxf, sxf.stride(0), sxf.stride(1)
    elif sxf.numel() in (1, m):
        sa, sam, sak = sxf.reshape(-1), (0 if sxf.numel() == 1 else 1), 0
    else:
        raise ValueError(f"activation scale has {sxf.numel()} entries for M={m}")
    if swf.dim() == 2:
        if swf.shape[1] < kb:
            raise ValueError(f"2-D weight scale needs >= {kb} columns, got {tuple(swf.shape)}")
        if swf.shape[0] == n:  # per-row (1xB) weight blocks, same convention as expand_scale
            sb_div = 1
        elif swf.shape[0] >= -(-n // block):
            sb_div = block
        else:
            raise ValueError(f"2-D weight scale has {swf.shape[0]} rows for N={n}, block={block}")
        sb, sbn, sbk = swf, swf.stride(0), swf.stride(1)
    elif swf.numel() in (1, n):
        sb, sbn, sbk, sb_div = swf.reshape(-1), (0 if swf.numel() == 1 else 1), 0, 1
    else:
        raise ValueError(f"weight scale has {swf.numel()} entries for N={n}")
    return 2, sa, sb, (sam, sak, sbn, sbk), sb_div


def fp8_gemm_triton(
    xq: torch.Tensor,
    sx: torch.Tensor,
    wq: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    block: int = 128,
    config: dict[str, Any] | None = None,
    sm: int | None = None,
) -> torch.Tensor:
    """FP8 GEMM launcher.  ``xq [..., K]`` / ``wq [N, K]`` are ``torch.float8_e4m3fn``; scale conventions of
    :mod:`forge.kernels.ref.gemm_fp8`.  K is zero-padded to a multiple of 16 when needed (zero columns are exact)."""
    require_triton()
    if xq.dtype != torch.float8_e4m3fn or wq.dtype != torch.float8_e4m3fn:
        raise ValueError("Triton FP8 GEMM supports float8_e4m3fn operands only (use the fallback for e5m2)")
    lead = xq.shape[:-1]
    a = xq.reshape(-1, xq.shape[-1])
    m, k = a.shape
    n = wq.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=xq.device)
    if m == 0:
        return out.reshape(*lead, n)
    mode, sa, sb, strides, sb_div = prepare_fp8_scales(sx, sw, m, n, k, block)
    cfg = dict(default_mm_config(m, n, k, sm=sm))
    if config:
        cfg.update({kk: v for kk, v in config.items() if v is not None})
    k_pad = -(-k // 16) * 16
    if k_pad != k:
        a = torch.nn.functional.pad(a.view(torch.uint8), (0, k_pad - k)).view(torch.float8_e4m3fn)
        wq = torch.nn.functional.pad(wq.view(torch.uint8), (0, k_pad - k)).view(torch.float8_e4m3fn)
    a, wq = a.contiguous(), wq.contiguous()
    zero = torch.zeros(1, dtype=torch.float32, device=xq.device)

    def _launch(c: dict[str, Any]) -> torch.Tensor:
        bm, bn, bk = int(c["tile_m"]), int(c["tile_n"]), int(c["tile_k"])
        if mode == 2:  # a K tile must not straddle two scale blocks
            bk = min(bk, block)
            while block % bk:
                bk //= 2
        _fp8_gemm_kernel[(cdiv(m, bm) * cdiv(n, bn),)](
            a, wq, sa, sb, bias if bias is not None else zero, out,
            m, n, k_pad,
            a.stride(0), a.stride(1), wq.stride(0), wq.stride(1), out.stride(0), out.stride(1),
            strides[0], strides[1], strides[2], strides[3],
            SCALE_MODE=mode, SCALE_BLOCK=block, SB_DIV=sb_div, IMPRECISE=min(128, bk) if sm == 90 else 0, HAS_BIAS=bias is not None,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=int(c.get("warps", 4)), num_stages=int(c.get("stages", 3)),
        )
        return out

    launch_fitting(("fp8", mode, sb_div, str(out_dtype), sm, tuple(sorted((kk, v) for kk, v in cfg.items() if isinstance(v, (int, str))))),
                   cfg, _launch, min_tile_k=32)
    return out.reshape(*lead, n)


def act_quant_fp8_triton(
    x: torch.Tensor,
    *,
    granularity: str = "per_token",
    block: int = 128,
    scale: torch.Tensor | None = None,
    rms_weight: torch.Tensor | None = None,
    rms_eps: float | None = None,
    num_warps: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise ``x [..., K]`` to ``float8_e4m3fn``.  With ``rms_eps`` set the RMSNorm (optionally with
    ``rms_weight``) is fused in front of the absmax/quantise (per-token only).  Returns ``(q, scale)`` in the
    conventions of the oracle (per-token ``[M]``, per-tensor ``[1]``, block ``[M, ceil(K/B)]``)."""
    require_triton()
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    m, k = x2.shape
    q = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    fused_rms = rms_eps is not None
    if fused_rms and granularity != "per_token":
        raise ValueError("fused RMSNorm requires per-token granularity")
    if granularity == "per_token":
        s = torch.empty(m, dtype=torch.float32, device=x.device)
        if m:
            blk = min(4096, max(128, next_pow2(k)))
            _act_quant_rowwise_kernel[(m,)](
                x2, rms_weight if rms_weight is not None else x2, q, s, k, float(rms_eps or 0.0),
                x2.stride(0), q.stride(0),
                FMAX=FP8_E4M3_MAX, RMS=fused_rms, HAS_W=rms_weight is not None, BLOCK=blk, num_warps=num_warps,
            )
    elif granularity == "block":
        if block != next_pow2(block):
            raise ValueError(f"Triton block quantiser needs a power-of-two block, got {block}")  # dispatcher falls back to the oracle
        kb = cdiv(k, block)
        s = torch.empty((m, kb), dtype=torch.float32, device=x.device)
        if m:
            _act_quant_blockwise_kernel[(m, kb)](
                x2, q, s, k, x2.stride(0), q.stride(0), s.stride(0),
                FMAX=FP8_E4M3_MAX, BLOCK=next_pow2(block), num_warps=num_warps,
            )
    elif granularity == "per_tensor":
        if scale is None:
            amax = x2.abs().amax().float().reshape(1)
            s = torch.where(amax > 0, amax / FP8_E4M3_MAX, torch.ones_like(amax))
        else:
            s = scale.float().reshape(1).to(x.device)
        xc = x2.contiguous()
        if xc.numel():
            _act_quant_static_kernel[(cdiv(xc.numel(), 1024),)](xc, q, s, xc.numel(), FMAX=FP8_E4M3_MAX, BLOCK=1024)
    else:
        raise ValueError(f"unsupported granularity {granularity!r}")
    return q.reshape(shape), s


__all__ = ["fp8_gemm_triton", "act_quant_fp8_triton", "HAS_TRITON"]
