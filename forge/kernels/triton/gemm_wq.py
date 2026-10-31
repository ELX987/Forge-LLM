"""Weight-only quantised GEMM (W4A16 / W8A16) Triton kernel with in-kernel dequantisation (Plan 6.4 K1).

Math::   Y[M, N] = X[M, K] @ dequant(Wq[N, K])^T + bias,   dequant = (q - zero) * scale      (fp32 accumulate)

Consumes the ``portable_row`` layout (see :mod:`forge.kernels.layouts.w4`):

* 4-bit: ``qweight int32 [N, ceil(K/8)]`` (nibble ``k % 8`` of word ``k // 8``), zeros ``int32 [G, ceil(N/8)]``
* 8-bit: ``qweight int8 [N, K]``
* scales ``[G, N]`` column-contiguous.

Schedule: one program per (BLOCK_M x BLOCK_N) output tile and split-K slice; ``tl.dot`` on dequantised bf16/fp16
tiles; software pipelining is left to Triton (``num_stages``).  Per-channel scales (single group, no ``g_idx``) are
applied once in the epilogue (they commute with the K sum, Plan 6.4); grouped scales are applied before the MMA.
Split-K: partial sums go to an fp32 workspace ``[S, M, N]`` and a second tiny kernel reduces them in a fixed
order (``deterministic=True``, bit reproducible) or via fp32 atomics (``deterministic=False``, faster, run-to-run
rounding differences possible).

Status: implemented; logic verified with the Triton CPU interpreter; NOT compiled or benchmarked on a GPU.
Notes: ``tl.dot`` needs BLOCK_M, BLOCK_N, BLOCK_K >= 16.  bf16 ``tl.dot`` needs sm_80+ (use fp16 activations on
sm_70/75).  The dequantised tile is rounded to the activation dtype before the MMA (integers x scale), so results
match the fp32 oracle to ~2^-9 relative per weight.
"""
from __future__ import annotations

from typing import Any

import torch

from .gemm_common import HAS_TRITON, cdiv, default_wq_config, launch_fitting, next_pow2, require_triton, tl, triton

if HAS_TRITON:

    @triton.jit
    def _wq_gemm_kernel(
        x_ptr, qw_ptr, s_ptr, z_ptr, g_ptr, bias_ptr, y_ptr, ws_ptr,
        M, N, K,
        stride_xm, stride_xk, stride_qn, stride_sg, stride_zg, stride_ym, stride_yn,
        GROUP: tl.constexpr, BITS: tl.constexpr, HAS_ZEROS: tl.constexpr, HAS_GIDX: tl.constexpr,
        HAS_BIAS: tl.constexpr, PER_CHANNEL: tl.constexpr, SPLIT_K: tl.constexpr, DETERMINISTIC: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        offs_n64 = offs_n.to(tl.int64)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        k_tiles = tl.cdiv(K, BLOCK_K)
        tiles_per_split = tl.cdiv(k_tiles, SPLIT_K)
        k_tile_start = pid_k * tiles_per_split
        k_tile_end = tl.minimum(k_tile_start + tiles_per_split, k_tiles)

        for kt in range(k_tile_start, k_tile_end):
            offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x = tl.load(
                x_ptr + offs_m[:, None].to(tl.int64) * stride_xm + offs_k[None, :] * stride_xk,
                mask=m_mask[:, None] & k_mask[None, :], other=0.0,
            )
            w_mask = k_mask[:, None] & n_mask[None, :]
            if BITS == 4:
                word = tl.load(qw_ptr + offs_n64[None, :] * stride_qn + (offs_k // 8)[:, None], mask=w_mask, other=0)
                q = (word >> ((offs_k % 8) * 4)[:, None]) & 0xF
            else:
                q = tl.load(qw_ptr + offs_n64[None, :] * stride_qn + offs_k[:, None], mask=w_mask, other=0)
            qf = q.to(tl.float32)

            if PER_CHANNEL:
                g = offs_k * 0
            elif HAS_GIDX:
                g = tl.load(g_ptr + offs_k, mask=k_mask, other=0)
            else:
                g = offs_k // GROUP

            if BITS == 4:
                if HAS_ZEROS:
                    zw = tl.load(z_ptr + g[:, None] * stride_zg + (offs_n // 8)[None, :], mask=w_mask, other=0)
                    z = (zw >> ((offs_n % 8) * 4)[None, :]) & 0xF
                    qf = qf - z.to(tl.float32)
                else:
                    qf = qf - 8.0
            if not PER_CHANNEL:
                s = tl.load(s_ptr + g[:, None] * stride_sg + offs_n[None, :], mask=w_mask, other=0.0)
                qf = qf * s.to(tl.float32)
            acc += tl.dot(x, qf.to(x.dtype))

        if PER_CHANNEL:
            sc = tl.load(s_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc * sc[None, :]

        out_mask = m_mask[:, None] & n_mask[None, :]
        if SPLIT_K == 1:
            if HAS_BIAS:
                b = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
                acc = acc + b[None, :]
            tl.store(y_ptr + offs_m[:, None].to(tl.int64) * stride_ym + offs_n[None, :] * stride_yn, acc, mask=out_mask)
        else:
            if DETERMINISTIC:
                ws = ws_ptr + (pid_k * M + offs_m[:, None].to(tl.int64)) * N + offs_n[None, :]
                tl.store(ws, acc, mask=out_mask)
            else:
                ws = ws_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :]
                tl.atomic_add(ws, acc, mask=out_mask)

    @triton.jit
    def _splitk_reduce_kernel(
        ws_ptr, bias_ptr, y_ptr, N, total, stride_ym, stride_yn,
        SPLITS: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """y = sum_s ws[s] (+ bias), summed in fixed order s = 0..SPLITS-1 (deterministic).

        ``total = M * N`` is computed on the host: Triton specialises integer arguments equal to 1 into
        ``constexpr`` (so ``M.to(tl.int64)`` fails to compile for the very common decode shape M == 1).
        """
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        p = ws_ptr + offs
        for _s in range(0, SPLITS):
            acc += tl.load(p, mask=mask, other=0.0)
            p += total
        row = offs // N
        col = offs % N
        if HAS_BIAS:
            acc += tl.load(bias_ptr + col, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * stride_ym + col * stride_yn, acc, mask=mask)


def wq_gemm(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor | None,
    g_idx: torch.Tensor | None,
    bias: torch.Tensor | None,
    *,
    bits: int,
    n: int,
    k: int,
    group_size: int,
    config: dict[str, Any] | None = None,
    deterministic: bool = True,
    sm: int | None = None,
    num_sms: int = 0,
) -> torch.Tensor:
    """Launch the weight-only GEMM on 2-D ``x [M, K]`` (fp16/bf16/fp32 accepted by Triton; fp16/bf16 recommended).

    Returns ``[M, N]`` in ``x.dtype``.  Raises if Triton is unavailable (callers fall back to the oracle).
    """
    require_triton()
    if x.dim() != 2 or x.shape[1] != k:
        raise ValueError(f"x must be [M, {k}], got {tuple(x.shape)}")
    if bits not in (4, 8):
        raise ValueError("bits must be 4 or 8")
    m = x.shape[0]
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return y
    cfg = dict(default_wq_config(m, n, k, sm=sm, num_sms=num_sms, bits=bits))
    if config:
        cfg.update({kk: v for kk, v in config.items() if v is not None})
    per_channel = scales.shape[0] == 1 and g_idx is None
    gsz = k if group_size <= 0 else int(group_size)
    has_zeros = zeros is not None and bits == 4
    dummy = qweight

    def _launch(c: dict[str, Any]) -> torch.Tensor:
        bm, bn, bk = int(c["tile_m"]), int(c["tile_n"]), int(c["tile_k"])
        split = min(max(1, int(c.get("split_k", 1))), max(1, cdiv(k, bk)))
        ws = None
        if split > 1:
            ws = torch.empty((split, m, n), dtype=torch.float32, device=x.device) if deterministic else torch.zeros(
                (m, n), dtype=torch.float32, device=x.device)
        grid = (cdiv(m, bm) * cdiv(n, bn), split)
        _wq_gemm_kernel[grid](
            x, qweight, scales, zeros if has_zeros else dummy, g_idx if g_idx is not None else dummy,
            bias if bias is not None else dummy, y, ws if ws is not None else dummy,
            m, n, k,
            x.stride(0), x.stride(1), qweight.stride(0), scales.stride(0), zeros.stride(0) if has_zeros else 0,
            y.stride(0), y.stride(1),
            GROUP=gsz, BITS=bits, HAS_ZEROS=has_zeros, HAS_GIDX=g_idx is not None, HAS_BIAS=bias is not None,
            PER_CHANNEL=per_channel, SPLIT_K=split, DETERMINISTIC=deterministic,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=int(c.get("warps", 4)), num_stages=int(c.get("stages", 3)),
        )
        if split > 1:
            block = 1024
            _splitk_reduce_kernel[(cdiv(m * n, block),)](
                ws, bias if bias is not None else dummy, y, n, m * n, y.stride(0), y.stride(1),
                SPLITS=split if deterministic else 1, HAS_BIAS=bias is not None, BLOCK=block, num_warps=4,
            )
        return y

    return launch_fitting(("wq", bits, has_zeros, g_idx is not None, per_channel, str(x.dtype), sm,
                           tuple(sorted((kk, v) for kk, v in cfg.items() if isinstance(v, (int, str))))), cfg, _launch)


__all__ = ["wq_gemm", "next_pow2"]
