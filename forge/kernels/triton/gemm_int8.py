"""INT8 W8A8 GEMM and per-token dynamic activation quantisation in Triton (Plan 6.4 K3, INT8 branch).

``Y = (Xq @ Wq^T)_int32 * sx[:, None] * sw[None, :] + bias``.  ``tl.dot`` on int8 operands accumulates in int32
(exact; reduction order irrelevant), the fp32 epilogue applies per-token x per-channel scales.  Needs sm_75+ (int8
tensor cores) and BLOCK_K >= 32.  Status: logic verified with the Triton CPU interpreter, not run on a GPU.
"""
from __future__ import annotations

from typing import Any

import torch

from .gemm_common import HAS_TRITON, cdiv, default_mm_config, launch_fitting, next_pow2, require_triton, tl, triton

if HAS_TRITON:

    @triton.jit
    def _int8_gemm_kernel(
        a_ptr, b_ptr, sa_ptr, sb_ptr, bias_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
        PER_TENSOR_A: tl.constexpr, PER_TENSOR_B: tl.constexpr, HAS_BIAS: tl.constexpr,
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
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for kt in range(0, tl.cdiv(K, BLOCK_K)):
            offs_k = kt * BLOCK_K + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak,
                        mask=m_mask[:, None] & k_mask[None, :], other=0)
            b = tl.load(b_ptr + offs_n[None, :].to(tl.int64) * stride_bn + offs_k[:, None] * stride_bk,
                        mask=n_mask[None, :] & k_mask[:, None], other=0)
            acc = tl.dot(a, b, acc)
        accf = acc.to(tl.float32)
        if PER_TENSOR_A:
            accf = accf * tl.load(sa_ptr)
        else:
            accf = accf * tl.load(sa_ptr + offs_m, mask=m_mask, other=0.0)[:, None]
        if PER_TENSOR_B:
            accf = accf * tl.load(sb_ptr)
        else:
            accf = accf * tl.load(sb_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
        if HAS_BIAS:
            accf = accf + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn, accf,
                 mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _int8_quant_rowwise_kernel(
        x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs0 = tl.arange(0, BLOCK)
        amaxv = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(0, K, BLOCK):
            offs = start + offs0
            xv = tl.load(x_ptr + row * stride_xm + offs, mask=offs < K, other=0.0).to(tl.float32)
            amaxv = tl.maximum(amaxv, tl.abs(xv))
        amax = tl.max(amaxv, axis=0)
        scale = tl.where(amax > 0.0, amax / 127.0, 1.0)
        tl.store(s_ptr + row, scale)
        for start in range(0, K, BLOCK):
            offs = start + offs0
            xv = tl.load(x_ptr + row * stride_xm + offs, mask=offs < K, other=0.0).to(tl.float32)
            v = tl.minimum(tl.maximum(xv / scale, -127.0), 127.0)
            # round-half-to-even via the 1.5 * 2**23 trick (|v| <= 127 << 2**22); matches torch.round
            r = (v + 12582912.0) - 12582912.0
            tl.store(q_ptr + row * stride_qm + offs, r.to(tl.int8), mask=offs < K)


def int8_gemm_triton(
    xq: torch.Tensor,
    sx: torch.Tensor,
    wq: torch.Tensor,
    sw: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
    config: dict[str, Any] | None = None,
    sm: int | None = None,
) -> torch.Tensor:
    """INT8 GEMM launcher: ``xq [..., K] int8``, ``wq [N, K] int8``, scales ``[M]``/``[1]`` and ``[N]``/``[1]``."""
    require_triton()
    lead = xq.shape[:-1]
    a = xq.reshape(-1, xq.shape[-1]).contiguous()
    wq = wq.contiguous()  # strides below are those of the contiguous copy
    m, k = a.shape
    n = wq.shape[0]
    out = torch.empty((m, n), dtype=out_dtype, device=xq.device)
    if m == 0:
        return out.reshape(*lead, n)
    cfg = dict(default_mm_config(m, n, k, sm=sm))
    if config:
        cfg.update({kk: v for kk, v in config.items() if v is not None})
    sa, sb = sx.float().reshape(-1).contiguous(), sw.float().reshape(-1).contiguous()
    if sa.numel() not in (1, m) or sb.numel() not in (1, n):
        raise ValueError(f"scale sizes {sa.numel()}/{sb.numel()} do not match M={m}/N={n}")
    zero = torch.zeros(1, dtype=torch.float32, device=xq.device)

    def _launch(c: dict[str, Any]) -> torch.Tensor:
        bm, bn, bk = int(c["tile_m"]), int(c["tile_n"]), max(32, int(c["tile_k"]))
        _int8_gemm_kernel[(cdiv(m, bm) * cdiv(n, bn),)](
            a, wq, sa, sb, bias if bias is not None else zero, out,
            m, n, k, a.stride(0), a.stride(1), wq.stride(0), wq.stride(1), out.stride(0), out.stride(1),
            PER_TENSOR_A=sa.numel() == 1, PER_TENSOR_B=sb.numel() == 1, HAS_BIAS=bias is not None,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
            num_warps=int(c.get("warps", 4)), num_stages=int(c.get("stages", 3)),
        )
        return out

    launch_fitting(("int8", str(out_dtype), sm, tuple(sorted((kk, v) for kk, v in cfg.items() if isinstance(v, (int, str))))),
                   cfg, _launch, min_tile_k=32)
    return out.reshape(*lead, n)


def int8_quant_triton(x: torch.Tensor, num_warps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token symmetric int8 quantisation of ``x [..., K]`` -> ``(q int8, scale fp32 [M])``."""
    require_triton()
    shape = x.shape
    x2 = x.reshape(-1, shape[-1]).contiguous()
    m, k = x2.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty(m, dtype=torch.float32, device=x.device)
    if m:
        _int8_quant_rowwise_kernel[(m,)](x2, q, s, k, x2.stride(0), q.stride(0),
                                        BLOCK=min(4096, max(128, next_pow2(k))), num_warps=num_warps)
    return q.reshape(shape), s


__all__ = ["int8_gemm_triton", "int8_quant_triton", "HAS_TRITON"]
