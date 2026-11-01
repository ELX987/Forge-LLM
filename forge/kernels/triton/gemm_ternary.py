"""Ternary W1.58A8 GEMM and per-token activation quantiser in Triton (ADR 0007, kernel ``ternary_w158a8_gemm``).

``Y[m, n] = sx[m] * sum_g sw[n, g] * sum_{k in g} xq[m, k] * T[n, k]``; ``T in {-1, 0, +1}`` is stored as 2-bit codes, four per byte along K
(layout ``ternary2_row``: 00 -> 0, 01 -> +1, 11 -> -1, 10 reserved -> 0).  ``tl.dot`` on int8 operands accumulates in int32 (exact, order
independent); the per-group fp32 scale is applied when a group is complete, the per-token scale in the epilogue.

Two kernels (K-permutation invariance of the dot product lets us split a tile by code *plane* instead of unpacking interleaved codes):

* ``_t158_planes_kernel``  -- groups that are a multiple of 128, per-row (``group_size == -1``) or per-tensor (``0``) scales.  Each byte column is
  loaded once; the four code planes ``(byte >> 2j) & 3`` are multiplied with the matching stride-4 activation columns (4 int8 dots of K = 32).
* ``_t158_gather_kernel``  -- groups of 32 / 64 / 128 / 256 (tile K = group): bytes gathered at ``k // 4`` and shifted by ``2 * (k % 4)``.

Constraints: ``K % 16 == 0``; int8 ``tl.dot`` needs sm_75+ and tile K >= 32.  Status: logic verified with the Triton CPU interpreter and offline PTX
compilation; NOT run on a GPU (docs/KERNELS.md validation table: all cells 'no').  The activation quantiser follows tern158 exactly
(``q = round_half_even(x * (127 / max|x|))``, ``s = max / 127``), which differs from Forge's int8 quantiser (``x / s``) in rare tie cases.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from .gemm_common import HAS_TRITON, cdiv, launch_fitting, next_pow2, require_triton, tl, triton

if HAS_TRITON:
    if os.environ.get("TRITON_INTERPRET") == "1":  # CPU interpreter: numpy division is already IEEE

        @triton.jit
        def _div_rn(x, y):
            return x / y

    else:
        try:
            from triton.language.extra import libdevice as _libdevice

            _div_rn = _libdevice.div_rn
        except Exception:  # older Triton
            _div_rn = tl.math.div_rn

    @triton.jit
    def _decode(code):
        """int32 2-bit code (0..3) -> int8 ternary: 00 -> 0, 01 -> +1, 11 -> -1, 10 -> 0."""
        return ((code & 1) * (1 - (code & 2))).to(tl.int8)

    @triton.jit
    def _t158_planes_kernel(
        a_ptr, w_ptr, sa_ptr, sw_ptr, bias_ptr, c_ptr,
        M, N, K, G, stride_am, stride_ak, stride_wn, stride_swn, stride_cm, stride_cn,
        TENSOR_SCALE: tl.constexpr, HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        PK: tl.constexpr = BLOCK_K // 4
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_p = tl.arange(0, PK)
        m_mask = offs_m < M
        n_mask = offs_n < N
        kbytes = K // 4
        facc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, tl.cdiv(K, G)):
            iacc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
            for kt in range(0, tl.cdiv(G, BLOCK_K)):
                kbase = g * G + kt * BLOCK_K
                byte_idx = kbase // 4 + offs_p
                wb = tl.load(w_ptr + offs_n[None, :].to(tl.int64) * stride_wn + byte_idx[:, None],
                             mask=n_mask[None, :] & (byte_idx[:, None] < kbytes), other=0).to(tl.int32)
                for j in tl.static_range(4):
                    t = _decode((wb >> (2 * j)) & 3)
                    kidx = kbase + offs_p * 4 + j
                    a = tl.load(a_ptr + offs_m[:, None].to(tl.int64) * stride_am + kidx[None, :] * stride_ak,
                                mask=m_mask[:, None] & (kidx[None, :] < K) & (kidx[None, :] < (g + 1) * G), other=0)
                    iacc = tl.dot(a, t, iacc)
            if TENSOR_SCALE:
                facc += iacc.to(tl.float32) * tl.load(sw_ptr)
            else:
                swv = tl.load(sw_ptr + offs_n * stride_swn + g, mask=n_mask, other=0.0)
                facc += iacc.to(tl.float32) * swv[None, :]
        facc = facc * tl.load(sa_ptr + offs_m, mask=m_mask, other=0.0)[:, None]
        if HAS_BIAS:
            facc = facc + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn, facc, mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _t158_gather_kernel(
        a_ptr, w_ptr, sa_ptr, sw_ptr, bias_ptr, c_ptr,
        M, N, K, stride_am, stride_ak, stride_wn, stride_swn, stride_cm, stride_cn,
        HAS_BIAS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < M
        n_mask = offs_n < N
        kbytes = K // 4
        facc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for g in range(0, tl.cdiv(K, GROUP)):
            offs_k = g * GROUP + tl.arange(0, GROUP)
            k_mask = offs_k < K
            a = tl.load(a_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak, mask=m_mask[:, None] & k_mask[None, :], other=0)
            wb = tl.load(w_ptr + offs_n[None, :].to(tl.int64) * stride_wn + (offs_k // 4)[:, None],
                         mask=n_mask[None, :] & ((offs_k // 4)[:, None] < kbytes), other=0).to(tl.int32)
            t = _decode((wb >> ((offs_k % 4) * 2)[:, None]) & 3)
            iacc = tl.dot(a, t)
            swv = tl.load(sw_ptr + offs_n * stride_swn + g, mask=n_mask, other=0.0)
            facc += iacc.to(tl.float32) * swv[None, :]
        facc = facc * tl.load(sa_ptr + offs_m, mask=m_mask, other=0.0)[:, None]
        if HAS_BIAS:
            facc = facc + tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(c_ptr + offs_m[:, None].to(tl.int64) * stride_cm + offs_n[None, :] * stride_cn, facc, mask=m_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _t158_act_quant_kernel(x_ptr, q_ptr, s_ptr, K, stride_xm, stride_qm, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        offs0 = tl.arange(0, BLOCK)
        amaxv = tl.zeros((BLOCK,), dtype=tl.float32)
        for start in range(0, K, BLOCK):
            offs = start + offs0
            xv = tl.load(x_ptr + row * stride_xm + offs, mask=offs < K, other=0.0).to(tl.float32)
            amaxv = tl.maximum(amaxv, tl.abs(xv))
        mx = tl.max(amaxv, axis=0)
        inv = tl.where(mx > 0.0, _div_rn(127.0, mx), 0.0)
        tl.store(s_ptr + row, _div_rn(mx, 127.0))
        for start in range(0, K, BLOCK):
            offs = start + offs0
            xv = tl.load(x_ptr + row * stride_xm + offs, mask=offs < K, other=0.0).to(tl.float32)
            v = tl.minimum(tl.maximum(xv * inv, -127.0), 127.0)
            r = (v + 12582912.0) - 12582912.0  # round-half-to-even (|v| <= 127 << 2**22), matches torch.round
            tl.store(q_ptr + row * stride_qm + offs, r.to(tl.int8), mask=offs < K)


def supported_group(group_size: int, k: int) -> bool:
    """Scale layouts the Triton kernels implement (others use the torch oracle)."""
    if k % 16:
        return False
    if group_size in (-1, 0):
        return True
    if group_size in (32, 64):
        return k % group_size == 0
    return group_size % 128 == 0 and k % group_size == 0


def default_t158_config(m: int, n: int, k: int, group_size: int, sm: int | None = 89) -> dict[str, Any]:
    """Conservative heuristic config (the autotuner is the place to be clever; every number is UNMEASURED)."""
    stages = 2 if (sm is not None and (sm < 80 or sm in (86, 89, 120, 121))) else 3
    if m <= 16:
        return dict(tile_m=16, tile_n=64, tile_k=128, warps=4, stages=stages)
    if m <= 64:
        return dict(tile_m=64, tile_n=64, tile_k=128, warps=4, stages=stages)
    return dict(tile_m=128, tile_n=64, tile_k=128, warps=8, stages=stages)


def ternary_gemm_triton(
    xq: torch.Tensor, sx: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, group_size: int, k: int,
    bias: torch.Tensor | None = None, *, out_dtype: torch.dtype = torch.bfloat16, config: dict[str, Any] | None = None, sm: int | None = None,
) -> torch.Tensor:
    """``xq int8 [..., K]``, ``sx fp32 [M]``, packed weight ``uint8 [N, K/4]``, ``scales fp32 [N, G]`` (``[1, 1]`` per tensor) -> ``[..., N]``."""
    require_triton()
    if not supported_group(group_size, k):
        raise NotImplementedError(f"ternary Triton kernel: unsupported group_size={group_size} for K={k}")
    lead = xq.shape[:-1]
    a = xq.reshape(-1, xq.shape[-1]).contiguous()
    w = qweight.contiguous()
    m, kk = a.shape
    n = w.shape[0]
    if kk != k or w.shape[1] * 4 != k:
        raise ValueError(f"shape mismatch: activations K={kk}, packed weight {tuple(w.shape)}, K={k}")
    out = torch.empty((m, n), dtype=out_dtype, device=xq.device)
    if m == 0:
        return out.reshape(*lead, n)
    sa = sx.float().reshape(-1).contiguous()
    if sa.numel() != m:
        raise ValueError(f"sx has {sa.numel()} entries for M={m}")
    sw = scales.float().contiguous()
    tensor_scale = group_size == 0
    G = k if group_size in (-1, 0) else group_size
    if tensor_scale:
        sw = sw.reshape(1, 1)
    elif tuple(sw.shape) != (n, k // G):
        raise ValueError(f"scales must be [{n}, {k // G}], got {tuple(sw.shape)}")
    cfg = dict(default_t158_config(m, n, k, group_size, sm))
    if config:
        cfg.update({kk_: v for kk_, v in config.items() if v is not None})
    zero = torch.zeros(1, dtype=torch.float32, device=xq.device)
    use_planes = group_size in (-1, 0) or group_size % 128 == 0

    def _launch(c: dict[str, Any]) -> torch.Tensor:
        bm, bn = int(c["tile_m"]), int(c["tile_n"])
        grid = (cdiv(m, bm) * cdiv(n, bn),)
        common = dict(num_warps=int(c.get("warps", 4)), num_stages=int(c.get("stages", 3)))
        if use_planes:
            bk = max(128, int(c["tile_k"]))
            if G % bk and group_size not in (-1, 0):
                bk = 128
            _t158_planes_kernel[grid](a, w, sa, sw, bias if bias is not None else zero, out, m, n, k, G, a.stride(0), a.stride(1), w.stride(0),
                                      0 if tensor_scale else sw.stride(0), out.stride(0), out.stride(1),
                                      TENSOR_SCALE=tensor_scale, HAS_BIAS=bias is not None, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, **common)
        else:
            _t158_gather_kernel[grid](a, w, sa, sw, bias if bias is not None else zero, out, m, n, k, a.stride(0), a.stride(1), w.stride(0), sw.stride(0),
                                      out.stride(0), out.stride(1), HAS_BIAS=bias is not None, BLOCK_M=bm, BLOCK_N=bn, GROUP=group_size, **common)
        return out

    launch_fitting(("t158", str(out_dtype), sm, group_size, tuple(sorted((a_, b_) for a_, b_ in cfg.items() if isinstance(b_, (int, str))))), cfg, _launch, min_tile_k=128)
    return out.reshape(*lead, n)


def ternary_act_quant_triton(x: torch.Tensor, num_warps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token int8 quantisation exactly as tern158 / :func:`forge.formats.ternary.act_quant_int8`: ``(q int8 [M, K], s fp32 [M])``."""
    require_triton()
    x2 = x.reshape(-1, x.shape[-1]).contiguous()
    m, k = x2.shape
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    s = torch.empty(m, dtype=torch.float32, device=x.device)
    if m:
        _t158_act_quant_kernel[(m,)](x2, q, s, k, x2.stride(0), q.stride(0), BLOCK=min(4096, max(128, next_pow2(k))), num_warps=num_warps)
    return q, s


__all__ = ["HAS_TRITON", "default_t158_config", "supported_group", "ternary_act_quant_triton", "ternary_gemm_triton"]
