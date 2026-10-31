"""Triton quantisation utilities (Plan K8 / §6.7): row/group absmax-quantise (INT8, FP8-E4M3), fused RMSNorm(+residual)+quantise,
and the counter-based-RNG stochastic-rounding quantiser (INT8/INT4 grids and the FP4 E2M1 grid).

**Status: implemented, not yet validated on NVIDIA hardware** (INT8 / SR paths run in the Triton CPU interpreter in the unit
tests; FP8 stores need ``float8e4nv`` = sm_89+ and are GPU-test only).  Min Triton 2.1.

Rounding conventions (mirrored by the oracles in ``ref/util_quant.py``): INT rounding is *half away from zero*; FP8 uses the
hardware/torch round-to-nearest-even conversion after clamping to +-448; all-zero groups get ``scale = 1``; the absmax ignores
non-finite values.  SR draws ``u = (philox_r0(seed, offset + flat_index) >> 8) * 2^-24`` per element.
"""
from __future__ import annotations

import torch

from forge.kernels.triton.attn_common import HAVE_TRITON
from forge.kernels.triton.util_dispatch import cdiv, next_pow2, require_fp8_store

if HAVE_TRITON:
    import triton
    import triton.language as tl

    from forge.kernels.triton.attn_common import philox_uniform

    @triton.jit
    def _round_half_away(y):
        pos = (y + 0.5).to(tl.int32)
        neg = -((0.5 - y).to(tl.int32))
        return tl.where(y >= 0.0, pos, neg)

    @triton.jit
    def _quant_rowwise_kernel(
        x_ptr, q_ptr, s_ptr, G,
        QMAX: tl.constexpr, QKIND: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """One program per row/group of ``G`` contiguous elements. ``QKIND`` 0 = int8, 1 = fp8 e4m3."""
        r = tl.program_id(0)
        base = r.to(tl.int64) * G
        amax = tl.zeros([1], tl.float32)
        for st in range(0, G, BLOCK):
            offs = st + tl.arange(0, BLOCK)
            m = offs < G
            x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            ax = tl.abs(x)
            ax = tl.where(ax < float("inf"), ax, 0.0)  # drops inf and NaN (comparison is False for NaN)
            amax = tl.maximum(amax, tl.max(ax, axis=0))
        sc0 = amax / QMAX
        scale = tl.where(sc0 > 0.0, sc0, 1.0)  # all-zero or underflowing groups: scale 1 -> values flush to 0
        for st in range(0, G, BLOCK):
            offs = st + tl.arange(0, BLOCK)
            m = offs < G
            x = tl.load(x_ptr + base + offs, mask=m, other=0.0).to(tl.float32)
            y = x / scale
            if QKIND == 0:
                qv = _round_half_away(tl.minimum(tl.maximum(y, -128.0), 127.0))
                tl.store(q_ptr + base + offs, qv.to(tl.int8), mask=m)
            else:
                yc = tl.minimum(tl.maximum(y, -448.0), 448.0)
                tl.store(q_ptr + base + offs, yc.to(q_ptr.dtype.element_ty), mask=m)
        tl.store(s_ptr + r + tl.arange(0, 1), scale)

    @triton.jit
    def _rmsnorm_quant_kernel(
        x_ptr, res_ptr, w_ptr, q_ptr, s_ptr, res_out_ptr, y_ptr, K, eps,
        sx, sr, sq, so, sy,
        BLOCK_K: tl.constexpr, QMAX: tl.constexpr, QKIND: tl.constexpr, HAS_RES: tl.constexpr, STORE_Y: tl.constexpr,
    ):
        r = tl.program_id(0)
        offs = tl.arange(0, BLOCK_K)
        m = offs < K
        x = tl.load(x_ptr + r.to(tl.int64) * sx + offs, mask=m, other=0.0)
        if HAS_RES:
            rr = tl.load(res_ptr + r.to(tl.int64) * sr + offs, mask=m, other=0.0)
            h = (x.to(tl.float32) + rr.to(tl.float32)).to(x.dtype)  # residual stream lives in the activation dtype
            tl.store(res_out_ptr + r.to(tl.int64) * so + offs, h, mask=m)
        else:
            h = x
        hf = h.to(tl.float32)
        var = tl.sum(hf * hf, axis=0) / K
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        y = hf * rstd * w
        if STORE_Y:
            tl.store(y_ptr + r.to(tl.int64) * sy + offs, y, mask=m)
        ax = tl.where(m, tl.abs(y), 0.0)
        ax = tl.where(ax < float("inf"), ax, 0.0)
        amax = tl.max(ax, axis=0)
        sc0 = amax / QMAX
        scale = tl.where(sc0 > 0.0, sc0, 1.0)  # all-zero or underflowing groups: scale 1 -> values flush to 0
        yq = y / scale
        if QKIND == 0:
            qv = _round_half_away(tl.minimum(tl.maximum(yq, -128.0), 127.0))
            tl.store(q_ptr + r.to(tl.int64) * sq + offs, qv.to(tl.int8), mask=m)
        else:
            yc = tl.minimum(tl.maximum(yq, -448.0), 448.0)
            tl.store(q_ptr + r.to(tl.int64) * sq + offs, yc.to(q_ptr.dtype.element_ty), mask=m)
        tl.store(s_ptr + r, scale)

    @triton.jit
    def _sr_kernel(
        x_ptr, scale_ptr, q_ptr, n, seed, offset, G,
        FMT: tl.constexpr, BLOCK: tl.constexpr,
    ):
        """Stochastic rounding of flat elements. FMT 0 = int8, 1 = int4 (codes in [-8, 7]), 2 = fp4 E2M1 codes."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        sc = tl.load(scale_ptr + offs // G, mask=m, other=1.0)
        y = x / sc
        ctr = (offs + offset).to(tl.uint32)
        u = philox_uniform(seed, ctr)
        if FMT == 2:
            a = tl.minimum(tl.abs(y), 6.0)
            lo = a * 0.0
            hi = lo + 0.5
            lidx = lo
            ge = a >= 0.5
            lo = tl.where(ge, 0.5, lo)
            lidx = tl.where(ge, 1.0, lidx)
            hi = tl.where(ge, 1.0, hi)
            ge = a >= 1.0
            lo = tl.where(ge, 1.0, lo)
            lidx = tl.where(ge, 2.0, lidx)
            hi = tl.where(ge, 1.5, hi)
            ge = a >= 1.5
            lo = tl.where(ge, 1.5, lo)
            lidx = tl.where(ge, 3.0, lidx)
            hi = tl.where(ge, 2.0, hi)
            ge = a >= 2.0
            lo = tl.where(ge, 2.0, lo)
            lidx = tl.where(ge, 4.0, lidx)
            hi = tl.where(ge, 3.0, hi)
            ge = a >= 3.0
            lo = tl.where(ge, 3.0, lo)
            lidx = tl.where(ge, 5.0, lidx)
            hi = tl.where(ge, 4.0, hi)
            ge = a >= 4.0
            lo = tl.where(ge, 4.0, lo)
            lidx = tl.where(ge, 6.0, lidx)
            hi = tl.where(ge, 6.0, hi)
            ge = a >= 6.0
            lo = tl.where(ge, 6.0, lo)
            lidx = tl.where(ge, 7.0, lidx)
            hi = tl.where(ge, 6.0, hi)
            denom = hi - lo
            p_up = tl.where(denom > 0.0, (a - lo) / tl.where(denom > 0.0, denom, 1.0), 0.0)
            up = (u < p_up) & (denom > 0.0)
            idx = tl.where(up, lidx + 1.0, lidx).to(tl.int32)
            code = idx | tl.where(y < 0.0, 8, 0)
            tl.store(q_ptr + offs, code.to(tl.uint8), mask=m)
        else:
            if FMT == 0:
                lo_b = -128.0
                hi_b = 127.0
            else:
                lo_b = -8.0
                hi_b = 7.0
            yc = tl.minimum(tl.maximum(y, lo_b), hi_b)
            t = yc.to(tl.int32).to(tl.float32)  # truncation toward zero
            fl = tl.where(t > yc, t - 1.0, t)  # floor
            qv = fl + tl.where(u < (yc - fl), 1.0, 0.0)
            qv = tl.minimum(tl.maximum(qv, lo_b), hi_b)
            tl.store(q_ptr + offs, qv.to(tl.int8), mask=m)


def quantize_rowwise_triton(x: torch.Tensor, qdtype: str = "int8", group_size: int | None = None):
    """Row/group absmax quantisation -> ``(q, scale float32 [..., K // group])``. ``qdtype`` ``"int8"`` or ``"fp8"``
    (FP8 requires sm_89+ for the Triton store)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    if qdtype != "int8":
        require_fp8_store(x.device, "quantize_rowwise_triton")
    k = x.shape[-1]
    g = k if group_size is None else group_size
    if k % g:
        raise ValueError("last dim must be divisible by group_size")
    xc = x.contiguous()
    rows = xc.numel() // g
    q = torch.empty(xc.shape, dtype=torch.int8 if qdtype == "int8" else torch.float8_e4m3fn, device=x.device)
    s = torch.empty(rows, dtype=torch.float32, device=x.device)
    _quant_rowwise_kernel[(rows,)](xc, q, s, g, QMAX=127.0 if qdtype == "int8" else 448.0,
                                   QKIND=0 if qdtype == "int8" else 1, BLOCK=min(4096, max(16, next_pow2(g))), num_warps=4)
    return q, s.view(*x.shape[:-1], k // g)


def rmsnorm_quant_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, qdtype: str = "int8",
                         residual: torch.Tensor | None = None, return_norm: bool = False):
    """Fused (residual add +) RMSNorm + per-row quantise. ``x [R, K]`` (K <= 16384)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    if qdtype != "int8":
        require_fp8_store(x.device, "rmsnorm_quant_triton")
    r, k = x.shape
    if k > 16384:
        raise ValueError("hidden size above 16384 is not supported by the fused kernel")
    xc = x.contiguous()
    q = torch.empty(r, k, dtype=torch.int8 if qdtype == "int8" else torch.float8_e4m3fn, device=x.device)
    s = torch.empty(r, dtype=torch.float32, device=x.device)
    rc = residual.contiguous() if residual is not None else xc
    res_out = torch.empty_like(xc) if residual is not None else xc
    y = torch.empty(r, k, dtype=torch.float32, device=x.device) if return_norm else xc
    _rmsnorm_quant_kernel[(r,)](
        xc, rc, weight.contiguous(), q, s, res_out, y, k, float(eps),
        xc.stride(0), rc.stride(0), q.stride(0), res_out.stride(0), y.stride(0),
        BLOCK_K=max(16, next_pow2(k)), QMAX=127.0 if qdtype == "int8" else 448.0, QKIND=0 if qdtype == "int8" else 1,
        HAS_RES=residual is not None, STORE_Y=return_norm, num_warps=8 if k > 4096 else 4)
    out = (q, s, res_out if residual is not None else None)
    return out + (y,) if return_norm else out


def stochastic_round_triton(x: torch.Tensor, scale: torch.Tensor, group_size: int, fmt: str, seed: int, offset: int = 0):
    """SR quantise ``x`` with precomputed ``scale`` (one per ``group_size`` contiguous elements). Returns codes
    (``int8`` for int8/int4, ``uint8`` E2M1 codes for fp4)."""
    if not HAVE_TRITON:
        raise RuntimeError("triton is not available")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    xc = x.contiguous()
    sc = scale.to(torch.float32).contiguous()
    n = xc.numel()
    q = torch.empty(xc.shape, dtype=torch.uint8 if fmt == "fp4" else torch.int8, device=x.device)
    block = 1024
    _sr_kernel[(cdiv(n, block),)](xc, sc, q, n, int(seed), int(offset), group_size,
                                  FMT={"int8": 0, "int4": 1, "fp4": 2}[fmt], BLOCK=block, num_warps=4)
    return q
