"""Quantisation utility oracles (Plan K8): absmax/scale, row/group-wise FP8 & INT8 quantisation, counter-based stochastic
rounding (INT grids and the FP4 E2M1 grid), and fused RMSNorm + quantise.

Conventions
-----------
* ``scale = absmax / qmax`` with ``qmax = 448`` (E4M3FN), ``127`` (int8), ``7`` (int4), ``6`` (E2M1); all-zero groups get
  ``scale = 1`` so that dequantisation is exactly zero (no 0/0).  Non-finite inputs are ignored for the absmax and are
  *unspecified* in the Triton kernels (finite inputs are a precondition; NaN stays NaN only in the torch path).
* INT8 rounding of the quantisers is *half away from zero* (the Triton kernels' convention, see :func:`round_half_away`);
  FP8 uses the round-to-nearest-even conversion; the KV-page quantisers in ``attn_kv`` (offline, torch only) use ``torch.round``.
* ``group_size=None`` means one scale per row of the last dimension; otherwise one scale per ``group_size`` contiguous
  elements of the last dimension (scale shape ``[..., K // group_size]``).
* Stochastic rounding uses :func:`forge.kernels.ref.util_rng.philox_uniform` with offset ``offset + flat_element_index``.
"""
from __future__ import annotations

import torch

from forge.kernels.ref.util_rng import philox_uniform

E4M3_MAX = 448.0
E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
QMAX = {"fp8": E4M3_MAX, "int8": 127.0, "int4": 7.0, "fp4": 6.0}


def _grouped(x: torch.Tensor, group_size: int | None):
    k = x.shape[-1]
    g = k if group_size is None else group_size
    if k % g:
        raise ValueError(f"last dim {k} not divisible by group_size {g}")
    return x.reshape(*x.shape[:-1], k // g, g)


def absmax_ref(x: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    """Per-row (or per-group) max |x| over finite elements -> ``[..., groups]`` float32."""
    xg = _grouped(x.float(), group_size).abs()
    xg = torch.where(torch.isfinite(xg), xg, torch.zeros_like(xg))
    return xg.amax(-1)


def round_half_away(y: torch.Tensor) -> torch.Tensor:
    """Round half away from zero (what the Triton kernels implement; ``torch.round`` is half-to-even)."""
    return torch.sign(y) * torch.floor(y.abs() + 0.5)


def scale_from_absmax(amax: torch.Tensor, fmt: str) -> torch.Tensor:
    qmax = QMAX[fmt]
    s = amax / qmax
    return torch.where(s > 0, s, torch.ones_like(s))  # all-zero / underflowing group -> scale 1 (values flush to 0)


def quantize_rowwise_ref(x: torch.Tensor, qdtype: str = "fp8", group_size: int | None = None):
    """Row/group-wise symmetric quantisation. Returns ``(q, scale)``: ``q`` is ``float8_e4m3fn`` or ``int8`` with the
    shape of ``x`` and ``scale`` is float32 ``[..., K // group]``."""
    amax = absmax_ref(x, group_size)
    scale = scale_from_absmax(amax, qdtype)
    xg = _grouped(x.float(), group_size)
    y = xg / scale.unsqueeze(-1)
    if qdtype == "fp8":
        q = y.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    elif qdtype == "int8":
        q = round_half_away(y).clamp(-128, 127).to(torch.int8)
    else:
        raise ValueError("qdtype must be 'fp8' or 'int8'")
    return q.reshape(x.shape), scale


def dequantize_rowwise_ref(q: torch.Tensor, scale: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
    qf = (q.view(torch.float8_e4m3fn) if q.dtype == torch.uint8 else q).float()
    return (_grouped(qf, group_size) * scale.unsqueeze(-1)).reshape(q.shape)


# ---- stochastic rounding ------------------------------------------------------------------------------------
def fp4_values(codes: torch.Tensor) -> torch.Tensor:
    """Decode E2M1 codes (0..15: sign bit 3, magnitude index 0..7) to float32 grid values."""
    grid = torch.tensor(E2M1_GRID, dtype=torch.float32, device=codes.device)
    mag = grid[(codes.long() & 7)]
    return torch.where((codes.long() & 8) != 0, -mag, mag)


def sr_uniform(shape, seed: int, offset: int, device) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= int(s)
    idx = torch.arange(n, dtype=torch.int64, device=device) + int(offset)
    return philox_uniform(seed, idx).reshape(shape)


def stochastic_round_ref(x: torch.Tensor, fmt: str = "int8", *, group_size: int | None = None,
                         scale: torch.Tensor | None = None, seed: int = 0, offset: int = 0):
    """Stochastic-rounding quantiser (unbiased: ``E[dequant] = clip(x)``).

    ``fmt``: ``"int8"``, ``"int4"`` (codes ``int8`` in ``[-8, 7]``) or ``"fp4"`` (E2M1 codes ``uint8`` in ``0..15``).
    If ``scale`` is ``None`` it is derived per group from the absmax (``absmax / qmax``).  Returns ``(codes, scale)``.
    The uniform used for flat element ``i`` of ``x`` (row-major) is ``philox_uniform(seed, offset + i)``.
    """
    scale_t = scale if scale is not None else scale_from_absmax(absmax_ref(x, group_size), fmt)
    xg = _grouped(x.float(), group_size)
    y = xg / scale_t.float().unsqueeze(-1)
    u = sr_uniform(xg.shape, seed, offset, x.device)
    if fmt in ("int8", "int4"):
        lo_b, hi_b = (-128.0, 127.0) if fmt == "int8" else (-8.0, 7.0)
        y = y.clamp(lo_b, hi_b)
        fl = torch.floor(y)
        q = fl + (u < (y - fl)).float()
        q = q.clamp(lo_b, hi_b).to(torch.int8)
    elif fmt == "fp4":
        a = y.abs().clamp(max=6.0)
        lo = torch.zeros_like(a)
        hi = torch.full_like(a, 0.5)
        lidx = torch.zeros_like(a)
        for i in range(1, 8):
            g = E2M1_GRID[i]
            ge = a >= g
            lo = torch.where(ge, torch.full_like(a, g), lo)
            lidx = torch.where(ge, torch.full_like(a, float(i)), lidx)
            hi = torch.where(ge, torch.full_like(a, E2M1_GRID[min(i + 1, 7)]), hi)
        denom = hi - lo
        p_up = torch.where(denom > 0, (a - lo) / denom.clamp_min(1e-30), torch.zeros_like(a))
        up = (u < p_up) & (denom > 0)
        idx = torch.where(up, lidx + 1, lidx).to(torch.int32)
        code = idx | torch.where(y < 0, torch.full_like(idx, 8), torch.zeros_like(idx))
        q = code.to(torch.uint8)
    else:
        raise ValueError("fmt must be int8|int4|fp4")
    return q.reshape(x.shape), scale_t


def dequantize_sr_ref(q: torch.Tensor, scale: torch.Tensor, fmt: str, group_size: int | None = None) -> torch.Tensor:
    vals = fp4_values(q) if fmt == "fp4" else q.float()
    return (_grouped(vals, group_size) * scale.unsqueeze(-1)).reshape(q.shape)


# ---- rmsnorm + quant ----------------------------------------------------------------------------------------
def rmsnorm_quant_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, qdtype: str = "fp8",
                      residual: torch.Tensor | None = None, return_norm: bool = False):
    """Fused ``(residual add) -> RMSNorm -> per-row quantise`` oracle.

    ``h = x (+ residual)`` (rounded to the activation dtype, which is how the residual stream is stored);
    ``y = h * rsqrt(mean(h^2) + eps) * weight``; ``q, scale = quantize_rowwise(y)``.
    Returns ``(q, scale[R], h or None)`` (+ ``y`` if ``return_norm``).  ``h`` is the updated residual stream.
    """
    if residual is not None:
        h = (x.float() + residual.float()).to(x.dtype).float()
    else:
        h = x.float()
    var = (h * h).mean(-1, keepdim=True)
    y = h * torch.rsqrt(var + eps) * weight.float()
    q, scale = quantize_rowwise_ref(y, qdtype, None)
    res_out = h.to(x.dtype) if residual is not None else None
    if return_norm:
        return q, scale.squeeze(-1), res_out, y
    return q, scale.squeeze(-1), res_out
