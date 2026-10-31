"""FP8 (E4M3 "FN" and E5M2) quantisation with scaling.

``x ~= x8.float() * scale``.  Scaling granularity: per tensor (default), per slice (``dim=``: reduce over that dim, e.g.
``dim=-1`` = one scale per row) or per block of ``block`` elements along the last dim.  Values are clamped (saturating)
before the cast, so there is never an inf/NaN from overflow.  The cast itself is torch's round-to-nearest-even
(``torch.float8_e4m3fn`` / ``torch.float8_e5m2``); :func:`emulate_fp8` is a pure-arithmetic re-implementation used to
prove the cast bit-identical in the tests and as a fallback on builds without float8 dtypes.

E4M3: 1-4-3, bias 7, max 448, min subnormal 2**-9 (no inf; S.1111.111 is NaN).  E5M2: 1-5-2, bias 15, max 57344.
"""
from __future__ import annotations

import torch

from forge.core.errors import ConfigError

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
_FMT_PARAMS = {"e4m3": (4, 3, 7, 448.0), "e5m2": (5, 2, 15, 57344.0)}  # exp bits, mantissa bits, bias, max
TINY = 1e-12


def fp8_dtype(fmt: str) -> torch.dtype:
    fmt = _norm_fmt(fmt)
    return torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2


def _norm_fmt(fmt: str) -> str:
    f = fmt.lower().replace("fp8_", "").replace("fp8", "").strip("_- ")
    if f not in FP8_MAX:
        raise ConfigError(f"unknown fp8 format {fmt!r} (e4m3|e5m2)")
    return f


def emulate_fp8(x: torch.Tensor, fmt: str = "e4m3") -> torch.Tensor:
    """Round-to-nearest-even onto the FP8 grid with saturation, computed in float32 arithmetic (returns float32)."""
    e_bits, m_bits, bias, vmax = _FMT_PARAMS[_norm_fmt(fmt)]
    xf = x.float()
    a = xf.abs().clamp(max=vmax)
    min_exp = 1 - bias
    e = torch.floor(torch.log2(a.clamp(min=2.0**-30))).clamp(min=min_exp)
    step = torch.pow(2.0, e - m_bits)
    q = torch.round(a / step) * step  # torch.round = half to even
    q = q.clamp(max=vmax)
    return torch.copysign(q, xf)


def round_fp8(x: torch.Tensor, fmt: str = "e4m3") -> torch.Tensor:
    """Saturating round-trip through the FP8 dtype, returned as float32 (no scaling)."""
    vmax = FP8_MAX[_norm_fmt(fmt)]
    try:
        return x.float().clamp(-vmax, vmax).to(fp8_dtype(fmt)).float()
    except (AttributeError, RuntimeError, TypeError):  # pragma: no cover - torch build without float8
        return emulate_fp8(x, fmt)


def round_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Non-negative-friendly E4M3 rounding used for NVFP4 block scales (float32 in/out)."""
    return round_fp8(x, "e4m3")


def compute_fp8_scale(
    x: torch.Tensor, fmt: str = "e4m3", dim: int | None = None, block: int | None = None, pow2: bool = False
) -> torch.Tensor:
    """Scale such that ``amax / scale == FP8_MAX`` (per tensor / per ``dim`` slice / per ``block``)."""
    vmax = FP8_MAX[_norm_fmt(fmt)]
    xf = x.detach().float()
    if block is not None:
        k = xf.shape[-1]
        if k % block:
            raise ConfigError(f"last dim {k} not divisible by block {block}")
        amax = xf.reshape(*xf.shape[:-1], k // block, block).abs().amax(dim=-1)
    elif dim is None:
        amax = xf.abs().amax()
    else:
        amax = xf.abs().amax(dim=dim, keepdim=True)
    scale = (amax / vmax).clamp(min=TINY)
    if pow2:
        scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    return scale


def _expand_scale(scale: torch.Tensor, x: torch.Tensor, block: int | None) -> torch.Tensor:
    if block is not None and scale.ndim == x.ndim and scale.shape[-1] != x.shape[-1] and scale.shape[-1] * block == x.shape[-1]:
        return scale.repeat_interleave(block, dim=-1)
    return scale


def to_fp8(
    x: torch.Tensor,
    fmt: str = "e4m3",
    scale: torch.Tensor | float | None = None,
    *,
    dim: int | None = None,
    block: int | None = None,
    pow2: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise to FP8. Returns ``(x8, scale)`` with ``x ~= x8.float() * scale``.

    ``scale=None`` computes it from the absolute maximum (see :func:`compute_fp8_scale`); an explicit ``scale`` (e.g. a static
    calibrated activation scale) is used as given and must broadcast against ``x`` (block scales: ``[..., K/block]``).
    """
    fmt = _norm_fmt(fmt)
    vmax = FP8_MAX[fmt]
    if scale is None:
        s = compute_fp8_scale(x, fmt, dim, block, pow2)
    else:
        s = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    se = _expand_scale(s, x, block)
    y = (x.float() / se).clamp(-vmax, vmax)
    try:
        x8 = y.to(fp8_dtype(fmt))
    except (AttributeError, RuntimeError, TypeError):  # pragma: no cover
        raise ConfigError("this torch build has no float8 dtypes; use emulate_fp8 for fake-quantisation") from None
    return x8, s


def from_fp8(x8: torch.Tensor, scale: torch.Tensor | float, dtype: torch.dtype = torch.float32, block: int | None = None) -> torch.Tensor:
    s = torch.as_tensor(scale, dtype=torch.float32, device=x8.device)
    s = _expand_scale(s, x8, block)
    return (x8.float() * s).to(dtype)


def fake_quant_fp8(
    x: torch.Tensor,
    fmt: str = "e4m3",
    scale: torch.Tensor | float | None = None,
    *,
    dim: int | None = None,
    block: int | None = None,
    ste: bool = False,
) -> torch.Tensor:
    """Quantise-dequantise in the dtype of ``x`` (float arithmetic via :func:`round_fp8`, so works without float8 dtypes)."""
    fmt = _norm_fmt(fmt)
    if scale is None:
        s = compute_fp8_scale(x, fmt, dim, block)
    else:
        s = torch.as_tensor(scale, dtype=torch.float32, device=x.device)
    se = _expand_scale(s, x, block)
    d = (round_fp8(x.detach().float() / se, fmt) * se).to(x.dtype)
    return x + (d - x).detach() if ste else d
