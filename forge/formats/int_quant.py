"""Group-wise integer quantisation (symmetric / asymmetric), numerics shared by PTQ, QAT and kernel oracles.

Conventions (Plan CP-1/CP-3; GPTQ/AWQ-compatible)
------------------------------------------------
* Codes are **unsigned** (``uint8`` for ``bits <= 8``, ``int16`` above) in ``[0, 2**bits - 1]`` and
  ``dequant = (q - zero) * scale``.
* Symmetric: ``scale = amax / (2**(bits-1) - 1)`` per group and ``zero = 2**(bits-1)`` (offset-binary).  Absmax scaling produces codes in
  ``[1, 2**bits - 1]``; the most negative code ``0`` is representable but only reached by stochastic rounding / explicit clipping.
* Asymmetric: ``xmin = min(group.min, 0)``, ``xmax = max(group.max, 0)``, ``scale = (xmax - xmin) / (2**bits - 1)``,
  ``zero = round(-xmin / scale)`` (so that 0.0 is exactly representable, as in GPTQ).
* Groups run along ``dim``; ``group_size <= 0`` means one group per row of that dim ("per-channel").
  ``scale``/``zero`` have the shape of ``w`` with ``dim`` replaced by the number of groups.
* Rounding is round-half-even (``torch.round``) or stochastic (counter-based RNG, see :mod:`forge.formats.rng`).
"""
from __future__ import annotations

import torch

from forge.core.errors import ConfigError
from forge.formats.rng import counter_uniform


def int_range(bits: int, sym: bool = True) -> tuple[int, int]:
    """``(qmin, qmax)`` of the *unsigned* code space."""
    if not 1 <= bits <= 16:
        raise ConfigError(f"bits must be in [1, 16], got {bits}")
    return 0, (1 << bits) - 1


def code_dtype(bits: int) -> torch.dtype:
    return torch.uint8 if bits <= 8 else torch.int16 if bits < 16 else torch.int32


def _grouped(w: torch.Tensor, group_size: int, dim: int) -> tuple[torch.Tensor, int]:
    """Move ``dim`` last and split into groups -> ``[..., n_groups, G]``."""
    x = w.movedim(dim, -1)
    k = x.shape[-1]
    g = k if group_size is None or group_size <= 0 or group_size > k else int(group_size)
    if k % g:
        raise ConfigError(f"dimension of size {k} is not divisible by group_size {g}")
    return x.reshape(*x.shape[:-1], k // g, g), g


def _div_const(x: torch.Tensor, d: float) -> torch.Tensor:
    """``x / d`` as a true IEEE division on every device.

    ``x / python_scalar`` is not device-invariant in PyTorch: the CUDA kernel rewrites it as ``x * (1/d)`` (CPU scalar fast path), which can differ
    from the CPU result in the last ulp and therefore flip a rounding decision.  Dividing by a 0-dim tensor on ``x``'s device avoids that path."""
    return x / torch.full((), float(d), dtype=x.dtype, device=x.device)


def compute_qparams(
    xg: torch.Tensor, bits: int, sym: bool, clip_ratio: float | torch.Tensor = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """``scale, zero`` (float32, shape ``xg.shape[:-1] + (1,)``) for groups on the last dim of ``xg``."""
    xg = xg.float()
    maxq = (1 << bits) - 1
    if sym:
        amax = xg.abs().amax(dim=-1, keepdim=True) * clip_ratio
        scale = _div_const(amax, max((1 << (bits - 1)) - 1, 1))
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = torch.full_like(scale, float(1 << (bits - 1)))
    else:
        xmin = xg.amin(dim=-1, keepdim=True).clamp(max=0) * clip_ratio
        xmax = xg.amax(dim=-1, keepdim=True).clamp(min=0) * clip_ratio
        scale = _div_const(xmax - xmin, maxq)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        zero = torch.round(-xmin / scale).clamp(0, maxq)
    return scale, zero


def _round(v: torch.Tensor, rounding: str, seed: int, offset: int) -> torch.Tensor:
    if rounding == "nearest":
        return torch.round(v)
    if rounding == "stochastic":
        u = counter_uniform(v.shape, seed, offset, device=v.device)
        return torch.floor(v + u)
    raise ConfigError(f"unknown rounding {rounding!r} (nearest|stochastic)")


def quantize_with_params(
    xg: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, bits: int, rounding: str = "nearest", seed: int = 0, offset: int = 0
) -> torch.Tensor:
    """Float integer codes (still float32, clamped to the code range) for grouped input and given qparams."""
    maxq = (1 << bits) - 1
    return (_round(xg.float() / scale + zero, rounding, seed, offset)).clamp(0, maxq)


def mse_clip_search(
    xg: torch.Tensor, bits: int, sym: bool, grid: int = 20, max_shrink: float = 0.5
) -> torch.Tensor:
    """Per-group clip ratio in ``[1 - max_shrink, 1]`` minimising the reconstruction MSE (AWQ/GPTQ ``mse`` search).

    Returns a tensor shaped like the scale (``[..., n_groups, 1]``).
    """
    xg = xg.float()
    best = torch.full(xg.shape[:-1] + (1,), float("inf"), device=xg.device)
    best_r = torch.ones_like(best)
    for i in range(grid + 1):
        r = 1.0 - max_shrink * i / grid
        scale, zero = compute_qparams(xg, bits, sym, r)
        q = quantize_with_params(xg, scale, zero, bits)
        err = ((q - zero) * scale - xg).pow(2).sum(dim=-1, keepdim=True)
        better = err < best
        best = torch.where(better, err, best)
        best_r = torch.where(better, torch.full_like(best_r, r), best_r)
    return best_r


def quantize_int(
    w: torch.Tensor,
    bits: int,
    group_size: int = 128,
    sym: bool = True,
    dim: int = -1,
    *,
    rounding: str = "nearest",
    seed: int = 0,
    offset: int = 0,
    clip_search: bool = False,
    clip_ratio: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise ``w`` group-wise along ``dim``. Returns ``(q, scale, zero)`` (see module docstring).

    ``q`` has the shape of ``w`` and an unsigned integer dtype; ``scale`` is float32 and ``zero`` is ``uint8`` (``int16`` for
    >8 bit) with ``dim`` replaced by the number of groups.  ``clip_search=True`` picks a per-group clipping ratio by MSE.
    """
    xg, g = _grouped(w, group_size, dim)
    ratio: float | torch.Tensor = clip_ratio
    if clip_search:
        ratio = mse_clip_search(xg, bits, sym)
    scale, zero = compute_qparams(xg, bits, sym, ratio)
    q = quantize_with_params(xg, scale, zero, bits, rounding, seed, offset)
    dt = code_dtype(bits)
    q = q.reshape(*xg.shape[:-2], -1).to(dt).movedim(-1, dim)
    scale = scale.squeeze(-1).movedim(-1, dim).contiguous()
    zero = zero.squeeze(-1).movedim(-1, dim).to(dt).contiguous()
    return q.contiguous(), scale, zero


def dequantize_int(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor | None,
    group_size: int = 128,
    dim: int = -1,
    *,
    bits: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Inverse of :func:`quantize_int`. ``zero=None`` means symmetric (``2**(bits-1)``; ``bits`` then required)."""
    qd = q.movedim(dim, -1).float()
    k = qd.shape[-1]
    ng = scale.movedim(dim, -1).shape[-1]
    g = k // ng
    if group_size and group_size > 0 and group_size <= k and k % group_size == 0 and k // group_size == ng:
        g = group_size
    qg = qd.reshape(*qd.shape[:-1], ng, g)
    s = scale.movedim(dim, -1).float().unsqueeze(-1)
    if zero is None:
        if bits is None:
            raise ConfigError("dequantize_int: zero=None (symmetric) requires bits=")
        z = float(1 << (bits - 1))
    else:
        z = zero.movedim(dim, -1).float().unsqueeze(-1)
    out = ((qg - z) * s).reshape(*qd.shape).movedim(-1, dim)
    return out.to(dtype).contiguous()


def fake_quant_int(
    w: torch.Tensor,
    bits: int,
    group_size: int = 128,
    sym: bool = True,
    dim: int = -1,
    *,
    rounding: str = "nearest",
    seed: int = 0,
    offset: int = 0,
    clip_search: bool = False,
    ste: bool = False,
) -> torch.Tensor:
    """Quantise-dequantise in the dtype of ``w``. ``ste=True`` keeps the gradient of ``w`` (identity backward)."""
    q, s, z = quantize_int(w.detach(), bits, group_size, sym, dim, rounding=rounding, seed=seed, offset=offset, clip_search=clip_search)
    d = dequantize_int(q, s, z, group_size, dim, dtype=w.dtype)
    if ste:
        return w + (d - w).detach()
    return d


def quant_error(w: torch.Tensor, w_hat: torch.Tensor) -> dict[str, float]:
    """Relative Frobenius error, MSE and SQNR (dB) of an approximation."""
    wf, hf = w.float(), w_hat.float()
    err = (wf - hf).pow(2).sum().item()
    sig = wf.pow(2).sum().item()
    n = max(wf.numel(), 1)
    return {
        "rel_err": (err / sig) ** 0.5 if sig > 0 else 0.0,
        "mse": err / n,
        "sqnr_db": float(10.0 * torch.log10(torch.tensor(sig / err)).item()) if err > 0 and sig > 0 else float("inf"),
    }
