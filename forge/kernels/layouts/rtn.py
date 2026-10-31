"""Round-to-nearest weight quantisers used by tests, benchmarks and ``make_packed_linear`` convenience paths.

These are the simple, deterministic baselines.  GPTQ / AWQ live in ``forge.compress`` and hand their integer
codes to :func:`forge.kernels.layouts.pack_w4`.
"""
from __future__ import annotations

import torch

from forge.core.errors import ConfigError

from .w4 import effective_group


def _grouped(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int, int]:
    n, k = w.shape
    gs = effective_group(group_size, k)
    g = -(-k // gs)
    pad = g * gs - k
    wf = w.float()
    if pad:
        # pad with NaN-free edge replication-free zeros but exclude them from min/max by masking below
        wf = torch.nn.functional.pad(wf, (0, pad), value=float("nan"))
    return wf.view(n, g, gs), g, gs


def quantize_w4_rtn(
    w: torch.Tensor, group_size: int = 128, sym: bool = True, scale_dtype: torch.dtype = torch.float16
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """RTN int4: returns ``(q uint8 [N,K], scales [N,G], zeros uint8 [N,G] | None)``.

    Symmetric: ``scale = amax/7``, codes ``clamp(round(w/scale) + 8, 0, 15)`` (zero-point 8 implied -> ``zeros=None``).
    Asymmetric: ``scale = (max-min)/15``, ``zero = round(-min/scale)``.
    """
    if w.dim() != 2:
        raise ConfigError("w must be [N, K]")
    n, k = w.shape
    wg, g, gs = _grouped(w, group_size)
    valid = ~torch.isnan(wg)
    wz = torch.where(valid, wg, torch.zeros_like(wg))
    if sym:
        amax = wz.abs().amax(dim=-1)
        scale = (amax / 7.0).clamp_min(1e-12)
        scale = scale.to(scale_dtype).float().clamp_min(1e-12)
        q = torch.round(wz / scale.unsqueeze(-1)).clamp(-8, 7) + 8
        zeros = None
    else:
        big = torch.finfo(torch.float32).max
        wmin = torch.where(valid, wg, torch.full_like(wg, big)).amin(dim=-1).clamp(max=0)
        wmax = torch.where(valid, wg, torch.full_like(wg, -big)).amax(dim=-1).clamp(min=0)
        scale = ((wmax - wmin) / 15.0).clamp_min(1e-12)
        scale = scale.to(scale_dtype).float().clamp_min(1e-12)
        zf = torch.round(-wmin / scale).clamp(0, 15)
        q = torch.round(wz / scale.unsqueeze(-1) + zf.unsqueeze(-1)).clamp(0, 15)
        zeros = zf.to(torch.uint8)
    q = q.reshape(n, g * gs)[:, :k].to(torch.uint8)
    return q.contiguous(), scale.to(scale_dtype), zeros


def quantize_w8_rtn(
    w: torch.Tensor, group_size: int = -1, scale_dtype: torch.dtype = torch.float16
) -> tuple[torch.Tensor, torch.Tensor]:
    """RTN symmetric int8: ``(q int8 [N,K], scales [N,G])`` with ``scale = amax/127``."""
    if w.dim() != 2:
        raise ConfigError("w must be [N, K]")
    n, k = w.shape
    wg, g, gs = _grouped(w, group_size)
    wz = torch.where(torch.isnan(wg), torch.zeros_like(wg), wg)
    scale = (wz.abs().amax(dim=-1) / 127.0).clamp_min(1e-12)
    scale = scale.to(scale_dtype).float().clamp_min(1e-12)
    q = torch.round(wz / scale.unsqueeze(-1)).clamp(-128, 127)
    return q.reshape(n, g * gs)[:, :k].to(torch.int8).contiguous(), scale.to(scale_dtype)
