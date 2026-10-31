"""TR-2 FP4-training diagnostics as pure functions (Plan 3.4 TR-2, NVFP4 paper section 15.4).

* :func:`zero_grad_tile_stats`   -- fraction of 16x16 weight tiles whose gradient is entirely zero (+ element zero fraction)
* :func:`scale_stats`            -- E4M3 block-scale flush / subnormal fractions and exponent-range occupancy
* :func:`sr_bias_test`           -- empirical bias of stochastic rounding vs round-to-nearest on a tensor
* :func:`relative_loss_gap`      -- (loss - reference) / reference, scalar or aligned series
* :func:`low_norm_channel_experiment` / :func:`predicted_zero_fraction_2d` -- reproduction of the "low-norm channels inside
  16x16 tiles" failure: total MSE barely moves while whole channels are erased (paper Table 15.2, Exp. 12)
* :func:`update_weight_ratio`    -- ||dW|| / ||W|| over a parameter list
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from forge.train.fp4sim import (
    E4M3_MIN_NORMAL,
    FP4Rng,
    fake_quant_nvfp4,
)

__all__ = [
    "A_MAX_256", "low_norm_channel_experiment", "predicted_zero_fraction_2d", "relative_loss_gap", "scale_stats",
    "sr_bias_test", "update_weight_ratio", "zero_grad_tile_stats",
]

A_MAX_256 = 3.04  # mean maximum of 256 |N(0,1)| draws (paper A.6, Monte Carlo)


def zero_grad_tile_stats(grad: torch.Tensor, tile: int = 16) -> dict[str, float]:
    """Zero-gradient statistics of a 2D gradient: fraction of ``tile x tile`` tiles that are *all* zero, and of zero elements.

    Partial edge tiles are included (zero-padded).  Tensors with ndim != 2 are flattened to ``[-1, last]``.
    """
    g = grad.detach()
    g2 = g.reshape(-1, g.shape[-1]) if g.ndim != 2 else g
    nz = (g2 != 0)
    R, C = nz.shape
    pr, pc = (-R) % tile, (-C) % tile
    if pr or pc:
        nz = torch.nn.functional.pad(nz, (0, pc, 0, pr), value=False)
    tiles = nz.reshape(nz.shape[0] // tile, tile, nz.shape[1] // tile, tile).any(dim=3).any(dim=1)
    return {
        "zero_tile_frac": float((~tiles).float().mean()),
        "zero_elem_frac": float((g2 == 0).float().mean()),
        "n_tiles": float(tiles.numel()),
    }


def scale_stats(x: torch.Tensor, block: tuple[int, int] = (1, 16)) -> dict[str, float]:
    """E4M3 block-scale health of NVFP4 quantisation of ``x``.

    * ``flush_frac``      -- blocks with non-zero data whose E4M3 scale flushed to zero (block erased)
    * ``subnormal_frac``  -- blocks whose scale landed in the E4M3 subnormal range (reduced precision)
    * ``occupancy``       -- fraction of the 18 binary exponents of the E4M3 range [2^-9, 2^8.8] holding at least one block scale
    * ``log2_span``       -- log2(max/min) of the non-zero block scales (the usable range is ~2^17.8)
    * ``saturated_frac``  -- elements clipped at 6 after scaling (scale rounding down)
    """
    _, info = fake_quant_nvfp4(x, block, "rtn", return_info=True)
    stored = info.stored_scales.flatten()
    amax = info.block_amax.flatten()
    live = amax > 0
    n_live = max(1, int(live.sum()))
    flushed = live & (stored == 0)
    sub = live & (stored > 0) & (stored < E4M3_MIN_NORMAL)
    pos = stored[stored > 0]
    if pos.numel():
        exps = torch.floor(torch.log2(pos)).long()
        occupied = len(set(exps.tolist()))
        span = float(torch.log2(pos.max() / pos.min()))
    else:
        occupied, span = 0, 0.0
    return {
        "flush_frac": float(flushed.sum()) / n_live,
        "subnormal_frac": float(sub.sum()) / n_live,
        "occupancy": occupied / 18.0,
        "log2_span": span,
        "saturated_frac": float(info.saturated_frac),
        "n_blocks": float(stored.numel()),
    }


def sr_bias_test(x: torch.Tensor, n_draws: int = 32, block: tuple[int, int] = (1, 16), seed: int = 0) -> dict[str, float]:
    """Compare the mean of ``n_draws`` stochastic-rounding quantisations against ``x`` (and against round-to-nearest).

    ``rel_bias``  = ||mean_SR - x|| / ||x||: should be at the Monte-Carlo noise level ``expected_noise``.
    ``rtn_rel_err`` = relative error of the (deterministic) RTN result, for contrast: RTN's error does not average out.
    ``z`` = rel_bias / expected_noise: values >> 1 flag a *biased* stochastic rounder (e.g. a broken RNG or a dead zone).
    """
    xf = x.detach().to(torch.float32)
    rng = FP4Rng(seed)
    acc = torch.zeros_like(xf)
    sq = torch.zeros_like(xf)
    for _ in range(n_draws):
        q = fake_quant_nvfp4(xf, block, "sr", rng)
        acc += q
        sq += q * q
    mean = acc / n_draws
    denom = float(xf.norm()) + 1e-30
    rel_bias = float((mean - xf).norm()) / denom
    # standard error of the mean per element, from the sample variance of the draws
    var = (sq / n_draws - mean * mean).clamp(min=0) * (n_draws / max(1, n_draws - 1))
    expected_noise = float((var / n_draws).sum().sqrt()) / denom
    rtn = fake_quant_nvfp4(xf, block, "rtn")
    return {
        "rel_bias": rel_bias, "expected_noise": expected_noise, "z": rel_bias / max(expected_noise, 1e-30),
        "rtn_rel_err": float((rtn - xf).norm()) / denom, "n_draws": float(n_draws),
    }


def relative_loss_gap(loss: float | Sequence[float], reference: float | Sequence[float]) -> float | list[float]:
    """``(loss - reference) / reference`` for scalars or aligned sequences (published FP4 gap: 1-2% vs BF16/FP8)."""
    if isinstance(loss, (int, float)) and isinstance(reference, (int, float)):
        return (float(loss) - float(reference)) / float(reference)
    a, b = list(loss), list(reference)  # type: ignore[arg-type]
    if len(a) != len(b):
        raise ValueError("loss and reference series must have equal length")
    return [(x - y) / y for x, y in zip(a, b, strict=False)]


def update_weight_ratio(params_before: Sequence[torch.Tensor], params_after: Sequence[torch.Tensor]) -> float:
    """Global ``||W_after - W_before|| / ||W_before||``."""
    num = 0.0
    den = 0.0
    for b, a in zip(params_before, params_after, strict=False):
        num += float((a.detach().float() - b.detach().float()).pow(2).sum())
        den += float(b.detach().float().pow(2).sum())
    return math.sqrt(num) / (math.sqrt(den) + 1e-30)


def predicted_zero_fraction_2d(r: float, a_max: float = A_MAX_256) -> float:
    """Closed form (paper eq. 15.1): fraction of a low-norm row's entries flushed to zero in a 16x16 NVFP4 tile whose maximum is
    set by high-norm rows: ``2*Phi(a_max/(24*r)) - 1`` with ``r`` = row std / tile std."""
    return math.erf(a_max / (24.0 * r) / math.sqrt(2.0))


def low_norm_channel_experiment(rows: int = 1024, cols: int = 1024, low_fraction: float = 0.1, norm_ratio: float = 0.3, seed: int = 0) -> dict[str, Any]:
    """Reproduce paper Exp. 12 / Table 15.2: Gaussian weights where a random fraction of rows has std ``norm_ratio``.

    Returns zero fractions inside the low-norm rows for 1x16 and 16x16 NVFP4 weight scaling, relative MSE restricted to the low
    rows and over the whole matrix, and the closed-form prediction.  The point (TR-2): ``rel_mse_all`` is almost unchanged
    between 1D and 2D scaling while the low rows are erased -- loss/MSE-based diagnostics miss it; the zero count does not.
    """
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(rows, generator=g) < low_fraction
    c = torch.where(low, torch.full((rows,), float(norm_ratio)), torch.ones(rows))
    w = torch.randn(rows, cols, generator=g) * c[:, None]
    q1 = fake_quant_nvfp4(w, (1, 16))
    q2 = fake_quant_nvfp4(w, (16, 16))

    def stats(q: torch.Tensor) -> dict[str, float]:
        wl, ql = w[low], q[low]
        return {
            "zero_low": float((ql == 0).float().mean()),
            "zero_high": float((q[~low] == 0).float().mean()),
            "rel_mse_low": float(((ql - wl) ** 2).sum() / (wl ** 2).sum().clamp(min=1e-30)),
            "rel_mse_all": float(((q - w) ** 2).sum() / (w ** 2).sum()),
        }

    return {
        "1d": stats(q1), "2d": stats(q2), "predicted_zero_low_2d": predicted_zero_fraction_2d(norm_ratio),
        "low_rows": int(low.sum()), "norm_ratio": norm_ratio, "low_fraction": low_fraction,
    }
