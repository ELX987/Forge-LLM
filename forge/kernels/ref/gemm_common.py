"""Shared helpers of the GEMM oracles: tolerance rules (Plan 6.6 step 4, spec ``tolerance``) and comparison."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class CompareResult:
    """Outcome of :func:`compare` (truthy when the candidate is within tolerance)."""

    ok: bool
    max_abs_err: float
    ref_max_abs: float
    rel_err: float  # max_abs_err / ref_max_abs
    threshold: float
    n_nonfinite_mismatch: int = 0
    detail: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.ok


def _tol(tol: Any) -> tuple[float, float, float]:
    """(rel, abs, rel_norm) from a Tolerance model, dict or None (defaults of the Plan C.2 spec)."""
    if tol is None:
        return 2e-3, 1e-3, 2e-2
    if isinstance(tol, dict):
        return float(tol.get("rel", 2e-3)), float(tol.get("abs", 1e-3)), float(tol.get("rel_norm", 2e-2))
    return float(tol.rel), float(tol.abs), float(tol.rel_norm)


def compare(
    out: torch.Tensor,
    ref: torch.Tensor,
    tol: Any = None,
    *,
    out_dtype: torch.dtype | None = None,
    exact: bool = False,
    slack: float = 1.0,
) -> CompareResult:
    """Compare a candidate kernel output with the fp32 oracle.

    Float rule (both must hold, ``eps`` = machine epsilon of the *output* dtype)::

        max|out-ref| <= abs + max(rel, 2*eps) * max|ref|     (dtype-aware tolerance)
        max|out-ref| <= abs + rel_norm * max|ref|            (hard cap)

    Non-finite entries must match positionally (NaN<->NaN, +-inf<->+-inf) and are excluded from the error norm,
    so NaN-propagation conventions are testable.  ``exact=True`` demands bit equality (integer paths).
    ``slack`` multiplies both thresholds (for deliberately loose paths such as split-K reorderings of bf16 data).
    """
    rel, atol, rel_norm = _tol(tol)
    if out.shape != ref.shape:
        return CompareResult(False, float("inf"), 0.0, float("inf"), 0.0, detail=f"shape {tuple(out.shape)} != {tuple(ref.shape)}")
    if exact:
        eq = torch.equal(out.cpu(), ref.cpu().to(out.dtype)) if out.dtype != ref.dtype else torch.equal(out.cpu(), ref.cpu())
        return CompareResult(eq, 0.0 if eq else float("inf"), 0.0, 0.0 if eq else float("inf"), 0.0,
                             detail="bit-exact" if eq else "bit mismatch")
    o = out.detach().float().cpu()
    r = ref.detach().float().cpu()
    fin_o, fin_r = torch.isfinite(o), torch.isfinite(r)
    nonfinite_bad = int(((fin_o != fin_r) | (~fin_o & ~fin_r & ~((o == r) | (torch.isnan(o) & torch.isnan(r))))).sum())
    both = fin_o & fin_r
    if both.any():
        err = (o[both] - r[both]).abs().max().item()
        ref_max = r[both].abs().max().item()
    else:
        err, ref_max = 0.0, 0.0
    dt = out_dtype or out.dtype
    eps = torch.finfo(dt).eps if dt.is_floating_point else 0.0
    thr1 = atol + max(rel, 2.0 * eps) * ref_max
    thr2 = atol + rel_norm * ref_max
    thr = min(thr1, thr2) * slack
    ok = nonfinite_bad == 0 and err <= thr
    detail = f"max_abs_err={err:.3e} thr={thr:.3e} ref_max={ref_max:.3e} nonfinite_mismatch={nonfinite_bad}"
    return CompareResult(ok, err, ref_max, err / ref_max if ref_max else 0.0, thr, nonfinite_bad, detail)


def next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (int(x) - 1).bit_length()
