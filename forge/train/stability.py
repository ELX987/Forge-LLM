"""Loss-spike / grad-norm-spike detection and divergence bookkeeping (Plan 7.4 failure handling)."""
from __future__ import annotations

import collections
import math
import statistics
from typing import Any

__all__ = ["SpikeDetector", "TrainingDiverged"]

from forge.core.errors import ForgeError


class TrainingDiverged(ForgeError):
    """Training produced non-finite values / persistent spikes and the restart budget is exhausted."""

    code = "training_diverged"
    http_status = 500


class SpikeDetector:
    """Rolling-window spike detector.

    A loss is a *spike* when ``(loss - mean) / max(std, 0.02*|mean|) > zscore`` **and** ``loss > mean * ratio``; a gradient norm is a
    spike when it exceeds ``grad_factor`` times the trailing median.  Spiking values are not added to the window (so a run of
    spikes cannot normalise itself).  Detection needs ``min_samples`` clean observations first.
    """

    def __init__(self, window: int = 50, zscore: float = 6.0, ratio: float = 1.5, grad_factor: float = 10.0, min_samples: int | None = None) -> None:
        self.window, self.zscore, self.ratio, self.grad_factor = int(window), float(zscore), float(ratio), float(grad_factor)
        self.min_samples = int(min_samples if min_samples is not None else max(8, window // 4))
        self.losses: collections.deque[float] = collections.deque(maxlen=self.window)
        self.gnorms: collections.deque[float] = collections.deque(maxlen=self.window)

    def check(self, loss: float, grad_norm: float | None = None) -> dict[str, Any]:
        """Classify an observation (without recording it)."""
        out = {"loss_spike": False, "grad_spike": False, "z": 0.0}
        if len(self.losses) >= self.min_samples and math.isfinite(loss):
            mean = statistics.fmean(self.losses)
            std = statistics.pstdev(self.losses)
            denom = max(std, 0.02 * abs(mean), 1e-8)
            z = (loss - mean) / denom
            out["z"] = z
            out["loss_spike"] = bool(z > self.zscore and loss > mean * self.ratio)
        if (self.grad_factor > 0 and grad_norm is not None and len(self.gnorms) >= self.min_samples and math.isfinite(grad_norm)):
            med = statistics.median(self.gnorms)
            out["grad_spike"] = bool(med > 0 and grad_norm > self.grad_factor * med)
        return out

    def record(self, loss: float, grad_norm: float | None = None) -> None:
        if math.isfinite(loss):
            self.losses.append(float(loss))
        if grad_norm is not None and math.isfinite(grad_norm):
            self.gnorms.append(float(grad_norm))

    def reset(self) -> None:
        self.losses.clear()
        self.gnorms.clear()

    def state_dict(self) -> dict[str, Any]:
        return {"losses": list(self.losses), "gnorms": list(self.gnorms)}

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.losses = collections.deque(sd.get("losses", []), maxlen=self.window)
        self.gnorms = collections.deque(sd.get("gnorms", []), maxlen=self.window)
