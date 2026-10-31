"""Measurement hygiene (Plan 6.6 step 3): warm-up, L2 flush, median / IQR, throttle detection.

``time_callable`` uses CUDA events on GPU and ``perf_counter`` on CPU (so the whole autotuner is unit-testable
without a GPU).  Everything here is injectable into :class:`forge.kernels.autotune.Autotuner`.
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


BLOCKING_THROTTLE_REASONS = frozenset({"hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown"})


@dataclass
class Stats:
    n: int
    median: float
    q1: float
    q3: float
    iqr: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return {"n": self.n, "median": self.median, "q1": self.q1, "q3": self.q3, "iqr": self.iqr, "min": self.min, "max": self.max}


def summarize(times: list[float]) -> Stats:
    """Median and inter-quartile range (linear-interpolated quartiles)."""
    if not times:
        raise ValueError("no timings")
    xs = sorted(times)
    if len(xs) >= 2:
        q = statistics.quantiles(xs, n=4, method="inclusive")
        q1, q3 = q[0], q[2]
    else:
        q1 = q3 = xs[0]
    return Stats(len(xs), statistics.median(xs), q1, q3, q3 - q1, xs[0], xs[-1])


def detect_throttle(
    times: list[float],
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    *,
    drift: float = 1.2,
    clock_drop: float = 0.9,
) -> list[str]:
    """Reasons the run looks throttled (empty list = clean).

    * telemetry reports a throttle reason bit/flag after the run;
    * SM clock fell below ``clock_drop`` x the clock before the run;
    * the median of the last third of the samples is ``drift`` x slower than the first third (thermal ramp).
    """
    reasons: list[str] = []
    for snap, tag in ((before, "before"), (after, "after")):
        if snap and snap.get("throttle"):
            reasons.append(f"telemetry throttle flag {tag}: {snap.get('throttle')}")
    if before and after:
        c0, c1 = before.get("clock_mhz"), after.get("clock_mhz")
        if c0 and c1 and c1 < clock_drop * c0:
            reasons.append(f"sm clock dropped {c0:.0f} -> {c1:.0f} MHz")
    n = len(times)
    if n >= 9:
        third = n // 3
        first = statistics.median(times[:third])
        last = statistics.median(times[-third:])
        if first > 0 and last > drift * first:
            reasons.append(f"timing drift: last third {last:.4g} vs first third {first:.4g}")
    return reasons


def make_l2_flusher(l2_bytes: int | None = None) -> Callable[[], None] | None:
    """Return a function that overwrites a buffer bigger than L2 (None when CUDA is unavailable)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        size = int(max(l2_bytes or 0, 128 * 2**20))  # >= 128 MiB covers every shipping NVIDIA L2 (max ~126 MB)
        buf = torch.empty(size // 4, dtype=torch.int32, device="cuda")

        def flush() -> None:
            buf.zero_()

        return flush
    except Exception:  # pragma: no cover - needs a GPU
        return None


def time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    repeats: int = 30,
    flush: Callable[[], None] | None = None,
) -> list[float]:
    """Milliseconds of ``repeats`` calls of ``fn`` (after ``warmup`` untimed calls), L2 flushed before each."""
    cuda = False
    torch = None
    try:
        import torch as _t

        torch = _t
        cuda = bool(_t.cuda.is_available())
    except Exception:
        pass
    for _ in range(max(0, warmup)):
        fn()
    if cuda:  # pragma: no cover - needs a GPU
        torch.cuda.synchronize()
        out: list[float] = []
        for _ in range(repeats):
            if flush is not None:
                flush()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            out.append(float(s.elapsed_time(e)))
        return out
    out = []
    for _ in range(repeats):
        if flush is not None:
            flush()
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def default_telemetry(device: int = 0) -> Callable[[], dict[str, Any]]:
    """Snapshot function ``() -> {clock_mhz, temp_c, power_w, throttle}`` (empty dict when no GPU telemetry)."""

    def snap() -> dict[str, Any]:
        try:
            from forge.hw.telemetry import GpuSampler  # type: ignore

            rows = GpuSampler().sample()
            if rows:
                r = rows[min(device, len(rows) - 1)]
                out = {k: r.get(k) for k in ("clock_mhz", "temp_c", "power_w") if r.get(k) is not None}
                # sw_power_cap is the normal steady state of a loaded GPU; only thermal / hardware slowdowns invalidate a run
                blocking = [str(x) for x in (r.get("throttle_reasons") or []) if x in BLOCKING_THROTTLE_REASONS]
                if blocking:
                    out["throttle"] = ",".join(blocking)
                return out
        except Exception:
            pass
        return {}

    return snap
