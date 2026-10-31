"""forge.kernels.autotune: correctness-gated, cached kernel autotuning (Plan 6.6).

``Autotuner(cache_path).tune(spec, shapes, run_fn, ref_fn) -> TuneResult`` -- see :mod:`forge.kernels.autotune.tuner`.
"""
from __future__ import annotations

from .cache import AutotuneCache, default_cache_path, environment_versions, make_key
from .space import enumerate_configs, is_feasible, normalise_shape, shape_bucket
from .timing import Stats, default_telemetry, detect_throttle, make_l2_flusher, summarize, time_callable
from .tuner import Autotuner, Measurement, TuneResult

__all__ = [
    "AutotuneCache", "Autotuner", "Measurement", "Stats", "TuneResult", "default_cache_path", "default_telemetry",
    "detect_throttle", "enumerate_configs", "environment_versions", "is_feasible", "make_key", "make_l2_flusher",
    "normalise_shape", "shape_bucket", "summarize", "time_callable",
]
