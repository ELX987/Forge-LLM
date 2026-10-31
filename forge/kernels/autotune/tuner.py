"""The autotuner (Plan 6.6).

Protocol implemented::

    1. search space      : spec.search, infeasible points dropped analytically (smem / registers / async-copy)
    2. model-guided prune: optional ``rank_fn(cfg, shape) -> predicted time``; measure the top-N plus a seeded
                           random sample (to catch model errors)
    3. hygiene           : warm-up, L2 flush, median + IQR over ``repeats`` runs, telemetry snapshot before/after,
                           throttle detection (clock drop / flagged reason / timing drift) with re-measurement
    4. correctness gate  : every candidate's output is compared with the oracle (``compare``: tolerance rules of the
                           spec; integer outputs bit-exact).  Failing/raising candidates are recorded in
                           ``rejected`` and NEVER cached.
    5. cache key         : (chip fingerprint, driver/compiler versions, op, dtype config, shape bucket, layout_id)
    6. regression check  : ``Autotuner.regression`` compares a fresh median to the stored baseline (> 5% slower)

``run_fn(cfg, shape) -> Tensor`` executes the kernel for a config; ``ref_fn(shape) -> Tensor`` returns the oracle.
Both are plain callables, so the whole thing is testable with fakes on a CPU-only machine.
"""
from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.core.errors import KernelCorrectnessError
from forge.kernels.layouts.chip import chip_info
from forge.kernels.ref.gemm_common import compare

from .cache import AutotuneCache, default_cache_path, environment_versions, make_key
from .space import Config, enumerate_configs, normalise_shape, shape_bucket
from .timing import default_telemetry, detect_throttle, make_l2_flusher, summarize, time_callable

log = logging.getLogger("forge.kernels.autotune")


@dataclass
class Measurement:
    config: Config
    correct: bool | None = None
    median_ms: float | None = None
    iqr_ms: float | None = None
    min_ms: float | None = None
    n: int = 0
    times_ms: list[float] = field(default_factory=list)
    throttled: list[str] = field(default_factory=list)
    error: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["times_ms"] = list(self.times_ms)
        return d


@dataclass
class TuneResult:
    key: str
    key_fields: dict[str, Any]
    shape: dict[str, int]
    best: Config | None
    best_measurement: Measurement | None
    measurements: list[Measurement] = field(default_factory=list)
    rejected: list[Measurement] = field(default_factory=list)
    from_cache: bool = False
    cached: bool = False  # True if this call wrote the cache
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "key_fields": self.key_fields, "shape": self.shape, "best": self.best,
            "best_median_ms": self.best_measurement.median_ms if self.best_measurement else None,
            "from_cache": self.from_cache, "cached": self.cached,
            "n_measured": len(self.measurements), "n_rejected": len(self.rejected), "notes": self.notes,
            "rejected": [{"config": m.config, "error": m.error, "detail": m.detail} for m in self.rejected],
            "measurements": [m.to_dict() for m in self.measurements],
        }


class Autotuner:
    """Measure, gate and cache kernel configs.  See module docstring for the protocol."""

    def __init__(
        self,
        cache_path: str | Path | None = None,
        *,
        chip: Any = None,
        warmup: int = 5,
        repeats: int = 30,
        flush_l2: bool = True,
        timer: Callable[..., list[float]] | None = None,
        telemetry: Callable[[], dict[str, Any]] | None = None,
        retries: int = 2,
        seed: int = 0,
        max_candidates: int | None = 48,
        random_extra: int = 4,
        top_n: int = 8,
    ) -> None:
        self.chip = chip_info(chip)
        self.cache = AutotuneCache(cache_path if cache_path is not None else default_cache_path(self.chip.fingerprint))
        self.warmup, self.repeats, self.retries = warmup, repeats, retries
        self.flush_l2 = flush_l2
        self._timer = timer
        self._telemetry = telemetry if telemetry is not None else default_telemetry()
        self._rng = random.Random(seed)
        self.max_candidates, self.random_extra, self.top_n = max_candidates, random_extra, top_n
        self._flusher: Callable[[], None] | None | bool = False  # lazily created

    # -- keys ----------------------------------------------------------------------------------------
    def key_fields(self, spec: Any, shape: Any, *, dtype_cfg: str = "", layout_id: str = "") -> dict[str, Any]:
        env = environment_versions()
        return {
            "chip": self.chip.fingerprint,
            "driver": env.get("driver", "unknown"),
            "cuda": env.get("cuda", "none"),
            "triton": env.get("triton", "none"),
            "op": spec.kernel,
            "dtype_cfg": dtype_cfg,
            "bucket": shape_bucket(shape),
            "layout_id": layout_id,
        }

    def key_for(self, spec: Any, shape: Any, **kw: Any) -> str:
        return make_key(self.key_fields(spec, shape, **kw))

    def lookup(self, spec: Any, shape: Any, **kw: Any) -> Config | None:
        """Cached best config for this (op, bucket, dtype, layout) or ``None``."""
        e = self.cache.get(self.key_for(spec, shape, **kw))
        return dict(e["config"]) if e else None

    def regression(self, spec: Any, shape: Any, new_median_ms: float, *, tol: float = 0.05, **kw: Any) -> dict[str, Any]:
        return self.cache.check_regression(self.key_for(spec, shape, **kw), new_median_ms, tol)

    # -- measurement -----------------------------------------------------------------------------------
    def _get_flusher(self) -> Callable[[], None] | None:
        if self._flusher is False:
            self._flusher = make_l2_flusher() if self.flush_l2 else None
        return self._flusher  # type: ignore[return-value]

    def _time(self, fn: Callable[[], Any], cfg: Config) -> list[float]:
        if self._timer is not None:
            return list(self._timer(fn, cfg))
        return time_callable(fn, warmup=self.warmup, repeats=self.repeats, flush=self._get_flusher())

    def _measure(self, run_fn: Callable, cfg: Config, shape: dict[str, int]) -> Measurement:
        m = Measurement(config=cfg)
        for attempt in range(self.retries + 1):
            before = self._telemetry() if self._telemetry else {}
            times = self._time(lambda: run_fn(cfg, shape), cfg)
            after = self._telemetry() if self._telemetry else {}
            reasons = detect_throttle(times, before, after)
            st = summarize(times)
            m.times_ms, m.n = times, st.n
            m.median_ms, m.iqr_ms, m.min_ms = st.median, st.iqr, st.min
            m.throttled = reasons
            if not reasons:
                break
            log.warning("throttle detected for %s (attempt %d): %s", cfg, attempt + 1, "; ".join(reasons))
        return m

    def _gate(self, spec: Any, run_fn: Callable, ref: Any, cfg: Config, shape: dict[str, int], exact: bool | None) -> tuple[bool, str]:
        out = run_fn(cfg, shape)
        use_exact = exact if exact is not None else (not out.dtype.is_floating_point and spec.tolerance.bitexact_int_paths)
        res = compare(out, ref, spec.tolerance, out_dtype=out.dtype, exact=use_exact)
        return bool(res.ok), res.detail

    def _select(self, cands: list[Config], shape: dict[str, int], rank_fn: Callable | None) -> list[Config]:
        if rank_fn is not None:
            ranked = sorted(cands, key=lambda c: rank_fn(c, shape))
            head, tail = ranked[: self.top_n], ranked[self.top_n:]
            extra = self._rng.sample(tail, min(self.random_extra, len(tail))) if tail else []
            return head + extra
        if self.max_candidates is not None and len(cands) > self.max_candidates:
            return self._rng.sample(cands, self.max_candidates)
        return cands

    # -- main entry --------------------------------------------------------------------------------------
    def tune(
        self,
        spec: Any,
        shapes: Any,
        run_fn: Callable[[Config, dict[str, int]], Any],
        ref_fn: Callable[[dict[str, int]], Any],
        *,
        dtype_cfg: str = "",
        layout_id: str = "",
        candidates: list[Config] | None = None,
        rank_fn: Callable[[Config, dict[str, int]], float] | None = None,
        exact: bool | None = None,
        force: bool = False,
        group: int = 128,
        raise_on_fail: bool = False,
    ) -> TuneResult | list[TuneResult]:
        """Tune one shape (dict / ``(M,N,K)``) -> :class:`TuneResult`, or a list of shapes -> list of results."""
        if isinstance(shapes, list):
            return [self.tune(spec, s, run_fn, ref_fn, dtype_cfg=dtype_cfg, layout_id=layout_id, candidates=candidates,
                              rank_fn=rank_fn, exact=exact, force=force, group=group, raise_on_fail=raise_on_fail)
                    for s in shapes]  # type: ignore[misc]
        shape = normalise_shape(shapes)
        fields = self.key_fields(spec, shape, dtype_cfg=dtype_cfg, layout_id=layout_id)
        key = make_key(fields)
        if not force:
            hit = self.cache.get(key)
            if hit is not None:
                meas = Measurement(config=hit["config"], correct=True, median_ms=hit["median_ms"], iqr_ms=hit["iqr_ms"], n=hit["n"])
                return TuneResult(key, fields, shape, dict(hit["config"]), meas, from_cache=True)

        cands = list(candidates) if candidates is not None else enumerate_configs(spec, self.chip, shape, group=group)
        res = TuneResult(key, fields, shape, None, None)
        if not cands:
            res.notes.append("no feasible candidates")
            return res
        cands = self._select(cands, shape, rank_fn)
        ref = ref_fn(shape)
        good: list[Measurement] = []
        for cfg in cands:
            m = Measurement(config=cfg)
            try:
                ok, detail = self._gate(spec, run_fn, ref, cfg, shape, exact)
            except Exception as e:  # compile failure, OOM, illegal config...
                m.correct, m.error = False, f"{type(e).__name__}: {e}"
                res.rejected.append(m)
                continue
            m.correct, m.detail = ok, detail
            if not ok:
                m.error = "failed oracle comparison"
                res.rejected.append(m)
                log.warning("autotune: candidate %s rejected by correctness gate (%s)", cfg, detail)
                continue
            try:
                timed = self._measure(run_fn, cfg, shape)
            except Exception as e:
                m.error = f"{type(e).__name__}: {e}"
                res.rejected.append(m)
                continue
            timed.correct, timed.detail = True, detail
            res.measurements.append(timed)
            good.append(timed)
        if not good:
            res.notes.append("every candidate failed the correctness gate or raised; nothing cached")
            if raise_on_fail and res.rejected and all(r.correct is False for r in res.rejected):
                raise KernelCorrectnessError(f"all {len(res.rejected)} candidates failed for {spec.kernel} {shape}",
                                             details={"rejected": [r.error for r in res.rejected][:8]})
            return res
        clean = [g for g in good if not g.throttled]
        pool = clean or good
        best = min(pool, key=lambda g: (g.median_ms, g.iqr_ms or 0.0))
        res.best, res.best_measurement = dict(best.config), best
        if best.throttled:
            res.notes.append("best candidate was measured under throttling: result NOT cached")
        else:
            self.cache.put(key, fields, res.best, {"median": best.median_ms, "iqr": best.iqr_ms, "n": best.n})
            res.cached = True
        return res
