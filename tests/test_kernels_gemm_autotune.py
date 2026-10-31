"""CPU tests of the autotuner (Plan 6.6) with fake run functions / fake clocks (no GPU needed)."""
from __future__ import annotations

import json

import pytest
import torch

from forge.core.errors import KernelCorrectnessError
from forge.kernels.autotune import (
    AutotuneCache,
    Autotuner,
    detect_throttle,
    enumerate_configs,
    is_feasible,
    shape_bucket,
    summarize,
    time_callable,
)
from forge.kernels.layouts import chip_from_sm
from forge.kernels.spec import load_spec

SHAPE = {"M": 4, "N": 64, "K": 128}
CHIP = chip_from_sm(90)


def fake_problem():
    """Oracle result + a family of 'kernels' whose speed is a fixed function of the config."""
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    w = torch.randn(64, 128)
    ref = x @ w.t()

    def ref_fn(shape):
        return ref

    return x, w, ref, ref_fn


def fixed_cfgs():
    return [
        {"tile_m": 16, "tile_n": 64, "tile_k": 64, "stages": 2, "warps": 4, "split_k": 1},
        {"tile_m": 16, "tile_n": 128, "tile_k": 64, "stages": 3, "warps": 4, "split_k": 1},
        {"tile_m": 32, "tile_n": 64, "tile_k": 128, "stages": 4, "warps": 8, "split_k": 2},
    ]


def timer_by_cfg(table):
    """Fake clock: median time (ms) per tile_n; slight jitter so IQR > 0 but ordering is stable."""
    def timer(fn, cfg):
        fn()
        base = table[cfg["tile_n"] * 1000 + cfg["tile_k"]]
        return [base * (1 + 0.01 * ((i * 7) % 5)) for i in range(15)]
    return timer


def make_tuner(tmp_path, **kw):
    kw.setdefault("telemetry", lambda: {})
    return Autotuner(tmp_path / "cache.json", chip=CHIP, **kw)


def test_summarize_median_and_iqr():
    s = summarize([1, 2, 3, 4, 5, 6, 7, 8, 9])
    assert s.median == 5 and s.q1 == 3 and s.q3 == 7 and s.iqr == 4 and s.min == 1 and s.max == 9 and s.n == 9
    assert summarize([2.5]).iqr == 0
    with pytest.raises(ValueError):
        summarize([])


def test_time_callable_counts_calls_and_flushes():
    calls, flushes = [], []
    out = time_callable(lambda: calls.append(1), warmup=3, repeats=7, flush=lambda: flushes.append(1))
    assert len(out) == 7 and len(calls) == 10 and len(flushes) == 7 and all(t >= 0 for t in out)


def test_tune_picks_fastest_correct_candidate_and_caches(tmp_path):
    x, w, ref, ref_fn = fake_problem()
    spec = load_spec("w4a16_gemm")
    table = {64064: 3.0, 128064: 1.0, 64128: 2.0}
    seen = []

    def run_fn(cfg, shape):
        seen.append(cfg["tile_n"])
        return x @ w.t()

    t = make_tuner(tmp_path, timer=timer_by_cfg(table))
    r = t.tune(spec, SHAPE, run_fn, ref_fn, candidates=fixed_cfgs(), dtype_cfg="t", layout_id="L")
    assert r.best == fixed_cfgs()[1] and r.cached and not r.from_cache and not r.rejected
    assert r.best_measurement.median_ms == pytest.approx(1.0, rel=0.05) and r.best_measurement.iqr_ms > 0
    assert len(r.measurements) == 3
    # second call is served from the cache without running the kernel
    n_before = len(seen)
    r2 = make_tuner(tmp_path, timer=timer_by_cfg(table)).tune(spec, SHAPE, run_fn, ref_fn, candidates=fixed_cfgs(), dtype_cfg="t", layout_id="L")
    assert r2.from_cache and r2.best == r.best and len(seen) == n_before
    assert t.lookup(spec, SHAPE, dtype_cfg="t", layout_id="L") == r.best
    assert t.lookup(spec, SHAPE, dtype_cfg="other", layout_id="L") is None
    # force re-tunes
    assert not t.tune(spec, SHAPE, run_fn, ref_fn, candidates=fixed_cfgs(), dtype_cfg="t", layout_id="L", force=True).from_cache


def test_wrong_kernel_is_rejected_and_never_cached(tmp_path):
    """C-040: the harness must reject a deliberately wrong kernel -- even if it is the fastest."""
    x, w, ref, ref_fn = fake_problem()
    spec = load_spec("w4a16_gemm")
    table = {64064: 3.0, 128064: 0.1, 64128: 2.0}  # the wrong one (tile_n=128) is by far the fastest

    def run_fn(cfg, shape):
        y = x @ w.t()
        return y * 1.05 if cfg["tile_n"] == 128 else y

    t = make_tuner(tmp_path, timer=timer_by_cfg(table))
    r = t.tune(spec, SHAPE, run_fn, ref_fn, candidates=fixed_cfgs())
    assert r.best["tile_n"] == 64 and r.best["tile_k"] == 128  # fastest *correct* one
    assert len(r.rejected) == 1 and r.rejected[0].config["tile_n"] == 128 and "oracle" in r.rejected[0].error
    cached = t.cache.entries()
    assert len(cached) == 1 and cached[0]["config"]["tile_n"] == 64

    # all candidates wrong: nothing selected, nothing cached
    t2 = make_tuner(tmp_path / "other", timer=timer_by_cfg(table))
    r2 = t2.tune(spec, SHAPE, lambda c, s: x @ w.t() + 1.0, ref_fn, candidates=fixed_cfgs())
    assert r2.best is None and not r2.cached and len(r2.rejected) == 3 and t2.cache.entries() == []
    with pytest.raises(KernelCorrectnessError):
        t2.tune(spec, SHAPE, lambda c, s: x @ w.t() + 1.0, ref_fn, candidates=fixed_cfgs(), raise_on_fail=True)


def test_candidate_exceptions_are_rejected_not_fatal(tmp_path):
    x, w, ref, ref_fn = fake_problem()
    spec = load_spec("w4a16_gemm")

    def run_fn(cfg, shape):
        if cfg["stages"] == 3:
            raise RuntimeError("out of resources")
        return x @ w.t()

    t = make_tuner(tmp_path, timer=lambda fn, cfg: [1.0 + cfg["stages"]] * 9)
    r = t.tune(spec, SHAPE, run_fn, ref_fn, candidates=fixed_cfgs())
    assert r.best["stages"] == 2 and len(r.rejected) == 1 and "out of resources" in r.rejected[0].error


def test_integer_outputs_must_be_bit_exact(tmp_path):
    spec = load_spec("int8_w8a8_gemm")
    ref = torch.arange(32, dtype=torch.int32).view(4, 8)

    def run_fn(cfg, shape):
        return ref + (1 if cfg["tile_n"] == 128 else 0)  # off by one: tolerable for floats, not for int paths

    t = make_tuner(tmp_path, timer=lambda fn, cfg: [1.0] * 9)
    r = t.tune(spec, SHAPE, run_fn, lambda s: ref, candidates=fixed_cfgs())
    assert all(m.config["tile_n"] == 64 for m in r.measurements) and len(r.rejected) == 1


def test_throttled_runs_are_remeasured_and_not_cached(tmp_path):
    x, w, ref, ref_fn = fake_problem()
    spec = load_spec("w4a16_gemm")
    clocks = iter([{"clock_mhz": 1800}, {"clock_mhz": 1000}] * 20)  # clock collapses during every measurement
    t = make_tuner(tmp_path, timer=lambda fn, cfg: [1.0] * 9, telemetry=lambda: next(clocks), retries=1)
    r = t.tune(spec, SHAPE, lambda c, s: x @ w.t(), ref_fn, candidates=fixed_cfgs()[:1])
    assert r.best is not None and r.best_measurement.throttled and not r.cached
    assert any("not cached" in n or "NOT cached" in n for n in r.notes) and t.cache.entries() == []


def test_throttle_prefers_clean_candidates(tmp_path):
    x, w, ref, ref_fn = fake_problem()
    spec = load_spec("w4a16_gemm")
    state = {"n": 0}

    def telemetry():
        state["n"] += 1
        return {"clock_mhz": 1800}

    def timer(fn, cfg):
        fn()
        slow = cfg["tile_n"] == 128  # the fastest-looking candidate shows a thermal ramp (drifting times)
        return [1.0 + (i * 0.2 if slow else 0.0) for i in range(12)] if slow else [2.0] * 12

    t = make_tuner(tmp_path, timer=timer, telemetry=telemetry, retries=0)
    r = t.tune(spec, SHAPE, lambda c, s: x @ w.t(), ref_fn, candidates=fixed_cfgs())
    assert r.best["tile_n"] == 64 and r.cached  # drift-flagged candidate lost to a clean one


def test_detect_throttle_signals():
    assert detect_throttle([1.0] * 12) == []
    drift = detect_throttle([1.0] * 4 + [1.1] * 4 + [1.5] * 4)
    assert drift and "drift" in drift[0]
    assert detect_throttle([1.0] * 3, {"clock_mhz": 1800}, {"clock_mhz": 1200})
    assert detect_throttle([1.0] * 3, {"clock_mhz": 1800}, {"clock_mhz": 1750}) == []
    assert detect_throttle([1.0] * 3, None, {"throttle": "hw_thermal_slowdown"})


def test_cache_key_buckets_layout_dtype_and_persistence(tmp_path):
    spec = load_spec("w4a16_gemm")
    t = make_tuner(tmp_path)
    k = lambda m, **kw: t.key_for(spec, {"M": m, "N": 4096, "K": 4096}, **kw)  # noqa: E731
    assert shape_bucket({"M": 3, "N": 7, "K": 9}) == {"M": 4, "N": 7, "K": 9}
    assert k(3) == k(4) and k(5) == k(8) and k(4) != k(5) and k(1) != k(2)
    assert k(16, layout_id="a") != k(16, layout_id="b") and k(16, dtype_cfg="x") != k(16, dtype_cfg="y")
    assert t.key_for(load_spec("w8a16_gemm"), SHAPE) != t.key_for(spec, SHAPE)
    other_chip = Autotuner(tmp_path / "c2.json", chip=chip_from_sm(80), telemetry=lambda: {})
    assert other_chip.key_for(spec, SHAPE) != t.key_for(spec, SHAPE)
    fields = t.key_fields(spec, SHAPE, dtype_cfg="d", layout_id="L")
    assert {"chip", "driver", "triton", "op", "dtype_cfg", "bucket", "layout_id"} <= set(fields)


def test_cache_file_roundtrip_corruption_and_regression(tmp_path):
    p = tmp_path / "a" / "cache.json"
    c = AutotuneCache(p)
    c.put("k1", {"op": "x"}, {"tile_m": 16}, {"median": 2.0, "iqr": 0.1, "n": 30})
    assert json.loads(p.read_text())["entries"]["k1"]["config"] == {"tile_m": 16}
    assert AutotuneCache(p).get("k1")["median_ms"] == 2.0 and len(AutotuneCache(p)) == 1
    assert c.check_regression("k1", 2.05)["regressed"] is False
    reg = c.check_regression("k1", 2.2)
    assert reg["regressed"] and reg["ratio"] == pytest.approx(1.1)
    assert c.check_regression("missing", 1.0)["regressed"] is False
    assert c.delete("k1") and not c.delete("k1")
    p.write_text("{not json", encoding="utf-8")
    assert AutotuneCache(p).entries() == []  # corrupt cache is only a cache
    c2 = AutotuneCache(p)
    c2.put("k", {}, {"a": 1}, {"median": 1, "iqr": 0, "n": 1})
    c2.clear()
    assert len(c2) == 0


def test_rank_fn_limits_measurements(tmp_path):
    x, w, ref, ref_fn = fake_problem()
    spec = load_spec("w4a16_gemm")
    cands = [{"tile_m": 16, "tile_n": 64, "tile_k": 64, "stages": 2, "warps": 4, "split_k": 1, "id": i} for i in range(30)]
    measured = []

    def run_fn(cfg, shape):
        return x @ w.t()

    def timer(fn, cfg):
        measured.append(cfg["id"])
        return [float(cfg["id"] + 1)] * 9

    t = make_tuner(tmp_path, timer=timer, top_n=3, random_extra=2)
    r = t.tune(spec, SHAPE, run_fn, ref_fn, candidates=cands, rank_fn=lambda c, s: c["id"])
    assert len(set(measured)) == 5 and {0, 1, 2} <= set(measured) and r.best["id"] == 0
    # without a rank function the sample is capped (seeded => reproducible)
    t2 = make_tuner(tmp_path / "x", timer=timer, max_candidates=7)
    measured.clear()
    assert len(t2.tune(spec, SHAPE, run_fn, ref_fn, candidates=cands).measurements) == 7


def test_tune_many_shapes_and_default_enumeration(tmp_path):
    x, w, ref, ref_fn = fake_problem()
    spec = load_spec("w4a16_gemm")
    t = make_tuner(tmp_path, timer=lambda fn, cfg: [cfg["tile_n"] / 64.0 + cfg["stages"]] * 9, max_candidates=6)
    res = t.tune(spec, [SHAPE, {"M": 32, "N": 64, "K": 128}], lambda c, s: x @ w.t(), lambda s: ref)
    assert isinstance(res, list) and len(res) == 2 and all(r.best for r in res)
    assert res[0].key != res[1].key and len(t.cache.entries()) == 2


def test_enumerate_configs_feasibility():
    spec = load_spec("w4a16_gemm")
    big = enumerate_configs(spec, chip_from_sm(90), {"M": 1, "N": 4096, "K": 4096})
    assert big and all(c["tile_m"] == 16 for c in big)  # skinny M -> no tile_m > 16 candidates
    assert any(c["split_k"] > 1 for c in big)  # 'auto' resolved for 132 SMs
    volta = enumerate_configs(spec, chip_from_sm(70), {"M": 64, "N": 4096, "K": 4096})
    assert volta and all(c["stages"] == 2 for c in volta)  # no cp.async ring pre-Ampere
    huge_tile = {"tile_m": 64, "tile_n": 256, "tile_k": 128, "stages": 5, "warps": 8, "split_k": 1}
    assert not is_feasible(huge_tile, spec, chip_from_sm(75), None) and is_feasible(huge_tile, spec, chip_from_sm(90), None)
    assert not is_feasible({**huge_tile, "tile_m": 8}, spec, chip_from_sm(90), None)
    assert not is_feasible({**huge_tile, "split_k": 8, "stages": 2}, spec, chip_from_sm(90), {"M": 1, "N": 64, "K": 128})
    assert len({tuple(c.values()) for c in big}) == len(big)
