"""CPU tests (no GPU, no Triton launch) for the large-M dequantise + cuBLAS path: routing policy, chunking, candidates, the
host-sync-free launch path (act-order prepare), the autotuner's early abort / deadline, and the bench script's --quick plan,
time budget and incremental report writing."""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import textwrap
from pathlib import Path

import pytest
import torch

from forge.kernels import ops_gemm
from forge.kernels.autotune import Autotuner
from forge.kernels.layouts import MISSING_SORT, cached_group_sort, chip_from_sm, pack_w4, quantize_w4_rtn
from forge.kernels.spec import load_spec
from forge.kernels.w4_tuning import (
    DEQUANT_MIN_M,
    default_w4_config,
    dequant_budget_bytes,
    dequant_min_m,
    limit_candidates,
    normalise_w4_config,
    plan_dequant_chunks,
    resolve_w4_config,
    w4_candidates,
    w4_feasible,
)

@pytest.fixture(autouse=True)
def _isolate_global_rng():
    """These tests draw random tensors; keep the global torch RNG stream of the rest of the suite untouched (some older tests
    use unseeded random inputs with tight tolerances)."""
    with torch.random.fork_rng():
        yield


CHIPS = [(None, 0), (70, 80), (80, 108), (86, 82), (89, 128), (90, 132), (100, 148), (120, 188)]


def _load_bench():
    spec = importlib.util.spec_from_file_location("bench_w4a16_t", Path(__file__).resolve().parent.parent / "scripts" / "bench_w4a16.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------------------------
# policy
# ------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sm,sms", CHIPS)
@pytest.mark.parametrize("n,k", [(5120, 5120), (5120, 17408), (151936, 5120)])
def test_default_routing_by_m(sm, sms, n, k, monkeypatch):
    monkeypatch.delenv("FORGE_W4_DEQUANT_MIN_M", raising=False)
    measured = sm is not None and sm >= 120  # sm_120: the user's sweep found dot winning from M=2 (see test_measured_sm120_defaults)
    for m in (1,) if measured else (1, 2, 4, 8):
        c = default_w4_config(m, n, k, sm=sm, num_sms=sms)
        assert c["impl"] == "gemv" and c["tile_m"] == m and w4_feasible(c, sm=sm), c
    for m in (9, 16, 17, 31, 32, 33, 63):
        c = default_w4_config(m, n, k, sm=sm, num_sms=sms)
        assert c["impl"] == "dot" and c["tile_m"] >= 16 and w4_feasible(c, sm=sm), c
    for m in (64, 128, 256, 1024, 8192):
        c = default_w4_config(m, n, k, sm=sm, num_sms=sms)
        assert c["impl"] == "dequant_matmul" and normalise_w4_config(c, m) == c and w4_feasible(c, sm=sm), c
        assert c["split_k"] == 1 and c["tile_k"] % 8 == 0


def test_threshold_is_per_chip_and_overridable(monkeypatch):
    monkeypatch.delenv("FORGE_W4_DEQUANT_MIN_M", raising=False)
    assert dequant_min_m(120) == DEQUANT_MIN_M == 64
    assert default_w4_config(32, 5120, 5120, sm=120)["impl"] == "dot"
    assert default_w4_config(64, 5120, 5120, sm=120)["impl"] == "dequant_matmul"
    monkeypatch.setenv("FORGE_W4_DEQUANT_MIN_M", "32")
    assert dequant_min_m(120) == 32
    assert default_w4_config(32, 5120, 5120, sm=120)["impl"] == "dequant_matmul"
    monkeypatch.setenv("FORGE_W4_DEQUANT_MIN_M", "0")  # disables the routing entirely
    assert default_w4_config(4096, 5120, 5120, sm=120)["impl"] == "dot"
    monkeypatch.setenv("FORGE_W4_DEQUANT_MIN_M", "garbage")
    assert dequant_min_m(120) == 64


def test_explicit_impl_wins_over_routing(monkeypatch):
    monkeypatch.delenv("FORGE_W4_DEQUANT_MIN_M", raising=False)
    assert resolve_w4_config(1024, 5120, 5120, {"impl": "dot", "tile_m": 128}, sm=120)["impl"] == "dot"
    assert resolve_w4_config(1024, 5120, 5120, {"impl": "legacy"}, sm=120)["impl"] == "legacy"
    assert resolve_w4_config(1, 5120, 5120, {"impl": "dequant_matmul", "tile_n": 64}, sm=120)["tile_n"] == 64


def test_dequant_config_is_clamped():
    c = normalise_w4_config({"impl": "dequant_matmul", "tile_n": 1000, "tile_k": 3, "warps": 64, "stages": 9, "split_k": 5}, 100)
    assert c["tile_n"] == 128 and c["tile_k"] == 64 and c["warps"] == 8 and c["stages"] == 1 and c["split_k"] == 1
    assert not w4_feasible(dict(impl="dequant_matmul", tile_m=16, tile_n=128, tile_k=512, warps=4, stages=1, split_k=1))


@pytest.mark.parametrize("m", [8, 9, 16, 32, 64, 256, 1024])
def test_candidates_include_dequant_floor_and_legacy(m):
    cands = w4_candidates(m, 5120, 5120, sm=120, num_sms=188)
    impls = {c["impl"] for c in cands}
    assert "dequant_matmul" in impls
    assert ("legacy" in impls) == (m >= 9)
    assert default_w4_config(m, 5120, 5120, sm=120, num_sms=188) in cands
    assert all(w4_feasible(c, sm=120) and normalise_w4_config(c, m) == c for c in cands)
    assert not any(c["impl"] == "dequant_matmul" for c in w4_candidates(1, 5120, 5120, sm=120, num_sms=188))


@pytest.mark.parametrize("m", [1, 8, 16, 64, 1024])
def test_limit_candidates_keeps_default_and_every_impl(m):
    cands = w4_candidates(m, 5120, 17408, sm=120, num_sms=188, level="full")
    lim = limit_candidates(cands, 8)
    assert len(lim) <= 8 and lim[0] == cands[0]
    assert {c["impl"] for c in lim} == {c["impl"] for c in cands} or len(lim) == 8
    assert len({tuple(sorted(c.items())) for c in lim}) == len(lim)
    assert limit_candidates(cands, 0) == cands and limit_candidates(cands[:3], 8) == cands[:3]


# ------------------------------------------------------------------------------------------------------------------
# chunking / memory budget
# ------------------------------------------------------------------------------------------------------------------
def test_lm_head_is_chunked_under_the_default_budget(monkeypatch):
    monkeypatch.delenv("FORGE_W4_DEQUANT_MB", raising=False)
    assert dequant_budget_bytes() == 1 << 30
    n, k, item = 151936, 5120, 2  # 1.45 GiB of bf16 > 1 GiB
    chunks = plan_dequant_chunks(n, k, item, dequant_budget_bytes())
    assert len(chunks) == 2 and chunks[0][0] == 0 and chunks[-1][1] == n
    assert all(r1 > r0 and (r1 - r0) * k * item <= 1 << 30 for r0, r1 in chunks)
    assert all(r0 % 64 == 0 for r0, _ in chunks)  # chunk starts are multiples of 8 (packed zero words)
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:], strict=False))


def test_chunk_planner_edges():
    assert plan_dequant_chunks(0, 128, 2, 1 << 20) == []
    assert plan_dequant_chunks(100, 128, 2, 1 << 30) == [(0, 100)]
    assert plan_dequant_chunks(200, 64, 2, 64 * 64 * 2 + 1) == [(0, 64), (64, 128), (128, 192), (192, 200)]
    assert plan_dequant_chunks(130, 1 << 20, 2, 1) == [(0, 64), (64, 128), (128, 130)]  # a row group never exceeds the floor of 64 rows
    cover = plan_dequant_chunks(12345, 777, 2, 5 << 20)
    assert sum(r1 - r0 for r0, r1 in cover) == 12345 and cover[0][0] == 0 and cover[-1][1] == 12345


def test_budget_env_and_config(monkeypatch):
    monkeypatch.setenv("FORGE_W4_DEQUANT_MB", "256")
    assert dequant_budget_bytes() == 256 << 20
    assert dequant_budget_bytes({"chunk_mb": 64}) == 64 << 20  # config wins
    monkeypatch.setenv("FORGE_W4_DEQUANT_MB", "nonsense")
    assert dequant_budget_bytes() == 1 << 30


# ------------------------------------------------------------------------------------------------------------------
# no host sync in the launch path
# ------------------------------------------------------------------------------------------------------------------
SYNC_ATTRS = {"cpu", "item", "tolist", "numpy", "synchronize", "wait_stream", "__array__"}


def _sync_calls(fn) -> list[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return [f"{ast.unparse(n.func)}" for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SYNC_ATTRS]


def test_launch_path_source_has_no_host_sync_calls():
    from forge.kernels.layouts import w4 as layouts_w4
    from forge.kernels.triton import gemm_w4_dequant as dq
    from forge.kernels.triton import gemm_w4_fast as fast
    from forge.kernels.triton import gemm_w4a16 as top

    launch = [top.w4a16_gemm_triton, top.group_sorted, top._capturing, fast.w4_fast_gemm, fast._resolve, dq.w4_dequant,
              dq.w4_dequant_matmul, dq.scale_mode, ops_gemm.w4a16_linear, ops_gemm.tuned_config, ops_gemm._portable,
              layouts_w4.cached_group_sort]
    for fn in launch:
        assert _sync_calls(fn) == [], f"{fn.__module__}.{fn.__name__} syncs the host"
    # the check is meaningful: the prepare-time sort DOES sync (and is therefore never on the launch path)
    assert any(c.endswith(".cpu") for c in _sync_calls(layouts_w4.sort_columns_by_group))


def _actorder_pack(n=40, k=256, gs=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, s, z = quantize_w4_rtn(torch.randn(n, k, generator=g), gs, sym=False)
    gi = (torch.arange(k) // gs)[torch.randperm(k, generator=g)].to(torch.int32)
    return pack_w4(q, s, z, gs, None, gi)


class _Boom(RuntimeError):
    pass


@pytest.fixture()
def no_sync(monkeypatch):
    def boom(*a, **k):
        raise _Boom("device->host sync in the launch path")

    for name in ("cpu", "item", "tolist", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, boom)
    monkeypatch.setattr(torch.cuda, "synchronize", boom)


def test_prepared_actorder_launch_is_sync_free(monkeypatch):
    import forge.kernels.triton.gemm_w4_dequant as dq
    import forge.kernels.triton.gemm_w4_fast as fast
    import forge.kernels.triton.gemm_w4a16 as top

    monkeypatch.setattr(top, "HAS_TRITON", True)
    p = _actorder_pack()
    top.prepare_w4(p)  # sync allowed here
    calls = []
    monkeypatch.setattr(fast, "w4_fast_gemm", lambda x, *a, **k: (calls.append(("fast", a[4] is not None)), torch.zeros(x.shape[0], k["n"], dtype=x.dtype))[1])
    monkeypatch.setattr(dq, "w4_dequant_matmul", lambda x, *a, **k: (calls.append(("dq", a[4] is not None)), torch.zeros(x.shape[0], k["n"], dtype=x.dtype))[1])
    x1, x64 = torch.randn(1, 256).half(), torch.randn(64, 256).half()
    with monkeypatch.context() as mp:
        for name in ("cpu", "item", "tolist", "numpy"):
            mp.setattr(torch.Tensor, name, lambda *a, _n=name, **k: (_ for _ in ()).throw(_Boom(_n)))
        y1 = top.w4a16_gemm_triton(x1, p, None)
        y64 = top.w4a16_gemm_triton(x64, p, None)
    assert y1.shape == (1, 40) and y64.shape == (64, 40)
    assert calls == [("fast", True), ("dq", True)]  # gemv got the precomputed permutation; dequant_matmul got g_idx directly


def test_cold_cache_under_graph_capture_falls_back_without_sync(monkeypatch):
    import forge.kernels.triton.gemm_w4a16 as top

    monkeypatch.setattr(top, "HAS_TRITON", True)
    monkeypatch.setattr(top, "_capturing", lambda t: True)  # pretend a CUDA graph is being captured
    p = _actorder_pack()  # NOT prepared
    assert cached_group_sort(p, compute=False) is MISSING_SORT
    legacy = []
    monkeypatch.setattr(top, "wq_gemm", lambda x, *a, **k: (legacy.append(1), torch.zeros(x.shape[0], p.n, dtype=x.dtype))[1])
    with monkeypatch.context() as mp:
        for name in ("cpu", "item", "tolist", "numpy"):
            mp.setattr(torch.Tensor, name, lambda *a, _n=name, **k: (_ for _ in ()).throw(_Boom(_n)))
        top.w4a16_gemm_triton(torch.randn(1, 256).half(), p, None)
    assert legacy == [1]
    assert cached_group_sort(p, compute=False) is MISSING_SORT  # nothing was computed lazily during "capture"


def test_prepare_fills_the_cache_once_and_survives_in_place_edits():
    p = _actorder_pack()
    assert cached_group_sort(p, compute=False) is MISSING_SORT
    top_prepare = ops_gemm.prepare_w4(p)
    assert top_prepare is p
    first = cached_group_sort(p, compute=False)
    assert first is not MISSING_SORT and first[1].dtype == torch.int32
    assert cached_group_sort(p, compute=False) is first  # same object: no recomputation
    p.qweight.add_(0)  # in-place op bumps the tensor version -> stale
    assert cached_group_sort(p, compute=False) is MISSING_SORT


def test_kernel_module_prepares_at_load_time():
    p = _actorder_pack()
    m = ops_gemm.KernelW4A16Linear.from_packed(p)
    assert cached_group_sort(m.packed(), compute=False) is not MISSING_SORT  # from_packed prepared it
    m.half()  # .to(): scales re-created -> re-prepared by _apply
    assert cached_group_sort(m.packed(), compute=False) is not MISSING_SORT
    m2 = ops_gemm.KernelW4A16Linear(p.k, p.n, False, torch.float16, None, group_size=p.group_size, sym=False, has_g_idx=True, n_groups=p.num_groups)
    assert cached_group_sort(m2.packed(), compute=False) is MISSING_SORT  # a freshly built (empty) module has nothing prepared
    m2.load_state_dict(m.state_dict())  # post-hook prepares
    assert cached_group_sort(m2.packed(), compute=False) is not MISSING_SORT
    import pickle

    pickle.dumps(m2)  # the load hook is a module-level function, so modules stay picklable


# ------------------------------------------------------------------------------------------------------------------
# autotuner: early abort, deadline, accept_throttled
# ------------------------------------------------------------------------------------------------------------------
def _fake_tuner(tmp_path, times_by_tn, **kw):
    ref = torch.ones(2, 2)
    calls = {"full": 0, "probe": 0}

    def timer(fn, cfg):
        fn()
        calls["full"] += 1
        return [times_by_tn[cfg["tile_n"]]] * 15

    timer.probe = lambda fn, cfg: (calls.__setitem__("probe", calls["probe"] + 1), [times_by_tn[cfg["tile_n"]]] * 3)[1]
    t = Autotuner(tmp_path / "c.json", chip=chip_from_sm(90), timer=timer, telemetry=kw.pop("telemetry", lambda: {}), **kw)
    return t, ref, calls


def _cfgs(*tns):
    return [dict(impl="dot", tile_m=16, tile_n=tn, tile_k=128, warps=4, stages=2, split_k=1) for tn in tns]


def test_tuner_early_abort_skips_clearly_slower_candidates(tmp_path):
    t, ref, calls = _fake_tuner(tmp_path, {16: 1.0, 32: 5.0, 64: 1.5, 128: 100.0})
    res = t.tune(load_spec("w4a16_gemm"), {"M": 4, "N": 64, "K": 128}, lambda cfg, s: ref, lambda s: ref, candidates=_cfgs(16, 32, 64, 128),
                 prune_factor=2.0)
    assert res.best["tile_n"] == 16
    assert sorted(m.config["tile_n"] for m in res.pruned) == [32, 128]  # > 2x slower than 1.0
    assert sorted(m.config["tile_n"] for m in res.measurements) == [16, 64]
    assert calls["full"] == 2 and calls["probe"] == 3  # the pruned ones never got the full measurement
    assert res.to_dict()["n_pruned"] == 2


def test_tuner_deadline_returns_partial_result(tmp_path, monkeypatch):
    import types

    import forge.kernels.autotune.tuner as tuner_mod

    t, ref, calls = _fake_tuner(tmp_path, {16: 1.0, 32: 0.5, 64: 0.1})
    now = {"t": 0.0}
    monkeypatch.setattr(tuner_mod, "time", types.SimpleNamespace(monotonic=lambda: now["t"]))
    inner = t._timer

    def slow_timer(fn, cfg):  # every full measurement "takes" 10 s of wall clock
        now["t"] += 10.0
        return inner(fn, cfg)

    t._timer = slow_timer
    res = t.tune(load_spec("w4a16_gemm"), {"M": 4, "N": 64, "K": 128}, lambda cfg, s: ref, lambda s: ref, candidates=_cfgs(16, 32, 64), deadline=5.0)
    assert res.best is not None and len(res.measurements) == 1 and any("time budget" in n for n in res.notes)
    assert res.cached  # the partial winner is still written


def test_tuner_accept_throttled(tmp_path):
    t, ref, _ = _fake_tuner(tmp_path, {16: 1.0}, telemetry=lambda: {"throttle": "hw_slowdown"}, retries=1)
    spec, shape = load_spec("w4a16_gemm"), {"M": 4, "N": 64, "K": 128}
    r1 = t.tune(spec, shape, lambda cfg, s: ref, lambda s: ref, candidates=_cfgs(16))
    assert r1.best and not r1.cached and r1.best_measurement.throttled  # default: not cached
    r2 = t.tune(spec, shape, lambda cfg, s: ref, lambda s: ref, candidates=_cfgs(16), accept_throttled=True)
    assert r2.cached
    entry = t.cache.entries()[0]
    assert entry["throttled"] is True


# ------------------------------------------------------------------------------------------------------------------
# bench script: --quick plan, budget, incremental report
# ------------------------------------------------------------------------------------------------------------------
def test_quick_plan_and_overrides(capsys):
    b = _load_bench()
    args = b.build_parser().parse_args(["--quick"])
    pl = b.resolve_plan(args)
    assert pl["ms"] == [1, 8, 16, 64, 256, 1024] and [s[1:] for s in pl["shapes"]] == [(5120, 5120), (5120, 17408)]
    assert pl["max_candidates"] == 8 and pl["refine"] == 0 and pl["reps"] < 9 and pl["prune_factor"] == 2.0
    rows = b.plan(pl["shapes"], pl["ms"], 128, "quick", 120, 188, pl["max_candidates"])
    assert len(rows) == 12 and all(0 < r["n_candidates"] <= 8 for r in rows)
    assert "dequant_matmul" in {i for r in rows if r["m"] >= 64 for i in r["impls"]}
    full = b.resolve_plan(b.build_parser().parse_args([]))
    assert len(full["shapes"]) == 5 and full["max_candidates"] == 0 and full["refine"] == 1
    mix = b.resolve_plan(b.build_parser().parse_args(["--quick", "--m", "2,4", "--shapes", "64:128", "--refine", "2", "--no-prune", "--time-budget-min", "1.5"]))
    assert mix["ms"] == [2, 4] and mix["shapes"][0][1:] == (64, 128) and mix["refine"] == 2 and mix["prune_factor"] is None and mix["budget_s"] == 90.0
    assert b.main(["--dry-run", "--quick"]) == 0
    out = capsys.readouterr().out
    assert "mode: quick" in out and "total candidates:" in out and "dequant_matmul" in out


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class FakeHooks:
    def __init__(self, clock, cost=60.0, fail_on=None, interrupt_on=None):
        self.clock, self.cost, self.fail_on, self.interrupt_on = clock, cost, fail_on, interrupt_on
        self.events = []

    def setup(self, role, n, k):
        self.events.append(("setup", role))
        return {"role": role}

    def teardown(self, st):
        self.events.append(("teardown", st["role"]))

    def row(self, st, role, n, k, m, budget):
        if (role, m) == self.fail_on:
            raise RuntimeError("boom")
        if (role, m) == self.interrupt_on:
            raise KeyboardInterrupt
        self.clock.t += self.cost
        best = {"impl": "gemv" if m <= 8 else ("dot" if m < 32 else "dequant_matmul")}
        return dict(role=role, n=n, k=k, m=m, best=best, bf16_us=100.0, tuned_us=50.0 * (1 if m <= 16 else 3), legacy_us=170.0, default_us=100.0,
                    speedup_vs_bf16=2.0 if m <= 16 else 0.8)


def test_budget_expiry_stops_gracefully_and_report_is_incremental(tmp_path):
    b = _load_bench()
    clock = FakeClock()
    sink = b.ReportSink(tmp_path / "r.json", {"chip": "fake"})
    assert json.loads((tmp_path / "r.json").read_text())["rows"] == [] and json.loads((tmp_path / "r.json").read_text())["partial"] is True
    hooks = FakeHooks(clock, cost=60.0)
    seen = []
    orig_add = sink.add_row

    def add_and_check(row):
        orig_add(row)
        data = json.loads((tmp_path / "r.json").read_text())  # valid JSON with every row so far, after EVERY row
        seen.append(len(data["rows"]))

    sink.add_row = add_and_check
    budget = b.Budget(150.0, clock=clock)  # 2.5 rows worth
    logs = []
    stopped = b.sweep([("a", 8, 8), ("b", 8, 8)], [1, 8, 64], hooks, sink, budget, log=logs.append)
    sink.finish(stopped)
    assert stopped == "time budget" and seen == [1, 2, 3]
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["partial"] is True and data["stopped"] == "time budget" and len(data["rows"]) == 3
    assert ("teardown", "a") in hooks.events and ("setup", "b") not in hooks.events  # shape a cleaned up, b never started
    assert data["summary"]["n_ok"] == 3 and any("time budget" in line for line in logs)


def test_failed_row_and_keyboard_interrupt_keep_partial_results(tmp_path):
    b = _load_bench()
    clock = FakeClock()
    sink = b.ReportSink(tmp_path / "r.json", {})
    stopped = b.sweep([("a", 8, 8)], [1, 8, 16], FakeHooks(clock, fail_on=("a", 8)), sink, b.Budget(0, clock=clock), log=lambda s: None)
    sink.finish(stopped)
    rows = json.loads((tmp_path / "r.json").read_text())["rows"]
    assert stopped is None and [("error" in r) for r in rows] == [False, True, False]
    sink2 = b.ReportSink(tmp_path / "r2.json", {})
    stopped = b.sweep([("a", 8, 8)], [1, 8, 16], FakeHooks(clock, interrupt_on=("a", 16)), sink2, b.Budget(0, clock=clock), log=lambda s: None)
    sink2.finish(stopped)
    d2 = json.loads((tmp_path / "r2.json").read_text())
    assert stopped == "interrupted" and len(d2["rows"]) == 2 and d2["partial"] is True


def test_summary_best_impl_geomean_and_threshold():
    b = _load_bench()

    def row(m, impl, sp, tuned, dq=None, role="x", dflt=100.0, leg=170.0):
        return dict(role=role, m=m, best={"impl": impl}, speedup_vs_bf16=sp, tuned_us=tuned, dequant_us=dq, default_us=dflt, legacy_us=leg, throttled=m == 1)

    rows = [row(1, "gemv", 2.0, 50), row(1, "gemv", 8.0, 10, role="y"), row(16, "dot", 1.3, 90), row(32, "dot", 1.0, 160, dq=300), row(64, "dequant_matmul", 0.9, 400),
            row(256, "dequant_matmul", 0.8, 900), row(1024, "dequant_matmul", 0.7, 5000)]
    s = b.summarize_rows(rows)
    assert s["by_m"][1]["best_impl"] == "gemv" and s["by_m"][1]["speedup_vs_bf16"] == pytest.approx(4.0)
    assert s["decode_geomean"] == pytest.approx((2.0 * 8.0 * 1.3) ** (1 / 3))
    assert s["prefill_ratio"] == pytest.approx((1.0 * 0.9 * 0.8 * 0.7) ** 0.25)
    assert s["suggested_dequant_min_m"] == 64 and s["n_throttled"] == 2
    rows.append(row(2048, "dot", 0.2, 10, role="z"))
    assert b.summarize_rows(rows)["suggested_dequant_min_m"] is None  # does not win at the top M
    assert b.summarize_rows([row(64, "dequant_matmul", 0.9, 400, dflt=300, leg=170)])["default_slower_than_legacy"] == [("x", 64)]
    text = "\n".join(b.format_summary(s))
    assert "best impl per M" in text and "decode (M<=16)" in text and "prefill (M>=32)" in text and "FORGE_W4_DEQUANT_MIN_M=64" in text
    assert b.summarize_rows([])["decode_geomean"] is None and b.format_summary(b.summarize_rows([]))


def test_budget_is_unlimited_when_zero():
    b = _load_bench()
    clock = FakeClock()
    bud = b.Budget(0, clock=clock)
    clock.t = 1e9
    assert not bud.expired() and bud.deadline is None
    assert b.Budget(10, clock=clock).deadline == 1e9 + 10


# ------------------------------------------------------------------------------------------------------------------
# review fixes: cache key, resolve seeding, measured defaults, dequant memory, bench robustness
# ------------------------------------------------------------------------------------------------------------------
def test_tune_tag_separates_act_order_and_symmetry():
    from forge.kernels.w4_tuning import W4_TUNE_TAG, w4_tune_tag

    assert W4_TUNE_TAG == "k4"
    tags = {w4_tune_tag(ao, sym) for ao in (False, True) for sym in (False, True)}
    assert len(tags) == 4 and w4_tune_tag(True, False) == "k4|ao=1|sym=0" and w4_tune_tag() == "k4|ao=0|sym=0"
    ds = {ops_gemm.dtype_cfg("w4a16_gemm", torch.bfloat16, 128, t) for t in tags}
    assert len(ds) == 4


def test_bench_cache_key_matches_the_dispatcher_key(monkeypatch):
    """bench_w4a16 and forge.kernels.bench build the dtype_cfg from the same helper as w4a16_linear (act-order + symmetry)."""
    import inspect

    from forge.kernels import bench as kbench

    b = _load_bench()
    for src in (inspect.getsource(b.main), inspect.getsource(kbench._build_w4), inspect.getsource(ops_gemm.w4a16_linear)):
        assert "w4_tune_tag(" in src


def test_resolve_seeds_from_the_named_impls_own_default(monkeypatch):
    monkeypatch.delenv("FORGE_W4_DEQUANT_MIN_M", raising=False)
    for sm, sms in ((120, 188), (90, 132), (None, 0)):
        c = resolve_w4_config(256, 5120, 5120, {"impl": "dot"}, sm=sm, num_sms=sms)  # heuristic would be dequant (tile 16x32)
        assert c["impl"] == "dot" and c["tile_m"] >= 64 and c["tile_n"] >= 64 and c["tile_k"] >= 128, c
        g = resolve_w4_config(4, 5120, 5120, {"impl": "gemv"}, sm=sm, num_sms=sms)
        assert g["impl"] == "gemv" and g["tile_m"] == 4 and g["tile_n"] >= 16
        d = resolve_w4_config(2, 5120, 5120, {"impl": "dequant_matmul"}, sm=sm, num_sms=sms)
        assert d["impl"] == "dequant_matmul" and d["tile_m"] == 16 and d["tile_n"] == 32
        # explicit entries still win over the seed
        assert resolve_w4_config(256, 5120, 5120, {"impl": "dot", "tile_n": 32}, sm=sm, num_sms=sms)["tile_n"] == 32


def test_resolved_cache_is_keyed_on_the_dequant_threshold_env(monkeypatch):
    import forge.kernels.triton.gemm_w4_fast as fast

    fast._RESOLVED.clear()
    monkeypatch.delenv("FORGE_W4_DEQUANT_MIN_M", raising=False)
    assert fast._resolve(32, 5120, 5120, None, 120, 188, 128)[0]["impl"] == "dot"
    monkeypatch.setenv("FORGE_W4_DEQUANT_MIN_M", "32")
    assert fast._resolve(32, 5120, 5120, None, 120, 188, 128)[0]["impl"] == "dequant_matmul"
    monkeypatch.delenv("FORGE_W4_DEQUANT_MIN_M")
    assert fast._resolve(32, 5120, 5120, None, 120, 188, 128)[0]["impl"] == "dot"


def test_dequant_min_m_follows_the_cost_model_vs_measured_dot():
    from forge.kernels.w4_tuning import DEQUANT_TIME_VS_BF16, _derived_dequant_min_m, chip_profile

    assert _derived_dequant_min_m({}) == 64  # unmeasured chip
    assert _derived_dequant_min_m({"dot_vs_bf16": {32: 0.77}}) == 64  # model (x2.25 bf16 time) is above dot (x1.3) at M=32
    assert _derived_dequant_min_m({"dot_vs_bf16": {32: 1.0 / (DEQUANT_TIME_VS_BF16 + 0.5)}}) == 32  # dot slower than the model
    assert chip_profile(120)["dequant_min_m"] == 64 and chip_profile(90)["dequant_min_m"] == 64


@pytest.mark.parametrize("n,k", [(5120, 5120), (17408, 5120), (5120, 17408)])
def test_measured_sm120_defaults(n, k, monkeypatch):
    monkeypatch.delenv("FORGE_W4_DEQUANT_MIN_M", raising=False)

    def cfg(m):
        c = default_w4_config(m, n, k, sm=120, num_sms=188)
        assert w4_feasible(c, sm=120) and normalise_w4_config(c, m) == c, c
        return c

    c1 = cfg(1)
    assert c1["impl"] == "gemv" and c1["tile_m"] == 1 and c1["warps"] == 4
    if n >= 8192:
        assert (c1["tile_n"], c1["tile_k"], c1["stages"]) == (64, 256, 5) and c1["split_k"] in (1, 2)
    else:
        assert (c1["tile_n"], c1["tile_k"], c1["stages"], c1["split_k"]) == (32, 128, 3, 8)
    for m in (2, 4):
        c = cfg(m)
        assert (c["impl"], c["tile_m"], c["tile_n"], c["tile_k"], c["warps"]) == ("dot", 16, 128, 128, 4) and c["stages"] in (2, 3) and 4 <= c["split_k"] <= 16
    for m in (8, 16):
        c = cfg(m)
        assert (c["impl"], c["tile_m"], c["tile_n"], c["tile_k"]) == ("dot", 16, 256, 128) and c["warps"] in (4, 8) and 8 <= c["split_k"] <= 16
    c32 = cfg(32)
    assert (c32["impl"], c32["tile_m"], c32["tile_n"], c32["tile_k"], c32["stages"]) == ("dot", 32, 256, 128, 3) and 4 <= c32["split_k"] <= 16
    for m in (1, 4, 8, 16, 32):  # the measured winner is among the sweep candidates
        assert default_w4_config(m, n, k, sm=120, num_sms=188) in w4_candidates(m, n, k, sm=120, num_sms=188)
    # other chips keep the first-principles heuristics (not measured)
    assert default_w4_config(4, n, k, sm=90, num_sms=132)["impl"] == "gemv"


def test_chunk_budget_includes_the_output_slice():
    k, item, m = 4096, 2, 1024
    budget = 256 << 20
    plain = plan_dequant_chunks(100000, k, item, budget)
    with_out = plan_dequant_chunks(100000, k, item, budget, m=m)
    assert max(r1 - r0 for r0, r1 in with_out) < max(r1 - r0 for r0, r1 in plain)
    assert all((r1 - r0) * (k + m) * item <= budget for r0, r1 in with_out)


class _FakeDq:
    """Stands in for the Triton dequant kernel: fills the chunk buffer from a dense weight; raises OOM above a row limit."""

    def __init__(self, dense, oom_rows=None):
        self.dense, self.oom_rows, self.calls = dense, oom_rows, []

    def __call__(self, qweight, scales, zeros, g_idx, *, n, k, group_size, row0=0, rows=None, out=None, dtype=None, config=None):
        rows = n - row0 if rows is None else rows
        self.calls.append(rows)
        if self.oom_rows is not None and rows > self.oom_rows:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1.00 GiB")
        w = self.dense[row0:row0 + rows]
        if out is None:
            return w.clone()
        out[:rows] = w
        return out[:rows]


def _dq_setup(monkeypatch, n=320, k=64, m=5, oom_rows=None):
    import forge.kernels.triton.gemm_w4_dequant as dq

    dense = torch.randn(n, k).bfloat16()
    fake = _FakeDq(dense, oom_rows)
    monkeypatch.setattr(dq, "w4_dequant", fake)
    monkeypatch.setattr(dq, "require_triton", lambda: None)
    monkeypatch.setattr(dq, "_BUDGET_CAP", {})
    monkeypatch.setattr(dq, "_WARNED", set())
    x = torch.randn(m, k).bfloat16()
    bias = torch.randn(n).bfloat16()
    return dq, fake, dense, x, bias


def test_chunked_dequant_matmul_writes_into_the_output_slice(monkeypatch):
    dq, fake, dense, x, bias = _dq_setup(monkeypatch)
    monkeypatch.setenv("FORGE_W4_DEQUANT_MB", "1")
    monkeypatch.setattr(dq, "dequant_budget_bytes", lambda cfg=None: 64 * (64 + 5) * 2 * 2)  # two 64-row chunks' worth -> 128 rows
    for b in (None, bias):
        y = dq.w4_dequant_matmul(x, None, None, None, b, None, n=320, k=64, group_size=64)
        ref = torch.nn.functional.linear(x, dense, b)
        torch.testing.assert_close(y, ref, atol=0.3, rtol=0.05)
        assert y.shape == (5, 320) and len(fake.calls) > 1 and max(fake.calls) <= 128
        fake.calls.clear()


def test_dequant_oom_falls_back_with_one_warning_and_remembers_the_cap(monkeypatch, caplog):
    import logging

    dq, fake, dense, x, bias = _dq_setup(monkeypatch, oom_rows=64)
    caplog.set_level(logging.WARNING, logger=dq.log.name)
    y = dq.w4_dequant_matmul(x, None, None, None, bias, None, n=320, k=64, group_size=64)
    torch.testing.assert_close(y, torch.nn.functional.linear(x, dense, bias), atol=0.3, rtol=0.05)
    assert fake.calls[0] == 320 and fake.calls[-1] <= 64  # the first attempt OOMed, the retry used small chunks
    warned = [r for r in caplog.records if "out of memory" in r.getMessage()]
    assert warned, "OOM fallback must not be silent"
    cap = dict(dq._BUDGET_CAP)
    assert cap and next(iter(cap.values())) < dense.numel() * 2
    fake.calls.clear()
    n_warn = len(warned)
    dq.w4_dequant_matmul(x, None, None, None, bias, None, n=320, k=64, group_size=64)
    assert max(fake.calls) <= 64, "the shrunken chunk size is remembered: no 1 GiB retry on every call"
    assert len([r for r in caplog.records if "out of memory" in r.getMessage()]) == n_warn  # warned once only


def test_dequant_oom_at_the_minimum_chunk_is_raised(monkeypatch):
    dq, fake, dense, x, bias = _dq_setup(monkeypatch, oom_rows=0)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        dq.w4_dequant_matmul(x, None, None, None, bias, None, n=320, k=64, group_size=64)


def test_tuner_picks_global_min_and_prefers_clean_only_within_3pct(tmp_path):
    def run(times, throttled_tns):
        ref = torch.ones(2, 2)
        state = {"cfg": None}

        def timer(fn, cfg):
            fn()
            return [times[cfg["tile_n"]]] * 15

        def run_fn(cfg, s):
            state["cfg"] = cfg["tile_n"]
            return ref

        def tele():
            return {"throttle": "hw_slowdown"} if state["cfg"] in throttled_tns else {}

        t = Autotuner(tmp_path / f"c{times[32]}{sorted(throttled_tns)}.json", chip=chip_from_sm(90), timer=timer, telemetry=tele, retries=0)
        return t.tune(load_spec("w4a16_gemm"), {"M": 4, "N": 64, "K": 128}, run_fn, lambda s: ref, candidates=_cfgs(16, 32, 64),
                      accept_throttled=True)

    # throttled winner is 20% faster than the best clean one -> the global minimum wins, flagged throttled and cached flagged
    r = run({16: 1.0, 32: 0.8, 64: 1.5}, {32})
    assert r.best["tile_n"] == 32 and r.best_measurement.throttled and r.cached
    # throttled winner only 2% faster -> the clean one is preferred
    r = run({16: 1.0, 32: 0.98, 64: 1.5}, {32})
    assert r.best["tile_n"] == 16 and not r.best_measurement.throttled
    # a (faster) clean candidate is never beaten by a slower throttled one
    r = run({16: 1.0, 32: 1.2, 64: 1.5}, {32, 64})
    assert r.best["tile_n"] == 16


def test_tuner_store_false_writes_nothing_until_commit(tmp_path):
    t, ref, _ = _fake_tuner(tmp_path, {16: 1.0, 32: 2.0})
    spec, shape = load_spec("w4a16_gemm"), {"M": 4, "N": 64, "K": 128}
    r = t.tune(spec, shape, lambda cfg, s: ref, lambda s: ref, candidates=_cfgs(16, 32), store=False)
    assert r.best["tile_n"] == 16 and not r.cached and t.cache.entries() == []
    assert t.commit(r) and r.cached and len(t.cache.entries()) == 1
    # a worse refinement result must not replace the stored best: the caller simply does not commit it
    r2 = t.tune(spec, shape, lambda cfg, s: ref, lambda s: ref, candidates=_cfgs(32), force=True, store=False)
    assert t.cache.entries()[0]["config"]["tile_n"] == 16 and r2.best["tile_n"] == 32


def test_report_sink_never_truncates_an_existing_report(tmp_path):
    b = _load_bench()
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"rows": [{"role": "a", "n": 8, "k": 8, "m": 1, "best": {"impl": "gemv"}}], "partial": True}), encoding="utf-8")
    before = p.read_text(encoding="utf-8")
    s = b.ReportSink(p, {"chip": "x"})
    assert s.path != p and p.read_text(encoding="utf-8") == before  # a new sibling file, the old one untouched
    s.add_row(dict(role="a", n=8, k=8, m=1))
    assert p.read_text(encoding="utf-8") == before
    # resume merges
    s2 = b.ReportSink(p, {"chip": "x"}, resume=True)
    assert s2.path == p and len(s2.report["rows"]) == 1 and s2.finished("a", 8, 8, 1) and not s2.finished("a", 8, 8, 8)
    s2.add_row(dict(role="a", n=8, k=8, m=8, best={"impl": "dot"}))
    s2.add_row(dict(role="a", n=8, k=8, m=1, error="boom"))  # same key replaces
    rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
    assert len(rows) == 2 and sum("error" in r for r in rows) == 1
    # an unreadable report is not resumed (and not overwritten either)
    p.write_text("{not json", encoding="utf-8")
    s3 = b.ReportSink(p, {}, resume=True)
    assert s3.path != p and p.read_text(encoding="utf-8") == "{not json"


def test_sweep_keep_skips_finished_rows_and_survives_report_write_errors(tmp_path):
    b = _load_bench()
    p = tmp_path / "r.json"
    clock = FakeClock()
    sink = b.ReportSink(p, {})
    b.sweep([("a", 8, 8)], [1, 8], FakeHooks(clock), sink, b.Budget(0, clock=clock), log=lambda s: None)
    sink.finish(None)
    hooks = FakeHooks(clock)
    sink2 = b.ReportSink(p, {}, resume=True)
    logs = []
    b.sweep([("a", 8, 8), ("b", 8, 8)], [1, 8], hooks, sink2, b.Budget(0, clock=clock), log=logs.append, skip_finished=True)
    assert ("setup", "a") not in hooks.events and ("setup", "b") in hooks.events and any("already finished" in line for line in logs)
    assert len(sink2.report["rows"]) == 4

    class BadSink(b.ReportSink):
        def add_row(self, row):
            raise OSError("disk full")

    bad = BadSink(tmp_path / "bad.json", {})
    logs = []
    assert b.sweep([("a", 8, 8)], [1, 8, 16], FakeHooks(clock), bad, b.Budget(0, clock=clock), log=logs.append) is None
    assert sum("report write failed" in line for line in logs) == 3  # every row attempted; the sweep did not abort


def test_bench_source_refines_without_store_and_no_cache_never_deletes():
    import inspect

    src = inspect.getsource(_load_bench().main)
    assert "store=False" in src and "tuner.commit(" in src and "cache.delete" not in src
    assert "TemporaryDirectory" in src and "not res.from_cache" in src


def test_run_gpu_tests_script_is_set_u_safe_and_preflight_warns_about_torch():
    root = Path(__file__).resolve().parent.parent
    sh = (root / "scripts" / "run_gpu_tests.sh").read_text(encoding="utf-8")
    assert '"${CPU_TIMEOUT_ARGS[@]}"' not in sh.replace('${CPU_TIMEOUT_ARGS[@]+"${CPU_TIMEOUT_ARGS[@]}"}', "")
    assert '"${TIMEOUT_ARGS[@]}"' not in sh.replace('${TIMEOUT_ARGS[@]+"${TIMEOUT_ARGS[@]}"}', "")
    assert "\r" not in sh
    spec = importlib.util.spec_from_file_location("gpu_report_t", root / "scripts" / "gpu_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    banner = mod.format_banner({"python": "py", "deps": {"fastapi": "missing (x)"}, "missing_required": ["fastapi"], "missing_optional": [],
                                "install_command": mod.install_command("py")})
    assert "--no-deps" in banner and "torch" in banner and "NVIDIA" in banner
