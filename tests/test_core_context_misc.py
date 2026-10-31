from __future__ import annotations

import os
import threading

import pytest

from forge.core import errors, paths, registry
from forge.core.context import NullContext, RunContext
from forge.core.settings import get_settings


def test_null_context_records_everything():
    ctx = NullContext(params={"a": 1})
    assert isinstance(ctx, RunContext)
    assert ctx.run_dir.exists() and ctx.params == {"a": 1}
    ctx.log("hello")
    ctx.log("oops", "warning")
    ctx.metric(1, loss=2.5, lr=1e-3)
    ctx.metric(2, loss=2.0)
    ctx.progress(0.5, "half")
    ctx.progress(7)  # clamps
    ctx.set_output("k", {"v": 1})
    assert ctx.logs == ["hello", "oops"]
    assert ctx.log_records[1] == ("warning", "oops")
    assert ctx.metrics[0] == (1, {"loss": 2.5, "lr": 1e-3})
    assert ctx.metric_series("loss") == [(1, 2.5), (2, 2.0)]
    assert ctx.progress_events[0] == (0.5, "half") and ctx.last_progress == 1.0
    assert ctx.outputs == {"k": {"v": 1}}
    assert ctx.store is None
    assert not ctx.should_stop()
    ctx.stop = True
    assert ctx.should_stop()


def test_null_context_threadsafe():
    ctx = NullContext()

    def work():
        for i in range(200):
            ctx.metric(i, x=i)

    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(ctx.metrics) == 800


def test_paths_follow_env(forge_home):
    assert paths.forge_home() == forge_home
    for fn, name in [
        (paths.runs_dir, "runs"), (paths.store_dir, "store"), (paths.models_dir, "models"),
        (paths.datasets_dir, "datasets"), (paths.cache_dir, "cache"),
    ]:
        p = fn()
        assert p == forge_home / name and p.is_dir()
    assert paths.data_dir("a/b").is_dir()
    for bad in ("../x", "/abs", "a/../b", ""):
        with pytest.raises(ValueError):
            paths.data_dir(bad)
    assert paths.run_dir("r1") == forge_home / "runs" / "r1"


def test_settings_env(forge_home, monkeypatch):
    monkeypatch.setenv("FORGE_API_TOKEN", "sekrit")
    monkeypatch.setenv("FORGE_API_PORT", "9999")
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    monkeypatch.setenv("FORGE_MAX_CONCURRENT_JOBS", "3")
    s = get_settings()
    assert s.home == forge_home and s.db_url.startswith("sqlite:///") and s.db_url.endswith("forge.db")
    assert s.api_port == 9999 and s.api_token == "sekrit" and s.hf_token == "hf_x" and s.max_concurrent_jobs == 3
    pub = s.public_dict()
    assert pub["api_token"] == "set" and "sekrit" not in str(pub) and "hf_x" not in str(pub)
    monkeypatch.setenv("FORGE_API_PORT", "bad")
    monkeypatch.delenv("FORGE_API_TOKEN")
    s2 = get_settings()
    assert s2.api_port == 8765 and s2.api_token is None


def test_errors_hierarchy():
    for cls in (errors.StoreError, errors.IntegrityError, errors.GateFailed, errors.LicenceBlocked,
                errors.HardwareUnsupported, errors.KernelCorrectnessError, errors.ConfigError, errors.NotFound,
                errors.StageFailed, errors.Cancelled):
        assert issubclass(cls, errors.ForgeError)
    assert issubclass(errors.IntegrityError, errors.StoreError)
    e = errors.NotFound("nope", details={"x": 1})
    assert str(e) == "nope" and e.to_dict() == {"error": "not_found", "message": "nope", "details": {"x": 1}}
    assert isinstance(e, LookupError) and e.http_status == 404
    assert isinstance(errors.ConfigError("x"), ValueError)
    assert errors.GateFailed("g", verdict="v").verdict == "v"


def test_registry_roundtrip():
    kind = "testkind"

    @registry.register(kind, "a", meta={"x": 1})
    def fa():
        """Doc of a.

        More."""
        return 1

    @registry.register(kind, "b")
    class B:
        pass

    assert registry.get(kind, "a") is fa and registry.get(kind, "b") is B
    assert registry.names(kind) == ["a", "b"]
    assert registry.meta(kind, "a") == {"x": 1}
    assert registry.has(kind, "a") and not registry.has(kind, "zz")
    d = registry.describe(kind)
    assert d[0]["name"] == "a" and d[0]["doc"] == "Doc of a." and d[0]["meta"] == {"x": 1}
    assert kind in registry.kinds()
    with pytest.raises(errors.NotFound):
        registry.get(kind, "missing")
    with pytest.raises(errors.NotFound):
        registry.get("nokind", "x")
    # replacement
    registry.register_obj(kind, "a", 42)
    assert registry.get(kind, "a") == 42
    registry.unregister(kind, "a")
    assert not registry.has(kind, "a")


def test_registry_lazy():
    registry.register_lazy("lazykind", "dumps", "json:dumps")
    assert registry.describe("lazykind")[0]["lazy"] == "json:dumps"
    import json

    assert registry.get("lazykind", "dumps") is json.dumps
    with pytest.raises(ValueError):
        registry.register_lazy("lazykind", "bad", "nocolon")


def test_pytest_session_home_is_isolated():
    # conftest points FORGE_HOME at a temp dir; the real ~/.forge must never be used in tests
    assert "forge_test_home_" in os.environ["FORGE_HOME"] or "forge_home" in os.environ["FORGE_HOME"]
