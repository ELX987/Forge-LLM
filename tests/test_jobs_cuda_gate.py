"""Worker CUDA pre-flight: GPU jobs fail fast with the diagnosis; CPU jobs and FORGE_SKIP_CUDA_CHECK are untouched (hermetic)."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import home  # noqa: E402,F401

from forge.core.errors import HardwareUnsupported  # noqa: E402
from forge.hw import cuda_compat  # noqa: E402
from forge.jobs import cuda_gate  # noqa: E402

FINDINGS = [{"level": "fail", "code": "torch_cuda_newer_minor", "message": "torch CUDA 13.4 > driver CUDA 13.0", "fix": "use driver >= 595.91.07"}]
MSG = "CUDA stack problem: torch CUDA 13.4 > driver CUDA 13.0 -> use driver >= 595.91.07"


@pytest.fixture(autouse=True)
def no_disk_cache(monkeypatch):
    """Default: no fingerprint -> the on-disk success cache is bypassed (the cache tests below opt in)."""
    monkeypatch.setattr(cuda_gate, "fingerprint", lambda env: None)


@pytest.fixture
def probe(monkeypatch):
    """Replace assert_cuda_usable with a counting fake; ``probe.fail`` toggles the failure."""
    state = {"calls": 0, "fail": True}

    def fake(**_kw):
        state["calls"] += 1
        if state["fail"]:
            raise cuda_compat.CudaUnusable(MSG, FINDINGS)
        return {"ok": True}

    monkeypatch.setattr(cuda_compat, "assert_cuda_usable", fake)
    monkeypatch.delenv("FORGE_SKIP_CUDA_CHECK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    return state


# -- the helper ---------------------------------------------------------------------------------------------------------------
def test_gpu_job_raises_hardware_unsupported_with_findings(probe):
    with pytest.raises(HardwareUnsupported) as ei:
        cuda_gate.gate_cuda(True)
    assert str(ei.value) == MSG and ei.value.details["findings"] == FINDINGS and ei.value.code == "hardware_unsupported"


def test_cpu_job_type_not_gated(probe):
    cuda_gate.gate_cuda(False)
    assert probe["calls"] == 0


@pytest.mark.parametrize("cvd", ["", "-1", "  "])
def test_scheduler_forced_cpu_not_gated(probe, monkeypatch, cvd):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cvd)
    cuda_gate.gate_cuda(True)
    assert probe["calls"] == 0


def test_unset_visible_devices_is_gated(probe, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(HardwareUnsupported):
        cuda_gate.gate_cuda(True)


@pytest.mark.parametrize("val", ["1", "true", "YES"])
def test_skip_env_var(probe, monkeypatch, val):
    monkeypatch.setenv("FORGE_SKIP_CUDA_CHECK", val)
    cuda_gate.gate_cuda(True)
    assert probe["calls"] == 0


def test_skip_env_zero_does_not_skip(probe, monkeypatch):
    monkeypatch.setenv("FORGE_SKIP_CUDA_CHECK", "0")
    with pytest.raises(HardwareUnsupported):
        cuda_gate.gate_cuda(True)


def test_no_gpu_host_not_blocked(monkeypatch):
    """The real assert_cuda_usable only warns on a host without NVIDIA GPUs, so the gate lets the job through."""
    monkeypatch.delenv("FORGE_SKIP_CUDA_CHECK", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(cuda_compat, "_CACHE", {})
    monkeypatch.setattr(cuda_compat, "detect_stack", lambda **_k: {"smi_present": False, "devices": [], "torch": {"installed": True}})
    cuda_gate.gate_cuda(True)


def test_success_cached_per_process_not_reprobed(monkeypatch):
    monkeypatch.delenv("FORGE_SKIP_CUDA_CHECK", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(cuda_compat, "_CACHE", {})
    calls = []
    monkeypatch.setattr(cuda_compat, "check", lambda **_k: calls.append(1) or {"ok": True, "findings": [{"level": "ok"}]})
    cuda_gate.gate_cuda(True)
    cuda_gate.gate_cuda(True)
    cuda_gate.gate_cuda(True)
    assert len(calls) == 1


def test_failure_is_rechecked_each_time(probe):
    for _ in range(2):
        with pytest.raises(HardwareUnsupported):
            cuda_gate.gate_cuda(True)
    assert probe["calls"] == 2
    probe["fail"] = False
    cuda_gate.gate_cuda(True)


@pytest.mark.parametrize("dev", ["cpu", "CPU", "cpu:0", " cpu "])
def test_params_device_cpu_not_gated(probe, dev):
    cuda_gate.gate_cuda(True, params={"device": dev})
    assert probe["calls"] == 0


@pytest.mark.parametrize("params", [None, {}, {"device": "auto"}, {"device": "cuda:1"}, {"device": None}, {"device": 3}])
def test_params_without_cpu_still_gated(probe, params):
    with pytest.raises(HardwareUnsupported):
        cuda_gate.gate_cuda(True, params=params)


# -- on-disk success cache ----------------------------------------------------------------------------------------------------
FP = {"driver_version": "580.65.06", "driver_cuda": "13.0", "torch": "2.11.0", "cuda_visible_devices": "0"}


@pytest.fixture
def disk(monkeypatch, tmp_path):
    state = {"fp": dict(FP)}
    monkeypatch.setattr(cuda_gate, "fingerprint", lambda env: state["fp"])
    state["path"] = tmp_path / "cache" / "cuda_gate.json"
    return state


def _gate(disk, now):
    cuda_gate.gate_cuda(True, now=now, path=disk["path"])


def test_disk_cache_skips_check_within_ttl(probe, disk):
    probe["fail"] = False
    _gate(disk, 1000.0)
    _gate(disk, 1000.0 + cuda_gate.CACHE_TTL_S - 1)
    assert probe["calls"] == 1 and disk["path"].exists()


def test_disk_cache_expires(probe, disk):
    probe["fail"] = False
    _gate(disk, 1000.0)
    _gate(disk, 1000.0 + cuda_gate.CACHE_TTL_S + 1)
    assert probe["calls"] == 2


def test_disk_cache_future_timestamp_ignored(probe, disk):
    probe["fail"] = False
    _gate(disk, 1000.0)
    _gate(disk, 500.0)  # clock went backwards
    assert probe["calls"] == 2


@pytest.mark.parametrize("key,val", [("driver_version", "590.1"), ("driver_cuda", "13.1"), ("torch", "2.12.0"), ("cuda_visible_devices", "1")])
def test_disk_cache_invalidated_by_fingerprint_change(probe, disk, key, val):
    probe["fail"] = False
    _gate(disk, 1000.0)
    disk["fp"] = {**FP, key: val}
    _gate(disk, 1001.0)
    assert probe["calls"] == 2


def test_failure_never_cached_and_rechecked(probe, disk):
    for t in (1000.0, 1001.0):
        with pytest.raises(HardwareUnsupported):
            _gate(disk, t)
    assert probe["calls"] == 2 and not disk["path"].exists()
    probe["fail"] = False
    _gate(disk, 1002.0)
    assert probe["calls"] == 3 and disk["path"].exists()


def test_failure_after_cached_success_is_not_masked_after_change(probe, disk):
    probe["fail"] = False
    _gate(disk, 1000.0)
    disk["fp"] = {**FP, "driver_version": "470.0"}
    probe["fail"] = True
    with pytest.raises(HardwareUnsupported):
        _gate(disk, 1001.0)


@pytest.mark.parametrize("blob", [b"", b"{not json", b"[]", b'{"ts": "x", "fingerprint": {}}', b'{"fingerprint": null}', b"\xff\xfe"])
def test_corrupt_cache_ignored_and_rewritten(probe, disk, blob):
    probe["fail"] = False
    disk["path"].parent.mkdir(parents=True)
    disk["path"].write_bytes(blob)
    _gate(disk, 1000.0)
    _gate(disk, 1001.0)
    assert probe["calls"] == 1  # first run re-checked and healed the file, second hit it


def test_unwritable_cache_does_not_break_gate(probe, disk, tmp_path):
    probe["fail"] = False
    blocker = tmp_path / "file"
    blocker.write_text("x")
    cuda_gate.gate_cuda(True, now=1.0, path=blocker / "sub" / "cuda_gate.json")  # parent is a file -> cannot write
    assert probe["calls"] == 1


def test_no_fingerprint_means_no_cache(probe, disk):
    probe["fail"] = False
    disk["fp"] = None
    _gate(disk, 1000.0)
    _gate(disk, 1001.0)
    assert probe["calls"] == 2 and not disk["path"].exists()


def test_skip_env_does_not_touch_cache(probe, disk, monkeypatch):
    monkeypatch.setenv("FORGE_SKIP_CUDA_CHECK", "1")
    _gate(disk, 1000.0)
    assert probe["calls"] == 0 and not disk["path"].exists()


def test_default_cache_path_under_forge_home(home):
    assert cuda_gate.cache_path() == Path(home) / "cache" / "cuda_gate.json" or cuda_gate.cache_path().name == "cuda_gate.json"
    assert cuda_gate.cache_path().parent.name == "cache"


def test_fingerprint_uses_nvml_then_smi_without_torch_or_probe(monkeypatch):
    monkeypatch.undo()  # drop the autouse stub
    monkeypatch.setattr(cuda_compat, "_nvml_facts", lambda: {"driver_version": "580.65.06"})
    monkeypatch.setattr(cuda_compat, "_smi_facts", lambda: {"devices": [{"driver": "x"}], "driver_cuda": "13.0"})
    monkeypatch.setattr(cuda_compat, "detect_stack", lambda **_: pytest.fail("must not run the full detection"))
    monkeypatch.setattr(cuda_compat, "probe_devices", lambda *_a, **_k: pytest.fail("must not probe"))
    fp = cuda_gate.fingerprint({"CUDA_VISIBLE_DEVICES": "0"})
    assert fp["driver_version"] == "580.65.06" and fp["driver_cuda"] == "13.0" and fp["cuda_visible_devices"] == "0"


def test_fingerprint_none_without_driver(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(cuda_compat, "_nvml_facts", lambda: {})
    monkeypatch.setattr(cuda_compat, "_smi_facts", lambda: {"present": False, "devices": []})
    assert cuda_gate.fingerprint({}) is None


# -- through the worker -------------------------------------------------------------------------------------------------------
def _run_worker(home, monkeypatch, *, needs_gpu: bool):
    from forge.core.db import get_db
    from forge.core.schemas import JobSpec
    from forge.jobs import registry, worker
    from forge.jobs.repo import JobsRepo

    real = registry.get_job_type
    monkeypatch.setattr(registry, "get_job_type", lambda name: dataclasses.replace(real(name), needs_gpu=needs_gpu))
    monkeypatch.setattr(worker, "_write_env", lambda run_dir: None)
    repo = JobsRepo()
    from forge.core.paths import run_dir as run_dir_for
    from forge.core.util import new_id

    jid = new_id("job")
    rd = run_dir_for(jid)
    repo.create(JobSpec(type="selftest", params={"steps": 2, "step_delay_s": 0.0}), run_dir=str(rd), job_id=jid)
    code = worker.main([jid])
    return code, get_db().query_one("SELECT status, error, exit_code FROM jobs WHERE id=?", (jid,))


def test_worker_gpu_job_fails_fast_with_reason(home, probe, monkeypatch, capsys):
    code, row = _run_worker(home, monkeypatch, needs_gpu=True)
    assert code == 1 and row["status"] == "failed" and row["exit_code"] == 1
    assert row["error"] == f"[hardware_unsupported] HardwareUnsupported: {MSG}"
    assert f"job failed: [hardware_unsupported] HardwareUnsupported: {MSG}" in capsys.readouterr().out
    assert probe["calls"] == 1


def test_worker_cpu_job_unaffected(home, probe, monkeypatch):
    code, row = _run_worker(home, monkeypatch, needs_gpu=False)
    assert code == 0 and row["status"] == "succeeded" and row["error"] is None
    assert probe["calls"] == 0


def test_worker_skip_env_runs_gpu_job(home, probe, monkeypatch):
    monkeypatch.setenv("FORGE_SKIP_CUDA_CHECK", "1")
    code, row = _run_worker(home, monkeypatch, needs_gpu=True)
    assert code == 0 and row["status"] == "succeeded" and probe["calls"] == 0
