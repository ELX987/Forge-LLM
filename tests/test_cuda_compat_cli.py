"""`forge cuda-check`, the CUDA checks inside `forge doctor`, and the driver-CUDA field of run environments (hermetic: detection is monkeypatched)."""
from __future__ import annotations

import json

import pytest

from forge import cli
from forge.core import env
from forge.hw import cuda_compat

GPU = {"index": 0, "name": "NVIDIA H100", "driver": "580.65.06", "sm": 90, "compute_mode": "Default", "mem_mib": 81559}


def _stack(torch_cuda: str, driver_cuda: str = "13.0", probe: dict | None = None) -> dict:
    st = {"driver_version": "580.65.06", "driver_cuda": driver_cuda, "smi_present": True, "smi_ok": True, "devices": [GPU],
          "torch": {"installed": True, "version": "2.9.0", "cuda": torch_cuda, "hip": None, "container_build": False,
                    "arch_list": ["sm_80", "sm_90", "compute_90"]},
          "triton": "3.4.0", "nvidia_ml_py": "13.0.0", "nvcc": None, "cuda_visible_devices": None, "platform": "linux"}
    if probe is not None:
        st["probe"] = probe
    return st


HEALTHY = _stack("13.0", probe={"available": True, "device_count": 1, "devices": [{"torch_index": 0, "ok": True}]})
BROKEN = _stack("13.4", probe={"available": True, "device_count": 1,
                               "devices": [{"torch_index": 0, "ok": False, "error": "CUDA-capable device(s) is/are busy or unavailable"}]})
MISMATCH = _stack("13.4")  # driver 13.0 vs torch 13.4: warn only (no probe result)
CPU_ONLY = {"driver_version": None, "driver_cuda": None, "smi_present": False, "smi_ok": False, "devices": [],
            "torch": {"installed": True, "version": "2.9.0+cpu", "cuda": None, "hip": None, "container_build": False, "arch_list": []},
            "triton": None, "nvidia_ml_py": None, "nvcc": None, "cuda_visible_devices": None, "platform": "win32"}


@pytest.fixture
def stack_of(monkeypatch):
    """Make cuda_compat.check use a synthetic stack (never touches nvidia-smi or CUDA); returns the list of probe flags it was called with."""
    calls: list[bool] = []
    real = cuda_compat.check

    def install(stack: dict) -> list[bool]:
        monkeypatch.setattr(cuda_compat, "check", lambda *, probe=True, probe_timeout=120.0: (calls.append(probe), real(stack=stack))[1])
        return calls

    return install


def run(argv, capsys):
    code = cli.main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cuda_check_healthy_exit_zero(stack_of, capsys):
    stack_of(HEALTHY)
    code, out, _ = run(["cuda-check"], capsys)
    assert code == 0 and "[ ok ]" in out and "FAIL" not in out


def test_cuda_check_broken_exit_one_and_explains(stack_of, capsys):
    stack_of(BROKEN)
    code, out, _ = run(["cuda-check"], capsys)
    assert code == 1 and "[FAIL]" in out and "busy or unavailable" in out and "newer driver" in out


def test_cuda_check_strict_fails_on_warn(stack_of, capsys):
    stack_of(MISMATCH)
    code, out, _ = run(["cuda-check"], capsys)
    assert code == 0 and "[warn]" in out
    code, _, _ = run(["cuda-check", "--strict"], capsys)
    assert code == 1


def test_cuda_check_json_and_no_probe(stack_of, capsys):
    calls = stack_of(BROKEN)
    code, out, _ = run(["cuda-check", "--json", "--no-probe"], capsys)
    rep = json.loads(out)
    assert code == 1 and rep["ok"] is False and rep["worst"] == "fail"
    assert {"stack", "findings", "recommend"} <= set(rep) and any(f["code"] == "devices_unusable" for f in rep["findings"])
    assert calls == [False]
    run(["cuda-check"], capsys)
    assert calls == [False, True]


def test_doctor_fails_on_broken_cuda_stack(stack_of):
    stack_of(BROKEN)
    rep = cli.collect_doctor()
    cuda = {c["name"]: c for c in rep["checks"] if c["name"].startswith("cuda:")}
    assert cuda["cuda:devices_unusable"]["level"] == "fail" and "busy or unavailable" in cuda["cuda:devices_unusable"]["detail"]
    assert rep["info"]["cuda_compat"]["ok"] is False
    assert any(c["level"] == "fail" for c in rep["checks"])


def test_doctor_passes_cuda_part_on_healthy_stack(stack_of):
    stack_of(HEALTHY)
    rep = cli.collect_doctor()
    cuda = [c for c in rep["checks"] if c["name"].startswith("cuda:")]
    assert cuda and all(c["level"] in ("ok", "info", "warn") for c in cuda)
    assert {c["name"] for c in cuda} >= {"cuda:driver", "cuda:torch", "cuda:devices_usable"}
    assert rep["info"]["cuda_compat"]["ok"] is True


def test_doctor_cpu_only_host_warns_without_duplicates(stack_of):
    stack_of(CPU_ONLY)
    rep = cli.collect_doctor()
    assert rep["info"]["cuda_compat"]["findings"][0]["code"] == "no_nvidia_driver"
    assert not [c for c in rep["checks"] if c["name"] == "cuda:no_nvidia_driver"]  # the nvidia-smi / gpu checks already say it
    assert not [c for c in rep["checks"] if c["name"].startswith("cuda:") and c["level"] == "fail"]


def test_doctor_survives_a_crashing_cuda_check(monkeypatch):
    def boom(**_):
        raise RuntimeError("nvml exploded")

    monkeypatch.setattr(cuda_compat, "check", boom)
    rep = cli.collect_doctor()
    c = next(c for c in rep["checks"] if c["name"] == "cuda:check")
    assert c["level"] == "warn" and "nvml exploded" in c["detail"]


def test_capture_environment_has_driver_cuda(monkeypatch):
    monkeypatch.setattr(cuda_compat, "detect_stack", lambda **_: {"driver_cuda": "13.0"})
    assert env.capture_environment()["cuda_driver_cuda"] == "13.0"

    def boom(**_):
        raise RuntimeError("no")

    monkeypatch.setattr(cuda_compat, "detect_stack", boom)
    assert env.capture_environment()["cuda_driver_cuda"] is None
