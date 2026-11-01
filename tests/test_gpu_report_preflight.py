"""CPU tests of the GPU-host environment preflight (scripts/gpu_report.py, scripts/run_gpu_tests.sh) and of the
importorskip guards that turn a missing optional-in-practice dependency into a skip with a reason."""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def rep_mod():
    spec = importlib.util.spec_from_file_location("gpu_report_t", ROOT / "scripts" / "gpu_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_versions(monkeypatch, present: dict[str, str]):
    def version(dist):
        if dist in present:
            return present[dist]
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr(md, "version", version)


def test_forge_from_source_tree_is_reported_as_such(rep_mod, monkeypatch):
    _fake_versions(monkeypatch, {})
    assert rep_mod.dep_status("forge", "forge-llm") == "not installed (running from source tree)"
    assert "ModuleNotFoundError" not in rep_mod.dep_status("forge", "forge-llm")
    _fake_versions(monkeypatch, {"forge-llm": "0.1.0"})
    assert rep_mod.dep_status("forge", "forge-llm") == "0.1.0"


def test_missing_dependency_is_reported_with_reason(rep_mod, monkeypatch):
    _fake_versions(monkeypatch, {})
    st = rep_mod.dep_status("definitely_not_a_module_xyz", "definitely-not-a-dist")
    assert st.startswith("missing")


def test_preflight_banner_has_the_exact_install_command(rep_mod, monkeypatch, capsys, tmp_path):
    _fake_versions(monkeypatch, {"pytest": "8", "fastapi": "0.1", "uvicorn": "0.1"})  # forge, zstandard, pytest-timeout absent
    monkeypatch.setattr(rep_mod, "dep_status", lambda module, dist: {"forge-llm": "not installed (running from source tree)",
                                                                   "zstandard": "missing (ModuleNotFoundError)",
                                                                   "pytest-timeout": "missing (ModuleNotFoundError)"}.get(dist, "1.0"))
    assert rep_mod.cmd_preflight([str(tmp_path)]) == 0  # never aborts the run
    out = capsys.readouterr().out
    expected = f'{sys.executable if " " not in sys.executable else chr(34) + sys.executable + chr(34)} -m pip install -e ".[gpu,serve,parquet,dev]"'
    assert expected in out
    assert "zstandard" in out and "forge-llm" in out and "pytest-timeout" in out and "optional" in out
    data = json.loads((tmp_path / "preflight.json").read_text())
    assert data["ok"] is False and set(data["missing_required"]) == {"forge-llm", "zstandard"} and data["missing_optional"] == ["pytest-timeout"]


def test_preflight_quiet_when_everything_is_installed(rep_mod, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(rep_mod, "dep_status", lambda module, dist: "1.2.3")
    assert rep_mod.cmd_preflight([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "pip install" not in out and "all dependencies present" in out
    assert json.loads((tmp_path / "preflight.json").read_text())["ok"] is True


def test_env_json_marks_source_tree_and_missing_deps(rep_mod, monkeypatch, tmp_path):
    _fake_versions(monkeypatch, {})
    rep_mod.cmd_env([str(tmp_path)])
    env = json.loads((tmp_path / "env.json").read_text())
    assert env["forge"] == "not installed (running from source tree)" and env["forge_source_version"] != "?"
    assert "forge-llm" in env["missing_dependencies"] and "pip install -e" in env["install_command"]


def test_run_gpu_tests_script_runs_the_preflight_before_anything_else():
    text = (ROOT / "scripts" / "run_gpu_tests.sh").read_text(encoding="utf-8")
    assert '".[gpu,serve,parquet,dev]"' in text
    i_env, i_pre, i_cuda = text.index('"$HELPER" env'), text.index('"$HELPER" preflight'), text.index("torch.cuda.is_available")
    assert i_env < i_pre < i_cuda
    assert "preflight" in (ROOT / "scripts" / "gpu_report.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("path,missing", [
    ("tests/test_api_resources.py", "fastapi"),
    ("tests/test_serve_server.py", "fastapi"),
    ("tests_gpu/test_store_gpu.py", "zstandard"),
])
def test_missing_dependency_skips_instead_of_collection_error(tmp_path, path, missing):
    """Collect the module in a subprocess in which ``missing`` cannot be imported: exit code 0/5 with a skip, never 2 (error)."""
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / f"{missing}.py").write_text("raise ModuleNotFoundError('simulated: not installed')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(shim), str(ROOT), env.get("PYTHONPATH", "")])
    proc = subprocess.run([sys.executable, "-m", "pytest", path, "-q", "-rs", "-p", "no:cacheprovider", "--no-header", "--collect-only"],
                          capture_output=True, text=True, cwd=ROOT, env=env, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode in (0, 5), out[-2000:]
    assert "ERROR" not in out, out[-2000:]
    assert f"{missing} not installed in this interpreter" in out and "SKIPPED [1]" in out, out[-2000:]


def test_cuda_health_flags_unusable_devices_and_explains(rep_mod, monkeypatch):
    monkeypatch.setattr(rep_mod, "_smi", lambda *a: ["0, Exclusive_Process, Enabled, N/A"])
    rep = {"devices": [{"index": 0, "ok": False, "error": "CUDA error: CUDA-capable device(s) is/are busy or unavailable"},
                       {"index": 1, "ok": True, "sum": 1024.0}],
           "compute_mode": ["0, Exclusive_Process, Enabled, N/A"], "compute_apps": ["GPU-x, 99, python3, 100 MiB"],
           "ok": False, "usable_devices": [1]}
    banner = rep_mod.format_cuda_banner(rep)
    assert "cuda:0: FAILED" in banner and "cuda:1: ok" in banner
    assert "Exclusive_Process" in banner and "python3" in banner
    assert "CUDA_VISIBLE_DEVICES=1" in banner


def test_cuda_health_cmd_on_cpu_host_writes_json_and_fails(rep_mod, tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("only meaningful on CPU-only torch")
    monkeypatch.setattr(rep_mod, "_smi", lambda *a: [])
    assert rep_mod.cmd_cuda_health([str(tmp_path)]) == 3
    data = json.loads((tmp_path / "cuda_health.json").read_text(encoding="utf-8"))
    assert data["ok"] is False and data["devices"] == []
    assert "no CUDA device" in capsys.readouterr().out


def test_finalize_fails_when_cuda_was_unusable(rep_mod, tmp_path):
    (tmp_path / "cuda_health.json").write_text(json.dumps({"ok": False, "devices": [], "compute_mode": []}), encoding="utf-8")
    assert rep_mod.cmd_finalize([str(tmp_path)]) == 1
    assert "Overall: **FAIL**" in (tmp_path / "report.md").read_text(encoding="utf-8")


class _FakeCompat:
    """Stand-in for forge.hw.cuda_compat: a driver-CUDA-13.0 host with a CUDA 13.4 torch whose devices cannot allocate."""

    def __init__(self, ok: bool = False):
        self._ok = ok

    def check(self, **_kw):
        stack = {"driver_version": "580.65.06", "driver_cuda": "13.0", "nvcc": None, "triton": "3.4.0",
                 "devices": [{"index": 0, "name": "NVIDIA B200", "sm": 100, "compute_mode": "Default"}],
                 "torch": {"version": "2.99+cu134", "cuda": "13.4", "arch_list": ["sm_90", "sm_100"]},
                 "probe": {"devices": [{"torch_index": 0, "ok": self._ok, **({"sum": 1024.0} if self._ok else {"error": "CUDA error: busy or unavailable"})}]}}
        findings = [{"level": "ok", "code": "driver", "message": "driver 580.65.06", "fix": None}]
        if not self._ok:
            findings.append({"level": "fail", "code": "devices_unusable", "message": "1 of 1 CUDA device(s) failed", "fix": "rent a host with a newer driver"})
        return {"ok": self._ok, "worst": "ok" if self._ok else "fail", "stack": stack, "findings": findings, "recommend": {}}

    def format_report(self, rep):
        return "\n".join(f"[{f['level']}] {f['message']}" for f in rep["findings"])


def test_cuda_health_uses_cuda_compat_and_records_stack(rep_mod, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rep_mod, "_load_cuda_compat", lambda: _FakeCompat(ok=False))
    monkeypatch.setattr(rep_mod, "_smi", lambda *a: ["0, Default, Enabled, N/A"])
    assert rep_mod.cmd_cuda_health([str(tmp_path)]) == 3
    out = capsys.readouterr().out
    assert "cuda:0: FAILED" in out and "devices_unusable" not in out and "1 of 1 CUDA device(s) failed" in out
    data = json.loads((tmp_path / "cuda_health.json").read_text(encoding="utf-8"))
    assert data["ok"] is False and data["stack"]["driver_cuda"] == "13.0" and data["findings"][-1]["code"] == "devices_unusable"
    assert data["devices"] == [{"index": 0, "ok": False, "error": "CUDA error: busy or unavailable"}]


def test_cuda_health_ok_path_and_stack_section_in_report_md(rep_mod, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rep_mod, "_load_cuda_compat", lambda: _FakeCompat(ok=True))
    monkeypatch.setattr(rep_mod, "_smi", lambda *a: [])
    assert rep_mod.cmd_cuda_health([str(tmp_path)]) == 0
    assert "1 device(s) usable" in capsys.readouterr().out
    assert rep_mod.cmd_finalize([str(tmp_path)]) == 0
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## CUDA stack" in md and "driver CUDA: `13.0`" in md and "torch CUDA: `13.4`" in md and "sm_90 sm_100" in md


def test_stack_section_lists_failing_findings_with_fixes(rep_mod, monkeypatch, tmp_path):
    monkeypatch.setattr(rep_mod, "_load_cuda_compat", lambda: _FakeCompat(ok=False))
    monkeypatch.setattr(rep_mod, "_smi", lambda *a: [])
    rep_mod.cmd_cuda_health([str(tmp_path)])
    assert rep_mod.cmd_finalize([str(tmp_path)]) == 1
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## CUDA stack" in md and "**FAIL** `devices_unusable`" in md and "fix: rent a host with a newer driver" in md


def test_cuda_health_falls_back_to_builtin_logic_when_import_fails(rep_mod, monkeypatch):
    monkeypatch.setattr(rep_mod, "_load_cuda_compat", lambda: None)
    monkeypatch.setattr(rep_mod, "_smi", lambda *a: [])
    rep = rep_mod.cuda_health_report()
    assert "stack" not in rep and "findings" not in rep and "ok" in rep


def test_cuda_health_real_module_is_importable_from_source_tree(rep_mod):
    assert rep_mod._load_cuda_compat() is not None


def test_shell_scripts_use_cuda_compat():
    setup = (ROOT / "scripts" / "vast_setup.sh").read_text(encoding="utf-8")
    assert "forge.hw.cuda_compat" in setup and "FORGE_SKIP_CUDA_CHECK" in setup and "cu124" not in setup.split("TORCH_INDEX=")[1].splitlines()[0]
    smoke = (ROOT / "scripts" / "gpu_smoke.sh").read_text(encoding="utf-8")
    assert smoke.index("forge.hw.cuda_compat") < smoke.index("step torch_cuda")
