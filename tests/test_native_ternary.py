"""Optional native ternary CUDA tier: it must never be required, must degrade gracefully, and its sources must at least type-check with the stub headers."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import forge.native.ternary as nt

ROOT = Path(nt.ROOT)


def test_loader_degrades_gracefully_without_the_extension():
    nt.load(force=True)
    assert nt.available() is False and nt.error() and "forge_ternary_C" in nt.error()


def test_force_ref_disables_the_extension(monkeypatch):
    monkeypatch.setenv("FORGE_FORCE_REF", "1")
    assert nt.load(force=True) is None and nt.error() == "FORGE_FORCE_REF=1"
    monkeypatch.delenv("FORGE_FORCE_REF")
    nt.load(force=True)


def test_sources_are_present_and_renamed():
    assert "PYBIND11_MODULE(forge_ternary_C, m)" in (ROOT / "kernel" / "torch_ext" / "bindings.cpp").read_text(encoding="utf-8")
    assert 'name="forge_ternary_C"' in (ROOT / "setup.py").read_text(encoding="utf-8")
    assert sorted(p.name for p in (ROOT / "kernel" / "src").glob("*.cu")) == ["activations.cu", "device.cu", "gemm.cu", "loss.cu", "optimizer.cu", "quantize.cu", "validate.cu"]
    assert "Apache License" in (ROOT.parents[2] / "NOTICE").read_text(encoding="utf-8")


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ not available")
def test_cuda_sources_type_check_against_the_stub_headers():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "syntax_check.py")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
