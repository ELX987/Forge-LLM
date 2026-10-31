"""forge.hw package surface: lazy exports, python-version-safe imports, no heavy imports at import time."""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_lazy_exports():
    import forge.hw as hw

    assert hw.KNOWN_CHIPS["h100-sxm5-80g"]["sm"] == 90
    for name in ("current_capabilities", "list_gpus", "run_probe", "GpuSampler", "save_descriptor", "load_descriptor",
                 "list_descriptors", "Capabilities", "ProbeConfig", "descriptor_from_chip", "support_matrix"):
        assert callable(getattr(hw, name)) or isinstance(getattr(hw, name), type)
    assert hw.microbench.stream_copy_bw and hw.telemetry.GpuSampler
    assert "probe" in dir(hw)
    with pytest.raises(AttributeError):
        hw.nope  # noqa: B018


def test_import_is_cheap_and_has_no_hard_optional_deps():
    code = (
        "import sys, forge.hw, forge.hw.chips, forge.hw.capabilities, forge.hw.probe, forge.hw.microbench, "
        "forge.hw.telemetry, forge.hw.descriptor_store;"
        "bad=[m for m in ('triton','pynvml','transformers') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""


def test_hip_torch_probe_falls_back_to_cpu(monkeypatch):
    import torch

    from forge.hw import capabilities as cap
    from forge.hw import probe as pr
    from forge.hw.config import ProbeConfig

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(cap, "software_versions", lambda refresh=False: {"hip": "6.0", "torch": "x", "cuda": None,
                                                                         "triton": None})
    monkeypatch.setattr(cap, "list_gpus", lambda: [])
    res = pr.run_probe_full(config=ProbeConfig(microbench=False, save=False))
    assert res.descriptor.vendor == "cpu" and res.descriptor.tier == "C"
