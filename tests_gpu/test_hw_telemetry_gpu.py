"""GPU telemetry on a real host (auto-skipped without CUDA)."""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

from forge.hw.telemetry import GpuSampler  # noqa: E402


def test_sampler_sees_gpus():
    import torch

    with GpuSampler() as s:
        rows = s.sample()
        assert s.backend in ("pynvml", "nvidia-smi")
        assert len(rows) >= torch.cuda.device_count() >= 1
        r = rows[0]
        assert r["mem_total_mb"] and r["mem_total_mb"] > 1000
        assert r["temp_c"] is None or 0 < r["temp_c"] < 120
        assert r["name"]


def test_capabilities_match_torch():
    import torch

    from forge.hw.capabilities import current_capabilities, list_gpus

    c = current_capabilities(0)
    p = torch.cuda.get_device_properties(0)
    assert c.sm == p.major * 10 + p.minor and c.num_sms == p.multi_processor_count
    assert len(list_gpus()) == torch.cuda.device_count()
