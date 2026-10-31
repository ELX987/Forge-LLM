"""End-to-end hardware probe on a real NVIDIA GPU (auto-skipped without CUDA)."""
from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

from forge.core.db import Database  # noqa: E402
from forge.core.schemas import Provenance  # noqa: E402
from forge.hw import capabilities as cap  # noqa: E402
from forge.hw import descriptor_store as ds  # noqa: E402
from forge.hw.config import ProbeConfig  # noqa: E402
from forge.hw.probe import run_probe_full  # noqa: E402


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    db = Database(tmp_path_factory.mktemp("probe") / "p.db")
    db.migrate()
    return run_probe_full(config=ProbeConfig(quick=True), db=db), db


def test_probe_builds_measured_descriptor(result):
    res, db = result
    d = res.descriptor
    c = cap.current_capabilities(0)
    assert d.sm == c.sm and d.compute_units == c.num_sms and d.tier in ("A", "B", "C")
    assert d.memory.hbm_bw_gbs.measured and d.memory.hbm_bw_gbs.measured > 0
    assert d.probe_provenance["memory.hbm_bw_gbs"] == Provenance.measured
    assert d.supports("fp16") or c.sm < 70
    assert all(v > 0 for v in d.derived.ridge_flop_per_byte.values())
    assert d.launch_latency_us is None or d.launch_latency_us > 0
    assert ds.load_descriptor(d.id, db=db) == d


def test_fp8_fp4_only_listed_when_confirmed(result):
    res, _ = result
    d, f = res.descriptor, res.report["features"]
    if f.get("fp4") is not True:
        assert not d.supports("fp4") or res.report["benchmarks"].get("mma_fp4") is None
    if f.get("fp8") is False:
        assert not d.supports("fp8")


def test_effective_ridge_not_above_datasheet_ridge(result):
    res, _ = result
    eff, dsr = res.report["effective_ridge"], res.report["datasheet_ridge"]
    for k, v in eff.items():
        if k in dsr and res.descriptor.matrix_unit_for(k).peak_tflops.measured:
            # measured stacks sit at or below the datasheet ridge (Plan HW-5); above it means the catalogue peak is too
            # low (report), above 1.5x means the catalogue is implausibly wrong (fail).
            assert v < 1.5 * dsr[k], (k, v, dsr[k])
            if v > dsr[k]:
                warnings.warn(f"catalogue too low: effective ridge {k} {v} > datasheet ridge {dsr[k]}", stacklevel=1)


def test_throttle_summary_present_or_explained(result):
    res, _ = result
    t = res.report["throttle"]
    assert t["available"] or any("telemetry" in n for n in res.report["notes"])
