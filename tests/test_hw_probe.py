"""Probe: CPU-only path, nvidia-smi-only path, and the full GPU path with mocked torch.cuda / NVML / microbenchmarks."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from forge.core.context import NullContext
from forge.core.db import Database
from forge.core.jsonschema import defaults_for, schema_for
from forge.core.schemas import Provenance
from forge.hw import capabilities as cap
from forge.hw import descriptor_store as ds
from forge.hw import microbench as mb
from forge.hw import probe as pr
from forge.hw import telemetry as tm
from forge.hw.config import ProbeConfig


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "probe.db")
    d.migrate()
    yield d
    d.close()


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv("FORGE_DISABLE_FEATURES", raising=False)
    monkeypatch.setattr(tm, "_run_smi", lambda args, timeout=5.0: (-1, ""))
    monkeypatch.setattr(tm, "_load_pynvml", lambda: None)
    monkeypatch.setattr(pr, "_CHECKS", {})
    cap.clear_cache()
    yield
    cap.clear_cache()


def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def fake_cuda(monkeypatch, name, major, minor, gib, sms, **extra):
    props = SimpleNamespace(name=name, major=major, minor=minor, total_memory=int(gib * 2**30),
                            multi_processor_count=sms, **extra)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props)
    cap.clear_cache()
    return props


class MiniNvml:
    NVML_CLOCK_SM = 1

    def nvmlInit(self): ...
    def nvmlShutdown(self): ...
    def nvmlDeviceGetCount(self): return 1
    def nvmlDeviceGetHandleByIndex(self, i): return i
    def nvmlDeviceGetName(self, h): return "x"
    def nvmlDeviceGetUUID(self, h): return "GPU-abcd"
    def nvmlDeviceGetMaxClockInfo(self, h, k): return 1980
    def nvmlDeviceGetClockInfo(self, h, k): return 1700
    def nvmlDeviceGetEnforcedPowerLimit(self, h): return 700000
    def nvmlDeviceGetTemperature(self, h, k): return 70
    def nvmlDeviceGetPowerUsage(self, h): return 650000
    def nvmlDeviceGetCurrentClocksEventReasons(self, h): return 0x4


def ok(name, value, unit, **params):
    return mb.BenchResult(name=name, ok=True, value=value, unit=unit, params=params, samples=[value],
                          stats={"median": value})


def bad(name, unit, err, skipped=False):
    return mb.BenchResult(name=name, ok=False, unit=unit, error=err, skipped=skipped)


def patch_bench(monkeypatch, mma: dict[str, mb.BenchResult], bw=2900.0):
    monkeypatch.setattr(mb, "stream_copy_bw", lambda device=0, kind="copy", **k: (
        ok(f"stream_{kind}_bw", bw if kind == "copy" else bw - 100, "GB/s")))
    monkeypatch.setattr(mb, "mma_throughput", lambda dtype, device=0, **k: mma.get(
        mb.canon_dtype(dtype), bad(f"mma_{dtype}", "TFLOP/s", "n/a", skipped=True)))
    monkeypatch.setattr(mb, "smem_bw", lambda device=0, **k: ok("smem_bw_proxy", 20000.0, "GB/s"))
    monkeypatch.setattr(mb, "launch_latency", lambda device=0, **k: ok("launch_latency", 2.5, "us", method="cuda_graph"))
    monkeypatch.setattr(mb, "dequant_alu_throughput", lambda device=0, **k: ok(
        "dequant_alu", 5000.0, "Gelem/s", simple_ops_per_s_t=25.0))
    monkeypatch.setattr(mb, "p2p_bw", lambda s=0, d=1, **k: ok("p2p", 300.0, "GB/s"))


H100_MMA = {
    "fp32": ok("mma_fp32", 48.0, "TFLOP/s"), "tf32": ok("mma_tf32", 380.0, "TFLOP/s"),
    "fp16": ok("mma_fp16", 700.0, "TFLOP/s"), "bf16": ok("mma_bf16", 720.0, "TFLOP/s"),
    "int8": ok("mma_int8", 1500.0, "TOP/s"), "fp8": ok("mma_fp8", 1400.0, "TFLOP/s"),
}


# ---------------------------------------------------------------------------------------------------------------
def test_config_schema_for_ui():
    sch = schema_for(ProbeConfig)
    props = sch["properties"]
    assert {"device", "quick", "microbench", "benchmarks", "dtypes", "save"} <= set(props)
    assert all(p.get("description") for p in props.values())
    assert defaults_for(ProbeConfig)["quick"] is False
    assert pr.ProbeConfig is ProbeConfig


def test_cpu_probe_end_to_end(monkeypatch, db):
    no_cuda(monkeypatch)
    ctx = NullContext()
    res = pr.run_probe_full(ctx, quick=True, db=db)
    d = res.descriptor
    assert d.tier == "C" and d.vendor == "cpu" and d.sm is None and d.compute_units >= 1
    assert d.memory.hbm_bw_gbs.measured and d.memory.hbm_bw_gbs.measured > 0
    assert d.probe_provenance["memory.hbm_bw_gbs"] == Provenance.measured
    assert d.matrix_units[0].dtype == "fp32" and d.matrix_units[0].peak_tflops.measured > 0
    assert d.derived.ridge_flop_per_byte["fp32"] > 0
    assert not d.supports("bf16")
    assert res.saved_id == d.id and ds.load_descriptor(d.id, db=db) == d
    assert ctx.outputs["descriptor_id"] == d.id and ctx.outputs["tier"] == "C"
    assert ctx.last_progress == 1.0 and any("CPU" in m for m in ctx.logs)
    assert any("no usable GPU" in n or "tier C" in n for n in res.report["notes"])
    assert ds.load_report(d.id, db=db)["features"]["fp8"] is False


def test_cpu_probe_without_microbench_and_no_save(monkeypatch, db):
    no_cuda(monkeypatch)
    d = pr.run_probe_full(config=ProbeConfig(microbench=False, save=False), db=db).descriptor
    assert d.matrix_units == [] and d.memory.hbm_bw_gbs.measured is None and ds.list_descriptors(db=db) == []


def test_run_probe_signature_returns_descriptor(monkeypatch):
    no_cuda(monkeypatch)
    d = pr.run_probe(None, 0, True)
    assert d.tier == "C"
    ds.delete_descriptor(d.id)


def test_job_entry(monkeypatch):
    no_cuda(monkeypatch)
    ctx = NullContext()
    out = pr.run_probe_job(ProbeConfig(quick=True, save=False), ctx)
    assert out["tier"] == "C" and out["descriptor_id"] and "summary" in out
    assert ctx.outputs["summary"]["tier"] == "C"


def test_smi_only_host_gets_datasheet_tier_c(monkeypatch, db):
    no_cuda(monkeypatch)
    row = {"index": 0, "name": "NVIDIA H100 80GB HBM3", "sm": 90, "num_sms": 132, "total_mem_gb": 79.6,
           "source": "nvidia-smi"}
    monkeypatch.setattr(cap, "list_gpus", lambda: [row])
    res = pr.run_probe_full(config=ProbeConfig(save=False), db=db)
    d = res.descriptor
    assert d.tier == "C" and d.name.startswith("NVIDIA H100") and d.sm == 90
    assert d.memory.hbm_bw_gbs.measured is None and not d.supports("fp4")
    assert any("torch.cuda is unavailable" in n for n in res.report["notes"])


# ---------------------------------------------------------------------------------------------------------------
# GPU path (mocked)
# ---------------------------------------------------------------------------------------------------------------
def test_gpu_probe_h100_full(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132, shared_memory_per_block_optin=232448,
              L2_cache_size=50 * 2**20, uuid="abcd")
    monkeypatch.setattr(tm, "_load_pynvml", lambda: MiniNvml())
    patch_bench(monkeypatch, H100_MMA)
    ctx = NullContext()
    res = pr.run_probe_full(ctx, config=ProbeConfig(quick=False, record_throttling=True), db=db)
    d, rep = res.descriptor, res.report
    assert d.sm == 90 and d.tier == "A" and d.compute_units == 132 and d.name == "NVIDIA H100 80GB HBM3"
    assert rep["match"]["key"] == "h100-sxm5-80g" and rep["match"]["kind"] == "exact"
    # measured numbers sit next to datasheet ones
    bw = d.memory.hbm_bw_gbs
    assert bw.datasheet == 3350 and bw.measured == 2900 and bw.src == "microbench/stream_copy_bw"
    mu16 = d.matrix_unit_for("bf16")
    assert mu16.peak_tflops.datasheet == 989.0 and mu16.peak_tflops.measured == 710.0  # mean of fp16 700 / bf16 720
    assert d.matrix_unit_for("fp8").peak_tflops.measured == 1400.0
    assert d.matrix_unit_for("int8").peak_tflops.measured == 1500.0
    assert d.matrix_unit_for("tf32").peak_tflops.measured == 380.0
    assert d.alu.fp32_tflops_measured == 48.0 and d.alu.simple_ops_per_s_t == 25.0
    assert d.alu.onchip_reread_gbs == 20000.0
    assert d.launch_latency_us == 2.5
    # provenance
    pp = d.probe_provenance
    assert pp["memory.hbm_bw_gbs"] == pp["matrix_units.fp16"] == pp["matrix_units.fp8"] == Provenance.measured
    assert pp["launch_latency_us"] == Provenance.measured and pp["feature.fp8"] == Provenance.measured
    assert pp["memory.hbm_gb"] == Provenance.vendor and pp["tier"] == Provenance.derived
    # effective ridge uses measured numbers, and is lower than the datasheet one (Plan HW-5)
    assert d.derived.ridge_flop_per_byte["bf16"] == pytest.approx(710e12 / 2900e9, rel=1e-3)
    assert d.derived.ridge_flop_per_byte["bf16"] < rep["datasheet_ridge"]["bf16"]
    # driver facts + telemetry
    assert d.on_chip.smem_per_cu_kib == 227.0 and d.memory.l2_mb == 50.0 and d.power_limit_w == 700.0
    assert rep["throttle"]["available"] and rep["throttle"]["throttled"] is True
    assert pp["clock_mhz"] == Provenance.measured and d.clock_mhz == 1700.0
    assert any("throttling observed" in n for n in rep["notes"])
    assert rep["features"]["fp8"] is True and rep["features"]["fp4"] is None
    # persisted and visible to the capability layer
    assert res.saved_id == d.id and ds.load_descriptor(d.id, db=db) == d
    assert ctx.outputs["tier"] == "A" and ctx.last_progress == 1.0
    assert rep["config"]["quick"] is False and rep["schema"] == pr.REPORT_SCHEMA


def test_gpu_probe_quick_skips_p2p_and_shrinks(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132)
    calls = {}
    patch_bench(monkeypatch, H100_MMA)
    orig = mb.mma_throughput
    monkeypatch.setattr(mb, "mma_throughput", lambda dtype, device=0, **k: calls.setdefault(dtype, k) and orig(dtype, device))
    monkeypatch.setattr(mb, "p2p_bw", lambda *a, **k: pytest.fail("p2p must not run in quick mode"))
    res = pr.run_probe_full(config=ProbeConfig(quick=True, save=False), db=db)
    assert all(k["quick"] is True for k in calls.values()) and res.descriptor.tier == "A"


def test_gpu_probe_fp4_failure_drops_unit_and_vetoes_capability(monkeypatch):
    fake_cuda(monkeypatch, "NVIDIA GeForce RTX 5090", 12, 0, 31.3, 170)
    mma = {"fp32": ok("m", 100.0, "TFLOP/s"), "tf32": ok("m", 90.0, "TFLOP/s"), "fp16": ok("m", 200.0, "TFLOP/s"),
           "bf16": ok("m", 205.0, "TFLOP/s"), "int8": ok("m", 700.0, "TOP/s"),
           "fp8": bad("mma_fp8", "TFLOP/s", "torch lacks float8", skipped=True),
           "fp4": bad("mma_fp4", "TFLOP/s", "fp4 result != K")}
    patch_bench(monkeypatch, mma)
    assert cap.current_capabilities().has_fp4_block_scaled  # hypothesis before probing
    res = pr.run_probe_full(config=ProbeConfig(quick=True))
    d = res.descriptor
    try:
        assert not d.supports("fp4") and not d.supports("nvfp4")  # Appendix B: only if the probe confirms
        assert d.supports("fp8")  # skipped (not failed) -> hypothesis kept, flagged untested
        assert d.probe_provenance["matrix_units.fp8"] == Provenance.untested
        assert d.probe_provenance["feature.fp4"] == Provenance.untested
        assert res.report["features"]["fp4"] is False
        assert res.report["features"]["fp8"] is None
        assert any("fp4 MMA NOT confirmed" in n for n in res.report["notes"])
        assert d.tier == "A"
        # the stored report now vetoes fp4 in current_capabilities (fp8 untouched)
        c = cap.current_capabilities()
        assert c.fp4_confirmed is False and not c.has_fp4_block_scaled and c.has_fp8
    finally:
        ds.delete_descriptor(d.id)


def test_gpu_probe_fp4_confirmed_adds_unit(monkeypatch, db):
    fake_cuda(monkeypatch, "Mystery Blackwell", 10, 0, 100, 150)  # unknown name -> sm fallback, no datasheet numbers
    mma = dict(H100_MMA, fp4=ok("mma_fp4", 7000.0, "TFLOP/s", recipe="nvfp4"))
    patch_bench(monkeypatch, mma)
    res = pr.run_probe_full(config=ProbeConfig(save=False), db=db)
    d = res.descriptor
    assert res.report["match"]["kind"] == "sm_fallback"
    mu = d.matrix_unit_for("fp4")
    assert mu is not None and mu.peak_tflops.measured == 7000.0 and mu.peak_tflops.datasheet is None
    assert mu.block_scaled["scale"] == "e4m3" and d.probe_provenance["matrix_units.fp4"] == Provenance.measured
    assert d.tier == "A" and d.derived.ridge_flop_per_byte["fp4"] > 0


def test_gpu_probe_correctness_failure_demotes_tier(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA A100-SXM4-80GB", 8, 0, 79.1, 108)
    mma = dict(H100_MMA)
    mma["bf16"] = bad("mma_bf16", "TFLOP/s", "numerical check failed: rel err 0.9")
    mma["fp16"] = bad("mma_fp16", "TFLOP/s", "numerical check failed: rel err 0.9")
    patch_bench(monkeypatch, mma)
    res = pr.run_probe_full(config=ProbeConfig(save=False), db=db)
    assert res.descriptor.tier == "C"  # B demoted to C
    assert res.report["features"]["bf16"] is False
    assert res.descriptor.matrix_unit_for("bf16").peak_tflops.measured is None


def test_gpu_probe_measured_exceeds_datasheet_is_flagged(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA A10G", 8, 6, 22.3, 80)
    patch_bench(monkeypatch, {"fp16": ok("m", 110.0, "TFLOP/s"), "bf16": ok("m", 110.0, "TFLOP/s")}, bw=640.0)
    res = pr.run_probe_full(config=ProbeConfig(save=False, dtypes=["fp16", "bf16"]), db=db)
    assert any("exceeds datasheet" in n for n in res.report["notes"])
    assert res.descriptor.probe_provenance["matrix_units.fp16"] == Provenance.measured
    assert "mma_fp8" not in res.report["benchmarks"] and "mma_fp32" not in res.report["benchmarks"]


def test_gpu_probe_microbench_off_keeps_everything_unconfirmed(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA B200", 10, 0, 178, 148)
    monkeypatch.setattr(mb, "mma_throughput", lambda *a, **k: pytest.fail("no benchmarks expected"))
    res = pr.run_probe_full(config=ProbeConfig(microbench=False, save=False), db=db)
    d = res.descriptor
    assert Provenance.measured not in d.probe_provenance.values()
    assert d.supports("fp4") and res.report["features"]["fp4"] is None
    assert d.memory.hbm_gb == pytest.approx(178.0, abs=0.01)  # driver-reported capacity replaces nominal 192
    assert d.compute_units == 148 and d.tier == "A"


def test_gpu_probe_cancel(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132)
    patch_bench(monkeypatch, H100_MMA)
    ctx = NullContext()
    ctx.stop = True
    res = pr.run_probe_full(ctx, config=ProbeConfig(save=False), db=db)
    assert any("cancelled" in n for n in res.report["notes"]) and res.report["benchmarks"] == {}


def test_gpu_probe_sm_count_mismatch_noted(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA H100 80GB HBM3", 9, 0, 79.1, 120)
    res = pr.run_probe_full(config=ProbeConfig(microbench=False, save=False), db=db)
    assert res.report["match"]["kind"] == "name_adjusted" and res.descriptor.compute_units == 120
    assert any("rescaled" in n for n in res.report["notes"])


def test_extra_checks_registered_by_other_modules(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132)
    patch_bench(monkeypatch, H100_MMA)
    pr.register_check("tma_smoke", lambda device, quick: {"ok": True, "device": device})
    pr.register_check("explodes", lambda device, quick: 1 / 0)
    res = pr.run_probe_full(config=ProbeConfig(save=False), db=db)
    assert res.report["checks"]["tma_smoke"] == {"ok": True, "device": 0}
    assert res.report["checks"]["explodes"]["ok"] is False and "ZeroDivisionError" in res.report["checks"]["explodes"]["error"]
    res2 = pr.run_probe_full(config=ProbeConfig(save=False, benchmarks=["bandwidth"]), db=db)
    assert res2.report["checks"] == {} and "mma_bf16" not in res2.report["benchmarks"]


def test_revision_increments_and_ids_are_unique(monkeypatch, db):
    fake_cuda(monkeypatch, "NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132)
    cfg = ProbeConfig(microbench=False)
    a = pr.run_probe_full(config=cfg, db=db).descriptor
    b = pr.run_probe_full(config=cfg, db=db).descriptor
    assert a.id.endswith("-r1") and b.id.endswith("-r2") and a.fingerprint == b.fingerprint
    assert len(ds.list_descriptors(db=db)) == 2


def test_main_cli(monkeypatch, capsys):
    no_cuda(monkeypatch)
    assert pr.main(["--quick", "--no-save"]) == 0
    out = capsys.readouterr().out
    assert "tier=C" in out
    assert pr.main(["--quick", "--no-save", "--json"]) == 0
    assert '"descriptor"' in capsys.readouterr().out
