"""Capability probe: builds a full :class:`ChipDescriptor` (Plan Appendix B) for a device, with per-field provenance.

Flow (GPU): ``torch.cuda.get_device_properties`` + NVML -> catalogue match -> datasheet descriptor -> overwrite with
driver-reported facts (SM count, memory, shared memory, L2, clocks, power limit) -> run micro-benchmarks
(:mod:`forge.hw.microbench`) while sampling clocks/temperature/throttle reasons -> write ``measured`` values next to the
datasheet ones -> confirm or drop FP8/FP4 -> recompute effective ridge points -> assign tier -> persist.

No NVIDIA GPU: a tier-"C" CPU descriptor (cores, RAM, optional tiny copy/GEMM measurements).  If ``nvidia-smi`` sees a
GPU that torch cannot use, a datasheet-only descriptor with tier "C" is returned (it cannot run our kernels).

Provenance tags in ``ChipDescriptor.probe_provenance``:
``measured`` = produced by a micro-benchmark in this probe run; ``vendor`` = datasheet or driver-reported;
``derived`` = computed (ridge points, SMs x clock x rate); ``untested`` = unverified/approximate figure or a feature that
could not be confirmed.  ``feature.<name>`` keys record whether fp8/fp4/bf16/int8 were confirmed by running them.

STATUS: GPU paths are implemented but not validated on NVIDIA hardware by the author; CPU paths and the logic around
mocked ``torch.cuda`` / ``pynvml`` are unit-tested.

CLI: ``python -m forge.hw.probe [--device N] [--quick] [--no-save] [--json]``.
"""
from __future__ import annotations

import logging
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from forge.core.schemas import (
    AluInfo,
    ChipDescriptor,
    InterconnectInfo,
    MatrixUnit,
    MeasuredPair,
    MemoryInfo,
    Provenance,
)
from forge.hw import capabilities as caps_mod
from forge.hw import chips, descriptor as desc_mod
from forge.hw import microbench as mb
from forge.hw import telemetry
from forge.hw.config import ProbeConfig

log = logging.getLogger("forge.hw.probe")

__all__ = ["ProbeConfig", "ProbeResult", "run_probe", "run_probe_full", "run_probe_job", "register_check",
           "cpu_descriptor", "main"]

REPORT_SCHEMA = 1

# extra checks other modules may register: name -> fn(device:int, quick:bool) -> dict (should contain "ok": bool)
_CHECKS: dict[str, Callable[[int, bool], dict[str, Any]]] = {}


def register_check(name: str, fn: Callable[[int, bool], dict[str, Any]]) -> None:
    """Register an additional probe check (e.g. a kernel owner's 'does tl.dot_scaled work here?' test).  ``fn`` gets
    ``(device, quick)`` and returns a JSON-able dict; ``{"ok": bool}`` is the convention.  Results land in
    ``report['checks'][name]``; exceptions are caught and recorded."""
    _CHECKS[name] = fn


@dataclass
class ProbeResult:
    """Descriptor plus the raw report (benchmark results, throttling, feature confirmations, notes)."""

    descriptor: ChipDescriptor
    report: dict[str, Any]
    saved_id: str | None = None


class _Reporter:
    """Adapter so the probe works with or without a RunContext."""

    def __init__(self, ctx: Any | None) -> None:
        self.ctx = ctx

    def log(self, msg: str, level: str = "info") -> None:
        if self.ctx is not None:
            self.ctx.log(msg, level)
        else:
            log.log(getattr(logging, level.upper(), logging.INFO), msg)

    def progress(self, frac: float, msg: str = "") -> None:
        if self.ctx is not None:
            self.ctx.progress(frac, msg)

    def stop(self) -> bool:
        return bool(self.ctx is not None and self.ctx.should_stop())


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _unit_for(desc: ChipDescriptor, short: str) -> MatrixUnit | None:
    for mu in desc.matrix_units:
        if short in desc_mod.shorts_of(mu):
            return mu
    return None


def _set_measured(mu: MatrixUnit, value: float, src: str) -> None:
    mu.peak_tflops = MeasuredPair(datasheet=mu.peak_tflops.datasheet, measured=round(float(value), 3), src=src)


def _ensure_unit(desc: ChipDescriptor, short: str) -> MatrixUnit:
    """Matrix unit for ``short`` (created without a datasheet number if the catalogue had none)."""
    mu = _unit_for(desc, short)
    if mu is None:
        mu = MatrixUnit(dtype=desc_mod.SHORT_TO_MU[short], atom=desc_mod._atoms(desc.sm or 0).get(short),
                        peak_tflops=MeasuredPair(), block_scaled=None)
        desc.matrix_units.append(mu)
    return mu


def _gpu_visible_info(device: int) -> dict[str, Any]:
    import torch

    p = torch.cuda.get_device_properties(device)
    return {
        "name": str(getattr(p, "name", "unknown")), "major": int(getattr(p, "major", 0)),
        "minor": int(getattr(p, "minor", 0)), "total_memory": int(getattr(p, "total_memory", 0)),
        "multi_processor_count": int(getattr(p, "multi_processor_count", 0)),
        "smem_optin": getattr(p, "shared_memory_per_block_optin", None),
        "smem_per_sm": getattr(p, "shared_memory_per_multiprocessor", None),
        "l2_bytes": getattr(p, "L2_cache_size", None), "uuid": str(getattr(p, "uuid", "") or "") or None,
        "regs_per_sm": getattr(p, "regs_per_multiprocessor", None),
        "clock_khz": getattr(p, "clock_rate", None),
    }


def _environment() -> dict[str, Any]:
    try:
        from forge.core.env import capture_environment

        env = capture_environment()
        t = env.get("torch", {})
        return {"python": env.get("python"), "platform": env.get("platform"), "torch": t.get("version"),
                "cuda_runtime": t.get("cuda_version"), "cudnn": t.get("cudnn_version"), "triton": env.get("triton"),
                "cuda_driver": env.get("cuda_driver"), "git_sha": env.get("git_sha"),
                "packages": env.get("packages", {})}
    except Exception as e:  # pragma: no cover - defensive
        return {"error": repr(e)}


def _expected_dtypes(sm: int) -> list[str]:
    out = ["fp32"]
    if sm >= 80:
        out.append("tf32")
    out.append("fp16")
    if sm >= 80:
        out.append("bf16")
    if sm >= 75:
        out.append("int8")
    if sm >= 89:
        out.append("fp8")
    if sm >= 100:
        out.append("fp4")
    return out


def _tier_after_probe(desc: ChipDescriptor, features: dict[str, bool | None], gemm_attempted: bool,
                      gemm_ok: bool) -> str:
    if desc.sm is None or desc.sm < 70:
        return "C"
    tier = caps_mod.tier_for_sm(desc.sm)
    if features.get("fp16") is False or features.get("bf16") is False or (gemm_attempted and not gemm_ok):
        tier = caps_mod.demote_tier(tier)
    if desc.sm >= 90 and features.get("fp8") is False:
        tier = caps_mod.demote_tier(tier)
    return tier


# --------------------------------------------------------------------------------------------
# GPU probe
# --------------------------------------------------------------------------------------------
def _build_gpu_descriptor(device: int, cfg: ProbeConfig, rep: _Reporter, db: Any | None) -> tuple[ChipDescriptor, dict]:
    info = _gpu_visible_info(device)
    sm = info["major"] * 10 + info["minor"]
    nsm = info["multi_processor_count"]
    mem_gib = info["total_memory"] / 2**30
    notes: list[str] = []
    sampler = telemetry.GpuSampler()
    try:
        rows = sampler.sample()
        nvml_idx = telemetry.resolve_nvml_index(device, info["uuid"], rows)
        mine = next((r for r in rows if r["index"] == nvml_idx), None)
    finally:
        sampler.close()
    clock_mhz = (mine or {}).get("clock_max_mhz") or (info["clock_khz"] / 1000.0 if info["clock_khz"] else None)
    match = chips.find_chip(info["name"], sm=sm, num_sms=nsm or None, mem_gb=mem_gib or None, clock_mhz=clock_mhz)
    notes.extend(match.notes)
    fp = desc_mod.chip_fingerprint(info["name"], sm, nsm, info["total_memory"] / 2**20)
    try:
        from forge.hw import descriptor_store

        rev = descriptor_store.next_revision(fp, db=db)
    except Exception as e:
        log.debug("revision lookup failed: %s", e)
        rev = 1
    desc = desc_mod.descriptor_from_spec(match.spec, name=info["name"], total_mem_gb=round(mem_gib, 2), fingerprint=fp,
                                         revision=rev)
    prov = desc.probe_provenance
    field_src: dict[str, str] = {k: ("datasheet" if v == Provenance.vendor else v.value) for k, v in prov.items()}

    # driver-reported facts replace catalogue guesses -------------------------------------------------------
    desc.compute_units = nsm or desc.compute_units
    prov["compute_units"] = Provenance.vendor
    field_src["compute_units"] = "driver_query"
    desc.memory.hbm_gb = round(mem_gib, 2)
    prov["memory.hbm_gb"] = Provenance.vendor
    field_src["memory.hbm_gb"] = "driver_query"
    if info["smem_optin"]:
        desc.on_chip.smem_per_cu_kib = round(info["smem_optin"] / 1024.0, 2)
        field_src["on_chip.smem_per_cu_kib"] = "driver_query"
    if info["l2_bytes"]:
        desc.memory.l2_mb = round(info["l2_bytes"] / 2**20, 2)
        prov["memory.l2_mb"] = Provenance.vendor
        field_src["memory.l2_mb"] = "driver_query"
    if info["regs_per_sm"]:
        desc.on_chip.regs_per_cu = int(info["regs_per_sm"])
    if clock_mhz:
        desc.clock_mhz = float(clock_mhz)
        prov["clock_mhz"] = Provenance.vendor
        field_src["clock_mhz"] = "driver_query (max SM clock)"
    if mine and mine.get("power_limit_w"):
        desc.power_limit_w = float(mine["power_limit_w"])
        prov["power_limit_w"] = Provenance.vendor
        field_src["power_limit_w"] = "driver_query"
    if mine is None:
        notes.append("NVML/nvidia-smi telemetry unavailable: clocks, power limit and throttling not recorded")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA, "created_at": time.time(), "device": device, "nvml_index": mine["index"] if mine else None,
        "quick": cfg.quick, "label": cfg.label, "match": match.to_dict(), "props": {k: v for k, v in info.items()},
        "benchmarks": {}, "checks": {}, "features": {}, "throttle": {"available": False}, "notes": notes,
        "field_sources": field_src, "datasheet_ridge": desc_mod.ridge_points(desc, datasheet_only=True),
        "environment": _environment(),
    }

    if not cfg.microbench:
        notes.append("microbench disabled: descriptor is datasheet + driver facts only; fp8/fp4 unconfirmed")
        report["features"] = {"fp8": None, "fp4": None, "bf16": None, "fp16": None, "int8": None}
        _finalise(desc, report, {})
        return desc, report

    want = set(cfg.benchmarks)
    results: dict[str, mb.BenchResult] = {}
    dts = [d for d in _expected_dtypes(sm) if not cfg.dtypes or d in {mb.canon_dtype(x) for x in cfg.dtypes}]
    steps: list[tuple[str, Callable[[], mb.BenchResult]]] = []
    q = cfg.quick
    if "bandwidth" in want:
        steps.append(("stream_copy_bw", lambda: mb.stream_copy_bw(device, iters=10 if q else 30,
                                                                  max_seconds=1.5 if q else 4.0)))
        steps.append(("stream_write_bw", lambda: mb.stream_copy_bw(device, kind="write", iters=10 if q else 30,
                                                                   max_seconds=1.0 if q else 3.0)))
    if "gemm" in want:
        for d in dts:
            steps.append((f"mma_{d}", lambda d=d: mb.mma_throughput(d, device, quick=q, max_seconds=0.8 if q else 2.0)))
    if "onchip" in want:
        steps.append(("smem_bw_proxy", lambda: mb.smem_bw(device, iters=5 if q else 10)))
    if "launch" in want:
        steps.append(("launch_latency", lambda: mb.launch_latency(device, repeats=3 if q else 5)))
    if "dequant" in want:
        steps.append(("dequant_alu", lambda: mb.dequant_alu_throughput(device, iters=5 if q else 10)))

    try:
        import torch

        n_gpu = int(torch.cuda.device_count())
    except Exception:
        n_gpu = 1
    if "p2p" in want and not q and n_gpu >= 2:
        other = 1 if device == 0 else 0
        steps.append((f"p2p_bw_{device}_{other}", lambda: mb.p2p_bw(device, other)))
    n_checks = len(_CHECKS) if "extra_checks" in want else 0
    total_steps = max(1, len(steps) + n_checks)

    monitor = telemetry.ThrottleMonitor(mine["index"]) if (cfg.record_throttling and mine) else None
    if monitor is not None:
        monitor.__enter__()
    try:
        for i, (name, fn) in enumerate(steps):
            if rep.stop():
                notes.append("probe cancelled before completion; partial results")
                break
            rep.progress(0.1 + 0.8 * i / total_steps, name)
            res = fn()
            results[name] = res
            rep.log(f"{name}: " + (f"{res.value:.2f} {res.unit}" if res.ok and res.value is not None
                                   else f"unavailable ({res.error})"))
        for j, (cname, cfn) in enumerate(list(_CHECKS.items()) if n_checks else []):
            if rep.stop():
                break
            rep.progress(0.1 + 0.8 * (len(steps) + j) / total_steps, f"check {cname}")
            try:
                report["checks"][cname] = cfn(device, q)
            except Exception as e:
                report["checks"][cname] = {"ok": False, "error": repr(e)}
    finally:
        if monitor is not None:
            monitor.__exit__(None, None, None)
            report["throttle"] = monitor.summary()
    report["benchmarks"] = {k: v.to_dict() for k, v in results.items()}
    _apply_measurements(desc, report, results, sm, notes)
    _finalise(desc, report, results)
    return desc, report


def _apply_measurements(desc: ChipDescriptor, report: dict[str, Any], results: dict[str, mb.BenchResult], sm: int,
                        notes: list[str]) -> None:
    prov = desc.probe_provenance
    src = report["field_sources"]
    features: dict[str, bool | None] = {}

    bw = results.get("stream_copy_bw")
    if bw and bw.ok and bw.value:
        desc.memory.hbm_bw_gbs = MeasuredPair(datasheet=desc.memory.hbm_bw_gbs.datasheet, measured=round(bw.value, 2),
                                              src="microbench/stream_copy_bw")
        prov["memory.hbm_bw_gbs"] = Provenance.measured
        src["memory.hbm_bw_gbs"] = "microbench"
        ds = desc.memory.hbm_bw_gbs.datasheet
        if ds and bw.value > 1.05 * ds:
            notes.append(f"measured copy bandwidth {bw.value:.0f} GB/s exceeds datasheet {ds:.0f} GB/s: the catalogue "
                         "figure is probably wrong for this SKU")
    elif bw and not bw.ok:
        notes.append(f"stream copy bandwidth not measured: {bw.error}")

    per_unit: dict[int, list[float]] = {}
    units_by_id: dict[int, tuple[MatrixUnit, str]] = {}
    for dt in ("fp16", "bf16", "tf32", "fp8", "int8", "fp4"):
        r = results.get(f"mma_{dt}")
        if r is None:
            continue
        if r.ok and r.value:
            mu = _ensure_unit(desc, dt)
            if dt == "fp4" and not mu.block_scaled:
                mu.block_scaled = {**desc_mod.NVFP4_BLOCK, "mxfp4": dict(desc_mod.MXFP4_BLOCK)}
            per_unit.setdefault(id(mu), []).append(r.value)
            units_by_id[id(mu)] = (mu, dt)
            features[dt] = True
            prov[f"matrix_units.{dt}"] = Provenance.measured
            src[f"matrix_units.{dt}"] = "microbench"
        else:
            features[dt] = None if r.skipped else False
    for uid, vals in per_unit.items():
        mu, dt = units_by_id[uid]
        _set_measured(mu, sum(vals) / len(vals), f"microbench/mma_{dt}")
        ds = mu.peak_tflops.datasheet
        if ds and mu.peak_tflops.measured and mu.peak_tflops.measured > 1.1 * ds:
            notes.append(f"measured {dt} throughput {mu.peak_tflops.measured:.0f} exceeds datasheet {ds:.0f}: "
                         "catalogue figure likely too low (e.g. fp16- vs fp32-accumulate rate)")
    f32 = results.get("mma_fp32")
    if f32 and f32.ok and f32.value:
        extra = desc.alu.model_dump()
        extra["fp32_tflops_measured"] = round(f32.value, 3)
        desc.alu = AluInfo(**extra)
        prov["alu.fp32_tflops_measured"] = Provenance.measured
    tf32 = results.get("mma_tf32")
    if tf32 and tf32.ok:
        features["tf32"] = True

    # FP8 / FP4 confirmation policy (Plan HW-4): a low-precision unit is kept only if it ran and was numerically right.
    #   fp8: dropped only when the run FAILED (a skipped run - e.g. old torch - leaves it as an unconfirmed hypothesis)
    #   fp4: dropped unless confirmed (Appendix B: "only if probe confirms"), when the probe attempted it
    for dt in ("fp8", "fp4"):
        attempted = f"mma_{dt}" in results
        if not attempted:
            features.setdefault(dt, None)
        if features.get(dt) is True:
            prov[f"feature.{dt}"] = Provenance.measured
            continue
        prov[f"feature.{dt}"] = Provenance.untested
        mu = _unit_for(desc, dt)
        if mu is None:
            continue
        reason = results[f"mma_{dt}"].error if attempted else "not run"
        drop = attempted and (dt == "fp4" or features.get(dt) is False)
        if drop:
            desc.matrix_units = [m_ for m_ in desc.matrix_units if m_ is not mu]
            prov.pop(f"matrix_units.{dt}", None)
            notes.append(f"{dt} MMA NOT confirmed by the probe ({reason}): unit removed from the descriptor; "
                         "capabilities keep the SM-level hypothesis unless the run reported a numerical failure")
        else:
            prov[f"matrix_units.{dt}"] = Provenance.untested

    for dt in ("bf16", "fp16", "int8"):
        features.setdefault(dt, None)

    ll = results.get("launch_latency")
    if ll and ll.ok and ll.value is not None:
        desc.launch_latency_us = round(ll.value, 3)
        prov["launch_latency_us"] = Provenance.measured
        src["launch_latency_us"] = "microbench/" + str(ll.params.get("method"))
    dq = results.get("dequant_alu")
    if dq and dq.ok and dq.value:
        extra = desc.alu.model_dump()
        extra["simple_ops_per_s_t"] = dq.params.get("simple_ops_per_s_t")
        extra["dequant_gelem_per_s"] = round(dq.value, 2)
        desc.alu = AluInfo(**extra)
        prov["alu.simple_ops_per_s_t"] = Provenance.measured
        src["alu.simple_ops_per_s_t"] = "microbench/dequant_alu (5 ops/element)"
    sm_bw = results.get("smem_bw_proxy")
    if sm_bw and sm_bw.ok and sm_bw.value:
        extra = desc.alu.model_dump()
        extra["onchip_reread_gbs"] = round(sm_bw.value, 1)
        desc.alu = AluInfo(**extra)
        prov["alu.onchip_reread_gbs"] = Provenance.measured
    for k, r in results.items():
        if k.startswith("p2p_bw") and r.ok and r.value:
            desc.interconnect = InterconnectInfo(intra_gbs=round(r.value, 2), inter_gbs=desc.interconnect.inter_gbs,
                                                 topology=desc.interconnect.topology)
            prov["interconnect"] = Provenance.measured
            src["interconnect"] = "microbench/p2p_bw (unidirectional copy)"
    thr = report.get("throttle", {})
    if thr.get("available") and thr.get("clock_sm_mhz_median"):
        desc.clock_mhz = float(thr["clock_sm_mhz_median"])
        prov["clock_mhz"] = Provenance.measured
        src["clock_mhz"] = "telemetry median SM clock during benchmarks"
        if thr.get("throttled"):
            notes.append("throttling observed during benchmarking (" + ", ".join(thr.get("throttle_reasons_seen", []))
                         + "): measured throughput may understate steady-state capability; re-run on a cool, "
                           "unconstrained GPU for datasheet-comparable numbers")
    report["features"] = features


def _finalise(desc: ChipDescriptor, report: dict[str, Any], results: dict[str, mb.BenchResult]) -> None:
    desc_mod.refresh_derived(desc)
    gemm = [v for k, v in results.items() if k.startswith("mma_")]
    desc.tier = _tier_after_probe(desc, report.get("features", {}), bool(gemm), any(v.ok for v in gemm))
    desc.probe_provenance["tier"] = Provenance.derived
    report["effective_ridge"] = dict(desc.derived.ridge_flop_per_byte)
    report["tier"] = desc.tier


# --------------------------------------------------------------------------------------------
# no-torch-CUDA paths
# --------------------------------------------------------------------------------------------
def cpu_descriptor(*, microbench: bool = True, quick: bool = True, db: Any | None = None) -> tuple[ChipDescriptor, dict]:
    """Tier-C descriptor of the host CPU (+ optional tiny copy/GEMM measurements through torch)."""
    ci = caps_mod.cpu_info()
    fp = desc_mod.chip_fingerprint(ci["name"], None, ci["logical_cores"], ci["ram_gb"] * 1024, vendor="cpu")
    try:
        from forge.hw import descriptor_store

        rev = descriptor_store.next_revision(fp, db=db)
    except Exception:
        rev = 1
    prov: dict[str, Provenance] = {"name": Provenance.vendor, "compute_units": Provenance.vendor,
                                   "memory.hbm_gb": Provenance.vendor, "tier": Provenance.derived}
    bw = MeasuredPair()
    units: list[MatrixUnit] = []
    bench: dict[str, Any] = {}
    notes = ["no usable NVIDIA GPU: tier C portable fallback descriptor from CPU information"]
    if microbench:
        try:
            import torch

            n = (16 if quick else 64) * 2**20 // 4
            a, b = torch.ones(n), torch.empty(n)
            b.copy_(a)
            ts = []
            for _ in range(3 if quick else 6):
                t = time.perf_counter()
                b.copy_(a)
                ts.append(time.perf_counter() - t)
            gbs = 2.0 * n * 4 / min(ts) / 1e9
            bw = MeasuredPair(measured=round(gbs, 2), src="microbench/cpu_copy (best of N)")
            prov["memory.hbm_bw_gbs"] = Provenance.measured
            bench["cpu_copy_gbs"] = gbs
            d = 256 if quick else 1024
            x, y = torch.randn(d, d), torch.randn(d, d)
            x @ y
            ts = []
            for _ in range(3 if quick else 5):
                t = time.perf_counter()
                x @ y
                ts.append(time.perf_counter() - t)
            tf = 2.0 * d**3 / min(ts) / 1e12
            units.append(MatrixUnit(dtype="fp32", atom=None, peak_tflops=MeasuredPair(measured=round(tf, 4),
                                                                                       src="microbench/cpu_gemm")))
            prov["matrix_units.fp32"] = Provenance.measured
            bench["cpu_gemm_fp32_tflops"] = tf
        except Exception as e:
            notes.append(f"cpu microbench failed: {e!r}")
    desc = ChipDescriptor(
        id=desc_mod.make_id(ci["name"], fp, rev), fingerprint=fp, vendor="cpu", family=ci["family"], name=ci["name"],
        sm=None, compute_capability="", compute_units=int(ci["logical_cores"]), clock_mhz=ci["max_freq_mhz"],
        memory=MemoryInfo(hbm_gb=ci["ram_gb"], hbm_bw_gbs=bw, l2_mb=0.0, transaction_bytes=64,
                          max_vector_load_bytes=64 if ci["has_avx512"] else 32),
        matrix_units=units, tier="C", probe_provenance=prov,
    )
    desc_mod.refresh_derived(desc)
    report = {"schema": REPORT_SCHEMA, "created_at": time.time(), "device": None, "cpu": ci, "benchmarks": bench,
              "checks": {}, "features": {"fp8": False, "fp4": False, "bf16": False, "fp16": False, "int8": False},
              "throttle": {"available": False}, "notes": notes, "tier": "C",
              "effective_ridge": dict(desc.derived.ridge_flop_per_byte), "environment": _environment()}
    return desc, report


def _smi_only_descriptor(row: dict[str, Any], db: Any | None) -> tuple[ChipDescriptor, dict]:
    """nvidia-smi sees a GPU but torch cannot use CUDA: datasheet-only descriptor, tier C."""
    sm = row.get("sm")
    mem = row.get("total_mem_gb") or None
    match = chips.find_chip(row["name"], sm=sm, num_sms=row.get("num_sms"), mem_gb=mem)
    fp = desc_mod.chip_fingerprint(row["name"], sm, row.get("num_sms"), (mem or 0) * 1024)
    desc = desc_mod.descriptor_from_spec(match.spec, name=row["name"], total_mem_gb=mem, fingerprint=fp,
                                         include_fp4=False)
    desc.tier = "C"
    notes = [f"nvidia-smi reports {row['name']} but torch.cuda is unavailable (CPU-only torch build or driver "
             "mismatch): descriptor is datasheet-only and tier C because Forge GPU kernels cannot run"] + match.notes
    report = {"schema": REPORT_SCHEMA, "created_at": time.time(), "device": row.get("index"), "match": match.to_dict(),
              "benchmarks": {}, "checks": {}, "features": {"fp8": None, "fp4": None}, "notes": notes,
              "throttle": {"available": False}, "tier": "C", "effective_ridge": dict(desc.derived.ridge_flop_per_byte),
              "environment": _environment(), "field_sources": {}}
    return desc, report


# --------------------------------------------------------------------------------------------
# public entry points
# --------------------------------------------------------------------------------------------
def run_probe_full(ctx: Any | None = None, device: int = 0, quick: bool = False, *, config: ProbeConfig | None = None,
                   db: Any | None = None, **overrides: Any) -> ProbeResult:
    """Run the probe and return descriptor + report.  ``config`` (or keyword ``overrides`` such as
    ``microbench=False``) refine behaviour; ``save`` persists to the store (``db`` for tests)."""
    cfg = config or ProbeConfig(device=device, quick=quick, **overrides)
    rep = _Reporter(ctx)
    rep.progress(0.0, "probing hardware")
    cuda_ok, why = mb.cuda_ready(cfg.device)
    if cuda_ok and caps_mod.software_versions().get("hip"):
        cuda_ok, why = False, "ROCm/HIP build of torch (Forge GPU kernels and this probe are NVIDIA-only)"
    if cuda_ok:
        rep.log(f"probing CUDA device {cfg.device}")
        desc, report = _build_gpu_descriptor(cfg.device, cfg, rep, db)
    else:
        smi = [g for g in caps_mod.list_gpus() if g.get("source") == "nvidia-smi"]
        row = next((g for g in smi if g["index"] == cfg.device), None)
        if row is not None:
            rep.log(f"torch CUDA unavailable ({why}); building datasheet descriptor from nvidia-smi", "warning")
            desc, report = _smi_only_descriptor(row, db)
        else:
            rep.log(f"no usable GPU ({why}); building CPU tier-C descriptor", "warning")
            desc, report = cpu_descriptor(microbench=cfg.microbench, quick=cfg.quick, db=db)
    report["config"] = cfg.model_dump(mode="json")
    saved = None
    if cfg.save:
        try:
            from forge.hw import descriptor_store

            saved = descriptor_store.save_descriptor(desc, report=report, source="probe", db=db)
            caps_mod.clear_cache()
            rep.log(f"saved descriptor {saved}")
        except Exception as e:
            report.setdefault("notes", []).append(f"descriptor not saved: {e!r}")
            rep.log(f"could not save descriptor: {e!r}", "warning")
    rep.progress(1.0, "done")
    if ctx is not None:
        try:
            ctx.set_output("descriptor_id", desc.id)
            ctx.set_output("tier", desc.tier)
            ctx.set_output("summary", desc_mod.summarize(desc))
        except Exception:  # pragma: no cover
            pass
    return ProbeResult(descriptor=desc, report=report, saved_id=saved)


def run_probe(ctx: Any | None = None, device: int = 0, quick: bool = False) -> ChipDescriptor:
    """ARCHITECTURE.md 5.2 entry point.  GPU present -> measured descriptor; otherwise a tier-C CPU descriptor.
    The descriptor is saved to the Forge store (see :class:`ProbeConfig` to opt out)."""
    return run_probe_full(ctx, device, quick).descriptor


def run_probe_job(cfg: ProbeConfig, ctx: Any) -> dict[str, Any]:
    """Job entry point for job type ``probe`` (config model :class:`ProbeConfig`)."""
    res = run_probe_full(ctx, config=cfg)
    return {"descriptor_id": res.descriptor.id, "tier": res.descriptor.tier,
            "summary": desc_mod.summarize(res.descriptor), "notes": res.report.get("notes", [])}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(prog="python -m forge.hw.probe", description="Probe the local GPU/CPU")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-microbench", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full descriptor + report as JSON")
    a = ap.parse_args(argv)
    cfg = ProbeConfig(device=a.device, quick=a.quick, save=not a.no_save, microbench=not a.no_microbench)
    res = run_probe_full(config=cfg)
    if a.json:
        print(json.dumps({"descriptor": res.descriptor.model_dump(mode="json"), "report": res.report}, indent=2,
                         default=str))
    else:
        d = res.descriptor
        print(f"{d.name}  [{d.id}]  sm={d.sm}  tier={d.tier}  mem={d.memory.hbm_gb} GB")
        print(f"  HBM bw  datasheet={d.memory.hbm_bw_gbs.datasheet} measured={d.memory.hbm_bw_gbs.measured} GB/s")
        for mu in d.matrix_units:
            print(f"  {mu.dtype:<20} datasheet={mu.peak_tflops.datasheet} measured={mu.peak_tflops.measured} TFLOP/s")
        print(f"  ridge (effective): {d.derived.ridge_flop_per_byte}")
        for n in res.report.get("notes", []):
            print(f"  note: {n}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
