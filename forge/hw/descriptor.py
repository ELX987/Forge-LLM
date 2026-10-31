"""Build :class:`forge.core.schemas.ChipDescriptor` objects (Plan Appendix B) from catalogue specs, and helpers
for fingerprints, ridge points and provenance.

A descriptor built here from the catalogue alone is *datasheet-only*: every number is tagged ``vendor``
(datasheet), ``derived`` (computed from SMs x clock x per-SM rate) or ``untested`` (an approximate/uncertain
figure - treat as a hint only).  :mod:`forge.hw.probe` overwrites fields with ``measured`` values.

Conventions
-----------
* ``on_chip.smem_per_cu_kib`` = usable opt-in shared memory per block (227 on H100, as in Plan Appendix B).
* ``interconnect.intra_gbs`` = per-direction GB/s (NVLink 900 GB/s total -> 450), matching Appendix B.
* ``derived.ridge_flop_per_byte[dtype]`` = peak FLOP/s / HBM bytes/s using the *measured* number when present,
  else the datasheet one.  Keys: fp32, tf32, fp16, bf16, fp8, int8, int4, fp4.
* ``matrix_units[*].dtype`` combines dtypes with identical hardware rate (``fp16|bf16``, ``fp8_e4m3|fp8_e5m2``).
"""
from __future__ import annotations

import re
from typing import Any

from forge.core.hashing import hash_json
from forge.core.schemas import (
    AluInfo,
    ChipDescriptor,
    DerivedInfo,
    InterconnectInfo,
    MatrixUnit,
    MeasuredPair,
    MemoryInfo,
    OnChipInfo,
    Provenance,
    SparsityInfo,
)
from forge.hw import chips
from forge.hw.capabilities import sm_features, tier_for_sm

_PCIE_GBS_PER_DIR = {3: 15.75, 4: 31.5, 5: 63.0, 6: 121.0}

#: short dtype key -> MatrixUnit.dtype string used for that key
SHORT_TO_MU = {
    "fp16": "fp16|bf16", "bf16": "fp16|bf16", "tf32": "tf32", "fp8": "fp8_e4m3|fp8_e5m2", "int8": "int8",
    "int4": "int4", "fp4": "fp4_e2m1",
}
_MU_TO_SHORTS = {
    "fp16|bf16": ["fp16", "bf16"], "fp16": ["fp16"], "tf32": ["tf32"], "fp8_e4m3|fp8_e5m2": ["fp8"], "int8": ["int8"],
    "int4": ["int4"], "fp4_e2m1": ["fp4"], "fp32": ["fp32"],
}

NVFP4_BLOCK = {"block": 16, "scale": "e4m3", "scale_atom": [128, 4]}
MXFP4_BLOCK = {"block": 32, "scale": "e8m0", "scale_atom": [128, 4]}


def shorts_of(mu: MatrixUnit) -> list[str]:
    """Short dtype keys (``fp16``, ``bf16``, ``fp8`` ...) covered by a matrix-unit entry."""
    if mu.dtype in _MU_TO_SHORTS:
        return list(_MU_TO_SHORTS[mu.dtype])
    return [p.strip() for p in mu.dtype.split("|")]


# --------------------------------------------------------------------------------------------
# fingerprints / ids
# --------------------------------------------------------------------------------------------
def chip_fingerprint(name: str, sm: int | None, num_sms: int | None, mem_mb: int | float | None,
                     vendor: str = "nvidia") -> str:
    """Stable hash identifying a device *model* (same for two identical boards; independent of driver, UUID,
    clocks and power settings).  Autotune caches key on this plus driver/triton versions."""
    ident = {
        "v": 1, "vendor": vendor, "name": chips.normalize_name(name), "sm": sm, "num_sms": num_sms,
        "mem_mb": int(round(mem_mb / 64.0)) * 64 if mem_mb else None,  # tolerate small reserved-memory jitter
    }
    return hash_json(ident)


def make_id(name: str, fingerprint: str, revision: int = 1) -> str:
    """``'h100-80gb-hbm3-1a2b3c4d-r1'``."""
    slug = re.sub(r"[^a-z0-9]+", "-", chips.normalize_name(name)).strip("-")[:40] or "device"
    short = fingerprint.split(":", 1)[-1][:8]
    return f"{slug}-{short}-r{int(revision)}"


# --------------------------------------------------------------------------------------------
# matrix units
# --------------------------------------------------------------------------------------------
def _atoms(sm: int) -> dict[str, list[int]]:
    if sm >= 120:
        return {"fp16": [16, 8, 16], "tf32": [16, 8, 8], "int8": [16, 8, 32], "fp8": [16, 8, 32], "fp4": [16, 8, 64]}
    if sm >= 100:
        return {"fp16": [128, 0, 16], "tf32": [128, 0, 8], "int8": [128, 0, 32], "fp8": [128, 0, 32],
                "fp4": [128, 0, 64]}
    if sm >= 90:
        return {"fp16": [64, 0, 16], "tf32": [64, 0, 8], "int8": [64, 0, 32], "fp8": [64, 0, 32]}
    if sm >= 89:
        return {"fp16": [16, 8, 16], "tf32": [16, 8, 8], "int8": [16, 8, 32], "fp8": [16, 8, 32]}
    if sm >= 80:
        return {"fp16": [16, 8, 16], "tf32": [16, 8, 8], "int8": [16, 8, 32], "int4": [16, 8, 64]}
    if sm >= 75:
        return {"fp16": [16, 8, 8], "int8": [8, 8, 16], "int4": [8, 8, 32]}
    return {"fp16": [8, 8, 4]}


def _prov_of(basis: str | None) -> Provenance:
    return {"datasheet": Provenance.vendor, "derived": Provenance.derived}.get(basis or "", Provenance.untested)


def build_matrix_units(sm: int, tflops: dict[str, float], basis: dict[str, str], *,
                       include_fp4: bool = True) -> tuple[list[MatrixUnit], dict[str, Provenance]]:
    """Matrix-unit list for an SM level using catalogue dense TFLOP/s.  Returns (units, provenance by short key)."""
    atoms = _atoms(sm)
    feats = sm_features(sm)
    units: list[MatrixUnit] = []
    prov: dict[str, Provenance] = {}

    def add(short: str, dtype: str, block: Any = None, key: str | None = None) -> None:
        v = tflops.get(key or short)
        if v is None:
            return
        b = basis.get(key or short)
        src = {"datasheet": "datasheet", "derived": "derived:num_sms*clock*rate"}.get(b or "", "approximate")
        units.append(MatrixUnit(dtype=dtype, atom=atoms.get(short), peak_tflops=MeasuredPair(datasheet=float(v), src=src),
                                block_scaled=block))
        prov[f"matrix_units.{short}"] = _prov_of(b)

    if feats["fp16_tensor"]:
        if sm >= 80:
            add("fp16", "fp16|bf16")
        else:
            add("fp16", "fp16")
    if feats["tf32"]:
        add("tf32", "tf32")
    if feats["int8_tensor"]:
        add("int8", "int8")
    if sm in (75, 80, 86, 87, 89):
        add("int4", "int4")
    if feats["fp8"]:
        add("fp8", "fp8_e4m3|fp8_e5m2", block={"mxfp8": {"block": 32, "scale": "e8m0"}} if sm >= 100 else None)
    if feats["fp4_block_scaled"] and include_fp4:
        add("fp4", "fp4_e2m1", block={**NVFP4_BLOCK, "mxfp4": dict(MXFP4_BLOCK)})
    return units, prov


# --------------------------------------------------------------------------------------------
# ridge / derived
# --------------------------------------------------------------------------------------------
def ridge_points(desc: ChipDescriptor, *, datasheet_only: bool = False) -> dict[str, float]:
    """FLOP/byte ridge per dtype key.  Uses measured bandwidth/throughput when available (unless
    ``datasheet_only``), else the datasheet numbers."""
    bw = desc.memory.hbm_bw_gbs
    bw_v = bw.datasheet if datasheet_only else bw.best
    out: dict[str, float] = {}
    if not bw_v or bw_v <= 0:
        return out
    for mu in desc.matrix_units:
        peak = mu.peak_tflops.datasheet if datasheet_only else mu.peak_tflops.best
        if not peak or peak <= 0:
            continue
        for short in shorts_of(mu):
            out[short] = round(peak * 1e12 / (bw_v * 1e9), 3)
    return out


def refresh_derived(desc: ChipDescriptor) -> ChipDescriptor:
    """Recompute ``derived.ridge_flop_per_byte`` from the current best numbers (in place)."""
    desc.derived = DerivedInfo(ridge_flop_per_byte=ridge_points(desc))
    if desc.derived.ridge_flop_per_byte:
        desc.probe_provenance["derived.ridge_flop_per_byte"] = Provenance.derived
    return desc


# --------------------------------------------------------------------------------------------
# spec -> descriptor
# --------------------------------------------------------------------------------------------
def descriptor_from_spec(spec: dict[str, Any], *, id: str | None = None, name: str | None = None,
                         total_mem_gb: float | None = None, fingerprint: str | None = None,
                         revision: int = 1, include_fp4: bool = True) -> ChipDescriptor:
    """Datasheet-only descriptor from a catalogue/generic spec dict (see :mod:`forge.hw.chips`)."""
    sm = spec.get("sm")
    smi = int(sm) if sm else 0
    disp = name or spec.get("display") or "unknown"
    nsm = int(spec.get("num_sms") or 0)
    mem_gb = float(total_mem_gb if total_mem_gb is not None else (spec.get("mem_gb") or 0.0))
    fp = fingerprint or chip_fingerprint(disp, sm, nsm, mem_gb * 1024)
    tflops = dict(spec.get("tflops") or {})
    basis = dict(spec.get("tflops_basis") or {})
    prov: dict[str, Provenance] = {}
    conf = (spec.get("confidence") or {}).get("specs", "approximate")
    spec_prov = Provenance.vendor if conf == "datasheet" else Provenance.untested

    units: list[MatrixUnit] = []
    if spec.get("tensor_cores", True) and smi >= 70:
        units, mprov = build_matrix_units(smi, tflops, basis, include_fp4=include_fp4)
        prov.update(mprov)

    feats = sm_features(smi)
    async_copy: list[str] = []
    if feats["cp_async"]:
        async_copy.append("cp_async")
    if smi in (90, 100, 101, 103, 110, 120, 121):
        async_copy.append("bulk_async")

    nv = spec.get("nvlink_gbs")
    form = spec.get("form") or "pcie"
    if nv:
        intra = float(nv) / 2.0
        if form in ("sxm", "module"):
            topology = "nvlink_mesh" if smi < 80 else "switch"
        else:
            topology = "bridge"
    elif form == "module":  # SoC / superchip: no GPU-GPU link described
        intra, topology = None, None
    else:
        intra = _PCIE_GBS_PER_DIR.get(int(spec.get("pcie_gen") or 0))
        topology = "pcie" if intra else None

    fp32_tf = tflops.get("fp32")
    lanes = spec.get("fp32_lanes_per_sm")
    alu_ops = None
    if nsm and spec.get("boost_mhz") and lanes:
        alu_ops = round(nsm * lanes * spec["boost_mhz"] / 1e6, 2)  # T simple ops/s (1 op/lane/clk)
    alu = AluInfo(simple_ops_per_s_t=alu_ops, fp32_tflops=fp32_tf, fp32_lanes_per_sm=lanes)
    if alu_ops is not None:
        prov["alu.simple_ops_per_s_t"] = Provenance.derived

    bw = spec.get("mem_bw_gbs")
    desc = ChipDescriptor(
        id=id or make_id(disp, fp, revision),
        fingerprint=fp,
        vendor="nvidia",
        family=spec.get("family") or "unknown",
        name=disp,
        sm=smi or None,
        compute_capability=f"{smi // 10}.{smi % 10}" if smi else "",
        compute_units=nsm,
        clock_mhz=spec.get("boost_mhz"),
        memory=MemoryInfo(
            hbm_gb=mem_gb,
            hbm_bw_gbs=MeasuredPair(datasheet=bw, src="datasheet" if conf == "datasheet" else "approximate"),
            l2_mb=float(spec.get("l2_mb") or 0.0), transaction_bytes=128, max_vector_load_bytes=16),
        on_chip=OnChipInfo(
            smem_per_cu_kib=float(spec.get("smem_per_sm_kib") or 0.0), smem_banks=32, bank_width_bytes=4,
            regs_per_cu=int(spec.get("regs_per_sm") or 65536), max_regs_per_thread=255, async_copy=async_copy,
            cluster_launch=feats["cluster_launch"]),
        matrix_units=units,
        sparsity=SparsityInfo(two_four=bool(spec.get("two_four")) and bool(units)),
        alu=alu,
        launch_latency_us=None,
        interconnect=InterconnectInfo(intra_gbs=intra, inter_gbs=None, topology=topology),
        power_limit_w=spec.get("tdp_w"),
        tier=tier_for_sm(smi) if units else "C",
    )
    for key, val in (("name", spec_prov), ("compute_units", spec_prov), ("clock_mhz", spec_prov),
                     ("memory.hbm_gb", spec_prov), ("memory.hbm_bw_gbs", spec_prov), ("memory.l2_mb", spec_prov),
                     ("on_chip.smem_per_cu_kib", Provenance.vendor), ("interconnect", spec_prov),
                     ("power_limit_w", spec_prov), ("tier", Provenance.derived)):
        prov[key] = val
    desc.probe_provenance = prov
    refresh_derived(desc)
    return desc


def descriptor_from_chip(key_or_name: str, *, id: str | None = None) -> ChipDescriptor:
    """Datasheet-only descriptor for a catalogue key or free-form device name (for the planner / docs / tests)."""
    m = chips.resolve_key_or_name(key_or_name)
    d = descriptor_from_spec(m.spec, id=id)
    return d


def summarize(desc: ChipDescriptor) -> dict[str, Any]:
    """Small dict for lists/UI: id, name, sm, tier, memory, measured flags."""
    measured = [k for k, v in desc.probe_provenance.items() if v == Provenance.measured]
    return {
        "id": desc.id, "name": desc.name, "family": desc.family, "sm": desc.sm, "tier": desc.tier,
        "fingerprint": desc.fingerprint, "compute_units": desc.compute_units, "hbm_gb": desc.memory.hbm_gb,
        "hbm_bw_gbs": desc.memory.hbm_bw_gbs.best, "measured_fields": measured, "n_measured": len(measured),
        "dtypes": sorted({s for mu in desc.matrix_units for s in shorts_of(mu)}),
    }
