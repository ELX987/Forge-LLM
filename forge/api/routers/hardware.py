"""/hardware : chip catalogue, support matrix, measured descriptors, probe."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from forge.api.common import lazy_module, require_module
from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.core.errors import NotFound
from forge.jobs.repo import to_jsonable

router = APIRouter(prefix="/hardware", tags=["hardware"])

# docs/ARCHITECTURE.md section 3 (used only if forge.hw.capabilities.support_matrix is unavailable)
_FALLBACK_MATRIX = [
    {"sm": 70, "examples": "V100", "bf16": "emu", "fp8_mma": False, "fp4_block_scaled_mma": False, "notes": "fp16 only; Triton ok"},
    {"sm": 75, "examples": "T4, RTX 20xx", "bf16": "emu", "fp8_mma": False, "fp4_block_scaled_mma": False, "notes": "fp16 only"},
    {"sm": 80, "examples": "A100, A30", "bf16": True, "fp8_mma": False, "fp4_block_scaled_mma": False, "notes": "cp.async, 2:4 sparse"},
    {"sm": 86, "examples": "RTX 30xx, A10, A40", "bf16": True, "fp8_mma": False, "fp4_block_scaled_mma": False, "notes": ""},
    {"sm": 87, "examples": "Jetson Orin", "bf16": True, "fp8_mma": False, "fp4_block_scaled_mma": False, "notes": ""},
    {"sm": 89, "examples": "RTX 40xx, L4, L40S", "bf16": True, "fp8_mma": True, "fp4_block_scaled_mma": False, "notes": ""},
    {"sm": 90, "examples": "H100, H200, GH200", "bf16": True, "fp8_mma": True, "fp4_block_scaled_mma": False, "notes": "TMA, wgmma"},
    {"sm": 100, "examples": "B200, GB200", "bf16": True, "fp8_mma": True, "fp4_block_scaled_mma": True, "notes": "tcgen05"},
    {"sm": 103, "examples": "B300", "bf16": True, "fp8_mma": True, "fp4_block_scaled_mma": True, "notes": ""},
    {"sm": 120, "examples": "RTX 50xx, RTX PRO 6000 Blackwell", "bf16": True, "fp8_mma": True, "fp4_block_scaled_mma": "verify by probe", "notes": "consumer Blackwell"},
    {"sm": 121, "examples": "GB10 (DGX Spark)", "bf16": True, "fp8_mma": True, "fp4_block_scaled_mma": "verify by probe", "notes": ""},
]


@router.get("/chips", summary="Chip catalogue (vendor datasheet data) + GPU support matrix")
def chips(sm: int | None = Query(None), family: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"chips": [], "support_matrix": _FALLBACK_MATRIX, "source": None, "note": "Datasheet values are provenance=vendor; the probe replaces them with measured values."}
    hw = lazy_module("forge.hw.chips")
    if hw is not None and hasattr(hw, "chip_summary"):
        rows = hw.chip_summary()
        out["chips"] = [r for r in rows if (sm is None or r.get("sm") == sm) and (family is None or r.get("family") == family)]
        out["source"] = "forge.hw.chips"
    caps = lazy_module("forge.hw.capabilities")
    if caps is not None and hasattr(caps, "support_matrix"):
        try:
            out["support_matrix"] = to_jsonable(caps.support_matrix())
        except Exception:
            pass
    return out


@router.get("/chips/{key}", summary="One catalogue entry")
def chip(key: str) -> dict[str, Any]:
    hw = require_module("forge.hw.chips", "chip catalogue")
    try:
        return to_jsonable(hw.get_chip(key))
    except KeyError:
        raise NotFound(f"no chip {key!r} in the catalogue") from None


@router.get("/descriptors", summary="Measured chip descriptors (newest first)")
def descriptors(limit: int = Query(100, ge=1, le=1000)) -> list[dict[str, Any]]:
    ds = lazy_module("forge.hw.descriptor_store")
    if ds is None:
        return []
    return to_jsonable(ds.list_descriptors(limit=limit))


@router.get("/descriptors/{descriptor_id}", summary="One descriptor (+ raw probe report when stored)")
def descriptor(descriptor_id: str, report: bool = False) -> dict[str, Any]:
    ds = require_module("forge.hw.descriptor_store", "descriptors")
    d = ds.load_descriptor(descriptor_id)
    out = {"descriptor": to_jsonable(d)}
    if report and hasattr(ds, "load_report"):
        out["report"] = to_jsonable(ds.load_report(descriptor_id))
    return out


@router.post("/probe", status_code=201, summary="Run the hardware probe (job type `probe`)")
def probe(params: dict[str, Any] | None = None, svc: Services = Depends(get_services)) -> dict[str, Any]:
    return submit(svc, "probe", params or {}, name="hardware probe")
