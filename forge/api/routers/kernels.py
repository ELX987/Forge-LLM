"""/kernels : catalog, benchmark results, bench job, autotune cache."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, Query

from forge.api.common import lazy_module
from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.routers.evals import _columns, _decode_row
from forge.api.schemas import BenchRequest
from forge.core.paths import autotune_dir
from forge.jobs.repo import to_jsonable

router = APIRouter(prefix="/kernels", tags=["kernels"])


@router.get("", summary="Kernel catalog with availability on this host")
def catalog() -> dict[str, Any]:
    mod = lazy_module("forge.kernels.catalog")
    if mod is None or not hasattr(mod, "list_kernels"):
        return {"kernels": [], "available": False, "reason": "forge.kernels.catalog is not importable on this host", "gpu_note": _GPU_NOTE}
    try:
        return {"kernels": to_jsonable(mod.list_kernels()), "available": True, "reason": None, "gpu_note": _GPU_NOTE}
    except Exception as e:
        return {"kernels": [], "available": False, "reason": f"{type(e).__name__}: {e}", "gpu_note": _GPU_NOTE}


_GPU_NOTE = "Triton kernels are implemented but have not been validated on NVIDIA hardware until GPU tests (tests_gpu/) pass on your host."


@router.get("/results", summary="Benchmark results (table kernel_bench)")
def results(
    kernel: str | None = None,
    chip: str | None = None,
    limit: int = Query(500, ge=1, le=5000),
    svc: Services = Depends(get_services),
) -> list[dict[str, Any]]:
    db = svc.metrics.db
    lazy_module("forge.kernels.bench")  # importing it registers the kernel_bench table
    cols = _columns(db, "kernel_bench")
    if not cols:
        return []
    conds, params = [], []
    for col, val in (("kernel", kernel), ("chip", chip)):
        if val and col in cols:
            conds.append(f"{col}=?")
            params.append(val)
        elif val and col == "chip":
            for alt in ("chip_id", "fingerprint", "chip_fingerprint"):
                if alt in cols:
                    conds.append(f"{alt}=?")
                    params.append(val)
                    break
    order = "created_at DESC" if "created_at" in cols else "rowid DESC"
    sql = "SELECT * FROM kernel_bench" + (" WHERE " + " AND ".join(conds) if conds else "") + f" ORDER BY {order} LIMIT ?"
    try:
        return [_decode_row(r) for r in db.query(sql, (*params, limit))]
    except Exception:
        return []


@router.post("/bench", status_code=201, summary="Run kernel benchmarks (job type `bench`)")
def bench(body: BenchRequest | None = None, svc: Services = Depends(get_services)) -> dict[str, Any]:
    body = body or BenchRequest()
    return submit(svc, "bench", body.params, name=body.name or "kernel bench", gpus=body.gpus)


@router.get("/autotune-cache", summary="Autotune cache files (JSON) under FORGE_HOME/autotune")
def autotune_cache(include: bool = Query(True, description="Include file contents")) -> list[dict[str, Any]]:
    out = []
    d = autotune_dir()
    for f in sorted(d.rglob("*.json")):
        item: dict[str, Any] = {"file": str(f.relative_to(d)), "size": f.stat().st_size, "modified": f.stat().st_mtime}
        if include and f.stat().st_size < 5 * 2**20:
            try:
                item["content"] = json.loads(f.read_text(encoding="utf-8"))
            except ValueError:
                item["content"] = None
        out.append(item)
    return out
