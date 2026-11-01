"""/planner (estimate, plan), /serve/simulate and /claims : thin delegation to forge.planner / forge.serve.capacity."""
from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from forge.api.common import NotImplementedYet, call_flexible, lazy_module, require_module
from forge.api.deps import Services, get_services
from forge.api.schemas import ClaimUpdate, PlannerRequest, ServeSimulateRequest
from forge.core.errors import ConfigError, NotFound
from forge.jobs.modelinfo import model_local_path
from forge.jobs.repo import to_jsonable
from forge.planner import trainsim
from forge.planner.trainsim import TrainSimConfig

router = APIRouter(tags=["planner"])


# --------------------------------------------------------------------------------------------
# input resolution
# --------------------------------------------------------------------------------------------
def resolve_model_cfg(svc: Services, req: PlannerRequest) -> Any:
    """ModelConfig from an inline HF config, a registered model id, or a local directory."""
    models = require_module("forge.models", "model-config resolution")
    MC = models.ModelConfig
    if req.hf_config:
        try:
            return MC.from_hf_config(req.hf_config)
        except Exception:
            return MC.model_validate(req.hf_config)
    if not req.model:
        raise ConfigError("provide `model` (a registered model id or a local directory) or `hf_config`")
    path: str | None = None
    row = svc.models.get(req.model)
    if row:
        path = model_local_path(row)
        if not path:
            raise ConfigError(f"model {req.model} is not downloaded; download it first or pass `hf_config`")
    elif Path(req.model).expanduser().is_dir():
        path = req.model
    else:
        raise NotFound(f"model {req.model!r} is not a registered model id or an existing directory")
    cfg_file = Path(path) / "config.json"
    if not cfg_file.exists():
        raise ConfigError(f"{path} has no config.json")
    return MC.from_hf_config(json.loads(cfg_file.read_text(encoding="utf-8")))


def resolve_chip(req: PlannerRequest) -> Any:
    """ChipDescriptor (measured), Capabilities (catalogue key/name) or the current device's capabilities."""
    hw_ds = lazy_module("forge.hw.descriptor_store")
    hw_caps = lazy_module("forge.hw.capabilities")
    if req.descriptor_id:
        if hw_ds is None:
            raise NotImplementedYet("forge.hw.descriptor_store is not available")
        return hw_ds.load_descriptor(req.descriptor_id)
    if req.chip:
        if hw_caps is None or not hasattr(hw_caps, "capabilities_for_chip"):
            raise NotImplementedYet("forge.hw.capabilities is not available")
        return hw_caps.capabilities_for_chip(req.chip)
    if hw_caps is not None and hasattr(hw_caps, "current_capabilities"):
        return hw_caps.current_capabilities()
    raise ConfigError("provide `chip` (catalogue key or name) or `descriptor_id`")


def resolve_workload(req: PlannerRequest) -> Any:
    from forge.core.schemas import Workload

    if not req.workload:
        return Workload()  # schema defaults

    try:
        return Workload.model_validate(req.workload)
    except Exception as e:
        raise ConfigError(f"invalid workload: {e}") from e


def _callable(mod: Any, name: str) -> Any:
    obj = getattr(mod, name, None)
    if inspect.ismodule(obj):
        obj = getattr(obj, name, None)
    if obj is None or not callable(obj):
        raise NotImplementedYet(f"{mod.__name__}.{name} is not available", details={"module": mod.__name__})
    return obj


def _aliases(svc: Services, req: PlannerRequest) -> dict[tuple[str, ...], Any]:
    return {
        ("model_cfg", "model_config", "cfg", "model", "config"): resolve_model_cfg(svc, req),
        ("chip", "caps", "capabilities", "descriptor", "chip_descriptor", "hw"): resolve_chip(req),
        ("recipe", "quant", "scheme", "quant_scheme", "recipe_or_scheme", "quant_scheme_or_recipe"): req.quant,
        ("workload", "wl"): resolve_workload(req),
        ("kv_dtype", "kv"): req.kv_dtype,
        ("budget",): req.budget or None,
    }


@router.post("/planner/estimate", summary="Quick fit / bound / ceiling estimate for model + chip + scheme")
def estimate(body: PlannerRequest, svc: Services = Depends(get_services)) -> Any:
    mod = require_module("forge.planner", "the planner")
    fn = _callable(mod, "estimate")
    return to_jsonable(call_flexible(fn, _aliases(svc, body), **body.options))


@router.post("/planner/plan", summary="Ranked recipe candidates for a model + chip + workload + budget")
def plan(body: PlannerRequest, svc: Services = Depends(get_services)) -> Any:
    mod = require_module("forge.planner", "the planner")
    search = lazy_module("forge.planner.search")
    fn = None
    for m in (mod, search):
        if m is not None and callable(getattr(m, "plan", None)):
            fn = m.plan
            break
    if fn is None:
        raise NotImplementedYet("forge.planner.search.plan is not available")
    return to_jsonable(call_flexible(fn, _aliases(svc, body), **body.options))


class TrainSimRequest(BaseModel):
    """Body of ``POST /planner/train-sim``."""

    config: TrainSimConfig = Field(default_factory=TrainSimConfig, description="Model + training scheme + batch settings.")
    chip: str | dict[str, Any] = Field("h100-sxm-80g", description="Chip preset / alias / name, or a ChipSpec dict.")
    n_gpu: int = Field(1, ge=1, le=4096, description="GPUs.")
    price_per_gpu_hour: float = Field(0.0, ge=0, description="USD per GPU-hour (0 = unknown).")
    link: Literal["auto", "nvlink", "pcie"] = "auto"
    calibration: float = Field(1.0, gt=0, description="Throughput calibration factor from a measured run (see forge.planner.trainsim.calibrate).")


@router.post("/planner/train-sim", summary="Training memory / time / cost estimate for a scheme (bf16, fp8, lora, qlora, ternary ...); first-order and uncalibrated")
def train_sim(body: TrainSimRequest) -> Any:
    try:
        return to_jsonable(trainsim.simulate(body.config, body.chip, body.n_gpu, body.price_per_gpu_hour, body.link, body.calibration))
    except ValueError as e:
        raise ConfigError(str(e)) from e


@router.post("/serve/simulate", summary="Capacity simulation (weights/KV/free memory, max concurrency, decode ceiling)")
def simulate(body: ServeSimulateRequest, svc: Services = Depends(get_services)) -> Any:
    mod = require_module("forge.serve.capacity", "the capacity planner")
    fn = _callable(mod, "capacity_plan")
    aliases = _aliases(svc, body)
    aliases[("weight_bits", "bits")] = body.weight_bits
    cd: Any = body.context_dist
    if isinstance(cd, dict):  # {"mean_ctx": 4096, "p99_ctx": 8192, "prefix_tokens": 0}
        try:
            cd = mod.ContextDist(**cd)
        except TypeError as e:
            raise ConfigError(f"invalid context_dist: {e}") from e
    elif isinstance(cd, list) and cd and hasattr(mod.ContextDist, "from_samples"):
        cd = mod.ContextDist.from_samples([float(x) for x in cd])
    aliases[("context_dist", "context_distribution", "contexts")] = cd
    return to_jsonable(call_flexible(fn, aliases, **body.options))


# --------------------------------------------------------------------------------------------
# claims ledger
# --------------------------------------------------------------------------------------------
def _claims_mod() -> Any:
    mod = require_module("forge.planner.claims", "the claims ledger")
    return mod


def _pick(mod: Any, *names: str) -> Any:
    for n in names:
        f = getattr(mod, n, None)
        if callable(f):
            return f
    raise NotImplementedYet(f"{mod.__name__} has none of {names}")


@router.get("/claims", summary="Claims ledger (seeded from the papers on first use)")
def list_claims(
    status: str | None = Query(None, description="untested | reproduced | contradicted | partial"),
    paper: str | None = None,
    prior: str | None = Query(None, description="trusted | untrusted"),
    rule: str | None = Query(None, description="Planner rule id that depends on the claim"),
    q: str | None = Query(None, description="Substring of statement / source"),
) -> list[Any]:
    mod = _claims_mod()
    if callable(getattr(mod, "ensure_seeded", None)):
        mod.ensure_seeded()  # idempotent: inserts the paper claims when the ledger is empty
    fn = _pick(mod, "list_claims", "all_claims", "get_claims")
    aliases = {("status",): status, ("paper",): paper, ("prior",): prior, ("rule",): rule, ("q",): q}
    rows = to_jsonable(call_flexible(fn, aliases))
    rows = rows if isinstance(rows, list) else list(rows or [])
    # defensive client-side filtering for ledgers that ignore some filters
    if status:
        rows = [r for r in rows if not isinstance(r, dict) or r.get("status") == status]
    if q:
        ql = q.lower()
        rows = [r for r in rows if not isinstance(r, dict) or ql in str(r.get("statement", "")).lower() or ql in str(r.get("source_paper", "")).lower()]
    return rows


@router.get("/claims/summary", summary="Ledger summary + claims needing attention (dashboard)")
def claims_summary() -> dict[str, Any]:
    mod = _claims_mod()
    if callable(getattr(mod, "ensure_seeded", None)):
        mod.ensure_seeded()
    out: dict[str, Any] = {}
    if callable(getattr(mod, "ledger_summary", None)):
        out["summary"] = to_jsonable(mod.ledger_summary())
    if callable(getattr(mod, "claims_needing_attention", None)):
        out["attention"] = to_jsonable(mod.claims_needing_attention())
    return out


@router.get("/claims/{claim_id}")
def get_claim(claim_id: str) -> Any:
    mod = _claims_mod()
    fn = _pick(mod, "get_claim", "claim")
    out = fn(claim_id)
    if out is None:
        raise NotFound(f"claim {claim_id!r} not found")
    return to_jsonable(out)


@router.put("/claims/{claim_id}", summary="Update one claim (same body as PUT /claims)")
def update_claim_by_id(claim_id: str, body: ClaimUpdate) -> Any:
    return update_claim(body.model_copy(update={"id": claim_id}))


@router.put("/claims", summary="Update a claim's status / notes / linked experiments")
def update_claim(body: ClaimUpdate) -> Any:
    if not body.id:
        raise ConfigError("`id` is required")
    mod = _claims_mod()
    fields = body.model_dump(exclude_unset=True, exclude={"id"})
    fn = None
    for n in ("update_claim", "set_claim", "upsert_claim"):
        if callable(getattr(mod, n, None)):
            fn = getattr(mod, n)
            break
    if fn is None:
        raise NotImplementedYet("the claims ledger has no update function")
    sig = inspect.signature(fn)
    params = list(sig.parameters)
    try:
        if "claim_id" in sig.parameters or (params and params[0] in ("id", "claim_id", "cid")):
            out = fn(body.id, **{k: v for k, v in fields.items() if k in sig.parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())})
        else:
            cur = mod.get_claim(body.id)
            if cur is None:
                raise NotFound(f"claim {body.id!r} not found")
            data = to_jsonable(cur)
            data.update(fields)
            out = fn(type(cur).model_validate(data) if hasattr(type(cur), "model_validate") else data)
    except KeyError:
        raise NotFound(f"claim {body.id!r} not found") from None
    return to_jsonable(out if out is not None else mod.get_claim(body.id))
