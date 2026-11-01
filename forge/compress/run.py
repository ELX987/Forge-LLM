"""``run_compress(cfg, ctx)``: the ``compress`` job entry (Plan CP-12).

Loads the input model, executes a recipe (Appendix C.1 YAML, inline or from a file) through :class:`forge.core.recipe.RecipeEngine` with the stages of
:mod:`forge.compress.stages`, saves the resulting model directory, and registers it in the artifact store with lineage, calibration hash, evaluation
evidence and licence metadata.  Without a store everything still runs (no stage cache, no registration) -- this is the CPU test / scripting mode.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.compress import state
from forge.compress.calibrate import register_calibration_set
from forge.compress.config import CompressJobConfig
from forge.compress.modelutil import count_params, model_fingerprint, model_nbytes
from forge.core.context import NullContext, RunContext
from forge.core.errors import ConfigError, GateFailed, NotFound
from forge.core.hashing import is_digest
from forge.core.recipe import RecipeEngine, load_recipe
from forge.core.schemas import Recipe

log = logging.getLogger("forge.compress.run")

__all__ = ["CompressJobConfig", "run_compress"]


def _import_stages() -> None:
    import forge.compress.stages  # noqa: F401  (registers the recipe stages)


def _resolve_device(dev: str) -> str:
    if dev != "auto":
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_input_model(cfg: CompressJobConfig, ctx: RunContext, store: Any, device: str) -> tuple[nn.Module, Any, dict[str, Any], str | None]:
    """Returns ``(model, tokenizer, meta, store digest of the input or None)``."""
    src = cfg.model_source
    ref = cfg.model
    meta: dict[str, Any] = {"licence": cfg.licence, "distill_allowed": cfg.distill_allowed, "params_b": cfg.params_b, "source": ref}
    if src == "auto":
        if ref.startswith("tiny"):
            src = "tiny"
        elif is_digest(ref):
            src = "store"
        elif Path(ref).exists():
            src = "local"
        else:
            src = "hf"
    dt = getattr(torch, cfg.dtype)
    tok = None
    if src == "tiny":
        from forge.models.tiny import build_tiny_model

        seed = int(ref.split(":")[1]) if ":" in ref else cfg.seed
        model = build_tiny_model(seed, dtype=dt, **cfg.tiny_overrides).to(device)
        meta["locally_owned"] = True
        meta["source"] = f"tiny:{seed}"
        return model, tok, meta, None
    if src == "store":
        if store is None or not store.has(ref):
            raise NotFound(f"artifact {ref} not found in the artifact store")
        tmp = Path(tempfile.mkdtemp(prefix="forge_compress_in_"))
        try:
            store.materialize_artifact(ref, tmp)
            model, tok = state.load_model_dir(tmp, dtype=cfg.dtype, device=device)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        try:
            m = store.get_manifest(ref)
            meta["licence"] = meta["licence"] or getattr(m, "licence", None)
        except Exception:
            pass
        return model, tok, meta, ref
    from forge.models.loader import load_model

    model, tok = load_model(ref, dtype=cfg.dtype, device=device, revision=cfg.revision, backend="forge")
    return model, tok, meta, None


def _register_input(cfg: CompressJobConfig, ctx: RunContext, store: Any, model: nn.Module, tokenizer: Any, meta: dict[str, Any]) -> str | None:
    """Ingest the input model so the whole lineage has a root in the store."""
    from forge.compress.serialize import save_quantized

    d = Path(ctx.run_dir) / "input_model"
    try:
        save_quantized(model, d, tokenizer=tokenizer, extra={"role": "recipe_input", "source": meta.get("source")})
        kw: dict[str, Any] = {"licence": meta["licence"]} if meta.get("licence") else {}
        h = store.put_tensor_dir(d, kind="model", extra={"role": "recipe_input", "source": meta.get("source"), "name": cfg.name}, **kw)
        ctx.log(f"registered input model {h}")
        return h
    except Exception as e:
        ctx.log(f"could not register the input model ({e}); lineage will start at the first stage", "warning")
        return None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _stage_summary(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for rec in result.get("stages", []):
        o = rec.get("outputs") or {}
        keep = {k: v for k, v in o.items() if k != "model" and (k.endswith("_report") or k in ("gate", "gate_verdict", "layout_id", "kv", "compression", "bytes_after", "calib", "delta"))}
        out[f"{rec['index']:02d}_{rec['name']}"] = {"cached": rec.get("cached"), "seconds": round(rec.get("seconds", 0.0), 3), **keep}
    return out


def run_compress(
    cfg: CompressJobConfig | dict[str, Any],
    ctx: RunContext | None = None,
    *,
    model: nn.Module | None = None,
    tokenizer: Any = None,
    store: Any = None,
) -> dict[str, Any]:
    """Run a compression recipe. Returns a summary dict (also written to ``ctx`` outputs).

    ``model`` / ``tokenizer``: pass a live model instead of ``cfg.model`` (scripts, tests); ``cfg.model`` is then only a label.
    ``store``: artifact store (default: ``ctx.store``, else ``forge.store.get_store()`` when ``cfg.use_store``).
    """
    cfg = CompressJobConfig.model_validate(cfg) if isinstance(cfg, dict) else cfg
    ctx = ctx or NullContext()
    if cfg.save_dir:  # fail fast, before hours of compression, instead of at the very end
        sd = Path(cfg.save_dir)
        if sd.exists() and any(sd.iterdir()) and not (sd / "forge_compress.json").exists():
            raise ConfigError(f"save_dir {sd} exists, is not empty and is not a previous forge compress artifact; refusing to overwrite it "
                              "(choose an empty or new directory)")
    _import_stages()
    t0 = time.perf_counter()
    from forge.core.env import seed_everything

    seed_everything(cfg.seed)
    if store is None:
        store = getattr(ctx, "store", None)
    if store is None and cfg.use_store:
        try:
            from forge.store import get_store

            store = get_store()
        except Exception as e:
            ctx.log(f"artifact store unavailable ({e}); running without registration or stage cache", "warning")
    if store is not None and getattr(ctx, "store", None) is None:
        try:
            ctx.store = store  # type: ignore[misc]
        except Exception:
            pass

    for name, texts in cfg.calibration_texts.items():
        register_calibration_set(name, texts)
    for name, path in cfg.calibration_files.items():
        register_calibration_set(name, path=path)

    recipe: Recipe = load_recipe(cfg.recipe if cfg.recipe is not None else cfg.recipe_path)  # type: ignore[arg-type]
    device = _resolve_device(cfg.device)
    meta: dict[str, Any]
    in_digest: str | None = None
    if model is None:
        ctx.log(f"loading {cfg.model} ({cfg.dtype}) on {device}")
        model, tokenizer_loaded, meta, in_digest = _load_input_model(cfg, ctx, store, device)
        tokenizer = tokenizer if tokenizer is not None else tokenizer_loaded
    else:
        # a live model handed in by the caller is treated as locally owned unless licence information says otherwise
        meta = {"licence": cfg.licence, "distill_allowed": cfg.distill_allowed, "params_b": cfg.params_b, "source": cfg.model,
                "locally_owned": cfg.licence is None and cfg.distill_allowed is None}
    meta["params_b"] = meta.get("params_b") or round(count_params(model) / 1e9, 6)
    bytes_in = model_nbytes(model)
    base_digest = in_digest
    if base_digest is None and store is not None and cfg.use_store and cfg.register_input:
        base_digest = _register_input(cfg, ctx, store, model, tokenizer, meta)
    base_digest = base_digest or model_fingerprint(model)

    run_id = str(getattr(ctx, "run_id", "run"))
    state.begin_run(run_id, persist_intermediate=cfg.persist_intermediate, copy_below_gb=cfg.copy_below_gb)
    try:
        state.put_live(base_digest, model, tokenizer, **meta, base=True)
        inputs = dict(recipe.inputs)
        inputs.update(cfg.inputs)
        if "model" in inputs and inputs["model"] != base_digest:
            ctx.log(f"recipe input 'model' ({inputs['model']!r}) replaced by the job's model {base_digest[:23]}")
        inputs.update({"model": base_digest, "base_model": base_digest})
        if cfg.chip:
            inputs["chip"] = cfg.chip
        policy = dict(recipe.policy)
        policy.setdefault("licence_check", True)
        rd = recipe.model_dump()
        rd.update({"inputs": inputs, "policy": policy})
        if cfg.stage_overrides:
            for st in rd["stages"]:
                st["params"] = {**st["params"], **cfg.stage_overrides.get(st["name"], {})}
        if not rd.get("seeds"):
            rd["seeds"] = [cfg.seed]
        recipe = Recipe.model_validate(rd)

        engine = RecipeEngine(store, cache=bool(cfg.use_cache and store is not None))
        problems = engine.validate(recipe)
        if problems:
            raise (NotFound if any("not registered" in x for x in problems) else ConfigError)("; ".join(problems))
        result = engine.run(recipe, ctx)
        if result.get("cancelled"):
            ctx.log("recipe cancelled before completion", "warning")
        final_digest = result["outputs"].get("model", base_digest)
        lm = state.resolve_model(final_digest, ctx, dtype=cfg.dtype, device=device)
        summary = _finish(cfg, ctx, store, recipe, result, lm, base_digest, final_digest, bytes_in, meta, tokenizer, t0)
    finally:
        state.end_run(run_id)
    if summary.get("gate_failed") and cfg.fail_on_gate:
        raise GateFailed(f"recipe '{recipe.name}': gate verdict 'fail' ({summary.get('gate_reasons')}); the artifact {summary.get('artifact')} was registered with the failing evidence",
                         verdict=summary.get("gate"))
    return summary


def _finish(cfg, ctx, store, recipe, result, lm, base_digest, final_digest, bytes_in, meta, tokenizer, t0) -> dict[str, Any]:
    from forge.compress.modelutil import iter_packed_linears
    from forge.compress.serialize import save_quantized

    stages = _stage_summary(result)
    gates = [rec["outputs"]["gate"] for rec in result["stages"] if isinstance(rec.get("outputs"), dict) and rec["outputs"].get("gate")]
    gate = gates[-1] if gates else None
    gate_failed = any((g or {}).get("verdict") == "fail" for g in gates)
    eval_hashes = [rec["outputs"].get("eval_report_hash") or (rec["outputs"].get("gate_detail") or {}).get("eval_report_hash")
                   for rec in result["stages"] if isinstance(rec.get("outputs"), dict)]
    eval_hashes = [h for h in eval_hashes if h]
    calib = result["outputs"].get("calib")
    kv = lm.meta.get("kv")
    layout = result["outputs"].get("layout_id")
    name = cfg.output_name or f"{cfg.name}-{recipe.name}"
    out_dir = Path(cfg.save_dir) if cfg.save_dir else Path(ctx.run_dir) / "artifact"
    report = {"recipe": recipe.name, "recipe_hash": result["recipe_hash"], "stages": stages, "input": cfg.model, "base_digest": base_digest}
    extra = {"name": name, "recipe": recipe.name, "kv": kv, "layout_id": layout, "stages": [s.name for s in recipe.stages], "gate": gate,
             "licence": lm.meta.get("licence") or cfg.licence, "distill_allowed": lm.meta.get("distill_allowed"), "params_b": lm.meta.get("params_b"),
             "input": cfg.model, "base_digest": base_digest, "calibration": calib, "compress_summary": {k: v for k, v in stages.items()}}
    if out_dir.exists():
        # `save_dir` comes from the job config (UI/API): never recursively delete a directory that is not a previous compress artifact
        if any(out_dir.iterdir()) and not (out_dir / "forge_compress.json").exists():
            raise ConfigError(f"save_dir {out_dir} exists, is not empty and is not a previous forge compress artifact; refusing to overwrite it "
                              "(choose an empty or new directory)")
        shutil.rmtree(out_dir, ignore_errors=True)
    save_quantized(lm.model, out_dir, report=report, tokenizer=lm.tokenizer or tokenizer, extra=_clean(extra))
    ctx.set_output("model_dir", str(out_dir))
    artifact = None
    if store is not None and cfg.use_store:
        parents = [d for d in (base_digest, lm.meta.get("artifact")) if d and is_digest(d) and store.has(d)]
        parents = list(dict.fromkeys(parents))
        kw: dict[str, Any] = {"licence": extra["licence"]} if extra["licence"] else {}
        if calib and isinstance(calib, dict):
            kw["calib_hash"] = calib.get("set_hash")
        if eval_hashes:
            kw["eval_report_hashes"] = eval_hashes
        if cfg.chip:
            kw["hardware_families"] = [cfg.chip]
        kind = "quantized" if any(True for _ in iter_packed_linears(lm.model)) else "model"
        try:
            artifact = store.put_tensor_dir(out_dir, kind=kind, recipe_hash=result["recipe_hash"], parents=parents, extra=_clean(extra), **kw)
            for p in parents:
                try:
                    store.add_edge(artifact, p, result["recipe_hash"])
                except Exception as e:
                    log.debug("lineage edge failed: %s", e)
            ctx.log(f"registered artifact {artifact}")
        except Exception as e:
            ctx.log(f"could not register the artifact in the store: {e}", "warning")
    model_id = _register_model_row(cfg, ctx, out_dir, artifact, name, bytes_in)
    after = model_nbytes(lm.model)
    summary = {
        "artifact": artifact, "model_id": model_id, "model_dir": str(out_dir), "name": name, "recipe": recipe.name, "recipe_hash": result["recipe_hash"],
        "bytes_input": bytes_in, "bytes_output": after, "compression": bytes_in / max(after, 1), "params_output": count_params(lm.model),
        "gate": gate, "gate_failed": gate_failed, "gate_reasons": (gate or {}).get("reasons"), "kv": kv, "layout_id": layout, "stages": stages,
        "cache_hits": result.get("cache_hits", 0), "cancelled": bool(result.get("cancelled")), "seconds": round(time.perf_counter() - t0, 2),
    }
    for k in ("artifact", "model_id", "recipe_hash", "bytes_input", "bytes_output", "compression", "gate", "kv", "layout_id", "cache_hits"):
        try:
            ctx.set_output(k, summary[k])
        except Exception:
            pass
    ctx.set_output("summary", _clean(summary))
    ctx.progress(1.0, "compress done")
    return summary


def _register_model_row(cfg: CompressJobConfig, ctx: RunContext, out_dir: Path, artifact: str | None, name: str, bytes_in: int) -> str | None:
    """Job runs only (not scripts / tests with a NullContext): make the output a selectable model (``models`` table, source 'artifact')."""
    if not getattr(cfg, "register_model", True):
        return None
    try:
        from forge.jobs.context import WorkerContext
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(ctx, WorkerContext):
        return None
    try:
        from forge.jobs.modelinfo import register_artifact_dir

        row = register_artifact_dir(out_dir, artifact_hash=artifact, name=name, job_id=str(ctx.run_id), bytes_input=bytes_in)
        ctx.log(f"registered model {row['id']} ({row['name']}, {(row.get('info') or {}).get('quantization', {}).get('label') or 'dense'}) - selectable on the Inference page")
        return row["id"]
    except Exception as e:  # noqa: BLE001 - registration is a convenience; `POST /models/import-artifacts` backfills it
        ctx.log(f"could not register the output as a model ({type(e).__name__}: {e}); use Models -> Import artifacts", "warning")
        return None


def _clean(obj: Any) -> Any:
    import json

    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)
