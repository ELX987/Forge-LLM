"""Job types implemented by the jobs layer itself: probe wrapper, recipe, model_download, gc,
checkpoint_export and the synthetic ``selftest``.

Every config is a pydantic model with ``Field(description=...)`` because the UI renders forms from the schema.
"""
from __future__ import annotations

import fnmatch
import math
import os
import random
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.core.errors import Cancelled, ConfigError, ForgeError, NotFound
from forge.core.jsonschema import ui
from forge.jobs.registry import UnavailableModule
from forge.jobs.repo import to_jsonable


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


_BAD_NAME_CHARS = re.compile(r'[\\/:*?"<>|]+')


def safe_dirname(name: str, what: str = "name") -> str:
    """Turn a user-supplied display name into ONE directory component below the models dir.

    Path separators / drive colons become ``--`` (``org/name`` -> ``org--name``); ``.``/``..``/empty are rejected.
    Without this, ``local_name="../../x"`` (or an absolute path, which ``Path.__truediv__`` lets replace the base)
    would download into - and ``checkpoint_export`` would ``rmtree`` - an arbitrary directory."""
    n = _BAD_NAME_CHARS.sub("--", "".join(c for c in str(name) if c.isprintable())).strip().strip(".").strip()
    if not n or n in (".", "..") or len(n) > 200:
        raise ConfigError(f"invalid {what} {name!r}: must be a plain directory name")
    return n


def _inside(child: Path, root: Path) -> bool:
    try:
        child.resolve().relative_to(root.resolve())
        return child.resolve() != root.resolve()
    except ValueError:
        return False




# ============================================================================================
# selftest (synthetic)
# ============================================================================================
class SelftestConfig(_Cfg):
    """Synthetic CPU-only job that emits fake metrics, progress and logs. Used for UI demos and API tests.
    Every number it produces is generated, not measured."""

    steps: int = Field(default=30, ge=1, le=1_000_000, description="Number of synthetic steps to run.", json_schema_extra=ui(group="Run"))
    step_delay_s: float = Field(default=0.1, ge=0, le=60, description="Sleep per step (seconds).", json_schema_extra=ui(group="Run", unit="s"))
    log_every: int = Field(default=5, ge=1, description="Emit a log line every N steps.", json_schema_extra=ui(group="Run"))
    seed: int = Field(default=0, description="Seed of the synthetic noise.", json_schema_extra=ui(group="Run", advanced=True))
    noise: float = Field(default=0.02, ge=0, le=1, description="Relative noise added to the synthetic loss curve.", json_schema_extra=ui(group="Curve", advanced=True))
    fail_at_step: int | None = Field(default=None, ge=0, description="Raise an error at this step (tests failure handling).", json_schema_extra=ui(group="Fault injection", advanced=True))
    ignore_stop: bool = Field(default=False, description="Ignore cooperative cancellation (tests SIGKILL escalation).", json_schema_extra=ui(group="Fault injection", advanced=True))
    result_value: str = Field(default="ok", description="Value stored in the job outputs as `echo`.", json_schema_extra=ui(group="Run", advanced=True))


def run_selftest(cfg: SelftestConfig, ctx: Any) -> dict[str, Any]:
    ctx.log("selftest: SYNTHETIC job - the metrics below are generated, not measured")
    rng = random.Random(cfg.seed)
    ctx.set_output("echo", cfg.result_value)
    ctx.set_output("synthetic", True)
    done = 0
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        if ctx.should_stop() and not cfg.ignore_stop:
            ctx.log(f"selftest: stop requested at step {step}", "warning")
            return {"steps_done": done, "stopped": True}
        if cfg.fail_at_step is not None and step >= cfg.fail_at_step:
            raise RuntimeError(f"injected failure at step {step}")
        frac = step / cfg.steps
        loss = 4.0 * math.exp(-3.0 * frac) + 0.5
        loss *= 1.0 + cfg.noise * rng.gauss(0, 1)
        lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * frac))
        ctx.metric(
            step,
            loss=loss,
            lr=lr,
            grad_norm=1.0 / (1.0 + 5 * frac) * (1 + 0.1 * rng.random()),
            tokens_per_s=50_000 * (1 + 0.05 * rng.gauss(0, 1)),
            **{"fp4/zero_grad_tile_frac": 0.02 * (1 - frac) + 0.001 * rng.random()},
        )
        ctx.progress(frac, f"step {step}/{cfg.steps}")
        if step % cfg.log_every == 0 or step == cfg.steps:
            ctx.log(f"step {step}/{cfg.steps} loss={loss:.4f} lr={lr:.2e}")
        done = step
        if cfg.step_delay_s:
            time.sleep(cfg.step_delay_s)
    ctx.set_output("steps_done", done)
    ctx.set_output("seconds", round(time.time() - t0, 3))
    return {"steps_done": done, "stopped": False}


# ============================================================================================
# probe
# ============================================================================================
class ProbeJobConfig(_Cfg):
    """Probe the local GPU: capabilities plus micro-benchmarks, stored as a measured chip descriptor."""

    device: int = Field(default=0, ge=0, description="CUDA device index to probe (relative to CUDA_VISIBLE_DEVICES).", json_schema_extra=ui(group="Probe"))
    quick: bool = Field(default=False, description="Run the shortened micro-benchmark set (fewer repeats, smaller sizes).", json_schema_extra=ui(group="Probe"))
    save: bool = Field(default=True, description="Save the resulting descriptor (sqlite + yaml in FORGE_HOME).", json_schema_extra=ui(group="Probe"))


def run_probe_job(cfg: ProbeJobConfig, ctx: Any) -> dict[str, Any]:
    try:
        from forge.hw.probe import run_probe
    except Exception as e:
        raise UnavailableModule(f"forge.hw.probe is not available: {e}") from e
    desc = run_probe(ctx, device=cfg.device, quick=cfg.quick)
    out: dict[str, Any] = {}
    d = to_jsonable(desc)
    if isinstance(d, dict):
        out = {"descriptor_id": d.get("id"), "name": d.get("name"), "sm": d.get("sm"), "tier": d.get("tier")}
        ctx.set_output("descriptor", d)
    if cfg.save:
        try:
            from forge.hw.descriptor_store import save_descriptor

            res = save_descriptor(desc)
            ctx.set_output("saved", to_jsonable(res) if res is not None else True)
        except Exception as e:
            ctx.log(f"could not save descriptor: {e}", "warning")
    for k, v in out.items():
        ctx.set_output(k, v)
    return out


# ============================================================================================
# recipe
# ============================================================================================
class RecipeJobConfig(_Cfg):
    """Run a recipe with the stage-cached RecipeEngine. Provide a saved recipe id, inline YAML, or a dict."""

    recipe_id: str | None = Field(default=None, description="Id of a saved recipe (see the Recipes page).", json_schema_extra=ui(group="Recipe"))
    recipe_yaml: str | None = Field(default=None, description="Inline recipe in YAML (used when recipe_id is empty).", json_schema_extra=ui(group="Recipe", format="yaml"))
    recipe: dict[str, Any] | None = Field(default=None, description="Inline recipe as a JSON object (used when neither id nor YAML is given).", json_schema_extra=ui(group="Recipe", advanced=True))
    force: bool = Field(default=False, description="Ignore the stage cache and re-run every stage.", json_schema_extra=ui(group="Execution"))
    cache: bool = Field(default=True, description="Use the content-addressed stage cache (unchanged upstream stages are skipped).", json_schema_extra=ui(group="Execution"))
    input_overrides: dict[str, Any] = Field(default_factory=dict, description="Values merged over the recipe `inputs` before running.", json_schema_extra=ui(group="Recipe", advanced=True))

    @model_validator(mode="after")
    def _one_source(self) -> RecipeJobConfig:
        if not (self.recipe_id or self.recipe_yaml or self.recipe):
            raise ValueError("provide recipe_id, recipe_yaml or recipe")
        return self


def _load_recipe_for(cfg: RecipeJobConfig) -> Any:
    from forge.core.recipe import load_recipe

    if cfg.recipe_id:
        from forge.jobs.repo import RecipesRepo

        row = RecipesRepo().require(cfg.recipe_id, "recipe")
        return load_recipe(row["yaml"])
    if cfg.recipe_yaml:
        return load_recipe(cfg.recipe_yaml)
    return load_recipe(cfg.recipe)  # type: ignore[arg-type]


def import_stage_modules() -> list[str]:
    """Import modules that register recipe stages (best effort). Returns the modules that failed."""
    failed = []
    for m in ("forge.compress.stages", "forge.eval.run", "forge.train.run"):
        try:
            __import__(m)
        except Exception as e:  # missing module or heavy dep missing
            failed.append(f"{m}: {type(e).__name__}: {e}")
    return failed


def run_recipe_job(cfg: RecipeJobConfig, ctx: Any) -> dict[str, Any]:
    from forge.core.recipe import RecipeEngine

    recipe = _load_recipe_for(cfg)
    if cfg.input_overrides:
        recipe = recipe.model_copy(update={"inputs": {**recipe.inputs, **cfg.input_overrides}})
    for f in import_stage_modules():
        ctx.log(f"stage module not importable (stages it registers are unavailable): {f}", "warning")
    engine = RecipeEngine(ctx.store, cache=cfg.cache)
    res = engine.run(recipe, ctx, force=cfg.force)
    ctx.set_output("recipe_hash", res.get("recipe_hash"))
    ctx.set_output("cache_hits", res.get("cache_hits"))
    ctx.set_output("stages", [{k: v for k, v in s.items() if k != "outputs"} for s in res.get("stages", [])])
    ctx.set_output("outputs", to_jsonable(res.get("outputs")))
    if res.get("cancelled"):
        ctx.log("recipe cancelled before all stages ran", "warning")
    return {"recipe_hash": res.get("recipe_hash"), "cache_hits": res.get("cache_hits"), "cancelled": res.get("cancelled")}


# ============================================================================================
# model_download
# ============================================================================================
class ModelDownloadConfig(_Cfg):
    """Download a Hugging Face repository snapshot into the models directory and register it."""

    repo_id: str = Field(min_length=3, description="Hugging Face repo id, e.g. `meta-llama/Llama-3.1-8B`.", json_schema_extra=ui(group="Source"))
    revision: str | None = Field(default=None, description="Branch, tag or commit hash (default: main).", json_schema_extra=ui(group="Source"))
    allow_patterns: list[str] | None = Field(default=None, description="Only download files matching these glob patterns.", json_schema_extra=ui(group="Filter", advanced=True))
    ignore_patterns: list[str] = Field(default_factory=lambda: ["*.msgpack", "*.h5", "*.ot", "original/*", "*.onnx"], description="Skip files matching these glob patterns.", json_schema_extra=ui(group="Filter", advanced=True))
    prefer_safetensors: bool = Field(default=True, description="If the repo has .safetensors weights, skip .bin/.pt duplicates.", json_schema_extra=ui(group="Filter"))
    local_name: str | None = Field(default=None, description="Directory / display name (default: repo name).", json_schema_extra=ui(group="Target"))
    register_model: bool = Field(default=True, description="Register the model in the models table after download.", json_schema_extra=ui(group="Target"))
    use_token: bool = Field(default=True, description="Use the HF_TOKEN environment variable if set (needed for gated repos).", json_schema_extra=ui(group="Source"))
    max_workers: int = Field(default=8, ge=1, le=64, description="Parallel download workers.", json_schema_extra=ui(group="Transfer", advanced=True))
    licence: str | None = Field(default=None, description="Licence override recorded with the model (default: from the repo card).", json_schema_extra=ui(group="Target", advanced=True))
    model_id: str | None = Field(default=None, description="Existing models-table id to update instead of creating a new row.", json_schema_extra=ui(group="Target", advanced=True))


def _dir_bytes(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _matches(name: str, patterns: list[str] | None) -> bool:
    return bool(patterns) and any(fnmatch.fnmatch(name, pat) for pat in patterns)  # type: ignore[union-attr]


def run_model_download(cfg: ModelDownloadConfig, ctx: Any) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        raise UnavailableModule("huggingface_hub is not installed (`pip install huggingface_hub`)") from e
    from forge.core.paths import models_dir
    from forge.core.settings import get_settings

    token = get_settings().hf_token if cfg.use_token else None
    name = cfg.local_name or cfg.repo_id.split("/")[-1]
    target = models_dir() / safe_dirname(name, "local_name")
    if not _inside(target, models_dir()):
        raise ConfigError(f"refusing to download outside the models directory: {target}")
    target.mkdir(parents=True, exist_ok=True)
    ctx.log(f"downloading {cfg.repo_id}@{cfg.revision or 'main'} -> {target}")

    ignore = list(cfg.ignore_patterns)
    total_bytes = 0
    licence = cfg.licence
    try:
        info = HfApi().model_info(cfg.repo_id, revision=cfg.revision, files_metadata=True, token=token)
        files = [(s.rfilename, s.size or 0) for s in (info.siblings or [])]
        if cfg.prefer_safetensors and any(n.endswith(".safetensors") for n, _ in files):
            ignore += ["*.bin", "*.pt", "*.pth"]
        keep = [
            (n, s) for n, s in files
            if (not cfg.allow_patterns or _matches(n, cfg.allow_patterns)) and not _matches(n, ignore)
        ]
        total_bytes = sum(s for _, s in keep)
        ctx.log(f"{len(keep)} files, {total_bytes / 2**30:.2f} GiB to fetch")
        ctx.set_output("files", len(keep))
        ctx.set_output("total_bytes", total_bytes)
        if licence is None:
            card = getattr(info, "card_data", None) or getattr(info, "cardData", None)
            lic = getattr(card, "license", None) if card is not None else None
            if not lic:
                for tg in getattr(info, "tags", None) or []:
                    if str(tg).startswith("license:"):
                        lic = str(tg).split(":", 1)[1]
            licence = str(lic) if lic else None
    except Exception as e:
        ctx.log(f"could not query repo metadata ({type(e).__name__}: {e}); progress will be indeterminate", "warning")

    err: list[BaseException] = []
    done = threading.Event()
    result: list[str] = []

    def work() -> None:
        try:
            result.append(
                snapshot_download(
                    repo_id=cfg.repo_id,
                    revision=cfg.revision,
                    local_dir=str(target),
                    allow_patterns=cfg.allow_patterns,
                    ignore_patterns=ignore,
                    token=token,
                    max_workers=cfg.max_workers,
                )
            )
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    th = threading.Thread(target=work, name="hf-download", daemon=True)
    th.start()
    last = 0
    t_last, b_last = time.time(), 0
    while not done.wait(0.5):
        if ctx.should_stop():
            raise Cancelled("model download cancelled")
        cur = _dir_bytes(target)
        now_t = time.time()
        rate = (cur - b_last) / max(1e-6, now_t - t_last) if cur >= b_last else 0.0
        t_last, b_last = now_t, cur
        if total_bytes:
            ctx.progress(min(0.99, cur / total_bytes), f"{cur / 2**30:.2f}/{total_bytes / 2**30:.2f} GiB ({rate / 2**20:.0f} MiB/s)")
        else:
            ctx.progress(0.0, f"{cur / 2**30:.2f} GiB downloaded")
        if cur != last and int(cur / 2**28) != int(last / 2**28):
            ctx.log(f"downloaded {cur / 2**30:.2f} GiB")
        last = cur
    if err:
        raise err[0]
    ctx.progress(1.0, "download complete")
    ctx.set_output("path", str(target))
    out: dict[str, Any] = {"path": str(target), "bytes": _dir_bytes(target)}
    if cfg.register_model:
        from forge.jobs.modelinfo import register_local_model
        from forge.jobs.repo import ModelsRepo

        row = register_local_model(
            target, name=name, source="hf", hf_repo=cfg.repo_id, revision=cfg.revision, licence=licence
        )
        if cfg.model_id and cfg.model_id != row["id"]:
            repo = ModelsRepo()
            if repo.get(cfg.model_id):
                repo.update(cfg.model_id, **{k: row[k] for k in ("size_bytes", "params_b", "status", "info", "licence")})
                repo.delete(row["id"])
                row = repo.require(cfg.model_id)
        ctx.set_output("model_id", row["id"])
        out["model_id"] = row["id"]
        ctx.log(f"registered model {row['id']} ({row.get('params_b')} B params)")
    return out


# ============================================================================================
# gc
# ============================================================================================
class GcJobConfig(_Cfg):
    """Mark-and-sweep garbage collection of the content-addressed store."""

    dry_run: bool = Field(default=True, description="Only report what would be deleted; delete nothing.", json_schema_extra=ui(group="GC"))


def run_gc(cfg: GcJobConfig, ctx: Any) -> dict[str, Any]:
    try:
        from forge.jobs.repo import open_store

        store = ctx.store or open_store()
    except Exception as e:
        raise UnavailableModule(f"forge.store is not available: {e}") from e
    ctx.log(f"store gc (dry_run={cfg.dry_run})")
    rep = to_jsonable(store.gc(dry_run=cfg.dry_run))
    ctx.set_output("report", rep)
    return {"dry_run": cfg.dry_run, "report": rep}


# ============================================================================================
# checkpoint_export
# ============================================================================================
class CheckpointExportConfig(_Cfg):
    """Export a training checkpoint to a HF-compatible safetensors model directory and register it as a model."""

    run_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$", description="Training job / run id that owns the checkpoint.", json_schema_extra=ui(group="Source"))
    checkpoint: str = Field(default="latest", description="Checkpoint name or step (`latest`, `best`, `1000`, `step-1000`).", json_schema_extra=ui(group="Source"))
    name: str | None = Field(default=None, description="Model name (default: `<run_id>-<checkpoint>`).", json_schema_extra=ui(group="Target"))
    dtype: str | None = Field(default=None, description="Cast floating tensors to this dtype on export (bfloat16, float16, float32); default keeps stored dtype.", json_schema_extra=ui(group="Target", advanced=True))
    register_model: bool = Field(default=True, description="Register the exported directory in the models table.", json_schema_extra=ui(group="Target"))


def find_checkpoint_dirs(run_id: str) -> list[Path]:
    from forge.core.paths import runs_dir

    base = runs_dir() / run_id
    found = []
    for sub in ("checkpoints", "ckpt", "checkpoint"):
        d = base / sub
        if d.is_dir():
            found += sorted(x for x in d.iterdir())
    return found


def _store_checkpoint_export(cfg: CheckpointExportConfig, ctx: Any, dest: Path) -> bool:
    """Export from forge.store.checkpoints when the run has store-backed checkpoints. Returns False if it has none."""
    try:
        from forge.jobs.repo import open_store
        from forge.store.checkpoints import CheckpointReader
    except Exception:
        return False
    try:
        reader = CheckpointReader(open_store(), cfg.run_id)
        if not reader.rows():
            return False
        want = cfg.checkpoint.replace("step-", "").replace("step_", "")
        row = reader.resolve(int(want) if want.isdigit() else cfg.checkpoint)
    except NotFound:
        return False
    step = row["step"]
    ctx.log(f"exporting store checkpoint run={cfg.run_id} step={step}")
    hf_cfg = None
    try:
        meta = reader.load_meta(step)
        cand = (meta or {}).get("hf_config") or (meta or {}).get("model_config")
        if isinstance(cand, dict):
            try:
                from forge.models import ModelConfig

                hf_cfg = ModelConfig.model_validate(cand).to_hf_config()
            except Exception:
                hf_cfg = cand
    except Exception as e:
        ctx.log(f"no model config in checkpoint metadata ({e}); exporting weights only", "warning")
    reader.export_safetensors(step, dest, config=hf_cfg)
    ctx.set_output("step", step)
    if cfg.dtype:
        import torch
        from safetensors.torch import load_file, save_file

        dt = getattr(torch, cfg.dtype, None)
        if not isinstance(dt, torch.dtype):
            raise ConfigError(f"unknown dtype {cfg.dtype!r}")
        f = dest / "model.safetensors"
        tensors = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in load_file(str(f)).items()}
        save_file(tensors, str(f), metadata={"format": "pt"})
    return True


def run_checkpoint_export(cfg: CheckpointExportConfig, ctx: Any) -> dict[str, Any]:
    from forge.core.paths import models_dir

    name = safe_dirname(cfg.name or f"{cfg.run_id}-{cfg.checkpoint}", "name")
    dest = models_dir() / name
    if not _inside(dest, models_dir()):
        raise ConfigError(f"refusing to export outside the models directory: {dest}")
    if dest.exists():
        shutil.rmtree(dest)
    if not _store_checkpoint_export(cfg, ctx, dest):
        cands = find_checkpoint_dirs(cfg.run_id)
        chosen: Path | None = None
        want = cfg.checkpoint.replace("step-", "").replace("step_", "")
        for c in cands:
            if c.name == cfg.checkpoint or c.name.replace("step-", "").replace("step_", "") == want:
                chosen = c
        dirs = [c for c in cands if c.is_dir()]
        if chosen is None and cfg.checkpoint in ("latest", "best") and dirs:
            if cfg.checkpoint == "best":
                chosen = next((d for d in dirs if d.name == "best"), None)
            chosen = chosen or max(dirs, key=lambda d: d.stat().st_mtime)
        if chosen is None or not chosen.is_dir() or not any(chosen.glob("*.safetensors")):
            raise NotFound(
                f"no exportable checkpoint {cfg.checkpoint!r} for run {cfg.run_id!r} (looked in the artifact store and runs/{cfg.run_id}/checkpoints)"
            )
        ctx.log(f"copying {chosen} -> {dest}")
        shutil.copytree(chosen, dest)
    ctx.set_output("path", str(dest))
    out: dict[str, Any] = {"path": str(dest)}
    if cfg.register_model:
        from forge.jobs.modelinfo import register_local_model

        try:
            row = register_local_model(dest, name=name)
            ctx.set_output("model_id", row["id"])
            out["model_id"] = row["id"]
        except (ConfigError, NotFound, ForgeError) as e:
            ctx.log(f"exported but could not register: {e}", "warning")
    return out
