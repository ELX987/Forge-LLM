"""Job-type registry: maps a job type to (ConfigModel, entrypoint), both imported lazily.

``config`` / ``entry`` are ``"module:Attr"`` strings (``config`` may be a list of candidates, first importable
wins, because sibling modules may place the config class in ``forge.<mod>.config`` or next to the entrypoint).
A missing module/attribute raises :class:`UnavailableModule` (an HTTP 501 in the API), never crashes import.

Entry points are called as ``entry(cfg: ConfigModel, ctx: RunContext)``.
"""
from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge.core.errors import ForgeError, NotFound

log = logging.getLogger("forge.jobs.registry")


class UnavailableModule(ForgeError):
    """The module implementing a job type (or its config model) is not importable on this host."""

    code = "module_unavailable"
    http_status = 501


@dataclass
class JobTypeSpec:
    name: str
    title: str
    description: str
    config: str | list[str]  # "module:Class" (or candidates)
    entry: str  # "module:function"
    needs_gpu: bool = False  # scheduler assigns a free GPU when spec.gpus is None (only if GPUs exist)
    light: bool = False  # does not count against max_concurrent_jobs (downloads, gc, ...)
    group: str = "general"
    template_dirs: tuple[str, ...] = ()
    template_wrap: str | None = None  # "recipe_yaml": template files are whole recipes -> params {"recipe_yaml": text}
    experimental: bool = False
    _cfg_cache: Any = field(default=None, repr=False, compare=False)

    # -- lazy resolution ------------------------------------------------------------------
    @staticmethod
    def _import(target: str) -> Any:
        mod_name, _, attr = target.partition(":")
        mod = importlib.import_module(mod_name)
        obj: Any = mod
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj

    def config_model(self) -> type:
        if self._cfg_cache is not None:
            return self._cfg_cache
        errors = []
        for target in [self.config] if isinstance(self.config, str) else list(self.config):
            try:
                cls = self._import(target)
                self._cfg_cache = cls
                return cls
            except Exception as e:  # ImportError, AttributeError, or a bug inside the module
                errors.append(f"{target}: {type(e).__name__}: {e}")
        raise UnavailableModule(
            f"config model for job type '{self.name}' is not available ({'; '.join(errors)})",
            details={"job_type": self.name, "errors": errors},
        )

    def entrypoint(self) -> Any:
        try:
            return self._import(self.entry)
        except Exception as e:
            raise UnavailableModule(
                f"entrypoint {self.entry} for job type '{self.name}' is not available ({type(e).__name__}: {e})",
                details={"job_type": self.name, "entry": self.entry},
            ) from e

    def availability(self) -> tuple[bool, str | None]:
        try:
            self.config_model()
            self.entrypoint()
            return True, None
        except UnavailableModule as e:
            return False, e.message

    def templates_dirs(self) -> list[str]:
        return list(self.template_dirs or (self.name,))


_LOCK = threading.RLock()
_TYPES: dict[str, JobTypeSpec] = {}


def register_job_type(spec: JobTypeSpec) -> JobTypeSpec:
    with _LOCK:
        _TYPES[spec.name] = spec
    try:  # mirror into the generic registry so `forge.core.registry.names("job_type")` works
        from forge.core import registry as core_registry

        core_registry.register_obj("job_type", spec.name, spec, meta={"title": spec.title})
    except Exception:  # pragma: no cover
        pass
    return spec


def get_job_type(name: str) -> JobTypeSpec:
    _ensure_builtin()
    with _LOCK:
        spec = _TYPES.get(name)
    if spec is None:
        raise NotFound(
            f"unknown job type {name!r} (available: {', '.join(job_type_names())})", details={"job_type": name}
        )
    return spec


def job_type_names() -> list[str]:
    _ensure_builtin()
    with _LOCK:
        return sorted(_TYPES)


def list_job_types() -> list[JobTypeSpec]:
    _ensure_builtin()
    with _LOCK:
        return [_TYPES[k] for k in sorted(_TYPES)]


_BUILTIN_DONE = False


def _ensure_builtin() -> None:
    global _BUILTIN_DONE
    if _BUILTIN_DONE:
        return
    with _LOCK:
        if _BUILTIN_DONE:
            return
        _BUILTIN_DONE = True
        _register_all()
        _load_plugins()


def _register_all() -> None:
    J = JobTypeSpec
    # --- implemented by other modules (Plan section 6 mapping) ---------------------------------
    register_job_type(
        J(
            "train", "Pre-train / continue training",
            "Train a decoder-only model from scratch or continue pre-training (precision ladder R0 bf16 / R1 FP8 / R2 NVFP4).",
            ["forge.train.config:TrainConfig", "forge.train.run:TrainConfig"],
            "forge.train.run:run_training", needs_gpu=True, group="training",
        )
    )
    register_job_type(
        J(
            "finetune", "Fine-tune (full / LoRA)",
            "Supervised fine-tuning with full weights or LoRA adapters.",
            ["forge.train.config:FinetuneConfig", "forge.train.run:FinetuneConfig"],
            "forge.train.run:run_finetune", needs_gpu=True, group="training",
        )
    )
    register_job_type(
        J(
            "distill", "Distillation",
            "Offline-logit / sequence-level / on-policy distillation from a teacher model (licence gated).",
            ["forge.train.config:DistillConfig", "forge.train.run:DistillConfig"],
            "forge.train.run:run_distill", needs_gpu=True, group="training",
        )
    )
    register_job_type(
        J(
            "dataset_ingest", "Dataset ingest",
            "Tokenise, pack and shard a text/jsonl/parquet/HF dataset into a content-addressed token dataset.",
            ["forge.train.data:IngestConfig", "forge.train.config:IngestConfig"],
            "forge.train.data:ingest_dataset", group="data", template_dirs=("dataset_ingest", "dataset", "datasets"),
        )
    )
    register_job_type(
        J(
            "compress", "Compress / quantise",
            "Run a compression recipe (calibrate, PTQ, KV quant, prune, low-rank, delta, lossless ...).",
            ["forge.compress.run:CompressJobConfig", "forge.compress.config:CompressJobConfig"],
            "forge.compress.run:run_compress", needs_gpu=True, group="compression",
            template_dirs=("compress",),
        )
    )
    register_job_type(
        J(
            "eval", "Evaluate / gate",
            "Compare a candidate against its parent on suites (distribution, MCQ, long-context, safety ...) and apply gates.",
            ["forge.eval.run:EvalConfig", "forge.eval.config:EvalConfig"],
            "forge.eval.run:run_eval", needs_gpu=True, group="evaluation", template_dirs=("eval",),
        )
    )
    register_job_type(
        J(
            "serve", "Serve (inference engine)",
            "Start the continuous-batching OpenAI-compatible inference engine for a model or artifact.",
            ["forge.serve.config:ServeConfig", "forge.serve.run:ServeConfig"],
            "forge.serve.run:run_server", needs_gpu=True, group="inference",
        )
    )
    register_job_type(
        J(
            "bench", "Kernel benchmark",
            "Benchmark kernels against roofline and the reference oracle; results land in the kernel lab.",
            ["forge.kernels.bench:BenchConfig", "forge.kernels.config:BenchConfig"],
            "forge.kernels.bench:run_bench", needs_gpu=True, group="hardware",
        )
    )
    # --- implemented here (forge.jobs.builtin) -------------------------------------------------
    b = "forge.jobs.builtin"
    register_job_type(
        J(
            "probe", "Hardware probe",
            "Probe the GPU (capabilities + micro-benchmarks) and store a measured chip descriptor.",
            f"{b}:ProbeJobConfig", f"{b}:run_probe_job", needs_gpu=True, group="hardware",
        )
    )
    register_job_type(
        J(
            "recipe", "Run recipe",
            "Execute a saved or inline recipe with the stage-cached RecipeEngine.",
            f"{b}:RecipeJobConfig", f"{b}:run_recipe_job", group="compression", template_dirs=("recipes",),
            template_wrap="recipe_yaml",
        )
    )
    register_job_type(
        J(
            "model_download", "Download model (HF)",
            "Download a Hugging Face repo snapshot into the models directory and register it.",
            f"{b}:ModelDownloadConfig", f"{b}:run_model_download", light=True, group="data",
        )
    )
    register_job_type(
        J(
            "gc", "Storage garbage collection",
            "Mark-and-sweep the content-addressed store from pins (dry-run by default).",
            f"{b}:GcJobConfig", f"{b}:run_gc", light=True, group="storage",
        )
    )
    register_job_type(
        J(
            "checkpoint_export", "Export checkpoint",
            "Export a training checkpoint to a HF-compatible safetensors model directory and register it.",
            f"{b}:CheckpointExportConfig", f"{b}:run_checkpoint_export", light=True, group="storage",
        )
    )
    register_job_type(
        J(
            "selftest", "Self-test (synthetic)",
            "CPU-only synthetic job that emits fake metrics/progress/logs. For UI demos and API tests; "
            "the numbers are generated, NOT measurements.",
            f"{b}:SelftestConfig", f"{b}:run_selftest", group="diagnostics",
        )
    )


def _load_plugins() -> None:
    """Import modules named in ``FORGE_JOB_PLUGINS`` (comma separated). They register extra job types by calling
    :func:`register_job_type` at import time; worker processes load the same list, so plugin job types run too."""
    import os

    for mod in [m.strip() for m in os.environ.get("FORGE_JOB_PLUGINS", "").split(",") if m.strip()]:
        try:
            importlib.import_module(mod)
        except Exception as e:  # a broken plugin must not take the server down
            log.warning("job plugin %s could not be imported: %s: %s", mod, type(e).__name__, e)


def templates_root() -> Path | None:
    """Directory holding ``configs/<type>/*.yaml`` (env ``FORGE_CONFIGS_DIR`` or the source checkout)."""
    import os

    env = os.environ.get("FORGE_CONFIGS_DIR", "").strip()
    if env and Path(env).is_dir():
        return Path(env)
    from forge.core.paths import repo_root

    root = repo_root()
    if root and (root / "configs").is_dir():
        return root / "configs"
    return None
