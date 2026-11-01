from __future__ import annotations

import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge.core.errors import ConfigError, NotFound

Family = Literal["pretrain", "low_precision_training", "peft", "distillation", "ptq", "qat", "compression", "ternary"]
Status = Literal["supported", "experimental", "planned"]


class MethodSpec(BaseModel):
    """One method.  ``status``: supported = implemented and CPU-tested; experimental = implemented behind a flag/emulated, GPU-unvalidated;
    planned = reserved slot, not implemented (shown so the roadmap is honest)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    family: Family
    summary: str
    status: Status
    job_type: str | None = Field(None, description="Job type that runs it (forge.jobs.registry), if any.")
    template: str | None = Field(None, description="Template file under configs/ (relative), if any.")
    config_hint: dict = Field(default_factory=dict, description="Dotted-path -> value overrides that select this method inside its job config.")
    produces: str = Field("model", description="Artifact kind produced: model | quantized | adapter | delta | recipe.")
    needs: list[str] = Field(default_factory=list, description="Requirements, e.g. 'cuda', 'sm>=89', 'calibration data', 'teacher model'.")
    diagnostics_prefix: str | None = Field(None, description="Metric namespace emitted while running (e.g. fp4, ternary).")
    planner_scheme: str | None = Field(None, description="forge.planner Scheme name that models its deployment, if any.")
    gpu_validated: bool = Field(False, description="True only after a recorded hardware run (docs/KERNELS.md validation tables).")
    docs: str | None = None


_LOCK = threading.RLock()
_METHODS: dict[str, MethodSpec] = {}
_LOADED = False


def register_method(spec: MethodSpec, replace: bool = False) -> MethodSpec:
    with _LOCK:
        if spec.id in _METHODS and not replace:
            raise ConfigError(f"method {spec.id!r} already registered")
        _METHODS[spec.id] = spec
    return spec


def _load() -> None:
    global _LOADED
    with _LOCK:
        if not _LOADED:
            _LOADED = True
            from . import builtin_methods  # noqa: F401  (registers on import)


def list_methods(family: str | None = None, status: str | None = None) -> list[MethodSpec]:
    _load()
    with _LOCK:
        out = [m for m in _METHODS.values() if (family is None or m.family == family) and (status is None or m.status == status)]
    return sorted(out, key=lambda m: (m.family, m.id))


def get_method(method_id: str) -> MethodSpec:
    _load()
    with _LOCK:
        try:
            return _METHODS[method_id]
        except KeyError:
            raise NotFound(f"unknown method {method_id!r}", details={"known": sorted(_METHODS)}) from None
