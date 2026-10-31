"""forge.core: schemas, hashing, paths, settings, errors, registry, db, run context, recipe engine, env capture.

Light re-exports only; ``forge.core.db``, ``forge.core.recipe``, ``forge.core.env`` and ``forge.core.jsonschema``
are imported explicitly by users (they pull in sqlite3 / yaml / subprocess machinery).
"""
from __future__ import annotations

from forge.core.context import NullContext, RunContext
from forge.core.errors import (
    Cancelled,
    ConfigError,
    ForgeError,
    GateFailed,
    HardwareUnsupported,
    IntegrityError,
    KernelCorrectnessError,
    LicenceBlocked,
    NotFound,
    StageFailed,
    StoreError,
)
from forge.core.hashing import Hasher, hash_bytes, hash_file, hash_json
from forge.core.paths import (
    cache_dir,
    data_dir,
    datasets_dir,
    forge_home,
    models_dir,
    runs_dir,
    store_dir,
)
from forge.core.registry import get as registry_get
from forge.core.registry import names as registry_names
from forge.core.registry import register
from forge.core.schemas import (
    ArtifactManifest,
    Claim,
    DatasetManifest,
    Deployment,
    ChipDescriptor,
    EvalReport,
    GateVerdict,
    JobRecord,
    JobSpec,
    JobStatus,
    MetricValue,
    ModelRef,
    Provenance,
    Recipe,
    StageSpec,
    TensorEntry,
    Workload,
)
from forge.core.settings import get_settings

__all__ = [
    "ArtifactManifest", "Cancelled", "ChipDescriptor", "Claim", "ConfigError", "DatasetManifest", "Deployment",
    "EvalReport", "ForgeError", "GateFailed", "GateVerdict", "HardwareUnsupported", "Hasher", "IntegrityError",
    "JobRecord", "JobSpec", "JobStatus", "KernelCorrectnessError", "LicenceBlocked", "MetricValue", "ModelRef",
    "NotFound", "NullContext", "Provenance", "Recipe", "RunContext", "StageFailed", "StageSpec", "StoreError",
    "TensorEntry", "Workload", "cache_dir", "data_dir", "datasets_dir", "forge_home", "get_settings",
    "hash_bytes", "hash_file", "hash_json", "models_dir", "register", "registry_get", "registry_names",
    "runs_dir", "store_dir",
]
