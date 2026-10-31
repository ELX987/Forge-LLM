"""Request bodies for the REST API (responses are plain JSON documents described in the OpenAPI schema)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class JobCreate(_Req):
    type: str = Field(description="Job type (see GET /schemas/job-types).")
    name: str = Field(default="", description="Human-friendly name.")
    params: dict[str, Any] = Field(default_factory=dict, description="Job configuration; validated against the job type's model.")
    gpus: list[int] | None = Field(default=None, description="GPU indices to reserve (null = scheduler decides, [] = CPU only).")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")
    config_id: str | None = Field(default=None, description="Start from a saved config; `params` are merged over it.")
    template: str | None = Field(default=None, description="Start from a template name (configs/<type>/<name>.yaml); `params` merged over it.")


class ConfigCreate(_Req):
    job_type: str = Field(description="Job type this config belongs to.")
    name: str = Field(min_length=1, description="Config name.")
    params: dict[str, Any] = Field(default_factory=dict, description="Parameters (validated against the job type's model).")
    description: str = Field(default="", description="Free text.")


class ConfigUpdate(_Req):
    name: str | None = None
    params: dict[str, Any] | None = None
    description: str | None = None


class ValidateRequest(_Req):
    job_type: str
    params: dict[str, Any] = Field(default_factory=dict)


class ModelCreate(_Req):
    name: str = Field(min_length=1)
    source: str = Field(default="hf", description="`hf` (repo id, downloaded on demand), `local` (directory) or `artifact` (store hash).")
    path_or_repo: str = Field(description="HF repo id, local directory or artifact hash.")
    revision: str | None = None
    licence: str | None = None
    distill_allowed: bool | None = None
    notes: str = ""
    tags: list[str] = Field(default_factory=list)


class ModelPatch(_Req):
    name: str | None = None
    licence: str | None = None
    distill_allowed: bool | None = None
    notes: str | None = None
    tags: list[str] | None = None
    revision: str | None = None


class RegisterLocal(_Req):
    path: str = Field(description="Local model directory (config.json + weights).")
    name: str | None = None
    licence: str | None = None
    distill_allowed: bool | None = None
    notes: str = ""


class ModelDownloadRequest(_Req):
    revision: str | None = None
    allow_patterns: list[str] | None = None
    prefer_safetensors: bool = True
    gpus: list[int] | None = None


class DatasetCreate(_Req):
    name: str = Field(min_length=1)
    path: str | None = Field(default=None, description="Register an existing local file/directory (no copy). Omit to create an empty dataset for uploads.")
    format: str = Field(default="", description="jsonl | text | parquet | csv | tokens | ... (auto-detected from the extension when empty).")
    licence: str | None = None


class IngestRequest(_Req):
    params: dict[str, Any] = Field(default_factory=dict, description="dataset_ingest parameters; the dataset path is filled in automatically.")
    name: str = ""
    gpus: list[int] | None = None


class RecipeCreate(_Req):
    name: str | None = None
    description: str = ""
    yaml: str | None = Field(default=None, description="Recipe as YAML text.")
    recipe: dict[str, Any] | None = Field(default=None, description="Recipe as a JSON object (alternative to yaml).")


class RecipeValidate(_Req):
    yaml: str | None = None
    json_: dict[str, Any] | None = Field(default=None, alias="json")
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class RecipeRun(_Req):
    force: bool = False
    cache: bool = True
    input_overrides: dict[str, Any] = Field(default_factory=dict)
    gpus: list[int] | None = None
    name: str = ""


class PlannerRequest(_Req):
    model: str | None = Field(default=None, description="Model id from /models (or a local path).")
    hf_config: dict[str, Any] | None = Field(default=None, description="Inline HF config.json (or ModelConfig dict) instead of `model`.")
    chip: str | None = Field(default=None, description="Chip name/id (see /hardware/chips).")
    descriptor_id: str | None = Field(default=None, description="Measured descriptor id (see /hardware/descriptors).")
    quant: str | dict[str, Any] | None = Field(default=None, description="Quant scheme name or recipe dict.")
    kv_dtype: str | None = None
    workload: dict[str, Any] = Field(default_factory=dict)
    budget: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict, description="Extra keyword arguments forwarded when the planner function accepts them.")


class ClaimUpdate(BaseModel):
    """Accepts the whole claim object as the UI sends it; only the editable fields below are applied."""

    model_config = ConfigDict(extra="ignore")
    id: str | None = Field(default=None, description="Claim id (may also come from the URL).")
    status: str | None = Field(default=None, description="untested | reproduced | contradicted | partial")
    prior: str | None = Field(default=None, description="trusted | untrusted")
    notes: str | None = None
    linked_experiments: list[str] | None = None


class StorageGcRequest(_Req):
    dry_run: bool = True


class PinRequest(_Req):
    reason: str = "pinned from UI"


class TierRequest(_Req):
    to: str = Field(description="hot | warm | cold")


class EvalCompareRequest(_Req):
    ids: list[str] = Field(min_length=1)
    parent: str | None = Field(default=None, description="Eval id used as the baseline (default: first id).")
    metrics: list[str] | None = Field(default=None, description="Restrict the table to these metrics.")
    quality_metric: str | None = Field(default=None, description="Metric on the quality axis of the Pareto front (forge.eval).")
    pareto: dict[str, Any] = Field(
        default_factory=dict,
        description="{x: metric, y: metric, x_min: bool, y_min: bool} - axes for the Pareto front (default: first two shared metrics).",
    )


class BenchRequest(_Req):
    params: dict[str, Any] = Field(default_factory=dict)
    gpus: list[int] | None = None
    name: str = ""


class DeploymentCreate(_Req):
    name: str = ""
    artifact_ref: str = Field(description="Model id (from /models), local directory, or store artifact hash.")
    serve_config: dict[str, Any] = Field(default_factory=dict, description="Overrides merged into ServeConfig.")
    gpus: list[int] | None = None
    template: str | None = None


class ServeSimulateRequest(PlannerRequest):
    weight_bits: float | None = None
    context_dist: dict[str, Any] | list[Any] | None = None


class CheckpointExportRequest(_Req):
    name: str | None = None
    dtype: str | None = None
    register_model: bool = True
