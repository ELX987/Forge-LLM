"""Configuration models for the evaluation job (``EvalConfig``) - rendered as a form by the UI."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelSource(BaseModel):
    """Where a model comes from."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kind: Literal["path", "artifact", "hf", "none"] = Field(default="path", description="path: local HF-format directory; artifact: content hash in the artifact store; hf: Hugging Face repo id; none: not used.")
    ref: str = Field(default="", description="Directory, artifact manifest hash or repo id (per kind).")
    label: str = Field(default="", description="Display label in comparisons (default: the ref).")
    artifact_hash: str | None = Field(default=None, description="Artifact hash recorded in the report (defaults to the ref for kind=artifact).")
    backend: Literal["forge", "hf"] = Field(default="forge", description="Model implementation used to load the weights.")
    dtype: str = Field(default="bfloat16", description="Compute dtype: bfloat16 | float16 | float32.")
    device: str = Field(default="auto", description="'auto' (cuda if available else cpu), 'cuda', 'cuda:1' or 'cpu'.")
    revision: str | None = Field(default=None, description="Hub revision for kind=hf.")
    params_b: float | None = Field(default=None, ge=0, description="Parameter count in billions (used by size-dependent gate thresholds).")
    serve_url: str | None = Field(default=None, description="Base URL of a running OpenAI-compatible server, used by the throughput suite.")
    serve_model: str = Field(default="default", description="Model name sent to the server.")
    serve_api_key: str | None = Field(default=None, description="Bearer token for the server (not stored in reports).")


class SuiteSelection(BaseModel):
    """One suite to run and its parameters (validated against the suite's own config model at run time)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Registered suite name: distribution, mcq, longgen, longctx, safety, contamination, throughput, agentic.")
    params: dict[str, Any] = Field(default_factory=dict, description="Suite-specific parameters (see the suite's schema).")


class EvalConfig(BaseModel):
    """Evaluate a candidate against its parent (Plan 3.6 EV-1..EV-7, 10.1) and optionally apply a gate."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    name: str = Field(default="eval", description="Name of this evaluation.")
    candidate: ModelSource = Field(default_factory=ModelSource, description="The model / artifact under test.")
    parent: ModelSource | None = Field(default=None, description="The 16-bit (or previous) parent used as reference. Omit for absolute scores only.")
    tokenizer: str | None = Field(default=None, description="Tokenizer path / repo (default: the candidate's own; byte-level fallback for tiny test models).")
    suites: list[SuiteSelection] = Field(default_factory=lambda: [SuiteSelection(name="distribution"), SuiteSelection(name="mcq")], description="Suites to run, in order.")
    gate: str | None = Field(default=None, description="Gate id or YAML path (g-lossless, g-fp8, g-w4, g-struct, g-long, g-spec, g-fp4train, ...). Empty = no gate.")
    gate_overrides: dict[str, Any] = Field(default_factory=dict, description="Deep-merged into the gate definition (e.g. {'metrics': {'suite_score_ratio': {'min': 0.92}}}).")
    seeds: list[int] = Field(default_factory=lambda: [0], min_length=1, description="Pre-registered seeds: seeds[0] drives the main run, the rest seed the noise-floor repeats.")
    n_bootstrap: int = Field(default=2000, ge=100, le=100_000, description="Bootstrap resamples for CIs.")
    alpha: float = Field(default=0.05, gt=0, lt=0.5, description="Two-sided level of confidence intervals (0.05 = 95% CI).")
    noise_repeats: int | None = Field(default=None, ge=0, le=50, description="Parent repeats used to estimate the noise floor (None = the gate's value, 0 = disabled).")
    noise_perturb: Literal["seed", "batch", "both"] = Field(default="both", description="What changes between noise repeats: generator seeds, batch composition (padding / batch split), or both.")
    deployment_length: int | None = Field(default=None, ge=1, description="Context length of the deployment; long-context ratios are reported at this length.")
    context: dict[str, Any] = Field(default_factory=dict, description="Facts for gate rules, e.g. {'params_b': 8, 'storage_transform': true}.")
    extra_metrics: dict[str, float] = Field(default_factory=dict, description="Externally measured metrics to include (e.g. loss_gap_rel from a training run).")
    extra_flags: dict[str, bool] = Field(default_factory=dict, description="Externally established evidence flags (e.g. licence_gate_passed).")
    quality_metric: str = Field(default="suite_score", description="Metric used as the quality axis of the Pareto plot.")
    save_generations: bool = Field(default=True, description="Store all generations compressed in the artifact store / run dir (EV-7).")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")

    @field_validator("suites", mode="before")
    @classmethod
    def _coerce_suites(cls, v: Any) -> Any:
        if isinstance(v, list):
            return [{"name": s} if isinstance(s, str) else s for s in v]
        return v
