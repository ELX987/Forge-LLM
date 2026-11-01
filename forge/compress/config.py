"""Configuration models of the compression engine: job config and the per-stage parameter models.

``CompressJobConfig`` is the config model of the ``compress`` job type (``forge.compress.run.run_compress``).  Every recipe stage validates
its ``params`` mapping against one of the models in :data:`STAGE_PARAM_MODELS`, so the UI can render stage forms from their JSON schema
(:func:`stage_schemas`).  Models for the algorithms live next to the algorithms (``ptq.PTQConfig``, ``prune.PruneConfig``, ...) and are
re-exported here.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.compress.delta import DeltaConfig
from forge.compress.kvquant import KVQuantConfig
from forge.compress.lossless import LosslessConfig
from forge.compress.lowrank import LowRankConfig
from forge.compress.prune import PruneConfig, SparsityConfig
from forge.compress.ptq import PTQConfig
from forge.compress.rotation import RotationConfig


class CalibrateParams(BaseModel):
    """Stage ``calibrate``: choose and fingerprint the calibration data (and optionally fill the statistics cache)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    set: str | None = Field(None, description="Registered calibration set name (synthetic, builtin-mix, builtin-prose|code|math|chat, file:<path>, or one registered by the user). Default: the recipe input 'calib' if it names a set, else synthetic.")
    n_seqs: int = Field(128, ge=1, description="Number of calibration sequences.")
    seq_len: int = Field(512, ge=8, description="Tokens per sequence.")
    stats: list[Literal["hessian", "channel_scale", "router"]] = Field(["hessian", "channel_scale"], description="Statistics later stages will use. hessian / channel_scale are computed on demand by ptq, awq, smoothquant and lowrank (sequentially, on the already-compressed block inputs); 'router' is collected here (needed for expert pruning).")
    eager: bool = Field(False, description="Also collect hessian / channel_scale now into the statistics cache (analysis, UI). GPTQ does not need this.")
    batch_size: int = Field(4, ge=1, description="Batch size of the statistics pass.")
    seed: int = Field(0, description="Sampling seed.")


class TernarizeParams(BaseModel):
    """Stage ``ternarize``: data-free post-training ternarisation to packed ``t158`` layers ({-1,0,+1} weights, fp32 group scales; EXPERIMENTAL).

    One-shot ternarisation of a pretrained model is lossy: precede it with a `finetune` / `distill` stage that sets
    ``precision.quant_aware: ternary`` (QAT recovery), or use this stage on a model already trained with ternary QAT
    (then it reproduces the QAT forward weights exactly)."""

    model_config = ConfigDict(extra="forbid")

    group_size: int = Field(128, ge=-1, description="Scale group along K (>0), -1 = one scale per output row, 0 = one per tensor. Converted pretrained models need 128.")
    rule: Literal["absmean", "twn", "lloyd"] = Field("lloyd", description="Weight quantiser: absmean, TWN, or Lloyd (absmean start + refinement).")
    iters: int = Field(1, ge=1, le=16, description="Lloyd refinement steps.")
    exclude: list[str] = Field(["lm_head", "(^|\\.)gate$", "router", "shared_expert_gate", "embed"], description="Regexes of module names kept in high precision.")
    include: list[str] | None = Field(None, description="If set, only modules whose name matches one of these regexes are ternarised.")


class PackParams(BaseModel):
    """Stage ``pack``: derive the device layout of packed weights for a target chip."""

    model_config = ConfigDict(extra="forbid")

    layout: str = Field("auto", description="Layout id, or 'auto' to derive it from the kernel spec and the target chip (forge.kernels.layouts.derive_layout).")
    target: str | None = Field(None, description="Chip descriptor id (measured descriptor) or null = the local GPU / portable layout.")
    executor: Literal["triton", "native"] = Field("triton", description="Kernel executor the layout is derived for (triton = portable row layout).")
    validate_roundtrip: bool = Field(True, description="Check unpack(pack(w)) reproduces the codes bit-exactly for every packed layer.")


class EvalParams(BaseModel):
    """Stage ``eval``: parent-vs-candidate evaluation (delegates to forge.eval.run.run_eval)."""

    model_config = ConfigDict(extra="forbid")

    suites: list[str] = Field(["distribution"], description="Suites of forge.eval to run (distribution, mcq, longgen, longctx, safety, contamination, ...).")
    suite_params: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-suite parameters, keyed by suite name.")
    parent: str | None = Field(None, description="Digest of the reference model (default: the recipe's original input model).")
    gate: str | None = Field(None, description="Optional forge.eval gate id (g-w4, g-fp8, g-lossless, g-struct, ...) evaluated together with the suites.")
    seeds: list[int] = Field([0], description="Evaluation seeds (first = main run).")
    n_bootstrap: int = Field(1000, ge=100, description="Bootstrap resamples for confidence intervals.")
    context: dict[str, Any] = Field(default_factory=dict, description="Extra facts for gate rules (params_b, ...).")
    persist: bool = Field(True, description="Store the EvalReport in the database.")
    builtin: bool = Field(False, description="Skip forge.eval and use the minimal built-in comparison (KL / flip / perplexity). Used automatically if forge.eval is unavailable.")
    n_seqs: int = Field(16, ge=1, description="Built-in comparison: sequences.")
    seq_len: int = Field(128, ge=8, description="Built-in comparison: tokens per sequence.")
    set: str = Field("builtin-mix", description="Built-in comparison: held-out text set.")
    seed: int = Field(10_000, description="Built-in comparison: sampling seed (distinct from calibration).")


class GateParams(BaseModel):
    """Stage ``gate``: pass/fail/inconclusive verdict for the candidate against its parent."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    suite: str = Field("g-w4", alias="gate", description="Gate id (forge.eval gate: g-w4, g-fp8, g-lossless, g-struct, g-long, g-spec, g-fp4train) or path of a gate YAML. Appendix-C names like G-W4-v2 resolve to their base gate.")
    parent: str | None = Field(None, description="Digest of the reference model (default: the recipe's original input model).")
    seeds: int | list[int] = Field(1, description="Seeds (a number n means seeds 0..n-1).")
    stats: str = Field("bootstrap_2000", description="Statistics recipe, e.g. bootstrap_2000 (n resamples).")
    suites: list[str] = Field(["distribution"], description="Evaluation suites feeding the gate.")
    suite_params: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-suite parameters.")
    overrides: dict[str, Any] = Field(default_factory=dict, description="Deep-merged into the gate definition (thresholds).")
    context: dict[str, Any] = Field(default_factory=dict, description="Facts for gate rules (params_b, ...).")
    enforce: bool = Field(False, description="Raise GateFailed in the stage on a failing verdict (otherwise the verdict is recorded and the job policy decides).")
    builtin_thresholds: dict[str, float] = Field(default_factory=lambda: {"kl_p99_max": 0.05, "top1_flip_max": 0.04}, description="Thresholds used by the built-in fallback gate (when forge.eval is unavailable): kl_p99_max, top1_flip_max, ppl_ratio_max.")


class DistillParams(BaseModel):
    """Stage ``distill``: recovery after pruning / low-rank / quantisation."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["builtin", "delegate"] = Field("builtin", description="builtin = short logit-KL recovery on the calibration set (smoke / healing); delegate = forge.train.run.run_distill with the 'config' mapping (production).")
    teacher: str | None = Field(None, description="Digest of the teacher model (default: 'teacher' output of the prune stage, else the recipe's original input model).")
    steps: int = Field(50, ge=1, description="builtin: optimisation steps.")
    lr: float = Field(1e-4, gt=0, description="builtin: learning rate.")
    temperature: float = Field(1.0, gt=0, description="builtin: distillation temperature.")
    batch_size: int = Field(4, ge=1, description="builtin: batch size.")
    n_seqs: int = Field(64, ge=1, description="builtin: sequences drawn from the calibration set.")
    seq_len: int = Field(128, ge=8, description="builtin: tokens per sequence.")
    set: str = Field("synthetic", description="builtin: calibration set to distil on.")
    seed: int = Field(0, description="builtin: sampling seed.")
    trainable: Literal["all", "norms_and_factors"] = Field("all", description="builtin: parameters to train (norms_and_factors = RMSNorm weights and low-rank factors only).")
    config: dict[str, Any] = Field(default_factory=dict, description="delegate: DistillConfig overrides (data section, optimiser, ...). model.path / teacher.path are filled in automatically.")
    licence_check: bool | None = Field(None, description="Block when the teacher's licence forbids distillation (default: the recipe policy 'licence_check').")


class FinetuneParams(BaseModel):
    """Stage ``finetune``: delegates to forge.train.run.run_finetune."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any] = Field(default_factory=dict, description="FinetuneConfig mapping (data, LoRA, optimiser...). model.path is filled in with the current model.")
    merge_lora: bool = Field(True, description="Use the merged model produced by the run as the stage output.")


class CompressJobConfig(BaseModel):
    """Compression job: run a recipe (Plan App. C.1) on a model and register the resulting artifact with lineage and licence metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field("compress", description="Job / artifact base name.")
    model: str = Field(..., description="Model to compress: local directory, Hugging Face repo id, artifact-store hash, or 'tiny[:seed]' for the built-in CPU smoke model.")
    model_source: Literal["auto", "local", "hf", "store", "tiny"] = Field("auto", description="How to interpret 'model' (auto = tiny: prefix / store digest / existing directory / HF repo id).")
    revision: str | None = Field(None, description="Hugging Face revision for hf models.")
    tiny_overrides: dict[str, Any] = Field(default_factory=dict, description="ModelConfig overrides when model='tiny' (e.g. {'vocab_size': 512, 'num_layers': 4}).")
    recipe: str | dict[str, Any] | None = Field(None, description="Inline recipe (YAML text or mapping, Appendix C.1 layout). Exactly one of recipe / recipe_path.")
    recipe_path: str | None = Field(None, description="Path of a recipe YAML file (e.g. configs/recipes/w4a16-gptq.yaml).")
    output_name: str | None = Field(None, description="Name of the produced artifact (default: <name>-<recipe name>).")
    dtype: Literal["bfloat16", "float16", "float32"] = Field("bfloat16", description="Dtype the model is loaded in (scales of packed layers use it).")
    device: str = Field("auto", description="auto (cuda if available), cuda, cuda:N or cpu.")
    seed: int = Field(0, description="Global seed (also the recipe seed unless the recipe lists seeds).")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Extra recipe inputs (override the recipe's own, e.g. {'calib': 'my-set'}).")
    stage_overrides: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-stage parameter overrides merged into every stage of that name (e.g. {'calibrate': {'n_seqs': 16, 'seq_len': 64}} to smoke-run a production recipe on a small model).")
    chip: str | None = Field(None, description="Target chip descriptor id (recorded as the recipe input 'chip' and used by the pack stage).")
    licence: str | None = Field(None, description="SPDX licence id of the input model (recorded in the artifact manifest and checked by licence-gated stages).")
    distill_allowed: bool | None = Field(None, description="Explicit attestation whether distillation from this model is allowed (null = decide from licence).")
    params_b: float | None = Field(None, ge=0, description="Parameter count in billions (for size-dependent gate thresholds); default: computed.")
    tokenizer: str | None = Field(None, description="Tokenizer path / repo (default: the model's own; byte fallback for tiny models).")
    calibration_texts: dict[str, list[str]] = Field(default_factory=dict, description="Extra calibration sets to register for this job: name -> list of texts.")
    calibration_files: dict[str, str] = Field(default_factory=dict, description="Extra calibration sets from text/jsonl files: name -> path.")
    use_cache: bool = Field(True, description="Use the recipe stage cache (needs an artifact store; re-running unchanged stages is instant).")
    use_store: bool = Field(True, description="Register intermediate and final artifacts in the artifact store (when one is available).")
    register_input: bool = Field(True, description="Ingest the input model into the artifact store first (lineage root, stable digest) when it did not come from the store.")
    persist_intermediate: bool = Field(True, description="Write every model-producing stage to the store (required for stage caching across processes). Disable for very large models to save disk.")
    register_model: bool = Field(True, description="When run as a job: also register the output directory in the model registry (source 'artifact', with quantisation / base-model metadata) so the Inference page can select it.")
    save_dir: str | None = Field(None, description="Also write the final model directory here (default: <run_dir>/artifact).")
    fail_on_gate: bool = Field(True, description="Mark the job failed (after registering the artifact) when a gate verdict is 'fail'.")
    copy_below_gb: float = Field(2.0, ge=0, description="Stages copy the model when it is smaller than this (keeps earlier digests usable); larger models are modified in place.")

    @model_validator(mode="after")
    def _one_recipe(self) -> CompressJobConfig:
        if (self.recipe is None) == (self.recipe_path is None):
            raise ValueError("give exactly one of 'recipe' (inline) or 'recipe_path'")
        return self


STAGE_PARAM_MODELS: dict[str, type[BaseModel]] = {
    "calibrate": CalibrateParams,
    "ptq": PTQConfig,
    "pack": PackParams,
    "ternarize": TernarizeParams,
    "kv": KVQuantConfig,
    "rotate": RotationConfig,
    "prune": PruneConfig,
    "lowrank": LowRankConfig,
    "delta": DeltaConfig,
    "lossless": LosslessConfig,
    "eval": EvalParams,
    "gate": GateParams,
    "distill": DistillParams,
    "finetune": FinetuneParams,
}

#: parameter aliases accepted in recipes (Appendix C.1 spelling -> model field)
PARAM_ALIASES: dict[str, dict[str, str]] = {"ptq": {"group": "group_size"}, "kv": {"dtype": "scheme"}}


def stage_schemas() -> dict[str, dict[str, Any]]:
    """``{stage: JSON schema}`` for the UI (via ``forge.core.jsonschema.schema_for``)."""
    from forge.core.jsonschema import schema_for

    return {name: schema_for(cls) for name, cls in STAGE_PARAM_MODELS.items()}


__all__ = ["CompressJobConfig", "STAGE_PARAM_MODELS", "stage_schemas", "PARAM_ALIASES", "SparsityConfig"] + [c.__name__ for c in STAGE_PARAM_MODELS.values()]
