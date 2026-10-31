"""Pydantic v2 data models shared by every Forge module (Plan Appendix B/C and ARCHITECTURE.md section 4.2).

Conventions
-----------
* ``extra="forbid"`` everywhere unless stated; every field carries a ``description`` (UI forms are generated
  from the JSON schema).
* All timestamps are ``float`` UNIX epoch seconds.
* ``ArtifactManifest.model_cfg`` is the contract's ``model_config`` field: pydantic reserves that name, so it is
  stored as ``model_cfg`` and accepted as ``model_config`` on input (``AliasChoices``).
"""
from __future__ import annotations

import enum
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from forge.core.hashing import hash_json
from forge.core.util import new_id, now


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), validate_assignment=False)


# ============================================================================================
# provenance / metrics
# ============================================================================================
class Provenance(str, enum.Enum):
    """Where a number came from. Every paper-derived number shown in the UI carries one."""

    measured = "measured"
    reproduced = "reproduced"
    paper_reported = "paper_reported"
    derived = "derived"
    vendor = "vendor"
    target = "target"
    untested = "untested"


class MetricValue(_Base):
    """A metric with optional confidence interval and provenance."""

    value: float = Field(description="Point estimate.")
    ci_low: float | None = Field(default=None, description="Lower bound of the confidence interval.")
    ci_high: float | None = Field(default=None, description="Upper bound of the confidence interval.")
    n: int | None = Field(default=None, ge=0, description="Number of samples/seeds behind the estimate.")
    provenance: Provenance = Field(default=Provenance.measured, description="Origin of the number.")
    within_noise: bool | None = Field(
        default=None, description="True if the difference to the parent is within the measured noise floor."
    )

    @model_validator(mode="after")
    def _check_ci(self) -> MetricValue:
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError("ci_low must be <= ci_high")
        return self


# ============================================================================================
# chip descriptor (Plan Appendix B)
# ============================================================================================
class MeasuredPair(_Base):
    """A datasheet number paired with the probe-measured one."""

    datasheet: float | None = Field(default=None, ge=0, description="Vendor datasheet value.")
    measured: float | None = Field(default=None, ge=0, description="Value measured by the Forge microbenchmark.")
    src: str | None = Field(default=None, description="Provenance string, e.g. 'microbench/stream_copy@r3'.")

    @property
    def best(self) -> float | None:
        """Measured if available, else datasheet."""
        return self.measured if self.measured is not None else self.datasheet


class MemoryInfo(_Base):
    hbm_gb: float = Field(default=0.0, ge=0, description="Device memory capacity in GB.")
    hbm_bw_gbs: MeasuredPair = Field(default_factory=MeasuredPair, description="Memory bandwidth in GB/s.")
    l2_mb: float = Field(default=0.0, ge=0, description="L2 cache size in MB.")
    transaction_bytes: int = Field(default=128, ge=1, description="Memory transaction granularity in bytes.")
    max_vector_load_bytes: int = Field(default=16, ge=1, description="Widest vectorised load in bytes.")


class OnChipInfo(_Base):
    smem_per_cu_kib: float = Field(default=0.0, ge=0, description="Shared memory per compute unit (KiB).")
    smem_banks: int = Field(default=32, ge=1, description="Number of shared-memory banks.")
    bank_width_bytes: int = Field(default=4, ge=1, description="Width of a shared-memory bank in bytes.")
    regs_per_cu: int = Field(default=65536, ge=0, description="32-bit registers per compute unit.")
    max_regs_per_thread: int = Field(default=255, ge=0, description="Max registers per thread.")
    async_copy: list[str] = Field(
        default_factory=list, description="Async copy mechanisms available, e.g. ['cp_async', 'bulk_async']."
    )
    cluster_launch: bool = Field(default=False, description="Thread-block clusters supported.")


class MatrixUnit(_Base):
    """One tensor-core data type capability."""

    dtype: str = Field(description="Element type, e.g. 'fp16', 'bf16', 'fp8_e4m3', 'int8', 'fp4_e2m1'. "
                       "'a|b' lists dtypes sharing one entry.")
    atom: list[int] | None = Field(default=None, description="MMA atom shape [M, N, K] (N may be 0 for 'any').")
    peak_tflops: MeasuredPair = Field(default_factory=MeasuredPair, description="Peak throughput in TFLOP/s (TOP/s).")
    block_scaled: dict[str, Any] | bool | None = Field(
        default=None,
        description="Block-scaling support, e.g. {block: 16, scale: e4m3, scale_atom: [128, 4]}; None/False if absent.",
    )


class SparsityInfo(_Base):
    two_four: bool = Field(default=False, description="2:4 structured sparse tensor cores available.")


class AluInfo(_Base):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    simple_ops_per_s_t: float | None = Field(default=None, description="Simple integer/float ops per second (T).")


class InterconnectInfo(_Base):
    intra_gbs: float | None = Field(default=None, ge=0, description="Intra-node GPU-GPU bandwidth (GB/s).")
    inter_gbs: float | None = Field(default=None, ge=0, description="Inter-node bandwidth (GB/s).")
    topology: str | None = Field(default=None, description="Topology label: switch, pcie, ring, none...")


class DerivedInfo(_Base):
    ridge_flop_per_byte: dict[str, float] = Field(
        default_factory=dict, description="Roofline ridge point (FLOP/byte) per dtype: peak_flops / hbm_bw."
    )


def _canon_dtype(d: str) -> str:
    d = d.strip().lower()
    table = {
        "bfloat16": "bf16", "float16": "fp16", "half": "fp16", "float32": "fp32", "tf32": "tf32",
        "float8_e4m3fn": "fp8_e4m3", "float8_e4m3": "fp8_e4m3", "e4m3": "fp8_e4m3",
        "float8_e5m2": "fp8_e5m2", "e5m2": "fp8_e5m2", "float4_e2m1": "fp4_e2m1", "e2m1": "fp4_e2m1",
    }
    return table.get(d, d)


class ChipDescriptor(_Base):
    """Hardware description (Plan Appendix B). Datasheet values may be filled from ``forge.hw.chips``;
    measured values come from the probe."""

    id: str = Field(description="Unique id, e.g. 'h100-sxm-80g-r3'.")
    fingerprint: str = Field(default="", description="Stable hash of the identifying properties of this device.")
    vendor: str = Field(default="nvidia", description="Vendor name.")
    family: str = Field(default="unknown", description="Family label: ampere, ada, hopper, blackwell, cpu_avx, unknown...")
    name: str = Field(default="unknown", description="Marketing/device name.")
    sm: int | None = Field(default=None, description="CUDA SM level as an integer (e.g. 90 for sm_90). None for non-NVIDIA.")
    compute_capability: str = Field(default="", description="Compute capability string, e.g. '9.0'.")
    compute_units: int = Field(default=0, ge=0, description="Number of SMs / compute units.")
    clock_mhz: float | None = Field(default=None, ge=0, description="Sustained clock in MHz (locked if known).")
    memory: MemoryInfo = Field(default_factory=MemoryInfo, description="Memory subsystem.")
    on_chip: OnChipInfo = Field(default_factory=OnChipInfo, description="On-chip resources.")
    matrix_units: list[MatrixUnit] = Field(default_factory=list, description="Tensor-core capabilities per dtype.")
    sparsity: SparsityInfo = Field(default_factory=SparsityInfo, description="Structured sparsity support.")
    alu: AluInfo = Field(default_factory=AluInfo, description="Scalar ALU throughput.")
    launch_latency_us: float | None = Field(default=None, ge=0, description="Kernel launch latency in microseconds.")
    interconnect: InterconnectInfo = Field(default_factory=InterconnectInfo, description="Interconnect.")
    derived: DerivedInfo = Field(default_factory=DerivedInfo, description="Values derived from the above.")
    power_limit_w: float | None = Field(default=None, ge=0, description="Board power limit in watts.")
    tier: Literal["A", "B", "C"] = Field(
        default="C", description="Kernel tier: A native fast path confirmed by probe, B partially, C portable fallback."
    )
    probe_provenance: dict[str, Provenance] = Field(
        default_factory=dict, description="Per-field provenance (measured/vendor/...)."
    )

    def _dtype_set(self) -> list[tuple[set[str], MatrixUnit]]:
        out = []
        for mu in self.matrix_units:
            out.append(({_canon_dtype(p) for p in mu.dtype.split("|")}, mu))
        return out

    def matrix_unit_for(self, dtype: str) -> MatrixUnit | None:
        """Matrix-unit entry that handles ``dtype`` (families like 'fp8' match 'fp8_e4m3'), else None."""
        q = _canon_dtype(dtype)
        fp4_family = q in ("fp4", "nvfp4", "mxfp4")
        for names, mu in self._dtype_set():
            for t in names:
                if t == q or t.startswith(q + "_"):
                    return mu
                if fp4_family and t.startswith("fp4"):
                    if q == "fp4" or mu.block_scaled:
                        return mu
        return None

    def supports(self, dtype: str) -> bool:
        """True if the descriptor lists a matrix unit for ``dtype``."""
        return self.matrix_unit_for(dtype) is not None


# ============================================================================================
# model / workload refs
# ============================================================================================
class ModelRef(_Base):
    id: str = Field(description="Unique id within Forge.")
    name: str = Field(default="", description="Display name.")
    source: Literal["local", "hf"] = Field(default="local", description="Where the weights live.")
    path_or_repo: str = Field(default="", description="Local path or Hugging Face repo id.")
    revision: str | None = Field(default=None, description="Git revision / commit for Hugging Face models.")
    hash: str | None = Field(default=None, description="Content hash of the weights (artifact manifest hash).")
    licence: str | None = Field(default=None, description="SPDX-like licence id or free text.")
    distill_allowed: bool | None = Field(default=None, description="Whether the licence permits distillation; None = unknown.")
    params_b: float | None = Field(default=None, ge=0, description="Parameter count in billions.")


class Workload(_Base):
    context_lengths: list[int] = Field(default_factory=lambda: [4096], description="Typical context lengths (tokens).")
    batch_sizes: list[int] = Field(default_factory=lambda: [1, 8, 32], description="Concurrent sequences to plan for.")
    ttft_slo_ms: float | None = Field(default=None, ge=0, description="Time-to-first-token SLO in ms.")
    itl_slo_ms: float | None = Field(default=None, ge=0, description="Inter-token latency SLO in ms.")
    prefix_reuse: float = Field(default=0.0, ge=0, le=1, description="Fraction of prompt tokens shareable via prefix cache.")
    task_mix: dict[str, float] = Field(default_factory=dict, description="Task name -> weight, e.g. {'chat': 0.7}.")


# ============================================================================================
# recipe
# ============================================================================================
class StageSpec(_Base):
    name: str = Field(min_length=1, description="Registered stage name (kind 'stage' in the registry).")
    params: dict[str, Any] = Field(default_factory=dict, description="Stage parameters.")


class Recipe(_Base):
    """An ordered compression/training recipe (Plan Appendix C.1). ``hash()`` is a pure function of content."""

    name: str = Field(min_length=1, description="Recipe name.")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Named inputs (artifact hashes, ids, literals).")
    stages: list[StageSpec] = Field(default_factory=list, description="Stages executed in order.")
    baselines: list[str] = Field(default_factory=list, description="Baselines to compare against.")
    budget: dict[str, Any] = Field(default_factory=dict, description="Budget, e.g. {gpu_hours: 40, max_wall_hours: 12}.")
    policy: dict[str, Any] = Field(default_factory=dict, description="Policies, e.g. {licence_check: true}.")
    seeds: list[int] = Field(default_factory=lambda: [0], description="Seeds for stochastic stages / repeated evals.")

    def canonical(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def hash(self) -> str:
        """Content hash (digest string). Equal recipes -> equal hash regardless of key order."""
        return hash_json(self.canonical())

    def stage_names(self) -> list[str]:
        return [s.name for s in self.stages]


# ============================================================================================
# artifacts
# ============================================================================================
TensorRole = Literal["payload", "scale", "zero", "tensor_scale", "perm", "codebook", "other"]


class TensorEntry(_Base):
    name: str = Field(description="Tensor name within the artifact.")
    dtype: str = Field(description="Storage dtype string, e.g. 'bf16', 'uint8', 'int32'.")
    shape: list[int] = Field(description="Tensor shape.")
    hash: str = Field(description="Content hash of the tensor bytes.")
    layout_id: str | None = Field(default=None, description="Packed-layout id (see forge.kernels.layouts).")
    role: TensorRole = Field(default="payload", description="Role of the tensor inside a packed weight.")


class ArtifactManifest(_Base):
    """Manifest of a stored artifact. ``hash`` is computed from content, excluding ``created_at``
    (so identical content always has an identical hash)."""

    kind: str = Field(description="Artifact kind: model, checkpoint, adapter, dataset, calib, eval, ...")
    model_cfg: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("model_cfg", "model_config"),
        description="Model configuration dict (HF-style config.json content). Contract name: 'model_config'.",
    )
    tensors: list[TensorEntry] = Field(default_factory=list, description="Tensors making up the artifact.")
    recipe_hash: str | None = Field(default=None, description="Hash of the recipe that produced this artifact.")
    parents: list[str] = Field(default_factory=list, description="Parent artifact hashes.")
    calib_hash: str | None = Field(default=None, description="Hash of the calibration set used.")
    licence: str | None = Field(default=None, description="Licence inherited from the parents.")
    hardware_families: list[str] = Field(default_factory=list, description="Hardware families the layouts target.")
    eval_report_hashes: list[str] = Field(default_factory=list, description="Eval reports attached to the artifact.")
    created_at: float = Field(default_factory=now, description="Creation time (epoch seconds); not part of the hash.")
    extra: dict[str, Any] = Field(default_factory=dict, description="Free-form extra metadata.")

    @model_validator(mode="before")
    @classmethod
    def _drop_hash(cls, data: Any) -> Any:
        if isinstance(data, dict) and "hash" in data:
            data = {k: v for k, v in data.items() if k != "hash"}
        return data

    def content_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"created_at", "hash"})

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hash(self) -> str:
        return hash_json(self.content_dict())

    def tensor(self, name: str) -> TensorEntry | None:
        for t in self.tensors:
            if t.name == name:
                return t
        return None


# ============================================================================================
# evaluation
# ============================================================================================
class GateVerdict(_Base):
    gate_id: str = Field(description="Gate identifier, e.g. 'G-W4-v2'.")
    verdict: Literal["pass", "fail", "inconclusive"] = Field(description="Outcome.")
    reasons: list[str] = Field(default_factory=list, description="Human-readable reasons for the verdict.")
    details: dict[str, Any] = Field(default_factory=dict, description="Per-metric details.")


class EvalReport(_Base):
    id: str = Field(default_factory=lambda: new_id("eval"), description="Report id.")
    artifact_hash: str = Field(description="Candidate artifact evaluated.")
    parent_hash: str | None = Field(default=None, description="Parent artifact used as reference.")
    suite_versions: dict[str, str] = Field(default_factory=dict, description="Suite name -> version.")
    metrics: dict[str, MetricValue] = Field(default_factory=dict, description="Metric name -> value.")
    gate: GateVerdict | None = Field(default=None, description="Gate verdict if a gate was applied.")
    noise_floor: dict[str, float] = Field(default_factory=dict, description="Metric -> noise floor from parent repeats.")
    created_at: float = Field(default_factory=now, description="Creation time (epoch seconds).")
    env: dict[str, Any] = Field(default_factory=dict, description="Captured environment (see core.env).")


class Claim(_Base):
    """An entry of the claims ledger (what the papers assert vs. what Forge verified)."""

    id: str = Field(description="Claim id, e.g. 'P1-W4-001'.")
    source_paper: str = Field(description="Paper identifier (P1..P7).")
    location: str = Field(default="", description="Section/figure/table inside the paper.")
    statement: str = Field(description="The claim as a testable statement.")
    status: Literal["untested", "reproduced", "contradicted", "partial"] = Field(default="untested", description="Verification status.")
    prior: Literal["trusted", "untrusted"] = Field(default="untrusted", description="Prior trust before testing.")
    linked_experiments: list[str] = Field(default_factory=list, description="Eval report / job ids testing the claim.")
    depends_rules: list[str] = Field(default_factory=list, description="Rulebase rule ids relying on the claim.")
    notes: str = Field(default="", description="Free-form notes.")


# ============================================================================================
# jobs / deployments / datasets
# ============================================================================================
class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled)


class JobSpec(_Base):
    type: str = Field(min_length=1, description="Job type (probe, train, finetune, distill, compress, eval, serve, ...).")
    name: str = Field(default="", description="Human-friendly job name.")
    params: dict[str, Any] = Field(default_factory=dict, description="Job configuration (validated by the job type's model).")
    gpus: list[int] | None = Field(default=None, description="GPU indices to reserve; None = scheduler decides / CPU job.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags for filtering.")


class JobRecord(_Base):
    id: str = Field(default_factory=lambda: new_id("job"), description="Job id.")
    spec: JobSpec = Field(description="What was requested.")
    status: JobStatus = Field(default=JobStatus.queued, description="Lifecycle state.")
    created_at: float = Field(default_factory=now, description="Submission time (epoch seconds).")
    started_at: float | None = Field(default=None, description="Start time.")
    finished_at: float | None = Field(default=None, description="End time.")
    pid: int | None = Field(default=None, description="Worker process id.")
    exit_code: int | None = Field(default=None, description="Worker exit code.")
    progress: float = Field(default=0.0, ge=0, le=1, description="Progress fraction 0..1.")
    progress_msg: str = Field(default="", description="Latest progress message.")
    error: str | None = Field(default=None, description="Error text for failed jobs.")
    outputs: dict[str, Any] = Field(default_factory=dict, description="Outputs set by the job via ctx.set_output.")
    run_dir: str | None = Field(default=None, description="Run directory holding logs and checkpoints.")


class Deployment(_Base):
    id: str = Field(default_factory=lambda: new_id("dep"), description="Deployment id.")
    name: str = Field(default="", description="Display name.")
    artifact_ref: str = Field(description="Model id / artifact hash / path being served.")
    serve_config: dict[str, Any] = Field(default_factory=dict, description="ServeConfig as a dict.")
    job_id: str | None = Field(default=None, description="Serve job running the engine.")
    port: int | None = Field(default=None, ge=1, le=65535, description="Port the engine listens on.")
    status: str = Field(default="created", description="created | starting | running | stopped | failed.")
    created_at: float = Field(default_factory=now, description="Creation time (epoch seconds).")


class ShardEntry(BaseModel):
    """One dataset shard. Extra keys allowed (e.g. offsets sidecar hash)."""

    model_config = ConfigDict(extra="allow", protected_namespaces=())
    hash: str = Field(description="Content hash of the shard file.")
    tokens: int = Field(default=0, ge=0, description="Tokens in the shard.")
    docs: int = Field(default=0, ge=0, description="Documents in the shard.")


class DatasetManifest(_Base):
    id: str = Field(default_factory=lambda: new_id("ds"), description="Dataset id.")
    name: str = Field(default="", description="Display name.")
    tokenizer_hash: str = Field(default="", description="Hash of the tokenizer used to build the shards.")
    shards: list[ShardEntry] = Field(default_factory=list, description="Token shards.")
    mixture: dict[str, Any] = Field(default_factory=dict, description="Source name -> weight/token counts.")
    filtering_version: str = Field(default="", description="Version of the filtering pipeline.")
    licence: str | None = Field(default=None, description="Licence of the data.")
    dedup_report: dict[str, Any] = Field(default_factory=dict, description="Dedup statistics.")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens across shards.")

    @model_validator(mode="after")
    def _fill_total(self) -> DatasetManifest:
        if self.total_tokens == 0 and self.shards:
            self.total_tokens = sum(s.tokens for s in self.shards)
        return self

    def content_hash(self) -> str:
        return hash_json(self.model_dump(mode="json", exclude={"id"}))
