"""The papers' decision procedures as DATA (Plan 5.2): Figure 5.3 (quantisation flow Q1-Q6) and Table 5.1 (rules beyond
quantisation), each rule linked to claims-ledger entries, plus the engine that turns them into candidate recipes.

* :data:`FLOW` - the six questions of Fig. 5.3 (adapted from P1 Fig. 19.1) with their yes/no outcomes.
* :data:`RULES` - every rule: id, plan reference, ``when`` predicate (restricted expression over the planning context), the
  proposals it makes in first-try order, the claim ids it relies on (``forge.planner.claims_seed``), and flags
  (``training``, ``experimental``, ``requires`` hardware capabilities).
* :func:`propose` / :func:`propose_with_trace` - evaluate the rules for ``(model form, binding resource, chip capabilities,
  budget, workload)`` and return :class:`Candidate` objects whose ``recipe`` is a :class:`forge.core.schemas.Recipe`.

Questions are not mutually exclusive (Fig. 5.3 caption): a deployment may pass through several, e.g. FP8 weights/activations with a
4-bit KV cache - so all firing rules contribute candidates and the search ranks them.

Stage names and parameter keys follow ARCHITECTURE.md 5.6 (``calibrate``, ``ptq``, ``pack``, ``kv``, ``prune``, ``distill``,
``lowrank``, ``delta``, ``lossless``, ``eval``, ``gate``).  The exact parameter names are those the planner *requests*; the compress
module owns their validation.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from forge.core.schemas import Recipe, StageSpec, Workload
from forge.planner.spec import ChipSpec, ModelSpec, Scheme, parse_scheme

# --------------------------------------------------------------------------------------------
# Figure 5.3 - quantisation decision flow
# --------------------------------------------------------------------------------------------
FLOW: list[dict[str, Any]] = [
    {"id": "Q1", "question": "Does the BF16 model + KV fit at the target batch/context and meet cost + latency?",
     "yes": {"action": "Stay BF16/FP16 (FP8 only for throughput)", "rule": "R-Q1-BF16"}, "no": {"next": "Q2"}, "binding_hint": ["compute", "accuracy"],
     "source": "Plan Fig 5.3 / P1 Fig 19.1", "claims": ["P1-W8-01"]},
    {"id": "Q2", "question": "Does the chip have native FP8 tensor cores?",
     "yes": {"action": "FP8 W8A8 + FP8 KV (first try)", "rule": "R-Q2-FP8"}, "no": {"action": "go to Q3 (W4A16 needs no special tensor cores)", "next": "Q3"},
     "source": "Plan Fig 5.3", "claims": ["P1-W8-01", "P1-DEC-01"]},
    {"id": "Q3", "question": "Still too big, or is small-batch decode bandwidth-bound?",
     "yes": {"action": "W4A16 g64-128 (AWQ / GPTQ) + Marlin-class kernel", "rule": "R-Q3-W4A16"}, "no": {"next": "Q4"},
     "source": "Plan Fig 5.3", "claims": ["P1-DEC-01", "P1-MARLIN-01", "P1-W4-01", "P1-GPTQ-01"]},
    {"id": "Q4", "question": "Large-batch, compute-bound, on 4-bit-capable cores (Blackwell / MI350-class)?",
     "yes": {"action": "W4A4 block-scaled (NVFP4, or MXFP4 + MR-GPTQ) or W4A8; QAD if data is available", "rule": "R-Q4-FP4"}, "no": {"next": "Q5"},
     "source": "Plan Fig 5.3", "claims": ["P1-FP4-01", "P1-FP4-02", "P5-OVH-01", "P5-FMT-01", "P5-RHT-01"]},
    {"id": "Q5", "question": "Is the KV cache now the largest tensor (long context / many sequences)?",
     "yes": {"action": "FP8 KV -> 4-bit KV; sparse/latent attention; test at full length", "rule": "R-Q5-KV"}, "no": {"next": "Q6"},
     "source": "Plan Fig 5.3", "claims": ["P1-KV-01", "P3-KV-01", "P3-EVICT-01"]},
    {"id": "Q6", "question": "Must the model fit under 3.5 bits per weight?",
     "yes": {"action": "Vector / trellis PTQ or QAT / distillation; budget the evaluation effort", "rule": "R-Q6-SUB35"}, "no": {"action": "done", "next": None},
     "source": "Plan Fig 5.3", "claims": ["P1-SUB4-01", "P1-EVAL-01"]},
]
FLOW_FOOTER = "Every outcome: validate the deployed artifact (KL, flip rate, task suite incl. long-context + safety, free-form generation) vs the 16-bit parent; iterate or fall back one level."

# --------------------------------------------------------------------------------------------
# rules (Fig 5.3 outcomes + Table 5.1 rows)
# --------------------------------------------------------------------------------------------
RULES: list[dict[str, Any]] = [
    # ---- Figure 5.3 outcomes ------------------------------------------------------------------
    {"id": "R-Q1-BF16", "node": "Q1", "table": "Fig 5.3", "title": "Stay BF16/FP16", "forms": ["dense", "moe", "fleet"], "order": 0,
     "when": "bf16_fits and binding in ('compute', 'accuracy')", "claims": ["P1-W8-01"], "training": False, "experimental": False, "requires": [],
     "paper_refs": ["P1 s19"], "proposals": [{"kind": "bf16", "gate": None}], "note": "FP8 only for throughput; this is also the parent baseline arm."},
    {"id": "R-Q2-FP8", "node": "Q2", "table": "Fig 5.3", "title": "FP8 W8A8 + FP8 KV", "forms": ["dense", "moe", "fleet"], "order": 1,
     "when": "fp8", "claims": ["P1-W8-01", "P1-DEC-01"], "training": False, "experimental": False, "requires": ["fp8"], "paper_refs": ["P1 F2"],
     "proposals": [{"kind": "fp8", "kv": "fp8", "gate": "g-fp8"}, {"kind": "fp8", "kv": "bf16", "gate": "g-fp8", "when": "binding != 'kv'"}],
     "note": "First try on chips with FP8 tensor cores."},
    {"id": "R-Q3-W4A16", "node": "Q3", "table": "Fig 5.3", "title": "W4A16 g64-128 (AWQ/GPTQ) + Marlin-class kernel", "forms": ["dense", "moe", "fleet"], "order": 2,
     "when": "binding in ('capacity', 'bandwidth', 'kv') or not bf16_fits", "claims": ["P1-DEC-01", "P1-MARLIN-01", "P1-W4-01", "P1-GPTQ-01"], "training": False,
     "experimental": False, "requires": [], "paper_refs": ["P1 F2", "P1 F3", "P1 s14.2"],
     "proposals": [{"kind": "w4a16", "group": 128, "method": "gptq", "gate": "g-w4"}, {"kind": "w4a16", "group": 128, "method": "awq", "gate": "g-w4"},
                   {"kind": "w4a16", "group": 64, "method": "gptq", "gate": "g-w4", "when": "tight_accuracy"}],
     "note": "Weight-only 4-bit decides whether the model fits at all and gives ~3-4x at small batch if the kernel dequantises at full bandwidth."},
    {"id": "R-Q4-FP4", "node": "Q4", "table": "Fig 5.3", "title": "Block-scaled W4A4 (NVFP4 / MXFP4 + MR-GPTQ) or W4A8", "forms": ["dense", "moe"], "order": 3,
     "when": "binding == 'compute' and (fp4 or fp8)", "claims": ["P1-FP4-01", "P1-FP4-02", "P5-OVH-01", "P5-FMT-01", "P5-RHT-01", "P1-ROT-01"], "training": False, "experimental": False,
     "requires": [], "paper_refs": ["P1 F6", "P5 s11.3"],
     "proposals": [{"kind": "nvfp4", "gate": "g-w4", "when": "fp4"}, {"kind": "mxfp4", "gate": "g-w4", "when": "fp4"}, {"kind": "w4a8", "gate": "g-w4", "when": "fp8"}],
     "note": "FP4 compute only pays when the activation quantiser is fused (unfused: 3x peak -> 2.3x at N=4096). Without FP4 cores NVFP4 is a numerics study only."},
    {"id": "R-Q5-KV", "node": "Q5", "table": "Fig 5.3", "title": "FP8 KV -> 4-bit KV; sparse/latent attention", "forms": ["dense", "moe", "fleet"], "order": 4,
     "when": "binding == 'kv' or kv_dominates", "claims": ["P1-KV-01", "P3-KV-01", "P3-EVICT-01", "P1-EVAL-01"], "training": False, "experimental": False, "requires": [],
     "paper_refs": ["P1 F8", "P3 F1-F4"],
     "proposals": [{"kind": "kv", "kv": "fp8", "gate": "g-fp8"}, {"kind": "kv", "kv": "int4", "gate": "g-w4", "experimental": True}],
     "note": "KV precision decides concurrency/context once weights fit. 4-bit KV is experimental: test at full deployment length (RULER-style, multi-hop)."},
    {"id": "R-Q6-SUB35", "node": "Q6", "table": "Fig 5.3", "title": "Vector/trellis PTQ or QAT below 3.5 bits/weight", "forms": ["dense", "moe"], "order": 5,
     "when": "max_bits_per_weight is not None and max_bits_per_weight < 3.5", "claims": ["P1-SUB4-01", "P1-EVAL-01"], "training": False, "experimental": True, "requires": [],
     "paper_refs": ["P1 F5"],
     "proposals": [{"kind": "sub35", "gate": "g-w4", "experimental": True}, {"kind": "sub35_qat", "gate": "g-w4", "experimental": True, "training": True}],
     "note": "Tier 3 (Plan 1.3): scalar grids waste capacity at 2-3 bits; budget extra evaluation effort."},
    # ---- Table 5.1 -----------------------------------------------------------------------------
    {"id": "T-PRUNE-DISTILL", "node": None, "table": "Table 5.1 row 1", "title": "Structured width pruning (GQA-aware) + distillation", "forms": ["dense"], "order": 10,
     "when": "binding in ('capacity', 'bandwidth', 'kv')", "claims": ["P2-PD-01", "P2-DEPTH-01", "P4-BASE-01", "P2-SCALE-01", "P4-OPD-01"], "training": True,
     "experimental": False, "requires": [], "paper_refs": ["P2 s1.2", "P4 s1.3", "P2 F5"],
     "proposals": [{"kind": "prune_distill", "keep": 0.75, "gate": "g-struct"}, {"kind": "prune_distill", "keep": 0.5, "gate": "g-struct"}],
     "note": "Needs several B, ideally tens of B tokens of distillation from the unpruned teacher; always compare with the 'distil into a fresh smaller student' arm."},
    {"id": "T-MOE-PRUNE", "node": None, "table": "Table 5.1 row 2", "title": "Router-aware expert pruning/merging, then MoE-aware quantisation", "forms": ["moe"], "order": 11,
     "when": "binding in ('capacity', 'bandwidth')", "claims": ["P2-MOE-01", "P2-EVAL-01"], "training": False, "experimental": False, "requires": [], "paper_refs": ["P2 F5", "P4 s1.3"],
     "proposals": [{"kind": "expert_prune", "ratio": 0.25, "gate": "g-struct"}, {"kind": "expert_prune", "ratio": 0.5, "gate": "g-struct", "experimental": True}],
     "note": "Calibrate on mixed domains incl. generative samples; expert-offload tiers are the memory-bound alternative."},
    {"id": "T-ACT-SPARSE", "node": None, "table": "Table 5.1 row 3", "title": "Training-free activation sparsity (TEAL-style) on the quantised model", "forms": ["dense"], "order": 12,
     "when": "allow_custom_kernels and max_batch <= 16 and binding in ('bandwidth', 'capacity')", "claims": ["P2-ACT-01", "P2-SP-02"], "training": False, "experimental": True, "requires": [],
     "paper_refs": ["P2 F5"], "proposals": [{"kind": "act_sparsity", "sparsity": 0.4, "gate": "g-w4", "experimental": True}],
     "note": "40-50% activation sparsity gave 1.3-1.8x in the papers' own kernels; needs custom kernels."},
    {"id": "T-DELTA", "node": None, "table": "Table 5.1 row 4", "title": "Store fine-tunes as deltas; multi-tenant adapter serving", "forms": ["dense", "fleet"], "order": 13,
     "when": "many_finetunes", "claims": ["P4-DELTA-01"], "training": False, "experimental": False, "requires": [], "paper_refs": ["P4"],
     "proposals": [{"kind": "delta", "delta": "bitdelta", "gate": "g-struct", "experimental": True}, {"kind": "delta", "delta": "lowrank", "gate": "g-struct"}],
     "note": "Low-rank or 1-bit sign + scale deltas against the shared base."},
    {"id": "T-LOSSLESS", "node": None, "table": "Table 5.1 row 5", "title": "Entropy-code weights (zero behaviour change)", "forms": ["dense", "moe", "fleet"], "order": 14,
     "when": "optimise_storage", "claims": ["P4-LOSSLESS-01", "P3-SERVE-01"], "training": False, "experimental": False, "requires": [], "paper_refs": ["P4 F7"],
     "proposals": [{"kind": "lossless", "gate": "g-lossless"}],
     "note": "~30% for BF16 (P4); decode in a fused kernel only if one exists for the chip, otherwise cold storage only."},
    {"id": "T-LONG-NOTRAIN", "node": None, "table": "Table 5.1 row 6", "title": "Paging + prefix caching + chunked prefill; FP8/4-bit KV after full-length validation", "forms": ["dense", "moe", "fleet"], "order": 15,
     "when": "binding == 'kv' and not allow_training", "claims": ["P3-SERVE-01", "P3-EVICT-01", "P3-EVICT-02", "P1-KV-01"], "training": False, "experimental": False, "requires": [],
     "paper_refs": ["P3 F4-F5", "P1 F8"],
     "proposals": [{"kind": "serving_lossless", "gate": "g-lossless"}, {"kind": "kv_evict", "gate": "g-long", "experimental": True, "when": "retrieval_style"}],
     "note": "Conservative eviction only for single-query retrieval-style workloads (it fails on multi-hop / long CoT)."},
    {"id": "T-LONG-TRAIN", "node": None, "table": "Table 5.1 row 7", "title": "MLA conversion / local-global / hybrid linear / trainable sparse attention", "forms": ["dense", "moe"], "order": 16,
     "when": "binding == 'kv'", "claims": ["P3-ARCH-01", "P3-SPARSE-01", "P6-IDX-01", "P6-PARITY-01", "P3-EVAL-01"], "training": True,
     "experimental": True, "requires": [], "paper_refs": ["P3 F2-F3", "P6"],
     "proposals": [{"kind": "long_arch", "arch": "mla_convert", "gate": "g-long", "experimental": True}, {"kind": "long_arch", "arch": "local_global", "gate": "g-long", "experimental": True},
                   {"kind": "long_arch", "arch": "sparse_shared_indexer", "gate": "g-long", "experimental": True}],
     "note": "Each against a same-token-budget dense baseline and multi-hop tests; selector (indexer) bytes must be reported explicitly (P6)."},
    {"id": "T-SPEC-LOW", "node": None, "table": "Table 5.1 row 8", "title": "Speculative decoding at low concurrency", "forms": ["dense", "moe", "fleet"], "order": 17,
     "when": "max_batch <= 8", "claims": ["P7-EXACT-01", "P7-EN-01", "P7-EAGLE-01", "P7-NGRAM-01"], "training": False, "experimental": False, "requires": [], "paper_refs": ["P7 s19.3"],
     "proposals": [{"kind": "speculation", "mode": "ngram", "gate": "g-spec", "when": "context_copying"}, {"kind": "speculation", "mode": "auto", "gate": "g-spec"}],
     "note": "Measure acceptance per position, tune gamma; n-gram lookup first for context-copying workloads (code editing, structured agents)."},
    {"id": "T-SPEC-HIGH", "node": None, "table": "Table 5.1 row 9", "title": "High-concurrency serving: speculation off first, adaptive including 0", "forms": ["dense", "moe", "fleet"], "order": 18,
     "when": "max_batch > 8", "claims": ["P7-BATCH-01", "P7-ENGINE-01"], "training": False, "experimental": False, "requires": [], "paper_refs": ["P7 s5.8"],
     "proposals": [{"kind": "speculation", "mode": "adaptive0", "gate": "g-spec"}],
     "note": "Benchmark with speculation off first; the load-aware controller must be allowed to choose gamma = 0."},
    {"id": "T-TRAIN-PRECISION", "node": None, "table": "Table 5.1 row 10", "title": "Training precision: FP8 default, NVFP4 gated", "forms": ["dense", "moe"], "order": 19,
     "when": "purpose == 'training'", "claims": ["P1-TRAIN-01", "P5-LOSS-01", "P5-ZGT-01", "P5-NOISE-01", "P5-RHT-01"], "training": True, "experimental": False, "requires": [],
     "paper_refs": ["P1 F7", "P5"],
     "proposals": [{"kind": "train_fp8", "gate": None, "when": "fp8"}, {"kind": "train_nvfp4", "gate": "g-fp4train", "experimental": True, "when": "fp4"}],
     "note": "FP8 default; NVFP4 only after the TR-2 diagnostics pass, with BF16 exemptions (last layers) and a high-precision tail phase."},
    {"id": "T-LOWRANK", "node": None, "table": "Table 5.1 row 11", "title": "Low-rank on a frozen model (whitened SVD + LoRA healing)", "forms": ["dense"], "order": 20,
     "when": "", "claims": ["P4-LR-01", "P4-LR-02"], "training": False, "experimental": True, "requires": [], "paper_refs": ["P4 s1.3"],
     "proposals": [{"kind": "lowrank", "gate": "g-struct", "experimental": True}],
     "note": "Whiten with calibration activations, allocate ranks heterogeneously, add short LoRA healing; measure wall-clock, not parameter count (rank must beat mn/(m+n))."},
]

RULES_BY_ID = {r["id"]: r for r in RULES}
FLOW_BY_ID = {n["id"]: n for n in FLOW}


def rule_ids_for_claim(claim_id: str) -> list[str]:
    """Rules (and flow nodes) that rely on ``claim_id``."""
    return sorted({r["id"] for r in RULES if claim_id in r["claims"]})


def all_claim_ids() -> set[str]:
    return {c for r in RULES for c in r["claims"]} | {c for n in FLOW for c in n["claims"]}


# --------------------------------------------------------------------------------------------
# budget / candidate models
# --------------------------------------------------------------------------------------------
class Budget(BaseModel):
    """What the planner may spend and how much accuracy it may give up (Plan 5.3 ``budget``)."""

    model_config = ConfigDict(extra="forbid")

    allow_training: bool = Field(default=False, description="Training / distillation compute is available (enables prune+distill, QAT, architecture changes).")
    allow_experimental: bool = Field(default=False, description="Include gated / experimental tracks (4-bit KV, FP4 training, sub-3.5-bit, trainable sparse attention).")
    allow_custom_kernels: bool = Field(default=True, description="Custom kernels may be written/used (activation sparsity, fused FP4).")
    gpu_hours: float | None = Field(default=None, ge=0, description="GPU-hour budget for the search (None = unlimited).")
    distill_tokens_b: float = Field(default=20.0, ge=0, description="Distillation tokens (billions) available for pruned students.")
    max_quality_loss_pct: float = Field(default=4.0, ge=0, le=100, description="Largest acceptable loss of suite score vs the 16-bit parent, in percent.")
    max_bits_per_weight: float | None = Field(default=None, gt=0, description="Hard cap on effective bits per weight (e.g. 3.0 to force sub-4-bit methods).")
    optimise_storage: bool = Field(default=False, description="Also propose lossless entropy coding for cold storage.")
    many_finetunes: bool = Field(default=False, description="Many fine-tunes share one base (enables delta storage).")
    purpose: Literal["serving", "training"] = Field(default="serving", description="Plan for serving a model or for choosing a training precision.")
    context_copying: bool = Field(default=False, description="Workload copies from its context (code editing, structured agents): n-gram speculation first.")
    retrieval_style: bool = Field(default=False, description="Workload is single-query retrieval/summarisation (conservative eviction is acceptable).")
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2], description="Pre-registered seeds for every evaluation rung.")
    max_candidates: int = Field(default=16, ge=1, le=200, description="Cap on the number of candidates kept after the rulebase.")
    gates: dict[str, str] = Field(default_factory=dict, description="Override gate ids by rung name or scheme family.")

    @property
    def tight_accuracy(self) -> bool:
        return self.max_quality_loss_pct <= 2.0


class Candidate(BaseModel):
    """A recipe proposal with provenance (which rules and claims it rests on)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str = Field(description="Stable id: <kind>-<short recipe hash>.")
    name: str = Field(description="Human-readable name.")
    kind: str = Field(description="Proposal kind (fp8, w4a16, nvfp4, kv, prune_distill, ...).")
    scheme: Scheme = Field(description="Precision scheme used by the cost model.")
    recipe: Recipe = Field(description="Executable recipe (stage list).")
    rule_ids: list[str] = Field(default_factory=list, description="Rules that proposed it.")
    claim_ids: list[str] = Field(default_factory=list, description="Claims the proposal depends on.")
    gate: str | None = Field(default=None, description="Gate id that must pass before promotion.")
    requires_training: bool = Field(default=False, description="Needs training compute.")
    experimental: bool = Field(default=False, description="Gated / experimental track.")
    arm: Literal["rule", "baseline"] = Field(default="rule", description="rule: proposed by a rule; baseline: comparison arm (Plan 5.3 baseline_arms).")
    footprint_ratio: float = Field(default=1.0, gt=0, description="Parameter/weight-footprint ratio relative to the parent from structural changes (pruning).")
    serving: dict[str, Any] = Field(default_factory=dict, description="Serving-level policy (speculation, prefix caching ...).")
    notes: list[str] = Field(default_factory=list, description="Explanations and caveats.")


# --------------------------------------------------------------------------------------------
# recipe construction
# --------------------------------------------------------------------------------------------
def _calib(n: int = 512, seq: int = 2048) -> StageSpec:
    return StageSpec(name="calibrate", params={"set": "default", "n_samples": n, "seq_len": seq, "stats": ["hessian", "channel_scale"]})


def _eval(gate: str | None, suites: list[str] | None = None) -> list[StageSpec]:
    out = [StageSpec(name="eval", params={"suites": suites or ["distribution", "mcq", "longctx", "longgen", "safety"], "noise_repeats": 5})]
    if gate:
        out.append(StageSpec(name="gate", params={"gate": gate}))
    return out


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-")


def build_candidate(model: ModelSpec, kind: str, prop: dict[str, Any], rule: dict[str, Any], budget: Budget, ctx: dict[str, Any]) -> Candidate | None:
    """Turn one proposal descriptor into a Candidate (None if it cannot apply)."""
    gate = budget.gates.get(kind) or prop.get("gate")
    training = bool(prop.get("training", rule.get("training", False)))
    experimental = bool(prop.get("experimental", rule.get("experimental", False)))
    inputs = {"model": "$parent"}
    stages: list[StageSpec] = []
    notes: list[str] = []
    serving: dict[str, Any] = {}
    footprint = 1.0
    kv = prop.get("kv")
    kvs = {"fp8": "+kvfp8", "int4": "+kv4", "bf16": ""}.get(kv or "bf16", "")

    def kv_stage() -> list[StageSpec]:
        if kv in ("fp8", "int4"):
            return [StageSpec(name="kv", params={"dtype": "fp8_e4m3" if kv == "fp8" else "int4", "method": "per_channel_key_per_token_value", "residual_window": 128})]
        return []

    if kind == "bf16":
        scheme, name = parse_scheme("bf16"), "BF16 (parent)"
    elif kind == "fp8":
        scheme = parse_scheme("fp8" + kvs)
        name = "FP8 W8A8" + (" + FP8 KV" if kv == "fp8" else "")
        stages = [_calib(256), StageSpec(name="ptq", params={"method": "rtn", "weight_format": "fp8_e4m3", "act_format": "fp8_e4m3", "granularity": "per_channel"}),
                  StageSpec(name="pack", params={"format": "fp8", "layout": "auto"})] + kv_stage()
    elif kind == "w4a16":
        g, meth = int(prop.get("group", 128)), prop.get("method", "gptq")
        scheme = parse_scheme(f"w4a16-g{g}").model_copy(update={"method": meth})
        name = f"W4A16 g{g} ({meth.upper()})"
        stages = [_calib(), StageSpec(name="ptq", params={"method": meth, "bits": 4, "group_size": g, "sym": True, "act_order": meth == "gptq"}),
                  StageSpec(name="pack", params={"format": "w4a16", "layout": "auto"})]
    elif kind == "w4a8":
        scheme = parse_scheme("w4a8")
        name = "W4A8 (FP8 activations)"
        stages = [_calib(), StageSpec(name="ptq", params={"method": "gptq", "bits": 4, "group_size": 128, "act_format": "fp8_e4m3"}), StageSpec(name="pack", params={"format": "w4a8", "layout": "auto"})]
    elif kind in ("nvfp4", "mxfp4"):
        scheme = parse_scheme(kind)
        name = f"{kind.upper()} W4A4 (MR-GPTQ)"
        stages = [_calib(), StageSpec(name="ptq", params={"method": "mr_gptq", "weight_format": kind, "act_format": kind, "rotation": "hadamard"}),
                  StageSpec(name="pack", params={"format": kind, "layout": "auto"})]
        if not ctx.get("fp4"):
            notes.append("no native FP4 tensor cores on this chip: fake-quant numerics study only, no speed-up claimed")
            experimental = True
    elif kind == "kv":
        base = parse_scheme(ctx.get("base_scheme", "bf16"))
        scheme = base.with_kv(8.0 if kv == "fp8" else 4.25, f"+kv{'fp8' if kv == 'fp8' else '4'}")
        name = ("FP8" if kv == "fp8" else "4-bit") + " KV cache" + ("" if base.name == "bf16" else f" on {base.name}")
        stages = kv_stage()
        if kv == "int4":
            notes.append("4-bit KV must be validated at the full deployment length (multi-hop, aggregation) before promotion (P1 F8)")
    elif kind in ("sub35", "sub35_qat"):
        scheme = parse_scheme("w3a16-g128").model_copy(update={"experimental": True})
        scheme = scheme.model_copy(update={"weight_bits": 2.5, "weight_scale_bits": 0, "weight_group": None, "name": "sub35-vq", "family": "sub4", "method": "vq"})
        name = "Vector/trellis PTQ (~2.5 bit)" + (" + QAT" if kind == "sub35_qat" else "")
        stages = [_calib(1024), StageSpec(name="ptq", params={"method": "vq", "bits": 2.5, "codebook": "trellis"})]
        if kind == "sub35_qat":
            stages.append(StageSpec(name="distill", params={"teacher": "$parent", "mode": "logit", "tokens_b": budget.distill_tokens_b}))
            training = True
    elif kind == "prune_distill":
        keep = float(prop.get("keep", 0.75))
        footprint = keep
        scheme = parse_scheme("bf16")
        name = f"Prune width to {keep:.0%} + distill {budget.distill_tokens_b:g}B tokens"
        stages = [_calib(512), StageSpec(name="prune", params={"target": "width", "keep_ratio": keep, "gqa_aware": True, "importance": "activation"}),
                  StageSpec(name="distill", params={"teacher": "$parent", "mode": "logit", "kl": "forward", "tokens_b": budget.distill_tokens_b, "topk_cache": 64})]
        training = True
        inputs["teacher"] = "$parent"
        if model.distill_allowed is False:
            notes.append("LICENCE: the parent's licence does not permit distillation (LicenceBlocked at run time)")
        elif model.distill_allowed is None:
            notes.append("licence terms for distillation unknown - the licence gate must pass before this runs")
    elif kind == "expert_prune":
        ratio = float(prop.get("ratio", 0.25))
        footprint = 1.0 - ratio * (0.9 if model.form == "moe" else 0.0)
        scheme = parse_scheme("bf16")
        name = f"Prune {ratio:.0%} of experts (router-aware)"
        stages = [_calib(1024), StageSpec(name="prune", params={"target": "expert", "ratio": ratio, "criterion": "router_weighted_activation", "merge": True})]
    elif kind == "act_sparsity":
        scheme = parse_scheme(ctx.get("base_scheme", "w4a16-g128")).model_copy(update={"activation_sparsity": float(prop.get("sparsity", 0.4))})
        scheme = scheme.model_copy(update={"name": scheme.name + f"+sp{int(scheme.activation_sparsity * 100)}", "experimental": True})
        name = f"Activation sparsity {scheme.activation_sparsity:.0%} on {ctx.get('base_scheme', 'w4a16-g128')}"
        stages = [_calib(256), StageSpec(name="ptq", params={"method": "activation_sparsity", "sparsity": scheme.activation_sparsity, "calibrated_thresholds": True})]
        notes.append("needs custom sparse-activation kernels")
    elif kind == "delta":
        scheme = parse_scheme(ctx.get("base_scheme", "bf16"))
        name = "Delta storage (" + prop.get("delta", "bitdelta") + ")"
        stages = [StageSpec(name="delta", params={"kind": prop.get("delta", "bitdelta"), "base": "$parent"})]
    elif kind == "lossless":
        scheme = parse_scheme("bf16")
        scheme = scheme.model_copy(update={"lossless_ratio": 0.7, "name": "bf16+lossless"})
        name = "Lossless entropy coding"
        stages = [StageSpec(name="lossless", params={"codec": "exponent_split_zstd", "verify_roundtrip": True})]
        notes.append("storage-only transform: HBM bytes unchanged unless decoded in a fused kernel")
    elif kind == "serving_lossless":
        scheme = parse_scheme(ctx.get("base_scheme", "bf16"))
        name = "Paging + prefix caching + chunked prefill"
        serving = {"paged_kv": True, "prefix_caching": True, "chunked_prefill": True}
    elif kind == "kv_evict":
        scheme = parse_scheme(ctx.get("base_scheme", "bf16"))
        name = "Conservative KV eviction (retrieval-style workloads only)"
        serving = {"kv_eviction": "conservative"}
        notes.append("fails on multi-hop / long CoT (P3 F4): G-LONG must pass")
    elif kind == "long_arch":
        arch = prop.get("arch", "mla_convert")
        scheme = parse_scheme("bf16")
        name = {"mla_convert": "Convert GQA -> MLA (latent KV) + healing", "local_global": "Local/global attention pattern + healing",
                "sparse_shared_indexer": "Trainable sparse attention with shared indexers"}.get(arch, arch)
        stages = [StageSpec(name="finetune", params={"mode": "architecture_change", "arch": arch, "tokens_b": budget.distill_tokens_b})]
        notes.append("compare against a same-token-budget dense baseline and multi-hop/aggregation tests")
        training = True
    elif kind == "speculation":
        mode = prop.get("mode", "auto")
        scheme = parse_scheme(ctx.get("base_scheme", "bf16"))
        name = {"ngram": "N-gram speculation", "auto": "Speculation (auto: ngram/draft)", "adaptive0": "Load-aware speculation (gamma may be 0)"}.get(mode, mode)
        serving = {"speculation": {"mode": "ngram" if mode == "ngram" else "auto", "max_gamma": 5, "allow_zero": True}}
    elif kind in ("train_fp8", "train_nvfp4"):
        scheme = parse_scheme("fp8" if kind == "train_fp8" else "nvfp4")
        name = "FP8 mixed-precision training (R1)" if kind == "train_fp8" else "NVFP4 training (R2, gated)"
        stages = [StageSpec(name="finetune", params={"precision": "R1" if kind == "train_fp8" else "R2", "bf16_exempt_last_layers": 0.15, "high_precision_tail": kind == "train_nvfp4"})]
        training = True
    elif kind == "lowrank":
        scheme = parse_scheme("bf16")
        name = "Whitened-SVD low-rank + LoRA healing"
        stages = [_calib(512), StageSpec(name="lowrank", params={"method": "whitened_svd", "rank_allocation": "heterogeneous", "break_even_guard": True}),
                  StageSpec(name="finetune", params={"mode": "lora", "steps": 500})]
    else:
        return None

    stages = stages + _eval(gate)
    recipe = Recipe(name=_slug(f"{kind}-{name}")[:60] or kind, inputs=inputs, stages=stages, baselines=["bf16-parent", "fresh-student"] if kind == "prune_distill" else ["bf16-parent"],
                    budget={"gpu_hours": budget.gpu_hours, "max_quality_loss_pct": budget.max_quality_loss_pct},
                    policy={"licence_check": True, "gate": gate, "serving": serving} if serving else {"licence_check": True, "gate": gate}, seeds=list(budget.seeds))
    cid = f"{kind}-{recipe.hash().split(':')[-1][:8]}"
    return Candidate(id=cid, name=name, kind=kind, scheme=scheme, recipe=recipe, rule_ids=[rule["id"]], claim_ids=sorted(rule["claims"]), gate=gate, requires_training=training,
                     experimental=experimental, footprint_ratio=footprint, serving=serving, notes=notes + ([rule["note"]] if rule.get("note") else []))


# --------------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------------
def planning_context(model: ModelSpec, binding: str, chip: ChipSpec, budget: Budget, workload: Workload | None = None, *, bf16_fits: bool = True,
                     kv_dominates: bool = False) -> dict[str, Any]:
    wl = workload or Workload()
    return {
        "form": model.form, "binding": binding, "fp8": chip.supports("fp8"), "fp4": chip.supports("fp4"), "two_four": chip.two_four_sparse,
        "bf16_fits": bf16_fits, "kv_dominates": kv_dominates, "params_b": model.params_b, "allow_training": budget.allow_training,
        "allow_experimental": budget.allow_experimental, "allow_custom_kernels": budget.allow_custom_kernels, "max_batch": max(wl.batch_sizes or [1]),
        "max_bits_per_weight": budget.max_bits_per_weight, "optimise_storage": budget.optimise_storage, "many_finetunes": budget.many_finetunes or model.form == "fleet",
        "purpose": budget.purpose, "context_copying": budget.context_copying, "retrieval_style": budget.retrieval_style, "tight_accuracy": budget.tight_accuracy,
    }


def _truthy(expr: str | None, ctx: dict[str, Any]) -> bool:
    """Evaluate a rule condition; names missing from the context count as False and any error as False."""
    if not expr:
        return True
    full = dict(ctx)
    try:
        for n in _names_in(expr):
            full.setdefault(n, False)
        return bool(_eval_with_none(expr, full))
    except Exception:
        return False


def _names_in(expr: str) -> set[str]:
    import ast

    return {n.id for n in ast.walk(ast.parse(expr, mode="eval")) if isinstance(n, ast.Name) and n.id not in ("True", "False", "None")}


def _eval_with_none(expr: str, ctx: dict[str, Any]) -> Any:
    import ast
    import operator

    cmp = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b, ast.Is: operator.is_, ast.IsNot: operator.is_not}

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in ("True", "False", "None"):
                return {"True": True, "False": False, "None": None}[n.id]
            return ctx[n.id]
        if isinstance(n, ast.Tuple):
            return tuple(ev(e) for e in n.elts)
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, comp in zip(n.ops, n.comparators):
                right = ev(comp)
                if type(op) in (ast.Lt, ast.LtE, ast.Gt, ast.GtE) and (left is None or right is None):
                    return False
                if not cmp[type(op)](left, right):
                    return False
                left = right
            return True
        raise ValueError(f"unsupported syntax in rule condition {expr!r}")

    return ev(ast.parse(expr, mode="eval"))


def propose_with_trace(
    model: ModelSpec, binding: str, chip: ChipSpec, budget: Budget | None = None, workload: Workload | None = None, *, bf16_fits: bool = True,
    kv_dominates: bool = False,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Evaluate every rule; returns ``(candidates, trace)`` where trace explains each rule (fired / skipped and why)."""
    budget = budget or Budget()
    ctx = planning_context(model, binding, chip, budget, workload, bf16_fits=bf16_fits, kv_dominates=kv_dominates)
    cands: list[Candidate] = []
    trace: list[dict[str, Any]] = []
    seen: dict[str, Candidate] = {}
    # candidates that later rules build on (activation sparsity / speculation / KV start from the best weight scheme)
    for rule in sorted(RULES, key=lambda r: r["order"]):
        rec: dict[str, Any] = {"rule": rule["id"], "title": rule["title"], "table": rule["table"], "claims": rule["claims"], "fired": False, "reason": ""}
        trace.append(rec)
        if model.form not in rule["forms"]:
            rec["reason"] = f"rule is for forms {rule['forms']}, model is {model.form}"
            continue
        missing = [c for c in rule["requires"] if not ctx.get(c)]
        if missing:
            rec["reason"] = f"chip lacks required capability: {', '.join(missing)}"
            continue
        if rule.get("training") and not budget.allow_training:
            rec["reason"] = "needs training budget (allow_training = false)"
            continue
        if rule.get("experimental") and not budget.allow_experimental and rule["id"] not in ("R-Q6-SUB35",):
            rec["reason"] = "experimental / gated track (allow_experimental = false)"
            continue
        if rule["id"] == "R-Q6-SUB35" and not budget.allow_experimental:
            rec["reason"] = "sub-3.5-bit methods are Tier 3 / gated (allow_experimental = false)"
            continue
        if not _truthy(rule["when"], ctx):
            rec["reason"] = f"condition not met: {rule['when']}"
            continue
        # base scheme for rules that stack on a weight scheme
        base_cand = _best_weight_candidate(cands)
        stack_base = base_cand.scheme.name.split("+")[0] if base_cand else "bf16"
        rctx = dict(ctx, base_scheme=stack_base)
        n_before = len(cands)
        for prop in rule["proposals"]:
            if not _truthy(prop.get("when"), ctx):
                continue
            if prop.get("experimental") and not budget.allow_experimental:
                rec.setdefault("skipped_proposals", []).append(f"{prop['kind']}: experimental")
                continue
            if prop.get("training") and not budget.allow_training:
                rec.setdefault("skipped_proposals", []).append(f"{prop['kind']}: needs training")
                continue
            c = build_candidate(model, prop["kind"], prop, rule, budget, rctx)
            if c is None:
                continue
            if c.kind in STACKING_KINDS and base_cand is not None:
                c = stack_on(base_cand, c)
            if c.id in seen:
                seen[c.id].rule_ids = sorted(set(seen[c.id].rule_ids) | set(c.rule_ids))
                seen[c.id].claim_ids = sorted(set(seen[c.id].claim_ids) | set(c.claim_ids))
                continue
            seen[c.id] = c
            cands.append(c)
        rec["fired"] = len(cands) > n_before
        rec["reason"] = "fired" if rec["fired"] else "all proposals filtered"
        rec["n_candidates"] = len(cands) - n_before
    return cands[: budget.max_candidates], trace


def propose(model: ModelSpec, binding: str, chip: ChipSpec, budget: Budget | None = None, workload: Workload | None = None, **kw: Any) -> list[Candidate]:
    """Candidate recipes (``Candidate.recipe`` is a :class:`Recipe`) for (model form, binding resource, chip capabilities, budget)."""
    return propose_with_trace(model, binding, chip, budget, workload, **kw)[0]


STACKING_KINDS = ("kv", "act_sparsity", "delta", "serving_lossless", "kv_evict", "speculation")


def _best_weight_candidate(cands: list[Candidate]) -> Candidate | None:
    """The first-try weight-quantisation candidate that serving-level rules (KV, speculation, activation sparsity ...) build on."""
    for kind in ("w4a16", "fp8", "w4a8", "nvfp4", "mxfp4"):
        for c in cands:
            if c.kind == kind and c.scheme.kv_bits == 16.0 and not c.scheme.activation_sparsity:
                return c
    return None


def stack_on(base: Candidate, c: Candidate) -> Candidate:
    """Make ``c`` (a KV / speculation / sparsity / serving candidate) run on top of the quantised ``base``: its recipe first produces the
    base artifact (the base's stages), then applies ``c``'s own stages and evaluation."""
    pre = [s for s in base.recipe.stages if s.name not in ("eval", "gate")]
    own = [s for s in c.recipe.stages if s.name not in ("eval", "gate")]
    tail = [s for s in c.recipe.stages if s.name in ("eval", "gate")]
    recipe = c.recipe.model_copy(update={"stages": pre + own + tail, "name": (c.recipe.name + "-on-" + base.recipe.name)[:80]})
    name = f"{c.name} on {base.name}" if base.kind != "bf16" else c.name
    cid = f"{c.kind}-{recipe.hash().split(':')[-1][:8]}"
    return c.model_copy(update={
        "recipe": recipe, "id": cid, "name": name, "rule_ids": sorted(set(c.rule_ids) | set(base.rule_ids)), "claim_ids": sorted(set(c.claim_ids) | set(base.claim_ids)),
        "requires_training": c.requires_training or base.requires_training, "experimental": c.experimental or base.experimental,
        "notes": c.notes + [f"stacked on {base.id} ({base.name})"],
    })


def describe_rulebase() -> dict[str, Any]:
    """JSON-serialisable rulebase for the UI (flow nodes, rules, and the claim ids each depends on)."""
    return {"flow": FLOW, "footer": FLOW_FOOTER, "rules": [{k: v for k, v in r.items()} for r in RULES]}
