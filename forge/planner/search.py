"""Plan 5.3: ``plan(model, chip, workload, budget) -> ranked recipes``.

Steps (mirroring the Plan's pseudocode):

1. ``binding_resource`` - which resource binds: capacity | bandwidth | compute | kv | accuracy.
2. ``rulebase.propose`` + ``baseline_arms`` - candidate recipes, including the bf16 parent, naive RTN arms and a *smaller dense model
   of equal footprint* ("fresh student", P4 F8).
3. ``fits`` - byte accounting (Example A): weights + KV at the workload + engine reserve must fit in HBM.
4. cost-model prediction (bytes first) -> ``prune_dominated`` on (bytes, latency, quality proxy, risk).
5. successive halving over rungs PROXY_KL -> SUITE_SHORT -> SUITE_LONGCTX_REASONING_SAFETY -> END_TO_END_SERVING with a pluggable
   :class:`Runner`.  The default :class:`CostModelRunner` is a **dry mode**: it uses the cost model and *quality priors* (gate
   targets and paper-reported ranges, never measurements) so the UI can show ranked candidates and the Pareto front without GPUs.
   :class:`RecipeEvalRunner` runs the recipes and the evaluation harness for real (GPU hosts).
6. ``keep_top`` is **noise-aware**: candidates whose scores are statistically indistinguishable from the k-th are kept together
   (cost breaks the tie); the planner refuses to rank within noise.
7. ``pareto_front`` over (bytes, latency, throughput, quality, risk).

Everything that comes from the cost model is tagged ``derived``; quality numbers in dry mode are tagged ``target``/``untested``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from forge.core.schemas import Provenance, Recipe, StageSpec, Workload
from forge.core.util import now
from forge.planner import costmodel as cm
from forge.planner.rulebase import Budget, Candidate, build_candidate, propose_with_trace
from forge.planner.spec import ChipSpec, ModelSpec, Scheme, parse_scheme, resolve_chip

log = logging.getLogger("forge.planner")

BindingName = Literal["capacity", "bandwidth", "compute", "kv", "accuracy"]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), arbitrary_types_allowed=True)


# --------------------------------------------------------------------------------------------
# byte accounting
# --------------------------------------------------------------------------------------------
class FitReport(_M):
    """Example A as a function: does (weights + KV at the workload + reserve) fit in HBM?"""

    fits: bool = Field(description="True if weights and the KV cache of the workload fit with the reserve.")
    hbm_gb: float = Field(description="Device memory (GB, 1e9 bytes).")
    weights_gb: float = Field(description="Resident weight bytes in GB.")
    reserve_gb: float = Field(description="Engine overhead assumed (CUDA context, activations, fragmentation).")
    free_for_kv_gb: float = Field(description="HBM left for the KV cache after weights and reserve (negative = does not fit).")
    kv_bytes_per_token: float = Field(description="KV bytes per token per sequence at the scheme's KV precision.")
    kv_gb_at_workload: float = Field(description="KV cache needed for batch x context of the workload (prefix reuse deducted).")
    batch: int = Field(description="Concurrent sequences planned for.")
    ctx: int = Field(description="Context length planned for.")
    max_concurrency_at_ctx: float = Field(description="Sequences of length ctx that fit in the free memory.")
    max_tokens_total: float = Field(description="Total cached tokens that fit in the free memory.")
    notes: list[str] = Field(default_factory=list, description="Explanations.")
    provenance: Provenance = Field(default=Provenance.derived, description="derived: cost-model byte accounting.")


def workload_shape(workload: Workload | None) -> tuple[int, int]:
    wl = workload or Workload()
    return max(wl.batch_sizes or [1]), max(wl.context_lengths or [4096])


def fits(
    model: ModelSpec, chip: ChipSpec, scheme: Scheme | str, workload: Workload | None = None, *, batch: int | None = None, ctx: int | None = None,
    overhead_gb: float | None = None, reserve_frac: float = 0.05, footprint_ratio: float = 1.0, head_bits: float = 16.0,
) -> FitReport:
    """Byte-accounting fit test.  Default reserve = 1 GB + 5% of HBM (an assumption standing in for CUDA context, activations and
    fragmentation: 'several GB in practice' per the Plan; replaced by measured engine overheads when available).
    ``overhead_gb=0, reserve_frac=0`` reproduces Example A exactly."""
    sch = parse_scheme(scheme) if not isinstance(scheme, Scheme) else scheme
    sch, notes = cm.effective_scheme(sch, chip)
    wl = workload or Workload()
    B, T = workload_shape(wl)
    B = batch if batch is not None else B
    T = ctx if ctx is not None else T
    wb = cm.model_weight_bytes(model, sch, head_bits=head_bits)["total"] * footprint_ratio
    reserve = (1.0 if overhead_gb is None else overhead_gb) + reserve_frac * chip.hbm_gb
    per_seq = model.kv_bytes_for_context(T, sch.kv_bits)
    reuse = min(max(wl.prefix_reuse, 0.0), 1.0)
    kv_total = per_seq * (B * (1.0 - reuse) + reuse)  # shared prefix cached once
    free = chip.hbm_bytes - wb - reserve * 1e9
    per_tok = model.kv_bytes_per_token(sch.kv_bits)
    max_conc = max(0.0, free / per_seq) if per_seq > 0 else math.inf
    max_tok = max(0.0, free / per_tok) if per_tok > 0 else math.inf
    ok = free > 0 and kv_total <= free
    if wb > chip.hbm_bytes:
        notes.append(f"weights ({wb / 1e9:.1f} GB) alone exceed device memory ({chip.hbm_gb:g} GB)")
    elif free <= 0:
        notes.append("weights plus reserve exceed device memory")
    elif kv_total > free:
        notes.append(f"KV for {B} x {T} tokens ({kv_total / 1e9:.1f} GB) exceeds the {free / 1e9:.1f} GB left")
    return FitReport(
        fits=bool(ok), hbm_gb=chip.hbm_gb, weights_gb=wb / 1e9, reserve_gb=reserve, free_for_kv_gb=free / 1e9, kv_bytes_per_token=per_tok, kv_gb_at_workload=kv_total / 1e9,
        batch=int(B), ctx=int(T), max_concurrency_at_ctx=max_conc, max_tokens_total=max_tok, notes=notes,
    )


class BindingReport(_M):
    binding: BindingName = Field(description="The resource that binds first for the bf16 parent on this chip and workload.")
    reasons: list[str] = Field(default_factory=list, description="Why.")
    scores: dict[str, float] = Field(default_factory=dict, description="Pressure per resource (>1 = binding).")
    bf16_fit: FitReport = Field(description="Fit report of the bf16 parent.")
    b_star_bf16: float | None = Field(default=None, description="Batch crossover of bf16 weights at the achievable FLOP efficiency.")
    kv_dominates: bool = Field(default=False, description="KV cache is larger than the weights at the workload.")


def binding_resource(model: ModelSpec, chip: ChipSpec, workload: Workload | None = None, budget: Budget | None = None, *, eta: cm.EtaFactors | None = None) -> BindingReport:
    """Which resource binds first (Plan 5.3 step 1)?

    * ``capacity``  - the bf16 weights (plus reserve) do not fit, and KV is not the bigger problem;
    * ``kv``        - the weights fit/are small but the KV cache of the workload is larger than the weights or does not fit;
    * ``bandwidth`` - everything fits and the workload batch is below the (efficiency-adjusted) batch crossover ``B*``;
    * ``compute``   - batch >= ``B*`` (prefill / large-batch regime);
    * ``accuracy``  - everything fits and the accuracy budget is under 0.5% - there is nothing to trade."""
    eta = eta or cm.EtaFactors()
    bf16 = parse_scheme("bf16")
    B, T = workload_shape(workload)
    fit = fits(model, chip, bf16, workload)
    chip_bf16 = chip.supports("bf16")
    peak = chip.peak_flops("bf16" if chip_bf16 else "fp16") or 1e12
    bstar = cm.batch_crossover(peak, chip.bw, 16.0, eta=eta.eta_mma / max(eta.eta_mem, 1e-9))
    w_gb = fit.weights_gb
    kv_gb = fit.kv_gb_at_workload
    kv_dom = kv_gb > w_gb
    scores = {
        "capacity": (w_gb + fit.reserve_gb) / chip.hbm_gb, "kv": (w_gb + fit.reserve_gb + kv_gb) / chip.hbm_gb, "bandwidth": 1.0 - min(B / bstar, 1.0) if bstar else 0.0,
        "compute": B / bstar if bstar else 0.0,
    }
    reasons: list[str] = []
    if not fit.fits:
        if w_gb + fit.reserve_gb > chip.hbm_gb or not kv_dom:
            binding: BindingName = "capacity"
            reasons.append(f"bf16 weights {w_gb:.1f} GB + reserve do not leave room in {chip.hbm_gb:g} GB" if w_gb + fit.reserve_gb > chip.hbm_gb else f"bf16 weights + KV ({w_gb + kv_gb:.1f} GB) exceed {chip.hbm_gb:g} GB")
        else:
            binding = "kv"
            reasons.append(f"KV cache for {B} x {T} tokens ({kv_gb:.1f} GB) exceeds the weights ({w_gb:.1f} GB) and the free memory")
    elif kv_dom:
        binding = "kv"
        reasons.append(f"KV cache ({kv_gb:.1f} GB) is larger than the weights ({w_gb:.1f} GB): context/concurrency are the lever")
    elif budget is not None and budget.max_quality_loss_pct <= 0.5:
        binding = "accuracy"
        reasons.append(f"accuracy budget {budget.max_quality_loss_pct}% leaves no room to trade quality for bytes")
    elif B >= bstar:
        binding = "compute"
        reasons.append(f"batch {B} >= efficiency-adjusted crossover B* = {bstar:.0f}: tensor-core bound")
    else:
        binding = "bandwidth"
        reasons.append(f"everything fits and batch {B} < B* = {bstar:.0f}: decode is bandwidth-bound (bytes are the currency)")
    return BindingReport(binding=binding, reasons=reasons, scores={k: round(v, 4) for k, v in scores.items()}, bf16_fit=fit, b_star_bf16=bstar, kv_dominates=kv_dom)


# --------------------------------------------------------------------------------------------
# quality priors (dry mode) - TARGETS and paper-reported ranges, never measurements
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QualityPrior:
    ratio: float  # expected suite-score retention vs the parent (1.0 = lossless)
    unc: float  # +- uncertainty (half-width)
    provenance: Provenance
    source: str
    longctx_penalty: float = 0.0  # extra expected loss on long-context / multi-hop suites


_STACKING = ("kv", "act_sparsity", "delta", "serving_lossless", "kv_evict", "speculation")


def _combine(a: QualityPrior, b: QualityPrior, source: str) -> QualityPrior:
    prov = a.provenance if (a.provenance == Provenance.untested or b.provenance != Provenance.untested) else b.provenance
    if Provenance.untested in (a.provenance, b.provenance):
        prov = Provenance.untested
    elif Provenance.paper_reported in (a.provenance, b.provenance):
        prov = Provenance.paper_reported
    elif Provenance.target in (a.provenance, b.provenance):
        prov = Provenance.target
    return QualityPrior(a.ratio * b.ratio, math.hypot(a.unc, b.unc), prov, source, a.longctx_penalty + b.longctx_penalty)


def _weight_prior(s: Scheme, model: ModelSpec) -> QualityPrior:
    """Prior for the weight/activation scheme alone (family based)."""
    fam = s.family
    small, big = model.params_b < 10, model.params_b >= 30
    if fam == "fp8":
        return QualityPrior(0.995, 0.005, Provenance.target, "Plan 2.3: FP8 retains >= 99.5%")
    if fam == "w4a16":
        r = 0.98 if big else 0.96 if small else 0.97
        if s.method == "awq" or (s.weight_group or 128) <= 64:
            r += 0.002
        return QualityPrior(r, 0.02, Provenance.target, "Plan 2.3 / G-W4: >= 98% (>=30B), 96% (<10B)")
    if fam == "w4a8":
        return QualityPrior(0.96 if not small else 0.94, 0.03, Provenance.untested, "no direct source; between W4A16 and W4A4")
    if fam in ("nvfp4", "mxfp4"):
        r = (0.975 if big else 0.95) - (0.01 if fam == "mxfp4" else 0.0)
        return QualityPrior(r, 0.025, Provenance.paper_reported, "P1 F6 / P5: ~97-99% for 30B NVFP4; smaller models lose more; MXFP4 ~1.5 dB worse")
    if fam == "int8":
        return QualityPrior(0.995, 0.005, Provenance.paper_reported, "P1 F2: 8-bit is essentially lossless")
    if fam == "sub4":
        return QualityPrior(0.88, 0.06, Provenance.untested, "P1 F5: usable at 2 bits for large models only")
    if fam == "t158":
        return QualityPrior(0.85, 0.10, Provenance.untested, "ternary QAT: literature claims near-parity at scale after long training; nothing measured in Forge, wide placeholder")
    return QualityPrior(1.0, 0.0, Provenance.derived, "reference")


def quality_prior(c: Candidate, model: ModelSpec, budget: Budget | None = None) -> QualityPrior:
    """Prior expected score retention for a candidate.  Values come from Plan targets (provenance=target), P1/P5 reported ranges
    (paper_reported) or are pure guesses for methods nobody has measured here (untested, wide uncertainty).  Serving-level
    candidates (KV, speculation, sparsity ...) stack on the weight scheme's prior."""
    s = c.scheme
    if c.kind in _STACKING:
        base = _weight_prior(s, model)
        if c.kind in ("serving_lossless", "speculation"):
            return QualityPrior(base.ratio, base.unc, base.provenance, base.source + "; serving feature is exact by construction (G-LOSSLESS / G-SPEC verify)", base.longctx_penalty)
        if c.kind == "kv":
            int4 = s.kv_bits < 8
            own = QualityPrior(0.998 if not int4 else 0.99, 0.003 if not int4 else 0.02, Provenance.paper_reported if int4 else Provenance.target,
                               "P1 F8: limited loss on moderate-length tasks; long-context far less forgiving" if int4 else "FP8 KV ~lossless (G-FP8)", 0.0 if not int4 else 0.03)
        elif c.kind == "kv_evict":
            own = QualityPrior(0.97, 0.04, Provenance.paper_reported, "P3 F4: fine on retrieval, fails on multi-hop", 0.12)
        elif c.kind == "act_sparsity":
            own = QualityPrior(0.985, 0.015, Provenance.paper_reported, "P2 F5: 40-50% activation sparsity at small quality cost", 0.01)
        else:  # delta
            own = QualityPrior(0.99, 0.03, Provenance.untested, "P4: fleet-scale quality of 1-bit deltas not established")
        return _combine(base, own, f"{base.source}; {own.source}")
    if c.kind == "bf16":
        base = QualityPrior(1.0, 0.0, Provenance.derived, "reference")
    elif c.kind == "lossless":
        base = QualityPrior(1.0, 0.0, Provenance.derived, "bit-exact by construction (gate G-LOSSLESS verifies)")
    elif c.kind in ("fp8", "w4a16", "w4a8", "nvfp4", "mxfp4"):
        base = _weight_prior(s, model)
    elif c.kind in ("sub35", "sub35_qat"):
        base = QualityPrior(0.93 if c.kind == "sub35_qat" else 0.88, 0.06, Provenance.untested, "P1 F5: usable at 2 bits for large models only")
    elif c.kind == "prune_distill":
        base = QualityPrior(max(0.5, 0.78 + 0.22 * c.footprint_ratio - 0.08), 0.06, Provenance.untested, "Plan G5 target: student meets the gate at <=25% of parent footprint; no measurements")
    elif c.kind == "expert_prune":
        base = QualityPrior(0.985 if c.footprint_ratio > 0.9 else 0.96, 0.03, Provenance.paper_reported, "P2 F5: 25-50% expert pruning with small loss")
    elif c.kind == "long_arch":
        base = QualityPrior(0.97, 0.05, Provenance.untested, "P3 F2-F3: parity reported for average benchmarks, thinly evidenced for multi-hop", 0.04)
    elif c.kind == "lowrank":
        base = QualityPrior(0.92, 0.06, Provenance.untested, "P4 F4-F6: rarely wins wall-clock; quality depends on rank allocation")
    elif c.kind in ("train_fp8", "train_nvfp4"):
        base = QualityPrior(0.995 if c.kind == "train_fp8" else 0.985, 0.01 if c.kind == "train_fp8" else 0.02, Provenance.paper_reported, "P5: 1-2% relative loss gap for NVFP4 recipes")
    elif c.kind == "fresh_student":
        base = QualityPrior(0.85, 0.08, Provenance.untested, "equal-footprint baseline arm (P4 F8): no measurements")
    else:
        base = QualityPrior(0.97, 0.05, Provenance.untested, "no prior for this kind")
    # a weight scheme that already carries a quantised KV cache (e.g. fp8 + kvfp8) pays the KV loss on top
    if s.kv_bits < 16:
        kvl = 0.002 if s.kv_bits >= 8 else 0.01
        base = QualityPrior(base.ratio - kvl, math.hypot(base.unc, 0.003 if s.kv_bits >= 8 else 0.02), base.provenance, base.source + f"; KV {s.kv_bits:g}-bit",
                            base.longctx_penalty + (0 if s.kv_bits >= 8 else 0.03))
    if s.activation_sparsity:
        base = QualityPrior(base.ratio - 0.01, math.hypot(base.unc, 0.015), base.provenance, base.source + "; activation sparsity", base.longctx_penalty)
    return base


def claim_risk(claim_ids: Sequence[str], db: Any = None) -> float:
    """0..1 evidence risk of a set of claims from the ledger: untested untrusted = 1, untested trusted = 0.6, partial 0.5, contradicted 1,
    reproduced 0 (averaged).  Falls back to the seed priors if the ledger table is empty/unavailable."""
    if not claim_ids:
        return 0.0
    statuses: dict[str, tuple[str, str]] = {}
    try:
        from forge.core.db import get_db

        d = db or get_db()
        for cid in claim_ids:
            r = d.query_one("SELECT status, prior FROM claims WHERE id=?", (cid,))
            if r:
                statuses[cid] = (r["status"], r["prior"])
    except Exception:  # ledger not available: use seed priors
        pass
    if len(statuses) < len(claim_ids):
        from forge.planner.claims_seed import SEED_CLAIMS

        prior = {c["id"]: c["prior"] for c in SEED_CLAIMS}
        for cid in claim_ids:
            statuses.setdefault(cid, ("untested", prior.get(cid, "untrusted")))
    vals = []
    for cid in claim_ids:
        st, pr = statuses[cid]
        vals.append({"reproduced": 0.0, "partial": 0.5, "contradicted": 1.0}.get(st, 1.0 if pr == "untrusted" else 0.6))
    return float(sum(vals) / len(vals))


# --------------------------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------------------------
def baseline_arms(model: ModelSpec, budget: Budget | None = None, *, footprints: Sequence[float] = ()) -> list[Candidate]:
    """Comparison arms every plan carries (Plan 5.3 / 1.2 principle 7): the 16-bit parent, naive round-to-nearest 4-bit and 8-bit PTQ,
    and a **fresh smaller dense model of equal footprint** for each distinct footprint ratio in ``footprints`` (P4 F8: a compression
    method must beat 'train or distil a smaller model')."""
    budget = budget or Budget()
    rule = {"id": "BASELINE", "claims": ["P4-BASE-01"], "note": "", "training": False, "experimental": False}
    out: list[Candidate] = []
    parent = build_candidate(model, "bf16", {"kind": "bf16"}, rule, budget, {})
    if parent:
        parent.arm = "baseline"
        parent.name = "BF16 parent (reference)"
        out.append(parent)
    for kind, prop, nm in (("w4a16", {"kind": "w4a16", "group": 128, "method": "rtn", "gate": None}, "Naive RTN W4A16 g128"),
                           ("fp8", {"kind": "fp8", "kv": "bf16", "gate": None}, "Naive FP8 W8A8 (RTN, no calibration)")):
        c = build_candidate(model, kind, prop, rule, budget, {})
        if c:
            c.arm, c.name = "baseline", nm
            c.id = "naive-" + c.id
            out.append(c)
    for r in sorted({round(float(f), 2) for f in footprints if 0.02 < f < 0.98}):
        stages = [StageSpec(name="finetune", params={"mode": "train_student", "params_ratio_bf16": r, "tokens_b": budget.distill_tokens_b, "teacher": "$parent", "init": "fresh"}),
                  StageSpec(name="eval", params={"suites": ["distribution", "mcq", "longctx", "longgen", "safety"]})]
        rec = Recipe(name=f"fresh-student-{r:.2f}", inputs={"model": "$parent"}, stages=stages, baselines=["bf16-parent"], budget={"gpu_hours": budget.gpu_hours}, seeds=list(budget.seeds))
        out.append(Candidate(id=f"fresh-student-{int(r * 100)}", name=f"Fresh smaller student at {r:.0%} of the bf16 footprint", kind="fresh_student", scheme=parse_scheme("bf16"), recipe=rec,
                             rule_ids=["BASELINE"], claim_ids=["P4-BASE-01"], requires_training=True, arm="baseline", footprint_ratio=r,
                             notes=["equal-footprint baseline arm (P4 F8); assumes the student keeps the parent's attention/KV shape"]))
    return out


# --------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------
class Prediction(_M):
    """Cost-model predictions for one candidate (all derived) plus the quality proxy (dry mode: prior)."""

    bytes: float = Field(description="Resident weight bytes (artifact size proxy).")
    weights_gb: float = Field(description="Resident weights in GB.")
    kv_gb_at_workload: float = Field(description="KV cache GB at the workload.")
    fits: bool = Field(description="Fits in HBM at the workload.")
    concurrency: int = Field(description="Concurrency used for throughput (min of planned batch and what fits).")
    max_concurrency_at_ctx: float = Field(description="Sequences of the planned context that fit.")
    step_ms_b1: float = Field(description="Predicted decode step at batch 1 (ms, calibrated form).")
    step_ms_ideal_b1: float = Field(description="Ideal (bytes / datasheet bandwidth) decode step at batch 1 (ms).")
    tok_s_stream_b1: float = Field(description="Predicted single-stream tokens/s at batch 1.")
    tok_s_total: float = Field(description="Predicted aggregate tokens/s at the planned (or fitting) concurrency.")
    bound: str = Field(description="mem | mma | alu at the planned concurrency.")
    spec_speedup: float = Field(default=1.0, description="Assumed speculation speed-up at that concurrency (alpha/c are ASSUMPTIONS).")
    quality_ratio: float = Field(description="Expected suite-score retention vs the parent (prior in dry mode).")
    quality_unc: float = Field(description="Half-width of the quality estimate.")
    quality_provenance: Provenance = Field(description="target / paper_reported / untested in dry mode; measured with a real runner.")
    longctx_penalty: float = Field(default=0.0, description="Extra expected loss on long-context suites.")
    risk: float = Field(description="Evidence risk 0..1 from the claims ledger.")
    notes: list[str] = Field(default_factory=list, description="Caveats.")


def predict(model: ModelSpec, chip: ChipSpec, cand: Candidate, workload: Workload | None, budget: Budget, *, eta: cm.EtaFactors | None = None,
            spec_alpha: float = 0.7, spec_c: float = 0.1, use_ledger: bool = True) -> tuple[Prediction, FitReport]:
    wl = workload or Workload()
    B, T = workload_shape(wl)
    m = model
    if cand.footprint_ratio != 1.0:
        m = model.model_copy(update={"params_b": model.params_b * cand.footprint_ratio,
                                     "active_params_b": None if model.active_params_b is None else model.active_params_b * cand.footprint_ratio})
    # an equal-footprint fresh student is a *bf16* model with fewer parameters
    fit = fits(m, chip, cand.scheme, wl)
    conc = int(max(1, min(B, math.floor(fit.max_concurrency_at_ctx)))) if fit.max_concurrency_at_ctx >= 1 else 1
    eta = eta or cm.EtaFactors()
    d1 = cm.predict_decode(m, chip, cand.scheme, batch=1, ctx=T // 2 or 1, eta=eta)
    dB = cm.predict_decode(m, chip, cand.scheme, batch=conc, ctx=T // 2 or 1, eta=eta)
    spec = 1.0
    notes = list(dict.fromkeys(d1.notes + fit.notes))
    sp = cand.serving.get("speculation")
    if sp:
        bstar_eta = cm.batch_crossover(chip.peak_flops(cand.scheme.compute_dtype) or chip.peak_flops("bf16") or 1e12, chip.bw, cand.scheme.weight_bits_eff,
                                       eta=eta.eta_mma / max(eta.eta_mem, 1e-9))
        g, _s = cm.optimal_gamma(spec_alpha, spec_c, int(sp.get("max_gamma", 5)))
        spec = cm.spec_speedup_batched(conc, g, spec_alpha, bstar_eta) if g else 1.0
        spec = min(spec, cm.spec_speedup(g, spec_alpha, spec_c)) if g else 1.0
        notes.append(f"speculation speed-up uses ASSUMED alpha={spec_alpha}, draft cost c={spec_c} (measure acceptance per position)")
    q = quality_prior(cand, model, budget)
    risk = claim_risk(cand.claim_ids) if use_ledger else claim_risk(cand.claim_ids, db=None)
    wbytes = cm.model_weight_bytes(m, cand.scheme)["total"]
    pred = Prediction(
        bytes=wbytes, weights_gb=wbytes / 1e9, kv_gb_at_workload=fit.kv_gb_at_workload, fits=fit.fits, concurrency=conc, max_concurrency_at_ctx=fit.max_concurrency_at_ctx,
        step_ms_b1=d1.step_time_pred_s * 1e3, step_ms_ideal_b1=d1.step_time_ideal_s * 1e3, tok_s_stream_b1=d1.tok_s_stream_pred * spec if sp else d1.tok_s_stream_pred,
        tok_s_total=dB.tok_s_total_pred * spec, bound=dB.bound, spec_speedup=spec, quality_ratio=q.ratio, quality_unc=q.unc, quality_provenance=q.provenance,
        longctx_penalty=q.longctx_penalty, risk=risk, notes=notes + [f"quality prior: {q.source}"],
    )
    return pred, fit


# --------------------------------------------------------------------------------------------
# rungs / runners
# --------------------------------------------------------------------------------------------
@dataclass
class Rung:
    name: str
    keep: float  # fraction (0<k<=1) or absolute count (>1) of survivors, before tie-keeping
    suites: list[str]
    seeds: list[int] = field(default_factory=lambda: [0])
    metric: str = "quality"  # what this rung scores: kl | quality | longctx | serving


PROXY_KL = Rung("PROXY_KL", 0.75, ["distribution"], metric="kl")
SUITE_SHORT = Rung("SUITE_SHORT", 0.6, ["distribution", "mcq"], metric="quality")
SUITE_LONGCTX_REASONING_SAFETY = Rung("SUITE_LONGCTX_REASONING_SAFETY", 0.6, ["mcq", "longctx", "longgen", "safety", "agentic"], metric="longctx")
END_TO_END_SERVING = Rung("END_TO_END_SERVING", 1.0, ["throughput"], metric="serving")
DEFAULT_RUNGS = [PROXY_KL, SUITE_SHORT, SUITE_LONGCTX_REASONING_SAFETY, END_TO_END_SERVING]


class RungResult(_M):
    candidate_id: str = Field(description="Candidate evaluated.")
    rung: str = Field(description="Rung name.")
    score: float = Field(description="Higher is better; scale depends on the rung.")
    unc: float = Field(default=0.0, description="Half-width of the score's uncertainty (CI half-width or prior band).")
    metrics: dict[str, float] = Field(default_factory=dict, description="Metric name -> value.")
    passed_gate: bool | None = Field(default=None, description="Gate outcome if a gate was applied (None = not evaluated).")
    gate_verdict: str | None = Field(default=None, description="pass | fail | inconclusive.")
    provenance: Provenance = Field(default=Provenance.derived, description="measured for real runners, derived/target in dry mode.")
    cost_gpu_hours: float = Field(default=0.0, description="Compute spent (0 in dry mode).")
    notes: list[str] = Field(default_factory=list, description="Notes.")


class Runner(Protocol):
    name: str
    mode: str  # "dry" | "measured"

    def run(self, scored: list["ScoredCandidate"], rung: Rung, ctx: "PlanContext") -> list[RungResult]: ...


class ScoredCandidate(_M):
    candidate: Candidate
    fit: FitReport
    pred: Prediction
    status: str = Field(default="kept", description="kept | no_fit | dominated | below_gate | not_run (needs training budget).")
    reasons: list[str] = Field(default_factory=list, description="Why it was dropped / flagged.")
    rungs: dict[str, RungResult] = Field(default_factory=dict, description="Results per rung it survived.")
    rank: int | None = Field(default=None, description="Final rank among survivors (ties share a tie_group).")
    tie_group: int | None = Field(default=None, description="Candidates in one group are statistically indistinguishable on the final score.")
    on_pareto: bool = Field(default=False, description="On the Pareto front.")
    verdict: str = Field(default="inconclusive", description="Expected gate outcome: pass | fail | inconclusive (dry mode: from the quality prior band).")

    @property
    def id(self) -> str:
        return self.candidate.id


@dataclass
class PlanContext:
    model: ModelSpec
    chip: ChipSpec
    workload: Workload
    budget: Budget
    eta: cm.EtaFactors
    binding: BindingReport


def _gate_verdict_from_prior(pred: Prediction, budget: Budget) -> str:
    floor = 1.0 - budget.max_quality_loss_pct / 100.0
    if pred.quality_ratio - pred.quality_unc >= floor:
        return "pass"
    if pred.quality_ratio + pred.quality_unc < floor:
        return "fail"
    return "inconclusive"


class CostModelRunner:
    """Dry mode: no GPUs, no models.  Scores come from the cost model and quality PRIORS (provenance target/paper_reported/untested)."""

    name = "cost-model"
    mode = "dry"

    def run(self, scored: list[ScoredCandidate], rung: Rung, ctx: PlanContext) -> list[RungResult]:
        out = []
        for sc in scored:
            p = sc.pred
            if rung.metric == "kl":
                # lossy proxy: expected KL grows with expected quality loss (illustrative monotone map, NOT a measurement)
                loss = max(0.0, 1.0 - p.quality_ratio)
                score, unc, metrics = -loss * 0.5, p.quality_unc * 0.5, {"expected_loss": loss}
            elif rung.metric == "quality":
                score, unc, metrics = p.quality_ratio, p.quality_unc, {"quality_ratio": p.quality_ratio}
            elif rung.metric == "longctx":
                score, unc = p.quality_ratio - p.longctx_penalty, math.hypot(p.quality_unc, p.longctx_penalty * 0.5)
                metrics = {"quality_ratio": p.quality_ratio, "longctx_penalty": p.longctx_penalty}
            else:
                floor = 1.0 - ctx.budget.max_quality_loss_pct / 100.0
                gate_ok = (p.quality_ratio - p.longctx_penalty + p.quality_unc) >= floor  # only candidates the prior band rules out score 0
                score = p.tok_s_total if (gate_ok and p.fits) else 0.0
                unc = score * 0.1  # 10% model error band on tokens/s until calibrated
                metrics = {"tok_s_total": p.tok_s_total, "step_ms_b1": p.step_ms_b1}
            gv = _gate_verdict_from_prior(p, ctx.budget)
            out.append(RungResult(candidate_id=sc.id, rung=rung.name, score=score, unc=unc, metrics=metrics, passed_gate=None if gv == "inconclusive" else gv == "pass",
                                  gate_verdict=gv, provenance=p.quality_provenance if rung.metric != "serving" else Provenance.derived,
                                  notes=["dry mode: cost model + quality prior"]))
        return out


class RecipeEvalRunner:
    """Real runner (GPU host): execute each candidate's recipe with :class:`forge.core.recipe.RecipeEngine`, evaluate with
    :func:`forge.eval.run.run_eval` using the rung's suites, and score from the report.  UNTESTED on hardware.

    ``engine_factory(ctx)`` and ``eval_fn(cfg, ctx)`` are injectable for tests.  ``parent`` is a :class:`forge.eval.config.ModelSource`
    (or dict) describing the parent used as reference in every evaluation."""

    name = "recipe-eval"
    mode = "measured"

    def __init__(self, parent: Any, run_ctx: Any, *, engine_factory: Callable[[Any], Any] | None = None, eval_fn: Callable[..., Any] | None = None,
                 suite_params: dict[str, dict[str, Any]] | None = None, deployment_length: int | None = None, context: dict[str, Any] | None = None) -> None:
        self.parent = parent
        self.run_ctx = run_ctx
        self.engine_factory = engine_factory
        self.eval_fn = eval_fn
        self.suite_params = suite_params or {}
        self.deployment_length = deployment_length
        self.context = context or {}
        self._artifacts: dict[str, str] = {}

    def _engine(self) -> Any:
        if self.engine_factory is not None:
            return self.engine_factory(self.run_ctx)
        from forge.core.recipe import RecipeEngine

        return RecipeEngine(getattr(self.run_ctx, "store", None))

    def _artifact_for(self, cand: Candidate) -> str | None:
        key = cand.recipe.hash()
        if key in self._artifacts:
            return self._artifacts[key]
        pre = [s for s in cand.recipe.stages if s.name not in ("eval", "gate")]
        if not pre:
            return None  # serving-level candidates evaluate the parent weights
        recipe = cand.recipe.model_copy(update={"stages": pre})
        outputs = self._engine().run(recipe, self.run_ctx)
        art = None
        for k in ("artifact", "model", "artifact_hash", "output"):
            if isinstance(outputs, dict) and outputs.get(k):
                art = str(outputs[k])
                break
        if art is None and isinstance(outputs, dict) and outputs:
            art = str(list(outputs.values())[-1])
        self._artifacts[key] = art or ""
        return art

    def run(self, scored: list[ScoredCandidate], rung: Rung, ctx: PlanContext) -> list[RungResult]:
        from forge.eval.config import EvalConfig, ModelSource, SuiteSelection

        run_eval = self.eval_fn
        if run_eval is None:
            from forge.eval.run import run_eval as run_eval  # type: ignore[no-redef]
        results = []
        parent = self.parent if isinstance(self.parent, ModelSource) else ModelSource(**self.parent)
        for sc in scored:
            c = sc.candidate
            art = self._artifact_for(c) if not c.requires_training or ctx.budget.allow_training else None
            cand_src = ModelSource(kind="artifact", ref=art, label=c.name, params_b=ctx.model.params_b) if art else parent.model_copy(update={"label": c.name})
            cfg = EvalConfig(name=f"plan-{c.id}-{rung.name}", candidate=cand_src, parent=parent, suites=[SuiteSelection(name=s, params=self.suite_params.get(s, {})) for s in rung.suites],
                             gate=c.gate if rung.metric in ("longctx", "quality") else None, seeds=rung.seeds or [0], deployment_length=self.deployment_length,
                             context={"params_b": ctx.model.params_b, **self.context}, tags=["planner", rung.name])
            rep = run_eval(cfg, self.run_ctx)
            m = {k: v.value for k, v in rep.metrics.items()}
            if rung.metric == "kl":
                mv = rep.metrics.get("kl_mean")
                score, unc = (-mv.value, 0.5 * ((mv.ci_high or mv.value) - (mv.ci_low or mv.value))) if mv else (-math.inf, 0.0)
            elif rung.metric == "quality":
                mv = rep.metrics.get("suite_score_ratio") or rep.metrics.get("suite_score")
                score, unc = (mv.value, 0.5 * ((mv.ci_high or mv.value) - (mv.ci_low or mv.value))) if mv else (0.0, 0.0)
            elif rung.metric == "longctx":
                vals = [rep.metrics[k] for k in ("suite_score_ratio", "longctx_ratio", "longgen_reasoning") if k in rep.metrics]
                worst = min(vals, key=lambda v: v.value) if vals else None
                score, unc = (worst.value, 0.5 * ((worst.ci_high or worst.value) - (worst.ci_low or worst.value))) if worst else (0.0, 0.0)
            else:
                mv = rep.metrics.get("tokens_per_s")
                score, unc = (mv.value, 0.0) if mv else (0.0, 0.0)
            gv = rep.gate.verdict if rep.gate else None
            results.append(RungResult(candidate_id=c.id, rung=rung.name, score=score, unc=unc, metrics=m, passed_gate=None if gv is None else gv == "pass", gate_verdict=gv,
                                      provenance=Provenance.measured, notes=[f"eval report {rep.id}"]))
        return results


# --------------------------------------------------------------------------------------------
# selection helpers
# --------------------------------------------------------------------------------------------
def indistinguishable(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Scores (value, half-width uncertainty) are indistinguishable if their difference is within the combined uncertainty."""
    return abs(a[0] - b[0]) <= math.hypot(a[1], b[1])


def keep_top(results: Sequence[RungResult], k: int, *, require_gate: bool = False, noise_aware: bool = True) -> tuple[list[str], list[list[str]]]:
    """Keep the top ``k`` candidate ids by score.  With ``noise_aware`` every candidate statistically indistinguishable from the k-th is
    kept as well (the planner never ranks within noise).  Returns ``(kept_ids, tie_groups)``; gate failures are removed first when
    ``require_gate`` (inconclusive survives: it needs the next rung's evidence)."""
    pool = [r for r in results if not (require_gate and r.gate_verdict == "fail")]
    pool.sort(key=lambda r: (-r.score, r.candidate_id))
    if not pool:
        return [], []
    k = max(1, min(k, len(pool)))
    kth = pool[k - 1]
    kept = pool[:k]
    if noise_aware:
        for r in pool[k:]:
            if indistinguishable((r.score, r.unc), (kth.score, kth.unc)):
                kept.append(r)
    groups: list[list[str]] = []
    for r in kept:
        placed = False
        for g in groups:
            lead = next(x for x in kept if x.candidate_id == g[0])
            if indistinguishable((r.score, r.unc), (lead.score, lead.unc)):
                g.append(r.candidate_id)
                placed = True
                break
        if not placed:
            groups.append([r.candidate_id])
    return [r.candidate_id for r in kept], groups


def pareto_front(items: Sequence[dict[str, Any]], axes: Sequence[tuple[str, str]], tol: dict[str, float] | None = None) -> list[dict[str, Any]]:
    """Flag the Pareto-optimal ``items`` (dicts) over ``axes = [(key, 'min'|'max'), ...]``.  ``tol[key]`` is an absolute tolerance:
    differences within it are ties (a point is dominated only if it is no better on every axis within tolerance and strictly better
    beyond tolerance on at least one).  Returns copies with ``on_frontier`` and ``dominated_by``."""
    tol = tol or {}
    out = [dict(it, dominated_by=[], on_frontier=True) for it in items]

    def better(a: dict[str, Any], b: dict[str, Any], key: str, sense: str, strict: bool) -> bool:
        t = tol.get(key, 0.0)
        da = (b[key] - a[key]) if sense == "min" else (a[key] - b[key])
        return da > t if strict else da >= -t

    for i, b in enumerate(out):
        for j, a in enumerate(out):
            if i == j:
                continue
            if all(better(a, b, k, s, False) for k, s in axes) and any(better(a, b, k, s, True) for k, s in axes):
                b["dominated_by"].append(a.get("id", j))
        b["on_frontier"] = not b["dominated_by"]
    return out


# --------------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------------
class PlanResult(_M):
    """Ranked candidate recipes with the full explanation (UI: ranked table, Pareto chart, 'why did Forge choose this?')."""

    model: ModelSpec
    chip: ChipSpec
    workload: Workload
    budget: Budget
    binding: BindingReport
    ranked: list[ScoredCandidate] = Field(default_factory=list, description="Survivors, best first (ties share tie_group).")
    dropped: list[ScoredCandidate] = Field(default_factory=list, description="Dropped candidates with reasons.")
    baselines: list[ScoredCandidate] = Field(default_factory=list, description="Baseline arms (parent, naive PTQ, equal-footprint fresh student).")
    pareto: list[str] = Field(default_factory=list, description="Ids on the Pareto front over (bytes, latency, throughput, quality, risk).")
    rungs: list[dict[str, Any]] = Field(default_factory=list, description="Per-rung summary: candidates in/out, ties.")
    trace: list[dict[str, Any]] = Field(default_factory=list, description="Rulebase trace: which rules fired and why not.")
    mode: str = Field(default="dry", description="dry (cost model + priors) or measured.")
    notes: list[str] = Field(default_factory=list, description="Caveats shown with the result.")
    generated_at: float = Field(default_factory=now, description="Timestamp.")

    def top(self) -> ScoredCandidate | None:
        return self.ranked[0] if self.ranked else None


def plan(
    model: ModelSpec | str | dict | Any,
    chip: ChipSpec | str | dict | Any,
    workload: Workload | dict | None = None,
    budget: Budget | dict | None = None,
    *,
    runner: Runner | None = None,
    rungs: Sequence[Rung] | None = None,
    eta: cm.EtaFactors | None = None,
    include_baselines: bool = True,
    update_claims: bool = False,
    claims_mapping: dict[str, dict[str, str]] | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> PlanResult:
    """Rank validated (or, in dry mode, *predicted*) recipes for ``model`` on ``chip`` under ``workload`` and ``budget``.

    ``runner=None`` is dry mode (:class:`CostModelRunner`).  ``update_claims`` (measured mode only) writes reproduced/contradicted
    statuses to the ledger using ``claims_mapping = {candidate_kind: {claim_id: metric_name}}``."""
    msp = ModelSpec.from_config(model)
    csp = resolve_chip(chip)
    wl = workload if isinstance(workload, Workload) else Workload(**(workload or {}))
    bd = budget if isinstance(budget, Budget) else Budget(**(budget or {}))
    runner = runner or CostModelRunner()
    rungs = list(rungs or DEFAULT_RUNGS)
    eta = eta or cm.EtaFactors()
    bind = binding_resource(msp, csp, wl, bd, eta=eta)
    cands, trace = propose_with_trace(msp, bind.binding, csp, bd, wl, bf16_fits=bind.bf16_fit.fits, kv_dominates=bind.kv_dominates)
    ctx = PlanContext(msp, csp, wl, bd, eta, bind)
    notes: list[str] = []
    if runner.mode == "dry":
        notes.append("DRY MODE: throughput/latency are cost-model predictions (derived); quality numbers are PRIORS (Plan gate targets and paper-reported ranges), not measurements. Run the recipes through the eval harness to replace them.")

    def score(c: Candidate) -> ScoredCandidate:
        pred, fit = predict(msp, csp, c, wl, bd, eta=eta)
        sc = ScoredCandidate(candidate=c, fit=fit, pred=pred, verdict=_gate_verdict_from_prior(pred, bd))
        return sc

    scored = [score(c) for c in cands]
    base_cands = baseline_arms(msp, bd, footprints=sorted({round(s.pred.bytes / max(cm.model_weight_bytes(msp, parse_scheme("bf16"))["total"], 1.0), 2) for s in scored if s.candidate.kind != "bf16"})) if include_baselines else []
    baselines = [score(c) for c in base_cands]
    for sc in baselines:
        if sc.candidate.kind == "fresh_student":
            sc.pred.notes.append("baseline arm: needs training to realise; predictions assume the same attention/KV shape")
    # fits() filter, baselines excluded from selection but reported
    dropped: list[ScoredCandidate] = []
    live: list[ScoredCandidate] = []
    for sc in scored:
        if not sc.fit.fits:
            sc.status, sc.reasons = "no_fit", sc.fit.notes or ["does not fit in HBM at the workload"]
            dropped.append(sc)
        elif sc.candidate.requires_training and not bd.allow_training:
            sc.status, sc.reasons = "not_run", ["needs training compute; budget.allow_training is false"]
            dropped.append(sc)
        else:
            live.append(sc)
    # dominated pruning on (bytes, latency, quality, risk) with quality tolerance = prior uncertainty
    pts = [{"id": s.id, "bytes": s.pred.bytes, "lat": s.pred.step_ms_b1, "quality": s.pred.quality_ratio, "thr": s.pred.tok_s_total, "risk": s.pred.risk, "unc": s.pred.quality_unc} for s in live]
    flagged = pareto_front(pts, [("bytes", "min"), ("lat", "min"), ("quality", "max"), ("thr", "max")], tol={"quality": max([p["unc"] for p in pts] + [0.0]) * 0.5, "lat": 0.02 * max([p["lat"] for p in pts] + [0.0]),
                                                                                                           "bytes": 0.02 * max([p["bytes"] for p in pts] + [0.0]), "thr": 0.02 * max([p["thr"] for p in pts] + [0.0])})
    stacked_on = {s.id: {x.id for x in live if f"stacked on {s.id} " in " ".join(x.candidate.notes)} for s in live}  # descendants never prune their own base
    stackers = {s.id for s in live if s.candidate.kind in _STACKING}  # serving-level candidates never prune anything (they depend on a base)
    dom = {p["id"]: [d for d in p["dominated_by"] if d not in stacked_on.get(p["id"], set()) and d not in stackers] for p in flagged if not p["on_frontier"]}
    dom = {k: v for k, v in dom.items() if v}
    survivors = []
    for s in live:
        if s.id in dom and s.candidate.kind not in ("bf16",):
            s.status, s.reasons = "dominated", [f"dominated on (bytes, latency, throughput, quality proxy) by {', '.join(map(str, dom[s.id]))}"]
            dropped.append(s)
        else:
            survivors.append(s)
    rung_log: list[dict[str, Any]] = []
    cur = survivors
    for ri, rung in enumerate(rungs):
        if not cur:
            break
        if progress:
            progress(ri / max(1, len(rungs)), f"rung {rung.name}")
        res = runner.run(cur, rung, ctx)
        by_id = {r.candidate_id: r for r in res}
        for s in cur:
            if s.id in by_id:
                s.rungs[rung.name] = by_id[s.id]
        k = int(rung.keep) if rung.keep > 1 else max(1, math.ceil(rung.keep * len(cur)))
        kept_ids, groups = keep_top(res, k, require_gate=(rung.metric in ("quality", "longctx")), noise_aware=True)
        out_rung = {"rung": rung.name, "n_in": len(cur), "n_kept": len(kept_ids), "kept": kept_ids, "tie_groups": [g for g in groups if len(g) > 1], "mode": runner.mode}
        rung_log.append(out_rung)
        nxt = []
        for s in cur:
            if s.id in kept_ids:
                nxt.append(s)
            else:
                r = by_id.get(s.id)
                s.status = "below_gate" if (r and r.gate_verdict == "fail") else "dominated"
                s.reasons = [f"eliminated at rung {rung.name} (score {r.score:.4g} ± {r.unc:.3g})" if r else f"eliminated at rung {rung.name}"]
                dropped.append(s)
        cur = nxt
        if runner.mode == "measured":
            _calibrate(res, cur, ctx, notes)
            if update_claims:
                _update_claims(res, cur, claims_mapping, notes)
    # final ranking: gate verdict, then throughput (noise-aware tie groups), then bytes, then risk
    order = {"pass": 0, "inconclusive": 1, "fail": 2}
    final_rung = rungs[-1].name if rungs else ""

    def final_key(s: ScoredCandidate) -> tuple:
        r = s.rungs.get(final_rung)
        gv = (r.gate_verdict if r and r.gate_verdict else s.verdict)
        return (order.get(gv, 1), -(r.score if r else s.pred.tok_s_total), s.pred.bytes, s.pred.risk)

    cur.sort(key=final_key)
    gid = 0
    prev: ScoredCandidate | None = None
    for i, s in enumerate(cur):
        s.rank = i + 1
        r = s.rungs.get(final_rung)
        if prev is not None:
            pr = prev.rungs.get(final_rung)
            same_gate = order.get((r.gate_verdict if r and r.gate_verdict else s.verdict), 1) == order.get((pr.gate_verdict if pr and pr.gate_verdict else prev.verdict), 1)
            if not (same_gate and r and pr and indistinguishable((r.score, r.unc), (pr.score, pr.unc))):
                gid += 1
        s.tie_group = gid
        prev = s
    ties = {}
    for s in cur:
        ties.setdefault(s.tie_group, []).append(s.id)
    for g, ids in ties.items():
        if len(ids) > 1:
            notes.append(f"candidates {', '.join(ids)} are statistically indistinguishable on the final score; ordered by bytes then risk, not by score")
    # pareto on the survivors + baselines
    allp = [{"id": s.id, "bytes": s.pred.bytes, "lat": s.pred.step_ms_b1, "thr": s.pred.tok_s_total, "quality": s.pred.quality_ratio, "risk": s.pred.risk}
            for s in cur + [b for b in baselines if b.fit.fits]]
    front = pareto_front(allp, [("bytes", "min"), ("lat", "min"), ("thr", "max"), ("quality", "max"), ("risk", "min")],
                         tol={"quality": 0.005, "risk": 0.05, "lat": 0.02 * max([p["lat"] for p in allp] + [0.0]), "bytes": 0.02 * max([p["bytes"] for p in allp] + [0.0]),
                              "thr": 0.02 * max([p["thr"] for p in allp] + [0.0])})
    on = {p["id"] for p in front if p["on_frontier"]}
    for s in cur + baselines:
        s.on_pareto = s.id in on
    if any(s.candidate.experimental for s in cur):
        notes.append("experimental / gated candidates are included: they must pass their gate and are not defaults (Plan 1.3 Tier 3)")
    if not cur:
        notes.append("no candidate survived: " + ("; ".join(sorted({r for d in dropped for r in d.reasons[:1]})) or "empty rulebase result"))
    if progress:
        progress(1.0, "done")
    return PlanResult(model=msp, chip=csp, workload=wl, budget=bd, binding=bind, ranked=cur, dropped=dropped, baselines=baselines, pareto=sorted(on), rungs=rung_log, trace=trace,
                      mode=runner.mode, notes=notes)


# ---- measured-mode feedback (Plan 5.3: cost_model.calibrate(results); claims.update(results)) -----------
def _calibrate(res: Sequence[RungResult], cur: Sequence[ScoredCandidate], ctx: PlanContext, notes: list[str]) -> None:
    """Fit eta from END_TO_END results that report measured tokens/s: one Measurement per candidate at its predicted bytes/FLOPs."""
    ms = []
    by = {s.id: s for s in cur}
    for r in res:
        tps = r.metrics.get("tokens_per_s")
        s = by.get(r.candidate_id)
        if not tps or s is None or r.provenance != Provenance.measured:
            continue
        p = cm.predict_decode(ctx.model, ctx.chip, s.candidate.scheme, batch=s.pred.concurrency, ctx=max(ctx.workload.context_lengths or [4096]) // 2, eta=cm.IDEAL)
        ms.append(cm.Measurement(s.candidate.kind, s.pred.concurrency / tps, bytes_mem=p.bytes_total, flops=p.flops, dequant_ops=p.dequant_ops, dtype=s.candidate.scheme.compute_dtype))
    if len(ms) >= 3:
        fit = cm.fit_eta(ms, ctx.chip, prior=ctx.eta)
        cm.save_eta(ctx.chip, "decode", fit.eta)
        notes.append(f"cost model recalibrated from {len(ms)} measurements: eta_mem={fit.eta.eta_mem:.2f}, eta_mma={fit.eta.eta_mma:.2f}, {sum(fit.alerts)} residual alerts")


def _update_claims(res: Sequence[RungResult], cur: Sequence[ScoredCandidate], mapping: dict[str, dict[str, str]] | None, notes: list[str]) -> None:
    from forge.planner import claims as ledger

    by = {s.id: s for s in cur}
    for r in res:
        s = by.get(r.candidate_id)
        if s is None or r.provenance != Provenance.measured:
            continue
        for cid, metric in (mapping or {}).get(s.candidate.kind, {}).items():
            if metric in r.metrics:
                try:
                    out = ledger.evaluate_claim(cid, r.metrics[metric], experiment=next((n.split()[-1] for n in r.notes if n.startswith("eval report")), None))
                    notes.append(out["message"])
                except Exception as e:  # missing claim / expectation: report, do not fail the plan
                    notes.append(f"claim {cid} not updated: {e}")
