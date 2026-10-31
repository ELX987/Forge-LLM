"""Bytes-first cost model (Plan 5.1 and 5.4, Appendix D).  Pure functions, SI units: bytes, seconds, FLOP.

Formulas implemented (all "standard", from P1 2.2 / 14.1 / 19, P3 Ch. 2, P5 eq. 11.1, P7 Ch. 4-5):

====================  =========================================================  ====================================
quantity              formula                                                    function
====================  =========================================================  ====================================
weight bytes          ``W = P * b_eff / 8`` (b_eff includes scales)              :func:`weight_bytes`
KV bytes / token      ``2 L Hkv dh b / 8`` (MLA: latent; hybrids: cache layers)  :func:`kv_bytes_per_token`
decode step (ideal)   ``(W_active + B T k + selector) / (eta beta)``             :func:`decode_step_time`
GEMM intensity        ``2 B / (b/8)`` FLOP per weight byte                       :func:`gemm_intensity`
attention intensity   ``2 g / (b_kv/8)`` FLOP/byte (``= g`` at 16 bit)           :func:`attention_intensity`
batch crossover       ``B* = pi (b/8) / (2 beta)``                               :func:`batch_crossover`
quantiser overhead    ``Tq/Tg = (b_in + b_out) F / (2 N beta)`` (P5 eq. 11.1)    :func:`quantizer_overhead`
speculation           ``E[N] = (1 - a^(g+1)) / (1 - a)``; ``S = E[N]/(c g + 1)``  :func:`spec_expected_tokens`, :func:`spec_speedup`
Amdahl                ``1 / sum_i (f_i / s_i)``                                  :func:`amdahl`
calibrated time       ``max(bytes/(eta_mem beta), flops/(eta_mma pi),``         :func:`t_pred`
                      ``dequant/(eta_alu alu)) + launch + selector/(eta_mem beta)``
====================  =========================================================  ====================================

Every function returns *derived* numbers (Provenance.derived); the eta factors are priors until :func:`fit_eta` has seen measurements.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

from forge.planner.spec import ChipSpec, ModelSpec, Scheme

RESIDUAL_BAND = (0.8, 1.25)  # Plan 5.4: alert if residual outside [0.8, 1.25]


# --------------------------------------------------------------------------------------------
# bytes
# --------------------------------------------------------------------------------------------
def weight_bytes(params: float, bits_eff: float) -> float:
    """``W = P * b_eff / 8`` bytes for ``params`` parameters (absolute count) at ``bits_eff`` bits per weight."""
    return float(params) * bits_eff / 8.0


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, bits: float = 16.0, *, mla_latent: int | None = None, mla_rope: int = 0) -> float:
    """KV bytes per token per sequence.  Dense GQA: ``2 L Hkv dh b/8`` (Llama-3-8B: 128 KiB; Llama-3.1-70B: 320 KiB).
    MLA: ``L (latent + rope) b/8`` (DeepSeek-V3: 576 values per layer).  For hybrids pass only the layers that keep a cache."""
    if mla_latent is not None:
        return layers * (mla_latent + mla_rope) * bits / 8.0
    return 2.0 * layers * kv_heads * head_dim * bits / 8.0


def expert_fraction(model: ModelSpec) -> float:
    """Fraction of the non-embedding parameters that sit in routed experts.  Derived from active/total when both are known:
    ``active/total = (1 - f) + f k/E``  =>  ``f = (1 - active/total) / (1 - k/E)``; otherwise ``expert_params_fraction`` or 0.9."""
    if model.expert_params_fraction is not None:
        return model.expert_params_fraction
    if model.num_experts and model.experts_per_token and model.active_params_b is not None and model.experts_per_token < model.num_experts:
        emb = model.embed_params_b
        body_total = max(model.params_b - emb, 1e-9)
        body_active = max(model.active_params_b - emb, 0.0)
        r = min(max(body_active / body_total, 0.0), 1.0)
        f = (1.0 - r) / (1.0 - model.experts_per_token / model.num_experts)
        return min(max(f, 0.05), 1.0)
    return 0.9


def model_weight_bytes(model: ModelSpec, scheme: Scheme, *, head_bits: float = 16.0, quantize_embeddings: bool = False) -> dict[str, float]:
    """Weight bytes of a model under a scheme.  Embeddings / output head stay at ``head_bits`` unless ``quantize_embeddings``
    (the near-universal practice for PTQ).  Returns ``{total, body, embed, head_read}`` in bytes: ``total`` = resident weights
    (fit test), ``head_read`` = bytes of the output projection read every step.  (The input-embedding table is only indexed.)"""
    emb_b = model.embed_params_b * 1e9
    body = max(0.0, model.params_b * 1e9 - emb_b)
    eff = scheme.weight_bits_eff
    body_bytes = weight_bytes(body * scheme.weight_density, eff)
    ebits = eff if quantize_embeddings else head_bits
    emb_bytes = weight_bytes(emb_b, ebits)
    head_read = weight_bytes(model.vocab_size * model.hidden_size, ebits)
    return {"total": body_bytes + emb_bytes, "body": body_bytes, "embed": emb_bytes, "head_read": head_read}


def moe_expected_active_experts(num_experts: int, top_k: int, batch: int) -> float:
    """Expected number of distinct experts touched by ``batch`` tokens with uniform top-k routing: ``E (1 - (1 - k/E)^B)``."""
    if batch <= 0:
        return 0.0
    return num_experts * (1.0 - (1.0 - top_k / num_experts) ** batch)


def active_weight_bytes(model: ModelSpec, scheme: Scheme, batch: int = 1, *, head_bits: float = 16.0) -> float:
    """Weight bytes read in ONE decode step with ``batch`` sequences.  Dense: every body weight + the output head.  MoE: the
    shared part + the experts hit by the batch (more tokens touch more experts, so MoE decode loses its bandwidth advantage as
    the batch grows - the planner models this explicitly)."""
    wb = model_weight_bytes(model, scheme, head_bits=head_bits)
    if model.form != "moe" or not model.num_experts or not model.experts_per_token:
        return wb["body"] + wb["head_read"]
    f = expert_fraction(model)
    touched = moe_expected_active_experts(model.num_experts, model.experts_per_token, max(1, batch)) / model.num_experts
    return wb["body"] * (1.0 - f) + wb["body"] * f * touched + wb["head_read"]


# --------------------------------------------------------------------------------------------
# intensities / crossover / time
# --------------------------------------------------------------------------------------------
def gemm_intensity(batch: float, weight_bits: float) -> float:
    """FLOP per weight byte of a decode GEMM with ``batch`` rows: ``2B / (b/8)``."""
    return 2.0 * batch / (weight_bits / 8.0)


def attention_intensity(group: float, kv_bits: float = 16.0) -> float:
    """Decode attention FLOP/byte for ``group`` query heads per KV head: ``2 g / (b_kv/8)`` (equals g at 16-bit)."""
    return 2.0 * group / (kv_bits / 8.0)


def batch_crossover(peak_flops: float, bandwidth: float, weight_bits: float, eta: float = 1.0) -> float:
    """``B* = pi_eff (b/8) / (2 beta)``: the batch at which a weight-only GEMM stops being bandwidth-bound
    (P1 14.1: 74 for 4-bit vs 295 for 16-bit on H100-class numbers; pass ``eta`` < 1 for achievable FLOP efficiency)."""
    return (peak_flops * eta) * (weight_bits / 8.0) / (2.0 * bandwidth)


def decode_step_time(
    active_weight_b: float, batch: int, ctx: float, kv_bytes_tok: float, bandwidth: float, *, eta_mem: float = 1.0, selector_bytes: float = 0.0,
) -> float:
    """Ideal (bandwidth-only) time of one decode step: ``(W_active + B T k + selector) / (eta beta)`` seconds."""
    return (active_weight_b + batch * ctx * kv_bytes_tok + selector_bytes) / (eta_mem * bandwidth)


def quantizer_overhead(b_in: float, b_out: float, peak_flops: float, n: float, bandwidth: float) -> float:
    """``Tq/Tg = (b_in + b_out) F / (2 N beta)`` (P5 eq. 11.1): standalone activation-quantiser time over the GEMM time for
    hidden size ``N``; ``b_in``/``b_out`` are bytes read/written per element."""
    return (b_in + b_out) * peak_flops / (2.0 * n * bandwidth)


def fp4_vs_fp8_speedup(
    n: float, *, f4: float = 15e15, f8: float = 5e15, bandwidth: float = 8e12, fp4_out_bytes: float = 0.5625, fp8_out_bytes: float = 1.0,
    in_bytes: float = 2.0, fused_fp4: bool = False,
) -> dict[str, float]:
    """P5 Table 11.2: FP8-time over FP4-time for one forward GEMM with unfused quantisers (default numbers: Blackwell Ultra vendor peaks).

    Returns ``{tq_tg_fp4, tq_tg_fp8, peak_ratio, speedup}``; ``fused_fp4=True`` credits FP4 with a free quantiser (an upper bound)."""
    r4 = quantizer_overhead(in_bytes, fp4_out_bytes, f4, n, bandwidth)
    r8 = quantizer_overhead(in_bytes, fp8_out_bytes, f8, n, bandwidth)
    ratio = f4 / f8
    t8 = 1.0 + r8
    t4 = (1.0 if fused_fp4 else 1.0 + r4) / ratio
    return {"tq_tg_fp4": r4, "tq_tg_fp8": r8, "peak_ratio": ratio, "speedup": t8 / t4}


# --------------------------------------------------------------------------------------------
# speculation (P7)
# --------------------------------------------------------------------------------------------
def spec_expected_tokens(alpha: float, gamma: int) -> float:
    """``E[N] = (1 - alpha^(gamma+1)) / (1 - alpha)`` tokens per verification step (alpha = per-token acceptance rate)."""
    if gamma < 0:
        raise ValueError("gamma must be >= 0")
    if alpha >= 1.0:
        return gamma + 1.0
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


def spec_speedup(gamma: int, alpha: float, c: float) -> float:
    """Memory-bound wall-clock speed-up of speculative decoding: ``S = E[N] / (c gamma + 1)`` where ``c`` is the draft-step cost
    relative to a target step.  ``S > 1`` iff alpha > c (for gamma = 1); gamma = 0 means speculation off (S = 1)."""
    return spec_expected_tokens(alpha, gamma) / (c * gamma + 1.0)


def spec_speedup_batched(batch: float, gamma: int, alpha: float, n_eff: float) -> float:
    """Weights-only batch dependence of the speed-up with a free draft (P7 Theorem 5.1): ``S = E[N]`` up to ``B = n/(gamma+1)``,
    ``E[N] n / (B (gamma+1))`` in between, and ``E[N]/(gamma+1)`` (a slowdown) beyond ``B = n``.  ``n_eff`` is the effective
    crossover batch of the system (``B*`` times FLOP efficiency)."""
    e = spec_expected_tokens(alpha, gamma)
    if batch <= n_eff / (gamma + 1):
        return e
    if batch <= n_eff:
        return e * n_eff / (batch * (gamma + 1))
    return e / (gamma + 1)


def spec_break_even_batch(gamma: int, alpha: float, n_eff: float) -> float:
    """``B_be = E[N] n / (gamma + 1)``: above this batch speculation is a net slowdown (alpha=0.8, gamma=4, n=221 -> ~149)."""
    return spec_expected_tokens(alpha, gamma) * n_eff / (gamma + 1.0)


def optimal_gamma(alpha: float, c: float, gamma_max: int = 16, *, hysteresis: float = 0.03) -> tuple[int, float]:
    """Best draft length by the memory-bound speed-up; gamma = 0 (off) unless the gain beats 1 by ``hysteresis`` (Plan 7.5: 3%)."""
    best_g, best_s = 0, 1.0
    for g in range(1, gamma_max + 1):
        s = spec_speedup(g, alpha, c)
        if s > best_s * (1.0 + 1e-12):
            best_g, best_s = g, s
    if best_s < 1.0 + hysteresis:
        return 0, 1.0
    return best_g, best_s


# --------------------------------------------------------------------------------------------
# Amdahl
# --------------------------------------------------------------------------------------------
def amdahl(fractions: Sequence[float], speedups: Sequence[float], *, normalise: bool = False) -> float:
    """End-to-end speed-up ``1 / sum_i (f_i / s_i)`` over GEMM / attention / KV / communication / sampling shares."""
    if len(fractions) != len(speedups) or not fractions:
        raise ValueError("fractions and speedups must be equally long and non-empty")
    tot = sum(fractions)
    if normalise:
        fractions = [f / tot for f in fractions]
    elif abs(tot - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1 (got {tot:.6g}); pass normalise=True to rescale")
    if any(s <= 0 for s in speedups) or any(f < 0 for f in fractions):
        raise ValueError("speedups must be positive and fractions non-negative")
    return 1.0 / sum(f / s for f, s in zip(fractions, speedups))


# --------------------------------------------------------------------------------------------
# calibrated prediction (Plan 5.4)
# --------------------------------------------------------------------------------------------
@dataclass
class EtaFactors:
    """Efficiency factors per (chip, kernel class).  Defaults are PRIORS (not measurements): memory 0.85 (the G3 target),
    tensor cores 0.6 (the efficiency assumed in P7's roofline), ALU 0.5."""

    eta_mem: float = 0.85
    eta_mma: float = 0.60
    eta_alu: float = 0.50
    source: str = "prior"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


IDEAL = EtaFactors(1.0, 1.0, 1.0, "ideal")


@dataclass
class TimeBreakdown:
    t: float
    t_mem: float
    t_mma: float
    t_alu: float
    t_launch: float
    t_selector: float
    bound: str  # mem | mma | alu

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def t_pred(
    bytes_mem: float, flops: float, dequant_ops: float, chip: ChipSpec, *, dtype: str = "bf16", eta: EtaFactors | None = None,
    launch_s: float | None = None, selector_bytes: float = 0.0, launches: int = 1,
) -> TimeBreakdown:
    """Plan 5.4: ``t = max(bytes/(eta_mem beta), flops/(eta_mma pi[dtype]), dequant_ops/(eta_alu alu)) + launch + selector/(eta_mem beta)``.

    ``dtype`` that the chip lacks natively falls back to bf16 -> fp16 peak (the caller should already have chosen an emulation scheme)."""
    eta = eta or EtaFactors()
    pi = chip.peak_flops(dtype) or chip.peak_flops("bf16") or chip.peak_flops("fp16") or 1e12
    t_mem = bytes_mem / (eta.eta_mem * chip.bw)
    t_mma = flops / (eta.eta_mma * pi) if flops else 0.0
    t_alu = dequant_ops / (eta.eta_alu * chip.alu_ops) if dequant_ops else 0.0
    t_launch = (chip.launch_us * 1e-6 if launch_s is None else launch_s) * launches
    t_sel = selector_bytes / (eta.eta_mem * chip.bw) if selector_bytes else 0.0
    terms = {"mem": t_mem, "mma": t_mma, "alu": t_alu}
    bound = max(terms, key=lambda k: terms[k])
    return TimeBreakdown(max(terms.values()) + t_launch + t_sel, t_mem, t_mma, t_alu, t_launch, t_sel, bound)


def residual(t_measured: float, t_predicted: float) -> float:
    """``t_measured / t_pred`` (1.0 = the model is exact; > 1.25 = slower than modelled)."""
    return t_measured / t_predicted


def residual_alert(res: float, band: tuple[float, float] = RESIDUAL_BAND) -> bool:
    """True if the residual leaves ``[0.8, 1.25]``: the model or the kernel is wrong - investigate."""
    return not (band[0] <= res <= band[1])


@dataclass
class Measurement:
    """One kernel measurement used to fit eta."""

    kernel_class: str
    t_measured: float
    bytes_mem: float = 0.0
    flops: float = 0.0
    dequant_ops: float = 0.0
    dtype: str = "bf16"
    selector_bytes: float = 0.0
    launch_s: float | None = None
    shape_bucket: str = ""


@dataclass
class EtaFit:
    eta: EtaFactors
    n_samples: int
    n_by_term: dict[str, int]
    residuals: list[float]
    alerts: list[bool]
    converged: bool
    mad: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _huber_slope(x: Sequence[float], y: Sequence[float], k: float = 1.345, iters: int = 50) -> float:
    """Robust slope ``a`` of ``y ~ a x`` (through the origin) by Huber IRLS."""
    xs = [float(v) for v in x]
    ys = [float(v) for v in y]
    ratios = [yy / xx for xx, yy in zip(xs, ys) if xx > 0]
    if not ratios:
        return 1.0
    a = statistics.median(ratios)
    for _ in range(iters):
        res = [yy - a * xx for xx, yy in zip(xs, ys)]
        s = 1.4826 * statistics.median([abs(r) for r in res]) or 1e-30
        w = [1.0 if abs(r) <= k * s else k * s / abs(r) for r in res]
        num = sum(wi * xx * yy for wi, xx, yy in zip(w, xs, ys))
        den = sum(wi * xx * xx for wi, xx in zip(w, xs))
        if den <= 0:
            break
        new = num / den
        if abs(new - a) <= 1e-12 * max(1.0, abs(a)):
            a = new
            break
        a = new
    return a


def fit_eta(measurements: Iterable[Measurement], chip: ChipSpec, *, prior: EtaFactors | None = None, min_samples: int = 3,
            eta_bounds: tuple[float, float] = (0.02, 1.5), max_rounds: int = 8) -> EtaFit:
    """Fit ``eta_mem, eta_mma, eta_alu`` of one kernel class from measurements by robust (Huber) regression.

    Each sample is attributed to the term that binds under the current eta; per term the slope of
    ``(t_measured - launch)`` against the ideal term time gives ``1/eta``.  Terms with fewer than ``min_samples`` keep the prior.
    The attribution is iterated until stable.  Samples that disagree with the fit show up as residual alerts."""
    ms = list(measurements)
    if not ms:
        raise ValueError("no measurements")
    prior = prior or EtaFactors()
    eta = EtaFactors(prior.eta_mem, prior.eta_mma, prior.eta_alu, "fit")
    ideal = []
    for m in ms:
        pi = chip.peak_flops(m.dtype) or chip.peak_flops("bf16") or chip.peak_flops("fp16") or 1e12
        launch = chip.launch_us * 1e-6 if m.launch_s is None else m.launch_s
        ideal.append({
            "mem": (m.bytes_mem + m.selector_bytes) / chip.bw, "mma": m.flops / pi, "alu": m.dequant_ops / chip.alu_ops, "y": max(m.t_measured - launch, 1e-12),
        })
    prev: list[str] | None = None
    converged = False
    n_by: dict[str, int] = {"mem": 0, "mma": 0, "alu": 0}
    for _ in range(max_rounds):
        etas = {"mem": eta.eta_mem, "mma": eta.eta_mma, "alu": eta.eta_alu}
        assign = [max(("mem", "mma", "alu"), key=lambda t: d[t] / etas[t]) for d in ideal]
        n_by = {t: assign.count(t) for t in ("mem", "mma", "alu")}
        new = dict(etas)
        for t in ("mem", "mma", "alu"):
            idx = [i for i, a in enumerate(assign) if a == t and ideal[i][t] > 0]
            if len(idx) >= min_samples:
                slope = _huber_slope([ideal[i][t] for i in idx], [ideal[i]["y"] for i in idx])
                if slope > 0:
                    new[t] = min(max(1.0 / slope, eta_bounds[0]), eta_bounds[1])
        eta = EtaFactors(new["mem"], new["mma"], new["alu"], "fit")
        if assign == prev:
            converged = True
            break
        prev = assign
    res, alerts = [], []
    for m in ms:
        tp = t_pred(m.bytes_mem, m.flops, m.dequant_ops, chip, dtype=m.dtype, eta=eta, launch_s=m.launch_s, selector_bytes=m.selector_bytes).t
        r = residual(m.t_measured, tp)
        res.append(r)
        alerts.append(residual_alert(r))
    mad = statistics.median([abs(r - 1.0) for r in res])
    return EtaFit(eta, len(ms), n_by, res, alerts, converged, mad)


# ---- persistence of calibrated eta per (chip fingerprint, kernel class) ------------------------
def save_eta(chip: ChipSpec, kernel_class: str, eta: EtaFactors, db: Any = None) -> None:
    from forge.core.db import get_db

    (db or get_db()).kv_set(f"planner.eta.{chip.fingerprint or chip.name}.{kernel_class}", eta.to_dict())


def load_eta(chip: ChipSpec, kernel_class: str, db: Any = None) -> EtaFactors:
    """Calibrated eta for the chip/kernel class if present, else the priors."""
    from forge.core.db import get_db

    v = (db or get_db()).kv_get(f"planner.eta.{chip.fingerprint or chip.name}.{kernel_class}")
    return EtaFactors(**v) if v else EtaFactors()


# --------------------------------------------------------------------------------------------
# model-level predictions
# --------------------------------------------------------------------------------------------
def effective_scheme(scheme: Scheme, chip: ChipSpec) -> tuple[Scheme, list[str]]:
    """Map a scheme to what the chip can really execute, with notes: FP8 compute without FP8 cores becomes W8A16 (dequantise to
    bf16), FP4 compute without block-scaled FP4 becomes weight-only W4 with bf16 compute, bf16 on fp16-only chips uses fp16."""
    notes: list[str] = []
    s = scheme
    if s.act == "fp8" and not chip.supports("fp8"):
        s = s.model_copy(update={"act": "bf16" if chip.supports("bf16") else "fp16"})
        notes.append(f"{chip.name} has no native FP8 tensor cores: FP8 weights are dequantised to {s.act} (weight-only path; no compute gain)")
    if s.act == "fp4" and not chip.supports("fp4"):
        s = s.model_copy(update={"act": "bf16" if chip.supports("bf16") else "fp16"})
        notes.append(f"{chip.name} has no block-scaled FP4 tensor cores: FP4 emulated as weight-only 4-bit with {s.act} compute (numerics study only; no FP4 speed-up)")
    if s.act == "int8" and not chip.supports("int8"):
        s = s.model_copy(update={"act": "bf16" if chip.supports("bf16") else "fp16"})
        notes.append("no INT8 tensor cores: activations stay 16-bit")
    if s.act == "bf16" and not chip.supports("bf16") and chip.supports("fp16"):
        s = s.model_copy(update={"act": "fp16"})
        notes.append(f"{chip.name} has no bf16 tensor cores: using fp16")
    return s, notes


@dataclass
class DecodePrediction:
    batch: int
    ctx: int
    scheme: str
    bytes_weights: float
    bytes_kv: float
    bytes_total: float
    flops: float
    dequant_ops: float
    step_time_ideal_s: float
    step_time_pred_s: float
    tok_s_stream_ideal: float
    tok_s_stream_pred: float
    tok_s_total_pred: float
    bound: str
    intensity_flop_per_byte: float
    ridge_flop_per_byte: float | None
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decode_flops(model: ModelSpec, batch: int, ctx: int, scheme: Scheme) -> float:
    """FLOPs of one decode step: ``2 * P_active * B`` for the GEMMs plus attention ``4 B T Hq dh`` per cache layer."""
    gemm = 2.0 * model.active_b * 1e9 * batch * (1.0 - scheme.activation_sparsity) * scheme.weight_density
    n_q = model.num_heads * model.head_dim
    attn = 4.0 * batch * ctx * n_q * model.kv_layers
    return gemm + attn


def predict_decode(
    model: ModelSpec, chip: ChipSpec, scheme: Scheme, *, batch: int = 1, ctx: int = 4096, eta: EtaFactors | None = None, head_bits: float = 16.0,
    cuda_graphs: bool = True, kernels_per_layer: int = 8, selector_bytes_per_seq: float = 0.0,
) -> DecodePrediction:
    """Bandwidth/compute/ALU roofline prediction of one decode step for (model, chip, scheme, batch, context).

    ``ideal`` = pure bytes / datasheet bandwidth (Plan Example B); ``pred`` = the calibrated form of Plan 5.4 (prior eta unless
    ``eta`` given).  Launch overhead assumes ``kernels_per_layer`` (8) kernels per layer at ~1 us each with CUDA graphs or the chip's
    launch latency without - both are documented assumptions to be replaced by measurements."""
    sch, notes = effective_scheme(scheme, chip)
    eta = eta or EtaFactors()
    w_active = active_weight_bytes(model, sch, batch, head_bits=head_bits)
    kv_tok = model.kv_bytes_for_context(ctx, sch.kv_bits)
    bytes_kv = batch * kv_tok
    sel = batch * selector_bytes_per_seq
    flops = decode_flops(model, batch, ctx, sch)
    # dequantisation work: weight-only low-bit weights are expanded to the compute dtype on CUDA cores (~2 ops/element: unpack+scale)
    n_w_read = w_active * 8.0 / max(sch.weight_bits_eff, 1e-9)
    dequant = 0.0
    if sch.weight_bits < 16 and sch.act in ("bf16", "fp16"):
        dequant = 2.0 * n_w_read
    elif sch.weight_bits < 8 and sch.act == "fp8":
        dequant = 1.0 * n_w_read
    if sch.kv_bits < 16:
        dequant += 2.0 * bytes_kv * 8.0 / max(sch.kv_bits, 1e-9)
    launches = kernels_per_layer * model.num_layers
    each = 1e-6 if cuda_graphs else chip.launch_us * 1e-6  # replayed graph nodes cost ~1 us each (assumption); eager launches cost launch_us
    ideal_t = (w_active + bytes_kv + sel) / chip.bw
    tb = t_pred(w_active + bytes_kv, flops, dequant, chip, dtype=sch.compute_dtype, eta=eta, launch_s=each, selector_bytes=sel, launches=launches)
    total_bytes = w_active + bytes_kv + sel
    intensity = flops / total_bytes if total_bytes else 0.0
    return DecodePrediction(
        batch=batch, ctx=ctx, scheme=sch.name, bytes_weights=w_active, bytes_kv=bytes_kv, bytes_total=total_bytes, flops=flops, dequant_ops=dequant,
        step_time_ideal_s=ideal_t, step_time_pred_s=tb.t, tok_s_stream_ideal=1.0 / ideal_t if ideal_t else math.inf, tok_s_stream_pred=1.0 / tb.t if tb.t else math.inf,
        tok_s_total_pred=batch / tb.t if tb.t else math.inf, bound=tb.bound, intensity_flop_per_byte=intensity,
        ridge_flop_per_byte=chip.ridge(sch.compute_dtype), breakdown=tb.to_dict(), notes=notes,
    )


def predict_prefill(model: ModelSpec, chip: ChipSpec, scheme: Scheme, *, tokens: int = 2048, batch: int = 1, eta: EtaFactors | None = None) -> dict[str, float]:
    """Time-to-first-token estimate: compute-bound GEMMs + quadratic attention, versus weight/KV traffic."""
    sch, notes = effective_scheme(scheme, chip)
    eta = eta or EtaFactors()
    flops = 2.0 * model.active_b * 1e9 * tokens * batch * (1.0 - sch.activation_sparsity)
    flops += 2.0 * batch * tokens * tokens * model.num_heads * model.head_dim * model.kv_layers  # causal QK^T + PV ~ 2 * T^2/2 * 2
    w = active_weight_bytes(model, sch, max(batch * tokens, 1))
    kv_write = batch * model.kv_bytes_for_context(tokens, sch.kv_bits)
    dq = 2.0 * (w * 8.0 / max(sch.weight_bits_eff, 1e-9)) if (sch.weight_bits < 16 and sch.act in ("bf16", "fp16")) else 0.0
    tb = t_pred(w + kv_write, flops, dq, chip, dtype=sch.compute_dtype, eta=eta)
    return {"ttft_s": tb.t, "bound": tb.bound, "flops": flops, "bytes": w + kv_write, "tok_s_prefill": tokens * batch / tb.t, "notes": notes}  # type: ignore[dict-item]
