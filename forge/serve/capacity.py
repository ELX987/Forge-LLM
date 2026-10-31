"""Capacity planner (Plan SV-5, Worked examples A and B) and plot-ready simulators for the UI.

Everything here is *derived* arithmetic (provenance ``derived``): byte accounting of weights / KV / activations on a
chip, maximum concurrency from the leftover memory (shared prefixes counted once), and roofline decode ceilings
``tokens/s = B / max(bytes/BW, 2*params*B/peak)``. Chip numbers come from the descriptor / capabilities you pass, from
``forge.hw.chips`` (datasheet, NOT measured) or from explicit values; measured values from the hardware probe should be
preferred by the caller.

Unit convention (matches Plan Example A): GB = 1e9 bytes, KiB = 1024 bytes
(``43 GB / 320 KiB = 131 k tokens``).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from forge.core.errors import ConfigError

KIB = 1024
GB = 1e9

#: KV storage bits per element and the extra bytes-per-element fraction spent on scales/zero points.
KV_BITS = {"bf16": 16, "fp16": 16, "fp32": 32, "fp8": 8, "fp8_e4m3": 8, "int8": 8, "int4": 4}
KV_SCALE_OVERHEAD = {"bf16": 0.0, "fp16": 0.0, "fp32": 0.0, "fp8": 0.0, "fp8_e4m3": 0.0, "int8": 0.0625, "int4": 0.125}


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, kv_dtype: str = "bf16",
                       scale_overhead: float | None = None) -> float:
    """``2 * layers * kv_heads * head_dim * bits/8 * (1 + overhead)`` (Llama-3.1-70B @16 bit = 320 KiB)."""
    if kv_dtype not in KV_BITS:
        raise ConfigError(f"unknown kv_dtype {kv_dtype!r}; expected one of {sorted(KV_BITS)}")
    ov = KV_SCALE_OVERHEAD[kv_dtype] if scale_overhead is None else scale_overhead
    return 2.0 * num_layers * num_kv_heads * head_dim * KV_BITS[kv_dtype] / 8.0 * (1.0 + ov)


def weight_bytes(params: float, bits: float = 16.0, *, group_size: int | None = 128, scale_bits: int = 16,
                 zero_bits: int = 0, high_precision_params: float = 0.0, high_precision_bits: float = 16.0) -> float:
    """Bytes of a weight-quantised model: packed payload + per-group scales/zeros + layers kept at high precision."""
    quant = max(params - high_precision_params, 0.0)
    b = quant * bits / 8.0
    if bits < 16 and group_size:
        b += quant / group_size * (scale_bits + zero_bits) / 8.0
    b += high_precision_params * high_precision_bits / 8.0
    return b


# --------------------------------------------------------------------------------------------------
# model / chip descriptions
# --------------------------------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    params: float                      # total parameters
    num_layers: int
    num_kv_heads: int
    head_dim: int
    active_params: float | None = None  # per-token active parameters (MoE); default = params
    hidden_size: int | None = None
    moe_experts: int = 0
    moe_top_k: int = 0
    expert_param_fraction: float = 0.0  # fraction of params living in routed experts
    kv_layers: int | None = None        # layers that own a KV cache (hybrid / KV-shared models); None = num_layers
    state_bytes_per_seq: float = 0.0    # context-independent per-sequence recurrent state (lfm2 conv layers)
    kv_elems_per_token: float | None = None  # K+V elements per token summed over layers (per-layer geometry: gemma4 / hybrid); None = uniform

    @property
    def active(self) -> float:
        return self.params if self.active_params is None else self.active_params

    def kv_bytes_per_token(self, kv_dtype: str = "bf16", scale_overhead: float | None = None) -> float:
        if self.kv_elems_per_token is not None:
            return kv_bytes_per_token(1, 1, self.kv_elems_per_token / 2.0, kv_dtype, scale_overhead)
        n = self.num_layers if self.kv_layers is None else self.kv_layers
        return kv_bytes_per_token(n, self.num_kv_heads, self.head_dim, kv_dtype, scale_overhead)

    @classmethod
    def from_any(cls, m: Any) -> ModelSpec:
        if isinstance(m, ModelSpec):
            return m
        if isinstance(m, dict):
            if "params" in m:
                return cls(**{k: v for k, v in m.items() if k in cls.__dataclass_fields__})  # type: ignore[attr-defined]
            from forge.models.config import ModelConfig

            m = ModelConfig.from_hf_config(m) if "num_hidden_layers" in m else ModelConfig(**m)
        # ModelConfig-like
        layers = int(getattr(m, "num_layers", None) or m.num_hidden_layers)
        kvh = int(getattr(m, "kv_heads", None) or getattr(m, "num_kv_heads", None) or m.num_heads)
        hd = int(getattr(m, "head_dim", None) or (m.hidden_size // m.num_heads))
        params = float(m.param_count()) if hasattr(m, "param_count") else float(m.params)
        active = float(m.active_param_count()) if hasattr(m, "active_param_count") else None
        moe = getattr(m, "moe", None)
        experts = int(getattr(moe, "num_experts", 0) or 0) if moe is not None else 0
        topk = int(getattr(moe, "top_k", 0) or 0) if moe is not None else 0
        frac = 0.0
        if experts and active is not None and params > 0:
            # active = shared + (top_k/E) * routed  and  params = shared + routed
            denom = 1.0 - topk / experts
            routed = (params - active) / denom if denom > 0 else 0.0
            frac = max(0.0, min(1.0, routed / params))
        kvl = getattr(m, "kv_layers", None)
        kvb = getattr(m, "kv_bytes_per_token", None)
        elems = float(kvb(16)) / 2.0 if callable(kvb) else None  # per-layer aware (gemma4 widths, KV sharing, conv layers)
        state = getattr(m, "recurrent_state_bytes_per_seq", None)
        return cls(name=str(getattr(m, "arch", "model")), params=params, num_layers=layers, num_kv_heads=kvh, head_dim=hd,
                   active_params=active, hidden_size=getattr(m, "hidden_size", None), moe_experts=experts, moe_top_k=topk,
                   expert_param_fraction=frac, kv_layers=len(kvl) if isinstance(kvl, (list, tuple)) else None,
                   state_bytes_per_seq=float(state()) if callable(state) else 0.0, kv_elems_per_token=elems)


@dataclass
class ChipSpec:
    name: str
    hbm_gb: float                       # decimal GB
    bw_gbs: float | None                # GB/s (decimal)
    tflops: dict[str, float] = field(default_factory=dict)  # dense tensor TFLOP/s by dtype ("bf16", "fp16", "fp8", "fp4")
    price_per_hour: float | None = None
    provenance: str = "vendor"

    def peak_flops(self, compute: str = "bf16") -> float | None:
        v = self.tflops.get(compute) or self.tflops.get("bf16") or self.tflops.get("fp16")
        return v * 1e12 if v else None

    @classmethod
    def from_any(cls, chip: Any, **overrides: Any) -> ChipSpec:
        """Accepts ChipSpec, a chip name/key, a ``forge.hw.chips`` spec dict, a ``ChipDescriptor`` or ``Capabilities``."""
        spec: ChipSpec
        if isinstance(chip, ChipSpec):
            spec = chip
        elif isinstance(chip, str):
            spec = cls._from_name(chip)
        elif isinstance(chip, dict):
            spec = cls._from_spec_dict(chip)
        elif hasattr(chip, "memory") and hasattr(chip, "matrix_units"):  # ChipDescriptor
            mem = chip.memory
            bw = getattr(mem.hbm_bw_gbs, "measured", None) or getattr(mem.hbm_bw_gbs, "datasheet", None) \
                if not isinstance(mem.hbm_bw_gbs, (int, float)) else mem.hbm_bw_gbs
            tf: dict[str, float] = {}
            for mu in chip.matrix_units:
                pk = mu.peak_tflops
                v = (getattr(pk, "measured", None) or getattr(pk, "datasheet", None)) if not isinstance(pk, (int, float)) else pk
                if v:
                    tf[str(mu.dtype).lower()] = float(v)
            spec = cls(str(getattr(chip, "name", "chip")), float(mem.hbm_gb), float(bw) if bw else None, tf, None, "measured")
        elif hasattr(chip, "total_mem_gb") and hasattr(chip, "sm"):  # Capabilities
            base = None
            key = getattr(chip, "chip_key", None)
            try:
                from forge.hw.chips import KNOWN_CHIPS, find_chip

                if key and key in KNOWN_CHIPS:
                    base = cls._from_spec_dict(KNOWN_CHIPS[key])
                else:
                    base = cls._from_spec_dict(find_chip(chip.name, sm=chip.sm).spec)
            except Exception:
                base = None
            bw = getattr(chip, "mem_bw_gbs_datasheet", None)
            if base is None:
                base = cls(chip.name, float(chip.total_mem_gb) * 1.073741824, bw, {}, None, "vendor")
            spec = cls(chip.name, float(chip.total_mem_gb) * 1.073741824 if chip.total_mem_gb else base.hbm_gb,
                       bw or base.bw_gbs, base.tflops, None, base.provenance)
        else:
            raise ConfigError(f"cannot interpret chip {chip!r}")
        if overrides:
            d = asdict(spec)
            d.update({k: v for k, v in overrides.items() if v is not None})
            spec = cls(**d)
        return spec

    @classmethod
    def _from_name(cls, name: str) -> ChipSpec:
        from forge.hw.chips import KNOWN_CHIPS, find_chip

        if name in KNOWN_CHIPS:
            return cls._from_spec_dict(KNOWN_CHIPS[name])
        m = find_chip(name)
        if m.kind == "unknown":
            raise ConfigError(f"unknown chip {name!r}")
        return cls._from_spec_dict(m.spec)

    @staticmethod
    def _from_spec_dict(d: dict[str, Any]) -> ChipSpec:
        return ChipSpec(str(d.get("display") or d.get("name") or d.get("key") or "chip"), float(d.get("mem_gb") or 0.0),
                        float(d["mem_bw_gbs"]) if d.get("mem_bw_gbs") else None,
                        {k: float(v) for k, v in (d.get("tflops") or {}).items() if v}, None, "vendor")


def ridge_for_device_name(name: str, compute: str = "bf16") -> float:
    """FLOP per byte at the roofline ridge for a device name (datasheet; 250 when unknown)."""
    try:
        c = ChipSpec.from_any(name)
        pf = c.peak_flops(compute)
        if pf and c.bw_gbs:
            return pf / (c.bw_gbs * 1e9)
    except Exception:
        pass
    return 250.0


# --------------------------------------------------------------------------------------------------
# context distribution
# --------------------------------------------------------------------------------------------------
@dataclass
class ContextDist:
    """KV tokens held per live sequence. ``prefix_tokens`` of every sequence are a shared prefix stored once."""

    mean_ctx: float = 4096.0
    p99_ctx: float | None = None
    prefix_tokens: float = 0.0
    output_tokens: float = 0.0   # informational

    @classmethod
    def from_samples(cls, lengths: Sequence[float], prefix_tokens: float = 0.0) -> ContextDist:
        if not len(lengths):
            raise ConfigError("empty context sample")
        s = sorted(lengths)
        return cls(mean_ctx=sum(s) / len(s), p99_ctx=s[min(len(s) - 1, int(0.99 * (len(s) - 1)))], prefix_tokens=prefix_tokens)


def fleet_capacity(free_kv_tokens: float, ctx_len: float, shared_prefix: float = 0.0, block_size: int = 16) -> dict[str, float]:
    """How many agents/sessions fit when each holds ``ctx_len`` tokens of which ``shared_prefix`` is common.

    ``N_shared = floor((T - P) / (L - P))`` with P and (L - P) rounded up to whole KV blocks, versus
    ``N_naive = floor(T / L)`` (no sharing). (Plan SV-7 fleet capacity; paper P3.)"""
    bs = block_size
    pb = math.ceil(shared_prefix / bs) * bs if shared_prefix > 0 else 0
    private = math.ceil(max(ctx_len - shared_prefix, 1) / bs) * bs
    naive_each = math.ceil(ctx_len / bs) * bs
    n_naive = int(free_kv_tokens // naive_each)
    n_shared = int(max(free_kv_tokens - pb, 0) // private)
    return {"agents": float(n_shared), "agents_without_sharing": float(n_naive),
            "sharing_gain": (n_shared / n_naive) if n_naive else float("inf"),
            "kv_tokens_used": float(pb + n_shared * private), "private_tokens_each": float(private),
            "shared_prefix_tokens": float(pb)}


# --------------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------------
@dataclass
class CapacityReport:
    model: str
    chip: str
    num_gpus: int
    weight_bits: float
    kv_dtype: str
    provenance: str
    hbm_gb: float
    weights_gb: float
    activations_gb: float
    overhead_gb: float
    free_for_kv_gb: float
    kv_bytes_per_token: float
    kv_tokens: float
    fits: bool
    mean_ctx: float
    shared_prefix_tokens: float
    max_concurrency: float
    max_concurrency_no_sharing: float
    decode_bytes_per_step_b1: float
    ideal_decode_ms_b1: float | None
    ideal_decode_tok_s_b1: float | None
    batch_crossover: float | None
    tok_s_at_max_concurrency: float | None
    step_ms_at_max_concurrency: float | None
    itl_slo_ms: float | None
    max_concurrency_under_slo: float | None
    cost_per_million_tokens: float | None
    breakdown_gb: dict[str, float]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _expert_coverage(experts: int, top_k: int, batch: float) -> float:
    """Expected fraction of routed experts touched by ``batch`` tokens with uniform top-k routing."""
    if experts <= 0 or top_k <= 0:
        return 1.0
    return 1.0 - (1.0 - top_k / experts) ** max(batch, 1.0)


def decode_step(model: ModelSpec, chip: ChipSpec, *, weight_bytes_total: float, kv_bpt: float, batch: float, ctx: float,
                compute: str = "bf16", overhead_ms: float = 0.0, num_gpus: int = 1) -> dict[str, float | None]:
    """Roofline decode step for ``batch`` sequences of context ``ctx``: returns bytes, times (ms) and bound type."""
    if model.moe_experts and model.expert_param_fraction > 0:
        shared = weight_bytes_total * (1 - model.expert_param_fraction)
        routed = weight_bytes_total * model.expert_param_fraction * _expert_coverage(model.moe_experts, model.moe_top_k, batch)
        w = shared + routed
    else:
        w = weight_bytes_total
    byts = w + batch * ctx * kv_bpt
    t_mem = byts / (chip.bw_gbs * 1e9 * num_gpus) if chip.bw_gbs else None
    pf = chip.peak_flops(compute)
    t_cmp = 2.0 * model.active * batch / (pf * num_gpus) if pf else None
    cands = [t for t in (t_mem, t_cmp) if t is not None]
    t = max(cands) if cands else None
    step_ms = (t * 1000.0 + overhead_ms) if t is not None else None
    return {"bytes": byts, "t_mem_ms": t_mem * 1000 if t_mem else None, "t_compute_ms": t_cmp * 1000 if t_cmp else None,
            "step_ms": step_ms, "bound": ("compute" if (t_cmp and t_mem and t_cmp > t_mem) else "memory"),
            "tok_s": (batch / (step_ms / 1000.0)) if step_ms else None}


def capacity_plan(model_cfg: Any, chip: Any, weight_bits: float = 16.0, kv_dtype: str = "bf16",
                  context_dist: ContextDist | float | None = None, *, group_size: int | None = 128,
                  weight_gb: float | None = None, num_gpus: int = 1, gpu_mem_util: float = 1.0,
                  activation_gb: float = 0.0, runtime_overhead_gb: float = 0.0, block_size: int = 16,
                  itl_slo_ms: float | None = None, price_per_hour: float | None = None, compute: str = "bf16",
                  step_overhead_ms: float = 0.0, selector_gb: float = 0.0, high_precision_params: float = 0.0,
                  kv_scale_overhead: float | None = None, bandwidth_gbs: float | None = None) -> CapacityReport:
    """Plan SV-5: weights / KV / free memory, max concurrency, decode ceilings and cost.

    * ``weight_gb`` overrides the computed weight size (use the measured artifact size).
    * ``context_dist``: :class:`ContextDist` or a mean context length (tokens held per sequence, incl. shared prefix).
    * ``gpu_mem_util`` scales usable HBM; ``activation_gb``/``runtime_overhead_gb``/``selector_gb`` are subtracted
      (Example A ignores them: pass the defaults of 0 and ``gpu_mem_util=1`` to reproduce the table).
    """
    model = ModelSpec.from_any(model_cfg)
    cs = ChipSpec.from_any(chip, bw_gbs=bandwidth_gbs, price_per_hour=price_per_hour)
    if isinstance(context_dist, (int, float)):
        context_dist = ContextDist(mean_ctx=float(context_dist))
    cd = context_dist or ContextDist()
    notes: list[str] = []

    w_bytes = weight_gb * GB if weight_gb is not None else weight_bytes(
        model.params, weight_bits, group_size=group_size, high_precision_params=high_precision_params)
    hbm = cs.hbm_gb * GB * num_gpus * gpu_mem_util
    act = activation_gb * GB
    ovh = runtime_overhead_gb * GB
    sel = selector_gb * GB
    free = hbm - w_bytes - act - ovh - sel
    fits = free > 0
    if not fits:
        notes.append("weights + overheads do not fit in memory")
    kvb = model.kv_bytes_per_token(kv_dtype, kv_scale_overhead)
    kv_tokens = max(free, 0.0) / kvb
    fc = fleet_capacity(kv_tokens, cd.mean_ctx, cd.prefix_tokens, block_size)
    max_conc = fc["agents"] if cd.prefix_tokens > 0 else float(int(kv_tokens // (math.ceil(cd.mean_ctx / block_size) * block_size)))
    no_share = fc["agents_without_sharing"]
    if model.state_bytes_per_seq > 0 and cd.prefix_tokens <= 0:  # hybrid: every sequence also pins a fixed conv state
        per_seq = math.ceil(cd.mean_ctx / block_size) * block_size * kvb + model.state_bytes_per_seq
        max_conc = float(int(max(free, 0.0) // per_seq))
        notes.append(f"hybrid model: {model.state_bytes_per_seq / 1024:.1f} KiB of conv state per sequence is included in max concurrency")

    b1 = decode_step(model, cs, weight_bytes_total=w_bytes, kv_bpt=kvb, batch=1, ctx=0.0, compute=compute,
                     overhead_ms=0.0, num_gpus=num_gpus)
    ideal_ms = (b1["t_mem_ms"] or None) if b1["t_mem_ms"] is not None else None
    ideal_toks = (1000.0 / ideal_ms) if ideal_ms else None
    pf = cs.peak_flops(compute)
    b_star = None
    if pf and cs.bw_gbs and model.active:
        b_star = pf * w_bytes / (2.0 * model.active * cs.bw_gbs * 1e9)

    at_max = None
    if cs.bw_gbs and max_conc >= 1:
        at_max = decode_step(model, cs, weight_bytes_total=w_bytes, kv_bpt=kvb, batch=max_conc, ctx=cd.mean_ctx,
                             compute=compute, overhead_ms=step_overhead_ms, num_gpus=num_gpus)
    under_slo = None
    if itl_slo_ms and cs.bw_gbs and max_conc >= 1:
        lo, hi = 0, int(max_conc)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            st = decode_step(model, cs, weight_bytes_total=w_bytes, kv_bpt=kvb, batch=mid, ctx=cd.mean_ctx, compute=compute,
                             overhead_ms=step_overhead_ms, num_gpus=num_gpus)
            if st["step_ms"] is not None and st["step_ms"] <= itl_slo_ms:
                lo = mid
            else:
                hi = mid - 1
        under_slo = float(lo)
    cost = None
    if cs.price_per_hour and at_max and at_max["tok_s"]:
        cost = cs.price_per_hour * num_gpus / (at_max["tok_s"] * 3600.0) * 1e6
    kv_used = min(kv_tokens, fc["kv_tokens_used"] if cd.prefix_tokens > 0 else max_conc * math.ceil(cd.mean_ctx / block_size) * block_size) * kvb
    breakdown = {"weights": w_bytes / GB, "kv_cache_used_at_max_concurrency": kv_used / GB,
                 "kv_cache_free_slack": max(free - kv_used, 0.0) / GB, "selector": sel / GB, "activations": act / GB,
                 "runtime_overhead": ovh / GB}
    if cs.bw_gbs is None:
        notes.append("chip bandwidth unknown: throughput figures omitted")
    notes.append("derived from datasheet-level chip numbers; replace with probe-measured values for tighter bounds"
                 if cs.provenance != "measured" else "uses probe-measured chip numbers")
    return CapacityReport(
        model=model.name, chip=cs.name, num_gpus=num_gpus, weight_bits=weight_bits, kv_dtype=kv_dtype, provenance="derived",
        hbm_gb=hbm / GB, weights_gb=w_bytes / GB, activations_gb=act / GB, overhead_gb=ovh / GB, free_for_kv_gb=free / GB,
        kv_bytes_per_token=kvb, kv_tokens=kv_tokens, fits=fits, mean_ctx=cd.mean_ctx, shared_prefix_tokens=cd.prefix_tokens,
        max_concurrency=max_conc, max_concurrency_no_sharing=no_share, decode_bytes_per_step_b1=float(b1["bytes"] or 0.0),
        ideal_decode_ms_b1=ideal_ms, ideal_decode_tok_s_b1=ideal_toks, batch_crossover=b_star,
        tok_s_at_max_concurrency=at_max["tok_s"] if at_max else None,
        step_ms_at_max_concurrency=at_max["step_ms"] if at_max else None, itl_slo_ms=itl_slo_ms,
        max_concurrency_under_slo=under_slo, cost_per_million_tokens=cost, breakdown_gb=breakdown, notes=notes)


# --------------------------------------------------------------------------------------------------
# simulators for the UI (plot-ready lists)
# --------------------------------------------------------------------------------------------------
def speculation_gain_vs_batch(*, alpha: float = 0.7, draft_cost_ratio: float = 0.05, model: Any | None = None,
                              chip: Any | None = None, weight_bits: float = 16.0, kv_dtype: str = "bf16",
                              batches: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256), ctx: float = 1024.0,
                              max_gamma: int = 8, position_decay: float = 1.0, hysteresis: float = 0.03,
                              ridge_flop_per_byte: float | None = None, weight_gb: float | None = None) -> dict[str, Any]:
    """Modelled speculation speed-up vs batch size (the load-aware controller's trade-off, Plan 7.5 / P7 5).

    Conditional acceptance at draft position ``k`` is ``alpha * position_decay**k``. Without ``model``/``chip`` a
    generic ``ridge`` of 250 FLOP/B and a 8B-parameter bf16 model are used. Returns lists indexed like ``batch``:
    ``best_gamma``, ``gain`` (best modelled speed-up, >= 1) and ``gain_by_gamma[g]``."""
    from forge.serve.spec import SpecController, StepCostModel

    if model is not None and chip is not None:
        ms, cs = ModelSpec.from_any(model), ChipSpec.from_any(chip)
        wb = weight_gb * GB if weight_gb is not None else weight_bytes(ms.params, weight_bits)
        pf = cs.peak_flops()
        ridge = ridge_flop_per_byte or ((pf / (cs.bw_gbs * 1e9)) if pf and cs.bw_gbs else 250.0)
        cost = StepCostModel(weight_bytes=wb, params=ms.active, ridge_flop_per_byte=ridge)
        kv_bpt = ms.kv_bytes_per_token(kv_dtype)
    else:
        cost = StepCostModel(weight_bytes=16e9, params=8e9, ridge_flop_per_byte=ridge_flop_per_byte or 250.0)
        kv_bpt = 128 * KIB
    ctrl = SpecController(max_gamma, cost=cost, draft_cost_ratio=draft_cost_ratio, hysteresis=hysteresis, max_batch_for_spec=10**9)
    alphas = [min(1.0, max(0.0, alpha * position_decay ** k)) for k in range(max_gamma)]
    ctrl.est._a["sim"] = alphas
    ctrl.est._n["sim"] = [1] * max_gamma
    best_g, best_gain = [], []
    by_g: dict[int, list[float]] = {g: [] for g in range(1, max_gamma + 1)}
    for b in batches:
        kvb = b * ctx * kv_bpt
        g = ctrl.choose_gamma("sim", b, kvb)
        best_g.append(g)
        best_gain.append(ctrl.modelled_gain(alphas, g, b, kvb) if g else 1.0)
        for gg in by_g:
            by_g[gg].append(ctrl.modelled_gain(alphas, gg, b, kvb))
    return {"batch": list(batches), "best_gamma": best_g, "gain": best_gain, "gain_by_gamma": by_g, "alpha": alpha,
            "position_decay": position_decay, "draft_cost_ratio": draft_cost_ratio, "ridge_flop_per_byte": cost.ridge_flop_per_byte,
            "provenance": "derived"}


def capacity_curve(model_cfg: Any, chip: Any, weight_bits: float = 16.0, kv_dtype: str = "bf16", *,
                   contexts: Sequence[int] = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072),
                   itl_slo_ms: float | None = None, shared_prefix: float = 0.0, **plan_kw: Any) -> dict[str, Any]:
    """Max concurrency, aggregate tokens/s and step time vs context length (plot-ready arrays)."""
    out: dict[str, Any] = {"context": list(contexts), "max_concurrency": [], "tokens_per_s": [], "step_ms": [],
                           "kv_gb_per_seq": [], "max_concurrency_under_slo": [], "kv_tokens": [], "provenance": "derived"}
    for c in contexts:
        r = capacity_plan(model_cfg, chip, weight_bits, kv_dtype, ContextDist(mean_ctx=float(c), prefix_tokens=min(shared_prefix, c)),
                          itl_slo_ms=itl_slo_ms, **plan_kw)
        out["max_concurrency"].append(r.max_concurrency)
        out["tokens_per_s"].append(r.tok_s_at_max_concurrency)
        out["step_ms"].append(r.step_ms_at_max_concurrency)
        out["kv_gb_per_seq"].append(c * r.kv_bytes_per_token / GB)
        out["max_concurrency_under_slo"].append(r.max_concurrency_under_slo)
        out["kv_tokens"].append(r.kv_tokens)
    return out


def throughput_vs_batch(model_cfg: Any, chip: Any, weight_bits: float = 16.0, kv_dtype: str = "bf16", *, ctx: float = 2048.0,
                        batches: Sequence[int] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512), compute: str = "bf16",
                        weight_gb: float | None = None, num_gpus: int = 1) -> dict[str, Any]:
    """Roofline tokens/s and step time vs batch at fixed context (shows the memory->compute crossover B*)."""
    ms, cs = ModelSpec.from_any(model_cfg), ChipSpec.from_any(chip)
    wb = weight_gb * GB if weight_gb is not None else weight_bytes(ms.params, weight_bits)
    kvb = ms.kv_bytes_per_token(kv_dtype)
    res = [decode_step(ms, cs, weight_bytes_total=wb, kv_bpt=kvb, batch=b, ctx=ctx, compute=compute, num_gpus=num_gpus) for b in batches]
    return {"batch": list(batches), "tokens_per_s": [r["tok_s"] for r in res], "step_ms": [r["step_ms"] for r in res],
            "bound": [r["bound"] for r in res], "provenance": "derived"}
