"""``estimate``: the quick fit / bound / ceiling API behind the UI's recipe-builder panel and ``POST /planner/estimate``.

``estimate(model_cfg, chip, recipe_or_scheme, workload)`` answers, without any GPU or model weights:

* does it fit (Example A byte accounting), how much KV is left, how many sequences / tokens fit,
* which resource binds (capacity / bandwidth / compute / kv / accuracy),
* ideal and calibrated decode ceilings per (batch, context) with the bound type and the roofline ridge,
* the W4 batch crossover B*, and, if speculation is requested, the load-dependent speed-up curve.

Every number is *derived* from the cost model (``Provenance.derived``); nothing here is a measurement.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forge.core.schemas import Provenance, Recipe, Workload
from forge.planner import costmodel as cm
from forge.planner.search import BindingReport, FitReport, binding_resource, fits, workload_shape
from forge.planner.spec import ChipSpec, ModelSpec, Scheme, parse_scheme, resolve_chip


class DecodePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch: int = Field(description="Concurrent sequences.")
    ctx: int = Field(description="Context length per sequence.")
    step_ms_ideal: float = Field(description="Ideal bytes/bandwidth step time (ms).")
    step_ms_pred: float = Field(description="Calibrated prediction (prior eta unless calibrated).")
    tok_s_stream_ideal: float = Field(description="Ideal per-stream tokens/s.")
    tok_s_stream_pred: float = Field(description="Predicted per-stream tokens/s.")
    tok_s_total_pred: float = Field(description="Predicted aggregate tokens/s.")
    bound: str = Field(description="mem | mma | alu.")
    intensity: float = Field(description="Arithmetic intensity of the step (FLOP/byte).")
    bytes_weights_gb: float = Field(description="Weight bytes read per step (GB).")
    bytes_kv_gb: float = Field(description="KV bytes read per step (GB).")
    fits: bool = Field(description="Fits in HBM.")


class Estimate(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str = Field(description="Model name.")
    chip: str = Field(description="Chip name.")
    scheme: str = Field(description="Effective scheme name.")
    weight_bits_eff: float = Field(description="Effective bits per weight including scales.")
    kv_bits: float = Field(description="KV bits per element.")
    fit: FitReport = Field(description="Byte accounting at the workload.")
    binding: BindingReport = Field(description="Binding resource of the bf16 parent (what the scheme must relieve).")
    kv_bytes_per_token: float = Field(description="KV bytes per token per sequence.")
    ridge_flop_per_byte: float | None = Field(description="Roofline ridge point for the compute dtype.")
    b_star: float | None = Field(description="Weight-only batch crossover B* at datasheet FLOP rate.")
    b_star_effective: float | None = Field(description="B* with the prior FLOP/bandwidth efficiency ratio.")
    points: list[DecodePoint] = Field(description="Decode ceilings over the workload's (batch, context) grid.")
    speculation: list[dict[str, float]] = Field(default_factory=list, description="Speed-up vs batch for the requested speculation setting (assumed alpha, c).")
    spec_break_even_batch: float | None = Field(default=None, description="Batch above which speculation is a net slowdown (weights-only model).")
    warnings: list[str] = Field(default_factory=list, description="Emulation notes and caveats.")
    provenance: Provenance = Field(default=Provenance.derived, description="All numbers are derived from the cost model.")


def scheme_from_recipe(recipe: Recipe | dict[str, Any]) -> Scheme:
    """Infer the precision scheme of a recipe from its ptq / pack / kv / prune stages (best effort)."""
    r = recipe if isinstance(recipe, Recipe) else Recipe(**recipe)
    name = "bf16"
    kv = None
    sparsity = 0.0
    for st in r.stages:
        p = st.params
        if st.name == "ptq":
            wf = str(p.get("weight_format", "")).lower()
            bits = p.get("bits")
            if "nvfp4" in wf:
                name = "nvfp4"
            elif "mxfp4" in wf:
                name = "mxfp4"
            elif "fp8" in wf:
                name = "fp8"
            elif bits is not None and float(bits) == 4:
                g = int(p.get("group_size", 128))
                name = "w4a8" if "fp8" in str(p.get("act_format", "")).lower() else f"w4a16-g{g}"
            elif bits is not None and float(bits) == 8:
                name = "w8a16"
            elif bits is not None:
                name = f"w{int(float(bits))}a16-g{int(p.get('group_size', 128))}"
            if p.get("method") == "activation_sparsity":
                sparsity = float(p.get("sparsity", 0.0))
        elif st.name == "kv":
            d = str(p.get("dtype", "")).lower()
            kv = "fp8" if "fp8" in d else "4" if "4" in d else None
    s = parse_scheme(name)
    if kv:
        s = parse_scheme(f"{s.name}+kv{kv}")
    if sparsity:
        s = s.model_copy(update={"activation_sparsity": sparsity, "name": s.name + f"+sp{int(sparsity * 100)}"})
    return s


def estimate(
    model_cfg: Any,
    chip: Any,
    quant: Recipe | Scheme | str | dict | None = None,
    workload: Workload | dict | None = None,
    *,
    kv_dtype: str | None = None,
    eta: cm.EtaFactors | None = None,
    spec: dict[str, float] | None = None,
    overhead_gb: float | None = None,
) -> Estimate:
    """Quick fit / bound / ceiling estimate for the UI.

    ``quant``: a scheme string (``'w4a16-g128+kv8'``), :class:`Scheme`, :class:`Recipe` (or its dict) or None (bf16).
    ``kv_dtype`` ('fp8', '4', 'bf16') overrides the KV precision.  ``spec={'alpha': 0.8, 'gamma': 4, 'c': 0.1}`` adds the
    speculation speed-up curve (alpha / c are assumptions you provide)."""
    model = ModelSpec.from_config(model_cfg)
    csp = resolve_chip(chip)
    wl = workload if isinstance(workload, Workload) else Workload(**(workload or {}))
    if isinstance(quant, dict):
        scheme = scheme_from_recipe(quant)
    elif isinstance(quant, Recipe):
        scheme = scheme_from_recipe(quant)
    else:
        scheme = parse_scheme(quant)
    if kv_dtype:
        key = kv_dtype.lower()
        scheme = parse_scheme(f"{scheme.name.split('+kv')[0]}+kv{ {'fp8': 'fp8', '8': 'fp8', 'int8': 'fp8', '4': '4', 'int4': '4', 'kv4': '4', 'bf16': '16', '16': '16', 'fp16': '16'}.get(key, key) }")
    eff, notes = cm.effective_scheme(scheme, csp)
    eta = eta or cm.EtaFactors()
    B, T = workload_shape(wl)
    fit = fits(model, csp, scheme, wl, overhead_gb=overhead_gb)
    bind = binding_resource(model, csp, wl, eta=eta)
    pts: list[DecodePoint] = []
    batches = sorted(set(wl.batch_sizes or [1]))
    ctxs = sorted(set(wl.context_lengths or [4096]))
    for b in batches:
        for c in ctxs:
            p = cm.predict_decode(model, csp, scheme, batch=b, ctx=c, eta=eta)
            f = fits(model, csp, scheme, wl, batch=b, ctx=c, overhead_gb=overhead_gb)
            pts.append(DecodePoint(batch=b, ctx=c, step_ms_ideal=p.step_time_ideal_s * 1e3, step_ms_pred=p.step_time_pred_s * 1e3, tok_s_stream_ideal=p.tok_s_stream_ideal,
                                   tok_s_stream_pred=p.tok_s_stream_pred, tok_s_total_pred=p.tok_s_total_pred, bound=p.bound, intensity=p.intensity_flop_per_byte,
                                   bytes_weights_gb=p.bytes_weights / 1e9, bytes_kv_gb=p.bytes_kv / 1e9, fits=f.fits))
    dtype = eff.compute_dtype
    peak = csp.peak_flops(dtype) or csp.peak_flops("bf16") or csp.peak_flops("fp16")
    bstar = cm.batch_crossover(peak, csp.bw, eff.weight_bits_eff) if peak else None
    bstar_eff = cm.batch_crossover(peak, csp.bw, eff.weight_bits_eff, eta=eta.eta_mma / max(eta.eta_mem, 1e-9)) if peak else None
    spec_rows: list[dict[str, float]] = []
    be = None
    if spec and bstar_eff:
        a, g, c = float(spec.get("alpha", 0.7)), int(spec.get("gamma", 4)), float(spec.get("c", 0.1))
        be = cm.spec_break_even_batch(g, a, bstar_eff)
        for b in sorted(set(batches + [1, 2, 4, 8, 16, 32, 64, 128, 256, 512])):
            spec_rows.append({"batch": float(b), "speedup": cm.spec_speedup_batched(b, g, a, bstar_eff), "memory_bound_speedup": cm.spec_speedup(g, a, c)})
        notes = notes + [f"speculation curve uses ASSUMED alpha={a}, gamma={g}, c={c}"]
    return Estimate(
        model_name=model.name or "model", chip=csp.name, scheme=eff.name, weight_bits_eff=eff.weight_bits_eff, kv_bits=eff.kv_bits, fit=fit, binding=bind,
        kv_bytes_per_token=model.kv_bytes_per_token(eff.kv_bits), ridge_flop_per_byte=csp.ridge(dtype), b_star=bstar, b_star_effective=bstar_eff, points=pts,
        speculation=spec_rows, spec_break_even_batch=be, warnings=notes + ([csp.notes] if csp.provenance == Provenance.vendor and csp.notes and "APPROXIMATE" in csp.notes else []),
    )
