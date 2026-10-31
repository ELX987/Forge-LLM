"""Recipe stages of the compression engine, registered in ``forge.core.registry`` under kind ``"stage"`` (Plan CP-12, Appendix C.1).

Registered names: ``calibrate, ptq, pack, kv, rotate, prune, lowrank, delta, lossless, eval, gate, distill, finetune``.

Contract (see ``forge.core.recipe``): ``fn(inputs, params, ctx) -> dict``.  The recipe namespace carries

``model``       digest of the current model (live object in :mod:`forge.compress.state`; a store manifest hash when a store is present),
``base_model``  digest of the recipe's original input (default parent for ``eval`` / ``gate``),
``calib``       descriptor produced by ``calibrate`` (set name + content hash + sizes),
``teacher``     set by ``prune``/``lowrank`` to the pre-compression model (default teacher of ``distill``).

Model-producing stages work on a deep copy of the model while it is smaller than ``copy_below_gb`` (earlier digests stay valid for
``eval``/``gate``), larger models are modified in place.  Parameter models are in :data:`forge.compress.config.STAGE_PARAM_MODELS`.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.compress import state
from forge.compress.calibrate import StatsCache, collect_stats, get_calibration_set, get_calibration_tokens
from forge.compress.config import PARAM_ALIASES, STAGE_PARAM_MODELS
from forge.compress.modelutil import count_params, model_nbytes
from forge.core.context import RunContext
from forge.core.errors import ConfigError, ForgeError, GateFailed, LicenceBlocked, NotFound
from forge.core.registry import register

log = logging.getLogger("forge.compress.stages")

STATS_CACHE = StatsCache()  # process-wide; keyed (model fingerprint, set hash, layer)
PERMISSIVE_LICENCES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "cc-by-4.0", "unlicense"}


# ------------------------------------------------------------------------------------------------ helpers
def _params(stage: str, params: dict[str, Any]) -> Any:
    p = dict(params or {})
    for src, dst in PARAM_ALIASES.get(stage, {}).items():
        if src in p and dst not in p and src != dst:
            p[dst] = p.pop(src)
    if stage == "kv" and p.get("scheme") in ("fp8", "e4m3"):
        p["scheme"] = "fp8_e4m3"
    if stage == "kv" and p.get("scheme") == "e5m2":
        p["scheme"] = "fp8_e5m2"
    try:
        return STAGE_PARAM_MODELS[stage].model_validate(p)
    except Exception as e:
        raise ConfigError(f"invalid parameters for stage '{stage}': {e}") from e


def _meta(inputs: dict[str, Any]) -> dict[str, Any]:
    return inputs.get("_recipe", {})


def _cur(inputs: dict[str, Any], ctx: RunContext, key: str = "model") -> tuple[str, state.LiveModel]:
    d = inputs.get(key)
    if d is None:
        raise ConfigError(f"recipe namespace has no '{key}' (set the recipe input '{key}' or run a stage that produces it)")
    return d, state.resolve_model(d, ctx)


def _fork(lm: state.LiveModel) -> nn.Module:
    """Model to modify: a deep copy below the copy threshold, else the live object itself."""
    if model_nbytes(lm.model) < state.options().get("copy_below_bytes", 2 << 30):
        return copy.deepcopy(lm.model)
    return lm.model


def _is_compressed(model: nn.Module) -> bool:
    from forge.compress.modelutil import iter_packed_linears

    return any(True for _ in iter_packed_linears(model))


def _emit(
    inputs: dict[str, Any], ctx: RunContext, stage: str, params: dict[str, Any], parent: str, parent_lm: state.LiveModel, model: nn.Module,
    *, report: dict[str, Any] | None = None, meta: dict[str, Any] | None = None, tokenizer: Any = None,
) -> str:
    """Register ``model`` as the output of ``stage``: persist to the store (digest = manifest hash) or derive a pure digest; make it live."""
    rm = _meta(inputs)
    idx = int(rm.get("stage_index", 0))
    tok = tokenizer if tokenizer is not None else parent_lm.tokenizer
    new_meta = {**parent_lm.meta, **(meta or {})}
    extra = {"stage": stage, "stage_index": idx, "recipe": rm.get("name"), "recipe_hash": rm.get("hash"), "parent": parent,
             "licence": new_meta.get("licence"), "distill_allowed": new_meta.get("distill_allowed"), "kv": new_meta.get("kv"),
             "report": _jsonable(report) if report else None}
    kind = "quantized" if _is_compressed(model) else "model"
    h = state.persist_model(ctx, model, tok, name=f"{idx:02d}_{stage}", kind=kind, parents=[parent], recipe_hash=rm.get("hash"), extra=_jsonable(extra),
                            report=_jsonable(report) if report else None, licence=new_meta.get("licence"))
    digest = h or state.stage_digest(parent, stage, params)
    if digest == parent:  # a stage that changed nothing material still needs a distinct identity when metadata changed
        digest = state.stage_digest(parent, stage, {**params, "_i": idx})
    new_meta["artifact"] = h
    state.put_live(digest, model, tok, **new_meta)
    return digest


def _jsonable(obj: Any) -> Any:
    try:
        return json.loads(json.dumps(obj, default=_default))
    except Exception:
        return str(obj)


def _default(o: Any) -> Any:
    if isinstance(o, torch.Tensor):
        return o.tolist() if o.numel() <= 64 else f"<tensor {tuple(o.shape)}>"
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    if hasattr(o, "__dict__"):
        return {k: v for k, v in vars(o).items() if not k.startswith("_")}
    return str(o)


def _calib_tokens(inputs: dict[str, Any], model: nn.Module, tokenizer: Any, n_seqs: int | None = None, seq_len: int | None = None, seed: int | None = None):
    d = inputs.get("calib")
    desc = d if isinstance(d, dict) else {}
    name = desc.get("set") or (d if isinstance(d, str) else None) or "synthetic"
    return get_calibration_tokens(name, model, tokenizer, n_seqs=n_seqs or desc.get("n_seqs", 64), seq_len=seq_len or desc.get("seq_len", 256),
                                  seed=desc.get("seed", 0) if seed is None else seed)


def _calib_defaults(inputs: dict[str, Any], keys: tuple[str, ...] = ("n_seqs", "seq_len", "seed")) -> dict[str, Any]:
    d = inputs.get("calib")
    return {k: d[k] for k in keys if isinstance(d, dict) and k in d}


def _stage(name: str, inputs: list[str], doc: str, version: str = "1"):
    def deco(fn):
        fn.code_version = version
        register("stage", name, meta={"inputs": inputs, "code_version": version, "doc": doc})(fn)
        return fn

    return deco


# ------------------------------------------------------------------------------------------------ calibrate
@_stage("calibrate", ["model", "calib"], "Choose + fingerprint the calibration set; optionally pre-collect statistics.")
def stage_calibrate(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    p = _params("calibrate", params)
    raw_in = inputs.get("calib")
    name = p.set or (raw_in if isinstance(raw_in, str) else None) or "synthetic"
    try:
        cs = get_calibration_set(name)
    except Exception as e:
        raise ConfigError(f"calibration set {name!r} is not registered ({e}). Register it with forge.compress.calibrate.register_calibration_set "
                          "or give calibration_texts / calibration_files in the job config.") from e
    desc = {"set": cs.name, "set_hash": cs.content_hash(), "kind": cs.kind, "n_seqs": p.n_seqs, "seq_len": p.seq_len, "seed": p.seed, "stats": list(p.stats)}
    out: dict[str, Any] = {"calib": desc}
    want = [s for s in p.stats if s == "router"] + (["hessian", "channel_scale"] if p.eager else [])
    if want:
        digest, lm = _cur(inputs, ctx)
        has_router = hasattr(lm.model, "layers") and any(hasattr(b.mlp, "experts") for b in lm.model.layers) if hasattr(lm.model, "layers") else False
        want = [s for s in want if s != "router" or has_router]
        if want:
            collect_stats(lm.model, lm.tokenizer, cs, stats=want, n_seqs=min(p.n_seqs, 64), seq_len=min(p.seq_len, 256), batch_size=p.batch_size, seed=p.seed,
                          cache=STATS_CACHE, should_stop=ctx.should_stop)
            desc["prefetched"] = want
    ctx.log(f"calibration set '{cs.name}' ({cs.kind}) hash {cs.content_hash()[:19]}, {p.n_seqs} x {p.seq_len} tokens")
    return out


# ------------------------------------------------------------------------------------------------ ptq
@_stage("ptq", ["model", "calib"], "Post-training quantisation (rtn/gptq/awq/smoothquant/mrgptq).")
def stage_ptq(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from forge.compress.ptq import quantize_model

    merged = {**_calib_defaults(inputs), **dict(params or {})}
    p = _params("ptq", merged)
    parent, lm = _cur(inputs, ctx)
    d = inputs.get("calib")
    calib = (d.get("set") if isinstance(d, dict) else d) or "synthetic"
    model = _fork(lm)
    qm = quantize_model(model, config=p, calib=calib, tokenizer=lm.tokenizer, ctx=ctx)
    rep = qm.report.to_dict()
    digest = _emit(inputs, ctx, "ptq", p.model_dump(), parent, lm, model, report={k: v for k, v in rep.items() if k != "layers"}, meta={"quant": {"method": p.method, "format": p.format, "bits": p.bits, "group_size": p.group_size}})
    summary = {k: v for k, v in rep.items() if k != "layers"}
    summary["layer_kinds"] = sorted({r.kind for r in qm.report.layers})
    return {"model": digest, "ptq_report": summary, "bytes_after": rep["bytes_after"], "compression": rep["compression"]}


# ------------------------------------------------------------------------------------------------ pack
@_stage("pack", ["model"], "Derive/validate the device layout of packed weights for a target chip.")
def stage_pack(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from forge.compress.modelutil import iter_packed_linears
    from forge.compress.packed import LAYOUT_ID, IntLinear, NVFP4Linear

    p = _params("pack", params)
    parent, lm = _cur(inputs, ctx)
    model = _fork(lm)
    layers = list(iter_packed_linears(model))
    int4 = [(n, m) for n, m in layers if isinstance(m, IntLinear) and m.bits == 4 and m.group_size % 8 == 0]
    fp4 = [(n, m) for n, m in layers if isinstance(m, NVFP4Linear)]
    notes: list[str] = []
    layout_id = LAYOUT_ID
    packed_bytes = 0
    checked = 0
    try:
        from forge.kernels import layouts as KL

        chip = None
        if p.target:
            try:
                from forge.hw.descriptor_store import load_descriptor

                chip = load_descriptor(p.target)
            except Exception as e:
                notes.append(f"chip descriptor '{p.target}' not found ({e}); using the local GPU / portable defaults")
        else:
            try:
                from forge.hw.capabilities import current_capabilities

                chip = current_capabilities()
            except Exception:
                chip = None
        if int4:
            if p.layout == "auto":
                from forge.kernels.spec import load_spec

                spec = load_spec("w4a16_gemm")
                lay = KL.derive_layout(spec, chip, executor=p.executor, group_size=int4[0][1].group_size, bits=4)
            else:
                lay = KL.get_layout(p.layout)
            layout_id = lay.id
            for name, m in int4:
                m.kernel_layout = lay
                pk = KL.pack_w4(m.codes(), m.scales, None if m.sym else m.zeros, m.group_size, lay)
                if p.validate_roundtrip:
                    q2, s2, z2 = KL.unpack_w4(pk)
                    if not torch.equal(q2, m.codes()) or not torch.equal(s2.to(m.scales.dtype), m.scales):
                        raise ForgeError(f"pack: unpack(pack(w)) differs for layer {name} in layout {lay.id}")
                    checked += 1
                m._kernel_cache = ((m.qweight.device, m.qweight._version), pk)
                packed_bytes += pk.nbytes
        if fp4:
            lay4 = KL.fp4_layout("nvfp4", "swizzled")
            layout_id = lay4.id if not int4 else f"{layout_id}+{lay4.id}"
            for name, m in fp4:
                sc = m.block_scales.view(torch.uint8)
                sw = KL.swizzle_scales(sc)
                if p.validate_roundtrip:
                    back = KL.unswizzle_scales(sw, sc.shape[0], sc.shape[1])
                    if not torch.equal(back, sc):
                        raise ForgeError(f"pack: scale swizzle round trip differs for layer {name}")
                    checked += 1
                packed_bytes += m.qweight.numel() + sw.numel() + 4
    except ImportError as e:
        notes.append(f"forge.kernels.layouts unavailable ({e}); weights stay in the portable layout '{LAYOUT_ID}'")
    except ForgeError:
        raise
    except Exception as e:  # kernels owner's API drift must not break a compression run: stay portable, say so
        notes.append(f"kernel layout derivation failed ({type(e).__name__}: {e}); weights stay in the portable layout '{LAYOUT_ID}'")
        layout_id = LAYOUT_ID
        for _n, m in int4:
            m.kernel_layout = None
            m._kernel_cache = None
    if not layers:
        notes.append("model has no packed layers; nothing to lay out")
    digest = _emit(inputs, ctx, "pack", p.model_dump(), parent, lm, model, report={"layout_id": layout_id, "notes": notes}, meta={"layout_id": layout_id})
    for n in notes:
        ctx.log(n, "warning")
    return {"model": digest, "layout_id": layout_id, "packed_layers": len(layers), "layout_checked": checked, "packed_bytes": packed_bytes, "pack_notes": notes}


# ------------------------------------------------------------------------------------------------ kv
@_stage("kv", ["model", "calib"], "KV-cache quantisation scheme + accuracy-vs-bytes report.")
def stage_kv(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from forge.compress.kvquant import KVQuantConfig, model_kv_report

    p = _params("kv", params)
    parent, lm = _cur(inputs, ctx)
    tokens, _ = _calib_tokens(inputs, lm.model, lm.tokenizer, p.report_seqs, p.report_len)
    rows = model_kv_report(lm.model, tokens, [KVQuantConfig(scheme="fp16"), p])
    cfg = getattr(lm.model, "config", None)
    kv_bytes: dict[str, Any] = {}
    if cfg is not None and hasattr(cfg, "kv_bytes_per_token"):
        bits = {"fp16": 16, "fp8_e4m3": 8, "fp8_e5m2": 8, "int8": 8, "kivi": p.bits}[p.scheme]
        kv_bytes = {"fp16": cfg.kv_bytes_per_token(16), "quantised": cfg.kv_bytes_per_token(bits, group_size=p.group_size if p.scheme == "kivi" else None)}
    kvd = {**p.model_dump(), "kv_bytes_per_token": kv_bytes}
    if p.scheme == "kivi" and p.long_context_gated:
        kvd["requires_gate"] = "G-LONG"
        ctx.log("4-bit/2-bit KV is gated: validate long-context quality (gate g-long) before serving", "warning")
    model = _fork(lm)
    model.forge_kv_config = kvd  # type: ignore[assignment]
    digest = _emit(inputs, ctx, "kv", p.model_dump(), parent, lm, model, report={"kv": kvd, "rows": rows["rows"]}, meta={"kv": kvd})
    return {"model": digest, "kv": kvd, "kv_report": rows}


# ------------------------------------------------------------------------------------------------ rotate
@_stage("rotate", ["model", "calib"], "Fuse norms and rotate the residual stream (Hadamard / learned), optionally online Hadamard before down_proj.")
def stage_rotate(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from forge.compress.rotation import rotate_model

    p = _params("rotate", params)
    parent, lm = _cur(inputs, ctx)
    if _is_compressed(lm.model):
        raise ConfigError("rotate must run before quantisation (the model already contains packed layers)")
    tokens = _calib_tokens(inputs, lm.model, lm.tokenizer, 16, 128)[0] if p.mode == "learned" else None
    model = _fork(lm)
    rep = rotate_model(model, p, tokens)
    digest = _emit(inputs, ctx, "rotate", p.model_dump(), parent, lm, model, report=rep)
    return {"model": digest, "rotate_report": _jsonable(rep)}


# ------------------------------------------------------------------------------------------------ prune
@_stage("prune", ["model", "calib"], "Structured pruning (width/heads/depth/hidden/experts) and optional unstructured sparsity.")
def stage_prune(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from forge.compress.prune import prune_model, sparsify_model

    p = _params("prune", {**_calib_defaults(inputs, ("seed",)), **dict(params or {})})
    parent, lm = _cur(inputs, ctx)
    if _is_compressed(lm.model):
        raise ConfigError("prune must run before quantisation (the model already contains packed layers)")
    tokens, _ = _calib_tokens(inputs, lm.model, lm.tokenizer, p.n_seqs, p.seq_len, p.seed)
    report: dict[str, Any] = {}
    warnings: list[str] = []
    new_model = lm.model
    meta: dict[str, Any] = {}
    if p.any_structure():
        router = None
        if p.expert_keep < 1.0:
            cs_name = (inputs.get("calib") or {}).get("set", "synthetic") if isinstance(inputs.get("calib"), dict) else "synthetic"
            cache = collect_stats(lm.model, lm.tokenizer, cs_name, stats=("router",), n_seqs=min(p.n_seqs, 64), seq_len=min(p.seq_len, 256),
                                  batch_size=p.batch_size, seed=p.seed, cache=StatsCache(), track_coactivation=False)
            router = {n: cache.get(cache.model_hash, cache.set_hash, n, "router") for n in cache.layers(cache.model_hash, cache.set_hash) if n != "__meta__"}
        res = prune_model(lm.model, p, tokens, router_stats=router)
        new_model, report = res.model, res.report
    else:
        new_model = _fork(lm)
    if p.sparsity is not None:
        srep = sparsify_model(new_model, p.sparsity, tokens)
        warnings += srep["warnings"]
        report["sparsity"] = srep
    for w in warnings:
        ctx.log(w, "warning")
    meta["needs_recovery"] = bool(report.get("needs_recovery"))
    digest = _emit(inputs, ctx, "prune", p.model_dump(), parent, lm, new_model, report={k: v for k, v in report.items() if k != "kept"}, meta=meta)
    try:
        (Path(ctx.run_dir) / "prune_report.json").write_text(json.dumps(_jsonable(report), indent=1))
    except OSError:
        pass
    out = {"model": digest, "prune_report": _jsonable({k: v for k, v in report.items() if k not in ("new_config", "kept")}), "needs_recovery": meta["needs_recovery"],
           "teacher": parent, "params_after": count_params(new_model), "warnings": warnings}
    return out


# ------------------------------------------------------------------------------------------------ lowrank
@_stage("lowrank", ["model", "calib"], "Activation-whitened SVD with heterogeneous rank allocation and optional healing.")
def stage_lowrank(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from forge.compress.lowrank import lowrank_compress_model

    p = _params("lowrank", {**_calib_defaults(inputs), **dict(params or {})})
    parent, lm = _cur(inputs, ctx)
    if _is_compressed(lm.model):
        raise ConfigError("lowrank must run before quantisation")
    d = inputs.get("calib")
    calib = (d.get("set") if isinstance(d, dict) else d) or "synthetic"
    model = _fork(lm)
    teacher = lm.model if (p.heal_steps > 0 and model is not lm.model) else None
    rep = lowrank_compress_model(model, p, calib, tokenizer=lm.tokenizer, ctx=ctx, teacher=teacher)
    rep.pop("rel_energy_lost", None)
    digest = _emit(inputs, ctx, "lowrank", p.model_dump(), parent, lm, model, report={k: v for k, v in rep.items() if k not in ("ranks",)})
    return {"model": digest, "lowrank_report": _jsonable({k: v for k, v in rep.items() if k != "ranks"}), "teacher": parent}


# ------------------------------------------------------------------------------------------------ delta
@_stage("delta", ["model", "base_model", "calib"], "Compress a fine-tune against its base (BitDelta / low-rank delta).")
def stage_delta(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from forge.compress.delta import calibrate_bitdelta, compute_delta, delta_size_report

    p = _params("delta", params)
    ft_digest = p.finetuned or inputs.get("model")
    base_digest = p.base or inputs.get("base_model")
    ft = state.require_live(ft_digest, ctx, "fine-tuned model")
    base = state.require_live(base_digest, ctx, "base model")
    if ft_digest == base_digest:
        raise ConfigError("delta stage: base and fine-tuned model are the same artifact (set params.base / params.finetuned)")
    sd_b, sd_f = base.model.state_dict(), ft.model.state_dict()
    d = compute_delta(sd_b, sd_f, p)
    cal = None
    if p.method == "bitdelta" and p.calibrate_steps > 0:
        tokens, _ = get_calibration_tokens(p.calib_set, ft.model, ft.tokenizer, n_seqs=p.n_seqs, seq_len=p.seq_len, seed=p.seed)
        cal = calibrate_bitdelta(base.model, ft.model, d, tokens, steps=p.calibrate_steps, lr=p.calibrate_lr, batch_size=p.batch_size, ctx=ctx)
    rep = delta_size_report(d, sd_f)
    rep["calibration"] = {k: v for k, v in (cal or {}).items() if k != "losses"}
    out_dir = Path(ctx.run_dir) / "delta"
    d.save(out_dir)
    ref: str = str(out_dir)
    store = getattr(ctx, "store", None)
    if store is not None:
        try:
            ref = store.put_tensor_dir(out_dir, kind="delta", recipe_hash=_meta(inputs).get("hash"), parents=[x for x in (base_digest,) if x], extra={"method": p.method, "report": _jsonable(rep)})
        except Exception as e:
            log.warning("could not store the delta artifact: %s", e)
    ctx.log(f"delta ({p.method}): {rep['delta_bytes'] / 1e6:.3f} MB vs dense {rep['dense_finetune_bytes'] / 1e6:.3f} MB ({rep['compression_vs_dense']:.1f}x)")
    return {"delta": ref, "delta_report": _jsonable(rep)}


# ------------------------------------------------------------------------------------------------ lossless
@_stage("lossless", ["model"], "Measure/apply bit-exact entropy coding of the current model's tensors.")
def stage_lossless(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    from forge.compress.lossless import lossless_report

    p = _params("lossless", params)
    _, lm = _cur(inputs, ctx)
    out = Path(ctx.run_dir) / "lossless" if p.write_container else None
    rep = lossless_report(lm.model, p, out)
    rows = rep.pop("rows")
    (Path(ctx.run_dir) / "lossless_report.json").write_text(json.dumps({**rep, "rows": rows}, indent=1, default=str))
    ctx.log(f"lossless: {rep['raw_bytes'] / 1e6:.2f} MB -> {rep['encoded_bytes'] / 1e6:.2f} MB (ratio {rep['ratio']:.3f}, verified={rep['verified']})")
    return {"lossless_report": _jsonable(rep)}


# ------------------------------------------------------------------------------------------------ eval / gate
def _parent_for(inputs: dict[str, Any], ctx: RunContext, explicit: str | None) -> state.LiveModel | None:
    """Reference model: explicit digest if it can be resolved, else the job's input model (recipes often carry symbolic digests such as
    ``sha256:base70b_bf16``; the substitution is logged)."""
    base = inputs.get("base_model")
    if explicit:
        try:
            return state.resolve_model(explicit, ctx)
        except NotFound:
            ctx.log(f"parent '{explicit}' is not a known artifact; using the job's input model as the reference", "warning")
    if base is None or base == inputs.get("model"):
        return None
    return state.resolve_model(base, ctx)


def _eval_report_dict(rep: Any) -> dict[str, Any]:
    d = rep.model_dump(mode="json") if hasattr(rep, "model_dump") else dict(rep)
    d.pop("env", None)
    return d


def _run_forge_eval(inputs, ctx, lm, parent_lm, *, suites, suite_params, gate, overrides, seeds, n_boot, context, persist) -> Any:
    from forge.eval.config import EvalConfig, ModelSource, SuiteSelection
    from forge.eval.run import run_eval

    ctxd = dict(context)
    pb = lm.meta.get("params_b") or round(count_params(lm.model) / 1e9, 6)
    ctxd.setdefault("params_b", pb)
    cfg = EvalConfig(
        name=f"{_meta(inputs).get('name', 'recipe')}-eval", candidate=ModelSource(kind="none", label="candidate", params_b=pb),
        parent=ModelSource(kind="none", label="parent") if parent_lm is not None else None,
        suites=[SuiteSelection(name=s, params=dict(suite_params.get(s, {}))) for s in suites], gate=gate, gate_overrides=overrides, seeds=list(seeds),
        n_bootstrap=n_boot, context=ctxd, save_generations=False,
    )
    providers = {"candidate": lm.model}
    if parent_lm is not None:
        providers["parent"] = parent_lm.model
    return run_eval(cfg, ctx, providers=providers, tokenizer=lm.tokenizer, persist=persist)


def _store_report(ctx: RunContext, rep_dict: dict[str, Any]) -> str | None:
    store = getattr(ctx, "store", None)
    if store is None:
        return None
    try:
        return store.put_bytes(json.dumps(rep_dict, sort_keys=True, default=str).encode(), kind="eval_report")
    except Exception:
        return None


@_stage("eval", ["model", "base_model", "calib"], "Evaluate the current model against its parent (forge.eval suites).")
def stage_eval(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    p = _params("eval", params)
    digest, lm = _cur(inputs, ctx)
    parent_lm = _parent_for(inputs, ctx, p.parent)
    if not p.builtin:
        try:
            import forge.eval.run  # noqa: F401
        except Exception as e:
            ctx.log(f"forge.eval unavailable ({e}); using the built-in comparison", "warning")
            p = p.model_copy(update={"builtin": True})
    if p.builtin:
        return _builtin_eval(inputs, ctx, p, lm, parent_lm)
    rep = _run_forge_eval(inputs, ctx, lm, parent_lm, suites=p.suites, suite_params=p.suite_params, gate=p.gate, overrides={}, seeds=p.seeds, n_boot=p.n_bootstrap,
                          context=p.context, persist=p.persist)
    d = _eval_report_dict(rep)
    h = _store_report(ctx, d)
    return {"eval_report": _jsonable(d), "eval_report_hash": h, "eval_report_id": d.get("id")}


def _builtin_eval(inputs, ctx, p, lm, parent_lm) -> dict[str, Any]:
    from forge.compress.quickeval import compare_models

    tokens, _ = get_calibration_tokens(p.set, lm.model, lm.tokenizer, n_seqs=p.n_seqs, seq_len=p.seq_len, seed=p.seed)
    stats = compare_models(lm.model, parent_lm.model if parent_lm else None, tokens, n_boot=p.n_bootstrap)
    ctx.log("eval (built-in): " + ", ".join(f"{k}={v['value']:.4g}" for k, v in stats.items() if isinstance(v, dict) and "value" in v))
    return {"eval_report": {"metrics": stats, "builtin": True, "provenance": "measured"}}


@_stage("gate", ["model", "base_model", "calib"], "Pass/fail/inconclusive verdict against the parent (forge.eval gates).")
def stage_gate(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    p = _params("gate", params)
    digest, lm = _cur(inputs, ctx)
    parent_lm = _parent_for(inputs, ctx, p.parent)
    seeds = list(range(p.seeds)) if isinstance(p.seeds, int) else list(p.seeds)
    m = re.search(r"(\d+)", p.stats or "")
    n_boot = max(int(m.group(1)), 100) if m else 1000
    verdict: dict[str, Any]
    detail: dict[str, Any] = {}
    try:
        import forge.eval.run  # noqa: F401

        use_builtin = False
    except Exception as e:
        ctx.log(f"forge.eval unavailable ({e}); using the built-in gate", "warning")
        use_builtin = True
    if not use_builtin:
        rep = _run_forge_eval(inputs, ctx, lm, parent_lm, suites=p.suites, suite_params=p.suite_params, gate=p.suite, overrides=p.overrides, seeds=seeds or [0],
                              n_boot=n_boot, context=p.context, persist=True)
        d = _eval_report_dict(rep)
        verdict = d.get("gate") or {"gate_id": p.suite, "verdict": "inconclusive", "reasons": ["the evaluation produced no gate verdict"], "details": {}}
        detail = {"eval_report_id": d.get("id"), "eval_report_hash": _store_report(ctx, d), "metrics": d.get("metrics")}
    else:
        from forge.compress.quickeval import builtin_gate, compare_models

        tokens, _ = get_calibration_tokens("builtin-mix", lm.model, lm.tokenizer, n_seqs=16, seq_len=128, seed=10_000)
        stats = compare_models(lm.model, parent_lm.model if parent_lm else None, tokens, n_boot=n_boot)
        bg = builtin_gate(stats, p.builtin_thresholds)
        verdict = {"gate_id": f"builtin:{p.suite}", **bg, "details": {"metrics": stats}}
        detail = {"metrics": stats}
    ctx.log(f"gate {verdict.get('gate_id')}: {verdict.get('verdict')} ({'; '.join(verdict.get('reasons', [])[:3])})")
    try:
        ctx.set_output("gate_verdict", verdict.get("verdict"))
    except Exception:
        pass
    if p.enforce and verdict.get("verdict") == "fail":
        raise GateFailed(f"gate {verdict.get('gate_id')} failed: " + "; ".join(verdict.get("reasons", [])), verdict=verdict)
    return {"gate": _jsonable(verdict), "gate_verdict": verdict.get("verdict"), "gate_detail": _jsonable(detail)}


# ------------------------------------------------------------------------------------------------ distill / finetune
class _SubContext:
    """Forwards to the job context but records ``set_output`` calls (to find the model a delegated train run produced)."""

    def __init__(self, ctx: RunContext) -> None:
        self._ctx = ctx
        self.outputs: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)

    def set_output(self, key: str, value: Any) -> None:
        self.outputs[key] = value
        self._ctx.set_output(key, value)


def _teacher_allowed(meta: dict[str, Any], policy: dict[str, Any], check: bool | None) -> None:
    if check is None:
        check = bool(policy.get("licence_check", True))
    if not check:
        return
    allowed = meta.get("distill_allowed")
    if meta.get("locally_owned") or allowed is True:
        return
    lic = (meta.get("licence") or "").lower()
    if allowed is False:
        raise LicenceBlocked(f"teacher licence '{lic or 'unknown'}' forbids distillation (distill_allowed=false)")
    if allowed is None and lic in PERMISSIVE_LICENCES:
        return
    raise LicenceBlocked(
        f"teacher licence is '{lic or 'unknown'}': distillation needs a permissive licence or an explicit attestation "
        "(set job config distill_allowed=true after reviewing the terms, or policy.licence_check=false)"
    )


def _delegate_cfg(cls: type, user: dict[str, Any], student_dir: Path, teacher_dir: Path | None, teacher_meta: dict[str, Any] | None, name: str) -> Any:
    cfg = copy.deepcopy(user)
    cfg.setdefault("name", name)
    m = cfg.setdefault("model", {})
    m.update({"source": "pretrained", "path": str(student_dir)})
    if teacher_dir is not None:
        t = cfg.setdefault("teacher", {})
        t.update({"path": str(teacher_dir)})
        tm = teacher_meta or {}
        if tm.get("licence") and not t.get("licence"):
            t["licence"] = tm["licence"]
        if tm.get("distill_allowed") is not None and t.get("distill_allowed") is None:
            t["distill_allowed"] = tm["distill_allowed"]
        if tm.get("locally_owned"):
            t["locally_owned"] = True
    try:
        return cls.model_validate(cfg)
    except Exception as e:
        raise ConfigError(f"invalid {cls.__name__} for the delegated train run: {e}") from e


def _export_for_train(lm: state.LiveModel, ctx: RunContext, tag: str) -> Path:
    from forge.compress.serialize import save_quantized

    d = Path(ctx.run_dir) / "delegate" / tag
    save_quantized(lm.model, d, tokenizer=lm.tokenizer)
    return d


def _collect_model_dir(sub: _SubContext, fallback_dir: Path) -> Path:
    for key in ("model_dir", "final_model_dir"):
        if sub.outputs.get(key):
            return Path(sub.outputs[key])
    raise ForgeError(f"the delegated train run did not report an output model directory (outputs: {sorted(sub.outputs)})")


@_stage("distill", ["model", "base_model", "calib", "teacher"], "Distillation recovery (builtin logit-KL smoke recovery, or delegate to forge.train.run.run_distill).")
def stage_distill(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    p = _params("distill", params)
    parent, lm = _cur(inputs, ctx)
    t_digest = p.teacher or inputs.get("teacher") or inputs.get("base_model")
    if t_digest is None or t_digest == parent:
        raise ConfigError("distill needs a teacher: set params.teacher, run prune/lowrank before it, or provide a different base model")
    tlm = state.resolve_model(t_digest, ctx)
    _teacher_allowed(tlm.meta, _meta(inputs).get("policy", {}), p.licence_check)
    if p.mode == "builtin":
        from forge.compress.quickeval import distill_kl

        tokens, _ = get_calibration_tokens(p.set, lm.model, lm.tokenizer, n_seqs=p.n_seqs, seq_len=p.seq_len, seed=p.seed)
        student = _fork(lm)
        rep = distill_kl(student, tlm.model, tokens, steps=p.steps, lr=p.lr, temperature=p.temperature, batch_size=p.batch_size, trainable=p.trainable, ctx=ctx)
        digest = _emit(inputs, ctx, "distill", p.model_dump(), parent, lm, student, report={k: v for k, v in rep.items() if k != "losses"}, meta={"needs_recovery": False, "teacher": t_digest})
        return {"model": digest, "distill_report": _jsonable({k: v for k, v in rep.items() if k != "losses"}), "needs_recovery": False}
    try:
        from forge.train.config import DistillConfig
        from forge.train.run import run_distill
    except Exception as e:
        raise ForgeError(f"distill(mode=delegate) needs forge.train ({e})") from e
    sdir = _export_for_train(lm, ctx, "student")
    tdir = _export_for_train(tlm, ctx, "teacher")
    cfg = _delegate_cfg(DistillConfig, p.config, sdir, tdir, tlm.meta, f"{_meta(inputs).get('name', 'recipe')}-distill")
    sub = _SubContext(ctx)
    result = run_distill(cfg, sub)
    out_dir = _collect_model_dir(sub, sdir)
    model, tok = state.load_model_dir(out_dir, dtype="float32" if not torch.cuda.is_available() else "bfloat16", device="cuda" if torch.cuda.is_available() else "cpu")
    digest = _emit(inputs, ctx, "distill", p.model_dump(), parent, lm, model, report={"result": _jsonable(result)}, tokenizer=tok or lm.tokenizer,
                   meta={"needs_recovery": False, "teacher": t_digest})
    return {"model": digest, "distill_report": _jsonable({"result": result, "model_dir": str(out_dir)}), "needs_recovery": False}


@_stage("finetune", ["model"], "Fine-tune the current model (delegates to forge.train.run.run_finetune).")
def stage_finetune(inputs: dict[str, Any], params: dict[str, Any], ctx: RunContext) -> dict[str, Any]:
    p = _params("finetune", params)
    parent, lm = _cur(inputs, ctx)
    try:
        from forge.train.config import FinetuneConfig
        from forge.train.run import run_finetune
    except Exception as e:
        raise ForgeError(f"finetune needs forge.train ({e})") from e
    sdir = _export_for_train(lm, ctx, "finetune_in")
    cfg = _delegate_cfg(FinetuneConfig, p.config, sdir, None, None, f"{_meta(inputs).get('name', 'recipe')}-finetune")
    sub = _SubContext(ctx)
    result = run_finetune(cfg, sub)
    out_dir = _collect_model_dir(sub, sdir)
    model, tok = state.load_model_dir(out_dir, dtype="float32" if not torch.cuda.is_available() else "bfloat16", device="cuda" if torch.cuda.is_available() else "cpu")
    digest = _emit(inputs, ctx, "finetune", p.model_dump(), parent, lm, model, report={"result": _jsonable(result)}, tokenizer=tok or lm.tokenizer)
    return {"model": digest, "finetune_report": _jsonable({"result": result, "model_dir": str(out_dir)})}


STAGE_NAMES = ["calibrate", "ptq", "pack", "kv", "rotate", "prune", "lowrank", "delta", "lossless", "eval", "gate", "distill", "finetune"]
