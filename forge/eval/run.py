"""Evaluation harness: ``run_eval(cfg, ctx)`` (job entry for job type ``eval``).

Pipeline: load parent / candidate -> run the selected suites (paired on identical items) -> pool the task suites into
``suite_score`` -> estimate the noise floor from perturbed parent repeats -> mark differences "within noise" -> apply the gate
-> persist an :class:`forge.core.schemas.EvalReport` (table ``evals``) and the compressed generations (EV-7) -> emit metrics
and progress through the :class:`RunContext`.

Models are loaded with ``forge.models.loader.load_model`` (lazy; GPU hosts only for real models).  Tests and notebooks can
bypass loading by passing ``providers={"candidate": ..., "parent": ...}`` (any :class:`LogitsProvider`, module or callable).
"""
from __future__ import annotations

import copy
import json
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.context import NullContext
from forge.core.errors import ConfigError, ForgeError
from forge.core.hashing import hash_json
from forge.core.schemas import EvalReport, GateVerdict, MetricValue, Provenance
from forge.eval import stats
from forge.eval.config import EvalConfig, ModelSource
from forge.eval.gates import Gate, Measurements, load_gate
from forge.eval.metrics import as_provider, provider_nbytes
from forge.eval.suites import get_suite
from forge.eval.suites.base import SuiteEnv, SuiteResult, adapt_tokenizer, paired_metric

log = logging.getLogger("forge.eval")

NOISE_METHOD = (
    "score-type metrics: t_{n-1,0.975} * std * sqrt(2) over n parent repeats (half-width of the difference between two independent "
    "evaluations of the same model); divergence-type metrics (kl_*, top1_flip, entropy_shift): one-sided prediction bound "
    "mean + t * std * sqrt(1 + 1/n) of parent-vs-perturbed-parent runs."
)
_DIVERGENCE = ("kl_mean", "kl_p99", "top1_flip", "entropy_shift")


# --------------------------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------------------------
def _resolve_device(dev: str) -> str:
    if dev != "auto":
        return dev
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_source(src: ModelSource, ctx: Any = None) -> tuple[Any, Any]:
    """Load a model source -> (LogitsProvider, tokenizer|None)."""
    from forge.eval.metrics import ModuleProvider

    path = src.ref
    if src.kind == "artifact":
        store = getattr(ctx, "store", None)
        if store is None:
            from forge.store import get_store

            store = get_store()
        dest = Path(getattr(ctx, "run_dir", ".")) / "materialized" / src.ref.replace(":", "_")
        path = str(store.materialize_artifact(src.ref, dest))
    if not path:
        raise ConfigError("model source has no ref")
    from forge.models.loader import load_model  # lazy: only on real runs

    model, tok = load_model(path, dtype=src.dtype, device=_resolve_device(src.device), revision=src.revision, backend=src.backend)
    prov = ModuleProvider(model, name=src.label or src.ref, eos_token_id=getattr(tok, "eos_token_id", None))
    return prov, tok


def _make_serve(src: ModelSource | None) -> Any:
    if src is None or not src.serve_url:
        return None
    from forge.eval.serve_client import HTTPServeClient

    return HTTPServeClient(src.serve_url, model=src.serve_model, api_key=src.serve_api_key, name=src.label or src.serve_url)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_gate_overrides(gate: Gate, overrides: dict[str, Any]) -> Gate:
    if not overrides:
        return gate
    from forge.eval.gates import gate_from_dict

    return gate_from_dict(_deep_merge(gate.spec.model_dump(mode="python", exclude_none=False), overrides))


def _suite_params_with_variation(suite_name: str, params: dict[str, Any], cls: Any, perturb_seed: bool) -> dict[str, Any]:
    p = dict(params)
    fields = cls.Config.model_fields
    if perturb_seed and "vary_seed_with_run" in fields:
        p["vary_seed_with_run"] = True
    if perturb_seed and "shuffle_choices" in fields:
        p["shuffle_choices"] = True
    return p


def _pool_suite_score(results: list[SuiteResult], env: SuiteEnv) -> SuiteResult | None:
    """suite_score over all contributing suites (paired where a parent exists)."""
    contrib = [r for r in results if r.contributes_to_suite_score and "score" in r.items]
    if not contrib:
        return None
    cand = np.concatenate([r.items["score"][1] for r in contrib])
    par_parts = [r.items["score"][0] for r in contrib]
    par = np.concatenate(par_parts) if all(len(p) for p in par_parts) else None
    pooled = SuiteResult("suite_score", "1")
    paired_metric(pooled, "suite_score", par, cand, env, as_points=True)
    return pooled


def _attach_generations(rows: list[dict[str, Any]], ctx: Any, report_id: str) -> dict[str, str]:
    if not rows:
        return {}
    data = ("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows) + "\n").encode("utf-8")
    try:
        import zstandard

        blob = zstandard.ZstdCompressor(level=9).compress(data)
        ext = "jsonl.zst"
    except Exception:
        blob, ext = data, "jsonl"
    out: dict[str, str] = {}
    store = getattr(ctx, "store", None)
    if store is not None:
        try:
            out["generations_hash"] = store.put_bytes(blob, kind="eval")
        except Exception as e:  # store problems must not lose the report
            log.warning("could not store generations in the artifact store: %s", e)
    rd = getattr(ctx, "run_dir", None)
    if rd is not None:
        p = Path(rd) / f"generations_{report_id}.{ext}"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(blob)
        out["generations_path"] = str(p)
    out["generations_count"] = str(len(rows))
    return out


# --------------------------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------------------------
def run_eval(cfg: EvalConfig | dict[str, Any], ctx: Any | None = None, *, providers: dict[str, Any] | None = None,
             tokenizer: Any = None, serve_clients: dict[str, Any] | None = None, persist: bool = True) -> EvalReport:
    """Run an evaluation and return (and by default persist) the :class:`EvalReport`.

    ``providers`` may supply ``{"candidate": ..., "parent": ...}`` directly (modules, callables or LogitsProviders).  Returns the
    report; also ``ctx.set_output("report_id" | "verdict" | "gate")``."""
    if isinstance(cfg, dict):
        cfg = EvalConfig(**cfg)
    ctx = ctx or NullContext()
    t_start = time.time()
    providers = dict(providers or {})
    serve_clients = dict(serve_clients or {})

    gate: Gate | None = None
    if cfg.gate:
        gate = apply_gate_overrides(load_gate(cfg.gate), cfg.gate_overrides)
        ctx.log(f"gate {gate.id} ({gate.spec.status} thresholds)")

    # ---- models ---------------------------------------------------------------------------
    tok_from_model = None
    if "candidate" not in providers:
        needs_models = any(get_suite(s.name).needs_candidate_model for s in cfg.suites)
        if needs_models and (cfg.candidate.kind != "none") and not (cfg.candidate.serve_url and all(s.name == "throughput" for s in cfg.suites)):
            ctx.log(f"loading candidate {cfg.candidate.label or cfg.candidate.ref}")
            providers["candidate"], tok_from_model = load_source(cfg.candidate, ctx)
    if cfg.parent is not None and cfg.parent.kind != "none" and "parent" not in providers:
        if not (cfg.parent.serve_url and all(s.name == "throughput" for s in cfg.suites)):
            ctx.log(f"loading parent {cfg.parent.label or cfg.parent.ref}")
            providers["parent"], ptok = load_source(cfg.parent, ctx)
            tok_from_model = tok_from_model or ptok
    for k in ("candidate", "parent"):
        if k in providers and providers[k] is not None:
            providers[k] = as_provider(providers[k], name=k)
    tok = adapt_tokenizer(tokenizer if tokenizer is not None else tok_from_model)
    if tokenizer is None and cfg.tokenizer:
        from forge.models.loader import load_tokenizer

        tok = adapt_tokenizer(load_tokenizer(cfg.tokenizer))

    cand, parent = providers.get("candidate"), providers.get("parent")
    has_parent = parent is not None
    if cfg.candidate.serve_url and "candidate" not in serve_clients:
        serve_clients["candidate"] = _make_serve(cfg.candidate)
    if cfg.parent is not None and cfg.parent.serve_url and "parent" not in serve_clients:
        serve_clients["parent"] = _make_serve(cfg.parent)

    params_b = cfg.context.get("params_b", cfg.candidate.params_b)
    model_info = {"params_b": params_b}
    context = dict(cfg.context)
    context.update({"params_b": params_b, "has_parent": has_parent or bool(serve_clients.get("parent")), "deployment_length": cfg.deployment_length})

    def mkenv(seed: int, c: Any, p: Any, batch_override: int | None = None, progress: Any = None) -> SuiteEnv:
        return SuiteEnv(parent=p, candidate=c, tokenizer=tok, seed=seed, n_boot=cfg.n_bootstrap, alpha=cfg.alpha, ctx=ctx,
                        parent_serve=serve_clients.get("parent"), candidate_serve=serve_clients.get("candidate"),
                        deployment_length=cfg.deployment_length, model_info=model_info, batch_size_override=batch_override, sub_progress=progress)

    # ---- main run -------------------------------------------------------------------------
    seed0 = cfg.seeds[0]
    results: list[SuiteResult] = []
    errors: dict[str, str] = {}
    n_suites = len(cfg.suites)
    for si, sel in enumerate(cfg.suites):
        if ctx.should_stop():
            ctx.log("stop requested; finishing early", "warning")
            break
        cls = get_suite(sel.name)
        suite = cls()
        ctx.log(f"suite {sel.name} v{cls.version}")
        base, span = si / n_suites, 1 / n_suites

        def sub(f: float, msg: str = "", _b=base, _s=span) -> None:
            ctx.progress(0.9 * (_b + _s * f), msg)

        try:
            scfg = cls.make_config(sel.params)
            if cls.needs_candidate_model and cand is None and not (sel.name == "throughput" and serve_clients.get("candidate")):
                raise ConfigError(f"suite {sel.name} needs a candidate model")
            res = suite.run(scfg, mkenv(seed0, cand, parent, progress=sub))
            results.append(res)
            ctx.metric(si, **{f"{res.suite}/{k}": v for k, v in res.scalars.items() if np.isfinite(v)})
        except Exception as e:  # keep going: a failing suite makes dependent gate rules inconclusive
            errors[sel.name] = f"{type(e).__name__}: {e}"
            ctx.log(f"suite {sel.name} failed: {e}\n{traceback.format_exc()}", "error")
    if not results and errors:
        raise ForgeError("every suite failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()))

    metrics: dict[str, MetricValue] = {}
    boot: dict[str, np.ndarray] = {}
    flags: dict[str, bool | None] = {}
    generations: list[dict[str, Any]] = []
    notes: list[str] = []
    suite_versions: dict[str, str] = {}
    for r in results:
        metrics.update(r.metrics)
        boot.update(r.boot)
        flags.update(r.flags)
        generations += r.generations
        notes += [f"[{r.suite}] {n}" for n in r.notes]
        suite_versions[r.suite] = r.version
    env_main = mkenv(seed0, cand, parent)
    pooled = _pool_suite_score(results, env_main)
    if pooled is not None:
        metrics.update(pooled.metrics)
        boot.update(pooled.boot)
    names = {r.suite for r in results}
    flags["multi_benchmark_evidence"] = bool("longctx" in names and ({"agentic", "longgen"} & names))

    # ---- noise floor ------------------------------------------------------------------------
    n_rep = cfg.noise_repeats if cfg.noise_repeats is not None else (gate.spec.noise_floor.n if gate else 0)
    noise: dict[str, float] = {}
    noise_detail: dict[str, Any] = {}
    if n_rep and has_parent and not ctx.should_stop():
        noise, noise_detail = estimate_noise_floor(cfg, parent, tok, mkenv, n_rep, ctx)
    elif n_rep and not has_parent:
        notes.append("noise floor skipped: no parent model")

    _mark_within_noise(metrics, noise)

    # ---- external evidence --------------------------------------------------------------------
    for k, v in cfg.extra_metrics.items():
        metrics[k] = MetricValue(value=float(v), provenance=Provenance.measured, n=None)
    for k, v in cfg.extra_flags.items():
        flags[k] = bool(v)

    # ---- gate -------------------------------------------------------------------------------
    verdict: GateVerdict | None = None
    if gate is not None:
        m = Measurements(metrics=metrics, boot=boot, noise_floor=noise, flags=flags, context=context)
        verdict = gate.evaluate(m)
        ctx.log(f"gate {verdict.gate_id}: {verdict.verdict.upper()}" + ("".join(f"\n  - {r}" for r in verdict.reasons)))
        ctx.set_output("gate", verdict.verdict)

    # ---- report ---------------------------------------------------------------------------------
    cand_hash = cfg.candidate.artifact_hash or (cfg.candidate.ref if cfg.candidate.kind == "artifact" else None) or "local:" + hash_json(
        {"ref": cfg.candidate.ref, "label": cfg.candidate.label, "kind": cfg.candidate.kind, "name": cfg.name})[-16:]
    par_hash = None
    if cfg.parent is not None:
        par_hash = cfg.parent.artifact_hash or (cfg.parent.ref if cfg.parent.kind == "artifact" else None) or "local:" + hash_json(
            {"ref": cfg.parent.ref, "label": cfg.parent.label, "kind": cfg.parent.kind})[-16:]
    env_info: dict[str, Any] = {"suite_errors": errors, "notes": notes, "noise_method": NOISE_METHOD if noise else None,
                                "noise_detail": noise_detail, "seeds": cfg.seeds, "n_bootstrap": cfg.n_bootstrap, "alpha": cfg.alpha,
                                "has_parent": has_parent, "elapsed_s": round(time.time() - t_start, 2), "deployment_length": cfg.deployment_length,
                                "context": {k: v for k, v in context.items() if isinstance(v, (int, float, str, bool, type(None)))}}
    try:
        from forge.core.env import capture_environment

        env_info["environment"] = capture_environment()
    except Exception as e:  # pragma: no cover - environment capture is best effort
        env_info["environment_error"] = str(e)
    report = EvalReport(artifact_hash=cand_hash, parent_hash=par_hash, suite_versions=suite_versions, metrics=metrics, gate=verdict,
                        noise_floor={k: float(v) for k, v in noise.items() if np.isfinite(v)}, env=env_info)

    nbytes = provider_nbytes(cand) if cand is not None else None
    attachments: dict[str, str] = {}
    if cfg.save_generations:
        attachments = _attach_generations(generations, ctx, report.id)
    if persist:
        from forge.eval.db import save_report

        save_report(
            report, config=cfg.model_dump(mode="json", exclude={"candidate": {"serve_api_key"}, "parent": {"serve_api_key"}}), name=cfg.name,
            candidate_label=cfg.candidate.label or cfg.candidate.ref or cfg.name, parent_label=(cfg.parent.label or cfg.parent.ref) if cfg.parent else "",
            bytes=float(nbytes) if nbytes is not None else None, bytes_provenance="measured", params_b=params_b,
            quality_metric=cfg.quality_metric, run_id=getattr(ctx, "run_id", None), attachments=attachments, tags=cfg.tags,
        )
    ctx.set_output("report_id", report.id)
    ctx.set_output("suite_errors", errors)
    if verdict is not None:
        ctx.set_output("verdict", verdict.verdict)
    ctx.progress(1.0, "done")
    return report


# --------------------------------------------------------------------------------------------
# noise floor estimation
# --------------------------------------------------------------------------------------------
def estimate_noise_floor(cfg: EvalConfig, parent: Any, tok: Any, mkenv: Any, n_rep: int, ctx: Any) -> tuple[dict[str, float], dict[str, Any]]:
    """Run the suites ``n_rep`` times with the parent as candidate under perturbations; return floors by metric name."""
    from forge.eval.noise import BatchPerturbedProvider

    series: dict[str, list[float]] = {}
    pert_seed = cfg.noise_perturb in ("seed", "both")
    pert_batch = cfg.noise_perturb in ("batch", "both")
    extra_seeds = list(cfg.seeds[1:])
    for r in range(n_rep):
        if ctx.should_stop():
            break
        seed = extra_seeds[r] if r < len(extra_seeds) else (cfg.seeds[0] + 1000 + r)
        view = BatchPerturbedProvider(parent, seed) if pert_batch else parent
        ctx.log(f"noise repeat {r + 1}/{n_rep} (seed {seed})")
        for sel in cfg.suites:
            cls = get_suite(sel.name)
            if sel.name in ("contamination", "throughput") or not cls.needs_candidate_model:
                continue
            try:
                params = _suite_params_with_variation(sel.name, sel.params, cls, pert_seed)
                scfg = cls.make_config(params)
                if sel.name == "distribution":
                    # parent vs batch-perturbed parent: divergence-type noise
                    env = mkenv(seed, view, parent, batch_override=max(1, r + 1))
                else:
                    env = mkenv(seed, view, None)
                res = cls().run(scfg, env)
            except Exception as e:
                ctx.log(f"noise repeat of {sel.name} failed: {e}", "warning")
                continue
            for k, v in res.scalars.items():
                if np.isfinite(v):
                    series.setdefault(k, []).append(float(v))
            if res.contributes_to_suite_score and "score" in res.items:
                series.setdefault("suite_score", []).append(float(res.items["score"][1].mean()))
    floors: dict[str, float] = {}
    detail: dict[str, Any] = {}
    for k, vals in series.items():
        if len(vals) < 2:
            continue
        if k in _DIVERGENCE:
            arr = np.asarray(vals)
            t = stats.t_quantile(0.975, len(arr) - 1)
            floors[k] = float(arr.mean() + t * arr.std(ddof=1) * np.sqrt(1 + 1 / len(arr)))
            detail[k] = {"type": "divergence_bound", "n": len(arr), "mean": float(arr.mean()), "std": float(arr.std(ddof=1))}
        else:
            nf = stats.noise_floor_from_repeats(vals)
            floors[k] = nf.floor
            detail[k] = {"type": "score_halfwidth", **nf.to_dict()}
    # differences of scores are reported in points
    if "suite_score" in floors:
        floors["suite_score_diff_pts"] = floors["suite_score"] * 100.0
    for k in list(floors):
        if k.endswith(("_acc", "_score", "_success", "_rate_harmful", "_rate", "_pref_rate")) or k.startswith(("longctx_", "mcq_acc")):
            floors.setdefault(f"{k}_diff_pts", floors[k] * 100.0)
    return floors, detail


def _mark_within_noise(metrics: dict[str, MetricValue], noise: dict[str, float]) -> None:
    """Set ``within_noise`` on paired-difference metrics: indistinguishable if CI contains 0 or |diff| <= noise floor."""
    for name, mv in metrics.items():
        if not name.endswith(("_diff_pts", "_diff")):
            continue
        nf = noise.get(name)
        d = stats.distinguishable(mv.value, noise_floor=nf, ci_low=mv.ci_low, ci_high=mv.ci_high)
        if nf is None and mv.ci_low is None:
            continue
        mv.within_noise = not d.distinguishable
    sd = metrics.get("safety_delta")
    if sd is not None and sd.within_noise is not None:
        nf = noise.get("safety_delta") or noise.get("refusal_rate_harmful_diff_pts")
        if nf is not None and sd.value > nf:
            sd.within_noise = False
