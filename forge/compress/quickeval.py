"""Minimal built-in model comparison and logit-KL recovery used when ``forge.eval`` / ``forge.train`` are not wanted or unavailable.

* :func:`compare_models` -- per-sequence KL(parent || candidate), top-1 flip rate, perplexities on held-out tokens, with percentile bootstrap CIs.
  This is *not* a replacement for the evaluation harness (no task suites, no noise floor); the ``eval`` / ``gate`` stages use ``forge.eval``
  and fall back to this only if it cannot be imported or ``builtin=true`` is requested.
* :func:`distill_kl` -- short forward-KL recovery of a student against a teacher on calibration tokens (smoke-scale "healing"; production
  distillation is ``forge.train.run.run_distill``).
"""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.compress.modelutil import model_device
from forge.core.context import NullContext, RunContext


@torch.no_grad()
def compare_models(candidate: nn.Module, parent: nn.Module | None, tokens: torch.Tensor, batch_size: int = 4, n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    cd = model_device(candidate)
    pd = model_device(parent) if parent is not None else cd
    kl_seq: list[float] = []
    flip_seq: list[float] = []
    kl_tok: list[torch.Tensor] = []
    ce_c: list[float] = []
    ce_p: list[float] = []
    candidate.eval()
    if parent is not None:
        parent.eval()
    for b in tokens.split(batch_size):
        lc = candidate(b.to(cd)).logits.float().cpu()
        lp = parent(b.to(pd)).logits.float().cpu() if parent is not None else None
        tgt = b[:, 1:]
        ce_c += F.cross_entropy(lc[:, :-1].transpose(1, 2), tgt, reduction="none").mean(1).tolist()
        if lp is not None:
            ce_p += F.cross_entropy(lp[:, :-1].transpose(1, 2), tgt, reduction="none").mean(1).tolist()
            logp, logc = F.log_softmax(lp, -1), F.log_softmax(lc, -1)
            kl = (logp.exp() * (logp - logc)).sum(-1)
            kl_seq += kl.mean(1).tolist()
            kl_tok.append(kl.reshape(-1))
            flip_seq += (lp.argmax(-1) != lc.argmax(-1)).float().mean(1).tolist()
    out: dict[str, Any] = {"n_seqs": int(tokens.shape[0])}
    out["ppl"] = _stat([math.exp(x) for x in ce_c], n_boot, seed)
    if parent is not None:
        out["ppl_parent"] = _stat([math.exp(x) for x in ce_p], n_boot, seed)
        out["kl_mean"] = _stat(kl_seq, n_boot, seed)
        out["top1_flip"] = _stat(flip_seq, n_boot, seed)
        allkl = torch.cat(kl_tok)
        out["kl_p99"] = {"value": float(torch.quantile(allkl[: 1 << 20], 0.99)), "ci_low": None, "ci_high": None}
        out["ppl_ratio"] = {"value": out["ppl"]["value"] / max(out["ppl_parent"]["value"], 1e-12), "ci_low": None, "ci_high": None}
    return out


def _stat(vals: list[float], n_boot: int, seed: int, alpha: float = 0.05) -> dict[str, float]:
    t = torch.tensor(vals, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(vals), (n_boot, len(vals)), generator=g)
    means = t[idx].mean(1)
    return {"value": float(t.mean()), "ci_low": float(torch.quantile(means, alpha / 2)), "ci_high": float(torch.quantile(means, 1 - alpha / 2))}


def builtin_gate(stats: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """Pass/fail on the locally computable metrics only; always lists the checks that need task suites as not evaluated."""
    reasons: list[str] = []
    verdict = "pass"
    checks = {"kl_p99_max": "kl_p99", "top1_flip_max": "top1_flip", "ppl_ratio_max": "ppl_ratio"}
    for key, metric in checks.items():
        if key not in thresholds:
            continue
        if metric not in stats:
            verdict = "inconclusive" if verdict == "pass" else verdict
            reasons.append(f"{metric}: no parent to compare against")
            continue
        v = stats[metric]["value"]
        ok = v <= thresholds[key]
        reasons.append(f"{metric}={v:.4g} {'<=' if ok else '>'} {thresholds[key]}")
        if not ok:
            verdict = "fail"
    reasons.append("task-suite metrics (mcq, long-context, safety) were NOT evaluated by the built-in gate")
    return {"verdict": verdict, "reasons": reasons}


def distill_kl(
    student: nn.Module,
    teacher: nn.Module,
    tokens: torch.Tensor,
    *,
    steps: int = 50,
    lr: float = 1e-4,
    temperature: float = 1.0,
    batch_size: int = 4,
    trainable: str = "all",
    ctx: RunContext | None = None,
) -> dict[str, Any]:
    """Forward-KL distillation of ``student`` towards ``teacher`` logits for ``steps`` steps (AdamW, no weight decay)."""
    ctx = ctx or NullContext()
    dev = model_device(student)
    tdev = model_device(teacher)
    flags = [(p, p.requires_grad) for p in student.parameters()]
    params: list[nn.Parameter] = []
    if trainable == "all":
        params = [p for p in student.parameters() if p.is_floating_point()]
    else:
        for m in student.modules():
            if type(m).__name__.lower().endswith("norm") or type(m).__name__ == "LowRankLinear":
                params += [p for p in m.parameters(recurse=False) if p.is_floating_point()]
    if not params:
        return {"steps": 0, "note": "no trainable parameters", "kl_before": None, "kl_after": None, "losses": []}
    for p in student.parameters():
        p.requires_grad_(False)
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    batches = list(tokens.split(batch_size))
    teacher.eval()
    losses: list[float] = []

    def kl_of(b: torch.Tensor) -> torch.Tensor:
        ls = student(b.to(dev)).logits.float()
        with torch.no_grad():
            lt = teacher(b.to(tdev)).logits.float().to(dev)
        v = ls.shape[-1]
        return F.kl_div(F.log_softmax(ls / temperature, -1).reshape(-1, v), F.softmax(lt / temperature, -1).reshape(-1, v), reduction="batchmean") * temperature**2

    def eval_loss() -> float:
        student.eval()
        with torch.no_grad():
            return sum(float(kl_of(b)) for b in batches) / len(batches)

    before = eval_loss()
    try:
        student.train()
        for step in range(steps):
            if ctx.should_stop():
                break
            with torch.enable_grad():
                loss = kl_of(batches[step % len(batches)])
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(float(loss.detach()))
            ctx.metric(step, distill_kl=losses[-1])
            ctx.progress((step + 1) / steps, f"distill step {step + 1}/{steps}")
        after = eval_loss()
    finally:
        student.eval()
        for p, f in flags:
            p.requires_grad_(f)
    return {"steps": len(losses), "kl_before": before, "kl_after": after, "losses": losses, "temperature": temperature, "trainable": trainable}
