"""``mcq`` suite (EV-2): generic multiple-choice log-likelihood scorer driven by a local jsonl file.

jsonl item format::

    {"id": "q1", "question": "...", "choices": ["a", "b", "c", "d"], "answer": 2 | "C",
     "subject": "optional", "context": "optional passage placed before the question"}

Scoring modes (``scoring``): ``sum`` (total log-prob of the choice text), ``mean_token`` (per-token mean),
``byte_norm`` (log-prob per UTF-8 byte; the lm-eval ``acc_norm`` convention), ``letter`` (all options listed in the
prompt, compare the log-prob of the letters A/B/C/...).
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
from pydantic import Field

from forge.core.errors import ConfigError
from forge.core.registry import register
from forge.eval.metrics import ece_from_confidences
from forge.eval.suites.base import Suite, SuiteConfig, SuiteEnv, SuiteResult, paired_metric, read_jsonl, resolve_data_path, score_continuations

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class MCQConfig(SuiteConfig):
    """Multiple-choice accuracy and calibration from a local jsonl file."""

    data_path: str | None = Field(
        default=None, description="jsonl of {question, choices, answer}. Empty = bundled 24-item SEED set (smoke test, not a benchmark)."
    )
    scoring: str = Field(default="byte_norm", pattern="^(sum|mean_token|byte_norm|letter)$", description="sum | mean_token | byte_norm | letter.")
    template: str = Field(default="{context}Question: {question}\nAnswer:", description="Prompt template; fields {context} {question} {subject}.")
    batch_size: int = Field(default=8, ge=1, le=512, description="Sequences per forward pass.")
    max_len: int | None = Field(default=None, ge=8, description="Left-truncate prompts to this many tokens (None = no limit).")
    shuffle_choices: bool = Field(default=False, description="Shuffle choice order per item (seeded). Used by noise-floor repeats to vary the run.")
    n_bins: int = Field(default=10, ge=2, le=50, description="ECE bins.")
    group_by_subject: bool = Field(default=True, description="Also report per-subject accuracy (point estimates).")


def parse_answer(ans: Any, n_choices: int) -> int:
    if isinstance(ans, bool):
        raise ConfigError("answer must be an index or letter")
    if isinstance(ans, int):
        idx = ans
    elif isinstance(ans, str) and len(ans.strip()) == 1 and ans.strip().upper() in LETTERS:
        idx = LETTERS.index(ans.strip().upper())
    elif isinstance(ans, str) and ans.strip().isdigit():
        idx = int(ans.strip())
    else:
        raise ConfigError(f"cannot parse answer {ans!r}")
    if not 0 <= idx < n_choices:
        raise ConfigError(f"answer {ans!r} out of range for {n_choices} choices")
    return idx


def load_items(cfg: MCQConfig) -> list[dict[str, Any]]:
    path = resolve_data_path(cfg.data_path, "mcq_smoke.jsonl")
    rows = read_jsonl(path)  # type: ignore[arg-type]
    items = []
    for i, r in enumerate(rows):
        if "choices" not in r or "question" not in r or "answer" not in r:
            raise ConfigError(f"{path}: item {i} needs question/choices/answer")
        ch = [str(c) for c in r["choices"]]
        if len(ch) < 2:
            raise ConfigError(f"{path}: item {i} needs >= 2 choices")
        items.append({"id": str(r.get("id", f"q{i}")), "question": str(r["question"]), "choices": ch,
                      "answer": parse_answer(r["answer"], len(ch)), "subject": str(r.get("subject", "")),
                      "context": str(r.get("context", ""))})
        if cfg.max_items and len(items) >= cfg.max_items:
            break
    return items


def build_prompt(cfg: MCQConfig, it: dict[str, Any]) -> str:
    ctx = (it["context"].rstrip() + "\n\n") if it["context"] else ""
    base = cfg.template.format(context=ctx, question=it["question"], subject=it["subject"])
    if cfg.scoring == "letter":
        opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(it["choices"]))
        base = base.rsplit("Answer:", 1)[0] if "Answer:" in base else base
        base = f"{base.rstrip()}\n{opts}\nAnswer:"
    return base


def choice_scores(provider: Any, tok: Any, cfg: MCQConfig, items: list[dict[str, Any]]) -> list[np.ndarray]:
    """Per item array of per-choice scores (higher = preferred)."""
    ctxs: list[list[int]] = []
    conts: list[list[int]] = []
    owner: list[int] = []
    metas: list[tuple[float, int]] = []
    for ii, it in enumerate(items):
        prompt_ids = tok.encode(build_prompt(cfg, it))
        if not prompt_ids:
            prompt_ids = tok.encode("\n")
        for ci, ch in enumerate(it["choices"]):
            text = f" {LETTERS[ci]}" if cfg.scoring == "letter" else f" {ch}"
            cont = tok.encode(text)
            if not cont:
                raise ConfigError(f"choice {ch!r} encodes to zero tokens")
            ctxs.append(prompt_ids)
            conts.append(cont)
            owner.append(ii)
            metas.append((max(1, len(text.encode("utf-8"))), len(cont)))
    lp, nt = score_continuations(provider, ctxs, conts, batch_size=cfg.batch_size, max_len=cfg.max_len)
    out: list[np.ndarray] = [np.zeros(len(it["choices"])) for it in items]
    pos = [0] * len(items)
    for k, ii in enumerate(owner):
        nb, ntok = metas[k]
        if cfg.scoring == "sum" or cfg.scoring == "letter":
            s = lp[k]
        elif cfg.scoring == "mean_token":
            s = lp[k] / ntok
        else:
            s = lp[k] / nb
        out[ii][pos[ii]] = s
        pos[ii] += 1
    return out


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (x - x.max()) / temp
    e = np.exp(z)
    return e / e.sum()


@register("suite", "mcq", meta={"requirement": "EV-2"})
class MCQSuite(Suite):
    """Multiple-choice accuracy (paired with the parent on identical items) + calibration."""

    name = "mcq"
    version = "1"
    title = "Multiple-choice accuracy (log-likelihood)"
    Config = MCQConfig
    describes = "EV-2 task suite from local jsonl; paired items give paired bootstrap CIs."

    def run(self, cfg: MCQConfig, env: SuiteEnv) -> SuiteResult:  # type: ignore[override]
        res = SuiteResult(self.name, self.version, contributes_to_suite_score=True)
        items = load_items(cfg)
        if not items:
            res.notes.append("no items")
            return res
        if cfg.shuffle_choices:
            rng = random.Random(env.seed)
            for it in items:
                order = list(range(len(it["choices"])))
                rng.shuffle(order)
                it["choices"] = [it["choices"][i] for i in order]
                it["answer"] = order.index(it["answer"])
        tok = env.tok
        cand_scores = choice_scores(env.candidate, tok, cfg, items)
        env.progress(0.5 if env.parent is not None else 1.0, "mcq candidate")
        cand_correct = np.array([float(np.argmax(s) == it["answer"]) for s, it in zip(cand_scores, items)])
        parent_correct = None
        par_scores = None
        if env.parent is not None:
            par_scores = choice_scores(env.parent, tok, cfg, items)
            parent_correct = np.array([float(np.argmax(s) == it["answer"]) for s, it in zip(par_scores, items)])
        paired_metric(res, "mcq_acc", parent_correct, cand_correct, env, as_points=True)
        # calibration over the choice distribution (the score scale for byte_norm is nats/byte -> rescale by mean length)
        def _ece(scores: list[np.ndarray]) -> float:
            conf, corr = [], []
            for s, it in zip(scores, items):
                temp = 1.0 if cfg.scoring in ("sum", "letter") else 1.0 / max(1.0, np.mean([len(c.encode()) for c in it["choices"]]) if cfg.scoring == "byte_norm" else 1.0)
                p = softmax(s, temp)
                conf.append(p.max())
                corr.append(float(np.argmax(p) == it["answer"]))
            return ece_from_confidences(conf, corr, cfg.n_bins).ece

        res.add("mcq_ece", _ece(cand_scores), n=len(items))
        if par_scores is not None:
            res.add("mcq_ece_parent", _ece(par_scores), n=len(items))
            res.add("mcq_ece_delta", res.scalars["mcq_ece"] - res.scalars["mcq_ece_parent"], n=len(items))
        res.items["score"] = (parent_correct if parent_correct is not None else np.array([]), cand_correct)
        if cfg.group_by_subject:
            subj: dict[str, list[int]] = {}
            for i, it in enumerate(items):
                if it["subject"]:
                    subj.setdefault(it["subject"], []).append(i)
            res.details["by_subject"] = {s: {"n": len(ix), "acc": float(cand_correct[ix].mean()),
                                           **({"acc_parent": float(parent_correct[ix].mean())} if parent_correct is not None else {})}
                                       for s, ix in sorted(subj.items())}
        res.details.update({"n_items": len(items), "scoring": cfg.scoring, "chance": float(np.mean([1 / len(it["choices"]) for it in items]))})
        res.generations = [{"suite": "mcq", "id": it["id"], "cand_scores": [float(x) for x in cs], "pred": int(np.argmax(cs)), "answer": it["answer"]}
                           for it, cs in zip(items, cand_scores)]
        return res
