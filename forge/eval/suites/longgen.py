"""``longgen`` suite (EV-2 long-generation part): errors accumulate over long chains of reasoning (P1 F9).

Two views of the same generations:

1. **Verifiable multi-step arithmetic chains** (synthetic, seeded): the model writes ``step k: value`` lines and a final
   ``Answer: value``.  We score the final answer, the per-step accuracy and the index of the first wrong step.
   ``longgen_reasoning`` is the candidate/parent ratio of final-answer accuracy (the G-W4 gate metric).
2. **Parent-vs-candidate divergence** in free greedy generation: the position of the first differing token and the
   fraction of identical continuations (Plan 10.1 "first-divergence position in free generation").

Optional ``prompts_path`` jsonl ``{"prompt": ..., "answer": "regex or text"}`` adds user-supplied items to view 1 (answer is
a case-insensitive substring unless ``"regex": true``).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import Field

from forge.core.registry import register
from forge.eval.metrics import first_divergence, generate
from forge.eval.suites.base import Suite, SuiteConfig, SuiteEnv, SuiteResult, paired_metric, read_jsonl

_FEWSHOT = (
    "Solve each problem step by step. Write each intermediate value on its own line as 'step k: value' and finish with "
    "'Answer: value'.\n\n"
    "Problem: Start with 3. add 4, multiply by 2.\nstep 1: 7\nstep 2: 14\nAnswer: 14\n\n"
)
_STEP_RE = re.compile(r"step\s*(\d+)\s*:\s*(-?\d+)", re.I)
_ANS_RE = re.compile(r"answer\s*:\s*(-?\d+)", re.I)


@dataclass
class Chain:
    prompt: str
    values: list[int]
    steps: int


def make_chain(rng: random.Random, steps: int) -> Chain:
    v = rng.randint(2, 30)
    start = v
    ops_txt: list[str] = []
    vals: list[int] = []
    for _ in range(steps):
        kind = rng.choice(["add", "subtract", "multiply"] if abs(v) < 300 else ["add", "subtract"])
        if kind == "add":
            n = rng.randint(1, 20)
            v += n
            ops_txt.append(f"add {n}")
        elif kind == "subtract":
            n = rng.randint(1, 20)
            v -= n
            ops_txt.append(f"subtract {n}")
        else:
            n = rng.choice([2, 3])
            v *= n
            ops_txt.append(f"multiply by {n}")
        vals.append(v)
    prompt = f"{_FEWSHOT}Problem: Start with {start}. " + ", ".join(ops_txt) + ".\n"
    return Chain(prompt, vals, steps)


def score_chain(text: str, ch: Chain) -> dict[str, float]:
    steps = {int(k): int(v) for k, v in _STEP_RE.findall(text)}
    first_err = None
    correct = 0
    for i, truth in enumerate(ch.values, 1):
        if steps.get(i) == truth:
            correct += 1
        elif first_err is None:
            first_err = i
    m = _ANS_RE.search(text)
    final_ok = float(m is not None and int(m.group(1)) == ch.values[-1])
    return {
        "final": final_ok, "step_acc": correct / len(ch.values),
        "first_error": float(first_err if first_err is not None else len(ch.values) + 1) / (len(ch.values) + 1),
    }


class LongGenConfig(SuiteConfig):
    """Long-generation / error-accumulation suite."""

    steps: list[int] = Field(default_factory=lambda: [6, 16], description="Chain lengths (number of arithmetic steps) to generate.")
    n_per_length: int = Field(default=16, ge=1, le=10_000, description="Problems per chain length.")
    max_new_tokens: int = Field(default=256, ge=8, le=65_536, description="Generation budget per problem.")
    seed: int = Field(default=4321, ge=0, description="Problem generator seed.")
    vary_seed_with_run: bool = Field(default=False, description="Add the run seed (noise-floor repeats set this).")
    prompts_path: str | None = Field(default=None, description="Optional jsonl of {'prompt','answer','regex'?} extra items.")
    divergence_prompts: int = Field(default=8, ge=0, le=1000, description="How many of the problems also report parent-vs-candidate first-divergence.")


@register("suite", "longgen", meta={"requirement": "EV-2"})
class LongGenSuite(Suite):
    """Multi-step generation accuracy and parent/candidate divergence."""

    name = "longgen"
    version = "1"
    title = "Long generation (error accumulation)"
    Config = LongGenConfig
    describes = "Plan 10.1 long-generation suite; gate metric longgen_reasoning."

    def run(self, cfg: LongGenConfig, env: SuiteEnv) -> SuiteResult:  # type: ignore[override]
        res = SuiteResult(self.name, self.version)
        tok = env.tok
        rng = random.Random(cfg.seed + (env.seed if cfg.vary_seed_with_run else 0))
        chains: list[Chain] = [make_chain(rng, s) for s in cfg.steps for _ in range(cfg.n_per_length)]
        extra: list[dict[str, Any]] = read_jsonl(cfg.prompts_path) if cfg.prompts_path else []
        if cfg.max_items:
            chains = chains[: cfg.max_items]
        eos = getattr(tok, "eos_token_id", None)
        stop = [eos] if eos is not None else []
        n_total = len(chains) + len(extra)
        c_final, p_final = [], []
        c_step, p_step = [], []
        c_first, p_first = [], []
        divs: list[float] = []
        matches: list[float] = []
        for i, ch in enumerate(chains):
            if env.should_stop():
                break
            ids = tok.encode(ch.prompt)
            oc = generate(env.candidate, ids, cfg.max_new_tokens, stop_ids=stop)
            sc = score_chain(tok.decode(oc), ch)
            c_final.append(sc["final"]); c_step.append(sc["step_acc"]); c_first.append(sc["first_error"])
            rec = {"suite": "longgen", "steps": ch.steps, "truth": ch.values[-1], "candidate": tok.decode(oc)}
            if env.parent is not None:
                op = generate(env.parent, ids, cfg.max_new_tokens, stop_ids=stop)
                sp = score_chain(tok.decode(op), ch)
                p_final.append(sp["final"]); p_step.append(sp["step_acc"]); p_first.append(sp["first_error"])
                rec["parent"] = tok.decode(op)
                if i < cfg.divergence_prompts:
                    fd = first_divergence(op, oc)
                    horizon = max(1, max(len(op), len(oc)))
                    divs.append(1.0 if fd is None else fd / horizon)
                    matches.append(1.0 if fd is None else 0.0)
            res.generations.append(rec)
            env.progress((i + 1) / max(1, n_total), "longgen")
        for j, row in enumerate(extra):
            if env.should_stop():
                break
            ids = tok.encode(str(row["prompt"]))
            out = tok.decode(generate(env.candidate, ids, cfg.max_new_tokens, stop_ids=stop))
            ok = (re.search(str(row["answer"]), out, re.I) is not None) if row.get("regex") else (str(row["answer"]).lower() in out.lower())
            c_final.append(float(ok)); c_step.append(float(ok)); c_first.append(1.0 if ok else 0.0)
            if env.parent is not None:
                outp = tok.decode(generate(env.parent, ids, cfg.max_new_tokens, stop_ids=stop))
                okp = (re.search(str(row["answer"]), outp, re.I) is not None) if row.get("regex") else (str(row["answer"]).lower() in outp.lower())
                p_final.append(float(okp)); p_step.append(float(okp)); p_first.append(1.0 if okp else 0.0)
            res.generations.append({"suite": "longgen", "custom": True, "id": row.get("id", j), "candidate": out})
        if not c_final:
            return res
        cf = np.array(c_final)
        pf = np.array(p_final) if env.parent is not None else None
        paired_metric(res, "longgen_acc", pf, cf, env)
        paired_metric(res, "longgen_step_acc", np.array(p_step) if pf is not None else None, np.array(c_step), env)
        res.add("longgen_first_error_pos", float(np.mean(c_first)), n=len(c_first))
        if pf is not None:
            res.add("longgen_first_error_pos_parent", float(np.mean(p_first)), n=len(p_first))
            if "longgen_acc_ratio" in res.metrics:
                # the gate-facing alias: candidate/parent final-answer accuracy
                res.metrics["longgen_reasoning"] = res.metrics["longgen_acc_ratio"].model_copy()
                res.scalars["longgen_reasoning"] = res.scalars["longgen_acc_ratio"]
                res.boot["longgen_reasoning"] = res.boot["longgen_acc_ratio"]
            if divs:
                res.add("longgen_first_divergence", float(np.mean(divs)), n=len(divs))
                res.add("longgen_identical_rate", float(np.mean(matches)), n=len(matches))
            if float(pf.mean()) == 0.0:
                res.notes.append("parent scored 0 on every problem: the ratio is undefined (model too weak for this suite)")
        res.items["longgen"] = (pf if pf is not None else np.array([]), cf)
        res.details.update({"n_items": len(cf), "steps": list(cfg.steps)})
        return res
