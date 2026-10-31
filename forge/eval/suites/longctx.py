"""``longctx`` suite (EV-3): RULER-style synthetic long-context tasks, generated locally (no downloads).

Tasks (each scored by recall of the expected answers in the generated text, per-item score in [0, 1]):

* ``niah_single``     one needle (key -> 7-digit number) in filler text.
* ``niah_multikey``   many distractor key/number needles, query one key.
* ``niah_multivalue`` one key with several numbers spread over the context, recall all.
* ``niah_multiquery`` several keys queried in one question.
* ``variable_tracking`` chains of variable assignments (``VAR A1 = 123`` / ``VAR A2 = VAR A1`` ...) interleaved with noise;
  find every variable holding the value (multi-hop through the chain).
* ``aggregation``     common-words extraction: report the most frequent words of a long word list.
* ``multihop``        chained facts (A's mentor is B ... X lives in C) with distractors; answer by composing 2-3 facts.

Needle-in-a-haystack alone saturates and hides failures (P3 F7), hence the multi-key / multi-hop / aggregation tasks.
The filler is generated from a seeded template grammar, so a (seed, length, task) triple is reproducible.
These are synthetic sanity/regression suites in the style of RULER, **not** the RULER benchmark itself.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import Field

from forge.core.registry import register
from forge.eval.metrics import generate
from forge.eval.suites.base import Suite, SuiteConfig, SuiteEnv, SuiteResult, paired_metric

TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "niah_multiquery", "variable_tracking", "aggregation", "multihop")

_ADJ = "quiet bright old narrow heavy pale warm distant simple broad sharp gentle hollow rapid steady golden dusty frozen".split()
_NOUN = "river lantern orchard bridge harbour meadow engine window garden market tower valley ledger compass kettle ribbon anchor".split()
_VERB = "crossed watched followed lit carried sheltered measured painted circled remembered gathered mended".split()
_PLACE = "hill station shore village canal field square forest road quarry".split()
_SYL_C = "b d f g k l m n p r s t v z".split()
_SYL_V = "a e i o u".split()


def _name(rng: random.Random, syllables: int = 3) -> str:
    return "".join(rng.choice(_SYL_C) + rng.choice(_SYL_V) for _ in range(syllables)).capitalize()


def _key(rng: random.Random) -> str:
    return f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}-{rng.randint(10, 99)}"


def _number(rng: random.Random, digits: int = 7) -> str:
    return str(rng.randint(10 ** (digits - 1), 10**digits - 1))


def _filler_sentence(rng: random.Random) -> str:
    return (f"The {rng.choice(_ADJ)} {rng.choice(_NOUN)} {rng.choice(_VERB)} the {rng.choice(_ADJ)} {rng.choice(_NOUN)} "
            f"near the {rng.choice(_PLACE)}.")


@dataclass
class LongCtxItem:
    task: str
    length: int  # requested prompt length in tokens
    prompt: str
    answers: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    actual_tokens: int = 0


def _assemble(tok: Any, rng: random.Random, target_tokens: int, needles: list[tuple[float, str]], reserve: int) -> str:
    """Filler text of about ``target_tokens - reserve`` tokens with ``needles`` inserted at depth fractions."""
    budget = max(8, target_tokens - reserve)
    sample = [_filler_sentence(rng) + " " for _ in range(16)]
    avg = max(1.0, float(np.mean([len(tok.encode(s)) for s in sample])))
    needle_tok = sum(len(tok.encode(t + " ")) for _, t in needles)
    n_fill = max(1, int((budget - needle_tok) / avg))
    fill = [_filler_sentence(rng) for _ in range(n_fill)]

    def build(fs: list[str]) -> str:
        seq: list[tuple[bool, str]] = [(False, s) for s in fs]
        for depth, text in sorted(needles, key=lambda x: -x[0]):
            pos = min(len(seq), max(0, int(round(depth * len(seq)))))
            seq.insert(pos, (True, text))
        return " ".join(t for _, t in seq)

    text = build(fill)
    for _ in range(8):
        over = len(tok.encode(text)) - budget
        if over <= 0:
            break
        drop = max(1, math.ceil(over / avg))
        fill = fill[: max(1, len(fill) - drop)]
        text = build(fill)
    return text


def _depths(rng: random.Random, k: int) -> list[float]:
    base = [(i + 0.5) / k for i in range(k)]
    return [min(0.98, max(0.02, b + rng.uniform(-0.3, 0.3) / k)) for b in base]


def _finish(task: str, length: int, haystack: str, question: str, answers: list[str], tok: Any, meta: dict[str, Any], instr: str = "") -> LongCtxItem:
    prompt = f"{instr}{haystack}\n\n{question}\nAnswer:"
    return LongCtxItem(task, length, prompt, answers, meta, actual_tokens=len(tok.encode(prompt)))


def gen_niah_single(tok: Any, length: int, seed: int, max_new: int) -> LongCtxItem:
    rng = random.Random(seed)
    key, val = _key(rng), _number(rng)
    needle = f"One of the special magic numbers for {key} is: {val}."
    q = f"What is the special magic number for {key} mentioned in the text above?"
    hay = _assemble(tok, rng, length, [(rng.uniform(0.05, 0.95), needle)], reserve=len(tok.encode(q)) + max_new + 24)
    return _finish("niah_single", length, hay, q, [val], tok, {"key": key})


def _cap(n: int, length: int, per_item_tokens: int = 70, frac: float = 0.35, minimum: int = 1) -> int:
    """Cap the number of inserted needles so that they use at most ``frac`` of the context."""
    return max(minimum, min(n, int(frac * length / per_item_tokens)))


def gen_niah_multikey(tok: Any, length: int, seed: int, max_new: int, n_distractors: int = 8) -> LongCtxItem:
    rng = random.Random(seed)
    n_distractors = _cap(n_distractors, length, minimum=2)
    keys = list({_key(rng) for _ in range(n_distractors + 1)})
    keys = keys[: n_distractors + 1]
    vals = [_number(rng) for _ in keys]
    target = rng.randrange(len(keys))
    needles = [(d, f"One of the special magic numbers for {k} is: {v}.") for d, k, v in zip(_depths(rng, len(keys)), keys, vals)]
    q = f"What is the special magic number for {keys[target]} mentioned in the text above?"
    hay = _assemble(tok, rng, length, needles, reserve=len(tok.encode(q)) + max_new + 24)
    return _finish("niah_multikey", length, hay, q, [vals[target]], tok, {"key": keys[target], "n_keys": len(keys)})


def gen_niah_multivalue(tok: Any, length: int, seed: int, max_new: int, n_values: int = 4) -> LongCtxItem:
    rng = random.Random(seed)
    n_values = _cap(n_values, length, minimum=2)
    key = _key(rng)
    vals = list({_number(rng) for _ in range(n_values)})
    needles = [(d, f"One of the special magic numbers for {key} is: {v}.") for d, v in zip(_depths(rng, len(vals)), vals)]
    q = f"What are all the special magic numbers for {key} mentioned in the text above?"
    hay = _assemble(tok, rng, length, needles, reserve=len(tok.encode(q)) + max_new + 24)
    return _finish("niah_multivalue", length, hay, q, vals, tok, {"key": key})


def gen_niah_multiquery(tok: Any, length: int, seed: int, max_new: int, n_queries: int = 3) -> LongCtxItem:
    rng = random.Random(seed)
    n_queries = _cap(n_queries, length, minimum=2)
    keys = list({_key(rng) for _ in range(n_queries)})
    vals = [_number(rng) for _ in keys]
    needles = [(d, f"One of the special magic numbers for {k} is: {v}.") for d, k, v in zip(_depths(rng, len(keys)), keys, vals)]
    q = "What are the special magic numbers for " + ", ".join(keys) + " mentioned in the text above?"
    hay = _assemble(tok, rng, length, needles, reserve=len(tok.encode(q)) + max_new + 24)
    return _finish("niah_multiquery", length, hay, q, vals, tok, {"keys": keys})


def gen_variable_tracking(tok: Any, length: int, seed: int, max_new: int, hops: int = 4) -> LongCtxItem:
    rng = random.Random(seed)
    hops = _cap(hops, length, per_item_tokens=90, minimum=2)
    letters = rng.sample("ABCDEFGHJKLMNPQRSTUVWXYZ", 2)
    names = [f"{letters[0]}{rng.randint(1, 9)}{i}" for i in range(hops + 1)]
    val = _number(rng, 5)
    lines = [f"VAR {names[0]} = {val}"] + [f"VAR {names[i]} = VAR {names[i - 1]}" for i in range(1, hops + 1)]
    # a second chain with a different value acts as a distractor
    other = _number(rng, 5)
    onames = [f"{letters[1]}{rng.randint(1, 9)}{i}" for i in range(hops + 1)]
    olines = [f"VAR {onames[0]} = {other}"] + [f"VAR {onames[i]} = VAR {onames[i - 1]}" for i in range(1, hops + 1)]
    needles = [(d, t + ".") for d, t in zip(_depths(rng, len(lines)), lines)] + [(d, t + ".") for d, t in zip(_depths(rng, len(olines)), olines)]
    q = f"Find all variables that are assigned the value {val} in the text above, directly or through other variables."
    hay = _assemble(tok, rng, length, needles, reserve=len(tok.encode(q)) + max_new + 24)
    return _finish("variable_tracking", length, hay, q, names, tok, {"value": val, "hops": hops})


def gen_aggregation(tok: Any, length: int, seed: int, max_new: int, top_k: int = 3) -> LongCtxItem:
    rng = random.Random(seed)
    vocab = list({_name(rng, 2).lower() for _ in range(40)})
    heavy, light = vocab[:top_k], vocab[top_k : top_k + 12]
    budget = max(32, length - max_new - 60)
    words: list[str] = []
    avg = max(1.0, len(tok.encode("word ")) + 1)
    n_words = max(top_k * 6, int(budget / avg))
    for i in range(n_words):
        r = rng.random()
        words.append(rng.choice(heavy) if r < 0.45 else rng.choice(light))
    text = " ".join(words)
    for _ in range(8):
        over = len(tok.encode(text)) - budget
        if over <= 0:
            break
        words = words[: max(top_k * 6, len(words) - max(1, int(over / avg)))]
        text = " ".join(words)
    cnt = {w: words.count(w) for w in set(words)}
    top = sorted(cnt, key=lambda w: -cnt[w])[:top_k]
    q = f"What are the {top_k} most frequently appearing words in the list above?"
    return _finish("aggregation", length, text, q, top, tok, {"counts": {w: cnt[w] for w in top}},
                   instr="Below is a long list of words. Count how often each appears.\n")


def gen_multihop(tok: Any, length: int, seed: int, max_new: int, hops: int = 3) -> LongCtxItem:
    rng = random.Random(seed)
    hops = _cap(hops, length, per_item_tokens=120, minimum=2)
    people = list({_name(rng) for _ in range(hops + 1)})[: hops + 1]
    while len(people) < hops + 1:
        people.append(_name(rng, 4))
    city = _name(rng, 3) + "ville"
    facts = [f"{people[i]}'s mentor is {people[i + 1]}." for i in range(hops - 1)]
    facts.append(f"{people[hops - 1]}'s mentor is {people[hops]}.")
    facts.append(f"{people[hops]} lives in {city}.")
    # distractors
    for _ in range(_cap(6, length, per_item_tokens=110, minimum=1)):
        a, b = _name(rng), _name(rng)
        facts.append(f"{a}'s mentor is {b}.")
        facts.append(f"{b} lives in {_name(rng, 3)}ville.")
    rng.shuffle(facts)
    chain = "the mentor of " * hops
    q = f"In which city does {chain}{people[0]} live?"
    needles = [(d, f) for d, f in zip(_depths(rng, len(facts)), facts)]
    hay = _assemble(tok, rng, length, needles, reserve=len(tok.encode(q)) + max_new + 24)
    return _finish("multihop", length, hay, q, [city], tok, {"hops": hops, "chain": people})


GENERATORS = {
    "niah_single": gen_niah_single, "niah_multikey": gen_niah_multikey, "niah_multivalue": gen_niah_multivalue,
    "niah_multiquery": gen_niah_multiquery, "variable_tracking": gen_variable_tracking, "aggregation": gen_aggregation,
    "multihop": gen_multihop,
}


def generate_items(task: str, length: int, n: int, seed: int, tok: Any, max_new: int = 32) -> list[LongCtxItem]:
    if task not in GENERATORS:
        raise ValueError(f"unknown longctx task {task!r}; choose from {sorted(GENERATORS)}")
    return [GENERATORS[task](tok, length, seed * 100003 + i * 7919 + _task_salt(task), max_new) for i in range(n)]


def _task_salt(task: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(task))


def score_output(text: str, answers: list[str]) -> float:
    """Recall of expected answers in the output (case-insensitive substring, digits compared as tokens)."""
    if not answers:
        return 0.0
    low = text.lower()
    hit = 0
    for a in answers:
        al = a.lower()
        if al.isdigit():
            hit += bool(re.search(rf"(?<!\d){re.escape(al)}(?!\d)", low))
        else:
            hit += al in low
    return hit / len(answers)


class LongCtxConfig(SuiteConfig):
    """RULER-style synthetic long-context suite."""

    lengths: list[int] = Field(default_factory=lambda: [1024, 4096], description="Prompt lengths in tokens to evaluate (run at the deployment length too).")
    tasks: list[str] = Field(default_factory=lambda: list(TASKS), description=f"Tasks to run: {', '.join(TASKS)}.")
    n_per_task: int = Field(default=8, ge=1, le=10_000, description="Items per task per length.")
    max_new_tokens: int = Field(default=32, ge=1, le=4096, description="Generation budget per item.")
    seed: int = Field(default=1234, ge=0, description="Base seed of the item generators (also varied by noise-floor repeats).")
    vary_seed_with_run: bool = Field(default=False, description="Add the run seed to the generator seed (noise-floor repeats set this).")


@register("suite", "longctx", meta={"requirement": "EV-3"})
class LongCtxSuite(Suite):
    """Long-context retrieval, multi-key, multi-hop, tracking and aggregation at the requested lengths."""

    name = "longctx"
    version = "1"
    title = "Long-context (RULER-style, synthetic)"
    Config = LongCtxConfig
    describes = "EV-3; gate metric longctx_ratio (candidate/parent at the deployment length)."

    def run(self, cfg: LongCtxConfig, env: SuiteEnv) -> SuiteResult:  # type: ignore[override]
        res = SuiteResult(self.name, self.version)
        tok = env.tok
        unknown = [t for t in cfg.tasks if t not in GENERATORS]
        if unknown:
            raise ValueError(f"unknown longctx tasks {unknown}")
        base_seed = cfg.seed + (env.seed if cfg.vary_seed_with_run else 0)
        all_items: list[LongCtxItem] = []
        for L in cfg.lengths:
            for t in cfg.tasks:
                all_items += generate_items(t, L, cfg.n_per_task, base_seed, tok, cfg.max_new_tokens)
        if cfg.max_items:
            all_items = all_items[: cfg.max_items]
        c_scores = np.zeros(len(all_items))
        p_scores = np.zeros(len(all_items)) if env.parent is not None else None
        eos = getattr(tok, "eos_token_id", None)
        for i, it in enumerate(all_items):
            if env.should_stop():
                all_items = all_items[:i]
                c_scores = c_scores[:i]
                p_scores = p_scores[:i] if p_scores is not None else None
                break
            ids = tok.encode(it.prompt)
            out_c = generate(env.candidate, ids, cfg.max_new_tokens, stop_ids=[eos] if eos is not None else [])
            txt_c = tok.decode(out_c)
            c_scores[i] = score_output(txt_c, it.answers)
            rec = {"suite": "longctx", "task": it.task, "length": it.length, "answers": it.answers, "candidate": txt_c, "prompt_tokens": len(ids)}
            if p_scores is not None:
                txt_p = tok.decode(generate(env.parent, ids, cfg.max_new_tokens, stop_ids=[eos] if eos is not None else []))
                p_scores[i] = score_output(txt_p, it.answers)
                rec["parent"] = txt_p
            res.generations.append(rec)
            env.progress((i + 1) / max(1, len(all_items)), f"longctx {it.task}@{it.length}")
        if len(all_items) == 0:
            return res
        paired_metric(res, "longctx_score", p_scores, c_scores, env)
        lengths = np.array([it.length for it in all_items])
        tasks = np.array([it.task for it in all_items])
        for L in sorted(set(lengths.tolist())):
            m = lengths == L
            res.add(f"longctx_len{L}", float(c_scores[m].mean()), n=int(m.sum()))
            if p_scores is not None:
                res.add(f"longctx_len{L}_parent", float(p_scores[m].mean()), n=int(m.sum()))
        for t in sorted(set(tasks.tolist())):
            m = tasks == t
            # candidate score, parent score, paired diff (points) with bootstrap, and ratio (G-LONG uses *_diff_pts)
            paired_metric(res, f"longctx_{t}", p_scores[m] if p_scores is not None else None, c_scores[m], env, as_points=True)
        # ratio at the deployment length (nearest configured length), else pooled
        dep = env.deployment_length
        match = True
        if dep is not None and len(cfg.lengths):
            L_star = min(cfg.lengths, key=lambda L: (abs(L - dep), L))
            match = L_star == dep
            sel = lengths == L_star
            res.details["ratio_length"] = int(L_star)
        else:
            sel = np.ones(len(all_items), dtype=bool)
            res.details["ratio_length"] = "pooled"
        res.flags["longctx_length_matches_deployment"] = bool(match)
        if not match:
            res.notes.append(f"deployment length {dep} not evaluated; ratio reported at {res.details['ratio_length']}")
        if p_scores is not None and sel.any():
            from forge.eval import stats

            r = stats.paired_bootstrap(p_scores[sel], c_scores[sel], env.n_boot, stat="ratio", alpha=env.alpha, seed=env.seed + 11)
            res.add("longctx_ratio", r.estimate if math.isfinite(r.estimate) else float("nan"), ci=(r.ci_low, r.ci_high), n=int(sel.sum()))
            res.boot["longctx_ratio"] = stats.bootstrap_distribution(p_scores[sel], c_scores[sel], stat="ratio", n=env.n_boot, seed=env.seed + 11)
        res.items["longctx"] = (p_scores if p_scores is not None else np.array([]), c_scores)
        res.details.update({"n_items": len(all_items), "lengths": list(cfg.lengths), "tasks": list(cfg.tasks),
                            "actual_prompt_tokens_mean": float(np.mean([it.actual_tokens for it in all_items]))})
        return res
