"""``agentic`` suite (EV-6): multi-step tool use against a **deterministic simulated tool environment**.

Hybrid / sparse / compressed models showed deficits on multi-hop and tool-use tasks that average benchmarks missed
(P3 F7), hence this mini-suite.  Everything is generated from a seed - no network, no real tools:

* tools: ``lookup(entity, attribute)``, ``search(prefix)``, ``calc(expression)``, ``submit(answer)``
* tasks: ``hop2`` (compose two lookups + a third), ``calc_total`` (two price lookups + arithmetic), ``search_lookup``
  (find an entity by prefix, then look it up), ``hop3`` (three chained facts).
* protocol: the model writes ``Action: tool("arg", ...)``; the environment answers ``Observation: ...``; the episode ends on
  ``submit`` or after ``max_turns``.
* a scripted :func:`oracle_actions` policy solves every task and is used to validate the environment (tests) and to build the
  one-shot demonstration in the prompt.
"""
from __future__ import annotations

import ast
import operator
import random
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import Field

from forge.core.registry import register
from forge.eval.metrics import generate
from forge.eval.suites.base import Suite, SuiteConfig, SuiteEnv, SuiteResult, paired_metric

_SYL_C = "b d f g k l m n p r s t v z".split()
_SYL_V = "a e i o u".split()


def _name(rng: random.Random, syl: int = 3) -> str:
    return "".join(rng.choice(_SYL_C) + rng.choice(_SYL_V) for _ in range(syl)).capitalize()


class SimEnvironment:
    """Seeded knowledge base + tool executor."""

    def __init__(self, seed: int, n_people: int = 12, n_cities: int = 6, n_products: int = 6) -> None:
        rng = random.Random(seed)
        self.cities = sorted({_name(rng, 3) + "ton" for _ in range(n_cities)})
        self.people = sorted({_name(rng, 3) for _ in range(n_people)})
        self.products = sorted({_name(rng, 2).lower() + "s" for _ in range(n_products)})
        self.kb: dict[str, dict[str, Any]] = {}
        for c in self.cities:
            self.kb[c] = {"population": rng.randint(10, 900) * 1000, "country": _name(rng, 2) + "ia"}
        for p in self.people:
            self.kb[p] = {"manager": rng.choice([q for q in self.people if q != p]), "city": rng.choice(self.cities), "age": rng.randint(22, 64)}
        for pr in self.products:
            self.kb[pr] = {"price": rng.randint(2, 60), "stock": rng.randint(0, 500)}
        self.submitted: str | None = None
        self.calls: list[tuple[str, tuple]] = []

    # tools -----------------------------------------------------------------------------
    def lookup(self, entity: str, attribute: str) -> str:
        rec = self.kb.get(str(entity))
        if rec is None:
            return f"ERROR: unknown entity '{entity}'"
        if attribute not in rec:
            return f"ERROR: entity '{entity}' has no attribute '{attribute}' (available: {', '.join(sorted(rec))})"
        return str(rec[attribute])

    def search(self, prefix: str) -> str:
        hits = sorted(k for k in self.kb if k.lower().startswith(str(prefix).lower()))
        return ", ".join(hits) if hits else "no matches"

    def calc(self, expression: str) -> str:
        try:
            return str(_safe_eval(str(expression)))
        except Exception as e:
            return f"ERROR: {e}"

    def submit(self, answer: Any) -> str:
        self.submitted = str(answer)
        return "submitted"

    def execute(self, tool: str, args: tuple) -> str:
        self.calls.append((tool, args))
        fn = {"lookup": self.lookup, "search": self.search, "calc": self.calc, "submit": self.submit}.get(tool)
        if fn is None:
            return f"ERROR: unknown tool '{tool}'"
        try:
            return fn(*args)
        except TypeError:
            return f"ERROR: wrong arguments for {tool}"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.USub: operator.neg}


def _safe_eval(expr: str) -> int | float:
    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")

    v = ev(ast.parse(expr.strip(), mode="eval"))
    return int(v) if isinstance(v, float) and v.is_integer() else v


@dataclass
class AgentTask:
    kind: str
    question: str
    answer: str
    seed: int


def make_task(env: SimEnvironment, kind: str, rng: random.Random, seed: int) -> AgentTask:
    if kind == "hop2":
        p = rng.choice(env.people)
        mgr = env.kb[p]["manager"]
        city = env.kb[mgr]["city"]
        return AgentTask(kind, f"What is the population of the city where {p}'s manager lives?", str(env.kb[city]["population"]), seed)
    if kind == "hop3":
        p = rng.choice(env.people)
        mgr = env.kb[env.kb[p]["manager"]]["manager"]
        city = env.kb[mgr]["city"]
        return AgentTask(kind, f"What country is the city in which {p}'s manager's manager lives located in?", str(env.kb[city]["country"]), seed)
    if kind == "calc_total":
        a, b = rng.sample(env.products, 2)
        ka, kb = rng.randint(2, 9), rng.randint(2, 9)
        total = ka * env.kb[a]["price"] + kb * env.kb[b]["price"]
        return AgentTask(kind, f"What is the total price of {ka} units of {a} and {kb} units of {b}?", str(total), seed)
    if kind == "search_lookup":
        p = rng.choice(env.people)
        prefix = p
        for k in range(3, len(p) + 1):
            if sum(1 for key in env.kb if key.lower().startswith(p[:k].lower())) == 1:
                prefix = p[:k]
                break
        return AgentTask(kind, f"Find the person whose name starts with '{prefix}' and report the city they live in.", str(env.kb[p]["city"]), seed)
    raise ValueError(f"unknown agentic task kind {kind!r}")


KINDS = ("hop2", "calc_total", "search_lookup", "hop3")


def oracle_actions(env: SimEnvironment, task: AgentTask) -> list[tuple[str, tuple]]:
    """Scripted optimal tool-call sequence for ``task`` (used to validate the environment and to demonstrate the format)."""
    q = task.question
    if task.kind == "hop2":
        p = re.search(r"where (\w+)'s manager", q).group(1)  # type: ignore[union-attr]
        mgr = env.kb[p]["manager"]
        city = env.kb[mgr]["city"]
        return [("lookup", (p, "manager")), ("lookup", (mgr, "city")), ("lookup", (city, "population")), ("submit", (task.answer,))]
    if task.kind == "hop3":
        p = re.search(r"in which (\w+)'s manager's manager", q) or re.search(r"city in which (\w+)'s manager's manager", q)
        p = p.group(1)  # type: ignore[union-attr]
        m1 = env.kb[p]["manager"]
        m2 = env.kb[m1]["manager"]
        city = env.kb[m2]["city"]
        return [("lookup", (p, "manager")), ("lookup", (m1, "manager")), ("lookup", (m2, "city")), ("lookup", (city, "country")), ("submit", (task.answer,))]
    if task.kind == "calc_total":
        m = re.search(r"of (\d+) units of (\w+) and (\d+) units of (\w+)\?", q)
        ka, a, kb, b = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)  # type: ignore[union-attr]
        return [("lookup", (a, "price")), ("lookup", (b, "price")), ("calc", (f"{ka} * {env.kb[a]['price']} + {kb} * {env.kb[b]['price']}",)), ("submit", (task.answer,))]
    if task.kind == "search_lookup":
        prefix = re.search(r"starts with '([^']+)'", q).group(1)  # type: ignore[union-attr]
        name = env.search(prefix).split(", ")[0]  # the prefix is unique across the whole KB by construction
        return [("search", (prefix,)), ("lookup", (name, "city")), ("submit", (task.answer,))]
    raise ValueError(task.kind)


def fmt_action(tool: str, args: tuple) -> str:
    return f"Action: {tool}(" + ", ".join(repr(a) if not isinstance(a, str) else '"' + a.replace('"', "'") + '"' for a in args) + ")"


_ACTION_RE = re.compile(r"Action:\s*([A-Za-z_]\w*)\((.*)\)")

SYSTEM = (
    "You are an agent that answers questions by calling tools. Available tools:\n"
    "  lookup(entity, attribute)  - read a fact (attributes: manager, city, age, population, country, price, stock)\n"
    "  search(prefix)             - list entities whose name starts with prefix\n"
    "  calc(expression)           - evaluate arithmetic\n"
    "  submit(answer)             - give the final answer and stop\n"
    "Reply with exactly one line 'Action: tool(args)' per turn; the system replies with an Observation.\n\n"
)


def parse_action(text: str) -> tuple[str, tuple] | None:
    """First well-formed ``Action: tool(args)`` line in ``text`` (None if absent or unparsable)."""
    m = _ACTION_RE.search(text)
    if not m:
        return None
    try:
        parsed = ast.literal_eval("(" + m.group(2) + ",)")
    except Exception:
        return None
    return m.group(1), tuple(parsed)


def demo_trace(seed: int = 99991) -> str:
    env = SimEnvironment(seed)
    rng = random.Random(seed)
    task = make_task(env, "hop2", rng, seed)
    lines = [f"Question: {task.question}"]
    for tool, args in oracle_actions(env, task):
        lines.append(fmt_action(tool, args))
        lines.append(f"Observation: {env.execute(tool, args)}")
    return "Example:\n" + "\n".join(lines) + "\n\n"


class AgenticConfig(SuiteConfig):
    """Deterministic simulated tool-use episodes."""

    kinds: list[str] = Field(default_factory=lambda: list(KINDS), description=f"Task kinds: {', '.join(KINDS)}.")
    n_per_kind: int = Field(default=8, ge=1, le=10_000, description="Episodes per task kind.")
    max_turns: int = Field(default=8, ge=1, le=64, description="Maximum tool calls before the episode is a failure.")
    max_new_tokens_per_turn: int = Field(default=48, ge=4, le=1024, description="Generation budget per turn.")
    seed: int = Field(default=2468, ge=0, description="Environment / task seed.")
    vary_seed_with_run: bool = Field(default=False, description="Add the run seed (noise-floor repeats set this).")
    one_shot: bool = Field(default=True, description="Include a worked example in the prompt.")


def run_episode(provider: Any, tok: Any, env_sim: SimEnvironment, task: AgentTask, cfg: AgenticConfig) -> dict[str, Any]:
    prompt = SYSTEM + (demo_trace() if cfg.one_shot else "") + f"Question: {task.question}\n"
    eos = getattr(tok, "eos_token_id", None)
    oracle = oracle_actions(env_sim, task)
    valid = 0
    calls = 0
    matched = 0
    transcript: list[str] = []
    for turn in range(cfg.max_turns):
        ids = tok.encode(prompt)
        out = tok.decode(generate(provider, ids, cfg.max_new_tokens_per_turn, stop_ids=[eos] if eos is not None else []))
        out = out.split("Observation:")[0]
        act = parse_action(out)
        calls += 1
        if act is None:
            transcript.append(out.strip()[:200])
            prompt += out.split("\n")[0] + "\nObservation: ERROR: malformed action. Use Action: tool(args)\n"
            continue
        tool, args = act
        valid += 1
        if turn < len(oracle) and oracle[turn][0] == tool and tuple(map(str, oracle[turn][1])) == tuple(map(str, args)):
            matched += 1
        obs = env_sim.execute(tool, args)
        line = fmt_action(tool, args)
        transcript.append(line + " -> " + obs[:120])
        prompt += line + f"\nObservation: {obs}\n"
        if tool == "submit":
            break
    correct = env_sim.submitted is not None and env_sim.submitted.strip() == task.answer
    return {"success": float(correct), "valid_rate": valid / max(1, calls), "step_match": matched / max(1, len(oracle)), "turns": calls,
            "transcript": transcript}


@register("suite", "agentic", meta={"requirement": "EV-6"})
class AgenticSuite(Suite):
    """Multi-step tool use in a seeded simulated environment (success, valid-call rate, oracle step match)."""

    name = "agentic"
    version = "1"
    title = "Agentic tool use (simulated)"
    Config = AgenticConfig
    describes = "EV-6; deterministic environment, scripted oracle validates the tasks."

    def run(self, cfg: AgenticConfig, env: SuiteEnv) -> SuiteResult:  # type: ignore[override]
        res = SuiteResult(self.name, self.version)
        tok = env.tok
        seed0 = cfg.seed + (env.seed if cfg.vary_seed_with_run else 0)
        tasks: list[tuple[SimEnvironment, AgentTask]] = []
        for ki, kind in enumerate(cfg.kinds):
            for i in range(cfg.n_per_kind):
                s = seed0 * 1009 + ki * 101 + i
                e = SimEnvironment(s)
                tasks.append((e, make_task(e, kind, random.Random(s), s)))
        if cfg.max_items:
            tasks = tasks[: cfg.max_items]
        c_s, p_s, c_v, c_m, c_t = [], [], [], [], []
        by_kind: dict[str, list[float]] = {}
        for i, (e, t) in enumerate(tasks):
            if env.should_stop():
                break
            rc = run_episode(env.candidate, tok, e, t, cfg)
            c_s.append(rc["success"]); c_v.append(rc["valid_rate"]); c_m.append(rc["step_match"]); c_t.append(rc["turns"])
            by_kind.setdefault(t.kind, []).append(rc["success"])
            rec = {"suite": "agentic", "kind": t.kind, "question": t.question, "answer": t.answer, "candidate": rc["transcript"], "success": rc["success"]}
            if env.parent is not None:
                e2 = SimEnvironment(t.seed)
                rp = run_episode(env.parent, tok, e2, t, cfg)
                p_s.append(rp["success"])
                rec["parent"] = rp["transcript"]
            res.generations.append(rec)
            env.progress((i + 1) / len(tasks), "agentic")
        if not c_s:
            return res
        paired_metric(res, "agentic_success", np.array(p_s) if p_s else None, np.array(c_s), env, as_points=True)
        res.add("agentic_valid_call_rate", float(np.mean(c_v)), n=len(c_v))
        res.add("agentic_step_match", float(np.mean(c_m)), n=len(c_m))
        res.add("agentic_mean_turns", float(np.mean(c_t)), n=len(c_t))
        res.details["by_kind"] = {k: float(np.mean(v)) for k, v in by_kind.items()}
        res.items["agentic"] = (np.array(p_s), np.array(c_s))
        return res
