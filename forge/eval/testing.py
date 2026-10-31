"""Synthetic providers for tests, notebooks and UI demos of the evaluation harness (no model weights needed).

* :class:`BigramProvider`  - a random-but-deterministic bigram "language model" (logits = table[token]); ``perturbed(scale)``
  yields a noisy copy, a stand-in for a lossy compression of the parent.
* :class:`SolverProvider`  - a *scripted solver* that reads the prompts produced by the built-in suites (long-context needles,
  arithmetic chains, tool-use episodes, refusal probes) and answers them, failing a deterministic fraction ``1 - skill`` of the
  time.  It is NOT a model: it exists to prove that suites, scoring, pairing, noise floors and gates work end to end.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Sequence

from forge.eval.metrics import CallableProvider
from forge.eval.suites.base import ByteTokenizer, DATA_DIR, read_jsonl


class BigramProvider(CallableProvider):
    """logits[b, t] = table[input_ids[b, t]]  (position-independent, causal by construction)."""

    def __init__(self, vocab: int = 257, seed: int = 0, scale: float = 2.0, name: str = "bigram", table: Any = None) -> None:
        import torch

        g = torch.Generator().manual_seed(seed)
        self.table = table if table is not None else torch.randn(vocab, vocab, generator=g) * scale
        self.vocab = vocab
        self.seed = seed
        self.scale = scale
        super().__init__(lambda ids: self.table[ids], name=name, nbytes=int(self.table.numel() * 4))

    def perturbed(self, noise: float, seed: int = 1, name: str | None = None) -> "BigramProvider":
        import torch

        g = torch.Generator().manual_seed(seed)
        return BigramProvider(self.vocab, self.seed, self.scale, name or f"{self.name}+noise{noise}", table=self.table + noise * torch.randn(self.table.shape, generator=g))


def _fails(prompt: str, skill: float, salt: str = "") -> bool:
    if skill >= 1.0:
        return False
    h = int(hashlib.sha256((salt + prompt).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h >= skill


class SolverProvider(BigramProvider):
    """Scripted solver for the built-in suites; ``skill`` in [0, 1] is the probability of answering correctly."""

    def __init__(self, skill: float = 1.0, seed: int = 0, name: str = "solver", refuse: bool = True) -> None:
        super().__init__(257, seed, 1.0, name)
        self.skill = skill
        self.refuse = refuse
        self.tok = ByteTokenizer()
        self.eos_token_id = self.tok.eos_token_id
        self._harmful = {r["prompt"] for r in read_jsonl(DATA_DIR / "safety_refusal_seed.jsonl") if r.get("kind") == "harmful_generic"}

    # ---- generation ------------------------------------------------------------------------
    def generate(self, prompt_ids: Sequence[int], max_new_tokens: int, *, stop_ids: Sequence[int] = (), temperature: float = 0.0, seed: int | None = None) -> list[int]:
        text = self.tok.decode(prompt_ids)
        out = self.respond(text)
        return self.tok.encode(out)[:max_new_tokens]

    def respond(self, text: str) -> str:
        if self.refuse and any(h in text for h in self._harmful):
            return "I'm sorry, I can't help with that." if not _fails(text, self.skill, "ref") else "Sure, here you go."
        if "Problem: Start with" in text and "step k: value" in text:
            return self._chain(text)
        if "You are an agent that answers questions by calling tools" in text:
            return self._agent(text)
        if text.rstrip().endswith("Answer:"):
            return " " + self._ctx_answer(text)
        return "ok"

    # ---- long generation -----------------------------------------------------------------
    def _chain(self, text: str) -> str:
        m = list(re.finditer(r"Problem: Start with (-?\d+)\. ([^\n]*)\.\n", text))[-1]
        v = int(m.group(1))
        lines = []
        wrong = _fails(text, self.skill, "chain")
        for i, op in enumerate(m.group(2).split(", "), 1):
            k, n = re.match(r"(add|subtract|multiply by) (\d+)", op).groups()  # type: ignore[union-attr]
            v = v + int(n) if k == "add" else v - int(n) if k == "subtract" else v * int(n)
            shown = v + 1 if (wrong and i == 3) else v
            lines.append(f"step {i}: {shown}")
        ans = v + 1 if wrong else v
        return "\n".join(lines) + f"\nAnswer: {ans}\n"

    # ---- agentic ----------------------------------------------------------------------------
    def _agent(self, text: str) -> str:
        q = text.rsplit("Question:", 1)[1]
        question = q.split("\n", 1)[0]
        obs = re.findall(r"Observation: ([^\n]*)", q)
        step = len(obs)
        if _fails(text, self.skill, "agent") and step >= 1:
            return 'Action: submit("unknown")'
        a = lambda t, *args: "Action: " + t + "(" + ", ".join('"' + str(x).replace('"', "'") + '"' for x in args) + ")"
        m = re.search(r"where (\w+)'s manager lives", question)
        if m:
            p = m.group(1)
            return (a("lookup", p, "manager") if step == 0 else a("lookup", obs[0], "city") if step == 1
                    else a("lookup", obs[1], "population") if step == 2 else a("submit", obs[2]))
        m = re.search(r"city in which (\w+)'s manager's manager", question)
        if m:
            return (a("lookup", m.group(1), "manager") if step == 0 else a("lookup", obs[0], "manager") if step == 1 else a("lookup", obs[1], "city")
                    if step == 2 else a("lookup", obs[2], "country") if step == 3 else a("submit", obs[3]))
        m = re.search(r"of (\d+) units of (\w+) and (\d+) units of (\w+)\?", question)
        if m:
            ka, pa, kb, pb = m.groups()
            return (a("lookup", pa, "price") if step == 0 else a("lookup", pb, "price") if step == 1 else
                    a("calc", f"{ka} * {obs[0]} + {kb} * {obs[1]}") if step == 2 else a("submit", obs[2]))
        m = re.search(r"starts with '([^']+)'", question)
        if m:
            return a("search", m.group(1)) if step == 0 else a("lookup", obs[0].split(", ")[0], "city") if step == 1 else a("submit", obs[1])
        return "Action: submit(\"unknown\")"

    # ---- long context --------------------------------------------------------------------------
    def _ctx_answer(self, text: str) -> str:
        if _fails(text, self.skill, "ctx"):
            return "I do not know."
        body, qa = text.rsplit("\n\n", 1)
        question = qa.split("\nAnswer:")[0].strip()
        m = re.match(r"What are all the special magic numbers for (\S+) mentioned", question)
        if m:
            return " ".join(re.findall(rf"magic numbers for {re.escape(m.group(1))} is: (\d+)", body))
        m = re.match(r"What are the special magic numbers for (.+) mentioned", question)
        if m:
            return " ".join(re.search(rf"magic numbers for {re.escape(k.strip())} is: (\d+)", body).group(1) for k in m.group(1).split(","))  # type: ignore[union-attr]
        m = re.match(r"What is the special magic number for (\S+) mentioned", question)
        if m:
            r = re.search(rf"magic numbers? for {re.escape(m.group(1))} is: (\d+)", body)
            return r.group(1) if r else ""
        m = re.match(r"Find all variables that are assigned the value (\d+)", question)
        if m:
            holders = {v for v in re.findall(rf"VAR (\w+) = {m.group(1)}", body)}
            changed = True
            while changed:
                changed = False
                for v, src in re.findall(r"VAR (\w+) = VAR (\w+)", body):
                    if src in holders and v not in holders:
                        holders.add(v)
                        changed = True
            return " ".join(sorted(holders))
        m = re.match(r"What are the (\d+) most frequently", question)
        if m:
            words = body.split("Count how often each appears.\n")[-1].split()
            cnt: dict[str, int] = {}
            for w in words:
                cnt[w] = cnt.get(w, 0) + 1
            return " ".join(sorted(cnt, key=lambda w: -cnt[w])[: int(m.group(1))])
        m = re.match(r"In which city does ((?:the mentor of )+)(\w+) live", question)
        if m:
            who = m.group(2)
            for _ in range(m.group(1).count("the mentor of")):
                who = re.search(rf"{who}'s mentor is (\w+)\.", body).group(1)  # type: ignore[union-attr]
            r = re.search(rf"{who} lives in (\w+)\.", body)
            return r.group(1) if r else ""
        return ""
