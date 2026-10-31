"""Clients used by the ``throughput`` suite to talk to a serving engine.

A client implements ``stream(prompt_ids, max_tokens, temperature=0.0, seed=None) -> Iterator`` yielding one item per
generated token (or per streamed chunk).  Timing is taken by the *caller* when each item arrives, so any transport works.

* :class:`HTTPServeClient`    OpenAI-compatible ``/v1/completions`` with SSE streaming (stdlib ``urllib`` only).
* :class:`EngineServeClient`  drives an in-process ``forge.serve.engine.LLMEngine`` (step loop).  **Not validated against
  the real engine** (it was written against the interface in docs/ARCHITECTURE.md 5.8) - it reads result fields defensively.
* :class:`ProviderServeClient` greedy generation from any ``LogitsProvider`` (CPU tests, parity checks).
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

from forge.core.errors import ForgeError


@runtime_checkable
class ServeClient(Protocol):
    name: str

    def stream(self, prompt_ids: Sequence[int], max_tokens: int, *, temperature: float = 0.0, seed: int | None = None) -> Iterator[Any]: ...


class HTTPServeClient:
    """OpenAI-compatible streaming completions client (works with ``forge.serve.server`` and other compatible servers)."""

    def __init__(self, base_url: str, *, model: str = "default", api_key: str | None = None, timeout: float = 600.0, name: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.name = name or self.base_url

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def stream(self, prompt_ids: Sequence[int], max_tokens: int, *, temperature: float = 0.0, seed: int | None = None) -> Iterator[Any]:
        body: dict[str, Any] = {"model": self.model, "prompt": [int(t) for t in prompt_ids], "max_tokens": int(max_tokens),
                                "temperature": float(temperature), "stream": True}
        if seed is not None:
            body["seed"] = int(seed)
        req = urllib.request.Request(f"{self.base_url}/v1/completions", data=json.dumps(body).encode(), headers=self._headers(), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or []
                    if choices and (choices[0].get("text") or choices[0].get("finish_reason") is None):
                        yield choices[0].get("text", "")
        except urllib.error.URLError as e:
            raise ForgeError(f"serve request to {self.base_url} failed: {e}") from e

    def stats(self) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}/stats", headers=self._headers())
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())


class ProviderServeClient:
    """Serve client over a LogitsProvider (greedy decode); useful for tests and for CPU parity checks."""

    def __init__(self, provider: Any, *, name: str = "provider") -> None:
        self.provider = provider
        self.name = name

    def stream(self, prompt_ids: Sequence[int], max_tokens: int, *, temperature: float = 0.0, seed: int | None = None) -> Iterator[Any]:
        import torch

        ids = list(int(t) for t in prompt_ids)
        for _ in range(max_tokens):
            lg = self.provider.logits(torch.tensor([ids], dtype=torch.long))[0, -1]
            nxt = int(torch.argmax(lg).item())
            ids.append(nxt)
            yield nxt


class EngineServeClient:
    """Drive a ``forge.serve.engine.LLMEngine`` in-process.  The step loop is serialised by a lock; for real concurrency
    use :meth:`run_concurrent`, which submits all prompts then steps until every request finishes, recording per-token
    arrival times.  UNVALIDATED against the real engine."""

    supports_concurrency = False

    def __init__(self, engine: Any, *, name: str = "engine") -> None:
        self.engine = engine
        self.name = name
        self._lock = threading.Lock()
        self._counter = 0

    def _params(self, max_tokens: int, temperature: float, seed: int | None) -> Any:
        from forge.serve.engine import SamplingParams  # lazy; module may expose it elsewhere

        try:
            return SamplingParams(max_tokens=max_tokens, temperature=temperature, seed=seed)
        except TypeError:
            return SamplingParams(max_tokens=max_tokens, temperature=temperature)

    @staticmethod
    def _tokens(out: Any) -> list[int]:
        for attr in ("output_token_ids", "token_ids", "new_token_ids", "tokens"):
            v = getattr(out, attr, None)
            if v is not None:
                return list(v)
        return []

    def stream(self, prompt_ids: Sequence[int], max_tokens: int, *, temperature: float = 0.0, seed: int | None = None) -> Iterator[Any]:
        with self._lock:
            self._counter += 1
            rid = f"tp-{self._counter}"
            self.engine.add_request(list(prompt_ids), self._params(max_tokens, temperature, seed), rid)
            seen = 0
            while True:
                outs = self.engine.step()
                mine = [o for o in outs if getattr(o, "request_id", None) == rid]
                for o in mine:
                    toks = self._tokens(o)
                    for t in toks[seen:]:
                        yield t
                    seen = max(seen, len(toks))
                    if getattr(o, "finished", False):
                        return
                if not outs and not mine:
                    return

    def run_concurrent(self, prompts: Sequence[Sequence[int]], max_tokens: int) -> list[list[float]]:
        """Submit all prompts at once; returns per-request lists of token arrival times (perf_counter seconds)."""
        t0 = time.perf_counter()
        rids = []
        with self._lock:
            for p in prompts:
                self._counter += 1
                rid = f"tp-{self._counter}"
                rids.append(rid)
                self.engine.add_request(list(p), self._params(max_tokens, 0.0, None), rid)
            times: dict[str, list[float]] = {r: [] for r in rids}
            done: set[str] = set()
            seen: dict[str, int] = {r: 0 for r in rids}
            while len(done) < len(rids):
                outs = self.engine.step()
                now = time.perf_counter()
                if not outs:
                    break
                for o in outs:
                    rid = getattr(o, "request_id", None)
                    if rid not in times:
                        continue
                    toks = self._tokens(o)
                    new = max(0, len(toks) - seen[rid])
                    times[rid] += [now] * new
                    seen[rid] = max(seen[rid], len(toks))
                    if getattr(o, "finished", False):
                        done.add(rid)
        return [[t - t0 for t in times[r]] for r in rids]
