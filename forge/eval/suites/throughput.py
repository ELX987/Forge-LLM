"""``throughput`` suite (EV-1 system metrics): TTFT, inter-token latency and tokens/s through a serve client.

Measures what the plan insists on (Plan 10.1 "Throughput & latency"): the stack's *real* numbers at stated
concurrency and context, not the bandwidth ceiling.  Needs a serve client for the candidate (and optionally the parent),
supplied via ``SuiteEnv.candidate_serve`` / ``parent_serve``.  If no client is given but a model provider exists, a
(slow, unbatched, CPU-friendly) :class:`ProviderServeClient` is used and the result is labelled accordingly.

Prompts are synthetic token-id sequences (seeded), so the measurement is independent of any dataset.
"""
from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from pydantic import Field

from forge.core.registry import register
from forge.eval.serve_client import ProviderServeClient
from forge.eval.suites.base import Suite, SuiteConfig, SuiteEnv, SuiteResult


class ThroughputConfig(SuiteConfig):
    """Serving latency / throughput measurement."""

    concurrency: list[int] = Field(default_factory=lambda: [1, 4], description="Numbers of simultaneous requests to test.")
    prompt_len: int = Field(default=128, ge=1, le=1_000_000, description="Prompt length in tokens (synthetic ids).")
    gen_len: int = Field(default=32, ge=1, le=65_536, description="Tokens to generate per request.")
    requests_per_level: int = Field(default=8, ge=1, le=100_000, description="Requests issued per concurrency level.")
    warmup_requests: int = Field(default=1, ge=0, le=1000, description="Discarded warm-up requests per client.")
    prompt_vocab: int = Field(default=1000, ge=2, description="Synthetic prompt ids are drawn from [1, prompt_vocab).")
    seed: int = Field(default=0, ge=0, description="Seed for prompt generation.")


def _pct(x: list[float] | np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(x, dtype=np.float64), q)) if len(x) else float("nan")


def measure_client(client: Any, cfg: ThroughputConfig, concurrency: int, seed: int) -> dict[str, Any]:
    """Run ``requests_per_level`` requests at ``concurrency``; returns raw per-request timings."""
    rng = random.Random(seed)
    prompts = [[rng.randrange(1, cfg.prompt_vocab) for _ in range(cfg.prompt_len)] for _ in range(cfg.requests_per_level)]
    if hasattr(client, "run_concurrent") and concurrency > 1:
        t0 = time.perf_counter()
        per_req: list[list[float]] = []
        for i in range(0, len(prompts), concurrency):
            per_req += client.run_concurrent(prompts[i : i + concurrency], cfg.gen_len)
        wall = time.perf_counter() - t0
        ttft = [t[0] for t in per_req if t]
        itl = [b - a for t in per_req for a, b in zip(t, t[1:])]
        toks = [len(t) for t in per_req]
        rates = [len(t) / t[-1] for t in per_req if t and t[-1] > 0]
        return {"ttft": ttft, "itl": itl, "tokens": sum(toks), "wall": wall, "req_rates": rates}
    ttfts: list[float] = []
    itls: list[float] = []
    toks: list[int] = []
    rates: list[float] = []
    lock = threading.Lock()

    def one(p: list[int]) -> None:
        t_start = time.perf_counter()
        stamps: list[float] = []
        for _ in client.stream(p, cfg.gen_len):
            stamps.append(time.perf_counter())
        if not stamps:
            return
        with lock:
            ttfts.append(stamps[0] - t_start)
            itls.extend(b - a for a, b in zip(stamps, stamps[1:]))
            toks.append(len(stamps))
            if stamps[-1] > t_start:
                rates.append(len(stamps) / (stamps[-1] - t_start))

    t0 = time.perf_counter()
    if concurrency <= 1:
        for p in prompts:
            one(p)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            list(ex.map(one, prompts))
    wall = time.perf_counter() - t0
    return {"ttft": ttfts, "itl": itls, "tokens": sum(toks), "wall": wall, "req_rates": rates}


def summarize(raw: dict[str, Any]) -> dict[str, float]:
    return {
        "ttft_p50_ms": _pct(raw["ttft"], 0.5) * 1e3, "ttft_p99_ms": _pct(raw["ttft"], 0.99) * 1e3,
        "itl_p50_ms": _pct(raw["itl"], 0.5) * 1e3, "itl_p99_ms": _pct(raw["itl"], 0.99) * 1e3,
        "tokens_per_s": raw["tokens"] / raw["wall"] if raw["wall"] > 0 else float("nan"),
    }


@register("suite", "throughput", meta={"requirement": "EV-1"})
class ThroughputSuite(Suite):
    """TTFT / ITL / tokens-per-second at several concurrency levels, and the candidate/parent speed-up."""

    name = "throughput"
    version = "1"
    title = "Serving throughput and latency"
    Config = ThroughputConfig
    describes = "Measured on the real stack (never the roofline ceiling)."

    def _client(self, serve: Any, provider: Any, label: str, res: SuiteResult) -> Any:
        if serve is not None:
            return serve
        if provider is not None:
            res.notes.append(f"{label}: no serve client supplied, using an unbatched LogitsProvider client - numbers are NOT serving-engine numbers")
            return ProviderServeClient(provider, name=f"{label}-provider")
        return None

    def run(self, cfg: ThroughputConfig, env: SuiteEnv) -> SuiteResult:  # type: ignore[override]
        res = SuiteResult(self.name, self.version)
        cand = self._client(env.candidate_serve, env.candidate, "candidate", res)
        if cand is None:
            res.notes.append("no candidate client or model: throughput not measured")
            return res
        par = self._client(env.parent_serve, None, "parent", res) if env.parent_serve is not None else None
        levels = sorted(set(int(c) for c in cfg.concurrency))
        res.details["client"] = getattr(cand, "name", type(cand).__name__)
        for ci, c in enumerate(levels):
            for who, client in (("", cand), ("_parent", par)):
                if client is None:
                    continue
                for _ in range(cfg.warmup_requests):
                    try:
                        for _t in client.stream([1] * min(cfg.prompt_len, 8), 2):
                            pass
                    except Exception as e:  # warmup failures should not kill the suite
                        res.notes.append(f"warm-up failed: {e}")
                        break
                raw = measure_client(client, cfg, c, cfg.seed + ci)
                s = summarize(raw)
                for k, v in s.items():
                    res.add(f"{k}{who}_c{c}", v, n=len(raw["ttft"]))
                if who == "":
                    res.details.setdefault("raw_req_rates", {})[c] = raw["req_rates"]
                else:
                    res.details.setdefault("raw_req_rates_parent", {})[c] = raw["req_rates"]
                env.progress((ci + 1) / len(levels), f"throughput c={c}")
        top = levels[-1]
        res.add("tokens_per_s", res.scalars[f"tokens_per_s_c{top}"], n=cfg.requests_per_level)
        res.add("ttft_p50_ms", res.scalars[f"ttft_p50_ms_c{top}"], n=cfg.requests_per_level)
        res.add("itl_p50_ms", res.scalars[f"itl_p50_ms_c{top}"], n=cfg.requests_per_level)
        res.add("itl_p99_ms", res.scalars[f"itl_p99_ms_c{top}"], n=cfg.requests_per_level)
        if par is not None:
            for c in levels:
                a, b = res.scalars.get(f"tokens_per_s_parent_c{c}"), res.scalars.get(f"tokens_per_s_c{c}")
                if a and b:
                    res.add(f"throughput_speedup_c{c}", b / a, n=cfg.requests_per_level)
            ca = np.array(res.details["raw_req_rates_parent"].get(top, []))
            cb = np.array(res.details["raw_req_rates"].get(top, []))
            if ca.size and cb.size:
                rng = np.random.default_rng(env.seed)
                vals = [rng.choice(cb, cb.size).mean() / max(rng.choice(ca, ca.size).mean(), 1e-300) for _ in range(env.n_boot // 2 or 1)]
                lo, hi = np.quantile(vals, [env.alpha / 2, 1 - env.alpha / 2])
                res.add("throughput_speedup", float(cb.mean() / ca.mean()), ci=(float(lo), float(hi)), n=int(min(ca.size, cb.size)))
        return res
