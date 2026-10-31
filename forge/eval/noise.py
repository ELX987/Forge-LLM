"""Noise-floor machinery (EV-5): evaluate the *parent* under perturbed conditions and measure how much metrics move.

Perturbations (Plan 10.1: "different seeds, batch compositions and kernels"):

* ``seed``  - generator seeds of synthetic suites / choice order (suites expose ``vary_seed_with_run`` / ``shuffle_choices``)
* ``batch`` - :class:`BatchPerturbedProvider` evaluates the same inputs in different batch splits and with extra right padding,
  which changes tensor shapes (and so GPU kernel selection / reduction order) while leaving the maths unchanged.

On a deterministic CPU the batch perturbation will usually produce exactly zero noise - that is reported as such (a floor of 0
means "this stack is deterministic under these perturbations", not "the metric is exact": the seed perturbation still moves
sampled-instance scores).
"""
from __future__ import annotations

import random
from typing import Any


class BatchPerturbedProvider:
    """Wrap a LogitsProvider so each call is evaluated with a different batch composition."""

    def __init__(self, inner: Any, seed: int = 0, *, extra_pad: int = 3) -> None:
        self.inner = inner
        self.rng = random.Random(seed)
        self.extra_pad = extra_pad
        self.name = f"{getattr(inner, 'name', 'model')}+batchperturb{seed}"
        self.eos_token_id = getattr(inner, "eos_token_id", None)

    def nbytes(self) -> int | None:
        fn = getattr(self.inner, "nbytes", None)
        return fn() if callable(fn) else None

    def logits(self, input_ids: Any, attention_mask: Any = None) -> Any:
        import torch

        B, T = input_ids.shape
        pad_cols = self.rng.randint(0, self.extra_pad)
        outs = []
        i = 0
        while i < B:
            size = self.rng.randint(1, max(1, B))
            chunk = input_ids[i : i + size]
            mask = attention_mask[i : i + size] if attention_mask is not None else None
            if pad_cols:
                chunk = torch.cat([chunk, torch.zeros(chunk.shape[0], pad_cols, dtype=chunk.dtype)], dim=1)
                if mask is not None:
                    mask = torch.cat([mask, torch.zeros(mask.shape[0], pad_cols, dtype=mask.dtype)], dim=1)
            lg = self.inner.logits(chunk, mask)
            outs.append(lg[:, :T])
            i += size
        return torch.cat(outs, dim=0)

    def generate(self, prompt_ids: Any, max_new_tokens: int, **kw: Any) -> list[int]:
        from forge.eval.metrics import generate

        inner_gen = getattr(self.inner, "generate", None)
        if callable(inner_gen):
            return list(inner_gen(prompt_ids, max_new_tokens, **kw))
        from forge.eval.metrics import greedy_generate

        return greedy_generate(self, prompt_ids, max_new_tokens, eos_token_id=self.eos_token_id,
                               stop_ids=kw.get("stop_ids", ()), temperature=kw.get("temperature", 0.0), seed=kw.get("seed"))
