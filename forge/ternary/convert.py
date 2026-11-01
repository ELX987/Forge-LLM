"""Post-training ternarisation of a Forge model (data-free): dense linears -> packed ``t158`` layers, with a reconstruction report.

This is the engine behind the ``ternarize`` recipe stage and ``ptq`` with ``format: ternary``.  Per layer it reports the reconstruction
SQNR (``10 log10(|W|^2 / |W - Q(W)|^2)``; tern158 expects roughly 5.5-7.5 dB on real Llama weights with Lloyd/G=128 -- a TARGET, not
measured here) and the zero fraction.  Quality recovery after ternarising a pretrained model needs QAT / distillation (see the
``ternary-convert`` recipe): one-shot ternarisation alone is lossy.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import torch
import torch.nn as nn

from forge.formats import ternary as tn
from forge.models.linear import replace_linears

from .linear import T158Linear

DEFAULT_EXCLUDE = (r"lm_head", r"(^|\.)gate$", r"router", r"shared_expert_gate", r"embed")


@dataclass
class TernarizeLayer:
    name: str
    shape: tuple[int, int]
    sqnr_db: float
    zero_frac: float
    bytes_before: int
    bytes_after: int


@dataclass
class TernarizeReport:
    config: dict[str, Any]
    layers: list[TernarizeLayer] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def params(self) -> int:
        return sum(r.shape[0] * r.shape[1] for r in self.layers)

    @property
    def bytes_before(self) -> int:
        return sum(r.bytes_before for r in self.layers)

    @property
    def bytes_after(self) -> int:
        return sum(r.bytes_after for r in self.layers)

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.bytes_after / max(1, self.params)

    def sqnr_by_type(self) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for r in self.layers:
            acc.setdefault(r.name.rsplit(".", 1)[-1], []).append(r.sqnr_db)
        return {k: sum(v) / len(v) for k, v in sorted(acc.items())}

    def summary(self) -> dict[str, Any]:
        n = max(1, len(self.layers))
        return {"n_layers": len(self.layers), "n_skipped": len(self.skipped), "params": self.params, "bits_per_weight": self.bits_per_weight,
                "compression_vs_source": self.bytes_before / max(1, self.bytes_after), "mean_sqnr_db": sum(r.sqnr_db for r in self.layers) / n,
                "mean_zero_frac": sum(r.zero_frac for r in self.layers) / n, "sqnr_db_by_type": self.sqnr_by_type(),
                "bytes_before": self.bytes_before, "bytes_after": self.bytes_after}

    def to_dict(self) -> dict[str, Any]:
        return {**self.summary(), "config": self.config, "skipped": dict(self.skipped), "layers": [asdict(r) for r in self.layers]}


def sqnr_db(w: torch.Tensor, wq: torch.Tensor) -> float:
    num = float(w.float().pow(2).sum())
    den = float((w.float() - wq.float()).pow(2).sum())
    if den <= 0.0:
        return float("inf")
    return 10.0 * math.log10(max(num, 1e-30) / den)


def _matches(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) for p in patterns)


@torch.no_grad()
def ternarize_model(model: nn.Module, group_size: int = 128, rule: str = "lloyd", iters: int = 1, exclude: Iterable[str] = DEFAULT_EXCLUDE,
                    include: Iterable[str] | None = None) -> TernarizeReport:
    """Replace eligible ``nn.Linear`` layers of ``model`` in place by :class:`T158Linear`; returns the report."""
    exclude, include = tuple(exclude), (tuple(include) if include is not None else None)
    rep = TernarizeReport({"group_size": group_size, "rule": rule, "iters": iters, "exclude": list(exclude), "include": list(include) if include else None})

    def factory(name: str, lin: nn.Linear) -> nn.Module | None:
        if lin.weight.is_meta:
            return None
        if _matches(name, exclude) or (include is not None and not _matches(name, include)):
            rep.skipped[name] = "excluded"
            return None
        k = lin.in_features
        if k % tn.CODES_PER_WORD:
            rep.skipped[name] = f"in_features {k} is not a multiple of {tn.CODES_PER_WORD}"
            return None
        if group_size > 0 and k % group_size:
            rep.skipped[name] = f"group_size {group_size} does not divide in_features {k}"
            return None
        w = lin.weight.detach()
        t, s = tn.quantize_ternary(w, group_size, rule, iters)
        new = T158Linear.from_dense(lin, group_size=group_size, rule=rule, iters=iters)
        wq = tn.dequantize_ternary(t, s, group_size)
        rep.layers.append(TernarizeLayer(name, (lin.out_features, k), sqnr_db(w, wq), float((t == 0).float().mean()),
                                         w.numel() * w.element_size(), new.qweight.numel() + new.scales.numel() * 4))
        return new

    replace_linears(model, factory)
    return rep
