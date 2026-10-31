"""forge.eval - evaluation harness (Plan 3.6 EV-1..EV-7, 10.1).

Modules: ``metrics`` (KL / flip / ECE / entropy / perplexity + LogitsProvider), ``stats`` (bootstrap, Wilson, BH-FDR, noise floor,
non-inferiority), ``suites`` (registry + distribution, mcq, longgen, longctx, safety, contamination, throughput, agentic),
``gates`` (YAML gates -> GateVerdict), ``run`` (``run_eval``), ``db`` (``evals`` table, list / get / compare / Pareto).

Names below are resolved lazily so ``import forge.eval`` stays cheap (no torch import).
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "run_eval": "forge.eval.run",
    "EvalConfig": "forge.eval.config",
    "ModelSource": "forge.eval.config",
    "SuiteSelection": "forge.eval.config",
    "kl_divergence": "forge.eval.metrics",
    "top1_flip_rate": "forge.eval.metrics",
    "ece": "forge.eval.metrics",
    "entropy_shift": "forge.eval.metrics",
    "perplexity": "forge.eval.metrics",
    "LogitsProvider": "forge.eval.metrics",
    "as_provider": "forge.eval.metrics",
    "paired_bootstrap": "forge.eval.stats",
    "binomial_ci": "forge.eval.stats",
    "benjamini_hochberg": "forge.eval.stats",
    "noise_floor_from_repeats": "forge.eval.stats",
    "distinguishable": "forge.eval.stats",
    "load_gate": "forge.eval.gates",
    "list_gates": "forge.eval.gates",
    "Gate": "forge.eval.gates",
    "Measurements": "forge.eval.gates",
    "list_suites": "forge.eval.suites",
    "get_suite": "forge.eval.suites",
    "compare": "forge.eval.db",
    "list_reports": "forge.eval.db",
    "get_report": "forge.eval.db",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'forge.eval' has no attribute {name!r}")
    return getattr(importlib.import_module(mod), name)
