"""forge.planner - cost model, rulebase, search, claims ledger (Plan chapter 5).

* ``costmodel``  bytes-first formulas of Plan 5.1/5.4 (weight/KV bytes, decode time, B*, quantiser overhead, speculation, Amdahl, t_pred, eta fitting)
* ``spec``       ModelSpec / ChipSpec / Scheme + preset tables
* ``rulebase``   Fig 5.3 Q1-Q6 and Table 5.1 as data + ``propose`` (returns ``forge.core.schemas.Recipe`` candidates)
* ``search``     ``plan`` (successive halving, pluggable runner, dry mode by default), ``fits``, ``binding_resource``, ``baseline_arms``
* ``quick``      ``estimate`` for the UI
* ``claims``     claims ledger (db table ``claims``), ``seed_claims``, ``claims_needing_attention``

Public names are resolved lazily so ``import forge.planner`` is cheap.
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    "estimate": ("forge.planner.quick", "estimate"),
    "plan": ("forge.planner.search", "plan"),
    "fits": ("forge.planner.search", "fits"),
    "binding_resource": ("forge.planner.search", "binding_resource"),
    "baseline_arms": ("forge.planner.search", "baseline_arms"),
    "PlanResult": ("forge.planner.search", "PlanResult"),
    "Budget": ("forge.planner.rulebase", "Budget"),
    "propose": ("forge.planner.rulebase", "propose"),
    "ModelSpec": ("forge.planner.spec", "ModelSpec"),
    "ChipSpec": ("forge.planner.spec", "ChipSpec"),
    "Scheme": ("forge.planner.spec", "Scheme"),
    "parse_scheme": ("forge.planner.spec", "parse_scheme"),
    "resolve_chip": ("forge.planner.spec", "resolve_chip"),
    "seed_claims": ("forge.planner.claims", "seed_claims"),
    "claims_needing_attention": ("forge.planner.claims", "claims_needing_attention"),
    "list_claims": ("forge.planner.claims", "list_claims"),
    "get_claim": ("forge.planner.claims", "get_claim"),
    "update_claim": ("forge.planner.claims", "update_claim"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'forge.planner' has no attribute {name!r}")
    return getattr(importlib.import_module(target[0]), target[1])
