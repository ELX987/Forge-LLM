"""forge.serve - inference engine, paged KV / prefix cache, speculative decoding, capacity planner, OpenAI-compatible server.

Public API (lazily imported so ``import forge.serve`` stays cheap and never needs a GPU):

* ``ServeConfig`` (``serve.config``), ``SamplingParams`` (``serve.sampling``)
* ``LLMEngine``, ``RequestOutput`` (``serve.engine``), ``Scheduler`` (``serve.scheduler``)
* ``BlockPool``, ``KVBlockManager`` (``serve.kv``)
* ``NgramDrafter``, ``DraftModelDrafter``, ``EagleDrafter``, ``EagleHead``, ``SpecController``, ``rejection_sample`` (``serve.spec``)
* ``capacity_plan``, ``CapacityReport``, ``fleet_capacity``, ``speculation_gain_vs_batch``, ``capacity_curve`` (``serve.capacity``)
* ``create_app`` (``serve.server``), ``run_server`` (``serve.run``), ``ForgeServeClient`` (``serve.client``)
"""
from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "ServeConfig": "forge.serve.config",
    "SamplingDefaults": "forge.serve.config",
    "SpeculationConfig": "forge.serve.config",
    "SamplingParams": "forge.serve.sampling",
    "LLMEngine": "forge.serve.engine",
    "RequestOutput": "forge.serve.engine",
    "Scheduler": "forge.serve.scheduler",
    "Request": "forge.serve.scheduler",
    "BlockPool": "forge.serve.kv",
    "KVBlockManager": "forge.serve.kv",
    "NgramDrafter": "forge.serve.spec",
    "DraftModelDrafter": "forge.serve.spec",
    "EagleDrafter": "forge.serve.spec",
    "EagleHead": "forge.serve.spec",
    "SpecController": "forge.serve.spec",
    "rejection_sample": "forge.serve.spec",
    "capacity_plan": "forge.serve.capacity",
    "CapacityReport": "forge.serve.capacity",
    "ContextDist": "forge.serve.capacity",
    "fleet_capacity": "forge.serve.capacity",
    "speculation_gain_vs_batch": "forge.serve.capacity",
    "capacity_curve": "forge.serve.capacity",
    "create_app": "forge.serve.server",
    "run_server": "forge.serve.run",
    "ForgeServeClient": "forge.serve.client",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    mod = _EXPORTS.get(name)
    if mod is None:
        raise AttributeError(f"module 'forge.serve' has no attribute {name!r}")
    return getattr(importlib.import_module(mod), name)


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
