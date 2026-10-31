"""forge.hw - NVIDIA hardware knowledge: chip database, capabilities, probe + micro-benchmarks, descriptor store,
GPU telemetry (see docs/ARCHITECTURE.md sections 3 and 5.2).

Importing ``forge.hw`` is cheap: submodules and the convenience names below are resolved lazily on first access.

    from forge.hw import current_capabilities, KNOWN_CHIPS, run_probe, GpuSampler
    from forge.hw.capabilities import Capabilities
"""
from __future__ import annotations

import importlib
from typing import Any

_LAZY = {
    # chips
    "KNOWN_CHIPS": ("forge.hw.chips", "KNOWN_CHIPS"),
    "find_chip": ("forge.hw.chips", "find_chip"),
    "get_chip": ("forge.hw.chips", "get_chip"),
    "list_chips": ("forge.hw.chips", "list_chips"),
    "chip_summary": ("forge.hw.chips", "chip_summary"),
    # capabilities
    "Capabilities": ("forge.hw.capabilities", "Capabilities"),
    "current_capabilities": ("forge.hw.capabilities", "current_capabilities"),
    "capabilities_for_chip": ("forge.hw.capabilities", "capabilities_for_chip"),
    "list_gpus": ("forge.hw.capabilities", "list_gpus"),
    "support_matrix": ("forge.hw.capabilities", "support_matrix"),
    "tier_for_sm": ("forge.hw.capabilities", "tier_for_sm"),
    # descriptors
    "descriptor_from_chip": ("forge.hw.descriptor", "descriptor_from_chip"),
    "descriptor_from_spec": ("forge.hw.descriptor", "descriptor_from_spec"),
    "save_descriptor": ("forge.hw.descriptor_store", "save_descriptor"),
    "load_descriptor": ("forge.hw.descriptor_store", "load_descriptor"),
    "list_descriptors": ("forge.hw.descriptor_store", "list_descriptors"),
    # probe / telemetry
    "run_probe": ("forge.hw.probe", "run_probe"),
    "run_probe_full": ("forge.hw.probe", "run_probe_full"),
    "ProbeConfig": ("forge.hw.config", "ProbeConfig"),
    "GpuSampler": ("forge.hw.telemetry", "GpuSampler"),
}
_SUBMODULES = ("chips", "capabilities", "descriptor", "descriptor_store", "microbench", "probe", "telemetry", "config")

__all__ = sorted(_LAZY) + list(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        value = getattr(importlib.import_module(mod), attr)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return importlib.import_module(f"forge.hw.{name}")
    raise AttributeError(f"module 'forge.hw' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
