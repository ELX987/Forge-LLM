"""forge.telemetry: metrics store (sqlite + optional parquet), log capture, system/GPU sampling.

Public names (lazy re-exports keep ``import forge.telemetry`` cheap):

* :class:`forge.telemetry.metrics.MetricsStore`, :func:`get_metrics_store`
* :mod:`forge.telemetry.logs` (``read_log``, ``tail_lines``, ``LogTailer``, ``setup_worker_logging``)
* :class:`forge.telemetry.system.SystemMonitor`
"""
from __future__ import annotations

from typing import Any

__all__ = ["MetricsStore", "get_metrics_store", "SystemMonitor"]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial
    if name in ("MetricsStore", "get_metrics_store"):
        from forge.telemetry import metrics

        return getattr(metrics, name)
    if name == "SystemMonitor":
        from forge.telemetry.system import SystemMonitor

        return SystemMonitor
    raise AttributeError(name)
