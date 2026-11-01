"""CUDA pre-flight for job workers: fail a GPU job in seconds with a diagnosis instead of an opaque ``AcceleratorError`` mid-run.

Called by :mod:`forge.jobs.worker` before the job's entrypoint (and before anything in the worker initialises CUDA - the device probe
itself runs in a subprocess).  Jobs that do not use a GPU (``needs_gpu`` false, the scheduler forced CPU with ``CUDA_VISIBLE_DEVICES``
empty / ``-1``, or the job's own ``params['device']`` is ``cpu``, as ``eval`` and ``serve`` allow) are never gated; hosts without any
NVIDIA GPU are not blocked either (the stack check only warns there).  Job types whose CPU mode is not selected through a top-level
``device`` param (e.g. ``bench``, ``probe``) are still gated: set ``FORGE_SKIP_CUDA_CHECK=1`` to disable the gate for them.

Workers are a fresh process per job, so the in-process success cache of ``assert_cuda_usable`` never helps across jobs.  A passing check
is therefore also remembered on disk (``<FORGE_HOME>/cache/cuda_gate.json``, ``CACHE_TTL_S`` seconds) under a cheap fingerprint - driver
version + driver CUDA (NVML / nvidia-smi), torch version (package metadata, no import) and ``CUDA_VISIBLE_DEVICES``.  Only successes are
cached, a failure is always re-checked, a changed fingerprint or an unreadable / corrupt file means a full check.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.errors import HardwareUnsupported

SKIP_ENV = "FORGE_SKIP_CUDA_CHECK"
CACHE_TTL_S = 600.0
CACHE_FILE = "cuda_gate.json"


def cpu_requested(params: Mapping[str, Any] | None) -> bool:
    """True when the job's own params select the CPU (``device: cpu`` / ``cpu:0``)."""
    dev = params.get("device") if isinstance(params, Mapping) else None
    return isinstance(dev, str) and dev.strip().lower().startswith("cpu")


def uses_gpu(needs_gpu: bool, env: Mapping[str, str] | None = None, params: Mapping[str, Any] | None = None) -> bool:
    """True when the job will touch CUDA: its type needs a GPU, it did not ask for the CPU and the scheduler did not hide them all."""
    if not needs_gpu or cpu_requested(params):
        return False
    cvd = (os.environ if env is None else env).get("CUDA_VISIBLE_DEVICES")
    return cvd is None or cvd.strip() not in ("", "-1")


# -- on-disk success cache ----------------------------------------------------------------------------------------------------
def cache_path() -> Path:
    from forge.core.paths import cache_dir

    return cache_dir() / CACHE_FILE


def fingerprint(env: Mapping[str, str]) -> dict[str, str | None] | None:
    """Cheap identity of the CUDA stack (no torch import, no probe); None when the driver cannot be read (then nothing is cached)."""
    try:
        from importlib import metadata

        from forge.hw import cuda_compat

        facts = dict(cuda_compat._nvml_facts())
        if not (facts.get("driver_version") and facts.get("driver_cuda")):
            smi = cuda_compat._smi_facts()
            devs = smi.get("devices") or []
            facts["driver_version"] = facts.get("driver_version") or (devs[0]["driver"] if devs else None)
            facts["driver_cuda"] = facts.get("driver_cuda") or smi.get("driver_cuda")
        if not facts.get("driver_version"):
            return None
        try:
            torch_v: str | None = metadata.version("torch")
        except Exception:  # noqa: BLE001 - not installed
            torch_v = None
        return {"driver_version": facts.get("driver_version"), "driver_cuda": facts.get("driver_cuda"), "torch": torch_v,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES")}
    except Exception:  # noqa: BLE001 - never let the cache break the gate
        return None


def _cache_hit(path: Path, fp: dict[str, Any], now: float) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        age = now - float(data["ts"])
        return data["fingerprint"] == fp and 0 <= age < CACHE_TTL_S
    except Exception:  # noqa: BLE001 - missing / corrupt / unreadable -> full check
        return False


def _cache_store(path: Path, fp: dict[str, Any], now: float) -> None:
    try:
        from forge.core.util import atomic_write_text

        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps({"ts": now, "fingerprint": fp}))
    except Exception:  # noqa: BLE001 - read-only home etc.: just no caching
        pass


def gate_cuda(needs_gpu: bool, env: Mapping[str, str] | None = None, params: Mapping[str, Any] | None = None, *,
              now: float | None = None, path: Path | None = None) -> None:
    """Raise :class:`HardwareUnsupported` (message = the full diagnosis, ``details['findings']`` = the failing rules) when this job
    would use a GPU on a host whose CUDA stack is broken.  A success is remembered on disk (see module doc); ``now`` / ``path``
    exist for tests."""
    e = os.environ if env is None else env
    if (e.get(SKIP_ENV) or "").strip().lower() in ("1", "true", "yes", "on") or not uses_gpu(needs_gpu, e, params):
        return
    t = time.time() if now is None else now
    fp = fingerprint(e)
    cp = None
    if fp is not None:
        try:
            cp = path or cache_path()
        except Exception:  # noqa: BLE001
            cp = None
        if cp is not None and _cache_hit(cp, fp, t):
            return
    from forge.hw import cuda_compat

    try:
        cuda_compat.assert_cuda_usable()
    except cuda_compat.CudaUnusable as exc:
        raise HardwareUnsupported(str(exc), details={"findings": exc.findings}) from exc
    if fp is not None and cp is not None:
        _cache_store(cp, fp, t)
