"""CUDA pre-flight for job workers: fail a GPU job in seconds with a diagnosis instead of an opaque ``AcceleratorError`` mid-run.

Called by :mod:`forge.jobs.worker` before the job's entrypoint (and before anything in the worker initialises CUDA - the device probe
itself runs in a subprocess).  Jobs that do not use a GPU (``needs_gpu`` false, or the scheduler forced CPU with
``CUDA_VISIBLE_DEVICES`` empty / ``-1``) are never gated; hosts without any NVIDIA GPU are not blocked either (the stack check only
warns there).  ``FORGE_SKIP_CUDA_CHECK=1`` disables the gate.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

from forge.core.errors import HardwareUnsupported

SKIP_ENV = "FORGE_SKIP_CUDA_CHECK"


def uses_gpu(needs_gpu: bool, env: Mapping[str, str] | None = None) -> bool:
    """True when the job will touch CUDA: its type needs a GPU and the scheduler did not hide them all."""
    if not needs_gpu:
        return False
    cvd = (os.environ if env is None else env).get("CUDA_VISIBLE_DEVICES")
    return cvd is None or cvd.strip() not in ("", "-1")


def gate_cuda(needs_gpu: bool, env: Mapping[str, str] | None = None) -> None:
    """Raise :class:`HardwareUnsupported` (message = the full diagnosis, ``details['findings']`` = the failing rules) when this job
    would use a GPU on a host whose CUDA stack is broken.  Success is cached per process by ``assert_cuda_usable``."""
    e = os.environ if env is None else env
    if (e.get(SKIP_ENV) or "").strip().lower() in ("1", "true", "yes", "on") or not uses_gpu(needs_gpu, e):
        return
    from forge.hw import cuda_compat

    try:
        cuda_compat.assert_cuda_usable()
    except cuda_compat.CudaUnusable as exc:
        raise HardwareUnsupported(str(exc), details={"findings": exc.findings}) from exc
