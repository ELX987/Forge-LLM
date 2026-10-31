"""System/GPU telemetry sampled in the API process into a ring buffer.

Sources (first that works): ``forge.hw.telemetry.GpuSampler`` -> ``pynvml`` -> ``nvidia-smi`` -> ``[]``.
While jobs run, per-job series ``sys/gpu{i}/util|mem|temp|power`` plus ``sys/cpu_percent`` / ``sys/ram_gb`` are
written to the metrics store (run_id = job id) via the ``running_jobs`` callback.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

log = logging.getLogger("forge.telemetry.system")


def _nvidia_smi_sample() -> list[dict[str, Any]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        return []
    rows = []

    def f(x: str) -> float | None:
        try:
            return float(x)
        except ValueError:
            return None

    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "util": f(parts[2]),
                "mem_used_mb": f(parts[3]),
                "mem_total_mb": f(parts[4]),
                "temp_c": f(parts[5]),
                "power_w": f(parts[6]),
                "clock_mhz": f(parts[7]),
            }
        )
    return rows


def _pynvml_sample() -> list[dict[str, Any]]:
    import pynvml  # type: ignore[import-not-found]

    pynvml.nvmlInit()
    try:
        rows = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            row: dict[str, Any] = {
                "index": i,
                "name": name,
                "util": float(util.gpu),
                "mem_used_mb": mem.used / 2**20,
                "mem_total_mb": mem.total / 2**20,
            }
            for key, fn in (
                ("temp_c", lambda hh=h: float(pynvml.nvmlDeviceGetTemperature(hh, pynvml.NVML_TEMPERATURE_GPU))),
                ("power_w", lambda hh=h: pynvml.nvmlDeviceGetPowerUsage(hh) / 1000.0),
                ("clock_mhz", lambda hh=h: float(pynvml.nvmlDeviceGetClockInfo(hh, pynvml.NVML_CLOCK_SM))),
            ):
                try:
                    row[key] = fn()
                except Exception:
                    row[key] = None
            rows.append(row)
        return rows
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


_SAMPLER: Any = None
_SAMPLER_LOCK = threading.Lock()


def _shared_sampler() -> Any:
    """One long-lived GpuSampler: constructing a new one per sample re-runs nvmlInit() every time (slow on real
    drivers and never balanced by nvmlShutdown)."""
    global _SAMPLER
    with _SAMPLER_LOCK:
        if _SAMPLER is None:
            from forge.hw.telemetry import GpuSampler  # type: ignore[import-not-found]

            _SAMPLER = GpuSampler()
        return _SAMPLER


def sample_gpus() -> list[dict[str, Any]]:
    """One telemetry sample for all GPUs ([] when none / no tooling)."""
    try:
        rows = _shared_sampler().sample()
        if rows:
            return list(rows)
    except Exception:
        pass
    try:
        return _pynvml_sample()
    except Exception:
        pass
    return _nvidia_smi_sample()


def sample_host() -> dict[str, float]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "cpu_percent": float(psutil.cpu_percent(interval=None)),
            "ram_used_gb": (vm.total - vm.available) / 2**30,
            "ram_total_gb": vm.total / 2**30,
        }
    except Exception:
        return {}


class SystemMonitor:
    """Background sampler. ``latest()`` / ``history()`` are cheap and thread-safe."""

    def __init__(
        self,
        interval: float = 1.5,
        history: int = 900,
        *,
        sampler: Callable[[], list[dict[str, Any]]] | None = None,
        running_jobs: Callable[[], list[tuple[str, list[int] | None]]] | None = None,
        metrics_store: Any = None,
    ) -> None:
        self.interval = max(0.2, float(interval))
        self._sampler = sampler or sample_gpus
        self._running_jobs = running_jobs
        self._metrics = metrics_store
        self._ring: deque[dict[str, Any]] = deque(maxlen=history)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._step = 0
        self._job_t0: dict[str, float] = {}
        self._latest: dict[str, Any] = {"ts": time.time(), "gpus": [], "host": {}}

    # -- control --------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.sample_once()  # have data immediately
        self._thread = threading.Thread(target=self._loop, name="forge-sysmon", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.sample_once()
            except Exception as e:  # pragma: no cover
                log.debug("sampling failed: %s", e)

    # -- sampling -------------------------------------------------------------------------
    def sample_once(self) -> dict[str, Any]:
        gpus = self._sampler() or []
        host = sample_host()
        snap = {"ts": time.time(), "gpus": gpus, "host": host}
        with self._lock:
            self._ring.append(snap)
            self._latest = snap
        self._step += 1
        if self._running_jobs and self._metrics is not None:
            try:
                for job_id, idxs in self._running_jobs():
                    vals: dict[str, float] = {}
                    for g in gpus:
                        if idxs is not None and g.get("index") not in idxs:
                            continue
                        i = g.get("index")
                        for src, dst in (("util", "util"), ("mem_used_mb", "mem"), ("temp_c", "temp"), ("power_w", "power")):
                            v = g.get(src)
                            if v is not None:
                                vals[f"sys/gpu{i}/{dst}"] = float(v)
                    if "cpu_percent" in host:
                        vals["sys/cpu_percent"] = host["cpu_percent"]
                        vals["sys/ram_gb"] = host["ram_used_gb"]
                    if vals:
                        t0 = self._job_t0.setdefault(job_id, snap["ts"])
                        # step = seconds since this job's first system sample (keeps chart x-axes small)
                        self._metrics.write(job_id, int(snap["ts"] - t0), vals, ts=snap["ts"])
            except Exception as e:  # pragma: no cover
                log.debug("job telemetry failed: %s", e)
        return snap

    # -- read -----------------------------------------------------------------------------
    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def gpus(self) -> list[dict[str, Any]]:
        return list(self.latest().get("gpus") or [])

    def gpu_indices(self) -> list[int]:
        return [int(g["index"]) for g in self.gpus() if "index" in g]

    def history(self, n: int = 120) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._ring)[-n:]
