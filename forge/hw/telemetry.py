"""Live GPU telemetry: ``GpuSampler`` (pynvml -> ``nvidia-smi`` -> ``[]``), throttle-reason decoding and a
background ``ThrottleMonitor`` used by the probe to record clocks/temperature/power while benchmarking.

Nothing here requires a GPU or ``pynvml``: every failure degrades to the next backend and finally to an empty
list.  Sample dict keys (``None`` when the driver does not expose a value)::

    index, name, uuid, util (GPU %), mem_util (%), mem_used_mb, mem_total_mb, temp_c, power_w, power_limit_w,
    clock_mhz (SM), clock_mem_mhz, clock_max_mhz (max SM clock), fan_pct, pstate, pcie_gen, pcie_width,
    throttle_mask, throttle_reasons (list[str]), throttling (bool: a power/thermal limiter is active),
    energy_j (cumulative board energy, pynvml only), ts (unix seconds), source ('pynvml' | 'nvidia-smi')
"""
from __future__ import annotations

import csv
import io
import logging
import shutil
import statistics
import subprocess
import threading
import time
from collections import deque
from typing import Any

log = logging.getLogger("forge.hw.telemetry")

# NVML clocks-event / throttle-reason bit masks (stable across driver versions)
THROTTLE_BITS: dict[int, str] = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}
#: reasons that actually limit performance (idle / application-clock / sync-boost / display are benign)
THROTTLING_MASK = 0x4 | 0x8 | 0x20 | 0x40 | 0x80


def decode_throttle_reasons(mask: int | None) -> list[str]:
    """``0x24 -> ['sw_power_cap', 'sw_thermal_slowdown']``; ``None``/0 -> []."""
    if not mask:
        return []
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


def is_throttling(mask: int | None) -> bool:
    """True if a power/thermal limiter bit is set."""
    return bool(mask and (mask & THROTTLING_MASK))


# --------------------------------------------------------------------------------------------
# helpers (module level so tests can monkeypatch them)
# --------------------------------------------------------------------------------------------
def _load_pynvml() -> Any | None:
    """Import ``pynvml`` (package ``nvidia-ml-py``); None if unavailable."""
    try:
        import pynvml  # type: ignore

        return pynvml
    except Exception:
        return None


def _run_smi(args: list[str], timeout: float = 5.0) -> tuple[int, str]:
    """Run ``nvidia-smi`` with ``args``; returns (returncode, stdout). rc=-1 if the binary is missing/fails."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return -1, ""
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout, check=False)
        return out.returncode, out.stdout
    except Exception:
        return -1, ""


def _txt(v: Any) -> str:
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def _num(s: str) -> float | None:
    s = s.strip()
    if not s or s.startswith("[") or s.upper() in ("N/A", "NOT SUPPORTED"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


_SMI_BASE_COLUMNS = [
    "index", "name", "uuid", "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
    "temperature.gpu", "power.draw", "power.limit", "clocks.sm", "clocks.mem", "clocks.max.sm", "fan.speed",
    "pstate", "pcie.link.gen.current", "pcie.link.width.current",
]
_SMI_THROTTLE_COLUMNS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")


def parse_smi_csv(text: str, columns: list[str]) -> list[dict[str, Any]]:
    """Parse ``nvidia-smi --query-gpu=<columns> --format=csv,noheader,nounits`` output into sample dicts."""
    rows = []
    for rec in csv.reader(io.StringIO(text.strip()), skipinitialspace=True):
        if not rec or len(rec) < len(columns):
            continue
        raw = dict(zip(columns, (x.strip() for x in rec)))
        if not raw["index"].isdigit():
            continue
        mask = None
        for c in _SMI_THROTTLE_COLUMNS:
            if c in raw:
                try:
                    mask = int(raw[c], 16) if raw[c].lower().startswith("0x") else int(raw[c])
                except ValueError:
                    mask = None
        pstate = raw.get("pstate", "")
        rows.append({
            "index": int(raw["index"]), "name": raw["name"], "uuid": raw["uuid"],
            "util": _num(raw["utilization.gpu"]), "mem_util": _num(raw["utilization.memory"]),
            "mem_used_mb": _num(raw["memory.used"]), "mem_total_mb": _num(raw["memory.total"]),
            "temp_c": _num(raw["temperature.gpu"]), "power_w": _num(raw["power.draw"]),
            "power_limit_w": _num(raw["power.limit"]), "clock_mhz": _num(raw["clocks.sm"]),
            "clock_mem_mhz": _num(raw["clocks.mem"]), "clock_max_mhz": _num(raw["clocks.max.sm"]),
            "fan_pct": _num(raw["fan.speed"]), "pstate": pstate if pstate and not pstate.startswith("[") else None,
            "pcie_gen": _num(raw["pcie.link.gen.current"]), "pcie_width": _num(raw["pcie.link.width.current"]),
            "throttle_mask": mask, "throttle_reasons": decode_throttle_reasons(mask), "throttling": is_throttling(mask),
            "energy_j": None, "ts": time.time(), "source": "nvidia-smi",
        })
    return rows


# --------------------------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------------------------
class GpuSampler:
    """Poll GPU state.  ``backend`` is ``'auto'`` (pynvml, then nvidia-smi, then none), ``'pynvml'``,
    ``'nvidia-smi'`` or ``'none'``.  Thread-safe; ``close()`` releases NVML.

    ``sample()`` returns one dict per GPU (see module docstring) or ``[]`` when no backend works - callers must
    treat an empty list as "no GPUs / no telemetry".  ``start()`` runs a background thread filling a ring buffer
    (``history()``) for UIs.
    """

    def __init__(self, backend: str = "auto", *, include_processes: bool = False, smi_timeout: float = 5.0) -> None:
        if backend not in ("auto", "pynvml", "nvidia-smi", "none"):
            raise ValueError(f"unknown telemetry backend {backend!r}")
        self._requested = backend
        self.include_processes = include_processes
        self.smi_timeout = smi_timeout
        self._lock = threading.RLock()
        self._nvml: Any | None = None
        self._backend: str | None = None
        self._smi_columns: list[str] | None = None
        self._ring: deque[list[dict[str, Any]]] = deque(maxlen=600)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------------------------
    @property
    def backend(self) -> str:
        """Active backend after first use: 'pynvml' | 'nvidia-smi' | 'none' (initialises lazily)."""
        self._ensure()
        return self._backend or "none"

    def _ensure(self) -> None:
        with self._lock:
            if self._backend is not None:
                return
            req = self._requested
            if req in ("auto", "pynvml") and self._init_nvml():
                self._backend = "pynvml"
            elif req in ("auto", "nvidia-smi") and self._init_smi():
                self._backend = "nvidia-smi"
            else:
                self._backend = "none"

    def _init_nvml(self) -> bool:
        mod = _load_pynvml()
        if mod is None:
            return False
        try:
            mod.nvmlInit()
            mod.nvmlDeviceGetCount()
        except Exception as e:
            log.debug("pynvml init failed: %s", e)
            return False
        self._nvml = mod
        return True

    def _init_smi(self) -> bool:
        rc, out = _run_smi(["--query-gpu=index", "--format=csv,noheader,nounits"], self.smi_timeout)
        if rc != 0:
            return False
        cols = list(_SMI_BASE_COLUMNS)
        for tcol in _SMI_THROTTLE_COLUMNS:  # newer drivers renamed the field; probe once which works
            rc2, out2 = _run_smi([f"--query-gpu={','.join(cols + [tcol])}", "--format=csv,noheader,nounits"],
                                 self.smi_timeout)
            if rc2 == 0 and out2.strip():
                cols.append(tcol)
                break
        self._smi_columns = cols
        return True

    def close(self) -> None:
        self.stop()
        with self._lock:
            if self._nvml is not None:
                try:
                    self._nvml.nvmlShutdown()
                except Exception:
                    pass
            self._nvml = None
            self._backend = None

    def __enter__(self) -> GpuSampler:
        return self

    def __exit__(self, *a: Any) -> None:
        self.close()

    # -- sampling ----------------------------------------------------------------------------
    def sample(self, indices: list[int] | None = None) -> list[dict[str, Any]]:
        """One snapshot per GPU (optionally only ``indices``). Never raises."""
        try:
            self._ensure()
            with self._lock:
                rows = self._sample_nvml() if self._backend == "pynvml" else (
                    self._sample_smi() if self._backend == "nvidia-smi" else [])
        except Exception as e:  # driver hiccup: try to fall back once
            log.debug("telemetry sample failed on %s: %s", self._backend, e)
            rows = []
            with self._lock:
                if self._backend == "pynvml":
                    self._backend = None
                    self._requested = "nvidia-smi"
        if indices is not None:
            rows = [r for r in rows if r["index"] in indices]
        return rows

    def _sample_nvml(self) -> list[dict[str, Any]]:
        n = self._nvml
        assert n is not None
        count = int(n.nvmlDeviceGetCount())
        sm_clk = getattr(n, "NVML_CLOCK_SM", 1)
        mem_clk = getattr(n, "NVML_CLOCK_MEM", 2)
        temp_gpu = getattr(n, "NVML_TEMPERATURE_GPU", 0)
        rows = []
        for i in range(count):
            h = n.nvmlDeviceGetHandleByIndex(i)

            def g(fn: str, *args: Any, _h: Any = h) -> Any:
                f = getattr(n, fn, None)
                if f is None:
                    return None
                try:
                    return f(_h, *args)
                except Exception:
                    return None

            name = g("nvmlDeviceGetName")
            uuid = g("nvmlDeviceGetUUID")
            ur = g("nvmlDeviceGetUtilizationRates")
            mi = g("nvmlDeviceGetMemoryInfo")
            pw = g("nvmlDeviceGetPowerUsage")
            pl = g("nvmlDeviceGetEnforcedPowerLimit")
            if pl is None:
                pl = g("nvmlDeviceGetPowerManagementLimit")
            mask = g("nvmlDeviceGetCurrentClocksEventReasons")
            if mask is None:
                mask = g("nvmlDeviceGetCurrentClocksThrottleReasons")
            ps = g("nvmlDeviceGetPerformanceState")
            en = g("nvmlDeviceGetTotalEnergyConsumption")
            row = {
                "index": i, "name": _txt(name) if name is not None else None,
                "uuid": _txt(uuid) if uuid is not None else None,
                "util": float(ur.gpu) if ur is not None else None,
                "mem_util": float(ur.memory) if ur is not None else None,
                "mem_used_mb": round(mi.used / 2**20, 1) if mi is not None else None,
                "mem_total_mb": round(mi.total / 2**20, 1) if mi is not None else None,
                "temp_c": _f(g("nvmlDeviceGetTemperature", temp_gpu)),
                "power_w": round(pw / 1000.0, 2) if pw is not None else None,
                "power_limit_w": round(pl / 1000.0, 2) if pl is not None else None,
                "clock_mhz": _f(g("nvmlDeviceGetClockInfo", sm_clk)),
                "clock_mem_mhz": _f(g("nvmlDeviceGetClockInfo", mem_clk)),
                "clock_max_mhz": _f(g("nvmlDeviceGetMaxClockInfo", sm_clk)),
                "fan_pct": _f(g("nvmlDeviceGetFanSpeed")),
                "pstate": f"P{int(ps)}" if isinstance(ps, int) and 0 <= ps <= 15 else None,
                "pcie_gen": _f(g("nvmlDeviceGetCurrPcieLinkGeneration")),
                "pcie_width": _f(g("nvmlDeviceGetCurrPcieLinkWidth")),
                "throttle_mask": int(mask) if mask is not None else None,
                "throttle_reasons": decode_throttle_reasons(int(mask)) if mask is not None else [],
                "throttling": is_throttling(int(mask)) if mask is not None else False,
                "energy_j": round(en / 1000.0, 3) if en is not None else None,
                "ts": time.time(), "source": "pynvml",
            }
            if self.include_processes:
                procs = g("nvmlDeviceGetComputeRunningProcesses") or []
                row["processes"] = [{"pid": int(p.pid), "used_mb": (round(p.usedGpuMemory / 2**20, 1)
                                                                      if getattr(p, "usedGpuMemory", None) else None)}
                                    for p in procs]
            rows.append(row)
        return rows

    def _sample_smi(self) -> list[dict[str, Any]]:
        cols = self._smi_columns or list(_SMI_BASE_COLUMNS)
        rc, out = _run_smi([f"--query-gpu={','.join(cols)}", "--format=csv,noheader,nounits"], self.smi_timeout)
        if rc != 0:
            return []
        rows = parse_smi_csv(out, cols)
        if self.include_processes and rows:
            rc, pout = _run_smi(["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
                                self.smi_timeout)
            procs: dict[str, list[dict[str, Any]]] = {}
            if rc == 0:
                for rec in csv.reader(io.StringIO(pout.strip()), skipinitialspace=True):
                    if len(rec) >= 3 and rec[1].strip().isdigit():
                        procs.setdefault(rec[0].strip(), []).append({"pid": int(rec[1]), "used_mb": _num(rec[2])})
            for r in rows:
                r["processes"] = procs.get(r["uuid"], [])
        return rows

    # -- background ring buffer --------------------------------------------------------------
    def start(self, interval_s: float = 1.0, history: int = 600) -> None:
        """Sample every ``interval_s`` seconds in a daemon thread into a ring buffer of ``history`` snapshots."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ring = deque(self._ring, maxlen=int(history))
            self._stop.clear()

            def loop() -> None:
                while not self._stop.is_set():
                    snap = self.sample()
                    if snap:
                        self._ring.append(snap)
                    self._stop.wait(max(0.05, interval_s))

            self._thread = threading.Thread(target=loop, name="forge-gpu-sampler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3.0)
        self._thread = None

    def history(self, n: int | None = None) -> list[list[dict[str, Any]]]:
        """Most recent ring-buffer snapshots (oldest first)."""
        snaps = list(self._ring)
        return snaps[-n:] if n else snaps

    def latest(self) -> list[dict[str, Any]]:
        return self._ring[-1] if self._ring else []


def _f(v: Any) -> float | None:
    return None if v is None else float(v)


# --------------------------------------------------------------------------------------------
# throttle monitor (used by the probe)
# --------------------------------------------------------------------------------------------
class ThrottleMonitor:
    """Context manager that samples one GPU in a background thread and summarises clocks / temperature / power /
    throttle reasons seen in the window.  ``summary()['available']`` is False when no telemetry backend works."""

    def __init__(self, index: int = 0, interval_s: float = 0.25, sampler: GpuSampler | None = None) -> None:
        self.index = index
        self.interval_s = interval_s
        self._sampler = sampler or GpuSampler()
        self._own = sampler is None
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ThrottleMonitor:
        first = self._sampler.sample([self.index])
        if first:
            self.samples.append(first[0])
            self._thread = threading.Thread(target=self._loop, daemon=True, name="forge-throttle-monitor")
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            s = self._sampler.sample([self.index])
            if s:
                self.samples.append(s[0])

    def __exit__(self, *a: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        last = self._sampler.sample([self.index])
        if last:
            self.samples.append(last[0])
        if self._own:
            self._sampler.close()

    def summary(self) -> dict[str, Any]:
        s = self.samples
        if not s:
            return {"available": False, "n_samples": 0}

        def col(k: str) -> list[float]:
            return [x[k] for x in s if x.get(k) is not None]

        reasons: set[str] = set()
        thr = 0
        for x in s:
            reasons.update(x.get("throttle_reasons") or [])
            thr += 1 if x.get("throttling") else 0
        clk, tmp, pw = col("clock_mhz"), col("temp_c"), col("power_w")
        return {
            "available": True, "n_samples": len(s), "source": s[0].get("source"),
            "clock_sm_mhz_min": min(clk) if clk else None, "clock_sm_mhz_max": max(clk) if clk else None,
            "clock_sm_mhz_median": statistics.median(clk) if clk else None,
            "clock_max_mhz": s[-1].get("clock_max_mhz"),
            "temp_c_max": max(tmp) if tmp else None, "power_w_max": max(pw) if pw else None,
            "power_limit_w": s[-1].get("power_limit_w"),
            "throttle_reasons_seen": sorted(reasons), "throttled_fraction": thr / len(s),
            "throttled": thr > 0,
        }


def read_gpu_state(index: int = 0) -> dict[str, Any] | None:
    """One-off sample of GPU ``index`` (None if no telemetry)."""
    with GpuSampler() as s:
        r = s.sample([index])
        return r[0] if r else None


# --------------------------------------------------------------------------------------------
# NVLink
# --------------------------------------------------------------------------------------------
def nvlink_active_links(index: int = 0) -> int | None:
    """Number of active NVLink links of GPU ``index`` via pynvml (0 on PCIe-only boards, None if unknown)."""
    mod = _load_pynvml()
    if mod is None:
        return None
    try:
        mod.nvmlInit()
    except Exception:
        return None
    try:
        h = mod.nvmlDeviceGetHandleByIndex(index)
        active = 0
        for link in range(int(getattr(mod, "NVML_NVLINK_MAX_LINKS", 18))):
            try:
                if mod.nvmlDeviceGetNvLinkState(h, link):
                    active += 1
            except Exception:
                continue
        return active
    except Exception:
        return None
    finally:
        try:
            mod.nvmlShutdown()
        except Exception:
            pass


# --------------------------------------------------------------------------------------------
# torch index -> NVML index (CUDA_VISIBLE_DEVICES remapping)
# --------------------------------------------------------------------------------------------
def resolve_nvml_index(torch_index: int, uuid: str | None = None, rows: list[dict[str, Any]] | None = None,
                       env: str | None = None) -> int:
    """NVML/nvidia-smi index of torch CUDA device ``torch_index``.

    Matching order: device UUID (torch ``props.uuid`` vs NVML ``GPU-...``), then ``CUDA_VISIBLE_DEVICES``
    (integer list or UUID list), else identity."""
    import os

    if uuid and rows:
        u = str(uuid).lower().replace("gpu-", "").strip("'\"")
        for r in rows:
            ru = (r.get("uuid") or "").lower().replace("gpu-", "")
            if ru and (ru == u or u.endswith(ru) or ru.endswith(u)):
                return int(r["index"])
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "") if env is None else env
    toks = [t.strip() for t in cvd.split(",") if t.strip()]
    if 0 <= torch_index < len(toks):
        t = toks[torch_index]
        if t.isdigit():
            return int(t)
        if rows:
            tu = t.lower().replace("gpu-", "")
            for r in rows:
                ru = (r.get("uuid") or "").lower().replace("gpu-", "")
                if ru and (ru.startswith(tu) or tu.startswith(ru)):
                    return int(r["index"])
    return torch_index
