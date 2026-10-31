"""GpuSampler backends (fake pynvml, fixture nvidia-smi output), throttle decoding, monitor, NVML index mapping."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from forge.hw import telemetry as tm

SMI_OUT = """\
0, NVIDIA H100 80GB HBM3, GPU-11111111-aaaa-bbbb-cccc-000000000000, 97, 41, 61234, 81559, 63, 512.34, 700.00, 1980, 2619, 1980, [N/A], P0, 5, 16, 0x0000000000000004
1, NVIDIA GeForce RTX 4090, GPU-22222222-aaaa-bbbb-cccc-000000000000, 0, 0, 512, 24564, 31, 22.5, 450.00, 210, 405, 3105, 30, P8, 4, 16, 0x0000000000000001
"""
SMI_COLS = tm._SMI_BASE_COLUMNS + ["clocks_event_reasons.active"]


class FakeNvml:
    """Minimal pynvml look-alike with 2 GPUs. Some calls raise like real NVML on unsupported hardware."""

    NVML_CLOCK_SM, NVML_CLOCK_MEM, NVML_TEMPERATURE_GPU = 1, 2, 0
    NVML_NVLINK_MAX_LINKS = 18

    def __init__(self, count=2, init_fail=False, nvlink_active=12):
        self.count, self.init_fail, self.nvlink_active = count, init_fail, nvlink_active
        self.inited = self.shut = False

    def nvmlInit(self):
        if self.init_fail:
            raise RuntimeError("NVML Shared Library Not Found")
        self.inited = True

    def nvmlShutdown(self):
        self.shut = True

    def nvmlDeviceGetCount(self):
        return self.count

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetName(self, h):
        return [b"NVIDIA H100 80GB HBM3", "NVIDIA GeForce RTX 4090"][h]

    def nvmlDeviceGetUUID(self, h):
        return f"GPU-{h}{h}{h}{h}-uuid"

    def nvmlDeviceGetUtilizationRates(self, h):
        return SimpleNamespace(gpu=80 - h * 10, memory=40)

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(used=(h + 1) * 1024 * 2**20, total=(h + 1) * 8192 * 2**20)

    def nvmlDeviceGetTemperature(self, h, kind):
        return 60 + h

    def nvmlDeviceGetPowerUsage(self, h):
        return 300500

    def nvmlDeviceGetEnforcedPowerLimit(self, h):
        return 700000

    def nvmlDeviceGetClockInfo(self, h, kind):
        return 1900 if kind == self.NVML_CLOCK_SM else 2600

    def nvmlDeviceGetMaxClockInfo(self, h, kind):
        return 1980

    def nvmlDeviceGetFanSpeed(self, h):
        raise RuntimeError("Not Supported")

    def nvmlDeviceGetPerformanceState(self, h):
        return 0

    def nvmlDeviceGetCurrentClocksEventReasons(self, h):
        return 0x4 | 0x1

    def nvmlDeviceGetCurrPcieLinkGeneration(self, h):
        return 5

    def nvmlDeviceGetCurrPcieLinkWidth(self, h):
        return 16

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        return 5_000_000  # mJ

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [SimpleNamespace(pid=4242, usedGpuMemory=2048 * 2**20)]

    def nvmlDeviceGetNvLinkState(self, h, link):
        if link < self.nvlink_active:
            return 1
        raise RuntimeError("link inactive/unsupported")


@pytest.fixture()
def no_smi(monkeypatch):
    monkeypatch.setattr(tm, "_run_smi", lambda args, timeout=5.0: (-1, ""))


def test_decode_throttle():
    assert tm.decode_throttle_reasons(0) == [] and tm.decode_throttle_reasons(None) == []
    assert tm.decode_throttle_reasons(0x24) == ["sw_power_cap", "sw_thermal_slowdown"]
    assert tm.is_throttling(0x4) and tm.is_throttling(0x40) and tm.is_throttling(0x8)
    assert not tm.is_throttling(0x1) and not tm.is_throttling(0x2) and not tm.is_throttling(0x102) and not tm.is_throttling(None)


def test_pynvml_backend(monkeypatch, no_smi):
    fake = FakeNvml()
    monkeypatch.setattr(tm, "_load_pynvml", lambda: fake)
    s = tm.GpuSampler()
    assert s.backend == "pynvml" and fake.inited
    rows = s.sample()
    assert [r["index"] for r in rows] == [0, 1]
    r = rows[0]
    assert r["name"] == "NVIDIA H100 80GB HBM3" and r["util"] == 80 and r["mem_used_mb"] == 1024
    assert r["mem_total_mb"] == 8192 and r["temp_c"] == 60 and r["power_w"] == 300.5 and r["power_limit_w"] == 700
    assert r["clock_mhz"] == 1900 and r["clock_max_mhz"] == 1980 and r["clock_mem_mhz"] == 2600
    assert r["fan_pct"] is None  # NVML raised -> None, not a crash
    assert r["pstate"] == "P0" and r["pcie_gen"] == 5 and r["pcie_width"] == 16
    assert r["throttle_mask"] == 5 and r["throttle_reasons"] == ["gpu_idle", "sw_power_cap"] and r["throttling"]
    assert r["energy_j"] == 5000.0 and r["source"] == "pynvml" and isinstance(r["ts"], float)
    assert rows[1]["name"] == "NVIDIA GeForce RTX 4090"
    assert [x["index"] for x in s.sample([1])] == [1]
    s.close()
    assert fake.shut


def test_pynvml_processes(monkeypatch, no_smi):
    monkeypatch.setattr(tm, "_load_pynvml", lambda: FakeNvml(count=1))
    with tm.GpuSampler(include_processes=True) as s:
        assert s.sample()[0]["processes"] == [{"pid": 4242, "used_mb": 2048.0}]


def test_falls_back_to_smi_when_nvml_missing_or_broken(monkeypatch):
    calls = []

    def fake_smi(args, timeout=5.0):
        calls.append(args)
        q = [a for a in args if a.startswith("--query-gpu=")][0]
        if q == "--query-gpu=index":
            return 0, "0\n1\n"
        if "clocks_event_reasons.active" in q:
            return 0, SMI_OUT
        return 1, ""

    monkeypatch.setattr(tm, "_run_smi", fake_smi)
    for loader in (lambda: None, lambda: FakeNvml(init_fail=True)):
        monkeypatch.setattr(tm, "_load_pynvml", loader)
        s = tm.GpuSampler()
        assert s.backend == "nvidia-smi"
        rows = s.sample()
        assert len(rows) == 2 and rows[0]["source"] == "nvidia-smi"
        assert rows[0]["util"] == 97 and rows[0]["mem_used_mb"] == 61234 and rows[0]["mem_total_mb"] == 81559
        assert rows[0]["temp_c"] == 63 and rows[0]["power_w"] == 512.34 and rows[0]["power_limit_w"] == 700
        assert rows[0]["clock_mhz"] == 1980 and rows[0]["clock_max_mhz"] == 1980
        assert rows[0]["fan_pct"] is None and rows[1]["fan_pct"] == 30
        assert rows[0]["pstate"] == "P0" and rows[0]["pcie_gen"] == 5
        assert rows[0]["throttle_reasons"] == ["sw_power_cap"] and rows[0]["throttling"]
        assert rows[1]["throttle_reasons"] == ["gpu_idle"] and not rows[1]["throttling"]
        s.close()


def test_smi_old_driver_field_name(monkeypatch):
    """Drivers that only know ``clocks_throttle_reasons.active`` are handled (query without it as last resort)."""

    def fake_smi(args, timeout=5.0):
        q = [a for a in args if a.startswith("--query-gpu=")][0]
        if q == "--query-gpu=index":
            return 0, "0\n"
        if "clocks_event_reasons" in q:
            return 2, ""
        if "clocks_throttle_reasons.active" in q:
            return 0, SMI_OUT.splitlines()[0] + "\n"
        return 0, ""

    monkeypatch.setattr(tm, "_run_smi", fake_smi)
    monkeypatch.setattr(tm, "_load_pynvml", lambda: None)
    rows = tm.GpuSampler().sample()
    assert rows[0]["throttle_reasons"] == ["sw_power_cap"]


def test_no_backend_returns_empty(monkeypatch, no_smi):
    monkeypatch.setattr(tm, "_load_pynvml", lambda: None)
    s = tm.GpuSampler()
    assert s.backend == "none" and s.sample() == []
    s.close()
    assert tm.GpuSampler("none").sample() == []
    with pytest.raises(ValueError):
        tm.GpuSampler("bogus")


def test_forced_backend(monkeypatch):
    monkeypatch.setattr(tm, "_load_pynvml", lambda: FakeNvml(count=1))
    monkeypatch.setattr(tm, "_run_smi", lambda args, timeout=5.0: (-1, ""))
    assert tm.GpuSampler("nvidia-smi").backend == "none"  # not allowed to use NVML, smi absent
    assert tm.GpuSampler("pynvml").backend == "pynvml"


def test_sample_never_raises(monkeypatch, no_smi):
    class Bad(FakeNvml):
        def nvmlDeviceGetCount(self):
            if self.inited and getattr(self, "boom", False):
                raise RuntimeError("GPU fell off the bus")
            return 1

    bad = Bad()
    monkeypatch.setattr(tm, "_load_pynvml", lambda: bad)
    s = tm.GpuSampler()
    assert len(s.sample()) == 1
    bad.boom = True
    assert s.sample() == []


def test_parse_smi_csv_na_values():
    row = "0, GPU X, GPU-1, [Not Supported], [N/A], 100, 200, N/A, [N/A], [N/A], 1, 2, 3, [N/A], [N/A], 4, 16, 0x0"
    cols = tm._SMI_BASE_COLUMNS + ["clocks_throttle_reasons.active"]
    r = tm.parse_smi_csv(row, cols)[0]
    assert r["util"] is None and r["temp_c"] is None and r["pstate"] is None and r["throttle_reasons"] == []
    assert tm.parse_smi_csv("", cols) == [] and tm.parse_smi_csv("x,y", cols) == []


def test_background_ring_buffer(monkeypatch, no_smi):
    monkeypatch.setattr(tm, "_load_pynvml", lambda: FakeNvml(count=1))
    s = tm.GpuSampler()
    s.start(interval_s=0.05, history=5)
    deadline = time.time() + 5
    while len(s.history()) < 3 and time.time() < deadline:
        time.sleep(0.02)
    s.stop()
    h = s.history()
    assert 3 <= len(h) <= 5 and h[-1][0]["index"] == 0 and s.latest() == h[-1]
    s.close()


def test_throttle_monitor_summary():
    class Seq:
        def __init__(self):
            self.i = 0
            self.closed = False

        def sample(self, idx=None):
            self.i += 1
            return [{"index": 0, "clock_mhz": 1900 - 100 * self.i, "temp_c": 60 + self.i, "power_w": 300 + self.i,
                     "throttle_reasons": ["sw_power_cap"] if self.i >= 3 else [], "throttling": self.i >= 3,
                     "clock_max_mhz": 1980, "power_limit_w": 700, "source": "fake"}]

        def close(self):
            self.closed = True

    seq = Seq()
    with tm.ThrottleMonitor(0, interval_s=0.01, sampler=seq) as mon:
        time.sleep(0.15)
    s = mon.summary()
    assert s["available"] and s["n_samples"] >= 3 and s["throttled"] and s["throttle_reasons_seen"] == ["sw_power_cap"]
    assert s["clock_sm_mhz_max"] > s["clock_sm_mhz_min"] and s["temp_c_max"] >= 62 and 0 < s["throttled_fraction"] <= 1
    assert not seq.closed  # caller-supplied sampler is not closed


def test_throttle_monitor_without_telemetry(monkeypatch, no_smi):
    monkeypatch.setattr(tm, "_load_pynvml", lambda: None)
    with tm.ThrottleMonitor(0) as mon:
        pass
    assert mon.summary() == {"available": False, "n_samples": 0}


def test_nvlink_active_links(monkeypatch):
    monkeypatch.setattr(tm, "_load_pynvml", lambda: FakeNvml(nvlink_active=12))
    assert tm.nvlink_active_links(0) == 12
    monkeypatch.setattr(tm, "_load_pynvml", lambda: FakeNvml(nvlink_active=0))
    assert tm.nvlink_active_links(0) == 0
    monkeypatch.setattr(tm, "_load_pynvml", lambda: None)
    assert tm.nvlink_active_links(0) is None
    monkeypatch.setattr(tm, "_load_pynvml", lambda: FakeNvml(init_fail=True))
    assert tm.nvlink_active_links(0) is None


def test_resolve_nvml_index(monkeypatch):
    rows = [{"index": 0, "uuid": "GPU-aaaa"}, {"index": 1, "uuid": "GPU-bbbb"}, {"index": 2, "uuid": "GPU-cccc"}]
    assert tm.resolve_nvml_index(0, "bbbb", rows) == 1
    assert tm.resolve_nvml_index(0, "GPU-cccc", rows) == 2
    assert tm.resolve_nvml_index(0, None, rows, env="2,3") == 2
    assert tm.resolve_nvml_index(1, None, rows, env="2,0") == 0
    assert tm.resolve_nvml_index(0, None, rows, env="GPU-bbbb") == 1
    assert tm.resolve_nvml_index(0, None, rows, env="") == 0
    assert tm.resolve_nvml_index(1, None, None, env="") == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert tm.resolve_nvml_index(0, None, None) == 3
