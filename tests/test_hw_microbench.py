"""Micro-benchmarks must degrade gracefully without CUDA / triton and never raise or fabricate numbers.
The real measurements are exercised in tests_gpu/test_hw_microbench_gpu.py (needs an NVIDIA GPU)."""
from __future__ import annotations

import json

import pytest
import torch

from forge.hw import microbench as mb


@pytest.fixture()
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_all_benchmarks_skip_without_cuda(no_cuda):
    results = [
        mb.stream_copy_bw(), mb.stream_copy_bw(kind="write"), mb.smem_bw(), mb.launch_latency(),
        mb.dequant_alu_throughput(), mb.p2p_bw(0, 1),
    ] + [mb.mma_throughput(d) for d in mb.SUPPORTED_MMA_DTYPES]
    for r in results:
        assert r.ok is False and r.skipped is True and r.value is None and r.error
        assert r.provenance == "measured" and r.samples == []
        json.dumps(r.to_dict())


def test_cuda_ready_reasons(no_cuda):
    ok, why = mb.cuda_ready(0)
    assert not ok and "not available" in why


def test_device_out_of_range(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    ok, why = mb.cuda_ready(5)
    assert not ok and "not found" in why


def test_runtime_errors_become_failed_results_not_exceptions(monkeypatch):
    """CUDA 'available' (mock) but the CPU-only torch cannot actually run anything -> ok=False with the error text."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=0: (9, 0), raising=False)
    for r in (mb.stream_copy_bw(), mb.mma_throughput("bf16"), mb.mma_throughput("fp8"), mb.launch_latency(),
              mb.p2p_bw(0, 0)):
        assert r.ok is False and r.value is None
        assert r.error and "Traceback" not in r.error


def test_unknown_dtype_is_a_failed_result():
    r = mb.mma_throughput("fp6")
    assert not r.ok and "unknown dtype" in r.error


def test_arch_gating_reports_skip_not_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    for sm, dt in ((70, "bf16"), (75, "tf32"), (80, "fp8"), (86, "fp8"), (90, "fp4"), (70, "int8")):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=0, sm=sm: (sm // 10, sm % 10), raising=False)
        r = mb.mma_throughput(dt)
        assert r.skipped and not r.ok and f"sm_{sm}" in r.error, (sm, dt, r.error)


def test_canon_dtype():
    assert mb.canon_dtype("bfloat16") == "bf16" and mb.canon_dtype("torch.float16") == "fp16"
    assert mb.canon_dtype("float8_e4m3fn") == "fp8" and mb.canon_dtype("NVFP4") == "fp4"
    assert mb.canon_dtype("tf32") == "tf32" and mb.canon_dtype("int8") == "int8"
    with pytest.raises(ValueError):
        mb.canon_dtype("fp6")


def test_stats_helper():
    st = mb._stats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    assert st["median"] == 4.5 and st["min"] == 1.0 and st["max"] == 8.0 and st["n"] == 8.0
    assert st["sustained_median"] == 6.5 and st["iqr"] == 4.0
    assert mb._stats([]) == {} and mb._stats([3.0])["median"] == 3.0


def test_bench_result_dict_is_compact():
    r = mb.BenchResult(name="x", ok=True, value=1.0, unit="u", samples=[0.1234567] * 200)
    d = r.to_dict()
    assert len(d["samples"]) == 64 and d["samples"][0] == 0.1235


def test_round_up():
    assert mb._round_up(1, 128) == 128 and mb._round_up(128, 128) == 128 and mb._round_up(129, 4) == 132


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful on CPU-only torch")
def test_timed_events_needs_cuda_and_is_guarded():
    # the helper itself requires CUDA; callers wrap it. On CPU torch it raises a clean error.
    with pytest.raises(Exception):
        mb._timed_events(lambda: None, 1, 0, 0)
