"""Micro-benchmarks on a real NVIDIA GPU (auto-skipped without CUDA).  These are the first-ever executions of the
code paths in forge/hw/microbench.py - failures here are bugs to fix, not flakiness."""
from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

from forge.hw import capabilities as cap  # noqa: E402
from forge.hw import chips  # noqa: E402
from forge.hw import microbench as mb  # noqa: E402


def _spec():
    c = cap.current_capabilities(0)
    return c, chips.find_chip(c.name, sm=c.sm, num_sms=c.num_sms, mem_gb=c.total_mem_gb)


def test_stream_copy_bandwidth_is_plausible():
    c, m = _spec()
    r = mb.stream_copy_bw(0, iters=10, max_seconds=2.0)
    assert r.ok, r.error
    assert r.unit == "GB/s" and r.value > 20
    ds = m.spec.get("mem_bw_gbs")
    if m.known and ds:
        assert 0.3 * ds < r.value < 1.1 * ds, (r.value, ds)


def test_stream_write_bandwidth():
    r = mb.stream_copy_bw(0, kind="write", iters=10, max_seconds=2.0)
    assert r.ok, r.error


@pytest.mark.parametrize("dtype", ["fp32", "tf32", "fp16", "bf16", "fp8", "int8", "fp4"])
def test_mma_throughput_each_dtype(dtype):
    c, m = _spec()
    r = mb.mma_throughput(dtype, 0, quick=True, max_seconds=1.0)
    expected_ok = {
        "fp32": True, "tf32": c.sm >= 80, "fp16": c.sm >= 70, "bf16": c.sm >= 80, "int8": c.sm >= 75,
        "fp8": c.sm >= 89, "fp4": c.sm >= 100,
    }[dtype]
    if not expected_ok:
        assert r.skipped and not r.ok
        return
    if dtype == "fp4" and r.skipped:
        pytest.skip(f"fp4 path unavailable in this torch build: {r.error}")
    assert r.ok, f"{dtype}: {r.error}"
    assert r.value > 0
    ds = m.spec.get("tflops", {}).get(dtype) if m.known else None
    if ds:
        basis = (m.spec.get("tflops_basis") or {}).get(dtype)
        # A measurement above the catalogue figure means the catalogue is too low (wrong per-SM rate / clock), which
        # is a data bug to report, not a hardware fault: only fail when it is implausible (>1.5x), or when the figure
        # is a real vendor datasheet number (basis "datasheet") and is exceeded by >1.25x.  Derived/approximate
        # (estimated) figures just warn.
        limit = 1.25 if basis == "datasheet" else 1.5
        assert r.value < limit * ds, (f"{dtype}: measured {r.value} vs catalogue {ds} ({basis}) - exceeds by more "
                                      f"than {limit}x; catalogue figure is wrong")
        if r.value > ds:
            warnings.warn(f"catalogue too low: {dtype} measured {r.value:.1f} > catalogue {ds} ({basis}) for {m.key}",
                          stacklevel=1)


def test_launch_latency():
    r = mb.launch_latency(0, n_kernels=500, repeats=3)
    assert r.ok, r.error
    assert 0.2 < r.value < 100  # microseconds per tiny kernel


@pytest.mark.skipif(not mb.triton_ready()[0], reason="triton not installed")
def test_smem_proxy_and_dequant():
    r = mb.smem_bw(0, iters=3)
    assert r.ok, r.error
    d = mb.dequant_alu_throughput(0, iters=3)
    assert d.ok, d.error
    assert d.params["simple_ops_per_s_t"] > 0


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
def test_p2p():
    r = mb.p2p_bw(0, 1, size_mb=64, iters=3)
    assert r.ok, r.error
