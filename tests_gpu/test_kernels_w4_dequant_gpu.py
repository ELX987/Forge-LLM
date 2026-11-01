"""GPU tests (auto-skipped without CUDA): W4 dequantise kernel and the large-M ``dequant_matmul`` path vs the fp32 oracle, row
chunking / temp-memory bound, CUDA graph capture without host sync (act-order), routing.

NEVER run by the developer (no NVIDIA GPU on the dev machine); logic is covered by the Triton CPU interpreter tests in
tests/test_kernels_w4_dequant_interp.py.  This file is the compile + numerics + memory gate for the GPU host:

    pytest tests_gpu/test_kernels_w4_dequant_gpu.py -q -s
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
pytest.importorskip("triton")

from forge.kernels import ops_gemm  # noqa: E402
from forge.kernels.layouts import PORTABLE_W4, pack_w4, quantize_w4_rtn  # noqa: E402
from forge.kernels.ref import compare, w4a16_gemm_ref  # noqa: E402
from forge.kernels.spec import load_spec  # noqa: E402
from forge.kernels.triton.gemm_w4_dequant import w4_dequant  # noqa: E402
from forge.kernels.triton.gemm_w4a16 import prepare_w4, w4a16_gemm_triton  # noqa: E402
from forge.kernels.w4_tuning import default_w4_config, plan_dequant_chunks  # noqa: E402

DEV = "cuda"
TOL = load_spec("w4a16_gemm").tolerance
LARGE_MS = [32, 64, 256, 1024]


def _chip():
    from forge.kernels.layouts.chip import chip_info

    return chip_info(None, 0)


def _dtypes():
    out = [torch.float16]
    if torch.cuda.get_device_capability()[0] >= 8:
        out.append(torch.bfloat16)
    return out


def make(n, k, gs, sym, act_order=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.05).to(DEV)
    q, s, z = quantize_w4_rtn(w, gs, sym=sym)
    gsz = k if gs <= 0 else gs
    gi = (torch.arange(k, device=DEV) // gsz)[torch.randperm(k, generator=g).to(DEV)].to(torch.int32) if act_order else None
    return pack_w4(q, s, z, gs, PORTABLE_W4, gi)


def run(x, p, bias=None, **cfg):
    chip = _chip()
    return w4a16_gemm_triton(x, p, bias, config=cfg or None, sm=chip.sm, num_sms=chip.num_sms)


SHAPES = [(4096, 4096, 128), (1000, 4096, 64), (4096, 1000, 32), (777, 2056, 128), (333, 131, 32), (256, 128, -1), (5120, 5120, 128)]


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("n,k,gs", SHAPES)
def test_dequant_kernel_matches_oracle_weight(n, k, gs, act_order, sym):
    if act_order and gs <= 0:
        pytest.skip("act-order needs groups")
    p = make(n, k, gs, sym, act_order=act_order)
    for dtype in _dtypes():
        ref = p.dequantize(torch.float32)
        for cfg in (None, dict(tile_n=16, tile_k=64), dict(tile_n=64, tile_k=256, warps=8)):
            w = w4_dequant(p.qweight, p.scales, p.zeros, p.g_idx, n=p.n, k=p.k, group_size=p.group_size, dtype=dtype, config=cfg)
            assert w.shape == (n, k) and w.dtype == dtype
            # one fp32 multiply + one rounding to the 16-bit type: at most 1 ulp from the correctly rounded oracle
            torch.testing.assert_close(w.float(), ref.to(dtype).float(), rtol=2.0 ** -8 if dtype == torch.bfloat16 else 2.0 ** -11, atol=0.0)


@pytest.mark.parametrize("row0,rows", [(0, 64), (64, 1000), (4096 - 72, 72), (8, 8)])
def test_dequant_row_chunk_writes_only_its_rows(row0, rows):
    p = make(4096, 1024, 128, False)
    buf = torch.full((max(rows, 1100), 1024), float("nan"), device=DEV, dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    out = w4_dequant(p.qweight, p.scales, p.zeros, None, n=p.n, k=p.k, group_size=p.group_size, dtype=buf.dtype, row0=row0, rows=rows, out=buf)
    assert out.shape == (rows, 1024)
    torch.testing.assert_close(out.float(), p.dequantize(torch.float32)[row0:row0 + rows].to(buf.dtype).float(), rtol=2.0 ** -8, atol=0.0)
    assert torch.isnan(buf[rows:]).all()


@pytest.mark.parametrize("act_order", [False, True])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("m", LARGE_MS)
@pytest.mark.parametrize("n,k,gs", SHAPES)
def test_dequant_matmul_matches_oracle(n, k, gs, m, sym, act_order):
    if act_order and gs <= 0:
        pytest.skip("act-order needs groups")
    p = make(n, k, gs, sym, act_order=act_order)
    for dtype in _dtypes():
        x = torch.randn(m, k, device=DEV, dtype=dtype)
        bias = torch.randn(n, device=DEV, dtype=dtype)
        ref = w4a16_gemm_ref(x, p, bias)
        for cfg in ({"impl": "dequant_matmul"}, {}):  # explicit and via the default routing (M >= 32 -> dequant_matmul)
            y = run(x, p, bias, **cfg)
            res = compare(y, ref, TOL, out_dtype=dtype)
            assert res.ok, (cfg, dtype, res.detail)


def test_default_routing_is_dequant_matmul_for_large_m(monkeypatch):
    chip = _chip()
    monkeypatch.delenv("FORGE_W4_DEQUANT_MIN_M", raising=False)
    assert default_w4_config(1024, 5120, 5120, sm=chip.sm, num_sms=chip.num_sms)["impl"] == "dequant_matmul"
    p = make(2048, 2048, 128, False)
    x = torch.randn(256, 2048, device=DEV, dtype=torch.float16)
    ops_gemm.w4a16_linear(x, p)
    assert ops_gemm.last_path("w4a16_linear") == "triton"
    from forge.kernels.triton import gemm_w4a16

    assert not gemm_w4a16._FAST_BROKEN, gemm_w4a16._FAST_BROKEN


def test_multi_chunk_matmul_is_correct_and_bounds_the_temp_buffer(monkeypatch):
    n, k, m = 65536, 4096, 64
    monkeypatch.setenv("FORGE_W4_DEQUANT_MB", "64")  # 64 MiB of bf16 = 8192 rows of 4096 -> 8 chunks
    chunks = plan_dequant_chunks(n, k, 2, 64 << 20)
    assert len(chunks) == 8
    p = make(n, k, 128, False)
    dtype = torch.float16
    x = torch.randn(m, k, device=DEV, dtype=dtype)
    ref = w4a16_gemm_ref(x, p)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    y = run(x, p, impl="dequant_matmul")
    torch.cuda.synchronize()
    extra = torch.cuda.max_memory_allocated() - base
    assert compare(y, ref, TOL, out_dtype=dtype).ok
    out_bytes = m * n * 2
    assert extra <= (64 << 20) + out_bytes + (64 << 20) + (32 << 20), f"temporary memory {extra / 2**20:.0f} MiB exceeds the 64 MiB budget + output"
    full_weight = n * k * 2
    assert extra < full_weight / 2  # the dense weight was never materialised


def test_actorder_cuda_graph_capture_without_host_sync():
    """Prepared act-order weights: gemv (M=1) and dequant_matmul (M=64) capture and replay in a CUDA graph (a .cpu()/.item()
    inside the launch path would raise 'Cannot copy between CPU and CUDA tensors during CUDA graph capture')."""
    p = make(2048, 2048, 64, False, act_order=True)
    prepare_w4(p)
    for m in (1, 8, 64):
        x = torch.randn(m, 2048, device=DEV, dtype=torch.float16)
        ref = w4a16_gemm_ref(x, p)
        for _ in range(2):  # warm-up: compile outside capture
            run(x, p)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            y = run(x, p)
        g.replay()
        torch.cuda.synchronize()
        assert compare(y, ref, TOL, out_dtype=torch.float16).ok, m


def test_unprepared_actorder_capture_falls_back_instead_of_failing():
    p = make(1024, 2048, 64, False, act_order=True)  # NOT prepared: cold sort cache
    x = torch.randn(1, 2048, device=DEV, dtype=torch.float16)
    ref = w4a16_gemm_ref(x, p)
    run(x, p, impl="legacy")  # compile the fallback outside capture
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = run(x, p)
    g.replay()
    torch.cuda.synchronize()
    assert compare(y, ref, TOL, out_dtype=torch.float16).ok


def _graph_ms(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    g.replay()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters


def test_large_m_default_is_not_slower_than_legacy_speed_report():
    """Prints bf16 / legacy / default timings at large M; asserts only the floor the routing exists for: the default must not be
    slower than the legacy kernel (measured sm_120: dot was 3-7x slower than bf16 at M >= 64, legacy 3-6x)."""
    import torch.nn.functional as F  # noqa: N812

    chip = _chip()
    n, k = 5120, 5120
    p = make(n, k, 128, False)
    dense = p.dequantize(torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    for m in (32, 64, 256, 1024):
        x = torch.randn(m, k, device=DEV, dtype=dense.dtype)
        t_bf = _graph_ms(lambda x=x: F.linear(x, dense))
        t_leg = _graph_ms(lambda x=x: w4a16_gemm_triton(x, p, None, config={"impl": "legacy"}, sm=chip.sm, num_sms=chip.num_sms))
        t_new = _graph_ms(lambda x=x: w4a16_gemm_triton(x, p, None, sm=chip.sm, num_sms=chip.num_sms))
        print(f"M={m:<5} bf16 {t_bf * 1e3:8.1f}us  legacy {t_leg * 1e3:8.1f}us (x{t_bf / t_leg:.2f})  default {t_new * 1e3:8.1f}us (x{t_bf / t_new:.2f})")
        assert t_new <= 1.1 * t_leg, f"M={m}: default slower than legacy; run scripts/bench_w4a16.py --quick"
