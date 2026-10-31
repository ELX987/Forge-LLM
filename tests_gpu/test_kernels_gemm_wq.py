"""GPU tests (auto-skipped without CUDA): W4A16 / W8A16 Triton kernels vs the fp32 oracle over the Plan 6.7 edge-case list.

These tests have NEVER been run by the developer (no NVIDIA GPU on the dev machine); they are the validation gate
for the user's vast.ai host:  scripts/run_gpu_tests.sh  or  pytest tests_gpu -q -k kernels_gemm
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
triton = pytest.importorskip("triton")

from forge.kernels import ops_gemm  # noqa: E402
from forge.kernels.layouts import (  # noqa: E402
    FRAGMENT_W4,
    PORTABLE_W4,
    pack_w4,
    pack_w8,
    quantize_w4_rtn,
    quantize_w8_rtn,
    repack,
    unpack_w4,
)
from forge.kernels.ref import compare, w4a16_gemm_ref, w8a16_gemm_ref  # noqa: E402
from forge.kernels.spec import load_spec  # noqa: E402

DEV = "cuda"
DTYPES = [torch.float16, torch.bfloat16]


def _supported(dtype):
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bf16 tensor cores need sm_80+")


def make_w4(n, k, gs, sym, seed=0, device=DEV):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.05).to(device)
    q, s, z = quantize_w4_rtn(w, gs, sym=sym)
    return pack_w4(q, s, z, gs, PORTABLE_W4)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [0, 1, 2, 7, 16, 33, 128, 256])
@pytest.mark.parametrize("n,k,gs", [(4096, 4096, 128), (1000, 4096, 64), (4096, 1000, 32), (777, 2056, 128), (256, 128, -1)])
@pytest.mark.parametrize("sym", [True, False])
def test_w4a16_matches_oracle_edge_cases(dtype, m, n, k, gs, sym):
    """K not a multiple of the group / tile, odd N, M in {0, 1..256}: within the spec tolerance of the fp32 oracle."""
    _supported(dtype)
    p = make_w4(n, k, gs, sym)
    x = torch.randn(m, k, device=DEV, dtype=dtype)
    bias = torch.randn(n, device=DEV, dtype=dtype)
    y = ops_gemm.w4a16_linear(x, p, bias)
    assert ops_gemm.last_path("w4a16_linear") == "triton"
    ref = w4a16_gemm_ref(x, p, bias)
    res = compare(y, ref, load_spec("w4a16_gemm").tolerance, out_dtype=dtype)
    assert res.ok, res.detail


@pytest.mark.parametrize("split_k", [1, 2, 4, 8])
def test_split_k_is_deterministic_and_correct(split_k):
    p = make_w4(2048, 4096, 128, True)
    x = torch.randn(1, 4096, device=DEV, dtype=torch.float16)
    cfg = {"tile_m": 16, "tile_n": 64, "tile_k": 128, "split_k": split_k, "stages": 3, "warps": 4}
    runs = [ops_gemm.w4a16_linear(x, p, config=cfg, deterministic=True) for _ in range(5)]
    assert all(torch.equal(runs[0], r) for r in runs[1:]), "deterministic split-K must be bit reproducible"
    assert compare(runs[0], w4a16_gemm_ref(x, p), load_spec("w4a16_gemm").tolerance, out_dtype=torch.float16).ok
    nd = ops_gemm.w4a16_linear(x, p, config=cfg, deterministic=False)  # atomics: may differ in the last bits only
    assert compare(nd, runs[0], load_spec("w4a16_gemm").tolerance, out_dtype=torch.float16).ok


def test_group_boundary_inside_tile_and_act_order():
    n, k, gs = 512, 1024, 32  # group (32) smaller than tile_k (128): several scales per K tile
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(n, k, generator=g) * 0.05).to(DEV)
    q, s, z = quantize_w4_rtn(w, gs, sym=False)
    g_idx = torch.randint(0, s.shape[1], (k,), dtype=torch.int32, device=DEV)
    p = pack_w4(q, s, z, gs, PORTABLE_W4, g_idx)
    x = torch.randn(4, k, device=DEV, dtype=torch.float16)
    y = ops_gemm.w4a16_linear(x, p, config={"tile_m": 16, "tile_n": 64, "tile_k": 128, "stages": 3, "warps": 4, "split_k": 1})
    assert compare(y, w4a16_gemm_ref(x, p), load_spec("w4a16_gemm").tolerance, out_dtype=torch.float16).ok


def test_channels_much_smaller_than_neighbours():
    """Plan 6.7: channels 10-30x smaller inside a tile must not be flushed or mis-scaled by the dequant path."""
    n, k = 256, 512
    w = torch.randn(n, k, device=DEV) * 0.05
    w[:, ::7] *= 1 / 20.0
    q, s, z = quantize_w4_rtn(w, 128, sym=True)
    p = pack_w4(q, s, z, 128, PORTABLE_W4)
    x = torch.randn(8, k, device=DEV, dtype=torch.float16)
    assert compare(ops_gemm.w4a16_linear(x, p), w4a16_gemm_ref(x, p), load_spec("w4a16_gemm").tolerance, out_dtype=torch.float16).ok


def test_nan_inf_in_activations_propagate_like_the_oracle():
    p = make_w4(128, 256, 64, True)
    x = torch.randn(4, 256, device=DEV, dtype=torch.float16)
    x[1, 3] = float("nan")
    x[2, 5] = float("inf")
    y = ops_gemm.w4a16_linear(x, p)
    ref = w4a16_gemm_ref(x, p)
    assert torch.isnan(y[1]).all() == torch.isnan(ref[1]).all()
    assert torch.isfinite(y[0]).all() and torch.isfinite(y[3]).all()
    assert not torch.isfinite(y[2]).any() or torch.isnan(y[2]).any() or torch.isinf(y[2]).any()


def test_fragment_layout_packed_on_gpu_equals_cpu_and_dispatcher_repacks():
    n, k = 300, 520
    w = torch.randn(n, k) * 0.05
    q, s, z = quantize_w4_rtn(w, 64, sym=False)
    cpu = pack_w4(q, s, z, 64, FRAGMENT_W4)
    gpu = pack_w4(q.to(DEV), s.to(DEV), z.to(DEV), 64, FRAGMENT_W4)
    assert torch.equal(cpu.qweight, gpu.qweight.cpu()), "pack must be device independent"
    assert all(torch.equal(a, b.cpu()) for a, b in zip(unpack_w4(cpu), unpack_w4(gpu)))
    x = torch.randn(3, k, device=DEV, dtype=torch.float16)
    y_frag = ops_gemm.w4a16_linear(x, gpu)
    y_port = ops_gemm.w4a16_linear(x, repack(gpu, PORTABLE_W4))
    assert torch.equal(y_frag, y_port)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("m", [0, 1, 16, 64, 300])
@pytest.mark.parametrize("n,k,gs", [(4096, 4096, -1), (1001, 2056, 128), (512, 777, 32)])
def test_w8a16_matches_oracle(dtype, m, n, k, gs):
    _supported(dtype)
    w = (torch.randn(n, k) * 0.05).to(DEV)
    q, s = quantize_w8_rtn(w, gs)
    p = pack_w8(q, s, gs)
    x = torch.randn(m, k, device=DEV, dtype=dtype)
    bias = torch.randn(n, device=DEV, dtype=dtype)
    y = ops_gemm.w8a16_linear(x, p, bias)
    assert ops_gemm.last_path("w8a16_linear") == "triton"
    res = compare(y, w8a16_gemm_ref(x, p, bias), load_spec("w8a16_gemm").tolerance, out_dtype=dtype)
    assert res.ok, res.detail
    assert compare(ops_gemm.w8a16_linear(x, p), w8a16_gemm_ref(x, p), load_spec("w8a16_gemm").tolerance, out_dtype=dtype).ok


def test_decode_speedup_vs_bf16_is_reported_not_asserted():
    """KPI (C-043): 3.2x vs BF16 at M=1..16 is a TARGET.  Print the measured number; never fail on it."""
    from forge.kernels.autotune.timing import summarize, time_callable

    n = k = 8192
    p = make_w4(n, k, 128, True)
    dense = p.dequantize(torch.float16).to(DEV)
    x = torch.randn(1, k, device=DEV, dtype=torch.float16)
    t4 = summarize(time_callable(lambda: ops_gemm.w4a16_linear(x, p), warmup=10, repeats=30, flush=None)).median
    t16 = summarize(time_callable(lambda: torch.nn.functional.linear(x, dense), warmup=10, repeats=30, flush=None)).median
    print(f"\n[KPI] w4a16 M=1 {n}x{k}: {t4:.4f} ms vs fp16 {t16:.4f} ms -> {t16 / t4:.2f}x (target >= 3.2x)")
    assert t4 > 0
