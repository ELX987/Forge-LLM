"""GPU tests (auto-skipped without CUDA) for the issues found by the independent GEMM review.

NEVER run by the reviewer (no NVIDIA GPU available); run on the GPU host with  pytest tests_gpu -q -k gemm_review
Each test asserts the *Triton* path was taken (not a silent fallback) AND that the result matches the fp32 oracle.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
pytest.importorskip("triton")

from forge.kernels import ops_gemm  # noqa: E402
from forge.kernels.layouts import PORTABLE_W4, pack_w4, quantize_w4_rtn  # noqa: E402
from forge.kernels.ref import compare, fp8_gemm_ref, quantize_fp8_ref, w4a16_gemm_ref  # noqa: E402
from forge.kernels.spec import load_spec  # noqa: E402

DEV = "cuda"


@pytest.mark.parametrize("m", [1, 2, 8, 64])
@pytest.mark.parametrize("sym", [True, False])
def test_w4a16_decode_shapes_take_triton_path_without_fallback(m, sym):
    """M == 1 used to fail to compile the split-K reduce kernel; asymmetric 3-stage tiles exceeded 99 KiB smem on
    sm_86/89/120.  Neither may fall back to the reference path."""
    ops_gemm.reset_failure_memo()
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(4096, 4096, generator=g) * 0.05).to(DEV)
    q, s, z = quantize_w4_rtn(w, 128, sym=sym)
    p = pack_w4(q, s, z, 128, PORTABLE_W4)
    x = torch.randn(m, 4096, device=DEV, dtype=torch.float16)
    y = ops_gemm.w4a16_linear(x, p)
    assert ops_gemm.last_path("w4a16_linear") == "triton"
    assert not ops_gemm._FAILED
    res = compare(y, w4a16_gemm_ref(x, p), load_spec("w4a16_gemm").tolerance, out_dtype=torch.float16)
    assert res.ok, res.detail


@pytest.mark.parametrize("wg", ["per_tensor", "per_token", "block2d"])
@pytest.mark.parametrize("m", [1, 17])
def test_fp8_linear_all_weight_granularities(wg, m):
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native fp8 needs sm_89+")
    ops_gemm.reset_failure_memo()
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(512, 1024, generator=g) * 0.05).to(DEV)
    wq, sw = quantize_fp8_ref(w, wg)
    x = torch.randn(m, 1024, device=DEV, dtype=torch.bfloat16)
    y = ops_gemm.fp8_linear(x, wq, sw, out_dtype=torch.bfloat16)
    assert ops_gemm.last_path("fp8_linear") == "triton"
    xq, sx = ops_gemm.quantize_act_fp8(x, granularity="block" if wg == "block2d" else "per_token")
    ref = fp8_gemm_ref(xq, sx, wq, sw)
    res = compare(y, ref, load_spec("fp8_gemm").tolerance, out_dtype=torch.bfloat16)
    assert res.ok, res.detail


def test_fp8_hopper_accumulation_is_precise_for_large_k():
    """sm_90: Triton's default imprecise fp8 accumulation over K=16384 must not exceed the spec tolerance."""
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip("native fp8 needs sm_89+")
    g = torch.Generator().manual_seed(2)
    w = torch.rand(256, 16384, generator=g).to(DEV)  # same-sign data: worst case for reduced-precision accumulators
    x = torch.rand(8, 16384, generator=g).to(DEV).to(torch.bfloat16)
    wq, sw = quantize_fp8_ref(w, "per_token")
    y = ops_gemm.fp8_linear(x, wq, sw, out_dtype=torch.float32)
    xq, sx = ops_gemm.quantize_act_fp8(x)
    ref = fp8_gemm_ref(xq, sx, wq, sw)
    assert ops_gemm.last_path("fp8_linear") == "triton"
    rel = float((y - ref).abs().max() / ref.abs().max())
    assert rel < 2e-3, rel


def test_native_mxfp4_selftest_decides_path():
    """Reports (does not require) whether tl.dot_scaled passes the on-device self-test; the dispatcher must agree."""
    x = torch.zeros(1, device=DEV)
    avail = ops_gemm.native_fp4_available(x)
    key = (x.device.type, x.device.index)
    if ops_gemm._NATIVE_FP4_OK.get(key) is False:
        assert avail is False  # self-test failed -> never used
    print("native MXFP4 dot_scaled available:", avail)
