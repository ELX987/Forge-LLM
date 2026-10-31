"""GPU tests (auto-skipped without CUDA): FP8 GEMM + fused activation quantiser, INT8 W8A8.  Never run by the developer."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
triton = pytest.importorskip("triton")

from forge.kernels import ops_gemm  # noqa: E402
from forge.kernels.ref import (  # noqa: E402
    compare,
    dequantize_fp8_ref,
    fp8_gemm_ref,
    int8_gemm_ref,
    int8_quant_agreement,
    quantize_fp8_ref,
    quantize_int8_ref,
    rmsnorm_act_quant_fp8_ref,
)
from forge.kernels.spec import load_spec  # noqa: E402

DEV = "cuda"


def _sm():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


needs_fp8 = pytest.mark.skipif(not torch.cuda.is_available() or _sm() < 89, reason="native FP8 MMA needs sm_89+")


@needs_fp8
@pytest.mark.parametrize("gx,gw", [("per_tensor", "per_tensor"), ("per_token", "per_channel"), ("block", "block2d")])
@pytest.mark.parametrize("m", [0, 1, 7, 64, 300])
@pytest.mark.parametrize("n,k", [(4096, 4096), (1000, 2056), (256, 777)])
def test_fp8_gemm_native_matches_oracle(gx, gw, m, n, k):
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=DEV) * 0.05
    wq, sw = quantize_fp8_ref(w, gw)
    xq, sx = quantize_fp8_ref(x, gx)
    y = ops_gemm.fp8_linear(xq, wq, sw, x_scale=sx, out_dtype=torch.bfloat16)
    assert ops_gemm.last_path("fp8_linear") == "triton"
    res = compare(y, fp8_gemm_ref(xq, sx, wq, sw), load_spec("fp8_gemm").tolerance, out_dtype=torch.bfloat16)
    assert res.ok, res.detail


@needs_fp8
def test_fp8_gemm_is_run_to_run_bit_reproducible():
    x = torch.randn(256, 4096, device=DEV, dtype=torch.bfloat16)
    wq, sw = quantize_fp8_ref(torch.randn(4096, 4096, device=DEV) * 0.05, "per_channel")
    outs = [ops_gemm.fp8_linear(x, wq, sw) for _ in range(5)]
    assert all(torch.equal(outs[0], o) for o in outs[1:])


@needs_fp8
def test_fp8_gemm_vs_vendor_scaled_mm_if_available():
    if not hasattr(torch, "_scaled_mm"):
        pytest.skip("torch._scaled_mm missing")
    m, n, k = 128, 4096, 4096
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16)
    wq, sw = quantize_fp8_ref(torch.randn(n, k, device=DEV) * 0.05, "per_tensor")
    xq, sx = quantize_fp8_ref(x, "per_tensor")
    mine = ops_gemm.fp8_linear(xq, wq, sw, x_scale=sx, out_dtype=torch.bfloat16)
    try:
        vendor = torch._scaled_mm(xq, wq.t(), scale_a=sx.reshape(1, 1), scale_b=sw.reshape(1, 1), out_dtype=torch.bfloat16)
        vendor = vendor[0] if isinstance(vendor, tuple) else vendor
    except Exception as e:
        pytest.skip(f"_scaled_mm signature differs on this torch: {e}")
    assert compare(mine, vendor, load_spec("fp8_gemm").tolerance, out_dtype=torch.bfloat16).ok


@needs_fp8
@pytest.mark.parametrize("k", [64, 1000, 4096, 12288])
def test_fused_rmsnorm_fp8_quant_matches_oracle(k):
    x = torch.randn(33, k, device=DEV, dtype=torch.bfloat16) * 3
    w = torch.rand(k, device=DEV, dtype=torch.bfloat16) + 0.5
    q, s = ops_gemm.quantize_act_fp8(x, rms_weight=w, rms_eps=1e-6)
    assert ops_gemm.last_path("quantize_act_fp8") == "triton"
    qr, sr, _ = rmsnorm_act_quant_fp8_ref(x, w, 1e-6)
    assert torch.allclose(s, sr, rtol=1e-5)
    d, dr = dequantize_fp8_ref(q, s), dequantize_fp8_ref(qr, sr)
    # fp32 division is not IEEE-rounded in Triton: codes may differ by one at rounding boundaries (<= 1 fp8 ulp)
    assert (d - dr).abs().max() <= (dr.abs() * 0.07 + 1e-3 * sr.max()).max()
    assert (q.view(torch.uint8) != qr.view(torch.uint8)).float().mean() < 0.01


@needs_fp8
def test_activation_quant_edge_cases():
    x = torch.zeros(4, 256, device=DEV, dtype=torch.bfloat16)  # all-zero rows
    x[1] = 1e4  # large values -> scale, no overflow
    x[2, 0] = 1e-6
    q, s = ops_gemm.quantize_act_fp8(x)
    assert torch.isfinite(q.float()).all() and s[0].item() == 1.0 and q[0].float().abs().max() == 0
    assert q[1].float().abs().max().item() == 448.0
    xn = torch.randn(2, 128, device=DEV, dtype=torch.bfloat16)
    xn[0, 5] = float("nan")
    qn, sn = ops_gemm.quantize_act_fp8(xn)
    assert qn.shape == xn.shape and torch.isfinite(qn[1].float()).all()  # NaN row: E4M3FN has a NaN code; the clean row is unaffected


needs_int8 = pytest.mark.skipif(not torch.cuda.is_available() or _sm() < 75, reason="int8 tensor cores need sm_75+")


@needs_int8
@pytest.mark.parametrize("m", [0, 1, 16, 100, 513])
@pytest.mark.parametrize("n,k", [(4096, 4096), (777, 1000)])
def test_int8_gemm_is_exact_given_identical_codes_and_scales(m, n, k):
    """The GEMM is integer-exact: feed the oracle's codes/scales to the kernel (independent of quantiser rounding)."""
    from forge.kernels.triton.gemm_int8 import int8_gemm_triton

    x = torch.randn(m, k, device=DEV, dtype=torch.float16)
    wq, sw = quantize_int8_ref(torch.randn(n, k, device=DEV) * 0.05, "per_channel")
    xr, sr = quantize_int8_ref(x, "per_token")
    y = int8_gemm_triton(xr, sr, wq, sw, None, out_dtype=torch.float32)
    assert torch.equal(y, int8_gemm_ref(xr, sr, wq, sw)), "int32 accumulation + fp32 epilogue must be exact"


@needs_int8
@pytest.mark.parametrize("m", [0, 1, 16, 100, 513])
@pytest.mark.parametrize("k", [1000, 4096])
def test_int8_activation_quantiser_agrees_with_oracle(m, k):
    """Scales within 1 ulp; codes equal except (at most) at exact rounding ties."""
    x = torch.randn(m, k, device=DEV, dtype=torch.float16)
    xq, sx = ops_gemm.quantize_act_int8(x)
    assert ops_gemm.last_path("quantize_act_int8") == "triton"
    xr, sr = quantize_int8_ref(x, "per_token")
    st = int8_quant_agreement(x, xq, sx, xr, sr)
    print(f"int8 quantiser agreement m={m} k={k}: {st}")
    assert st["scale_ulp_max"] <= 1.0, st
    assert st["untied_mismatches"] == 0, st
    assert st["max_code_diff"] <= 1, st


@needs_int8
@pytest.mark.parametrize("m", [0, 1, 16, 100, 513])
@pytest.mark.parametrize("n,k", [(4096, 4096), (777, 1000)])
def test_int8_linear_end_to_end_within_tolerance(m, n, k):
    x = torch.randn(m, k, device=DEV, dtype=torch.float16)
    wq, sw = quantize_int8_ref(torch.randn(n, k, device=DEV) * 0.05, "per_channel")
    xr, sr = quantize_int8_ref(x, "per_token")
    y16 = ops_gemm.int8_linear(x, wq, sw)
    assert compare(y16, int8_gemm_ref(xr, sr, wq, sw), load_spec("int8_w8a8_gemm").tolerance, out_dtype=torch.float16).ok


def test_fp8_fallback_path_on_pre_ada_gpus():
    if _sm() >= 89:
        pytest.skip("this GPU has native FP8")
    w = torch.randn(256, 512, device=DEV) * 0.05
    wq, sw = quantize_fp8_ref(w, "per_channel")
    x = torch.randn(8, 512, device=DEV, dtype=torch.float16)
    y = ops_gemm.fp8_linear(x, wq, sw)
    assert ops_gemm.last_path("fp8_linear") == "dequant_matmul"
    xq, sx = quantize_fp8_ref(x, "per_token")
    assert compare(y, fp8_gemm_ref(xq, sx, wq, sw), load_spec("fp8_gemm").tolerance, out_dtype=torch.float16, slack=1.5).ok
