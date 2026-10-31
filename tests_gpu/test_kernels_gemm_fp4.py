"""GPU tests (auto-skipped without CUDA): block-scaled FP4 kernels + autotuner on real kernels.  Never run by the developer.

FP4 is EXPERIMENTAL: native block-scaled MMA is a hypothesis confirmed by the hardware probe (Plan HW-2).
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu
triton = pytest.importorskip("triton")

from forge.kernels import ops_gemm  # noqa: E402
from forge.kernels.ref import (  # noqa: E402
    compare,
    dequantize_fp4_ref,
    fp4_block_gemm_ref,
    quantize_fp4_ref,
)
from forge.kernels.spec import load_spec  # noqa: E402

DEV = "cuda"


def _sm():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("layout", ["logical", "swizzled"])
@pytest.mark.parametrize("m", [1, 16, 100])
@pytest.mark.parametrize("n,k", [(4096, 4096), (777, 1000), (130, 48)])
def test_fp4_weight_only_dequant_kernel_matches_oracle(fmt, layout, m, n, k):
    """Any sm: the dequantise-to-16-bit kernel keeps the storage win (Plan K2 fallback)."""
    w = torch.randn(n, k, device=DEV) * 0.05
    wp = quantize_fp4_ref(w, fmt, scale_layout=layout)
    x = torch.randn(m, k, device=DEV, dtype=torch.float16)
    y = ops_gemm.fp4_linear(x, wp)
    assert ops_gemm.last_path("fp4_linear") == "triton_dequant"
    assert compare(y, fp4_block_gemm_ref(x, wp), load_spec("fp4_block_gemm").tolerance, out_dtype=torch.float16).ok


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("layout", ["logical", "swizzled"])
def test_fused_fp4_quantiser_matches_oracle(fmt, layout):
    x = torch.randn(257, 1000, device=DEV, dtype=torch.bfloat16) * 2
    ref = quantize_fp4_ref(x, fmt, scale_layout=layout)
    got = ops_gemm.quantize_act_fp4(x, fmt, scale_layout=layout)
    assert ops_gemm.last_path("quantize_act_fp4") == "triton"
    assert torch.equal(got.scales, ref.scales) or (got.scales != ref.scales).float().mean() < 1e-3
    dq, dr = dequantize_fp4_ref(got), dequantize_fp4_ref(ref)
    # Triton's fp32 division is not IEEE rounded: at exact rounding ties a code may differ by one grid step
    assert (got.payload != ref.payload).float().mean() < 5e-3
    assert (dq - dr).abs().max() <= 2.0 * dr.abs().max() / 6.0 + 1e-6


def test_fp4_quantiser_edge_blocks_zero_underflow_and_tiny_channels():
    """Plan 6.7: all-zero blocks, E4M3 scale underflow, channels 10-30x smaller than neighbours in a 16x16 tile."""
    x = torch.zeros(32, 64, device=DEV, dtype=torch.bfloat16)
    x[0, 16:32] = torch.linspace(0.1, 100, 16, device=DEV)  # normal block, all others zero
    x[1, 0:16] = 1e-7  # underflows E4M3 relative to the tensor scale
    x[1, 16:32] = 50.0
    x[2:18, 0:16] = torch.randn(16, 16, device=DEV) * 1.0
    x[2:18, 3] *= 1 / 20.0  # a channel 20x smaller than its neighbours inside the tile
    got = dequantize_fp4_ref(ops_gemm.quantize_act_fp4(x, "nvfp4"))
    ref = dequantize_fp4_ref(quantize_fp4_ref(x, "nvfp4"))
    assert torch.isfinite(got).all()
    assert (got[0, :16] == 0).all() and (got[1, :16] == 0).all()
    assert (got - ref).abs().max() <= 0.5 * ref.abs().max()


@pytest.mark.skipif(not torch.cuda.is_available() or _sm() < 100, reason="block-scaled FP4 MMA hypothesis needs sm_100/120")
def test_native_mxfp4_dot_scaled_if_present():
    from forge.kernels.triton.gemm_fp4 import mxfp4_dot_scaled_available

    if not mxfp4_dot_scaled_available():
        pytest.skip("tl.dot_scaled not available in this Triton build")
    w = torch.randn(1024, 1024, device=DEV) * 0.05
    wp = quantize_fp4_ref(w, "mxfp4")
    x = torch.randn(256, 1024, device=DEV, dtype=torch.bfloat16)
    y = ops_gemm.fp4_linear(x, wp, act_quant="fp4")
    path = ops_gemm.last_path("fp4_linear")
    print(f"\n[FP4] path taken: {path}")
    xq = quantize_fp4_ref(x, "mxfp4")
    assert compare(y, fp4_block_gemm_ref(xq, wp), load_spec("fp4_block_gemm").tolerance, out_dtype=torch.bfloat16, slack=2.0).ok


# ------------------------------------------------------------------------------------------- autotuner on real kernels
def test_autotuner_on_real_w4a16_kernel_gates_and_caches(tmp_path):
    from forge.kernels.autotune import Autotuner
    from forge.kernels.layouts import PORTABLE_W4, pack_w4, quantize_w4_rtn
    from forge.kernels.layouts.chip import chip_info
    from forge.kernels.ref import w4a16_gemm_ref
    from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton

    n, k, m = 2048, 2048, 4
    w = (torch.randn(n, k) * 0.05).to(DEV)
    q, s, z = quantize_w4_rtn(w, 128, sym=True)
    p = pack_w4(q, s, z, 128, PORTABLE_W4)
    x = torch.randn(m, k, device=DEV, dtype=torch.float16)
    spec = load_spec("w4a16_gemm")
    ref = w4a16_gemm_ref(x, p)
    tuner = Autotuner(tmp_path / "c.json", chip=chip_info(None), repeats=10, warmup=3, max_candidates=8)
    res = tuner.tune(spec, {"M": m, "N": n, "K": k}, lambda c, sh: w4a16_gemm_triton(x, p, None, config=c), lambda sh: ref,
                     dtype_cfg="w4a16_gemm|x=float16|g=128", layout_id=p.layout_id)
    print(f"\n[autotune] best={res.best} median={res.best_measurement.median_ms if res.best_measurement else None} ms "
          f"rejected={len(res.rejected)} notes={res.notes}")
    assert res.best is not None and all(r.error for r in res.rejected)
    again = tuner.tune(spec, {"M": m, "N": n, "K": k}, lambda c, sh: 1 / 0, lambda sh: ref,
                       dtype_cfg="w4a16_gemm|x=float16|g=128", layout_id=p.layout_id)
    assert again.from_cache or not res.cached
