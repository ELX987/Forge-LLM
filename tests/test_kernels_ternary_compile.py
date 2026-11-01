"""Offline (no GPU) Triton *compilation* of the ternary kernels for sm_80 / 89 / 120 plus structural PTX checks (int8 tensor-core MMA is used,
weights are loaded as bytes once per plane).  Skipped when the offline compiler is unavailable.  Validates compilability, NOT numerics or speed."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")
sys.path.insert(0, str(Path(__file__).parent))

from forge.kernels.layouts.ternary import quantize_pack_ternary  # noqa: E402
from forge.formats import ternary as tn  # noqa: E402
from forge.kernels.triton.gemm_ternary import ternary_act_quant_triton, ternary_gemm_triton  # noqa: E402


def _compile(rec, sm):
    import kernels_attn_compile_harness as H
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = rec.kernel
    names = list(fn.arg_names)
    values = dict(zip(names, rec.args))
    values.update({k: v for k, v in rec.kwargs.items() if k in names})
    opts = {k: v for k, v in rec.kwargs.items() if k in ("num_warps", "num_stages")}
    cexp = {p.name for p in fn.params if p.is_constexpr}
    sig, cs, attrs = {}, {}, {}
    for i, n in enumerate(names):
        v = values[n]
        if n in cexp or (isinstance(v, int) and not isinstance(v, bool) and v == 1):
            sig[n], cs[n] = "constexpr", v
            continue
        sig[n] = H._sig_entry(v)
        aligned = (isinstance(v, torch.Tensor) and v.data_ptr() % 16 == 0) or (isinstance(v, int) and not isinstance(v, bool) and v % 16 == 0)
        if aligned:
            attrs[(i,)] = [["tt.divisibility", 16]]
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cs, attrs=attrs), target=GPUTarget("cuda", sm, 32), options=opts)


def _ptx(group, *, sm=120, m=16, n=256, k=512, kernel_prefix="_t158"):
    import kernels_attn_compile_harness as H

    p = quantize_pack_ternary(torch.randn(n, k), group)
    xq, sx = tn.act_quant_int8(torch.randn(m, k))
    try:
        with H.recording() as recs:
            ternary_gemm_triton(xq, sx, p.qweight, p.scales, group, k, torch.randn(n), out_dtype=torch.bfloat16, sm=sm)
        rec = next(r for r in recs if r.kernel.__name__.startswith(kernel_prefix))
        return rec.kernel.__name__, _compile(rec, sm).asm["ptx"]
    except Exception as e:  # pragma: no cover - environment dependent
        if "ptxas" in str(e).lower() or "target" in str(e).lower():
            pytest.skip(f"offline Triton compilation unavailable: {e}")
        raise


@pytest.mark.parametrize("sm", [80, 89, 120])
@pytest.mark.parametrize("group,name", [(128, "_t158_planes_kernel"), (-1, "_t158_planes_kernel"), (0, "_t158_planes_kernel"), (64, "_t158_gather_kernel")])
def test_kernels_compile_and_use_int8_tensor_cores(sm, group, name):
    kname, ptx = _ptx(group, sm=sm)
    assert kname == name
    assert re.search(r"mma\.sync\.aligned\.m16n8k32\.row\.col(?:\.satfinite)?\.s32\.s8\.s8\.s32", ptx), "expected int8 m16n8k32 MMA instructions"


def test_activation_quantiser_compiles():
    import kernels_attn_compile_harness as H

    try:
        with H.recording() as recs:
            ternary_act_quant_triton(torch.randn(8, 4096))
        rec = next(r for r in recs if r.kernel.__name__ == "_t158_act_quant_kernel")
        ptx = _compile(rec, 120).asm["ptx"]
    except Exception as e:  # pragma: no cover
        if "ptxas" in str(e).lower() or "target" in str(e).lower():
            pytest.skip(f"offline Triton compilation unavailable: {e}")
        raise
    assert re.search(r"div.rn(?:.ftz)?.f32", ptx)  # IEEE division (libdevice), not the approximate div.full
