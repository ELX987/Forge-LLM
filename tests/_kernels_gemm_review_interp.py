"""Review regression checks run in the Triton CPU interpreter (subprocess; TRITON_INTERPRET must precede the triton import).

Covers bugs found in review: fp8 scale-granularity combinations that read scales out of bounds, non-contiguous INT8
weights, FP4 NaN-poisoned scale blocks, malformed FP4 containers.  Prints ``REPORT_JSON=...``.
"""
from __future__ import annotations

import json
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from forge.kernels.ref import (  # noqa: E402
    fp4_block_gemm_ref,
    fp8_gemm_ref,
    int8_gemm_ref,
    quantize_fp4_ref,
    quantize_fp8_ref,
    quantize_int8_ref,
)
from forge.kernels.triton.gemm_fp4 import fp4_dequant_gemm_triton, quantize_act_fp4_triton  # noqa: E402
from forge.kernels.triton.gemm_fp8 import fp8_gemm_triton  # noqa: E402
from forge.kernels.triton.gemm_int8 import int8_gemm_triton  # noqa: E402

torch.manual_seed(3)
CFG = dict(tile_m=16, tile_n=32, tile_k=32, stages=2, warps=4)
rep: dict = {}


def err(a, b):
    return float((a.float() - b.float()).abs().max())


n, k = 40, 200
w = torch.randn(n, k)
wgrans = {"per_tensor": "per_tensor", "per_channel": "per_token", "block2d": "block2d"}
res = []
for m in (1, 5):
    x = torch.randn(m, k)
    for xg in ("per_tensor", "per_token", "block"):
        xq, sx = quantize_fp8_ref(x, xg)
        for wname, wg in wgrans.items():
            wq, sw = quantize_fp8_ref(w, wg)
            ref = fp8_gemm_ref(xq, sx, wq, sw)
            try:
                y = fp8_gemm_triton(xq, sx, wq, sw, None, out_dtype=torch.float32, config=CFG)
                e = err(y, ref)
            except Exception as ex:  # noqa: BLE001
                e = float("inf")
                wname = f"{wname}!{type(ex).__name__}"
            res.append(dict(m=m, x=xg, w=wname, err=e, scale=float(ref.abs().max())))
rep["fp8_combos"] = res

# bias + per-token x per-channel, K not a multiple of 16
xq, sx = quantize_fp8_ref(torch.randn(3, k), "per_token")
wq, sw = quantize_fp8_ref(w, "per_token")
b = torch.randn(n)
rep["fp8_bias"] = err(fp8_gemm_triton(xq, sx, wq, sw, b, out_dtype=torch.float32, config=CFG), fp8_gemm_ref(xq, sx, wq, sw, b))

# mismatched scale sizes must raise (never read out of bounds)
try:
    fp8_gemm_triton(xq, sx[:2], wq, sw, None, out_dtype=torch.float32, config=CFG)
    rep["fp8_bad_scale_raises"] = False
except ValueError:
    rep["fp8_bad_scale_raises"] = True

# INT8 with a non-contiguous (transposed view) weight
xi, sxi = quantize_int8_ref(torch.randn(6, 64), "per_token")
wi, swi = quantize_int8_ref(torch.randn(48, 64), "per_channel")
wt = wi.t().contiguous().t()  # same values, column-major strides
assert not wt.is_contiguous() and torch.equal(wt, wi)
rep["int8_noncontig_err"] = err(int8_gemm_triton(xi, sxi, wt, swi, None, out_dtype=torch.float32, config=CFG),
                                int8_gemm_ref(xi, sxi, wi, swi))
try:
    int8_gemm_triton(xi, sxi[:2], wi, swi, None, out_dtype=torch.float32)
    rep["int8_bad_scale_raises"] = False
except ValueError:
    rep["int8_bad_scale_raises"] = True

# FP4: a NaN in a weight block poisons the block's scale; the kernel must reproduce the oracle's NaNs
fp4 = {}
for fmt in ("nvfp4", "mxfp4"):
    ww = torch.randn(24, 96)
    ww[3, 5] = float("nan")
    wp = quantize_fp4_ref(ww, fmt)
    x = torch.randn(4, 96).half()
    ref = fp4_block_gemm_ref(x, wp)
    y = fp4_dequant_gemm_triton(x, wp, None, config=dict(tile_m=16, tile_n=32, tile_k=32, stages=2, warps=4)).float()
    fp4[fmt] = dict(ref_nan_cols=torch.isnan(ref).any(0).nonzero().flatten().tolist(),
                    out_nan_cols=torch.isnan(y).any(0).nonzero().flatten().tolist())
    bad = wp.with_scale_layout("logical")
    broken = type(bad)(bad.fmt, bad.payload[:, :-1], bad.scales, bad.tensor_scale, bad.rows, bad.k, bad.scale_layout)
    try:
        fp4_dequant_gemm_triton(x, broken, None)
        fp4[fmt]["malformed_raises"] = False
    except ValueError:
        fp4[fmt]["malformed_raises"] = True
rep["fp4_nan"] = fp4

# fused FP4 activation quantiser: NaN / Inf inputs must poison their block exactly like the oracle (not silently become numbers)
q4 = {}
for fmt in ("nvfp4", "mxfp4"):
    xa = torch.randn(7, 70)
    xa[2, 3] = float("nan")
    xa[5, 40] = float("inf")
    for lay in ("logical", "swizzled"):
        a = quantize_act_fp4_triton(xa, fmt, scale_layout=lay)
        r = quantize_fp4_ref(xa, fmt, scale_layout=lay)
        q4[f"{fmt}_{lay}"] = dict(scales_equal=bool(torch.equal(a.scales, r.scales)), payload_equal=bool(torch.equal(a.payload, r.payload)),
                                  ts_equal=bool(torch.equal(a.tensor_scale, r.tensor_scale)))
rep["fp4_quant_nonfinite"] = q4

print("REPORT_JSON=" + json.dumps(rep))
