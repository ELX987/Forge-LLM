"""Runs the GEMM Triton kernels in the Triton *CPU interpreter* (TRITON_INTERPRET=1) and prints a JSON report.

Executed in a subprocess by tests/test_kernels_gemm_triton_interp.py (the interpreter switch must be set before
triton is imported).  This validates the kernel LOGIC (indexing, masking, packing conventions, split-K, scale
application) against the torch oracles -- it is NOT a GPU compile/performance test.  Known interpreter limits:
bf16 ``tl.dot`` and the fp32->fp8 cast are not faithfully emulated, so those parts are exercised on fp16 / compared loosely.
"""
from __future__ import annotations

import json
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from forge.kernels import ops_gemm  # noqa: E402
from forge.kernels.layouts import (  # noqa: E402
    FRAGMENT_W4,
    pack_w4,
    pack_w8,
    quantize_w4_rtn,
    quantize_w8_rtn,
)
from forge.kernels.ref import (  # noqa: E402
    dequantize_fp4_ref,
    fp4_block_gemm_ref,
    fp8_gemm_ref,
    int8_gemm_ref,
    quantize_fp4_ref,
    quantize_fp8_ref,
    quantize_int8_ref,
    w4a16_gemm_ref,
    w8a16_gemm_ref,
)
from forge.kernels.triton.gemm_fp4 import (  # noqa: E402
    fp4_dequant_gemm_triton,
    mxfp4_dot_scaled_available,
    mxfp4_gemm_native_triton,
    quantize_act_fp4_triton,
)
from forge.kernels.triton.gemm_fp8 import act_quant_fp8_triton, fp8_gemm_triton  # noqa: E402
from forge.kernels.triton.gemm_int8 import int8_gemm_triton, int8_quant_triton  # noqa: E402
from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton  # noqa: E402
from forge.kernels.triton.gemm_w8a16 import w8a16_gemm_triton  # noqa: E402

report: dict = {}
torch.manual_seed(0)
CFG = dict(tile_m=16, tile_n=32, tile_k=32)


def err(a, b):
    return float((a.float() - b.float()).abs().max())


# ---- W4A16: groups, zeros, g_idx, per-channel, split-K (deterministic and atomic), odd N, K not multiple of group/tile
cases = [
    dict(m=3, n=40, k=100, gs=32, sym=False, gidx=False),
    dict(m=17, n=70, k=64, gs=-1, sym=True, gidx=False),
    dict(m=1, n=33, k=128, gs=64, sym=True, gidx=True),
    dict(m=20, n=37, k=96, gs=32, sym=False, gidx=False),
]
w4 = []
for c in cases:
    w = torch.randn(c["n"], c["k"])
    q, s, z = quantize_w4_rtn(w, c["gs"], sym=c["sym"])
    gi = torch.randint(0, s.shape[1], (c["k"],), dtype=torch.int32) if c["gidx"] else None
    p = pack_w4(q, s, z, c["gs"], None, gi)
    x = torch.randn(c["m"], c["k"]).half()
    b = torch.randn(c["n"]).half()
    ref = w4a16_gemm_ref(x, p, b)
    for sk in (1, 3):
        for det in (True, False):
            y = w4a16_gemm_triton(x, p, b, config={**CFG, "split_k": sk}, deterministic=det)
            w4.append(dict(case=c, split_k=sk, det=det, err=err(y, ref), scale=float(ref.abs().max())))
    # determinism: two deterministic split-K launches are bit identical
    y1 = w4a16_gemm_triton(x, p, b, config={**CFG, "split_k": 3}, deterministic=True)
    y2 = w4a16_gemm_triton(x, p, b, config={**CFG, "split_k": 3}, deterministic=True)
    w4.append(dict(case=c, bit_identical=bool(torch.equal(y1, y2))))
report["w4a16"] = w4

# fragment layout input gets repacked through the dispatcher
w = torch.randn(48, 96)
q, s, z = quantize_w4_rtn(w, 32, sym=False)
pf = pack_w4(q, s, z, 32, FRAGMENT_W4)
x = torch.randn(4, 96).half()
y = ops_gemm.w4a16_linear(x, pf, config=CFG)
report["dispatch_w4a16"] = dict(path=ops_gemm.last_path("w4a16_linear"), err=err(y, w4a16_gemm_ref(x, pf)))

# ---- W8A16
w8 = []
for gs in (-1, 32):
    w = torch.randn(50, 70)
    q, s = quantize_w8_rtn(w, gs)
    p8 = pack_w8(q, s, gs)
    x = torch.randn(5, 70).half()
    ref = w8a16_gemm_ref(x, p8)
    for sk in (1, 2):
        y = w8a16_gemm_triton(x, p8, torch.ones(50).half(), config={**CFG, "split_k": sk})
        w8.append(dict(gs=gs, split_k=sk, err=err(y, ref + 1), scale=float(ref.abs().max())))
report["w8a16"] = w8

# ---- INT8 (bit-exact integer path)
x = torch.randn(9, 150) * 2
w = torch.randn(41, 150)
q, s = int8_quant_triton(x)
qr, sr = quantize_int8_ref(x)
wq, sw = quantize_int8_ref(w, "per_channel")
bias = torch.randn(41)
y = int8_gemm_triton(qr, sr, wq, sw, bias, out_dtype=torch.float32, config=CFG)
report["int8"] = dict(quant_code_mismatch=int((q != qr).sum()), quant_scale_err=err(s, sr),
                      gemm_err=err(y, int8_gemm_ref(qr, sr, wq, sw, bias)))

# ---- FP8 GEMM (scale modes) + activation quant scales (fp32->fp8 cast not faithfully emulated: codes not compared)
x = torch.randn(5, 200) * 3
w = torch.randn(70, 200)
f8 = {}
for gx, gw in (("per_tensor", "per_tensor"), ("per_token", "per_channel"), ("block", "block2d")):
    xq, sx = quantize_fp8_ref(x, gx)
    wq, sw = quantize_fp8_ref(w, gw)
    y = fp8_gemm_triton(xq, sx, wq, sw, torch.ones(70), out_dtype=torch.float32, config=CFG)
    f8[gx] = dict(err=err(y, fp8_gemm_ref(xq, sx, wq, sw, torch.ones(70))), scale=float(y.abs().max()))
qa, sa = act_quant_fp8_triton(x, granularity="per_token")
f8["quant_scale_err"] = err(sa, quantize_fp8_ref(x, "per_token")[1])
wt = torch.rand(200) + 0.5
_, sr2 = act_quant_fp8_triton(x, rms_weight=wt, rms_eps=1e-6)
from forge.kernels.ref import rmsnorm_act_quant_fp8_ref  # noqa: E402

f8["rms_scale_err"] = err(sr2, rmsnorm_act_quant_fp8_ref(x, wt, 1e-6)[1])
_, sb = act_quant_fp8_triton(x, granularity="block")
f8["block_scale_err"] = err(sb, quantize_fp8_ref(x, "block")[1])
report["fp8"] = f8

# ---- FP4
x = torch.randn(130, 100) * 2
w = torch.randn(70, 100)
xa = torch.randn(5, 100)
f4 = {}
for fmt in ("mxfp4", "nvfp4"):
    for lay in ("logical", "swizzled"):
        ref = quantize_fp4_ref(x, fmt, scale_layout=lay)
        got = quantize_act_fp4_triton(x, fmt, scale_layout=lay)
        wp = quantize_fp4_ref(w, fmt, scale_layout=lay)
        y = fp4_dequant_gemm_triton(xa.half(), wp, torch.ones(70).half(), config=CFG)
        f4[f"{fmt}_{lay}"] = dict(
            payload_mismatch=int((got.payload != ref.payload).sum()), payload_total=int(ref.payload.numel()),
            scale_mismatch=int((got.scales != ref.scales).sum()), scale_total=int(ref.scales.numel()),
            gemm_err=err(y, fp4_block_gemm_ref(xa, wp, torch.ones(70))), gemm_scale=float(y.abs().max()),
        )
f4["dot_scaled_available"] = bool(mxfp4_dot_scaled_available())
if mxfp4_dot_scaled_available():
    try:
        wp = quantize_fp4_ref(w, "mxfp4")
        xq = quantize_fp4_ref(xa, "mxfp4")
        y = mxfp4_gemm_native_triton(xq, wp, out_dtype=torch.float32, tile=(16, 32, 32))
        f4["native_dot_scaled_err"] = err(y, fp4_block_gemm_ref(xq, wp))
    except Exception as e:  # interpreter support for dot_scaled varies by version
        f4["native_dot_scaled_error"] = f"{type(e).__name__}: {e}"[:300]
report["fp4"] = f4

print("REPORT_JSON=" + json.dumps(report))
