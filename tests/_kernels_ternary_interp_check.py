"""Subprocess helper: run the ternary Triton kernels in the CPU interpreter and compare with the torch oracle (prints REPORT_JSON=...)."""
from __future__ import annotations

import json
import os

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402

from forge.formats import ternary as tn  # noqa: E402
from forge.kernels.layouts.ternary import quantize_pack_ternary  # noqa: E402
from forge.kernels.ref.gemm_ternary import ternary_gemm_ref  # noqa: E402
from forge.kernels.triton.gemm_ternary import ternary_act_quant_triton, ternary_gemm_triton  # noqa: E402

g = torch.Generator().manual_seed(0)
rows = []
CASES = [(-1, 128), (-1, 272), (0, 256), (32, 128), (64, 256), (128, 256), (128, 384), (256, 512)]
for group, k in CASES:
    for m in (1, 5, 17, 70):
        n = 40 if m < 17 else 70
        w = torch.randn(n, k, generator=g) * 0.05
        p = quantize_pack_ternary(w, group)
        x = torch.randn(m, k, generator=g)
        xq, sx = tn.act_quant_int8(x)
        bias = torch.randn(n, generator=g)
        for use_bias in (False, True):
            ref = ternary_gemm_ref(xq, sx, p.qweight, p.scales, group, k, torch.float32) + (bias if use_bias else 0)
            out = ternary_gemm_triton(xq, sx, p.qweight, p.scales, group, k, bias if use_bias else None, out_dtype=torch.float32)
            err = float((out - ref).abs().max())
            rows.append({"group": group, "k": k, "m": m, "n": n, "bias": use_bias, "err": err, "scale": float(ref.abs().max())})

quant = []
for shape in ((1, 16), (7, 96), (5, 4100), (3, 128)):
    x = torch.randn(*shape, generator=g) * 3
    x[0] = 0.0  # all-zero row
    if shape[0] > 2:
        x[2, 5] = 100.0  # outlier
    q, s = ternary_act_quant_triton(x)
    q0, s0 = tn.act_quant_int8(x)
    quant.append({"shape": list(shape), "q_equal": bool(torch.equal(q, q0)), "s_equal": bool(torch.equal(s, s0))})

# strided activations and bf16 output
w = torch.randn(32, 128, generator=g) * 0.05
p = quantize_pack_ternary(w, 128)
big = torch.randn(6, 256, generator=g)
xq, sx = tn.act_quant_int8(big[:, :128])
ref = ternary_gemm_ref(xq, sx, p.qweight, p.scales, 128, 128, torch.float32)
out = ternary_gemm_triton(xq, sx, p.qweight, p.scales, 128, 128, out_dtype=torch.bfloat16).float()
bf = float((out - ref).abs().max() / ref.abs().max())
# dispatcher: float activations -> quantiser -> GEMM, Triton path in interpreter mode, shapes with leading dims, bias
from forge.kernels import ops_gemm as O  # noqa: E402
from forge.kernels.ref.gemm_ternary import ternary_linear_ref  # noqa: E402

x3 = torch.randn(2, 3, 128, generator=g)
bias = torch.randn(32, generator=g)
y = O.ternary_linear(x3, p.qweight, p.scales, 128, 128, bias)
disp = {"path": O.last_path("ternary_linear"), "quant_path": O.last_path("quantize_act_ternary"), "shape": list(y.shape),
        "err": float((y - ternary_linear_ref(x3, p.qweight, p.scales, 128, 128, bias)).abs().max())}
print("REPORT_JSON=" + json.dumps({"gemm": rows, "quant": quant, "bf16_rel": bf, "dispatch": disp}))
