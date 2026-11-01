"""Runs the W4 dequantise kernel + the dequant_matmul dispatch in the Triton CPU interpreter and prints a JSON report.

Executed in a subprocess by tests/test_kernels_w4_dequant_interp.py (TRITON_INTERPRET must be set before triton is imported).
Validates kernel LOGIC against the fp32 oracle (group sizes 8..K, odd K / N, sym / asym,
act-order g_idx, row chunks, chunked matmul) -- NOT GPU compilation or speed.  fp16 / bf16 outputs.
"""
from __future__ import annotations

import json
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import forge.kernels.triton.gemm_w4a16 as w4mod  # noqa: E402
from forge.kernels.layouts import pack_w4, quantize_w4_rtn  # noqa: E402
from forge.kernels.ref import w4a16_gemm_ref  # noqa: E402
from forge.kernels.triton.gemm_w4_dequant import scale_mode, w4_dequant, w4_dequant_matmul  # noqa: E402

torch.manual_seed(0)
CASES = [
    dict(n=33, k=128, gs=128, sym=True, gidx=False),   # single group, odd N
    dict(n=40, k=100, gs=32, sym=False, gidx=False),   # K not a multiple of 8, group multiple of 8 (mode 1)
    dict(n=70, k=256, gs=64, sym=False, gidx=False),   # several groups per 256-wide tile
    dict(n=37, k=384, gs=128, sym=False, gidx=True),   # act-order, odd N
    dict(n=20, k=96, gs=-1, sym=True, gidx=False),     # per-channel scales
    dict(n=48, k=520, gs=40, sym=True, gidx=False),    # group size 40: multiple of 8 but not of the tile
    dict(n=16, k=90, gs=8, sym=False, gidx=False),     # tiny groups: many groups per tile, odd K
    dict(n=24, k=384, gs=32, sym=False, gidx=True),    # act-order, small groups
]
rows = []
for c in CASES:
    w = torch.randn(c["n"], c["k"])
    q, s, z = quantize_w4_rtn(w, c["gs"], sym=c["sym"])
    gsz = c["k"] if c["gs"] <= 0 else c["gs"]
    gi = (torch.arange(c["k"]) // gsz)[torch.randperm(c["k"])].to(torch.int32) if c["gidx"] else None
    p = pack_w4(q, s, z, c["gs"], None, gi)
    dense = p.dequantize(torch.float32)
    for dt in (torch.float16, torch.bfloat16):
        for tn, tk in ((16, 64), (32, 128), (8, 256)):
            cfg = dict(tile_n=tn, tile_k=tk, warps=4)
            out = w4_dequant(p.qweight, p.scales, p.zeros, p.g_idx, n=p.n, k=p.k, group_size=p.group_size, dtype=dt, config=cfg)
            tol = 2.0 ** -7 if dt == torch.bfloat16 else 2.0 ** -10  # the interpreter truncates (GPU rounds to nearest): 1 ulp
            rel = float(((out.float() - dense).abs() / dense.abs().clamp_min(1e-2 * float(dense.abs().max()))).max())
            rows.append(dict(kind="dequant", case=c, dt=str(dt), tile=[tn, tk], rel=rel, tol=tol * 1.01,
                             mode=scale_mode(p.group_size, p.k, p.num_groups, tk, p.g_idx is not None)))
    # row chunk (row0 multiple of 8) written into a caller-owned oversized buffer
    if c["n"] >= 24:
        buf = torch.full((c["n"], c["k"]), float("nan"), dtype=torch.float16)
        part = w4_dequant(p.qweight, p.scales, p.zeros, p.g_idx, n=p.n, k=p.k, group_size=p.group_size, dtype=torch.float16,
                          row0=8, rows=16, out=buf, config=dict(tile_n=16, tile_k=64))
        rows.append(dict(kind="chunk", case=c, ok=bool(torch.equal(part.float(), dense[8:24].half().float()) or
                                                      float((part.float() - dense[8:24]).abs().max()) < 2e-3 * float(dense.abs().max())),
                         rest_untouched=bool(torch.isnan(buf[16:]).all())))
    # dequant_matmul through the public dispatch, single chunk and forced multi-chunk
    x = torch.randn(5, c["k"]).half()
    b = torch.randn(c["n"]).half()
    ref = w4a16_gemm_ref(x, p, b)
    scale = float(ref.abs().max())
    y = w4mod.w4a16_gemm_triton(x, p, b, config={"impl": "dequant_matmul", "tile_n": 16, "tile_k": 64})
    rows.append(dict(kind="matmul", case=c, err=float((y.float() - ref.float()).abs().max()), scale=scale))

# genuinely multi-chunk: N=200, K=64 (fp16: 128 B per row); the budget is patched to 64 rows + 1 byte -> 4 chunks of 64/64/64/8 rows
w = torch.randn(200, 64)
q, s, z = quantize_w4_rtn(w, 32, sym=False)
p = pack_w4(q, s, z, 32)
x = torch.randn(4, 64).half()
ref = w4a16_gemm_ref(x, p)
import forge.kernels.triton.gemm_w4_dequant as dq  # noqa: E402
import forge.kernels.w4_tuning as t  # noqa: E402

chunks = t.plan_dequant_chunks(200, 64, 2, 64 * 64 * 2 + 1)
y = w4_dequant_matmul(x, p.qweight, p.scales, p.zeros, None, None, n=200, k=64, group_size=32)  # default budget: one chunk
real_budget = dq.dequant_budget_bytes
dq.dequant_budget_bytes = lambda cfg=None: 64 * 64 * 2 + 1
y2 = w4_dequant_matmul(x, p.qweight, p.scales, p.zeros, None, None, n=200, k=64, group_size=32)
multi = dict(n_chunks=len(chunks), err1=float((y.float() - ref.float()).abs().max()), err2=float((y2.float() - ref.float()).abs().max()),
             scale=float(ref.abs().max()), identical=bool(torch.equal(y, y2)))

# default routing: M >= 64 -> dequant_matmul without any config; kill switch; forced impl
dq.dequant_budget_bytes = real_budget
calls = []
orig_fn = dq.w4_dequant_matmul
dq.w4_dequant_matmul = lambda *a, **k: (calls.append(1), orig_fn(*a, **k))[1]
xl = torch.randn(72, 64).half()
yl = w4mod.w4a16_gemm_triton(xl, p, None)
big_default = dict(routed=len(calls) == 1, err=float((yl.float() - w4a16_gemm_ref(xl, p).float()).abs().max()))
calls.clear()
ys = w4mod.w4a16_gemm_triton(xl[:3], p, None)
small_default = dict(not_routed=len(calls) == 0)
os.environ["FORGE_W4_IMPL"] = "dequant_matmul"
yf = w4mod.w4a16_gemm_triton(xl[:3], p, None)
forced = dict(routed=len(calls) == 1, err=float((yf.float() - w4a16_gemm_ref(xl[:3], p).float()).abs().max()))

print("REPORT_JSON=" + json.dumps(dict(rows=rows, multi=multi, big_default=big_default, small_default=small_default, forced=forced)))
