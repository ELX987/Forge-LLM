"""Runs the nibble-split W4A16 kernels (impl='dot' / 'gemv') in the Triton CPU interpreter and prints a JSON report.

Executed in a subprocess by tests/test_kernels_w4_fast_interp.py (TRITON_INTERPRET must be set before triton is
imported).  Validates kernel LOGIC (nibble/word indexing, stride-8 activation gather, group/zero handling, act-order
column sort + in-kernel gather, split-K, masking of odd N / K) against the fp32 oracle -- NOT GPU compilation or speed.
fp16 activations (the interpreter does not emulate bf16 ``tl.dot``; the gemv path is also checked with bf16).
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

torch.manual_seed(0)
CASES = [
    dict(m=1, n=33, k=128, gs=128, sym=True, gidx=False),
    dict(m=3, n=40, k=100, gs=32, sym=False, gidx=False),
    dict(m=17, n=70, k=256, gs=64, sym=False, gidx=False),
    dict(m=5, n=37, k=384, gs=128, sym=False, gidx=True),
    dict(m=2, n=20, k=96, gs=-1, sym=True, gidx=False),
    dict(m=8, n=48, k=520, gs=128, sym=True, gidx=False),
    dict(m=20, n=16, k=384, gs=32, sym=False, gidx=True),
]
rows = []
for c in CASES:
    w = torch.randn(c["n"], c["k"])
    q, s, z = quantize_w4_rtn(w, c["gs"], sym=c["sym"])
    gsz = c["k"] if c["gs"] <= 0 else c["gs"]
    gi = (torch.arange(c["k"]) // gsz)[torch.randperm(c["k"])].to(torch.int32) if c["gidx"] else None
    p = pack_w4(q, s, z, c["gs"], None, gi)
    x = torch.randn(c["m"], c["k"]).half()
    b = torch.randn(c["n"]).half()
    ref = w4a16_gemm_ref(x, p, b)
    scale = float(ref.abs().max())
    for impl in ("dot", "gemv", "legacy", None):
        if impl == "gemv" and c["m"] > 8:
            continue
        for tk in (64, 128, 256):
            for sk in (1, 3):
                for det in (True, False):
                    cfg = dict(tile_m=16, tile_n=32, tile_k=tk, split_k=sk, stages=2, warps=4)
                    if impl:
                        cfg["impl"] = impl
                    y = w4mod.w4a16_gemm_triton(x, p, b, config=cfg, deterministic=det)
                    rows.append(dict(case=c, impl=impl, tile_k=tk, split_k=sk, det=det, scale=scale,
                                     err=float((y.float() - ref.float()).abs().max())))
    cfg = dict(impl="dot", tile_m=16, tile_n=32, tile_k=128, split_k=3)
    y1 = w4mod.w4a16_gemm_triton(x, p, b, config=cfg)
    y2 = w4mod.w4a16_gemm_triton(x, p, b, config=cfg)
    rows.append(dict(case=c, bit_identical=bool(torch.equal(y1, y2))))
    # default config (no impl): must take the nibble-split path, not the legacy fallback
    yd = w4mod.w4a16_gemm_triton(x, p, b)
    rows.append(dict(case=c, default_err=float((yd.float() - ref.float()).abs().max()), scale=scale,
                     fast_broken=len(w4mod._FAST_BROKEN)))
    if c["gidx"]:
        rows.append(dict(case=c, sorted_cached=w4mod.group_sorted(p) is not None))

# bf16 activations through the FMA kernel (fp32 math inside; only the output rounds to bf16)
w = torch.randn(40, 256)
q, s, z = quantize_w4_rtn(w, 128, sym=False)
p = pack_w4(q, s, z, 128)
xb = torch.randn(4, 256).bfloat16()
yb = w4mod.w4a16_gemm_triton(xb, p, None, config=dict(impl="gemv", tile_n=16, tile_k=128, split_k=2))
refb = w4a16_gemm_ref(xb, p)
bf16 = dict(err=float((yb.float() - refb).abs().max()), scale=float(refb.abs().max()))

# FORGE_W4_IMPL kill switch
os.environ["FORGE_W4_IMPL"] = "legacy"
yl = w4mod.w4a16_gemm_triton(xb.half(), p, None)
kill = dict(err=float((yl.float() - w4a16_gemm_ref(xb.half(), p)).abs().max()))

print("REPORT_JSON=" + json.dumps(dict(rows=rows, bf16_gemv=bf16, kill_switch=kill)))
