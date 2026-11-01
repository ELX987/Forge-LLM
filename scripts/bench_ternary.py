#!/usr/bin/env python
"""Ternary (1.58-bit) GEMM benchmark for the GPU host: CUDA-graph timing over a ring of weight copies larger than L2.

Why not ``forge.kernels.bench``?  Decode-size GEMMs take 5-40 us on the device while one eager Triton launch costs 20-60 us of host time, so eager
timing measures the launch path (see ``scripts/bench_w4a16.py``).  This script captures ``--graph-iters`` calls in a CUDA graph, replays it, and cycles
through enough distinct weight copies that no kernel is served from L2.

For every shape (N, K) and M it times: (a) the Triton ternary kernel (``ops_gemm.ternary_linear`` with pre-quantised activations excluded), (b) the torch
dequantise-then-matmul reference route, (c) a dense bf16 ``F.linear`` baseline, and prints TOPS and effective GB/s next to the numbers tern158 measured with
its hand-written CUDA kernels on an RTX PRO 6000 Blackwell (sm_120): mma 346 TOPS (M=4096, N=K=4096, G=128), bf16-cuBLAS route 332 TOPS, GEMV 4.8 TOPS /
~600 GB/s at M=1.  Those are the targets, not Forge results.

Without a CUDA GPU the script only prints the plan.  NOT executed on hardware by its author; every number it prints on your machine is the first measurement.

    python scripts/bench_ternary.py --quick
    python scripts/bench_ternary.py --m 1,16,512,4096 --shapes 4096:4096,11008:4096 --group 128 --out ternary_bench.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TERN158_SM120 = {  # tern158 logs, 2026-10-27: (M, N, K) -> {backend: TOPS}
    (4096, 4096, 4096): {"mma_cuda": 346.0, "bf16_cublas": 332.0, "wmma_cuda": 187.0},
    (4096, 11008, 4096): {"mma_cuda": 377.0, "bf16_cublas": 359.0, "wmma_cuda": 191.0},
    (512, 4096, 4096): {"mma_cuda": 257.0, "bf16_cublas": 260.0},
    (16, 4096, 4096): {"gemv_cuda": 19.9, "bf16_cublas": 21.6},
    (1, 4096, 4096): {"gemv_cuda": 4.8, "bf16_cublas": 1.46},
}


def _time_graph(fn, iters: int, repeats: int = 5) -> float:
    """Median seconds per call of ``fn`` replayed ``iters`` times inside a CUDA graph."""
    import torch

    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    ts = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g.replay()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) / iters)
    ts.sort()
    return ts[len(ts) // 2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", default="1,16,128,512,4096")
    ap.add_argument("--shapes", default="4096:4096,11008:4096", help="N:K pairs")
    ap.add_argument("--group", type=int, default=128, help="scale group (-1 per row, 0 per tensor)")
    ap.add_argument("--graph-iters", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ms = [int(v) for v in args.m.split(",")]
    shapes = [tuple(int(t) for t in s.split(":")) for s in args.shapes.split(",")]
    if args.quick:
        ms, shapes = [1, 16, 4096], shapes[:1]
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    if not has_gpu:
        print("no CUDA GPU: plan only (this script times with CUDA graphs)")
        for (n, k) in shapes:
            for m in ms:
                print(f"  would time M={m} N={n} K={k} group={args.group}; tern158 sm_120 reference: {TERN158_SM120.get((m, n, k), 'n/a')}")
        return 0

    import torch.nn.functional as F

    from forge.formats import ternary as tn
    from forge.kernels import ops_gemm as O
    from forge.kernels.layouts.ternary import quantize_pack_ternary

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    l2 = getattr(props, "L2_cache_size", 64 * 2**20)
    rows = []
    print(f"device {props.name} sm_{props.major}{props.minor}; Forge ternary_linear vs torch dequant+matmul vs dense bf16 (graph-timed)")
    for (n, k) in shapes:
        w = torch.randn(n, k, device=dev) * 0.02
        p = quantize_pack_ternary(w, args.group)
        ring = max(2, int(2 * l2 / max(1, p.nbytes)) + 1)
        packs = [quantize_pack_ternary(torch.randn(n, k, device=dev) * 0.02, args.group) for _ in range(min(ring, 8))]
        denses = [q.dequantize(torch.float32).to(torch.bfloat16) for q in packs[:2]]
        for m in ms:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            xq, sx = tn.act_quant_int8(x)
            idx = [0]

            def tri():
                q = packs[idx[0] % len(packs)]
                idx[0] += 1
                return O.ternary_linear(x, q.qweight, q.scales, args.group, k, out_dtype=torch.bfloat16)

            def deq():
                q = packs[idx[0] % len(packs)]
                idx[0] += 1
                return F.linear(x, q.dequantize(torch.float32).to(torch.bfloat16))

            def bf16():
                idx[0] += 1
                return F.linear(x, denses[idx[0] % len(denses)])

            row = {"m": m, "n": n, "k": k, "group": args.group}
            for name, fn in (("triton", tri), ("dequant_matmul", deq), ("bf16", bf16)):
                try:
                    sec = _time_graph(fn, args.graph_iters)
                except Exception as e:  # noqa: BLE001
                    row[name] = {"error": f"{type(e).__name__}: {e}"}
                    continue
                row[name] = {"us": sec * 1e6, "tops": 2.0 * m * n * k / sec / 1e12, "gbs": (p.nbytes + 2 * m * (k + n)) / sec / 1e9}
            row["path"] = O.last_path("ternary_linear")
            row["tern158_sm120"] = TERN158_SM120.get((m, n, k))
            rows.append(row)
            f = lambda r: f"{r['us']:9.1f} us {r['tops']:7.1f} TOPS {r['gbs']:7.0f} GB/s" if isinstance(r, dict) and "us" in r else str(r)  # noqa: E731
            print(f"M={m:5d} N={n:6d} K={k:6d} | triton[{row['path']}] {f(row.get('triton'))} | dequant {f(row.get('dequant_matmul'))} | bf16 {f(row.get('bf16'))}")
    if args.out:
        Path(args.out).write_text(json.dumps({"device": props.name, "sm": props.major * 10 + props.minor, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
