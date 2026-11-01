#!/usr/bin/env python
"""Standalone W4A16 GEMM sweep + autotune for the GPU host (writes the winners into the autotune cache).

For every real Qwen3-14B linear shape (hidden 5120, q/o 5120x5120, k/v 1024x5120, gate/up 17408x5120, down
5120x17408, lm_head 151936x5120 -- override with ``--shapes N:K,...``) and every M in ``--m`` (default
1,4,8,16,32,64,128,256,1024) it

1. builds a synthetic ``portable_row`` PackedW4 of that shape (random nibbles, g128, asymmetric zeros; ``--act-order``
   adds a shuffled ``g_idx``), the fp32 oracle and a torch bf16 ``F.linear`` baseline on the dequantised weight;
2. times every candidate config (:func:`forge.kernels.w4_tuning.w4_candidates`, then a one-parameter-at-a-time refinement
   of the winner) through the correctness-gated :class:`forge.kernels.autotune.Autotuner`;
3. writes the best config per (chip, M bucket, N, K) into the autotune cache that
   :func:`forge.kernels.ops_gemm.tuned_config` reads at run time (``<FORGE_HOME>/autotune/<chip>.json``);
4. prints a table: bf16 / legacy kernel / default heuristic / tuned time, speed-up vs bf16, achieved GB/s and % of the
   measured copy bandwidth, and dumps everything to ``--out`` (JSON).

TIMING METHOD (important).  Decode GEMMs take 5-40 us on the device while one eager Triton launch costs 20-60 us of
host time, so eager per-call timing (what ``forge.kernels.bench`` does) mostly measures the Python / launch path and
cannot rank configs.  This script times with CUDA graphs (``--graph-iters`` calls captured and replayed, which is
what a CUDA-graph decode loop sees) over a ring of distinct weight copies sized to exceed the L2 cache, so neither the
bf16 baseline nor the W4 kernel is served from L2.  The eager (launch-bound) numbers are reported alongside.

Without a CUDA GPU the script only prints the plan (``--dry-run`` is implied).  Not executed on hardware by its author.

Examples::

    python scripts/bench_w4a16.py                                  # full Qwen3-14B sweep, quick candidate level
    python scripts/bench_w4a16.py --m 1,8,16 --shapes 5120:5120    # one shape, decode only
    python scripts/bench_w4a16.py --level full --refine 2          # exhaustive warps/stages/split sweep
    python scripts/bench_w4a16.py --dry-run                        # list candidates, no GPU needed
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QWEN3_14B_SHAPES: list[tuple[str, int, int]] = [  # (role, N, K)
    ("q_proj/o_proj", 5120, 5120),
    ("k_proj/v_proj", 1024, 5120),
    ("gate_proj/up_proj", 17408, 5120),
    ("down_proj", 5120, 17408),
    ("lm_head", 151936, 5120),
]
DEFAULT_M = [1, 4, 8, 16, 32, 64, 128, 256, 1024]


def parse_shapes(text: str | None) -> list[tuple[str, int, int]]:
    if not text or text == "qwen3-14b":
        return list(QWEN3_14B_SHAPES)
    out = []
    for part in text.split(","):
        n, k = part.lower().replace("x", ":").split(":")
        out.append((f"{n}x{k}", int(n), int(k)))
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", default="qwen3-14b", help="'qwen3-14b' or comma list N:K,N:K")
    ap.add_argument("--m", default=",".join(map(str, DEFAULT_M)), help="comma list of M (tokens)")
    ap.add_argument("--group", type=int, default=128, help="quantisation group size (-1 = per channel)")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--no-zeros", action="store_true", help="symmetric weights (no zero-point tensor)")
    ap.add_argument("--act-order", action="store_true", help="shuffled g_idx (external GPTQ desc_act checkpoints; Forge's own act-order artifacts use static groups and carry no g_idx)")
    ap.add_argument("--level", choices=["quick", "full"], default="quick", help="candidate enumeration level")
    ap.add_argument("--refine", type=int, default=1, help="refinement rounds around the winner (0 = none)")
    ap.add_argument("--graph-iters", type=int, default=32, help="calls captured per CUDA graph")
    ap.add_argument("--reps", type=int, default=15, help="graph replays timed per candidate")
    ap.add_argument("--l2-factor", type=float, default=3.0, help="weight ring size in multiples of the L2 cache")
    ap.add_argument("--cache-path", default=None, help="autotune JSON (default: <FORGE_HOME>/autotune/<chip>.json, the one the kernel loads)")
    ap.add_argument("--no-cache", action="store_true", help="measure only, do not write the autotune cache")
    ap.add_argument("--keep", action="store_true", help="keep existing cache entries instead of re-tuning (default: re-tune)")
    ap.add_argument("--out", default="w4a16_bench.json", help="JSON report path")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the plan (candidate counts) and exit; works without a GPU")
    return ap


def m_list(text: str) -> list[int]:
    return sorted({int(t) for t in text.split(",") if t.strip()})


def plan(shapes: list[tuple[str, int, int]], ms: list[int], group: int, level: str, sm: int | None, num_sms: int) -> list[dict[str, Any]]:
    from forge.kernels.w4_tuning import default_w4_config, m_bucket, w4_candidates

    rows = []
    for role, n, k in shapes:
        for m in ms:
            cands = w4_candidates(m, n, k, sm=sm, num_sms=num_sms, group=group, level=level)
            rows.append(dict(role=role, n=n, k=k, m=m, bucket=m_bucket(m), n_candidates=len(cands),
                             default=default_w4_config(m, n, k, sm=sm, num_sms=num_sms, group=group),
                             impls=sorted({c["impl"] for c in cands})))
    return rows


# ------------------------------------------------------------------------------------------------------------------
# GPU helpers (torch imported lazily so --dry-run / import work anywhere)
# ------------------------------------------------------------------------------------------------------------------
def synth_packed(n: int, k: int, group: int, device: str, *, zeros: bool, act_order: bool, seed: int):
    import torch

    from forge.kernels.layouts import PORTABLE_W4, PackedW4

    if k % 8:
        raise ValueError("synthetic weights need K % 8 == 0")
    gs = k if group <= 0 else group
    g = -(-k // gs)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    qw = torch.randint(-(2**31), 2**31, (n, k // 8), generator=gen, dtype=torch.int32).to(device)
    sc = (torch.rand(g, n, generator=gen) * 0.02 + 0.002).to(torch.float16).to(device)
    zw = torch.randint(-(2**31), 2**31, (g, -(-n // 8)), generator=gen, dtype=torch.int32).to(device) if zeros else None
    gidx = (torch.arange(k) // gs)[torch.randperm(k, generator=gen)].to(torch.int32).to(device) if act_order else None
    return PackedW4(qw, sc, zw, PORTABLE_W4, n, k, group, gidx)


def clone_packed(p):
    import torch  # noqa: F401

    from forge.kernels.layouts import PackedW4

    c = lambda t: None if t is None else t.clone()  # noqa: E731
    return PackedW4(c(p.qweight), c(p.scales), c(p.zeros), p.layout, p.n, p.k, p.group_size, c(p.g_idx))


class GraphTimer:
    """Callable ``(fn, cfg) -> [ms per call]`` using CUDA graphs; falls back to eager events if capture fails."""

    def __init__(self, iters: int, reps: int) -> None:
        self.iters, self.reps = iters, reps
        self.used_graph = True

    def __call__(self, fn, cfg=None) -> list[float]:
        import torch

        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(self.iters):
                    fn()
            g.replay()
            torch.cuda.synchronize()
            out = []
            for _ in range(self.reps):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                g.replay()
                e.record()
                e.synchronize()
                out.append(s.elapsed_time(e) / self.iters)
            return out
        except Exception as ex:  # noqa: BLE001 - capture can fail for exotic kernels; eager numbers are still usable
            self.used_graph = False
            print(f"  [warn] CUDA graph capture failed ({type(ex).__name__}: {ex}); eager timing", file=sys.stderr)
            out = []
            for _ in range(self.reps):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(self.iters):
                    fn()
                e.record()
                e.synchronize()
                out.append(s.elapsed_time(e) / self.iters)
            return out


def ring_size(nbytes: int, l2_bytes: int, factor: float, free_bytes: int, cap: int = 32) -> int:
    """Number of distinct weight copies so that cycling through them defeats the L2 cache (bounded by free memory)."""
    want = math.ceil(factor * max(l2_bytes, 64 << 20) / max(1, nbytes))
    fit = max(1, int(0.30 * free_bytes // max(1, nbytes)))
    return max(1, min(cap, want, fit))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shapes, ms = parse_shapes(args.shapes), m_list(args.m)

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    from forge.kernels.layouts.chip import chip_info

    chip = chip_info(None, args.device) if has_cuda else chip_info(None)
    rows_plan = plan(shapes, ms, args.group, args.level, chip.sm, chip.num_sms)
    if args.dry_run or not has_cuda:
        print(f"chip: {chip.name} sm={chip.sm} sms={chip.num_sms} fingerprint={chip.fingerprint}")
        for r in rows_plan:
            d = r["default"]
            print(f"{r['role']:>18} N={r['n']:<7} K={r['k']:<6} M={r['m']:<5} bucket={r['bucket']:<5} candidates={r['n_candidates']:<4} "
                  f"default={d['impl']}/{d['tile_m']}x{d['tile_n']}x{d['tile_k']} w{d['warps']} s{d['stages']} sk{d['split_k']}")
        if not has_cuda:
            print("\nNo CUDA device: plan only (nothing measured).")
        return 0 if (args.dry_run or not has_cuda) else 1

    import torch.nn.functional as F  # noqa: N812

    from forge.kernels import ops_gemm
    from forge.kernels.autotune import Autotuner, make_l2_flusher, summarize, time_callable
    from forge.kernels.bench import measure_peak_bandwidth
    from forge.kernels.ref import compare
    from forge.kernels.spec import load_spec
    from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton
    from forge.kernels.w4_tuning import W4_TUNE_TAG, m_bucket, refine_candidates, w4_candidates

    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    spec = load_spec("w4a16_gemm")
    props = torch.cuda.get_device_properties(args.device)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 96 << 20)
    peak = measure_peak_bandwidth(dev)
    print(f"chip {chip.name} sm_{chip.sm} {chip.num_sms} SMs, L2 {l2_bytes >> 20} MiB, measured copy bandwidth "
          f"{peak:.0f} GB/s ({'datasheet ' + format(chip.hbm_gbs, '.0f') if chip.hbm_gbs else 'datasheet unknown'}), dtype {args.dtype}")
    timer = GraphTimer(args.graph_iters, args.reps)
    tuner = Autotuner(None if args.cache_path is None else args.cache_path, chip=chip, timer=timer, max_candidates=None,
                      retries=1)
    if args.cache_path is not None:
        print(f"[warn] custom --cache-path: ops_gemm.tuned_config() only reads {tuner.cache.path.parent}/<chip>.json by default")
    flush = make_l2_flusher()
    dtype_cfg = ops_gemm.dtype_cfg("w4a16_gemm", dtype, args.group, W4_TUNE_TAG)
    report: dict[str, Any] = dict(chip=chip.to_dict(), dtype=args.dtype, group=args.group, peak_copy_gbs=peak, rows=[],
                                  cache=str(tuner.cache.path), created=time.strftime("%Y-%m-%d %H:%M:%S"),
                                  method="cuda-graph replay over an L2-defeating weight ring; eager numbers include launch overhead")
    table: list[str] = []

    for role, n, k in shapes:
        try:
            p0 = synth_packed(n, k, args.group, dev, zeros=not args.no_zeros, act_order=args.act_order, seed=args.seed)
            dense32 = p0.dequantize(torch.float32)
            dense = dense32.to(dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"[skip] {role} N={n} K={k}: out of memory building the oracle")
            torch.cuda.empty_cache()
            continue
        free = torch.cuda.mem_get_info(args.device)[0]
        w4_bytes = p0.nbytes
        ring_w4 = ring_size(w4_bytes, l2_bytes, args.l2_factor, free // 2)
        ring_bf = ring_size(dense.numel() * dense.element_size(), l2_bytes, args.l2_factor, free // 2)
        packs = [p0] + [clone_packed(p0) for _ in range(ring_w4 - 1)]
        denses = [dense] + [dense.clone() for _ in range(ring_bf - 1)]
        for m in ms:
            x = torch.randn(m, k, device=dev, dtype=dtype, generator=torch.Generator(device=dev).manual_seed(args.seed + m))
            ref = (x.float() @ dense32.t())
            shape = {"M": m, "N": n, "K": k}
            ctr = {"i": 0}

            def run_cfg(cfg, _shape=None, x=x, packs=packs, ctr=ctr):
                ctr["i"] += 1
                return w4a16_gemm_triton(x, packs[ctr["i"] % len(packs)], None, config=cfg, sm=chip.sm, num_sms=chip.num_sms)

            def bf16_call(x=x, denses=denses, ctr=ctr):
                ctr["i"] += 1
                return F.linear(x, denses[ctr["i"] % len(denses)])

            cands = w4_candidates(m, n, k, sm=chip.sm, num_sms=chip.num_sms, group=args.group, level=args.level)
            res = tuner.tune(spec, shape, run_cfg, lambda s, ref=ref: ref, dtype_cfg=dtype_cfg, layout_id=p0.layout_id,
                             candidates=cands, force=not args.keep, group=args.group)
            best = res.best
            for _ in range(args.refine if best else 0):
                extra = refine_candidates(best, m, n, k, sm=chip.sm, group=args.group) + [best]
                r2 = tuner.tune(spec, shape, run_cfg, lambda s, ref=ref: ref, dtype_cfg=dtype_cfg, layout_id=p0.layout_id,
                                candidates=extra, force=True, group=args.group)
                if r2.best_measurement and res.best_measurement and r2.best_measurement.median_ms <= res.best_measurement.median_ms:
                    res, best = r2, r2.best
                elif r2.best:
                    break
            rejected = [dict(config=r.config, error=r.error) for r in res.rejected[:5]]
            row: dict[str, Any] = dict(role=role, n=n, k=k, m=m, bucket=m_bucket(m), best=best, n_measured=len(res.measurements),
                                       n_rejected=len(res.rejected), rejected_sample=rejected, from_cache=res.from_cache,
                                       cached=res.cached, notes=res.notes)
            if best is None:
                row["error"] = "no candidate passed the correctness gate"
                report["rows"].append(row)
                table.append(f"{role:>18} N={n:<7} K={k:<6} M={m:<5} NO VALID CONFIG ({len(res.rejected)} rejected)")
                continue
            graph_ms = lambda fn: float(summarize(timer(fn)).median)  # noqa: E731
            bf_ms = graph_ms(bf16_call)
            tuned_ms = graph_ms(lambda: run_cfg(best))
            legacy_ms = graph_ms(lambda: run_cfg({"impl": "legacy"}))
            dflt_ms = graph_ms(lambda: run_cfg(None))
            eager_tuned = float(summarize(time_callable(lambda: run_cfg(best), warmup=5, repeats=30, flush=flush)).median)
            eager_bf = float(summarize(time_callable(bf16_call, warmup=5, repeats=30, flush=flush)).median)
            ops_gemm.clear_config_memo()
            picked = ops_gemm.tuned_config("w4a16_gemm", chip, m, n, k, dtype_cfg_str=dtype_cfg, layout_id=p0.layout_id)
            nbytes = w4_bytes + x.numel() * x.element_size() + m * n * x.element_size()
            gbs = nbytes / (tuned_ms * 1e-3) / 1e9
            row.update(bf16_us=bf_ms * 1e3, legacy_us=legacy_ms * 1e3, default_us=dflt_ms * 1e3, tuned_us=tuned_ms * 1e3,
                       speedup_vs_bf16=bf_ms / tuned_ms, speedup_legacy_vs_bf16=bf_ms / legacy_ms, speedup_default_vs_bf16=bf_ms / dflt_ms,
                       eager_tuned_us=eager_tuned * 1e3, eager_bf16_us=eager_bf * 1e3, eager_speedup=eager_bf / eager_tuned,
                       achieved_gbs=gbs, pct_peak=gbs / peak if peak else None, bytes=nbytes,
                       dispatcher_picks_cache=bool(picked) and picked == best, graph_timing=timer.used_graph,
                       top=[dict(config=q.config, us=q.median_ms * 1e3) for q in sorted(res.measurements, key=lambda q: q.median_ms)[:5]])
            report["rows"].append(row)
            c = best
            table.append(f"{role:>18} N={n:<7} K={k:<6} M={m:<5} bf16 {bf_ms * 1e3:8.1f}us legacy {legacy_ms * 1e3:8.1f}us "
                         f"dflt {dflt_ms * 1e3:8.1f}us tuned {tuned_ms * 1e3:8.1f}us  x{bf_ms / tuned_ms:5.2f} (legacy x{bf_ms / legacy_ms:4.2f}, "
                         f"eager x{eager_bf / eager_tuned:4.2f})  {gbs:6.0f} GB/s {100 * gbs / peak:4.0f}%  "
                         f"{c['impl']} {c['tile_m']}x{c['tile_n']}x{c['tile_k']} w{c['warps']} s{c['stages']} sk{c['split_k']}"
                         f"{'' if row['dispatcher_picks_cache'] else '  [cache not picked up!]'}")
            print(table[-1], flush=True)
            if args.no_cache:
                tuner.cache.delete(res.key)
            Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
        del packs, denses, dense, dense32
        torch.cuda.empty_cache()

    print("\n".join(["", "=" * 40 + " summary " + "=" * 40, *table]))
    if report["rows"]:
        ok = [r for r in report["rows"] if r.get("speedup_vs_bf16")]
        dec = [r for r in ok if r["m"] <= 16]
        if dec:
            gm = math.exp(sum(math.log(r["speedup_vs_bf16"]) for r in dec) / len(dec))
            print(f"\ndecode (M<=16) geometric-mean speed-up vs bf16 F.linear: x{gm:.2f}  (target 3.2x, spec kpi)")
    Path(args.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"report: {Path(args.out).resolve()}\nautotune cache: {tuner.cache.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
