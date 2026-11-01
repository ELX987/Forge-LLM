#!/usr/bin/env python
"""Standalone W4A16 GEMM sweep + autotune for the GPU host (writes the winners into the autotune cache).

For every real Qwen3-14B linear shape (hidden 5120, q/o 5120x5120, k/v 1024x5120, gate/up 17408x5120, down
5120x17408, lm_head 151936x5120 -- override with ``--shapes N:K,...``) and every M in ``--m`` (default
1,4,8,16,32,64,128,256,1024) it

1. builds a synthetic ``portable_row`` PackedW4 of that shape (random nibbles, g128, asymmetric zeros; ``--act-order``
   adds a shuffled ``g_idx``), the fp32 oracle and a torch bf16 ``F.linear`` baseline on the dequantised weight;
2. times the candidate configs (:func:`forge.kernels.w4_tuning.w4_candidates`: gemv / dot / dequant_matmul / legacy, then a
   one-parameter-at-a-time refinement of the winner) through the correctness-gated :class:`forge.kernels.autotune.Autotuner`;
3. writes the best config per (chip, M bucket, N, K) into the autotune cache that
   :func:`forge.kernels.ops_gemm.tuned_config` reads at run time (``<FORGE_HOME>/autotune/<chip>.json``) -- after EVERY row,
   so a killed run still leaves usable tuned configs;
4. prints a table: bf16 / legacy kernel / default heuristic / tuned time, speed-up vs bf16, achieved GB/s and % of the
   measured copy bandwidth, and dumps everything to ``--out`` (JSON, rewritten atomically after every row); a final summary
   gives the best impl per M, the decode (M<=16) geometric-mean speed-up and the prefill (M>=32) ratio.

TIMING METHOD (important).  Decode GEMMs take 5-40 us on the device while one eager Triton launch costs 20-60 us of
host time, so eager per-call timing (what ``forge.kernels.bench`` does) mostly measures the Python / launch path and
cannot rank configs.  This script times with CUDA graphs (``--graph-iters`` calls captured and replayed, which is
what a CUDA-graph decode loop sees) over a ring of distinct weight copies sized to exceed the L2 cache, so neither the
bf16 baseline nor the W4 kernel is served from L2.  The eager (launch-bound) numbers are reported alongside.  Act-order
weights are prepared (:func:`forge.kernels.ops_gemm.prepare_w4`) before capture, so capture never syncs the host.

SWEEP COST.  The full sweep takes hours on a power-capped card.  ``--quick`` runs M in {1,8,16,64,256,1024} on two shapes
(q/o 5120x5120, down 5120x17408) with at most 8 candidates per bucket, no refinement, fewer repeats; ``--time-budget-min N``
stops gracefully after N minutes (results so far stay in the cache and the JSON); candidates that are > ``--prune-factor``
(2x) slower than the best so far are dropped after one cheap probe; throttle detection retries a candidate at most once and
marks the row ``throttled`` (cached unless ``--strict-throttle``) instead of looping.

Without a CUDA GPU the script only prints the plan (``--dry-run`` is implied).  Not executed on hardware by its author.

Examples::

    python scripts/bench_w4a16.py --quick                          # ~minutes: the routing-relevant subset
    python scripts/bench_w4a16.py --quick --time-budget-min 30     # ... but never longer than 30 minutes
    python scripts/bench_w4a16.py                                  # full Qwen3-14B sweep, quick candidate level
    python scripts/bench_w4a16.py --m 1,8,16 --shapes 5120:5120    # one shape, decode only
    python scripts/bench_w4a16.py --level full --refine 2          # exhaustive warps/stages/split sweep
    python scripts/bench_w4a16.py --dry-run [--quick]              # list candidates, no GPU needed
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable
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
QUICK_M = [1, 8, 16, 64, 256, 1024]
QUICK_SHAPES = [QWEN3_14B_SHAPES[0], QWEN3_14B_SHAPES[3]]
QUICK_MAX_CANDIDATES = 8


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
    ap.add_argument("--shapes", default=None, help="'qwen3-14b' (default) or comma list N:K,N:K")
    ap.add_argument("--m", default=None, help=f"comma list of M (tokens); default {','.join(map(str, DEFAULT_M))} ({','.join(map(str, QUICK_M))} with --quick)")
    ap.add_argument("--quick", action="store_true", help=f"fast sweep: M in {QUICK_M}, 2 shapes (5120x5120, 5120x17408), <= {QUICK_MAX_CANDIDATES} candidates "
                    "per bucket, no refinement, fewer repeats (explicit --m / --shapes / --refine / --reps still win)")
    ap.add_argument("--time-budget-min", type=float, default=0.0, help="stop gracefully after this many minutes (0 = unlimited); partial results are kept")
    ap.add_argument("--max-candidates", type=int, default=None, help="cap the candidates per bucket (0 = unlimited; --quick: 8)")
    ap.add_argument("--prune-factor", type=float, default=2.0, help="drop a candidate whose probe is this many times slower than the best so far")
    ap.add_argument("--no-prune", action="store_true", help="disable the early abort of clearly slower candidates")
    ap.add_argument("--strict-throttle", action="store_true", help="do not cache winners measured under throttling (default: cache them, flagged)")
    ap.add_argument("--group", type=int, default=128, help="quantisation group size (-1 = per channel)")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--no-zeros", action="store_true", help="symmetric weights (no zero-point tensor)")
    ap.add_argument("--act-order", action="store_true", help="shuffled g_idx (external GPTQ desc_act checkpoints; Forge's own act-order artifacts use static groups and carry no g_idx)")
    ap.add_argument("--level", choices=["quick", "full"], default="quick", help="candidate enumeration level")
    ap.add_argument("--refine", type=int, default=None, help="refinement rounds around the winner (0 = none; default 1, --quick: 0)")
    ap.add_argument("--graph-iters", type=int, default=None, help="calls captured per CUDA graph (default 16, --quick 8)")
    ap.add_argument("--reps", type=int, default=None, help="graph replays timed per candidate (default 9, --quick 5)")
    ap.add_argument("--l2-factor", type=float, default=3.0, help="weight ring size in multiples of the L2 cache")
    ap.add_argument("--cache-path", default=None, help="autotune JSON (default: <FORGE_HOME>/autotune/<chip>.json, the one the kernel loads)")
    ap.add_argument("--no-cache", action="store_true", help="measure only: tuning writes to a throwaway cache file, your autotune cache is never read or touched")
    ap.add_argument("--keep", action="store_true", help="resume: keep existing cache entries instead of re-tuning (default: re-tune) and skip rows "
                    "already finished in the existing --out report (their rows are carried over)")
    ap.add_argument("--out", default=None, help="JSON report path (default: w4a16_bench_<timestamp>.json; an existing file is never truncated: "
                    "without --keep a timestamped sibling is used, with --keep its finished rows are merged)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the plan (candidate counts) and exit; works without a GPU")
    return ap


def m_list(text: str) -> list[int]:
    return sorted({int(t) for t in text.split(",") if t.strip()})


def resolve_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Effective sweep parameters: ``--quick`` only supplies defaults for what the user did not pass."""
    q = bool(args.quick)
    shapes = parse_shapes(args.shapes) if args.shapes else (list(QUICK_SHAPES) if q else list(QWEN3_14B_SHAPES))
    ms = m_list(args.m) if args.m else (list(QUICK_M) if q else list(DEFAULT_M))
    return dict(
        shapes=shapes, ms=ms,
        max_candidates=args.max_candidates if args.max_candidates is not None else (QUICK_MAX_CANDIDATES if q else 0),
        refine=args.refine if args.refine is not None else (0 if q else 1),
        graph_iters=args.graph_iters if args.graph_iters is not None else (8 if q else 16),
        reps=args.reps if args.reps is not None else (5 if q else 9),
        eager_reps=10 if q else 20,
        prune_factor=None if args.no_prune else float(args.prune_factor),
        budget_s=max(0.0, float(args.time_budget_min)) * 60.0,
    )


def plan(shapes: list[tuple[str, int, int]], ms: list[int], group: int, level: str, sm: int | None, num_sms: int,
         max_candidates: int = 0) -> list[dict[str, Any]]:
    from forge.kernels.w4_tuning import default_w4_config, limit_candidates, m_bucket, w4_candidates

    rows = []
    for role, n, k in shapes:
        for m in ms:
            cands = w4_candidates(m, n, k, sm=sm, num_sms=num_sms, group=group, level=level)
            if max_candidates:
                cands = limit_candidates(cands, max_candidates)
            rows.append(dict(role=role, n=n, k=k, m=m, bucket=m_bucket(m), n_candidates=len(cands),
                             default=default_w4_config(m, n, k, sm=sm, num_sms=num_sms, group=group),
                             impls=sorted({c["impl"] for c in cands})))
    return rows


# ------------------------------------------------------------------------------------------------------------------
# incremental results, time budget, sweep driver, summary (pure Python: unit-tested on CPU)
# ------------------------------------------------------------------------------------------------------------------
class Budget:
    """Wall-clock budget of the sweep (``seconds <= 0`` = unlimited).  ``clock`` is injectable for tests."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.seconds = float(seconds)
        self.start = clock()
        self.deadline: float | None = self.start + self.seconds if self.seconds > 0 else None

    def expired(self) -> bool:
        return self.deadline is not None and self.clock() >= self.deadline

    def elapsed_min(self) -> float:
        return (self.clock() - self.start) / 60.0


class ReportSink:
    """Collects rows and rewrites the JSON report atomically after every change, so a killed run leaves a valid file."""

    def __init__(self, path: str | Path, header: dict[str, Any], resume: bool = False) -> None:
        """``resume``: an existing report at ``path`` is loaded and its rows kept (new rows replace same-key rows).  Otherwise an
        existing file is never truncated: the report goes to a timestamped sibling path (``self.path``)."""
        self.path = Path(path)
        rows: list[dict[str, Any]] = []
        if self.path.exists():
            loaded = None
            if resume:
                try:
                    loaded = json.loads(self.path.read_text(encoding="utf-8")).get("rows")
                except (OSError, ValueError, AttributeError):
                    loaded = None
            if isinstance(loaded, list):
                rows = [r for r in loaded if isinstance(r, dict)]
            else:
                self.path = unique_path(self.path)
        self.report: dict[str, Any] = dict(header, rows=rows, partial=True, stopped=None)
        if rows:
            self.report["summary"] = summarize_rows(rows)
        self.flush()

    @staticmethod
    def row_key(row: dict[str, Any]) -> tuple:
        return (row.get("role"), row.get("n"), row.get("k"), row.get("m"))

    def finished(self, role: str, n: int, k: int, m: int) -> bool:
        """A previous (resumed) run already measured this row successfully."""
        return any(self.row_key(r) == (role, n, k, m) and r.get("best") and not r.get("error") for r in self.report["rows"])

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(self.report, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.path)

    def add_row(self, row: dict[str, Any]) -> None:
        key = self.row_key(row)
        self.report["rows"] = [r for r in self.report["rows"] if self.row_key(r) != key] + [row]
        self.report["summary"] = summarize_rows(self.report["rows"])
        self.flush()

    def finish(self, stopped: str | None) -> None:
        self.report["partial"] = bool(stopped)
        self.report["stopped"] = stopped
        self.report["summary"] = summarize_rows(self.report["rows"])
        self.flush()


def unique_path(path: Path) -> Path:
    """``path`` with a timestamp (and a counter if needed) inserted before the suffix: the first name that does not exist yet."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cand = path.with_name(f"{path.stem}.{stamp}{path.suffix}")
    i = 1
    while cand.exists():
        cand = path.with_name(f"{path.stem}.{stamp}-{i}{path.suffix}")
        i += 1
    return cand


class SweepHooks:
    """What :func:`sweep` needs from the measuring side (the GPU implementation lives in :func:`main`)."""

    def setup(self, role: str, n: int, k: int) -> Any:  # one per shape
        return None

    def row(self, state: Any, role: str, n: int, k: int, m: int, budget: Budget) -> dict[str, Any]:
        raise NotImplementedError

    def teardown(self, state: Any) -> None:
        pass


def sweep(shapes: list[tuple[str, int, int]], ms: list[int], hooks: SweepHooks, sink: ReportSink, budget: Budget,
          log: Callable[[str], None] = print, skip_finished: bool = False) -> str | None:
    """Run the (shape x M) grid; write the report after every row.  Returns why it stopped early (``None`` = complete):
    ``'time budget'`` or ``'interrupted'`` (Ctrl-C).  ``skip_finished``: rows the sink already holds (resumed report) are not
    measured again."""
    try:
        for role, n, k in shapes:
            if budget.expired():
                log(f"[stop] time budget reached before {role} N={n} K={k}")
                return "time budget"
            todo = [m for m in ms if not (skip_finished and sink.finished(role, n, k, m))]
            if len(todo) < len(ms):
                log(f"[skip] {role} N={n} K={k}: M={[m for m in ms if m not in todo]} already finished in the resumed report")
            if not todo:
                continue
            state = hooks.setup(role, n, k)
            try:
                for m in todo:
                    if budget.expired():
                        log(f"[stop] time budget reached at {role} N={n} K={k} M={m}")
                        return "time budget"
                    try:
                        row = hooks.row(state, role, n, k, m, budget)
                    except Exception as e:  # noqa: BLE001 - one bad row (OOM, driver hiccup) must not lose the rest of the sweep
                        row = dict(role=role, n=n, k=k, m=m, error=f"{type(e).__name__}: {e}")
                        log(f"[row failed] {role} N={n} K={k} M={m}: {row['error']}")
                    try:
                        sink.add_row(row)
                    except Exception as e:  # noqa: BLE001 - a report write error (disk full, ...) must not abort the sweep either
                        log(f"[report write failed] {role} N={n} K={k} M={m}: {type(e).__name__}: {e}")
            finally:
                hooks.teardown(state)
    except KeyboardInterrupt:
        log("[stop] interrupted; partial results are kept")
        return "interrupted"
    return None


def _gmean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-M best impl (majority over shapes), geometric-mean tuned speed-up vs bf16, decode (M<=16) geomean, prefill (M>=32)
    ratio, how often the heuristic default was slower than the legacy kernel, and the suggested ``dequant_min_m``."""
    ok = [r for r in rows if r.get("best") and r.get("speedup_vs_bf16")]
    by_m: dict[int, dict[str, Any]] = {}
    for m in sorted({r["m"] for r in ok}):
        rs = [r for r in ok if r["m"] == m]
        counts = Counter(r["best"]["impl"] for r in rs)
        by_m[m] = dict(best_impl=counts.most_common(1)[0][0], impl_counts=dict(counts), n=len(rs),
                       speedup_vs_bf16=_gmean([r["speedup_vs_bf16"] for r in rs]),
                       default_vs_legacy=_gmean([r["default_us"] and r["legacy_us"] / r["default_us"] for r in rs if r.get("default_us") and r.get("legacy_us")]))
    dec = [r["speedup_vs_bf16"] for r in ok if r["m"] <= 16]
    pre = [r["speedup_vs_bf16"] for r in ok if r["m"] >= 32]
    slower = [(r["role"], r["m"]) for r in ok if r.get("default_us") and r.get("legacy_us") and r["default_us"] > 1.05 * r["legacy_us"]]
    return dict(by_m=by_m, decode_geomean=_gmean(dec), prefill_ratio=_gmean(pre), n_rows=len(rows), n_ok=len(ok),
                n_throttled=sum(1 for r in rows if r.get("throttled")), default_slower_than_legacy=slower,
                suggested_dequant_min_m=suggest_dequant_min_m(ok))


def suggest_dequant_min_m(rows: list[dict[str, Any]]) -> int | None:
    """Smallest measured M such that ``dequant_matmul`` won (or tied the winner within 3%) at that M and at every larger measured
    M, for every shape -> the value for ``FORGE_W4_DEQUANT_MIN_M`` / the chip profile.  ``None`` if it never wins at the top M."""
    def wins(r: dict[str, Any]) -> bool:
        if r["best"]["impl"] == "dequant_matmul":
            return True
        dq = r.get("dequant_us")
        return bool(dq and r.get("tuned_us") and dq <= 1.03 * r["tuned_us"])

    ms = sorted({r["m"] for r in rows})
    out = None
    for m in reversed(ms):
        if all(wins(r) for r in rows if r["m"] == m):
            out = m
        else:
            break
    return out


def format_summary(s: dict[str, Any]) -> list[str]:
    lines = ["", "=" * 34 + " best impl per M " + "=" * 34,
             f"{'M':>6}  {'best impl':<15} {'impl counts':<40} {'x vs bf16':>9}  {'dflt vs legacy':>14}"]
    for m, d in s["by_m"].items():
        sp = d["speedup_vs_bf16"]
        dl = d["default_vs_legacy"]
        lines.append(f"{m:>6}  {d['best_impl']:<15} {json.dumps(d['impl_counts']):<40} {sp:9.2f}  {(f'x{dl:.2f}') if dl else '-':>14}")
    if s["decode_geomean"]:
        lines.append(f"decode (M<=16) geometric-mean speed-up vs bf16 F.linear: x{s['decode_geomean']:.2f}  (target 3.2x, spec kpi)")
    if s["prefill_ratio"]:
        lines.append(f"prefill (M>=32) geometric-mean ratio vs bf16 F.linear: x{s['prefill_ratio']:.2f}  (<1 = slower than dense bf16)")
    if s["default_slower_than_legacy"]:
        lines.append(f"WARNING: the heuristic default was >5% slower than the legacy kernel in {len(s['default_slower_than_legacy'])} row(s): "
                     f"{s['default_slower_than_legacy'][:6]}")
    if s["suggested_dequant_min_m"]:
        lines.append(f"dequant_matmul wins from M={s['suggested_dequant_min_m']} up on every swept shape "
                     f"(set FORGE_W4_DEQUANT_MIN_M={s['suggested_dequant_min_m']} or the chip profile if it differs from the default)")
    if s["n_throttled"]:
        lines.append(f"{s['n_throttled']} row(s) were measured under throttling (power-capped card): rankings are valid, absolute times pessimistic")
    return lines


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
    """Callable ``(fn, cfg) -> [ms per call]`` using CUDA graphs; falls back to eager events if capture fails.
    ``probe(fn, cfg)`` is the cheap variant used for the early abort (small graph, 3 replays)."""

    def __init__(self, iters: int, reps: int) -> None:
        self.iters, self.reps = iters, reps
        self.used_graph = True
        self._warned: set[str] = set()

    def _run(self, fn, iters: int, reps: int, warm: int) -> list[float]:
        import torch

        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        try:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(iters):
                    fn()
            g.replay()
            torch.cuda.synchronize()
            out = []
            for _ in range(reps):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                g.replay()
                e.record()
                e.synchronize()
                out.append(s.elapsed_time(e) / iters)
            return out
        except Exception as ex:  # noqa: BLE001 - capture can fail for exotic kernels; eager numbers are still usable
            self.used_graph = False
            msg = f"{type(ex).__name__}: {str(ex)[:160]}"
            if msg not in self._warned:
                self._warned.add(msg)
                print(f"  [warn] CUDA graph capture failed ({msg}); eager timing", file=sys.stderr)
            out = []
            for _ in range(reps):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                for _ in range(iters):
                    fn()
                e.record()
                e.synchronize()
                out.append(s.elapsed_time(e) / iters)
            return out

    def __call__(self, fn, cfg=None) -> list[float]:
        return self._run(fn, self.iters, self.reps, 3)

    def probe(self, fn, cfg=None) -> list[float]:
        return self._run(fn, max(2, self.iters // 2), 3, 1)


def ring_size(nbytes: int, l2_bytes: int, factor: float, free_bytes: int, cap: int = 32) -> int:
    """Number of distinct weight copies so that cycling through them defeats the L2 cache (bounded by free memory)."""
    want = math.ceil(factor * max(l2_bytes, 64 << 20) / max(1, nbytes))
    fit = max(1, int(0.30 * free_bytes // max(1, nbytes)))
    return max(1, min(cap, want, fit))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pl = resolve_plan(args)
    shapes, ms = pl["shapes"], pl["ms"]

    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:
        has_cuda = False

    from forge.kernels.layouts.chip import chip_info

    chip = chip_info(None, args.device) if has_cuda else chip_info(None)
    rows_plan = plan(shapes, ms, args.group, args.level, chip.sm, chip.num_sms, pl["max_candidates"])
    if args.dry_run or not has_cuda:
        print(f"chip: {chip.name} sm={chip.sm} sms={chip.num_sms} fingerprint={chip.fingerprint}")
        print(f"mode: {'quick' if args.quick else 'full'}  shapes={len(shapes)} M={ms} max_candidates/bucket={pl['max_candidates'] or 'unlimited'} "
              f"refine={pl['refine']} prune={pl['prune_factor']} budget={'%.0f min' % (pl['budget_s'] / 60) if pl['budget_s'] else 'unlimited'}")
        for r in rows_plan:
            d = r["default"]
            print(f"{r['role']:>18} N={r['n']:<7} K={r['k']:<6} M={r['m']:<5} bucket={r['bucket']:<5} candidates={r['n_candidates']:<4} "
                  f"default={d['impl']}/{d['tile_m']}x{d['tile_n']}x{d['tile_k']} w{d['warps']} s{d['stages']} sk{d['split_k']}  "
                  f"impls={','.join(r['impls'])}")
        print(f"total candidates: {sum(r['n_candidates'] for r in rows_plan)} over {len(rows_plan)} (shape, M) rows")
        if not has_cuda:
            print("\nNo CUDA device: plan only (nothing measured).")
        return 0 if (args.dry_run or not has_cuda) else 1

    import torch.nn.functional as F  # noqa: N812

    from forge.kernels import ops_gemm
    from forge.kernels.autotune import Autotuner, make_l2_flusher, summarize, time_callable
    from forge.kernels.bench import measure_peak_bandwidth
    from forge.kernels.spec import load_spec
    from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton
    from forge.kernels.w4_tuning import (
        limit_candidates,
        m_bucket,
        refine_candidates,
        w4_candidates,
        w4_tune_tag,
    )

    torch.cuda.set_device(args.device)
    dev = f"cuda:{args.device}"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    spec = load_spec("w4a16_gemm")
    props = torch.cuda.get_device_properties(args.device)
    l2_bytes = int(getattr(props, "L2_cache_size", 0) or 96 << 20)
    peak = measure_peak_bandwidth(dev)
    print(f"chip {chip.name} sm_{chip.sm} {chip.num_sms} SMs, L2 {l2_bytes >> 20} MiB, measured copy bandwidth "
          f"{peak:.0f} GB/s ({'datasheet ' + format(chip.hbm_gbs, '.0f') if chip.hbm_gbs else 'datasheet unknown'}), dtype {args.dtype}")
    timer = GraphTimer(pl["graph_iters"], pl["reps"])
    scratch = None
    if args.no_cache:  # measure only: a throwaway cache file, so the user's real entries are neither read, overwritten nor deleted
        import tempfile

        scratch = tempfile.TemporaryDirectory(prefix="forge_w4_nocache_")
        cache_path: str | Path | None = Path(scratch.name) / "autotune.json"
    else:
        cache_path = args.cache_path
    tuner = Autotuner(cache_path, chip=chip, timer=timer, max_candidates=None,
                      retries=1)  # at most ONE re-measurement of a throttled candidate
    if args.cache_path is not None and not args.no_cache:
        print(f"[warn] custom --cache-path: ops_gemm.tuned_config() only reads {tuner.cache.path.parent}/<chip>.json by default")
    flush = make_l2_flusher()
    dtype_cfg = ops_gemm.dtype_cfg("w4a16_gemm", dtype, args.group, w4_tune_tag(args.act_order, args.no_zeros))
    budget = Budget(pl["budget_s"])
    out_path = args.out or f"w4a16_bench_{time.strftime('%Y%m%d-%H%M%S')}.json"
    sink = ReportSink(out_path, dict(
        chip=chip.to_dict(), dtype=args.dtype, group=args.group, peak_copy_gbs=peak,
        cache="(none: --no-cache)" if args.no_cache else str(tuner.cache.path),
        created=time.strftime("%Y-%m-%d %H:%M:%S"), mode="quick" if args.quick else "full", plan={k: v for k, v in pl.items() if k != "shapes"},
        method="cuda-graph replay over an L2-defeating weight ring; eager numbers include launch overhead"), resume=args.keep)
    if sink.path != Path(out_path):
        print(f"[warn] {out_path} already exists: writing the report to {sink.path} instead (pass --keep to resume/merge into it)")
    table: list[str] = []

    class GpuHooks(SweepHooks):
        def setup(self, role, n, k):
            try:
                p0 = synth_packed(n, k, args.group, dev, zeros=not args.no_zeros, act_order=args.act_order, seed=args.seed)
                dense32 = p0.dequantize(torch.float32)
                dense = dense32.to(dtype)
            except torch.cuda.OutOfMemoryError:
                print(f"[skip] {role} N={n} K={k}: out of memory building the oracle")
                torch.cuda.empty_cache()
                return None
            free = torch.cuda.mem_get_info(args.device)[0]
            ring_w4 = ring_size(p0.nbytes, l2_bytes, args.l2_factor, free // 2)
            ring_bf = ring_size(dense.numel() * dense.element_size(), l2_bytes, args.l2_factor, free // 2)
            packs = [p0] + [clone_packed(p0) for _ in range(ring_w4 - 1)]
            for pk in packs:  # act-order column sort etc. BEFORE any graph capture (it syncs the host)
                ops_gemm.prepare_w4(pk)
            return dict(p0=p0, dense32=dense32, packs=packs, denses=[dense] + [dense.clone() for _ in range(ring_bf - 1)])

        def teardown(self, st):
            del st
            torch.cuda.empty_cache()

        def row(self, st, role, n, k, m, budget):
            if st is None:
                return dict(role=role, n=n, k=k, m=m, bucket=m_bucket(m), error="skipped: out of memory building the oracle")
            p0, dense32, packs, denses = st["p0"], st["dense32"], st["packs"], st["denses"]
            w4_bytes = p0.nbytes
            x = torch.randn(m, k, device=dev, dtype=dtype, generator=torch.Generator(device=dev).manual_seed(args.seed + m))
            ref = x.float() @ dense32.t()
            shape = {"M": m, "N": n, "K": k}
            ctr = {"i": 0}
            timer.used_graph = True

            def run_cfg(cfg, _shape=None):
                ctr["i"] += 1
                return w4a16_gemm_triton(x, packs[ctr["i"] % len(packs)], None, config=cfg, sm=chip.sm, num_sms=chip.num_sms)

            def bf16_call():
                ctr["i"] += 1
                return F.linear(x, denses[ctr["i"] % len(denses)])

            cands = w4_candidates(m, n, k, sm=chip.sm, num_sms=chip.num_sms, group=args.group, level=args.level)
            if pl["max_candidates"]:
                cands = limit_candidates(cands, pl["max_candidates"])
            # store=False everywhere: the cache is written ONCE below with the final best (a refinement round that finds nothing
            # better, or a worse throttled one, can never overwrite a better entry)
            tkw = dict(dtype_cfg=dtype_cfg, layout_id=p0.layout_id, group=args.group, prune_factor=pl["prune_factor"],
                       deadline=budget.deadline, accept_throttled=not args.strict_throttle, store=False)
            res = tuner.tune(spec, shape, run_cfg, lambda s: ref, candidates=cands, force=not args.keep, **tkw)
            best = res.best
            for _ in range(pl["refine"] if (best and not res.from_cache) else 0):  # a cache hit (--keep) is not refined
                if budget.expired() or best.get("impl") == "legacy":
                    break
                extra = refine_candidates(best, m, n, k, sm=chip.sm, group=args.group) + [best]
                r2 = tuner.tune(spec, shape, run_cfg, lambda s: ref, candidates=extra, force=True, **tkw)
                if r2.best_measurement and res.best_measurement and r2.best_measurement.median_ms <= res.best_measurement.median_ms:
                    res, best = r2, r2.best
                elif r2.best:
                    break
            if best and not res.from_cache:
                tuner.commit(res, accept_throttled=not args.strict_throttle)
            rejected = [dict(config=r.config, error=r.error) for r in res.rejected[:5]]
            row: dict[str, Any] = dict(role=role, n=n, k=k, m=m, bucket=m_bucket(m), best=best, n_measured=len(res.measurements),
                                       n_rejected=len(res.rejected), n_pruned=len(res.pruned), rejected_sample=rejected,
                                       from_cache=res.from_cache, cached=res.cached, notes=res.notes)
            if best is None:
                row["error"] = "no candidate passed the correctness gate"
                table.append(f"{role:>18} N={n:<7} K={k:<6} M={m:<5} NO VALID CONFIG ({len(res.rejected)} rejected)")
                print(table[-1], flush=True)
                return row
            row["throttled"] = bool(res.best_measurement and res.best_measurement.throttled)
            graph_ms = lambda fn: float(summarize(timer(fn)).median)  # noqa: E731
            bf_ms = graph_ms(bf16_call)
            tuned_ms = graph_ms(lambda: run_cfg(best))
            legacy_ms = graph_ms(lambda: run_cfg({"impl": "legacy"}))
            dflt_ms = graph_ms(lambda: run_cfg(None))
            dq_us = None
            if best["impl"] != "dequant_matmul" and m >= 8:  # the dequant floor at its default tile, for the routing threshold
                from forge.kernels.w4_tuning import normalise_w4_config

                try:
                    dq_us = graph_ms(lambda: run_cfg(normalise_w4_config({"impl": "dequant_matmul"}, m))) * 1e3
                except Exception:  # noqa: BLE001
                    dq_us = None
            eager_tuned = float(summarize(time_callable(lambda: run_cfg(best), warmup=3, repeats=pl["eager_reps"], flush=flush)).median)
            eager_bf = float(summarize(time_callable(bf16_call, warmup=3, repeats=pl["eager_reps"], flush=flush)).median)
            ops_gemm.clear_config_memo()
            picked = ops_gemm.tuned_config("w4a16_gemm", chip, m, n, k, dtype_cfg_str=dtype_cfg, layout_id=p0.layout_id)
            nbytes = w4_bytes + x.numel() * x.element_size() + m * n * x.element_size()
            gbs = nbytes / (tuned_ms * 1e-3) / 1e9
            row.update(bf16_us=bf_ms * 1e3, legacy_us=legacy_ms * 1e3, default_us=dflt_ms * 1e3, tuned_us=tuned_ms * 1e3, dequant_us=dq_us,
                       speedup_vs_bf16=bf_ms / tuned_ms, speedup_legacy_vs_bf16=bf_ms / legacy_ms, speedup_default_vs_bf16=bf_ms / dflt_ms,
                       eager_tuned_us=eager_tuned * 1e3, eager_bf16_us=eager_bf * 1e3, eager_speedup=eager_bf / eager_tuned,
                       achieved_gbs=gbs, pct_peak=gbs / peak if peak else None, bytes=nbytes,
                       dispatcher_picks_cache=bool(picked) and picked == best, graph_timing=timer.used_graph,
                       top=[dict(config=q.config, us=q.median_ms * 1e3) for q in sorted(res.measurements, key=lambda q: q.median_ms)[:5]])
            c = best
            table.append(f"{role:>18} N={n:<7} K={k:<6} M={m:<5} bf16 {bf_ms * 1e3:8.1f}us legacy {legacy_ms * 1e3:8.1f}us "
                         f"dflt {dflt_ms * 1e3:8.1f}us tuned {tuned_ms * 1e3:8.1f}us  x{bf_ms / tuned_ms:5.2f} (legacy x{bf_ms / legacy_ms:4.2f}, "
                         f"eager x{eager_bf / eager_tuned:4.2f})  {gbs:6.0f} GB/s {100 * gbs / peak:4.0f}%  "
                         f"{c['impl']} {c['tile_m']}x{c['tile_n']}x{c['tile_k']} w{c['warps']} s{c['stages']} sk{c['split_k']}"
                         f"{'' if row['dispatcher_picks_cache'] or args.no_cache else '  [cache not picked up!]'}"
                         f"{'  [throttled]' if row['throttled'] else ''}{'' if timer.used_graph else '  [eager timing]'}")
            print(table[-1], flush=True)
            return row

    try:
        stopped = sweep(shapes, ms, GpuHooks(), sink, budget, skip_finished=args.keep)
    finally:
        if scratch is not None:
            scratch.cleanup()
    sink.finish(stopped)
    print("\n".join(["", "=" * 40 + " summary " + "=" * 40, *table]))
    print("\n".join(format_summary(sink.report["summary"])))
    if stopped:
        print(f"\nSTOPPED EARLY ({stopped}) after {budget.elapsed_min():.1f} min: {len(sink.report['rows'])} row(s) done; "
              "tuned configs found so far are in the cache. Re-run (add --keep to skip finished rows).")
    print(f"report: {sink.path.resolve()}\nautotune cache: {'(none: --no-cache)' if args.no_cache else tuner.cache.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
