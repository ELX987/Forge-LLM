"""Multi-process collectives check (the old tern158 ``nccl_test.py``): every collective the sharded trainer uses, with known answers.

``python -m forge.train.collectives [--backend nccl|gloo|auto] [--nproc N] [--size-mib 16]`` spawns ``N`` ranks, runs ``broadcast``,
``all_reduce``, ``all_gather`` (uint8 packed codes), ``reduce_scatter_tensor`` (NCCL only; gloo is recorded as unsupported, not as a
failure) and times an ``all_reduce``.  Also registered as probe check ``collectives`` (needs >= 2 GPUs; otherwise reported as skipped,
never as passed).  The gloo run is CPU-tested; the NCCL run is validation gate G1 and has not been executed in this repository.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

__all__ = ["main", "probe_check", "run_collectives_check"]


def _rank_main(rank: int, world: int, backend: str, init_file: str, size_mib: float, out_q: Any) -> None:
    import torch
    import torch.distributed as dist

    res: dict[str, Any] = {"rank": rank}
    try:
        dev = torch.device("cuda", rank) if backend == "nccl" else torch.device("cpu")
        if dev.type == "cuda":
            torch.cuda.set_device(dev)
        dist.init_process_group(backend, init_method=Path(init_file).as_uri(), rank=rank, world_size=world)
        checks: dict[str, Any] = {}

        t = torch.full((8,), float(rank + 1), device=dev)
        dist.broadcast(t, src=0)
        checks["broadcast"] = bool(torch.all(t == 1.0))

        t = torch.full((1024,), float(rank + 1), device=dev)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        checks["all_reduce"] = bool(torch.all(t == world * (world + 1) / 2))

        mine = torch.full((4096,), rank + 1, dtype=torch.uint8, device=dev)
        outs = [torch.empty_like(mine) for _ in range(world)]
        dist.all_gather(outs, mine)
        checks["all_gather_uint8"] = all(bool(torch.all(o == r + 1)) for r, o in enumerate(outs))

        try:
            inp = torch.cat([torch.full((64, 4), float(rank + 1) * (r + 1), device=dev) for r in range(world)])
            out = torch.empty(64, 4, device=dev)
            dist.reduce_scatter_tensor(out, inp, op=dist.ReduceOp.SUM)
            checks["reduce_scatter"] = bool(torch.all(out == (rank + 1) * world * (world + 1) / 2))
        except (RuntimeError, NotImplementedError, ValueError) as e:  # gloo has no reduce_scatter_tensor in every torch build
            checks["reduce_scatter"] = None
            res["reduce_scatter_note"] = f"unsupported on {backend}: {str(e)[:120]}"

        n = max(1, int(size_mib * 2**20 // 4))
        buf = torch.ones(n, device=dev)
        for _ in range(2):
            dist.all_reduce(buf)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        iters = 5
        for _ in range(iters):
            dist.all_reduce(buf)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        dt = (time.perf_counter() - t0) / iters
        res["all_reduce_ms"] = 1000.0 * dt
        res["all_reduce_busbw_gbs"] = (2.0 * (world - 1) / world * n * 4 / dt / 1e9) if dt > 0 else None
        dist.barrier()
        res["checks"] = checks
        dist.destroy_process_group()
    except BaseException:  # noqa: BLE001
        import traceback

        res["error"] = traceback.format_exc()[-600:]
    out_q.put(res)


def run_collectives_check(backend: str = "auto", nproc: int | None = None, size_mib: float = 16.0, timeout: float = 180.0) -> dict[str, Any]:
    """Spawn the ranks and return ``{ok, backend, world, checks, all_reduce_busbw_gbs, ...}``; ``ok`` is None when it could not run."""
    import torch
    import torch.multiprocessing as mp

    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else "gloo"
    if backend == "nccl":
        have = torch.cuda.device_count() if torch.cuda.is_available() else 0
        nproc = nproc or have
        if have < 2 or nproc > have:
            return {"ok": None, "backend": backend, "skipped": f"nccl needs >= 2 visible GPUs (have {have}, requested {nproc})"}
    nproc = max(2, int(nproc or 2))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    init_file = str(Path(tempfile.mkdtemp(prefix="forge_coll_")) / "rdzv")
    procs = [ctx.Process(target=_rank_main, args=(r, nproc, backend, init_file, size_mib, q)) for r in range(nproc)]
    for p in procs:
        p.start()
    results: list[dict[str, Any]] = []
    try:
        for _ in procs:
            results.append(q.get(timeout=timeout))
    except Exception:  # noqa: BLE001 - queue.Empty: a rank never reported (hang or crash)
        for p in procs:
            if p.is_alive():
                p.terminate()
        return {"ok": False, "backend": backend, "world": nproc, "error": f"ranks did not report within {timeout:.0f} s (exit codes {[p.exitcode for p in procs]})"}
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    errs = [r["error"] for r in results if r.get("error")]
    if errs:
        return {"ok": False, "backend": backend, "world": nproc, "error": errs[0]}
    agg: dict[str, Any] = {}
    for name in results[0]["checks"]:
        vals = [r["checks"][name] for r in results]
        agg[name] = None if all(v is None for v in vals) else all(v is True for v in vals)
    bw = [r["all_reduce_busbw_gbs"] for r in results if r.get("all_reduce_busbw_gbs")]
    out = {"ok": all(v is not False for v in agg.values()) and any(v is True for v in agg.values()), "backend": backend, "world": nproc, "checks": agg,
           "all_reduce_ms": max(r["all_reduce_ms"] for r in results), "all_reduce_busbw_gbs": min(bw) if bw else None, "size_mib": size_mib}
    notes = [r["reduce_scatter_note"] for r in results if r.get("reduce_scatter_note")]
    if notes:
        out["notes"] = notes[:1]
    if backend == "nccl" and agg.get("reduce_scatter") is not True:
        out["ok"] = False  # the zero_ternary NCCL path depends on reduce_scatter_tensor
    return out


def probe_check(device: int, quick: bool) -> dict[str, Any]:
    """``forge probe`` extra check: NCCL collectives across all visible GPUs (skipped, not passed, with fewer than two)."""
    return run_collectives_check("nccl", None, size_mib=4.0 if quick else 64.0, timeout=120.0 if quick else 300.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forge.train.collectives", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="auto")
    ap.add_argument("--nproc", type=int, default=None)
    ap.add_argument("--size-mib", type=float, default=16.0)
    ap.add_argument("--timeout", type=float, default=180.0)
    ns = ap.parse_args(argv)
    res = run_collectives_check(ns.backend, ns.nproc, ns.size_mib, ns.timeout)
    print(json.dumps(res, indent=1))
    return 0 if res.get("ok") else (2 if res.get("ok") is None else 1)


if __name__ == "__main__":  # pragma: no cover
    if sys.platform.startswith("linux"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")  # avoids hostname-resolution stalls on rented containers
    sys.exit(main())
