"""GPU micro-benchmarks that measure what a device *achieves* (Plan HW-3, HW-6): stream-copy bandwidth with an L2
flush, tensor-core GEMM throughput per dtype, an on-chip SRAM re-read proxy, kernel launch latency, a dequantisation
ALU benchmark, and peer-to-peer copy bandwidth.

STATUS: implemented, **not validated on NVIDIA hardware by the author** (the development machine has no NVIDIA
GPU).  Every function is guarded: with no CUDA device, missing triton, an unsupported dtype or any runtime error
it returns a :class:`BenchResult` with ``ok=False`` and a reason - it never raises and never fabricates numbers.
Numerical correctness of each GEMM path is checked on a small problem before timing (all-ones inputs make the
expected output exactly K), so a path that "runs" but computes garbage is reported as failed.

Public API (all take ``device=0``)::

    stream_copy_bw(...)        GB/s     achievable HBM/GDDR copy bandwidth (bytes read + written / time)
    mma_throughput(dtype, ...) TFLOP/s  dtype in fp32, tf32, fp16, bf16, fp8, int8, fp4 (TOP/s for int8)
    smem_bw(...)               GB/s     on-chip SRAM re-read proxy (needs triton)
    launch_latency(...)        us       GPU-side per-kernel time of back-to-back tiny kernels (CUDA graph)
    dequant_alu_throughput(..) Gelem/s  int4 dequantisation rate (needs triton) + derived simple-ops/s
    p2p_bw(src, dst, ...)      GB/s     device-to-device copy bandwidth

Requires torch >= 2.1 (``torch._scaled_mm`` FP8 on sm_89+, ``torch._int_mm`` INT8, FP4 needs torch >= 2.8 with
``torch.float4_e2m1fn_x2``).
"""
from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

log = logging.getLogger("forge.hw.microbench")

_DTYPE_ALIASES = {
    "bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "half": "fp16", "fp16": "fp16",
    "float32": "fp32", "fp32": "fp32", "tf32": "tf32", "fp8": "fp8", "fp8_e4m3": "fp8", "float8_e4m3fn": "fp8",
    "e4m3": "fp8", "int8": "int8", "i8": "int8", "fp4": "fp4", "nvfp4": "fp4", "mxfp4": "fp4", "fp4_e2m1": "fp4",
}
SUPPORTED_MMA_DTYPES = ("fp32", "tf32", "fp16", "bf16", "fp8", "int8", "fp4")


def canon_dtype(d: str) -> str:
    """'bfloat16' -> 'bf16', 'float8_e4m3fn' -> 'fp8', 'nvfp4' -> 'fp4' ..."""
    k = str(d).lower().replace("torch.", "")
    if k not in _DTYPE_ALIASES:
        raise ValueError(f"unknown dtype {d!r}; choose from {sorted(set(_DTYPE_ALIASES.values()))}")
    return _DTYPE_ALIASES[k]


@dataclass
class BenchResult:
    """Outcome of one micro-benchmark.  ``value`` is None unless ``ok``; ``provenance`` is always 'measured'
    for numbers produced here."""

    name: str
    ok: bool
    value: float | None = None
    unit: str = ""
    samples: list[float] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    skipped: bool = False
    duration_s: float = 0.0
    provenance: str = "measured"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["samples"] = [round(x, 4) for x in self.samples[:64]]  # keep reports small
        return d


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    q1, q3 = s[n // 4], s[(3 * n) // 4] if n >= 4 else s[-1]
    half = xs[n // 2:] if n >= 4 else xs
    return {"median": statistics.median(s), "min": s[0], "max": s[-1], "mean": statistics.fmean(s), "iqr": q3 - q1,
            "n": float(n), "sustained_median": statistics.median(half)}


def _fail(name: str, unit: str, msg: str, t0: float, *, skipped: bool = False, params: dict | None = None) -> BenchResult:
    return BenchResult(name=name, ok=False, unit=unit, error=msg, skipped=skipped, params=params or {},
                       duration_s=time.time() - t0)


def cuda_ready(device: int = 0) -> tuple[bool, str]:
    """(usable, reason).  True only if torch is importable, CUDA is available and ``device`` exists."""
    try:
        import torch
    except Exception as e:
        return False, f"torch unavailable: {e}"
    try:
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        if not 0 <= device < torch.cuda.device_count():
            return False, f"CUDA device {device} not found"
    except Exception as e:
        return False, f"CUDA query failed: {e}"
    return True, ""


def triton_ready() -> tuple[bool, str]:
    try:
        import triton  # noqa: F401
    except Exception as e:
        return False, f"triton unavailable: {e}"
    return True, ""


def _free_bytes(device: int) -> int:
    import torch

    try:
        return int(torch.cuda.mem_get_info(device)[0])
    except Exception:
        return 2**30


def _l2_bytes(device: int) -> int:
    import torch

    try:
        v = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", 0)
        if v:
            return int(v)
    except Exception:
        pass
    return 64 * 2**20  # conservative guess: bigger than every pre-Blackwell L2 except H200/B200


def _timed_events(fn: Callable[[], None], iters: int, warmup: int, device: int,
                  between: Callable[[], None] | None = None, max_seconds: float | None = None) -> list[float]:
    """Run ``fn`` ``iters`` times (after ``warmup``) timing each call with CUDA events; returns ms per call.
    ``between`` runs (untimed, synchronised) before every timed call (e.g. an L2 flush)."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    out: list[float] = []
    t_end = time.time() + max_seconds if max_seconds else None
    for _ in range(iters):
        if between is not None:
            between()
            torch.cuda.synchronize(device)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        fn()
        e1.record()
        e1.synchronize()
        out.append(e0.elapsed_time(e1))
        if t_end is not None and time.time() > t_end and len(out) >= 3:
            break
    return out


# --------------------------------------------------------------------------------------------
# stream copy bandwidth
# --------------------------------------------------------------------------------------------
def stream_copy_bw(device: int = 0, *, size_mb: float | None = None, iters: int = 30, warmup: int = 5,
                   flush_l2: bool = True, kind: str = "copy", max_seconds: float = 4.0) -> BenchResult:
    """Achievable memory bandwidth in GB/s (1e9 B/s).

    ``kind='copy'`` times ``dst.copy_(src)`` (bytes = read + write); ``kind='write'`` times ``dst.fill_(x)``.
    Buffers are >= 8x the L2 size (so L2 hits are negligible) and, with ``flush_l2``, a 2x-L2 buffer is zeroed
    before every timed iteration.  Reports the median over iterations (``stats['max']`` is the best iteration)."""
    name = f"stream_{kind}_bw"
    t0 = time.time()
    ok, why = cuda_ready(device)
    if not ok:
        return _fail(name, "GB/s", why, t0, skipped=True)
    try:
        import torch

        with torch.cuda.device(device):
            l2 = _l2_bytes(device)
            free = _free_bytes(device)
            want = int(size_mb * 2**20) if size_mb else max(256 * 2**20, 8 * l2)
            size = max(8 * 2**20, min(want, 2**30, int(free * 0.2)))
            n = size // 4
            src = torch.empty(n, dtype=torch.float32, device="cuda")
            src.fill_(1.0)
            dst = torch.empty_like(src)
            flush = torch.empty((2 * l2) // 4, dtype=torch.float32, device="cuda") if flush_l2 else None

            if kind == "copy":
                def fn() -> None:
                    dst.copy_(src)
                moved = 2.0 * n * 4
            elif kind == "write":
                def fn() -> None:
                    dst.fill_(2.0)
                moved = 1.0 * n * 4
            else:
                return _fail(name, "GB/s", f"unknown kind {kind!r}", t0)
            between = (lambda: flush.zero_()) if flush is not None else None  # type: ignore[union-attr]
            ms = _timed_events(fn, iters, warmup, device, between, max_seconds)
            gbs = [moved / (m * 1e-3) / 1e9 for m in ms if m > 0]
            if kind == "copy" and not bool(torch.equal(dst[:1024], src[:1024])):
                return _fail(name, "GB/s", "copy result mismatch", t0)
            st = _stats(gbs)
            del src, dst, flush
            torch.cuda.empty_cache()
        return BenchResult(name=name, ok=bool(gbs), value=st.get("median"), unit="GB/s", samples=gbs, stats=st,
                           params={"bytes_per_buffer": size, "l2_bytes": l2, "flush_l2": flush_l2, "kind": kind,
                                   "iters": len(gbs)}, duration_s=time.time() - t0,
                           error=None if gbs else "no samples")
    except Exception as e:
        log.debug("stream_copy_bw failed", exc_info=True)
        return _fail(name, "GB/s", repr(e), t0)


# --------------------------------------------------------------------------------------------
# GEMM throughput
# --------------------------------------------------------------------------------------------
def _mma_dims(dtype: str, device: int, quick: bool) -> tuple[int, int, int]:
    free = _free_bytes(device)
    big = 8192 if (not quick and free > 3 * 2**30) else 4096
    if free < 512 * 2**20:
        big = 2048
    if dtype in ("fp32", "tf32") and big > 4096:
        big = 4096 if free < 2**31 else 6144
    return big, big, big


def mma_throughput(dtype: str = "bf16", device: int = 0, *, m: int | None = None, n: int | None = None,
                   k: int | None = None, iters: int | None = None, warmup: int = 3, max_seconds: float = 2.0,
                   quick: bool = False) -> BenchResult:
    """Sustained dense GEMM throughput in TFLOP/s (TOP/s for int8) of ``dtype``.

    fp16/bf16/fp32/tf32 use ``torch.matmul`` (FP32 accumulation; tf32 toggles
    ``torch.backends.cuda.matmul.allow_tf32``), fp8 uses ``torch._scaled_mm`` (e4m3, per-tensor scale 1.0, default
    accumulation), int8 uses ``torch._int_mm``, fp4 tries ``torch._scaled_mm`` with ``torch.float4_e2m1fn_x2``
    and block scales (NVFP4: e4m3 / 16, MXFP4: e8m0 / 32).  ``stats['sustained_median']`` is the median over the
    second half of the run (compare with ``median`` to see throttling)."""
    t0 = time.time()
    try:
        dt = canon_dtype(dtype)
    except ValueError as e:
        return _fail(f"mma_{dtype}", "TFLOP/s", str(e), t0)
    name = f"mma_{dt}"
    unit = "TOP/s" if dt == "int8" else "TFLOP/s"
    ok, why = cuda_ready(device)
    if not ok:
        return _fail(name, unit, why, t0, skipped=True)
    try:
        import torch

        cc = torch.cuda.get_device_capability(device)
        sm = cc[0] * 10 + cc[1]
        if dt == "bf16" and sm < 80:
            return _fail(name, unit, f"bf16 tensor cores need sm_80+ (device is sm_{sm})", t0, skipped=True)
        if dt in ("tf32",) and sm < 80:
            return _fail(name, unit, f"tf32 needs sm_80+ (device is sm_{sm})", t0, skipped=True)
        if dt == "fp8" and sm < 89:
            return _fail(name, unit, f"fp8 MMA needs sm_89+ (device is sm_{sm})", t0, skipped=True)
        if dt == "int8" and sm < 75:
            return _fail(name, unit, f"int8 tensor cores need sm_75+ (device is sm_{sm})", t0, skipped=True)
        if dt == "fp4" and sm < 100:
            return _fail(name, unit, f"block-scaled fp4 MMA needs sm_100+ (device is sm_{sm})", t0, skipped=True)
        dm, dn, dk = _mma_dims(dt, device, quick)
        m, n, k = m or dm, n or dn, k or dk
        iters = iters or (6 if quick else 20)
        with torch.cuda.device(device):
            runner, params = _make_gemm(dt, m, n, k, torch)
            if isinstance(runner, str):  # reason string
                return _fail(name, unit, runner, t0, skipped=True, params=params)
            fn, check, restore = runner
            try:
                err = check()
                if err:
                    return _fail(name, unit, f"numerical check failed: {err}", t0, params=params)
                ms = _timed_events(fn, iters, warmup, device, None, max_seconds)
            finally:
                restore()
            flop = 2.0 * m * n * k
            tf = [flop / (x * 1e-3) / 1e12 for x in ms if x > 0]
            torch.cuda.empty_cache()
        st = _stats(tf)
        params.update(m=m, n=n, k=k, iters=len(tf), sm=sm)
        return BenchResult(name=name, ok=bool(tf), value=st.get("median"), unit=unit, samples=tf, stats=st,
                           params=params, duration_s=time.time() - t0, error=None if tf else "no samples")
    except Exception as e:
        log.debug("mma_throughput(%s) failed", dtype, exc_info=True)
        return _fail(name, unit, repr(e), t0)


def _make_gemm(dt: str, m: int, n: int, k: int, torch: Any):
    """Return ((timed_fn, check_fn, restore_fn), params) or (reason_str, params)."""
    dev = "cuda"
    params: dict[str, Any] = {"dtype": dt}
    noop = lambda: None  # noqa: E731

    if dt in ("fp16", "bf16", "fp32", "tf32"):
        tdt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "tf32": torch.float32}[dt]
        a = torch.randn(m, k, device=dev, dtype=tdt)
        b = torch.randn(k, n, device=dev, dtype=tdt)
        c = torch.empty(m, n, device=dev, dtype=tdt)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = dt == "tf32"
        params["accumulate"] = "fp32"

        def fn() -> None:
            torch.matmul(a, b, out=c)

        def check() -> str | None:
            sa, sb = a[:128, :256].float(), b[:256, :128].float()
            got = torch.matmul(a[:128, :256], b[:256, :128]).float()
            ref = sa @ sb
            tol = {"fp32": 1e-3, "tf32": 5e-2, "fp16": 5e-2, "bf16": 1.5e-1}[dt]
            rel = float((got - ref).norm() / ref.norm().clamp_min(1e-12))
            return None if rel < tol else f"rel err {rel:.3g} > {tol}"

        def restore() -> None:
            torch.backends.cuda.matmul.allow_tf32 = prev

        return (fn, check, restore), params

    if dt == "int8":
        if not hasattr(torch, "_int_mm"):
            return "torch._int_mm not available", params
        a = torch.ones(m, k, device=dev, dtype=torch.int8)
        b = torch.ones(k, n, device=dev, dtype=torch.int8)
        params["accumulate"] = "int32"

        def fn() -> None:
            torch._int_mm(a, b)

        def check() -> str | None:
            out = torch._int_mm(a[:64, :256].contiguous(), b[:256, :64].contiguous())
            return None if bool((out == 256).all()) else "int8 _int_mm result != K"

        return (fn, check, noop), params

    if dt == "fp8":
        if not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
            return "torch lacks float8_e4m3fn / _scaled_mm", params
        a = torch.ones(m, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        bt = torch.ones(n, k, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        sa = torch.tensor(1.0, device=dev, dtype=torch.float32)
        sb = torch.tensor(1.0, device=dev, dtype=torch.float32)
        params["accumulate"] = "fp32 (default, not fast-accum)"

        def run(x: Any, wt: Any) -> Any:
            r = torch._scaled_mm(x, wt.t(), scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            return r[0] if isinstance(r, tuple) else r

        def fn() -> None:
            run(a, bt)

        def check() -> str | None:
            out = run(a[:64, :256].contiguous(), bt[:64, :256].contiguous())
            return None if bool((out.float() == 256.0).all()) else "fp8 _scaled_mm result != K"

        return (fn, check, noop), params

    if dt == "fp4":
        if not hasattr(torch, "float4_e2m1fn_x2") or not hasattr(torch, "_scaled_mm"):
            return "torch lacks float4_e2m1fn_x2 (needs torch >= 2.8)", params
        errors: list[str] = []
        for recipe in ("nvfp4", "mxfp4"):
            try:
                built = _build_fp4(recipe, m, n, k, torch)
            except Exception as e:  # recipe not supported by this torch build
                errors.append(f"{recipe}: {e!r}")
                continue
            fn, check = built
            err = None
            try:
                err = check()
            except Exception as e:
                err = repr(e)
            if err is None:
                params["recipe"] = recipe
                params["accumulate"] = "fp32"
                return (fn, check, noop), params
            errors.append(f"{recipe}: {err}")
        return "no fp4 _scaled_mm recipe worked on this torch/driver: " + "; ".join(errors), params

    return f"unsupported dtype {dt}", params


def _round_up(x: int, y: int) -> int:
    return (x + y - 1) // y * y


def _build_fp4(recipe: str, m: int, n: int, k: int, torch: Any):
    """FP4 GEMM via ``torch._scaled_mm`` with all-ones data and unit scales (swizzle-agnostic)."""
    dev = "cuda"
    kb = 16 if recipe == "nvfp4" else 32
    if recipe == "nvfp4":
        def scale(rows: int) -> Any:
            return torch.ones(_round_up(rows, 128) * _round_up(k // kb, 4), device=dev,
                              dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    else:
        e8 = getattr(torch, "float8_e8m0fnu", None)
        if e8 is None:
            raise RuntimeError("torch lacks float8_e8m0fnu")

        def scale(rows: int) -> Any:
            return torch.full((_round_up(rows, 128) * _round_up(k // kb, 4),), 127, device=dev,
                              dtype=torch.uint8).view(e8)
    # E2M1 value 1.0 has code 0b0010; two codes per byte -> 0x22
    a = torch.full((m, k // 2), 0x22, device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    bt = torch.full((n, k // 2), 0x22, device=dev, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    sa, sb = scale(m), scale(n)

    def run(x: Any, wt: Any, s1: Any, s2: Any) -> Any:
        r = torch._scaled_mm(x, wt.t(), scale_a=s1, scale_b=s2, out_dtype=torch.bfloat16)
        return r[0] if isinstance(r, tuple) else r

    def fn() -> None:
        run(a, bt, sa, sb)

    def check() -> str | None:
        out = run(a, bt, sa, sb)
        return None if bool((out[:64, :64].float() == float(k)).all()) else f"{recipe} result != K"

    return fn, check


# --------------------------------------------------------------------------------------------
# on-chip SRAM re-read proxy and dequant ALU (triton)
# --------------------------------------------------------------------------------------------
def smem_bw(device: int = 0, *, reps: int = 2048, iters: int = 10, warmup: int = 3,
            programs_per_sm: int = 4) -> BenchResult:
    """On-chip SRAM re-read bandwidth proxy (GB/s) - see :mod:`forge.hw._bench_kernels` for what it does and
    does not measure.  Each program re-reads a 16 KiB working set ``reps`` times; bytes = programs x 16 KiB x reps."""
    name = "smem_bw_proxy"
    t0 = time.time()
    ok, why = cuda_ready(device)
    if not ok:
        return _fail(name, "GB/s", why, t0, skipped=True)
    tok, twhy = triton_ready()
    if not tok:
        return _fail(name, "GB/s", twhy, t0, skipped=True)
    try:
        import torch

        from forge.hw import _bench_kernels as bk

        BLOCK, NT = 1024, 4
        with torch.cuda.device(device):
            nsm = torch.cuda.get_device_properties(device).multi_processor_count
            grid = nsm * programs_per_sm
            x = torch.randn(grid * BLOCK * NT, device="cuda", dtype=torch.float32)
            out = torch.empty(grid * BLOCK, device="cuda", dtype=torch.float32)

            def fn() -> None:
                bk.onchip_reread_kernel[(grid,)](x, out, BLOCK=BLOCK, NT=NT, REPS=reps)

            fn()
            torch.cuda.synchronize(device)
            # correctness: out = sum over reps of tile (i % NT)
            ref = torch.zeros(BLOCK, device="cuda")
            for i in range(reps):
                ref += x[(i % NT) * BLOCK:(i % NT + 1) * BLOCK]
            if not torch.allclose(out[:BLOCK], ref, rtol=1e-3, atol=1e-2 * max(1.0, reps ** 0.5)):
                return _fail(name, "GB/s", "on-chip kernel numerical check failed", t0)
            ms = _timed_events(fn, iters, warmup, device)
            nbytes = float(grid) * BLOCK * 4 * reps
            gbs = [nbytes / (m_ * 1e-3) / 1e9 for m_ in ms if m_ > 0]
            del x, out
            torch.cuda.empty_cache()
        st = _stats(gbs)
        return BenchResult(name=name, ok=bool(gbs), value=st.get("median"), unit="GB/s", samples=gbs, stats=st,
                           params={"grid": grid, "reps": reps, "working_set_bytes_per_program": BLOCK * NT * 4,
                                   "note": "L1/shared SRAM re-read proxy; Triton cannot address shared memory"},
                           duration_s=time.time() - t0, error=None if gbs else "no samples")
    except Exception as e:
        log.debug("smem_bw failed", exc_info=True)
        return _fail(name, "GB/s", repr(e), t0)


def dequant_alu_throughput(device: int = 0, *, reps: int = 256, iters: int = 10, warmup: int = 3,
                           programs_per_sm: int = 8) -> BenchResult:
    """int4 -> fp32 dequantisation rate in Gelem/s (``params['simple_ops_per_s_t']`` = 5 ops/element, in 1e12/s)."""
    name = "dequant_alu"
    t0 = time.time()
    ok, why = cuda_ready(device)
    if not ok:
        return _fail(name, "Gelem/s", why, t0, skipped=True)
    tok, twhy = triton_ready()
    if not tok:
        return _fail(name, "Gelem/s", twhy, t0, skipped=True)
    try:
        import torch

        from forge.hw import _bench_kernels as bk

        BLOCK = 1024
        with torch.cuda.device(device):
            nsm = torch.cuda.get_device_properties(device).multi_processor_count
            grid = nsm * programs_per_sm
            w = torch.randint(-(2**31), 2**31 - 1, (grid * BLOCK,), device="cuda", dtype=torch.int32)
            out = torch.empty(grid * BLOCK, device="cuda", dtype=torch.float32)
            scale = 0.5

            def fn() -> None:
                bk.dequant_alu_kernel[(grid,)](w, out, scale, BLOCK=BLOCK, REPS=reps)

            fn()
            torch.cuda.synchronize(device)
            # numerical check of the first 256 words against a torch reference
            ww = w[:256].to(torch.int64)
            ref = torch.zeros(256, device="cuda", dtype=torch.float64)
            for r in range(reps):
                x = ww ^ r
                for j in range(8):
                    q = (x >> (4 * j)) & 0xF
                    ref += (q - 8).double() * scale
            if not torch.allclose(out[:256].double(), ref, rtol=1e-4, atol=1e-2):
                return _fail(name, "Gelem/s", "dequant kernel numerical check failed", t0)
            ms = _timed_events(fn, iters, warmup, device)
            elems = float(grid) * BLOCK * 8 * reps
            ge = [elems / (m_ * 1e-3) / 1e9 for m_ in ms if m_ > 0]
            del w, out
            torch.cuda.empty_cache()
        st = _stats(ge)
        med = st.get("median", 0.0)
        return BenchResult(name=name, ok=bool(ge), value=med, unit="Gelem/s", samples=ge, stats=st,
                           params={"grid": grid, "reps": reps, "ops_per_element": 5,
                                   "simple_ops_per_s_t": round(med * 5 / 1e3, 3)},
                           duration_s=time.time() - t0, error=None if ge else "no samples")
    except Exception as e:
        log.debug("dequant_alu_throughput failed", exc_info=True)
        return _fail(name, "Gelem/s", repr(e), t0)


# --------------------------------------------------------------------------------------------
# launch latency
# --------------------------------------------------------------------------------------------
def launch_latency(device: int = 0, *, n_kernels: int = 1000, repeats: int = 5) -> BenchResult:
    """Per-kernel GPU-side time (microseconds) of ``n_kernels`` back-to-back 1-element kernels replayed from a CUDA
    graph (removes Python/driver launch cost); ``params['eager_us_per_launch']`` is the wall-clock cost of
    launching the same kernels eagerly (CPU enqueue rate).  ``value`` is the graph number, or the eager one if
    graph capture fails."""
    name = "launch_latency"
    t0 = time.time()
    ok, why = cuda_ready(device)
    if not ok:
        return _fail(name, "us", why, t0, skipped=True)
    try:
        import torch

        with torch.cuda.device(device):
            x = torch.zeros(1, device="cuda")
            for _ in range(20):
                x.add_(1)
            torch.cuda.synchronize(device)
            eager = []
            for _ in range(repeats):
                s = time.perf_counter()
                for _ in range(n_kernels):
                    x.add_(1)
                torch.cuda.synchronize(device)
                eager.append((time.perf_counter() - s) / n_kernels * 1e6)
            graph_us: list[float] = []
            gerr = None
            try:
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        x.add_(1)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize(device)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for _ in range(n_kernels):
                        x.add_(1)
                g.replay()
                torch.cuda.synchronize(device)
                for _ in range(repeats):
                    e0 = torch.cuda.Event(enable_timing=True)
                    e1 = torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    graph_us.append(e0.elapsed_time(e1) * 1e3 / n_kernels)
            except Exception as e:
                gerr = repr(e)
        samples = graph_us or eager
        st = _stats(samples)
        return BenchResult(name=name, ok=True, value=st["median"], unit="us", samples=samples, stats=st,
                           params={"n_kernels": n_kernels, "eager_us_per_launch": statistics.median(eager),
                                   "method": "cuda_graph" if graph_us else "eager", "graph_error": gerr},
                           duration_s=time.time() - t0)
    except Exception as e:
        log.debug("launch_latency failed", exc_info=True)
        return _fail(name, "us", repr(e), t0)


# --------------------------------------------------------------------------------------------
# peer to peer
# --------------------------------------------------------------------------------------------
def p2p_bw(src: int = 0, dst: int = 1, *, size_mb: int = 256, iters: int = 10, warmup: int = 2) -> BenchResult:
    """Device-to-device copy bandwidth in GB/s from ``src`` to ``dst`` (NVLink or PCIe, whichever the driver
    routes over).  Skipped if fewer than two GPUs are visible."""
    name = f"p2p_bw_{src}_{dst}"
    t0 = time.time()
    ok, why = cuda_ready(max(src, dst))
    if not ok:
        return _fail(name, "GB/s", why, t0, skipped=True)
    try:
        import torch

        n = size_mb * 2**20 // 4
        can = bool(torch.cuda.can_device_access_peer(src, dst))
        a = torch.ones(n, device=f"cuda:{src}")
        b = torch.empty(n, device=f"cuda:{dst}")

        def fn() -> None:
            b.copy_(a, non_blocking=True)

        for _ in range(warmup):
            fn()
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        gbs = []
        for _ in range(iters):
            s = time.perf_counter()
            fn()
            torch.cuda.synchronize(src)
            torch.cuda.synchronize(dst)
            gbs.append(n * 4 / (time.perf_counter() - s) / 1e9)
        st = _stats(gbs)
        return BenchResult(name=name, ok=True, value=st["median"], unit="GB/s", samples=gbs, stats=st,
                           params={"peer_access": can, "size_mb": size_mb}, duration_s=time.time() - t0)
    except Exception as e:
        return _fail(name, "GB/s", repr(e), t0)
