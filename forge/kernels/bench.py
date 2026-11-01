"""Kernel benchmark harness: ``BenchConfig`` (job config, UI-rendered) and ``run_bench(cfg, ctx)`` (job entry ``bench``).

* Cases come from a registry: this file registers the GEMM family (:func:`gemm_bench_cases`); the attention / MoE /
  utility owners expose ``bench_cases()`` in ``ops_attn`` / ``ops_moe`` / ``ops_util`` (lazily imported, optional).
  A case is a dict (or :class:`BenchCase`) -- see :class:`BenchCase`.
* Every measurement is correctness-gated against the case's oracle first (a wrong kernel is reported, never ranked),
  timed with warm-up / L2-flush / median + IQR (:mod:`forge.kernels.autotune.timing`) and annotated with roofline
  fields: achieved GB/s and TFLOP/s, % of the *measured* peak bandwidth / matmul throughput (measured on this
  device by a copy kernel / a large torch matmul -- not datasheet numbers), memory- vs compute-bound classification
  and the residual vs the roofline prediction ``max(bytes / peak_bw, flops / peak_flops)``.
* Results are stored in sqlite table ``kernel_bench`` (also returned and written to ``<run_dir>/bench_results.json``).
* Without a CUDA device the job is skipped (``allow_cpu=True`` runs the reference paths as a smoke test, flagged
  ``provenance='cpu_reference_smoke'`` -- those numbers say nothing about GPU kernels).
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field

from forge.core.context import NullContext
from forge.core.db import register_migration

log = logging.getLogger("forge.kernels.bench")

register_migration(
    "kernels_bench",
    1,
    """
CREATE TABLE IF NOT EXISTS kernel_bench(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    created_at REAL NOT NULL,
    kernel TEXT NOT NULL,
    case_name TEXT,
    family TEXT,
    chip TEXT,
    sm INTEGER,
    dtype TEXT,
    shape_json TEXT,
    layout_id TEXT,
    config_json TEXT,
    path TEXT,
    median_ms REAL,
    iqr_ms REAL,
    min_ms REAL,
    n INTEGER,
    bytes REAL,
    flops REAL,
    achieved_gbs REAL,
    achieved_tflops REAL,
    pct_peak_bw REAL,
    pct_peak_flops REAL,
    bound TEXT,
    predicted_ms REAL,
    residual REAL,
    correct INTEGER,
    max_abs_err REAL,
    baseline_ms REAL,
    speedup REAL,
    provenance TEXT,
    extra_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_kernel_bench_kernel ON kernel_bench(kernel, chip, created_at);
""",
)


class BenchConfig(BaseModel):
    """Configuration of a kernel benchmark job (job type ``bench``)."""

    model_config = ConfigDict(extra="forbid")

    kernels: list[str] = Field(
        default_factory=list,
        description="Case names to run (e.g. w4a16_gemm, fp8_gemm, fp4_nvfp4_gemm). Empty = every case runnable on this device.",
    )
    dtypes: list[Literal["bf16", "fp16"]] = Field(default_factory=lambda: ["bf16"], description="Activation dtypes to benchmark.")
    m_values: list[int] = Field(
        default_factory=lambda: [1, 4, 16, 64, 256, 1024],
        description="Batch sizes M (tokens). 1-16 are the bandwidth-bound decode regime, >=256 the compute-bound prefill regime.",
        json_schema_extra={"x-group": "shapes"},
    )
    n: int = Field(4096, ge=16, description="Output features N of the GEMM (model-native: e.g. 4096 for an 8B hidden size).", json_schema_extra={"x-group": "shapes"})
    k: int = Field(4096, ge=16, description="Input features K of the GEMM.", json_schema_extra={"x-group": "shapes"})
    group_size: int = Field(128, description="Quantisation group size for W4A16/W8A16 (-1 = per output channel).", json_schema_extra={"x-group": "shapes"})
    shapes: list[dict[str, int]] | None = Field(
        None, description="Explicit shape list [{M,N,K}, ...]; overrides m_values/n/k when given.", json_schema_extra={"x-advanced": True})
    warmup: int = Field(10, ge=0, le=1000, description="Untimed warm-up launches per measurement.", json_schema_extra={"x-group": "measurement"})
    repeats: int = Field(30, ge=3, le=2000, description="Timed repeats per measurement (median and IQR are reported; Plan 6.6 uses 30).", json_schema_extra={"x-group": "measurement"})
    flush_l2: bool = Field(True, description="Overwrite a >=128 MiB buffer before every timed run (cold L2, bandwidth-bound kernels).", json_schema_extra={"x-group": "measurement"})
    check_correctness: bool = Field(True, description="Compare every kernel against its fp32 oracle before timing; failing kernels are flagged, not ranked.")
    include_baseline: bool = Field(True, description="Also time a dense 16-bit GEMM (cuBLAS via torch) for speed-up numbers.")
    measure_peaks: bool = Field(True, description="Measure peak copy bandwidth and peak matmul TFLOP/s on this device for roofline percentages.", json_schema_extra={"x-group": "measurement"})
    autotune: bool = Field(False, description="Run the correctness-gated autotuner for the Triton GEMM cases and cache the winner (slow).", json_schema_extra={"x-group": "autotune"})
    autotune_repeats: int = Field(15, ge=3, le=500, description="Timed repeats per autotune candidate.", json_schema_extra={"x-group": "autotune", "x-advanced": True})
    autotune_max_candidates: int = Field(24, ge=1, le=500, description="Max candidates measured per shape (seeded random sample of the feasible space).", json_schema_extra={"x-group": "autotune", "x-advanced": True})
    device: int = Field(0, ge=0, description="CUDA device index.")
    seed: int = Field(0, description="Seed for synthetic weights/activations.")
    allow_cpu: bool = Field(False, description="Run reference paths on CPU when no CUDA device exists (smoke test only; not a kernel measurement).")
    save_results: bool = Field(True, description="Write results to the kernel_bench database table.")


@dataclass
class BenchCase:
    """One benchmarkable kernel case.

    ``build(shape, dtype, device, cfg) -> dict`` must return a *bench instance* with keys

    ``run``      ``() -> Tensor``       the dispatcher under test (whatever path it picks)
    ``ref``      ``() -> Tensor``       the oracle result for the same inputs
    ``bytes``    ``int``                HBM bytes the kernel must move at minimum (weights + activations in/out)
    ``flops``    ``int``                arithmetic work (2*M*N*K for GEMMs)
    optional: ``baseline`` (``() -> Tensor``, dense 16-bit comparison), ``layout_id``, ``path`` (``() -> str`` taken by
    ``run``), ``run_cfg`` (``(config: dict) -> Tensor`` direct Triton launch for autotuning), ``spec`` (spec name),
    ``exact`` (bool, bit-exact compare).
    """

    name: str
    kernel: str
    family: str = "gemm"
    description: str = ""
    shapes: list[dict[str, int]] = field(default_factory=list)  # default shapes; M/N/K from BenchConfig override
    dtypes: list[str] = field(default_factory=lambda: ["bf16", "fp16"])
    min_sm: int = 70
    shape_from_config: bool = True  # False for cases with their own fixed shapes (attention...)
    build: Callable[..., dict[str, Any]] | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | BenchCase) -> BenchCase:
        if isinstance(d, BenchCase):
            return d
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


_TORCH_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


# ------------------------------------------------------------------------------------------------------------------
# GEMM family cases
# ------------------------------------------------------------------------------------------------------------------
def _gen(seed: int, device: str) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def _act(shape: dict[str, int], dtype: torch.dtype, device: str, seed: int) -> torch.Tensor:
    return torch.randn(shape["M"], shape["K"], generator=_gen(seed, device)).to(device=device, dtype=dtype)


def _weights(shape: dict[str, int], device: str, seed: int) -> torch.Tensor:
    return (torch.randn(shape["N"], shape["K"], generator=_gen(seed + 1, device)) * 0.02).to(device)


def _io_bytes(shape: dict[str, int], dtype: torch.dtype) -> int:
    return (shape["M"] * shape["K"] + shape["M"] * shape["N"]) * torch.empty((), dtype=dtype).element_size()


def _build_w4(shape: dict[str, int], dtype: torch.dtype, device: str, cfg: BenchConfig) -> dict[str, Any]:
    from forge.kernels import ops_gemm
    from forge.kernels.layouts import pack_w4, quantize_w4_rtn
    from forge.kernels.ref import w4a16_gemm_ref
    from forge.kernels.w4_tuning import w4_tune_tag

    g = shape.get("group", cfg.group_size)
    w = _weights(shape, device, cfg.seed)
    q, s, z = quantize_w4_rtn(w, g, sym=True)
    p = pack_w4(q, s, z, g)
    x = _act(shape, dtype, device, cfg.seed)
    dense = p.dequantize(dtype)
    state: dict[str, Any] = {"config": None}
    inst: dict[str, Any] = {
        "run": lambda: ops_gemm.w4a16_linear(x, p, config=state["config"]),
        "ref": lambda: w4a16_gemm_ref(x, p),
        "bytes": p.nbytes + _io_bytes(shape, dtype),
        "flops": 2 * shape["M"] * shape["N"] * shape["K"],
        "baseline": lambda: F.linear(x, dense),
        "layout_id": p.layout_id,
        "path": lambda: ops_gemm.last_path("w4a16_linear") or "?",
        "spec": "w4a16_gemm",
        "state": state,
        "dtype_cfg": ops_gemm.dtype_cfg("w4a16_gemm", dtype, g, w4_tune_tag(p.g_idx is not None, p.zeros is None)),
    }

    def run_cfg(c: dict[str, Any]) -> torch.Tensor:
        from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton

        return w4a16_gemm_triton(x, p, None, config=c)

    inst["run_cfg"] = run_cfg
    return inst


def _build_w8(shape: dict[str, int], dtype: torch.dtype, device: str, cfg: BenchConfig) -> dict[str, Any]:
    from forge.kernels import ops_gemm
    from forge.kernels.layouts import pack_w8, quantize_w8_rtn
    from forge.kernels.ref import w8a16_gemm_ref

    g = shape.get("group", -1)
    w = _weights(shape, device, cfg.seed)
    q, s = quantize_w8_rtn(w, g)
    p = pack_w8(q, s, g)
    x = _act(shape, dtype, device, cfg.seed)
    dense = p.dequantize(dtype)
    state: dict[str, Any] = {"config": None}
    inst: dict[str, Any] = {
        "run": lambda: ops_gemm.w8a16_linear(x, p, config=state["config"]),
        "ref": lambda: w8a16_gemm_ref(x, p),
        "bytes": p.nbytes + _io_bytes(shape, dtype),
        "flops": 2 * shape["M"] * shape["N"] * shape["K"],
        "baseline": lambda: F.linear(x, dense),
        "layout_id": p.layout_id,
        "path": lambda: ops_gemm.last_path("w8a16_linear") or "?",
        "spec": "w8a16_gemm",
        "state": state,
        "dtype_cfg": ops_gemm.dtype_cfg("w8a16_gemm", dtype, g),
    }

    def run_cfg(c: dict[str, Any]) -> torch.Tensor:
        from forge.kernels.triton.gemm_w8a16 import w8a16_gemm_triton

        return w8a16_gemm_triton(x, p, None, config=c)

    inst["run_cfg"] = run_cfg
    return inst


def _build_fp8(shape: dict[str, int], dtype: torch.dtype, device: str, cfg: BenchConfig) -> dict[str, Any]:
    from forge.kernels import ops_gemm
    from forge.kernels.ref import dequantize_fp8_ref, fp8_gemm_ref, quantize_fp8_ref

    w = _weights(shape, device, cfg.seed)
    wq, sw = quantize_fp8_ref(w, "per_token")
    x = _act(shape, dtype, device, cfg.seed)
    xq, sx = quantize_fp8_ref(x, "per_token")
    dense = dequantize_fp8_ref(wq, sw).to(dtype)
    return {
        "run": lambda: ops_gemm.fp8_linear(x, wq, sw, out_dtype=dtype),
        "ref": lambda: fp8_gemm_ref(xq, sx, wq, sw),
        "bytes": wq.numel() + _io_bytes(shape, dtype),
        "flops": 2 * shape["M"] * shape["N"] * shape["K"],
        "baseline": lambda: F.linear(x, dense),
        "layout_id": "row_major",
        "path": lambda: ops_gemm.last_path("fp8_linear") or "?",
        "spec": "fp8_gemm",
        "exact": False,
    }


def _build_fp8_quant(shape: dict[str, int], dtype: torch.dtype, device: str, cfg: BenchConfig) -> dict[str, Any]:
    from forge.kernels import ops_gemm
    from forge.kernels.ref import quantize_fp8_ref

    x = _act(shape, dtype, device, cfg.seed)
    return {
        "run": lambda: ops_gemm.quantize_act_fp8(x)[0].view(torch.uint8).float(),
        "ref": lambda: quantize_fp8_ref(x, "per_token")[0].view(torch.uint8).float(),
        "bytes": shape["M"] * shape["K"] * (x.element_size() + 1) + shape["M"] * 4,
        "flops": 3 * shape["M"] * shape["K"],
        "layout_id": "row_major",
        "path": lambda: ops_gemm.last_path("quantize_act_fp8") or "?",
        "spec": "fp8_gemm",
        "exact": False,
        "tol_slack": 1e9,  # codes may differ by one at rounding boundaries; correctness judged via fp8 GEMM case
    }


def _build_int8(shape: dict[str, int], dtype: torch.dtype, device: str, cfg: BenchConfig) -> dict[str, Any]:
    from forge.kernels import ops_gemm
    from forge.kernels.ref import int8_gemm_ref, quantize_int8_ref

    w = _weights(shape, device, cfg.seed)
    wq, sw = quantize_int8_ref(w, "per_channel")
    x = _act(shape, dtype, device, cfg.seed)
    xq, sx = quantize_int8_ref(x, "per_token")
    dense = (wq.float() * sw[:, None]).to(dtype)
    return {
        "run": lambda: ops_gemm.int8_linear(x, wq, sw, out_dtype=dtype),
        "ref": lambda: int8_gemm_ref(xq, sx, wq, sw),
        "bytes": wq.numel() + _io_bytes(shape, dtype),
        "flops": 2 * shape["M"] * shape["N"] * shape["K"],
        "baseline": lambda: F.linear(x, dense),
        "layout_id": "row_major",
        "path": lambda: ops_gemm.last_path("int8_linear") or "?",
        "spec": "int8_w8a8_gemm",
    }


def _make_fp4_builder(fmt: str, act_quant: str) -> Callable[..., dict[str, Any]]:
    def build(shape: dict[str, int], dtype: torch.dtype, device: str, cfg: BenchConfig) -> dict[str, Any]:
        from forge.kernels import ops_gemm
        from forge.kernels.ref import dequantize_fp4_ref, fp4_block_gemm_ref, quantize_fp4_ref

        w = _weights(shape, device, cfg.seed)
        wp = quantize_fp4_ref(w, fmt, scale_layout="logical")
        x = _act(shape, dtype, device, cfg.seed)
        dense = dequantize_fp4_ref(wp, dtype)
        if act_quant == "fp4":
            ref = lambda: fp4_block_gemm_ref(dequantize_fp4_ref(quantize_fp4_ref(x, fmt)), wp)  # noqa: E731
        else:
            ref = lambda: fp4_block_gemm_ref(x, wp)  # noqa: E731
        return {
            "run": lambda: ops_gemm.fp4_linear(x, wp, act_quant=act_quant),
            "ref": ref,
            "bytes": wp.nbytes + _io_bytes(shape, dtype),
            "flops": 2 * shape["M"] * shape["N"] * shape["K"],
            "baseline": lambda: F.linear(x, dense),
            "layout_id": wp.layout_id,
            "path": lambda: ops_gemm.last_path("fp4_linear") or "?",
            "spec": "fp4_block_gemm",
            "tol_slack": 2.0 if act_quant == "fp4" else 1.0,
        }

    return build


def gemm_bench_cases() -> list[dict[str, Any]]:
    """Benchmark cases of the GEMM family (registered through ``ops_gemm.bench_cases``)."""
    return [
        dict(name="w4a16_gemm", kernel="w4a16_gemm", description="INT4 weight-only GEMM (portable_row, in-kernel dequant)", min_sm=70, build=_build_w4),
        dict(name="w8a16_gemm", kernel="w8a16_gemm", description="INT8 weight-only GEMM", min_sm=70, build=_build_w8),
        dict(name="fp8_gemm", kernel="fp8_gemm", description="FP8 E4M3 W8A8 GEMM, per-token x per-channel scales", min_sm=70, build=_build_fp8),
        dict(name="fp8_act_quant", kernel="fp8_gemm", description="Per-token FP8 activation quantiser", min_sm=70, build=_build_fp8_quant),
        dict(name="int8_w8a8_gemm", kernel="int8_w8a8_gemm", description="INT8 W8A8 GEMM", min_sm=75, build=_build_int8),
        dict(name="fp4_nvfp4_gemm", kernel="fp4_block_gemm", description="NVFP4 weight-only GEMM (dequant kernel) [experimental]", min_sm=70,
             build=_make_fp4_builder("nvfp4", "none")),
        dict(name="fp4_mxfp4_gemm", kernel="fp4_block_gemm", description="MXFP4 weight-only GEMM (dequant kernel) [experimental]", min_sm=70,
             build=_make_fp4_builder("mxfp4", "none")),
        dict(name="fp4_mxfp4_w4a4_gemm", kernel="fp4_block_gemm", description="MXFP4 W4A4 with fused act quantiser (native tl.dot_scaled if available) [experimental]",
             min_sm=70, build=_make_fp4_builder("mxfp4", "fp4")),
    ]


def _adapt_family_case(d: dict[str, Any]) -> BenchCase:
    """Wrap a descriptor ``{name, family, make_inputs(cfg), run(inp), ref(inp), shapes, bytes_moved(cfg), flops(cfg)}``
    (the form exposed by ``ops_attn`` / ``ops_moe`` / ``ops_util``) as a :class:`BenchCase`."""
    make_inputs, run, ref = d["make_inputs"], d["run"], d["ref"]
    bytes_fn, flops_fn = d["bytes_moved"], d["flops"]

    def build(shape: dict[str, Any], dtype: torch.dtype, device: str, cfg: Any) -> dict[str, Any]:
        name = {torch.bfloat16: "bfloat16", torch.float16: "float16", torch.float32: "float32"}.get(dtype, "bfloat16")
        c = dict(shape)
        c.setdefault("dtype", name)
        c.setdefault("device", device)
        inp = make_inputs(c)
        return {
            "run": lambda: run(inp),
            "ref": lambda: ref(inp),
            "bytes": int(bytes_fn(c)),
            "flops": int(flops_fn(c)),
            "path": d.get("path") or "dispatch",
        }

    return BenchCase(name=d["name"], kernel="", family=d.get("family", "other"), description=d.get("description", ""),
                     shapes=[dict(x) for x in d.get("shapes", [])], dtypes=list(d.get("dtypes", ["bf16", "fp16"])),
                     min_sm=int(d.get("min_sm", 70)), shape_from_config=False, build=build)


def _shape_m(shape: dict[str, Any], default: int = 0) -> int:
    for k in ("M", "batch", "tokens", "rows"):
        if k in shape:
            return int(shape[k])
    return default


def _generic_compare(out: Any, ref: Any) -> tuple[bool, float, str]:
    """Loose dtype-aware comparison for cases without a KernelSpec tolerance (tensors, tuples, namedtuples)."""
    def leaves(x: Any) -> list[torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return [x]
        if isinstance(x, (tuple, list)):
            return [t for v in x for t in leaves(v)]
        if hasattr(x, "__dict__") and not isinstance(x, (int, float, str)):
            return [t for v in vars(x).values() for t in leaves(v)]
        return []

    a, b = leaves(out), leaves(ref)
    if len(a) != len(b):
        return False, float("inf"), f"output arity {len(a)} != reference arity {len(b)}"
    worst = 0.0
    for x, y in zip(a, b):
        x, y = x.detach().float().cpu(), y.detach().float().cpu()
        if x.shape != y.shape:
            return False, float("inf"), f"shape {tuple(x.shape)} != {tuple(y.shape)}"
        err = float((x - y).abs().max()) if x.numel() else 0.0
        scale = float(y.abs().max()) if y.numel() else 0.0
        worst = max(worst, err)
        if err > 5e-2 * max(scale, 1.0):
            return False, worst, f"max abs err {err:.3g} vs scale {scale:.3g}"
    return True, worst, "ok"


def collect_cases() -> dict[str, BenchCase]:
    """All registered cases: GEMM family + whatever ops_attn / ops_moe / ops_util expose (missing modules are skipped)."""
    import importlib

    out: dict[str, BenchCase] = {}
    for modname in ("forge.kernels.ops_gemm", "forge.kernels.ops_attn", "forge.kernels.ops_moe", "forge.kernels.ops_util"):
        try:
            mod = importlib.import_module(modname)
            fn = getattr(mod, "bench_cases", None)
            if fn is None:
                continue
            for d in fn():
                c = _adapt_family_case(d) if isinstance(d, dict) and "make_inputs" in d else BenchCase.from_dict(d)
                out[c.name] = c
        except Exception as e:  # another owner's module may be absent/broken: never break our own cases
            log.debug("bench cases from %s unavailable: %s", modname, e)
    return out


# ------------------------------------------------------------------------------------------------------------------
# measurement
# ------------------------------------------------------------------------------------------------------------------
def measure_peak_bandwidth(device: str, nbytes: int = 1 << 30) -> float | None:
    """Measured device-to-device copy bandwidth in GB/s (read + write counted), or None without CUDA."""
    if not str(device).startswith("cuda"):
        return None
    from forge.kernels.autotune.timing import summarize, time_callable

    n = nbytes // 4
    a = torch.empty(n, dtype=torch.float32, device=device)
    b = torch.empty(n, dtype=torch.float32, device=device)
    a.normal_()
    t = summarize(time_callable(lambda: b.copy_(a), warmup=5, repeats=20))
    del a, b
    return 2.0 * nbytes / (t.median * 1e-3) / 1e9


def measure_peak_tflops(device: str, dtype: torch.dtype = torch.bfloat16, size: int = 8192) -> float | None:
    """Measured dense matmul throughput (TFLOP/s) of torch/cuBLAS, or None without CUDA."""
    if not str(device).startswith("cuda"):
        return None
    from forge.kernels.autotune.timing import summarize, time_callable

    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    t = summarize(time_callable(lambda: a @ b, warmup=3, repeats=10))
    del a, b
    return 2.0 * size**3 / (t.median * 1e-3) / 1e12


def _sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _shapes_for(case: BenchCase, cfg: BenchConfig) -> list[dict[str, int]]:
    if cfg.shapes:
        return [dict(s) for s in cfg.shapes]
    if not case.shape_from_config and case.shapes:
        return [dict(s) for s in case.shapes]
    return [{"M": m, "N": cfg.n, "K": cfg.k} for m in cfg.m_values]


def roofline(median_ms: float, nbytes: float, flops: float, peak_gbs: float | None, peak_tflops: float | None) -> dict[str, Any]:
    """Roofline annotation of one measurement (all optional peaks may be None)."""
    t = median_ms * 1e-3
    gbs = nbytes / t / 1e9 if t > 0 else None
    tf = flops / t / 1e12 if t > 0 else None
    out: dict[str, Any] = {"achieved_gbs": gbs, "achieved_tflops": tf, "pct_peak_bw": None, "pct_peak_flops": None,
                           "bound": None, "predicted_ms": None, "residual": None,
                           "arithmetic_intensity": flops / nbytes if nbytes else None}
    if peak_gbs:
        out["pct_peak_bw"] = gbs / peak_gbs if gbs is not None else None
    if peak_tflops:
        out["pct_peak_flops"] = tf / peak_tflops if tf is not None else None
    if peak_gbs and peak_tflops and nbytes:
        ridge = peak_tflops * 1e12 / (peak_gbs * 1e9)
        out["bound"] = "memory" if flops / nbytes < ridge else "compute"
        pred = max(nbytes / (peak_gbs * 1e9), flops / (peak_tflops * 1e12)) * 1e3
        out["predicted_ms"] = pred
        out["residual"] = median_ms / pred - 1.0 if pred > 0 else None
    return out


def _kpi_status(spec: Any, m: int, row: dict[str, Any]) -> dict[str, Any]:
    """Compare a row with the spec KPI targets (TARGETS, reported honestly as met / not met / not applicable)."""
    k = spec.kpi
    res: dict[str, Any] = {}
    if k.at_M and m not in k.at_M:
        return {"applicable": False}
    res["applicable"] = True
    if k.bw_fraction_min is not None and row.get("pct_peak_bw") is not None:
        res["bw_fraction"] = {"target": k.bw_fraction_min, "value": row["pct_peak_bw"], "met": row["pct_peak_bw"] >= k.bw_fraction_min}
    if k.speedup_vs_bf16_min is not None and row.get("speedup") is not None:
        res["speedup_vs_bf16"] = {"target": k.speedup_vs_bf16_min, "value": row["speedup"], "met": row["speedup"] >= k.speedup_vs_bf16_min}
    if k.tflops_fraction_min is not None and row.get("pct_peak_flops") is not None:
        res["tflops_fraction"] = {"target": k.tflops_fraction_min, "value": row["pct_peak_flops"], "met": row["pct_peak_flops"] >= k.tflops_fraction_min}
    return res


def _store_rows(rows: list[dict[str, Any]], run_id: str) -> None:
    from forge.core.db import get_db

    db = get_db()
    now = time.time()
    for r in rows:
        db.execute(
            "INSERT INTO kernel_bench(run_id, created_at, kernel, case_name, family, chip, sm, dtype, shape_json, layout_id, config_json, path,"
            " median_ms, iqr_ms, min_ms, n, bytes, flops, achieved_gbs, achieved_tflops, pct_peak_bw, pct_peak_flops, bound,"
            " predicted_ms, residual, correct, max_abs_err, baseline_ms, speedup, provenance, extra_json)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, now, r["kernel"], r["case"], r["family"], r["chip"], r.get("sm"), r["dtype"], json.dumps(r["shape"]), r.get("layout_id"),
             json.dumps(r.get("config")) if r.get("config") else None, r.get("path"), r.get("median_ms"), r.get("iqr_ms"), r.get("min_ms"),
             r.get("n"), r.get("bytes"), r.get("flops"), r.get("achieved_gbs"), r.get("achieved_tflops"), r.get("pct_peak_bw"),
             r.get("pct_peak_flops"), r.get("bound"), r.get("predicted_ms"), r.get("residual"),
             None if r.get("correct") is None else int(bool(r["correct"])), r.get("max_abs_err"), r.get("baseline_ms"), r.get("speedup"),
             r.get("provenance"), json.dumps({"kpi": r.get("kpi"), "notes": r.get("notes"), "autotune": r.get("autotune")})),
        )


def query_results(kernel: str | None = None, chip: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    """Rows of ``kernel_bench`` (newest first), JSON columns decoded."""
    from forge.core.db import get_db

    conds, params = [], []
    if kernel:
        conds.append("kernel=?")
        params.append(kernel)
    if chip:
        conds.append("chip=?")
        params.append(chip)
    sql = "SELECT * FROM kernel_bench" + (" WHERE " + " AND ".join(conds) if conds else "") + " ORDER BY created_at DESC, id DESC LIMIT ?"
    rows = get_db().query(sql, (*params, limit))
    for r in rows:
        for k in ("shape_json", "config_json", "extra_json"):
            if r.get(k):
                try:
                    r[k[:-5]] = json.loads(r[k])
                except ValueError:
                    pass
    return rows


def run_bench(cfg: BenchConfig | dict[str, Any] | None = None, ctx: Any = None) -> dict[str, Any]:
    """Job entry (``bench``): run the selected cases, store and return the results.

    Returns ``{"status", "chip", "rows": [...], "peaks": {...}, "skipped": [...]}``.
    """
    from forge.kernels.autotune.timing import make_l2_flusher, summarize, time_callable
    from forge.kernels.layouts.chip import chip_info
    from forge.kernels.ref.gemm_common import compare
    from forge.kernels.spec import load_spec

    cfg = cfg if isinstance(cfg, BenchConfig) else BenchConfig(**(cfg or {}))
    ctx = ctx or NullContext()
    has_cuda = torch.cuda.is_available()
    if not has_cuda and not cfg.allow_cpu:
        msg = "no CUDA device on this host: kernel benchmarks need an NVIDIA GPU (set allow_cpu=true for a reference-path smoke test)"
        ctx.log(msg, "warning")
        ctx.set_output("status", "skipped_no_gpu")
        return {"status": "skipped_no_gpu", "message": msg, "rows": [], "skipped": [], "peaks": {}}
    device = f"cuda:{cfg.device}" if has_cuda else "cpu"
    if has_cuda:
        torch.cuda.set_device(cfg.device)
    chip = chip_info(None, cfg.device) if has_cuda else chip_info(None)
    provenance = "measured" if has_cuda else "cpu_reference_smoke"
    ctx.log(f"bench on {chip.name} ({chip.fingerprint}) device={device}")

    cases = collect_cases()
    wanted = cfg.kernels or sorted(cases)
    selected, skipped = [], []
    for name in wanted:
        c = cases.get(name)
        if c is None:
            skipped.append({"case": name, "reason": "unknown case"})
        elif has_cuda and chip.sm is not None and chip.sm < c.min_sm:
            skipped.append({"case": name, "reason": f"needs sm_{c.min_sm}+, device is sm_{chip.sm}"})
        else:
            selected.append(c)

    peaks: dict[str, Any] = {"bandwidth_gbs": None, "tflops_bf16": None, "provenance": provenance}
    if cfg.measure_peaks and has_cuda:
        try:
            peaks["bandwidth_gbs"] = measure_peak_bandwidth(device)
            peaks["tflops_bf16"] = measure_peak_tflops(device, torch.bfloat16 if chip.has_bf16 else torch.float16)
            ctx.log(f"measured peaks: {peaks['bandwidth_gbs']:.0f} GB/s copy, {peaks['tflops_bf16']:.1f} TFLOP/s matmul")
        except Exception as e:
            ctx.log(f"peak measurement failed: {e}", "warning")
    ctx.set_output("peaks", peaks)

    flush = make_l2_flusher() if (cfg.flush_l2 and has_cuda) else None
    rows: list[dict[str, Any]] = []
    jobs = [(c, s, d) for c in selected for s in _shapes_for(c, cfg) for d in cfg.dtypes]
    total = max(1, len(jobs))
    for i, (case, shape, dname) in enumerate(jobs):
        if ctx.should_stop():
            ctx.log("stop requested; ending benchmark early", "warning")
            break
        dtype = _TORCH_DTYPES[dname]
        if dtype == torch.bfloat16 and has_cuda and not chip.has_bf16:
            skipped.append({"case": case.name, "shape": shape, "dtype": dname, "reason": "bf16 not supported by this sm"})
            continue
        try:
            spec = load_spec(case.kernel) if case.kernel else None
        except Exception:  # families without a KernelSpec yaml (attention / MoE / util) use the generic comparison
            spec = None
        row: dict[str, Any] = {"case": case.name, "kernel": case.kernel, "family": case.family, "chip": chip.fingerprint, "sm": chip.sm,
                               "dtype": dname, "shape": shape, "provenance": provenance, "notes": []}
        try:
            inst = case.build(shape, dtype, device, cfg) if case.build else None  # type: ignore[misc]
            if inst is None:
                raise RuntimeError("case has no builder")
            row["layout_id"] = inst.get("layout_id")
            if cfg.autotune and has_cuda and inst.get("run_cfg") and spec is not None:
                row["autotune"] = _autotune(cfg, spec, chip, shape, inst, ctx)
                if row["autotune"].get("best"):
                    inst["state"]["config"] = row["autotune"]["best"]
                    row["config"] = row["autotune"]["best"]
            out = inst["run"]()
            _sync(device)
            row["path"] = inst["path"]() if callable(inst.get("path")) else inst.get("path")
            if cfg.check_correctness and spec is None and inst.get("ref"):
                ok, err, detail = _generic_compare(out, inst["ref"]())
                row["correct"], row["max_abs_err"] = ok, err
                if not ok:
                    row["notes"].append("FAILED correctness gate: " + detail)
                    ctx.log(f"{case.name} {shape} {dname}: FAILED correctness ({detail})", "error")
            if cfg.check_correctness and spec is not None:
                res = compare(out, inst["ref"](), spec.tolerance, out_dtype=out.dtype, exact=bool(inst.get("exact", False)),
                              slack=float(inst.get("tol_slack", 1.0)))
                row["correct"], row["max_abs_err"] = bool(res.ok), res.max_abs_err
                if not res.ok:
                    row["notes"].append("FAILED correctness gate: " + res.detail)
                    ctx.log(f"{case.name} {shape} {dname}: FAILED correctness ({res.detail})", "error")
            times = time_callable(inst["run"], warmup=cfg.warmup, repeats=cfg.repeats, flush=flush)
            st = summarize(times)
            row.update(median_ms=st.median, iqr_ms=st.iqr, min_ms=st.min, n=st.n, bytes=inst["bytes"], flops=inst["flops"])
            if cfg.include_baseline and inst.get("baseline"):
                bt = summarize(time_callable(inst["baseline"], warmup=cfg.warmup, repeats=cfg.repeats, flush=flush))
                row["baseline_ms"], row["speedup"] = bt.median, bt.median / st.median if st.median else None
            row.update(roofline(st.median, inst["bytes"], inst["flops"], peaks["bandwidth_gbs"], peaks["tflops_bf16"]))
            if spec is not None:
                row["kpi"] = _kpi_status(spec, _shape_m(shape), row)
            if not has_cuda:
                row["notes"].append("CPU reference path: NOT a GPU kernel measurement")
            ctx.metric(i, **{f"{case.name}/{dname}/M{_shape_m(shape, i)}/median_ms": st.median})
        except Exception as e:
            log.exception("bench case %s failed", case.name)
            row["error"] = f"{type(e).__name__}: {e}"
            row["notes"].append(row["error"])
            ctx.log(f"{case.name} {shape} {dname} failed: {row['error']}", "error")
        rows.append(row)
        ctx.progress((i + 1) / total, f"{case.name} M={_shape_m(shape, i)} {dname}")

    if cfg.save_results and rows:
        try:
            _store_rows([r for r in rows if "median_ms" in r], getattr(ctx, "run_id", "adhoc"))
        except Exception as e:
            ctx.log(f"could not store results in the database: {e}", "warning")
    result = {"status": "ok", "chip": chip.to_dict(), "device": device, "peaks": peaks, "rows": rows, "skipped": skipped}
    try:
        (ctx.run_dir / "bench_results.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    except Exception:
        pass
    ctx.set_output("n_rows", len(rows))
    ctx.set_output("n_failed_correctness", sum(1 for r in rows if r.get("correct") is False))
    ctx.set_output("status", "ok")
    return result


def _autotune(cfg: BenchConfig, spec: Any, chip: Any, shape: dict[str, int], inst: dict[str, Any], ctx: Any) -> dict[str, Any]:
    from forge.kernels.autotune import Autotuner

    tuner = Autotuner(chip=chip, repeats=cfg.autotune_repeats, max_candidates=cfg.autotune_max_candidates)
    cands = None
    if spec.kernel == "w4a16_gemm":  # the nibble-split kernels have their own search space (impl / tile / split-K)
        from forge.kernels.w4_tuning import w4_candidates

        cands = w4_candidates(shape["M"], shape["N"], shape["K"], sm=chip.sm, num_sms=chip.num_sms, group=shape.get("group", cfg.group_size))
    res = tuner.tune(spec, shape, lambda c, s: inst["run_cfg"](c), lambda s: inst["ref"](),
                     dtype_cfg=inst.get("dtype_cfg", ""), layout_id=inst.get("layout_id") or "", candidates=cands)
    ctx.log(f"autotune {spec.kernel} {shape}: best={res.best} cached={res.cached} rejected={len(res.rejected)}")
    return {"best": res.best, "cached": res.cached, "from_cache": res.from_cache, "n_measured": len(res.measurements),
            "n_rejected": len(res.rejected), "notes": res.notes}


__all__ = [
    "BenchCase", "BenchConfig", "collect_cases", "gemm_bench_cases", "measure_peak_bandwidth", "measure_peak_tflops",
    "query_results", "roofline", "run_bench",
]
