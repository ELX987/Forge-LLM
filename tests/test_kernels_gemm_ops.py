"""CPU tests: dispatchers (fallback paths), packed-linear modules/factory, catalog, bench harness, ops re-exports."""
from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.core.context import NullContext
from forge.core.errors import ConfigError
from forge.kernels import ops, ops_gemm
from forge.kernels.layouts import (
    FRAGMENT_W4,
    PORTABLE_W4,
    PackedFP4,
    pack_w4,
    pack_w8,
    quantize_w4_rtn,
    quantize_w8_rtn,
)
from forge.kernels.ref import quantize_fp4_ref, quantize_fp8_ref, w4a16_gemm_ref


def rnd(*shape, seed=0, scale=1.0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * scale


@pytest.fixture(autouse=True)
def _no_triton_env(monkeypatch):
    """These tests exercise the fallback paths; make sure the interpreter switch of a dev shell does not leak in."""
    monkeypatch.delenv("TRITON_INTERPRET", raising=False)
    monkeypatch.delenv("FORGE_FORCE_REF", raising=False)
    monkeypatch.delenv("FORGE_KERNEL_STRICT", raising=False)


# ------------------------------------------------------------------------------------------ dispatch
def test_cpu_tensors_use_reference_path_for_every_op():
    w = rnd(48, 96)
    x = rnd(5, 96, seed=1)
    q, s, z = quantize_w4_rtn(w, 32, sym=False)
    p = pack_w4(q, s, z, 32)
    y = ops_gemm.w4a16_linear(x, p, torch.ones(48))
    assert ops_gemm.last_path("w4a16_linear") == "reference" and y.dtype == x.dtype
    assert torch.allclose(y, w4a16_gemm_ref(x, p, torch.ones(48)), atol=1e-6)

    q8, s8 = quantize_w8_rtn(w, -1)
    p8 = pack_w8(q8, s8, -1)
    assert torch.allclose(ops_gemm.w8a16_linear(x, p8), x @ p8.dequantize().t(), atol=1e-5)
    assert ops_gemm.last_path("w8a16_linear") == "reference"

    wq, sw = quantize_fp8_ref(w, "per_token")
    y8 = ops_gemm.fp8_linear(x, wq, sw, torch.zeros(48))
    assert ops_gemm.last_path("fp8_linear") == "reference" and (y8 - x @ w.t()).abs().max() < 0.2 * (x @ w.t()).abs().max()

    yi = ops_gemm.int8_linear(x, *_int8_w(w))
    assert yi.shape == (5, 48) and ops_gemm.last_path("int8_linear") == "reference"

    wp = quantize_fp4_ref(w, "nvfp4")
    y4 = ops_gemm.fp4_linear(x, wp)
    assert ops_gemm.last_path("fp4_linear") == "reference" and (y4 - x @ w.t()).norm() < 0.3 * (x @ w.t()).norm()


def _int8_w(w):
    from forge.kernels.ref import quantize_int8_ref

    return quantize_int8_ref(w, "per_channel")


def test_force_ref_and_debug_env(monkeypatch, caplog):
    monkeypatch.setenv("FORGE_FORCE_REF", "1")
    assert ops_gemm.force_ref() and ops_gemm.triton_usable(torch.zeros(1)) == (False, "FORGE_FORCE_REF=1")
    monkeypatch.setenv("FORGE_KERNEL_DEBUG", "1")
    q, s, _ = quantize_w4_rtn(rnd(8, 32), 32)
    with caplog.at_level("INFO", logger="forge.kernels.ops_gemm"):
        ops_gemm.w4a16_linear(rnd(2, 32), pack_w4(q, s, None, 32))
    assert any("w4a16_linear -> reference" in r.message for r in caplog.records)


def test_w4a16_linear_accepts_any_layout_and_nd_inputs():
    w = rnd(70, 100)
    q, s, z = quantize_w4_rtn(w, 32, sym=False)
    a = pack_w4(q, s, z, 32, PORTABLE_W4)
    b = pack_w4(q, s, z, 32, FRAGMENT_W4)
    x = rnd(2, 3, 100, seed=2)
    ya, yb = ops_gemm.w4a16_linear(x, a), ops_gemm.w4a16_linear(x, b)
    assert ya.shape == (2, 3, 70) and torch.equal(ya, yb)
    assert ops_gemm.w4a16_linear(torch.zeros(0, 100), a).shape == (0, 70)  # M = 0


def test_fp8_linear_variants():
    w, x = rnd(40, 200), rnd(6, 200, seed=1)
    ref = x @ w.t()
    for gw in ("per_tensor", "per_channel", "block2d"):
        wq, sw = ops_gemm.quantize_weight_fp8(w, gw)
        y = ops_gemm.fp8_linear(x, wq, sw)
        assert (y - ref).abs().max() < 0.15 * ref.abs().max(), gw
    wq, sw = ops_gemm.quantize_weight_fp8(w, "per_channel")
    xq, sx = ops_gemm.quantize_act_fp8(x)
    y2 = ops_gemm.fp8_linear(xq, wq, sw, x_scale=sx, out_dtype=torch.float32)
    assert torch.allclose(y2, ops_gemm.fp8_linear(x, wq, sw), atol=1e-5)
    with pytest.raises(ConfigError):
        ops_gemm.fp8_linear(xq, wq, sw)  # pre-quantised without a scale
    # fused rmsnorm quantiser == oracle
    wt = rnd(200).abs() + 0.5
    qf, sf = ops_gemm.quantize_act_fp8(x, rms_weight=wt, rms_eps=1e-6)
    from forge.kernels.ref import rmsnorm_act_quant_fp8_ref

    qr, sr, _ = rmsnorm_act_quant_fp8_ref(x, wt, 1e-6)
    assert torch.equal(qf.view(torch.uint8), qr.view(torch.uint8)) and torch.equal(sf, sr)


def test_int8_and_fp4_linear_variants():
    w, x = rnd(32, 128), rnd(4, 128, seed=3)
    wq, sw = _int8_w(w)
    xq, sx = ops_gemm.quantize_act_int8(x)
    y = ops_gemm.int8_linear(xq, wq, sw, x_scale=sx, out_dtype=torch.float32)
    assert (y - x @ w.t()).abs().max() < 0.05 * (x @ w.t()).abs().max()
    with pytest.raises(ConfigError):
        ops_gemm.int8_linear(xq, wq, sw)
    for fmt in ("nvfp4", "mxfp4"):
        wp = quantize_fp4_ref(w, fmt, scale_layout="swizzled")
        y0 = ops_gemm.fp4_linear(x, wp)
        y1 = ops_gemm.fp4_linear(x, wp, act_quant="fp4")
        assert y0.shape == y1.shape == (4, 32) and not torch.allclose(y0, y1)
        assert (y1 - x @ w.t()).norm() < 0.5 * (x @ w.t()).norm()
    with pytest.raises(ConfigError):
        ops_gemm.fp4_linear(rnd(2, 64), wp)
    with pytest.raises(ConfigError):
        ops_gemm.as_packed_fp4(object())
    q = ops_gemm.quantize_act_fp4(x, "mxfp4", scale_layout="swizzled")
    assert isinstance(q, PackedFP4) and q.scale_layout == "swizzled"


def test_tuned_config_lookup_is_safe(forge_home):
    from forge.kernels.autotune import Autotuner
    from forge.kernels.layouts import chip_from_sm
    from forge.kernels.spec import load_spec

    chip = chip_from_sm(90, "Fake H100")
    assert ops_gemm.tuned_config("w4a16_gemm", chip, 1, 64, 128, dtype_cfg_str="d", layout_id="L") is None
    Autotuner(chip=chip).cache.put(
        Autotuner(chip=chip).key_for(load_spec("w4a16_gemm"), {"M": 1, "N": 64, "K": 128}, dtype_cfg="d", layout_id="L"),
        {"op": "w4a16_gemm"}, {"tile_m": 16, "tile_n": 64, "tile_k": 64}, {"median": 1.0, "iqr": 0.0, "n": 1},
    )
    ops_gemm.clear_config_memo()
    ops_gemm._TUNERS.clear()
    assert ops_gemm.tuned_config("w4a16_gemm", chip, 1, 64, 128, dtype_cfg_str="d", layout_id="L")["tile_n"] == 64


# ------------------------------------------------------------------------------------------ modules / factory
@pytest.mark.parametrize(
    "kind,kw,tol",
    [("w4a16", {"group_size": 32}, 0.25), ("w4a16", {"group_size": 32, "sym": False, "layout": FRAGMENT_W4}, 0.2),
     ("w8a16", {}, 0.02), ("w8a16", {"group_size": 32}, 0.02), ("fp8", {}, 0.1), ("fp8", {"granularity": "block2d", "block": 32}, 0.1),
     ("int8", {}, 0.03), ("nvfp4", {}, 0.35), ("mxfp4", {}, 0.4)],
)
def test_make_packed_linear_from_dense_and_state_dict_roundtrip(kind, kw, tol):
    lin = nn.Linear(96, 40, bias=True)
    lin.weight.data = rnd(40, 96, seed=5, scale=0.2)
    lin.bias.data = rnd(40, seed=6)
    m = ops.make_packed_linear(kind, lin, **kw)
    x = rnd(3, 96, seed=7)
    y = m(x)
    ref = F.linear(x, lin.weight, lin.bias)
    assert y.shape == (3, 40) and (y - ref).norm() / ref.norm() < tol
    assert m.in_features == 96 and m.out_features == 40 and m.dequantize().shape == (40, 96)
    assert m.packed_nbytes() < lin.weight.numel() * 4 if hasattr(m, "packed_nbytes") else True
    assert "bias" in m.state_dict() and not any(isinstance(p, nn.Parameter) and p.numel() > 40 for p in m.parameters())
    # rebuild an empty module of the same shape and load the state dict (artifact loading path)
    fresh = ops.make_packed_linear(kind, in_features=96, out_features=40, bias=lin.bias, **kw)
    fresh.load_state_dict(m.state_dict())
    assert torch.equal(fresh(x), y)


def test_make_packed_linear_from_packed_and_errors():
    w = rnd(24, 64)
    q, s, z = quantize_w4_rtn(w, 32, sym=False)
    p = pack_w4(q, s, z, 32, FRAGMENT_W4)
    m = ops.make_packed_linear("w4a16", packed=p, bias=torch.ones(24))
    x = rnd(2, 64)
    assert torch.allclose(m(x), w4a16_gemm_ref(x, p, torch.ones(24)), atol=1e-4)
    assert m.quant_spec()["layout"] == FRAGMENT_W4.id and torch.equal(m.packed().dequantize(), p.dequantize())
    p8 = pack_w8(*quantize_w8_rtn(w, -1), -1)
    assert ops.make_packed_linear("w8a16", packed=p8)(x).shape == (2, 24)
    pf = quantize_fp4_ref(w, "mxfp4")
    mf = ops.make_packed_linear("mxfp4", packed=pf)
    from forge.kernels.ref import dequantize_fp4_ref

    assert mf.fmt == "mxfp4" and torch.equal(mf.dequantize(), dequantize_fp4_ref(pf))
    with pytest.raises(ConfigError):
        ops.make_packed_linear("bogus", nn.Linear(4, 4))
    with pytest.raises(ConfigError):
        ops.make_packed_linear("w4a16")
    with pytest.raises(ConfigError):
        ops.make_packed_linear("w8a16", packed=p)
    with pytest.raises(ConfigError):
        ops_gemm.KernelFp8Linear(8, 8, granularity="weird")


def test_packed_linear_is_a_models_packed_linear_when_models_present():
    models_linear = pytest.importorskip("forge.models.linear")
    m = ops.make_packed_linear("w4a16", nn.Linear(32, 8), group_size=32)
    assert isinstance(m, models_linear.PackedLinear) and m.kind == "w4a16"
    assert m.weight.shape == (8, 32)


# ------------------------------------------------------------------------------------------ ops re-exports
def test_ops_reexports_and_lazy_external_names():
    avail = ops.available_ops()
    for n in ("w4a16_linear", "w8a16_linear", "fp8_linear", "fp4_linear", "int8_linear", "quantize_act_fp8", "make_packed_linear"):
        assert avail[n] and callable(getattr(ops, n))
    assert set(avail) >= {"paged_decode_attention", "moe_route", "hadamard_transform"}  # keys exist even if not (yet) importable
    with pytest.raises(AttributeError):
        ops.definitely_not_an_op  # noqa: B018


# ------------------------------------------------------------------------------------------ catalog
def test_catalog_rows_are_json_serialisable_and_honest_on_cpu():
    from forge.kernels.catalog import get_kernel, list_kernels, register_catalog_entry

    rows = list_kernels()
    by = {r["name"]: r for r in rows}
    for n in ("w4a16_gemm", "w8a16_gemm", "fp8_gemm", "fp4_block_gemm", "int8_w8a8_gemm"):
        r = by[n]
        assert r["family"] == "gemm" and r["reference_available"] is True and r["min_sm"] >= 70
        assert r["available"] is False and r["reasons"]  # CPU host: native path unavailable, reasons given
        assert "not validated" in r["validation"]["gpu"]
    assert by["fp4_block_gemm"]["experimental"] is True
    json.dumps(rows, default=str)
    register_catalog_entry({"name": "zz_test_kernel", "family": "util", "available": True, "reasons": []})
    assert get_kernel("zz_test_kernel")["family"] == "util"
    with pytest.raises(ValueError):
        register_catalog_entry({"family": "x"})
    from forge.core import registry

    assert registry.has("kernel", "w4a16_gemm")


# ------------------------------------------------------------------------------------------ bench
def test_bench_config_schema_for_ui():
    from forge.core.jsonschema import defaults_for, schema_for
    from forge.kernels.bench import BenchConfig

    sch = schema_for(BenchConfig)
    props = sch["properties"]
    assert all("description" in p for p in props.values())
    assert {"kernels", "m_values", "n", "k", "warmup", "repeats", "autotune", "device", "allow_cpu"} <= set(props)
    assert defaults_for(BenchConfig)["repeats"] == 30


def test_run_bench_skips_without_gpu_and_cpu_smoke_works(forge_home):
    from forge.core.db import get_db
    from forge.kernels.bench import BenchConfig, query_results, run_bench

    ctx = NullContext()
    res = run_bench(BenchConfig(), ctx)
    assert res["status"] == "skipped_no_gpu" and res["rows"] == [] and ctx.outputs["status"] == "skipped_no_gpu"

    cfg = BenchConfig(kernels=["w4a16_gemm", "w8a16_gemm", "fp8_gemm", "fp8_act_quant", "int8_w8a8_gemm", "fp4_nvfp4_gemm", "fp4_mxfp4_w4a4_gemm"],
                      dtypes=["fp16"], m_values=[1, 3], n=64, k=128, group_size=32, warmup=0, repeats=3, allow_cpu=True, measure_peaks=False)
    ctx = NullContext()
    res = run_bench(cfg, ctx)
    assert res["status"] == "ok" and len(res["rows"]) == 14
    for r in res["rows"]:
        assert "error" not in r, r
        assert r["correct"] is True, r
        assert r["provenance"] == "cpu_reference_smoke" and r["path"] == "reference"
        assert r["median_ms"] > 0 and r["pct_peak_bw"] is None and r["bound"] is None  # no peaks without a GPU
        assert any("NOT a GPU kernel measurement" in n for n in r["notes"])
    assert ctx.last_progress == 1.0 and ctx.outputs["n_failed_correctness"] == 0
    assert (ctx.run_dir / "bench_results.json").exists()
    db_rows = query_results(kernel="w4a16_gemm")
    assert len(db_rows) == 2 and db_rows[0]["shape"]["N"] == 64 and get_db().query("SELECT COUNT(*) AS c FROM kernel_bench")[0]["c"] == 14


def test_bench_reports_failing_kernel_and_unknown_cases(forge_home, monkeypatch):
    from forge.kernels import bench

    cases = bench.collect_cases()
    assert {"w4a16_gemm", "fp8_gemm", "fp4_mxfp4_gemm"} <= set(cases)

    def broken_build(shape, dtype, device, cfg):
        inst = bench._build_w8(shape, dtype, device, cfg)
        good = inst["run"]
        inst["run"] = lambda: good() * 1.5  # deliberately wrong kernel
        return inst

    monkeypatch.setattr(bench, "collect_cases", lambda: {"w8a16_gemm": bench.BenchCase(name="w8a16_gemm", kernel="w8a16_gemm", build=broken_build)})
    res = bench.run_bench(bench.BenchConfig(kernels=["w8a16_gemm", "nope"], dtypes=["fp16"], m_values=[2], n=32, k=64,
                                            warmup=0, repeats=3, allow_cpu=True, measure_peaks=False), NullContext())
    assert res["rows"][0]["correct"] is False and any("FAILED" in n for n in res["rows"][0]["notes"])
    assert res["skipped"] == [{"case": "nope", "reason": "unknown case"}]


def test_roofline_math():
    from forge.kernels.bench import roofline

    r = roofline(1.0, nbytes=1e9, flops=2e9, peak_gbs=2000.0, peak_tflops=100.0)
    assert r["achieved_gbs"] == pytest.approx(1000.0) and r["pct_peak_bw"] == pytest.approx(0.5)
    assert r["bound"] == "memory" and r["predicted_ms"] == pytest.approx(0.5) and r["residual"] == pytest.approx(1.0)
    c = roofline(10.0, nbytes=1e6, flops=1e12, peak_gbs=2000.0, peak_tflops=100.0)
    assert c["bound"] == "compute" and c["achieved_tflops"] == pytest.approx(100.0) and c["pct_peak_flops"] == pytest.approx(1.0)
    assert roofline(1.0, 1e9, 1e9, None, None)["bound"] is None
