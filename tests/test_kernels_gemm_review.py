"""Regression tests for bugs found by the independent GEMM-kernel review (CPU only; Triton parts run in the CPU interpreter
or the offline compiler and need ``triton`` importable)."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from forge.core.errors import ConfigError, NotFound
from forge.kernels import ops_gemm
from forge.kernels.layouts import FRAGMENT_W4, PORTABLE_W4, pack_w4, quantize_w4_rtn, repack
from forge.kernels.layouts.fp4 import swizzle_scales, unswizzle_scales
from forge.kernels.ref import quantize_fp8_ref
from forge.kernels.ref.gemm_fp8 import expand_scale
from forge.kernels.spec import load_spec
from forge.kernels.triton.gemm_common import launch_fitting
from forge.kernels.triton.gemm_fp8 import prepare_fp8_scales

HAS_TRITON = importlib.util.find_spec("triton") is not None


# ---------------------------------------------------------------------------------------------------------------------
# launch_fitting: consumer GPUs (~99 KiB smem) must shrink the schedule instead of failing
# ---------------------------------------------------------------------------------------------------------------------
class OutOfResources(Exception):  # same class name as triton.runtime.errors.OutOfResources
    pass


def test_launch_fitting_shrinks_stages_then_tile_k_and_memoises():
    seen = []

    def launch(c):
        seen.append((c["stages"], c["tile_k"]))
        if c["stages"] * c["tile_k"] > 128:
            raise OutOfResources("out of resource: shared memory")
        return "ok"

    cfg = dict(tile_m=16, tile_n=64, tile_k=128, stages=3, warps=4)
    assert launch_fitting(("t1",), cfg, launch) == "ok"
    assert seen[0] == (3, 128) and seen[-1][0] * seen[-1][1] <= 128
    n_first = len(seen)
    assert launch_fitting(("t1",), cfg, launch) == "ok"
    assert len(seen) == n_first + 1  # memoised: a single launch the second time


def test_launch_fitting_does_not_swallow_other_errors_and_gives_up():
    def boom(c):
        raise ValueError("boom")

    with pytest.raises(ValueError):
        launch_fitting(("t2",), dict(tile_k=64, stages=2), boom)

    def always(c):
        raise OutOfResources("shared memory")

    with pytest.raises(OutOfResources):
        launch_fitting(("t3",), dict(tile_k=64, stages=2), always)


# ---------------------------------------------------------------------------------------------------------------------
# FP8 scale addressing (was: per-tensor weight scale with per-token activations read sb[0..N) out of bounds)
# ---------------------------------------------------------------------------------------------------------------------
def _emulate_scale_addressing(mode, sa, sb, strides, sb_div, m, n, k, block):
    """Per-(row, k-block) activation / weight scale the kernel would load, using its exact index arithmetic."""
    sam, sak, sbn, sbk = strides
    kb = -(-k // block)
    fa, fb = sa.reshape(-1), sb.reshape(-1)
    a_out = torch.zeros(m, kb)
    b_out = torch.zeros(n, kb)
    for i in range(m):
        for j in range(kb):
            idx = i * sam + (j * sak if mode == 2 else 0)
            assert 0 <= idx < fa.numel(), f"activation scale index {idx} out of bounds ({fa.numel()})"
            a_out[i, j] = fa[idx]
    for r in range(n):
        for j in range(kb):
            idx = (r // sb_div if mode == 2 else r) * sbn + (j * sbk if mode == 2 else 0)
            assert 0 <= idx < fb.numel(), f"weight scale index {idx} out of bounds ({fb.numel()})"
            b_out[r, j] = fb[idx]
    if mode == 0:
        a_out, b_out = a_out[:1, :1].expand(m, kb), b_out[:1, :1].expand(n, kb)
    return a_out, b_out


@pytest.mark.parametrize("m", [1, 5])
@pytest.mark.parametrize("xg", ["per_tensor", "per_token", "block"])
@pytest.mark.parametrize("wg", ["per_tensor", "per_token", "block2d"])
def test_prepare_fp8_scales_matches_oracle_for_every_granularity_mix(m, xg, wg):
    n, k, block = 40, 300, 128
    torch.manual_seed(0)
    _, sx = quantize_fp8_ref(torch.randn(m, k), xg)
    _, sw = quantize_fp8_ref(torch.randn(n, k), wg)
    mode, sa, sb, strides, sb_div = prepare_fp8_scales(sx, sw, m, n, k, block)
    got_a, got_b = _emulate_scale_addressing(mode, sa, sb, strides, sb_div, m, n, k, block)
    exp_a = expand_scale(sx, m, k, block)[:, ::block]
    exp_b = expand_scale(sw, n, k, block)[:, ::block]
    assert torch.equal(got_a, exp_a) and torch.equal(got_b, exp_b), (mode, strides, sb_div)


def test_prepare_fp8_scales_rejects_mismatched_sizes():
    with pytest.raises(ValueError):
        prepare_fp8_scales(torch.ones(3), torch.ones(10), 5, 10, 64, 128)
    with pytest.raises(ValueError):
        prepare_fp8_scales(torch.ones(5), torch.ones(7), 5, 10, 64, 128)
    with pytest.raises(ValueError):
        prepare_fp8_scales(torch.ones(5, 1), torch.ones(2, 1), 5, 300, 300, 128)


# ---------------------------------------------------------------------------------------------------------------------
# FP4 scale swizzle re-derived independently (torchao / cuBLAS "blocked" layout)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rows,cols", [(1, 1), (128, 4), (130, 9), (257, 33)])
def test_fp4_scale_swizzle_matches_independent_to_blocked(rows, cols):
    torch.manual_seed(1)
    s = torch.randint(0, 255, (rows, cols), dtype=torch.uint8)
    rp, cp = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = torch.zeros(rp, cp, dtype=torch.uint8)
    padded[:rows, :cols] = s
    blocks = padded.view(rp // 128, 128, cp // 4, 4).permute(0, 2, 1, 3)  # [rb, cb, 128, 4]
    expect = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)  # per 128x4 atom: (r%32, r//32, c)
    assert torch.equal(swizzle_scales(s), expect)
    assert torch.equal(unswizzle_scales(expect, rows, cols), s)


# ---------------------------------------------------------------------------------------------------------------------
# misc contract / safety
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", ["../secrets", "a/b", "..", "", "x.y", "w4a16_gemm\\..\\x"])
def test_load_spec_rejects_path_like_names(bad):
    with pytest.raises(NotFound):
        load_spec(bad)


def test_pack_w4_rejects_negative_zero_points():
    w = torch.randn(8, 32)
    q, s, z = quantize_w4_rtn(w, 16, sym=False)
    z = z.to(torch.int16)
    z[0, 0] = -1
    with pytest.raises(ConfigError):
        pack_w4(q, s, z, 16)


def test_portable_repack_cache_is_not_stale_after_inplace_weight_update():
    torch.manual_seed(0)
    n, k = 70, 96
    lin1, lin2 = torch.nn.Linear(k, n), torch.nn.Linear(k, n)
    m1 = ops_gemm.KernelW4A16Linear.from_dense(lin1, group_size=32, layout=FRAGMENT_W4)
    m2 = ops_gemm.KernelW4A16Linear.from_dense(lin2, group_size=32, layout=FRAGMENT_W4)
    p1 = ops_gemm._portable(m1.packed())
    assert p1.layout.id == PORTABLE_W4.id and torch.equal(p1.dequantize(), m1.dequantize())
    assert ops_gemm._portable(m1.packed()) is p1  # cached across forward calls (module keeps the meta dict)
    m1.load_state_dict(m2.state_dict())  # in-place update of the buffers
    p2 = ops_gemm._portable(m1.packed())
    assert p2 is not p1
    assert torch.equal(p2.dequantize(), m2.dequantize())
    assert torch.equal(p2.qweight, repack(m2.packed(), PORTABLE_W4).qweight)


def test_failed_triton_launch_is_memoised_and_skipped():
    ops_gemm.reset_failure_memo()
    x = torch.zeros(3, 8, dtype=torch.float16)
    fk = ops_gemm._fkey("w4a16_linear", x, 16, 8)
    ops_gemm._fail("w4a16_linear", RuntimeError("compile error"), fk)
    ok, why = ops_gemm.triton_usable(x, fk)
    assert not ok and "failed earlier" in why
    assert ops_gemm._fkey("w4a16_linear", torch.zeros(4, 8, dtype=torch.float16), 16, 8) == fk  # same pow2 row bucket (3 and 4 rows)
    assert ops_gemm._fkey("w4a16_linear", torch.zeros(9, 8, dtype=torch.float16), 16, 8) != fk
    ops_gemm.reset_failure_memo()
    assert fk not in ops_gemm._FAILED


def test_native_fp4_selftest_rejects_wrong_results(monkeypatch):
    import forge.kernels.triton.gemm_fp4 as g4
    from forge.kernels.ref import fp4_block_gemm_ref

    cpu = torch.device("cpu")
    monkeypatch.setattr(g4, "mxfp4_gemm_native_triton", lambda xq, w, bias=None, **kw: torch.zeros(xq.rows, w.rows))
    assert ops_gemm._native_fp4_selftest(cpu) is False

    monkeypatch.setattr(g4, "mxfp4_gemm_native_triton", lambda xq, w, bias=None, **kw: fp4_block_gemm_ref(xq, w, bias))
    assert ops_gemm._native_fp4_selftest(cpu) is True

    def boom(*a, **k):
        raise RuntimeError("API changed")

    monkeypatch.setattr(g4, "mxfp4_gemm_native_triton", boom)
    assert ops_gemm._native_fp4_selftest(cpu) is False


# ---------------------------------------------------------------------------------------------------------------------
# Triton: interpreter (logic) + offline compiler (M == 1 integer specialisation)
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def review_report():
    pytest.importorskip("triton")
    script = Path(__file__).with_name("_kernels_gemm_review_interp.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT_JSON="))
    return json.loads(line[len("REPORT_JSON="):])


def test_fp8_kernel_every_scale_granularity_mix(review_report):
    rows = review_report["fp8_combos"]
    assert len(rows) == 2 * 3 * 3
    for r in rows:
        assert "!" not in r["w"], r
        assert r["err"] / max(r["scale"], 1e-9) < 1e-4, r
    assert review_report["fp8_bias"] < 1e-3
    assert review_report["fp8_bad_scale_raises"] is True


def test_int8_kernel_noncontiguous_weight_and_bad_scales(review_report):
    assert review_report["int8_noncontig_err"] == 0.0
    assert review_report["int8_bad_scale_raises"] is True


def test_fp4_kernel_nan_poison_and_malformed_container(review_report):
    for fmt, r in review_report["fp4_nan"].items():
        assert r["ref_nan_cols"] == [3] and r["out_nan_cols"] == [3], (fmt, r)
        assert r["malformed_raises"] is True


def test_fp4_fused_quantiser_poisons_nonfinite_blocks_like_the_oracle(review_report):
    for key, r in review_report["fp4_quant_nonfinite"].items():
        assert r["scales_equal"] and r["payload_equal"] and r["ts_equal"], (key, r)


@pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")
def test_splitk_reduce_kernel_compiles_when_m_is_1_offline():
    """Triton specialises integer args equal to 1 into constexpr; the old reduce kernel called ``M.to(tl.int64)`` and
    failed to compile for the decode shape M == 1 (silently falling back to the slow reference path)."""
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import kernels_attn_compile_harness as H
        import triton
        import triton.language as tl
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource

        @triton.jit
        def _probe(x_ptr, B: tl.constexpr):
            tl.store(x_ptr + tl.arange(0, B), tl.zeros([B], tl.float32))

        triton.compile(ASTSource(fn=_probe, signature={"x_ptr": "*fp32", "B": "constexpr"}, constexprs={"B": 16}),
                       target=GPUTarget("cuda", 80, 32))
    except Exception as e:  # pragma: no cover - environment dependent
        pytest.skip(f"offline Triton compilation unavailable: {e}")

    from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton

    q, s, z = quantize_w4_rtn(torch.randn(256, 512), 128, sym=False)
    p = pack_w4(q, s, z, 128)
    with H.recording() as recs:
        w4a16_gemm_triton(torch.randn(1, 512).half(), p, torch.randn(256).half(), num_sms=108, sm=80)
    names = {r.kernel.__name__ for r in recs}
    assert "_splitk_reduce_kernel" in names, "M=1 must exercise split-K on 108 SMs"
    for rec in recs:
        fn = rec.kernel
        arg_names = list(fn.arg_names)
        values = dict(zip(arg_names, rec.args))
        values.update({k: v for k, v in rec.kwargs.items() if k in arg_names})
        cexp = {pp.name for pp in fn.params if pp.is_constexpr}
        sig, cs = {}, {}
        for nme in arg_names:
            v = values[nme]
            if nme in cexp or (isinstance(v, int) and not isinstance(v, bool) and v == 1):  # mimic the runtime specialisation
                sig[nme], cs[nme] = "constexpr", v
            else:
                sig[nme] = H._sig_entry(v)
        opts = {k: v for k, v in rec.kwargs.items() if k in ("num_warps", "num_stages")}
        triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cs), target=GPUTarget("cuda", 80, 32), options=opts)
