"""Triton GEMM kernels executed in the Triton CPU interpreter (needs ``triton`` importable; skipped otherwise).

This checks kernel LOGIC against the torch oracles.  It does not compile for, or run on, any NVIDIA GPU
(tests_gpu/ does that on the user's vast.ai host).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")
SCRIPT = Path(__file__).with_name("_kernels_gemm_interp_check.py")


@pytest.fixture(scope="module")
def report():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=600, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT_JSON="))
    return json.loads(line[len("REPORT_JSON="):])


def rel(r):
    return r["err"] / max(r["scale"], 1e-9)


def test_w4a16_kernel_matches_oracle(report):
    rows = [r for r in report["w4a16"] if "err" in r]
    assert len(rows) == 16
    for r in rows:
        assert rel(r) < 2e-3, r  # fp16 activations: dequantised tile rounded to fp16 (<= 2^-11 per weight)
    assert all(r["bit_identical"] for r in report["w4a16"] if "bit_identical" in r)


def test_w4a16_dispatcher_repacks_fragment_layout_and_takes_triton_path(report):
    d = report["dispatch_w4a16"]
    assert d["path"] == "triton" and d["err"] < 0.05


def test_w8a16_kernel_matches_oracle(report):
    for r in report["w8a16"]:
        assert rel(r) < 2e-3, r


def test_int8_kernel_is_bit_exact(report):
    r = report["int8"]
    assert r["quant_code_mismatch"] == 0 and r["quant_scale_err"] == 0.0 and r["gemm_err"] == 0.0


def test_fp8_kernel_scale_modes(report):
    f = report["fp8"]
    for mode in ("per_tensor", "per_token", "block"):
        assert f[mode]["err"] / f[mode]["scale"] < 1e-4, (mode, f[mode])
    assert f["quant_scale_err"] == 0.0 and f["block_scale_err"] == 0.0 and f["rms_scale_err"] < 1e-5


def test_fp4_kernels(report):
    f = report["fp4"]
    for key in ("mxfp4_logical", "mxfp4_swizzled", "nvfp4_logical", "nvfp4_swizzled"):
        r = f[key]
        assert r["gemm_err"] / r["gemm_scale"] < 5e-3, (key, r)  # fp16 output rounding
    for key in ("mxfp4_logical", "mxfp4_swizzled", "nvfp4_logical", "nvfp4_swizzled"):
        # E8M0 / E4M3 scale conversion and E2M1 rounding are integer + bit-cast arithmetic: bit exact vs the oracle quantiser
        assert f[key]["payload_mismatch"] == 0 and f[key]["scale_mismatch"] == 0, f[key]
    if f["dot_scaled_available"] and "native_dot_scaled_err" in f:
        assert f["native_dot_scaled_err"] < 1e-3  # packing convention of the dot_scaled kernel verified by the interpreter
