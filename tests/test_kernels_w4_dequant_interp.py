"""W4 dequantise kernel + dequant_matmul dispatch executed in the Triton CPU interpreter (needs ``triton``; skipped otherwise).

Checks kernel LOGIC against the fp32 oracle; does not compile for or run on a GPU (tests_gpu/test_kernels_w4_dequant_gpu.py does)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")
SCRIPT = Path(__file__).with_name("_kernels_w4_dequant_interp_check.py")


@pytest.fixture(scope="module")
def report():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT_JSON="))
    return json.loads(line[len("REPORT_JSON="):])


def test_dequant_matches_oracle_for_all_scale_modes(report):
    rows = [r for r in report["rows"] if r["kind"] == "dequant"]
    assert len(rows) == 8 * 2 * 3
    assert {r["mode"] for r in rows} == {0, 1, 2}  # single group / per-word groups / per-element (act-order)
    assert {r["dt"] for r in rows} == {"torch.float16", "torch.bfloat16"}
    for r in rows:
        assert r["rel"] <= r["tol"], r


def test_row_chunk_only_writes_its_rows(report):
    rows = [r for r in report["rows"] if r["kind"] == "chunk"]
    assert rows and all(r["ok"] and r["rest_untouched"] for r in rows)


def test_dequant_matmul_matches_oracle(report):
    rows = [r for r in report["rows"] if r["kind"] == "matmul"]
    assert len(rows) == 8
    for r in rows:
        assert r["err"] / r["scale"] < 2e-3, r  # fp16 output rounding ~5e-4 relative


def test_chunked_matmul_equals_single_chunk(report):
    m = report["multi"]
    assert m["n_chunks"] == 4 and m["identical"]
    assert m["err1"] / m["scale"] < 2e-3 and m["err2"] / m["scale"] < 2e-3


def test_default_routing_and_forced_impl(report):
    assert report["big_default"]["routed"] and report["big_default"]["err"] < 0.05  # M=72 -> dequant_matmul with no config
    assert report["small_default"]["not_routed"]  # M=3 keeps the fused gemv kernel
    assert report["forced"]["routed"] and report["forced"]["err"] < 0.05  # FORGE_W4_IMPL=dequant_matmul
