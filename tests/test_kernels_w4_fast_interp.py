"""Nibble-split W4A16 kernels executed in the Triton CPU interpreter (needs ``triton`` importable; skipped otherwise).

Checks kernel LOGIC against the fp32 oracle; does not compile for or run on a GPU (tests_gpu/ does that)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")
SCRIPT = Path(__file__).with_name("_kernels_w4_fast_interp_check.py")


@pytest.fixture(scope="module")
def report():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT_JSON="))
    return json.loads(line[len("REPORT_JSON="):])


def test_all_impls_match_oracle(report):
    rows = [r for r in report["rows"] if "tile_k" in r]
    assert {r["impl"] for r in rows} == {"dot", "gemv", "legacy", None}
    assert len(rows) > 200
    for r in rows:
        assert r["err"] / max(r["scale"], 1e-9) < 2e-3, r  # fp16 output rounding is ~5e-4 relative


def test_split_k_deterministic_is_bit_identical(report):
    flags = [r["bit_identical"] for r in report["rows"] if "bit_identical" in r]
    assert flags and all(flags)


def test_default_config_uses_nibble_split_kernel_without_fallback(report):
    rows = [r for r in report["rows"] if "default_err" in r]
    assert rows and all(r["default_err"] / r["scale"] < 2e-3 and r["fast_broken"] == 0 for r in rows)


def test_act_order_weights_are_column_sorted_once(report):
    assert all(r["sorted_cached"] for r in report["rows"] if "sorted_cached" in r)


def test_bf16_gemv_and_kill_switch(report):
    assert report["bf16_gemv"]["err"] / report["bf16_gemv"]["scale"] < 1e-2
    assert report["kill_switch"]["err"] < 0.05
