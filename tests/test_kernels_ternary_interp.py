"""Ternary Triton kernels in the Triton CPU interpreter vs the torch oracle (skipped without ``triton``).

Checks kernel LOGIC (code-plane decoding, group scales, tails, bias, masks), not GPU compilation speed or numerics on hardware."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("triton")
SCRIPT = Path(__file__).with_name("_kernels_ternary_interp_check.py")


@pytest.fixture(scope="module")
def report():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=900, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("REPORT_JSON="))
    return json.loads(line[len("REPORT_JSON="):])


def test_gemm_matches_oracle_for_every_scale_layout(report):
    rows = report["gemm"]
    assert len(rows) == 8 * 4 * 2
    assert {r["group"] for r in rows} == {-1, 0, 32, 64, 128, 256}
    for r in rows:
        assert r["err"] <= 1e-5 * max(1.0, r["scale"]), r  # int32 accumulation is exact; only the fp32 summation order over groups can differ


def test_row_scale_with_k_not_a_multiple_of_the_tile(report):
    assert any(r["group"] == -1 and r["k"] == 272 for r in report["gemm"])


def test_activation_quantiser_is_bit_exact_with_the_oracle(report):
    for r in report["quant"]:
        assert r["q_equal"] and r["s_equal"], r


def test_bf16_output_error_is_bf16_rounding_only(report):
    assert report["bf16_rel"] < 2 ** -7


def test_dispatcher_takes_the_triton_path_and_matches_the_oracle(report):
    d = report["dispatch"]
    assert d["path"] == "triton" and d["quant_path"] == "triton" and d["shape"] == [2, 3, 32] and d["err"] < 1e-5


def test_supported_group_rules():
    from forge.kernels.triton.gemm_ternary import supported_group

    assert all(supported_group(g, 512) for g in (-1, 0, 32, 64, 128, 256))
    assert not supported_group(16, 512) and not supported_group(48, 480) and not supported_group(128, 200) and not supported_group(64, 100)
    assert not supported_group(128, 192)  # 192 % 128 != 0
