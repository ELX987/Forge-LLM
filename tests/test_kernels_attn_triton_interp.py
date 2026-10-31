"""Runs the Triton-interpreter kernel suites (CPU only, no GPU) in subprocesses.

``TRITON_INTERPRET=1`` has to be set before ``triton`` is imported for the first time in a process, which cannot be guaranteed
inside a shared pytest session, so each suite (``interp_attn_cases.py``, ``interp_moe_util_cases.py``) is executed by a fresh
``python -m pytest`` child.  Skipped when ``triton`` is not installed (it is an optional dependency).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

pytestmark = pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="triton not installed")


def _run(path: Path) -> None:
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run([sys.executable, "-m", "pytest", str(path), "-q", "-x", "-p", "no:cacheprovider"], cwd=str(ROOT),
                          env=env, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, "interpreter suite failed:\n" + proc.stdout[-6000:] + proc.stderr[-2000:]


@pytest.mark.slow
def test_attention_kernels_in_triton_interpreter():
    _run(HERE / "interp_attn_cases.py")


@pytest.mark.slow
def test_moe_and_util_kernels_in_triton_interpreter():
    _run(HERE / "interp_moe_util_cases.py")
