"""Runs the adversarial-review Triton-interpreter suite (CPU only, no GPU) in a subprocess (see ``interp_attn_review_cases.py``)."""
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


@pytest.mark.slow
def test_review_cases_in_triton_interpreter():
    env = dict(os.environ)
    env["TRITON_INTERPRET"] = "1"
    env["PYTHONHASHSEED"] = "0"
    proc = subprocess.run([sys.executable, "-m", "pytest", str(HERE / "interp_attn_review_cases.py"), "-q", "-p",
                           "no:cacheprovider"], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1500)
    assert proc.returncode == 0, "review interpreter suite failed:\n" + proc.stdout[-8000:] + proc.stderr[-2000:]
