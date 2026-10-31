"""Integration: every module under forge/ imports on a CPU-only host without triton / pynvml, and the kernel ops
facade, catalog and bench registry cover all kernel families."""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

_SCRIPT = textwrap.dedent(
    """
    import importlib, pkgutil, sys
    sys.modules["triton"] = None      # simulate hosts without triton / pynvml
    sys.modules["pynvml"] = None
    import forge
    bad = []
    for m in pkgutil.walk_packages(forge.__path__, "forge."):
        if m.name.startswith("forge.web") or m.name.endswith("__main__"):
            continue
        if m.name == "forge.hw._bench_kernels":   # triton kernels, imported lazily by design
            continue
        try:
            importlib.import_module(m.name)
        except BaseException as e:
            bad.append((m.name, repr(e)[:200]))
    print(bad)
    sys.exit(1 if bad else 0)
    """
)


def test_every_module_imports_without_triton_or_pynvml():
    p = subprocess.run([sys.executable, "-c", _SCRIPT], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stdout + p.stderr


def test_ops_facade_reexports_every_family():
    from forge.kernels import ops

    assert all(ops.available_ops().values()), [k for k, v in ops.available_ops().items() if not v]
    for name in ("w4a16_linear", "fp4_linear", "paged_decode_attention", "grouped_gemm", "moe_unpermute", "hadamard_transform",
                 "segmented_lora", "rejection_verify"):
        assert callable(getattr(ops, name)), name


def test_catalog_and_bench_cover_all_families():
    from forge.kernels.bench import collect_cases
    from forge.kernels.catalog import list_kernels

    rows = list_kernels()
    assert {r["family"] for r in rows} >= {"gemm", "attention", "moe", "util", "sparse", "spec"}
    assert all("available" in r and "reasons" in r for r in rows)
    cases = collect_cases()
    fams = {c.family for c in cases.values()}
    assert fams >= {"gemm", "attention", "moe", "util"}, fams
