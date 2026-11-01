"""GPU tests (auto-skipped with fewer than 2 CUDA devices): gate G1 for the sharded ternary trainer -- NCCL collectives and loss/weight parity of
``parallel.strategy=zero_ternary`` against the unsharded trainer.  Never run by the developer; the first run on a multi-GPU host decides the
'GPU-validated' cells for zero_ternary in docs/TERNARY.md (all start at 'no')."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[1]
NGPU = torch.cuda.device_count() if torch.cuda.is_available() else 0
needs_two = pytest.mark.skipif(NGPU < 2, reason="needs >= 2 CUDA devices")


@needs_two
def test_nccl_collectives_used_by_zero_ternary():
    from forge.train.collectives import run_collectives_check

    r = run_collectives_check("nccl", min(NGPU, 4), size_mib=16.0, timeout=300)
    assert r["ok"] is True, r
    assert r["checks"]["reduce_scatter"] is True and r["checks"]["all_gather_uint8"] is True
    print(f"all_reduce bus bandwidth {r['all_reduce_busbw_gbs']} GB/s over {r['world']} GPUs")


@needs_two
@pytest.mark.parametrize("nproc", [2, 4])
def test_sharded_equals_unsharded_over_nccl(nproc, tmp_path):
    if NGPU < nproc:
        pytest.skip(f"needs {nproc} CUDA devices")
    env = {**os.environ, "FORGE_PARITY_DIR": str(tmp_path)}
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", str(ROOT / "scripts" / "zero_ternary_parity.py"),
           "--device", "cuda", "--backend", "nccl", "--steps", "4", "--atol", "1e-4"]
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    line = next(ln for ln in p.stdout.splitlines() if ln.startswith("{"))
    rep = json.loads(line)
    assert rep["ok"] and rep["world"] == nproc, rep
