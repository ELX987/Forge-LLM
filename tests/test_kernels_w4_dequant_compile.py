"""Offline (no GPU) Triton compilation of the W4 dequantise kernel for the target architectures + structural PTX checks.

Reuses the offline-compile helpers of test_kernels_w4_fast_compile.py.  Skipped when ``triton`` (or its offline compiler) is
unavailable.  Validates compilability and the memory-access shape (16-byte packed-word loads, 16-byte coalesced output
stores) -- NOT numerics (interpreter tests), NOT speed (scripts/bench_w4a16.py on the GPU host).

Also pins the PTX evidence behind the new ``dot`` default tile (tile_n 128 for M <= 16): the activation gather is a fixed
number of scalar 16-bit loads per program and K tile, independent of tile_n, while the MMA count scales with tile_n, so a wider
N tile amortises the gather over proportionally more tensor-core work.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")
sys.path.insert(0, str(Path(__file__).parent))

from test_kernels_w4_fast_compile import _count, _ptx  # noqa: E402

@pytest.fixture(autouse=True)
def _isolate_global_rng():
    """These tests draw random tensors; keep the global torch RNG stream of the rest of the suite untouched (some older tests
    use unseeded random inputs with tight tolerances)."""
    with torch.random.fork_rng():
        yield


DQ = dict(impl="dequant_matmul", tile_n=32, tile_k=128, warps=4)
LD_V4 = r"ld\.global(\.[a-z0-9]+)*\.v4\.b32"
ST_V4 = r"st\.global(\.[a-z0-9]+)*\.v4\.b32"
ST_B16 = r"st\.global(\.[a-z0-9]+)*\.b16"
LD_B16 = r"ld\.global(\.[a-z0-9]+)*\.b16"


def test_dequant_kernel_uses_16_byte_loads_and_stores():
    ptx = _ptx(DQ, sm=120, m=64, n=256, k=512)
    assert _count(ptx, LD_V4) >= 1, "packed weight words are not loaded as 16-byte vectors"
    assert _count(ptx, ST_V4) >= 1, "dequantised output is not stored as 16-byte vectors"
    assert _count(ptx, ST_B16) == 0, "scalar 16-bit output stores: the store is not vectorised"


@pytest.mark.parametrize("sm", [80, 89, 90, 120])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cfg", [DQ, {**DQ, "tile_n": 64, "tile_k": 256, "warps": 8}, {**DQ, "tile_n": 16, "tile_k": 64}])
def test_compiles_for_target(sm, dtype, cfg):
    assert ".entry" in _ptx(cfg, sm=sm, dtype=dtype, m=64)


@pytest.mark.parametrize("kw", [
    dict(n=70, k=520, gs=32, sym=False),      # ragged rows / K, several groups per tile
    dict(n=256, k=512, gs=128, sym=True),     # symmetric: no zero tensor
    dict(n=256, k=512, gs=64, sym=False),     # two groups per 128-wide tile -> per-word scale gather
    dict(n=256, k=512, gs=128, act_order=True),  # per-element g_idx
    dict(n=256, k=512, gs=-1, sym=True),      # per-channel
])
def test_compiles_for_odd_shapes_sym_group_and_actorder(kw):
    assert ".entry" in _ptx(DQ, sm=120, m=64, **kw)


def test_dot_gather_is_independent_of_tile_n_while_mma_scales():
    def stats(tn):
        ptx = _ptx(dict(impl="dot", tile_m=16, tile_n=tn, tile_k=128, split_k=1, stages=3, warps=4), sm=120, m=16, n=1024, k=1024)
        return _count(ptx, r"mma\.sync"), _count(ptx, LD_B16)

    (mma64, ld64), (mma128, ld128), (mma256, ld256) = stats(64), stats(128), stats(256)
    assert mma128 == 2 * mma64 and mma256 == 4 * mma64
    assert max(ld64, ld128, ld256) <= 1.5 * min(ld64, ld128, ld256)  # ~constant: the activation gather does not grow with tile_n
