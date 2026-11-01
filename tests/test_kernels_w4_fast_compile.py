"""Offline (no GPU) Triton *compilation* of the W4A16 kernels for the target architectures, with the runtime's
"divisible by 16" argument specialisation, plus structural checks of the generated PTX.

Skipped when ``triton`` (or its offline compiler) is unavailable.  Validates compilability and the memory-access shape
(vectorised 16-byte weight loads instead of the legacy kernel's scalar 4-byte gathers) -- NOT numerics, NOT speed.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("triton")
sys.path.insert(0, str(Path(__file__).parent))

from forge.kernels.layouts import pack_w4, quantize_w4_rtn  # noqa: E402
from forge.kernels.triton.gemm_w4a16 import w4a16_gemm_triton  # noqa: E402


def _compile(rec, sm):
    import kernels_attn_compile_harness as H
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    fn = rec.kernel
    names = list(fn.arg_names)
    values = dict(zip(names, rec.args))
    values.update({k: v for k, v in rec.kwargs.items() if k in names})
    opts = {k: v for k, v in rec.kwargs.items() if k in ("num_warps", "num_stages")}
    cexp = {p.name for p in fn.params if p.is_constexpr}
    sig, cs, attrs = {}, {}, {}
    for i, n in enumerate(names):
        v = values[n]
        if n in cexp or (isinstance(v, int) and not isinstance(v, bool) and v == 1):
            sig[n], cs[n] = "constexpr", v
            continue
        sig[n] = H._sig_entry(v)
        aligned = (isinstance(v, torch.Tensor) and v.data_ptr() % 16 == 0) or (isinstance(v, int) and not isinstance(v, bool) and v % 16 == 0)
        if aligned:  # what the JIT's specialisation records for pointers / ints divisible by 16
            attrs[(i,)] = [["tt.divisibility", 16]]
    return triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cs, attrs=attrs), target=GPUTarget("cuda", sm, 32), options=opts)


def _ptx(cfg, *, sm=120, m=1, n=256, k=512, gs=128, sym=False, dtype=torch.bfloat16, act_order=False):
    import kernels_attn_compile_harness as H

    q, s, z = quantize_w4_rtn(torch.randn(n, k), gs, sym=sym)
    gi = (torch.arange(k) // gs)[torch.randperm(k)].to(torch.int32) if act_order else None
    p = pack_w4(q, s, z, gs, None, gi)
    try:
        with H.recording() as recs:
            w4a16_gemm_triton(torch.randn(m, k).to(dtype), p, torch.randn(n).to(dtype), config=cfg, sm=sm, num_sms=188)
        rec = next(r for r in recs if r.kernel.__name__.startswith(("_w4", "_wq")))
        return _compile(rec, sm).asm["ptx"]
    except Exception as e:  # pragma: no cover - environment dependent (offline compiler / ptxas missing)
        if "ptxas" in str(e).lower() or "target" in str(e).lower():
            pytest.skip(f"offline Triton compilation unavailable: {e}")
        raise


def _count(ptx, pat):
    return len(re.findall(pat, ptx))


DOT = dict(impl="dot", tile_m=16, tile_n=64, tile_k=128, split_k=1, stages=3, warps=4)
GEMV = dict(impl="gemv", tile_n=32, tile_k=128, split_k=1, stages=3, warps=4)
LEGACY = dict(impl="legacy", tile_m=16, tile_n=64, tile_k=128, split_k=1, stages=3, warps=4)


def test_dot_kernel_loads_weights_as_16_byte_vectors_unlike_legacy():
    new, old = _ptx(DOT), _ptx(LEGACY)
    assert _count(new, r"cp\.async\.cg\.shared\.global [^;]*0x10") >= 2  # 16-byte weight vectors
    assert _count(new, r"cp\.async\.ca\.shared\.global [^;]*0x4,") < 20  # no per-element 4-byte weight gathers
    assert _count(old, r"cp\.async\.ca\.shared\.global [^;]*0x4,") > 100  # legacy: redundant scalar word loads (the diagnosis)


def test_gemv_kernel_loads_weights_as_16_byte_vectors():
    assert _count(_ptx(GEMV), r"ld\.global\.v4\.b32") >= 1


@pytest.mark.parametrize("sm", [80, 89, 90, 120])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cfg", [DOT, {**DOT, "split_k": 4, "tile_n": 32, "tile_k": 256}, GEMV, {**GEMV, "split_k": 4, "tile_k": 256}])
def test_compiles_for_target(sm, dtype, cfg):
    assert ".entry" in _ptx(cfg, sm=sm, dtype=dtype)


@pytest.mark.parametrize("cfg", [DOT, GEMV])
def test_compiles_for_odd_shapes_zeros_actorder_and_group32(cfg):
    assert ".entry" in _ptx(cfg, sm=120, m=3, n=70, k=520, gs=32, sym=False, act_order=False)  # EVEN_K / EVEN_N false, multi-group
    assert ".entry" in _ptx(cfg, sm=120, m=1, n=256, k=512, sym=True)  # symmetric, no zero tensor
    assert ".entry" in _ptx(cfg, sm=120, m=2, n=256, k=512, act_order=True)  # in-kernel x gather
