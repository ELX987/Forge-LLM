"""GPU validation of mixed-precision block-scaled attention (``ops_attn.mixed_attention``) vs the CPU oracle.

Auto-skipped without CUDA.  Run on the GPU host (``docs/VAST_AI.md``)::

    pytest tests_gpu/test_kernels_attn_mixed.py -q -s

This is the *gate* that upgrades the KERNELS.md validation cell for ``mixed_attention`` from "no" to a measured chip.
It compares the Triton kernel against the pure-torch oracle over the Plan §6.7 edge-case list (ragged lengths, empty
sequences, GQA group sizes, page sizes, head dims, windows, softcap, sink, split counts) for every block-scaled format,
and checks determinism and the fp8/int8/int4 delegation path.
"""
from __future__ import annotations

import pytest
import torch

from forge.kernels.ops_attn import mixed_attention
from forge.kernels.ref.attn_kv import build_paged_pools
from forge.kernels.ref.attn_mixed import mixed_attention_ref

pytestmark = pytest.mark.gpu

BLOCK_FORMATS = ("nvfp4", "mxfp4", "ternary")


def _setup(b, hq, hkv, d, lens, ps, fmt, seed=0, device="cuda"):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hq, d, generator=g)
    k = torch.randn(b, hkv, max(lens), d, generator=g)
    v = torch.randn(b, hkv, max(lens), d, generator=g)
    pools = build_paged_pools(k, v, lens, ps, fmt, generator=g)
    kw = dict(kv_dtype=fmt, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
              k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
              tensor_scale=pools.get("tensor_scale"))
    kw_gpu = {k_: (v_.to(device) if torch.is_tensor(v_) else v_) for k_, v_ in kw.items()}
    return (q.to(device), pools["k_pool"].to(device), pools["v_pool"].to(device),
            pools["block_tables"].to(device), pools["seq_lens"].to(device), kw_gpu)


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
@pytest.mark.parametrize("gsize", [1, 2, 4, 8])
@pytest.mark.parametrize("page", [16, 32])
def test_mixed_attention_matches_oracle(fmt, gsize, page):
    lens = [1, 17, 33, 128, 0]
    q, kp, vp, bt, sl, kw = _setup(len(lens), 2 * gsize, 2, 32, lens, page, fmt)
    ref = mixed_attention_ref(q.cpu(), kp.cpu(), vp.cpu(), bt.cpu(), sl.cpu(),
                              **{k_: (v_.cpu() if torch.is_tensor(v_) else v_) for k_, v_ in kw.items()})
    for splits in (1, 4):
        out = mixed_attention(q, kp, vp, bt, sl, num_splits=splits, **kw)
        torch.testing.assert_close(out.float().cpu(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
@pytest.mark.parametrize("window", [1, 5, 64])
def test_mixed_attention_window(fmt, window):
    q, kp, vp, bt, sl, kw = _setup(3, 4, 2, 32, [128, 64, 3], 16, fmt, seed=1)
    ref = mixed_attention_ref(q.cpu(), kp.cpu(), vp.cpu(), bt.cpu(), sl.cpu(), window=window,
                              **{k_: (v_.cpu() if torch.is_tensor(v_) else v_) for k_, v_ in kw.items()})
    out = mixed_attention(q, kp, vp, bt, sl, window=window, **kw)
    torch.testing.assert_close(out.float().cpu(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_mixed_attention_softcap_and_sink(fmt):
    q, kp, vp, bt, sl, kw = _setup(2, 4, 2, 32, [64, 40], 16, fmt, seed=2)
    sink = torch.tensor([0.5, -0.3, 1.0, 0.0], device="cuda")
    ref = mixed_attention_ref(q.cpu(), kp.cpu(), vp.cpu(), bt.cpu(), sl.cpu(), softcap=12.0, sink=sink.cpu(),
                              **{k_: (v_.cpu() if torch.is_tensor(v_) else v_) for k_, v_ in kw.items()})
    out = mixed_attention(q, kp, vp, bt, sl, softcap=12.0, sink=sink, **kw)
    torch.testing.assert_close(out.float().cpu(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("d,page", [(64, 32), (128, 16), (256, 64)])
def test_mixed_attention_head_dims(d, page):
    q, kp, vp, bt, sl, kw = _setup(2, 4, 2, d, [64, 33], page, "nvfp4", seed=3)
    ref = mixed_attention_ref(q.cpu(), kp.cpu(), vp.cpu(), bt.cpu(), sl.cpu(),
                              **{k_: (v_.cpu() if torch.is_tensor(v_) else v_) for k_, v_ in kw.items()})
    out = mixed_attention(q, kp, vp, bt, sl, **kw)
    torch.testing.assert_close(out.float().cpu(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("fmt", BLOCK_FORMATS)
def test_mixed_attention_deterministic(fmt):
    q, kp, vp, bt, sl, kw = _setup(2, 4, 2, 64, [96, 50], 16, fmt, seed=4)
    a = mixed_attention(q, kp, vp, bt, sl, num_splits=4, **kw)
    b = mixed_attention(q, kp, vp, bt, sl, num_splits=4, **kw)
    torch.testing.assert_close(a, b, atol=0, rtol=0)  # no atomics -> bit-reproducible for a fixed split count


@pytest.mark.parametrize("fmt", ["bf16", "fp8_e4m3", "int8", "int4"])
def test_mixed_attention_delegates_float_formats(fmt):
    q, kp, vp, bt, sl, kw = _setup(2, 4, 2, 32, [64, 40], 16, fmt, seed=5)
    ref = mixed_attention_ref(q.cpu(), kp.cpu(), vp.cpu(), bt.cpu(), sl.cpu(),
                              **{k_: (v_.cpu() if torch.is_tensor(v_) else v_) for k_, v_ in kw.items()})
    out = mixed_attention(q, kp, vp, bt, sl, **kw)
    tol = 2e-3 if fmt == "bf16" else 5e-2  # int4/fp8 carry real quantisation error vs the (also quantised) oracle
    torch.testing.assert_close(out.float().cpu(), ref.float(), atol=tol, rtol=tol)
