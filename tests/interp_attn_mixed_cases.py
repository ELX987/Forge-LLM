"""Triton mixed-precision attention kernel in the **Triton CPU interpreter** (no GPU) vs the oracle.

Not collected directly (no ``test_`` prefix): ``test_kernels_attn_triton_interp.py`` runs this file in a *subprocess* with
``TRITON_INTERPRET=1`` because the interpreter must be enabled before ``triton`` is first imported.  This validates the
kernel *logic* (packed-byte decoding, block-scale folding, two-pass online softmax, split-KV merge, window, softcap, sink,
GQA) against ``ref.attn_mixed`` for NVFP4 / MXFP4 / ternary; it does NOT validate GPU compilation or performance (see
``tests_gpu/test_kernels_attn_mixed.py``).
"""
from __future__ import annotations

import os

os.environ["TRITON_INTERPRET"] = "1"  # must precede the first `import triton`

import importlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton")

from forge.kernels.ref.attn_kv import build_paged_pools  # noqa: E402
from forge.kernels.ref.attn_mixed import mixed_attention_ref  # noqa: E402


@pytest.fixture(scope="module")
def km():
    return SimpleNamespace(attn_mixed=importlib.import_module("forge.kernels.triton.attn_mixed"))


def _setup(b, hq, hkv, d, lens, ps, fmt, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, hq, d, generator=g)
    k = torch.randn(b, hkv, max(lens), d, generator=g)
    v = torch.randn(b, hkv, max(lens), d, generator=g)
    pools = build_paged_pools(k, v, lens, ps, fmt, generator=g)
    kw = dict(kv_dtype=fmt, k_scale=pools["k_scale"], v_scale=pools["v_scale"],
              k_block_scale=pools.get("k_block_scale"), v_block_scale=pools.get("v_block_scale"),
              tensor_scale=pools.get("tensor_scale"))
    return q, pools, kw


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "ternary"])
@pytest.mark.parametrize("splits", [1, 2, 3])
def test_mixed_kernel_matches_oracle(km, fmt, splits):
    lens = [40, 19]
    q, p, kw = _setup(2, 4, 2, 32, lens, 16, fmt)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = mixed_attention_ref(*args, **kw)
    out = km.attn_mixed.mixed_attention_triton(*args, num_splits=splits, **kw)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "ternary"])
@pytest.mark.parametrize("gsize", [1, 2, 4, 8])
def test_mixed_kernel_gqa_groups(km, fmt, gsize):
    lens = [1, 17, 33, 0]
    q, p, kw = _setup(4, 2 * gsize, 2, 32, lens, 16, fmt, seed=1)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = mixed_attention_ref(*args, **kw)
    out = km.attn_mixed.mixed_attention_triton(*args, num_splits=2, **kw)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "ternary"])
@pytest.mark.parametrize("window", [1, 5, 16, 33])
def test_mixed_kernel_sliding_window(km, fmt, window):
    lens = [50, 20, 3]
    q, p, kw = _setup(3, 4, 2, 32, lens, 16, fmt, seed=2)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = mixed_attention_ref(*args, window=window, **kw)
    out = km.attn_mixed.mixed_attention_triton(*args, window=window, num_splits=1, **kw)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_mixed_kernel_softcap(km, fmt):
    lens = [40, 19]
    q, p, kw = _setup(2, 4, 2, 32, lens, 16, fmt, seed=3)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = mixed_attention_ref(*args, softcap=15.0, **kw)
    out = km.attn_mixed.mixed_attention_triton(*args, softcap=15.0, num_splits=2, **kw)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4", "ternary"])
def test_mixed_kernel_sink(km, fmt):
    lens = [40, 19]
    q, p, kw = _setup(2, 4, 2, 32, lens, 16, fmt, seed=4)
    sink = torch.tensor([0.5, -0.3, 1.0, 0.0])
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = mixed_attention_ref(*args, sink=sink, **kw)
    out = km.attn_mixed.mixed_attention_triton(*args, sink=sink, num_splits=2, **kw)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("fmt", ["nvfp4", "ternary"])
@pytest.mark.parametrize("d,ps", [(64, 32), (128, 16)])
def test_mixed_kernel_head_dims(km, fmt, d, ps):
    lens = [40, 19]
    q, p, kw = _setup(2, 4, 2, d, lens, ps, fmt, seed=5)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref = mixed_attention_ref(*args, **kw)
    out = km.attn_mixed.mixed_attention_triton(*args, num_splits=2, **kw)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)


def test_mixed_kernel_return_lse(km):
    lens = [45, 30]
    q, p, kw = _setup(2, 4, 2, 32, lens, 16, "nvfp4", seed=6)
    args = (q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"])
    ref, lse_ref = mixed_attention_ref(*args, return_lse=True, **kw)
    out, lse = km.attn_mixed.mixed_attention_triton(*args, return_lse=True, num_splits=2, **kw)
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(lse, lse_ref, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("bad", ["missing_scale", "bad_scale_shape", "oob_page", "too_many_pages"])
def test_mixed_kernel_rejects_malformed_launch(km, bad):
    # the launcher must reject out-of-bounds page ids / oversized seq lens and malformed block scales with a clean error
    q, p, kw = _setup(2, 4, 2, 32, [40, 19], 16, "nvfp4", seed=7)
    args = [q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"]]
    if bad == "missing_scale":
        kw = dict(kw, k_block_scale=None)
    elif bad == "bad_scale_shape":
        kw = dict(kw, v_block_scale=p["v_block_scale"][:, :, :, :1])  # truncated block axis
    elif bad == "oob_page":
        bt = p["block_tables"].clone()
        bt[0, 0] = p["k_pool"].shape[0]  # page id == P (out of range)
        args[3] = bt
    else:  # too_many_pages: a seq_len needing more pages than the block-table row holds
        args[4] = torch.full_like(p["seq_lens"], int(p["block_tables"].shape[1] * 16 + 16))
    with pytest.raises(ValueError):
        km.attn_mixed.mixed_attention_triton(*args, num_splits=1, **kw)


def test_mixed_kernel_requires_tensor_scale_for_nvfp4(km):
    # the nvfp4 pool stores its block scales pre-divided by the tensor scale, so the kernel refuses a missing one
    q, p, kw = _setup(1, 4, 2, 32, [40], 16, "nvfp4", seed=8)
    with pytest.raises(ValueError):
        km.attn_mixed.mixed_attention_triton(q, p["k_pool"], p["v_pool"], p["block_tables"], p["seq_lens"],
                                             num_splits=1, **dict(kw, tensor_scale=None))
