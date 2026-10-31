"""GPU-only checks for forge.models (auto-skipped without CUDA by conftest). NOT run on the development machine.

These compare the CUDA paths (SDPA/flash, paged decode kernel, grouped GEMM, FP8 linear) against the CPU/torch reference
paths of the same module.  Kernel-backed paths are skipped when ``forge.kernels`` ops are not importable on the host.
"""
from __future__ import annotations

import copy
import os

import pytest
import torch

pytestmark = pytest.mark.gpu

from forge.models import DenseKVCache, PagedKVView, SimplePagedPool, build_tiny_model, quantize_linears  # noqa: E402
from forge.models.attention import AttnMeta  # noqa: E402


def _cuda_model(dtype=torch.bfloat16, **ov):
    m = build_tiny_model(**ov)
    return m.to(device="cuda", dtype=dtype).eval()


def _tf32_forced_by_env() -> bool:
    """NVIDIA NGC PyTorch containers export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1, which forces cuBLAS TF32 for fp32
    matmuls regardless of ``torch.backends.cuda.matmul.allow_tf32`` / ``set_float32_matmul_precision`` (it cannot be
    undone in-process).  TF32 has a 10-bit mantissa (~1e-3 relative per matmul), so fp32 logits then differ from CPU by
    a few 1e-4 of the max logit - that is the host configuration, not a model bug."""
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "") == "1":
        return True
    # measure instead of trusting env alone: true fp32 matmul error vs an fp64 reference is ~1e-6, TF32 is ~1e-3
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(128, 256, generator=g), torch.randn(256, 128, generator=g)
    ref = a.double() @ b.double()
    got = (a.cuda() @ b.cuda()).double().cpu()
    return bool(((got - ref).abs().max() / ref.abs().max()) > 1e-4)


def _tol(dtype):
    if dtype in (torch.bfloat16, torch.float16):
        return 3e-2
    return 2e-3 if _tf32_forced_by_env() else 1e-4


@pytest.fixture
def highest_fp32_precision():
    """Pin true-fp32 matmuls for the test (forge itself never changes global matmul precision) and restore after."""
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_matches_cpu_fp32(dtype, highest_fp32_precision):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 unsupported on this GPU")
    ref_model = build_tiny_model().eval()
    gm = copy.deepcopy(ref_model).to(device="cuda", dtype=dtype).eval()
    ids = torch.randint(0, 256, (2, 33))
    ref = ref_model(ids).logits
    out = gm(ids.cuda()).logits.float().cpu()
    assert (out - ref).abs().max() / ref.abs().max() < _tol(dtype)


def test_cache_decode_on_gpu():
    m = _cuda_model(torch.float16, sliding_window=7)
    ids = torch.randint(0, 256, (2, 24), device="cuda")
    full = m(ids).logits
    cache = m.new_kv_cache(2, 32)
    m(ids[:, :16], kv_cache=cache)
    dec = torch.cat([m(ids[:, i : i + 1], kv_cache=cache).logits for i in range(16, 24)], 1)
    assert (dec - full[:, 16:]).abs().max() / full.abs().max() < 2e-2


@pytest.mark.parametrize("kv_dtype", ["fp8_e4m3", "int8", "int4"])
def test_quantised_dense_cache_on_gpu(kv_dtype):
    m = _cuda_model(torch.float16)
    ids = torch.randint(0, 256, (1, 20), device="cuda")
    full = m(ids).logits
    cache = DenseKVCache(1, 32, m.config, dtype=torch.float16, kv_dtype=kv_dtype, device="cuda", residual_window=4)
    m(ids[:, :12], kv_cache=cache)
    dec = torch.cat([m(ids[:, i : i + 1], kv_cache=cache).logits for i in range(12, 20)], 1)
    assert torch.isfinite(dec).all()
    assert (dec - full[:, 12:]).abs().max() / full.abs().max() < {"fp8_e4m3": 0.3, "int8": 0.05, "int4": 0.7}[kv_dtype]


def _paged(m, ids, use_kernel):
    cfg = m.config
    pool = SimplePagedPool(cfg, num_blocks=16, block_size=16, dtype=torch.float16, device="cuda")
    tables = torch.arange(1, 9, device="cuda", dtype=torch.int32).reshape(2, 4)
    view = PagedKVView(tables, torch.zeros(2, dtype=torch.long, device="cuda"), pool)
    m(ids[:, :20], kv_cache=view)
    outs = [m(ids[:, i : i + 1], kv_cache=view, attn_meta=AttnMeta(use_paged_kernel=use_kernel)).logits for i in range(20, 26)]
    return torch.cat(outs, 1)


def test_paged_decode_kernel_matches_reference_path():
    pytest.importorskip("triton")
    try:
        from forge.kernels import ops_attn  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"forge.kernels.ops_attn unavailable: {e}")
    m = _cuda_model(torch.float16)
    ids = torch.randint(0, 256, (2, 26), device="cuda")
    ref = _paged(m, ids, use_kernel=False)
    got = _paged(m, ids, use_kernel=True)
    assert all(layer.self_attn._paged_ok for layer in m.layers), "kernel path silently fell back; run with FORGE_KERNEL_STRICT=1"
    assert (got - ref).abs().max() / ref.abs().max() < 2e-2


def test_fp8_linear_native_vs_reference():
    cap = torch.cuda.get_device_capability()
    if cap[0] * 10 + cap[1] < 89:
        pytest.skip("fp8 needs sm89+")
    m = _cuda_model(torch.bfloat16)
    quantize_linears(m, "fp8")
    ids = torch.randint(0, 256, (2, 16), device="cuda")
    native = m(ids).logits.float()
    for mod in m.modules():
        if hasattr(mod, "_kernel_ok"):
            mod._kernel_ok = False  # force the reference path
    ref = m(ids).logits.float()
    assert (native - ref).abs().max() / ref.abs().max() < 5e-2


def test_moe_grouped_gemm_vs_loop():
    pytest.importorskip("triton")
    try:
        from forge.kernels import ops_moe  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(f"forge.kernels.ops_moe unavailable: {e}")
    m = _cuda_model(torch.bfloat16, moe=True)
    ids = torch.randint(0, 256, (2, 32), device="cuda")
    with torch.no_grad():
        a = m(ids).logits.float()
        for layer in m.layers:
            layer.mlp.cfg_moe.use_grouped_gemm = False
        b = m(ids).logits.float()
    assert (a - b).abs().max() / b.abs().max() < 3e-2


def test_flash_attn_matches_sdpa_if_installed():
    pytest.importorskip("flash_attn")
    m = _cuda_model(torch.float16, sliding_window=9)
    ids = torch.randint(0, 256, (2, 48), device="cuda")
    a = m(ids).logits
    m.set_attn_impl("flash_attn")
    b = m(ids).logits
    assert (a - b).abs().max() / a.abs().max() < 2e-2


def test_generate_and_training_step_on_gpu():
    m = build_tiny_model().cuda()
    ids = torch.randint(0, 256, (2, 16), device="cuda")
    m.gradient_checkpointing_enable()
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
        loss = m(ids, labels=ids).loss
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    out = m.eval().generate(ids[:, :8], max_new_tokens=4)
    assert out.shape == (2, 12)
