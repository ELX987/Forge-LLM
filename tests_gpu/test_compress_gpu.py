"""GPU-only checks for forge.formats and forge.compress (skipped automatically without CUDA by the root conftest).

NOTHING in this file has been executed by the authors: the development machine has no NVIDIA GPU.  Run on the rented host with
``pytest tests_gpu/test_compress_gpu.py -q`` (or ``scripts/run_gpu_tests.sh``).  A failure here is a genuine finding, not a flaky test:
the CPU suite proves the numerics; these tests prove that CUDA reproduces them and that the kernel dispatch paths agree with the references.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.gpu



def _dev() -> str:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return "cuda"


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).pow(2).sum() / b.float().pow(2).sum().clamp(min=1e-30)).item()


# ------------------------------------------------------------------------------------------------ formats: CUDA == CPU
def test_formats_cuda_matches_cpu_bit_exact():
    from forge.formats import dequantize_mxfp4, dequantize_nvfp4, fake_quant_int, quantize_mxfp4, quantize_nvfp4, to_fp8

    dev = _dev()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(257, 160, generator=g)
    xc = x.to(dev)
    for kw in ({}, {"rounding": "stochastic", "seed": 7, "offset": 5}, {"four_over_six": True}):
        a, b = quantize_nvfp4(x, **kw), quantize_nvfp4(xc, **kw)
        assert torch.equal(a.payload_u8, b.payload_u8.cpu()), kw
        assert torch.equal(a.block_scales_e4m3.view(torch.uint8), b.block_scales_e4m3.view(torch.uint8).cpu()), kw
        assert torch.equal(dequantize_nvfp4(a), dequantize_nvfp4(b).cpu()), kw
    for mode in ("ceil", "floor", "near"):
        assert torch.equal(dequantize_mxfp4(quantize_mxfp4(x, scale_mode=mode)), dequantize_mxfp4(quantize_mxfp4(xc, scale_mode=mode)).cpu())
    for bits, sym in ((4, True), (4, False), (8, True)):
        assert torch.equal(fake_quant_int(x, bits, 32, sym), fake_quant_int(xc, bits, 32, sym).cpu())
        assert torch.equal(fake_quant_int(x, bits, 32, sym, rounding="stochastic", seed=3), fake_quant_int(xc, bits, 32, sym, rounding="stochastic", seed=3).cpu())
    x8c, sc = to_fp8(xc, "e4m3", dim=-1)
    x8, s = to_fp8(x, "e4m3", dim=-1)
    assert torch.equal(x8.view(torch.uint8), x8c.view(torch.uint8).cpu()) and torch.allclose(s, sc.cpu())


def test_counter_rng_cuda_matches_cpu():
    from forge.formats import counter_uniform

    dev = _dev()
    assert torch.equal(counter_uniform((4096,), 11, 123), counter_uniform((4096,), 11, 123, device=dev).cpu())


# ------------------------------------------------------------------------------------------------ packed linears: kernel dispatch vs reference
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("m_rows", [1, 16, 64])
def test_w4a16_native_matches_dequant_reference(dtype, sym, m_rows):
    from forge.compress.packed import IntLinear

    dev = _dev()
    torch.manual_seed(0)
    lin = nn.Linear(512, 384, bias=True).to(dev, dtype)
    m = IntLinear.from_dense(lin, bits=4, group_size=128, sym=sym).to(dev)
    x = torch.randn(3, m_rows, 512, device=dev, dtype=dtype)
    y = m(x)  # CUDA: ops_gemm.w4a16_linear (Triton) with fallback
    ref = torch.nn.functional.linear(x, m.dequantize(dtype), m.bias.to(dtype))
    assert y.shape == ref.shape and rel(y, ref) < 2e-3
    assert m._kernel_ok, "the native path was disabled by an exception (see the warning in the log)"


def test_nvfp4_native_matches_reference():
    from forge.compress.packed import NVFP4Linear

    dev = _dev()
    torch.manual_seed(1)
    lin = nn.Linear(256, 128, bias=False).to(dev, torch.bfloat16)
    m = NVFP4Linear.from_dense(lin).to(dev)
    x = torch.randn(8, 256, device=dev, dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(x, m.dequantize(torch.bfloat16))
    assert rel(m(x), ref) < 5e-3


def test_pack_stage_layout_roundtrip_on_gpu():
    from forge.compress.config import CompressJobConfig
    from forge.compress.run import run_compress
    from forge.core.context import NullContext

    _dev()
    recipe = "recipe: gpu-pack\nstages:\n- ptq: {method: rtn, bits: 4, group: 64}\n- pack: {layout: auto}\n"
    res = run_compress(CompressJobConfig(model="tiny", dtype="float16", device="cuda", recipe=recipe, use_store=False), NullContext())
    assert res["layout_id"] and "pack" in "".join(res["stages"])


# ------------------------------------------------------------------------------------------------ algorithms on CUDA
def test_gptq_cuda_matches_cpu_proxy_loss():
    from forge.compress.ptq import PTQConfig, quantize_weight

    dev = _dev()
    torch.manual_seed(0)
    W = torch.randn(64, 256)
    X = torch.randn(1024, 256) * torch.linspace(0.2, 3, 256)
    H = 2 / 1024 * X.t() @ X
    cfg = PTQConfig(method="gptq", bits=4, group_size=64, act_order=True)
    cpu = quantize_weight(W, H, cfg, sdtype=torch.float32)
    gpu = quantize_weight(W.to(dev), H.to(dev), cfg, sdtype=torch.float32)

    def loss(q):
        d = W - q.deq.cpu()
        return float(((d @ H) * d).sum())

    assert abs(loss(cpu) - loss(gpu)) / loss(cpu) < 0.05
    assert (cpu.q != gpu.q.cpu()).float().mean() < 0.02  # float non-associativity may flip a few ties


def test_full_pipeline_on_gpu_bf16():
    from forge.compress.config import CompressJobConfig
    from forge.compress.run import run_compress
    from forge.core.context import NullContext

    _dev()
    recipe = """
recipe: gpu-smoke
stages:
- calibrate: {set: synthetic, n_seqs: 16, seq_len: 64}
- rotate: {mode: hadamard}
- ptq: {method: gptq, bits: 4, group: 64, act_order: true}
- kv: {scheme: fp8_e4m3}
- eval: {builtin: true, n_seqs: 4, seq_len: 64}
"""
    res = run_compress(CompressJobConfig(model="tiny", dtype="bfloat16", device="cuda", recipe=recipe, use_store=False), NullContext())
    m = res["stages"]["04_eval"]["eval_report"]["metrics"]
    assert torch.isfinite(torch.tensor(m["ppl"]["value"])) and res["compression"] > 1.0


def test_rotation_equivalence_on_gpu():
    """Rotated vs original logits.  Run at fp32 'highest' precision: with TF32 on (e.g. TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1) the
    forward passes alone differ by ~1e-3 relative, which has nothing to do with the rotation.  ``rel`` is a *squared* relative error,
    so 1e-10 means <= 1e-5 relative L2 error: fp32 rounding through a few layers of Hadamard-mixed matmuls (~1e-6) with margin."""
    from forge.compress.rotation import RotationConfig, rotate_model
    from forge.models.tiny import build_tiny_model

    dev = _dev()
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_float32_matmul_precision())
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        m = build_tiny_model(0, dtype=torch.float32).to(dev).eval()
        ids = torch.randint(0, 256, (2, 32), device=dev)
        with torch.no_grad():
            ref = m(ids).logits
            rotate_model(m, RotationConfig())
            out = m(ids).logits
        assert rel(out, ref) < 1e-10
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[0], saved[1]
        torch.set_float32_matmul_precision(saved[2])
