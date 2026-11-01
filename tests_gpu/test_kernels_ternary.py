"""GPU tests (auto-skipped without CUDA): ternary W1.58A8 Triton kernels vs the torch oracle.  Never run by the developer; the first run on hardware decides
the 'GPU-validated' cells of docs/KERNELS.md (all start at 'no').  Also cross-checks against tern158's golden vectors."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu
pytest.importorskip("triton")

from forge.formats import ternary as tn  # noqa: E402
from forge.kernels import ops_gemm  # noqa: E402
from forge.kernels.layouts.ternary import quantize_pack_ternary  # noqa: E402
from forge.kernels.ref.gemm_ternary import ternary_gemm_ref  # noqa: E402
from forge.kernels.triton.gemm_ternary import ternary_act_quant_triton, ternary_gemm_triton  # noqa: E402

DEV = "cuda"
GROUPS = [-1, 0, 32, 64, 128, 256]


def _sm():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@pytest.fixture(autouse=True)
def _need_int8():
    if _sm() < 75:
        pytest.skip("int8 tensor cores need sm_75+")


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("m", [1, 7, 16, 100, 512])
def test_gemm_equals_oracle(group, m):
    n, k = 200, 512
    g = torch.Generator(device="cpu").manual_seed(m + group)
    w = (torch.randn(n, k, generator=g) * 0.05).to(DEV)
    p = quantize_pack_ternary(w, group)
    xq, sx = tn.act_quant_int8((torch.randn(m, k, generator=g) * 2).to(DEV))
    ref = ternary_gemm_ref(xq, sx, p.qweight, p.scales, group, k, torch.float32)
    out = ternary_gemm_triton(xq, sx, p.qweight, p.scales, group, k, out_dtype=torch.float32, sm=_sm())
    torch.cuda.synchronize()
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5 * float(ref.abs().max())), float((out - ref).abs().max())


def test_activation_quantiser_bit_exact_on_gpu():
    x = (torch.randn(33, 4100, device=DEV) * 5)
    x[3] = 0
    q, s = ternary_act_quant_triton(x)
    q0, s0 = tn.act_quant_int8(x)
    assert torch.equal(q, q0) and torch.equal(s, s0)


def test_dispatcher_takes_triton_path_and_matches_oracle():
    p = quantize_pack_ternary(torch.randn(256, 512, device=DEV) * 0.05, 128)
    x = torch.randn(4, 3, 512, device=DEV, dtype=torch.bfloat16)
    y = ops_gemm.ternary_linear(x, p.qweight, p.scales, 128, 512)
    assert ops_gemm.last_path("ternary_linear") == "triton"
    xq, sx = tn.act_quant_int8(x)
    ref = ternary_gemm_ref(xq, sx, p.qweight, p.scales, 128, 512, torch.float32).reshape(4, 3, 256)
    assert torch.allclose(y.float(), ref, rtol=2 ** -7, atol=1e-3)


def test_matches_tern158_golden_vectors():
    from safetensors.torch import load_file

    gold = load_file(str(Path(__file__).resolve().parent.parent / "tests" / "data" / "ternary_golden.safetensors"))
    for rule, group in [("lloyd", 32), ("absmean", 64), ("twn", 0)]:
        key = f"{rule}_g{group}"
        qw = tn.words_to_packed(gold[key + "_words"], 8, 64).to(DEV)
        sc = (gold[key + "_scales"].reshape(1, 1) if group == 0 else gold[key + "_scales"].reshape(8, -1)).to(DEV)
        y = ternary_gemm_triton(gold["xq"].to(DEV), gold["sx"].to(DEV), qw, sc, group, 64, out_dtype=torch.float32, sm=_sm())
        assert torch.allclose(y.cpu(), gold[key + "_y"], rtol=1e-5, atol=1e-6), key
