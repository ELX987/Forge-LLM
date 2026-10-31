"""GPU tests (skipped without CUDA): MoE routing / grouped GEMM / combine and the utility kernels (Hadamard, quantisers,
stochastic rounding, RMSNorm+quant, RoPE+KV write, TEAL GEMV, segmented LoRA).  Edge cases of Plan 6.7: M = 0 experts, ragged
expert loads, counter-based RNG reproducibility, all-zero rows / blocks, E4M3 convention.
Run on the target GPU: ``pytest tests_gpu/test_kernels_attn_moe_util.py -q``."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from forge.kernels import ops_moe, ops_util  # noqa: E402
from forge.kernels.ref.moe_ref import (  # noqa: E402
    apply_expert_map_ref,
    moe_experts_forward_ref,
    moe_grouped_gemm_ref,
    moe_permute_ref,
    moe_route_ref,
    pack_w4_experts,
)
from forge.kernels.ref.util_adapters import segmented_lora_ref, teal_gemv_ref, teal_threshold  # noqa: E402
from forge.kernels.ref.util_hadamard import hadamard_ref, random_signs  # noqa: E402
from forge.kernels.ref.util_quant import (  # noqa: E402
    dequantize_rowwise_ref,
    dequantize_sr_ref,
    quantize_rowwise_ref,
    rmsnorm_quant_ref,
    stochastic_round_ref,
)
from forge.kernels.ref.util_rope import make_cos_sin_cache, rope_and_kv_write_ref  # noqa: E402
from forge.kernels.triton.util_dispatch import device_sm  # noqa: E402

DEV = torch.device("cuda")
SM = device_sm(DEV) or 0
DT = torch.bfloat16 if SM >= 80 else torch.float16
needs_sm80 = pytest.mark.skipif(SM < 80, reason="grouped GEMM tests need sm_80+")
needs_fp8 = pytest.mark.skipif(SM < 89, reason="FP8 stores need sm_89+")


@pytest.mark.parametrize("e,k", [(8, 2), (64, 8), (128, 8), (256, 8), (60, 4)])
def test_route_kernel(e, k):
    torch.manual_seed(e)
    logits = torch.randn(513, e)
    logits[7] = 0.5
    w, ids = ops_moe.moe_route(logits.to(DEV, torch.float32), k)
    rw, ri = moe_route_ref(logits, k)
    assert (ids.cpu() == ri).float().mean() > 0.999
    ok = ids.cpu() == ri
    assert (w.cpu()[ok] - rw[ok]).abs().max() < 1e-5


@needs_sm80
@pytest.mark.parametrize("tokens", [1, 7, 128, 4096])
@pytest.mark.parametrize("skew", [0.0, 3.0])
def test_grouped_gemm_ragged_loads_and_empty_experts(tokens, skew):
    torch.manual_seed(tokens)
    e, k, h, inter = 32, 8, 512, 384
    logits = torch.randn(tokens, e)
    logits[:, :4] += skew
    logits[:, 20:24] -= 50.0  # experts that never receive a token (M = 0)
    w, ids = moe_route_ref(logits, k)
    pr = moe_permute_ref(ids, e)
    x = torch.randn(tokens, h)
    wt = torch.randn(e, inter, h) * 0.05
    out = ops_moe.grouped_gemm(x.to(DEV, DT), wt.to(DEV, DT), pr.expert_offsets.to(DEV),
                               gather_idx=pr.token_idx.to(DEV), total_rows=tokens * k)
    ref = moe_grouped_gemm_ref(x.to(DT).float(), wt.to(DT).float(), pr.expert_offsets, gather_idx=pr.token_idx)
    assert (out.float().cpu() - ref).abs().max() < 2e-2 * max(1.0, float(ref.abs().max()))


@needs_sm80
def test_full_moe_with_pruned_experts_and_w4_experts():
    torch.manual_seed(1)
    t, h, i, e, k = 64, 256, 128, 16, 4
    x = torch.randn(t, h)
    wgu = torch.randn(e, 2 * i, h) * 0.1
    wd = torch.randn(e, h, i) * 0.1
    w, ids = moe_route_ref(torch.randn(t, e), k)
    out = ops_moe.moe_experts_forward(x.to(DEV, DT), w.to(DEV), ids.to(DEV), wgu.to(DEV, DT), wd.to(DEV, DT))
    ref = moe_experts_forward_ref(x.to(DT).float(), w, ids, wgu.to(DT).float(), wd.to(DT).float())
    assert (out.float().cpu() - ref).abs().max() < 5e-2
    emap = torch.arange(e)
    emap[[3, 9]] = -1
    keep = torch.tensor([i_ for i_ in range(e) if i_ not in (3, 9)])
    emap[keep] = torch.arange(keep.numel())
    out_p = ops_moe.moe_experts_forward(x.to(DEV, DT), w.to(DEV), ids.to(DEV), wgu[keep].to(DEV, DT), wd[keep].to(DEV, DT),
                                        expert_map=emap.to(DEV))
    mids, mw = apply_expert_map_ref(ids, w, emap)
    ref_p = moe_experts_forward_ref(x.to(DT).float(), mw, mids, wgu[keep].to(DT).float(), wd[keep].to(DT).float())
    assert (out_p.float().cpu() - ref_p).abs().max() < 5e-2
    q1, s1, _ = pack_w4_experts(wgu, 64)
    q2, s2, _ = pack_w4_experts(wd, 64)
    out4 = ops_moe.moe_experts_forward(x.to(DEV, DT), w.to(DEV), ids.to(DEV), q1.to(DEV), q2.to(DEV),
                                       gate_up_scales=s1.to(DEV, DT), down_scales=s2.to(DEV, DT), group_size=64)
    ref4 = moe_experts_forward_ref(x.to(DT).float(), w, ids, ops_moe.dequant_w4_experts(q1, s1, None, 64),
                                   ops_moe.dequant_w4_experts(q2, s2, None, 64))
    assert (out4.float().cpu() - ref4).abs().max() < 8e-2


@pytest.mark.parametrize("block", [16, 32, 64, 128])
def test_hadamard_kernel(block):
    x = torch.randn(33, 4096)
    s = random_signs(4096, 1)
    out = ops_util.hadamard_transform(x.to(DEV, DT), block, signs=s.to(DEV))
    ref = hadamard_ref(x.to(DT).float(), block, s)
    assert (out.float().cpu() - ref).abs().max() < 3e-2 * float(ref.abs().max())


def test_quantize_rowwise_int8_and_zero_rows():
    x = torch.randn(64, 4096) * 5
    x[3] = 0.0
    q, s = ops_util.quantize_rowwise(x.to(DEV, DT), "int8")
    qr, sr = quantize_rowwise_ref(x.to(DT).float(), "int8")
    assert (s.cpu() - sr).abs().max() < 1e-6 * float(sr.max())
    assert (q.cpu().int() - qr.int()).abs().max() <= 1 and (q[3] == 0).all()


@needs_fp8
def test_quantize_fp8_and_rmsnorm_quant_fp8():
    x = torch.randn(64, 4096) * 5
    x[3] = 0.0
    q, s = ops_util.quantize_fp8_rowwise(x.to(DEV, DT))
    xd = dequantize_rowwise_ref(q.cpu(), s.cpu())
    assert torch.isfinite(xd).all() and (xd[3] == 0).all()
    assert ((xd - x.to(DT).float()).abs().amax(-1)[[0, 1, 2]] / x.abs().amax(-1)[[0, 1, 2]]).max() < 0.08
    w = torch.rand(4096) + 0.5
    qn, sn, res = ops_util.rmsnorm_quant(x.to(DEV, DT), w.to(DEV), 1e-6, "fp8", residual=x.flip(0).to(DEV, DT))
    qr, sr, rr = rmsnorm_quant_ref(x.to(DT), w, 1e-6, "fp8", residual=x.flip(0).to(DT))
    assert (sn.cpu() - sr).abs().max() < 1e-3 * float(sr.max())
    assert torch.equal(res.cpu(), rr)


def test_rmsnorm_quant_int8():
    x = torch.randn(33, 5120)
    w = torch.rand(5120) + 0.5
    qn, sn, _ = ops_util.rmsnorm_quant(x.to(DEV, DT), w.to(DEV), 1e-6, "int8")
    qr, sr, _ = rmsnorm_quant_ref(x.to(DT), w, 1e-6, "int8")
    assert (sn.cpu() - sr).abs().max() < 2e-3 * float(sr.max())
    assert (qn.cpu().int() - qr.int()).abs().max() <= 2


@pytest.mark.parametrize("fmt,group", [("int8", 128), ("int4", 64), ("fp4", 16)])
def test_stochastic_rounding_kernel_reproducible_and_unbiased(fmt, group):
    x = torch.randn(512, 1024) * 2
    q1, s1 = ops_util.stochastic_round(x.to(DEV), fmt, group_size=group, seed=99)
    q2, _ = ops_util.stochastic_round(x.to(DEV), fmt, group_size=group, seed=99)
    q3, _ = ops_util.stochastic_round(x.to(DEV), fmt, group_size=group, seed=100)
    assert torch.equal(q1, q2) and not torch.equal(q1, q3)
    rq, rs = stochastic_round_ref(x, fmt, group_size=group, seed=99)
    assert (q1.cpu() == rq).float().mean() > 0.999
    acc = torch.zeros_like(x, device=DEV)
    for sd in range(64):
        q, sc = ops_util.stochastic_round(x.to(DEV), fmt, group_size=group, scale=s1, seed=1000 + sd)
        acc += dequantize_sr_ref(q, sc, fmt, group)
    step = float(s1.max()) * (2 if fmt == "fp4" else 1)
    assert (acc.cpu() / 64 - x).abs().mean() < 0.15 * step


@pytest.mark.parametrize("kvd", ["bf16", "int8", "fp8_e4m3"])
@pytest.mark.parametrize("interleaved,rd", [(False, None), (True, 64)])
def test_rope_and_kv_write(kvd, interleaved, rd):
    if kvd == "fp8_e4m3" and SM < 89:
        pytest.skip("FP8 stores need sm_89+")
    t, hq, hkv, d, ps = 33, 8, 4, 128, 16
    torch.manual_seed(3)
    q, k, v = torch.randn(t, hq, d), torch.randn(t, hkv, d), torch.randn(t, hkv, d)
    pos = torch.randint(0, 4000, (t,))
    cache = make_cos_sin_cache(4096, rd or d)
    slots = torch.randperm(8 * ps)[:t].int()
    slots[5] = -1
    pdt = {"bf16": DT, "int8": torch.int8, "fp8_e4m3": torch.float8_e4m3fn}[kvd]
    mk = lambda dev: (torch.zeros(8, hkv, ps, d, dtype=pdt, device=dev), torch.zeros(8, hkv, ps, d, dtype=pdt, device=dev),  # noqa: E731
                      torch.zeros(8, hkv, ps, device=dev))
    kp, vp, vs = mk(DEV)
    kr, vr, vsr = mk("cpu")
    ks = torch.full((hkv, d), 0.06)
    kw = dict(kv_dtype=kvd, k_scale=ks if kvd != "bf16" else None, rotary_dim=rd, interleaved=interleaved)
    qo = ops_util.rope_and_kv_write(q.to(DEV, DT), k.to(DEV, DT), v.to(DEV, DT), pos.to(DEV), cache.to(DEV), slots.to(DEV),
                                    kp, vp, v_scale_pool=vs, **{**kw, "k_scale": ks.to(DEV) if kvd != "bf16" else None})
    qref = rope_and_kv_write_ref(q.to(DT).float(), k.to(DT).float(), v.to(DT).float(), pos, cache, slots, kr, vr,
                                 v_scale_pool=vsr, **{**kw, "kv_dtype": "fp32" if kvd == "bf16" else kvd})
    assert (qo.float().cpu() - qref).abs().max() < 3e-2 * float(qref.abs().max())
    if kvd == "bf16":
        assert (kp.float().cpu() - kr).abs().max() < 3e-2 * float(kr.abs().max())
    else:
        assert (vs.cpu() - vsr).abs().max() < 1e-2 * float(vsr.abs().max())


def test_teal_gemv():
    x = torch.randn(4, 4096)
    w = torch.randn(4096, 4096) * 0.02
    thr = teal_threshold(x, 0.5)
    out = ops_util.teal_gemv(x.to(DEV, DT), w.to(DEV, DT), thr)
    ref = teal_gemv_ref(x.to(DT).float(), w.to(DT).float(), thr)
    assert (out.float().cpu() - ref).abs().max() < 3e-2 * float(ref.abs().max())


@needs_sm80
def test_segmented_lora_mixed_batch():
    torch.manual_seed(4)
    t, k, n, r = 96, 1024, 512, 16
    a = torch.randn(4, r, k) * 0.05
    b = torch.randn(4, n, r) * 0.05
    sc = torch.tensor([1.0, 0.5, 2.0, 0.25])
    x, y = torch.randn(t, k), torch.randn(t, n)
    ids = torch.randint(-1, 4, (t,))
    ids[ids == 2] = 3  # adapter 2 unused in this batch (empty segment)
    out = ops_util.segmented_lora(x.to(DEV, DT), y.to(DEV, DT), ids.to(DEV), a.to(DEV, DT), b.to(DEV, DT), sc.to(DEV))
    ref = segmented_lora_ref(x.to(DT).float(), y.to(DT).float(), ids, a.to(DT).float(), b.to(DT).float(), sc)
    assert (out.float().cpu() - ref).abs().max() < 5e-2 * max(1.0, float(ref.abs().max()))
