"""CPU tests of the utility-kernel oracles (Plan K8/K9): Hadamard, quantisers, stochastic rounding, RMSNorm+quant,
RoPE+KV write, TEAL gather-GEMV, segmented LoRA."""
from __future__ import annotations

import math

import pytest
import torch

from forge.kernels.ref.attn_kv import dequantize_rows, normalize_kv_dtype
from forge.kernels.ref.util_adapters import pad_adapters, segmented_lora_ref, teal_active_fraction, teal_gemv_ref, teal_threshold
from forge.kernels.ref.util_hadamard import fwht_butterfly, hadamard_matrix, hadamard_ref, random_signs
from forge.kernels.ref.util_quant import (
    absmax_ref,
    dequantize_rowwise_ref,
    dequantize_sr_ref,
    fp4_values,
    quantize_rowwise_ref,
    rmsnorm_quant_ref,
    stochastic_round_ref,
)
from forge.kernels.ref.util_rope import apply_rope_ref, make_cos_sin_cache, rope_and_kv_write_ref


@pytest.mark.parametrize("n", [16, 32, 128, 64])
def test_hadamard_matches_matrix_and_is_orthonormal(n):
    x = torch.randn(5, n)
    h = hadamard_matrix(n, normalize=True)
    torch.testing.assert_close(hadamard_ref(x, n), x @ h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hadamard_ref(hadamard_ref(x, n), n), x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(hadamard_ref(x, n).norm(dim=-1), x.norm(dim=-1), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(fwht_butterfly(x), x @ hadamard_matrix(n), atol=1e-4, rtol=1e-5)


def test_hadamard_block_diagonal_signs_and_dtype():
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    y = hadamard_ref(x, 32)
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    exp = torch.cat([x[:, i * 32 : (i + 1) * 32].float() @ hadamard_matrix(32, normalize=True) for i in range(8)], -1)
    torch.testing.assert_close(y.float(), exp.to(torch.bfloat16).float(), atol=2e-2, rtol=2e-2)
    s = random_signs(256, 3)
    xf = x.float()
    rht = hadamard_ref(xf, 256, signs=s)
    back = hadamard_ref(rht, 256) * s  # x = signs * H y
    torch.testing.assert_close(back, xf, atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError):
        hadamard_ref(torch.randn(2, 48), 32)


def test_hadamard_flattens_outliers():
    x = torch.randn(4, 128) * 0.1
    x[:, 5] = 50.0
    y = hadamard_ref(x, 128)
    assert (y.abs().amax(-1) / y.abs().mean(-1)).mean() < (x.abs().amax(-1) / x.abs().mean(-1)).mean() / 4


@pytest.mark.parametrize("qdt", ["fp8", "int8"])
@pytest.mark.parametrize("group", [None, 32])
def test_rowwise_quant_roundtrip_and_zero_rows(qdt, group):
    x = torch.randn(6, 128) * 3
    x[2] = 0.0
    q, s = quantize_rowwise_ref(x, qdt, group)
    xd = dequantize_rowwise_ref(q, s, group)
    assert s.shape == ((6, 1) if group is None else (6, 4))
    assert (xd[2] == 0).all() and torch.isfinite(s).all() and (s > 0).all()
    tol = 0.08 if qdt == "fp8" else 0.012
    assert ((xd - x).abs().amax(-1) / x.abs().amax(-1).clamp_min(1e-9))[[0, 1, 3, 4, 5]].max() < tol
    assert torch.equal(absmax_ref(x, group).amax(-1) if group else absmax_ref(x).squeeze(-1), x.abs().amax(-1))


def test_quant_ignores_nonfinite_in_absmax():
    x = torch.tensor([[1.0, float("inf"), -2.0, float("nan")]])
    assert absmax_ref(x).item() == 2.0


@pytest.mark.parametrize("fmt,group", [("int8", None), ("int4", 32), ("fp4", 16)])
def test_stochastic_rounding_unbiased_and_reproducible(fmt, group):
    torch.manual_seed(0)
    x = torch.randn(64, 64) * 2
    q1, s1 = stochastic_round_ref(x, fmt, group_size=group, seed=11)
    q2, _ = stochastic_round_ref(x, fmt, group_size=group, seed=11)
    q3, _ = stochastic_round_ref(x, fmt, group_size=group, seed=12)
    assert torch.equal(q1, q2) and not torch.equal(q1, q3)
    acc = torch.zeros_like(x)
    n = 200
    for sd in range(n):
        q, s = stochastic_round_ref(x, fmt, group_size=group, seed=1000 + sd)
        acc += dequantize_sr_ref(q, s, fmt, group)
    mean = acc / n
    step = (s.max() * (2.0 if fmt == "fp4" else 1.0)).item()  # grid spacing upper bound (E2M1 max gap = 2)
    assert (mean - x).abs().mean() < 0.12 * step, ((mean - x).abs().mean(), step)
    # deterministic RN would be biased towards grid points; SR output must hit both neighbours
    assert len(torch.unique(q1)) > 3


def test_stochastic_rounding_offset_composition_and_exact_grid_points():
    x = torch.randn(8, 32)
    full, s = stochastic_round_ref(x, "int8", seed=3)
    # chunking by rows with the right offset reproduces the whole result (launch-geometry independence)
    lo, _ = stochastic_round_ref(x[:4], "int8", scale=s[:4], seed=3, offset=0)
    hi, _ = stochastic_round_ref(x[4:], "int8", scale=s[4:], seed=3, offset=4 * 32)
    assert torch.equal(torch.cat([lo, hi]), full)
    grid = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -6.0, -0.5, -3.0, 0.0, 1.0, 2.0, 4.0, -1.5]])
    q, _ = stochastic_round_ref(grid, "fp4", scale=torch.ones(1, 1), seed=1)
    assert torch.equal(fp4_values(q), grid)  # values exactly on the grid never move


def test_rmsnorm_quant_ref_matches_unfused():
    torch.manual_seed(1)
    x = torch.randn(5, 64, dtype=torch.bfloat16)
    r = torch.randn(5, 64, dtype=torch.bfloat16)
    w = torch.rand(64) + 0.5
    q, s, res = rmsnorm_quant_ref(x, w, 1e-6, "fp8", residual=r)
    h = (x.float() + r.float()).to(torch.bfloat16).float()
    y = h * torch.rsqrt((h * h).mean(-1, keepdim=True) + 1e-6) * w
    qq, ss = quantize_rowwise_ref(y, "fp8")
    assert torch.equal(q.view(torch.uint8), qq.view(torch.uint8)) and torch.equal(s, ss.squeeze(-1))
    assert torch.equal(res, h.to(torch.bfloat16))
    q2, s2, none = rmsnorm_quant_ref(x, w, qdtype="int8")
    assert none is None and q2.dtype == torch.int8 and s2.shape == (5,)


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("rotary_dim", [None, 8])
def test_rope_is_norm_preserving_and_relative(interleaved, rotary_dim):
    d = 16
    cache = make_cos_sin_cache(64, rotary_dim or d)
    x = torch.randn(3, 2, d)
    pos = torch.tensor([0, 5, 9])
    r = apply_rope_ref(x, pos, cache, rotary_dim, interleaved)
    torch.testing.assert_close(r.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(r[0], x[0])  # position 0 = identity
    if rotary_dim:
        torch.testing.assert_close(r[..., rotary_dim:], x[..., rotary_dim:])
    # relative-position property: <rope(q,m), rope(k,n)> depends on m-n
    q = torch.randn(1, 1, d)
    k = torch.randn(1, 1, d)
    def dot(m, n):
        return (apply_rope_ref(q, torch.tensor([m]), cache, rotary_dim, interleaved) *
                apply_rope_ref(k, torch.tensor([n]), cache, rotary_dim, interleaved)).sum()
    torch.testing.assert_close(dot(10, 7), dot(20, 17), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("kvd", ["bf16", "fp8_e4m3", "int8", "int4"])
def test_rope_kv_write_ref_places_tokens_and_quantises(kvd):
    torch.manual_seed(2)
    t, hq, hkv, d, ps = 5, 4, 2, 16, 8
    q = torch.randn(t, hq, d)
    k = torch.randn(t, hkv, d)
    v = torch.randn(t, hkv, d)
    pos = torch.tensor([3, 4, 5, 6, 7])
    cache = make_cos_sin_cache(32, d)
    slots = torch.tensor([17, 18, -1, 9, 23])
    kvn = normalize_kv_dtype(kvd)
    sd = d // 2 if kvn == "int4" else d
    pdt = {"bf16": torch.bfloat16, "fp8_e4m3": torch.float8_e4m3fn, "int8": torch.int8, "int4": torch.uint8}[kvn]
    kp = torch.zeros(4, hkv, ps, sd, dtype=pdt)
    vp = torch.zeros(4, hkv, ps, sd, dtype=pdt)
    vs = torch.zeros(4, hkv, ps)
    ks = torch.full((hkv, d), 0.05)
    qr = rope_and_kv_write_ref(q, k, v, pos, cache, slots, kp, vp, kv_dtype=kvd, k_scale=ks if kvn != "bf16" else None,
                               v_scale_pool=vs)
    torch.testing.assert_close(qr, apply_rope_ref(q, pos, cache).to(q.dtype))
    k_rot = apply_rope_ref(k, pos, cache)
    for ti, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        pg, off = divmod(slot, ps)
        if kvn == "bf16":
            torch.testing.assert_close(kp[pg, :, off].float(), k_rot[ti].to(torch.bfloat16).float())
        else:
            kd = dequantize_rows(kp[pg, :, off][:, None, :], ks[:, :] , None, kvn, scale_axis="channel")[:, 0]
            vd = dequantize_rows(vp[pg, :, off][:, None, :], vs[pg, :, off][:, None], None, kvn, scale_axis="token")[:, 0]
            tol = {"fp8_e4m3": 0.15, "int8": 0.04, "int4": 0.4}[kvn]
            assert (vd - v[ti]).abs().max() / v[ti].abs().max() < tol
            if kvn != "int4":
                assert (kd - k_rot[ti]).abs().max() < tol * 2 + 0.05
    # the skipped slot (-1) wrote nothing: exactly 4 of the 5 tokens produced a value-scale entry (bf16 writes no scales)
    if kvn != "bf16":
        assert int((vs != 0).sum()) == 4 * hkv


def test_teal_gemv_ref_and_threshold():
    torch.manual_seed(0)
    x = torch.randn(3, 128)
    w_t = torch.randn(128, 32)
    thr = teal_threshold(x, 0.5)
    assert abs(teal_active_fraction(x, thr) - 0.5) < 0.02
    y = teal_gemv_ref(x, w_t, thr)
    keep = x.abs() > thr
    exp = torch.stack([(x[b][keep[b]]) @ w_t[keep[b]] for b in range(3)])
    torch.testing.assert_close(y, exp, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(teal_gemv_ref(x, w_t, 0.0), x @ w_t, atol=1e-4, rtol=1e-4)
    yb = teal_gemv_ref(x, w_t, thr, bias=torch.ones(32))
    torch.testing.assert_close(yb, y + 1, atol=1e-4, rtol=1e-4)
    # 50 % sparsity retains most of the output energy for Gaussian inputs
    full = x @ w_t
    assert ((y - full).norm() / full.norm()) < 0.7


def test_segmented_lora_ref_mixed_batch():
    torch.manual_seed(1)
    t, k, n = 9, 16, 12
    a_list = [torch.randn(4, k), torch.randn(8, k)]
    b_list = [torch.randn(n, 4), torch.randn(n, 8)]
    a_st, b_st = pad_adapters(a_list, b_list, 16)
    assert a_st.shape == (2, 16, k) and b_st.shape == (2, n, 16)
    x = torch.randn(t, k)
    y = torch.randn(t, n)
    ids = torch.tensor([0, 1, -1, 1, 0, 0, -1, 1, 1])
    sc = torch.tensor([0.5, 2.0])
    out = segmented_lora_ref(x, y, ids, a_st, b_st, sc)
    for i in range(t):
        a = int(ids[i])
        exp = y[i] if a < 0 else y[i] + sc[a] * (x[i] @ a_list[a].T) @ b_list[a].T
        torch.testing.assert_close(out[i], exp, atol=1e-4, rtol=1e-4)


def test_reference_names_resolve_through_the_package_lazily():
    import forge.kernels.ref as ref

    for name in ("paged_decode_attention_ref", "mla_decode_attention_ref", "sparse_attention_ref", "tree_attention_ref",
                 "moe_grouped_gemm_ref", "moe_route_ref", "hadamard_ref", "stochastic_round_ref", "rmsnorm_quant_ref",
                 "rope_kv_write_ref", "rejection_verify_ref", "kv_compact_ref", "prefill_attention_ref"):
        assert callable(getattr(ref, name)), name


@pytest.mark.parametrize("qdt", ["fp8", "int8"])
def test_scale_underflow_and_all_zero_blocks_flush_cleanly(qdt):
    """Plan 6.7: E4M3 scale underflow / flush blocks and all-zero blocks must not produce 0/0, inf or NaN."""
    x = torch.zeros(4, 32)
    x[1] = 1e-44          # denormal absmax: absmax / qmax underflows to 0 in float32
    x[2] = 1e-30          # tiny but representable scale
    x[3, :2] = torch.tensor([3.0, -3.0])
    q, s = quantize_rowwise_ref(x, qdt)
    d = dequantize_rowwise_ref(q, s)
    assert torch.isfinite(s).all() and (s > 0).all() and torch.isfinite(d).all()
    assert (d[0] == 0).all() and (d[1] == 0).all()           # flushed, no NaN
    assert d[2].abs().max() > 0 and (d[2] - x[2]).abs().max() < 0.1 * 1e-30
    assert (d[3, :2] - x[3, :2]).abs().max() < 0.1


def test_sr_keeps_small_channels_inside_a_fp4_tile_unlike_round_to_nearest():
    """Plan 6.7 / P5 zero-gradient failure: a 16-wide tile where one channel is 10-30x larger than its neighbours.
    Round-to-nearest flushes the small channels to 0 (zero gradient), stochastic rounding keeps them unbiased."""
    tile = torch.full((1, 16), 0.05)
    tile[0, 0] = 6.0                       # large neighbour sets the block scale (absmax / 6 = 1)
    scale = torch.ones(1, 1)
    from forge.kernels.ref.util_quant import dequantize_sr_ref

    rn_mag = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    rn = rn_mag[(tile[0, 1:].abs().unsqueeze(-1) - rn_mag).abs().argmin(-1)]
    assert (rn == 0).all()                  # RN: the small channels vanish
    acc = torch.zeros(1, 16)
    n = 4000
    for sd in range(n):
        q, sc = stochastic_round_ref(tile, "fp4", group_size=16, scale=scale, seed=sd)
        acc += dequantize_sr_ref(q, sc, "fp4", 16)
    mean_small = (acc / n)[0, 1:]
    assert (mean_small - 0.05).abs().max() < 0.012
