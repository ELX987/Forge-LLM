"""CPU tests: kernel specs, layout deriver (Plan 6.3), W4/W8 packers (property tests), FP4 scale swizzle."""
from __future__ import annotations

import pytest
import torch

from forge.core.errors import ConfigError, NotFound
from forge.kernels.layouts import (
    FRAGMENT_W4,
    MARLIN_INTERLEAVE,
    PORTABLE_W4,
    PORTABLE_W8,
    Layout,
    PackedFP4,
    chip_from_sm,
    derive_layout,
    dequantize_parts,
    fragment_tile_perm,
    fp4_layout,
    get_layout,
    known_layouts,
    pack_e2m1,
    pack_w4,
    pack_w8,
    padded_scale_shape,
    quantize_w4_rtn,
    quantize_w8_rtn,
    repack,
    swizzle_scales,
    unpack_e2m1,
    unpack_w4,
    unpack_w8,
    unswizzle_scales,
)
from forge.kernels.spec import list_specs, load_spec, load_spec_from_dict

GEMM_SPECS = ["fp4_block_gemm", "fp8_gemm", "int8_w8a8_gemm", "w4a16_gemm", "w8a16_gemm"]


# ----------------------------------------------------------------------------------------------- specs
def test_all_gemm_specs_load_and_are_consistent():
    names = list_specs()
    for n in GEMM_SPECS:
        assert n in names
        s = load_spec(n)
        assert s.kernel == n and s.family == "gemm"
        assert s.oracle.startswith("ref.")
        assert s.tolerance.rel > 0 and s.search.tile_m and s.search.tile_k
        assert min(s.search.tile_m) >= 16  # tl.dot minimum
    w4 = load_spec("w4a16_gemm")
    assert w4.quant.weight_bits == 4 and 128 in w4.quant.group and w4.kpi.speedup_vs_bf16_min == 3.2
    assert w4.split_k_values(default_auto=0) == [1, 2, 4, 0]
    assert load_spec("fp4_block_gemm").experimental is True


def test_spec_errors():
    with pytest.raises(NotFound):
        load_spec("does_not_exist")
    with pytest.raises(ConfigError):
        load_spec_from_dict({"kernel": "bad name!", "op": "x", "oracle": "ref.x"})
    with pytest.raises(ConfigError):
        load_spec_from_dict({"kernel": "ok", "oracle": "ref.x"})  # missing op


def test_spec_oracles_exist():
    import forge.kernels.ref as ref

    for n in GEMM_SPECS:
        assert hasattr(ref, load_spec(n).oracle.split(".", 1)[1])


# ----------------------------------------------------------------------------------------------- layouts
def test_layout_ids_deterministic_and_registered():
    assert PORTABLE_W4.id != FRAGMENT_W4.id != PORTABLE_W8.id
    assert get_layout(PORTABLE_W4.id) is not None
    assert get_layout(FRAGMENT_W4.id).kind == "mma_fragment"
    assert PORTABLE_W4.id.startswith("w4.portable_row.v1-")
    assert fp4_layout("nvfp4", "swizzled").id != fp4_layout("nvfp4", "logical").id != fp4_layout("mxfp4", "logical").id
    assert len({lay.id for lay in known_layouts()}) == len(known_layouts())
    with pytest.raises(NotFound):
        get_layout("nope")
    d = FRAGMENT_W4.to_dict()
    assert Layout.from_dict(d) == FRAGMENT_W4 and Layout.from_dict(d).interleave == MARLIN_INTERLEAVE


@pytest.mark.parametrize("sm", [None, 70, 75, 80, 86, 87, 89, 90, 100, 103, 120, 121])
def test_derive_layout_for_every_chip_family(sm):
    spec = load_spec("w4a16_gemm")
    chip = chip_from_sm(sm) if sm else None
    lay = derive_layout(spec, chip if chip is not None else chip_from_sm(None), executor="triton", shape=(1, 4096, 4096))
    assert lay.kind == "portable_row" and lay.id == PORTABLE_W4.id and lay.executor == "triton"
    s = lay.schedule
    assert s["tile_m"] >= 16 and s["tile_n"] in spec.search.tile_n and s["tile_k"] in spec.search.tile_k
    assert s["stages"] in spec.search.stages and isinstance(s["split_k"], int)
    nat = derive_layout(spec, chip_from_sm(sm) if sm else chip_from_sm(None), executor="native", shape=(1, 4096, 4096))
    if sm is not None and sm >= 75:
        assert nat.kind == "mma_fragment" and nat.id == FRAGMENT_W4.id and nat.swizzle["rule"] == "vec_col ^= row % 8"
    else:  # Volta (different fragment order) and CPU fall back to the portable layout
        assert nat.kind == "portable_row"


def test_derived_schedule_respects_capacity_and_wave_quantisation():
    from forge.kernels.layouts import tile_smem_bytes

    spec = load_spec("w4a16_gemm")
    for sm in (75, 80, 86, 89, 90, 120):
        chip = chip_from_sm(sm)
        lay = derive_layout(spec, chip, shape=(1, 4096, 4096))
        t = lay.schedule
        used = tile_smem_bytes((t["tile_m"], t["tile_n"], t["tile_k"]), t["stages"], 4, 128)
        assert used <= chip.smem_per_sm_kib * 1024
        if sm < 80:
            assert t["stages"] == 2  # no cp.async ring before Ampere
    # skinny decode on many SMs -> split-K > 1; large M -> 1
    assert derive_layout(spec, chip_from_sm(90), shape=(1, 1024, 8192)).schedule["split_k"] > 1
    assert derive_layout(spec, chip_from_sm(90), shape=(4096, 4096, 4096)).schedule["split_k"] == 1


def test_derive_layout_fp4_and_errors():
    s = load_spec("fp4_block_gemm")
    blackwell = derive_layout(s, chip_from_sm(100), fmt="nvfp4")
    older = derive_layout(s, chip_from_sm(80), fmt="mxfp4")
    assert blackwell.scales["layout"] == "swizzled" and blackwell.scales["format"] == "e4m3"
    assert older.scales["layout"] == "logical" and older.scales["format"] == "e8m0"
    with pytest.raises(ConfigError):
        derive_layout(load_spec("w4a16_gemm"), chip_from_sm(90), executor="bogus")


# ----------------------------------------------------------------------------------------------- W4 packer
def _rand_w4(n, k, gs, sym, scale_dtype=torch.float16, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n, k, generator=g)
    return w, *quantize_w4_rtn(w, gs, sym=sym, scale_dtype=scale_dtype)


@pytest.mark.parametrize("layout", [PORTABLE_W4, FRAGMENT_W4], ids=["portable", "fragment"])
@pytest.mark.parametrize("n", [1, 7, 63, 64, 65, 130])
@pytest.mark.parametrize("k", [8, 16, 33, 100, 128, 257])
@pytest.mark.parametrize("gs", [-1, 16, 64])
@pytest.mark.parametrize("sym", [True, False])
def test_pack_unpack_identity_property(layout, n, k, gs, sym):
    """C-041: pack -> unpack is the identity for random shapes, incl. K not a multiple of the group and odd N."""
    _, q, s, z = _rand_w4(n, k, gs, sym, seed=n * 1000 + k)
    p = pack_w4(q, s, z, gs, layout)
    q2, s2, z2 = unpack_w4(p)
    assert torch.equal(q, q2) and torch.equal(s, s2)
    assert (z is None and z2 is None) or torch.equal(z, z2)
    assert p.qweight.dtype == torch.int32 and p.layout_id == layout.id
    if layout.kind == "mma_fragment":
        assert p.qweight.shape == ((k + 15) // 16, ((n + 63) // 64) * 128)  # Marlin shape (K/16, N*16/8)


@pytest.mark.parametrize("scale_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_scale_dtype_preserved_and_g_idx_roundtrip(scale_dtype):
    n, k, gs = 40, 96, 32
    _, q, s, z = _rand_w4(n, k, gs, False, scale_dtype)
    g_idx = torch.randint(0, s.shape[1], (k,), dtype=torch.int32)
    for lay in (PORTABLE_W4, FRAGMENT_W4):
        p = pack_w4(q, s, z, gs, lay, g_idx)
        q2, s2, z2 = unpack_w4(p)
        assert s2.dtype == scale_dtype and torch.equal(s, s2) and torch.equal(q, q2) and torch.equal(z, z2)
        assert torch.equal(p.g_idx, g_idx)
        ref = dequantize_parts(q, s, z, gs, g_idx)
        assert torch.equal(p.dequantize(torch.float32), ref)


@pytest.mark.parametrize("n,k,gs", [(33, 100, 32), (64, 128, 64), (7, 50, -1)])
def test_repack_is_a_pure_permutation(n, k, gs):
    _, q, s, z = _rand_w4(n, k, gs, False)
    a = pack_w4(q, s, z, gs, PORTABLE_W4)
    b = repack(a, FRAGMENT_W4)
    assert b.layout_id == FRAGMENT_W4.id and not torch.equal(a.qweight.flatten()[:1], torch.tensor([123456]))
    # logical content untouched => no re-quantisation
    for x, y in zip(unpack_w4(a), unpack_w4(b)):
        assert (x is None and y is None) or torch.equal(x, y)
    back = repack(b, PORTABLE_W4)
    assert torch.equal(back.qweight, a.qweight) and torch.equal(back.scales, a.scales) and torch.equal(back.zeros, a.zeros)
    assert repack(a, PORTABLE_W4) is a
    # same bytes are re-used: the multiset of nibbles is identical
    na = torch.bincount(unpack_w4(a)[0].flatten().long(), minlength=16)
    nb = torch.bincount(unpack_w4(b)[0].flatten().long(), minlength=16)
    assert torch.equal(na, nb)


def test_fragment_permutation_is_bijective_and_mma_ordered():
    idx, inv = fragment_tile_perm(MARLIN_INTERLEAVE)
    assert sorted(idx.tolist()) == list(range(1024)) and torch.equal(inv[idx], torch.arange(1024))
    # lane l owns k = 2*(l%4)+{0,1}(+8) and n = 8*nb + l//4 : check the natural (pre-interleave) order of lane 5
    ident, _ = fragment_tile_perm((0, 1, 2, 3, 4, 5, 6, 7))
    lane = 5
    mine = ident[lane * 32:(lane + 1) * 32]
    ks, ns = mine // 64, mine % 64
    assert set((ns % 8).tolist()) == {lane // 4}
    assert sorted(set(ks.tolist())) == sorted({2 * (lane % 4) + a + 8 * h for a in (0, 1) for h in (0, 1)})
    assert ns.tolist()[:4] == [lane // 4] * 4 and ks.tolist()[:4] == [2 * (lane % 4), 2 * (lane % 4) + 1, 2 * (lane % 4) + 8, 2 * (lane % 4) + 9]


def test_marlin_interleave_gives_ready_made_pairs():
    """(word & 0x000f000f) must hold natural elements (e0, e1) = consecutive k of the same column."""
    n, k = 64, 16
    q = (torch.arange(k).view(1, k).expand(n, k) % 16).to(torch.uint8)  # value == k
    s = torch.ones(n, 1, dtype=torch.float16)
    p = pack_w4(q, s, None, -1, FRAGMENT_W4)
    w0 = int(p.qweight[0, 0]) & 0xFFFFFFFF  # lane 0, word 0: natural e0..e7 = k 0,1,8,9 @ n0 then k 0,1,8,9 @ n8
    lo, hi = w0 & 0xF, (w0 >> 16) & 0xF
    assert (lo, hi) == (0, 1)
    assert ((w0 >> 4) & 0xF, (w0 >> 20) & 0xF) == (8, 9)  # e2, e3
    assert ((w0 >> 8) & 0xF, (w0 >> 24) & 0xF) == (0, 1)  # e4, e5


def test_packer_validation_errors():
    _, q, s, z = _rand_w4(8, 32, 16, True)
    with pytest.raises(ConfigError):
        pack_w4(q + 16, s, z, 16)  # code > 15
    with pytest.raises(ConfigError):
        pack_w4(q, s[:, :1], z, 16)  # wrong group count
    with pytest.raises(ConfigError):
        pack_w4(q, s, z, 12)  # group not multiple of 8
    with pytest.raises(ConfigError):
        pack_w4(q.float(), s, z, 16)
    with pytest.raises(ConfigError):
        pack_w4(q, s, z, 8, FRAGMENT_W4)  # fragment layout needs groups multiple of 16
    with pytest.raises(ConfigError):
        pack_w4(q, s, z, 16, PORTABLE_W8)


def test_packed_w4_state_roundtrip_and_size():
    _, q, s, z = _rand_w4(64, 256, 128, False)
    p = pack_w4(q, s, z, 128, FRAGMENT_W4)
    from forge.kernels.layouts import PackedW4

    p2 = PackedW4.from_tensors(p.tensors(), p.state())
    assert p2.layout_id == p.layout_id and torch.equal(p2.dequantize(), p.dequantize())
    assert PackedW4.tensor_roles()["qweight"] == "payload"
    # 4 bits + (16 bit scale + 4 bit zero)/128 per weight
    assert abs(p.bits_per_weight - (4 + 20 / 128)) < 0.2
    assert p.to(torch.device("cpu"), torch.float32).scales.dtype == torch.float32


# ----------------------------------------------------------------------------------------------- W8 packer
@pytest.mark.parametrize("n,k,gs", [(5, 33, -1), (64, 100, 32), (3, 8, 8)])
def test_w8_pack_unpack(n, k, gs):
    w = torch.randn(n, k)
    q, s = quantize_w8_rtn(w, gs)
    p = pack_w8(q, s, gs)
    q2, s2 = unpack_w8(p)
    assert torch.equal(q, q2) and torch.equal(s, s2) and p.layout_id == PORTABLE_W8.id
    err = (p.dequantize() - w).abs().max()
    assert err <= w.abs().max() / 127 + 1e-3
    with pytest.raises(ConfigError):
        pack_w8(q.to(torch.int16), s, gs)


def test_rtn_quantisers_bounds():
    w = torch.randn(16, 100)
    for sym in (True, False):
        q, s, z = quantize_w4_rtn(w, 32, sym=sym, scale_dtype=torch.float32)
        deq = dequantize_parts(q, s, z, 32)
        assert q.max() <= 15 and (z is None) == sym
        assert (deq - w).abs().max() <= s.max() * 0.51 + 1e-6


# ----------------------------------------------------------------------------------------------- FP4 scales
def to_blocked_reference(m: torch.Tensor) -> torch.Tensor:
    """Independent implementation of the cuBLAS block-scale swizzle (view/permute/reshape form)."""
    rows, cols = m.shape
    nrb, ncb = -(-rows // 128), -(-cols // 4)
    padded = torch.zeros(nrb * 128, ncb * 4, dtype=m.dtype)
    padded[:rows, :cols] = m
    blocks = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


@pytest.mark.parametrize("rows,cols", [(1, 1), (5, 3), (128, 4), (130, 7), (257, 13), (64, 64)])
def test_scale_swizzle_matches_reference_and_roundtrips(rows, cols):
    m = torch.randint(0, 256, (rows, cols), dtype=torch.uint8)
    sw = swizzle_scales(m)
    rp, cp = padded_scale_shape(rows, cols)
    assert rp % 128 == 0 and cp % 4 == 0 and sw.numel() == rp * cp
    assert torch.equal(sw, to_blocked_reference(m))
    assert torch.equal(unswizzle_scales(sw, rows, cols), m)


def test_scale_swizzle_known_positions():
    m = torch.zeros(130, 7, dtype=torch.uint8)
    m[5, 2] = 1
    m[40, 1] = 2
    m[130 - 1, 6] = 3
    sw = swizzle_scales(m)
    assert sw[82].item() == 1  # (5 % 32) * 16 + (5 // 32) * 4 + 2
    assert sw[133].item() == 2  # (40 % 32) * 16 + (40 // 32) * 4 + 1
    ncb = padded_scale_shape(130, 7)[1] // 4
    r, c = 129, 6
    assert sw[((r // 128) * ncb + c // 4) * 512 + (r % 128 % 32) * 16 + (r % 128 // 32) * 4 + c % 4].item() == 3


def test_e2m1_pack_roundtrip_and_odd_error():
    codes = torch.randint(0, 16, (3, 10), dtype=torch.uint8)
    packed = pack_e2m1(codes)
    assert packed.shape == (3, 5) and torch.equal(unpack_e2m1(packed), codes)
    assert packed[0, 0].item() == (codes[0, 0].item() | (codes[0, 1].item() << 4))  # low nibble = even index
    with pytest.raises(ConfigError):
        pack_e2m1(torch.zeros(2, 3, dtype=torch.uint8))


def test_packedfp4_scale_layout_conversion_is_bit_exact():
    from forge.kernels.ref import quantize_fp4_ref

    x = torch.randn(70, 100)
    for fmt in ("nvfp4", "mxfp4"):
        a = quantize_fp4_ref(x, fmt, scale_layout="logical")
        b = a.with_scale_layout("swizzled")
        assert b.scale_layout == "swizzled" and b.layout_id != a.layout_id
        assert torch.equal(b.logical_scales(), a.scales) and torch.equal(b.with_scale_layout("logical").scales, a.scales)
        assert isinstance(a, PackedFP4) and a.fmt == fmt
        p2 = PackedFP4.from_tensors(b.tensors(), b.state())
        assert torch.equal(p2.logical_scales(), a.scales)
