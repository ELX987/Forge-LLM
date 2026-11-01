"""CPU tests of the W4A16 launch-config policy, M bucketing, act-order column sort and the tuned-config fallbacks
(no GPU, no Triton needed)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from forge.kernels import ops_gemm
from forge.kernels.autotune import Autotuner
from forge.kernels.layouts import chip_from_sm, pack_w4, quantize_w4_rtn, sort_columns_by_group
from forge.kernels.spec import load_spec
from forge.kernels.w4_tuning import (
    DECODE_MAX_M,
    GEMV_MAX_M,
    MIN_TILE_K,
    W4_TUNE_TAG,
    chip_family,
    choose_split_k_decode,
    default_w4_config,
    m_bucket,
    nearest_entry,
    neighbor_buckets,
    normalise_w4_config,
    refine_candidates,
    resolve_w4_config,
    w4_candidates,
    w4_feasible,
)

CHIPS = [(80, 108), (86, 82), (89, 128), (90, 132), (100, 148), (120, 188), (70, 80)]
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1024]
SHAPES = [(5120, 5120), (1024, 5120), (17408, 5120), (5120, 17408), (151936, 5120)]


def test_m_bucket_and_neighbours_stay_in_regime():
    assert [m_bucket(m) for m in (0, 1, 2, 3, 5, 9, 17, 300)] == [1, 1, 2, 4, 8, 16, 32, 512]
    assert neighbor_buckets(3) == [4, 8, 2, 16, 1]
    assert max(neighbor_buckets(1)) <= DECODE_MAX_M
    assert all(b >= 32 for b in neighbor_buckets(100))  # prefill never borrows from decode
    assert all(b <= DECODE_MAX_M for b in neighbor_buckets(7))


def test_chip_families():
    assert [chip_family(s) for s in (None, 75, 80, 86, 89, 90, 100, 120, 121)] == [
        "generic", "pre_ampere", "ampere_dc", "ampere_consumer", "ada", "hopper", "blackwell_dc",
        "blackwell_consumer", "blackwell_consumer"]


@pytest.mark.parametrize("sm,sms", CHIPS)
@pytest.mark.parametrize("m", MS)
@pytest.mark.parametrize("n,k", SHAPES)
def test_default_config_is_valid_feasible_and_idempotent(sm, sms, m, n, k):
    c = default_w4_config(m, n, k, sm=sm, num_sms=sms)
    assert normalise_w4_config(c, m) == c
    assert c["impl"] in ("dot", "gemv") and c["tile_k"] >= MIN_TILE_K[c["impl"]] and c["tile_m"] >= 16
    assert c["tile_k"] % 128 == 0 and c["split_k"] >= 1
    assert w4_feasible(c, sm=sm), c
    if sm < 80:
        assert c["stages"] <= 2
    if m > 64:
        assert c["split_k"] == 1


@pytest.mark.parametrize("sm,sms", [(120, 188), (89, 128), (90, 132), (80, 108)])
def test_decode_defaults_fill_the_machine_for_skinny_n(sm, sms):
    c = default_w4_config(1, 1024, 5120, sm=sm, num_sms=sms)  # k/v projection: only 16 N tiles
    programs = -(-1024 // c["tile_n"]) * c["split_k"]
    assert programs >= sms, (c, programs)
    assert -(-5120 // c["tile_k"]) // c["split_k"] >= 2  # every split keeps >= 2 K tiles
    big = default_w4_config(1, 151936, 5120, sm=sm, num_sms=sms)  # lm_head: plenty of tiles
    assert big["split_k"] == 1


def test_consumer_blackwell_has_fewer_stages_than_hopper():
    assert default_w4_config(1, 5120, 5120, sm=120, num_sms=188)["stages"] < default_w4_config(1, 5120, 5120, sm=90, num_sms=132)["stages"]


def test_split_k_decode_rules():
    assert choose_split_k_decode(1000, 40, 108) == 1
    assert choose_split_k_decode(8, 40, 100, 2) in (8, 16)
    assert choose_split_k_decode(8, 3, 100) == 1  # never splits below two K tiles
    assert choose_split_k_decode(8, 40, 0) == 1


def test_legacy_style_and_odd_configs_are_repaired():
    c = resolve_w4_config(3, 40, 100, {"tile_m": 16, "tile_n": 32, "tile_k": 32, "split_k": 3}, sm=89, num_sms=128)
    assert c["impl"] == "dot" and c["tile_k"] == 128 and c["tile_n"] == 32 and c["split_k"] == 3
    g = resolve_w4_config(32, 4096, 4096, {"impl": "gemv"}, sm=90)
    assert g["impl"] == "dot"  # gemv is decode only
    g1 = resolve_w4_config(3, 4096, 4096, {"impl": "gemv", "tile_k": 32}, sm=90)
    assert g1["impl"] == "gemv" and g1["tile_m"] == 4 and g1["tile_k"] == 64
    leg = resolve_w4_config(1, 64, 64, {"impl": "legacy"})
    assert leg["impl"] == "legacy" and leg["tile_k"] in (64, 128)


@pytest.mark.parametrize("sm,sms", [(120, 188), (89, 128), (90, 132), (80, 108)])
@pytest.mark.parametrize("m", [1, 4, 8, 16, 64, 256])
def test_candidates_unique_feasible_and_contain_default(sm, sms, m):
    for level in ("quick", "full"):
        cands = w4_candidates(m, 5120, 5120, sm=sm, num_sms=sms, level=level)
        keys = [tuple(sorted(c.items())) for c in cands]
        assert len(keys) == len(set(keys)) and cands
        assert all(w4_feasible(c, sm=sm) and normalise_w4_config(c, m) == c for c in cands)
        assert all(not (c["impl"] == "gemv" and m > GEMV_MAX_M) for c in cands)
        assert default_w4_config(m, 5120, 5120, sm=sm, num_sms=sms) in cands
        if m <= GEMV_MAX_M:
            assert {"gemv", "dot"} <= {c["impl"] for c in cands}
    assert len(w4_candidates(1, 5120, 5120, sm=sm, num_sms=sms, level="full")) > len(w4_candidates(1, 5120, 5120, sm=sm, num_sms=sms))


def test_refine_candidates_are_neighbours_of_best():
    best = default_w4_config(1, 5120, 5120, sm=120, num_sms=188)
    ref = refine_candidates(best, 1, 5120, 5120, sm=120)
    assert ref and best not in ref
    for c in ref:
        diff = [k for k in best if best[k] != c[k]]
        assert len(diff) == 1 and w4_feasible(c, sm=120)


def test_feasibility_rejects_oversized_tiles():
    assert not w4_feasible(dict(impl="dot", tile_m=128, tile_n=256, tile_k=256, warps=4, stages=5, split_k=1), sm=120)
    assert not w4_feasible(dict(impl="dot", tile_m=16, tile_n=64, tile_k=128, warps=4, stages=3, split_k=1), sm=75)
    assert not w4_feasible(dict(impl="gemv", tile_m=8, tile_n=256, tile_k=256, warps=2, stages=2, split_k=1), sm=120)


def test_nearest_entry_picks_closest_shape_in_same_regime():
    def entry(n, k, m=1, tag="t", cfg=None):
        return {"fields": {"chip": "c", "op": "w", "dtype_cfg": tag, "bucket": {"M": m, "N": n, "K": k}}, "config": cfg or {"tile_n": n}}

    entries = [entry(5120, 5120), entry(1024, 5120), entry(17408, 5120), entry(5120, 5120, m=16), entry(8192, 8192, tag="other")]
    want = {"chip": "c", "op": "w", "dtype_cfg": "t", "bucket": {"M": 1, "N": 4096, "K": 4096}}
    assert nearest_entry(entries, want, 4096, 4096)["config"]["tile_n"] == 5120
    want2 = {**want, "bucket": {"M": 1, "N": 1280, "K": 5120}}
    assert nearest_entry(entries, want2, 1280, 5120)["config"]["tile_n"] == 1024
    assert nearest_entry([entry(5120, 5120, m=16)], want, 4096, 4096) is None
    assert nearest_entry([], want, 1, 1) is None


def test_tuned_config_borrows_neighbour_bucket_and_nearest_shape(forge_home):
    chip = chip_from_sm(120, "Fake RTX PRO 6000")
    tuner = Autotuner(chip=chip)
    spec = load_spec("w4a16_gemm")
    cfg = {"impl": "gemv", "tile_m": 4, "tile_n": 32, "tile_k": 128, "warps": 4, "stages": 3, "split_k": 4}
    dcfg = ops_gemm.dtype_cfg("w4a16_gemm", torch.bfloat16, 128, W4_TUNE_TAG)
    key = tuner.key_for(spec, {"M": 4, "N": 5120, "K": 5120}, dtype_cfg=dcfg, layout_id="L")
    fields = tuner.key_fields(spec, {"M": 4, "N": 5120, "K": 5120}, dtype_cfg=dcfg, layout_id="L")
    tuner.cache.put(key, fields, cfg, {"median": 0.01, "iqr": 0.0, "n": 5})
    ops_gemm.clear_config_memo()
    ops_gemm._TUNERS.clear()

    def look(m, n, k, dc=dcfg):
        return ops_gemm.tuned_config("w4a16_gemm", chip, m, n, k, dtype_cfg_str=dc, layout_id="L")

    assert look(4, 5120, 5120) == cfg  # exact
    assert look(3, 5120, 5120) == cfg  # M=3 -> bucket 4 exact
    assert look(2, 5120, 5120) == cfg  # bucket 2 untuned -> borrows bucket 4
    near = look(1, 1024, 5120)  # untuned N -> nearest shape, split-K recomputed by the launcher
    assert near is not None and "split_k" not in near and near["tile_n"] == 32
    assert look(64, 5120, 5120) is None  # prefill never borrows decode configs
    assert look(4, 5120, 5120, "old|k1") is None  # configs tuned for the legacy kernel (other tag) are ignored


def _parts(n, k, gs, sym=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    return quantize_w4_rtn(torch.randn(n, k, generator=g), gs, sym=sym)


@pytest.mark.parametrize("n,k,gs", [(24, 256, 64), (8, 100, 32), (16, 128, 128)])
def test_sort_columns_by_group_is_an_exact_permutation(n, k, gs):
    q, s, z = _parts(n, k, gs)
    base = torch.arange(k) // gs
    g_idx = base[torch.randperm(k, generator=torch.Generator().manual_seed(3))].to(torch.int32)
    p = pack_w4(q, s, z, gs, None, g_idx)
    res = sort_columns_by_group(p)
    assert res is not None
    ps, perm = res
    assert ps.g_idx is None and perm.dtype == torch.int32 and sorted(perm.tolist()) == list(range(k))
    x = torch.randn(5, k)
    y_ref = x @ p.dequantize(torch.float32).t()
    y_sorted = x[:, perm.long()] @ ps.dequantize(torch.float32).t()
    assert torch.allclose(y_ref, y_sorted, rtol=1e-4, atol=1e-4)
    q2, s2, _ = ps.unpack()
    assert torch.equal(q2, q[:, perm.long()]) and torch.equal(s2, s)


def test_sort_columns_rejects_missing_or_irregular_g_idx():
    q, s, z = _parts(8, 128, 32)
    assert sort_columns_by_group(pack_w4(q, s, z, 32)) is None
    irregular = torch.zeros(128, dtype=torch.int32)  # all columns in group 0
    assert sort_columns_by_group(pack_w4(q, s, z, 32, None, irregular)) is None


def test_bench_script_plans_without_a_gpu(capsys):
    spec = importlib.util.spec_from_file_location("bench_w4a16", Path(__file__).resolve().parent.parent / "scripts" / "bench_w4a16.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert [s[1:] for s in mod.parse_shapes("qwen3-14b")] == [(5120, 5120), (1024, 5120), (17408, 5120), (5120, 17408), (151936, 5120)]
    assert mod.parse_shapes("64x128,256:512")[1][1:] == (256, 512)
    assert mod.m_list("16,1,8,1") == [1, 8, 16]
    assert mod.ring_size(50 << 20, 96 << 20, 3.0, 80 << 30) == 6
    assert mod.ring_size(2 << 30, 96 << 20, 3.0, 80 << 30) == 1  # bigger than L2 already
    assert mod.ring_size(1 << 20, 96 << 20, 3.0, 1 << 30) <= 32
    rows = mod.plan(mod.parse_shapes("1024:5120"), [1, 16], 128, "quick", 120, 188)
    assert [r["bucket"] for r in rows] == [1, 16] and all(r["n_candidates"] > 0 for r in rows)
    assert mod.main(["--dry-run", "--m", "1", "--shapes", "64:128"]) == 0
    assert "candidates=" in capsys.readouterr().out
