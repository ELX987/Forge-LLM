"""Capacity planner reproduces Plan Worked examples A and B numerically; fleet formula; UI simulators."""
from __future__ import annotations

import pytest

from forge.serve.capacity import (
    GB,
    KIB,
    ChipSpec,
    ContextDist,
    ModelSpec,
    capacity_curve,
    capacity_plan,
    fleet_capacity,
    kv_bytes_per_token,
    speculation_gain_vs_batch,
    throughput_vs_batch,
    weight_bytes,
)

H100 = ChipSpec("H100-80GB", 80.0, 3350.0, {"bf16": 989.0, "fp8": 1979.0})
LLAMA70 = ModelSpec("llama-3.1-70b", params=70e9, num_layers=80, num_kv_heads=8, head_dim=128)
LLAMA8 = ModelSpec("llama-3-8b", params=8e9, num_layers=32, num_kv_heads=8, head_dim=128)


def test_kv_bytes_per_token_matches_plan():
    assert kv_bytes_per_token(80, 8, 128, "bf16") == 320 * KIB          # Llama-3.1-70B: 320 KiB/token
    assert kv_bytes_per_token(80, 8, 128, "fp8") == 160 * KIB
    assert kv_bytes_per_token(80, 8, 128, "int4", scale_overhead=0.0) == 80 * KIB
    assert kv_bytes_per_token(80, 8, 128, "int4") == 90 * KIB           # raw + 12.5 % scales = upper end of "80-90 KiB"
    assert kv_bytes_per_token(32, 8, 128, "bf16") == 128 * KIB          # Llama-3-8B: 128 KiB (Fig 5.2)
    assert kv_bytes_per_token(36, 8, 128, "bf16") == 144 * KIB          # Qwen3-8B: 144 KiB (Fig 5.2)


def row(weights_gb, kv):
    return capacity_plan(LLAMA70, H100, 16, kv, ContextDist(mean_ctx=4096), weight_gb=weights_gb, kv_scale_overhead=None)


def test_example_a_fp8_weights_row():
    # FP8/INT8 weights: 70 GB -> 10 GB free -> 30 k tokens @16-bit KV (< one 32k sequence), 61 k @FP8, 108-122 k @4-bit
    r16, r8 = row(70, "bf16"), row(70, "fp8")
    assert r16.fits and r16.free_for_kv_gb == pytest.approx(10.0)
    assert round(r16.kv_tokens / 1000) == 31 or round(r16.kv_tokens / 1000) == 30
    assert r16.kv_tokens < 32768
    assert round(r8.kv_tokens / 1000) == 61
    r4_lo = capacity_plan(LLAMA70, H100, 16, "int4", 4096, weight_gb=70, kv_scale_overhead=0.0)   # 80 KiB
    r4_hi = row(70, "int4")                                                                      # 90 KiB
    assert round(r4_lo.kv_tokens / 1000) == 122 and round(r4_hi.kv_tokens / 1000) == 109


def test_example_a_w4a16_row():
    # W4A16 (g128): 37 GB -> 43 GB free -> 131 k @16-bit, 262 k @FP8, 470-525 k @4-bit
    r16, r8 = row(37, "bf16"), row(37, "fp8")
    assert r16.free_for_kv_gb == pytest.approx(43.0)
    assert round(r16.kv_tokens / 1000) == 131
    assert round(r8.kv_tokens / 1000) == 262
    lo = capacity_plan(LLAMA70, H100, 16, "int4", 4096, weight_gb=37, kv_scale_overhead=0.0).kv_tokens
    hi = row(37, "int4").kv_tokens
    assert round(lo / 1000) == 525 and 465 <= hi / 1000 <= 470


def test_example_a_bf16_does_not_fit_one_gpu():
    r = capacity_plan(LLAMA70, H100, 16, "bf16", 4096, weight_gb=140)
    assert not r.fits and r.kv_tokens == 0 and r.max_concurrency == 0
    assert any("do not fit" in n for n in r.notes)


def test_example_a_computed_weight_sizes_close_to_plan_table():
    w16 = weight_bytes(70e9, 16)
    w8 = weight_bytes(70e9, 8, group_size=None)
    w4 = weight_bytes(70e9, 4, group_size=128)
    assert w16 / GB == pytest.approx(140.0)
    assert w8 / GB == pytest.approx(70.0)
    assert 35.0 < w4 / GB < 37.5            # plan rounds W4A16 g128 (with a few layers at 16-bit) to 37 GB
    r = capacity_plan(LLAMA70, H100, 4, "bf16", 4096, group_size=128, high_precision_params=1.0e9)
    assert 36.0 < r.weights_gb < 38.0 and 41.0 < r.free_for_kv_gb < 44.0


def test_example_a_from_real_model_config():
    from forge.models.config import ModelConfig

    mc = ModelConfig.from_hf_config(dict(vocab_size=128256, hidden_size=8192, intermediate_size=28672, num_hidden_layers=80,
                                         num_attention_heads=64, num_key_value_heads=8, rope_theta=500000.0,
                                         max_position_embeddings=131072, rms_norm_eps=1e-5, tie_word_embeddings=False,
                                         model_type="llama"))
    ms = ModelSpec.from_any(mc)
    assert ms.params / 1e9 == pytest.approx(70.55, abs=0.1)
    assert ms.kv_bytes_per_token() == 320 * KIB


def test_example_b_ideal_decode_ceilings():
    # 8B BF16 (16 GB): 4.8 ms, 210 tok/s; 8B W4A16 (~4.5 GB): 1.3 ms, 740 tok/s; 70B W4A16 (37 GB): 11 ms, 90 tok/s
    cases = [(LLAMA8, 16.0, 4.8, 210), (LLAMA8, 4.5, 1.3, 740), (LLAMA70, 37.0, 11.0, 90)]
    for model, gb, ms, toks in cases:
        r = capacity_plan(model, H100, 16, "bf16", 1024, weight_gb=gb)
        assert r.ideal_decode_ms_b1 == pytest.approx(ms, abs=0.06)
        assert r.ideal_decode_tok_s_b1 == pytest.approx(toks, rel=0.02)


def test_batch_crossover_and_roofline_bounds():
    r = capacity_plan(LLAMA8, H100, 16, "bf16", 1024, weight_gb=16.0)
    # B* = peak * W / (2 P BW): memory-bound below, compute-bound above
    assert r.batch_crossover == pytest.approx(989e12 * 16e9 / (2 * 8e9 * 3350e9), rel=1e-6)
    curve = throughput_vs_batch(LLAMA8, H100, 16, "bf16", ctx=0.0, weight_gb=16.0, batches=(1, 8, 64, 512, 4096))
    assert curve["bound"][0] == "memory" and curve["bound"][-1] == "compute"
    tps = curve["tokens_per_s"]
    assert all(b >= a for a, b in zip(tps, tps[1:]))


def test_fleet_capacity_counts_shared_prefix_once():
    f = fleet_capacity(100_000, ctx_len=8000, shared_prefix=6000, block_size=16)
    assert f["agents"] == 47 and f["agents_without_sharing"] == 12     # floor(94000/2000) vs floor(100000/8000)
    assert f["sharing_gain"] == pytest.approx(47 / 12)
    assert f["kv_tokens_used"] == 6000 + 47 * 2000
    # no sharing configured -> same as naive
    g = fleet_capacity(100_000, 8000, 0.0)
    assert g["agents"] == g["agents_without_sharing"]


def test_capacity_plan_uses_sharing_slo_and_cost():
    base = capacity_plan(LLAMA8, H100, 16, "bf16", ContextDist(mean_ctx=8000), weight_gb=16.0)
    shared = capacity_plan(LLAMA8, H100, 16, "bf16", ContextDist(mean_ctx=8000, prefix_tokens=7000), weight_gb=16.0)
    assert shared.max_concurrency > 3 * base.max_concurrency
    assert shared.max_concurrency_no_sharing == base.max_concurrency
    fp8 = capacity_plan(LLAMA8, H100, 16, "fp8", ContextDist(mean_ctx=8000), weight_gb=16.0)
    assert fp8.max_concurrency >= 2 * base.max_concurrency - 1
    slo = capacity_plan(LLAMA8, H100, 16, "bf16", ContextDist(mean_ctx=8000), weight_gb=16.0, itl_slo_ms=10.0, price_per_hour=3.0)
    assert 0 < slo.max_concurrency_under_slo <= slo.max_concurrency
    assert slo.step_ms_at_max_concurrency is not None and slo.cost_per_million_tokens > 0
    d = slo.to_dict()
    assert d["breakdown_gb"]["weights"] == pytest.approx(16.0) and d["provenance"] == "derived"
    assert sum(v for k, v in d["breakdown_gb"].items() if k != "kv_cache_used_at_max_concurrency") <= H100.hbm_gb + 1e-6


def test_chip_inputs_accepted():
    assert ChipSpec.from_any("NVIDIA H100 80GB HBM3").bw_gbs == 3350
    assert ChipSpec.from_any("NVIDIA GeForce RTX 4090").hbm_gb == 24
    from forge.hw.capabilities import capabilities_for_chip

    cap = capabilities_for_chip("h100-sxm5-80g")
    cs = ChipSpec.from_any(cap)
    assert cs.bw_gbs == 3350 and cs.hbm_gb > 80
    r = capacity_plan(LLAMA8, cap, 16, "bf16", 2048)
    assert r.fits and r.max_concurrency > 0
    with pytest.raises(Exception):
        ChipSpec.from_any(12345)


def test_speculation_simulator_is_load_aware():
    sim = speculation_gain_vs_batch(alpha=0.8, draft_cost_ratio=0.05, max_gamma=8, ctx=128.0,
                                    batches=(1, 2, 4, 8, 16, 64, 256, 1024, 4096))
    assert sim["batch"][0] == 1 and sim["best_gamma"][0] >= 3
    assert sim["best_gamma"][-1] == 0 and sim["gain"][-1] == 1.0
    assert all(g >= 1.0 for g in sim["gain"])
    gam = sim["best_gamma"]
    assert all(b <= a for a, b in zip(gam, gam[1:]))             # optimal gamma never grows with batch
    assert set(sim["gain_by_gamma"]) == set(range(1, 9)) and len(sim["gain_by_gamma"][3]) == len(sim["batch"])
    bad = speculation_gain_vs_batch(alpha=0.2, draft_cost_ratio=0.3)
    assert set(bad["best_gamma"]) == {0}
    with_model = speculation_gain_vs_batch(alpha=0.8, model=LLAMA8, chip=H100, weight_gb=16.0)
    assert with_model["ridge_flop_per_byte"] == pytest.approx(989e12 / 3350e9)


def test_capacity_curve_plot_arrays():
    cur = capacity_curve(LLAMA8, H100, 16, "bf16", contexts=(1024, 4096, 16384, 65536), weight_gb=16.0, itl_slo_ms=20.0)
    n = len(cur["context"])
    for key in ("max_concurrency", "tokens_per_s", "step_ms", "kv_gb_per_seq", "max_concurrency_under_slo", "kv_tokens"):
        assert len(cur[key]) == n
    mc = cur["max_concurrency"]
    assert all(b <= a for a, b in zip(mc, mc[1:])) and mc[0] > mc[-1]
    assert cur["kv_gb_per_seq"][1] == pytest.approx(4096 * 128 * KIB / GB)
