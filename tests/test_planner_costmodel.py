from __future__ import annotations

import math
import random

import pytest

from forge.planner import costmodel as cm
from forge.planner.spec import CHIP_PRESETS, MODEL_PRESETS, ChipSpec, ModelSpec, Scheme, parse_scheme, resolve_chip

H100 = CHIP_PRESETS["h100-sxm-80g"]
KiB = 1024


# ------------------------------------------------------------------ bits / bytes / KV (Appendix D, Examples A and B)
def test_effective_bits_match_appendix_d():
    assert parse_scheme("nvfp4").weight_bits_eff == pytest.approx(4.5)
    assert parse_scheme("mxfp4").weight_bits_eff == pytest.approx(4.25)
    assert parse_scheme("w4a16-g128").weight_bits_eff == pytest.approx(4.125)
    assert parse_scheme("w4a16-g64").weight_bits_eff == pytest.approx(4.25)
    assert parse_scheme("fp8").weight_bits_eff == 8 and parse_scheme("bf16").weight_bits_eff == 16
    assert parse_scheme("w4a16+kv8").kv_bits == 8 and parse_scheme("fp8+kv4").kv_bits == pytest.approx(4.25)
    assert parse_scheme("w3a16-g32").weight_bits_eff == pytest.approx(3.5)
    assert parse_scheme("fp8+sp40").activation_sparsity == 0.4
    with pytest.raises(Exception):
        parse_scheme("nope")


def test_kv_bytes_per_token_numbers_from_the_plan():
    kib = lambda spec, bits=16: MODEL_PRESETS[spec].kv_bytes_per_token(bits) / KiB
    assert kib("llama-3-8b") == 128
    assert kib("qwen3-8b") == 144
    assert kib("llama-3.1-70b") == 320
    assert kib("qwen3-next-80b-a3b") == 24
    assert kib("kimi-linear-48b-a3b") == pytest.approx(8, abs=0.2)  # 7 MLA layers x 576 x 2 B = 7.875 KiB ("about 8")
    assert kib("deepseek-v3") == pytest.approx(61 * 576 * 2 / KiB)  # MLA caches 576 values per token per layer
    # free functions agree
    assert cm.kv_bytes_per_token(32, 8, 128, 16) == 131072
    assert cm.kv_bytes_per_token(61, 0, 0, 16, mla_latent=512, mla_rope=64) == 61 * 576 * 2
    # 128K tokens of Llama-3-8B = 16 GiB, "as large as the bf16 weights themselves"
    assert 128 * 1024 * cm.kv_bytes_per_token(32, 8, 128) == 16 * 1024**3
    # architecture, not post-hoc compression, accounts for most of the 5-18x spread
    assert 144 / 24 == 6 and 144 / 8 == 18


def test_sliding_window_layers_cap_kv():
    m = ModelSpec(name="x", params_b=1, num_layers=10, num_kv_heads=2, head_dim=64, sliding_window=1024, sliding_layers=8)
    full = m.kv_bytes_for_context(100_000, 16)
    per_layer = 2 * 2 * 64 * 2
    assert full == per_layer * (2 * 100_000 + 8 * 1024)
    assert m.kv_bytes_for_context(512, 16) == per_layer * 10 * 512


def test_example_a_which_weights_fit_and_how_much_kv_is_left():
    hbm = 80e9
    kv16 = cm.kv_bytes_per_token(80, 8, 128, 16)
    assert kv16 == 320 * KiB
    assert cm.weight_bytes(70e9, 16) / 1e9 == 140  # does not fit one 80 GB device
    assert cm.weight_bytes(70e9, 16) > hbm
    w_fp8 = cm.weight_bytes(70e9, 8)
    assert w_fp8 == 70e9
    free = hbm - w_fp8
    assert free / kv16 == pytest.approx(30.5e3, rel=0.01)  # "30 k (< one 32 k sequence)"
    assert free / cm.kv_bytes_per_token(80, 8, 128, 8) == pytest.approx(61e3, rel=0.01)
    lo, hi = free / (90 * KiB), free / (80 * KiB)
    assert lo == pytest.approx(108e3, rel=0.01) and hi == pytest.approx(122e3, rel=0.01)  # "108-122 k"
    # W4A16 g128: plan says 37 GB; our b_eff = 4.125 gives 36.1 GB for exactly 70B parameters (within rounding)
    w4 = cm.weight_bytes(70e9, 4.125)
    assert w4 / 1e9 == pytest.approx(37, rel=0.03)
    free4 = hbm - 37e9
    assert free4 / kv16 == pytest.approx(131e3, rel=0.01)
    assert free4 / cm.kv_bytes_per_token(80, 8, 128, 8) == pytest.approx(262e3, rel=0.01)
    assert free4 / (90 * KiB) == pytest.approx(467e3, rel=0.02) and free4 / (80 * KiB) == pytest.approx(525e3, rel=0.01)  # "470-525 k"


def test_example_b_ideal_decode_ceilings():
    bw = 3.35e12
    t = cm.decode_step_time(cm.weight_bytes(8e9, 16), 1, 0, 0, bw)
    assert t * 1e3 == pytest.approx(4.8, abs=0.05) and 1 / t == pytest.approx(210, rel=0.01)
    t = cm.decode_step_time(4.5e9, 1, 0, 0, bw)
    assert t * 1e3 == pytest.approx(1.3, abs=0.05) and 1 / t == pytest.approx(740, rel=0.02)
    t = cm.decode_step_time(37e9, 1, 0, 0, bw)
    assert t * 1e3 == pytest.approx(11, abs=0.1) and 1 / t == pytest.approx(90, rel=0.01)


def test_decode_step_includes_kv_and_selector_and_eta():
    base = cm.decode_step_time(10e9, 1, 0, 0, 1e12)
    assert cm.decode_step_time(10e9, 8, 4096, 1e5, 1e12) == pytest.approx(base + 8 * 4096 * 1e5 / 1e12)
    assert cm.decode_step_time(10e9, 1, 0, 0, 1e12, selector_bytes=5e9) == pytest.approx(1.5 * base)
    assert cm.decode_step_time(10e9, 1, 0, 0, 1e12, eta_mem=0.5) == pytest.approx(2 * base)


# ------------------------------------------------------------------ intensity / crossover (P1 14.1)
def test_batch_crossover_74_and_295():
    bstar16 = cm.batch_crossover(989e12, 3.35e12, 16)
    bstar4 = cm.batch_crossover(989e12, 3.35e12, 4)
    assert bstar16 == pytest.approx(295, abs=0.5)  # ridge point of Fig 5.1
    assert bstar4 == pytest.approx(74, abs=0.5)
    assert round(bstar4) == 74 and round(bstar16) == 295
    assert cm.batch_crossover(989e12, 3.35e12, 8) == pytest.approx(147.6, abs=0.2)
    assert H100.ridge("bf16") == pytest.approx(295, abs=0.5)
    # the speed-up of 4-bit weights is the full 4x only below B*, then decays like 295/B in between (P4 F5 / Appendix D)
    assert cm.gemm_intensity(1, 16) == 1 and cm.gemm_intensity(1, 4) == 4
    assert cm.batch_crossover(989e12, 3.35e12, 4, eta=0.5) == pytest.approx(bstar4 / 2)


def test_attention_intensity_is_g_at_16_bit():
    assert cm.attention_intensity(8, 16) == 8
    assert cm.attention_intensity(8, 8) == 16
    assert MODEL_PRESETS["llama-3-8b"].group_size == 4
    assert MODEL_PRESETS["llama-3.1-70b"].group_size == 8


# ------------------------------------------------------------------ quantiser overhead (P5 Table 11.2)
@pytest.mark.parametrize(
    "n,r4,r8,s_unf,s_fused",
    [(2048, 1.17, 0.46, 2.01, 4.37), (4096, 0.59, 0.23, 2.32, 3.69), (8192, 0.29, 0.11, 2.59, 3.34), (16384, 0.15, 0.06, 2.77, 3.17), (32768, 0.07, 0.03, 2.88, 3.09)],
)
def test_fp4_quantiser_overhead_reproduces_table_11_2(n, r4, r8, s_unf, s_fused):
    unf = cm.fp4_vs_fp8_speedup(n)
    assert unf["tq_tg_fp4"] == pytest.approx(r4, abs=0.006)
    assert unf["tq_tg_fp8"] == pytest.approx(r8, abs=0.006)
    assert unf["peak_ratio"] == 3.0
    assert unf["speedup"] == pytest.approx(s_unf, abs=0.006)
    assert cm.fp4_vs_fp8_speedup(n, fused_fp4=True)["speedup"] == pytest.approx(s_fused, abs=0.006)


def test_59_percent_and_2_3x_headline_numbers():
    r = cm.fp4_vs_fp8_speedup(4096)
    assert r["tq_tg_fp4"] == pytest.approx(0.59, abs=0.005)  # "unfused FP4 quantisation costs 59% of GEMM time at N=4096"
    assert r["speedup"] == pytest.approx(2.3, abs=0.03)  # "cuts a 3x peak ratio to 2.3x"
    # constants of eq. 11.1: 2402/N for FP4 and 937/N for FP8
    k4 = cm.quantizer_overhead(2, 0.5625, 15e15, 1, 8e12)
    k8 = cm.quantizer_overhead(2, 1.0, 5e15, 1, 8e12)
    assert k4 == pytest.approx(2402, abs=1) and k8 == pytest.approx(937, abs=1)
    assert 16 / 4.5 == pytest.approx(3.56, abs=0.005) and 8 / 4.5 == pytest.approx(1.78, abs=0.005)  # compression ratios in Appendix D


# ------------------------------------------------------------------ speculation (P7)
def test_spec_expected_tokens_and_speedup():
    assert cm.spec_expected_tokens(0.8, 4) == pytest.approx(3.36, abs=0.005)  # P7: E[N] = 3.36
    assert cm.spec_expected_tokens(0.0, 5) == 1.0 and cm.spec_expected_tokens(1.0, 5) == 6.0
    assert cm.spec_expected_tokens(0.8, 0) == 1.0
    assert cm.spec_speedup(0, 0.8, 0.1) == 1.0
    assert cm.spec_speedup(4, 0.8, 0.05) == pytest.approx(3.3616 / 1.2, rel=1e-4)
    assert cm.spec_speedup(1, 0.3, 0.5) < 1  # alpha < c: a slowdown
    assert cm.spec_speedup(1, 0.6, 0.5) > 1


def test_spec_batch_dependence_matches_p7_table():
    n = 221
    assert cm.spec_speedup_batched(1, 4, 0.8, n) == pytest.approx(3.362, abs=0.001)
    assert cm.spec_speedup_batched(44, 4, 0.8, n) == pytest.approx(3.362, abs=0.001)  # plateau to n/(gamma+1) = 44
    assert cm.spec_speedup_batched(64, 4, 0.8, n) == pytest.approx(2.326, rel=0.005)
    assert cm.spec_speedup_batched(100, 4, 0.8, n) == pytest.approx(1.489, rel=0.005)
    assert cm.spec_speedup_batched(300, 4, 0.8, n) == pytest.approx(0.672, abs=0.001)  # floor E[N]/(gamma+1)
    assert cm.spec_break_even_batch(4, 0.8, n) == pytest.approx(149, abs=1)  # P7: B_be ~ 149
    # at the break-even batch the speed-up is exactly 1
    assert cm.spec_speedup_batched(cm.spec_break_even_batch(4, 0.8, n), 4, 0.8, n) == pytest.approx(1.0)


def test_optimal_gamma_includes_zero():
    g, s = cm.optimal_gamma(0.8, 0.05, 8)
    assert g >= 3 and s == pytest.approx(cm.spec_speedup(g, 0.8, 0.05))
    assert cm.optimal_gamma(0.3, 0.5, 8) == (0, 1.0)  # adaptive that includes 0 is the safe default
    assert cm.optimal_gamma(0.5, 0.45, 8, hysteresis=0.03)[0] == 0 or cm.optimal_gamma(0.5, 0.45, 8)[1] >= 1.03


def test_amdahl():
    assert cm.amdahl([0.6, 0.3, 0.1], [4, 1, 1]) == pytest.approx(1 / (0.15 + 0.3 + 0.1))
    assert cm.amdahl([1.0], [3.0]) == 3.0
    assert cm.amdahl([2, 1], [2, 1], normalise=True) == pytest.approx(1 / (1 / 3 + 1 / 3))
    with pytest.raises(ValueError):
        cm.amdahl([0.5, 0.4], [2, 2])
    with pytest.raises(ValueError):
        cm.amdahl([1.0], [0.0])


# ------------------------------------------------------------------ t_pred / residual / eta fitting
def test_t_pred_terms_and_bound():
    chip = H100
    eta = cm.EtaFactors(0.8, 0.5, 0.5)
    tb = cm.t_pred(10e9, 1e9, 0, chip, eta=eta, launch_s=0.0)
    assert tb.bound == "mem" and tb.t == pytest.approx(10e9 / (0.8 * chip.bw))
    tb = cm.t_pred(1e6, 1e15, 0, chip, eta=eta, launch_s=0.0)
    assert tb.bound == "mma" and tb.t == pytest.approx(1e15 / (0.5 * 989e12))
    tb = cm.t_pred(1e6, 1e6, 1e13, chip, eta=eta, launch_s=0.0)
    assert tb.bound == "alu"
    tb = cm.t_pred(10e9, 0, 0, chip, eta=cm.IDEAL, launch_s=10e-6, selector_bytes=1e9)
    assert tb.t == pytest.approx(10e9 / chip.bw + 10e-6 + 1e9 / chip.bw)
    assert cm.t_pred(1e9, 0, 0, chip, eta=cm.IDEAL, launches=10, launch_s=1e-6).t_launch == pytest.approx(10e-6)


def test_residual_alert_band():
    assert not cm.residual_alert(cm.residual(1.0, 1.0))
    assert not cm.residual_alert(0.8) and not cm.residual_alert(1.25)
    assert cm.residual_alert(0.79) and cm.residual_alert(1.26) and cm.residual_alert(2.0)
    assert cm.RESIDUAL_BAND == (0.8, 1.25)


def _synth(chip, true: cm.EtaFactors, n_mem=24, n_mma=12, outliers=3, seed=0):
    rng = random.Random(seed)
    ms = []
    for i in range(n_mem):
        b = rng.uniform(2e8, 4e9)
        t = b / (true.eta_mem * chip.bw) * rng.gauss(1.0, 0.01) + 5e-6
        ms.append(cm.Measurement("gemv", t, bytes_mem=b, flops=2 * b / 2, dtype="bf16"))
    for i in range(n_mma):
        f = rng.uniform(5e12, 5e13)
        t = f / (true.eta_mma * chip.peak_flops("bf16")) * rng.gauss(1.0, 0.01) + 5e-6
        ms.append(cm.Measurement("gemv", t, bytes_mem=f / 800.0, flops=f, dtype="bf16"))
    for i in range(outliers):  # stragglers (thermal throttling / OS noise)
        m = ms[i]
        ms[i] = cm.Measurement(m.kernel_class, m.t_measured * 6, m.bytes_mem, m.flops, m.dequant_ops, m.dtype)
    return ms


def test_fit_eta_recovers_efficiencies_robustly():
    true = cm.EtaFactors(0.72, 0.55, 0.4)
    fit = cm.fit_eta(_synth(H100, true), H100)
    assert fit.eta.eta_mem == pytest.approx(0.72, rel=0.04)
    assert fit.eta.eta_mma == pytest.approx(0.55, rel=0.04)
    assert fit.eta.eta_alu == 0.5  # never observed -> keeps the prior
    assert fit.n_by_term["mem"] >= 20 and fit.n_by_term["mma"] >= 10 and fit.converged
    assert fit.eta.source == "fit" and fit.n_samples == 36
    # the 3 stragglers are flagged, the rest are not
    assert sum(fit.alerts) == 3
    with pytest.raises(ValueError):
        cm.fit_eta([], H100)


def test_eta_persistence_roundtrip():
    chip = H100.model_copy(update={"fingerprint": "fp-test-1"})
    assert cm.load_eta(chip, "w4a16_gemv").source == "prior"
    cm.save_eta(chip, "w4a16_gemv", cm.EtaFactors(0.7, 0.4, 0.3, "fit"))
    got = cm.load_eta(chip, "w4a16_gemv")
    assert got.eta_mem == 0.7 and got.source == "fit"
    assert cm.load_eta(chip, "other").source == "prior"


# ------------------------------------------------------------------ model-level predictions
def test_moe_expert_touching_grows_with_batch():
    assert cm.moe_expected_active_experts(64, 8, 1) == pytest.approx(8)
    assert cm.moe_expected_active_experts(64, 8, 1000) == pytest.approx(64, abs=1e-3)
    assert cm.moe_expected_active_experts(64, 8, 0) == 0
    m = MODEL_PRESETS["qwen3-30b-a3b"]
    s = parse_scheme("bf16")
    w1, w64, w512 = (cm.active_weight_bytes(m, s, b) for b in (1, 64, 512))
    assert w1 < w64 < w512 <= cm.model_weight_bytes(m, s)["total"] * 1.001
    assert w1 / cm.model_weight_bytes(m, s)["total"] < 0.2  # B=1 reads only a small fraction of the model
    assert 0.5 < cm.expert_fraction(m) <= 1.0


def test_predict_decode_llama8b_matches_example_b_within_model_assumptions():
    m = MODEL_PRESETS["llama-3-8b"]
    p = cm.predict_decode(m, H100, parse_scheme("bf16"), batch=1, ctx=0)
    # the plan's naive 16 GB / 3.35 TB/s = 4.8 ms; the refined model does not read the input-embedding table (~0.5B params)
    assert p.step_time_ideal_s * 1e3 == pytest.approx(4.5, abs=0.3) and p.bound == "mem"
    assert p.tok_s_stream_ideal == pytest.approx(220, rel=0.08)
    assert p.tok_s_stream_pred < p.tok_s_stream_ideal  # eta < 1 and launch overhead
    w4 = cm.predict_decode(m, H100, parse_scheme("w4a16-g128"), batch=1, ctx=0)
    assert 2.5 < w4.tok_s_stream_ideal / p.tok_s_stream_ideal < 3.9  # embeddings stay 16-bit, so < 4x
    # long context makes KV matter
    long = cm.predict_decode(m, H100, parse_scheme("w4a16-g128"), batch=32, ctx=32768)
    assert long.bytes_kv > long.bytes_weights and long.step_time_pred_s > w4.step_time_pred_s
    kv8 = cm.predict_decode(m, H100, parse_scheme("w4a16-g128+kv8"), batch=32, ctx=32768)
    assert kv8.bytes_kv == pytest.approx(long.bytes_kv / 2)
    # large batch becomes compute-bound
    big = cm.predict_decode(m, H100, parse_scheme("bf16"), batch=2048, ctx=128)
    assert big.bound == "mma"


def test_effective_scheme_emulates_missing_hardware():
    a100 = CHIP_PRESETS["a100-80g"]
    s, notes = cm.effective_scheme(parse_scheme("fp8"), a100)
    assert s.act == "bf16" and notes and "FP8" in notes[0]
    s, notes = cm.effective_scheme(parse_scheme("nvfp4"), H100)
    assert s.act == "bf16" and "no block-scaled FP4" in notes[0]
    s, notes = cm.effective_scheme(parse_scheme("nvfp4"), CHIP_PRESETS["b200"])
    assert s.act == "fp4" and not notes
    s, notes = cm.effective_scheme(parse_scheme("bf16"), CHIP_PRESETS["v100-32g"])
    assert s.act == "fp16"
    p = cm.predict_decode(MODEL_PRESETS["llama-3-8b"], a100, parse_scheme("fp8"), batch=1, ctx=0)
    assert p.notes


def test_predict_prefill_is_compute_bound():
    r = cm.predict_prefill(MODEL_PRESETS["llama-3-8b"], H100, parse_scheme("bf16"), tokens=4096)
    assert r["bound"] == "mma" and 0.01 < r["ttft_s"] < 1.0


def test_chip_resolution():
    assert resolve_chip("H100").name == "h100-sxm-80g"
    assert resolve_chip("NVIDIA GeForce RTX 4090").name == "rtx-4090"
    assert resolve_chip("NVIDIA L40S").name == "l40s" and resolve_chip("L4").name == "l4" and resolve_chip("NVIDIA A100-SXM4-40GB").name == "a100-40g"
    assert resolve_chip("NVIDIA A10").name == "a10" and resolve_chip("NVIDIA H100 PCIe").name == "h100-pcie-80g"
    assert resolve_chip({"name": "x", "hbm_gb": 10, "hbm_bw_gbs": 100, "peak_tflops": {"bf16": 10}}).hbm_gb == 10
    with pytest.raises(Exception):
        resolve_chip("nonexistent-gpu")
    from forge.core.schemas import ChipDescriptor

    d = ChipDescriptor(id="h100-r3", name="NVIDIA H100 80GB HBM3", sm=90, compute_units=132,
                       memory={"hbm_gb": 80, "hbm_bw_gbs": {"datasheet": 3350, "measured": 3000}},
                       matrix_units=[{"dtype": "bf16|fp16", "peak_tflops": {"datasheet": 989, "measured": 700}}])
    c = resolve_chip(d)
    assert c.hbm_bw_gbs == 3000 and c.peak_flops("bf16") == 700e12 and c.peak_flops("fp8") is not None  # gap filled from the preset
    assert c.provenance.value == "measured"
    with pytest.raises(Exception):
        resolve_chip(ChipDescriptor(id="mystery"))


def test_model_spec_from_hf_config_and_estimates():
    cfg = {"num_hidden_layers": 32, "hidden_size": 4096, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256}
    m = ModelSpec.from_config(cfg)
    assert m.head_dim == 128 and m.params_estimated and m.params_b == pytest.approx(8.03, rel=0.02)
    assert m.kv_bytes_per_token() == 128 * KiB
    moe = ModelSpec.from_config({"num_hidden_layers": 48, "hidden_size": 2048, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
                                 "num_experts": 128, "num_experts_per_tok": 8, "moe_intermediate_size": 768, "vocab_size": 151936, "intermediate_size": 6144})
    assert moe.form == "moe" and moe.active_params_b < moe.params_b / 4 and moe.params_b == pytest.approx(30.5, rel=0.1)
    mla = ModelSpec.from_config({"num_hidden_layers": 61, "hidden_size": 7168, "num_attention_heads": 128, "num_key_value_heads": 128, "kv_lora_rank": 512, "qk_rope_head_dim": 64})
    assert mla.attention == "mla" and mla.kv_bytes_per_token() == 61 * 576 * 2

    class Cfg:  # forge.models.ModelConfig-like object
        num_layers = 2
        hidden_size = 64
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        vocab_size = 100
        intermediate_size = 128

    assert ModelSpec.from_config(Cfg()).num_kv_heads == 2
    assert ModelSpec.from_config("llama-3-8b") is MODEL_PRESETS["llama-3-8b"]
    with pytest.raises(Exception):
        ModelSpec.from_config("gpt-unknown")
    with pytest.raises(Exception):
        ModelSpec.from_config({"hidden_size": 1})
