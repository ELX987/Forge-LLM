"""Training simulator (phase P3): arithmetic invariants, scheme behaviour, chip table extensions, API route.  It is an ESTIMATE: tests check
structure and invariants, never accuracy against hardware."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from forge.planner.spec import CHIP_PRESETS, SCHEMES, parse_scheme, resolve_chip
from forge.planner.trainsim import TrainSimConfig, calibrate, memory_per_gpu, param_counts, simulate

BASE = dict(hidden=768, layers=12, heads=12, kv_heads=12, ffn=2048, vocab=32000, seq_len=2048, global_batch_tokens=262144, grad_checkpoint=True, max_tokens=2e8)


def cfg(**kw):
    return TrainSimConfig(**{**BASE, **kw})


def test_param_counts_125m_matches_tern158_preset():
    pc = param_counts(cfg())
    assert pc["linear"] == 12 * (3 * 768 * 768 + 768 * 768 + 2 * 2048 * 768 + 2048 * 768)
    assert 84e6 < pc["linear"] < 86e6 and pc["embed"] == 32000 * 768 and pc["head"] == 0  # tern158 prints ternary 0.085B / embed 25M


def test_ternary_native_memory_matches_documented_formula():
    """tern158 docs: codes 0.25 B/param (+4 B per group scale), master/Adam/grad divided by the GPU count when sharded."""
    c = cfg(scheme="ternary_native", shard_state=True, master_dtype="bf16", state_dtype="bf16", group_size=128)
    chip = resolve_chip("h100-sxm-80g")
    n = param_counts(c)["linear"]
    for world in (1, 4):
        m = memory_per_gpu(c, chip, world, 1)
        gb = 1024 ** 3
        assert m["packed codes + scales"] == pytest.approx((n * 0.25 + n / 128 * 4) / gb)
        assert m["latent master"] == pytest.approx(n / world * 2 / gb) and m["adam states"] == pytest.approx(n / world * 4 / gb)


def test_sharding_and_dtypes_reduce_memory_monotonically():
    chip = resolve_chip("h100-sxm-80g")
    full = sum(memory_per_gpu(cfg(scheme="bf16"), chip, 8, 4).values())
    sharded = sum(memory_per_gpu(cfg(scheme="bf16", shard_state=True), chip, 8, 4).values())
    assert sharded < full
    t32 = sum(memory_per_gpu(cfg(scheme="ternary_native", shard_state=True), chip, 8, 4).values())
    t16 = sum(memory_per_gpu(cfg(scheme="ternary_native", shard_state=True, master_dtype="bf16", state_dtype="bf16"), chip, 8, 4).values())
    assert t16 < t32


def test_lora_and_qlora_need_far_less_state_than_full_training():
    chip = resolve_chip("h100-sxm-80g")
    big = dict(hidden=4096, layers=32, heads=32, kv_heads=8, ffn=14336, vocab=128256, tie_embeddings=False)
    full = sum(memory_per_gpu(cfg(scheme="bf16", **big), chip, 1, 1).values())
    lora = sum(memory_per_gpu(cfg(scheme="lora", **big), chip, 1, 1).values())
    qlora = sum(memory_per_gpu(cfg(scheme="qlora", **big), chip, 1, 1).values())
    assert qlora < lora < full / 2


def test_results_are_sane_and_flagged_as_estimates():
    r = simulate(cfg(scheme="ternary_native"), "rtx-pro-6000-blackwell", 1, price_per_gpu_hour=1.5)
    assert r.ok and r.estimate and r.tokens_per_s > 0 and 0.02 < r.mfu < 0.9 and r.cost_usd == pytest.approx(r.total_hours * 1.5)
    assert any("ESTIMATE" in n for n in r.notes) and any("Forge has no such trainer" in n for n in r.notes)
    assert r.energy_kwh == pytest.approx(r.total_hours * 600 * 0.8 / 1000)


def test_does_not_fit_gives_advice():
    r = simulate(cfg(scheme="bf16", hidden=8192, layers=80, heads=64, kv_heads=8, ffn=28672, vocab=128256, tie_embeddings=False), "rtx-4090", 1)
    assert not r.ok and any("does NOT fit" in n for n in r.notes) and r.suggestions


def test_int8_throttled_chip_routes_ternary_to_bf16_path():
    c = cfg(scheme="ternary_native")
    b300 = simulate(c, "b300", 1)
    assert any("throttled" in n for n in b300.notes) and resolve_chip("b300").int8_tc_fast is False
    h100 = simulate(c, "h100-sxm-80g", 1)
    assert not any("throttled" in n for n in h100.notes)
    assert resolve_chip("h100-sxm-80g").int8_tc_fast is True


def test_emulated_schemes_are_not_faster_than_bf16():
    t = {s: simulate(cfg(scheme=s), "a100-80g", 1).tokens_per_s for s in ("bf16", "ternary_qat", "nvfp4_emulated")}
    assert t["ternary_qat"] < t["bf16"] and t["nvfp4_emulated"] < t["bf16"]


def test_calibration_scales_throughput_and_multi_gpu_adds_comm():
    r = simulate(cfg(), "h100-sxm-80g", 1)
    k = calibrate(r.tokens_per_s, r.tokens_per_s * 0.5)
    r2 = simulate(cfg(), "h100-sxm-80g", 1, calibration=k)
    assert r2.tokens_per_s == pytest.approx(r.tokens_per_s * 0.5, rel=1e-6)
    r8 = simulate(cfg(), "h100-sxm-80g", 8)
    assert r8.breakdown_s["exposed comm"] > 0 and r8.tokens_per_s > r.tokens_per_s


def test_chip_table_extensions():
    c = resolve_chip("NVIDIA RTX PRO 6000 Blackwell Server Edition")
    assert c.name == "rtx-pro-6000-blackwell" and c.tdp_w == 600 and c.sm == 120 and "NOT confirmed" in c.notes
    assert all(p.int8_tc_fast for n, p in CHIP_PRESETS.items() if n != "b300")


def test_t158_scheme_is_experimental_with_expected_bits():
    s = parse_scheme("ternary")
    assert s is SCHEMES["t158"] and s.experimental and s.act == "int8" and s.weight_bits_eff == pytest.approx(2.25)
    assert parse_scheme("t158+kv8").kv_bits == 8.0


# ------------------------------------------------------------------------------------------------ API
pytest.importorskip("fastapi")
sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home, make_client  # noqa: E402,F401


def test_train_sim_endpoint(client):
    r = client.post("/api/v1/planner/train-sim", json={"config": {"scheme": "ternary_qat", "hidden": 512, "layers": 6, "heads": 8, "kv_heads": 4, "ffn": 1536, "vocab": 50304,
                                                                 "seq_len": 512, "global_batch_tokens": 32768, "max_tokens": 3e7}, "chip": "rtx-pro-6000-blackwell", "n_gpu": 1})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["ok"] and b["estimate"] and b["scheme"] == "ternary_qat" and b["tokens_per_s"] > 0 and "activations" in b["mem_gb"]
    bad = client.post("/api/v1/planner/train-sim", json={"config": {"scheme": "nope"}})
    assert bad.status_code in (400, 422)


# ------------------------------------------------------------------------------------------------ chip table reconciliation
# tern158's python/tern158/gpus.py values (vendor datasheet / its report) for the GPUs both projects know: (preset, vram_gb, bw_gbs, bf16, int8).
_TERN158 = {
    "h100-sxm-80g": (80, 3350, 989, 1979), "h100-pcie-80g": (80, 2000, 756, 1513), "h200-141g": (141, 4800, 989, 1979), "a100-80g": (80, 2039, 312, 624),
    "a100-40g": (40, 1555, 312, 624), "a10": (24, 600, 125, 250), "l4": (24, 300, 121, 242), "l40s": (48, 864, 362, 733), "rtx-3090": (24, 936, 71, 284),
    "rtx-4090": (24, 1008, 165, 661), "rtx-5090": (32, 1792, 209, None), "b200": (180, 8000, 2250, 4500), "b300": (288, 8000, 2250, None),
}
# Known disagreements, deliberately NOT resolved silently: Forge keeps its own number until the chip probe measures it.
#   rtx-3090 / rtx-4090 int8: Forge lists fp32-accumulate-halved GeForce rates (142 / 330); tern158 lists 284 / 661 (headline INT8 dense).
#   b200 vram: 192 GB (Forge) vs 180 GB (tern158).   b300 bf16: 2500 (Forge, P5 Table 11.2) vs 2250 (tern158).   rtx-5090 bf16 210 vs 209 is rounding.
_KNOWN = {("rtx-3090", "int8"), ("rtx-4090", "int8"), ("b200", "vram"), ("b300", "bf16")}


def test_chip_tables_agree_or_disagreement_is_documented():
    diffs = set()
    for name, (vram, bw, bf16, int8) in _TERN158.items():
        c = CHIP_PRESETS[name]
        pairs = {"vram": (c.hbm_gb, vram), "bw": (c.hbm_bw_gbs, bw), "bf16": (c.peak_tflops.get("bf16"), bf16), "int8": (c.peak_tflops.get("int8"), int8)}
        for k, (fv, tv) in pairs.items():
            if fv is None or tv is None:
                continue
            if abs(fv - tv) / max(fv, tv) > 0.03:
                diffs.add((name, k))
    assert diffs == _KNOWN, (diffs ^ _KNOWN)
