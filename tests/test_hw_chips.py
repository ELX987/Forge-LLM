"""Chip database: coverage of every required family, name matching, SM fallback, consistency of the catalogue."""
from __future__ import annotations

import json

import pytest

from forge.hw import chips
from forge.hw.chips import KNOWN_CHIPS, find_chip, normalize_name


def test_required_families_present():
    required = [
        "v100", "t4", "rtx-2080-ti", "rtx-2060", "a100-sxm4-40g", "a100-sxm4-80g", "a100-pcie-80g", "a30", "a10",
        "a10g", "a40", "rtx-3060", "rtx-3070", "rtx-3080-10g", "rtx-3090", "jetson-agx-orin-64g", "rtx-4060",
        "rtx-4070", "rtx-4080", "rtx-4090", "l4", "l40", "l40s", "rtx-6000-ada", "h100-sxm5-80g", "h100-pcie-80g",
        "h100-nvl-94g", "h200-sxm-141g", "gh200-96g", "b200", "gb200", "b300", "rtx-5070", "rtx-5080", "rtx-5090",
        "rtx-pro-6000-blackwell", "gb10",
    ]
    for r in required:
        assert any(k == r or k.startswith(r) for k in KNOWN_CHIPS), r
    sms = {v["sm"] for v in KNOWN_CHIPS.values()}
    assert {70, 75, 80, 86, 87, 89, 90, 100, 103, 120, 121} <= sms


@pytest.mark.parametrize(
    "name,sm,nsm,mem,key",
    [
        ("NVIDIA H100 80GB HBM3", 90, 132, 79.1, "h100-sxm5-80g"),
        ("NVIDIA H100 PCIe", 90, 114, 79.0, "h100-pcie-80g"),
        ("NVIDIA H100 NVL", 90, 132, 93.0, "h100-nvl-94g"),
        ("NVIDIA H200", 90, 132, 139.0, "h200-sxm-141g"),
        ("NVIDIA A100-SXM4-40GB", 80, 108, 39.4, "a100-sxm4-40g"),
        ("NVIDIA A100-SXM4-80GB", 80, 108, 79.2, "a100-sxm4-80g"),
        ("NVIDIA A100 80GB PCIe", 80, 108, 79.0, "a100-pcie-80g"),
        ("NVIDIA A100", 80, 108, 79.0, "a100-sxm4-80g"),  # no capacity hint in the name -> memory decides
        ("NVIDIA GeForce RTX 4090", 89, 128, 23.5, "rtx-4090"),
        ("NVIDIA GeForce RTX 4070 Ti SUPER", 89, 66, 15.6, "rtx-4070-ti-super"),
        ("NVIDIA GeForce RTX 4070 Ti", 89, 60, 11.7, "rtx-4070-ti"),
        ("NVIDIA GeForce RTX 4070", 89, 46, 11.7, "rtx-4070"),
        ("NVIDIA GeForce RTX 3080", 86, 68, 9.7, "rtx-3080-10g"),
        ("NVIDIA GeForce RTX 3080", 86, 70, 11.7, "rtx-3080-12g"),
        ("NVIDIA GeForce RTX 5090", 120, 170, 31.3, "rtx-5090"),
        ("NVIDIA GeForce RTX 5070 Ti", 120, 70, 15.5, "rtx-5070-ti"),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 120, 188, 95, "rtx-pro-6000-blackwell"),
        ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", 120, 188, 95, "rtx-pro-6000-blackwell-maxq"),
        ("NVIDIA GB10", 121, 48, 119, "gb10"),
        ("NVIDIA B200", 100, 148, 178, "b200"),
        ("NVIDIA GB200", 100, 148, 184, "gb200"),
        ("NVIDIA B300", 103, 160, 280, "b300"),
        ("NVIDIA GH200 480GB", 90, 132, 95, "gh200-96g"),
        ("NVIDIA GH200 480GB", 90, 132, 143, "gh200-144g"),
        ("Tesla T4", 75, 40, 14.6, "t4"),
        ("Tesla V100-SXM2-16GB", 70, 80, 15.7, "v100-sxm2-16g"),
        ("Tesla V100-PCIE-32GB", 70, 80, 31, "v100-pcie-32g"),
        ("NVIDIA A10G", 86, 80, 22.3, "a10g"),
        ("NVIDIA A10", 86, 72, 22.3, "a10"),
        ("NVIDIA L4", 89, 58, 22.3, "l4"),
        ("NVIDIA L40S", 89, 142, 44.4, "l40s"),
        ("NVIDIA L40", 89, 142, 44.4, "l40"),
        ("NVIDIA RTX 6000 Ada Generation", 89, 142, 47.5, "rtx-6000-ada"),
        ("NVIDIA RTX A6000", 86, 84, 47.5, "rtx-a6000"),
        ("Orin", 87, 16, 61, "jetson-agx-orin-64g"),
        ("Orin", 87, 8, 7.4, "jetson-orin-nx-8g"),
        ("NVIDIA GeForce RTX 2080 Ti", 75, 68, 10.7, "rtx-2080-ti"),
        ("NVIDIA GeForce GTX 1660 SUPER", 75, 22, 5.7, "gtx-16xx"),
    ],
)
def test_name_matching(name, sm, nsm, mem, key):
    m = find_chip(name, sm=sm, num_sms=nsm, mem_gb=mem)
    assert m.key == key, (name, m.key, m.notes)
    assert m.kind == "exact"
    assert m.spec["num_sms"] == nsm or m.spec["num_sms"] is None


def test_sm_filter_prevents_cross_arch_match():
    # name says 4090 but the driver says sm_90 -> do not trust the name
    m = find_chip("NVIDIA GeForce RTX 4090", sm=90, num_sms=132)
    assert m.key is None and m.kind == "sm_fallback"


def test_laptop_and_unknown_use_sm_fallback():
    m = find_chip("NVIDIA GeForce RTX 4090 Laptop GPU", sm=89, num_sms=76, clock_mhz=2040)
    assert m.kind == "sm_fallback" and m.key is None
    assert m.spec["sm"] == 89
    assert m.spec["confidence"]["tflops"] == "approximate"
    assert m.spec["tflops"]["fp16"] == pytest.approx(76 * 512 * 2040 / 1e6, rel=1e-3)
    assert all(b == "derived" for b in m.spec["tflops_basis"].values())
    assert m.notes


def test_sm_fallback_without_clock_has_no_numbers():
    m = find_chip("Mystery Accelerator", sm=90, num_sms=100)
    assert m.kind == "sm_fallback" and m.spec["tflops"] == {}
    assert m.spec["family"] == "hopper" and m.spec["smem_per_sm_kib"] == 227


def test_unknown_without_sm():
    m = find_chip("Some AMD thing")
    assert m.kind == "unknown" and m.spec["sm"] is None and not m.known


def test_sm_count_mismatch_rescales_and_flags():
    m = find_chip("NVIDIA H100 80GB HBM3", sm=90, num_sms=120)
    assert m.kind == "name_adjusted" and m.known
    assert m.spec["num_sms"] == 120
    assert m.spec["tflops"]["fp16"] == pytest.approx(989.0 * 120 / 132, abs=0.1)
    assert m.spec["confidence"]["tflops"] == "approximate"
    # original catalogue untouched (deep copy)
    assert KNOWN_CHIPS["h100-sxm5-80g"]["num_sms"] == 132
    assert KNOWN_CHIPS["h100-sxm5-80g"]["tflops"]["fp16"] == 989.0


def test_returned_specs_are_copies():
    a = find_chip("Tesla T4", sm=75)
    a.spec["tflops"]["fp16"] = 1.0
    assert KNOWN_CHIPS["t4"]["tflops"]["fp16"] == 65.0
    assert chips.get_chip("t4")["tflops"]["fp16"] == 65.0


def test_normalize_name():
    assert normalize_name("NVIDIA GeForce RTX 4090") == "rtx 4090"
    assert normalize_name("NVIDIA A100-SXM4-80GB") == "a100 sxm4 80gb"
    assert normalize_name("Tesla V100-SXM2-16GB") == "v100 sxm2 16gb"
    assert normalize_name("NVIDIA RTX 6000 Ada Generation") == "rtx 6000 ada"


def test_resolve_key_or_name():
    assert chips.resolve_key_or_name("h100-sxm5-80g").key == "h100-sxm5-80g"
    assert chips.resolve_key_or_name("H100_SXM5_80G").key == "h100-sxm5-80g"
    assert chips.resolve_key_or_name("NVIDIA L4").key == "l4"
    with pytest.raises(KeyError):
        chips.get_chip("nope")


# ---------------------------------------------------------------------------------------------------------------
# catalogue consistency
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(KNOWN_CHIPS))
def test_entry_sanity(key):
    c = KNOWN_CHIPS[key]
    assert c["sm"] in {70, 75, 80, 86, 87, 89, 90, 100, 103, 120, 121}
    assert c["confidence"]["specs"] in ("datasheet", "approximate")
    assert c["confidence"]["tflops"] in ("datasheet", "derived", "mixed", "approximate")
    assert set(c["tflops"]) == set(c["tflops_basis"])
    assert all(b in ("datasheet", "derived", "approximate") for b in c["tflops_basis"].values())
    if c["tensor_cores"]:
        assert c["tflops"].get("fp16", 0) > 0 or key in ("b300",), key
    for dt, v in c["tflops"].items():
        assert v > 0, (key, dt)
    if c["mem_bw_gbs"] and c["tflops"].get("fp16"):
        ridge = c["tflops"]["fp16"] * 1e3 / c["mem_bw_gbs"]
        assert 5 < ridge < 1000, (key, ridge)  # FLOP/byte sanity
    if c["confidence"]["specs"] == "approximate":
        assert c["notes"], f"{key}: approximate entries must say why in notes"
    # tensor-core rates grow with narrower types
    t = c["tflops"]
    if "fp8" in t and "fp16" in t:
        assert t["fp8"] >= t["fp16"]
    if "fp4" in t and "fp8" in t:
        assert t["fp4"] >= t["fp8"]
    # architecture feature gating
    if c["sm"] < 89:
        assert "fp8" not in t
    if c["sm"] < 100:
        assert "fp4" not in t
    if c["sm"] < 80:
        assert "bf16" not in t and "tf32" not in t
    # smem: usable per block never exceeds the physical per-SM size
    assert c["smem_per_sm_kib"] <= c["smem_total_per_sm_kib"]
    # patterns compile
    import re

    for p in c["patterns"] + c["exclude"]:
        re.compile(p)


def test_known_values_match_published_datasheets():
    h = KNOWN_CHIPS["h100-sxm5-80g"]
    assert (h["num_sms"], h["mem_bw_gbs"], h["tflops"]["bf16"], h["tflops"]["fp8"], h["nvlink_gbs"]) == (
        132, 3350, 989.0, 1979.0, 900)
    a = KNOWN_CHIPS["a100-sxm4-80g"]
    assert (a["num_sms"], a["mem_bw_gbs"], a["tflops"]["bf16"], a["l2_mb"]) == (108, 2039, 312.0, 40)
    r = KNOWN_CHIPS["rtx-4090"]
    assert (r["num_sms"], r["mem_gb"], r["mem_bw_gbs"], r["l2_mb"]) == (128, 24, 1008, 72)
    # Plan Example: H100 effective datasheet ridge ~295 FLOP/B (fp16)
    assert h["tflops"]["fp16"] * 1e3 / h["mem_bw_gbs"] == pytest.approx(295.2, abs=0.5)


def test_every_model_in_task_scope_has_provenance_labels():
    for k, c in KNOWN_CHIPS.items():
        assert "specs" in c["confidence"] and "tflops" in c["confidence"], k
    # consumer Blackwell fp4 numbers must be flagged approximate
    assert KNOWN_CHIPS["rtx-5090"]["confidence"]["tflops"] == "approximate"


def test_summary_is_json_serialisable_and_ordered():
    rows = chips.chip_summary()
    json.dumps(rows)
    assert len(rows) == len(KNOWN_CHIPS)
    assert [r["sm"] for r in rows] == sorted(r["sm"] for r in rows)
    assert all(r["provenance"] == "vendor" for r in rows)
    assert chips.list_chips(sm=90) and all(c["sm"] == 90 for c in chips.list_chips(sm=90))
    assert all(c["family"] == "blackwell" for c in chips.list_chips(family="blackwell"))


def test_registered_in_registry():
    from forge.core import registry

    assert "h100-sxm5-80g" in registry.names("chip")
    assert registry.get("chip", "t4")["sm"] == 75
