"""Capabilities: CPU-only behaviour, mocked torch.cuda for each architecture, feature gating, listing."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from forge.core.errors import HardwareUnsupported
from forge.hw import capabilities as cap


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FORGE_DISABLE_FEATURES", raising=False)
    cap.clear_cache()
    yield
    cap.clear_cache()


def fake_cuda(monkeypatch, props_list):
    props_list = list(props_list)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(props_list))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: props_list[i])
    cap.clear_cache()


def P(name, major, minor, gib, sms, **kw):
    return SimpleNamespace(name=name, major=major, minor=minor, total_memory=int(gib * 2**30),
                           multi_processor_count=sms, **kw)


def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# ---------------------------------------------------------------------------------------------------------------
def test_cpu_only_capabilities(monkeypatch):
    no_cuda(monkeypatch)
    c = cap.current_capabilities()
    assert c.tier == "C" and c.device_type == "cpu" and c.sm == 0
    assert not (c.has_bf16 or c.has_fp8 or c.has_fp4_block_scaled or c.has_cp_async or c.has_tma or c.has_two_four)
    assert c.num_sms >= 1 and c.total_mem_gb >= 0 and c.name
    assert not c.is_gpu and c.preferred_dtype() == "float32"
    assert cap.current_capabilities(3).device_type == "cpu"  # any index on a CPU host is fine
    assert c.to_dict()["tier"] == "C"


def test_real_environment_does_not_raise():
    # whatever this host is (CPU-only in CI, GPU on vast.ai) this must work
    c = cap.current_capabilities()
    assert c.tier in ("A", "B", "C")
    assert isinstance(cap.list_gpus(), list)


@pytest.mark.parametrize(
    "props,sm,tier,bf16,fp8,fp4,cp,tma,sparse",
    [
        (P("Tesla V100-SXM2-32GB", 7, 0, 31.7, 80), 70, "C", False, False, False, False, False, False),
        (P("Tesla T4", 7, 5, 14.6, 40), 75, "C", False, False, False, False, False, False),
        (P("NVIDIA A100-SXM4-80GB", 8, 0, 79.1, 108), 80, "B", True, False, False, True, False, True),
        (P("NVIDIA GeForce RTX 3090", 8, 6, 23.6, 82), 86, "B", True, False, False, True, False, True),
        (P("Orin", 8, 7, 61, 16), 87, "B", True, False, False, True, False, True),
        (P("NVIDIA GeForce RTX 4090", 8, 9, 23.5, 128), 89, "B", True, True, False, True, False, True),
        (P("NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132), 90, "A", True, True, False, True, True, True),
        (P("NVIDIA B200", 10, 0, 178, 148), 100, "A", True, True, True, True, True, True),
        (P("NVIDIA B300", 10, 3, 280, 160), 103, "A", True, True, True, True, True, True),
        (P("NVIDIA GeForce RTX 5090", 12, 0, 31.3, 170), 120, "A", True, True, True, True, True, True),
        (P("NVIDIA GB10", 12, 1, 119, 48), 121, "A", True, True, True, True, True, True),
    ],
)
def test_mocked_cuda_matrix(monkeypatch, props, sm, tier, bf16, fp8, fp4, cp, tma, sparse):
    fake_cuda(monkeypatch, [props])
    c = cap.current_capabilities(0)
    assert c.sm == sm and c.tier == tier and c.device_type == "cuda"
    assert (c.has_bf16, c.has_fp8, c.has_fp4_block_scaled, c.has_cp_async, c.has_tma, c.has_two_four) == (
        bf16, fp8, fp4, cp, tma, sparse)
    assert c.num_sms == props.multi_processor_count
    assert c.total_mem_gb == pytest.approx(props.total_memory / 2**30, abs=0.01)
    assert c.bf16_emulated == (sm in (70, 75))
    assert c.compute_capability == f"{props.major}.{props.minor}"
    assert c.has_wgmma == (sm == 90)
    assert c.has_tcgen05 == (sm in (100, 103))
    assert c.is_gpu
    assert c.preferred_dtype() == ("bfloat16" if bf16 else "float16")
    assert c.chip_key  # all of these are in the catalogue


def test_driver_smem_overrides_catalogue(monkeypatch):
    fake_cuda(monkeypatch, [P("NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132, shared_memory_per_block_optin=232448,
                              shared_memory_per_multiprocessor=233472, L2_cache_size=50 * 2**20)])
    c = cap.current_capabilities()
    assert c.smem_per_sm_kib == 227.0 and c.smem_total_per_sm_kib == 228.0 and c.l2_mb == 50.0


def test_catalogue_smem_when_driver_silent(monkeypatch):
    fake_cuda(monkeypatch, [P("NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132)])
    assert cap.current_capabilities().smem_per_sm_kib == 227.0
    fake_cuda(monkeypatch, [P("NVIDIA A100-SXM4-80GB", 8, 0, 79.1, 108)])
    assert cap.current_capabilities().smem_per_sm_kib == 163.0


def test_unknown_gpu_gets_sm_based_caps(monkeypatch):
    fake_cuda(monkeypatch, [P("NVIDIA Future 9000", 9, 0, 40, 100)])
    c = cap.current_capabilities()
    assert c.sm == 90 and c.tier == "A" and c.has_fp8 and c.spec_match == "sm_fallback" and c.chip_key is None


def test_pre_volta_is_tier_c_without_triton(monkeypatch):
    fake_cuda(monkeypatch, [P("Tesla P100", 6, 0, 15.9, 56)])
    c = cap.current_capabilities()
    assert c.tier == "C" and not c.triton_ok and not c.has_bf16 and not c.has_fp16 and c.notes


def test_out_of_range_device(monkeypatch):
    fake_cuda(monkeypatch, [P("NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132)])
    with pytest.raises(HardwareUnsupported):
        cap.current_capabilities(1)


def test_broken_torch_cuda_degrades_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: (_ for _ in ()).throw(RuntimeError("driver")))
    assert cap.current_capabilities().device_type == "cpu"


def test_multi_gpu(monkeypatch):
    fake_cuda(monkeypatch, [P("NVIDIA H100 80GB HBM3", 9, 0, 79.1, 132), P("NVIDIA GeForce RTX 4090", 8, 9, 23.5, 128)])
    assert cap.current_capabilities(0).sm == 90 and cap.current_capabilities(1).sm == 89
    rows = cap.list_gpus()
    assert [r["index"] for r in rows] == [0, 1] and rows[0]["source"] == "torch" and rows[1]["features"]["fp8"]
    assert rows[0]["tier"] == "A"


def test_disable_features_env(monkeypatch):
    fake_cuda(monkeypatch, [P("NVIDIA B200", 10, 0, 178, 148)])
    monkeypatch.setenv("FORGE_DISABLE_FEATURES", "fp4, TMA")
    c = cap.current_capabilities()
    assert not c.has_fp4_block_scaled and not c.has_tma and c.has_fp8
    assert "fp4" in c.disabled


def test_supports_api():
    c = cap.capabilities_for_chip("h100-sxm5-80g")
    assert c.supports("fp8") and c.supports("tma") and c.supports("wgmma") and not c.supports("fp4")
    assert c.supports("two_four") and c.supports("bf16") and c.supports("triton")
    with pytest.raises(KeyError):
        c.supports("warp_specialisation_9000")
    assert c.at_least(90) and not c.at_least(100) and c.sm_tag == "sm_90"


def test_capabilities_for_chip_hypothetical():
    b = cap.capabilities_for_chip("b200")
    assert b.has_fp4_block_scaled and b.has_tcgen05 and b.tier == "A" and b.index == -1
    assert b.notes and "hypothetical" in b.notes[0]
    v = cap.capabilities_for_chip("v100-sxm2-32g")
    assert v.tier == "C" and not v.has_bf16 and v.bf16_emulated and v.has_fp16
    g = cap.capabilities_for_chip("gtx-16xx")
    assert not g.has_fp16 and g.tier == "C"
    assert cap.capabilities_for_chip("NVIDIA GeForce RTX 5090").sm == 120


def test_probe_veto_of_unconfirmed_features(monkeypatch):
    """A stored probe report saying fp4 failed turns the SM-level hypothesis off (fp8 stays on)."""
    from forge.hw import descriptor as dm
    from forge.hw import descriptor_store as ds

    p = P("NVIDIA GeForce RTX 5090", 12, 0, 31.3, 170)
    fp = dm.chip_fingerprint(p.name, 120, 170, round(p.total_memory / 2**20))
    d = dm.descriptor_from_chip("rtx-5090")
    d = d.model_copy(update={"id": "veto-test-r1", "fingerprint": fp})
    ds.save_descriptor(d, report={"features": {"fp8": True, "fp4": False}}, write_files=False)
    fake_cuda(monkeypatch, [p])
    c = cap.current_capabilities()
    assert c.fp4_confirmed is False and not c.has_fp4_block_scaled
    assert c.fp8_confirmed is True and c.has_fp8
    ds.delete_descriptor("veto-test-r1")
    cap.clear_cache()
    c2 = cap.current_capabilities()
    assert c2.fp4_confirmed is None and c2.has_fp4_block_scaled and any("hypothesis" in n for n in c2.notes)


def test_tier_and_features_pure_functions():
    assert [cap.tier_for_sm(s) for s in (None, 0, 61, 70, 75, 80, 86, 87, 89, 90, 100, 103, 120, 121)] == list(
        "CCCCCBBBBAAAAA")
    assert cap.demote_tier("A") == "B" and cap.demote_tier("B") == "C" and cap.demote_tier("C") == "C"
    f = cap.sm_features(89)
    assert f["fp8"] and not f["fp4_block_scaled"] and not f["tma"]
    assert not cap.sm_features(None)["triton_ok"]


def test_support_matrix_rows():
    rows = cap.support_matrix()
    assert [r["sm"] for r in rows] == [70, 75, 80, 86, 87, 89, 90, 100, 103, 120, 121]
    r90 = next(r for r in rows if r["sm"] == 90)
    assert r90["tier"] == "A" and r90["tma"] and "h100-sxm5-80g" in r90["chips"]
    assert next(r for r in rows if r["sm"] == 120)["fp4_block_scaled_mma"].startswith("yes (verify")


def test_parse_smi_gpu_list_fixture():
    text = ("0, NVIDIA H100 80GB HBM3, GPU-aaaa-1111, 81559, 550.54.15, 9.0\n"
            "1, NVIDIA GeForce RTX 4090, GPU-bbbb-2222, 24564, 550.54.15, 8.9\n"
            "garbage line\n")
    rows = cap.parse_smi_gpu_list(text)
    assert len(rows) == 2
    assert rows[0]["sm"] == 90 and rows[0]["chip_key"] == "h100-sxm5-80g" and rows[0]["tier"] == "A"
    assert rows[1]["sm"] == 89 and rows[1]["total_mem_gb"] == pytest.approx(23.99, abs=0.05)
    assert rows[0]["source"] == "nvidia-smi" and rows[0]["driver"] == "550.54.15"


def test_list_gpus_falls_back_to_smi(monkeypatch):
    no_cuda(monkeypatch)
    monkeypatch.setattr(cap, "_smi_gpus", lambda: [{"index": 0, "name": "X", "source": "nvidia-smi"}])
    assert cap.list_gpus()[0]["source"] == "nvidia-smi"
    monkeypatch.setattr(cap, "_smi_gpus", lambda: [])
    assert cap.list_gpus() == []


def test_hip_build_is_not_treated_as_nvidia(monkeypatch):
    monkeypatch.setattr(cap, "software_versions", lambda refresh=False: {"hip": "6.0", "cuda": None, "torch": "x",
                                                                         "triton": None})
    fake_cuda(monkeypatch, [P("AMD Instinct MI300X", 9, 4, 191, 304)])
    c = cap.current_capabilities()
    assert c.vendor == "amd" and c.tier == "C" and not c.has_fp8 and not c.is_gpu


def test_cpu_info_shape():
    ci = cap.cpu_info()
    assert ci["logical_cores"] >= 1 and ci["family"] in ("cpu_avx", "cpu_arm", "cpu_generic") and ci["name"]
