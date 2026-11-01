"""CUDA stack rules (forge.hw.cuda_compat): pure-function tests on synthetic stacks, including the two real hosts seen in the field."""
from __future__ import annotations

import json

import pytest

from forge.hw import cuda_compat as cc


def _dev(i=0, name="NVIDIA GeForce RTX 5090", sm=120, mode="Default"):
    return {"index": i, "name": name, "driver": "580.178.04", "sm": sm, "compute_mode": mode, "mem_mib": 32607}


ARCHS_13 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]


def stack(**kw):
    s = {"driver_version": "595.91.07", "driver_cuda": "13.4", "smi_present": True, "smi_ok": True, "devices": [_dev()],
         "torch": {"installed": True, "version": "2.14.0a0+nv26.08", "cuda": "13.4", "container_build": True, "arch_list": ARCHS_13},
         "triton": "3.8.0", "nvidia_ml_py": "13.610.43", "nvcc": "13.4",
         "probe": {"available": True, "device_count": 1, "devices": [{"torch_index": 0, "ok": True}]}}
    for k, v in kw.items():
        if k == "torch":
            s["torch"] = {**s["torch"], **v}
        else:
            s[k] = v
    return s


def codes(st, level=None):
    return {f["code"] for f in cc.evaluate(st) if level is None or f["level"] == level}


def test_parsing():
    assert cc.parse_version("580.178.04") == (580, 178, 4)
    assert cc.fmt_driver((580, 65, 6)) == "580.65.06" and cc.fmt_driver((595, 91, 7)) == "595.91.07" and cc.fmt_driver(None) == "?"
    assert cc.parse_version("13.4") == (13, 4)
    assert cc.parse_version("2.14.0a0+x") == (2, 14, 0)
    assert cc.parse_version("") is None and cc.parse_version(None) is None and cc.parse_version("n/a") is None
    assert cc.cuda_int_to_version(13000) == (13, 0) and cc.cuda_int_to_version(12080) == (12, 8)
    assert cc.sm_from_compute_cap("12.0") == 120 and cc.sm_from_compute_cap("8.9") == 89 and cc.sm_from_compute_cap("") is None


def test_healthy_host_is_ok():
    f = cc.evaluate(stack())
    assert cc.worst(f) == "ok", f


def test_report7_host_old_driver_with_cuda134_torch_busy_devices():
    st = stack(driver_version="580.178.04", driver_cuda="13.0",
               probe={"available": True, "device_count": 4, "devices": [
                   {"torch_index": i, "ok": False, "error": "CUDA error: CUDA-capable device(s) is/are busy or unavailable"} for i in range(4)]},
               devices=[_dev(i) for i in range(4)])
    f = {x["code"]: x for x in cc.evaluate(st)}
    # NVIDIA documents a CUDA 13.4 app on a 580.65.06 driver as supported: informational, never the reason for a failure by itself
    assert f["torch_cuda_newer_minor"]["level"] == "info"
    assert "580.65.06" in f["torch_cuda_newer_minor"]["message"] and "R615" in f["torch_cuda_newer_minor"]["fix"]
    d = f["devices_unusable"]
    assert d["level"] == "fail" and "4 of 4" in d["message"]
    assert "minor-version compatibility" in d["fix"] and "Exclusive_Process" in d["fix"] and "/dev/nvidia" in d["fix"]
    assert not cc.check(stack=st)["ok"]


def test_cuda13_family_boundaries():
    # every 13.x torch on a 13.0 driver is supported by minor-version compatibility: informational, never a failure by itself
    for minor in range(0, 5):
        st = stack(driver_version="580.65.06", driver_cuda="13.0", torch={"cuda": f"13.{minor}"})
        c = codes(st)
        assert "torch_cuda_newer_major" not in c and "driver_below_family_floor" not in c
        assert ("torch_cuda_newer_minor" in c) == (minor > 0)
    # driver below the 13.x floor
    st = stack(driver_version="575.51.03", driver_cuda="12.9", torch={"cuda": "13.0"})
    assert "driver_below_family_floor" in codes(st, "fail")
    st = stack(driver_version="570.26", driver_cuda="12.8", torch={"cuda": "13.1"})
    assert "driver_below_family_floor" in codes(st, "fail")
    # newer driver than torch is fine
    st = stack(driver_version="595.91.07", driver_cuda="13.4", torch={"cuda": "12.8"}, nvcc=None)
    assert "driver_torch_match" in codes(st) and cc.worst(cc.evaluate(st)) == "ok"


def test_driver_major_older_than_torch_major_fails():
    st = stack(driver_version="560.35.03", driver_cuda="12.6", torch={"cuda": "13.0"})
    assert "driver_below_family_floor" in codes(st, "fail")


def test_major_mismatch_when_driver_version_string_is_unknown():
    st = stack(driver_version=None, driver_cuda="12.6", torch={"cuda": "13.0"})
    assert "torch_cuda_newer_major" in codes(st, "fail")


def test_blackwell_needs_cuda_128_torch():
    st = stack(torch={"cuda": "12.4", "container_build": False, "arch_list": ["sm_80", "sm_90"]}, driver_cuda="13.4")
    assert "arch_needs_newer_cuda" in codes(st, "fail")
    st = stack(torch={"cuda": "12.8", "container_build": False, "arch_list": ["sm_80", "sm_90", "sm_100", "sm_120"]}, nvcc=None)
    assert cc.worst(cc.evaluate(st)) == "ok"


def test_arch_missing_from_torch_build():
    st = stack(torch={"arch_list": ["sm_80", "sm_90"], "cuda": "12.8"})  # 12.8 torch with no sm_120
    assert "arch_not_in_torch" in codes(st, "fail")
    st = stack(torch={"arch_list": ["sm_80", "compute_90"], "cuda": "12.8"})
    assert "arch_ptx_only" in codes(st, "warn")
    st = stack(devices=[_dev(sm=103, name="B300")], torch={"arch_list": ["sm_100", "compute_100"], "cuda": "13.0"})
    assert "arch_family_sass" in codes(st, "info") and "arch_not_in_torch" not in codes(st)


def test_cuda13_drops_volta():
    st = stack(devices=[_dev(name="Tesla V100", sm=70)], torch={"cuda": "13.0", "arch_list": ["sm_75", "sm_80"]})
    assert "cuda13_dropped_arch" in codes(st, "fail")
    st = stack(devices=[_dev(name="Tesla V100", sm=70)], torch={"cuda": "12.6", "arch_list": ["sm_70", "sm_80"]},
               driver_cuda="12.6", driver_version="560.28.03")
    assert "cuda13_dropped_arch" not in codes(st)


def test_cpu_only_torch_and_missing_torch_and_rocm():
    assert "torch_cpu_only" in codes(stack(torch={"cuda": None, "arch_list": []}), "fail")
    assert "torch_missing" in codes(stack(torch={"installed": False}), "fail")
    assert "torch_rocm" in codes(stack(torch={"hip": "6.2"}), "fail")


def test_no_driver_is_a_warning_not_a_failure():
    st = {"smi_present": False, "devices": [], "torch": {"installed": True, "cuda": None}}
    f = cc.evaluate(st)
    assert [x["code"] for x in f] == ["no_nvidia_driver"] and cc.worst(f) == "warn" and cc.check(stack=st)["ok"]


def test_triton_rules():
    assert "triton_missing" in codes(stack(triton=None), "warn")
    assert "triton_too_old_for_blackwell" in codes(stack(triton="3.2.0"), "warn")
    h100 = stack(triton="3.2.0", devices=[_dev(sm=90, name="H100")])
    assert "triton_too_old_for_blackwell" not in codes(h100)


def test_nvml_bindings_and_nvcc():
    assert "nvml_bindings_old" in codes(stack(nvidia_ml_py="12.575.51"), "warn")
    assert "nvml_missing" in codes(stack(nvidia_ml_py=None), "info")
    assert "nvcc_major_mismatch" in codes(stack(nvcc="12.6"), "warn")
    assert "nvcc_minor_mismatch" in codes(stack(nvcc="13.0"), "info")


def test_compute_modes():
    assert "compute_mode_exclusive" in codes(stack(devices=[_dev(mode="Exclusive_Process")]), "warn")
    assert "compute_mode_prohibited" in codes(stack(devices=[_dev(mode="Prohibited")]), "fail")


def test_partially_usable_devices_name_the_good_ones():
    st = stack(devices=[_dev(0), _dev(1)], probe={"available": True, "device_count": 2, "devices": [
        {"torch_index": 0, "ok": False, "error": "x busy or unavailable"}, {"torch_index": 1, "ok": True}]})
    f = {x["code"]: x for x in cc.evaluate(st)}
    assert f["devices_unusable"]["level"] == "fail" and "CUDA_VISIBLE_DEVICES=1" in f["some_devices_usable"]["message"]


def test_torch_sees_no_gpu_unless_hidden_on_purpose():
    probe = {"available": False, "device_count": 0, "devices": []}
    assert "torch_sees_no_gpu" in codes(stack(probe=probe), "fail")
    assert "torch_sees_no_gpu" not in codes(stack(probe=probe, cuda_visible_devices=""))
    assert "torch_sees_no_gpu" not in codes(stack(probe=probe, cuda_visible_devices="-1"))


def test_recommend_wheel_index():
    r = cc.recommend(stack(driver_cuda="13.0", torch={"container_build": False}))
    assert r["torch_index"] == "cu130" and r["torch_index_url"].endswith("/cu130") and "nvidia-ml-py>=13" in r["requirements"]
    assert "triton>=3.3" in r["requirements"]
    assert cc.recommend(stack(driver_cuda="12.9"))["torch_index"] == "cu129"
    assert cc.recommend(stack(driver_cuda="12.8"))["torch_index"] == "cu128"
    assert cc.recommend(stack(driver_cuda="12.6", devices=[_dev(sm=90, name="H100")]))["torch_index"] == "cu126"
    # Blackwell on a driver that cannot run CUDA 12.8: no index, a note says upgrade the driver
    r = cc.recommend(stack(driver_cuda="12.6"))
    assert r["torch_index"] is None and any("driver must be upgraded" in n for n in r["notes"])
    # Volta present: stay on the CUDA 12 line even with a 13.x driver
    r = cc.recommend(stack(driver_cuda="13.4", devices=[_dev(sm=70, name="V100")]))
    assert r["torch_index"] == "cu126"
    # CUDA 12.0-12.5 drivers still get a wheel line through minor-version compatibility
    assert cc.recommend(stack(driver_cuda="12.2", devices=[_dev(sm=90, name="H100")]))["torch_index"] == "cu126"
    assert cc.recommend(stack(driver_cuda="11.8", devices=[_dev(sm=90, name="H100")]))["torch_index"] is None
    assert cc.recommend({"devices": []})["torch_index"] is None


def test_recommend_never_suggests_replacing_a_container_torch():
    r = cc.recommend(stack())
    assert r["reinstall_torch"] is False and any("container build" in n for n in r["notes"])
    r = cc.recommend(stack(torch={"container_build": False, "cuda": "13.0"}, driver_cuda="12.8", driver_version="570.26"))
    assert r["reinstall_torch"] is True and r["torch_index"] == "cu128"


def test_native_driver_lookup_is_conservative_for_unlisted_releases():
    assert cc.native_driver_for((13, 0)) == ((580, 65, 6), "branch", True)
    # NVIDIA documents driver branches for the 13.x releases: 13.1 R590, 13.2 R595, 13.3 R610, 13.4 R615
    assert [cc.native_driver_for((13, m))[0] for m in (1, 2, 3, 4)] == [(590,), (595,), (610,), (615,)]
    assert cc.native_driver_for((12, 7))[0] == (560, 28, 3) and cc.native_driver_for((12, 7))[2] is False
    assert cc.native_driver_for((13, 9))[0] == (615,) and cc.native_driver_for((13, 9))[2] is False
    assert cc.fmt_driver((615,)) == "R615"
    assert cc.native_driver_for((11, 8))[0] == (450, 80, 2) or cc.native_driver_for((11, 8))[0] == cc.DRIVER_FLOOR[11]
    assert cc.native_driver_for((11, 8))[2] is False


def test_assert_cuda_usable_raises_with_diagnosis_and_caches_success(monkeypatch):
    cc._CACHE.clear()
    bad = stack(probe={"available": True, "device_count": 1, "devices": [{"torch_index": 0, "ok": False, "error": "busy or unavailable"}]})
    monkeypatch.setattr(cc, "detect_stack", lambda **k: bad)
    with pytest.raises(cc.CudaUnusable) as ei:
        cc.assert_cuda_usable()
    assert "busy or unavailable" in str(ei.value) and ei.value.findings
    good = stack()
    calls = []
    monkeypatch.setattr(cc, "detect_stack", lambda **k: calls.append(1) or good)
    cc.assert_cuda_usable()
    cc.assert_cuda_usable()
    assert len(calls) == 1  # success cached, failure was not
    cc._CACHE.clear()


def test_detect_stack_never_raises_on_a_cpu_host_and_is_json_serialisable():
    st = cc.detect_stack(probe=False)
    json.dumps(st)
    assert set(st) >= {"driver_version", "driver_cuda", "devices", "torch", "triton"}
    rep = cc.check(stack=st)
    assert rep["worst"] in cc.LEVELS and isinstance(cc.format_report(rep), str)


def test_main_flags(capsys):
    assert cc.main(["--no-probe", "--json"]) in (0, 1)
    assert '"findings"' in capsys.readouterr().out
    assert cc.main(["--torch-index"]) == 0
    capsys.readouterr()
    assert cc.main(["--requirements"]) == 0
    assert "nvidia-ml-py" in capsys.readouterr().out


def test_smi_installed_but_failing_is_a_hard_failure_not_a_cpu_host():
    st = {"smi_present": True, "smi_ok": False, "devices": [], "torch": {"installed": True, "cuda": "13.0"}}
    assert [f["code"] for f in cc.evaluate(st)] == ["nvidia_smi_failed"] and not cc.check(stack=st)["ok"]


def test_rules_only_see_devices_visible_to_the_process():
    mixed = [_dev(0, name="Tesla V100", sm=70), _dev(1, name="NVIDIA A100", sm=80)]
    st = stack(devices=mixed, torch={"cuda": "13.0", "arch_list": ["sm_75", "sm_80", "sm_90"]}, driver_cuda="13.0",
               driver_version="580.65.06", nvcc=None, cuda_visible_devices="1")
    assert "cuda13_dropped_arch" not in codes(st) and cc.worst(cc.evaluate(st)) == "ok"
    st["cuda_visible_devices"] = None  # nothing pinned: the V100 is visible and CUDA 13 cannot run on it
    assert "cuda13_dropped_arch" in codes(st, "fail")
    st["cuda_visible_devices"] = "GPU-abc"  # UUID form: do not guess, keep all devices
    assert "cuda13_dropped_arch" in codes(st, "fail")
    st["cuda_visible_devices"] = "-1"
    assert codes(st) == {"gpus_hidden"}


def test_two_part_driver_string_at_the_floor_is_not_below_it():
    st = stack(driver_version="580.65", driver_cuda="13.0", torch={"cuda": "13.0"}, nvcc=None)
    assert "driver_below_family_floor" not in codes(st)
    st = stack(driver_version="580.64.99", driver_cuda="13.0", torch={"cuda": "13.0"}, nvcc=None)
    assert "driver_below_family_floor" in codes(st, "fail")


def test_relative_index_advice_when_visible_devices_already_set():
    st = stack(devices=[_dev(2), _dev(3)], cuda_visible_devices="2,3", probe={"available": True, "device_count": 2, "devices": [
        {"torch_index": 0, "ok": False, "error": "busy or unavailable"}, {"torch_index": 1, "ok": True}]})
    msg = {f["code"]: f for f in cc.evaluate(st)}["some_devices_usable"]["message"]
    assert "relative to the current CUDA_VISIBLE_DEVICES=2,3" in msg
