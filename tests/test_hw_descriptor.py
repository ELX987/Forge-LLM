"""Descriptor construction from the catalogue, fingerprints, ridge points, and the sqlite/yaml store."""
from __future__ import annotations

import pytest

from forge.core.db import Database
from forge.core.errors import NotFound
from forge.core.schemas import ChipDescriptor, MeasuredPair, Provenance
from forge.hw import chips
from forge.hw import descriptor as dm
from forge.hw import descriptor_store as ds


@pytest.fixture()
def db(tmp_path):
    d = Database(tmp_path / "hw.db")
    d.migrate()
    yield d
    d.close()


def test_h100_descriptor_matches_plan_appendix_b():
    d = dm.descriptor_from_chip("h100-sxm5-80g")
    assert d.sm == 90 and d.compute_capability == "9.0" and d.compute_units == 132 and d.family == "hopper"
    assert d.memory.hbm_gb == 80 and d.memory.hbm_bw_gbs.datasheet == 3350 and d.memory.hbm_bw_gbs.measured is None
    assert d.memory.l2_mb == 50 and d.on_chip.smem_per_cu_kib == 227
    assert d.on_chip.async_copy == ["cp_async", "bulk_async"] and d.on_chip.cluster_launch
    assert d.sparsity.two_four
    assert d.interconnect.intra_gbs == 450 and d.interconnect.topology == "switch"
    assert d.power_limit_w == 700 and d.tier == "A"
    assert d.supports("bf16") and d.supports("fp16") and d.supports("fp8") and d.supports("fp8_e4m3")
    assert d.supports("int8") and d.supports("tf32") and not d.supports("fp4") and not d.supports("nvfp4")
    mu = d.matrix_unit_for("bf16")
    assert mu.dtype == "fp16|bf16" and mu.peak_tflops.datasheet == 989.0 and mu.atom == [64, 0, 16]
    # ridge = 989e12 / 3350e9 = 295 (Plan example) ; fp8 = 590
    assert d.derived.ridge_flop_per_byte["fp16"] == pytest.approx(295.2, abs=0.2)
    assert d.derived.ridge_flop_per_byte["bf16"] == d.derived.ridge_flop_per_byte["fp16"]
    assert d.derived.ridge_flop_per_byte["fp8"] == pytest.approx(590.7, abs=0.5)
    # provenance: datasheet numbers are tagged vendor, never measured
    assert d.probe_provenance["memory.hbm_bw_gbs"] == Provenance.vendor
    assert Provenance.measured not in d.probe_provenance.values()
    assert d.probe_provenance["derived.ridge_flop_per_byte"] == Provenance.derived


def test_blackwell_descriptor_has_block_scaled_fp4():
    d = dm.descriptor_from_chip("b200")
    mu = d.matrix_unit_for("fp4")
    assert mu is not None and d.supports("nvfp4") and d.supports("mxfp4")
    assert mu.block_scaled["block"] == 16 and mu.block_scaled["scale"] == "e4m3"
    assert mu.block_scaled["scale_atom"] == [128, 4] and mu.block_scaled["mxfp4"]["block"] == 32
    assert d.interconnect.intra_gbs == 900 and d.tier == "A"
    fp8 = d.matrix_unit_for("fp8")
    assert "mxfp8" in fp8.block_scaled
    assert dm.descriptor_from_chip("b200").derived.ridge_flop_per_byte["fp4"] == pytest.approx(9000e12 / 8000e9, rel=1e-3)


def test_old_and_consumer_chips():
    v = dm.descriptor_from_chip("v100-sxm2-32g")
    assert v.tier == "C" and v.supports("fp16") and not v.supports("bf16") and not v.sparsity.two_four
    assert v.interconnect.topology == "nvlink_mesh"
    t = dm.descriptor_from_chip("t4")
    assert t.supports("int8") and t.supports("int4") and not t.supports("bf16") and t.tier == "C"
    a = dm.descriptor_from_chip("rtx-3090")
    assert a.tier == "B" and a.supports("bf16") and not a.supports("fp8") and a.interconnect.topology == "bridge"
    assert a.matrix_unit_for("fp16").peak_tflops.src.startswith("derived")
    assert a.probe_provenance["matrix_units.fp16"] == Provenance.derived
    ada = dm.descriptor_from_chip("rtx-4090")
    assert ada.supports("fp8") and ada.tier == "B" and ada.matrix_unit_for("fp8").atom == [16, 8, 32]
    assert ada.interconnect.topology == "pcie" and ada.interconnect.intra_gbs == pytest.approx(31.5)
    nt = dm.descriptor_from_chip("gtx-16xx")
    assert nt.matrix_units == [] and nt.tier == "C"
    orin = dm.descriptor_from_chip("jetson-agx-orin-64g")
    assert orin.sm == 87 and orin.interconnect.topology is None and orin.probe_provenance["name"] == Provenance.untested
    gb10 = dm.descriptor_from_chip("gb10")
    assert gb10.sm == 121 and gb10.supports("fp4")


@pytest.mark.parametrize("key", sorted(chips.KNOWN_CHIPS))
def test_every_catalogue_chip_builds_a_valid_descriptor(key):
    d = dm.descriptor_from_chip(key)
    again = ChipDescriptor.model_validate(d.model_dump(mode="json"))
    assert again == d
    assert d.tier in ("A", "B", "C") and d.id and d.fingerprint
    for k, v in d.derived.ridge_flop_per_byte.items():
        assert v > 0, (key, k)


def test_approximate_numbers_are_not_tagged_vendor():
    d = dm.descriptor_from_chip("a10g")
    assert d.probe_provenance["memory.hbm_bw_gbs"] == Provenance.untested
    assert d.probe_provenance["matrix_units.fp16"] == Provenance.untested
    assert d.matrix_unit_for("fp16").peak_tflops.src == "approximate"


def test_fingerprint_and_id():
    a = dm.chip_fingerprint("NVIDIA H100 80GB HBM3", 90, 132, 81559)
    b = dm.chip_fingerprint("NVIDIA H100 80GB HBM3", 90, 132, 81560)  # small reserved-memory jitter
    c = dm.chip_fingerprint("NVIDIA H100 PCIe", 90, 114, 81559)
    assert a == b and a != c and a.startswith(("blake3:", "b2:"))
    assert dm.chip_fingerprint("x", 90, 1, 1000) != dm.chip_fingerprint("x", 90, 1, 1000, vendor="cpu")
    i = dm.make_id("NVIDIA GeForce RTX 4090", a, 3)
    assert i.startswith("rtx-4090-") and i.endswith("-r3")


def test_ridge_prefers_measured_and_refresh():
    d = dm.descriptor_from_chip("a100-sxm4-80g")
    ds_ridge = dm.ridge_points(d, datasheet_only=True)
    d.memory.hbm_bw_gbs = MeasuredPair(datasheet=2039, measured=1800, src="microbench/stream_copy_bw")
    mu = d.matrix_unit_for("bf16")
    mu.peak_tflops = MeasuredPair(datasheet=312, measured=250, src="microbench/mma_bf16")
    dm.refresh_derived(d)
    eff = d.derived.ridge_flop_per_byte
    assert eff["bf16"] == pytest.approx(250e12 / 1800e9, rel=1e-3)
    assert eff["bf16"] < ds_ridge["bf16"]
    assert eff["tf32"] == pytest.approx(156e12 / 1800e9, rel=1e-3)  # unmeasured unit uses datasheet peak, measured bw
    assert dm.ridge_points(ChipDescriptor(id="x")) == {}


def test_summarize():
    d = dm.descriptor_from_chip("l40s")
    s = dm.summarize(d)
    assert s["sm"] == 89 and "fp8" in s["dtypes"] and s["n_measured"] == 0


# ---------------------------------------------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------------------------------------------
def test_store_roundtrip_list_delete(db):
    d = dm.descriptor_from_chip("h100-sxm5-80g")
    d.probe_provenance["memory.hbm_bw_gbs"] = Provenance.measured
    rep = {"features": {"fp8": True}, "benchmarks": {"x": 1}}
    assert ds.save_descriptor(d, report=rep, db=db, write_files=False) == d.id
    assert ds.load_descriptor(d.id, db=db) == d
    assert ds.load_report(d.id, db=db) == rep
    rows = ds.list_descriptors(db=db)
    assert len(rows) == 1 and rows[0]["id"] == d.id and rows[0]["n_measured"] == 1 and rows[0]["sm"] == 90
    assert ds.latest_for_fingerprint(d.fingerprint, db=db).id == d.id
    assert ds.latest_for_fingerprint(d.fingerprint, db=db, measured_only=True).id == d.id
    assert ds.latest_report_for_fingerprint(d.fingerprint, db=db)["features"]["fp8"] is True
    assert ds.next_revision(d.fingerprint, db=db) == 2
    # upsert keeps created_at, replaces content
    d2 = d.model_copy(update={"name": "renamed"})
    ds.save_descriptor(d2, db=db, write_files=False)
    assert len(ds.list_descriptors(db=db)) == 1 and ds.load_descriptor(d.id, db=db).name == "renamed"
    assert ds.list_descriptors(db=db, fingerprint="nope") == []
    assert ds.delete_descriptor(d.id, db=db) and not ds.delete_descriptor(d.id, db=db)
    with pytest.raises(NotFound):
        ds.load_descriptor(d.id, db=db)
    with pytest.raises(NotFound):
        ds.load_report("missing", db=db)
    assert ds.latest_for_fingerprint("none", db=db) is None


def test_store_newest_first_and_measured_filter(db):
    a = dm.descriptor_from_chip("t4", id="t4-a")
    b = dm.descriptor_from_chip("t4", id="t4-b")
    b.probe_provenance["memory.hbm_bw_gbs"] = Provenance.measured
    ds.save_descriptor(a, db=db, write_files=False)
    ds.save_descriptor(b, db=db, write_files=False)
    assert ds.list_descriptors(db=db)[0]["id"] == "t4-b"
    assert ds.latest_for_fingerprint(a.fingerprint, db=db, measured_only=True).id == "t4-b"
    assert ds.latest_for_fingerprint(a.fingerprint, db=db).id == "t4-b"


def test_yaml_roundtrip_and_files(db, tmp_path):
    d = dm.descriptor_from_chip("b200")
    text = ds.descriptor_to_yaml(d)
    assert "fp4_e2m1" in text and "block_scaled" in text
    assert ds.descriptor_from_yaml(text) == d
    assert ds.descriptor_from_yaml(d.model_dump_json()) == d  # JSON is accepted too
    with pytest.raises(ValueError):
        ds.descriptor_from_yaml("- just\n- a list")
    ds.save_descriptor(d, report={"a": 1}, db=db)  # writes files under FORGE_HOME
    base = ds.descriptors_dir()
    assert (base / f"{d.id}.yaml").exists() and (base / f"{d.id}.report.json").exists()
    assert ds.load_descriptor_file(base / f"{d.id}.yaml") == d
    ds.delete_descriptor(d.id, db=db)
    assert not (base / f"{d.id}.yaml").exists()


def test_store_uses_default_database():
    d = dm.descriptor_from_chip("l4", id="default-db-l4")
    ds.save_descriptor(d, write_files=False)
    assert ds.load_descriptor("default-db-l4").sm == 89
    ds.delete_descriptor("default-db-l4")
