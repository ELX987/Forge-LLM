from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from forge.core import schemas as S
from forge.core.hashing import canonical_json, hash_json


def _recipe() -> S.Recipe:
    return S.Recipe(
        name="w4a16-fp8kv",
        inputs={"model": "sha256:" + "a" * 64, "calib": "mix_v3"},
        stages=[
            S.StageSpec(name="calibrate", params={"set": "mix_v3", "n_seqs": 512}),
            S.StageSpec(name="ptq", params={"method": "gptq", "bits": 4, "group": 128}),
        ],
        baselines=["fp8_w8a8"],
        budget={"gpu_hours": 40},
        policy={"licence_check": True},
        seeds=[0, 1],
    )


GOLDEN_RECIPE_JSON = (
    '{"baselines":["fp8_w8a8"],"budget":{"gpu_hours":40},"inputs":{"calib":"mix_v3","model":"sha256:'
    + "a" * 64
    + '"},"name":"w4a16-fp8kv","policy":{"licence_check":true},"seeds":[0,1],"stages":[{"name":"calibrate",'
    '"params":{"n_seqs":512,"set":"mix_v3"}},{"name":"ptq","params":{"bits":4,"group":128,"method":"gptq"}}]}'
)


def test_recipe_canonical_golden_and_hash_determinism():
    r = _recipe()
    assert canonical_json(r.canonical()) == GOLDEN_RECIPE_JSON
    assert r.hash() == _recipe().hash()
    # key order of params must not matter
    r2 = _recipe()
    r2.stages[1].params = {"group": 128, "method": "gptq", "bits": 4}
    assert r2.hash() == r.hash()
    # any content change must
    r3 = _recipe()
    r3.stages[1].params["bits"] = 8
    assert r3.hash() != r.hash()
    r4 = _recipe()
    r4.seeds = [0]
    assert r4.hash() != r.hash()
    assert r.hash() == hash_json(json.loads(GOLDEN_RECIPE_JSON))


def test_recipe_forbids_extra_and_empty_names():
    with pytest.raises(ValidationError):
        S.Recipe(name="x", bogus=1)
    with pytest.raises(ValidationError):
        S.StageSpec(name="")


def test_manifest_hash_computed_stable_and_excludes_created_at():
    t = S.TensorEntry(name="w", dtype="uint8", shape=[4, 4], hash="b2:" + "0" * 64, layout_id="L1", role="payload")
    m1 = S.ArtifactManifest(kind="model", model_cfg={"hidden_size": 64}, tensors=[t], created_at=1.0)
    m2 = S.ArtifactManifest(kind="model", model_cfg={"hidden_size": 64}, tensors=[t], created_at=999.0)
    assert m1.hash == m2.hash
    m3 = S.ArtifactManifest(kind="model", model_cfg={"hidden_size": 65}, tensors=[t])
    assert m3.hash != m1.hash
    # contract alias
    m4 = S.ArtifactManifest.model_validate({"kind": "model", "model_config": {"hidden_size": 64}, "tensors": [t.model_dump()]})
    assert m4.hash == m1.hash
    # roundtrip through dump (which includes the computed hash)
    dumped = m1.model_dump(mode="json")
    assert dumped["hash"] == m1.hash
    again = S.ArtifactManifest.model_validate(dumped)
    assert again.hash == m1.hash and again.created_at == 1.0


def test_manifest_golden_content():
    m = S.ArtifactManifest(kind="adapter", parents=["p"], created_at=5.0)
    assert canonical_json(m.content_dict()) == (
        '{"calib_hash":null,"eval_report_hashes":[],"extra":{},"hardware_families":[],"kind":"adapter",'
        '"licence":null,"model_cfg":{},"parents":["p"],"recipe_hash":null,"tensors":[]}'
    )


def test_tensor_role_validation():
    with pytest.raises(ValidationError):
        S.TensorEntry(name="a", dtype="u8", shape=[1], hash="h", role="nonsense")


def test_metric_value_ci_validation():
    S.MetricValue(value=1.0, ci_low=0.5, ci_high=1.5, n=10, provenance=S.Provenance.measured)
    with pytest.raises(ValidationError):
        S.MetricValue(value=1.0, ci_low=2.0, ci_high=1.0)
    assert S.MetricValue(value=1).provenance == S.Provenance.measured
    assert {p.value for p in S.Provenance} == {
        "measured", "reproduced", "paper_reported", "derived", "vendor", "target", "untested"
    }


def test_chip_descriptor_supports_and_roundtrip():
    chip = S.ChipDescriptor(
        id="h100",
        sm=90,
        compute_capability="9.0",
        compute_units=132,
        memory=S.MemoryInfo(hbm_gb=80, hbm_bw_gbs=S.MeasuredPair(datasheet=3350, measured=2900)),
        matrix_units=[
            S.MatrixUnit(dtype="fp16|bf16", atom=[16, 8, 16], peak_tflops=S.MeasuredPair(datasheet=989)),
            S.MatrixUnit(dtype="fp8_e4m3", peak_tflops=S.MeasuredPair(datasheet=1979), block_scaled=False),
            S.MatrixUnit(dtype="fp4_e2m1", block_scaled={"block": 16, "scale": "e4m3"}),
        ],
        derived=S.DerivedInfo(ridge_flop_per_byte={"fp16": 280.0}),
        tier="A",
        probe_provenance={"memory.hbm_bw_gbs": S.Provenance.measured},
    )
    assert chip.supports("bf16") and chip.supports("fp16") and chip.supports("bfloat16")
    assert chip.supports("fp8") and chip.supports("fp8_e4m3") and not chip.supports("fp8_e5m2")
    assert chip.supports("nvfp4") and chip.supports("fp4")
    assert not chip.supports("int8")
    assert chip.memory.hbm_bw_gbs.best == 2900
    again = S.ChipDescriptor.model_validate(json.loads(chip.model_dump_json()))
    assert again == chip
    assert not S.ChipDescriptor(id="cpu").supports("bf16")
    with pytest.raises(ValidationError):
        S.ChipDescriptor(id="x", tier="Z")


def test_blockscale_required_for_nvfp4_match():
    chip = S.ChipDescriptor(id="x", matrix_units=[S.MatrixUnit(dtype="fp4_e2m1")])
    assert chip.supports("fp4") and not chip.supports("nvfp4")


def test_job_models_defaults_and_status():
    spec = S.JobSpec(type="train", params={"a": 1})
    rec = S.JobRecord(spec=spec)
    assert rec.status == S.JobStatus.queued and rec.id.startswith("job_") and rec.progress == 0
    assert S.JobStatus.failed.terminal and not S.JobStatus.running.terminal
    with pytest.raises(ValidationError):
        S.JobRecord(spec=spec, progress=1.5)
    again = S.JobRecord.model_validate_json(rec.model_dump_json())
    assert again == rec
    assert S.JobStatus("succeeded") is S.JobStatus.succeeded


def test_eval_report_and_gate_roundtrip():
    rep = S.EvalReport(
        artifact_hash="b2:" + "1" * 64,
        metrics={"kl_p99": S.MetricValue(value=0.01, n=100)},
        gate=S.GateVerdict(gate_id="G-W4-v2", verdict="pass", reasons=["ok"]),
        noise_floor={"kl_p99": 0.002},
    )
    assert S.EvalReport.model_validate(rep.model_dump(mode="json")) == rep
    with pytest.raises(ValidationError):
        S.GateVerdict(gate_id="g", verdict="maybe")


def test_claim_model():
    c = S.Claim(id="P1-001", source_paper="P1", statement="W4 g128 is near-lossless")
    assert c.status == "untested" and c.prior == "untrusted"
    with pytest.raises(ValidationError):
        S.Claim(id="x", source_paper="P1", statement="s", status="done")


def test_dataset_manifest_total_tokens_filled():
    d = S.DatasetManifest(shards=[{"hash": "h1", "tokens": 10, "docs": 2, "offsets": "o"}, {"hash": "h2", "tokens": 5, "docs": 1}])
    assert d.total_tokens == 15
    assert d.shards[0].model_dump()["offsets"] == "o"
    assert d.content_hash() == S.DatasetManifest.model_validate(d.model_dump()).content_hash()


def test_model_ref_workload_deployment():
    S.ModelRef(id="m", source="hf", path_or_repo="org/model", params_b=8)
    with pytest.raises(ValidationError):
        S.ModelRef(id="m", source="ftp")
    w = S.Workload()
    assert w.context_lengths == [4096] and w.prefix_reuse == 0
    with pytest.raises(ValidationError):
        S.Workload(prefix_reuse=1.5)
    d = S.Deployment(artifact_ref="m", port=8000)
    assert d.id.startswith("dep_") and d.status == "created"


def test_every_field_has_a_description():
    """UI forms are generated from the schema: every field of every model must be described."""
    import inspect

    from pydantic import BaseModel

    missing = []
    for name, cls in inspect.getmembers(S, inspect.isclass):
        if issubclass(cls, BaseModel) and cls.__module__ == S.__name__ and cls is not S._Base:
            for fname, f in cls.model_fields.items():
                if not f.description:
                    missing.append(f"{name}.{fname}")
    assert not missing, missing
