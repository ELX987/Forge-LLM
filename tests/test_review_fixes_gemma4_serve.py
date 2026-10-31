"""Regression tests for the adversarial review of the Gemma 4 / LFM2 / dataset / model-resolve work. CPU only, hermetic."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home  # noqa: E402,F401
from test_models_gemma4 import _model as _g4_model  # noqa: E402
from test_models_gemma4 import _text_cfg  # noqa: E402
from test_models_lfm2 import _cfg as _lfm_cfg  # noqa: E402
from test_models_lfm2 import _engine as _lfm_engine  # noqa: E402
from test_models_lfm2 import _model as _lfm_model  # noqa: E402

from forge.core.errors import ConfigError, NotFound  # noqa: E402
from forge.models import ModelConfig  # noqa: E402

API = "/api/v1"


def _g4():
    cfg = ModelConfig.from_hf_config(_text_cfg())
    return _g4_model(cfg), cfg


# ---- 1. serving Gemma 4 is refused, softcap applied -------------------------------------------------
def test_model_adapter_refuses_gemma4():
    from forge.serve.runner import ModelAdapter

    m, cfg = _g4()
    with pytest.raises(ConfigError, match="Gemma 4.*not supported yet"):
        ModelAdapter(m, cfg)


def test_model_adapter_refuses_each_gemma4_feature():
    from forge.serve.runner import check_servable

    for kw in ({"sandwich_norm": True}, {"per_layer_input_size": 8}, {"num_kv_shared_layers": 2}):
        ns = types.SimpleNamespace(hidden_size=8, num_heads=2, num_layers=4, **kw)
        with pytest.raises(ConfigError):
            check_servable(ns)
    check_servable(types.SimpleNamespace(hidden_size=8, num_heads=2, num_layers=4))  # plain config passes


def test_adapter_logits_apply_softcap():
    from forge.models.tiny import build_tiny_model
    from forge.serve.runner import ModelAdapter

    m = build_tiny_model(1).eval()
    cfg = m.config.model_copy(update={"final_logit_softcapping": 5.0})
    ad = ModelAdapter(m, cfg)
    h = torch.randn(3, cfg.hidden_size) * 4
    raw = ModelAdapter(m, m.config).logits(h)
    got = ad.logits(h)
    assert torch.allclose(got, torch.tanh(raw / 5.0) * 5.0, atol=1e-5) and got.abs().max() <= 5.0


def test_deployment_create_surfaces_gemma4_error(client, tmp_path):
    d = tmp_path / "g4"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({**_text_cfg(), "model_type": "gemma4_text"}))
    r = client.post(f"{API}/deployments", json={"artifact_ref": str(d)})
    assert r.status_code == 400 and "Gemma 4" in r.text


# ---- 2. lfm2 conv state is not advanced twice on kernel fallback --------------------------------------
def test_kernel_decode_fallback_does_not_double_advance_conv_state():
    from forge.serve.runner import SeqInput

    m = _lfm_model(_lfm_cfg())
    e = _lfm_engine(m)
    p = list(range(3, 15))
    t = 21
    e.manager.allocate("a", p + [t])
    bt = e.manager.block_table("a")
    r = e.runner
    r.forward([SeqInput(p, 0, bt, state_key="a")])

    def boom(*_a, **_k):
        raise RuntimeError("forced kernel failure")

    r._kernel_ok = True
    r._kernel_attention_fn = lambda _bt, _lens: boom  # fails after the first conv layer already wrote its state
    out = r.forward([SeqInput([t], len(p), bt, state_key="a")]).logits[0][-1]
    assert r._kernel_ok is False and r.stats["reference_steps"] >= 1
    ref = m(torch.tensor([p + [t]])).logits[0, -1]
    assert torch.allclose(out, ref, atol=1e-4)


# ---- 3. compress guards ---------------------------------------------------------------------------------
def test_prune_rotation_guard_refuses_gemma4():
    from forge.compress.modelutil import require_attention_only

    m, _ = _g4()
    with pytest.raises(ConfigError, match="Gemma 4"):
        require_attention_only(m, "structured pruning")
    from forge.models.tiny import build_tiny_model

    require_attention_only(build_tiny_model(1), "structured pruning")  # llama-style still fine


def test_kv_report_handles_kv_shared_layers_and_per_layer_dims():
    from forge.compress.kvquant import model_kv_report

    m, cfg = _g4()
    tokens = torch.randint(0, 128, (2, 16))
    rep = model_kv_report(m, tokens, max_layers=8)
    owners = [i for i in range(cfg.num_layers) if not cfg.is_kv_shared_layer(i)]
    assert set(rep["layers"]) <= set(owners) and rep["rows"]


def test_ptq_rtn_works_on_gemma4_but_calibrated_methods_refuse():
    from forge.compress.ptq import quantize_model

    m, _ = _g4()
    with pytest.raises(ConfigError, match="Gemma 4"):
        quantize_model(m, method="gptq", bits=4)
    out = quantize_model(m, method="rtn", bits=8, group_size=16)
    assert out.report.bytes_after <= out.report.bytes_before


# ---- 4. source-preview confinement ---------------------------------------------------------------------
def test_source_preview_rejects_paths_outside_roots_and_non_data_files(client, home, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP SECRET")
    r = client.post(f"{API}/datasets/source-preview", json={"source": {"path": str(secret), "kind": "txt"}})
    assert r.status_code == 400 and "TOP SECRET" not in r.text
    ds = home / "datasets"
    ds.mkdir(parents=True, exist_ok=True)
    weird = ds / "id_rsa"
    weird.write_text("KEY")
    r = client.post(f"{API}/datasets/source-preview", json={"source": {"path": str(weird), "kind": "txt"}})
    assert r.status_code == 400 and "KEY" not in r.text
    ok = ds / "a.txt"
    ok.write_text("hello\n")
    assert client.post(f"{API}/datasets/source-preview", json={"source": {"path": str(ok), "kind": "txt"}}).status_code == 200


def test_source_preview_allows_registered_dataset_path(client, home, tmp_path):
    d = tmp_path / "elsewhere"
    d.mkdir()
    f = d / "x.jsonl"
    f.write_text(json.dumps({"text": "doc"}) + "\n")
    assert client.post(f"{API}/datasets/source-preview", json={"source": {"path": str(f)}}).status_code == 400
    r = client.post(f"{API}/datasets", json={"name": "ext", "path": str(d)})
    assert r.status_code in (200, 201), r.text
    assert client.post(f"{API}/datasets/source-preview", json={"source": {"path": str(f)}}).status_code == 200


# ---- 5. registry lookup + resolve order ---------------------------------------------------------------
def _insert(name, *, status, path=None, created, source="local"):
    from forge.core.util import new_id
    from forge.jobs.repo import ModelsRepo

    mid = new_id("mdl")
    ModelsRepo().insert(id=mid, name=name, source=source, path_or_repo=str(path or ""), revision=None, licence=None,
                        distill_allowed=None, status=status, info={"local_path": str(path)} if path else {}, tags=[], notes="",
                        created_at=created, updated_at=created)
    return mid


def test_find_by_name_prefers_ready_with_local_path_then_newest(home, tmp_path):
    from forge.jobs.repo import ModelsRepo

    d = tmp_path / "real"
    d.mkdir()
    ok = _insert("dup", status="ready", path=d, created=2.0)
    _insert("dup", status="failed", path=d, created=3.0)
    _insert("dup", status="ready", path=tmp_path / "gone", created=4.0)
    assert ModelsRepo().find_by_name("dup")["id"] == ok
    newest = _insert("only-failed", status="failed", created=9.0)
    _insert("only-failed", status="failed", created=1.0)
    assert ModelsRepo().find_by_name("only-failed")["id"] == newest


def test_resolve_registry_beats_cwd_dir_and_db_errors_surface(home, tmp_path, monkeypatch):
    from forge.core import modelref

    reg = tmp_path / "registered"
    reg.mkdir()
    _insert("shadowed", status="ready", path=reg, created=1.0)
    cwd = tmp_path / "cwd"
    (cwd / "shadowed").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    res = modelref.resolve_model_ref("shadowed")
    assert res.kind == "registry" and res.path == reg
    assert modelref.resolve_model_ref("./shadowed").kind == "local"  # an explicit path still means the directory
    (cwd / "plain").mkdir()
    assert modelref.resolve_model_ref("plain").kind == "local"

    class Boom:
        def get(self, _):
            raise RuntimeError("database is locked")

    with pytest.raises(RuntimeError, match="locked"):
        modelref.resolve_model_ref("anything", repo=Boom())

    with pytest.raises(NotFound):
        modelref.resolve_model_ref("./missing/dir")


# ---- 6. unverified unified variants ---------------------------------------------------------------------
def test_gemma4_unified_is_refused():
    for top in ({"model_type": "gemma4_unified", "text_config": _text_cfg()},
                {**_text_cfg(), "model_type": "gemma4_unified_text"},
                {"architectures": ["Gemma4UnifiedForConditionalGeneration"], "text_config": _text_cfg()}):
        with pytest.raises(ConfigError, match="unverified"):
            ModelConfig.from_hf_config(top)


# ---- 7. capacity uses per-layer KV ------------------------------------------------------------------------
def test_capacity_modelspec_uses_per_layer_kv_bytes():
    from forge.serve.capacity import ModelSpec

    cfg = ModelConfig.from_hf_config(_text_cfg())
    spec = ModelSpec.from_any(cfg)
    assert spec.kv_bytes_per_token("bf16") == pytest.approx(cfg.kv_bytes_per_token(16))
    naive = 2.0 * cfg.num_layers * cfg.kv_heads * cfg.head_dim * 2
    assert spec.kv_bytes_per_token("bf16") != naive


# ---- 8. hf hints + warnings sink ----------------------------------------------------------------------------
def test_hf_hint_classifies_by_status_and_class_not_substring():
    from forge.train.data import _hf_hint

    class Resp:
        status_code = 403

    class HttpErr(Exception):
        response = Resp()

    assert "authentication" in _hf_hint("o/n", HttpErr("nope"), None)

    class GatedRepoError(Exception):
        pass

    assert "authentication" in _hf_hint("o/n", GatedRepoError("x"), None)
    # digits / the word "gated" inside an unrelated message no longer misclassify
    h = _hf_hint("o/n", ValueError("parsed 401 rows of the gated column"), None)
    assert h.startswith("could not load")

    class NF(Exception):
        status_code = 404

    assert "could not be found" in _hf_hint("o/n", NF("m"), None)


def test_hf_token_comes_from_modelref():
    from forge.train import data

    assert data.get_hf_token.__module__ == "forge.core.modelref"


def test_resolution_warnings_reach_job_log():
    from forge.train import data

    got: list[tuple] = []
    ctx = types.SimpleNamespace(log=lambda m, level="info": got.append((level, m)))
    tok = data._WARN_SINK.set(lambda m: ctx.log(m, "warning"))
    try:
        data._emit_warning("hello warn")
    finally:
        data._WARN_SINK.reset(tok)
    assert got == [("warning", "hello warn")]


# ---- 9. download guards ---------------------------------------------------------------------------------------
def test_dir_bytes_skips_hf_cache(tmp_path):
    from forge.jobs.builtin import _dir_bytes

    (tmp_path / "a.bin").write_bytes(b"0" * 100)
    (tmp_path / ".cache" / "huggingface").mkdir(parents=True)
    (tmp_path / ".cache" / "huggingface" / "x.incomplete").write_bytes(b"0" * 5000)
    assert _dir_bytes(tmp_path) == 100


def test_download_guard_applies_with_model_id_and_ready_row_not_demoted(home, monkeypatch):
    from forge.core.context import NullContext
    from forge.core.paths import models_dir
    from forge.jobs import builtin
    from forge.jobs.repo import ModelsRepo

    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = lambda **kw: kw["local_dir"]
    fake.HfApi = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    target = models_dir() / "foreign"
    target.mkdir(parents=True)
    (target / "precious.txt").write_text("mine")
    mid = _insert("foreign", status="ready", path=target, created=1.0)
    with pytest.raises(ConfigError, match="already exists"):
        builtin.run_model_download(builtin.ModelDownloadConfig(repo_id="org/foreign", local_name="foreign", model_id=mid), NullContext())
    assert (target / "precious.txt").read_text() == "mine"
    assert ModelsRepo().require(mid)["status"] == "ready"  # the failed re-download did not demote a usable model
