"""Model reference resolution, repo-id validation and hardened model_download (huggingface_hub mocked, no network)."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import home  # noqa: E402,F401

from forge.core import modelref  # noqa: E402
from forge.core.context import NullContext  # noqa: E402
from forge.core.errors import ConfigError, NotFound  # noqa: E402
from forge.jobs import builtin  # noqa: E402

HF_CFG = {"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
          "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100}


def _model_dir(d: Path, weights: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(HF_CFG))
    if weights:
        (d / "model.safetensors").write_bytes(b"\0" * 2048)
    return d


def _fake_hub(monkeypatch, *, snapshot, api=None):
    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = snapshot
    fake.HfApi = api or (lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake


class RepositoryNotFoundError(Exception):  # matched by class name, like the real huggingface_hub one
    pass


def test_validate_repo_id_accepts_org_name_and_suggests_for_bare(monkeypatch):
    assert modelref.validate_repo_id("google/gemma-4-E4B") == "google/gemma-4-E4B"

    class Api:
        def list_models(self, search=None, limit=None, sort=None, token=None):
            return [types.SimpleNamespace(id="google/gemma-4-E4B"), types.SimpleNamespace(id="google/gemma-4-E4B-it")]

    _fake_hub(monkeypatch, snapshot=None, api=Api)
    with pytest.raises(ConfigError) as ei:
        modelref.validate_repo_id("gemma-4-E4B")
    assert "org/name" in str(ei.value) and "google/gemma-4-E4B-it" in str(ei.value)
    for bad in ("a/b/c", "../x", "/abs/path", "org/"):
        with pytest.raises(ConfigError):
            modelref.validate_repo_id(bad, suggest=False)


def test_download_config_rejects_bare_name(monkeypatch):
    _fake_hub(monkeypatch, snapshot=None)  # suggestions fail offline -> still a clear error
    with pytest.raises(ValueError, match="org/name"):
        builtin.ModelDownloadConfig(repo_id="gemma-4-E4B")


def test_hf_token_precedence(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "from-alt")
    assert modelref.get_hf_token() == "from-alt"
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert modelref.get_hf_token() == "from-env"
    assert modelref.get_hf_token("explicit") == "explicit"


def test_resolve_order_dir_registry_modelsdir_hf(home, monkeypatch, tmp_path):
    from forge.core.paths import models_dir
    from forge.jobs.modelinfo import register_local_model

    _fake_hub(monkeypatch, snapshot=None)
    d = _model_dir(tmp_path / "somewhere" / "mymodel")
    assert modelref.resolve_model_ref(str(d)).kind == "local"
    row = register_local_model(d, name="my-reg")
    r = modelref.resolve_model_ref("my-reg")  # by name
    assert r.kind == "registry" and r.path == Path(row["info"]["local_path"])
    assert modelref.resolve_model_ref(row["id"]).kind == "registry"  # by id
    assert modelref.resolve_model_ref("MY-REG").kind == "registry"  # case-insensitive
    _model_dir(models_dir() / "plain-dir")
    assert modelref.resolve_model_ref("plain-dir").kind == "models_dir"
    r = modelref.resolve_model_ref("google/gemma-4-E4B")
    assert r.kind == "hf" and r.repo_id == "google/gemma-4-E4B"
    with pytest.raises(NotFound, match="cannot resolve model"):
        modelref.resolve_model_ref("gemma-4-E4B")  # bare unknown name is never sent to the Hub
    with pytest.raises(NotFound):
        modelref.resolve_model_ref("/nope/abs")


def test_registered_but_undownloaded_hf_model_is_clear_error(home):
    from forge.core.util import new_id, now
    from forge.jobs.repo import ModelsRepo

    t = now()
    ModelsRepo().insert(id=new_id("mdl"), name="gemma-4-E4B", source="hf", path_or_repo="google/gemma-4-E4B", revision=None,
                        licence=None, distill_allowed=None, status="remote", info={}, tags=[], notes="", created_at=t, updated_at=t)
    with pytest.raises(NotFound, match="not downloaded"):
        modelref.resolve_model_ref("gemma-4-E4B")


def test_loader_resolve_model_path_uses_registry_and_token(home, monkeypatch, tmp_path):
    from forge.jobs.modelinfo import register_local_model
    from forge.models import loader

    d = _model_dir(tmp_path / "elsewhere" / "m2")
    row = register_local_model(d, name="nice-name")
    assert loader.resolve_model_path("nice-name") == Path(row["info"]["local_path"])
    seen = {}

    def snap(repo, **kw):
        seen.update(repo=repo, **kw)
        return str(d)

    fake = types.ModuleType("huggingface_hub")
    fake.snapshot_download = snap
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "tok")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert loader.resolve_model_path("org/name") == d
    assert seen["repo"] == "org/name" and seen["token"] == "tok"


def test_download_silent_return_of_nonempty_dir_is_an_error(home, monkeypatch):
    """snapshot_download returning an unrelated local_dir without fetching (Hub unreachable) must not count as success."""
    def snap(repo_id=None, revision=None, local_dir=None, **kw):
        return local_dir  # writes nothing, like huggingface_hub's 'Returning existing local_dir' branch

    _fake_hub(monkeypatch, snapshot=snap)
    with pytest.raises(ConfigError, match="no config.json"):
        builtin.run_model_download(builtin.ModelDownloadConfig(repo_id="google/gemma-4-E4B"), NullContext())


def test_download_requires_weights(home, monkeypatch):
    def snap(repo_id=None, revision=None, local_dir=None, **kw):
        (Path(local_dir) / "config.json").write_text(json.dumps(HF_CFG))
        return local_dir

    _fake_hub(monkeypatch, snapshot=snap)
    with pytest.raises(ConfigError, match="no weight files"):
        builtin.run_model_download(builtin.ModelDownloadConfig(repo_id="org/noweights"), NullContext())


def test_download_refuses_foreign_nonempty_target(home, monkeypatch):
    from forge.core.paths import models_dir

    _model_dir(models_dir() / "gemma-4-E4B")  # user's own local copy with the same dir name
    called = []
    _fake_hub(monkeypatch, snapshot=lambda **kw: called.append(1))
    with pytest.raises(ConfigError, match="already exists"):
        builtin.run_model_download(builtin.ModelDownloadConfig(repo_id="google/gemma-4-E4B"), NullContext())
    assert not called


def test_repo_not_found_in_metadata_probe_is_fatal_with_hint(home, monkeypatch):
    class Api:
        def model_info(self, *a, **k):
            raise RepositoryNotFoundError("404")

        def list_models(self, **k):
            return [types.SimpleNamespace(id="google/gemma-4-E4B")]

    called = []
    _fake_hub(monkeypatch, snapshot=lambda **kw: called.append(1), api=Api)
    with pytest.raises(ConfigError, match="not found") as ei:
        builtin.run_model_download(builtin.ModelDownloadConfig(repo_id="org/missing"), NullContext())
    assert "google/gemma-4-E4B" in str(ei.value) and not called


def test_failed_download_marks_registered_row_failed(home, monkeypatch):
    from forge.core.util import new_id, now
    from forge.jobs.repo import ModelsRepo

    t = now()
    mid = new_id("mdl")
    ModelsRepo().insert(id=mid, name="x", source="hf", path_or_repo="org/x", revision=None, licence=None, distill_allowed=None,
                        status="downloading", info={}, tags=[], notes="", created_at=t, updated_at=t)
    _fake_hub(monkeypatch, snapshot=lambda repo_id=None, revision=None, local_dir=None, **kw: local_dir)
    with pytest.raises(ConfigError):
        builtin.run_model_download(builtin.ModelDownloadConfig(repo_id="org/x", model_id=mid), NullContext())
    row = ModelsRepo().require(mid)
    assert row["status"] == "failed" and "config.json" in row["info"]["error"]


def test_inspect_surfaces_unsupported_architecture(tmp_path):
    from forge.jobs.modelinfo import inspect_model_dir

    d = tmp_path / "g4"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({
        "model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100}}))
    (d / "model.safetensors").write_bytes(b"\0" * 16)
    info = inspect_model_dir(d)
    if info["forge_supported"]:
        pytest.skip("gemma4 support was added by the models owner")
    assert "gemma4" in info["unsupported_reason"] and "Gemma4ForConditionalGeneration" in info["unsupported_reason"]
    assert info["warnings"][0] == info["unsupported_reason"]
