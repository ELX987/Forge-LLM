"""Regression tests for the review of KV-pool sizing, kernel failure memoisation, kernel-layout conversion, artifact KV schemes, the deployment form and
model-reference resolution. CPU only, hermetic."""
from __future__ import annotations

import gc
import logging
import shutil
import sys
import weakref
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import client, home  # noqa: E402,F401
from test_compress_recipes import _job  # noqa: E402

from forge.core.context import NullContext  # noqa: E402
from forge.core.errors import NotFound  # noqa: E402
from forge.serve.config import ServeConfig  # noqa: E402
from forge.serve.memory import GB, artifact_kv_choice, draft_pool_blocks, plan_kv_memory  # noqa: E402

API = "/api/v1"
PER_BLOCK_BF16 = 2 * 40 * 16 * 8 * 128 * 2  # Qwen3-14B, 16-token block


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Path:
    from forge.compress.run import run_compress

    out = tmp_path_factory.mktemp("rfs") / "artifact"
    run_compress(_job(save_dir=str(out)), NullContext())
    return out


def _plan(**kw):
    base = dict(device="cuda", bytes_per_block=PER_BLOCK_BF16, block_size=16, weight_bytes=int(10.08 * GB), packed_weight_bytes=int(8.5 * GB),
                gpu_mem_util=0.85, activation_reserve_gb=1.0, max_num_seqs=64, max_model_len=4096, total_bytes=int(102.6 * GB),
                free_bytes=int(102 * GB), allocated_bytes=int(10.08 * GB))
    base.update(kw)
    return plan_kv_memory(**base)


# ---- 1. default KV pool ------------------------------------------------------------------------------
def test_defaults_do_not_reserve_a_43gb_pool():
    sc = ServeConfig()
    assert sc.max_num_seqs == 16 and sc.kv_pool_cap_tokens == 32768
    p = _plan(max_num_seqs=sc.max_num_seqs, kv_pool_cap_tokens=sc.kv_pool_cap_tokens)
    assert p.limited_by == "workload_cap" and p.num_blocks == 2048 and p.kv_pool_bytes / GB < 6
    assert any("capped" in w for w in p.warnings)
    assert "x weights" in p.summary()
    # 0 = uncapped (explicit opt-in) and a cap never drops below one max-length sequence
    assert _plan(kv_pool_cap_tokens=0).num_blocks == 64 * 256
    assert _plan(kv_pool_cap_tokens=16, max_model_len=4096).num_blocks == 256


def test_warns_when_pool_exceeds_twice_the_weights():
    p = _plan(kv_pool_cap_tokens=0, weight_bytes=int(1 * GB), allocated_bytes=int(1 * GB))
    assert any("2x the model weights" in w for w in p.warnings)
    assert not any("2x the model weights" in w for w in _plan(num_gpu_blocks=64).warnings)


def test_scratch_block_and_draft_reservation_are_in_the_plan():
    a, b = _plan(kv_pool_cap_tokens=32768), _plan(kv_pool_cap_tokens=32768, extra_bytes=int(3 * GB))
    assert a.planned_total_bytes == a.weight_bytes + a.other_allocated_bytes + a.kv_pool_bytes + a.activation_reserve_bytes + PER_BLOCK_BF16
    assert b.planned_total_bytes - a.planned_total_bytes == int(3 * GB) and b.extra_bytes == int(3 * GB)
    f1, f2 = _plan(kv_pool="fill"), _plan(kv_pool="fill", extra_bytes=int(3 * GB))
    assert f2.num_blocks < f1.num_blocks and f1.planned_total_bytes <= int(0.85 * 102.6 * GB)
    assert draft_pool_blocks(max_num_seqs=16, max_model_len=4096, block_size=16, kv_pool_cap_tokens=32768) == 512


# ---- 2. failure memoisation --------------------------------------------------------------------------
def test_only_compile_errors_are_memoised():
    from forge.kernels import ops_gemm
    from forge.kernels.triton.gemm_common import is_compile_error

    class CompilationError(Exception):
        pass

    class OutOfResources(Exception):
        pass

    assert is_compile_error(CompilationError("x")) and is_compile_error(OutOfResources("x"))
    assert not is_compile_error(MemoryError()) and not is_compile_error(RuntimeError("CUDA error: an illegal memory access")) and not is_compile_error(ValueError("x"))
    if hasattr(torch.cuda, "OutOfMemoryError"):
        assert not is_compile_error(torch.cuda.OutOfMemoryError("out of memory"))
    ops_gemm.reset_failure_memo()
    k1, k2 = ("op", "float16", -1, 4, 64), ("op", "float16", -1, 8, 64)
    ops_gemm._fail("op", MemoryError("oom"), k1)
    assert k1 not in ops_gemm._FAILED
    ops_gemm._fail("op", CompilationError("bad"), k2)
    assert k2 in ops_gemm._FAILED and k1 not in ops_gemm._FAILED  # per shape key
    ops_gemm.reset_failure_memo()


# ---- 3. kernel-layout conversion ---------------------------------------------------------------------
def test_conversion_releases_each_portable_module_and_warns_with_layer_name(artifact, caplog, monkeypatch):
    from forge.compress.packed import IntLinear
    from forge.kernels.ops_gemm import KernelW4A16Linear
    from forge.models.loader import convert_packed_for_kernels, load_model

    m, _ = load_model(artifact, dtype="float32", device="cpu", with_tokenizer=False)
    refs = [weakref.ref(x) for x in m.modules() if isinstance(x, IntLinear)]
    assert refs
    out = convert_packed_for_kernels(m, require_cuda=False)
    gc.collect()
    assert out["converted"] == len(refs) and all(r() is None for r in refs)  # nothing keeps the old portable modules alive

    m2, _ = load_model(artifact, dtype="float32", device="cpu", with_tokenizer=False)
    names = [n for n, x in m2.named_modules() if isinstance(x, IntLinear)]
    real = KernelW4A16Linear.from_packed
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory (simulated)")
        return real(*a, **kw)

    monkeypatch.setattr(KernelW4A16Linear, "from_packed", staticmethod(flaky))
    with caplog.at_level(logging.WARNING, logger="forge.models.loader"):
        out = convert_packed_for_kernels(m2, require_cuda=False)
    assert out["kept"] == 1 and out["converted"] == len(names) - 1 and out["errors"]
    assert any(names[0] in r.getMessage() and "out of memory" in r.getMessage() for r in caplog.records)
    assert isinstance(m2.get_submodule(names[0]), IntLinear)


# ---- 4. artifact KV scheme ---------------------------------------------------------------------------
def test_artifact_kv_scheme_is_never_silently_dropped():
    assert artifact_kv_choice("auto", {"scheme": "int8"}) == ("int8", [])
    chosen, w = artifact_kv_choice("auto", {"scheme": "fp8_e5m2"})
    assert chosen is None and w and "fp8_e5m2" in w[0]
    chosen, w = artifact_kv_choice("auto", {"scheme": "kivi", "residual_window": 128})
    assert chosen is None and "kivi" in w[0]
    chosen, w = artifact_kv_choice("auto", {"scheme": "fp8_e4m3", "residual_window": 128})
    assert chosen == "fp8_e4m3" and "residual_window=128" in w[0]
    assert artifact_kv_choice("auto", None) == (None, []) and artifact_kv_choice("bf16", {"scheme": "int8"})[1]


def test_engine_reports_kv_warnings_and_status_shows_them(artifact):
    from forge.api.routers.deployments import _status_warnings
    from forge.serve._testing import ByteTokenizer
    from forge.serve.engine import LLMEngine

    eng = LLMEngine(ServeConfig(model=str(artifact), device="cpu", dtype="float32", max_model_len=64, max_num_seqs=2), tokenizer=ByteTokenizer())
    assert eng.identity["warnings"] == eng.kv_warnings
    w = _status_warnings(None, None, {"kv": {"scheme": "kivi"}}, "auto")
    assert w and "kivi" in w[0]
    assert _status_warnings(eng.identity, eng.memory.to_dict(), None) == list(eng.kv_warnings) + list(eng.memory.warnings)


# ---- 5. UI -------------------------------------------------------------------------------------------
def test_inference_page_preselects_enabled_option_and_flags_typed_override():
    js = (Path(__file__).parent.parent / "forge" / "web" / "static" / "js" / "pages" / "inference.js").read_text(encoding="utf-8")
    assert "!o.disabled" in js and "d.quant[0].value" not in js
    assert "overrideMsg" in js and "typed reference used" in js


# ---- 6. model references -----------------------------------------------------------------------------
def test_directory_without_config_json_points_at_artifact_subdir(tmp_path):
    from forge.core.modelref import resolve_model_ref

    run = tmp_path / "run"
    (run / "artifact").mkdir(parents=True)
    (run / "artifact" / "config.json").write_text("{}")
    with pytest.raises(NotFound, match="artifact"):
        resolve_model_ref(str(run))
    assert resolve_model_ref(str(run / "artifact")).path == run / "artifact"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert resolve_model_ref(str(empty)).kind == "local"  # no hint available: the loader reports what is missing


def test_preflight_servable_rejects_dirs_without_config(tmp_path):
    from forge.api.routers.deployments import _preflight_servable

    (tmp_path / "artifact").mkdir()
    (tmp_path / "artifact" / "config.json").write_text("{}")
    with pytest.raises(NotFound, match="select that path"):
        _preflight_servable(str(tmp_path))


def test_digest_materialisation_is_verified_and_replaces_stale_dirs(tmp_path):
    from forge.core.modelref import _DIGEST_MARK, _materialize_digest

    class Store:
        def __init__(self, ok=True):
            self.ok, self.mat = ok, 0

        def verify(self, ref):
            return self.ok

        def materialize_artifact(self, ref, dest):
            self.mat += 1
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / "config.json").write_text("{}")

    ref, dest = "blake3:" + "cd" * 32, tmp_path / "artifacts" / "cd"
    with pytest.raises(NotFound, match="verification"):
        _materialize_digest(Store(ok=False), ref, dest)
    assert not dest.exists()
    st = Store()
    _materialize_digest(st, ref, dest)
    assert (dest / _DIGEST_MARK).read_text() == ref and st.mat == 1
    _materialize_digest(st, ref, dest)  # complete + marked for this digest: reused
    assert st.mat == 1
    (dest / _DIGEST_MARK).write_text("blake3:other")  # stale / foreign dir at the destination: replaced (Windows cannot rename over it)
    (dest / "junk.bin").write_text("x")
    _materialize_digest(st, ref, dest)
    assert st.mat == 2 and not (dest / "junk.bin").exists() and (dest / _DIGEST_MARK).read_text() == ref


# ---- 7. /deployments/estimate ------------------------------------------------------------------------
def test_estimate_maps_bad_serve_config_to_422_and_reports_pool_vs_weights(client, home, artifact):
    from forge.jobs.modelinfo import register_artifact_dir

    d = home / "runs" / "job_x" / "artifact"
    shutil.copytree(artifact, d)
    row = register_artifact_dir(d, name="q")
    bad = client.post(f"{API}/deployments/estimate", json={"artifact_ref": row["id"], "serve_config": {"max_model_len": -1}})
    assert bad.status_code == 422, bad.text
    assert client.post(f"{API}/deployments/estimate", json={"artifact_ref": row["id"], "serve_config": {"bogus": 1}}).status_code == 422
    ok = client.post(f"{API}/deployments/estimate", json={"artifact_ref": row["id"], "total_gpu_gb": 96, "serve_config": {}})
    assert ok.status_code == 200, ok.text
    e = ok.json()
    assert e["fits"] and "kv_warnings" in e and e["limited_by"] in ("workload", "workload_cap")
