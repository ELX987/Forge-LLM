"""Root pytest configuration for Forge LLM.

* puts the repo root on ``sys.path``
* points ``FORGE_HOME`` at a throw-away temp dir for the whole session (before any forge import)
* registers the ``gpu`` marker and auto-skips ``tests_gpu/`` (and ``@pytest.mark.gpu``) without CUDA
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SESSION_HOME = Path(tempfile.mkdtemp(prefix="forge_test_home_"))
os.environ["FORGE_HOME"] = str(_SESSION_HOME)
os.environ.setdefault("FORGE_API_TOKEN", "")


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires an NVIDIA GPU (auto-skipped without CUDA)")
    config.addinivalue_line("markers", "slow: slow test")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _cuda_available():
        return
    skip = pytest.mark.skip(reason="no CUDA device available")
    for item in items:
        in_gpu_dir = "tests_gpu" in Path(str(item.fspath)).parts
        if in_gpu_dir or item.get_closest_marker("gpu"):
            item.add_marker(skip)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("FORGE_TEST_KEEP_HOME"):
        return
    try:
        from forge.core import db as _db

        _db.close_all()
    except Exception:
        pass
    shutil.rmtree(_SESSION_HOME, ignore_errors=True)


@pytest.fixture()
def forge_home(tmp_path, monkeypatch):
    """Isolated FORGE_HOME for a single test (fresh db, store, runs...)."""
    home = tmp_path / "forge_home"
    home.mkdir()
    monkeypatch.setenv("FORGE_HOME", str(home))
    return home


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    return _cuda_available()
