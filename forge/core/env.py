"""Reproducibility helpers: seeding and environment capture. Never requires a GPU; heavy imports are guarded."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import platform
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from forge.core.hashing import hash_json

_KEY_PACKAGES = (
    "torch", "triton", "transformers", "tokenizers", "safetensors", "numpy", "pydantic", "fastapi",
    "uvicorn", "huggingface_hub", "nvidia-ml-py", "pynvml", "zstandard", "blake3", "pyarrow", "torchao",
    "transformer_engine", "flash_attn", "accelerate", "peft",
)


def seed_everything(seed: int, *, deterministic: bool = False) -> int:
    """Seed python, numpy and torch (CPU + all CUDA devices). With ``deterministic`` also request
    deterministic algorithms (``torch.use_deterministic_algorithms(True, warn_only=True)``)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    return seed


def _pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:
        return None


def _run(cmd: list[str], cwd: str | None = None, timeout: float = 5.0) -> str | None:
    try:
        out = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_sha(path: str | os.PathLike | None = None) -> str | None:
    """HEAD commit of the checkout containing ``path`` (default: this package), or None."""
    if shutil.which("git") is None:
        return None
    root = Path(path) if path else Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    return _run(["git", "rev-parse", "HEAD"], cwd=str(root))


def _torch_info() -> dict[str, Any]:
    info: dict[str, Any] = {"installed": False}
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return info
    info.update(installed=True, version=getattr(torch, "__version__", None))
    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        info["hip_version"] = getattr(torch.version, "hip", None)
        try:
            info["cudnn_version"] = torch.backends.cudnn.version()
        except Exception:
            info["cudnn_version"] = None
        gpus = []
        if info["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": p.name,
                        "sm": p.major * 10 + p.minor,
                        "total_memory_mb": round(p.total_memory / 2**20),
                        "multi_processor_count": p.multi_processor_count,
                    }
                )
        info["gpus"] = gpus
    except Exception as e:  # pragma: no cover - driver quirks
        info["error"] = repr(e)
    return info


def _driver_version() -> str | None:
    try:
        pynvml = importlib.import_module("pynvml")
        pynvml.nvmlInit()
        try:
            v = pynvml.nvmlSystemGetDriverVersion()
            return v.decode() if isinstance(v, bytes) else str(v)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if out:
            return out.splitlines()[0].strip()
    return None


def _driver_cuda() -> str | None:
    """Newest CUDA version the driver supports (NVML / nvidia-smi header; no probing, never raises)."""
    try:
        from forge.hw import cuda_compat

        return cuda_compat.detect_stack(probe=False).get("driver_cuda")
    except Exception:
        return None


def gpu_fingerprint(gpus: list[dict[str, Any]] | None = None, driver: str | None = None) -> str | None:
    """Stable hash over GPU names/sm/memory + driver, or None without GPUs."""
    if gpus is None:
        gpus = _torch_info().get("gpus", [])
    if not gpus:
        return None
    ident = [[g.get("name"), g.get("sm"), g.get("total_memory_mb")] for g in gpus]
    return hash_json({"gpus": ident, "driver": driver})


def capture_environment() -> dict[str, Any]:
    """JSON-serialisable snapshot of the software/hardware environment (safe without a GPU)."""
    torch_info = _torch_info()
    driver = _driver_version()
    versions = {p: v for p in _KEY_PACKAGES if (v := _pkg_version(p))}
    return {
        "python": sys.version.split()[0],
        "python_impl": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch": torch_info,
        "triton": _pkg_version("triton"),
        "cuda_driver": driver,
        "cuda_driver_cuda": _driver_cuda(),
        "gpu_fingerprint": gpu_fingerprint(torch_info.get("gpus"), driver),
        "git_sha": git_sha(),
        "packages": versions,
        "env_flags": {
            k: os.environ[k]
            for k in ("FORGE_FORCE_REF", "FORGE_KERNEL_DEBUG", "FORGE_HASH_ALGO", "CUDA_VISIBLE_DEVICES")
            if k in os.environ
        },
    }
