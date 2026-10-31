"""Helpers shared by the run entry points: model construction, tokenizer wiring, export + store registration, lineage."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from forge.core.errors import ConfigError
from forge.core.hashing import hash_json
from forge.train.config import ModelSourceConfig
from forge.train.tokenization import TokenizerAdapter, load_tokenizer_any

log = logging.getLogger("forge.train.common")

__all__ = ["build_model", "coerce_cfg", "export_model", "register_outputs", "resolve_param_dtype", "write_lineage"]

_DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def coerce_cfg(cfg: Any, cls: type) -> Any:
    return cfg if isinstance(cfg, cls) else cls.model_validate(cfg)


def resolve_param_dtype(src: ModelSourceConfig, device: torch.device, *, n_params: float | None = None, frozen_base: bool = False,
                        compute_dtype: str = "auto") -> torch.dtype:
    """Concrete parameter dtype for ``dtype='auto'`` (fp32 master weights unless huge / frozen; CPU is always fp32)."""
    if src.dtype != "auto":
        return _DT[src.dtype]
    if device.type != "cuda":
        return torch.float32
    if frozen_base:
        return torch.bfloat16
    if n_params is not None and n_params > 2e9:
        return torch.bfloat16
    return torch.float32


def _estimate_params(path: str, revision: str | None) -> float | None:
    try:
        from forge.models.loader import read_config, resolve_model_path

        cfg, _ = read_config(resolve_model_path(path, revision))
        return float(cfg.param_count())
    except Exception:  # noqa: BLE001
        return None


def _materialize_artifact(ref: str, ctx: Any) -> str:
    """If ``ref`` is a store artifact hash, materialise it into a directory under the cache dir; else return it unchanged."""
    if not (ref.startswith("blake3:") or ref.startswith("b2:")):
        return ref
    store = getattr(ctx, "store", None)
    if store is None:
        try:
            from forge.store import get_store

            store = get_store()
        except Exception as e:  # noqa: BLE001
            raise ConfigError(f"model reference {ref} looks like an artifact hash but no store is available") from e
    from forge.core.paths import cache_dir

    dest = cache_dir() / "materialized" / ref.replace(":", "_")[:40]
    if not dest.exists():
        store.materialize_artifact(ref, dest)
    return str(dest)


def build_model(src: ModelSourceConfig, *, device: torch.device, seed: int, tok: TokenizerAdapter | None = None, ctx: Any = None,
                frozen_base: bool = False, role: str = "model") -> tuple[nn.Module, str | None]:
    """Instantiate the model described by ``src``. Returns ``(model, model_dir_or_None)``."""
    n_params: float | None = None
    path = src.path
    if src.source == "pretrained":
        if not path:
            raise ConfigError(f"{role}.path is required for source=pretrained")
        path = _materialize_artifact(path, ctx)
        n_params = _estimate_params(path, src.revision)
    dtype = resolve_param_dtype(src, device, n_params=n_params, frozen_base=frozen_base)
    if src.source in ("tiny", "scratch"):
        from forge.models.config import ModelConfig
        from forge.models.modeling import ForgeLM
        from forge.models.tiny import build_tiny_model

        ov = dict(src.arch_overrides)
        if tok is not None and "vocab_size" not in ov:
            ov["vocab_size"] = max(tok.vocab_size, 8)
        torch.manual_seed(seed)
        if src.source == "tiny":
            model = build_tiny_model(seed, dtype=dtype, **ov)
        else:
            if not ov.get("hidden_size"):
                raise ConfigError("source=scratch needs arch_overrides (hidden_size, num_layers, num_heads, ...)")
            model = ForgeLM(ModelConfig(**ov), dtype=dtype)
        return model.to(device), None
    if src.backend == "hf" or src.backend == "forge":
        from forge.models.loader import load_model

        model, _ = load_model(path, dtype=dtype, device=device, revision=src.revision, backend=src.backend, with_tokenizer=False)
        return model, path
    raise ConfigError(f"unknown backend {src.backend!r}")


def load_tokenizer_for(src: ModelSourceConfig, ctx: Any = None) -> TokenizerAdapter:
    spec = src.tokenizer or (src.path if src.source == "pretrained" else "byte")
    if src.tokenizer is None and src.source == "pretrained":
        spec = _materialize_artifact(src.path, ctx)
    return load_tokenizer_any(spec)


def check_vocab(model: nn.Module, tok: TokenizerAdapter) -> None:
    cfg = getattr(model, "config", None)
    v = getattr(cfg, "vocab_size", None)
    if v is not None and tok.vocab_size and int(v) < tok.vocab_size:
        raise ConfigError(f"model vocab ({v}) is smaller than the tokenizer vocab ({tok.vocab_size})")


def write_lineage(path: Path, lineage: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(lineage, indent=1, default=str), encoding="utf-8")


def export_model(model: nn.Module, out_dir: Path, tok: TokenizerAdapter | None, dtype: str, lineage: dict[str, Any] | None = None) -> Path:
    """Write a loadable model directory (safetensors + config.json [+ tokenizer])."""
    out_dir.mkdir(parents=True, exist_ok=True)
    as_dtype = None if dtype == "auto" else dtype
    inner = tok.tok if (tok is not None and not tok.is_byte) else None
    try:
        from forge.models.loader import save_model

        save_model(model, out_dir, as_dtype=as_dtype, tokenizer=inner)
    except Exception as e:  # noqa: BLE001
        log.warning("forge.models.loader.save_model failed (%r); writing a plain safetensors/state_dict export", e)
        sd = {k: (v.detach().cpu().to(_DT[as_dtype]) if as_dtype and v.is_floating_point() else v.detach().cpu()) for k, v in model.state_dict().items()}
        try:
            from safetensors.torch import save_file

            save_file({k: v.contiguous() for k, v in sd.items()}, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
        except Exception:  # noqa: BLE001
            torch.save(sd, out_dir / "model.pt")
        cfg = getattr(model, "config", None)
        if cfg is not None:
            try:
                (out_dir / "config.json").write_text(json.dumps(cfg.to_hf_config() if hasattr(cfg, "to_hf_config") else vars(cfg), indent=1, default=str), encoding="utf-8")
            except Exception:  # noqa: BLE001
                pass
    if tok is not None and tok.is_byte:
        (out_dir / "tokenizer_spec.json").write_text(json.dumps({"tokenizer": "byte", "vocab_size": tok.vocab_size}), encoding="utf-8")
    if lineage is not None:
        write_lineage(out_dir / "forge_lineage.json", lineage)
    return out_dir


def register_outputs(ctx: Any, kind: str, directory: Path, *, parents: list[str], extra: dict[str, Any], out_key: str, licence: str | None = None) -> str | None:
    """Put a model/adapter directory into the artifact store with lineage edges (best effort). Returns the artifact hash."""
    store = getattr(ctx, "store", None)
    ctx.set_output(f"{out_key}_dir", str(directory))
    if store is None:
        return None
    try:
        kw: dict[str, Any] = {}
        if licence:
            kw["licence"] = licence
        h = store.put_tensor_dir(directory, kind=kind, parents=[p for p in parents if p], extra=extra, **kw)
        ctx.set_output(f"{out_key}_hash", h)
        ctx.set_output("artifact_hash", h)
        for p in parents:
            if p and p != h:
                try:
                    store.add_edge(h, p)
                except Exception:  # noqa: BLE001
                    pass
        ctx.log(f"registered {kind} artifact {h}")
        return h
    except Exception as e:  # noqa: BLE001
        ctx.log(f"could not register {kind} in the store: {e!r}", "warning")
        return None


def base_lineage(cfg: Any, ctx: Any, result: dict[str, Any], data_fp: str | None, parent_hash: str | None) -> dict[str, Any]:
    return {
        "kind": type(cfg).__name__, "name": cfg.name, "parent_hash": parent_hash, "data_fingerprint": data_fp,
        "config_hash": hash_json(cfg.model_dump(mode="json")), "tokens_seen": result.get("tokens_seen"), "steps": result.get("steps"),
        "final_loss": result.get("final_loss"), "best_eval_loss": result.get("best_eval_loss"), "precision_rung": cfg.precision.rung,
        "run_id": getattr(ctx, "run_id", None), "created": time.time(), "eval_report_hashes": [],
    }
