"""Save / load compressed models (packed linears, low-rank factors, BitDelta layers) as safetensors + JSON manifest.

Directory layout (HF-style, so ``forge.store.put_tensor_dir`` can ingest it as-is)::

    config.json              architecture (``forge_arch_config`` = exact ModelConfig dump; HF-compatible keys when available)
                             + ``forge_quantization`` (module manifest, same schema as ``forge.models.linear.quantization_manifest``)
    model.safetensors        (or ``model-00001-of-N.safetensors`` + ``model.safetensors.index.json``)
    forge_compress.json      manifest, report, kv config, tied-weight map, format version
    tokenizer files          if a tokenizer with ``save_pretrained`` was supplied

Packed layers store their canonical "simple" layout (see :mod:`forge.compress.packed`); device-specific kernel layouts are derived
at load / pack time and never stored, so re-packing for another chip never changes the payload bytes (Plan 8.2: sidecars dedupe).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

from forge.compress.modelutil import is_packed_linear
from forge.compress.packed import PACKED_CLASSES
from forge.core.errors import ConfigError

log = logging.getLogger("forge.compress.serialize")

FORMAT_VERSION = 1
MODULE_KINDS: dict[str, type] = {}


def register_module_kind(kind: str) -> Callable[[type], type]:
    """Register a compress-owned module class (low-rank, BitDelta...) so :func:`load_quantized` can rebuild it from the manifest."""

    def deco(cls: type) -> type:
        cls.kind = kind  # type: ignore[attr-defined]
        MODULE_KINDS[kind] = cls
        return cls

    return deco


def resolve_kind(kind: str) -> type:
    if kind in PACKED_CLASSES:
        return PACKED_CLASSES[kind]
    if kind in MODULE_KINDS:
        return MODULE_KINDS[kind]
    try:
        from forge.models.linear import get_linear_cls

        return get_linear_cls(kind)
    except Exception as e:
        raise ConfigError(f"unknown module kind {kind!r} in manifest ({e})") from e


def compress_manifest(model: nn.Module) -> dict[str, dict[str, Any]]:
    """``{module name: {kind, spec, in_features, out_features, bias}}`` for every compressed (non-dense) linear-like module."""
    out: dict[str, dict[str, Any]] = {}
    for name, m in model.named_modules():
        if not name or not hasattr(m, "quant_spec"):
            continue
        own = getattr(m, "kind", None) in MODULE_KINDS
        if not own and (isinstance(m, nn.Linear) or not is_packed_linear(m)):
            continue
        out[name] = {
            "kind": m.kind, "spec": m.quant_spec(), "in_features": int(m.in_features), "out_features": int(m.out_features),
            "bias": getattr(m, "bias", None) is not None,
        }
    return out


def apply_manifest(model: nn.Module, manifest: dict[str, dict[str, Any]], dtype: torch.dtype | None = None, device: Any = "cpu") -> None:
    """Swap modules named in ``manifest`` for empty modules of the right class/shape (state is loaded afterwards)."""
    for name, ent in manifest.items():
        cls = resolve_kind(ent["kind"])
        new = cls.from_shape(  # type: ignore[attr-defined]
            ent["in_features"], ent["out_features"], bool(ent.get("bias")), dtype=dtype, device=device, **ent.get("spec", {})
        )
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, new)


def _config_dump(model: nn.Module) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise ConfigError("model has no .config; cannot save a self-describing artifact")
    d: dict[str, Any] = {}
    if hasattr(cfg, "to_hf_config"):
        try:
            d.update(cfg.to_hf_config())
        except Exception as e:  # pragma: no cover
            log.warning("to_hf_config failed: %s", e)
    d["forge_arch_config"] = cfg.model_dump(mode="json") if hasattr(cfg, "model_dump") else dict(cfg)
    return d


def _needs_own_writer(model: nn.Module) -> bool:
    return any(getattr(m, "kind", None) in MODULE_KINDS for m in model.modules())


def _loader_writer_ok(model: nn.Module) -> bool:
    if _needs_own_writer(model) or not hasattr(getattr(model, "config", None), "to_hf_config"):
        return False
    try:
        from forge.models import loader  # noqa: F401

        return True
    except Exception:
        return False


def save_quantized(
    model: nn.Module,
    out_dir: str | Path,
    *,
    report: dict[str, Any] | None = None,
    tokenizer: Any = None,
    extra: dict[str, Any] | None = None,
    shard_bytes: int = 4 << 30,
    prefer_loader: bool = True,
) -> Path:
    """Write ``model`` (any mix of dense and compressed modules) to ``out_dir``. Returns the directory.

    Pure packed-linear models are written by ``forge.models.loader.save_model`` (HF-compatible names, manifest in ``config.json`` ->
    ``forge.quantization``) so ``loader.load_model`` / the serving engine can load them directly; models containing compress-only modules
    (low-rank factors, online Hadamard) use the writer below.  ``forge_compress.json`` is always written next to the weights.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = compress_manifest(model)
    writer = "forge.models.loader" if (prefer_loader and _loader_writer_ok(model)) else "forge.compress.serialize"
    tied: dict[str, str] = {}
    if writer == "forge.models.loader":
        from forge.models import loader

        loader.save_model(model, out, shard_gb=shard_bytes / (1 << 30), tokenizer=tokenizer, extra_config={"compress": {"report_summary": {k: v for k, v in (report or {}).items() if k != "layers"}}})
    else:
        tied = _write_own(model, out, manifest, shard_bytes)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            try:
                tokenizer.save_pretrained(str(out))
            except Exception as e:  # pragma: no cover
                log.warning("tokenizer save failed: %s", e)
    meta = {
        "format_version": FORMAT_VERSION, "writer": writer, "manifest": manifest, "tied": tied, "report": report or {}, "extra": extra or {},
        "dtype": str(next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)).replace("torch.", ""),
    }
    (out / "forge_compress.json").write_text(json.dumps(meta, indent=1, default=str))
    return out


def _write_own(model: nn.Module, out: Path, manifest: dict[str, Any], shard_bytes: int) -> dict[str, str]:
    from safetensors.torch import save_file

    sd = model.state_dict()
    tensors: dict[str, torch.Tensor] = {}
    tied: dict[str, str] = {}
    seen: dict[tuple, str] = {}
    for k, v in sd.items():
        t = v.detach().cpu().contiguous()
        key = (v.data_ptr(), tuple(v.shape), str(v.dtype), v.device.type)
        if v.numel() and key in seen:
            tied[k] = seen[key]
            continue
        seen[key] = k
        tensors[k] = t
    shards: list[dict[str, torch.Tensor]] = [{}]
    size = 0
    for k, t in tensors.items():
        nb = t.numel() * t.element_size()
        if size and size + nb > shard_bytes:
            shards.append({})
            size = 0
        shards[-1][k] = t
        size += nb
    files: dict[str, str] = {}
    if len(shards) == 1:
        save_file(shards[0], str(out / "model.safetensors"), metadata={"format": "pt"})
        for k in shards[0]:
            files[k] = "model.safetensors"
    else:
        for i, sh in enumerate(shards):
            fn = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
            save_file(sh, str(out / fn), metadata={"format": "pt"})
            for k in sh:
                files[k] = fn
        total = sum(t.numel() * t.element_size() for t in tensors.values())
        (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": files}, indent=1))
    cfg = _config_dump(model)
    cfg["forge_quantization"] = manifest
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))
    return tied


def read_meta(path: str | Path) -> dict[str, Any]:
    p = Path(path) / "forge_compress.json"
    if not p.exists():
        raise ConfigError(f"{path} is not a forge compressed artifact (missing forge_compress.json)")
    return json.loads(p.read_text())


def load_state_dir(path: str | Path) -> dict[str, torch.Tensor]:
    """All tensors of a model directory (single file or sharded)."""
    from safetensors.torch import load_file

    p = Path(path)
    idx = p / "model.safetensors.index.json"
    sd: dict[str, torch.Tensor] = {}
    if idx.exists():
        for fn in sorted(set(json.loads(idx.read_text())["weight_map"].values())):
            sd.update(load_file(str(p / fn)))
    elif (p / "model.safetensors").exists():
        sd = load_file(str(p / "model.safetensors"))
    else:
        raise ConfigError(f"no safetensors found in {path}")
    return sd


def _default_factory(arch_config: dict[str, Any], dtype: torch.dtype, device: Any) -> nn.Module:
    from forge.models import ForgeLM
    from forge.models.config import ModelConfig

    return ForgeLM(ModelConfig.model_validate(arch_config), dtype=dtype, device=device, init_weights=False)


def load_quantized(
    path: str | Path,
    *,
    dtype: torch.dtype | str | None = None,
    device: str | torch.device = "cpu",
    model_factory: Callable[[dict[str, Any], torch.dtype, Any], nn.Module] | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild a model saved by :func:`save_quantized`. Returns ``(model, forge_compress.json contents)``.

    ``model_factory(arch_config_dict, dtype, device)`` builds the (meta-device) skeleton; default: ``ForgeLM(ModelConfig(...))``.
    """
    p = Path(path)
    meta = read_meta(p)
    if meta.get("writer") == "forge.models.loader":
        from forge.models import loader

        dt0 = dtype if dtype is not None else meta.get("dtype", "float32")
        model, _tok = loader.load_model(p, dtype=dt0, device=device, with_tokenizer=False)
        model.eval()
        return model, meta
    cfg_json = json.loads((p / "config.json").read_text())
    if "forge_arch_config" not in cfg_json:
        raise ConfigError("config.json has no forge_arch_config; cannot rebuild the architecture")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    dt = dtype or getattr(torch, meta.get("dtype", "float32"))
    model = (model_factory or _default_factory)(cfg_json["forge_arch_config"], dt, "meta")
    apply_manifest(model, meta["manifest"], dt, device)
    sd = load_state_dir(p)
    for dup, first in meta.get("tied", {}).items():
        sd[dup] = sd[first]
    res = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in res.missing_keys]
    if missing:
        # tied output head is re-tied below; anything else is an error
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        still = [k for k, v in list(model.named_parameters()) + list(model.named_buffers()) if v.is_meta]
        if still:
            raise ConfigError(f"checkpoint is missing tensors: {still[:6]}")
    if res.unexpected_keys:
        raise ConfigError(f"unexpected tensors in checkpoint: {res.unexpected_keys[:6]}")
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    model.to(device)
    model.eval()
    return model, meta
