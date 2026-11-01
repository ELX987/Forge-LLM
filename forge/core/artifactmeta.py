"""Read-only summary of a compressed ("quantised") model directory, from its JSON metadata only (no torch, no weights).

A directory written by ``forge.compress.serialize.save_quantized`` holds ``config.json`` + safetensors + ``forge_compress.json``.  The module
manifest (``{module name: {kind, spec, in_features, out_features, bias}}``) lives in ``forge_compress.json`` (``manifest``) and, for directories
written through ``forge.models.loader.save_model``, also in ``config.json`` (``forge.quantization``); the compress-only writer uses
``config.json`` -> ``forge_quantization``.  :func:`read_artifact_summary` turns whichever of these exist into one dict that the model registry,
the model resolver, the loader (to verify what it built) and the web UI (badges) share.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

META_FILE = "forge_compress.json"

_FP_LABELS = {"fp8": "FP8", "nvfp4": "NVFP4", "mxfp4": "MXFP4", "int8": "W8A8", "w8a8": "W8A8", "lowrank": "LowRank", "bitdelta": "BitDelta"}


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def quant_label(layer_kinds: dict[str, int], specs: dict[str, dict[str, Any]]) -> str:
    """Short badge text, e.g. ``W4A16 g128`` / ``W8A16`` / ``NVFP4`` / ``W4A16 g128 + W8A16`` (the dominant kind first)."""
    parts: list[str] = []
    for kind, _n in sorted(layer_kinds.items(), key=lambda kv: -kv[1]):
        spec = specs.get(kind, {})
        if kind in ("w4a16", "w8a16", "w3a16", "w2a16") or (kind.startswith("w") and kind.endswith("a16")):
            lab = kind.upper()
            g = spec.get("group_size")
            if isinstance(g, int) and g > 0:
                lab += f" g{g}"
        else:
            lab = _FP_LABELS.get(kind, kind)
        parts.append(lab)
    return " + ".join(parts)


def read_artifact_summary(path: str | Path) -> dict[str, Any] | None:
    """``None`` when ``path`` is not a directory with a ``config.json``; otherwise a dict (``is_quantized`` False for plain dense models).

    Keys: ``is_quantized``, ``is_compress_artifact`` (has ``forge_compress.json``), ``writer``, ``layer_kinds`` ({kind: count}), ``n_quantized_layers``,
    ``label``, ``scheme``, ``bits``, ``group_size``, ``sym``, ``layout_id``, ``kv`` (recipe KV scheme or None), ``recipe``, ``name``,
    ``base_model`` / ``base_digest`` (what was compressed), ``licence``, ``params_b``, ``gate``, ``bytes_input`` / ``bytes_output`` / ``compression``,
    ``manifest_source``.  Never raises on malformed metadata (fields are simply missing)."""
    p = Path(path)
    cfg_path = p / "config.json"
    if not p.is_dir() or not cfg_path.is_file():
        return None
    cfg = _read_json(cfg_path) or {}
    meta = _read_json(p / META_FILE)
    manifest: dict[str, Any] = {}
    source = None
    if meta and isinstance(meta.get("manifest"), dict) and meta["manifest"]:
        manifest, source = meta["manifest"], META_FILE
    elif isinstance((cfg.get("forge") or {}).get("quantization"), dict) and cfg["forge"]["quantization"]:
        manifest, source = cfg["forge"]["quantization"], "config.json:forge.quantization"
    elif isinstance(cfg.get("forge_quantization"), dict) and cfg["forge_quantization"]:
        manifest, source = cfg["forge_quantization"], "config.json:forge_quantization"
    ents = [e for e in manifest.values() if isinstance(e, dict) and e.get("kind")]
    kinds = Counter(str(e["kind"]) for e in ents)
    specs: dict[str, dict[str, Any]] = {}
    for e in ents:
        specs.setdefault(str(e["kind"]), dict(e.get("spec") or {}))
    dom = max(kinds, key=lambda k: kinds[k]) if kinds else None
    dspec = specs.get(dom, {}) if dom else {}
    extra = (meta or {}).get("extra") or {}
    report = (meta or {}).get("report") or {}
    gate = extra.get("gate")
    out: dict[str, Any] = {
        "is_quantized": bool(ents),
        "is_compress_artifact": meta is not None,
        "writer": (meta or {}).get("writer"),
        "manifest_source": source,
        "layer_kinds": dict(kinds),
        "n_quantized_layers": len(ents),
        "scheme": dom,
        "label": quant_label(dict(kinds), specs) if ents else None,
        "bits": dspec.get("bits"),
        "group_size": dspec.get("group_size"),
        "sym": dspec.get("sym"),
        "layout_id": extra.get("layout_id"),
        "kv": extra.get("kv"),
        "recipe": extra.get("recipe") or report.get("recipe"),
        "name": extra.get("name"),
        "base_model": extra.get("input") or report.get("input"),
        "base_digest": extra.get("base_digest") or report.get("base_digest"),
        "licence": extra.get("licence"),
        "params_b": extra.get("params_b"),
        "gate": gate.get("verdict") if isinstance(gate, dict) else gate,
        "dtype": (meta or {}).get("dtype") or cfg.get("torch_dtype") or cfg.get("dtype"),
    }
    cs = extra.get("compress_summary")
    if isinstance(cs, dict):
        out["compress_summary"] = cs
    return out
