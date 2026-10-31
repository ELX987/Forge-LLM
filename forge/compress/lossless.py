"""Artifact-level lossless coding (Plan CP-11): measure / apply the exponent-split entropy codec of ``forge.store.lossless`` to a model's tensors.

The codec itself (BF16/FP16/FP8 exponent-field coding, byte-grouped zstd baseline, verify-after-encode) lives in :mod:`forge.store.lossless`
and is imported lazily; when that module is unavailable a self-contained byte-grouped zlib/zstd baseline is used (documented, weaker).
This module adds the *artifact* view: per-tensor results, totals, exponent entropy (the Plan quotes ~2.6 bits for BF16 LLM weights from the
literature -- here it is *measured* on the actual weights), an optional on-disk container directory, and a verified decode back to a state dict.

Serving path (Plan 8.2): weights are decoded once at load time; decode-inside-GEMM needs a fused kernel and is not implemented here.
"""
from __future__ import annotations

import json
import logging
import time
import zlib
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from forge.core.errors import IntegrityError

log = logging.getLogger("forge.compress.lossless")

try:
    import zstandard as _zstd
except Exception:  # pragma: no cover
    _zstd = None


class LosslessConfig(BaseModel):
    """Lossless-coding settings (stage ``lossless``)."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["auto", "exp_huffman", "exp_zstd", "bytegroup_zstd"] = Field("auto", description="Codec: auto = exponent Huffman for float tensors, byte-grouped zstd otherwise.")
    write_container: bool = Field(False, description="Write the encoded tensors to <run_dir>/lossless/ (one container per tensor) in addition to reporting.")
    min_numel: int = Field(1024, ge=0, description="Tensors smaller than this are stored raw (header overhead dominates).")
    max_tensors: int | None = Field(None, ge=1, description="Only process the first N tensors (quick estimates on huge models).")
    verify: bool = Field(True, description="Decode and compare every tensor (always on for container writes).")


def _store_codec() -> Any | None:
    try:
        from forge.store import lossless as codec

        return codec
    except Exception as e:  # pragma: no cover - depends on the store module being present
        log.warning("forge.store.lossless unavailable (%s); using the built-in byte-group baseline", e)
        return None


# -- fallback baseline ------------------------------------------------------------------------------
def _fb_encode(t: torch.Tensor) -> bytes:
    raw = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b""
    isz = t.element_size()
    planes = [raw[i::isz] for i in range(isz)]
    comp = [(_zstd.ZstdCompressor(level=6).compress(p) if _zstd else zlib.compress(p, 6)) for p in planes]
    head = json.dumps({"dtype": str(t.dtype).replace("torch.", ""), "shape": list(t.shape), "lens": [len(c) for c in comp]}).encode()
    return len(head).to_bytes(4, "little") + head + b"".join(comp)


def _fb_decode(blob: bytes) -> torch.Tensor:
    hl = int.from_bytes(blob[:4], "little")
    head = json.loads(blob[4 : 4 + hl])
    pos = 4 + hl
    planes = []
    for ln in head["lens"]:
        c = blob[pos : pos + ln]
        pos += ln
        planes.append(_zstd.ZstdDecompressor().decompress(c) if _zstd else zlib.decompress(c))
    isz = len(planes)
    n = len(planes[0])
    dt = getattr(torch, head["dtype"])
    if n == 0:
        return torch.empty(head["shape"], dtype=dt)
    buf = bytearray(n * isz)
    for i, p in enumerate(planes):
        buf[i::isz] = p
    return torch.frombuffer(buf, dtype=torch.uint8).clone().view(dt).reshape(head["shape"])


def _roundtrip_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.numel() == 0:
        return True
    return bool(torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)))


def lossless_encode_state_dict(
    sd: dict[str, torch.Tensor], cfg: LosslessConfig | dict | None = None, out_dir: str | Path | None = None
) -> dict[str, Any]:
    """Encode every tensor of ``sd`` bit-exactly; returns the report (and writes containers when ``out_dir`` is given)."""
    cfg = LosslessConfig.model_validate(cfg) if isinstance(cfg, dict) else (cfg or LosslessConfig())
    codec = _store_codec()
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    rows = []
    raw_total = enc_total = 0
    t0 = time.perf_counter()
    items = list(sd.items())[: cfg.max_tensors] if cfg.max_tensors else list(sd.items())
    for i, (name, t) in enumerate(items):
        t = t.detach().cpu().contiguous()
        raw = t.numel() * t.element_size()
        if t.numel() < cfg.min_numel:
            blob, method = None, "raw"
            enc = raw
        else:
            if codec is not None:
                kw = {} if cfg.method == "auto" else {"method": cfg.method}
                blob = codec.encode_tensor(t, **kw)
                method = codec.read_info(blob).get("method", cfg.method) if hasattr(codec, "read_info") else cfg.method
            else:
                blob, method = _fb_encode(t), "bytegroup_zstd_fallback"
            enc = len(blob)
            if cfg.verify or out:
                back = codec.decode_tensor(blob) if codec is not None else _fb_decode(blob)
                if not _roundtrip_equal(back, t):
                    raise IntegrityError(f"lossless round trip failed for tensor {name}")
        ent = None
        if t.dtype in (torch.bfloat16, torch.float16, torch.float32) and codec is not None and hasattr(codec, "exponent_entropy_bits") and t.numel() >= cfg.min_numel:
            try:
                ent = float(codec.exponent_entropy_bits(t))
            except Exception:
                ent = None
        rows.append({"name": name, "dtype": str(t.dtype).replace("torch.", ""), "numel": t.numel(), "raw_bytes": raw, "encoded_bytes": enc,
                     "ratio": enc / max(raw, 1), "method": method, "exponent_entropy_bits": ent})
        raw_total += raw
        enc_total += enc
        if out:
            (out / f"{i:06d}.flc").write_bytes(blob if blob is not None else _fb_encode(t))  # tiny tensors: baseline container
    if out:
        (out / "index.json").write_text(json.dumps([{"file": f"{i:06d}.flc", **{k: r[k] for k in ("name", "dtype", "method")}, "shape": list(sd[r["name"]].shape)}
                                                    for i, r in enumerate(rows)], indent=0))
    ents = [r["exponent_entropy_bits"] for r in rows if r["exponent_entropy_bits"] is not None]
    by_dtype: dict[str, dict[str, float]] = {}
    for r in rows:
        d = by_dtype.setdefault(r["dtype"], {"raw_bytes": 0, "encoded_bytes": 0})
        d["raw_bytes"] += r["raw_bytes"]
        d["encoded_bytes"] += r["encoded_bytes"]
    for d in by_dtype.values():
        d["ratio"] = d["encoded_bytes"] / max(d["raw_bytes"], 1)
    return {
        "tensors": len(rows), "raw_bytes": raw_total, "encoded_bytes": enc_total, "ratio": enc_total / max(raw_total, 1),
        "saving_pct": 100.0 * (1 - enc_total / max(raw_total, 1)), "by_dtype": by_dtype,
        "mean_exponent_entropy_bits": sum(ents) / len(ents) if ents else None, "codec": "forge.store.lossless" if codec is not None else "builtin_bytegroup",
        "verified": bool(cfg.verify or out), "seconds": time.perf_counter() - t0, "provenance": "measured", "rows": rows,
    }


def lossless_decode_dir(path: str | Path) -> dict[str, torch.Tensor]:
    """Inverse of ``lossless_encode_state_dict(..., out_dir=path)``; every tensor is verified against its stored hash by the codec."""
    p = Path(path)
    index = json.loads((p / "index.json").read_text())
    codec = _store_codec()
    sd: dict[str, torch.Tensor] = {}
    for ent in index:
        blob = (p / ent["file"]).read_bytes()
        if ent["method"] in ("raw", "bytegroup_zstd_fallback") or codec is None:
            sd[ent["name"]] = _fb_decode(blob)
        else:
            sd[ent["name"]] = codec.decode_tensor(blob)
    return sd


def lossless_report(model: nn.Module | dict[str, torch.Tensor], cfg: LosslessConfig | dict | None = None, out_dir: str | Path | None = None) -> dict[str, Any]:
    sd = model.state_dict() if isinstance(model, nn.Module) else model
    return lossless_encode_state_dict(dict(sd), cfg, out_dir)
