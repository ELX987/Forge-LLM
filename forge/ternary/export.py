"""``.t158.safetensors`` import/export (tern158 interchange) at tensor level.

tern158 file layout: per ternary layer ``{name}.codes`` (int32 words, 16 codes each, little-endian), ``{name}.scales`` (fp32, flat
``[N*G]`` or a scalar for per-tensor), optional ``{name}.norm`` (fp32 norm gain), plus every non-ternary parameter by name; metadata
``format='t158'``, ``version='1'``, ``config`` (TOML), ``group`` (effective group of the first layer; 0 = per tensor).

These functions only translate storage (bit-exact, see ``forge.formats.ternary.words_to_packed``).  Mapping tern158's fused,
Llama-like ``TernaryLM`` onto a Forge model (qkv / gate-up splitting, SubLN gains, config) is part of the convert work (P3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from forge.core.errors import ConfigError
from forge.formats import ternary as tn
from forge.kernels.layouts.ternary import TERNARY2_ROW, PackedTernary

T158_FORMAT = "t158"


@dataclass
class T158File:
    layers: dict[str, PackedTernary] = field(default_factory=dict)
    norms: dict[str, torch.Tensor] = field(default_factory=dict)
    dense: dict[str, torch.Tensor] = field(default_factory=dict)
    meta: dict[str, str] = field(default_factory=dict)


def infer_shapes(cfg: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """``{layer_name: (N, K)}`` for tern158's ``l{i}.qkv/.o/.up/.down`` layers from a parsed tern158 config dict."""
    m = cfg["model"]
    hidden, layers, heads, kv = int(m["hidden"]), int(m["layers"]), int(m["heads"]), int(m["kv_heads"])
    hd = int(m.get("head_dim") or 0) or hidden // heads
    ffn = int(m["ffn"])
    up_out = 2 * ffn if m.get("act", "swiglu") == "swiglu" else ffn
    out: dict[str, tuple[int, int]] = {}
    for i in range(layers):
        out[f"l{i}.qkv"] = ((heads + 2 * kv) * hd, hidden)
        out[f"l{i}.o"] = (hidden, heads * hd)
        out[f"l{i}.up"] = (up_out, hidden)
        out[f"l{i}.down"] = (hidden, ffn)
    return out


def _shapes_from_meta(meta: dict[str, str]) -> dict[str, tuple[int, int]]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - python 3.10
        import tomli as tomllib  # type: ignore[no-redef]
    return infer_shapes(tomllib.loads(meta["config"]))


def read_t158(path: str | Path, shapes: dict[str, tuple[int, int]] | None = None) -> T158File:
    """Read a tern158 export.  ``shapes`` maps layer name -> (N, K); when omitted it is derived from the TOML config in the metadata."""
    from safetensors import safe_open

    out = T158File()
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        if meta.get("format") != T158_FORMAT:
            raise ConfigError(f"{path}: not a t158 file (metadata format={meta.get('format')!r})")
        out.meta = meta
        keys = list(f.keys())
        names = sorted(k[: -len(".codes")] for k in keys if k.endswith(".codes"))
        if names and shapes is None:
            shapes = _shapes_from_meta(meta)
        for name in names:
            if shapes is None or name not in shapes:
                raise ConfigError(f"{path}: no (N, K) shape known for layer {name!r}")
            n, k = shapes[name]
            words = f.get_tensor(f"{name}.codes")
            if words.numel() != n * k // tn.CODES_PER_WORD:
                raise ConfigError(f"{name}: codes hold {words.numel()} words, expected {n * k // tn.CODES_PER_WORD} for ({n}, {k})")
            sc = f.get_tensor(f"{name}.scales").float()
            if sc.numel() == 1:
                group, scales = 0, sc.reshape(1, 1)
            else:
                g = sc.numel() // n
                if g * n != sc.numel() or k % g:
                    raise ConfigError(f"{name}: {sc.numel()} scales do not fit N={n}, K={k}")
                group, scales = (-1 if g == 1 else k // g), sc.reshape(n, g)
            out.layers[name] = PackedTernary(tn.words_to_packed(words, n, k), scales.contiguous(), TERNARY2_ROW, n, k, group)
            if f"{name}.norm" in keys:
                out.norms[name] = f.get_tensor(f"{name}.norm")
        consumed = {f"{n}.{s}" for n in names for s in ("codes", "scales", "norm")}
        out.dense = {k: f.get_tensor(k) for k in keys if k not in consumed}
    return out


def write_t158(path: str | Path, file: T158File) -> int:
    """Write ``file`` in tern158's format; returns bytes written."""
    from safetensors.torch import save_file

    t: dict[str, torch.Tensor] = {}
    groups = set()
    for name, p in file.layers.items():
        t[f"{name}.codes"] = tn.pack_words_int32(p.qweight).contiguous()
        t[f"{name}.scales"] = (p.scales.reshape(1) if p.group_size == 0 else p.scales.reshape(-1)).float().contiguous()
        groups.add(p.group_size)
    for name, g in file.norms.items():
        t[f"{name}.norm"] = g.float().contiguous()
    t.update({k: v.contiguous() for k, v in file.dense.items()})
    meta = dict(file.meta)
    meta.update({"format": T158_FORMAT, "version": meta.get("version", "1")})
    if "group" not in meta and file.layers:
        first = next(iter(file.layers.values()))
        meta["group"] = str(first.k if first.group_size == -1 else max(first.group_size, 0))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(t, str(path), metadata=meta)
    return Path(path).stat().st_size


def split_rows(p: PackedTernary, sizes: list[int]) -> list[PackedTernary]:
    """Split a packed layer along output rows (rows are independent because scales run along K), e.g. fused qkv -> q, k, v."""
    if sum(sizes) != p.n:
        raise ConfigError(f"row sizes {sizes} do not sum to N={p.n}")
    if p.group_size == 0:
        raise ConfigError("per-tensor scales cannot be split by rows without sharing the scale; re-quantise per row/group first")
    out, r = [], 0
    for s in sizes:
        out.append(PackedTernary(p.qweight[r : r + s].contiguous(), p.scales[r : r + s].contiguous(), p.layout, s, p.k, p.group_size))
        r += s
    return out
