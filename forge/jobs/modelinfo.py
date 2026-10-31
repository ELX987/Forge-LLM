"""Model directory inspection + registration (shared by the API, CLI and the model_download job)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from forge.core.errors import ConfigError, NotFound
from forge.core.util import new_id, now
from forge.jobs.repo import ModelsRepo

WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.gguf")


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def estimate_from_hf_config(hf: dict[str, Any]) -> dict[str, Any]:
    """Parameter/KV estimate straight from a HF ``config.json`` (used when ``forge.models`` is unavailable
    or does not support the architecture). Dense and simple-MoE decoder-only transformers."""
    tc = hf.get("text_config") if isinstance(hf.get("text_config"), dict) else hf
    H = int(tc.get("hidden_size") or tc.get("n_embd") or tc.get("d_model") or 0)
    L = int(tc.get("num_hidden_layers") or tc.get("n_layer") or tc.get("num_layers") or 0)
    nh = int(tc.get("num_attention_heads") or tc.get("n_head") or 0)
    nkv = int(tc.get("num_key_value_heads") or nh or 0)
    hd = int(tc.get("head_dim") or (H // nh if nh else 0))
    inter = int(tc.get("intermediate_size") or tc.get("ffn_dim") or 4 * H)
    V = int(tc.get("vocab_size") or 0)
    experts = int(tc.get("num_local_experts") or tc.get("num_experts") or tc.get("n_routed_experts") or 0)
    moe_inter = int(tc.get("moe_intermediate_size") or inter)
    if not (H and L and nh):
        raise ConfigError("config.json lacks hidden_size / num_hidden_layers / num_attention_heads")
    attn = H * nh * hd + 2 * H * nkv * hd + nh * hd * H
    mlp = 3 * H * inter if not experts else experts * 3 * H * moe_inter + H * experts
    per_layer = attn + mlp + 2 * H
    emb = V * H
    total = L * per_layer + emb + (0 if tc.get("tie_word_embeddings", False) else emb) + H
    kv = {bits: 2 * L * nkv * hd * bits / 8 for bits in (16, 8, 4)}
    return {"params": int(total), "kv_bytes_per_token": kv, "estimated": True, "layers": L, "hidden": H, "heads": nh, "kv_heads": nkv, "head_dim": hd}


def _read_licence(p: Path) -> str | None:
    readme = p / "README.md"
    if readme.exists():
        try:
            head = readme.read_text(encoding="utf-8", errors="replace")[:4000]
            m = re.search(r"^license:\s*([^\n]+)$", head, re.M)
            if m:
                return m.group(1).strip().strip("'\"")
        except OSError:
            pass
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        f = p / name
        if f.exists():
            try:
                first = f.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0].strip()
                return first[:120]
            except (OSError, IndexError):
                return "see LICENSE file"
    return None


def inspect_model_dir(path: str | Path) -> dict[str, Any]:
    """Summarise a local HF-style model directory. Raises :class:`NotFound` / :class:`ConfigError`."""
    p = Path(path).expanduser()
    if not p.exists():
        raise NotFound(f"path does not exist: {p}", details={"path": str(p)})
    if p.is_file():
        raise ConfigError(f"{p} is a file; register a model directory (with config.json and weights)")
    cfg_file = p / "config.json"
    if not cfg_file.exists():
        raise ConfigError(f"{p} has no config.json (expected a HF-format model directory)")
    try:
        hf = json.loads(cfg_file.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ConfigError(f"config.json is not valid JSON: {e}") from e
    warnings: list[str] = []
    weight_files = sorted({f.name for pat in WEIGHT_PATTERNS for f in p.glob(pat)})
    weight_bytes = sum((p / n).stat().st_size for n in weight_files)
    if not weight_files:
        warnings.append("no weight files found (*.safetensors / *.bin / *.gguf)")
    info: dict[str, Any] = {
        "path": str(p),
        "model_type": hf.get("model_type"),
        "architectures": hf.get("architectures"),
        "dtype": hf.get("torch_dtype") or hf.get("dtype"),
        "quantization_config": hf.get("quantization_config"),
        "max_position_embeddings": hf.get("max_position_embeddings"),
        "vocab_size": hf.get("vocab_size"),
        "weight_files": weight_files[:200],
        "weight_bytes": weight_bytes,
        "has_tokenizer": any((p / n).exists() for n in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")),
        "licence": _read_licence(p),
        "forge_supported": False,
    }
    # --- preferred: forge.models.ModelConfig --------------------------------------------------
    try:
        from forge.models import ModelConfig  # type: ignore[attr-defined]

        mc = ModelConfig.from_hf_config(hf)
        info["forge_supported"] = True
        info["params"] = int(mc.param_count())
        info["kv_bytes_per_token"] = {str(b): float(mc.kv_bytes_per_token(b)) for b in (16, 8, 4)}
        info["layers"] = int(mc.num_layers)
        info["hidden"] = int(mc.hidden_size)
        info["heads"] = int(mc.num_heads)
        info["kv_heads"] = int(mc.num_kv_heads)
        info["head_dim"] = int(mc.head_dim)
        info["moe"] = bool(getattr(mc, "moe", None))
        info["estimated"] = False
        info["model_config"] = mc.model_dump(mode="json") if hasattr(mc, "model_dump") else None
    except Exception as e:
        try:
            est = estimate_from_hf_config(hf)
            info["params"] = est["params"]
            info["kv_bytes_per_token"] = {str(k): v for k, v in est["kv_bytes_per_token"].items()}
            for k in ("layers", "hidden", "heads", "kv_heads", "head_dim"):
                info[k] = est[k]
            info["estimated"] = True
            reason = "forge.models not importable" if isinstance(e, ImportError) else f"forge.models rejected config ({e})"
            warnings.append(f"parameter count estimated from config.json: {reason}")
        except ConfigError as e2:
            warnings.append(str(e2))
    if info.get("params"):
        info["params_b"] = round(info["params"] / 1e9, 4)
    info["warnings"] = warnings
    return info


def register_local_model(
    path: str | Path,
    *,
    name: str | None = None,
    source: str = "local",
    hf_repo: str | None = None,
    revision: str | None = None,
    licence: str | None = None,
    distill_allowed: bool | None = None,
    notes: str = "",
    models: ModelsRepo | None = None,
) -> dict[str, Any]:
    """Inspect ``path`` and insert (or refresh) its row in the ``models`` table. Returns the row.

    For ``source="hf"`` downloads, ``hf_repo`` is the repo id and the row's ``path_or_repo`` stays the repo id
    while ``info.local_path`` points at the downloaded directory."""
    repo = models or ModelsRepo()
    info = inspect_model_dir(path)
    p = str(Path(path).expanduser().resolve())
    info["path"] = p
    key = hf_repo if source == "hf" and hf_repo else p
    existing = repo.find_by_repo(hf_repo, revision) if source == "hf" and hf_repo else repo.find_by_path(p)
    fields: dict[str, Any] = {
        "name": name or (hf_repo.split("/")[-1] if hf_repo else Path(p).name),
        "source": source,
        "path_or_repo": key,
        "revision": revision,
        "licence": licence or info.get("licence"),
        "distill_allowed": distill_allowed,
        "params_b": info.get("params_b"),
        "status": "ready",
        "size_bytes": info.get("weight_bytes") or _dir_size(Path(p)),
        "info": {**info, "local_path": p},
        "notes": notes,
    }
    if existing:
        if existing.get("name") and not name:
            fields.pop("name")
        if distill_allowed is None:
            fields.pop("distill_allowed")
        return repo.update(existing["id"], **fields)
    t = now()
    return repo.insert(id=new_id("mdl"), created_at=t, updated_at=t, tags=[], **fields)


def model_local_path(row: dict[str, Any]) -> str | None:
    """Directory holding the model files for a ``models`` row (None for a not-yet-downloaded HF repo)."""
    info = row.get("info") or {}
    if info.get("local_path"):
        return info["local_path"]
    if row.get("source") == "local":
        return row.get("path_or_repo")
    return None
