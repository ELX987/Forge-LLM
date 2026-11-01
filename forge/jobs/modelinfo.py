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


def unsupported_architecture_message(hf: dict[str, Any], err: Exception | None = None) -> str:
    """Human-readable 'Forge cannot run this architecture' text naming the top-level type, the HF class and what is supported."""
    tc = hf.get("text_config") if isinstance(hf.get("text_config"), dict) else {}
    top = hf.get("model_type")
    inner = tc.get("model_type")
    archs = ", ".join(hf.get("architectures") or []) or "unknown class"
    name = f"{top} / {inner}" if inner and inner != top else str(top)
    try:
        from forge.models.config import SUPPORTED_ARCHS

        sup = ", ".join(SUPPORTED_ARCHS)
    except Exception:  # noqa: BLE001
        sup = "see docs/ARCHITECTURE.md"
    why = f" ({err})" if err is not None and not str(err).startswith("unsupported model_type") else ""
    return f"architecture '{name}' ({archs}) is not supported by Forge's own model code{why}; supported: {sup}. Jobs that load it will fail."


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
        "unsupported_reason": None,
    }
    try:  # packed Forge artifacts describe themselves (layer kinds, bits, group size, layout, KV scheme)
        from forge.core.artifactmeta import read_artifact_summary

        qs = read_artifact_summary(p)
        if qs and qs.get("is_quantized"):
            info["quantization"] = qs
    except Exception:  # noqa: BLE001
        pass
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
            if not isinstance(e, ImportError):
                info["unsupported_reason"] = unsupported_architecture_message(hf, e)
                warnings.insert(0, info["unsupported_reason"])
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


# ------------------------------------------------------------------------------------------------ compress outputs as models
def _unique_name(repo: ModelsRepo, want: str, path: str, suffix: str) -> str:
    """``want`` unless another registered model already uses that display name for a different path."""
    other = repo.find_by_name(want)
    if other is None or (other.get("info") or {}).get("local_path") == path or other.get("path_or_repo") == path:
        return want
    return f"{want}-{suffix}"[:200]


def register_artifact_dir(
    path: str | Path,
    *,
    artifact_hash: str | None = None,
    name: str | None = None,
    job_id: str | None = None,
    bytes_input: int | None = None,
    models: ModelsRepo | None = None,
) -> dict[str, Any]:
    """Register a compress output directory (``config.json`` + ``forge_compress.json`` + packed safetensors) in the ``models`` table so it is
    selectable for serving: ``source='artifact'``, ``hash`` = the artifact-store digest, ``info.kind='quantized'`` and ``info.quantization``
    (label such as ``W4A16 g128``, layer kinds, bits, group size, layout id, KV scheme), ``info.base_model`` (what was compressed, linked to a
    registered row as ``info.base_model_id`` when one matches), ``size_bytes`` = packed weight bytes. Idempotent per digest / directory."""
    from forge.core.artifactmeta import read_artifact_summary

    repo = models or ModelsRepo()
    p = Path(path).expanduser().resolve()
    summ = read_artifact_summary(p)
    if summ is None or not (summ["is_quantized"] or summ["is_compress_artifact"]):
        raise ConfigError(f"{p} is not a Forge compress artifact (needs config.json and forge_compress.json / a quantisation manifest)")
    info = inspect_model_dir(p)
    ps = str(p)
    base = summ.get("base_model")
    base_row = None
    if base:
        base_row = repo.find_by_repo(str(base)) or repo.find_by_path(str(base)) or repo.find_by_name(str(base))
    label = summ.get("label") or "dense"
    dname = name or summ.get("name") or p.parent.name or p.name
    existing = (repo.find_by_hash(artifact_hash) if artifact_hash else None) or repo.find_by_path(ps)
    suffix = (job_id or artifact_hash or p.parent.name or "x").split(":")[-1][:8]
    if existing is None:
        dname = _unique_name(repo, dname, ps, suffix)
    size = info.get("weight_bytes") or _dir_size(p)
    out_info: dict[str, Any] = {
        **info, "local_path": ps, "kind": "quantized" if summ["is_quantized"] else "model", "quantization": summ,
        "base_model": base, "base_model_id": base_row["id"] if base_row else None, "job_id": job_id, "artifact": artifact_hash,
        "weight_gb": round(size / 1e9, 3),
    }
    if bytes_input:
        out_info["bytes_input"] = int(bytes_input)
        out_info["compression"] = round(bytes_input / max(size, 1), 3)
    fields: dict[str, Any] = {
        "name": dname, "source": "artifact", "path_or_repo": ps, "hash": artifact_hash, "licence": summ.get("licence") or info.get("licence"),
        "params_b": summ.get("params_b") or info.get("params_b"), "status": "ready", "size_bytes": size, "info": out_info,
        "tags": ["quantized"] + ([label] if summ["is_quantized"] else []),
    }
    if base_row and base_row.get("distill_allowed") is not None:
        fields["distill_allowed"] = base_row["distill_allowed"]
    if existing:
        if existing.get("name") and not name:
            fields.pop("name")
        return repo.update(existing["id"], **fields)
    t = now()
    return repo.insert(id=new_id("mdl"), created_at=t, updated_at=t, notes=f"compress output of {base}" if base else "compress output", **fields)


def import_artifacts(*, models: ModelsRepo | None = None, jobs: Any = None, extra_dirs: list[str | Path] | None = None) -> dict[str, Any]:
    """Backfill: register every compress output found on disk. Sources: succeeded ``compress`` jobs (``outputs.model_dir`` / ``outputs.artifact``),
    ``FORGE_HOME/runs/*/artifact`` and ``extra_dirs``. Returns ``{imported: [rows], updated: n, skipped: [{path, reason}]}``; never raises for one bad dir."""
    from forge.core.paths import runs_dir

    repo = models or ModelsRepo()
    cands: dict[str, dict[str, Any]] = {}

    def add(d: Any, **kw: Any) -> None:
        try:
            rp = str(Path(str(d)).expanduser().resolve())
        except OSError:
            return
        cands.setdefault(rp, {}).update({k: v for k, v in kw.items() if v})

    try:
        if jobs is None:
            from forge.jobs.repo import JobsRepo

            jobs = JobsRepo()
        for j in jobs.list({"type": "compress", "status": "succeeded"}, limit=2000):
            o = j.get("outputs") or {}
            md = o.get("model_dir") or (o.get("summary") or {}).get("model_dir")
            if md:
                add(md, job_id=j["id"], artifact_hash=o.get("artifact") or (o.get("summary") or {}).get("artifact"),
                    bytes_input=o.get("bytes_input"), name=(o.get("summary") or {}).get("name"))
    except Exception:  # noqa: BLE001 - a missing jobs table must not stop the disk scan
        pass
    try:
        for d in sorted(runs_dir().glob("*/artifact")):
            add(d, job_id=d.parent.name)
    except OSError:
        pass
    for d in extra_dirs or []:
        add(d)
    imported: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    updated = 0
    for rp, kw in cands.items():
        pth = Path(rp)
        if not (pth / "forge_compress.json").is_file():
            skipped.append({"path": rp, "reason": "no forge_compress.json (not a compress artifact, or the run did not finish)"})
            continue
        try:
            was = repo.find_by_path(rp) is not None
            row = register_artifact_dir(pth, models=repo, **kw)
            updated += 1 if was else 0
            if not was:
                imported.append(row)
        except Exception as e:  # noqa: BLE001
            skipped.append({"path": rp, "reason": f"{type(e).__name__}: {e}"})
    return {"imported": imported, "updated": updated, "skipped": skipped, "scanned": len(cands)}
