"""Model reference resolution shared by every job type that takes a model, plus HF repo-id checks and token lookup.

``resolve_model_ref`` order: an explicit path (absolute, ``./``, ``~``, drive/backslash) -> registered model id / name
(``models`` table) -> an existing directory relative to the CWD -> a directory under ``FORGE_HOME/models`` -> a Hugging Face repo id, but only when the string
looks like ``org/name``. Anything else is an error that says what was tried, instead of a Hub 404 from a bare name."""
from __future__ import annotations

import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FutTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.errors import ConfigError, NotFound

_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]{0,95}/[\w.\-]{1,96}$")
_BAD_DIR_CHARS = re.compile(r'[\/:*?"<>|\x00-\x1f]+')


def get_hf_token(explicit: str | None = None) -> str | None:
    """Hugging Face token: explicit value > ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` env > the Forge settings. Never logged."""
    if explicit:
        return explicit
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        v = (os.environ.get(k) or "").strip()
        if v:
            return v
    try:
        from forge.core.settings import get_settings

        return get_settings().hf_token or None
    except Exception:  # noqa: BLE001
        return None


def looks_like_repo_id(s: str) -> bool:
    return bool(_REPO_RE.match(s)) and ".." not in s


def suggest_repo_ids(query: str, token: str | None = None, limit: int = 8, timeout: float = 6.0) -> list[str]:
    """Best-effort Hub search for repo ids resembling ``query`` (empty on any failure, bounded wait)."""
    def _search() -> list[str]:
        from huggingface_hub import HfApi

        q = query.strip().split("/")[-1]
        return [str(m.id) for m in HfApi().list_models(search=q, limit=limit, sort="downloads", token=token)]

    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-suggest")
    try:
        return ex.submit(_search).result(timeout=timeout)
    except (_FutTimeout, Exception):  # noqa: BLE001
        return []
    finally:
        ex.shutdown(wait=False)


def validate_repo_id(repo_id: str, token: str | None = None, suggest: bool = True) -> str:
    """Return ``repo_id`` if it is shaped like ``org/name``; otherwise raise :class:`ConfigError` with Hub suggestions."""
    rid = (repo_id or "").strip()
    if looks_like_repo_id(rid):
        return rid
    msg = f"{repo_id!r} is not a Hugging Face repo id (expected 'org/name', e.g. 'google/gemma-4-E4B')"
    hits = suggest_repo_ids(rid, token) if (suggest and rid) else []
    if hits:
        msg += "; did you mean: " + ", ".join(hits)
    raise ConfigError(msg, details={"suggestions": hits})


@dataclass
class ResolvedModel:
    kind: str  # local | registry | models_dir | hf
    path: Path | None = None
    repo_id: str | None = None
    row: dict[str, Any] | None = None


def _safe_dir(name: str) -> str | None:
    n = _BAD_DIR_CHARS.sub("--", name).strip().strip(".").strip()
    return n if n and len(n) <= 200 else None


def _registry_lookup(ref: str, repo: Any = None) -> dict[str, Any] | None:
    try:
        if repo is None:
            from forge.jobs.repo import ModelsRepo

            repo = ModelsRepo()
        return repo.get(ref) or repo.find_by_name(ref)
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():  # fresh home, no models table yet: not a registry hit
            return None
        raise ConfigError(f"model registry lookup failed: {e}") from e


def resolve_model_ref(ref: str | os.PathLike, *, repo: Any = None) -> ResolvedModel:
    """Map a user-supplied model string to a local directory or a (validated) Hub repo id. See the module docstring."""
    s = str(ref).strip()
    if not s:
        raise ConfigError("model reference is empty")
    p = Path(s).expanduser()
    path_like = p.is_absolute() or s.startswith((".", "~")) or "\\" in s or bool(re.match(r"^[A-Za-z]:", s))
    if path_like:  # explicit filesystem paths never consult the registry
        if p.exists():
            return ResolvedModel("local", path=p)
        raise NotFound(f"model path {s!r} does not exist")
    row = _registry_lookup(s, repo)
    if row:
        from forge.jobs.modelinfo import model_local_path

        lp = model_local_path(row)
        if lp and Path(lp).is_dir():
            return ResolvedModel("registry", path=Path(lp), row=row)
        if row.get("source") == "hf":
            raise NotFound(
                f"model {row.get('name')!r} ({row.get('id')}) is registered but not downloaded; run its Download first",
                details={"model_id": row.get("id"), "repo": row.get("path_or_repo")},
            )
        raise NotFound(f"model {row.get('name')!r} ({row.get('id')}) has no local files", details={"model_id": row.get("id")})
    if p.exists():  # a bare name / org/name: a registered model wins over a same-named directory in the CWD
        return ResolvedModel("local", path=p)
    d = _safe_dir(s)
    if d:
        try:
            from forge.core.paths import models_dir

            cand = models_dir() / d
            if cand.is_dir():
                return ResolvedModel("models_dir", path=cand)
        except Exception:  # noqa: BLE001
            pass
    if not looks_like_repo_id(s):
        hits = suggest_repo_ids(s, get_hf_token())
        msg = (f"cannot resolve model {s!r}: not a directory, not a registered model (id or name) and not a "
               "Hugging Face repo id ('org/name')")
        if hits:
            msg += "; did you mean: " + ", ".join(hits)
        raise NotFound(msg, details={"suggestions": hits})
    return ResolvedModel("hf", repo_id=s)
