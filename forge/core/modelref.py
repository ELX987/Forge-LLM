"""Model reference resolution shared by every job type that takes a model, plus HF repo-id checks and token lookup.

``resolve_model_ref`` order: an explicit path (absolute, ``./``, ``~``, drive/backslash) -> a content digest of the artifact store
(``blake3:...``; looked up in the ``models`` table first, otherwise materialised from the store) -> registered model id / name
(``models`` table; includes compress outputs registered as quantised artifacts) -> an existing directory relative to the CWD -> a directory under
``FORGE_HOME/models`` -> a Hugging Face repo id, but only when the string looks like ``org/name``. Anything else is an error that says what was tried,
instead of a Hub 404 from a bare name. Every local hit also carries ``ResolvedModel.artifact`` (the quantisation summary read from the directory's
own metadata, ``None`` for a plain model) so callers can detect a packed W4A16 artifact without guessing from names."""
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
    kind: str  # local | registry | models_dir | artifact | hf
    path: Path | None = None
    repo_id: str | None = None
    row: dict[str, Any] | None = None
    artifact: dict[str, Any] | None = None  # forge.core.artifactmeta.read_artifact_summary(path); None for plain models / hub repos

    @property
    def is_quantized(self) -> bool:
        return bool(self.artifact and self.artifact.get("is_quantized"))


def _with_summary(r: ResolvedModel) -> ResolvedModel:
    if r.path is not None and r.artifact is None:
        try:
            from forge.core.artifactmeta import read_artifact_summary

            r.artifact = read_artifact_summary(r.path)
        except Exception:  # noqa: BLE001 - metadata is advisory here; the loader verifies what it really builds
            r.artifact = None
    return r


def _digest_lookup(ref: str, repo: Any = None) -> ResolvedModel | None:
    """A store digest -> the registered artifact row (by ``hash``) or, failing that, a directory materialised from the artifact store."""
    try:
        if repo is None:
            from forge.jobs.repo import ModelsRepo

            repo = ModelsRepo()
        row = repo.find_by_hash(ref) if hasattr(repo, "find_by_hash") else None
    except sqlite3.OperationalError:
        row = None
    if row:
        from forge.jobs.modelinfo import model_local_path

        lp = model_local_path(row)
        if lp and Path(lp).is_dir():
            return ResolvedModel("registry", path=Path(lp), row=row)
    try:
        from forge.core.paths import models_dir
        from forge.jobs.repo import open_store

        st = open_store()
        if not st.has(ref):
            return None
        dest = models_dir() / "artifacts" / ref.split(":", 1)[-1][:24]
        _materialize_digest(st, ref, dest)
        return ResolvedModel("artifact", path=dest, row=row)
    except Exception as e:  # noqa: BLE001
        raise NotFound(f"artifact {ref} is in neither the model registry nor a loadable directory of the artifact store ({type(e).__name__}: {e})",
                       details={"artifact": ref}) from e


_DIGEST_MARK = ".forge_digest"


def _materialize_digest(st: Any, ref: str, dest: Path) -> None:
    """Make ``dest`` a verified, complete materialisation of artifact ``ref``. An existing ``dest`` is reused only when its marker names exactly this
    digest and it still has a config.json; otherwise it is replaced (a stale / foreign / half-written directory is removed first, because
    renaming a directory over an existing one fails on Windows)."""
    import shutil

    mark = dest / _DIGEST_MARK
    try:
        if (dest / "config.json").is_file() and mark.is_file() and mark.read_text(encoding="utf-8").strip() == ref:
            return
    except OSError:
        pass
    if not st.verify(ref):
        raise NotFound(f"artifact {ref} failed verification in the artifact store; it is not materialised", details={"artifact": ref})
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    st.materialize_artifact(ref, tmp)
    (tmp / _DIGEST_MARK).write_text(ref, encoding="utf-8")
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
        if dest.exists():
            raise ConfigError(f"cannot replace the stale artifact directory {dest}; remove it and retry")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp.replace(dest)


def _local_hit(p: Path, kind: str = "local") -> ResolvedModel:
    """A resolved filesystem location. A directory without config.json that has a model in ``<dir>/artifact`` (a job run directory) is refused with a
    pointer to that sub-directory instead of failing later in the loader; any other directory is returned as is (the loader reports what is missing)."""
    if p.is_dir() and not (p / "config.json").is_file() and (p / "artifact" / "config.json").is_file():
        hint = p / "artifact"
        raise NotFound(f"{str(p)!r} is not a model directory (no config.json); the model is in {str(hint)!r} - use that path",
                       details={"path": str(p), "suggestion": str(hint)})
    return _with_summary(ResolvedModel(kind, path=p))


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
            return _local_hit(p)
        raise NotFound(f"model path {s!r} does not exist")
    from forge.core.hashing import is_digest

    if is_digest(s):
        hit = _digest_lookup(s, repo)
        if hit is not None:
            return _with_summary(hit)
        raise NotFound(f"artifact {s} not found in the model registry or the artifact store", details={"artifact": s})
    row = _registry_lookup(s, repo)
    if row:
        from forge.jobs.modelinfo import model_local_path

        lp = model_local_path(row)
        if lp and Path(lp).is_dir():
            return _with_summary(ResolvedModel("registry", path=Path(lp), row=row))
        if row.get("source") == "hf":
            raise NotFound(
                f"model {row.get('name')!r} ({row.get('id')}) is registered but not downloaded; run its Download first",
                details={"model_id": row.get("id"), "repo": row.get("path_or_repo")},
            )
        raise NotFound(f"model {row.get('name')!r} ({row.get('id')}) has no local files", details={"model_id": row.get("id")})
    if p.exists():  # a bare name / org/name: a registered model wins over a same-named directory in the CWD
        return _local_hit(p)
    d = _safe_dir(s)
    if d:
        try:
            from forge.core.paths import models_dir

            cand = models_dir() / d
            if cand.is_dir():
                return _with_summary(ResolvedModel("models_dir", path=cand))
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
