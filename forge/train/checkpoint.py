"""Training checkpoints: atomic local files with retention, optionally mirrored to the content-addressed store.

Local layout (``<dir>/step_00000100/{state.pt, meta.json}``, markers ``BEST`` / ``PRE_ANNEAL`` / ``FINAL`` hold a step number).
The store mirror (delta-chained keyframes + chunk dedup, Plan 8.3) uses ``forge.store.checkpoints.CheckpointWriter`` lazily; it is
best-effort -- any failure is logged once and disables the mirror, local checkpoints stay authoritative.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from forge.core.errors import NotFound
from forge.train.config import CheckpointConfig

log = logging.getLogger("forge.train.checkpoint")

__all__ = ["CheckpointManager", "flatten_tensors"]


def flatten_tensors(obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    """Flatten tensors inside nested dict/list structures to ``{'a/b/0': tensor}`` (for the store mirror)."""
    out: dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        out[prefix or "t"] = obj.detach().cpu()
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_tensors(v, f"{prefix}/{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten_tensors(v, f"{prefix}/{i}" if prefix else str(i)))
    return out


def _step_dir(root: Path, step: int) -> Path:
    return root / f"step_{step:08d}"


class CheckpointManager:
    def __init__(self, root: str | Path, cfg: CheckpointConfig, *, store: Any = None, run_id: str = "") -> None:
        self.root = Path(root)
        self.cfg = cfg
        self.store = store
        self.run_id = run_id
        self._writer: Any = None
        self._mirror_ok = bool(cfg.use_store and store is not None)
        self._n_saved = 0

    # -- queries ---------------------------------------------------------------------------------------
    def steps(self) -> list[int]:
        out = []
        for p in self.root.glob("step_*"):
            if (p / "meta.json").exists():
                try:
                    out.append(int(p.name.split("_")[1]))
                except ValueError:
                    pass
        return sorted(out)

    def _marker(self, name: str) -> int | None:
        f = self.root / name
        try:
            return int(f.read_text().strip())
        except Exception:  # noqa: BLE001
            return None

    def meta(self, step: int) -> dict[str, Any]:
        return json.loads((_step_dir(self.root, step) / "meta.json").read_text(encoding="utf-8"))

    def latest(self) -> int | None:
        s = self.steps()
        return s[-1] if s else None

    def last_good(self) -> int | None:
        for s in reversed(self.steps()):
            try:
                if self.meta(s).get("good", True):
                    return s
            except Exception:  # noqa: BLE001
                continue
        return None

    def best(self) -> int | None:
        return self._marker("BEST")

    def pre_anneal(self) -> int | None:
        return self._marker("PRE_ANNEAL")

    # -- save ---------------------------------------------------------------------------------------------
    def save(self, step: int, state: dict[str, Any], *, good: bool = True, best: bool = False, pre_anneal: bool = False,
             final: bool = False, meta: dict[str, Any] | None = None) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        d = _step_dir(self.root, step)
        tmp = self.root / f".tmp_step_{step:08d}_{int(time.time() * 1000)}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(state, tmp / "state.pt")
            m = {"step": int(step), "good": bool(good), "time": time.time(), "run_id": self.run_id, **(meta or {})}
            (tmp / "meta.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
            if d.exists():
                shutil.rmtree(d)
            tmp.rename(d)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        if best:
            (self.root / "BEST").write_text(str(step))
        if pre_anneal:
            (self.root / "PRE_ANNEAL").write_text(str(step))
        if final:
            (self.root / "FINAL").write_text(str(step))
        self._n_saved += 1
        self._mirror(step, state, m)
        self._retain()
        return d

    def _mirror(self, step: int, state: dict[str, Any], meta: dict[str, Any]) -> None:
        if not self._mirror_ok:
            return
        try:
            if self._writer is None:
                from forge.store.checkpoints import CheckpointWriter  # type: ignore

                self._writer = CheckpointWriter(self.store, self.run_id or "default", keyframe_every=self.cfg.keyframe_every)
            m = {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool, type(None)))}
            ev = meta.get("eval_loss")
            self._writer.save(step, state, m, keyframe_every=self.cfg.keyframe_every, metric=float(ev) if isinstance(ev, (int, float)) else None)
        except Exception as e:  # noqa: BLE001
            log.warning("checkpoint store mirror disabled after error: %r", e)
            self._mirror_ok = False

    def _retain(self) -> None:
        steps = self.steps()
        protect = {s for s in (self.best(), self.pre_anneal(), self._marker("FINAL")) if s is not None}
        periodic = [s for s in steps if s not in protect]
        for s in periodic[: max(0, len(periodic) - self.cfg.keep_last)]:
            shutil.rmtree(_step_dir(self.root, s), ignore_errors=True)

    # -- load ---------------------------------------------------------------------------------------------
    def resolve(self, which: str | int | Path) -> int | Path:
        """Resolve 'latest' | 'best' | 'pre_anneal' | 'good' | <step> | <path> to a step number or a checkpoint directory."""
        if isinstance(which, int):
            return which
        w = str(which)
        if w in ("latest", "auto"):
            s = self.latest()
        elif w == "best":
            s = self.best()
        elif w == "pre_anneal":
            s = self.pre_anneal()
        elif w == "good":
            s = self.last_good()
        elif w.isdigit():
            s = int(w)
        else:
            p = Path(w)
            if p.is_dir() and (p / "state.pt").exists():
                return p
            if (p / "meta.json").exists() or p.suffix == ".pt":
                return p if p.is_dir() else p.parent
            raise NotFound(f"checkpoint not found: {which}")
        if s is None:
            raise NotFound(f"no checkpoint for {which!r} in {self.root}")
        return s

    def load(self, which: str | int | Path = "latest") -> tuple[dict[str, Any], dict[str, Any]]:
        r = self.resolve(which)
        d = r if isinstance(r, Path) else _step_dir(self.root, r)
        if not (d / "state.pt").exists():
            raise NotFound(f"checkpoint missing: {d}")
        state = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        return state, meta
