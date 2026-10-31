"""``python -m torch.distributed.run ... -m forge.train.launch --kind train --config cfg.json --run-dir DIR``

Per-process entry used by :func:`forge.train.dist.launch_torchrun` and for manual multi-GPU runs.  Rank 0 writes
``events.jsonl`` (metrics / progress / outputs, tailed by the launcher or by you) and ``result.json``; the run is cancelled
cooperatively by creating ``<run-dir>/STOP``.  Not executed in this repository's CPU tests beyond single-process mode.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


class FileContext:
    """RunContext writing events to ``events.jsonl`` and stopping on ``STOP`` (works without the Forge job system)."""

    def __init__(self, run_dir: Path, run_id: str = "") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or self.run_dir.name
        self.params: dict = {}
        self._events = self.run_dir / "events.jsonl"
        self._store: Any = None
        self._store_tried = False

    def _w(self, ev: dict[str, Any]) -> None:
        with open(self._events, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, default=str) + "\n")

    def log(self, msg: str, level: str = "info") -> None:
        print(f"[{level}] {msg}", flush=True)

    def metric(self, step: int, **values: float) -> None:
        self._w({"t": "metric", "step": int(step), "values": {k: float(v) for k, v in values.items()}})

    def progress(self, frac: float, msg: str = "") -> None:
        self._w({"t": "progress", "frac": float(frac), "msg": msg})

    def should_stop(self) -> bool:
        return (self.run_dir / "STOP").exists()

    def set_output(self, key: str, value: Any) -> None:
        self._w({"t": "output", "key": key, "value": value})

    @property
    def store(self) -> Any:
        if not self._store_tried:
            self._store_tried = True
            try:
                from forge.store import get_store

                self._store = get_store()
            except Exception:  # noqa: BLE001
                self._store = None
        return self._store


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="forge.train.launch")
    ap.add_argument("--kind", required=True, choices=["train", "finetune", "distill"])
    ap.add_argument("--config", required=True, help="JSON or YAML config file")
    ap.add_argument("--run-dir", required=True)
    ns = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    text = Path(ns.config).read_text(encoding="utf-8")
    if ns.config.endswith((".yaml", ".yml")):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    from forge.train import config as C
    from forge.train import run as R

    cls, fn = {"train": (C.TrainConfig, R.run_training), "finetune": (C.FinetuneConfig, R.run_finetune), "distill": (C.DistillConfig, R.run_distill)}[ns.kind]
    cfg = cls.model_validate(raw)
    # the launcher must not re-launch: mark this process as already distributed-aware
    rank = int(os.environ.get("RANK", "0"))
    ctx = FileContext(Path(ns.run_dir))
    res = fn(cfg, ctx)
    if rank == 0:
        (Path(ns.run_dir) / "result.json").write_text(json.dumps(res, default=str, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
