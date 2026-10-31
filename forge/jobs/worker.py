"""Job worker process: ``python -m forge.jobs.worker <job_id>``.

Reads the job row, validates ``spec.params`` against the job type's config model, runs the entrypoint with a
:class:`forge.jobs.context.WorkerContext` and records the final status/outputs/error. SIGTERM (POSIX) or a
``CANCEL`` file in the run dir asks the job to stop cooperatively (``ctx.should_stop()``); the JobManager
escalates to SIGKILL after the grace period.
"""
from __future__ import annotations

import json
import os
import signal
import sys
import threading
import traceback
from pathlib import Path
from typing import Any

from forge.core.errors import Cancelled, ForgeError
from forge.core.util import now
from forge.jobs.context import WorkerContext
from forge.jobs.repo import dumps, to_jsonable
from forge.telemetry.logs import format_line, setup_worker_logging


def _log(msg: str, level: str = "info") -> None:
    print(format_line(msg, level), flush=True)


def _finalize(db: Any, job_id: str, status: str, *, error: str | None, exit_code: int, result: Any, ctx: WorkerContext) -> None:
    row = db.query_one("SELECT status, outputs_json, started_at FROM jobs WHERE id=?", (job_id,)) or {}
    if row.get("status") in ("succeeded", "failed", "cancelled"):
        return  # the manager already decided (e.g. killed)
    try:
        outs = json.loads(row.get("outputs_json") or "{}")
    except ValueError:
        outs = {}
    if result is not None:
        outs["result"] = to_jsonable(result)
    prog = 1.0 if status == "succeeded" else None
    sets = "status=?, finished_at=?, exit_code=?, error=?, outputs_json=?, updated_at=?"
    params: list[Any] = [status, now(), exit_code, error, dumps(outs), now()]
    if prog is not None:
        sets += ", progress=?"
        params.append(prog)
    params.append(job_id)
    db.execute(f"UPDATE jobs SET {sets} WHERE id=?", params)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m forge.jobs.worker <job_id>", file=sys.stderr)
        return 2
    job_id = argv[0]
    from forge.core.db import get_db
    from forge.core.paths import run_dir as run_dir_for
    from forge.core.settings import get_settings
    from forge.jobs.registry import get_job_type
    from forge.telemetry.metrics import get_metrics_store

    setup_worker_logging(get_settings().log_level)
    db = get_db()
    row = db.query_one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if row is None:
        _log(f"job {job_id} not found", "error")
        return 2
    spec = json.loads(row["spec_json"])
    run_dir = Path(row["run_dir"] or run_dir_for(job_id))
    ctx = WorkerContext(job_id, run_dir, spec.get("params") or {}, db=db, metrics=get_metrics_store())

    def _on_term(signum: int, frame: Any) -> None:
        if ctx._stop_flag.is_set():  # second signal -> hard exit path
            raise SystemExit(143)
        _log(f"signal {signum} received: requesting cooperative stop", "warning")
        ctx.request_stop()

    for name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _on_term)
            except (ValueError, OSError):  # pragma: no cover - not main thread
                pass

    # make sure status shows running even if the manager did not set it (manual `python -m forge.jobs.worker`)
    db.execute(
        "UPDATE jobs SET status='running', pid=?, started_at=COALESCE(started_at, ?), updated_at=? "
        "WHERE id=? AND status IN ('queued','running')",
        (os.getpid(), now(), now(), job_id),
    )
    _log(f"worker pid={os.getpid()} job={job_id} type={spec.get('type')} name={spec.get('name')!r}")
    _log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} run_dir={run_dir}")

    status, error, exit_code, result = "succeeded", None, 0, None
    try:
        jt = get_job_type(spec["type"])
        (run_dir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
        if jt.needs_gpu:
            threading.Thread(target=_write_env, args=(run_dir,), daemon=True).start()
        cfg_cls = jt.config_model()
        cfg = cfg_cls.model_validate(spec.get("params") or {}) if hasattr(cfg_cls, "model_validate") else cfg_cls(**(spec.get("params") or {}))
        fn = jt.entrypoint()
        result = fn(cfg, ctx)
        if ctx.should_stop():
            status = "cancelled"
            _log("job stopped on request", "warning")
        else:
            _log("job finished")
    except Cancelled as e:
        status, error, exit_code = "cancelled", None, 0
        _log(f"cancelled: {e}", "warning")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 1
        if ctx.should_stop() or code == 143:
            status, error, exit_code = "cancelled", None, 0
        else:
            status, error, exit_code = ("succeeded", None, 0) if code == 0 else ("failed", f"SystemExit({e.code})", code)
    except BaseException as e:  # noqa: BLE001 - report everything
        status, exit_code = "failed", 1
        tb = traceback.format_exc()
        error = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        if isinstance(e, ForgeError):
            error = f"[{e.code}] {error}"
        print(tb, file=sys.stderr, flush=True)
        _log(f"job failed: {error}", "error")
    finally:
        ctx.close()
    try:
        _finalize(db, job_id, status, error=error, exit_code=exit_code, result=result, ctx=ctx)
    except Exception as e:  # pragma: no cover
        _log(f"could not record final status: {e}", "error")
        return 1
    return exit_code


def _write_env(run_dir: Path) -> None:
    try:
        from forge.core.env import capture_environment

        (run_dir / "env.json").write_text(json.dumps(to_jsonable(capture_environment()), indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover
        pass


if __name__ == "__main__":
    code = main()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:  # pragma: no cover
        pass
    os._exit(code)  # do not wait for non-daemon threads left behind by user code
