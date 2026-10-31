"""JobManager: schedules queued jobs onto worker subprocesses.

* one thread (``forge-job-manager``) polls the ``jobs`` table every ``poll_s`` seconds: reaps finished
  workers, escalates cancellations (SIGTERM -> grace -> SIGKILL of the whole process tree) and starts queued jobs
* GPUs are assigned exclusively (``CUDA_VISIBLE_DEVICES``); a job with explicit ``spec.gpus`` waits until all of
  them are free, a GPU job type with ``gpus=None`` gets the first free GPU (if the host has GPUs at all),
  ``gpus=[]`` forces CPU. ``max_concurrent`` limits non-"light" jobs
* every worker runs ``python -m forge.jobs.worker <id>`` in its own process group / session, stdout+stderr go
  to ``runs/<id>/log.txt``
* on start, jobs left ``running`` whose pid is gone are marked ``failed``; live ones are adopted and monitored
"""
from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge.core.errors import ConfigError, ForgeError, NotFound
from forge.core.jsonschema import validate_params
from forge.core.paths import run_dir as run_dir_for
from forge.core.schemas import JobSpec
from forge.core.util import now
from forge.jobs.context import CANCEL_FILE
from forge.jobs.registry import JobTypeSpec, get_job_type
from forge.jobs.repo import TERMINAL, JobsRepo, dumps

log = logging.getLogger("forge.jobs")


class InvalidParams(ConfigError):
    """Job params failed validation; ``details["errors"]`` is a list of ``{loc, msg, type}``."""

    code = "invalid_params"
    http_status = 422


class JobStateError(ForgeError):
    """The requested operation is not valid in the job's current state (HTTP 409)."""

    code = "job_state"
    http_status = 409


class _PopenHandle:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.pid = proc.pid

    def poll(self) -> int | None:
        return self.proc.poll()


class _PidHandle:
    """Handle to a worker started by a previous API process (no exit code available)."""

    def __init__(self, pid: int, job_id: str):
        self.pid = pid
        self.job_id = job_id

    def poll(self) -> int | None:
        return None if _is_worker_alive(self.pid, self.job_id) else -1


def _is_worker_alive(pid: int | None, job_id: str) -> bool:
    if not pid:
        return False
    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        p = psutil.Process(pid)
        if p.status() == psutil.STATUS_ZOMBIE:
            return False
        cmd = " ".join(p.cmdline())
        return "forge.jobs.worker" in cmd and job_id in cmd
    except Exception:
        return False


def kill_process_tree(pid: int, sig: int | None = None) -> None:
    """Terminate ``pid`` and all descendants (``sig=None``: hard kill)."""
    try:
        import psutil
    except ImportError:  # pragma: no cover
        try:
            os.kill(pid, sig or signal.SIGTERM)
        except OSError:
            pass
        return
    try:
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.Error:
        return
    for p in procs:
        try:
            if sig is None:
                p.kill()
            else:
                p.send_signal(sig)
        except psutil.Error:
            pass


class JobManager:
    def __init__(
        self,
        *,
        repo: JobsRepo | None = None,
        metrics: Any = None,
        max_concurrent: int | None = None,
        gpu_provider: Callable[[], list[int]] | None = None,
        grace_s: float | None = None,
        poll_s: float = 0.3,
        env_extra: dict[str, str] | None = None,
    ) -> None:
        from forge.core.settings import get_settings

        st = get_settings()
        self.repo = repo or JobsRepo()
        self._metrics = metrics
        self.max_concurrent = max(1, int(max_concurrent if max_concurrent is not None else st.max_concurrent_jobs))
        self.grace_s = float(grace_s if grace_s is not None else st.job_kill_grace_s)
        self.poll_s = float(poll_s)
        self._gpu_provider = gpu_provider
        self.env_extra = dict(env_extra or {})
        self._procs: dict[str, Any] = {}
        self._deadline: dict[str, float] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._recovered = False
        self._gpu_cache: tuple[float, list[int]] = (0.0, [])

    # -- lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.recover()
        self._thread = threading.Thread(target=self._loop, name="forge-job-manager", daemon=True)
        self._thread.start()

    def stop(self, *, kill_running: bool = False) -> None:
        """Stop the scheduling thread. Workers keep running unless ``kill_running`` (they are re-adopted on restart)."""
        self._stop.set()
        self._wake.set()
        if self._thread and self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)
        self._thread = None
        if kill_running:
            for h in list(self._procs.values()):
                kill_process_tree(h.pid)
                try:
                    h.proc.wait(timeout=5)  # type: ignore[union-attr]
                except Exception:
                    pass
            self._reap()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # pragma: no cover - keep the scheduler alive
                log.exception("job manager tick failed")
            self._wake.wait(self.poll_s)
            self._wake.clear()

    def wake(self) -> None:
        self._wake.set()

    # -- GPU inventory ----------------------------------------------------------------------
    def gpu_indices(self) -> list[int]:
        if self._gpu_provider is not None:
            return list(self._gpu_provider())
        t, cached = self._gpu_cache
        if time.time() - t < 15:
            return cached
        try:
            from forge.telemetry.system import sample_gpus

            idx = sorted(int(g["index"]) for g in sample_gpus() if "index" in g)
        except Exception:
            idx = []
        self._gpu_cache = (time.time(), idx)
        return idx

    def busy_gpus(self) -> set[int]:
        busy: set[int] = set()
        for r in self.repo.running():
            for g in r.get("gpus") or []:
                busy.add(int(g))
        return busy

    def running_jobs(self) -> list[tuple[str, list[int] | None]]:
        return [(r["id"], r.get("gpus")) for r in self.repo.running()]

    # -- submit / cancel / delete ---------------------------------------------------------------
    def validate(self, spec: JobSpec) -> tuple[JobTypeSpec, dict[str, Any]]:
        """Resolve the job type and validate params -> ``(type_spec, clean_params)``. Raises NotFound,
        :class:`UnavailableModule` (501) or :class:`InvalidParams` (422)."""
        jt = get_job_type(spec.type)
        cls = jt.config_model()
        if not hasattr(cls, "model_validate"):
            return jt, dict(spec.params)
        clean, errors = validate_params(cls, spec.params)
        if errors:
            raise InvalidParams(
                f"invalid parameters for job type '{spec.type}': " + "; ".join(f"{e['loc'] or '<root>'}: {e['msg']}" for e in errors[:5]),
                details={"errors": errors, "job_type": spec.type},
            )
        return jt, clean or {}

    def submit(self, spec: JobSpec, *, validate: bool = True) -> dict[str, Any]:
        """Validate and enqueue. Returns the job row."""
        if validate:
            _jt, clean = self.validate(spec)
            spec = spec.model_copy(update={"params": clean})
        else:
            get_job_type(spec.type)
        if spec.gpus is not None:
            if any((not isinstance(g, int)) or g < 0 for g in spec.gpus):
                raise ConfigError("gpus must be non-negative integers")
            known = self.gpu_indices()
            bad = [g for g in spec.gpus if known and g not in known]
            if bad:
                raise ConfigError(f"GPU index {bad} not present on this host (available: {known})")
        jid = None
        from forge.core.util import new_id

        jid = new_id("job")
        rd = run_dir_for(jid)
        row = self.repo.create(spec, run_dir=str(rd), job_id=jid)
        (rd / "log.txt").touch()
        log.info("job %s (%s) queued", jid, spec.type)
        self.wake()
        return row

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.repo.require(job_id, "job")
            st = row["status"]
            if st in TERMINAL:
                return row
            if st == "queued":
                return self.repo.update(job_id, status="cancelled", finished_at=now(), cancel_requested=1)
            # running
            rd = Path(row["run_dir"] or run_dir_for(job_id))
            rd.mkdir(parents=True, exist_ok=True)
            (rd / CANCEL_FILE).write_text(str(now()), encoding="utf-8")
            self._deadline.setdefault(job_id, time.time() + self.grace_s)
            row = self.repo.update(job_id, cancel_requested=1)
            pid = row.get("pid")
            if pid and os.name == "posix":
                try:
                    os.killpg(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            self.wake()
            return row

    def delete(self, job_id: str, *, purge_files: bool = True) -> None:
        row = self.repo.require(job_id, "job")
        if row["status"] in ("queued", "running"):
            raise JobStateError(f"job {job_id} is {row['status']}; cancel it first")
        self.repo.delete(job_id)
        try:
            from forge.telemetry.metrics import get_metrics_store

            (self._metrics or get_metrics_store()).delete_run(job_id)
        except Exception:  # pragma: no cover
            pass
        if purge_files and row.get("run_dir"):
            shutil.rmtree(row["run_dir"], ignore_errors=True)

    def clone(self, job_id: str, *, name: str | None = None) -> dict[str, Any]:
        row = self.repo.require(job_id, "job")
        spec = JobSpec.model_validate(row["spec"])
        spec = spec.model_copy(update={"name": name or f"{spec.name or spec.type} (clone)"})
        return self.submit(spec, validate=False)

    def wait(self, job_id: str, timeout: float = 60.0, poll: float = 0.05) -> dict[str, Any]:
        """Block until the job is terminal (or timeout) and return its row. Test/CLI helper."""
        end = time.time() + timeout
        while True:
            row = self.repo.require(job_id, "job")
            if row["status"] in TERMINAL or time.time() > end:
                return row
            time.sleep(poll)

    # -- the loop body ---------------------------------------------------------------------------
    def tick(self) -> None:
        with self._lock:
            self._reap()
            self._escalate()
            self._schedule()

    def recover(self) -> list[str]:
        """Handle jobs left ``running`` by a previous process. Returns ids marked failed."""
        failed: list[str] = []
        with self._lock:
            for r in self.repo.running():
                jid = r["id"]
                if jid in self._procs:
                    continue
                if _is_worker_alive(r.get("pid"), jid):
                    log.info("adopting running worker pid=%s for job %s", r.get("pid"), jid)
                    self._procs[jid] = _PidHandle(int(r["pid"]), jid)
                    if r.get("cancel_requested"):
                        self._deadline[jid] = time.time() + self.grace_s
                else:
                    self.repo.update(
                        jid, status="failed", finished_at=now(),
                        error="worker process is gone (API server restarted or worker crashed)",
                    )
                    failed.append(jid)
        self._recovered = True
        return failed

    def _reap(self) -> None:
        for jid, h in list(self._procs.items()):
            rc = h.poll()
            if rc is None:
                continue
            del self._procs[jid]
            self._deadline.pop(jid, None)
            row = self.repo.get(jid)
            if row is None:
                continue
            fields: dict[str, Any] = {}
            if row.get("exit_code") is None and rc >= 0 and isinstance(h, _PopenHandle):
                fields["exit_code"] = rc
            if row["status"] == "running":
                if row.get("cancel_requested"):
                    fields.update(status="cancelled", finished_at=now())
                else:
                    tail = _log_tail(row.get("run_dir"))
                    fields.update(
                        status="failed", finished_at=now(),
                        error=f"worker exited with code {rc} without reporting a result" + (f": {tail}" if tail else ""),
                    )
            if fields:
                self.repo.update(jid, **fields)

    def _escalate(self) -> None:
        t = time.time()
        for jid, dl in list(self._deadline.items()):
            h = self._procs.get(jid)
            if h is None:
                self._deadline.pop(jid, None)
                continue
            if t >= dl:
                log.warning("job %s ignored cancellation for %.0fs: killing process tree", jid, self.grace_s)
                kill_process_tree(h.pid)
                self._deadline.pop(jid, None)

    def _schedule(self) -> None:
        queued = self.repo.queued()
        if not queued:
            return
        running = self.repo.running()
        heavy_running = 0
        busy: set[int] = set()
        for r in running:
            try:
                if not get_job_type(r["type"]).light:
                    heavy_running += 1
            except NotFound:
                heavy_running += 1
            busy.update(int(g) for g in (r.get("gpus") or []))
        all_gpus = self.gpu_indices()
        for row in queued:
            jid = row["id"]
            fresh = self.repo.get(jid)
            if not fresh or fresh["status"] != "queued":
                continue
            try:
                jt = get_job_type(row["type"])
            except NotFound as e:
                self.repo.update(jid, status="failed", finished_at=now(), error=str(e))
                continue
            if not jt.light and heavy_running >= self.max_concurrent:
                continue
            req = (row.get("spec") or {}).get("gpus")
            assigned: list[int] | None
            if req is not None:
                if any(g in busy for g in req):
                    continue
                assigned = list(req)
            elif jt.needs_gpu and all_gpus:
                free = [g for g in all_gpus if g not in busy]
                if not free:
                    continue
                assigned = [free[0]]
            else:
                assigned = None
            if self._spawn(row, jt, assigned):
                busy.update(assigned or [])
                if not jt.light:
                    heavy_running += 1

    def _spawn(self, row: dict[str, Any], jt: JobTypeSpec, gpus: list[int] | None) -> bool:
        jid = row["id"]
        rd = Path(row.get("run_dir") or run_dir_for(jid))
        rd.mkdir(parents=True, exist_ok=True)
        stale = rd / CANCEL_FILE
        if stale.exists():
            stale.unlink()
        env = os.environ.copy()
        env.update(self.env_extra)
        env["PYTHONUNBUFFERED"] = "1"
        env["FORGE_JOB_ID"] = jid
        pkg_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        # GPU indices in the UI / scheduler come from NVML / nvidia-smi, which number devices by PCI bus id. CUDA's
        # default order is FASTEST_FIRST, so on a mixed-GPU host CUDA_VISIBLE_DEVICES=1 could be a different card.
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        else:
            kwargs["start_new_session"] = True
        try:
            with open(rd / "log.txt", "ab") as logf:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "forge.jobs.worker", jid],
                    stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, env=env, **kwargs,
                )
        except Exception as e:
            self.repo.update(jid, status="failed", finished_at=now(), error=f"could not start worker: {e}")
            return False
        self._procs[jid] = _PopenHandle(proc)
        # A very fast worker may already have recorded its final status: only a still-queued row becomes "running"
        # (an unconditional update would resurrect a finished job and _reap would then mark it failed).
        t = now()
        self.repo.db.execute(
            "UPDATE jobs SET pid=?, gpus_json=?, updated_at=? WHERE id=?", (proc.pid, dumps(gpus), t, jid)
        )
        self.repo.db.execute(
            "UPDATE jobs SET status='running', started_at=COALESCE(started_at, ?), updated_at=? WHERE id=? AND status='queued'",
            (t, t, jid),
        )
        log.info("job %s started pid=%s gpus=%s", jid, proc.pid, gpus)
        return True


def _log_tail(run_dir: str | None, n: int = 3) -> str:
    if not run_dir:
        return ""
    try:
        from forge.telemetry.logs import tail_lines

        lines, _ = tail_lines(Path(run_dir) / "log.txt", n)
        return " | ".join(l.strip() for l in lines if l.strip())[:500]
    except Exception:  # pragma: no cover
        return ""
