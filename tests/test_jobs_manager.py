"""JobManager + worker subprocesses (CPU selftest jobs)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_api_fixtures import home  # noqa: E402,F401

from forge.core.errors import ConfigError  # noqa: E402
from forge.core.schemas import JobSpec  # noqa: E402
from forge.jobs.manager import InvalidParams, JobManager  # noqa: E402


@pytest.fixture
def mgr(home):
    from forge.telemetry.metrics import get_metrics_store

    m = JobManager(poll_s=0.05, grace_s=1.5, max_concurrent=1, gpu_provider=lambda: [0, 1], metrics=get_metrics_store())
    m.start()
    yield m
    m.stop(kill_running=True)


def plugin_env(monkeypatch):
    """Make the extra job types of tests/jobs_test_plugin.py known to this process and to worker subprocesses."""
    here = str(Path(__file__).parent)
    monkeypatch.setenv("FORGE_JOB_PLUGINS", "jobs_test_plugin")
    monkeypatch.setenv("PYTHONPATH", here)
    import jobs_test_plugin  # noqa: F401  (registers in this process)


def selftest(**p):
    return JobSpec(type="selftest", params={"step_delay_s": 0.02, **p})


def test_lifecycle_success(mgr):
    row = mgr.submit(selftest(steps=6))
    assert row["status"] == "queued" and Path(row["run_dir"]).is_dir()
    done = mgr.wait(row["id"], 40)
    assert done["status"] == "succeeded", done
    assert done["exit_code"] == 0 and done["progress"] == 1.0 and done["error"] is None
    assert done["outputs"]["echo"] == "ok" and done["outputs"]["result"]["steps_done"] == 6
    assert done["started_at"] and done["finished_at"] >= done["started_at"] and done["pid"]
    from forge.telemetry.metrics import get_metrics_store

    m = get_metrics_store().query(row["id"], keys=["loss"])
    assert len(m["points"]["loss"]) == 6
    log = (Path(row["run_dir"]) / "log.txt").read_text()
    assert "SYNTHETIC" in log and "job finished" in log
    assert (Path(row["run_dir"]) / "spec.json").exists()


def test_failure_records_error_and_traceback(mgr):
    row = mgr.submit(selftest(steps=10, fail_at_step=3))
    done = mgr.wait(row["id"], 40)
    assert done["status"] == "failed" and done["exit_code"] == 1
    assert "injected failure at step 3" in done["error"]
    assert "Traceback" in (Path(row["run_dir"]) / "log.txt").read_text()


def test_validation_errors_are_field_level(mgr):
    with pytest.raises(InvalidParams) as ei:
        mgr.submit(selftest(steps=-5, bogus=1))
    errs = ei.value.details["errors"]
    assert {e["loc"] for e in errs} == {"steps", "bogus"}
    assert all(e["msg"] for e in errs)
    # nothing was queued
    assert mgr.repo.count() == 0


def test_unknown_type_and_bad_gpu(mgr):
    from forge.core.errors import NotFound

    with pytest.raises(NotFound):
        mgr.submit(JobSpec(type="nope"))
    with pytest.raises(ConfigError):
        mgr.submit(JobSpec(type="selftest", gpus=[7]))
    with pytest.raises(ConfigError):
        mgr.submit(JobSpec(type="selftest", gpus=[-1]))


def test_cancel_running_is_cooperative(mgr):
    row = mgr.submit(selftest(steps=5000))
    t0 = time.time()
    while mgr.repo.require(row["id"])["status"] != "running" and time.time() - t0 < 30:
        time.sleep(0.05)
    time.sleep(0.5)
    r = mgr.cancel(row["id"])
    assert r["cancel_requested"]
    done = mgr.wait(row["id"], 30)
    assert done["status"] == "cancelled"
    assert "stop requested" in (Path(row["run_dir"]) / "log.txt").read_text()
    assert mgr.cancel(row["id"])["status"] == "cancelled"  # idempotent


def test_cancel_escalates_to_kill(mgr):
    row = mgr.submit(selftest(steps=100000, ignore_stop=True))
    t0 = time.time()
    while mgr.repo.require(row["id"])["status"] != "running" and time.time() - t0 < 30:
        time.sleep(0.05)
    time.sleep(0.5)
    mgr.cancel(row["id"])
    done = mgr.wait(row["id"], 30)
    assert done["status"] == "cancelled"
    assert time.time() - t0 < 30


def test_cancel_queued_job(mgr):
    blocker = mgr.submit(selftest(steps=5000))
    waiting = mgr.submit(selftest(steps=3))
    time.sleep(0.4)
    assert mgr.repo.require(waiting["id"])["status"] == "queued"  # max_concurrent=1
    assert mgr.cancel(waiting["id"])["status"] == "cancelled"
    mgr.cancel(blocker["id"])
    mgr.wait(blocker["id"], 30)
    assert mgr.repo.require(waiting["id"])["status"] == "cancelled"  # never started
    assert mgr.repo.require(waiting["id"])["pid"] is None


def test_max_concurrency_and_fifo(mgr):
    a = mgr.submit(selftest(steps=8))
    b = mgr.submit(selftest(steps=3))
    seen_both_running = False
    t0 = time.time()
    while time.time() - t0 < 40:
        sa, sb = (mgr.repo.require(x["id"])["status"] for x in (a, b))
        if sa == "running" and sb == "running":
            seen_both_running = True
        if sa == "succeeded" and sb == "succeeded":
            break
        time.sleep(0.02)
    assert not seen_both_running
    ra, rb = mgr.repo.require(a["id"]), mgr.repo.require(b["id"])
    assert ra["status"] == rb["status"] == "succeeded"
    assert rb["started_at"] >= ra["finished_at"] - 0.5  # b started after a finished (poll granularity)


def test_gpu_exclusivity_and_cuda_visible_devices(home):
    from forge.telemetry.metrics import get_metrics_store

    m = JobManager(poll_s=0.05, grace_s=1.5, max_concurrent=4, gpu_provider=lambda: [0, 1], metrics=get_metrics_store())
    m.start()
    try:
        a = m.submit(JobSpec(type="selftest", gpus=[0], params={"steps": 40, "step_delay_s": 0.05}))
        b = m.submit(JobSpec(type="selftest", gpus=[0], params={"steps": 2, "step_delay_s": 0.0}))  # same GPU: must wait
        c = m.submit(JobSpec(type="selftest", gpus=[1], params={"steps": 2, "step_delay_s": 0.0}))  # other GPU: runs now
        rc = m.wait(c["id"], 40)
        assert rc["status"] == "succeeded"
        assert m.repo.require(b["id"])["status"] == "queued" or m.repo.require(a["id"])["status"] != "running"
        ra = m.wait(a["id"], 60)
        rb = m.wait(b["id"], 60)
        assert ra["status"] == rb["status"] == "succeeded"
        assert rb["started_at"] >= ra["finished_at"] - 0.5
        assert "CUDA_VISIBLE_DEVICES=0" in (Path(a["run_dir"]) / "log.txt").read_text()
        assert "CUDA_VISIBLE_DEVICES=1" in (Path(c["run_dir"]) / "log.txt").read_text()
        # gpus=[] -> CPU only, hides all devices
        d = m.submit(JobSpec(type="selftest", gpus=[], params={"steps": 1, "step_delay_s": 0.0}))
        m.wait(d["id"], 40)
        assert "CUDA_VISIBLE_DEVICES=\n" in (Path(d["run_dir"]) / "log.txt").read_text() + "\n" or "CUDA_VISIBLE_DEVICES= " in (Path(d["run_dir"]) / "log.txt").read_text()
    finally:
        m.stop(kill_running=True)


def test_gpu_job_type_gets_free_gpu_automatically(home, monkeypatch):
    from forge.telemetry.metrics import get_metrics_store

    plugin_env(monkeypatch)
    m = JobManager(poll_s=0.05, grace_s=1.5, max_concurrent=4, gpu_provider=lambda: [3], metrics=get_metrics_store())
    m.start()
    try:
        a = m.submit(JobSpec(type="selftest_gpu", params={"steps": 30, "step_delay_s": 0.05}))
        b = m.submit(JobSpec(type="selftest_gpu", params={"steps": 1, "step_delay_s": 0.0}))
        time.sleep(1.0)
        ra = m.repo.require(a["id"])
        assert ra["gpus"] == [3]  # auto-assigned
        assert m.repo.require(b["id"])["status"] == "queued"  # the only GPU is taken
        assert m.wait(b["id"], 60)["status"] == "succeeded"
    finally:
        m.stop(kill_running=True)


def test_light_jobs_do_not_count_against_limit(mgr):
    heavy = mgr.submit(selftest(steps=5000))
    gc = mgr.submit(JobSpec(type="gc", params={"dry_run": True}))  # light job type
    done = mgr.wait(gc["id"], 40)
    assert done["status"] in ("succeeded", "failed")  # ran although the heavy slot is occupied
    assert done["started_at"] is not None
    mgr.cancel(heavy["id"])
    mgr.wait(heavy["id"], 30)


def test_recover_marks_dead_running_jobs_failed(home):
    from forge.jobs.repo import JobsRepo

    repo = JobsRepo()
    row = repo.create(JobSpec(type="selftest"), run_dir=str(home / "runs" / "x"))
    repo.update(row["id"], status="running", pid=2_000_000_000, started_at=time.time())
    m = JobManager(repo=repo)
    failed = m.recover()
    assert failed == [row["id"]]
    r = repo.require(row["id"])
    assert r["status"] == "failed" and "gone" in r["error"]


def test_recover_adopts_live_worker(home):
    """A worker started by a previous API process is adopted, monitored and reaped."""
    import subprocess

    from forge.jobs.repo import JobsRepo
    from forge.telemetry.metrics import get_metrics_store

    repo = JobsRepo()
    row = repo.create(JobSpec(type="selftest", params={"steps": 25, "step_delay_s": 0.04}), run_dir=str(home / "runs" / "adopt"))
    (home / "runs" / "adopt").mkdir(parents=True)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    p = subprocess.Popen([sys.executable, "-m", "forge.jobs.worker", row["id"]], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while repo.require(row["id"])["status"] != "running" and time.time() - t0 < 30:
            time.sleep(0.05)
        m = JobManager(repo=repo, poll_s=0.05, metrics=get_metrics_store())
        assert m.recover() == []  # alive -> adopted, not failed
        assert row["id"] in m._procs
        m.start()
        done = m.wait(row["id"], 40)
        assert done["status"] == "succeeded"
        m.tick()
        assert row["id"] not in m._procs
        m.stop()
    finally:
        p.kill()
        p.wait(timeout=10)


def test_worker_crash_without_status_marks_failed(home, monkeypatch):
    from forge.telemetry.metrics import get_metrics_store

    plugin_env(monkeypatch)
    mgr = JobManager(poll_s=0.05, grace_s=1.5, metrics=get_metrics_store())
    mgr.start()
    row = mgr.submit(JobSpec(type="crasher", params={}))
    done = mgr.wait(row["id"], 40)
    assert done["status"] == "failed"
    assert "not available" in done["error"] and "tests_missing_mod" in done["error"]
    mgr.stop(kill_running=True)


def test_delete_and_clone(mgr):
    from forge.core.errors import NotFound

    row = mgr.submit(selftest(steps=2))
    mgr.wait(row["id"], 40)
    clone = mgr.clone(row["id"])
    assert clone["id"] != row["id"] and clone["spec"]["params"] == row["spec"]["params"] and "(clone)" in clone["name"]
    mgr.wait(clone["id"], 40)
    mgr.delete(row["id"])
    assert not Path(row["run_dir"]).exists()
    with pytest.raises(NotFound):
        mgr.repo.require(row["id"])
    running = mgr.submit(selftest(steps=5000))
    time.sleep(0.3)
    from forge.jobs.manager import JobStateError

    with pytest.raises(JobStateError):
        mgr.delete(running["id"])
