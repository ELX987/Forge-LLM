"""Test-only job-type plugin loaded through FORGE_JOB_PLUGINS (see test_jobs_manager.py)."""
from __future__ import annotations

from forge.jobs.registry import JobTypeSpec, register_job_type

register_job_type(
    JobTypeSpec("selftest_gpu", "t", "t", "forge.jobs.builtin:SelftestConfig", "forge.jobs.builtin:run_selftest", needs_gpu=True)
)
register_job_type(JobTypeSpec("crasher", "t", "t", "forge.jobs.builtin:SelftestConfig", "tests_missing_mod:nothing"))
