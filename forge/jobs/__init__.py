"""forge.jobs: job manager, worker process, job-type registry and built-in job types.

* :mod:`forge.jobs.registry`  job type -> (ConfigModel, entrypoint), lazily imported
* :mod:`forge.jobs.manager`   :class:`JobManager` (scheduling, GPU exclusivity, cancel, recovery)
* :mod:`forge.jobs.worker`    ``python -m forge.jobs.worker <job_id>``
* :mod:`forge.jobs.builtin`   selftest / probe / recipe / model_download / gc / checkpoint_export
* :mod:`forge.jobs.repo`      data access for jobs, models, datasets, configs, recipes, deployments
"""
from __future__ import annotations

from typing import Any

__all__ = ["JobManager", "get_job_type", "list_job_types", "register_job_type"]


def __getattr__(name: str) -> Any:  # lazy re-exports
    if name == "JobManager":
        from forge.jobs.manager import JobManager

        return JobManager
    if name in ("get_job_type", "list_job_types", "register_job_type"):
        from forge.jobs import registry

        return getattr(registry, name)
    raise AttributeError(name)
