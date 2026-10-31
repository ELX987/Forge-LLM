"""forge.api: FastAPI application (``create_app``) and routers. See docs/ARCHITECTURE.md section 7."""
from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):  # lazy: importing forge.api must stay cheap
    from forge.api.app import create_app as _create

    return _create(*args, **kwargs)
