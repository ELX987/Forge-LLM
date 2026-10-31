"""Exception hierarchy. Everything Forge raises on purpose derives from :class:`ForgeError`."""
from __future__ import annotations

from typing import Any


class ForgeError(Exception):
    """Base class of all Forge errors. ``code`` is a stable machine-readable identifier."""

    code: str = "forge_error"
    http_status: int = 500

    def __init__(self, message: str = "", *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}

    def __str__(self) -> str:  # message only (no tuple repr)
        return self.message


class StoreError(ForgeError):
    code = "store_error"


class IntegrityError(StoreError):
    """Content hash mismatch / corrupted object."""

    code = "integrity_error"


class GateFailed(ForgeError):
    """An evaluation gate returned a ``fail`` (or blocking ``inconclusive``) verdict."""

    code = "gate_failed"
    http_status = 422

    def __init__(self, message: str = "", *, verdict: Any = None, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, details=details)
        self.verdict = verdict


class LicenceBlocked(ForgeError):
    """A model/dataset licence forbids the requested operation (e.g. distillation)."""

    code = "licence_blocked"
    http_status = 403


class HardwareUnsupported(ForgeError):
    """The requested feature needs hardware (sm level, dtype, memory) that is not present."""

    code = "hardware_unsupported"
    http_status = 422


class KernelCorrectnessError(ForgeError):
    """A kernel disagreed with its reference oracle beyond tolerance."""

    code = "kernel_correctness_error"


class ConfigError(ForgeError, ValueError):
    """Invalid user/config input."""

    code = "config_error"
    http_status = 400


class NotFound(ForgeError, LookupError):
    """Requested object (model, job, artifact, registry entry...) does not exist."""

    code = "not_found"
    http_status = 404


class StageFailed(ForgeError):
    """A recipe stage raised a non-Forge exception; the original is chained as ``__cause__``."""

    code = "stage_failed"

    def __init__(
        self, message: str = "", *, stage: str = "", index: int = -1, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, details=details)
        self.stage = stage
        self.index = index


class Cancelled(ForgeError):
    """Cooperative cancellation requested through ``RunContext.should_stop``."""

    code = "cancelled"
    http_status = 409
