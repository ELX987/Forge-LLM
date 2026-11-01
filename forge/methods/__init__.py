"""forge.methods: catalog of training / fine-tuning / quantization / compression methods (ADR 0007).

A :class:`MethodSpec` is metadata only: it points at the job type, template and diagnostics that already implement a method, and states
honestly how far validated it is.  It powers the Method picker (UI), the planner and docs; it does not change how anything runs.
"""
from __future__ import annotations

from .catalog import MethodSpec, get_method, list_methods, register_method

__all__ = ["MethodSpec", "get_method", "list_methods", "register_method"]
