"""Evaluation suite registry.

Suites register themselves with ``@register("suite", name)`` (``forge.core.registry``).  Importing this package imports
every built-in suite module (cheap: torch is only touched lazily inside functions).
"""
from __future__ import annotations

from typing import Any

from forge.core.errors import NotFound
from forge.core.registry import get as _get
from forge.core.registry import names as _names
from forge.eval.suites.base import (  # noqa: F401
    ByteTokenizer,
    HFTokenizerAdapter,
    Suite,
    SuiteConfig,
    SuiteEnv,
    SuiteResult,
    adapt_tokenizer,
    read_jsonl,
    score_continuations,
)

# importing registers the suites
from forge.eval.suites import agentic, contamination, distribution, longctx, longgen, mcq, safety, throughput  # noqa: E402,F401

BUILTIN = ("distribution", "mcq", "longgen", "longctx", "safety", "contamination", "throughput", "agentic")


def list_suites() -> list[str]:
    return _names("suite")


def get_suite(name: str) -> type[Suite]:
    try:
        return _get("suite", name)
    except NotFound:
        raise NotFound(f"unknown evaluation suite {name!r}; available: {', '.join(list_suites())}") from None


def describe_suites() -> list[dict[str, Any]]:
    """For the UI: name, title, version, description and the JSON Schema of each suite's config."""
    from forge.core.jsonschema import defaults_for, schema_for

    out = []
    for n in list_suites():
        cls = get_suite(n)
        out.append({
            "name": n, "title": cls.title or n, "version": cls.version, "description": cls.describes or (cls.__doc__ or "").strip(),
            "needs_candidate_model": cls.needs_candidate_model, "schema": schema_for(cls.Config), "defaults": defaults_for(cls.Config),
        })
    return out
