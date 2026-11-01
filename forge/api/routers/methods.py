"""/methods: the method catalog (forge.methods) -- metadata only, read-only."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from forge.methods import get_method, list_methods

router = APIRouter(tags=["methods"])


@router.get("/methods", summary="Training / fine-tuning / quantization / compression methods with their honest status")
def methods(family: str | None = Query(None), status: str | None = Query(None, description="supported | experimental | planned")) -> list[dict[str, Any]]:
    return [m.model_dump(mode="json") for m in list_methods(family=family, status=status)]


@router.get("/methods/{method_id}", summary="One method")
def method(method_id: str) -> dict[str, Any]:
    return get_method(method_id).model_dump(mode="json")
