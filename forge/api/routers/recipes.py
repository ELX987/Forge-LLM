"""/recipes : saved recipes, validation, run."""
from __future__ import annotations

from typing import Any

import yaml
from fastapi import APIRouter, Depends, Response

from forge.api.deps import Services, get_services
from forge.api.jobsvc import submit
from forge.api.schemas import RecipeCreate, RecipeRun, RecipeValidate
from forge.core.errors import ConfigError, NotFound
from forge.core.recipe import RecipeEngine, dump_recipe, load_recipe
from forge.core.registry import describe as registry_describe
from forge.core.util import new_id, now
from forge.jobs.builtin import import_stage_modules
from forge.jobs.templates import find_repo_dir

router = APIRouter(prefix="/recipes", tags=["recipes"])


def check_recipe(text: str | None = None, obj: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse + validate a recipe. Never raises for user errors: returns ``{valid, errors, problems, recipe, hash, yaml}``."""
    import_stage_modules()
    try:
        recipe = load_recipe(text if text is not None else (obj or {}))
    except Exception as e:  # ConfigError / pydantic ValidationError
        msg = getattr(e, "message", None) or str(e)
        errs = []
        if hasattr(e, "errors"):
            try:
                errs = [{"loc": ".".join(str(p) for p in x["loc"]), "msg": x["msg"], "type": x["type"]} for x in e.errors()]  # type: ignore[attr-defined]
            except Exception:
                errs = []
        return {"valid": False, "errors": errs or [{"loc": "", "msg": msg, "type": "parse"}], "problems": [], "recipe": None, "hash": None, "yaml": None}
    problems = RecipeEngine(None).validate(recipe)
    return {
        "valid": not problems,
        "errors": [],
        "problems": problems,
        "recipe": recipe.model_dump(mode="json"),
        "hash": recipe.hash(),
        "yaml": dump_recipe(recipe),
    }


@router.get("/stages", summary="Registered recipe stages (the stage palette)")
def stages() -> list[dict[str, Any]]:
    import_stage_modules()
    return registry_describe("stage")


@router.get("/templates", summary="Recipe templates from configs/recipes/")
def templates() -> list[dict[str, Any]]:
    d = find_repo_dir("recipes")
    out = []
    if d:
        for f in sorted(list(d.glob("*.yaml")) + list(d.glob("*.yml"))):
            text = f.read_text(encoding="utf-8")
            out.append({"name": f.stem, "yaml": text, "source": f"recipes/{f.name}"})
    return out


@router.post("/validate", summary="Validate a recipe (YAML text or JSON object)")
def validate(body: RecipeValidate) -> dict[str, Any]:
    if body.yaml is None and body.json_ is None:
        raise ConfigError("provide `yaml` or `json`")
    return check_recipe(body.yaml, body.json_)


@router.get("", summary="List saved recipes")
def list_recipes(svc: Services = Depends(get_services)) -> list[dict[str, Any]]:
    return svc.recipes.list(order="updated_at DESC", limit=1000)


def _text_of(body: RecipeCreate) -> str:
    if body.yaml:
        return body.yaml
    if body.recipe is not None:
        return yaml.safe_dump(body.recipe, sort_keys=False)
    raise ConfigError("provide `yaml` or `recipe`")


@router.post("", status_code=201, summary="Save a recipe (parsed and checked for syntax; unknown stages are reported but allowed)")
def create(body: RecipeCreate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    text = _text_of(body)
    res = check_recipe(text)
    if res["recipe"] is None:
        raise ConfigError("recipe does not parse: " + "; ".join(e["msg"] for e in res["errors"]), details={"errors": res["errors"]})
    t = now()
    row = svc.recipes.insert(
        id=new_id("rcp"), name=body.name or res["recipe"]["name"], description=body.description, yaml=text, recipe_hash=res["hash"], created_at=t, updated_at=t
    )
    row["problems"] = res["problems"]
    return row


@router.get("/{recipe_id}")
def get(recipe_id: str, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = svc.recipes.require(recipe_id, "recipe")
    res = check_recipe(row["yaml"])
    row["problems"] = res["problems"]
    row["parsed"] = res["recipe"]
    return row


@router.put("/{recipe_id}")
def update(recipe_id: str, body: RecipeCreate, svc: Services = Depends(get_services)) -> dict[str, Any]:
    svc.recipes.require(recipe_id, "recipe")
    fields: dict[str, Any] = {"description": body.description}
    if body.yaml or body.recipe is not None:
        text = _text_of(body)
        res = check_recipe(text)
        if res["recipe"] is None:
            raise ConfigError("recipe does not parse: " + "; ".join(e["msg"] for e in res["errors"]), details={"errors": res["errors"]})
        fields.update(yaml=text, recipe_hash=res["hash"])
        if not body.name:
            fields["name"] = res["recipe"]["name"]
    if body.name:
        fields["name"] = body.name
    return svc.recipes.update(recipe_id, **fields)


@router.delete("/{recipe_id}", status_code=204)
def delete(recipe_id: str, svc: Services = Depends(get_services)) -> Response:
    if not svc.recipes.delete(recipe_id):
        raise NotFound(f"recipe {recipe_id!r} not found")
    return Response(status_code=204)


@router.post("/{recipe_id}/run", status_code=201, summary="Run a saved recipe (job type `recipe`)")
def run(recipe_id: str, body: RecipeRun | None = None, svc: Services = Depends(get_services)) -> dict[str, Any]:
    row = svc.recipes.require(recipe_id, "recipe")
    body = body or RecipeRun()
    params = {"recipe_id": recipe_id, "force": body.force, "cache": body.cache, "input_overrides": body.input_overrides}
    return submit(svc, "recipe", params, name=body.name or f"recipe {row['name']}", gpus=body.gpus)
