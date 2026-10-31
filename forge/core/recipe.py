"""Recipe engine: ordered stages with a content-addressed stage cache and a lineage hook.

Stage contract
--------------
::

    @register("stage", "ptq", meta={"inputs": ["model", "calib"], "code_version": "2"})
    def ptq(inputs: dict, params: dict, ctx: RunContext) -> dict: ...

* ``inputs``  the accumulated namespace (recipe inputs + outputs of earlier stages; later keys override),
  restricted to ``meta["inputs"]`` if given. A reserved key ``_recipe`` carries ``{name, hash, seeds, budget,
  policy, stage_index}`` and is *not* part of the cache key (seeds are).
* the returned dict is merged into the namespace and also kept per stage in the result.
* **cache key** = hash(stage name, params, hashes of consumed inputs, code version, seeds).
  A stage whose cache key is known is skipped (instant); changing a stage re-runs it and everything whose inputs
  changed as a consequence. ``meta["cache"] = False`` disables caching for a stage.
* **code version**: ``fn.code_version`` attribute, else ``meta["code_version"]``, else ``"1"``.
* Persisted cache entries hold the JSON-serialisable outputs. Non-JSON outputs are cached in-process only.
  Output values that look like artifact digests are verified with ``store.has`` before a cached entry is reused.
* **lineage**: for each stage, every output digest gets an edge to every input digest it consumed
  (``store.add_edge(child, parent, recipe_hash, "derived_from")``) and/or the optional ``lineage_hook``.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from forge.core import registry
from forge.core.context import NullContext, RunContext
from forge.core.errors import ConfigError, ForgeError, NotFound, StageFailed
from forge.core.hashing import canonical_json, hash_algo, hash_bytes, hash_json, is_digest
from forge.core.schemas import Recipe, StageSpec

log = logging.getLogger("forge.core.recipe")

RESERVED_INPUT = "_recipe"


# --------------------------------------------------------------------------------------------
# YAML (de)serialisation
# --------------------------------------------------------------------------------------------
def _stage_from_yaml(item: Any, idx: int) -> StageSpec:
    if isinstance(item, str):
        return StageSpec(name=item, params={})
    if not isinstance(item, dict):
        raise ConfigError(f"stage #{idx} must be a mapping, got {type(item).__name__}")
    if set(item.keys()) <= {"name", "params"} and "name" in item:
        return StageSpec(name=str(item["name"]), params=dict(item.get("params") or {}))
    if len(item) != 1:
        raise ConfigError(f"stage #{idx}: expected a single '<stage>: {{params}}' mapping, got keys {list(item)}")
    ((name, params),) = item.items()
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ConfigError(f"stage #{idx} ({name}): params must be a mapping")
    return StageSpec(name=str(name), params=params)


def recipe_from_dict(data: dict[str, Any]) -> Recipe:
    """Build a :class:`Recipe` from the Appendix C.1 layout (``recipe:`` name, ``stages: [- name: {params}]``)
    or from the canonical ``Recipe`` dict."""
    if not isinstance(data, dict):
        raise ConfigError("recipe must be a mapping")
    d = dict(data)
    if "recipe" in d and "name" not in d:
        d["name"] = d.pop("recipe")
    if "stages" not in d:
        raise ConfigError("recipe has no 'stages'")
    stages_raw = d["stages"]
    if not isinstance(stages_raw, list):
        raise ConfigError("'stages' must be a list")
    d["stages"] = [_stage_from_yaml(s, i) for i, s in enumerate(stages_raw)]
    try:
        return Recipe.model_validate(d)
    except Exception as e:
        raise ConfigError(f"invalid recipe: {e}") from e


def load_recipe(path_or_text: str | Path | dict) -> Recipe:
    """Load a recipe from a YAML/JSON file path, YAML text, or an already-parsed dict."""
    if isinstance(path_or_text, dict):
        return recipe_from_dict(path_or_text)
    if isinstance(path_or_text, Path) or (
        isinstance(path_or_text, str) and "\n" not in path_or_text and _is_file(path_or_text)
    ):
        text = Path(path_or_text).read_text(encoding="utf-8")
    else:
        text = str(path_or_text)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"recipe YAML parse error: {e}") from e
    return recipe_from_dict(data)


def _is_file(s: str) -> bool:
    try:
        return len(s) < 1024 and Path(s).is_file()
    except OSError:
        return False


def dump_recipe(recipe: Recipe) -> str:
    """YAML text in the Appendix C.1 compact layout. ``load_recipe(dump_recipe(r)) == r``."""
    d: dict[str, Any] = {"recipe": recipe.name}
    body = recipe.model_dump(mode="json")
    d["inputs"] = body["inputs"]
    d["stages"] = [{s["name"]: s["params"]} for s in body["stages"]]
    for k in ("baselines", "budget", "policy", "seeds"):
        d[k] = body[k]
    return yaml.safe_dump(d, sort_keys=False, default_flow_style=False)


# --------------------------------------------------------------------------------------------
# hashing of stage values
# --------------------------------------------------------------------------------------------
def hash_value(v: Any) -> str:
    """Identity of a value for cache keys: digests are their own identity; tensors/bytes by content; JSON by canonical form."""
    if is_digest(v):
        return v
    if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "dtype"):  # torch tensor
        try:
            t = v.detach().cpu().contiguous()
            raw = t.view(-1).view(__import__("torch").uint8).numpy().tobytes()
            return hash_json({"tensor": str(t.dtype), "shape": list(t.shape), "h": hash_bytes(raw)})
        except Exception:
            pass
    try:
        return hash_json(v)
    except TypeError:
        return hash_json({"unhashable": type(v).__qualname__, "id": id(v)})


def _collect_digests(v: Any, out: list[str]) -> None:
    if is_digest(v):
        if v not in out:
            out.append(v)
    elif isinstance(v, dict):
        for x in v.values():
            _collect_digests(x, out)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _collect_digests(x, out)


def _jsonable(v: Any) -> bool:
    try:
        canonical_json(v)
        import json

        json.dumps(v)
        return True
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------------
class RecipeEngine:
    """Executes a :class:`Recipe`. See module docstring for the stage contract."""

    def __init__(
        self,
        store: Any = None,
        *,
        cache: bool = True,
        db: Any = None,
        lineage_hook: Callable[[str, str, str, str], None] | None = None,
    ) -> None:
        """``store``: optional ArtifactStore (lineage + digest verification).
        ``cache``: enable the stage cache. ``db``: ``forge.core.db.Database`` (default: ``get_db()``, lazily).
        ``lineage_hook(child, parent, recipe_hash, stage_name)`` is called for every derived edge."""
        self.store = store
        self.cache_enabled = cache
        self._db = db
        self.lineage_hook = lineage_hook
        self._mem: dict[str, dict[str, Any]] = {}

    # -- cache plumbing --------------------------------------------------------------------
    @property
    def db(self) -> Any:
        if self._db is None:
            from forge.core.db import get_db

            self._db = get_db()
        return self._db

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        entry = self._mem.get(key)
        if entry is None:
            try:
                row = self.db.query_one("SELECT outputs_json FROM stage_cache WHERE cache_key=?", (key,))
            except Exception as e:  # db unavailable -> behave as cache miss
                log.warning("stage cache read failed: %s", e)
                row = None
            if row is not None:
                import json

                entry = {"outputs": json.loads(row["outputs_json"])}
        if entry is None:
            return None
        outs = entry["outputs"]
        if self.store is not None:
            digests: list[str] = []
            _collect_digests(outs, digests)
            prefix = hash_algo() + ":"
            for d in digests:
                if d.startswith(prefix):
                    try:
                        if not self.store.has(d):
                            log.info("cached output %s missing from store; invalidating", d)
                            self._cache_drop(key)
                            return None
                    except Exception:
                        pass
        if key not in self._mem:
            self._mem[key] = entry
        try:
            self.db.execute(
                "UPDATE stage_cache SET hits=hits+1, last_hit_at=? WHERE cache_key=?", (time.time(), key)
            )
        except Exception:
            pass
        return entry

    def _cache_put(self, key: str, stage: str, recipe_hash: str, outputs: dict[str, Any], seconds: float) -> None:
        self._mem[key] = {"outputs": outputs}
        if not _jsonable(outputs):
            return
        digests: list[str] = []
        _collect_digests(outputs, digests)
        import json

        try:
            self.db.execute(
                "INSERT OR REPLACE INTO stage_cache(cache_key, stage, recipe_hash, outputs_json, output_hashes_json,"
                " seconds, hits, created_at) VALUES(?,?,?,?,?,?,0,?)",
                (key, stage, recipe_hash, json.dumps(outputs), json.dumps(digests), seconds, time.time()),
            )
        except Exception as e:
            log.warning("stage cache write failed: %s", e)

    def _cache_drop(self, key: str) -> None:
        self._mem.pop(key, None)
        try:
            self.db.execute("DELETE FROM stage_cache WHERE cache_key=?", (key,))
        except Exception:
            pass

    def clear_cache(self, stage: str | None = None) -> int:
        """Delete cached entries (all, or one stage name). Returns rows removed from the database."""
        if stage is None:
            self._mem.clear()
            cur = self.db.execute("DELETE FROM stage_cache")
        else:
            cur = self.db.execute("DELETE FROM stage_cache WHERE stage=?", (stage,))
            self._mem = {k: v for k, v in self._mem.items()}  # keys are opaque; clear all memory to be safe
            self._mem.clear()
        return cur.rowcount

    # -- stage resolution ------------------------------------------------------------------
    @staticmethod
    def _resolve(spec: StageSpec) -> tuple[Callable[..., dict], dict[str, Any], str]:
        fn = registry.get("stage", spec.name)
        meta = registry.meta("stage", spec.name)
        version = str(getattr(fn, "code_version", None) or meta.get("code_version") or "1")
        return fn, meta, version

    def validate(self, recipe: Recipe) -> list[str]:
        """Problems that would prevent running (unknown stages, empty recipe). Empty list = OK."""
        problems = []
        if not recipe.stages:
            problems.append("recipe has no stages")
        for i, s in enumerate(recipe.stages):
            if not registry.has("stage", s.name):
                problems.append(f"stage #{i} '{s.name}' is not registered")
        return problems

    @staticmethod
    def _consumed(inputs: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        want = meta.get("inputs")
        if want:
            return {k: inputs[k] for k in want if k in inputs}
        return dict(inputs)

    def _cache_key(
        self, spec: StageSpec, consumed: dict[str, Any], version: str, recipe: Recipe
    ) -> str:
        return hash_json(
            {
                "stage": spec.name,
                "params": spec.params,
                "inputs": {k: hash_value(v) for k, v in sorted(consumed.items())},
                "code_version": version,
                "seeds": recipe.seeds,
            }
        )

    # -- planning --------------------------------------------------------------------------
    def plan(self, recipe: Recipe) -> list[dict[str, Any]]:
        """Dry run: which stages would be served from cache. After the first miss, downstream stages are
        reported ``cached=False`` (their inputs are not known yet)."""
        ns: dict[str, Any] = dict(recipe.inputs)
        known = True
        rows = []
        for i, spec in enumerate(recipe.stages):
            try:
                _, meta, version = self._resolve(spec)
            except NotFound:
                rows.append({"index": i, "name": spec.name, "cache_key": None, "cached": False, "registered": False})
                known = False
                continue
            row: dict[str, Any] = {"index": i, "name": spec.name, "cache_key": None, "cached": False, "registered": True}
            if known and self.cache_enabled and meta.get("cache", True):
                key = self._cache_key(spec, self._consumed(ns, meta), version, recipe)
                row["cache_key"] = key
                entry = self._cache_get(key)
                if entry is not None:
                    row["cached"] = True
                    ns.update(entry["outputs"])
                else:
                    known = False
            else:
                known = False
            rows.append(row)
        return rows

    # -- run -------------------------------------------------------------------------------
    def run(
        self, recipe: Recipe, ctx: RunContext | None = None, *, force: bool = False
    ) -> dict[str, Any]:
        """Execute all stages in order.

        Returns ``{"recipe_hash", "outputs": <final namespace>, "stages": [{index, name, cache_key, cached,
        seconds, outputs}], "cache_hits", "cancelled"}``. ``force=True`` ignores existing cache entries
        (results are still written). Raises :class:`NotFound` before starting if a stage is unregistered.
        """
        ctx = ctx or NullContext()
        problems = [p for p in self.validate(recipe) if "no stages" not in p]
        if problems:
            raise NotFound("; ".join(problems), details={"problems": problems})
        rhash = recipe.hash()
        ns: dict[str, Any] = dict(recipe.inputs)
        n = max(1, len(recipe.stages))
        records: list[dict[str, Any]] = []
        hits = 0
        cancelled = False
        ctx.log(f"recipe '{recipe.name}' ({rhash[:19]}...) with {len(recipe.stages)} stages")
        for i, spec in enumerate(recipe.stages):
            if ctx.should_stop():
                cancelled = True
                ctx.log("stop requested; recipe cancelled", "warning")
                break
            fn, meta, version = self._resolve(spec)
            consumed = self._consumed(ns, meta)
            cacheable = self.cache_enabled and bool(meta.get("cache", True))
            key = self._cache_key(spec, consumed, version, recipe) if cacheable else None
            ctx.progress(i / n, f"stage {i + 1}/{n}: {spec.name}")
            entry = self._cache_get(key) if (cacheable and not force and key) else None
            t0 = time.perf_counter()
            if entry is not None:
                outputs = entry["outputs"]
                cached = True
                hits += 1
                ctx.log(f"[{i + 1}/{n}] {spec.name}: cache hit")
            else:
                cached = False
                ctx.log(f"[{i + 1}/{n}] {spec.name}: running")
                stage_inputs = dict(consumed)
                stage_inputs[RESERVED_INPUT] = {
                    "name": recipe.name, "hash": rhash, "seeds": list(recipe.seeds),
                    "budget": dict(recipe.budget), "policy": dict(recipe.policy), "stage_index": i,
                }
                try:
                    outputs = fn(stage_inputs, dict(spec.params), ctx)
                except (ForgeError, KeyboardInterrupt):
                    raise
                except Exception as e:
                    raise StageFailed(
                        f"stage #{i} '{spec.name}' failed: {type(e).__name__}: {e}", stage=spec.name, index=i
                    ) from e
                if outputs is None:
                    outputs = {}
                if not isinstance(outputs, dict):
                    raise StageFailed(
                        f"stage '{spec.name}' must return a dict, got {type(outputs).__name__}", stage=spec.name, index=i
                    )
                if ctx.should_stop():
                    # a stage interrupted by cancellation may return partial results: do not cache them
                    cacheable = False
                if cacheable and key:
                    self._cache_put(key, spec.name, rhash, outputs, time.perf_counter() - t0)
            seconds = time.perf_counter() - t0
            self._emit_lineage(spec.name, rhash, consumed, outputs)
            ns.update(outputs)
            records.append(
                {"index": i, "name": spec.name, "cache_key": key, "cached": cached, "seconds": seconds,
                 "outputs": outputs}
            )
            ctx.progress((i + 1) / n, f"stage {i + 1}/{n}: {spec.name} done")
            if ctx.should_stop() and i + 1 < len(recipe.stages):
                cancelled = True
                ctx.log("stop requested; recipe cancelled", "warning")
                break
        result = {
            "recipe_hash": rhash, "outputs": ns, "stages": records, "cache_hits": hits, "cancelled": cancelled,
        }
        try:
            ctx.set_output("recipe_hash", rhash)
            ctx.set_output("cache_hits", hits)
        except Exception:
            pass
        return result

    def _emit_lineage(self, stage: str, rhash: str, consumed: dict[str, Any], outputs: dict[str, Any]) -> None:
        if self.store is None and self.lineage_hook is None:
            return
        parents: list[str] = []
        children: list[str] = []
        _collect_digests(consumed, parents)
        _collect_digests(outputs, children)
        for child in children:
            for parent in parents:
                if child == parent:
                    continue
                if self.lineage_hook is not None:
                    try:
                        self.lineage_hook(child, parent, rhash, stage)
                    except Exception as e:
                        log.warning("lineage hook failed: %s", e)
                if self.store is not None and hasattr(self.store, "add_edge"):
                    try:
                        self.store.add_edge(child, parent, recipe_hash=rhash, relation="derived_from")
                    except Exception as e:
                        log.warning("add_edge failed: %s", e)
