from __future__ import annotations

import pytest

from forge.core import registry
from forge.core.context import NullContext
from forge.core.errors import ConfigError, GateFailed, NotFound, StageFailed
from forge.core.hashing import hash_algo, hash_bytes
from forge.core.recipe import RecipeEngine, dump_recipe, load_recipe
from forge.core.schemas import Recipe, StageSpec

CALLS: dict[str, int] = {}


def _count(name):
    CALLS[name] = CALLS.get(name, 0) + 1


@pytest.fixture(autouse=True)
def stages(forge_home):
    CALLS.clear()

    @registry.register("stage", "t_load", meta={"inputs": ["src"]})
    def load(inputs, params, ctx):
        _count("t_load")
        return {"model": hash_bytes(f"model:{inputs['src']}:{params.get('v', 0)}".encode())}

    @registry.register("stage", "t_quant", meta={"inputs": ["model"]})
    def quant(inputs, params, ctx):
        _count("t_quant")
        return {"qmodel": hash_bytes(f"q:{inputs['model']}:{params['bits']}".encode()), "bits": params["bits"]}

    @registry.register("stage", "t_eval", meta={"inputs": ["qmodel"]})
    def ev(inputs, params, ctx):
        _count("t_eval")
        return {"score": float(len(inputs["qmodel"]))}

    @registry.register("stage", "t_nocache", meta={"cache": False})
    def nc(inputs, params, ctx):
        _count("t_nocache")
        return {"nc": 1}

    @registry.register("stage", "t_boom")
    def boom(inputs, params, ctx):
        raise RuntimeError("kaboom")

    @registry.register("stage", "t_gate")
    def gate(inputs, params, ctx):
        raise GateFailed("nope")

    @registry.register("stage", "t_seed")
    def seed(inputs, params, ctx):
        _count("t_seed")
        return {"seeds": inputs["_recipe"]["seeds"]}

    @registry.register("stage", "t_bad_return")
    def bad(inputs, params, ctx):
        return 5

    yield
    for n in ("t_load", "t_quant", "t_eval", "t_nocache", "t_boom", "t_gate", "t_seed", "t_bad_return"):
        registry.unregister("stage", n)


def _recipe(bits=4, v=0, seeds=(0,)) -> Recipe:
    return Recipe(
        name="r",
        inputs={"src": "hf:tiny"},
        stages=[
            StageSpec(name="t_load", params={"v": v}),
            StageSpec(name="t_quant", params={"bits": bits}),
            StageSpec(name="t_eval"),
        ],
        seeds=list(seeds),
    )


def test_run_basic_and_namespace():
    eng = RecipeEngine(None)
    ctx = NullContext()
    res = eng.run(_recipe(), ctx)
    assert [s["name"] for s in res["stages"]] == ["t_load", "t_quant", "t_eval"]
    assert res["outputs"]["bits"] == 4 and res["outputs"]["score"] > 0
    assert res["cache_hits"] == 0 and not res["cancelled"]
    assert ctx.last_progress == 1.0 and ctx.outputs["recipe_hash"] == _recipe().hash()
    assert CALLS == {"t_load": 1, "t_quant": 1, "t_eval": 1}


def test_cache_hit_second_run_and_persisted_across_engines():
    RecipeEngine(None).run(_recipe(), NullContext())
    CALLS.clear()
    res = RecipeEngine(None).run(_recipe(), NullContext())  # new engine: db-backed cache
    assert CALLS == {} and res["cache_hits"] == 3
    assert all(s["cached"] for s in res["stages"])


def test_change_one_stage_reruns_downstream_only():
    eng = RecipeEngine(None)
    eng.run(_recipe(bits=4), NullContext())
    CALLS.clear()
    res = eng.run(_recipe(bits=8), NullContext())
    assert CALLS == {"t_quant": 1, "t_eval": 1}  # load cached
    assert [s["cached"] for s in res["stages"]] == [True, False, False]
    CALLS.clear()
    eng.run(_recipe(bits=8, v=1), NullContext())  # change first stage -> everything reruns
    assert CALLS == {"t_load": 1, "t_quant": 1, "t_eval": 1}
    CALLS.clear()
    eng.run(_recipe(bits=4), NullContext())  # old config still cached
    assert CALLS == {}


def test_unchanged_upstream_output_keeps_downstream_cache():
    """Stage input hashing is by value: re-running t_load with a param that changes nothing it
    outputs (same model hash) must not invalidate downstream. Here v differs => different hash."""
    eng = RecipeEngine(None)
    eng.run(_recipe(v=0), NullContext())
    CALLS.clear()
    # explicit no-op param difference in t_eval only
    r = _recipe(v=0)
    r.stages[2].params = {"x": 1}
    eng.run(r, NullContext())
    assert CALLS == {"t_eval": 1}


def test_seeds_are_part_of_the_key_and_visible_to_stages():
    eng = RecipeEngine(None)
    r = Recipe(name="s", stages=[StageSpec(name="t_seed")], seeds=[1, 2])
    assert eng.run(r, NullContext())["outputs"]["seeds"] == [1, 2]
    CALLS.clear()
    eng.run(r, NullContext())
    assert CALLS == {}
    r2 = Recipe(name="s", stages=[StageSpec(name="t_seed")], seeds=[3])
    eng.run(r2, NullContext())
    assert CALLS == {"t_seed": 1}


def test_force_and_cache_disabled_and_nocache_meta():
    eng = RecipeEngine(None)
    eng.run(_recipe(), NullContext())
    CALLS.clear()
    eng.run(_recipe(), NullContext(), force=True)
    assert sum(CALLS.values()) == 3
    CALLS.clear()
    off = RecipeEngine(None, cache=False)
    off.run(_recipe(), NullContext())
    off.run(_recipe(), NullContext())
    assert CALLS["t_load"] == 2
    CALLS.clear()
    nc = Recipe(name="n", stages=[StageSpec(name="t_nocache")])
    eng.run(nc, NullContext())
    eng.run(nc, NullContext())
    assert CALLS["t_nocache"] == 2


def test_code_version_invalidates():
    eng = RecipeEngine(None)
    eng.run(_recipe(), NullContext())
    CALLS.clear()
    fn = registry.get("stage", "t_eval")
    fn.code_version = "2"
    try:
        eng.run(_recipe(), NullContext())
        assert CALLS == {"t_eval": 1}
    finally:
        del fn.code_version


def test_plan_reports_cache_state():
    eng = RecipeEngine(None)
    assert [r["cached"] for r in eng.plan(_recipe())] == [False, False, False]
    eng.run(_recipe(bits=4), NullContext())
    assert [r["cached"] for r in eng.plan(_recipe(bits=4))] == [True, True, True]
    assert [r["cached"] for r in eng.plan(_recipe(bits=8))] == [True, False, False]
    bad = Recipe(name="b", stages=[StageSpec(name="nope")])
    assert eng.plan(bad)[0]["registered"] is False


def test_unregistered_stage_fails_fast_without_running():
    r = Recipe(name="x", stages=[StageSpec(name="t_load"), StageSpec(name="does_not_exist")])
    with pytest.raises(NotFound):
        RecipeEngine(None).run(r, NullContext())
    assert CALLS == {}
    assert RecipeEngine(None).validate(r) == ["stage #1 'does_not_exist' is not registered"]


def test_error_wrapping():
    eng = RecipeEngine(None)
    with pytest.raises(StageFailed) as ei:
        eng.run(Recipe(name="x", stages=[StageSpec(name="t_boom")]), NullContext())
    assert ei.value.stage == "t_boom" and isinstance(ei.value.__cause__, RuntimeError)
    with pytest.raises(GateFailed):  # Forge errors pass through unwrapped
        eng.run(Recipe(name="x", stages=[StageSpec(name="t_gate")]), NullContext())
    with pytest.raises(StageFailed):
        eng.run(Recipe(name="x", stages=[StageSpec(name="t_bad_return")]), NullContext())


def test_cooperative_cancel_between_stages():
    ctx = NullContext()

    @registry.register("stage", "t_stopper")
    def stopper(inputs, params, ctx):
        ctx.stop = True
        return {"s": 1}

    try:
        r = Recipe(name="c", stages=[StageSpec(name="t_stopper"), StageSpec(name="t_load"), StageSpec(name="t_eval")])
        res = RecipeEngine(None).run(r, ctx)
        assert res["cancelled"] and len(res["stages"]) == 1
        assert CALLS == {}
    finally:
        registry.unregister("stage", "t_stopper")
    # a stage that observed a stop request must not have its (possibly partial) result cached
    @registry.register("stage", "t_stop_nocache")
    def sn(inputs, params, ctx):
        _count("t_stop_nocache")
        ctx.stop = True
        return {"p": 1}

    try:
        r = Recipe(name="c2", stages=[StageSpec(name="t_stop_nocache")])
        eng = RecipeEngine(None)
        eng.run(r, NullContext())
        eng.run(r, NullContext())
        assert CALLS["t_stop_nocache"] == 2
    finally:
        registry.unregister("stage", "t_stop_nocache")


def test_lineage_hook_and_store_edges():
    edges = []
    eng = RecipeEngine(None, lineage_hook=lambda c, p, rh, st: edges.append((c, p, rh, st)))
    r = _recipe()
    r.inputs = {"src": hash_bytes(b"source")}
    res = eng.run(r, NullContext())
    model = res["outputs"]["model"]
    qmodel = res["outputs"]["qmodel"]
    rh = r.hash()
    assert (model, r.inputs["src"], rh, "t_load") in edges
    assert (qmodel, model, rh, "t_quant") in edges
    # cache hits re-emit lineage
    edges.clear()
    eng.run(r, NullContext())
    assert (qmodel, model, rh, "t_quant") in edges

    class FakeStore:
        def __init__(self):
            self.edges, self.have = [], set()

        def add_edge(self, child, parent, recipe_hash=None, relation="derived_from"):
            self.edges.append((child, parent, recipe_hash, relation))

        def has(self, h):
            return h in self.have

    st = FakeStore()
    st.have = {model, qmodel, r.inputs["src"]}
    eng2 = RecipeEngine(st)
    eng2.run(r, NullContext())
    assert (qmodel, model, rh, "derived_from") in st.edges


def test_cached_output_missing_from_store_invalidates():
    class FakeStore:
        def __init__(self):
            self.have = None

        def has(self, h):
            return self.have is None or h in self.have

        def add_edge(self, *a, **k):
            pass

    st = FakeStore()
    eng = RecipeEngine(st)
    eng.run(_recipe(), NullContext())
    CALLS.clear()
    eng.run(_recipe(), NullContext())
    assert CALLS == {}
    st.have = set()  # store lost the artifacts (e.g. GC)
    CALLS.clear()
    eng.run(_recipe(), NullContext())
    assert CALLS["t_load"] == 1


def test_non_json_outputs_cached_in_process_only():
    @registry.register("stage", "t_obj")
    def obj(inputs, params, ctx):
        _count("t_obj")
        return {"thing": object()}

    try:
        eng = RecipeEngine(None)
        r = Recipe(name="o", stages=[StageSpec(name="t_obj")])
        eng.run(r, NullContext())
        eng.run(r, NullContext())
        assert CALLS["t_obj"] == 1
        RecipeEngine(None).run(r, NullContext())
        assert CALLS["t_obj"] == 2
    finally:
        registry.unregister("stage", "t_obj")


def test_clear_cache():
    eng = RecipeEngine(None)
    eng.run(_recipe(), NullContext())
    assert eng.clear_cache() >= 3
    CALLS.clear()
    eng.run(_recipe(), NullContext())
    assert sum(CALLS.values()) == 3


# ------------------------------------------------------------------ YAML

APPENDIX_C1 = """
recipe: w4a16-fp8kv-70b
inputs: {model: "sha256:base70b_bf16", calib: "sha256:mix_v3", chip: chipA-r3}
stages:
- calibrate: {set: mix_v3, n_seqs: 512, seq_len: 2048, stats: [hessian, channel_scale]}
- ptq: {method: gptq, bits: 4, group: 128, act_order: true, damp: 0.01, sym: false}
- pack: {layout: auto, target: chipA-r3}
- kv: {dtype: fp8_e4m3, residual_window: 128}
- gate: {suite: G-W4-v2, parent: "sha256:base70b_bf16", seeds: 3, stats: bootstrap_2000}
baselines: [fp8_w8a8, fresh_student_8b]
budget: {gpu_hours: 40, max_wall_hours: 12}
policy: {licence_check: true}
"""


def test_load_appendix_c1_and_roundtrip(tmp_path):
    r = load_recipe(APPENDIX_C1)
    assert r.name == "w4a16-fp8kv-70b" and r.stage_names() == ["calibrate", "ptq", "pack", "kv", "gate"]
    assert r.stages[1].params["bits"] == 4 and r.baselines == ["fp8_w8a8", "fresh_student_8b"]
    text = dump_recipe(r)
    assert "recipe: w4a16-fp8kv-70b" in text
    r2 = load_recipe(text)
    assert r2 == r and r2.hash() == r.hash()
    p = tmp_path / "r.yaml"
    p.write_text(APPENDIX_C1, encoding="utf-8")
    assert load_recipe(p) == r
    assert load_recipe(str(p)) == r
    assert load_recipe(r.model_dump(mode="json")) == r  # canonical dict form


def test_load_recipe_accepts_name_params_form_and_bare_names():
    r = load_recipe("name: x\nstages:\n- {name: a, params: {k: 1}}\n- b\n- c:\n")
    assert [(s.name, s.params) for s in r.stages] == [("a", {"k": 1}), ("b", {}), ("c", {})]


@pytest.mark.parametrize(
    "text",
    ["", "- 1", "name: x", "name: x\nstages: 5", "name: x\nstages:\n- {a: 1, b: 2}", "name: x\nstages:\n- a: 5",
     "name: x\nstages: []\nbogus: 1", "stages: []", "a: [unclosed"],
)
def test_load_recipe_errors(text):
    with pytest.raises(ConfigError):
        load_recipe(text)


def test_hash_algo_prefix_in_recipe_hash():
    assert Recipe(name="a").hash().startswith(hash_algo() + ":")
