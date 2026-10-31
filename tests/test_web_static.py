"""Static checks for the vanilla-JS front end in ``forge/web/static`` (no browser needed).

* index.html references existing files, is well formed and loads nothing from the network
* every JS module's relative imports resolve (import graph) and parses (``node --check`` when node exists)
* the router table (routes.js) matches the sidebar navigation in index.html and the page modules on disk
* every API path the UI calls belongs to the documented REST surface (docs/ARCHITECTURE.md section 7)
* pure-logic modules (yaml, schema validation, router matching) behave, executed under node when available
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parents[1] / "forge" / "web" / "static"
JS_FILES = sorted(STATIC.rglob("*.js"))
NODE = shutil.which("node")

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}

API_ROOTS = {
    "system", "hardware", "models", "datasets", "schemas", "configs", "jobs", "recipes", "planner", "claims", "artifacts",
    "storage", "evals", "gates", "kernels", "deployments", "serve", "checkpoints", "events",
}


class _Collector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.hrefs: list[str] = []
        self.srcs: list[str] = []
        self.nav_links: list[str] = []
        self.ids: list[str] = []
        self.in_nav = 0
        self.title = ""
        self._in_title = False
        self.meta: dict[str, str] = {}
        self.html_lang = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.html_lang = a.get("lang", "")
        if tag == "meta" and a.get("name"):
            self.meta[a["name"]] = a.get("content", "")
        if tag == "link" and a.get("href"):
            self.hrefs.append(a["href"])
        if tag == "script" and a.get("src"):
            self.srcs.append(a["src"])
        if a.get("id"):
            self.ids.append(a["id"])
        if tag == "nav":
            self.in_nav += 1
        if tag == "a" and self.in_nav and a.get("href"):
            self.nav_links.append(a["href"])
        if tag == "title":
            self._in_title = True
        if tag not in VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):  # <br/> style
        self.handle_starttag(tag, attrs)
        if tag not in VOID and self.stack and self.stack[-1] == tag:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if tag == "nav":
            self.in_nav -= 1
        if tag == "title":
            self._in_title = False
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()

    def handle_data(self, data):
        if self._in_title:
            self.title += data


@pytest.fixture(scope="module")
def index():
    p = STATIC / "index.html"
    assert p.exists(), "forge/web/static/index.html missing"
    c = _Collector()
    c.feed(p.read_text(encoding="utf-8"))
    c.close()
    return c


def test_index_html_is_valid_and_local(index: _Collector):
    assert not index.errors, index.errors
    assert not index.stack, f"unclosed tags: {index.stack}"
    assert index.html_lang, "<html lang> missing"
    assert index.title.strip(), "<title> missing"
    assert "viewport" in index.meta, "viewport meta missing"
    assert len(index.ids) == len(set(index.ids)), "duplicate ids in index.html"
    for ref in index.hrefs + index.srcs:
        assert not re.match(r"^(https?:)?//", ref), f"external resource not allowed: {ref}"
        if ref.startswith(("data:", "#")):
            continue
        assert (STATIC / ref).exists(), f"index.html references missing file {ref}"
    assert any(s.endswith("main.js") for s in index.srcs)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'type="module"' in html
    assert "view" in index.ids and "toasts" in index.ids and "nav" in index.ids


def test_css_has_light_dark_and_no_remote_imports():
    css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in css
    assert 'data-theme="dark"' in css and 'data-theme="light"' in css
    assert not re.search(r"@import\s+url\(\s*['\"]?https?:", css)
    assert not re.search(r"url\(\s*['\"]?https?:", css)
    assert "body" in css and "--bg" in css


IMPORT_RE = re.compile(r"""(?:^|[;\s}])import\s*(?:[\w*{}\s,$]+?\s*from\s*)?['"]([^'"]+)['"]""", re.M)
DYN_RE = re.compile(r"""\bimport\(\s*['"]([^'"]+)['"]\s*\)""")
EXPORT_FROM_RE = re.compile(r"""\bexport\s+(?:\*|\{[^}]*\})\s*from\s*['"]([^'"]+)['"]""")


def _strip_comments(src: str) -> str:
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"(?m)^\s*//.*$", "", src)


def test_js_files_exist():
    assert len(JS_FILES) >= 20
    names = {p.name for p in JS_FILES}
    for need in ("api.js", "charts.js", "forms.js", "router.js", "state.js", "main.js", "routes.js", "yaml.js", "mock.js", "ui.js"):
        assert need in names, f"{need} missing"


@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(STATIC)))
def test_js_import_graph_resolves(path: Path):
    src = _strip_comments(path.read_text(encoding="utf-8"))
    specs = IMPORT_RE.findall(src) + DYN_RE.findall(src) + EXPORT_FROM_RE.findall(src)
    for spec in specs:
        assert not re.match(r"^(https?:)?//", spec), f"{path.name}: remote import {spec}"
        assert spec.startswith("."), f"{path.name}: bare import {spec!r} (no bundler - use relative paths)"
        target = (path.parent / spec).resolve()
        assert target.exists(), f"{path.relative_to(STATIC)} imports missing module {spec}"
        assert target.suffix == ".js"
    assert "from __future__" not in src


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("path", JS_FILES, ids=lambda p: str(p.relative_to(STATIC)))
def test_js_syntax_with_node(path: Path):
    # copy to .mjs so old node versions also treat it as an ES module
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / (path.stem + ".mjs")
        tmp.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        r = subprocess.run([NODE, "--check", str(tmp)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, f"{path.name}: {r.stderr}"


def _routes():
    src = (STATIC / "js" / "routes.js").read_text(encoding="utf-8")
    entries = re.findall(r"\{\s*id:\s*'([^']+)',\s*path:\s*'([^']+)',\s*title:\s*'([^']+)',\s*nav:\s*(true|false)(.*?)module:\s*'([^']+)'\s*\}", src, flags=re.S)
    return [{"id": i, "path": p, "title": t, "nav": n == "true", "module": m} for i, p, t, n, _mid, m in entries]


def test_router_pages_exist_and_match_nav(index: _Collector):
    routes = _routes()
    assert len(routes) >= 16
    ids = [r["id"] for r in routes]
    assert len(ids) == len(set(ids)), "duplicate route ids"
    paths = [r["path"] for r in routes]
    assert len(paths) == len(set(paths)), "duplicate route paths"
    for r in routes:
        mod = STATIC / "js" / r["module"]
        assert mod.exists(), f"route {r['id']} -> missing page module {r['module']}"
        text = mod.read_text(encoding="utf-8")
        assert re.search(r"export\s+default\s+(async\s+)?function", text), f"{r['module']} must export a default render function"
    nav_routes = {"#" + r["path"] for r in routes if r["nav"]}
    nav_links = set(index.nav_links)
    assert nav_routes == nav_links, f"nav mismatch: only in routes={nav_routes - nav_links}, only in index.html={nav_links - nav_routes}"
    # every page required by ARCHITECTURE.md section 8 exists
    required = {"dashboard", "models", "datasets", "hardware", "new-job", "jobs", "job-detail", "finetune", "recipes", "compare", "inference", "kernels", "storage", "claims", "settings"}
    assert required <= set(ids), f"missing pages: {required - set(ids)}"
    # static paths must be declared before parameterised ones that could shadow them (/jobs/new before /jobs/:id)
    assert paths.index("/jobs/new") < paths.index("/jobs/:id")
    # every page module file on disk is routed
    routed = {r["module"].removeprefix("./") for r in routes}
    on_disk = {str(p.relative_to(STATIC / "js")).replace("\\", "/") for p in (STATIC / "js" / "pages").glob("*.js")}
    assert on_disk <= routed, f"unrouted page modules: {on_disk - routed}"


CALL_RE = re.compile(r"""\b(?:api\.(?:get|post|put|patch|del|upload)|tryGet|sse|request\(\s*'[A-Z]+',)\(?\s*[`'"](/[^`'"?$]*)""")


def test_api_paths_belong_to_the_documented_surface():
    seen = set()
    for p in JS_FILES:
        if p.name == "mock.js":
            continue
        src = _strip_comments(p.read_text(encoding="utf-8"))
        for m in CALL_RE.finditer(src):
            path = m.group(1)
            seen.add(path)
            root = path.strip("/").split("/")[0]
            assert root in API_ROOTS, f"{p.name}: undocumented API path {path}"
    assert len(seen) >= 30, f"expected the UI to call many endpoints, found {sorted(seen)}"
    # the contract's key streams are used
    joined = " ".join(seen)
    for need in ("/jobs/", "/system/gpus/stream", "/schemas/job-types", "/planner/estimate", "/storage/gc", "/evals/compare", "/serve/simulate"):
        assert any(need in s or s in need for s in seen) or need in joined, f"UI never calls {need}"


def test_no_network_resources_in_sources():
    allowed = ("http://www.w3.org/2000/svg",)
    for p in list(JS_FILES) + [STATIC / "index.html", STATIC / "css" / "app.css"]:
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"https?://[^\s'\"`)<>]+", text):
            assert m.group(0).startswith(allowed) or p.name == "mock.js" and "example" in m.group(0), f"{p.name}: network URL {m.group(0)}"


def test_mock_mode_is_labelled_demo():
    main = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    assert "DEMO DATA" in main
    mock = (STATIC / "js" / "mock.js").read_text(encoding="utf-8")
    assert "DEMO" in mock and "synthetic" in mock.lower()
    state = (STATIC / "js" / "state.js").read_text(encoding="utf-8")
    assert "mock" in state and "sessionStorage" in state and "try" in state


def test_localstorage_is_always_guarded():
    for p in JS_FILES:
        src = _strip_comments(p.read_text(encoding="utf-8"))
        if "localStorage" in src:
            for m in re.finditer(r"localStorage", src):
                window = src[max(0, m.start() - 300): m.start() + 200]
                assert "try" in window or "catch" in window, f"{p.name}: unguarded localStorage near: {window[-120:]!r}"


# --------------------------------------------------------------------------- logic executed under node
def _node_eval(script: str) -> dict:
    assert NODE
    r = subprocess.run([NODE, "--input-type=module", "-e", script], capture_output=True, text=True, timeout=60, cwd=str(STATIC))
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _uri(rel: str) -> str:
    return (STATIC / rel).resolve().as_uri()


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_yaml_roundtrip_and_parse():
    out = _node_eval(f"""
import {{ dump, parse, parseAny }} from '{_uri('js/yaml.js')}';
const obj = {{ name: 'w4 recipe', inputs: {{ model: 'meta/llama', n: 3, f: 0.00015, flag: true, none: null, s: '123', t: 'yes: no' }},
  stages: [{{ name: 'ptq', params: {{ bits: 4, groups: [1,2,3], nested: {{ a: [{{ x: 1, y: [{{ z: 2 }}] }}], e: [], o: {{}} }} }} }}, {{ name: 'eval', params: {{}} }}], seeds: [0, 1], multi: 'a\\nb' }};
const back = parse(dump(obj));
const hand = parse(`# comment\\nname: x  # trailing\\nlst:\\n- a\\n- b: 1\\n  c: [1, "two", {{k: v}}]\\nblk: |\\n  l1\\n  l2\\nfold: >-\\n  a\\n  b\\nq: 'it''s'\\nn: -3.5e2\\nempty:\\nz: {{}}\\n`);
let bad = null; try {{ parse('a: 1\\n  b: 2\\n'); }} catch (e) {{ bad = e.name; }}
console.log(JSON.stringify({{ same: JSON.stringify(back) === JSON.stringify(obj), hand, bad, any: parseAny('{{"a": [1, 2]}}') }}));
""")
    assert out["same"] is True
    assert out["hand"] == {"name": "x", "lst": ["a", {"b": 1, "c": [1, "two", {"k": "v"}]}], "blk": "l1\nl2\n", "fold": "a b", "q": "it's", "n": -350, "empty": None, "z": {}}
    assert out["bad"] == "YamlError"
    assert out["any"] == {"a": [1, 2]}


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_schema_validation_and_defaults():
    out = _node_eval(f"""
import {{ validate, defaultFor, resolve }} from '{_uri('js/forms.js')}';
const schema = {{ type: 'object', additionalProperties: false, required: ['model'], properties: {{
  model: {{ type: 'string', minLength: 1 }},
  lr: {{ type: 'number', minimum: 1e-7, maximum: 1e-2, default: 3e-4 }},
  steps: {{ type: 'integer', minimum: 1, 'x-nullable': true, default: null }},
  mode: {{ type: 'string', enum: ['a', 'b'] }},
  tags: {{ type: 'array', items: {{ type: 'string' }}, maxItems: 2 }},
  opt: {{ type: 'object', properties: {{ k: {{ type: 'integer', minimum: 0 }} }} }},
  u: {{ anyOf: [{{ type: 'integer' }}, {{ type: 'string', enum: ['auto'] }}] }},
  m: {{ type: 'object', additionalProperties: {{ type: 'number' }} }} }} }};
const good = {{ model: 'x', lr: 1e-4, steps: null, mode: 'a', tags: ['a'], opt: {{ k: 1 }}, u: 'auto', m: {{ a: 1.5 }} }};
const bad = {{ lr: 5, steps: 0, mode: 'z', tags: ['a', 'b', 'c'], opt: {{ k: -1 }}, u: 'nope', m: {{ a: 'x' }}, extra: 1 }};
const paths = validate(schema, bad).map(e => e.path).sort();
const nullable = resolve({{ anyOf: [{{ type: 'integer' }}, {{ type: 'null' }}] }});
console.log(JSON.stringify({{ good: validate(schema, good).length, paths, nullable: nullable.nullable && nullable.schema.type, d: defaultFor(schema.properties.lr), d2: defaultFor(schema.properties.steps), d3: defaultFor(schema), arr: defaultFor({{ type: 'array', items: {{ type: 'string' }} }}) }}));
""")
    assert out["good"] == 0
    assert out["paths"] == sorted(["model", "lr", "steps", "mode", "tags", "opt.k", "u", "m.a", "extra"])
    assert out["nullable"] == "integer"
    assert out["d"] == 3e-4 and out["d2"] is None and out["d3"] == {"model": ""} and out["arr"] == []


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_router_matching_and_compare_ranking():
    out = _node_eval(f"""
import {{ matchRoute, parseHash, buildHash }} from '{_uri('js/router.js')}';
import {{ rankWithTies }} from '{_uri('js/pages/compare.js')}';
const a = matchRoute('/jobs/new'); const b = matchRoute('/jobs/job_1'); const c = matchRoute('/recipes/rcp 1'); const d = matchRoute('/nope');
const p = parseHash('#/jobs/x?tab=logs&q=a%20b');
const rows = [
  {{ id: 'a', metrics: {{ m: {{ value: 0.50, ci_low: 0.48, ci_high: 0.52 }} }} }},
  {{ id: 'b', metrics: {{ m: {{ value: 0.51, ci_low: 0.49, ci_high: 0.53 }} }} }},
  {{ id: 'c', metrics: {{ m: {{ value: 0.70, ci_low: 0.69, ci_high: 0.71 }} }} }},
  {{ id: 'd', metrics: {{ m: {{ value: 0.90 }} }} }} ];
const r = rankWithTies(rows, 'm', 'min');
console.log(JSON.stringify({{ a: a && a.route.id, b: b && [b.route.id, b.params.id], c: c && c.params.id, d, p, h: buildHash('/x', {{ a: 1, b: '' }}),
  ranks: Object.fromEntries([...r.entries()].map(([k, v]) => [k, [v.group, v.tied, v.known]])) }}));
""")
    assert out["a"] == "new-job" and out["b"] == ["job-detail", "job_1"] and out["c"] == "rcp 1" and out["d"] is None
    assert out["p"] == {"path": "/jobs/x", "query": {"tab": "logs", "q": "a b"}}
    assert out["h"] == "#/x?a=1"
    # a and b have overlapping CIs -> same rank group & tied; c is clearly worse; d has no CI so no rank claim is made
    assert out["ranks"]["a"][0] == out["ranks"]["b"][0] == 1 and out["ranks"]["a"][1] and out["ranks"]["b"][1]
    assert out["ranks"]["c"][0] == 2 and out["ranks"]["c"][1] is False
    assert out["ranks"]["d"][2] is False
