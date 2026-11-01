"""``forge`` command line interface (argparse, no extra dependencies).

Job commands talk to a running ``forge serve-ui`` server when one is reachable (``--api URL`` or the default
host/port) and otherwise work directly on the local FORGE_HOME database, running the job in a foreground
in-process scheduler (so ``forge train --config x.yaml`` just works without a server).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

JOB_SHORTCUTS = ("train", "finetune", "distill", "compress", "eval", "serve", "bench", "probe")
TERMINAL = {"succeeded", "failed", "cancelled"}


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------
def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _set_nested(d: dict[str, Any], dotted: str, value: Any) -> None:
    cur = d
    parts = dotted.split(".")
    for p in parts[:-1]:
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = cur[p] = {}
        cur = nxt
    cur[parts[-1]] = value


def load_params(config: str | None, sets: list[str] | None) -> dict[str, Any]:
    """Params from a YAML/JSON file (plain params, or ``{params: {...}}``) plus ``--set a.b=value`` overrides."""
    import yaml

    params: dict[str, Any] = {}
    if config:
        p = Path(config)
        if not p.is_file():
            _die(f"config file not found: {config}")
        try:
            doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            _die(f"{config} is not valid YAML: {e}")
        if isinstance(doc, dict) and isinstance(doc.get("params"), dict) and set(doc) <= {"type", "name", "description", "params", "gpus", "tags"}:
            doc = doc["params"]
        if not isinstance(doc, dict):
            _die(f"{config} must contain a mapping at the top level")
        params = doc
    for item in sets or []:
        if "=" not in item:
            _die(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            val = yaml.safe_load(v)
        except yaml.YAMLError:
            val = v
        _set_nested(params, k.strip(), val)
    return params


def _parse_gpus(s: str | None) -> list[int] | None:
    if s is None:
        return None
    s = s.strip().lower()
    if s in ("", "none", "cpu"):
        return []
    if s == "auto":
        return None
    try:
        return [int(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        _die(f"--gpus expects a comma separated list of indices, 'cpu' or 'auto' (got {s!r})")
        return None


# --------------------------------------------------------------------------------------------
# backends: remote (HTTP) or local (database + in-process manager)
# --------------------------------------------------------------------------------------------
class RemoteBackend:
    kind = "remote"

    def __init__(self, url: str, token: str | None):
        import httpx

        self.url = url.rstrip("/")
        self.client = httpx.Client(
            base_url=self.url + "/api/v1", timeout=60.0, headers={"Authorization": f"Bearer {token}"} if token else {}
        )

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        r = self.client.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                body = r.json()
                msg = body.get("message") or body.get("detail") or r.text
                errs = body.get("errors") or []
            except ValueError:
                msg, errs = r.text, []
            for e in errs[:10]:
                print(f"  - {e.get('loc') or '<root>'}: {e.get('msg')}", file=sys.stderr)
            _die(f"{method} {path} -> HTTP {r.status_code}: {msg}", 1)
        return r.json() if r.content else None

    def submit(self, type_: str, params: dict[str, Any], name: str, gpus: list[int] | None, tags: list[str]) -> dict[str, Any]:
        return self._req("POST", "/jobs", json={"type": type_, "name": name, "params": params, "gpus": gpus, "tags": tags})

    def get(self, job_id: str) -> dict[str, Any]:
        return self._req("GET", f"/jobs/{job_id}")

    def list(self, status: str | None, type_: str | None, limit: int) -> list[dict[str, Any]]:
        return self._req("GET", "/jobs", params={k: v for k, v in (("status", status), ("type", type_), ("limit", limit)) if v})

    def logs(self, job_id: str, offset: int | None, tail: int = 200) -> tuple[list[str], int]:
        d = self._req("GET", f"/jobs/{job_id}/logs", params={"offset": offset} if offset is not None else {"tail": tail})
        return d["lines"], d["offset"]

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._req("POST", f"/jobs/{job_id}/cancel")

    def close(self) -> None:
        self.client.close()


class LocalBackend:
    kind = "local"

    def __init__(self) -> None:
        from forge.jobs.manager import JobManager
        from forge.jobs.repo import JobsRepo

        self.repo = JobsRepo()
        self.manager = JobManager(poll_s=0.2)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.manager.start()
            self._started = True

    def submit(self, type_: str, params: dict[str, Any], name: str, gpus: list[int] | None, tags: list[str]) -> dict[str, Any]:
        from forge.core.errors import ForgeError
        from forge.core.schemas import JobSpec

        try:
            row = self.manager.submit(JobSpec(type=type_, name=name, params=params, gpus=gpus, tags=tags))
        except ForgeError as e:
            for err in e.details.get("errors", [])[:10]:
                print(f"  - {err.get('loc') or '<root>'}: {err.get('msg')}", file=sys.stderr)
            _die(e.message, 1)
        return self.repo.public(row)

    def get(self, job_id: str) -> dict[str, Any]:
        from forge.core.errors import NotFound

        try:
            return self.repo.public(self.repo.require(job_id, "job"))
        except NotFound as e:
            _die(e.message, 1)
            raise

    def list(self, status: str | None, type_: str | None, limit: int) -> list[dict[str, Any]]:
        return [self.repo.public(r) for r in self.repo.list_jobs(status=status, type_=type_, limit=limit)]

    def logs(self, job_id: str, offset: int | None, tail: int = 200) -> tuple[list[str], int]:
        from forge.telemetry.logs import log_path, read_log, tail_lines

        row = self.repo.require(job_id, "job")
        path = log_path(row["run_dir"] or "")
        if offset is not None:
            text, new = read_log(path, offset)
            return text.splitlines(), new
        return tail_lines(path, tail)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self.repo.public(self.manager.cancel(job_id))

    def close(self) -> None:
        if self._started:
            self.manager.stop(kill_running=True)


def _server_alive(url: str, timeout: float = 0.4) -> bool:
    try:
        import httpx

        return httpx.get(url.rstrip("/") + "/api/v1/system/health", timeout=timeout).status_code == 200
    except Exception:
        return False


def get_backend(args: argparse.Namespace) -> RemoteBackend | LocalBackend:
    from forge.core.settings import get_settings

    st = get_settings()
    url = getattr(args, "api", None) or os.environ.get("FORGE_API_URL")
    token = getattr(args, "token", None) or st.api_token
    if url:
        if not _server_alive(url, 3.0):
            _die(f"no Forge server answering at {url}")
        return RemoteBackend(url, token)
    default = f"http://{'127.0.0.1' if st.api_host in ('0.0.0.0', '::') else st.api_host}:{st.api_port}"
    if _server_alive(default):
        return RemoteBackend(default, token)
    return LocalBackend()


def follow_job(be: RemoteBackend | LocalBackend, job_id: str, *, quiet_progress: bool = False) -> int:
    """Stream the log until the job ends; returns a process exit code (0 ok, 1 failed, 130 cancelled)."""
    offset: int | None = 0
    last_prog = -1.0
    try:
        while True:
            lines, offset = be.logs(job_id, offset)
            for ln in lines:
                print(ln, flush=True)
            job = be.get(job_id)
            if not quiet_progress and job["status"] == "running" and abs(job["progress"] - last_prog) >= 0.05:
                last_prog = job["progress"]
                print(f"[progress {job['progress'] * 100:5.1f}%] {job.get('progress_msg') or ''}", file=sys.stderr, flush=True)
            if job["status"] in TERMINAL:
                lines, offset = be.logs(job_id, offset)
                for ln in lines:
                    print(ln, flush=True)
                print(f"job {job_id}: {job['status']}" + (f" ({job['error']})" if job.get("error") else ""), file=sys.stderr)
                return {"succeeded": 0, "failed": 1, "cancelled": 130}[job["status"]]
            time.sleep(0.3)
    except KeyboardInterrupt:
        print(f"\ninterrupted: cancelling job {job_id} ...", file=sys.stderr)
        be.cancel(job_id)
        for _ in range(100):
            if be.get(job_id)["status"] in TERMINAL:
                break
            time.sleep(0.3)
        return 130


def _submit_and_maybe_follow(args: argparse.Namespace, type_: str, params: dict[str, Any], *, name: str | None = None) -> int:
    be = get_backend(args)
    try:
        gpus = _parse_gpus(getattr(args, "gpus", None))
        job = be.submit(type_, params, name or getattr(args, "name", None) or "", gpus, getattr(args, "tag", None) or [])
        print(f"submitted {job['id']} ({type_}) via {be.kind} backend")
        follow = getattr(args, "follow", None)
        if be.kind == "local":
            if follow is False:
                print("note: no server is running, so this queued job starts when `forge serve-ui` runs against this FORGE_HOME.")
                return 0
            be.start()  # type: ignore[union-attr]
            follow = True
        if follow is None:
            follow = type_ not in ("serve",)
        if not follow:
            return 0
        return follow_job(be, job["id"])
    finally:
        be.close()


# --------------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------------
def cmd_serve_ui(args: argparse.Namespace) -> int:
    if args.home:
        os.environ["FORGE_HOME"] = str(Path(args.home).expanduser())
    if args.token:
        os.environ["FORGE_API_TOKEN"] = args.token
    if args.max_concurrent:
        os.environ["FORGE_MAX_CONCURRENT_JOBS"] = str(args.max_concurrent)
    try:
        import uvicorn
    except ImportError:
        _die("uvicorn is not installed (`pip install uvicorn[standard]`)")
    from forge.core.settings import get_settings

    st = get_settings()
    host = args.host or st.api_host
    port = args.port or st.api_port
    print(f"Forge LLM UI/API: http://{'localhost' if host in ('0.0.0.0', '::') else host}:{port}  (home={st.home})")
    if host in ("0.0.0.0", "::") and not st.api_token:
        print("WARNING: listening on all interfaces without FORGE_API_TOKEN - anyone who can reach this port can run jobs. "
              "Set --token or use an SSH tunnel.", file=sys.stderr)
    from forge.api.app import create_app

    uvicorn.run(create_app(), host=host, port=port, log_level=args.log_level)
    return 0


def cmd_job_submit(args: argparse.Namespace) -> int:
    return _submit_and_maybe_follow(args, args.type, load_params(args.config, args.set))


def cmd_job_list(args: argparse.Namespace) -> int:
    be = get_backend(args)
    try:
        jobs = be.list(args.status, args.type, args.limit)
    finally:
        be.close()
    if args.json:
        _print_json(jobs)
        return 0
    if not jobs:
        print("no jobs")
        return 0
    print(f"{'ID':<18} {'TYPE':<14} {'STATUS':<10} {'PROG':>5}  NAME")
    for j in jobs:
        print(f"{j['id']:<18} {j['type']:<14} {j['status']:<10} {j['progress'] * 100:4.0f}%  {j['name']}")
    return 0


def cmd_job_show(args: argparse.Namespace) -> int:
    be = get_backend(args)
    try:
        _print_json(be.get(args.job_id))
    finally:
        be.close()
    return 0


def cmd_job_logs(args: argparse.Namespace) -> int:
    be = get_backend(args)
    try:
        if args.follow:
            return follow_job(be, args.job_id, quiet_progress=True)
        lines, _ = be.logs(args.job_id, None, args.tail)
        print("\n".join(lines))
    finally:
        be.close()
    return 0


def cmd_job_cancel(args: argparse.Namespace) -> int:
    be = get_backend(args)
    try:
        j = be.cancel(args.job_id)
        print(f"{j['id']}: {j['status']}" + (" (cancel requested)" if j.get("cancel_requested") and j["status"] == "running" else ""))
    finally:
        be.close()
    return 0


def cmd_shortcut(args: argparse.Namespace) -> int:
    params = load_params(getattr(args, "config", None), getattr(args, "set", None))
    if args.command == "probe":
        if args.quick:
            params["quick"] = True
    return _submit_and_maybe_follow(args, args.command, params)


def cmd_store(args: argparse.Namespace) -> int:
    try:
        from forge.jobs.repo import open_store

        store = open_store()
    except Exception as e:
        _die(f"the artifact store is not available: {type(e).__name__}: {e}", 1)
    from forge.jobs.repo import to_jsonable

    if args.store_cmd == "stats":
        _print_json(to_jsonable(store.stats()))
    elif args.store_cmd == "gc":
        rep = store.gc(dry_run=not args.execute)
        _print_json({"dry_run": not args.execute, "report": to_jsonable(rep)})
        if not args.execute:
            print("dry run: nothing deleted (pass --execute to delete)", file=sys.stderr)
    elif args.store_cmd == "verify":
        if hasattr(store, "verify_all"):
            rep = to_jsonable(store.verify_all(deep=not args.fast))
            _print_json(rep)
            bad = rep.get("bad") or rep.get("corrupt") or rep.get("failed") if isinstance(rep, dict) else None
            return 1 if bad else 0
        bad_list = [a["hash"] for a in store.list_artifacts(limit=100000) if not store.verify(a["hash"])]
        _print_json({"checked": "all", "bad": bad_list})
        return 1 if bad_list else 0
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from forge.core.errors import ForgeError
    from forge.jobs.repo import ModelsRepo

    repo = ModelsRepo()
    if args.models_cmd == "list":
        rows = repo.list(limit=1000)
        if args.json:
            _print_json(rows)
            return 0
        if not rows:
            print("no models registered")
        for r in rows:
            pb = f"{r['params_b']:.2f}B" if r.get("params_b") else "?"
            print(f"{r['id']:<16} {r['source']:<8} {pb:>8}  {r['name']}  [{r['status']}]  {r['path_or_repo']}")
        return 0
    if args.models_cmd == "register":
        from forge.jobs.modelinfo import register_local_model

        try:
            row = register_local_model(args.path, name=args.name, licence=args.licence)
        except ForgeError as e:
            _die(e.message, 1)
        _print_json({k: row[k] for k in ("id", "name", "path_or_repo", "params_b", "size_bytes", "licence")})
        return 0
    if args.models_cmd == "download":
        params: dict[str, Any] = {"repo_id": args.repo}
        if args.revision:
            params["revision"] = args.revision
        if args.name:
            params["local_name"] = args.name
        return _submit_and_maybe_follow(args, "model_download", params, name=f"download {args.repo}")
    return 2


def cmd_schema(args: argparse.Namespace) -> int:
    from forge.core.errors import ForgeError
    from forge.jobs.registry import get_job_type, job_type_names
    from forge.jobs.templates import job_type_payload

    if args.job_type == "list":
        for n in job_type_names():
            p = job_type_payload(get_job_type(n), with_schema=False)
            print(f"{n:<18} {'ok' if p['available'] else 'UNAVAILABLE':<12} {p['title']}")
        return 0
    try:
        p = job_type_payload(get_job_type(args.job_type))
    except ForgeError as e:
        _die(e.message, 1)
    if not p["available"]:
        _die(f"job type '{args.job_type}' is unavailable: {p['reason']}", 1)
    _print_json(p["defaults"] if args.defaults else p["schema"])
    return 0


def _try(fn: Any) -> Any:
    try:
        return fn()
    except Exception as e:
        return f"{type(e).__name__}: {e}"


def collect_doctor() -> dict[str, Any]:
    """Environment report (also used by tests). Never raises."""
    import importlib
    from importlib import metadata

    rep: dict[str, Any] = {"checks": [], "info": {}}

    def add(level: str, name: str, detail: str) -> None:
        rep["checks"].append({"level": level, "name": name, "detail": detail})

    from forge import __version__

    add("ok", "forge", f"version {__version__}")
    add("ok" if sys.version_info >= (3, 10) else "fail", "python", f"{sys.version.split()[0]} ({platform.platform()})")
    pk = {}
    for p in ("torch", "triton", "transformers", "huggingface_hub", "safetensors", "numpy", "pydantic", "fastapi", "uvicorn", "httpx",
              "pyarrow", "python-multipart", "nvidia-ml-py", "blake3", "zstandard", "pyyaml", "psutil"):
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pk[p] = None
    rep["info"]["packages"] = pk
    for p in ("torch", "triton"):
        if pk.get(p):
            add("ok", p, pk[p])
        else:
            add("warn" if p == "triton" else "fail", p, "not installed" + (" (needed for GPU kernels; kernels fall back to torch reference paths)" if p == "triton" else ""))
    for p in ("transformers", "huggingface_hub", "pyarrow", "python-multipart", "nvidia-ml-py"):
        add("ok" if pk.get(p) else "warn", p, pk.get(p) or "not installed (optional feature degrades gracefully)")

    # --- GPUs -----------------------------------------------------------------------------
    gpus: list[dict[str, Any]] = []
    torch_cuda: Any = None
    try:
        import torch

        torch_cuda = {"available": bool(torch.cuda.is_available()), "cuda_version": getattr(torch.version, "cuda", None)}
        if torch_cuda["available"]:
            for i in range(torch.cuda.device_count()):
                pr = torch.cuda.get_device_properties(i)
                gpus.append({"index": i, "name": pr.name, "sm": pr.major * 10 + pr.minor, "total_gb": round(pr.total_memory / 2**30, 1)})
    except Exception as e:
        torch_cuda = {"available": False, "error": f"{type(e).__name__}: {e}"}
    rep["info"]["torch_cuda"] = torch_cuda
    rep["info"]["gpus"] = gpus
    smi = shutil.which("nvidia-smi")
    add("ok" if smi else "warn", "nvidia-smi", smi or "not found (no NVIDIA driver on this host)")
    if gpus:
        for g in gpus:
            add("ok", f"gpu{g['index']}", f"{g['name']} sm_{g['sm']} {g['total_gb']} GiB")
    else:
        add("warn", "gpu", "no CUDA device visible to torch - training/serving would run on CPU (slow); fine for the UI and CPU tests")
    try:
        from forge.hw.capabilities import current_capabilities

        caps = current_capabilities()
        rep["info"]["capabilities"] = _try(lambda: {k: getattr(caps, k) for k in ("sm", "name", "tier", "has_bf16", "has_fp8", "has_fp4_block_scaled", "has_tma") if hasattr(caps, k)})
    except Exception as e:
        rep["info"]["capabilities"] = f"unavailable: {type(e).__name__}: {e}"

    # --- CUDA stack (driver <-> torch build <-> GPU arch <-> triton) -------------------------------
    try:
        from forge.hw import cuda_compat

        # probe (a throwaway subprocess that allocates on each GPU) only where there is something to probe
        crep = cuda_compat.check(probe=bool(smi) and bool(pk.get("torch")), probe_timeout=60.0)
        rep["info"]["cuda_compat"] = crep
        for f in crep["findings"]:
            if f["code"] == "no_nvidia_driver":
                continue  # already reported by the nvidia-smi / gpu checks above (CPU-only hosts stay at warn)
            add(f["level"], f"cuda:{f['code']}", f["message"] + (f" -> {f['fix']}" if f.get("fix") and f["level"] in ("warn", "fail") else ""))
    except Exception as e:
        add("warn", "cuda:check", f"CUDA compatibility check unavailable: {type(e).__name__}: {e}")

    # --- forge modules ------------------------------------------------------------------------
    mods = {}
    for m in ("store", "hw", "kernels", "formats", "models", "compress", "train", "serve", "eval", "planner"):
        try:
            importlib.import_module(f"forge.{m}")
            mods[m] = "ok"
        except Exception as e:
            mods[m] = f"{type(e).__name__}: {e}"
    rep["info"]["modules"] = mods
    bad = [m for m, v in mods.items() if v != "ok"]
    add("ok" if not bad else "warn", "forge modules", "all importable" if not bad else "not importable: " + ", ".join(bad))
    try:
        from forge.jobs.registry import list_job_types

        unavailable = []
        for s in list_job_types():
            ok, reason = s.availability()
            if not ok:
                unavailable.append(f"{s.name} ({reason})")
        rep["info"]["unavailable_job_types"] = unavailable
        add("ok" if not unavailable else "warn", "job types", "all available" if not unavailable else f"{len(unavailable)} unavailable: " + "; ".join(unavailable)[:400])
    except Exception as e:
        add("warn", "job types", f"{type(e).__name__}: {e}")

    # --- paths & disk ---------------------------------------------------------------------------
    try:
        from forge.core.paths import forge_home

        home = forge_home()
        usage = shutil.disk_usage(home)
        free_gb = usage.free / 2**30
        writable = os.access(home, os.W_OK)
        add("ok" if writable and free_gb > 20 else "warn", "FORGE_HOME", f"{home} ({free_gb:.0f} GiB free{'' if writable else ', NOT writable'})")
        rep["info"]["home"] = str(home)
    except Exception as e:
        add("fail", "FORGE_HOME", f"{type(e).__name__}: {e}")
    add("info", "GPU kernels", "untested on this host: Triton kernels are implemented against the torch reference oracles but "
        "have not been validated on NVIDIA hardware - run `forge probe` then the GPU tests (scripts/run_gpu_tests.sh) on the GPU host")
    return rep


def cmd_doctor(args: argparse.Namespace) -> int:
    rep = collect_doctor()
    if args.json:
        _print_json(rep)
    else:
        tag = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]", "info": "[info]"}
        for c in rep["checks"]:
            print(f"{tag.get(c['level'], '[ ?? ]')} {c['name']}: {c['detail']}")
    return 1 if any(c["level"] == "fail" for c in rep["checks"]) else 0


def cmd_cuda_check(args: argparse.Namespace) -> int:
    from forge.hw import cuda_compat

    rep = cuda_compat.check(probe=not args.no_probe)
    print(json.dumps(rep, indent=2) if args.json else cuda_compat.format_report(rep))
    return 1 if not rep["ok"] or (args.strict and rep["worst"] == "warn") else 0


# --------------------------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------------------------
def _add_job_opts(p: argparse.ArgumentParser, *, config: bool = True) -> None:
    if config:
        p.add_argument("--config", "-c", help="YAML/JSON file with the job parameters")
        p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a parameter (dotted keys; value parsed as YAML); repeatable")
    p.add_argument("--name", help="job name")
    p.add_argument("--gpus", help="GPU indices (e.g. 0,1), 'cpu' or 'auto' (default: scheduler decides)")
    p.add_argument("--tag", action="append", help="tag (repeatable)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--follow", dest="follow", action="store_true", default=None, help="stream the log until the job ends")
    g.add_argument("--detach", dest="follow", action="store_false", help="submit and return immediately")
    p.add_argument("--api", help="server URL (default: auto-detect http://127.0.0.1:8765, else run locally)")
    p.add_argument("--token", help="API token (default: $FORGE_API_TOKEN)")


def build_parser() -> argparse.ArgumentParser:
    from forge import __version__

    ap = argparse.ArgumentParser(prog="forge", description="Forge LLM: train, fine-tune, compress, evaluate and serve LLMs.")
    ap.add_argument("--version", action="version", version=f"forge {__version__}")
    ap.add_argument("--home", help="FORGE_HOME directory (default: $FORGE_HOME or ~/.forge)")
    sub = ap.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("serve-ui", help="start the web UI + REST API")
    p.add_argument("--host", help="bind address (default 127.0.0.1; use 0.0.0.0 on a rented box together with --token)")
    p.add_argument("--port", type=int, help="port (default 8765)")
    p.add_argument("--home", default=argparse.SUPPRESS, help="FORGE_HOME directory")
    p.add_argument("--token", help="require this bearer token for /api/v1")
    p.add_argument("--max-concurrent", type=int, dest="max_concurrent", help="max concurrent non-light jobs")
    p.add_argument("--log-level", default="info")
    p.set_defaults(fn=cmd_serve_ui)

    jp = sub.add_parser("job", help="submit / inspect / cancel jobs")
    jsub = jp.add_subparsers(dest="job_cmd", metavar="SUBCOMMAND", required=True)
    p = jsub.add_parser("submit", help="submit a job of any type")
    p.add_argument("type", help="job type (see `forge schema list`)")
    _add_job_opts(p)
    p.set_defaults(fn=cmd_job_submit)
    p = jsub.add_parser("list", help="list jobs")
    p.add_argument("--status")
    p.add_argument("--type")
    p.add_argument("--limit", type=int, default=30)
    p.add_argument("--json", action="store_true")
    p.add_argument("--api")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_job_list)
    p = jsub.add_parser("show", help="show one job as JSON")
    p.add_argument("job_id")
    p.add_argument("--api")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_job_show)
    p = jsub.add_parser("logs", help="print (or follow) a job log")
    p.add_argument("job_id")
    p.add_argument("--tail", type=int, default=200)
    p.add_argument("-f", "--follow", action="store_true")
    p.add_argument("--api")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_job_logs)
    p = jsub.add_parser("cancel", help="cancel a queued or running job")
    p.add_argument("job_id")
    p.add_argument("--api")
    p.add_argument("--token")
    p.set_defaults(fn=cmd_job_cancel)

    for name, helptext in (
        ("train", "pre-train / continue training"), ("finetune", "fine-tune (full or LoRA)"), ("distill", "distillation"),
        ("compress", "compress / quantise"), ("eval", "evaluate and gate"), ("serve", "start an inference engine"),
        ("bench", "kernel benchmarks"), ("probe", "hardware probe"),
    ):
        p = sub.add_parser(name, help=helptext)
        _add_job_opts(p)
        if name == "probe":
            p.add_argument("--quick", action="store_true", help="shortened micro-benchmarks")
        p.set_defaults(fn=cmd_shortcut)

    sp = sub.add_parser("store", help="artifact store maintenance")
    ssub = sp.add_subparsers(dest="store_cmd", metavar="SUBCOMMAND", required=True)
    ssub.add_parser("stats", help="sizes, dedup ratio, tiers").set_defaults(fn=cmd_store)
    p = ssub.add_parser("gc", help="garbage collect (dry-run unless --execute)")
    p.add_argument("--execute", action="store_true", help="actually delete unreachable objects")
    p.set_defaults(fn=cmd_store)
    p = ssub.add_parser("verify", help="verify artifact integrity")
    p.add_argument("--fast", action="store_true", help="skip deep (content hash) verification")
    p.set_defaults(fn=cmd_store)

    mp = sub.add_parser("models", help="model registry")
    msub = mp.add_subparsers(dest="models_cmd", metavar="SUBCOMMAND", required=True)
    p = msub.add_parser("list")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_models)
    p = msub.add_parser("register", help="register a local model directory")
    p.add_argument("path")
    p.add_argument("--name")
    p.add_argument("--licence")
    p.set_defaults(fn=cmd_models)
    p = msub.add_parser("download", help="download a Hugging Face repo into FORGE_HOME/models")
    p.add_argument("repo", help="org/name")
    p.add_argument("--revision")
    _add_job_opts(p, config=False)
    p.set_defaults(fn=cmd_models)

    p = sub.add_parser("schema", help="print the JSON schema of a job type's config (`list` lists types)")
    p.add_argument("job_type")
    p.add_argument("--defaults", action="store_true", help="print default values instead of the schema")
    p.set_defaults(fn=cmd_schema)

    p = sub.add_parser("doctor", help="check the environment (python, torch, GPUs, triton, modules, disk)")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("cuda-check", help="check the driver / torch / GPU / Triton CUDA stack (CUDA 12.x - 13.4) and say what to fix")
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.add_argument("--no-probe", action="store_true", help="skip the per-device allocation test (no torch import)")
    p.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    p.set_defaults(fn=cmd_cuda_check)
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if getattr(args, "home", None):
        os.environ["FORGE_HOME"] = str(Path(args.home).expanduser())
    if not getattr(args, "fn", None):
        ap.print_help()
        return 0
    try:
        return int(args.fn(args) or 0)
    except KeyboardInterrupt:
        return 130
    except SystemExit:
        raise
    except Exception as e:
        from forge.core.errors import ForgeError

        if isinstance(e, ForgeError):
            print(f"error: {e.message}", file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
