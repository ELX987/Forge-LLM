#!/usr/bin/env python3
"""Helper for scripts/gpu_smoke.sh and scripts/run_gpu_tests.sh.

Subcommands
-----------
``step <report_dir> <name> [--timeout S] [--allow-fail] -- <command...>``
    Run a command, tee its output to ``<report_dir>/<name>.log`` and append a record
    (name, command, exit code, seconds, last lines) to ``<report_dir>/steps.json``.
``env <report_dir>``
    Write ``<report_dir>/env.json`` (GPU, driver, torch, triton, python, forge versions, dependency status).
``preflight <report_dir>``
    Check that THIS interpreter has the project's hard dependencies (forge installed as a package, fastapi, uvicorn,
    zstandard, pytest; pytest-timeout is optional), print a banner with the exact ``pip install -e ".[gpu,serve,parquet,dev]"``
    command for this interpreter when something is missing, write ``<report_dir>/preflight.json``.  Always exits 0 (the
    caller continues; tests that need a missing package are skipped with the reason).
``bench-export <report_dir>``
    Dump the newest ``kernel_bench`` rows from ``$FORGE_HOME/forge.db`` to ``<report_dir>/bench_summary.json``.
``finalize <report_dir>``
    Combine env.json, steps.json and any ``junit_*.xml`` into ``report.json`` and ``report.md``.
    The markdown file is what you paste back when asking for help.

Only the standard library is required so the report can be produced even when the Forge install is broken.
Nothing here is a measurement: it only records what was run and what happened.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

TAIL_LINES = 40


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def cmd_step(argv: list[str]) -> int:
    if len(argv) < 3 or "--" not in argv:
        print("usage: step <report_dir> <name> [--timeout S] [--allow-fail] -- <command...>", file=sys.stderr)
        return 64
    sep = argv.index("--")
    head, command = argv[:sep], argv[sep + 1 :]
    report_dir, name = Path(head[0]), head[1]
    timeout = 1800.0
    allow_fail = False
    i = 2
    while i < len(head):
        if head[i] == "--timeout":
            timeout = float(head[i + 1])
            i += 2
        elif head[i] == "--allow-fail":
            allow_fail = True
            i += 1
        else:
            print(f"unknown option {head[i]}", file=sys.stderr)
            return 64
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = report_dir / f"{name}.log"
    print(f"\n=== [{name}] {' '.join(command)}", flush=True)
    t0 = time.time()
    code: int
    timed_out = False
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8", errors="replace") as logf:
        try:
            proc = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
            )
        except OSError as e:
            logf.write(f"failed to start: {e}\n")
            lines.append(f"failed to start: {e}")
            code = 127
            proc = None
        if proc is not None:
            assert proc.stdout is not None
            try:
                deadline = t0 + timeout
                for line in proc.stdout:
                    sys.stdout.write(line)
                    logf.write(line)
                    lines.append(line.rstrip("\n"))
                    if len(lines) > 4000:
                        del lines[:2000]
                    if time.time() > deadline:
                        timed_out = True
                        proc.kill()
                        break
                code = proc.wait()
            except KeyboardInterrupt:
                proc.kill()
                code = 130
    dur = time.time() - t0
    rec = {
        "name": name,
        "command": command,
        "exit_code": code,
        "timed_out": timed_out,
        "seconds": round(dur, 2),
        "allow_fail": allow_fail,
        "tail": lines[-TAIL_LINES:],
        "log": log_path.name,
        "finished": _now(),
    }
    steps_path = report_dir / "steps.json"
    steps = _load(steps_path, [])
    steps = [s for s in steps if s.get("name") != name] + [rec]
    steps_path.write_text(json.dumps(steps, indent=2), encoding="utf-8")
    status = "ok" if code == 0 else ("TIMEOUT" if timed_out else f"exit {code}")
    print(f"=== [{name}] {status} in {dur:.1f}s", flush=True)
    return 0 if (code == 0 or allow_fail) else code


INSTALL_EXTRAS = "gpu,serve,parquet,dev"
REPO_ROOT = Path(__file__).resolve().parent.parent
# (import name, distribution name, required)
DEPENDENCIES = [
    ("forge", "forge-llm", True),
    ("fastapi", "fastapi", True),
    ("uvicorn", "uvicorn", True),
    ("zstandard", "zstandard", True),
    ("pytest", "pytest", True),
    ("pytest_timeout", "pytest-timeout", False),
]


def install_command(python: str | None = None) -> str:
    """The exact command that installs everything the suite needs into the given interpreter."""
    py = python or sys.executable
    py = f'"{py}"' if " " in py else py
    return f'{py} -m pip install -e ".[{INSTALL_EXTRAS}]"'


def _source_tree_version() -> str:
    try:
        import re

        text = (REPO_ROOT / "forge" / "__version__.py").read_text(encoding="utf-8")
        m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)", text)
        return m.group(1) if m else "?"
    except OSError:
        return "?"


def dep_status(module: str, dist: str) -> str:
    """Version string, or ``missing (<reason>)``.  ``forge`` that is only importable from the source tree (no installed
    distribution) is reported as ``not installed (running from source tree)`` instead of ``missing (ModuleNotFoundError)``."""
    import importlib.metadata as md
    import importlib.util

    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - broken metadata: fall through to the import probe
        pass
    if module == "forge" and (REPO_ROOT / "forge" / "__init__.py").is_file():
        return "not installed (running from source tree)"
    try:
        if importlib.util.find_spec(module) is not None:
            m = __import__(module)
            return str(getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        return f"missing ({type(e).__name__})"
    return "missing (ModuleNotFoundError)"


def preflight_report() -> dict:
    deps: dict[str, str] = {}
    problems: list[str] = []
    optional_missing: list[str] = []
    for module, dist, required in DEPENDENCIES:
        st = dep_status(module, dist)
        deps[dist] = st
        if st.startswith("missing"):
            (problems if required else optional_missing).append(dist)
        elif st.startswith("not installed") and required:
            problems.append(dist)
    return {"python": sys.executable, "deps": deps, "missing_required": problems, "missing_optional": optional_missing,
            "install_command": install_command(), "ok": not problems}


TORCH_WARNING = ("WARNING: do not let pip replace the NVIDIA/CUDA torch build already installed on this GPU host (a plain install can pull a\n"
                 "CPU-only or mismatched torch from PyPI).  If torch is already the right build, pin it first, e.g.\n"
                 "    python -m pip freeze | grep -i -E '^(torch|triton|nvidia-)' > /tmp/gpu-pins.txt  &&  <install command above> -c /tmp/gpu-pins.txt\n"
                 "or install only the missing packages with --no-deps (e.g. python -m pip install --no-deps -e . ; python -m pip install fastapi uvicorn zstandard pytest).")


def format_banner(rep: dict) -> str:
    bar = "=" * 78
    lines = [bar, "FORGE ENVIRONMENT PREFLIGHT: this Python is missing project dependencies", bar,
             f"interpreter: {rep['python']}"]
    for dist in rep["missing_required"]:
        lines.append(f"  - {dist}: {rep['deps'][dist]}")
    for dist in rep["missing_optional"]:
        lines.append(f"  - {dist}: {rep['deps'][dist]}  (optional: per-test hang watchdog is disabled)")
    lines += ["", "Install everything the suite needs into THIS interpreter (run from the repository root):", "",
              f"    {rep['install_command']}", "",
              TORCH_WARNING, "",
              "Continuing anyway: tests that need a missing package are reported as skipped (with the reason), not as collection errors.", bar]
    return "\n".join(lines)


def cmd_preflight(argv: list[str]) -> int:
    report_dir = Path(argv[0])
    report_dir.mkdir(parents=True, exist_ok=True)
    rep = preflight_report()
    if rep["missing_required"] or rep["missing_optional"]:
        print(format_banner(rep), flush=True)
    else:
        print("[preflight] all dependencies present: " + ", ".join(f"{k} {v}" for k, v in rep["deps"].items()), flush=True)
    (report_dir / "preflight.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    return 0


def cmd_env(argv: list[str]) -> int:
    report_dir = Path(argv[0])
    report_dir.mkdir(parents=True, exist_ok=True)
    env: dict = {
        "generated": _now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap,power.limit,clocks.max.sm",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        env["nvidia_smi"] = out.stdout.strip().splitlines() if out.returncode == 0 else out.stderr.strip()
    except Exception as e:  # noqa: BLE001
        env["nvidia_smi"] = f"unavailable: {e}"
    for mod in ("torch", "triton", "transformers", "safetensors", "pydantic", "fastapi", "uvicorn", "zstandard", "numpy", "forge"):
        if mod == "forge":  # running from the source tree is fine, but say so instead of reporting a ModuleNotFoundError
            st = dep_status("forge", "forge-llm")
            env[mod] = st
            if st.startswith("not installed"):
                env["forge_source_version"] = _source_tree_version()
            continue
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            env[mod] = f"missing ({type(e).__name__})"
    pf = preflight_report()
    env["missing_dependencies"] = pf["missing_required"]
    if pf["missing_required"]:
        env["install_command"] = pf["install_command"]
    try:
        import torch  # type: ignore

        env["torch_cuda"] = torch.version.cuda
        env["cuda_available"] = bool(torch.cuda.is_available())
        # numerics-relevant global state: explains fp32 CUDA-vs-CPU tolerance failures (TF32 can be forced by env)
        env["fp32_matmul_precision"] = torch.get_float32_matmul_precision()
        env["allow_tf32_matmul"] = bool(torch.backends.cuda.matmul.allow_tf32)
        env["allow_tf32_cudnn"] = bool(torch.backends.cudnn.allow_tf32)
        env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "")
        env["NVIDIA_TF32_OVERRIDE"] = os.environ.get("NVIDIA_TF32_OVERRIDE", "")
        if torch.cuda.is_available():
            env["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "sm": "%d%d" % torch.cuda.get_device_capability(i),
                    "mem_gib": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception as e:  # noqa: BLE001
        env["torch_error"] = repr(e)
    (report_dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return 0


def cmd_bench_export(argv: list[str]) -> int:
    """Write bench_summary.json from the newest rows of the kernel_bench table (best effort, stdlib only)."""
    import os
    import sqlite3

    report_dir = Path(argv[0])
    since = float(argv[1]) if len(argv) > 1 else 0.0
    home = Path(os.environ.get("FORGE_HOME") or Path.home() / ".forge")
    db = home / "forge.db"
    if not db.exists():
        print(f"bench-export: no database at {db}")
        return 0
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cols = [r["name"] for r in con.execute("PRAGMA table_info(kernel_bench)")]
        if not cols:
            print("bench-export: table kernel_bench not found")
            return 0
        order = "created_at" if "created_at" in cols else "rowid"
        rows = [dict(r) for r in con.execute(f"SELECT * FROM kernel_bench ORDER BY {order} DESC LIMIT 200")]
        if since and "created_at" in cols:
            rows = [r for r in rows if (r.get("created_at") or 0) >= since]
        keep = ("kernel", "chip", "dtype", "shape_json", "layout_id", "path", "median_ms", "iqr_ms", "achieved_gbs",
                "pct_peak_bw", "bound", "correct", "max_abs_err", "speedup", "provenance")
        slim = [{k: r.get(k) for k in keep if k in r} for r in rows]
        (report_dir / "bench_summary.json").write_text(json.dumps({"rows": slim, "n": len(slim)}, indent=2), encoding="utf-8")
        print(f"bench-export: {len(slim)} rows")
    except Exception as e:  # noqa: BLE001
        print(f"bench-export failed: {e!r}")
    return 0


def _suite_label(name: str) -> str:
    """junit_cpu.xml -> 'CPU suite (tests/)'; junit_gpu.xml -> 'GPU suite (tests_gpu/)'."""
    stem = name[len("junit_"):-len(".xml")] if name.startswith("junit_") and name.endswith(".xml") else name
    return {"cpu": "CPU suite (tests/)", "gpu": "GPU suite (tests_gpu/)"}.get(stem, f"{stem} suite")


def _parse_junit(path: Path) -> dict:
    try:
        root = ET.parse(path).getroot()
    except Exception as e:  # noqa: BLE001
        return {"file": path.name, "suite": _suite_label(path.name), "error": repr(e)}
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    out = {"file": path.name, "suite": _suite_label(path.name), "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0, "failed": [], "skipped_reasons": {}}
    for s in suites:
        out["tests"] += int(s.get("tests", 0))
        out["failures"] += int(s.get("failures", 0))
        out["errors"] += int(s.get("errors", 0))
        out["skipped"] += int(s.get("skipped", 0))
        out["time"] += float(s.get("time", 0.0))
        for tc in s.iter("testcase"):
            ident = f"{tc.get('classname', '')}::{tc.get('name', '')}"
            bad = tc.find("failure")
            if bad is None:
                bad = tc.find("error")
            if bad is not None:
                msg = (bad.get("message") or bad.text or "").strip().splitlines()
                out["failed"].append({"test": ident, "message": (msg[0] if msg else "")[:300]})
            sk = tc.find("skipped")
            if sk is not None:
                reason = (sk.get("message") or "").strip()[:120]
                out["skipped_reasons"][reason] = out["skipped_reasons"].get(reason, 0) + 1
    out["time"] = round(out["time"], 2)
    out["passed"] = out["tests"] - out["failures"] - out["errors"] - out["skipped"]
    return out


def cmd_finalize(argv: list[str]) -> int:
    report_dir = Path(argv[0])
    env = _load(report_dir / "env.json", {})
    steps = _load(report_dir / "steps.json", [])
    junits = [_parse_junit(p) for p in sorted(report_dir.glob("junit_*.xml"))]
    bench = _load(report_dir / "bench_summary.json", None)
    failed_steps = [s["name"] for s in steps if s["exit_code"] != 0]
    n_fail = sum(j.get("failures", 0) + j.get("errors", 0) for j in junits)
    overall = "PASS" if not failed_steps and n_fail == 0 else "FAIL"
    report = {
        "generated": _now(),
        "overall": overall,
        "note": "Records what ran and its outcome on this host only. 'PASS' means no recorded failures; "
        "skipped tests are NOT passes.",
        "env": env,
        "steps": [{k: v for k, v in s.items() if k != "tail"} for s in steps],
        "tests": junits,
        "bench": bench,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md: list[str] = ["# Forge GPU report", "", f"Generated: {report['generated']}", f"Overall: **{overall}**", ""]
    md += ["## Environment", ""]
    for k in ("python", "platform", "torch", "torch_cuda", "triton", "transformers", "forge", "fastapi", "uvicorn", "zstandard",
              "cuda_available", "fp32_matmul_precision", "allow_tf32_matmul", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "NVIDIA_TF32_OVERRIDE"):
        if k in env:
            md.append(f"- {k}: `{env[k]}`")
    if env.get("missing_dependencies"):
        md.append(f"- MISSING DEPENDENCIES: {', '.join(env['missing_dependencies'])} -> `{env.get('install_command', '')}`")
    smi = env.get("nvidia_smi")
    if smi:
        md.append("- nvidia-smi (index, name, mem, driver, cc, power limit, max sm clock):")
        for line in smi if isinstance(smi, list) else [str(smi)]:
            md.append(f"  - `{line}`")
    md += ["", "## Steps", "", "| step | exit | seconds | log |", "|---|---|---|---|"]
    for s in steps:
        flag = "ok" if s["exit_code"] == 0 else ("TIMEOUT" if s.get("timed_out") else str(s["exit_code"]))
        md.append(f"| {s['name']} | {flag} | {s['seconds']} | {s['log']} |")
    if junits:
        md += ["", "## Test results", "", "| suite | junit file | passed | failed | errors | skipped | seconds |", "|---|---|---|---|---|---|---|"]
        for j in junits:
            if "error" in j:
                md.append(f"| {j['suite']} | {j['file']} | parse error: {j['error']} | | | | |")
                continue
            md.append(f"| {j['suite']} | {j['file']} | {j['passed']} | {j['failures']} | {j['errors']} | {j['skipped']} | {j['time']} |")
        for j in junits:
            for f in j.get("failed", [])[:50]:
                md.append(f"- FAILED [{j['suite']}] `{f['test']}`: {f['message']}")
            for reason, n in j.get("skipped_reasons", {}).items():
                md.append(f"- skipped x{n} [{j['suite']}]: {reason}")
    if bench:
        md += ["", "## Bench summary (this host, quick sweep; not a validated result)", "", "```json", json.dumps(bench, indent=2)[:6000], "```"]
    bad = [s for s in steps if s["exit_code"] != 0]
    if bad:
        md += ["", "## Failing step output (last lines)"]
        for s in bad:
            md += ["", f"### {s['name']} (exit {s['exit_code']})", "", "```", *s.get("tail", []), "```"]
    (report_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nReport written: {report_dir / 'report.md'}  (and report.json)")
    print(f"Overall: {overall}")
    return 0 if overall == "PASS" else 1


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("step", "env", "preflight", "finalize", "bench-export"):
        print(__doc__)
        return 64
    return {"step": cmd_step, "env": cmd_env, "preflight": cmd_preflight, "finalize": cmd_finalize,
            "bench-export": cmd_bench_export}[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
