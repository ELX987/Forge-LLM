#!/usr/bin/env python3
"""Type-check the .cu sources with g++ -fsyntax-only against tools/cuda_stub (no CUDA toolkit needed).
Strips <<<...>>> launch syntax first.  Catches typos, missing declarations, template and signature errors.
It does NOT validate device semantics, PTX, or performance - run tern158_selftest on a GPU for that."""
import re, subprocess, sys, tempfile, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
stub = root / "tools" / "cuda_stub"
inc = root / "kernel" / "include"
srcs = sorted((root / "kernel" / "src").glob("*.cu")) + [root / "kernel" / "tests" / "selftest.cu"]
rc = 0
for arch in ("0", "800"):
    for s in srcs:
        text = s.read_text(encoding="utf-8")
        text = re.sub(r"<<<[^>]*?>>>", "", text)
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / (s.stem + ".cpp")
            p.write_text(text, encoding="utf-8")
            cmd = ["g++", "-std=c++17", "-fsyntax-only", "-Wall", "-Wno-unused-function", "-Wno-unused-variable", "-Wno-unused-but-set-variable",
                   "-Wno-unknown-pragmas", "-Wno-attributes", f"-I{stub}", f"-I{inc}"]
            if arch != "0":
                cmd.append(f"-D__CUDA_ARCH__={arch}")
            cmd.append(str(p))
            r = subprocess.run(cmd, capture_output=True, text=True)
            out = (r.stdout + r.stderr).replace(str(p), s.name)
            if r.returncode != 0:
                rc = 1
                print(f"=== {s.name} (arch {arch}) FAILED")
                lines = [ln for ln in out.splitlines() if "error" in ln or "warning" in ln]
                print("\n".join(lines[:25]))
            else:
                print(f"ok  {s.name} (arch {arch})" + (f"\n{out}" if out.strip() else ""))
sys.exit(rc)
