#!/usr/bin/env bash
# Forge LLM - CPU-only developer setup (Linux/macOS/Git-Bash). Creates ./.venv and installs CPU torch.
# Windows users: prefer scripts/dev_setup_cpu.ps1.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYBIN="${PYTHON:-python3}"
command -v "$PYBIN" >/dev/null 2>&1 || PYBIN=python
"$PYBIN" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
    echo "Python >= 3.10 required" >&2
    exit 1
}

if [ ! -d .venv ]; then
    "$PYBIN" -m venv .venv
fi
if [ -x .venv/bin/python ]; then
    VPY=.venv/bin/python
else
    VPY=.venv/Scripts/python.exe
fi

"$VPY" -m pip install --upgrade pip wheel
"$VPY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$VPY" -m pip install -e ".[dev]"

echo
echo "CPU dev environment ready. Run the tests with:"
echo "  $VPY -m pytest tests -q"
echo "GPU code is NOT exercised here (tests_gpu/ auto-skips without CUDA)."
