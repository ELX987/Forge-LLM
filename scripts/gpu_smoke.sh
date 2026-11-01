#!/usr/bin/env bash
# Forge LLM - quick GPU health check (target: a few minutes).
#
# Runs: raw torch/CUDA sanity, a raw Triton sanity kernel, `forge doctor`, `forge probe --quick`, and a quick
# kernel benchmark sweep. Everything is logged under ./gpu_report/ (override with REPORT_DIR) and summarised in
# gpu_report/report.md + report.json - paste report.md back when something fails.
#
# This does NOT replace scripts/run_gpu_tests.sh (the correctness suite). Nothing here is a validated result.
set -euo pipefail

# shellcheck source=scripts/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

REPORT_DIR="${REPORT_DIR:-$PWD/gpu_report}"
PY="$(pick_python)" || die "python not found"
HELPER="$FORGE_REPO_ROOT/scripts/gpu_report.py"
mkdir -p "$REPORT_DIR"
cd "$FORGE_REPO_ROOT"

step() { "$PY" "$HELPER" step "$REPORT_DIR" "$@"; }

"$PY" "$HELPER" env "$REPORT_DIR"

# CUDA stack check (driver <-> torch build <-> GPU arch <-> triton + allocation probe): stop with its report when it fails,
# instead of letting every later step fail with an opaque CUDA error. FORGE_SKIP_CUDA_CHECK=1 runs anyway.
CUDA_RC=0
PYTHONPATH="$FORGE_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m forge.hw.cuda_compat >"$REPORT_DIR/cuda_compat.txt" 2>&1 || CUDA_RC=$?
cat "$REPORT_DIR/cuda_compat.txt"
if [ "$CUDA_RC" -ne 0 ]; then
    if [ "${FORGE_SKIP_CUDA_CHECK:-0}" != "1" ]; then
        err "CUDA stack check failed (report above, saved to $REPORT_DIR/cuda_compat.txt). Set FORGE_SKIP_CUDA_CHECK=1 to run anyway."
        "$PY" "$HELPER" finalize "$REPORT_DIR" || true
        exit 3
    fi
fi

step torch_cuda --allow-fail --timeout 300 -- "$PY" - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
x = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
y = (x @ x).float().abs().mean().item()
torch.cuda.synchronize()
print("fp16 matmul ok, mean|y| =", y)
print("bf16 supported:", torch.cuda.is_bf16_supported())
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
EOF

step triton_sanity --allow-fail --timeout 300 -- "$PY" scripts/triton_sanity.py

step forge_doctor --allow-fail --timeout 300 -- "$PY" -m forge.cli doctor
step forge_probe --allow-fail --timeout 900 -- "$PY" -m forge.cli probe --quick
step kernel_bench_quick --allow-fail --timeout 1800 -- "$PY" -m forge.cli bench --config configs/bench/bench-quick.yaml

"$PY" "$HELPER" bench-export "$REPORT_DIR"
"$PY" "$HELPER" finalize "$REPORT_DIR"
