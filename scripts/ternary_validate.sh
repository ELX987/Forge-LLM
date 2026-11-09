#!/usr/bin/env bash
# Forge LLM - ternary (1.58-bit) hardware validation ladder, stages G0/G1 (run on a rented GPU host; nothing here has ever run on hardware).
#
#   bash scripts/ternary_validate.sh                       # G0 kernel checks + G1 smoke on the visible GPU
#   FORGE_TERNARY_NATIVE=1 bash scripts/ternary_validate.sh   # also build + self-test the optional native CUDA tier (needs nvcc)
#
# Output: ./gpu_report/ (override REPORT_DIR) with per-step logs, junit_ternary.xml, ternary_bench.json and report.md.  Paste report.md back when something fails.
# G2 (TinyLlama conversion SQNR) and G3 (125M-token run vs the simulator, bf16 baseline) need models and data: see docs/runbooks/ternary-validation.md.
# Set FORGE_TERNARY_MULTI_GPU=1 to include the P5 multi-GPU NCCL parity and collectives tests.
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
"$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' || die "CUDA not available to torch - run scripts/vast_setup.sh first"
if ! "$PY" "$HELPER" cuda-health "$REPORT_DIR"; then
    if [ "${FORGE_ALLOW_UNUSABLE_CUDA:-0}" != "1" ] && [ "${FORGE_SKIP_CUDA_CHECK:-0}" != "1" ]; then
        echo "Stopping: fix the CUDA problem above first." >&2
        "$PY" "$HELPER" finalize "$REPORT_DIR" || true
        exit 3
    fi
fi

# ---- G0: kernels vs the torch oracle on the test device, tern158 golden vectors, bit-exact quantiser -----------------------------------------
step ternary_gpu_tests --allow-fail --timeout 1800 -- "$PY" -m pytest tests_gpu/test_kernels_ternary.py -q -rs "--junitxml=$REPORT_DIR/junit_ternary.xml"
step ternary_cpu_ref_tests --allow-fail --timeout 900 -- "$PY" -m pytest tests/test_ternary_format.py tests/test_ternary_artifact.py tests/test_ternary_compress.py -q
# timing method: CUDA graphs over a weight ring larger than L2 (eager launch overhead hides small-M kernels); prints tern158's sm_120 numbers as targets
step ternary_bench --allow-fail --timeout 3600 -- "$PY" scripts/bench_ternary.py --out "$REPORT_DIR/ternary_bench.json"

if [ "${FORGE_TERNARY_NATIVE:-0}" = "1" ]; then
    have nvcc || die "FORGE_TERNARY_NATIVE=1 needs nvcc (CUDA 12.8+ for sm_100/120, 12.9+ for sm_103/121)"
    step ternary_native_selftest --allow-fail --timeout 3600 -- make -C forge/native/ternary selftest PYTHON="$PY"
fi

# ---- G1 (single GPU): ingest -> ternary QAT -> ternarize + pack -> labelled artifact -------------------------------------------------------------------
step ternary_smoke --allow-fail --timeout 1800 -- env GPUS=auto FORGE_PYTHON="$PY" bash scripts/ternary_smoke.sh

if [ "${FORGE_TERNARY_MULTI_GPU:-0}" = "1" ]; then
    step ternary_multi_gpu --allow-fail --timeout 1800 -- "$PY" -m pytest tests_gpu/test_zero_ternary.py -v -rs "--junitxml=$REPORT_DIR/junit_zero_ternary.xml"
fi

"$PY" "$HELPER" finalize "$REPORT_DIR"
echo "Done. Fill the results table in docs/runbooks/ternary-validation.md from $REPORT_DIR/report.md and $REPORT_DIR/ternary_bench.json."
