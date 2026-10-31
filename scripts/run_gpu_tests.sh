#!/usr/bin/env bash
# Forge LLM - run the GPU correctness suite (tests_gpu/) and write a shareable report.
#
#   bash scripts/run_gpu_tests.sh                 # doctor + probe + all tests_gpu + quick bench
#   bash scripts/run_gpu_tests.sh -k w4a16        # extra args are passed to pytest
#   FORGE_SKIP_BENCH=1 bash scripts/run_gpu_tests.sh
#
# Output: ./gpu_report/{report.md,report.json,junit_*.xml,*.log} (override with REPORT_DIR).
# Exit code is non-zero if any step or test failed. Skipped tests are reported, never counted as passes.
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

"$PY" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' \
    || die "CUDA not available to torch - run scripts/vast_setup.sh first"

step forge_doctor --allow-fail --timeout 300 -- "$PY" -m forge.cli doctor
step forge_probe --allow-fail --timeout 900 -- "$PY" -m forge.cli probe --quick

# Also run the CPU suite on the GPU host: catches environment (version) problems early.
step cpu_tests --allow-fail --timeout 1800 -- "$PY" -m pytest tests -q -x --no-header -p no:cacheprovider \
    "--junitxml=$REPORT_DIR/junit_cpu.xml"

# GPU suite: do not stop at first failure, collect everything. -rs shows skip reasons.
step gpu_tests --allow-fail --timeout 5400 -- "$PY" -m pytest tests_gpu -v -rs --no-header -p no:cacheprovider \
    "--junitxml=$REPORT_DIR/junit_gpu.xml" "$@"

if [ "${FORGE_SKIP_BENCH:-0}" != "1" ]; then
    step kernel_bench_quick --allow-fail --timeout 1800 -- "$PY" -m forge.cli bench \
        --config configs/bench/bench-quick.yaml
    "$PY" "$HELPER" bench-export "$REPORT_DIR"
fi

set +e
"$PY" "$HELPER" finalize "$REPORT_DIR"
RC=$?
set -e
echo
echo "Paste $REPORT_DIR/report.md back if anything failed; attach $REPORT_DIR/*.log for details."
exit "$RC"
