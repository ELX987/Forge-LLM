#!/usr/bin/env bash
# Forge LLM - one-shot, idempotent setup for a rented Linux + NVIDIA host (vast.ai or similar).
#
#   bash scripts/vast_setup.sh                 # use the image's python/torch
#   FORGE_USE_VENV=1 bash scripts/vast_setup.sh   # create $FORGE_HOME/venv (--system-site-packages)
#
# Environment knobs (all optional):
#   FORGE_HOME           state directory (default: /workspace/forge-home if /workspace exists, else ~/.forge)
#   FORGE_USE_VENV       1 = install into a venv that can see the image's torch
#   FORGE_INSTALL_TORCH  1 = pip install torch if missing / CPU-only (uses FORGE_TORCH_INDEX)
#   FORGE_TORCH_INDEX    pip index URL for torch (default https://download.pytorch.org/whl/cu124)
#   FORGE_EXTRAS         pip extras (default: gpu,serve,parquet,dev)
#   FORGE_MIN_FREE_GIB   warn when FORGE_HOME has less free space (default 100)
#
# Safe to re-run: it only installs what is missing/out of date and rewrites ~/.forge_env.
set -euo pipefail

# shellcheck source=scripts/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

EXTRAS="${FORGE_EXTRAS:-gpu,serve,parquet,dev}"
MIN_FREE="${FORGE_MIN_FREE_GIB:-100}"
TORCH_INDEX="${FORGE_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

log "Forge setup in $FORGE_REPO_ROOT"

# ---- 1. NVIDIA driver -----------------------------------------------------------------------
if ! have nvidia-smi; then
    die "nvidia-smi not found. This must run on an NVIDIA host (choose a GPU instance, not CPU-only)."
fi
if ! nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi failed. Driver/GPU not usable in this container (check instance type and 'docker run --gpus')."
fi
log "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv,noheader || true

# ---- 2. Python ----------------------------------------------------------------------------------
BASE_PY=""
for cand in "${FORGE_PYTHON:-}" python3 python; do
    [ -n "$cand" ] || continue
    if have "$cand"; then
        BASE_PY="$cand"
        break
    fi
done
[ -n "$BASE_PY" ] || die "python3 not found"
if ! "$BASE_PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    die "Python >= 3.10 required, found $("$BASE_PY" -V 2>&1). Use a newer image (e.g. pytorch/pytorch:2.5+)."
fi
log "Python: $("$BASE_PY" -V 2>&1) at $(command -v "$BASE_PY")"

# ---- 3. FORGE_HOME on the big disk ----------------------------------------------------------------
FORGE_HOME="$(default_forge_home)"
export FORGE_HOME
mkdir -p "$FORGE_HOME"
FREE="$(free_gib "$FORGE_HOME")"
log "FORGE_HOME=$FORGE_HOME (free: ${FREE} GiB)"
if [ "$FREE" -lt "$MIN_FREE" ]; then
    warn "Only ${FREE} GiB free (< ${MIN_FREE}). 8B models need ~16 GiB each, plus checkpoints, datasets and the store."
    warn "Rent more disk or point FORGE_HOME at a larger volume, e.g. FORGE_HOME=/workspace/forge-home."
fi

# ---- 4. Optional venv --------------------------------------------------------------------------------
PY="$BASE_PY"
FORGE_VENV="${FORGE_VENV:-}"
if [ "${FORGE_USE_VENV:-0}" = "1" ]; then
    FORGE_VENV="${FORGE_VENV:-$FORGE_HOME/venv}"
    if [ ! -x "$FORGE_VENV/bin/python" ]; then
        log "Creating venv at $FORGE_VENV (inherits system site-packages so the image's torch is reused)"
        "$BASE_PY" -m venv --system-site-packages "$FORGE_VENV"
    fi
    PY="$FORGE_VENV/bin/python"
fi
"$PY" -m pip install --quiet --upgrade pip wheel >/dev/null

# ---- 5. torch with CUDA ------------------------------------------------------------------------------
torch_cuda_ok() {
    "$PY" - <<'EOF' >/dev/null 2>&1
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
EOF
}
if ! torch_cuda_ok; then
    if [ "${FORGE_INSTALL_TORCH:-0}" = "1" ]; then
        log "Installing torch from $TORCH_INDEX"
        "$PY" -m pip install torch --index-url "$TORCH_INDEX"
    fi
    if ! torch_cuda_ok; then
        err "torch is missing or cannot see CUDA. Either pick a PyTorch image, or re-run with FORGE_INSTALL_TORCH=1"
        err "(optionally FORGE_TORCH_INDEX=https://download.pytorch.org/whl/cu121|cu124|cu128; cu128 is needed for RTX 50xx / B200)."
        exit 2
    fi
fi

# ---- 6. Forge + triton -----------------------------------------------------------------------------
log "Installing forge[$EXTRAS] (editable)"
(cd "$FORGE_REPO_ROOT" && "$PY" -m pip install --quiet -e ".[$EXTRAS]")

if ! "$PY" -c 'import triton' >/dev/null 2>&1; then
    log "Installing triton (not shipped with this torch build)"
    "$PY" -m pip install --quiet "triton>=2.2"
fi

# ---- 7. Persist environment -------------------------------------------------------------------------
{
    echo "# written by scripts/vast_setup.sh - safe to delete, safe to edit"
    echo "export FORGE_HOME=\"$FORGE_HOME\""
    if [ -n "$FORGE_VENV" ]; then
        echo "export FORGE_VENV=\"$FORGE_VENV\""
    fi
    echo "export HF_HOME=\"\${HF_HOME:-$FORGE_HOME/hf_cache}\""
    echo "export TRITON_CACHE_DIR=\"\${TRITON_CACHE_DIR:-$FORGE_HOME/triton_cache}\""
    echo "export HF_HUB_ENABLE_HF_TRANSFER=\"\${HF_HUB_ENABLE_HF_TRANSFER:-0}\""
} >"$FORGE_ENV_FILE"
mkdir -p "$FORGE_HOME/hf_cache" "$FORGE_HOME/triton_cache"
BASHRC_LINE=". \"$FORGE_ENV_FILE\""
if [ -f "$HOME/.bashrc" ] && ! grep -Fqx "$BASHRC_LINE" "$HOME/.bashrc"; then
    echo "$BASHRC_LINE" >>"$HOME/.bashrc"
fi
log "Wrote $FORGE_ENV_FILE (sourced from ~/.bashrc)"

# ---- 8. Versions -------------------------------------------------------------------------------------
log "Versions:"
"$PY" - <<'EOF'
import importlib, platform, sys
print("  python      ", platform.python_version())
for name in ("torch", "triton", "transformers", "safetensors", "fastapi", "uvicorn", "pydantic", "numpy", "pynvml", "forge"):
    try:
        m = importlib.import_module(name)
        print(f"  {name:12s}", getattr(m, "__version__", "?"))
    except Exception as e:  # noqa: BLE001
        print(f"  {name:12s} MISSING ({type(e).__name__})")
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i}       {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB  torch CUDA {torch.version.cuda}")
EOF

log "Setup complete. Next:"
echo "  bash scripts/gpu_smoke.sh        # quick health check (a minute or two)"
echo "  bash scripts/run_gpu_tests.sh    # full GPU test suite + report in ./gpu_report"
echo "  bash scripts/run_server.sh       # start API + UI (prints the access token)"
