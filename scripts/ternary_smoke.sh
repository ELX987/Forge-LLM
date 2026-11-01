#!/usr/bin/env bash
# Forge LLM - ternary (1.58-bit) end-to-end smoke: ingest the bundled sample corpus -> ternary QAT job (tiny model) -> data-free ternarize + pack recipe ->
# a registered "Ternary 1.58b" artifact.  Works on CPU (default, about a minute) or a GPU host.  EXPERIMENTAL; it proves the pipeline runs, NOT that the
# resulting model is any good (the tiny model learns a 40-sentence corpus).
#
#   bash scripts/ternary_smoke.sh                 # CPU
#   GPUS=auto bash scripts/ternary_smoke.sh      # let the scheduler pick a GPU (the tiny model still trains fine; there is no ternary training speed-up)
#   FORGE_HOME=/tmp/forge-t bash scripts/ternary_smoke.sh
set -euo pipefail

# shellcheck source=scripts/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PY="$(pick_python)" || die "python not found"
cd "$FORGE_REPO_ROOT"
export FORGE_HOME="${FORGE_HOME:-$(mktemp -d)/forge_home}"
GPUS="${GPUS:-cpu}"
echo "FORGE_HOME=$FORGE_HOME  (GPUS=$GPUS)"

forge() { "$PY" -m forge.cli "$@"; }

forge job submit dataset_ingest -c configs/dataset/dataset-jsonl.yaml --gpus cpu --follow
forge ternary -c configs/ternary/tiny-smoke.yaml --gpus "$GPUS" --follow

MODEL_DIR="$(ls -d "$FORGE_HOME"/models/ternary-tiny-smoke-* | head -n1)"
[ -d "$MODEL_DIR" ] || die "the ternary job did not export a model under $FORGE_HOME/models"
RECIPE="$FORGE_HOME/ternary-smoke-recipe.yaml"
printf 'recipe: ternary-smoke\nstages:\n  - ternarize: {group_size: 32}\n  - pack: {}\n' > "$RECIPE"
forge compress --set model="$MODEL_DIR" --set recipe_path="$RECIPE" --set use_store=false --set save_dir="$FORGE_HOME/ternary_artifact" \
      --set dtype=float32 --set device=cpu --gpus cpu --follow

ART="$FORGE_HOME/ternary_artifact" "$PY" - <<'EOF'
import os, pathlib
from forge.core.artifactmeta import read_artifact_summary
s = read_artifact_summary(pathlib.Path(os.environ["ART"]))
assert s and s["is_quantized"] and s["label"] == "Ternary 1.58b", s
print("OK: artifact label", s["label"], "layers", s["n_quantized_layers"], "layout", s.get("layout_id"))
EOF
echo "ternary smoke finished (EXPERIMENTAL). Artifact: $FORGE_HOME/ternary_artifact"
