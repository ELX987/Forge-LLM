#!/usr/bin/env bash
# Shared helpers for the Forge scripts. Source this file; do not execute it.
# shellcheck shell=bash

# Repo root = parent of this scripts/ directory.
FORGE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FORGE_REPO_ROOT

# Persisted environment written by vast_setup.sh (FORGE_HOME, etc.).
FORGE_ENV_FILE="${FORGE_ENV_FILE:-$HOME/.forge_env}"
if [ -f "$FORGE_ENV_FILE" ]; then
    # shellcheck disable=SC1090
    . "$FORGE_ENV_FILE"
fi

log()  { printf '\033[1;34m[forge]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[forge:warn]\033[0m %s\n' "$*" >&2; }
err()  { printf '\033[1;31m[forge:error]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

# Pick the python interpreter: $FORGE_PYTHON, then the venv at $FORGE_VENV, then python3, then python.
pick_python() {
    if [ -n "${FORGE_PYTHON:-}" ]; then
        printf '%s' "$FORGE_PYTHON"
        return 0
    fi
    if [ -n "${FORGE_VENV:-}" ] && [ -x "$FORGE_VENV/bin/python" ]; then
        printf '%s' "$FORGE_VENV/bin/python"
        return 0
    fi
    if have python3; then
        printf '%s' "python3"
        return 0
    fi
    if have python; then
        printf '%s' "python"
        return 0
    fi
    return 1
}

# Default FORGE_HOME: a big persistent disk if present (vast.ai mounts /workspace), else ~/.forge.
default_forge_home() {
    if [ -n "${FORGE_HOME:-}" ]; then
        printf '%s' "$FORGE_HOME"
    elif [ -d /workspace ] && [ -w /workspace ]; then
        printf '%s' "/workspace/forge-home"
    else
        printf '%s' "$HOME/.forge"
    fi
}

# Print free GiB on the filesystem holding $1 (creating nothing).
free_gib() {
    local target="$1"
    while [ ! -d "$target" ] && [ "$target" != "/" ]; do
        target="$(dirname "$target")"
    done
    df -Pk "$target" | awk 'NR==2 {printf "%d", $4/1024/1024}'
}

# Generate a URL-safe random token.
gen_token() {
    local py
    if py="$(pick_python)"; then
        "$py" -c 'import secrets; print(secrets.token_urlsafe(32))'
    elif have openssl; then
        openssl rand -base64 32 | tr '+/' '-_' | tr -d '=\n'
    else
        head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n'
    fi
}
