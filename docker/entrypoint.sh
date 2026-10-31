#!/usr/bin/env bash
# Container entrypoint: `serve-ui` (default) starts API+UI; anything else is passed to the forge CLI
# or executed directly (e.g. `bash`, `pytest`).
set -euo pipefail

mkdir -p "${FORGE_HOME:-/data}"

if [ "${1:-serve-ui}" = "serve-ui" ]; then
    if [ -z "${FORGE_API_TOKEN:-}" ]; then
        FORGE_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
        export FORGE_API_TOKEN
        echo "[forge] generated API token: ${FORGE_API_TOKEN}"
    fi
    shift || true
    exec python -m forge.cli serve-ui --host "${FORGE_API_HOST:-0.0.0.0}" --port "${FORGE_API_PORT:-8765}" "$@"
fi

if command -v "$1" >/dev/null 2>&1; then
    exec "$@"
fi
exec python -m forge.cli "$@"
