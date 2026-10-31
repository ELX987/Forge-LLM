#!/usr/bin/env bash
# Forge LLM - start the API + web UI.
#
#   bash scripts/run_server.sh                 # foreground, 0.0.0.0:8765, token auto-generated and printed
#   bash scripts/run_server.sh --tmux          # inside tmux session "forge" (survives SSH disconnect)
#   bash scripts/run_server.sh --nohup         # background with nohup, log in $FORGE_HOME/server.log
#   bash scripts/run_server.sh --stop          # stop the nohup/tmux server
#   bash scripts/run_server.sh --port 9000 --host 127.0.0.1
#
# Token: $FORGE_API_TOKEN if set, else the one saved in $FORGE_HOME/api_token, else a new random one (saved, mode 600).
# SECURITY: binding 0.0.0.0 on a rented host exposes the port to the internet. Prefer --host 127.0.0.1 and an
# SSH tunnel (see docs/VAST_AI.md). The token is the only protection and there is no TLS.
set -euo pipefail

# shellcheck source=scripts/_common.sh
. "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

HOST="${FORGE_API_HOST:-0.0.0.0}"
PORT="${FORGE_API_PORT:-8765}"
MODE="foreground"

while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --tmux) MODE="tmux"; shift ;;
        --nohup) MODE="nohup"; shift ;;
        --stop) MODE="stop"; shift ;;
        --no-token) NO_TOKEN=1; shift ;;
        -h|--help) sed -n '2,13p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) die "unknown argument: $1" ;;
    esac
done

PY="$(pick_python)" || die "python not found"
FORGE_HOME="$(default_forge_home)"
export FORGE_HOME
mkdir -p "$FORGE_HOME"
PIDFILE="$FORGE_HOME/server.pid"
LOGFILE="$FORGE_HOME/server.log"
SESSION="forge"

if [ "$MODE" = "stop" ]; then
    if have tmux && tmux has-session -t "$SESSION" 2>/dev/null; then
        tmux kill-session -t "$SESSION"
        log "tmux session '$SESSION' stopped"
    fi
    if [ -f "$PIDFILE" ]; then
        PID="$(cat "$PIDFILE")"
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID"
            log "stopped server pid $PID"
        fi
        rm -f "$PIDFILE"
    fi
    exit 0
fi

# ---- token ---------------------------------------------------------------------------------------------
TOKEN_FILE="$FORGE_HOME/api_token"
if [ "${NO_TOKEN:-0}" = "1" ]; then
    FORGE_API_TOKEN=""
    warn "running WITHOUT authentication"
elif [ -z "${FORGE_API_TOKEN:-}" ]; then
    if [ -s "$TOKEN_FILE" ]; then
        FORGE_API_TOKEN="$(cat "$TOKEN_FILE")"
    else
        FORGE_API_TOKEN="$(gen_token)"
        (umask 077 && printf '%s' "$FORGE_API_TOKEN" >"$TOKEN_FILE")
    fi
fi
export FORGE_API_TOKEN FORGE_API_HOST="$HOST" FORGE_API_PORT="$PORT"

if [ "$HOST" = "0.0.0.0" ] && [ -n "${FORGE_API_TOKEN:-}" ]; then
    warn "listening on 0.0.0.0 - reachable from outside if the port is published. Prefer an SSH tunnel."
fi

CMD=("$PY" -m forge.cli serve-ui --host "$HOST" --port "$PORT")

banner() {
    echo
    echo "  Forge LLM server"
    echo "  ----------------"
    echo "  listen     : http://$HOST:$PORT"
    echo "  FORGE_HOME : $FORGE_HOME"
    if [ -n "${FORGE_API_TOKEN:-}" ]; then
        echo "  API token  : $FORGE_API_TOKEN"
        echo "               (paste into UI Settings -> API token; saved in $TOKEN_FILE)"
    fi
    echo "  tunnel     : ssh -N -L $PORT:127.0.0.1:$PORT -p <ssh_port> root@<host>   then open http://localhost:$PORT"
    echo
}

case "$MODE" in
    foreground)
        banner
        cd "$FORGE_REPO_ROOT"
        exec "${CMD[@]}"
        ;;
    nohup)
        banner
        cd "$FORGE_REPO_ROOT"
        nohup "${CMD[@]}" >>"$LOGFILE" 2>&1 &
        echo $! >"$PIDFILE"
        log "started in background, pid $(cat "$PIDFILE"), log: $LOGFILE   (stop: bash scripts/run_server.sh --stop)"
        ;;
    tmux)
        have tmux || die "tmux not installed (apt-get install -y tmux), or use --nohup"
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            die "tmux session '$SESSION' already exists (attach: tmux attach -t $SESSION; stop: $0 --stop)"
        fi
        banner
        tmux new-session -d -s "$SESSION" -c "$FORGE_REPO_ROOT" \
            "FORGE_HOME='$FORGE_HOME' FORGE_API_TOKEN='${FORGE_API_TOKEN:-}' FORGE_API_HOST='$HOST' FORGE_API_PORT='$PORT' ${CMD[*]} 2>&1 | tee -a '$LOGFILE'"
        log "started in tmux session '$SESSION' (attach: tmux attach -t $SESSION, detach: Ctrl-b d)"
        ;;
    *) die "internal error: mode $MODE" ;;
esac
