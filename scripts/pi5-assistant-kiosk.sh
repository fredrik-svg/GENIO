#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_HOST="${APP_HOST:-0.0.0.0}"
APP_PORT="${APP_PORT:-8080}"
UVICORN_CMD="$PROJECT_DIR/.venv/bin/uvicorn"
APP_MODULE="backend.app:app"
LOG_DIR="$PROJECT_DIR/logs"
BACKEND_STARTED=0
UVICORN_PID=""

mkdir -p "$LOG_DIR"

cleanup() {
    if [[ "$BACKEND_STARTED" -eq 1 && -n "$UVICORN_PID" ]]; then
        if kill -0 "$UVICORN_PID" >/dev/null 2>&1; then
            kill "$UVICORN_PID" >/dev/null 2>&1 || true
            wait "$UVICORN_PID" 2>/dev/null || true
        fi
    fi
}
trap cleanup EXIT

if [[ -x "$UVICORN_CMD" ]]; then
    if ! ss -ltn sport = :"$APP_PORT" >/dev/null 2>&1; then
        "$UVICORN_CMD" "$APP_MODULE" --host "$APP_HOST" --port "$APP_PORT" \
            >> "$LOG_DIR/kiosk-backend.log" 2>&1 &
        UVICORN_PID=$!
        BACKEND_STARTED=1
        # ge backend tid att starta upp
        for _ in {1..40}; do
            if ss -ltn sport = :"$APP_PORT" >/dev/null 2>&1; then
                break
            fi
            sleep 0.25
        done
        # sista korta väntan så att appen är redo
        sleep 1
    fi
fi

BROWSER_CMD=""
for candidate in chromium-browser chromium; do
    if command -v "$candidate" >/dev/null 2>&1; then
        BROWSER_CMD="$candidate"
        break
    fi
done

if [[ -z "$BROWSER_CMD" ]]; then
    zenity --error --text="Kunde inte hitta Chromium. Installera paketet 'chromium-browser'." 2>/dev/null || \
        notify-send "Pi5 Assistant" "Chromium hittades inte. Installera 'chromium-browser'." 2>/dev/null || \
        echo "[Pi5 Assistant] Chromium hittades inte. Installera 'chromium-browser'." >&2
    exit 1
fi

BROWSER_ARGS=(
    "--kiosk"
    "--start-fullscreen"
    "--disable-translate"
    "--noerrdialogs"
    "--no-first-run"
    "--disable-infobars"
    "--overscroll-history-navigation=0"
    "http://localhost:${APP_PORT}"
)

"$BROWSER_CMD" "${BROWSER_ARGS[@]}"
