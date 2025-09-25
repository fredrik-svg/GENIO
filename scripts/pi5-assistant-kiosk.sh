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
declare -a BROWSER_PIDS=()

mkdir -p "$LOG_DIR"

cleanup() {
    for pid in "${BROWSER_PIDS[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
            wait "$pid" 2>/dev/null || true
        fi
    done

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

PRIMARY_URL="${PRIMARY_URL:-http://localhost:${APP_PORT}}"
SECONDARY_URL="${SECONDARY_URL:-http://localhost:${APP_PORT}/display}"
PRIMARY_DISPLAY_TARGET="${PRIMARY_DISPLAY_TARGET:-${DISPLAY:-:0}}"
SECONDARY_DISPLAY_TARGET="${SECONDARY_DISPLAY_TARGET:-$PRIMARY_DISPLAY_TARGET}"
SECONDARY_DISPLAY_MODE="${SECONDARY_DISPLAY_MODE:-auto}"
CHROMIUM_PROFILE_BASE="${CHROMIUM_PROFILE_BASE:-$PROJECT_DIR/.chromium-profiles}"

mkdir -p "$CHROMIUM_PROFILE_BASE"

PRIMARY_BROWSER_EXTRA_ARGS_RAW="${PRIMARY_BROWSER_EXTRA_ARGS:-}"
SECONDARY_BROWSER_EXTRA_ARGS_RAW="${SECONDARY_BROWSER_EXTRA_ARGS:-}"

if [[ -n "$PRIMARY_BROWSER_EXTRA_ARGS_RAW" ]]; then
    read -r -a PRIMARY_BROWSER_EXTRA_ARGS <<<"$PRIMARY_BROWSER_EXTRA_ARGS_RAW"
else
    PRIMARY_BROWSER_EXTRA_ARGS=()
fi

if [[ -n "$SECONDARY_BROWSER_EXTRA_ARGS_RAW" ]]; then
    read -r -a SECONDARY_BROWSER_EXTRA_ARGS <<<"$SECONDARY_BROWSER_EXTRA_ARGS_RAW"
else
    SECONDARY_BROWSER_EXTRA_ARGS=()
fi

BROWSER_COMMON_ARGS=(
    "--kiosk"
    "--start-fullscreen"
    "--disable-translate"
    "--noerrdialogs"
    "--no-first-run"
    "--disable-infobars"
    "--overscroll-history-navigation=0"
)

should_launch_secondary=0
case "${SECONDARY_DISPLAY_MODE,,}" in
    always)
        should_launch_secondary=1
        ;;
    never)
        should_launch_secondary=0
        ;;
    auto)
        if [[ -n "$SECONDARY_URL" ]]; then
            if command -v xrandr >/dev/null 2>&1; then
                monitor_output=$(xrandr --listmonitors 2>/dev/null || true)
                monitor_count=$(awk 'NR==1 {print $2}' <<<"$monitor_output")
                if [[ "$monitor_count" =~ ^[0-9]+$ ]] && (( monitor_count >= 2 )); then
                    should_launch_secondary=1
                fi
            fi
        fi
        ;;
    1|true|yes)
        should_launch_secondary=1
        ;;
    0|false|no)
        should_launch_secondary=0
        ;;
    *)
        should_launch_secondary=0
        ;;
esac

launch_browser_instance() {
    local target_display="$1"
    local profile_name="$2"
    local url="$3"
    shift 3
    local -a extra_args=("$@")

    local user_data_dir="$CHROMIUM_PROFILE_BASE/$profile_name"
    mkdir -p "$user_data_dir"

    (
        if [[ -n "$target_display" ]]; then
            export DISPLAY="$target_display"
        fi
        exec "$BROWSER_CMD" "${BROWSER_COMMON_ARGS[@]}" "--user-data-dir=$user_data_dir" "${extra_args[@]}" "$url"
    ) &

    local pid=$!
    BROWSER_PIDS+=("$pid")
}

launch_browser_instance "$PRIMARY_DISPLAY_TARGET" "assistant" "$PRIMARY_URL" "${PRIMARY_BROWSER_EXTRA_ARGS[@]}"

if [[ $should_launch_secondary -eq 1 ]]; then
    sleep 1
    launch_browser_instance "$SECONDARY_DISPLAY_TARGET" "display" "$SECONDARY_URL" "${SECONDARY_BROWSER_EXTRA_ARGS[@]}"
fi

for pid in "${BROWSER_PIDS[@]}"; do
    wait "$pid"
done
