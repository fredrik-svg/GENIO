#!/usr/bin/env bash
set -e
WAV=${1:-/usr/share/sounds/alsa/Front_Center.wav}
echo "Trying to play: $WAV"
# Works for both ALSA (aplay) and Pulse/PipeWire (paplay) depending on PLAY_CMD
if command -v paplay >/dev/null 2>&1; then
  paplay "$WAV" || true
fi
if command -v aplay >/dev/null 2>&1; then
  aplay "$WAV" || true
fi
echo "If you didn't hear audio, set the default sink to your Bluetooth speaker (see bluetooth.md)"
