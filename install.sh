#!/usr/bin/env bash
set -e

# Install system deps
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 libasound2-dev ffmpeg bluez bluez-tools pulseaudio-utils pipewire-audio wireplumber

# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python deps
pip install -r requirements.txt

echo "✓ Installation klar. Skapa .env och starta med: uvicorn backend.app:app --host 0.0.0.0 --port 8080"
