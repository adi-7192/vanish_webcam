#!/bin/bash
# Double-click to start VanishCam in the macOS menu bar.
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"
PYTHON=""
for candidate in python3.12 python3.11 python3.10 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
  if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
done

if [ ! -x "$VENV/bin/python" ]; then
  if [ -z "$PYTHON" ]; then
    echo "[VanishCam] Python 3.10-3.12 is required (MediaPipe does not support newer versions yet)."
    echo "            Install it with:  brew install python@3.12"
    read -r -p "Press Enter to close..."; exit 1
  fi
  echo "[VanishCam] Creating the Python environment with $PYTHON..."
  "$PYTHON" -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import cv2, mediapipe, numpy, pyvirtualcam, rumps" >/dev/null 2>&1; then
  echo "[VanishCam] Installing components (internet needed only this time)..."
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -r requirements.txt
fi

if [ ! -d "/Library/SystemExtensions" ] || ! systemextensionsctl list 2>/dev/null | grep -q "com.obsproject.obs-studio.mac-camera-extension"; then
  echo "[VanishCam] OBS Virtual Camera not found. One-time setup:"
  echo "            1. Install OBS Studio 30+ (https://obsproject.com or: brew install --cask obs)"
  echo "            2. Open OBS, click 'Start Virtual Camera', allow the extension in System Settings"
  echo "            3. Quit OBS, then start VanishCam again."
fi

echo "[VanishCam] Running in the menu bar (◌ icon). Keep this window open; close it to quit."
exec "$VENV/bin/python" menubar.py
