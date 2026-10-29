#!/bin/bash
# Build dist/VanishCam.app. Usage: packaging/build_app.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3.12}"
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt "pyinstaller>=6,<7"

# Models are downloaded once and bundled, so the app works offline.
.venv/bin/python -c "import vanish_cam; vanish_cam.download_models()"
[ -f packaging/VanishCam.icns ] || .venv/bin/python packaging/make_icon.py

.venv/bin/python -m PyInstaller --noconfirm --clean \
  --distpath dist --workpath build packaging/VanishCam.spec
echo "Built dist/VanishCam.app"
