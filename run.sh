#!/usr/bin/env bash
# Dragon Ball Powers - start the app on macOS / Linux:  bash run.sh [options]
# All options are passed to the app ("bash run.sh --help" lists them).
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd -P)"
VENV_PY="$ROOT/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "[ERROR] Dragon Ball Powers is not set up yet."
    echo "        Run \"bash setup.sh\" first (folder: $ROOT)."
    exit 1
fi
if ! "$VENV_PY" -c "import importlib.util, sys; missing = [n for n in ('mediapipe', 'cv2', 'pygame', 'moderngl', 'numpy') if importlib.util.find_spec(n) is None]; print('[ERROR] Python packages missing in .venv: ' + ', '.join(missing)) if missing else None; sys.exit(3 if missing else 0)"; then
    echo "        Run \"bash setup.sh\" again to repair the installation."
    exit 1
fi
unset PYTHONHOME
export PYTHONPATH="$ROOT/src"
exec "$VENV_PY" -m dragon_ball_powers "$@"
