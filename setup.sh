#!/usr/bin/env bash
# ===========================================================================
#  Dragon Ball Powers - one-time setup for macOS (11 or newer) and Linux
#
#    [1/6] find Python 3.11 or 3.12 (64-bit) - or download a private copy into
#          .python (python-build-standalone, verified by SHA256)
#    [2/6] create the virtual environment .venv
#    [3/6] update pip
#    [4/6] pip install -r requirements.txt
#    [5/6] download + verify the MediaPipe models (scripts/download_models.py)
#    [6/6] self-check: python -m dragon_ball_powers --check
#
#  Run it from a terminal:  bash setup.sh   ("bash setup.sh --help" lists the
#  options). Running it again is safe: a healthy .venv is reused. Only this
#  folder is changed (pip keeps its usual download cache). Written for the
#  bash 3.2 that ships with macOS: no bash 4 features.
# ===========================================================================
set -u

cd "$(dirname "$0")" || exit 1
ROOT="$(pwd -P)"
VENV_DIR="$ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
LOCAL_PY_DIR="$ROOT/.python"
LOCAL_PY="$LOCAL_PY_DIR/bin/python3"
MODELS_SCRIPT="$ROOT/scripts/download_models.py"

# Private Python fallback: python-build-standalone (the CPython build used by uv),
# the same release as in setup.bat. "%2B" is the URL-encoded "+" of the release name.
PBS_TAG="20261013"
PBS_VERSION="3.12.15+20261013"
PBS_BASE="https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG"

pbs_sha256() {   # SHA256 of cpython-<PBS_VERSION>-<target>-install_only.tar.gz (release SHA256SUMS)
    case "$1" in
        aarch64-apple-darwin)      echo "ba32036ad8f4c434d97a1ee7feef8b7edc0fac16777639b97e39c74a108e4440" ;;
        x86_64-apple-darwin)       echo "9686ddd8639b855905447ea4a16eb54a804a9ef12afbe7c9ecb1412d9c1fbc98" ;;
        x86_64-unknown-linux-gnu)  echo "5d546fc004fc2435fa142eb8f42ed8de5eb4725963dc22f7e6671014a80eb560" ;;
        aarch64-unknown-linux-gnu) echo "34a43010cc05982d8b5debbad7d09f7b1200117ebf018f7acfd656895e408205" ;;
        *) return 1 ;;
    esac
}

# Prints "Python X.Y.Z" and exits 0 when the interpreter can build our .venv:
# 3.11 or 3.12, 64-bit, with venv + ensurepip (Debian/Ubuntu split them off).
PROBE_CODE="import struct, sys, importlib.util as u; v = sys.version_info; print('Python %d.%d.%d (%d-bit)' % (v[0], v[1], v[2], struct.calcsize('P') * 8)); sys.exit(0 if v[:2] in ((3, 11), (3, 12)) and struct.calcsize('P') == 8 and u.find_spec('venv') and u.find_spec('ensurepip') else 3)"

usage() {
    cat <<'EOF'
Dragon Ball Powers - setup for macOS and Linux

Usage: bash setup.sh [--dry-run] [--force] [--dev] [--help]

Prepares everything the app needs, inside this folder only:
  [1/6] find Python 3.11 or 3.12 (64-bit): an existing .venv, python3.12,
        python3.11, python3 or .python. If none is found, a private Python
        3.12 (python-build-standalone) is downloaded, verified (SHA256) and
        unpacked into .python
  [2/6] create the virtual environment .venv
  [3/6] update pip
  [4/6] install the pinned packages from requirements.txt (about 400 MB)
  [5/6] download and verify the MediaPipe models (53 MB)
  [6/6] run the self-check

Options:
  --dry-run   only show what would happen
  --force     re-create .venv from scratch
  --dev       also install the developer tools (requirements-dev.txt)
  --help      show this help

Start the app afterwards with:  bash run.sh
Exit codes: 0 = ready, 1 = a step failed, 2 = invalid option
EOF
}

DRY_RUN=0
FORCE=0
DEV=0
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --dev) DEV=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[ERROR] Unknown option \"$arg\". Run \"bash setup.sh --help\" to see the options."; exit 2 ;;
    esac
done
REQ_FILE="$ROOT/requirements.txt"
[ "$DEV" = 1 ] && REQ_FILE="$ROOT/requirements-dev.txt"

# Environment hygiene: a PYTHONHOME/PYTHONPATH from another tool breaks venv and pip;
# pip must never wait for input; a global "user = true" would break installs into .venv.
unset PYTHONHOME PYTHONPATH
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_USER=0

fail() { echo "[ERROR] $*"; exit 1; }
run() {   # run a command, or only print it with --dry-run
    if [ "$DRY_RUN" = 1 ]; then echo "      [dry-run] $*"; return 0; fi
    "$@"
}

probe() { "$1" -c "$PROBE_CODE" 2>/dev/null; }

detect_target() {   # python-build-standalone target triple of this machine
    case "$(uname -s)" in
        Darwin)
            # Apple Silicon even when this shell runs under Rosetta (uname -m = x86_64).
            if [ "$(sysctl -n hw.optional.arm64 2>/dev/null)" = 1 ]; then echo aarch64-apple-darwin; return 0; fi
            case "$(uname -m)" in x86_64) echo x86_64-apple-darwin ;; *) return 1 ;; esac ;;
        Linux)
            case "$(uname -m)" in
                x86_64|amd64) echo x86_64-unknown-linux-gnu ;;
                aarch64|arm64) echo aarch64-unknown-linux-gnu ;;
                *) return 1 ;;
            esac ;;
        *) return 1 ;;
    esac
}

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
    else sha256sum "$1" | awk '{print $1}'; fi
}

download_python() {   # private Python into .python; sets BASE_PY
    target="$(detect_target)" || fail "No private Python for $(uname -s) $(uname -m). Install Python 3.12 (64-bit) and run setup.sh again."
    sha="$(pbs_sha256 "$target")"
    url="$PBS_BASE/cpython-3.12.15%2B$PBS_TAG-$target-install_only.tar.gz"
    archive="${TMPDIR:-/tmp}/dbp-cpython-$PBS_VERSION-$target.tar.gz"
    echo "      No compatible Python found. Downloading a private Python $PBS_VERSION"
    echo "      ($target, about 30 MB) into .python - nothing is installed system-wide."
    if [ "$DRY_RUN" = 1 ]; then
        echo "      [dry-run] download $url"
        echo "      [dry-run] check its SHA256 = $sha and unpack it into .python"
        BASE_PY="$LOCAL_PY"
        return 0
    fi
    command -v curl >/dev/null 2>&1 || fail "curl is missing. Install curl (Linux: sudo apt install curl) and run setup.sh again."
    if [ ! -f "$archive" ] || [ "$(sha256_of "$archive")" != "$sha" ]; then
        rm -f "$archive" "$archive.part"
        curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 30 --progress-bar \
            --output "$archive.part" "$url" || fail "The download failed. Check the internet connection (behind a proxy: export HTTPS_PROXY=http://proxy:port) and run setup.sh again."
        mv -f "$archive.part" "$archive"
    else
        echo "      Using the archive downloaded earlier."
    fi
    [ "$(sha256_of "$archive")" = "$sha" ] || { rm -f "$archive"; fail "The downloaded file is damaged or not the expected release (SHA256 mismatch). Run setup.sh again."; }
    staging="$ROOT/.python.new"
    rm -rf "$staging"
    mkdir -p "$staging" && tar -xzf "$archive" -C "$staging" --strip-components=1 || { rm -rf "$staging"; fail "Could not unpack the Python archive."; }
    rm -rf "$LOCAL_PY_DIR"
    mv "$staging" "$LOCAL_PY_DIR" || fail "Could not move the private Python into .python."
    rm -f "$archive"
    BASE_PY="$LOCAL_PY"
    probe "$BASE_PY" >/dev/null || fail "The downloaded Python does not run on this machine."
}

echo "==========================================================================="
echo " Dragon Ball Powers - setup ($(uname -s) $(uname -m))"
echo "==========================================================================="
[ -f "$ROOT/requirements.txt" ] && [ -f "$MODELS_SCRIPT" ] || fail "This folder is incomplete: requirements.txt or scripts/download_models.py is missing."

# --------------------------------------------------------------------- [1/6]
echo "[1/6] Looking for Python 3.11 or 3.12 (64-bit) ..."
BASE_PY=""
REUSE_VENV=0
if [ "$FORCE" = 0 ] && [ -x "$VENV_PY" ] && probe "$VENV_PY" >/dev/null && "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    REUSE_VENV=1
    echo "      Found the existing .venv: $(probe "$VENV_PY")"
else
    for candidate in python3.12 python3.11 python3 "$LOCAL_PY"; do
        path="$(command -v "$candidate" 2>/dev/null)" || continue
        if info="$(probe "$path")"; then
            BASE_PY="$path"
            echo "      Using $path: $info"
            break
        fi
    done
    [ -n "$BASE_PY" ] || download_python
fi

# --------------------------------------------------------------------- [2/6]
echo "[2/6] Virtual environment .venv ..."
if [ "$REUSE_VENV" = 1 ]; then
    echo "      .venv is healthy - reused."
else
    if [ -e "$VENV_DIR" ]; then
        echo "      Re-creating .venv ..."
        run rm -rf "$VENV_DIR"
    fi
    run "$BASE_PY" -m venv "$VENV_DIR" || fail "Could not create .venv."
fi

# --------------------------------------------------------------------- [3/6]
echo "[3/6] Updating pip ..."
run "$VENV_PY" -m pip install --upgrade --quiet pip || fail "pip could not be updated."

# --------------------------------------------------------------------- [4/6]
echo "[4/6] Installing the Python packages from $(basename "$REQ_FILE") (this takes a few minutes) ..."
run "$VENV_PY" -m pip install -r "$REQ_FILE" || fail "Installing the packages failed. Check the internet connection and run setup.sh again."

# --------------------------------------------------------------------- [5/6]
echo "[5/6] Downloading and verifying the MediaPipe models ..."
run "$VENV_PY" "$MODELS_SCRIPT" || fail "The models could not be downloaded. Run setup.sh again."

# --------------------------------------------------------------------- [6/6]
echo "[6/6] Self-check ..."
if [ "$DRY_RUN" = 1 ]; then
    echo "      [dry-run] PYTHONPATH=src $VENV_PY -m dragon_ball_powers --check"
    echo
    echo "[OK] Dry run finished - nothing was changed."
    exit 0
fi
PYTHONPATH="$ROOT/src" "$VENV_PY" -m dragon_ball_powers --check || fail "The self-check found a problem (see above)."
echo
echo "[OK] Setup finished. Start the app with:  bash run.sh"
if [ "$(uname -s)" = Darwin ]; then
    echo "     macOS asks once whether Terminal may use the camera: allow it (System Settings >"
    echo "     Privacy & Security > Camera), then press START CAMERA in the app or restart it."
fi
exit 0
