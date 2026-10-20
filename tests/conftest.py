"""Shared pytest fixtures.

Tests never touch the real ``config/`` or ``logs/`` folders and never need a webcam.
pygame runs with dummy video/audio drivers unless a driver is already configured.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="dbp-tests-"))
os.environ.setdefault("DBP_CONFIG_DIR", str(_TMP_ROOT / "config"))
os.environ.setdefault("DBP_LOGS_DIR", str(_TMP_ROOT / "logs"))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pytest  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _create_gl_context():
    import moderngl

    errors = []
    backends = ["egl", None] if sys.platform.startswith("linux") else [None]
    for backend in backends:
        try:
            if backend:
                return moderngl.create_standalone_context(require=330, backend=backend)
            return moderngl.create_standalone_context(require=330)
        except Exception as exc:  # noqa: BLE001 - any failure means "no GL here"
            errors.append(f"{backend or 'default'}: {exc}")
    raise RuntimeError("; ".join(errors))


@pytest.fixture(scope="session")
def gl_context():
    """A headless OpenGL 3.3+ context, or skip the test when none is available."""
    try:
        ctx = _create_gl_context()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"No OpenGL 3.3 context available: {exc}")
    yield ctx
    ctx.release()


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    d = tmp_path / "config"
    d.mkdir()
    return d


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return PROJECT_ROOT / "assets" / "models"
