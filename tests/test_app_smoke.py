"""End-to-end smoke tests: run the real application loop headlessly.

The app is started in a subprocess with the generated test pattern (no webcam) and
``--smoke-test``, which drives every ability through the same keyboard code paths a
user would use (charge, fire, ki blast, burst, aura, palette and quality changes,
settings panel, debug overlay). The process must exit with code 0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run_app(tmp_path: Path, extra_args: list[str], video_driver: str, seconds: float = 9.0,
             screenshots: Path | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT / "src"),
        "SDL_VIDEODRIVER": video_driver,
        "SDL_AUDIODRIVER": "dummy",
        "DBP_CONFIG_DIR": str(tmp_path / "config"),
        "DBP_LOGS_DIR": str(tmp_path / "logs"),
    })
    if screenshots is not None:
        env["DBP_SMOKE_SCREENSHOTS"] = str(screenshots)
    cmd = [sys.executable, "-m", "dragon_ball_powers", "--test-pattern", "--smoke-test", str(seconds),
           "--windowed", *extra_args]
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=240)


def _gl_available() -> bool:
    try:
        import moderngl

        backend = {"backend": "egl"} if sys.platform.startswith("linux") else {}
        moderngl.create_standalone_context(require=330, **backend).release()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.slow
def test_app_runs_with_opengl_renderer(tmp_path):
    if not _gl_available():
        pytest.skip("no OpenGL 3.3")
    driver = "offscreen" if sys.platform.startswith("linux") else "windows"
    shots = tmp_path / "shots"
    proc = _run_app(tmp_path, ["--renderer", "gl"], driver, screenshots=shots)
    log_text = (tmp_path / "logs" / "dragon_ball_powers.log").read_text(encoding="utf-8", errors="replace") \
        if (tmp_path / "logs" / "dragon_ball_powers.log").exists() else ""
    assert proc.returncode == 0, proc.stderr[-4000:] + "\n" + log_text[-4000:]
    assert "Smoke test finished" in log_text and "PASS" in log_text
    assert "Traceback" not in log_text, log_text[-4000:]
    # The scripted run saves a few frames of the test pattern with effects.
    assert any(shots.glob("*.png"))


@pytest.mark.slow
def test_app_runs_with_cpu_renderer(tmp_path):
    proc = _run_app(tmp_path, ["--renderer", "cpu"], "dummy", seconds=6.0)
    log_path = tmp_path / "logs" / "dragon_ball_powers.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    assert proc.returncode == 0, proc.stderr[-4000:] + "\n" + log_text[-4000:]
    assert "Traceback" not in log_text, log_text[-4000:]


@pytest.mark.slow
def test_check_command_runs(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), SDL_AUDIODRIVER="dummy",
               DBP_CONFIG_DIR=str(tmp_path / "config"), DBP_LOGS_DIR=str(tmp_path / "logs"))
    proc = subprocess.run([sys.executable, "-m", "dragon_ball_powers", "--check"], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
    assert "self-check" in proc.stdout
    assert "Result:" in proc.stdout
    # In CI-like environments there is no camera; that is a warning, not a failure.
    assert "[FAIL] Python" not in proc.stdout
    assert "[FAIL] MediaPipe" not in proc.stdout
