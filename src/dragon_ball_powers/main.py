"""Command line entry point: ``python -m dragon_ball_powers [options]``."""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import platform
import struct
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# Must be set before pygame / OpenCV are imported anywhere.
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

from dragon_ball_powers import APP_NAME, __version__  # noqa: E402
from dragon_ball_powers import config  # noqa: E402

log = logging.getLogger("dragon_ball_powers")

LOG_FILE_NAME = "dragon_ball_powers.log"


@dataclass
class AppArgs:
    camera: int | None = None
    fullscreen: bool | None = None
    renderer: str = "auto"
    quality: str | None = None
    test_pattern: bool = False
    smoke_test: float | None = None
    no_audio: bool = False
    debug: bool = False


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="dragon_ball_powers",
        description=f"{APP_NAME} {__version__}: perform anime ki energy attacks with your webcam.",
    )
    parser.add_argument("--camera", type=int, metavar="N", help="camera index to use (see --list-cameras)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--fullscreen", action="store_true", default=None, help="start in fullscreen")
    mode.add_argument("--windowed", action="store_true", help="start in a window")
    parser.add_argument("--renderer", choices=("auto", "gl", "cpu"), default="auto",
                        help="rendering backend (default: auto = OpenGL with CPU fallback)")
    parser.add_argument("--quality", choices=config.QUALITY_LEVELS, help="effects quality preset")
    parser.add_argument("--no-audio", action="store_true", help="disable sound")
    parser.add_argument("--debug", action="store_true", help="debug logging and the landmark overlay")
    parser.add_argument("--log-level", default=None, choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                        help="console log level (default INFO, DEBUG with --debug)")
    tools = parser.add_argument_group("diagnostics")
    tools.add_argument("--check", action="store_true",
                       help="check Python, packages, models, OpenGL, cameras and audio, then exit")
    tools.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    tools.add_argument("--test-pattern", action="store_true",
                       help="use a generated test pattern instead of the webcam (diagnostics only)")
    tools.add_argument("--smoke-test", type=float, metavar="SECONDS",
                       help="run an automated self-test of the whole pipeline for SECONDS, then exit")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------ logging


def setup_logging(console_level: int, logs_dir: Path | None = None) -> Path | None:
    """Console + rotating file logging. Returns the log file path (None if unavailable)."""
    for stream in (sys.stdout, sys.stderr):
        # Windows consoles default to cp1252/cp850: never crash on non-ASCII log text.
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(console)

    log_path: Path | None = None
    logs_dir = logs_dir or config.LOGS_DIR
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / LOG_FILE_NAME
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s.%(msecs)03d %(levelname)-7s [%(threadName)s] %(name)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        root.addHandler(file_handler)
    except OSError as exc:
        log.warning("File logging disabled (%s)", exc)
        log_path = None

    # Third-party loggers that are very chatty at DEBUG level.
    for noisy in ("PIL", "matplotlib", "absl", "comtypes", "jax"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    def thread_excepthook(args: threading.ExceptHookArgs) -> None:
        log.critical("Unhandled exception in thread %s", getattr(args.thread, "name", "?"),
                     exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    threading.excepthook = thread_excepthook
    return log_path


# ------------------------------------------------------------------- diagnostics


def _status_line(label: str, ok: bool | None, detail: str = "") -> str:
    tag = {True: "OK  ", False: "FAIL", None: "WARN"}[ok]
    return f"  [{tag}] {label}{(': ' + detail) if detail else ''}"


ESSENTIAL_MODELS = ("hand_landmarker.task", "pose_landmarker_lite.task")


def _verify_models() -> list[tuple[str, bool | None, str]]:
    """Check every model in assets/models/models.json (size + SHA256).

    Reuses scripts/download_models.py (stdlib only, not part of the package). Falls
    back to a presence check if the script or its manifest is unavailable.
    """
    import importlib.util

    results: list[tuple[str, bool | None, str]] = []
    script = config.PROJECT_ROOT / "scripts" / "download_models.py"
    manifest = config.ASSETS_DIR / "models" / "models.json"
    try:
        spec = importlib.util.spec_from_file_location("dbp_download_models", script)
        if spec is None or spec.loader is None:
            raise ImportError(script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module  # dataclasses need the module registered
        spec.loader.exec_module(module)
        for model in module.load_manifest(manifest):
            check = module.verify_model(model, config.MODELS_DIR)
            if check.ok:
                results.append((model.file, True, f"{model.size / 1e6:.1f} MB, SHA256 verified"))
            else:
                essential = model.file in ESSENTIAL_MODELS
                results.append((model.file, False if essential else None,
                                f"{check.detail} - run setup.bat (or scripts\\download_models.py)"))
        return results
    except Exception as exc:  # noqa: BLE001 - fall back to a simple presence check
        log.debug("Model manifest check unavailable (%s); checking presence only", exc)
    for name in ESSENTIAL_MODELS + ("pose_landmarker_full.task", "pose_landmarker_heavy.task"):
        path = config.MODELS_DIR / name
        if path.is_file() and path.stat().st_size > 100_000:
            results.append((name, True, f"{path.stat().st_size / 1e6:.1f} MB"))
        else:
            results.append((name, False if name in ESSENTIAL_MODELS else None,
                            "missing - run setup.bat (or scripts\\download_models.py)"))
    return results


def run_check() -> int:
    """Environment self-test used by setup.bat. Returns 0 when the app can run."""
    print(f"{APP_NAME} {__version__} - self-check")
    print(f"  Project: {config.PROJECT_ROOT}")
    failures = 0
    warnings = 0

    def report(label: str, ok: bool | None, detail: str = "") -> None:
        nonlocal failures, warnings
        failures += ok is False
        warnings += ok is None
        print(_status_line(label, ok, detail), flush=True)

    bits = struct.calcsize("P") * 8
    py_ok = sys.version_info[:2] in ((3, 11), (3, 12)) and bits == 64
    report("Python", py_ok, f"{platform.python_version()} ({bits}-bit) on {platform.platform()}")

    packages = (("numpy", "numpy"), ("OpenCV", "cv2"), ("MediaPipe", "mediapipe"),
                ("pygame-ce", "pygame"), ("ModernGL", "moderngl"))
    for label, module_name in packages:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "?")
            if module_name == "pygame":
                version = f"{module.version.ver} (SDL {'.'.join(map(str, module.get_sdl_version()))})"
            report(label, True, str(version))
        except Exception as exc:  # noqa: BLE001 - report anything that breaks the import
            report(label, False, f"{type(exc).__name__}: {exc}")

    import importlib.util

    if importlib.util.find_spec("pygrabber") is not None:
        report("pygrabber (camera names)", True)
    else:
        report("pygrabber (camera names)", None, "not installed - cameras are listed as 'Camera N'")

    hand_model = config.MODELS_DIR / "hand_landmarker.task"
    for name, ok, detail in _verify_models():
        report(f"Model {name}", ok, detail)

    if hand_model.is_file():
        try:
            import numpy as np

            from dragon_ball_powers.config import TrackingSettings
            from dragon_ball_powers.datatypes import CameraFrame
            from dragon_ball_powers.tracking import HandPoseTracker

            tracker = HandPoseTracker(TrackingSettings())
            started = time.perf_counter()
            frame = CameraFrame(1, np.zeros((360, 640, 3), np.uint8), time.perf_counter())
            result = tracker.process_frame(frame)
            error = tracker.status.error  # process_frame reports model problems here, it never raises
            tracker.stop()
            if error:
                report("Tracking (MediaPipe inference)", False, error)
            else:
                report("Tracking (MediaPipe inference)", True,
                       f"models load and run ({(time.perf_counter() - started) * 1000:.0f} ms incl. load, "
                       f"{result.inference_ms.get('total', 0.0):.0f} ms per frame)")
        except Exception as exc:  # noqa: BLE001
            report("Tracking (MediaPipe inference)", False, f"{type(exc).__name__}: {exc}")

    try:
        import moderngl

        kwargs = {"backend": "egl"} if sys.platform.startswith("linux") else {}
        ctx = moderngl.create_standalone_context(require=330, **kwargs)
        info = ctx.info
        report("OpenGL 3.3 (GPU effects)", True,
               f"{info.get('GL_RENDERER')} / OpenGL {info.get('GL_VERSION')}")
        ctx.release()
    except Exception as exc:  # noqa: BLE001
        report("OpenGL 3.3 (GPU effects)", None,
               f"not available ({exc}); the slower CPU renderer will be used")

    try:
        from dragon_ball_powers.camera import list_cameras

        cameras = list_cameras()
        if cameras:
            names = ", ".join(f"{c.index}: {c.name}{' (IR)' if c.is_ir else ''}" for c in cameras)
            report("Cameras", True, f"{len(cameras)} found - {names}")
        else:
            report("Cameras", None, "none found (connect a webcam; keyboard test controls still work)")
    except Exception as exc:  # noqa: BLE001
        report("Cameras", None, f"enumeration failed: {exc}")

    try:
        import pygame

        pygame.mixer.init()
        report("Audio", True, f"mixer {pygame.mixer.get_init()}")
        pygame.mixer.quit()
    except Exception as exc:  # noqa: BLE001
        report("Audio", None, f"no audio device ({exc}); the app runs without sound")

    verdict = "READY" if failures == 0 else "NOT READY"
    print(f"Result: {verdict} ({failures} problem(s), {warnings} warning(s))")
    return 0 if failures == 0 else 1


def run_list_cameras() -> int:
    from dragon_ball_powers.camera import list_cameras

    cameras = list_cameras()
    if not cameras:
        print("No cameras found.")
        return 1
    for cam in cameras:
        print(f"{cam.index}: {cam.name}{'  (infrared - not suitable)' if cam.is_ir else ''}")
    return 0


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    console_level = getattr(logging, ns.log_level or ("DEBUG" if ns.debug else "INFO"))
    log_path = setup_logging(console_level)
    log.info("%s %s starting (Python %s, %s)", APP_NAME, __version__, platform.python_version(),
             platform.platform())
    if log_path:
        log.info("Log file: %s", log_path)

    try:
        if ns.check:
            return run_check()
        if ns.list_cameras:
            return run_list_cameras()

        from dragon_ball_powers.app import App

        settings = config.load_settings()
        args = AppArgs(
            camera=ns.camera,
            fullscreen=True if ns.fullscreen else (False if ns.windowed else None),
            renderer=ns.renderer,
            quality=ns.quality,
            test_pattern=ns.test_pattern,
            smoke_test=ns.smoke_test,
            no_audio=ns.no_audio,
            debug=ns.debug,
        )
        return App(args, settings).run()
    except KeyboardInterrupt:
        log.info("Interrupted")
        return 130
    except Exception:  # noqa: BLE001 - last line of defence: log and show the traceback
        log.critical("Fatal error", exc_info=True)
        print(f"\n{APP_NAME} crashed. The full error is in {log_path or 'the console output above'}.",
              file=sys.stderr)
        return 1
    finally:
        logging.shutdown()


if __name__ == "__main__":
    sys.exit(main())
