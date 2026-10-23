"""Tests for the video-call output: virtual_camera.py, its settings, UI and app wiring.

The device is always a fake (no OBS / v4l2loopback needed). The end-to-end test runs the
real application with the real pyvirtualcam ``Camera`` wrapper on a fake driver backend.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame
import pytest

from dragon_ball_powers import virtual_camera as vc
from dragon_ball_powers.config import Settings, VirtualCameraSettings, set_setting
from dragon_ball_powers.datatypes import (
    AbilityHUD,
    CalibrationHUD,
    CameraState,
    CameraStatus,
    TrackerStatus,
    VirtualCameraState,
    VirtualCameraStatus,
)
from dragon_ball_powers.fonts import FontManager
from dragon_ball_powers.interfaces import UI_ACTION_KINDS, UIContext
from dragon_ball_powers.primitives import SceneInput, UIText, compute_view_transform
from dragon_ball_powers.ui import UIController
from dragon_ball_powers.virtual_camera import VirtualCamera, friendly_error, probe

ROOT = Path(__file__).resolve().parents[1]
RUNNING = VirtualCameraStatus(VirtualCameraState.RUNNING, "OBS Virtual Camera", "obs", (1280, 720), 30.0, 12)


# ------------------------------------------------------------------- fakes


class FakeCam:
    """Stands in for ``pyvirtualcam.Camera``."""

    def __init__(self, kwargs: dict, device: str, backend: str) -> None:
        self.kwargs = kwargs
        self.device = device
        self.backend = backend
        self.frames: list[np.ndarray] = []
        self.closed = False

    def send(self, frame: np.ndarray) -> None:
        assert not self.closed, "send after close"
        self.frames.append(np.array(frame, copy=True))

    def close(self) -> None:
        self.closed = True


class FakeFactory:
    def __init__(self, fail: Exception | None = None, delay: float = 0.0, busy_times: int = 0,
                 backend: str = "fake") -> None:
        self.fail = fail
        self.delay = delay
        self.busy_times = busy_times              # first N opens fail like a briefly busy OBS camera
        self.backend = backend
        self.calls: list[dict] = []
        self.opened: list[FakeCam] = []

    def __call__(self, **kwargs) -> FakeCam:
        self.calls.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        if self.busy_times > 0:
            self.busy_times -= 1
            raise RuntimeError("'obs' backend: virtual camera output could not be started")
        if self.fail is not None:
            raise self.fail
        cam = FakeCam(kwargs, "Fake Cam", kwargs.get("backend") or self.backend)
        self.opened.append(cam)
        return cam


def wait_for(predicate, timeout: float = 5.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def running(cam: VirtualCamera) -> bool:
    return cam.status.state is VirtualCameraState.RUNNING


# ------------------------------------------------------------ VirtualCamera


def test_start_sends_a_steady_stream_and_stop_closes_the_device():
    factory = FakeFactory()
    cam = VirtualCamera(VirtualCameraSettings(enabled=True, width=320, height=180, fps=50), factory)
    assert cam.status.state is VirtualCameraState.OFF and not cam.frame_due(1.0)
    cam.start()
    assert wait_for(lambda: running(cam))
    status = cam.status
    assert (status.device, status.backend, status.resolution, status.fps) == ("Fake Cam", "fake", (320, 180), 50.0)
    assert status.active
    dev = factory.opened[0]
    assert dev.kwargs == {"width": 320, "height": 180, "fps": 50, "backend": None, "device": None}
    # Black first, then the last picture is repeated while the app sends nothing new.
    assert wait_for(lambda: len(dev.frames) >= 3)
    assert not dev.frames[0].any() and dev.frames[0].shape == (180, 320, 3)
    picture = np.full((180, 320, 3), 77, np.uint8)
    cam.submit(picture)
    assert wait_for(lambda: any(f.mean() == 77 for f in list(dev.frames)))
    assert wait_for(lambda: cam.status.frames >= 4)
    cam.stop()
    assert dev.closed and cam.status.state is VirtualCameraState.OFF
    sent = len(dev.frames)
    cam.submit(picture)                     # ignored once stopped
    cam.stop()                              # idempotent
    time.sleep(0.1)
    assert len(dev.frames) == sent


def test_pictures_are_scaled_to_the_call_size_and_bad_ones_skipped():
    factory = FakeFactory()
    cam = VirtualCamera(VirtualCameraSettings(width=320, height=180, fps=30), factory)
    cam.start()
    assert wait_for(lambda: running(cam))
    dev = factory.opened[0]
    cam.submit(np.full((360, 640, 4), 200, np.uint8))           # RGBA, twice the size
    assert wait_for(lambda: any(f.shape == (180, 320, 3) and f.mean() == 200 for f in list(dev.frames)))
    cam.submit(np.zeros((10, 10), np.float32))                  # unusable: the last picture stays
    time.sleep(0.25)
    assert list(dev.frames)[-1].mean() == 200
    cam.stop()


def test_open_failure_gives_an_actionable_error_and_a_retry_works():
    factory = FakeFactory(fail=RuntimeError(
        "'obs' backend: OBS Virtual Camera device not found! Did you install OBS?\n"
        "'unitycapture' backend: No camera registered. Did you install any camera?"))
    cam = VirtualCamera(VirtualCameraSettings(), factory)
    cam.start()
    assert wait_for(lambda: cam.status.state is VirtualCameraState.ERROR)
    assert cam.status.error == vc.INSTALL_HINT and not cam.status.active
    assert not cam.frame_due(time.perf_counter())
    factory.fail = None
    cam.start()                                                 # a fresh attempt after a failure
    assert wait_for(lambda: running(cam))
    cam.stop()
    assert cam.status.state is VirtualCameraState.OFF and cam.status.error is None


@pytest.mark.parametrize("exc, expected", [
    (RuntimeError("'obs' backend: virtual camera output could not be started\n"
                  "'unitycapture' backend: No camera registered. Did you install any camera?"), vc.IN_USE_HINT),
    (RuntimeError("'obs' backend: OBS Virtual Camera device not found! Did you install OBS?\n"
                  "'unitycapture' backend: No camera registered. Did you install any camera?"), vc.INSTALL_HINT),
    (RuntimeError("'unitycapture' backend: All cameras are already in use."), vc.IN_USE_HINT),
    (RuntimeError("'obs' backend: This backend supports only the 'OBS Virtual Camera' device.\n"
                  "'unitycapture' backend: No camera registered with this name."), vc.DEVICE_HINT),
    (RuntimeError("'obs' backend: This backend supports only the 'OBS Virtual Camera' device.\n"
                  "'unitycapture' backend: No camera registered. Did you install any camera?"), vc.DEVICE_HINT),
    (RuntimeError("'v4l2loopback' backend: No v4l2 loopback device found at /dev/video[0-99]. Did you run "
                  "'modprobe v4l2loopback'? See also pyvirtualcam's documentation."), vc.LINUX_HINT),
    (RuntimeError("'obs' backend: OBS Virtual Camera is not installed in your system. Use the Virtual Camera "
                  "function in OBS 30.0 or later to trigger the installation and activate the System Extension "
                  "in System Settings. You may need to restart your machine after that. [Device not found]"),
     vc.MAC_HINT),
    (ModuleNotFoundError("No module named 'pyvirtualcam'"), vc.NOT_INSTALLED_HINT),
])
def test_driver_errors_become_actionable_messages(exc, expected):
    assert friendly_error(exc) == expected


def test_unknown_errors_keep_their_first_line():
    assert "not available" in friendly_error(KeyError("obs"))
    assert friendly_error(RuntimeError("weird\nsecond line")) == "Virtual camera unavailable: weird"
    # The backend name prefix ('v4l2loopback') must not turn every Linux error into "load the module".
    assert friendly_error(RuntimeError("'v4l2loopback' backend: Device /dev/video0 could not be opened: "
                                       "Permission denied")) == \
        "Virtual camera unavailable: Device /dev/video0 could not be opened: Permission denied"
    # Installed but not loadable (missing Visual C++ runtime): setup.bat would not help.
    message = friendly_error(ImportError("DLL load failed while importing _native_windows_obs"))
    assert "DLL load failed" in message and "Visual C++" in message
    assert vc.is_busy_error(RuntimeError("'obs' backend: virtual camera output could not be started"))
    assert not vc.is_busy_error(RuntimeError("'obs' backend: OBS Virtual Camera device not found!"))


def test_a_briefly_busy_device_is_retried():
    # After a restart a call app holds the previous sender's OBS shared memory for ~1 frame.
    factory = FakeFactory(busy_times=2)
    cam = VirtualCamera(VirtualCameraSettings(), factory)
    cam.start()
    assert wait_for(lambda: running(cam))
    assert len(factory.calls) == 3
    cam.stop()


def test_a_device_that_stays_busy_reports_the_in_use_hint(monkeypatch):
    monkeypatch.setattr(vc, "OPEN_RETRY_SECONDS", 0.3)
    monkeypatch.setattr(vc, "OPEN_RETRY_INTERVAL", 0.05)
    factory = FakeFactory(busy_times=1000)
    cam = VirtualCamera(VirtualCameraSettings(), factory)
    cam.start()
    assert wait_for(lambda: cam.status.state is VirtualCameraState.ERROR)
    assert cam.status.error == vc.IN_USE_HINT and len(factory.calls) >= 3
    cam.stop()


def test_a_restart_reopens_the_same_driver_first():
    factory = FakeFactory(backend="obs")
    cam = VirtualCamera(VirtualCameraSettings(), factory)          # backend "auto"
    cam.start()
    assert wait_for(lambda: running(cam))
    assert factory.calls[0]["backend"] is None
    cam.start(VirtualCameraSettings(width=1920, height=1080))
    assert wait_for(lambda: running(cam) and len(factory.opened) == 2)
    assert factory.calls[-1]["backend"] == "obs"                   # not another driver via "auto"
    cam.stop()


@pytest.mark.parametrize("hz", [60.0, 59.94, 75.0, 144.0])
def test_frame_due_paces_rendering_to_the_call_frame_rate(hz):
    cam = VirtualCamera(VirtualCameraSettings(fps=30), FakeFactory())
    cam.start()
    assert wait_for(lambda: running(cam))
    t0 = 1000.0
    due = sum(cam.frame_due(t0 + i / hz) for i in range(int(hz * 4)))
    assert 118 <= due <= 121, due                # 30 fps over 4 s
    t = t0 + 10.0                                # after a stall: one frame, no catch-up burst
    assert cam.frame_due(t) and not cam.frame_due(t + 0.004)
    cam.stop()


def test_restart_with_new_settings_reopens_the_device():
    factory = FakeFactory()
    cam = VirtualCamera(VirtualCameraSettings(width=320, height=180), factory)
    cam.start()
    assert wait_for(lambda: running(cam))
    cam.start(VirtualCameraSettings(width=1920, height=1080, fps=25, backend="obs", device="OBS Virtual Camera"))
    assert wait_for(lambda: running(cam) and len(factory.opened) == 2)
    assert factory.opened[0].closed
    assert factory.opened[1].kwargs == {"width": 1920, "height": 1080, "fps": 25, "backend": "obs",
                                        "device": "OBS Virtual Camera"}
    assert cam.size == (1920, 1080) and cam.status.resolution == (1920, 1080)
    cam.start()                                  # no-op while running
    time.sleep(0.05)
    assert len(factory.opened) == 2
    cam.stop()


def test_stop_while_the_device_is_still_opening():
    factory = FakeFactory(delay=0.3)
    cam = VirtualCamera(VirtualCameraSettings(), factory)
    cam.start()
    assert cam.status.state is VirtualCameraState.STARTING and cam.status.active
    cam.stop()
    assert cam.status.state is VirtualCameraState.OFF
    assert factory.opened and factory.opened[0].closed and not factory.opened[0].frames


def test_no_sender_threads_are_left_behind():
    before = {t.ident for t in threading.enumerate() if t.name == "virtual-camera"}
    cam = VirtualCamera(VirtualCameraSettings(), FakeFactory())
    for _ in range(3):
        cam.start()
        assert wait_for(lambda: running(cam))
        cam.stop()
    left = {t.ident for t in threading.enumerate() if t.name == "virtual-camera"} - before
    assert not left


def test_probe_opens_and_closes_once():
    factory = FakeFactory()
    ok, detail = probe(VirtualCameraSettings(), factory)
    assert ok and "Fake Cam" in detail and "fake" in detail
    assert factory.opened[0].closed and factory.opened[0].kwargs["width"] == 1280
    ok, detail = probe(VirtualCameraSettings(), FakeFactory(fail=RuntimeError("x\n'obs' backend: Did you install OBS?")))
    assert not ok and detail == vc.INSTALL_HINT


def test_settings_defaults_and_validation():
    s = Settings()
    assert s.virtual_camera == VirtualCameraSettings()
    assert not s.virtual_camera.enabled and not s.virtual_camera.mirror
    assert (s.virtual_camera.width, s.virtual_camera.height, s.virtual_camera.fps) == (1280, 720, 30)
    assert set_setting(s, "virtual_camera.width", 1279) == 1278           # YUV 4:2:0 needs even sizes
    loaded = Settings.from_dict({"virtual_camera": {"enabled": True, "height": 719, "backend": "nope", "fps": 500}})
    v = loaded.virtual_camera
    assert v.enabled and v.height == 718 and v.backend == "auto" and v.fps == 60
    assert Settings.from_dict(s.to_dict()).virtual_camera == s.virtual_camera
    assert vc.even_size(VirtualCameraSettings(width=641, height=361)) == (640, 360)


# ---------------------------------------------------------------------- UI


@pytest.fixture(scope="module")
def fonts():
    pygame.init()
    return FontManager()


def ui_ctx(settings: Settings, size=(1920, 1080), vcam=None, callout: str | None = "KA... ME...") -> UIContext:
    return UIContext(
        now=100.0, window_size=size, view=compute_view_transform(1280, 720, *size), settings=settings,
        camera_status=CameraStatus(state=CameraState.RUNNING, index=0, name="Cam", resolution=(1280, 720)),
        tracker_status=TrackerStatus(running=True),
        ability=AbilityHUD(callout=callout, callout_intensity=1.0 if callout else 0.0),
        calibration=CalibrationHUD(), virtual_camera=vcam)


def texts(draw_list) -> str:
    return " ".join(c.text for c in draw_list.commands if isinstance(c, UIText))


def test_overlay_contains_only_the_callout(fonts):
    ui = UIController(Settings(), fonts)
    s = ui._settings
    ui.build(ui_ctx(s))
    widgets = ui.widgets
    overlay = ui.build_overlay(ui_ctx(s, size=(1280, 720)))
    assert overlay.commands and all(isinstance(c, UIText) for c in overlay.commands)
    assert {c.text for c in overlay.commands} == {"KA... ME..."}
    assert all(0 <= c.x <= 1280 and 0 <= c.y <= 720 for c in overlay.commands)
    assert ui.widgets == widgets                  # the interactive HUD state is untouched
    assert len(ui.build_overlay(ui_ctx(s, callout=None))) == 0


def test_hud_shows_vcam_chip_and_help(fonts):
    ui = UIController(Settings(), fonts)
    s = ui._settings
    assert "VCAM" not in texts(ui.build(ui_ctx(s)))
    assert "VCAM" in texts(ui.build(ui_ctx(s, vcam=RUNNING)))
    fullscreen = dataclasses.replace(ui_ctx(s, vcam=RUNNING), fullscreen=True)
    assert "VCAM" in texts(ui.build(fullscreen))                  # also in cinematic mode
    assert "VCAM" in texts(ui.build(ui_ctx(s, vcam=VirtualCameraStatus(VirtualCameraState.STARTING))))
    assert "VCAM" not in texts(ui.build(ui_ctx(s, vcam=VirtualCameraStatus(VirtualCameraState.ERROR, error="x"))))
    assert "V video calls" in texts(ui.build(ui_ctx(s)))        # key help strip (first seconds)


def test_settings_panel_video_call_section(fonts):
    ui = UIController(Settings(), fonts)
    s = ui._settings
    ui.toggle_settings()
    shown = texts(ui.build(ui_ctx(s, vcam=RUNNING)))
    assert "VIDEO CALLS" in shown and "OBS Virtual Camera" in shown
    toggle = ui.widget("settings:virtual_camera")
    assert toggle is not None and toggle.action.kind == "toggle_virtual_camera"
    res = ui.widget("settings:virtual_camera.resolution:1920x1080")
    assert res is not None and res.action.kind == "set_virtual_camera_resolution" and res.action.value == (1920, 1080)
    mirror = ui.widget("settings:virtual_camera.mirror")
    assert mirror is not None and mirror.action.value == ("virtual_camera.mirror", True)
    for widget in (toggle, res, mirror):
        assert widget.action.kind in UI_ACTION_KINDS
    error = VirtualCameraStatus(VirtualCameraState.ERROR, error=vc.INSTALL_HINT)
    assert "install_virtual_camera.bat" in texts(ui.build(ui_ctx(s, vcam=error)))
    assert "virtual camera" in texts(ui.build(ui_ctx(s)))      # the explanation while off


# --------------------------------------------------------------- app wiring


class RecordingRenderer:
    name = "recorder"

    def __init__(self, size, fail: bool = False) -> None:
        self.size = size
        self.fail = fail
        self.mirror_output = False
        self.calls: list[tuple] = []
        self.released = False

    def render(self, scene, effects, ui, view) -> None:
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append((scene, ui, view, self.mirror_output))

    def read_pixels(self) -> np.ndarray:
        return np.zeros((self.size[1], self.size[0], 3), np.uint8)

    def set_quality(self, preset) -> None:
        pass

    def release(self) -> None:
        self.released = True


class FakeVcam:
    size = (320, 180)

    def __init__(self) -> None:
        self.pictures: list[np.ndarray] = []
        self.stopped = False
        self.status = RUNNING

    def frame_due(self, now: float) -> bool:
        return True

    def submit(self, frame: np.ndarray) -> None:
        self.pictures.append(frame)

    def stop(self) -> None:
        self.stopped = True


def make_app(fonts, renderer: RecordingRenderer | None = None, vcam=None, **args):
    from dragon_ball_powers.app import App
    from dragon_ball_powers.main import AppArgs

    settings = Settings()
    app = App(AppArgs(**args), settings)
    app.ui = UIController(settings, fonts)
    app.vcam = vcam if vcam is not None else FakeVcam()
    if renderer is not None:
        app._vcam_renderer_for = lambda size: renderer  # no GL needed
    return app, settings


def test_app_sends_the_call_picture_the_right_way_round(fonts):
    rec = RecordingRenderer((320, 180))
    app, settings = make_app(fonts, rec)
    scene = SceneInput(None, -1, (1280, 720), 0.0)
    ctx = ui_ctx(settings, size=(1280, 720), callout="HA!")
    app._render_virtual_camera(1.0, scene, ctx)
    _, overlay, view, mirrored = rec.calls[-1]
    assert mirrored                                 # mirrored camera, plain call picture: flip back
    assert view == compute_view_transform(1280, 720, 320, 180, "cover")
    assert {c.text for c in overlay.commands} == {"HA!"}
    assert len(app.vcam.pictures) == 1 and app._vcam_submitted == 1
    settings.virtual_camera.mirror = True           # user wants the mirrored look in the call
    app._render_virtual_camera(2.0, scene, ctx)
    assert rec.calls[-1][3] is False
    settings.camera.mirror = False                  # unmirrored camera, mirrored call picture
    app._render_virtual_camera(3.0, scene, ctx)
    assert rec.calls[-1][3] is True


def test_app_stops_the_call_output_after_repeated_render_failures(fonts):
    app, settings = make_app(fonts, RecordingRenderer((320, 180), fail=True))
    settings.virtual_camera.enabled = True
    vcam = app.vcam
    scene = SceneInput(None, -1, (1280, 720), 0.0)
    for i in range(3):
        app._render_virtual_camera(float(i), scene, ui_ctx(settings))
    assert vcam.stopped and not settings.virtual_camera.enabled
    assert any("Virtual camera stopped" in t.text for t in app.ui._toasts)


def test_app_announces_live_and_failure_once(fonts):
    app, settings = make_app(fonts, RecordingRenderer((320, 180)))
    settings.virtual_camera.enabled = True
    assert app._poll_virtual_camera() is RUNNING
    app._poll_virtual_camera()
    assert sum("Virtual camera live" in t.text for t in app.ui._toasts) == 1
    app.vcam.status = VirtualCameraStatus(VirtualCameraState.ERROR, error=vc.IN_USE_HINT)
    app._poll_virtual_camera()
    assert any(t.text == vc.IN_USE_HINT and t.level == "error" for t in app.ui._toasts)
    assert not settings.virtual_camera.enabled      # not retried at the next start


def test_test_pattern_is_not_flipped_for_the_call(fonts):
    rec = RecordingRenderer((320, 180))
    app, settings = make_app(fonts, rec, test_pattern=True)    # the camera module never mirrors it
    app._render_virtual_camera(1.0, SceneInput(None, -1, (1280, 720), 0.0), ui_ctx(settings))
    assert rec.calls[-1][3] is False


def test_app_announces_a_repeated_failure_again(fonts):
    from dragon_ball_powers.interfaces import UIAction

    failing = VirtualCamera(VirtualCameraSettings(), FakeFactory(fail=RuntimeError(
        "'obs' backend: OBS Virtual Camera device not found! Did you install OBS?")))
    app, settings = make_app(fonts, vcam=failing)
    shown: list[str] = []
    notify = app.ui.notify
    app.ui.notify = lambda text, level="info", duration=3.0: (shown.append(text), notify(text, level, duration))
    for attempt in (1, 2):
        app.apply_action(UIAction("toggle_virtual_camera"), 0.0)
        assert settings.virtual_camera.enabled
        assert wait_for(lambda: failing.status.state is VirtualCameraState.ERROR)
        app._poll_virtual_camera()
        assert shown.count(vc.INSTALL_HINT) == attempt          # announced every time
        assert not settings.virtual_camera.enabled              # and not retried at the next start
    app._stop_virtual_camera(release_renderer=True)


def test_app_reopens_the_output_only_when_needed(fonts):
    from dragon_ball_powers.interfaces import UIAction

    factory = FakeFactory()
    app, settings = make_app(fonts, vcam=VirtualCamera(VirtualCameraSettings(), factory))
    app.apply_action(UIAction("toggle_virtual_camera"), 0.0)
    assert wait_for(lambda: running(app.vcam))
    app.apply_action(UIAction("set_virtual_camera_resolution", (1280, 720)), 0.0)     # unchanged
    app.apply_action(UIAction("set_setting", ("virtual_camera.mirror", True)), 0.0)    # read per frame
    time.sleep(0.05)
    assert len(factory.opened) == 1 and running(app.vcam)      # the call is not interrupted
    app.apply_action(UIAction("set_virtual_camera_resolution", (1920, 1080)), 0.0)
    assert wait_for(lambda: len(factory.opened) == 2 and running(app.vcam))
    assert factory.opened[1].kwargs["width"] == 1920 and factory.opened[0].closed
    app._stop_virtual_camera(release_renderer=True)


def test_cli_flag_is_for_one_run_only(fonts, monkeypatch):
    from dragon_ball_powers import app as app_module

    saved: list[dict] = []
    monkeypatch.setattr(app_module, "save_settings", lambda s, path=None: saved.append(s.to_dict()))
    app, settings = make_app(fonts, vcam=VirtualCamera(VirtualCameraSettings(), FakeFactory()), virtual_camera=True)
    app._apply_cli_overrides()
    assert settings.virtual_camera.enabled                     # on for this run ...
    app._mark_dirty()
    app._maybe_save_settings(10.0, force=True)
    assert saved[-1]["virtual_camera"]["enabled"] is False     # ... but the saved choice stays off
    app._set_virtual_camera(True)                              # the user's own choice is saved
    app._maybe_save_settings(20.0, force=True)
    assert saved[-1]["virtual_camera"]["enabled"] is True
    app._stop_virtual_camera(release_renderer=True)


def test_app_does_not_open_a_virtual_camera_as_its_webcam(fonts, monkeypatch):
    from dragon_ball_powers import app as app_module
    from dragon_ball_powers.datatypes import CameraInfo

    class FakeCameraManager:
        status = CameraStatus()

        def __init__(self) -> None:
            self.started: list[tuple] = []

        def start(self, index, name=None) -> None:
            self.started.append((index, name))

    monkeypatch.setattr(app_module, "list_cameras",
                        lambda backend="auto": [CameraInfo(0, "OBS Virtual Camera", is_virtual=True)])
    app, settings = make_app(fonts)
    app.camera = FakeCameraManager()
    app._scan_cameras(start_after=True)
    assert wait_for(lambda: not app._scanning)
    assert app.camera.started == []                    # its own output would loop back
    notice = app._camera_notice                         # shown by the main thread's next _draw
    assert notice is not None and "No webcam found" in notice[0] and notice[1] == "warning"


def test_cli_flag_enables_the_virtual_camera():
    from dragon_ball_powers.main import _parse_args

    assert _parse_args(["--virtual-camera"]).virtual_camera
    assert not _parse_args([]).virtual_camera


# --------------------------------------------------------------- end to end


def _gl_available() -> bool:
    try:
        import moderngl

        backend = {"backend": "egl"} if sys.platform.startswith("linux") else {}
        moderngl.create_standalone_context(require=330, **backend).release()
        return True
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.slow
@pytest.mark.parametrize("renderer", ["gl", "cpu"])
def test_app_sends_effects_to_a_virtual_camera(tmp_path, renderer):
    if renderer == "gl" and not _gl_available():
        pytest.skip("no OpenGL 3.3")
    pytest.importorskip("pyvirtualcam")
    out = tmp_path / "vcam"
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT / "src"),
        "SDL_VIDEODRIVER": ("offscreen" if sys.platform.startswith("linux") else "windows") if renderer == "gl"
        else "dummy",
        "SDL_AUDIODRIVER": "dummy",
        "DBP_CONFIG_DIR": str(tmp_path / "config"),
        "DBP_LOGS_DIR": str(tmp_path / "logs"),
        "DBP_FAKE_VCAM_DIR": str(out),
    })
    cmd = [sys.executable, str(ROOT / "tests" / "fake_virtual_camera.py"), "--test-pattern", "--smoke-test",
           "9" if renderer == "gl" else "6", "--windowed", "--renderer", renderer, "--virtual-camera"]
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=240)
    log_path = tmp_path / "logs" / "dragon_ball_powers.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    assert proc.returncode == 0, proc.stderr[-4000:] + "\n" + log_text[-4000:]
    assert "PASS" in log_text and "Traceback" not in log_text, log_text[-4000:]
    assert "Virtual camera started: 'Fake Virtual Camera' via fake, 1280x720 @ 30 fps" in log_text
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["size"] == [1280, 720] and stats["fps"] == 30
    assert stats["opened"] == 2 and stats["closed"] == 2           # switched off and on once
    assert stats["frames"] > 40
    best = np.load(out / "best.npy")
    assert best.shape == (720, 1280, 3) and best.std() > 20        # the test pattern + effects, not black
