"""Tests for camera.py: capture thread, device enumeration and the synthetic test pattern.

No webcam is needed: ``cv2.VideoCapture`` is replaced inside the camera module by a fake
that simulates devices (streaming, open failures, missing frames, disconnects, hangs).
Timing assertions are deliberately loose because many test runs share this machine.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

from dragon_ball_powers import camera
from dragon_ball_powers.config import CameraSettings
from dragon_ball_powers.datatypes import CameraInfo, CameraState

pytestmark = pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")

W, H = 64, 48
DSHOW, MSMF, ANY = int(cv2.CAP_DSHOW), int(cv2.CAP_MSMF), int(cv2.CAP_ANY)
MJPG = cv2.VideoWriter_fourcc(*"MJPG")
RUNNING, STARTING, STOPPED, ERROR = (CameraState.RUNNING, CameraState.STARTING,
                                     CameraState.STOPPED, CameraState.ERROR)

# ------------------------------------------------------------------------- helpers


def wait_until(predicate, timeout: float = 3.0, interval: float = 0.005) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def make_frame(n: int, width: int = W, height: int = H, marker: int = 0) -> np.ndarray:
    """Asymmetric BGR frame (bright left quarter); pixel (0, 0) encodes n and the device."""
    img = np.zeros((height, width, 3), np.uint8)
    img[:, : width // 4] = 255
    img[:, width // 4: width // 2, 2] = 128
    img[0, 0] = (n % 256, (n // 256) % 256, marker)
    return img


def decode(image: np.ndarray, mirrored: bool = True) -> tuple[int, int]:
    """(frame number, device marker) of a published frame."""
    px = image[0, -1] if mirrored else image[0, 0]
    return int(px[0]) + 256 * int(px[1]), int(px[2])


def cam_settings(**overrides) -> CameraSettings:
    values = dict(index=0, width=W, height=H, fps=30, backend="auto", mirror=True,
                  prefer_mjpg=True)
    values.update(overrides)
    return CameraSettings(**values)


def collect_frames(source, count: int, timeout: float = 3.0) -> list:
    frames, last = [], 0
    deadline = time.perf_counter() + timeout
    while len(frames) < count and time.perf_counter() < deadline:
        frame = source.wait_for_frame(last, 0.5)
        if frame is not None:
            frames.append(frame)
            last = frame.frame_id
    return frames


def frame_from(cam, marker: int, timeout: float = 3.0):
    """First frame published by the device with ``marker``."""
    deadline = time.perf_counter() + timeout
    last = 0
    while time.perf_counter() < deadline:
        frame = cam.wait_for_frame(last, 0.5)
        if frame is not None:
            last = frame.frame_id
            if decode(frame.image)[1] == marker:
                return frame
    return None


# ---------------------------------------------------------------------- fake devices


@dataclass
class FakeDevice:
    width: int = W
    height: int = H
    apis: set[int] | None = None          # backends that can open it (None = all)
    opens_allowed: int | None = None      # successful opens before it refuses
    frames_per_open: int | None = None    # frames per session, then read() fails
    limited_sessions: int | None = None   # frames_per_open applies to the first N sessions
    deliver_frames: bool = True           # False: opens, but read() never succeeds
    interval: float = 0.004               # seconds per read()
    reuse_buffer: bool = False            # return the same ndarray object on every read
    channels: int = 3                     # 1 -> (H, W) grey, 4 -> BGRA
    marker: int = 0                       # identifies the device in pixel (0, 0)
    honor_size: bool = False              # deliver the size requested via set()
    read_error: Exception | None = None
    set_error_props: frozenset = frozenset()
    open_gate: threading.Event | None = None
    read_gate: threading.Event | None = None
    opens: int = 0
    reads: int = 0
    at_gate: bool = False

    def image(self, n: int, size: tuple[int, int]) -> np.ndarray:
        img = make_frame(n, size[0], size[1], self.marker)
        if self.channels == 1:
            return np.ascontiguousarray(img[:, :, 2])
        if self.channels == 4:
            return np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
        return img


class FakeWorld:
    def __init__(self) -> None:
        self.devices: dict[int, FakeDevice] = {}
        self.captures: list = []
        self.lock = threading.Lock()

    def add(self, index: int, **kwargs) -> FakeDevice:
        device = FakeDevice(**kwargs)
        self.devices[index] = device
        return device

    def captures_for(self, index: int) -> list:
        with self.lock:
            return [c for c in self.captures if c.index == index]

    def open_gates(self) -> None:
        for device in self.devices.values():
            for gate in (device.open_gate, device.read_gate):
                if gate is not None:
                    gate.set()


def fake_capture_class(world: FakeWorld):
    class FakeCapture:
        """Mimics the parts of cv2.VideoCapture that camera.py uses."""

        def __init__(self, index, api=ANY):
            self.index, self.api = int(index), int(api)
            self.sets: list = []
            self.released = 0
            self.session_reads = 0
            self._buffer = None
            self.device = device = world.devices.get(self.index)
            with world.lock:
                world.captures.append(self)
            if device is not None and device.open_gate is not None:
                device.open_gate.wait(5.0)
            self._opened = bool(device is not None
                                and (device.apis is None or self.api in device.apis)
                                and (device.opens_allowed is None
                                     or device.opens < device.opens_allowed))
            self.size = (device.width, device.height) if device is not None else (0, 0)
            if self._opened:
                device.opens += 1
                limited = device.limited_sessions is None or device.opens <= device.limited_sessions
                self.frame_limit = device.frames_per_open if limited else None

        def isOpened(self):
            return self._opened and not self.released

        def set(self, prop, value):
            self.sets.append((prop, value))
            device = self.device
            if device is not None and prop in device.set_error_props:
                raise cv2.error("property not supported")
            if device is not None and device.honor_size:
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    self.size = (int(value), self.size[1])
                elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    self.size = (self.size[0], int(value))
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FOURCC:
                return float(MJPG)
            return 0.0

        def getBackendName(self):
            if not self._opened:
                raise cv2.error("not opened")
            return {DSHOW: "DSHOW", MSMF: "MSMF"}.get(self.api, "FAKE")

        def read(self):
            device = self.device
            if not self.isOpened():
                return False, None
            gate = device.read_gate
            if gate is not None and not gate.is_set():
                device.at_gate = True
                gate.wait(5.0)
                device.at_gate = False
            time.sleep(device.interval)
            if device.read_error is not None:
                raise device.read_error
            if not device.deliver_frames or (self.frame_limit is not None
                                             and self.session_reads >= self.frame_limit):
                return False, None
            self.session_reads += 1
            device.reads += 1
            img = device.image(device.reads, self.size)
            if device.reuse_buffer:
                if self._buffer is None or self._buffer.shape != img.shape:
                    self._buffer = img
                else:
                    self._buffer[...] = img
                img = self._buffer
            return True, img

        def release(self):
            self.released += 1

    return FakeCapture


@pytest.fixture
def world(monkeypatch):
    fake_world = FakeWorld()
    monkeypatch.setattr(camera.cv2, "VideoCapture", fake_capture_class(fake_world))
    # Keep error texts deterministic (the real check looks at /dev/video* or DirectShow).
    monkeypatch.setattr(camera, "_device_exists", lambda index, backend: None)
    yield fake_world
    fake_world.open_gates()


@pytest.fixture
def make_camera(world):
    created: list[camera.CameraManager] = []

    def factory(settings: CameraSettings | None = None, **kwargs) -> camera.CameraManager:
        kwargs.setdefault("open_timeout", 0.3)
        kwargs.setdefault("failure_timeout", 0.15)
        cam = camera.CameraManager(settings or cam_settings(), **kwargs)
        created.append(cam)
        return cam

    yield factory
    world.open_gates()
    for cam in created:
        cam.stop()


def install_fake_pygrabber(monkeypatch, names: list[str], error: Exception | None = None):
    package = types.ModuleType("pygrabber")
    module = types.ModuleType("pygrabber.dshow_graph")

    class FilterGraph:
        def get_input_devices(self):
            if error is not None:
                raise error
            return list(names)

    module.FilterGraph = FilterGraph
    package.dshow_graph = module
    monkeypatch.setitem(sys.modules, "pygrabber", package)
    monkeypatch.setitem(sys.modules, "pygrabber.dshow_graph", module)


# ------------------------------------------------------------------ module basics


def test_msmf_hw_transforms_disabled_before_cv2_import():
    assert "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" in os.environ
    source = Path(camera.__file__).read_text(encoding="utf-8")
    setting = source.index('os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")')
    assert setting < source.index("\nimport cv2")


def test_backend_id_mapping(monkeypatch):
    assert camera.backend_id("dshow") == DSHOW
    assert camera.backend_id("MSMF") == MSMF
    assert camera.backend_id(" any ") == ANY
    monkeypatch.setattr(sys, "platform", "win32")
    assert camera.backend_id("auto") == DSHOW
    monkeypatch.setattr(sys, "platform", "linux")
    assert camera.backend_id("auto") == ANY
    assert camera.backend_id("bogus") == ANY  # unknown names behave like "auto"


# ---------------------------------------------------------------------- streaming


def test_streaming_publishes_mirrored_private_frames(world, make_camera):
    world.add(0, reuse_buffer=True)  # the fake overwrites one buffer on every read
    cam = make_camera()
    cam.start()
    frames = collect_frames(cam, 6)
    assert len(frames) == 6
    ids = [f.frame_id for f in frames]
    assert ids == sorted(ids) and len(set(ids)) == 6
    assert [f.timestamp for f in frames] == sorted(f.timestamp for f in frames)
    for f in frames:
        assert f.image.shape == (H, W, 3) and f.image.dtype == np.uint8
        assert f.image.flags["C_CONTIGUOUS"] and f.size == (W, H)
        n, _ = decode(f.image, mirrored=True)
        source = make_frame(n)
        assert np.array_equal(f.image, source[:, ::-1])  # horizontally mirrored source frame
        assert not np.array_equal(f.image, source)
    # Published frames are private copies: later reads did not overwrite earlier frames.
    assert len({decode(f.image)[0] for f in frames}) == 6
    status = cam.status
    assert status.state is RUNNING and status.error is None
    assert status.resolution == (W, H) and status.frames >= 6 and status.fps > 0
    assert status.backend == "FAKE" and status.index == 0 and status.name == "Camera 0"
    assert cam.latest().frame_id >= frames[-1].frame_id and cam.is_running


def test_mirror_disabled_delivers_unflipped_private_copies(world, make_camera):
    world.add(0, reuse_buffer=True)
    cam = make_camera(cam_settings(mirror=False))
    cam.start()
    frames = collect_frames(cam, 4)
    assert len(frames) == 4
    for f in frames:
        n, _ = decode(f.image, mirrored=False)
        assert np.array_equal(f.image, make_frame(n))
    assert len({decode(f.image, mirrored=False)[0] for f in frames}) == 4
    buffer = world.captures_for(0)[0]._buffer
    assert not any(np.shares_memory(f.image, buffer) for f in frames)


@pytest.mark.parametrize("channels", [1, 4])
def test_grey_and_bgra_frames_are_converted_to_bgr(world, make_camera, channels):
    world.add(0, channels=channels)
    cam = make_camera()
    cam.start()
    frame = cam.wait_for_frame(0, 3.0)
    assert frame is not None
    assert frame.image.shape == (H, W, 3) and frame.image.dtype == np.uint8
    if channels == 1:
        image = frame.image.astype(int)
        assert (image[..., 0] == image[..., 1]).all() and (image[..., 1] == image[..., 2]).all()
        assert image[5, -1, 0] == 255 and image[5, 0, 0] == 0  # mirrored: bright band on the right
    else:
        n, _ = decode(frame.image)
        assert np.array_equal(frame.image, make_frame(n)[:, ::-1])


def test_configures_mjpg_size_fps_buffer_in_order(world, make_camera):
    world.add(0, set_error_props=frozenset({cv2.CAP_PROP_BUFFERSIZE}))  # one set() raises
    cam = make_camera(cam_settings(width=1280, height=720, fps=60))
    cam.start()
    assert wait_until(lambda: cam.status.state is RUNNING)
    assert world.captures_for(0)[0].sets == [
        (cv2.CAP_PROP_FOURCC, MJPG),
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
        (cv2.CAP_PROP_FPS, 60),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
    ]
    # The device cannot do 1280x720: the status reports the real frame size.
    assert cam.status.resolution == (W, H)
    cam.stop()
    cam = make_camera(cam_settings(prefer_mjpg=False))
    cam.start()
    assert wait_until(lambda: cam.status.state is RUNNING)
    assert [prop for prop, _ in world.captures_for(0)[-1].sets] == [
        cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS,
        cv2.CAP_PROP_BUFFERSIZE]


# ------------------------------------------------------------------------ failures


def test_open_failure_reports_actionable_error(world, make_camera, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    cam = make_camera()
    cam.start(0, name="HP 5MP Camera")
    assert wait_until(lambda: cam.status.state is ERROR)
    status = cam.status
    assert status.error == (
        "Could not open camera 0 (HP 5MP Camera). Close other apps that use the camera (Teams, "
        "Zoom, OBS, browser) and allow camera access in Windows Settings > Privacy & security > "
        "Camera > Let desktop apps access your camera.")
    assert status.index == 0 and status.name == "HP 5MP Camera"
    assert not cam.is_running and cam.latest() is None
    assert all(c.released for c in world.captures)


def test_missing_device_message(world, make_camera, monkeypatch):
    monkeypatch.setattr(camera, "_device_exists", lambda index, backend: False)
    cam = make_camera(cam_settings(index=3))
    cam.start()
    assert wait_until(lambda: cam.status.state is ERROR)
    assert cam.status.error == "Camera 3 does not exist. Select another camera in Settings (Tab)."
    assert cam.status.index == 3


@pytest.mark.parametrize("variant", ["read_fails", "read_raises_cv2_error"])
def test_no_frames_within_open_timeout(world, make_camera, variant):
    if variant == "read_fails":
        world.add(0, deliver_frames=False)
    else:
        world.add(0, read_error=cv2.error("grab failed"))
    cam = make_camera(open_timeout=0.3)
    started = time.perf_counter()
    cam.start(0, name="USB Cam")
    assert wait_until(lambda: cam.status.state is ERROR, timeout=5.0)
    assert time.perf_counter() - started >= 0.28  # waited the whole open_timeout
    error = cam.status.error
    assert error.startswith("Camera 0 (USB Cam) opened but delivered no frames within 0.3 s.")
    assert "Settings (Tab)" in error
    assert wait_until(lambda: all(c.released == 1 for c in world.captures_for(0)))


def test_no_frames_from_ir_camera_explains_why(world, make_camera):
    world.add(1, deliver_frames=False)
    cam = make_camera(cam_settings(index=1))
    cam.start(name="HP IR Camera")
    assert wait_until(lambda: cam.status.state is ERROR, timeout=5.0)
    assert "infrared (Windows Hello) camera" in cam.status.error


def test_ir_camera_open_failure_points_to_the_colour_camera(world, make_camera, monkeypatch):
    # Windows Hello usually holds its IR camera; "close Teams/Zoom" would send the user astray.
    monkeypatch.setattr(sys, "platform", "win32")
    cam = make_camera(cam_settings(index=1))
    cam.start(name="HP IR Camera")
    assert wait_until(lambda: cam.status.state is ERROR)
    assert cam.status.error == (
        "Could not open camera 1 (HP IR Camera). This is an infrared (Windows Hello) camera that "
        "cannot be used for tracking. Select your normal camera in Settings (Tab).")


def test_disconnect_triggers_one_reopen_then_error(world, make_camera):
    world.add(0, frames_per_open=5, opens_allowed=1)
    cam = make_camera(failure_timeout=0.15)
    cam.start(0, name="USB Cam")
    assert wait_until(lambda: cam.status.state is ERROR, timeout=5.0)
    status = cam.status
    assert status.error.startswith("Camera 0 (USB Cam) disconnected or stopped delivering frames.")
    assert status.frames == 5
    caps = world.captures_for(0)
    assert len(caps) == 2  # the initial open plus exactly one reopen attempt
    assert wait_until(lambda: all(c.released == 1 for c in caps))
    assert cam.latest() is None and not cam.is_running


def test_reconnect_after_temporary_failure_keeps_streaming(world, make_camera):
    world.add(0, frames_per_open=5, limited_sessions=1)
    cam = make_camera(failure_timeout=0.1)
    cam.start()
    assert wait_until(lambda: cam.status.frames >= 12 and cam.status.state is RUNNING, timeout=5.0)
    caps = world.captures_for(0)
    assert len(caps) == 2 and caps[0].released == 1 and caps[1].released == 0
    assert cam.latest().frame_id >= 12  # ids kept increasing across the reconnect
    assert cam.status.error is None


def test_reconnect_follows_renumbered_device_by_name(world, make_camera, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    # After the replug DirectShow lists the camera second; index 0 is another device now.
    install_fake_pygrabber(monkeypatch, ["HP 5MP Camera", "USB Cam"])
    world.add(0, frames_per_open=5, opens_allowed=1)
    world.add(1, marker=1)
    cam = make_camera(cam_settings(backend="dshow"), failure_timeout=0.1)
    cam.start(0, name="USB Cam")
    assert wait_until(lambda: cam.status.index == 1 and cam.status.state is RUNNING, timeout=5.0)
    assert frame_from(cam, marker=1) is not None
    assert len(world.captures_for(0)) == 1  # the stale index was not reopened


def test_reconnect_gives_up_when_device_is_gone(world, make_camera, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    install_fake_pygrabber(monkeypatch, ["HP 5MP Camera"])
    world.add(0, frames_per_open=5)
    cam = make_camera(cam_settings(backend="dshow"), failure_timeout=0.1)
    cam.start(0, name="USB Cam")
    assert wait_until(lambda: cam.status.state is ERROR, timeout=5.0)
    assert "disconnected" in cam.status.error
    assert len(world.captures_for(0)) == 1  # did not open whatever is at index 0 now


def test_restart_after_reconnect_reopens_the_moved_device(world, make_camera, monkeypatch):
    # Regression: restart() used to reopen the index of the original start(), which after a
    # DirectShow renumbering belongs to another camera, and still labelled it "USB Cam".
    monkeypatch.setattr(sys, "platform", "win32")
    install_fake_pygrabber(monkeypatch, ["HP IR Camera", "USB Cam"])  # order after the replug
    stale = world.add(0, frames_per_open=5, opens_allowed=1)
    world.add(1, marker=1)
    cam = make_camera(cam_settings(backend="dshow"), failure_timeout=0.1)
    cam.start(0, name="USB Cam")
    assert wait_until(lambda: cam.status.index == 1 and cam.status.state is RUNNING, timeout=5.0)
    stale.opens_allowed = stale.frames_per_open = None  # index 0 works - but is another camera
    cam.restart(cam_settings(backend="dshow", width=80, height=60))  # e.g. resolution changed
    assert frame_from(cam, marker=1) is not None
    assert (cam.status.index, cam.status.name) == (1, "USB Cam")
    assert len(world.captures_for(0)) == 1  # the stale index was never reopened


@pytest.mark.parametrize("start_name", ["USB Cam", "usb cam  "])
def test_reconnect_matches_padded_directshow_names(world, make_camera, monkeypatch, start_name):
    # Regression: names are stripped for CameraInfo but were compared unstripped on reconnect,
    # so a temporary dropout ended in "disconnected" although the camera never moved.
    monkeypatch.setattr(sys, "platform", "win32")
    install_fake_pygrabber(monkeypatch, ["  USB Cam ", "HP IR Camera   "])  # padded by the driver
    cameras = camera.list_cameras()
    assert [(c.name, c.is_ir) for c in cameras] == [("USB Cam", False), ("HP IR Camera", True)]
    world.add(0, frames_per_open=5, limited_sessions=1)  # one dropout, then fine again
    cam = make_camera(cam_settings(backend="dshow"), failure_timeout=0.1)
    cam.start(0, name=start_name)
    assert wait_until(lambda: cam.status.frames >= 12 and cam.status.state is RUNNING, timeout=5.0)
    assert cam.status.error is None and cam.status.index == 0
    assert len(world.captures_for(0)) == 2


def test_reconnect_reports_starting_until_frames_flow_again(world, make_camera):
    gate = threading.Event()
    gate.set()
    device = world.add(0, open_gate=gate)
    cam = make_camera(failure_timeout=0.1)
    cam.start()
    assert wait_until(lambda: cam.status.state is RUNNING)
    gate.clear()                   # the reopen will block inside the driver
    device.deliver_frames = False  # the stream dies
    assert wait_until(lambda: len(world.captures_for(0)) == 2, timeout=5.0)
    status = cam.status
    assert status.state is STARTING and status.error is None and status.fps == 0.0
    assert cam.is_running and world.captures_for(0)[0].released == 1  # old handle released first
    device.deliver_frames = True
    gate.set()
    assert wait_until(lambda: cam.status.state is RUNNING and cam.status.error is None)


def test_auto_backend_falls_back_dshow_msmf_any_on_windows(world, make_camera, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    world.add(0, apis={ANY})
    cam = make_camera()
    cam.start()
    assert wait_until(lambda: cam.status.state is RUNNING)
    caps = world.captures_for(0)
    assert [c.api for c in caps] == [DSHOW, MSMF, ANY]
    assert [c.released for c in caps] == [1, 1, 0]
    assert cam.status.backend == "FAKE"
    cam.stop()
    world.add(0, apis={MSMF})
    cam.start()
    assert wait_until(lambda: cam.status.state is RUNNING)
    assert [c.api for c in world.captures_for(0)[3:]] == [DSHOW, MSMF]
    assert cam.status.backend == "MSMF"


def test_explicit_backend_and_non_windows_auto_do_not_fall_back(world, make_camera, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    world.add(0, apis={MSMF})
    cam = make_camera(cam_settings(backend="dshow"))
    cam.start()
    assert wait_until(lambda: cam.status.state is ERROR)
    assert [c.api for c in world.captures_for(0)] == [DSHOW]
    monkeypatch.setattr(sys, "platform", "linux")
    cam = make_camera(cam_settings(backend="auto"))
    cam.start()
    assert wait_until(lambda: cam.status.state is ERROR)
    assert [c.api for c in world.captures_for(0)[1:]] == [ANY]


def test_unexpected_exception_sets_error_and_is_logged(world, make_camera, caplog):
    world.add(0, read_error=RuntimeError("driver exploded"))
    cam = make_camera()
    with caplog.at_level(logging.ERROR, logger="dragon_ball_powers.camera"):
        cam.start()
        assert wait_until(lambda: cam.status.state is ERROR)
    error = cam.status.error
    assert "unexpected error" in error and "driver exploded" in error
    assert any(record.exc_info for record in caplog.records)  # traceback in the log
    assert wait_until(lambda: world.captures_for(0)[0].released == 1)


def test_hanging_driver_is_reported_and_recovers(world, make_camera):
    gate = threading.Event()
    gate.set()
    device = world.add(0, read_gate=gate)
    cam = make_camera(hang_timeout=0.2)
    cam.start(0, name="USB Cam")
    assert wait_until(lambda: cam.status.state is RUNNING)
    gate.clear()
    assert wait_until(lambda: device.at_gate)
    assert wait_until(lambda: cam.status.state is ERROR)
    assert cam.status.error.startswith("Camera 0 (USB Cam) is not responding")
    assert cam.is_running  # derived view: the session is still alive
    gate.set()
    assert wait_until(lambda: cam.status.state is RUNNING and cam.status.error is None)


def test_negative_index_is_rejected_without_opening(world, make_camera):
    cam = make_camera()
    cam.start(-1)
    status = cam.status
    assert status.state is ERROR and "Invalid camera index -1" in status.error
    assert world.captures == []
    cam.stop()
    assert cam.status.state is STOPPED and cam.status.error is None


# ------------------------------------------------------------ lifecycle / waiting


def test_status_transitions_and_non_blocking_start(world, make_camera):
    gate = threading.Event()
    world.add(0, open_gate=gate)  # the fake driver blocks in the constructor
    cam = make_camera()
    assert cam.status.state is STOPPED and not cam.is_running
    started = time.perf_counter()
    cam.start(0, name="HP 5MP Camera")
    assert time.perf_counter() - started < 1.0  # opening happens in the capture thread
    status = cam.status
    assert status.state is STARTING and status.index == 0 and status.name == "HP 5MP Camera"
    assert cam.is_running and cam.latest() is None
    gate.set()
    assert wait_until(lambda: cam.status.state is RUNNING)
    cam.stop()
    assert cam.status.state is STOPPED


def test_status_is_a_snapshot(world, make_camera):
    world.add(0)
    cam = make_camera()
    cam.start()
    assert wait_until(lambda: cam.status.state is RUNNING)
    snapshot = cam.status
    snapshot.state, snapshot.error = ERROR, "tampered"
    assert cam.status.state is RUNNING and cam.status.error is None


def test_stop_is_idempotent_and_releases_device(world, make_camera):
    world.add(0)
    cam = make_camera()
    cam.stop()  # never started: harmless
    assert cam.status.state is STOPPED
    cam.start()
    assert wait_until(lambda: cam.status.state is RUNNING)
    thread = cam._run.thread
    assert thread.name == "camera-capture" and thread.daemon and thread.is_alive()
    cam.stop()
    cam.stop()
    assert world.captures_for(0)[0].released == 1
    assert not thread.is_alive()
    status = cam.status
    assert status.state is STOPPED and status.error is None and status.fps == 0.0
    assert cam.latest() is None and not cam.is_running


def test_wait_for_frame_semantics(world, make_camera):
    gate = threading.Event()
    gate.set()
    device = world.add(0, read_gate=gate, interval=0.01)
    cam = make_camera()
    started = time.perf_counter()
    assert cam.wait_for_frame(0, 2.0) is None  # nothing is capturing: prompt None
    assert time.perf_counter() - started < 1.0
    cam.start()
    first = cam.wait_for_frame(0, 3.0)  # blocks until the first frame exists
    assert first is not None and first.frame_id >= 1
    second = cam.wait_for_frame(first.frame_id, 3.0)
    assert second is not None and second.frame_id > first.frame_id

    gate.clear()  # freeze the stream
    assert wait_until(lambda: device.at_gate)
    frozen = cam.latest()
    assert cam.wait_for_frame(first.frame_id, 1.0) is frozen  # the newest, not the next one
    started = time.perf_counter()
    assert cam.wait_for_frame(frozen.frame_id, 0.25) is None  # timeout
    assert time.perf_counter() - started >= 0.2

    result: dict = {}
    waiter = threading.Thread(target=lambda: result.update(f=cam.wait_for_frame(frozen.frame_id, 3.0)))
    waiter.start()
    time.sleep(0.05)
    gate.set()  # a new frame wakes the blocked waiter
    waiter.join(3.0)
    assert result["f"] is not None and result["f"].frame_id > frozen.frame_id

    waiter = threading.Thread(target=lambda: result.update(
        g=cam.wait_for_frame(10**9, 5.0), t=time.perf_counter()))
    waiter.start()
    time.sleep(0.05)
    stop_called = time.perf_counter()
    cam.stop()  # wakes waiters promptly
    waiter.join(3.0)
    assert not waiter.is_alive() and result["g"] is None
    assert result["t"] - stop_called < 1.0
    started = time.perf_counter()
    assert cam.wait_for_frame(0, 5.0) is None  # after stop
    assert time.perf_counter() - started < 1.0


def test_restart_applies_settings_and_keeps_frame_ids_increasing(world, make_camera):
    world.add(0, honor_size=True)
    world.add(1, honor_size=True, marker=1)
    cam = make_camera(cam_settings())
    cam.start()
    first = cam.wait_for_frame(0, 3.0)
    assert first is not None and first.size == (W, H)

    cam.restart(cam_settings(width=80, height=60))  # same camera, new resolution
    frame = cam.wait_for_frame(first.frame_id, 3.0)
    assert frame is not None and frame.size == (80, 60) and frame.frame_id > first.frame_id
    caps = world.captures_for(0)
    assert len(caps) == 2 and caps[0].released == 1 and cam.settings.width == 80

    cam.restart(cam_settings(index=1))  # another camera selected through the settings
    assert frame_from(cam, marker=1) is not None and cam.status.index == 1

    cam.start(0)  # explicit index differs from settings.index (1)
    assert frame_from(cam, marker=0) is not None
    cam.restart()  # "same index unless given"
    assert frame_from(cam, marker=0) is not None and cam.status.index == 0
    cam.restart(index=1)
    assert frame_from(cam, marker=1) is not None


def test_stuck_thread_cannot_publish_into_new_session(world, make_camera, caplog):
    gate = threading.Event()
    gate.set()
    old = world.add(0, read_gate=gate)
    world.add(1, marker=1)
    cam = make_camera()
    cam.STOP_JOIN_TIMEOUT = 0.1
    cam.start(0)
    assert wait_until(lambda: cam.status.state is RUNNING)
    gate.clear()
    assert wait_until(lambda: old.at_gate)  # capture thread stuck in a driver call
    with caplog.at_level(logging.WARNING, logger="dragon_ball_powers.camera"):
        cam.start(1)
    assert "did not stop within" in caplog.text
    assert frame_from(cam, marker=1) is not None
    old_cap = world.captures_for(0)[0]
    gate.set()  # the stuck read returns a frame of the old device
    assert wait_until(lambda: old_cap.released == 1)
    frames = collect_frames(cam, 8)
    assert len(frames) == 8 and all(decode(f.image)[1] == 1 for f in frames)
    assert cam.status.index == 1 and cam.status.state is RUNNING


def test_concurrent_control_calls_are_safe(world, make_camera):
    world.add(0)
    cam = make_camera()
    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        try:
            for i in range(5):
                if (i + seed) % 2:
                    cam.start()
                else:
                    cam.stop()
                cam.status, cam.latest(), cam.wait_for_frame(0, 0.01)
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10.0)
    assert not errors
    cam.stop()
    assert cam.status.state is STOPPED and not cam.is_running
    assert wait_until(lambda: all(c.released == 1 for c in world.captures))


def test_restart_storm_leaks_no_threads_devices_or_registrations(world, make_camera):
    for index in range(3):
        world.add(index, marker=index, interval=0.002)
    cam = make_camera()
    before = set(threading.enumerate())
    errors: list[BaseException] = []

    def worker(seed: int) -> None:
        rnd = random.Random(seed)
        try:
            for _ in range(25):
                op = rnd.randrange(5)
                if op == 0:
                    cam.start(rnd.randrange(3))
                elif op == 1:
                    cam.stop()
                elif op == 2:
                    cam.restart(index=rnd.randrange(3))
                elif op == 3:
                    cam.restart()
                else:
                    cam.wait_for_frame(rnd.randrange(50), 0.01)
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20.0)
    assert not errors and not any(t.is_alive() for t in threads)
    cam.start(1)  # still fully usable afterwards
    assert frame_from(cam, marker=1) is not None
    cam.stop()
    assert cam.status.state is STOPPED and not cam.is_running and cam.latest() is None
    assert wait_until(lambda: not [t for t in threading.enumerate()
                                   if t not in before and t.name == "camera-capture"])
    assert world.captures and all(c.released == 1 for c in world.captures)  # each exactly once
    assert not any(index in camera._in_use for index in range(3))


def test_start_while_the_previous_open_is_blocked(world, make_camera):
    gate = threading.Event()
    world.add(0, open_gate=gate)  # the first driver call hangs in the constructor
    world.add(1, marker=1)
    cam = make_camera()
    cam.STOP_JOIN_TIMEOUT = 0.1
    cam.start(0)
    assert wait_until(lambda: len(world.captures_for(0)) == 1)  # stuck inside the driver
    cam.start(1)  # e.g. the user picked another camera meanwhile
    assert frame_from(cam, marker=1) is not None
    gate.set()  # the abandoned open completes late
    late = world.captures_for(0)[0]
    assert wait_until(lambda: late.released == 1)
    assert late.session_reads == 0  # never read, so it never published into the new session
    frames = collect_frames(cam, 5)
    assert len(frames) == 5 and all(decode(f.image)[1] == 1 for f in frames)
    assert cam.status.index == 1 and cam.status.state is RUNNING
    assert 0 not in camera._in_use


def test_stop_during_a_blocking_open(world, make_camera):
    gate = threading.Event()
    world.add(0, open_gate=gate)
    cam = make_camera()
    cam.STOP_JOIN_TIMEOUT = 0.1
    cam.start(0)
    assert wait_until(lambda: len(world.captures_for(0)) == 1)
    cam.stop()  # gives up joining after the timeout; the driver call is still blocked
    assert cam.status.state is STOPPED and not cam.is_running
    gate.set()
    late = world.captures_for(0)[0]
    assert wait_until(lambda: late.released == 1)  # released as soon as the call returned
    assert late.session_reads == 0 and cam.latest() is None
    assert cam.status.state is STOPPED and cam.status.error is None


def test_settings_edited_in_place_apply_on_restart(world, make_camera):
    world.add(0)
    settings = cam_settings(mirror=True)
    cam = make_camera(settings)
    cam.start()
    first = cam.wait_for_frame(0, 3.0)
    assert first is not None
    settings.mirror = False  # the settings panel edits the shared object
    settings.width = 99
    later = cam.wait_for_frame(first.frame_id, 3.0)
    n, _ = decode(later.image, mirrored=True)
    assert np.array_equal(later.image, make_frame(n)[:, ::-1])  # running session unaffected
    cam.restart()
    frame = cam.wait_for_frame(later.frame_id, 3.0)
    n, _ = decode(frame.image, mirrored=False)
    assert np.array_equal(frame.image, make_frame(n))  # the restart applied mirror=False
    assert (cv2.CAP_PROP_FRAME_WIDTH, 99) in world.captures_for(0)[-1].sets


def test_a_waiter_may_stop_the_camera_while_others_wait(world, make_camera):
    world.add(0, interval=0.01)
    cam = make_camera()
    cam.start()
    first = cam.wait_for_frame(0, 3.0)
    assert first is not None
    results: dict = {}

    def never_satisfied() -> None:
        results["other"] = cam.wait_for_frame(10**12, 5.0)

    def consumer_that_stops() -> None:
        results["frame"] = cam.wait_for_frame(first.frame_id, 3.0)
        cam.stop()  # e.g. a consumer shutting the pipeline down right after a frame
        results["stopped"] = True

    other = threading.Thread(target=never_satisfied)
    other.start()
    time.sleep(0.05)
    stopper = threading.Thread(target=consumer_that_stops)
    stopper.start()
    stopper.join(5.0)
    other.join(5.0)
    assert not stopper.is_alive() and not other.is_alive()  # no deadlock
    assert results["frame"] is not None and results["stopped"] and results["other"] is None
    assert cam.status.state is STOPPED


def test_wait_for_frame_tolerates_odd_arguments(world, make_camera):
    world.add(0)
    cam = make_camera()
    cam.start()
    assert cam.wait_for_frame(0, 3.0) is not None
    assert cam.wait_for_frame(None, 0.0) is not None            # None = any frame
    assert cam.wait_for_frame(float("-inf"), 0.0) is not None
    assert cam.wait_for_frame(float("nan"), 0.0) is not None    # garbage = any frame
    started = time.perf_counter()
    assert cam.wait_for_frame(float("inf"), 0.05) is None       # used to raise OverflowError
    # A bad timeout means "do not wait" - it used to discard after_id as well.
    for timeout in (None, float("nan"), -5, "soon"):
        assert cam.wait_for_frame(10**12, timeout) is None
    assert time.perf_counter() - started < 1.0


# --------------------------------------------------------------------- enumeration


def test_list_cameras_probes_until_two_consecutive_misses(world):
    for index in (0, 1, 3):
        world.add(index)
    cameras = camera.list_cameras(max_devices=8)
    assert [(c.index, c.name, c.is_ir, c.backend) for c in cameras] == [
        (0, "Camera 0", False, "auto"), (1, "Camera 1", False, "auto"),
        (3, "Camera 3", False, "auto")]
    assert [c.index for c in world.captures] == [0, 1, 2, 3, 4, 5]
    assert all(c.released == 1 for c in world.captures)


def test_list_cameras_without_devices_probes_every_index(world):
    assert camera.list_cameras(max_devices=5) == []
    assert [c.index for c in world.captures] == [0, 1, 2, 3, 4]
    assert camera.list_cameras(max_devices=0) == [] and len(world.captures) == 5  # no probe


def test_list_cameras_never_raises(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("broken driver")

    monkeypatch.setattr(camera.cv2, "VideoCapture", broken)
    assert camera.list_cameras(max_devices=3) == []
    monkeypatch.setattr(camera, "_probe_cameras", broken)
    assert camera.list_cameras() == []


def test_list_cameras_uses_pygrabber_names_and_flags_ir(world, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    install_fake_pygrabber(monkeypatch, ["HP 5MP Camera", "HP IR Camera"])
    assert camera.list_cameras() == [CameraInfo(0, "HP 5MP Camera", "auto", False),
                                     CameraInfo(1, "HP IR Camera", "auto", True)]
    assert world.captures == []  # names come from DirectShow: no device was opened


def test_pygrabber_initialises_com_only_in_worker_threads(world, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    install_fake_pygrabber(monkeypatch, ["HP 5MP Camera"])
    calls: list[tuple[str, str]] = []
    comtypes = types.ModuleType("comtypes")
    comtypes.CoInitialize = lambda: calls.append(("init", threading.current_thread().name))
    comtypes.CoUninitialize = lambda: calls.append(("uninit", threading.current_thread().name))
    monkeypatch.setitem(sys.modules, "comtypes", comtypes)
    result: dict = {}
    worker = threading.Thread(target=lambda: result.update(cams=camera.list_cameras()),
                              name="camera-scan")
    worker.start()
    worker.join(5.0)
    assert [c.name for c in result["cams"]] == ["HP 5MP Camera"]
    assert calls == [("init", "camera-scan"), ("uninit", "camera-scan")]
    calls.clear()
    assert [c.name for c in camera.list_cameras()] == ["HP 5MP Camera"]  # main thread
    assert calls == []

    def changed_mode():
        raise OSError("RPC_E_CHANGED_MODE")

    comtypes.CoInitialize = changed_mode  # COM already initialised differently: still works
    worker = threading.Thread(target=lambda: result.update(cams=camera.list_cameras()))
    worker.start()
    worker.join(5.0)
    assert [c.name for c in result["cams"]] == ["HP 5MP Camera"] and calls == []


def test_pygrabber_failure_or_absence_falls_back_to_probing(world, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    world.add(0)
    install_fake_pygrabber(monkeypatch, [], error=OSError("COM failure"))
    assert [(c.index, c.name) for c in camera.list_cameras(max_devices=3)] == [(0, "Camera 0")]
    assert {c.api for c in world.captures} == {DSHOW}
    monkeypatch.setitem(sys.modules, "pygrabber", None)  # not installed
    monkeypatch.delitem(sys.modules, "pygrabber.dshow_graph")
    assert [c.index for c in camera.list_cameras(max_devices=3)] == [0]


def test_dshow_names_are_not_used_for_other_backends(world, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    install_fake_pygrabber(monkeypatch, ["HP 5MP Camera"])
    world.add(0)
    cameras = camera.list_cameras(max_devices=2, backend="msmf")
    assert [(c.index, c.name, c.backend) for c in cameras] == [(0, "Camera 0", "msmf")]
    assert {c.api for c in world.captures} == {MSMF}


def test_probe_does_not_reopen_devices_this_process_uses(world, make_camera):
    world.add(0)
    world.add(1)
    cam = make_camera()
    cam.start(0, name="Studio Cam")
    assert wait_until(lambda: cam.status.state is RUNNING)
    opened_before = len(world.captures_for(0))
    cameras = camera.list_cameras(max_devices=4)
    assert [(c.index, c.name) for c in cameras] == [(0, "Studio Cam"), (1, "Camera 1")]
    assert len(world.captures_for(0)) == opened_before
    cam.stop()
    assert camera.list_cameras(max_devices=4)[0].name == "Camera 0"  # released: probed again


def test_probe_skips_a_device_this_process_is_still_opening(world, make_camera):
    # Regression: the device was registered only after the (slow) driver open returned, so a
    # rescan during the open opened it a second time - with DirectShow, releasing that second
    # handle stops the stream that is being started.
    gate = threading.Event()
    device = world.add(0, open_gate=gate)
    world.add(1)
    cam = make_camera()
    cam.start(0, name="Studio Cam")
    assert wait_until(lambda: len(world.captures_for(0)) == 1)  # inside the driver's open
    device.open_gate = None  # a second open would not block
    cameras = camera.list_cameras(max_devices=3)
    assert [(c.index, c.name) for c in cameras] == [(0, "Studio Cam"), (1, "Camera 1")]
    assert len(world.captures_for(0)) == 1  # not opened twice
    gate.set()
    assert wait_until(lambda: cam.status.state is RUNNING)
    cam.stop()
    assert 0 not in camera._in_use


def test_failed_open_leaves_no_registration(world, make_camera):
    cam = make_camera()
    cam.start(0, name="Ghost Cam")  # no such device
    assert wait_until(lambda: cam.status.state is ERROR)
    assert 0 not in camera._in_use  # unregistered before the ERROR became visible
    assert camera.list_cameras(max_devices=1) == []


@pytest.mark.parametrize("name, expected", [
    ("HP IR Camera", True), ("Integrated Infrared Camera", True), ("IR-Kamera", True),
    ("Camera (ir)", True), ("Infrarot-Kamera", True), ("HP 5MP Camera", False),
    ("Iriun Webcam", False), ("AIRcam", False), ("Mirror Cam", False), ("Camera 0", False),
])
def test_ir_name_detection(name, expected):
    assert camera._is_ir_name(name) is expected


def test_choose_camera_index_rules():
    def s(**kwargs) -> CameraSettings:
        return CameraSettings(**kwargs)

    hp = [CameraInfo(0, "HP 5MP Camera"), CameraInfo(1, "HP IR Camera", is_ir=True)]
    assert camera.choose_camera_index([], s(index=4)) == 4                 # nothing known
    assert camera.choose_camera_index(hp, s(index=0)) == 0                 # listed colour camera
    assert camera.choose_camera_index(hp, s(index=1)) == 0                 # IR -> colour camera
    assert camera.choose_camera_index(hp, s(index=5)) == 5                 # unlisted: keep
    only_ir = [CameraInfo(0, "HP IR Camera", is_ir=True)]
    assert camera.choose_camera_index(only_ir, s(index=0)) == 0            # no alternative

    moved = [CameraInfo(0, "OBS Virtual Camera"), CameraInfo(1, "HP 5MP Camera"),
             CameraInfo(2, "HP IR Camera", is_ir=True)]
    assert camera.choose_camera_index(moved, s(index=0, name_hint="hp 5mp")) == 1   # re-found
    assert camera.choose_camera_index(moved, s(index=2, name_hint="HP")) == 1       # colour wins
    assert camera.choose_camera_index(moved, s(index=0, name_hint="HP IR Camera")) == 2  # explicit
    assert camera.choose_camera_index(moved, s(index=0, name_hint="Logitech")) == 0  # not found
    twins = [CameraInfo(0, "USB Cam"), CameraInfo(1, "USB Cam")]
    assert camera.choose_camera_index(twins, s(index=1, name_hint="USB Cam")) == 1   # keep index


@pytest.mark.parametrize("name, expected", [
    ("OBS Virtual Camera", True), ("OBS-Camera", True), ("Unity Video Capture", True), ("Dragon Ball Powers", True),
    ("ManyCam Virtual Webcam", True), ("HP 5MP Camera", False), ("NVIDIA Broadcast", False),
    ("DroidCam Source 3", False), ("Logitech BRIO", False), ("Virtualis Cam", False), ("Camera 0", False),
])
def test_virtual_output_name_detection(name, expected):
    assert camera._is_virtual_name(name) is expected


def test_list_cameras_flags_virtual_outputs(world, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    install_fake_pygrabber(monkeypatch, ["OBS Virtual Camera", "HP 5MP Camera"])
    assert camera.list_cameras() == [CameraInfo(0, "OBS Virtual Camera", "auto", False, True),
                                     CameraInfo(1, "HP 5MP Camera", "auto", False, False)]


def test_choose_camera_index_never_picks_a_virtual_output_by_itself():
    # Installing OBS (for the video-call output) adds "OBS Virtual Camera", possibly at index 0.
    def s(**kwargs) -> CameraSettings:
        return CameraSettings(**kwargs)

    cams = [CameraInfo(0, "OBS Virtual Camera", is_virtual=True), CameraInfo(1, "HP 5MP Camera"),
            CameraInfo(2, "HP IR Camera", is_ir=True)]
    assert camera.choose_camera_index(cams, s(index=0)) == 1                               # the webcam
    assert camera.choose_camera_index(cams, s(index=0, name_hint="camera")) == 1           # webcam first
    assert camera.choose_camera_index(cams, s(index=1, name_hint="OBS Virtual Camera")) == 0  # explicit
    only_virtual = [CameraInfo(0, "OBS Virtual Camera", is_virtual=True)]
    assert camera.choose_camera_index(only_virtual, s(index=0)) == camera.NO_CAMERA        # webcam unplugged
    assert camera.choose_camera_index(only_virtual, s(index=0, name_hint="OBS Virtual Camera")) == 0  # picked
    ir_and_virtual = [CameraInfo(0, "HP IR Camera", is_ir=True), CameraInfo(1, "OBS Virtual Camera", is_virtual=True)]
    assert camera.choose_camera_index(ir_and_virtual, s(index=0)) == 0                     # as before for IR


# -------------------------------------------------------------------- test pattern


def test_synthetic_camera_generates_animated_frames():
    cam = camera.SyntheticCamera(CameraSettings(width=320, height=240, fps=60))
    try:
        assert cam.status.state is STOPPED
        cam.start()
        frames = collect_frames(cam, 3)
        assert len(frames) == 3
        for f in frames:
            assert f.image.shape == (240, 320, 3) and f.image.dtype == np.uint8
            assert f.size == (320, 240)
        assert frames[0].frame_id < frames[1].frame_id < frames[2].frame_id
        assert not np.array_equal(frames[0].image, frames[2].image)  # animated
        assert frames[0].image.max() == 255 and np.median(frames[0].image) < 90  # bright on dark
        status = cam.status
        assert status.state is RUNNING and status.name == camera.TEST_PATTERN_NAME == "Test pattern"
        assert status.backend == camera.SYNTHETIC_BACKEND == "synthetic"
        assert status.resolution == (320, 240) and status.index == -1 and cam.is_running
        assert any(t.name == "camera-test-pattern" for t in threading.enumerate())

        cam.restart(CameraSettings(width=160, height=120, fps=30))
        frame = cam.wait_for_frame(frames[-1].frame_id, 3.0)
        assert frame is not None and frame.image.shape == (120, 160, 3)
    finally:
        cam.stop()
    assert cam.status.state is STOPPED and cam.latest() is None
    started = time.perf_counter()
    assert cam.wait_for_frame(0, 5.0) is None
    assert time.perf_counter() - started < 1.0


def test_synthetic_camera_paces_frames_at_configured_fps():
    cam = camera.SyntheticCamera(CameraSettings(width=320, height=240, fps=50))
    try:
        cam.start()
        assert cam.wait_for_frame(0, 3.0) is not None
        time.sleep(0.6)
        status = cam.status
    finally:
        cam.stop()
    # Loose bounds: the machine is shared with other test runs.
    assert 15.0 < status.fps < 80.0
    assert 10 <= status.frames <= 60


# ------------------------------------------------------- retry and internal helpers


def test_restart_after_error_retries_the_device(world, make_camera):
    cam = make_camera()
    cam.start(0, name="USB Cam")
    assert wait_until(lambda: cam.status.state is ERROR)  # not plugged in yet
    world.add(0)
    cam.restart()  # e.g. the user plugged the camera in and pressed retry
    assert wait_until(lambda: cam.status.state is RUNNING)
    assert cam.status.name == "USB Cam" and cam.status.error is None


def test_named_missing_device_message(world, make_camera, monkeypatch):
    monkeypatch.setattr(camera, "_device_exists", lambda index, backend: False)
    cam = make_camera(cam_settings(index=2))
    cam.start(name="Logitech C920")
    assert wait_until(lambda: cam.status.state is ERROR)
    assert cam.status.error == ("Camera 2 (Logitech C920) is not connected. Plug it in or select "
                                "another camera in Settings (Tab).")


def test_device_exists_uses_directshow_names_on_windows(monkeypatch):
    assert camera._device_exists(0, "auto") is Path("/dev/video0").exists()  # Linux check
    monkeypatch.setattr(sys, "platform", "win32")
    install_fake_pygrabber(monkeypatch, ["HP 5MP Camera", "HP IR Camera"])
    assert camera._device_exists(1, "auto") is True
    assert camera._device_exists(2, "dshow") is False
    assert camera._device_exists(0, "msmf") is None  # MSMF numbering is unknown
    monkeypatch.setitem(sys.modules, "pygrabber", None)  # pygrabber not installed
    monkeypatch.delitem(sys.modules, "pygrabber.dshow_graph")
    assert camera._device_exists(0, "auto") is None


def test_probing_restores_the_opencv_log_level(world):
    before = cv2.getLogLevel()
    camera.list_cameras(max_devices=2)
    assert cv2.getLogLevel() == before
    with camera._quiet_opencv():
        with camera._quiet_opencv():  # nested / concurrent probes share the saved level
            assert cv2.getLogLevel() <= 1
        assert cv2.getLogLevel() <= 1
    assert cv2.getLogLevel() == before


def test_prepare_frame_rejects_garbage_and_normalises_formats():
    assert camera._prepare_frame(None, True) is None
    assert camera._prepare_frame(np.zeros((0, 4, 3), np.uint8), True) is None
    assert camera._prepare_frame(np.zeros((1, 5000), np.uint8), False) is None  # raw MJPG bytes
    assert camera._prepare_frame(np.zeros((4, 4, 2), np.uint8), False) is None
    assert camera._prepare_frame(np.zeros((2, 4, 4, 3), np.uint8), False) is None
    deep = np.full((4, 6), 0xFF00, np.uint16)  # 16-bit grey (e.g. IR sensors)
    out = camera._prepare_frame(deep, False)
    assert out.shape == (4, 6, 3) and out.dtype == np.uint8 and (out == 255).all()
    src = make_frame(7)
    assert np.array_equal(camera._prepare_frame(src, True), src[:, ::-1])
    copy = camera._prepare_frame(src, False)
    assert np.array_equal(copy, src) and not np.shares_memory(copy, src)


def test_fourcc_and_backend_labels():
    assert camera._fourcc_to_str(float(MJPG)) == "MJPG"
    assert camera._fourcc_to_str(0.0) == "unknown"
    assert camera._fourcc_to_str(float(0x01020304)).startswith("0x")
    assert camera._api_label(DSHOW) == "DSHOW" and camera._api_label(MSMF) == "MSMF"
    assert camera._api_label(ANY) == "ANY" and camera._api_label(int(cv2.CAP_V4L2)) == "V4L2"
