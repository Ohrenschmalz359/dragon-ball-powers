"""Virtual camera output: the effects picture as a webcam for video-call apps.

Discord, Zoom, Microsoft Teams, Google Meet and other browser calls (also FaceTime links)
... accept a webcam. This module offers them the composited picture (camera image, energy
effects and callouts, without the HUD) through a *virtual camera* device, so the effects
work in all of them without app-specific integrations. (Apps that refuse virtual cameras,
e.g. WhatsApp Desktop, can still get the app window through their screen sharing.)

The device comes from `pyvirtualcam <https://github.com/letmaik/pyvirtualcam>`_ (an
optional dependency: the app runs without it) plus a driver:

* Windows: ``install_virtual_camera.bat`` registers Unity Capture's small DirectShow
  filter as the camera "Dragon Ball Powers" (no OBS needed). Alternatively OBS Studio's
  "OBS Virtual Camera" (OBS itself need not run; its own "Start Virtual Camera" must be
  off - one sender at a time).
* macOS: OBS Studio 30+ (its camera system extension, activated once in OBS).
* Linux: v4l2loopback (``sudo modprobe v4l2loopback devices=1``).

Threading: the main thread renders a picture only when :meth:`VirtualCamera.frame_due`
says so and hands it over with :meth:`VirtualCamera.submit`, which never blocks. A sender
thread opens the device (that can take a moment), sends the newest picture and repeats
the last one while the app is busy, so call apps always receive a steady stream. Problems
end up in :attr:`VirtualCamera.status` as a message the user can act on.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from dragon_ball_powers.config import SETUP_SCRIPT, VirtualCameraSettings
from dragon_ball_powers.datatypes import VirtualCameraState, VirtualCameraStatus

log = logging.getLogger(__name__)

#: The last picture is sent again when no new one arrived for this many frame intervals.
REPEAT_AFTER_FRAMES = 2.5
#: A frame counts as due this much (s) before its slot: absorbs display-refresh jitter.
DUE_TOLERANCE = 0.002
#: stop() waits this long (s) for the sender thread.
STOP_TIMEOUT = 3.0
#: A busy device is retried this long (s) before the error is reported: after a restart a
#: call app holds the previous sender's OBS shared memory until its next frame tick (~33 ms).
OPEN_RETRY_SECONDS = 1.5
OPEN_RETRY_INTERVAL = 0.1

INSTALL_HINT = ("No virtual camera driver found. Run install_virtual_camera.bat once (it adds the camera "
                "'Dragon Ball Powers'), or install OBS Studio.")
MAC_HINT = ("No virtual camera found. Install OBS Studio 30 or newer, click 'Start Virtual Camera' once, "
            "allow the system extension in System Settings, then quit OBS.")
IN_USE_HINT = ("The virtual camera is busy. Stop 'Virtual Camera' in OBS if it runs, or switch the camera "
               "off and on in your call app, then press V again.")
DEVICE_HINT = ("The virtual camera device named in the settings (virtual_camera.device) does not exist. "
               "Leave it empty to use the default device.")
NOT_INSTALLED_HINT = f"The pyvirtualcam package is missing. Run {SETUP_SCRIPT} again to install it."
RUNTIME_HINT = ("The virtual camera module could not be loaded ({detail}). Install OBS Studio (it brings "
                "the Microsoft Visual C++ runtime) or the Microsoft Visual C++ Redistributable.")
LINUX_HINT = "No v4l2loopback device found. Load one with: sudo modprobe v4l2loopback devices=1"

#: ``factory(width=, height=, fps=, backend=, device=)`` -> an object with ``send(frame)``,
#: ``close()``, ``device`` and ``backend`` (a ``pyvirtualcam.Camera`` by default).
CameraFactory = Callable[..., Any]

#: pyvirtualcam reports one "'<backend>' backend: <message>" line per driver it tried.
_BACKEND_PREFIX = re.compile(r"^'[^']*' backend:\s*")


def _pyvirtualcam_camera(**kwargs: Any) -> Any:
    import pyvirtualcam  # optional dependency, imported on first use

    return pyvirtualcam.Camera(**kwargs)


def _error_lines(exc: BaseException) -> list[str]:
    return [_BACKEND_PREFIX.sub("", line).strip() for line in str(exc).splitlines() if line.strip()]


def _busy(line: str) -> bool:
    low = line.lower()
    return "already in use" in low or "could not be started" in low


def is_busy_error(exc: BaseException) -> bool:
    """True when a driver exists but its device is in use (possibly only for a moment)."""
    return not isinstance(exc, (ImportError, KeyError)) and any(_busy(line) for line in _error_lines(exc))


def friendly_error(exc: BaseException) -> str:
    """Turn an exception from opening the virtual camera into an actionable message."""
    if isinstance(exc, ModuleNotFoundError):
        return NOT_INSTALLED_HINT
    lines = _error_lines(exc)
    first = lines[0] if lines else type(exc).__name__
    if isinstance(exc, ImportError):         # e.g. "DLL load failed": installed but not loadable
        return RUNTIME_HINT.format(detail=first)
    if isinstance(exc, KeyError):            # pyvirtualcam: unknown backend on this platform
        return f"The virtual camera driver {exc} is not available on this computer; use 'auto'."
    lows = [line.lower() for line in lines]
    # With backend "auto" there is one line per driver. "Busy" wins: a driver is installed.
    if any(_busy(low) for low in lows):
        return IN_USE_HINT
    if any("supports only the" in low or "registered with this name" in low for low in lows):
        return DEVICE_HINT
    if any("no v4l2 loopback device" in low or "modprobe" in low for low in lows):
        return LINUX_HINT
    if any("not installed in your system" in low for low in lows):     # macOS: OBS extension missing
        return MAC_HINT
    if any("did you install" in low or "no camera registered" in low for low in lows):
        return INSTALL_HINT
    return f"Virtual camera unavailable: {first}"


def device_settings(settings: VirtualCameraSettings) -> tuple[Any, ...]:
    """The settings whose change needs the device to be re-opened (not mirror / enabled)."""
    return (even_size(settings), int(settings.fps), settings.backend, settings.device.strip())


def _open_kwargs(settings: VirtualCameraSettings, size: tuple[int, int]) -> dict[str, Any]:
    return {"width": size[0], "height": size[1], "fps": max(1, int(settings.fps)),
            "backend": None if settings.backend == "auto" else settings.backend,
            "device": settings.device.strip() or None}


def even_size(settings: VirtualCameraSettings) -> tuple[int, int]:
    """Output size (both even: virtual cameras transport YUV 4:2:0)."""
    w, h = max(2, int(settings.width)), max(2, int(settings.height))
    return (w - w % 2, h - h % 2)


def probe(settings: VirtualCameraSettings | None = None,
          camera_factory: CameraFactory | None = None) -> tuple[bool, str]:
    """Open and close the virtual camera once (``--check``). Returns ``(ok, detail)``."""
    s = settings or VirtualCameraSettings()
    factory = camera_factory or _pyvirtualcam_camera
    try:
        cam = factory(**_open_kwargs(s, even_size(s)))
    except Exception as exc:  # noqa: BLE001 - every driver problem is reported, never raised
        log.debug("Virtual camera probe failed: %s: %s", type(exc).__name__, exc)
        return False, friendly_error(exc)
    try:
        version = getattr(sys.modules.get("pyvirtualcam"), "__version__", "")
        detail = f"'{getattr(cam, 'device', '?')}' via {getattr(cam, 'backend', '?')}"
        return True, detail + (f", pyvirtualcam {version}" if version else "")
    finally:
        try:
            cam.close()
        except Exception as exc:  # noqa: BLE001
            log.debug("Closing the virtual camera probe failed: %s", exc)


class VirtualCamera:
    """Sends rendered pictures to a virtual webcam (see the module docstring).

    ``submit`` expects (H, W, 3) uint8 RGB pictures, top row first, of :attr:`size`
    (other sizes are scaled). The picture is sent as it is: mirroring is up to the
    renderer (``Renderer.mirror_output``) so that callout text stays readable.
    """

    def __init__(self, settings: VirtualCameraSettings, camera_factory: CameraFactory | None = None) -> None:
        self._settings = dataclasses.replace(settings)
        self._factory = camera_factory or _pyvirtualcam_camera
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._frame: np.ndarray | None = None      # newest picture, not yet sent
        self._status = VirtualCameraStatus()
        self._next_due = 0.0
        self._bad_frame_logged = False
        self._last_backend: str | None = None       # driver of the last successful open

    # ------------------------------------------------------------------ public API

    @property
    def settings(self) -> VirtualCameraSettings:
        """The settings of the current (or last) run (a private copy)."""
        return self._settings

    @property
    def size(self) -> tuple[int, int]:
        return even_size(self._settings)

    @property
    def status(self) -> VirtualCameraStatus:
        """Thread-safe snapshot."""
        with self._lock:
            return dataclasses.replace(self._status)

    def start(self, settings: VirtualCameraSettings | None = None) -> None:
        """Open the device in the background. With ``settings`` a running output is
        restarted with them; without, a no-op while starting or running."""
        if settings is not None:
            self.stop()
            self._settings = dataclasses.replace(settings)
        with self._lock:
            if self._thread is not None and self._status.active:
                return
            stop = threading.Event()
            self._stop_event = stop
            self._frame = None
            self._next_due = 0.0
            self._status = VirtualCameraStatus(VirtualCameraState.STARTING, resolution=self.size,
                                               fps=float(self._settings.fps))
            self._thread = threading.Thread(target=self._run, args=(stop,), name="virtual-camera", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Close the device and end the sender thread. Idempotent; the status becomes OFF."""
        with self._wake:
            thread, stop = self._thread, self._stop_event
            self._thread = self._stop_event = None
            if stop is not None:
                stop.set()
            self._wake.notify_all()
        if thread is not None and thread is not threading.current_thread():
            thread.join(STOP_TIMEOUT)
            if thread.is_alive():
                log.warning("The virtual camera thread did not stop within %.0f s", STOP_TIMEOUT)
        with self._lock:
            self._frame = None
            self._status = VirtualCameraStatus()

    def frame_due(self, now: float) -> bool:
        """True when the main loop should render the next picture (paced to ``fps``)."""
        with self._lock:
            if self._status.state is not VirtualCameraState.RUNNING:
                return False
        if now + DUE_TOLERANCE < self._next_due:
            return False
        interval = 1.0 / max(1, int(self._settings.fps))
        # Whole intervals keep the average rate exact; after a stall restart from now.
        behind = now - self._next_due
        self._next_due = (self._next_due if behind < interval else now) + interval
        return True

    def submit(self, frame: np.ndarray) -> None:
        """Hand over the next picture. Never blocks; ignored unless running."""
        with self._wake:
            if self._thread is None or self._status.state is not VirtualCameraState.RUNNING:
                return
            self._frame = frame
            self._wake.notify_all()

    # ------------------------------------------------------------ sender thread

    def _open(self, stop: threading.Event, settings: VirtualCameraSettings, size: tuple[int, int]) -> Any:
        """Open the device; a busy one is retried for a moment. None when stopped or failed."""
        kwargs = _open_kwargs(settings, size)
        pinned = kwargs
        if kwargs["backend"] is None and self._last_backend:
            # Re-open the driver of the last run: with "auto", a briefly busy OBS camera would
            # otherwise silently fall through to another driver the call app is not watching.
            pinned = dict(kwargs, backend=self._last_backend)
        deadline = time.monotonic() + OPEN_RETRY_SECONDS
        while True:
            attempt = pinned if time.monotonic() < deadline else kwargs
            try:
                return self._factory(**attempt)
            except Exception as exc:  # noqa: BLE001 - every driver problem becomes a status message
                if is_busy_error(exc) and attempt is pinned and time.monotonic() < deadline:
                    if stop.wait(OPEN_RETRY_INTERVAL):
                        return None
                    continue
                if attempt is pinned and pinned is not kwargs:
                    pinned = kwargs              # the old driver is gone: let "auto" choose
                    continue
                message = friendly_error(exc)
                log.warning("Virtual camera unavailable: %s (%s: %s)", message, type(exc).__name__,
                            str(exc).replace("\n", " | "))
                self._set_error(stop, message)
                return None

    def _run(self, stop: threading.Event) -> None:
        settings, size = self._settings, self.size
        fps = max(1, int(settings.fps))
        cam = None
        try:
            cam = self._open(stop, settings, size)
            if cam is None:
                return
            device = str(getattr(cam, "device", "") or "")
            backend = str(getattr(cam, "backend", "") or "")
            self._last_backend = backend or None
            with self._lock:
                if stop.is_set():
                    return
                self._status = VirtualCameraStatus(VirtualCameraState.RUNNING, device, backend, size, float(fps))
            log.info("Virtual camera started: '%s' via %s, %dx%d @ %d fps", device, backend, size[0], size[1], fps)
            self._send_loop(cam, stop, size, fps)
        except Exception as exc:  # noqa: BLE001 - e.g. the driver went away while sending
            log.error("Virtual camera failed while sending", exc_info=True)
            self._set_error(stop, f"Virtual camera stopped: {exc}")
        finally:
            if cam is not None:
                try:
                    cam.close()
                except Exception as exc:  # noqa: BLE001
                    log.debug("Closing the virtual camera failed: %s", exc)
                log.info("Virtual camera closed")

    def _send_loop(self, cam: Any, stop: threading.Event, size: tuple[int, int], fps: int) -> None:
        w, h = size
        repeat_after = REPEAT_AFTER_FRAMES / fps
        last = np.zeros((h, w, 3), np.uint8)      # black until the first picture arrives
        sent = 0
        while True:
            cam.send(last)
            sent += 1
            with self._wake:
                if not stop.is_set():
                    self._status.frames = sent
                if self._frame is None and not stop.is_set():
                    self._wake.wait(repeat_after)
                if stop.is_set():
                    return
                frame, self._frame = self._frame, None
            if frame is not None:
                fitted = self._fit(frame, w, h)
                if fitted is not None:
                    last = fitted

    def _fit(self, frame: np.ndarray, w: int, h: int) -> np.ndarray | None:
        """``frame`` as contiguous (h, w, 3) uint8 RGB, or None if it is unusable."""
        img = np.asarray(frame)
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] not in (3, 4) or img.size == 0:
            if not self._bad_frame_logged:
                self._bad_frame_logged = True
                log.warning("Virtual camera: unsupported picture %s %s skipped", img.dtype, img.shape)
            return None
        if img.shape[2] == 4:
            img = img[:, :, :3]
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img)

    def _set_error(self, stop: threading.Event, message: str) -> None:
        with self._lock:
            if stop.is_set():                     # stopped meanwhile: stop() owns the status
                return
            self._status = VirtualCameraStatus(VirtualCameraState.ERROR, resolution=self.size,
                                               fps=float(self._settings.fps), error=message)


__all__ = ["VirtualCamera", "friendly_error", "is_busy_error", "probe", "even_size", "device_settings",
           "INSTALL_HINT", "MAC_HINT", "IN_USE_HINT", "DEVICE_HINT", "NOT_INSTALLED_HINT", "RUNTIME_HINT",
           "LINUX_HINT"]
