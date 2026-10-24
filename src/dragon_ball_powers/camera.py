"""Camera capture thread and camera device enumeration.

:class:`CameraManager` owns one OpenCV ``VideoCapture`` in a background thread
(``"camera-capture"``) and publishes immutable :class:`CameraFrame` snapshots: BGR
uint8, mirrored when ``CameraSettings.mirror`` is set, and never aliasing a buffer
that is reused later. Consumers either poll :meth:`CameraManager.latest` (render loop)
or block in :meth:`CameraManager.wait_for_frame` (tracker thread, ``FrameSource``).

Opening a camera is slow (0.3-2 s on Windows) and drivers fail in many ways, so every
device call runs in the capture thread and every failure ends in a
:class:`CameraStatus` with an actionable ``error`` text instead of an exception:

* the device cannot be opened (busy in another app, privacy settings, unplugged),
* it opens but delivers no frame within ``open_timeout``,
* it stops delivering frames for ``failure_timeout`` (one reopen attempt, then ERROR),
* a single driver call blocks longer than ``hang_timeout`` ("not responding").

Windows: DirectShow (``CAP_DSHOW``) is the default backend because it opens quickly
and honours MJPG / resolution requests; with ``backend="auto"`` Media Foundation
(``CAP_MSMF``) and ``CAP_ANY`` are tried when DirectShow cannot open the device.
Device names come from DirectShow through the optional ``pygrabber`` package; its
device order equals OpenCV's DSHOW indices.

:class:`SyntheticCamera` has the same interface and renders an animated diagnostic
test pattern. It exists only for automated tests and ``--test-pattern`` diagnostics
and is never used as a stand-in for a missing camera.
"""

from __future__ import annotations

import os

# Media Foundation enumerates hardware (DXVA) video transforms when it opens a camera,
# which can delay the first frame by 5-30 s on some laptops. Software transforms are
# plenty fast for webcam resolutions. OpenCV reads this variable when its MSMF backend
# opens a device, so it has to be set before cv2 is used; ``setdefault`` keeps a value
# the user configured explicitly.
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import contextlib  # noqa: E402  (all imports intentionally follow the setting above)
import dataclasses  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from collections.abc import Iterator  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dragon_ball_powers.config import CameraSettings  # noqa: E402
from dragon_ball_powers.datatypes import CameraFrame, CameraInfo, CameraState, CameraStatus  # noqa: E402

log = logging.getLogger(__name__)

__all__ = [
    "NO_CAMERA",
    "SYNTHETIC_BACKEND",
    "TEST_PATTERN_NAME",
    "CameraManager",
    "SyntheticCamera",
    "backend_id",
    "choose_camera_index",
    "list_cameras",
]

#: ``CameraStatus.name`` / ``CameraStatus.backend`` of the :class:`SyntheticCamera`.
TEST_PATTERN_NAME = "Test pattern"
SYNTHETIC_BACKEND = "synthetic"
#: :func:`choose_camera_index` result when there is nothing suitable to open automatically.
NO_CAMERA = -1

_BACKEND_NAMES: tuple[str, ...] = ("auto", "dshow", "msmf", "any")
_IR_NAME = re.compile(r"\bIR\b", re.IGNORECASE)
# Virtual camera outputs: OBS, Unity Capture - also as "Dragon Ball Powers", the device
# install_virtual_camera.bat registers for this app, and "Dragon Ball Powers (Windows 11)"
# from install_win11_camera.bat -, ManyCam ...
_VIRTUAL_NAME = re.compile(r"\bvirtual\b|\bobs[- ]cam(era)?\b|unity video capture|dragon ball powers",
                           re.IGNORECASE)
_GENERIC_NAME = re.compile(r"Camera \d+")

_FPS_SMOOTHING = 0.1             # EMA factor for the measured frame interval
_IDLE_WAIT = 0.05                # wait_for_frame() pause while nothing captures (anti-spin)
_READ_RETRY_DELAY = 0.005        # pause after a failed read: a dead device must not spin a core
_FIRST_FRAME_RETRY_DELAY = 0.01
_MAX_CONDITION_WAIT = 3600.0     # Condition.wait() rejects huge timeouts (OverflowError)
_CV_LOG_LEVEL_FATAL = 1          # cv2.utils.logging.LOG_LEVEL_FATAL (not exposed by every build)
_SELECT_OTHER = "select another camera in Settings (Tab)"
_IR_HINT = ("This is an infrared (Windows Hello) camera that cannot be used for tracking. Select "
            "your normal camera in Settings (Tab).")
_RETRY_HINT = ("Select the camera again in Settings (Tab) or choose another one; details are in "
               "the log file.")


# =========================================================================== helpers


def _is_windows() -> bool:
    # Evaluated on every call (not cached) so tests can simulate Windows.
    return sys.platform == "win32"


def _normalize_backend(name: str) -> str:
    key = str(name or "auto").strip().lower()
    if key not in _BACKEND_NAMES:
        log.warning("Unknown camera backend %r; using 'auto'", name)
        return "auto"
    return key


def backend_id(name: str) -> int:
    """OpenCV ``CAP_*`` API id for a ``CameraSettings.backend`` value.

    ``"dshow"`` -> ``CAP_DSHOW``, ``"msmf"`` -> ``CAP_MSMF``, ``"any"`` -> ``CAP_ANY``;
    ``"auto"`` (and unknown names) -> ``CAP_DSHOW`` on Windows, ``CAP_ANY`` elsewhere.
    """
    key = _normalize_backend(name)
    if key == "dshow":
        return int(cv2.CAP_DSHOW)
    if key == "msmf":
        return int(cv2.CAP_MSMF)
    if key == "any":
        return int(cv2.CAP_ANY)
    return int(cv2.CAP_DSHOW) if _is_windows() else int(cv2.CAP_ANY)


def _backend_candidates(name: str) -> list[int]:
    """APIs to try in order. Only ``auto`` on Windows falls back to other backends."""
    key = _normalize_backend(name)
    first = backend_id(key)
    if key == "auto" and _is_windows():
        return [first, int(cv2.CAP_MSMF), int(cv2.CAP_ANY)]
    return [first]


def _api_label(api: int) -> str:
    labels = {int(cv2.CAP_DSHOW): "DSHOW", int(cv2.CAP_MSMF): "MSMF", int(cv2.CAP_ANY): "ANY"}
    label = labels.get(int(api))
    if label is not None:
        return label
    try:
        return str(cv2.videoio_registry.getBackendName(int(api)))
    except Exception:  # noqa: BLE001 - purely cosmetic
        return f"API {api}"


def _is_ir_name(name: str) -> bool:
    """Infrared (Windows Hello) cameras deliver grey IR images that are useless for tracking."""
    lowered = name.lower()
    # "infrarot" covers German driver names ("IR-Kamera" is matched by the regex).
    return bool(_IR_NAME.search(name)) or "infrared" in lowered or "infrarot" in lowered


def _is_virtual_name(name: str) -> bool:
    """Virtual camera outputs show another program's picture (or this app's own), not you."""
    return bool(_VIRTUAL_NAME.search(name))


def _usable(camera: CameraInfo) -> bool:
    """A real colour webcam: neither infrared nor a virtual camera output."""
    return not camera.is_ir and not camera.is_virtual


def _is_named(name: str) -> bool:
    """True for a real device name (not empty and not a generic "Camera N")."""
    return bool(name) and _GENERIC_NAME.fullmatch(name) is None


def _camera_ref(index: int, name: str) -> str:
    """'Camera 0 (HP 5MP Camera)' or 'Camera 0' when no real name is known."""
    return f"Camera {index} ({name})" if _is_named(name) else f"Camera {index}"


def _access_hint() -> str:
    """How to free a busy / blocked camera (no trailing period)."""
    if _is_windows():
        return ("Close other apps that use the camera (Teams, Zoom, OBS, browser) and allow camera "
                "access in Windows Settings > Privacy & security > Camera > Let desktop apps "
                "access your camera")
    if sys.platform == "darwin":
        return ("Close other apps that use the camera and allow camera access for Terminal (or the app "
                "you started from) in System Settings > Privacy & Security > Camera, then start the "
                "camera again")
    return ("Close other apps that use the camera and make sure this user is allowed to access "
            "the camera device")


def _safe_get(cap: Any, prop: int) -> float:
    try:
        value = float(cap.get(prop))
    except Exception:  # noqa: BLE001 - drivers may reject any property
        return 0.0
    return value if math.isfinite(value) else 0.0


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "unknown"
    chars = bytes((code >> (8 * i)) & 0xFF for i in range(4))
    text = chars.decode("ascii", errors="replace")
    return text if text.isprintable() else f"0x{code:08x}"


def _capture_backend_name(cap: Any, api: int) -> str:
    try:
        name = str(cap.getBackendName())
        if name:
            return name
    except Exception:  # noqa: BLE001 - getBackendName() raises for some captures
        pass
    return _api_label(api)


def _safe_release(cap: Any) -> None:
    if cap is None:
        return
    try:
        cap.release()
    except Exception as exc:  # noqa: BLE001
        log.debug("VideoCapture.release() failed: %s", exc)


def _as_frame_id(value: Any) -> int:
    """``wait_for_frame``'s ``after_id`` as an int (None / garbage -> -1 = any frame)."""
    if value is None:
        return -1
    try:
        return int(value)
    except OverflowError:  # +-inf: "nothing is newer" / "everything is newer"
        return sys.maxsize if value > 0 else -1
    except (TypeError, ValueError):
        return -1


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:  # e.g. raw IR / depth sensors
        return (img >> 8).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _prepare_frame(img: Any, mirror: bool) -> np.ndarray | None:
    """Validate a ``VideoCapture.read()`` image and turn it into a private BGR uint8 frame.

    Returns ``None`` for unusable data. The result never shares memory with ``img``, so
    a backend (or caller) that reuses its read buffer cannot modify published frames.
    """
    if not isinstance(img, np.ndarray) or img.size == 0:
        return None
    if img.ndim == 2:
        channels = 1
    elif img.ndim == 3:
        channels = int(img.shape[2])
    else:
        return None
    # A (1, N) array is an undecoded byte stream (raw MJPG), not an image.
    if img.shape[0] < 2 or img.shape[1] < 2 or channels not in (1, 3, 4):
        return None
    private = False
    if img.dtype != np.uint8:
        img = _to_uint8(img)
        private = True
    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        private = True
    elif channels == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        private = True
    if mirror:
        return cv2.flip(img, 1)  # always writes a new array
    if private:
        return np.ascontiguousarray(img)
    return np.array(img, order="C", copy=True)


# ---------------------------------------------------- devices opened by this process

_in_use_lock = threading.Lock()
_in_use: dict[int, list[str]] = {}


def _register_in_use(index: int, name: str) -> None:
    with _in_use_lock:
        _in_use.setdefault(index, []).append(name)


def _unregister_in_use(index: int, name: str) -> None:
    with _in_use_lock:
        names = _in_use.get(index)
        if not names:
            return
        try:
            names.remove(name)
        except ValueError:
            names.pop()
        if not names:
            del _in_use[index]


def _in_use_name(index: int) -> str | None:
    """Name of a device this process is opening or has open at ``index`` ('' if unnamed)."""
    with _in_use_lock:
        names = _in_use.get(index)
        if not names:
            return None
        return next((n for n in reversed(names) if n), "")


# ------------------------------------------------------------------ OpenCV logging

_cv_log_lock = threading.Lock()
_cv_log_users = 0
_cv_log_saved: int | None = None


@contextlib.contextmanager
def _quiet_opencv() -> Iterator[None]:
    """Hide OpenCV's console warnings while probing indices that are expected to fail.

    The OpenCV log level is process-global, so concurrent probes share one saved level.
    """
    global _cv_log_users, _cv_log_saved
    get_level = getattr(cv2, "getLogLevel", None)
    set_level = getattr(cv2, "setLogLevel", None)
    if get_level is None or set_level is None:
        yield
        return
    with _cv_log_lock:
        if _cv_log_users == 0:
            try:
                _cv_log_saved = int(get_level())
                set_level(min(_cv_log_saved, _CV_LOG_LEVEL_FATAL))
            except Exception:  # noqa: BLE001 - cosmetic only
                _cv_log_saved = None
        _cv_log_users += 1
    try:
        yield
    finally:
        with _cv_log_lock:
            _cv_log_users -= 1
            if _cv_log_users == 0 and _cv_log_saved is not None:
                with contextlib.suppress(Exception):
                    set_level(_cv_log_saved)
                _cv_log_saved = None


# =========================================================================== enumeration


def _dshow_device_names() -> list[str] | None:
    """DirectShow video input names in DirectShow order (= OpenCV ``CAP_DSHOW`` indices).

    Uses the optional Windows-only ``pygrabber`` package. Returns ``None`` when it is not
    installed or the enumeration failed (an empty list means "no cameras"). Names are
    stripped: some drivers pad their friendly names with spaces, and every name this
    module hands out or compares (``CameraInfo.name``, ``start(name=...)``) is stripped.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception as exc:  # noqa: BLE001 - ImportError, or COM errors at import time
        log.debug("pygrabber is not available (%s)", exc)
        return None
    comtypes_module: Any = None
    if threading.current_thread() is not threading.main_thread():
        # comtypes initialises COM only in the thread that first imports it; every other
        # thread needs its own apartment before it may create COM objects.
        try:
            import comtypes

            comtypes.CoInitialize()
            comtypes_module = comtypes
        except Exception as exc:  # noqa: BLE001
            # e.g. RPC_E_CHANGED_MODE: COM is already initialised (multithreaded) in this
            # thread, which works for device enumeration too.
            log.debug("CoInitialize failed (%s); trying the enumeration anyway", exc)
    try:
        names = FilterGraph().get_input_devices()
        return [str(n).strip() for n in (names or [])]
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read camera names from DirectShow (pygrabber): %s", exc)
        return None
    finally:
        if comtypes_module is not None:
            try:
                comtypes_module.CoUninitialize()
            except Exception as exc:  # noqa: BLE001
                log.debug("CoUninitialize failed: %s", exc)


def _probe_index(index: int, api: int) -> bool:
    cap = None
    try:
        cap = cv2.VideoCapture(index, api)
        return bool(cap.isOpened())
    except Exception as exc:  # noqa: BLE001 - cv2.error or driver errors
        log.debug("Probing camera %d failed: %s", index, exc)
        return False
    finally:
        _safe_release(cap)


def _probe_cameras(max_devices: int, backend: str) -> list[CameraInfo]:
    api = backend_id(backend)
    found: list[CameraInfo] = []
    misses = 0
    with _quiet_opencv():
        for index in range(max(0, int(max_devices))):
            in_use = _in_use_name(index)
            if in_use is not None:
                # Opened by this process right now: a second open would fail or disturb
                # the running stream, and the device certainly exists.
                name = in_use or f"Camera {index}"
                found.append(CameraInfo(index, name, backend, _is_ir_name(name), _is_virtual_name(name)))
                misses = 0
                continue
            if _probe_index(index, api):
                found.append(CameraInfo(index, f"Camera {index}", backend, False))
                misses = 0
                continue
            misses += 1
            # Indices are usually contiguous; one gap is tolerated (e.g. metadata nodes).
            if found and misses >= 2:
                break
    return found


def list_cameras(max_devices: int = 8, backend: str = "auto") -> list[CameraInfo]:
    """Enumerate cameras that can be opened with ``cv2.VideoCapture(index, backend)``.

    On Windows with the DirectShow backend (``"auto"``/``"dshow"``) the device names are
    read from DirectShow via ``pygrabber`` (fast, opens no device; the complete list is
    returned). Otherwise - or when pygrabber is unavailable or fails - indices
    ``0..max_devices-1`` are probed by opening them; probed devices are named
    ``"Camera N"``. Probing stops after two consecutive misses once a camera was found
    and can take about 0.5 s per device on Windows, so call this from a background
    thread (the app does). DirectShow names are not used for MSMF/ANY because those
    backends number devices differently. Devices this process has open are not
    re-opened. Infrared cameras are flagged with ``is_ir``.

    Never raises: failures are logged and yield an empty list.
    """
    try:
        name = _normalize_backend(backend)
        started = time.perf_counter()
        if _is_windows() and backend_id(name) == int(cv2.CAP_DSHOW):
            names = _dshow_device_names()
            if names is not None:
                cameras = [CameraInfo(i, n or f"Camera {i}", name, _is_ir_name(n), _is_virtual_name(n))
                           for i, n in enumerate(names)]
                log.info("DirectShow cameras: %s", ", ".join(c.label for c in cameras) or "none")
                return cameras
        cameras = _probe_cameras(max_devices, name)
        log.info("Camera probe (%s) finished in %.2f s: %s", _api_label(backend_id(name)),
                 time.perf_counter() - started, ", ".join(c.label for c in cameras) or "none found")
        return cameras
    except Exception:  # noqa: BLE001 - enumeration must never take the app down
        log.exception("Camera enumeration failed")
        return []


def choose_camera_index(cameras: list[CameraInfo], settings: CameraSettings) -> int:
    """Pick the camera index to open from an enumeration result.

    1. ``settings.name_hint`` re-finds a device by name (case-insensitive substring;
       exact names win, real webcams before virtual outputs before IR, the configured
       index on ties). A device the user picked by name is used even if it is virtual.
    2. The configured ``settings.index`` when it is listed and a real colour webcam.
    3. The first real colour webcam when the configured index is an IR camera or a
       virtual camera output (e.g. "OBS Virtual Camera", which this app may send to).
    4. :data:`NO_CAMERA` when the configured device is a virtual output and no webcam is
       listed: opening it would show another program's picture - or this app's own output
       in a feedback loop - instead of the user.
    5. Otherwise ``settings.index`` (the list may be incomplete, e.g. a busy device).
    """
    if not cameras:
        return settings.index
    hint = (settings.name_hint or "").strip().lower()
    if hint:
        matches = [c for c in cameras if hint in c.name.lower()]
        if matches:
            exact = [c for c in matches if c.name.strip().lower() == hint]
            pool = exact or matches
            candidates = [c for c in pool if _usable(c)] or [c for c in pool if not c.is_ir] or pool
            chosen = next((c for c in candidates if c.index == settings.index), candidates[0])
            if chosen.index != settings.index:
                log.info("Camera %r found at index %d (configured index %d)",
                         settings.name_hint, chosen.index, settings.index)
            return chosen.index
        log.info("No camera matches %r; using the configured index %d", settings.name_hint,
                 settings.index)
    configured = next((c for c in cameras if c.index == settings.index), None)
    if configured is not None and _usable(configured):
        return settings.index
    if configured is not None:
        webcam = next((c for c in cameras if _usable(c)), None)
        if webcam is not None:
            log.info("Camera %s is %s; using %s instead", configured.label,
                     "an infrared camera" if configured.is_ir else "a virtual camera output", webcam.label)
            return webcam.index
        if configured.is_virtual:
            log.warning("No webcam found, only virtual camera outputs (%s); none is opened automatically",
                        ", ".join(c.label for c in cameras))
            return NO_CAMERA
    return settings.index


def _device_exists(index: int, backend: str) -> bool | None:
    """Whether camera ``index`` exists, when the platform can tell cheaply; else ``None``."""
    try:
        if _is_windows():
            if backend_id(backend) == int(cv2.CAP_DSHOW):
                names = _dshow_device_names()
                if names is not None:
                    return 0 <= index < len(names)
            return None
        if sys.platform.startswith("linux"):
            return (Path("/dev") / f"video{index}").exists()
    except Exception as exc:  # noqa: BLE001
        log.debug("Could not check whether camera %d exists: %s", index, exc)
    return None


# =========================================================================== capture


@dataclass(eq=False)
class _Run:
    """One capture session (``start()`` until ``stop()`` or a terminal error).

    Each session has its own stop event and id, so a thread that outlives ``stop()``
    (stuck in a driver call) can never publish frames or status into a newer session.
    """

    run_id: int
    index: int
    name: str                          # device name given to start() ('' = unknown)
    settings: CameraSettings           # private snapshot taken by start()
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None
    cap: Any = None
    api: int | None = None
    registered_index: int | None = None
    blocked_since: float | None = None  # perf_counter() when the current driver call began
    hang_logged: float | None = None
    read_error_logged: bool = False
    last_ts: float | None = None
    interval: float | None = None       # smoothed frame interval (s)


class _CaptureSource:
    """Thread, frame-slot and status bookkeeping shared by the camera and the test pattern."""

    #: Seconds ``stop()`` waits for the capture thread to finish.
    STOP_JOIN_TIMEOUT: float = 3.0
    _thread_name = "camera-capture"

    def __init__(self, settings: CameraSettings) -> None:
        self._settings = settings
        self._control = threading.RLock()                # serialises start/stop/restart
        self._cond = threading.Condition(threading.Lock())  # guards the fields below
        self._run: _Run | None = None
        self._active_id: int | None = None               # run that may publish (None = idle)
        self._run_counter = 0
        self._frame_counter = 0                          # never reset: ids grow across restarts
        self._ended_runs = 0                             # bumped whenever a session ends
        self._latest: CameraFrame | None = None
        self._status = CameraStatus()
        # What the last start() opened, for restart(). Written under _control; a reconnect
        # of the active session also updates _started_index (under _cond).
        self._started_index: int | None = None
        self._started_name = ""
        self._started_settings_index: int | None = None

    # ------------------------------------------------------------------ public API

    @property
    def settings(self) -> CameraSettings:
        """The settings object used by the next ``start()``/``restart()``."""
        return self._settings

    @property
    def status(self) -> CameraStatus:
        """Thread-safe snapshot of the current state (a copy; never torn)."""
        with self._cond:
            return dataclasses.replace(self._status)

    @property
    def is_running(self) -> bool:
        """True while a session is active (opening, streaming or reconnecting)."""
        with self._cond:
            return self._active_id is not None

    def latest(self) -> CameraFrame | None:
        """Newest frame of the active session, or ``None`` (stopped, failed, no frame yet)."""
        with self._cond:
            return self._latest

    def wait_for_frame(self, after_id: int, timeout: float) -> CameraFrame | None:
        """Block until a frame with ``frame_id > after_id`` exists and return the newest one.

        Returns ``None`` after ``timeout`` seconds, immediately when the session is
        stopped or fails during the wait, and after at most ~50 ms when nothing is
        capturing at all (so polling loops cannot spin). Frame ids keep increasing
        across restarts, so ``after_id`` from an earlier session stays valid.
        """
        after = _as_frame_id(after_id)
        try:
            limit_s = float(timeout)
        except (TypeError, ValueError):
            limit_s = 0.0
        if not limit_s > 0.0:  # negative or NaN: just check once
            limit_s = 0.0
        start = time.perf_counter()
        deadline = start + limit_s
        with self._cond:
            ended_at_entry = self._ended_runs
            while True:
                frame = self._latest
                if frame is not None and frame.frame_id > after:
                    return frame
                if self._ended_runs != ended_at_entry:
                    return None  # stopped or failed while we were waiting
                limit = deadline if self._active_id is not None else min(deadline, start + _IDLE_WAIT)
                remaining = limit - time.perf_counter()
                if remaining <= 0.0:
                    return None
                self._cond.wait(min(remaining, _MAX_CONDITION_WAIT))

    def start(self, index: int | None = None, name: str | None = None) -> None:
        """Start capturing (non-blocking: the device is opened in the capture thread).

        A running session is stopped first. ``index`` defaults to ``settings.index``;
        ``name`` is the device name for status/error texts (e.g. ``CameraInfo.name``).
        """
        with self._control:
            self.stop()
            settings = dataclasses.replace(self._settings)  # the thread never sees later edits
            target_index, target_name = self._target(settings, index, name)
            self._run_counter += 1
            run = _Run(self._run_counter, target_index, target_name, settings)
            self._started_index = target_index
            self._started_name = target_name
            self._started_settings_index = settings.index
            with self._cond:
                self._run = run
                self._active_id = run.run_id
                self._latest = None
                self._status = self._starting_status(run)
                self._cond.notify_all()
            problem = self._precheck(run)
            if problem is not None:
                self._fail(run, problem)
                return
            thread = threading.Thread(target=self._thread_entry, args=(run,),
                                      name=self._thread_name, daemon=True)
            run.thread = thread
            log.info("%s: starting", self._describe(run))
            try:
                thread.start()
            except RuntimeError as exc:  # "can't start new thread"
                self._fail(run, f"Could not start the capture thread ({exc}). Restart the app.")

    def stop(self) -> None:
        """Stop capturing and release the device. Idempotent; waits up to 3 s for the thread."""
        with self._control:
            with self._cond:
                run, self._run = self._run, None
                if run is not None:
                    run.stop_event.set()
                    if self._active_id == run.run_id:
                        self._end_session_locked()  # wakes waiters, blocks further publishing
                elif self._status.state is not CameraState.STOPPED:
                    self._set_stopped_locked()  # e.g. clear an ERROR of a finished session
            if run is None:
                return
            thread = run.thread
            if thread is not None and thread.is_alive() and thread is not threading.current_thread():
                thread.join(self.STOP_JOIN_TIMEOUT)
                if thread.is_alive():
                    log.warning("%s: capture thread did not stop within %.1f s (a driver call is "
                                "still blocking); it releases the device when the call returns",
                                self._describe(run), self.STOP_JOIN_TIMEOUT)
            with self._cond:
                self._set_stopped_locked()
            log.info("%s: stopped", self._describe(run))

    def restart(self, settings: CameraSettings | None = None, index: int | None = None,
                name: str | None = None) -> None:
        """``stop()``, adopt ``settings`` (if given) and ``start()`` again.

        Reopens the same device unless ``index`` is given or ``settings.index`` differs
        from the index that was configured when the session was started. "The same device"
        follows a reconnect that found the camera at a new index.
        """
        with self._control:
            # End the session before reading the remembered index: a running reconnect may
            # still move the device (see _reconnect), an ended session cannot.
            self.stop()
            new_settings = settings if settings is not None else self._settings
            if index is None:
                if (self._started_settings_index is not None
                        and new_settings.index != self._started_settings_index):
                    index = new_settings.index  # the caller selected another camera
                elif self._started_index is not None:
                    index = self._started_index
                    if name is None:
                        name = self._started_name
            self._settings = new_settings
            self.start(index, name)

    # -------------------------------------------------------------- subclass hooks

    def _target(self, settings: CameraSettings, index: int | None,
                name: str | None) -> tuple[int, str]:
        raise NotImplementedError

    def _starting_status(self, run: _Run) -> CameraStatus:
        raise NotImplementedError

    def _thread_main(self, run: _Run) -> None:
        raise NotImplementedError

    def _precheck(self, run: _Run) -> str | None:
        return None

    def _cleanup(self, run: _Run) -> None:
        pass

    def _describe(self, run: _Run) -> str:
        return _camera_ref(run.index, run.name)

    # ------------------------------------------------------------------ internals

    def _thread_entry(self, run: _Run) -> None:
        try:
            self._thread_main(run)
        except Exception as exc:  # noqa: BLE001 - nothing may escape the thread
            log.exception("%s: capture thread failed", self._describe(run))
            self._fail(run, f"{self._describe(run)} stopped because of an unexpected error "
                            f"({type(exc).__name__}: {exc}). {_RETRY_HINT}")
        finally:
            try:
                self._cleanup(run)
            except Exception:  # noqa: BLE001
                log.exception("%s: cleanup failed", self._describe(run))
            if not run.stop_event.is_set():
                # Regular exits are stop() or _fail(); report anything else (no-op after _fail).
                self._fail(run, f"{self._describe(run)} stopped unexpectedly. {_RETRY_HINT}")

    def _set_stopped_locked(self) -> None:
        self._status = dataclasses.replace(self._status, state=CameraState.STOPPED, fps=0.0,
                                           error=None)

    def _end_session_locked(self) -> None:
        self._active_id = None
        self._latest = None
        self._ended_runs += 1
        self._cond.notify_all()

    def _fail(self, run: _Run, message: str) -> None:
        """Terminal error of ``run``: status ERROR with ``message``; logged once."""
        with self._cond:
            if self._active_id != run.run_id:
                return  # stopped or superseded meanwhile: the error is irrelevant
            self._status = dataclasses.replace(self._status, state=CameraState.ERROR, fps=0.0,
                                               error=message)
            self._end_session_locked()
        log.error("%s", message)

    def _set_status(self, run: _Run, **changes: Any) -> bool:
        with self._cond:
            if self._active_id != run.run_id:
                return False
            self._status = dataclasses.replace(self._status, **changes)
            return True

    def _reserve_frame_id(self, run: _Run) -> int | None:
        with self._cond:
            if self._active_id != run.run_id:
                return None
            self._frame_counter += 1
            return self._frame_counter

    def _publish(self, run: _Run, image: np.ndarray, timestamp: float,
                 frame_id: int | None = None) -> bool:
        """Make ``image`` the latest frame and wake waiters. False when ``run`` is stale."""
        height, width = image.shape[:2]
        with self._cond:
            if self._active_id != run.run_id:
                return False
            if frame_id is None:
                self._frame_counter += 1
                frame_id = self._frame_counter
            else:
                self._frame_counter = max(self._frame_counter, frame_id)
            self._latest = CameraFrame(frame_id, image, timestamp)
            if run.last_ts is not None:
                dt = timestamp - run.last_ts
                if dt > 0.0:
                    # Smooth the interval rather than 1/dt: bursty delivery (two frames
                    # 1 ms apart) would otherwise inflate the displayed fps.
                    run.interval = dt if run.interval is None else (
                        run.interval + _FPS_SMOOTHING * (dt - run.interval))
            run.last_ts = timestamp
            status = self._status
            status.state = CameraState.RUNNING
            status.resolution = (int(width), int(height))
            status.fps = 1.0 / run.interval if run.interval else 0.0
            status.frames += 1
            status.error = None
            self._cond.notify_all()
        return True


class CameraManager(_CaptureSource):
    """Captures a webcam in a background thread; implements ``FrameSource``.

    All public methods are thread-safe. ``start()`` returns immediately; progress and
    failures are reported through :attr:`status` (``STARTING`` -> ``RUNNING`` or
    ``ERROR`` with an actionable message). The manager keeps a reference to
    ``settings``; every ``start()`` takes a private snapshot of it.

    Args:
        settings: camera settings (index, backend, resolution, fps, MJPG, mirroring).
        open_timeout: seconds the first frame may take after the device opened.
        failure_timeout: seconds of consecutive failed reads before one reopen attempt.
        hang_timeout: seconds a single driver call may block before ``status`` reports
            the camera as not responding (default ``max(10, open + failure + 3)``).
    """

    def __init__(self, settings: CameraSettings, *, open_timeout: float = 5.0,
                 failure_timeout: float = 2.0, hang_timeout: float | None = None) -> None:
        super().__init__(settings)
        self.open_timeout = max(0.05, float(open_timeout))
        self.failure_timeout = max(0.05, float(failure_timeout))
        self.hang_timeout = (max(0.05, float(hang_timeout)) if hang_timeout is not None
                             else max(10.0, self.open_timeout + self.failure_timeout + 3.0))

    @property
    def status(self) -> CameraStatus:
        """Thread-safe snapshot; reports ``ERROR`` while a driver call hangs."""
        with self._cond:
            status = dataclasses.replace(self._status)
            run = self._run
            if run is not None and run.run_id != self._active_id:
                run = None
        if run is None or status.state not in (CameraState.STARTING, CameraState.RUNNING):
            return status
        since = run.blocked_since
        if since is None:
            return status
        stalled = time.perf_counter() - since
        if stalled <= self.hang_timeout:
            return status
        # The capture thread is stuck inside the driver; it recovers by itself if the call
        # ever returns, so this is a derived view and the session stays active.
        status.state = CameraState.ERROR
        status.fps = 0.0
        status.error = (f"{self._describe(run)} is not responding: the camera driver has been "
                        f"blocked for {stalled:.0f} s. Unplug and reconnect the camera, or "
                        f"{_SELECT_OTHER}.")
        if run.hang_logged != since:
            run.hang_logged = since
            log.warning("%s: driver call blocked for %.1f s", self._describe(run), stalled)
        return status

    # -------------------------------------------------------------------- hooks

    def _target(self, settings: CameraSettings, index: int | None,
                name: str | None) -> tuple[int, str]:
        raw = settings.index if index is None else index
        try:
            target = int(raw)
        except (TypeError, ValueError):
            target = -1
        return target, (name or "").strip()

    def _starting_status(self, run: _Run) -> CameraStatus:
        api = _backend_candidates(run.settings.backend)[0]
        return CameraStatus(state=CameraState.STARTING, index=run.index,
                            name=run.name or f"Camera {run.index}", backend=_api_label(api))

    def _precheck(self, run: _Run) -> str | None:
        if run.index < 0:
            return f"Invalid camera index {run.index}. Select a camera in Settings (Tab)."
        return None

    def _cleanup(self, run: _Run) -> None:
        self._release(run)

    # ------------------------------------------------------------ capture thread

    def _thread_main(self, run: _Run) -> None:
        if self._open_and_start(run, reconnect=False):
            self._read_loop(run)

    def _read_loop(self, run: _Run) -> None:
        # Hot path: read, flip, publish. No cap.get() or logging per frame.
        stop = run.stop_event
        mirror = bool(run.settings.mirror)
        last_good = time.perf_counter()
        while not stop.is_set():
            ok, img = self._read(run, run.cap)
            now = time.perf_counter()
            image = _prepare_frame(img, mirror) if ok else None
            if image is not None:
                self._publish(run, image, now)
                last_good = now
                continue
            if stop.is_set():
                return
            # Measured from the last good frame, so one read that blocked for longer than
            # failure_timeout and then failed counts as a sustained failure.
            if now - last_good <= self.failure_timeout:
                stop.wait(_READ_RETRY_DELAY)
                continue
            if not self._reconnect(run):
                return
            last_good = time.perf_counter()

    def _read(self, run: _Run, cap: Any) -> tuple[bool, Any]:
        run.blocked_since = time.perf_counter()
        try:
            ok, img = cap.read()
        except cv2.error as exc:
            if not run.read_error_logged:
                run.read_error_logged = True
                log.warning("%s: read failed: %s", self._describe(run), exc)
            return False, None
        finally:
            run.blocked_since = None
        return bool(ok), img

    def _open_and_start(self, run: _Run, *, reconnect: bool) -> bool:
        """Open + configure the device and deliver its first frame. False = gave up."""
        # Registered before opening, not after: a probe (list_cameras) that opened the same
        # index meanwhile would, with DirectShow's process-wide device table, stop our
        # stream when it releases its handle. _release() unregisters (also on failure).
        if run.registered_index is None:
            _register_in_use(run.index, run.name)
            run.registered_index = run.index
        cap, api = self._open_device(run, prefer=run.api if reconnect else None)
        if cap is None:
            self._release(run)  # nothing is open: unregister before the ERROR becomes visible
            if not run.stop_event.is_set():
                self._fail(run, self._disconnected_message(run) if reconnect
                           else self._open_failed_message(run))
            return False
        run.cap, run.api = cap, api
        self._configure(run, cap)
        backend = _capture_backend_name(cap, api)
        self._set_status(run, backend=backend)
        first = self._read_first_frame(run, cap)
        if first is None:
            if not run.stop_event.is_set():
                self._fail(run, self._disconnected_message(run) if reconnect
                           else self._no_frames_message(run))
            return False
        image, timestamp = first
        self._log_started(run, cap, image, backend, reconnect)
        self._publish(run, image, timestamp)
        return True

    def _open_device(self, run: _Run, prefer: int | None = None) -> tuple[Any, int | None]:
        candidates = _backend_candidates(run.settings.backend)
        if prefer is not None and prefer in candidates:
            candidates.remove(prefer)
            candidates.insert(0, prefer)  # reconnect with the backend that worked before
        for attempt, api in enumerate(candidates, start=1):
            if run.stop_event.is_set():
                return None, None
            label = _api_label(api)
            log.info("%s: opening with %s (attempt %d/%d)", self._describe(run), label, attempt,
                     len(candidates))
            cap: Any = None
            opened = False
            run.blocked_since = time.perf_counter()
            try:
                cap = cv2.VideoCapture(run.index, api)
                opened = bool(cap.isOpened())
            except Exception as exc:  # noqa: BLE001 - cv2.error and driver failures
                log.warning("%s: opening with %s raised %s", self._describe(run), label, exc)
            finally:
                run.blocked_since = None
            if opened and not run.stop_event.is_set():
                return cap, api
            _safe_release(cap)
            if opened:
                return None, None  # stop() arrived while the driver was opening
            log.warning("%s: could not be opened with %s", self._describe(run), label)
        return None, None

    def _configure(self, run: _Run, cap: Any) -> None:
        s = run.settings
        requests: list[tuple[int, float, str]] = []
        if s.prefer_mjpg:
            # Must precede the size: DirectShow picks the media type when width/height are
            # set, and uncompressed YUY2 is limited to low fps at 720p and above.
            requests.append((cv2.CAP_PROP_FOURCC, float(cv2.VideoWriter_fourcc(*"MJPG")), "MJPG"))
        requests += [
            (cv2.CAP_PROP_FRAME_WIDTH, float(s.width), f"width {s.width}"),
            (cv2.CAP_PROP_FRAME_HEIGHT, float(s.height), f"height {s.height}"),
            (cv2.CAP_PROP_FPS, float(s.fps), f"{s.fps} fps"),
            # One queued frame keeps latency low: we always want the newest image.
            (cv2.CAP_PROP_BUFFERSIZE, 1.0, "buffer size 1"),
        ]
        run.blocked_since = time.perf_counter()
        try:
            for prop, value, label in requests:
                if run.stop_event.is_set():
                    return
                try:
                    accepted = bool(cap.set(prop, value))
                except Exception as exc:  # noqa: BLE001 - some drivers raise for unsupported props
                    accepted = False
                    log.debug("%s: setting %s raised %s", self._describe(run), label, exc)
                if not accepted:
                    log.debug("%s: driver did not accept %s", self._describe(run), label)
        finally:
            run.blocked_since = None

    def _read_first_frame(self, run: _Run, cap: Any) -> tuple[np.ndarray, float] | None:
        deadline = time.perf_counter() + self.open_timeout
        mirror = bool(run.settings.mirror)
        while not run.stop_event.is_set():
            ok, img = self._read(run, cap)
            now = time.perf_counter()
            if ok:
                image = _prepare_frame(img, mirror)
                if image is not None:
                    return image, now
            if now >= deadline:
                return None
            run.stop_event.wait(_FIRST_FRAME_RETRY_DELAY)
        return None

    def _reconnect(self, run: _Run) -> bool:
        """One reopen attempt after a sustained read failure. False = ERROR (or stopped)."""
        log.warning("%s: no frames for %.1f s; reopening the device", self._describe(run),
                    self.failure_timeout)
        self._release(run)
        if run.stop_event.is_set():
            return False
        self._set_status(run, state=CameraState.STARTING, fps=0.0)
        target = self._reopen_index(run)
        if target is None:
            self._fail(run, self._disconnected_message(run))
            return False
        if target != run.index:
            log.info("%s is now at index %d", self._describe(run), target)
            run.index = target
            with self._cond:
                if self._active_id == run.run_id:
                    self._status = dataclasses.replace(self._status, index=target)
                    # restart() reopens "the same device": that is the new index now, the
                    # old one belongs to another camera (or to nothing).
                    self._started_index = target
        run.last_ts = None
        run.interval = None
        return self._open_and_start(run, reconnect=True)

    def _reopen_index(self, run: _Run) -> int | None:
        """Index to reopen, following the device by name; None when it is gone.

        After an unplug, DirectShow renumbers the remaining devices - reopening the old
        index blindly could silently switch to another camera (e.g. the IR camera).
        """
        if not (_is_windows() and run.api == int(cv2.CAP_DSHOW) and _is_named(run.name)):
            return run.index
        names = _dshow_device_names()
        if names is None:
            return run.index
        # Exact names only (ignoring case): a substring such as "Camera" would also match
        # "HP IR Camera" and switch to the wrong device.
        wanted = run.name.strip().casefold()
        keys = [n.casefold() for n in names]
        if 0 <= run.index < len(keys) and keys[run.index] == wanted:
            return run.index
        if wanted in keys:
            return keys.index(wanted)
        log.warning("%s is no longer connected (DirectShow devices: %s)", self._describe(run),
                    ", ".join(names) or "none")
        return None

    def _release(self, run: _Run) -> None:
        cap, run.cap = run.cap, None
        if cap is not None:
            run.blocked_since = time.perf_counter()  # release() can hang on broken drivers
            try:
                cap.release()
            except Exception as exc:  # noqa: BLE001
                log.warning("%s: releasing the device failed: %s", self._describe(run), exc)
            finally:
                run.blocked_since = None
        if run.registered_index is not None:
            _unregister_in_use(run.registered_index, run.name)
            run.registered_index = None

    def _log_started(self, run: _Run, cap: Any, image: np.ndarray, backend: str,
                     reconnect: bool) -> None:
        s = run.settings
        height, width = image.shape[:2]
        fps = _safe_get(cap, cv2.CAP_PROP_FPS)
        fourcc = _fourcc_to_str(_safe_get(cap, cv2.CAP_PROP_FOURCC))
        log.info("%s %s: backend %s, requested %dx%d @ %d fps%s, actual %dx%d @ %s fps "
                 "(driver), FOURCC %s, mirror %s", self._describe(run),
                 "reconnected" if reconnect else "running", backend, s.width, s.height, s.fps,
                 " MJPG" if s.prefer_mjpg else "", width, height,
                 f"{fps:.1f}" if fps > 0 else "?", fourcc, "on" if s.mirror else "off")
        if (width, height) != (s.width, s.height):
            log.info("%s does not support %dx%d; using %dx%d", self._describe(run), s.width,
                     s.height, width, height)

    # ------------------------------------------------------------------ messages

    def _open_failed_message(self, run: _Run) -> str:
        exists = _device_exists(run.index, run.settings.backend)
        if exists is False:
            if _is_named(run.name):
                return (f"{_camera_ref(run.index, run.name)} is not connected. Plug it in or "
                        f"{_SELECT_OTHER}.")
            return f"Camera {run.index} does not exist. Select another camera in Settings (Tab)."
        text = f"Could not open camera {run.index}{self._name_suffix(run)}. "
        if _is_ir_name(run.name):
            # Windows Hello usually holds its IR camera; closing apps would not help.
            return text + _IR_HINT
        return text + f"{_access_hint()}."

    def _no_frames_message(self, run: _Run) -> str:
        text = (f"{self._describe(run)} opened but delivered no frames within "
                f"{self.open_timeout:g} s. ")
        if _is_ir_name(run.name):
            return text + _IR_HINT
        return text + f"{_access_hint()}, or {_SELECT_OTHER}."

    def _disconnected_message(self, run: _Run) -> str:
        return (f"{self._describe(run)} disconnected or stopped delivering frames. Check the "
                "cable or USB port and close other apps that may use the camera, then select "
                "it again in Settings (Tab).")

    @staticmethod
    def _name_suffix(run: _Run) -> str:
        return f" ({run.name})" if _is_named(run.name) else ""


# ====================================================================== test pattern


class _TestPattern:
    """Static background rendered once + a cheap animated overlay (~1 ms per 720p frame)."""

    _FONT = cv2.FONT_HERSHEY_DUPLEX

    def __init__(self, width: int, height: int, fps: int) -> None:
        self.width, self.height, self.fps = width, height, fps
        self.scale = s = max(0.2, height / 720.0)
        self.thin = max(1, int(round(2 * s)))
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        img = np.empty((height, width, 3), np.uint8)
        # Dark blue gradient, brighter towards the bottom. One channel at a time keeps the
        # float temporary small (about 33 MB instead of 100 MB at 4K).
        for channel, (base, k_y, k_x) in enumerate(((38.0, 52.0, 14.0), (16.0, 18.0, 6.0),
                                                     (12.0, 6.0, 22.0))):
            img[..., channel] = np.clip(base + k_y * y + k_x * x, 0.0, 255.0)

        step = max(12, int(round(80 * s)))
        img[::step, :] = (88, 70, 60)
        img[:, ::step] = (88, 70, 60)
        cx, cy = width // 2, height // 2
        cv2.line(img, (cx, 0), (cx, height - 1), (150, 130, 110), self.thin)
        cv2.line(img, (0, cy), (width - 1, cy), (150, 130, 110), self.thin)
        cv2.circle(img, (cx, cy), int(min(width, height) * 0.3), (130, 110, 95), self.thin,
                   cv2.LINE_AA)

        # Pure B / G / R / W swatches make swapped colour channels obvious at a glance.
        size = max(8, int(round(54 * s)))
        gap = max(4, int(round(12 * s)))
        x0, y0 = int(24 * s), height - int(24 * s) - size
        swatches = (("B", (255, 0, 0)), ("G", (0, 255, 0)), ("R", (0, 0, 255)), ("W", (255, 255, 255)))
        for i, (label, colour) in enumerate(swatches):
            left = x0 + i * (size + gap)
            cv2.rectangle(img, (left, y0), (left + size, y0 + size), colour, -1)
            cv2.putText(img, label, (left + size // 3, y0 - int(8 * s)), self._FONT, 0.7 * s,
                        (235, 235, 235), self.thin, cv2.LINE_AA)

        # Orientation markers: the pattern is never mirrored.
        label_scale = 0.8 * s
        cv2.putText(img, "LEFT", (int(24 * s), cy - int(14 * s)), self._FONT, label_scale,
                    (210, 210, 210), self.thin, cv2.LINE_AA)
        (text_w, _), _ = cv2.getTextSize("RIGHT", self._FONT, label_scale, self.thin)
        cv2.putText(img, "RIGHT", (width - int(24 * s) - text_w, cy - int(14 * s)), self._FONT,
                    label_scale, (210, 210, 210), self.thin, cv2.LINE_AA)
        self._base = img

    def render(self, frame_number: int, t: float) -> np.ndarray:
        """Frame ``frame_number`` at animation time ``t`` seconds (new array)."""
        img = self._base.copy()
        w, h, s = self.width, self.height, self.scale

        # Bright orb on a Lissajous path; concentric discs fake a glow without a blur.
        cx = int(w * (0.5 + 0.36 * math.sin(0.83 * t)))
        cy = int(h * (0.5 + 0.30 * math.sin(1.31 * t + 0.7)))
        radius = max(3.0, 46.0 * s)
        for factor, colour in ((2.2, (90, 60, 30)), (1.6, (190, 130, 60)), (1.15, (255, 210, 140)),
                               (0.7, (255, 250, 235))):
            cv2.circle(img, (cx, cy), max(1, int(radius * factor)), colour, -1, cv2.LINE_AA)

        # Sweep hand around the centre: stutter and dropped frames are easy to see.
        angle = 1.6 * t
        length = min(w, h) * 0.3
        tip = (int(w / 2 + length * math.cos(angle)), int(h / 2 + length * math.sin(angle)))
        cv2.line(img, (w // 2, h // 2), tip, (60, 230, 255), max(2, int(round(5 * s))), cv2.LINE_AA)

        # Square bouncing between the screen edges (triangle wave).
        side = max(6, int(round(70 * s)))
        u = abs(((0.37 * t) % 2.0) - 1.0)
        left = int(u * max(0, w - side))
        top = int(h * 0.68)
        cv2.rectangle(img, (left, top), (left + side, top + side), (200, 80, 255), -1)

        cv2.putText(img, "TEST PATTERN", (int(24 * s), int(64 * s)), self._FONT, 1.5 * s,
                    (255, 255, 255), max(1, int(round(3 * s))), cv2.LINE_AA)
        cv2.putText(img, f"frame {frame_number}", (int(24 * s), int(112 * s)), self._FONT, 0.9 * s,
                    (200, 230, 255), self.thin, cv2.LINE_AA)
        cv2.putText(img, f"{w}x{h} @ {self.fps} fps   t = {t:7.2f} s", (int(24 * s), int(148 * s)),
                    self._FONT, 0.7 * s, (170, 190, 210), self.thin, cv2.LINE_AA)
        return img


class SyntheticCamera(_CaptureSource):
    """Animated diagnostic test pattern with the :class:`CameraManager` interface.

    Generates frames of ``settings.width`` x ``settings.height`` at ``settings.fps`` in
    its own thread: dark gradient, grid, moving bright shapes, colour swatches,
    "TEST PATTERN" and the frame id. Only for automated tests and ``--test-pattern``
    diagnostics - never present it to the user as their camera. The pattern is drawn in
    scene orientation and is not mirrored (its text would become unreadable).
    ``start()`` ignores ``index`` and ``name``.
    """

    _thread_name = "camera-test-pattern"

    def _target(self, settings: CameraSettings, index: int | None,
                name: str | None) -> tuple[int, str]:
        return -1, TEST_PATTERN_NAME

    def _describe(self, run: _Run) -> str:
        return TEST_PATTERN_NAME

    def _starting_status(self, run: _Run) -> CameraStatus:
        return CameraStatus(state=CameraState.STARTING, index=-1, name=TEST_PATTERN_NAME,
                            backend=SYNTHETIC_BACKEND)

    def _thread_main(self, run: _Run) -> None:
        s = run.settings
        width, height, fps = max(16, int(s.width)), max(16, int(s.height)), max(1, int(s.fps))
        pattern = _TestPattern(width, height, fps)
        period = 1.0 / fps
        stop = run.stop_event
        t0 = time.perf_counter()
        next_due = t0
        log.info("Test pattern running: %dx%d @ %d fps", width, height, fps)
        while not stop.is_set():
            frame_id = self._reserve_frame_id(run)  # drawn into the image, so reserve it first
            if frame_id is None:
                return
            image = pattern.render(frame_id, time.perf_counter() - t0)
            if not self._publish(run, image, time.perf_counter(), frame_id=frame_id):
                return
            next_due += period
            delay = next_due - time.perf_counter()
            if delay < -period:
                next_due = time.perf_counter()  # fell behind (busy machine): resync, no burst
                continue
            if delay > 0.0:
                stop.wait(delay)
