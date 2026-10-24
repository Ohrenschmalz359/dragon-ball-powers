"""Picture writer for the Windows 11 virtual camera "Dragon Ball Powers (Windows 11)".

The camera itself is a Media Foundation media source (``native/win11_camera``, installed by
``install_win11_camera.bat``) that the Windows camera frame server runs for every app -
also for apps such as WhatsApp Desktop and the Windows Camera app, which ignore the older
(DirectShow) virtual cameras of OBS / Unity Capture. This module hands it our pictures
through a named section; the layout must match ``native/win11_camera/src/SharedFrame.h``
(tests/test_win11_camera.py compares both).

The frame server creates the section when an app starts the camera (an ordinary user
process may not create objects in the ``Global`` namespace, but may open them), so the
writer looks for it about once per second and quietly waits until then.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

SHARED_NAME = "Global\\DragonBallPowersVirtualCamera"
MAGIC = 0x56504244                    # "DBPV" little-endian
VERSION = 1
MAX_WIDTH = 1920
MAX_HEIGHT = 1080
HEADER_SIZE = 64
SHARED_SIZE = HEADER_SIZE + MAX_WIDTH * MAX_HEIGHT * 4
FORMAT_BGRA = 1
CLSID = "{D61C8D41-AD63-4319-AE6F-EB46E5FEB187}"
DEVICE_NAME = "Dragon Ball Powers (Windows 11)"
BACKEND = "win11"
#: How often (s) the section is looked for while no app uses the camera.
ATTACH_INTERVAL = 1.0

#: magic, version, width, height, stride, format, sequence, timestamp, writer pid (+ reserved)
HEADER = struct.Struct("<IIIIIIqqI20x")
SEQUENCE_OFFSET = 24


def installed() -> bool:
    """True when ``install_win11_camera.bat`` registered the camera on this PC.

    ``DBP_WIN11_CAMERA=0`` makes the app ignore it (the tests set it).
    """
    if sys.platform != "win32" or os.environ.get("DBP_WIN11_CAMERA") == "0":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Classes\CLSID\{CLSID}\InprocServer32"):
            return True
    except OSError:
        return False


class _WindowsSection:
    """A writable view of the named section (Windows)."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenFileMappingW.restype = wintypes.HANDLE
        kernel32.OpenFileMappingW.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR)
        kernel32.MapViewOfFile.restype = ctypes.c_void_p
        kernel32.MapViewOfFile.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                           ctypes.c_size_t)
        kernel32.UnmapViewOfFile.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        file_map_write = 0x0002
        handle = kernel32.OpenFileMappingW(file_map_write, False, SHARED_NAME)
        if not handle:
            raise FileNotFoundError(ctypes.get_last_error(), "no app uses the camera yet")
        address = kernel32.MapViewOfFile(handle, file_map_write, 0, 0, SHARED_SIZE)
        if not address:
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            raise OSError(error, "MapViewOfFile failed")
        self._kernel32 = kernel32
        self._handle = handle
        self._address = address
        self.buffer: Any = (ctypes.c_ubyte * SHARED_SIZE).from_address(address)

    def close(self) -> None:
        self.buffer = None
        if self._address:
            self._kernel32.UnmapViewOfFile(self._address)
            self._address = None
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


class Win11CameraSink:
    """Writes pictures for the Windows 11 camera (a pyvirtualcam-like ``send`` / ``close``).

    ``send`` takes (H, W, 3) uint8 RGB pictures, top row first, at most 1920x1080.
    Before an app uses the camera there is nothing to write to: pictures are dropped.
    """

    device = DEVICE_NAME
    backend = BACKEND

    def __init__(self, width: int, height: int, opener: Callable[[], Any] | None = None) -> None:
        self.width = int(width)
        self.height = int(height)
        self._opener = opener or _WindowsSection
        self._section: Any = None
        self._memory: memoryview | None = None
        self._pixels: np.ndarray | None = None
        self._next_attempt = 0.0

    @property
    def attached(self) -> bool:
        """True once an app has started the camera (the section exists)."""
        return self._section is not None

    def _attach(self) -> bool:
        now = time.monotonic()
        if now < self._next_attempt:
            return False
        self._next_attempt = now + ATTACH_INTERVAL
        try:
            section = self._opener()
        except FileNotFoundError:
            return False                       # no app uses the camera yet
        except OSError as exc:
            log.debug("Windows 11 camera not reachable: %s", exc)
            return False
        self._section = section
        self._memory = memoryview(section.buffer).cast("B")
        self._pixels = np.frombuffer(section.buffer, dtype=np.uint8, count=MAX_WIDTH * MAX_HEIGHT * 4,
                                     offset=HEADER_SIZE)
        log.info("Windows 11 camera in use by an app: sending pictures")
        return True

    def send(self, frame: np.ndarray) -> None:
        if self._section is None and not self._attach():
            return
        assert self._memory is not None and self._pixels is not None
        image = np.asarray(frame)
        h, w = image.shape[:2]
        if w > MAX_WIDTH or h > MAX_HEIGHT:
            scale = min(MAX_WIDTH / w, MAX_HEIGHT / h)
            image = cv2.resize(image, (int(w * scale) & ~1, int(h * scale) & ~1), interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
        memory = self._memory
        writing = (struct.unpack_from("<q", memory, SEQUENCE_OFFSET)[0] + 1) | 1     # odd: being written
        struct.pack_into("<q", memory, SEQUENCE_OFFSET, writing)
        HEADER.pack_into(memory, 0, MAGIC, VERSION, w, h, w * 4, FORMAT_BGRA, writing,
                         time.perf_counter_ns() // 100, os.getpid())
        target = self._pixels[: h * w * 4].reshape(h, w, 4)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA, dst=target)
        if result is not target:
            target[...] = result
        struct.pack_into("<q", memory, SEQUENCE_OFFSET, writing + 1)                # even: complete

    def close(self) -> None:
        self._pixels = None                    # no view may outlive the mapping
        if self._memory is not None:
            self._memory.release()
            self._memory = None
        if self._section is not None:
            try:
                self._section.close()
            finally:
                self._section = None


__all__ = ["Win11CameraSink", "installed", "DEVICE_NAME", "BACKEND", "SHARED_NAME", "CLSID"]
