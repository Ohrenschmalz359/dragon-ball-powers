"""Tests for the Windows 11 camera writer (win11_camera.py), its native counterpart's
constants (native/win11_camera) and how virtual_camera.py feeds it together with the
pyvirtualcam drivers. The Windows APIs are replaced by a fake section."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

from dragon_ball_powers import virtual_camera as vc
from dragon_ball_powers import win11_camera as w11

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "native" / "win11_camera" / "src"


class FakeSection:
    def __init__(self) -> None:
        self.buffer = bytearray(w11.SHARED_SIZE)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeCamera:
    def __init__(self, device: str, backend: str, fail: bool = False) -> None:
        self.device = device
        self.backend = backend
        self.fail = fail
        self.frames = 0
        self.closed = False

    def send(self, frame) -> None:
        if self.fail:
            raise RuntimeError("driver gone")
        self.frames += 1

    def close(self) -> None:
        self.closed = True


def test_picture(width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, : width // 2] = (255, 0, 0)       # RGB red
    frame[:, width // 2:] = (0, 255, 0)        # RGB green
    return frame


test_picture.__test__ = False                  # a helper, not a test


def read_header(section: FakeSection) -> tuple:
    return w11.HEADER.unpack_from(section.buffer, 0)


# ------------------------------------------------------------ native constants


def test_layout_matches_the_native_media_source():
    shared = (NATIVE / "SharedFrame.h").read_text(encoding="utf-8")
    camera = (NATIVE / "Camera.h").read_text(encoding="utf-8")
    dllmain = (NATIVE / "dllmain.cpp").read_text(encoding="utf-8")

    def number(name: str) -> int:
        return int(re.search(rf"\b{name}\s*=\s*(0x[0-9A-Fa-f]+|\d+)", shared).group(1), 0)

    name = re.search(r'DBP_SHARED_NAME\[\]\s*=\s*L"([^"]+)"', shared).group(1)
    assert name.replace("\\\\", "\\") == w11.SHARED_NAME
    assert number("DBP_SHARED_MAGIC") == w11.MAGIC
    assert number("DBP_SHARED_VERSION") == w11.VERSION
    assert number("DBP_SHARED_MAX_WIDTH") == w11.MAX_WIDTH
    assert number("DBP_SHARED_MAX_HEIGHT") == w11.MAX_HEIGHT
    assert number("DBP_SHARED_HEADER_SIZE") == w11.HEADER_SIZE == w11.HEADER.size
    assert number("DBP_FORMAT_BGRA") == w11.FORMAT_BGRA

    struct_body = re.search(r"struct DbpFrameHeader\s*\{(.*?)\};", shared, re.S).group(1)
    fields = re.findall(r"(UINT32|LONG64)\s+(\w+)(\[\d+\])?;", struct_body)
    codes = "".join({"UINT32": "I", "LONG64": "q"}[t] for t, _, array in fields if not array)
    assert "<" + codes + "20x" == w11.HEADER.format
    assert [f[1] for f in fields].index("sequence") == 6 and w11.SEQUENCE_OFFSET == 24

    assert re.search(r'DBP_CAMERA_NAME\[\]\s*=\s*L"([^"]+)"', camera).group(1) == w11.DEVICE_NAME
    guid = re.search(r"CLSID_DbpCamera\s*=\s*\{\s*0x(\w+),\s*0x(\w+),\s*0x(\w+),\s*\{([^}]*)\}", dllmain)
    tail = [b.strip()[2:] for b in guid.group(4).split(",")]
    clsid = "{%s-%s-%s-%s-%s}" % (guid.group(1), guid.group(2), guid.group(3), "".join(tail[:2]), "".join(tail[2:]))
    assert clsid.upper() == w11.CLSID


# ------------------------------------------------------------------- the sink


def test_sink_waits_for_an_app_then_writes_bgra_frames(monkeypatch):
    monkeypatch.setattr(w11, "ATTACH_INTERVAL", 0.0)
    sections: list[FakeSection] = []
    available = {"yes": False}

    def opener() -> FakeSection:
        if not available["yes"]:
            raise FileNotFoundError(2, "no app uses the camera yet")
        sections.append(FakeSection())
        return sections[-1]

    sink = w11.Win11CameraSink(1280, 720, opener)
    assert (sink.device, sink.backend) == ("Dragon Ball Powers (Windows 11)", "win11")
    sink.send(test_picture())                  # nobody uses the camera: dropped quietly
    assert not sink.attached and not sections

    available["yes"] = True
    sink.send(test_picture())
    assert sink.attached and len(sections) == 1
    magic, version, width, height, stride, fmt, sequence, _, _ = read_header(sections[0])
    assert (magic, version, width, height, stride, fmt) == (w11.MAGIC, 1, 1280, 720, 1280 * 4, w11.FORMAT_BGRA)
    assert sequence > 0 and sequence % 2 == 0          # even: the frame is complete
    pixels = np.frombuffer(sections[0].buffer, np.uint8, count=1280 * 720 * 4, offset=w11.HEADER_SIZE)
    pixels = pixels.reshape(720, 1280, 4)
    assert tuple(pixels[100, 100]) == (0, 0, 255, 255)        # B G R A: red
    assert tuple(pixels[600, 1000]) == (0, 255, 0, 255)       # green

    sink.send(test_picture())
    assert read_header(sections[0])[6] == sequence + 2
    sink.close()
    assert sections[0].closed and not sink.attached
    sink.close()                                   # idempotent


def test_sink_looks_for_the_camera_about_once_per_second():
    attempts = []

    def opener():
        attempts.append(1)
        raise FileNotFoundError(2, "not yet")

    sink = w11.Win11CameraSink(1280, 720, opener)
    for _ in range(20):
        sink.send(test_picture())
    assert len(attempts) == 1


def test_sink_scales_pictures_larger_than_full_hd(monkeypatch):
    monkeypatch.setattr(w11, "ATTACH_INTERVAL", 0.0)
    section = FakeSection()
    sink = w11.Win11CameraSink(2560, 1440, lambda: section)
    sink.send(test_picture(2560, 1440))
    _, _, width, height, stride, _, _, _, _ = read_header(section)
    assert (width, height, stride) == (1920, 1080, 1920 * 4)


def test_installed_only_on_windows_and_never_in_tests(monkeypatch):
    assert w11.installed() is False                    # conftest sets DBP_WIN11_CAMERA=0
    monkeypatch.delenv("DBP_WIN11_CAMERA")
    if sys.platform != "win32":
        assert w11.installed() is False


# ------------------------------------------------- feeding several cameras


def test_multi_camera_feeds_all_and_drops_a_broken_one():
    good, broken = FakeCamera("Dragon Ball Powers", "unitycapture"), FakeCamera("Other", "obs", fail=True)
    multi = vc.MultiCamera([good, broken])
    assert multi.device == "Dragon Ball Powers / Other" and multi.backend == "unitycapture+obs"
    multi.send(test_picture())
    assert good.frames == 1 and broken.closed and multi.device == "Dragon Ball Powers"
    multi.close()
    assert good.closed


def test_multi_camera_raises_when_the_last_camera_fails():
    with pytest.raises(RuntimeError):
        vc.MultiCamera([FakeCamera("Other", "obs", fail=True)]).send(test_picture())


def test_default_camera_feeds_windows11_and_the_pyvirtualcam_driver(monkeypatch):
    calls: list[dict] = []

    def pyvirtualcam_camera(**kwargs):
        calls.append(kwargs)
        return FakeCamera("Dragon Ball Powers", kwargs["backend"] or "unitycapture")

    monkeypatch.setattr(w11, "installed", lambda: True)
    monkeypatch.setattr(vc, "_pyvirtualcam_camera", pyvirtualcam_camera)
    kwargs = {"width": 1280, "height": 720, "fps": 30, "device": None}

    both = vc._default_camera(backend=None, **kwargs)
    assert isinstance(both, vc.MultiCamera)
    assert both.device == "Dragon Ball Powers (Windows 11) / Dragon Ball Powers"
    assert both.backend == "win11+unitycapture" and calls[-1]["backend"] is None

    only = vc._default_camera(backend="win11", **kwargs)
    assert isinstance(only, w11.Win11CameraSink) and len(calls) == 1

    again = vc._default_camera(backend=both.backend, **kwargs)      # a restart keeps both
    assert isinstance(again, vc.MultiCamera) and calls[-1]["backend"] == "unitycapture"


def test_default_camera_uses_windows11_alone_without_another_driver(monkeypatch):
    def no_driver(**kwargs):
        raise RuntimeError("'unitycapture' backend: No camera registered. Did you install any camera?")

    monkeypatch.setattr(w11, "installed", lambda: True)
    monkeypatch.setattr(vc, "_pyvirtualcam_camera", no_driver)
    camera = vc._default_camera(width=1280, height=720, fps=30, backend=None, device=None)
    assert isinstance(camera, w11.Win11CameraSink)


def test_default_camera_without_windows11_is_plain_pyvirtualcam(monkeypatch):
    monkeypatch.setattr(w11, "installed", lambda: False)
    monkeypatch.setattr(vc, "_pyvirtualcam_camera", lambda **kwargs: FakeCamera("OBS Virtual Camera", "obs"))
    camera = vc._default_camera(width=1280, height=720, fps=30, backend=None, device=None)
    assert isinstance(camera, FakeCamera)
    with pytest.raises(RuntimeError) as info:
        vc._default_camera(width=1280, height=720, fps=30, backend="win11", device=None)
    assert vc.friendly_error(info.value) == vc.INSTALL_HINT
