"""Tests for the HUD / settings panel (pure draw commands, SDL dummy video driver)."""

from __future__ import annotations

import math
import time

import numpy as np
import pygame
import pytest

from dragon_ball_powers import config
from dragon_ball_powers.config import Settings
from dragon_ball_powers.datatypes import (
    AbilityHUD,
    BodyState,
    CalibrationHUD,
    CameraInfo,
    CameraState,
    CameraStatus,
    GestureSignals,
    HandObservation,
    PoseObservation,
    TrackerStatus,
    TrackingFrame,
)
from dragon_ball_powers.fonts import FontManager
from dragon_ball_powers.interfaces import UI_ACTION_KINDS, UIContext
from dragon_ball_powers.primitives import UIArc, UICircle, UILine, UIRect, UIText, compute_view_transform
from dragon_ball_powers.ui import UIController

SIZES = [(1280, 720), (1920, 1080), (1920, 1200), (2560, 1440), (3840, 2160), (800, 600)]
CAMERAS = [CameraInfo(0, "HP 5MP Camera"), CameraInfo(1, "HP IR Camera", is_ir=True)]


@pytest.fixture(scope="module")
def fonts():
    pygame.init()
    return FontManager()


@pytest.fixture
def ui(fonts):
    return UIController(Settings(), fonts)


def make_ctx(ui: UIController, size=(1920, 1080), cam=CameraState.RUNNING, calib=False, fullscreen=False,
             body=None, frame=None, now=None, error=None, phase="charging") -> UIContext:
    s = ui._settings if hasattr(ui, "_settings") else Settings()
    return UIContext(
        now=time.perf_counter() if now is None else now, window_size=size,
        view=compute_view_transform(1280, 720, *size), settings=s,
        camera_status=CameraStatus(state=cam, index=0, name="HP 5MP Camera", resolution=(1280, 720), fps=30.0,
                                   error=error or ("Could not open camera 0" if cam is CameraState.ERROR else None)),
        tracker_status=TrackerStatus(running=True, fps=29.5, latency_ms=41.0),
        ability=AbilityHUD(selected="kamehameha", selected_label="Kamehameha", phase=phase, charge=0.73,
                           charge_ready=True, power_level=9450, power_fraction=0.79, callout="KA... ME... HA...",
                           callout_intensity=0.8, message="Push both arms forward to fire!"),
        calibration=CalibrationHUD(active=calib, step=2, total_steps=3, title="Hands together",
                                   instruction="Bring both hands together", progress=0.4),
        cameras=CAMERAS, body=body, frame=frame, signals=GestureSignals(), render_fps=59.8,
        renderer_name="OpenGL", quality_name="high", fullscreen=fullscreen)


def check_commands(dl, size) -> None:
    assert len(dl) > 0
    w, h = size
    for c in dl.commands:
        if isinstance(c, UIRect):
            vals = (c.x, c.y, c.w, c.h, c.radius)
        elif isinstance(c, UIText):
            assert c.text, "empty text command"
            vals = (c.x, c.y, float(c.size))
            assert -w <= c.x <= 2 * w and -h <= c.y <= 2 * h, c
        elif isinstance(c, (UICircle, UIArc)):
            vals = (c.x, c.y, c.radius)
        elif isinstance(c, UILine):
            vals = tuple(v for p in c.points for v in p)
        else:
            raise AssertionError(f"unknown command {c!r}")
        assert all(math.isfinite(float(v)) for v in vals), c


@pytest.mark.parametrize("size", SIZES)
@pytest.mark.parametrize("cam", list(CameraState))
def test_hud_builds_for_every_state_and_size(ui, size, cam):
    for calib in (False, True):
        for fullscreen in (False, True):
            dl = ui.build(make_ctx(ui, size, cam, calib=calib, fullscreen=fullscreen))
            check_commands(dl, size)


def test_hud_shows_the_power_level_and_callout(ui):
    dl = ui.build(make_ctx(ui))
    texts = " ".join(c.text for c in dl.commands if isinstance(c, UIText))
    assert "9,450" in texts
    assert "KA... ME... HA..." in texts
    assert "POWER LEVEL" in texts


def test_debug_overlay_draws_skeletons(ui):
    ui._settings.ui.show_debug = True
    hand = HandObservation("left", np.tile(np.array([[600.0, 400.0, 0.0]], np.float32), (21, 1)))
    pose = PoseObservation(np.full((33, 3), 500.0, np.float32), np.ones(33, np.float32))
    frame = TrackingFrame(1, 0.0, 0.0, (1280, 720), None, hands=[hand], pose=pose)
    body = BodyState(0.0, (1280, 720), body_scale=200.0, hand_mid=(640.0, 400.0))
    dl = ui.build(make_ctx(ui, body=body, frame=frame))
    assert sum(isinstance(c, UILine) for c in dl.commands) > 20
    ui._settings.ui.show_debug = False


def test_every_widget_click_emits_a_valid_action(ui):
    ui.toggle_settings()
    ctx = make_ctx(ui)
    ui.build(ctx)
    seen = set()
    for widget in list(ui.widgets):
        x, y = widget.center
        actions = ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(x), int(y)), button=1))
        actions += ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(int(x), int(y)), button=1))
        for action in actions:
            assert action.kind in UI_ACTION_KINDS, action
            if action.kind == "set_setting":
                dotted, value = action.value
                config.get_setting(Settings(), dotted)  # the key must exist
            seen.add(action.kind)
        ui.build(make_ctx(ui))
    for kind in ("select_camera", "rescan_cameras", "toggle_camera", "set_resolution", "set_setting",
                 "recalibrate", "toggle_aura", "toggle_settings"):
        assert kind in seen, kind


def test_hud_buttons_select_ability(ui):
    ui.build(make_ctx(ui))
    widget = ui.widget("hud:ability:ki_charge")
    assert widget is not None
    x, y = widget.center
    actions = ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(x), int(y)), button=1))
    actions += ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(int(x), int(y)), button=1))
    assert any(a.kind == "select_ability" and a.value == "ki_charge" for a in actions)


def test_slider_drag_stays_within_constraints(ui):
    ui.toggle_settings()
    ui.build(make_ctx(ui))
    slider = next((w for w in ui.widgets if w.kind == "slider" and "sensitivity" in w.id), None)
    assert slider is not None
    x0, y0, w, h = slider.rect
    actions = ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(x0 + 2), int(y0 + h / 2)), button=1))
    for x in np.linspace(x0 - 50, x0 + w + 50, 12):
        actions += ui.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(int(x), int(y0 + h / 2)),
                                                      rel=(1, 0), buttons=(1, 0, 0)))
    actions += ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(int(x0 + w + 50), int(y0 + h / 2)),
                                                  button=1))
    values = [a.value[1] for a in actions if a.kind == "set_setting" and a.value[0] == "gestures.sensitivity"]
    assert values, "dragging the slider must emit set_setting"
    assert all(0.0 <= v <= 1.0 for v in values)


def test_reset_needs_two_clicks(ui):
    ui.toggle_settings()
    ui.build(make_ctx(ui))
    widget = ui.widget("settings:reset")
    assert widget is not None
    x, y = widget.center

    def click():
        a = ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=(int(x), int(y)), button=1))
        a += ui.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(int(x), int(y)), button=1))
        ui.build(make_ctx(ui))
        return [x for x in a if x.kind == "reset_settings"]

    assert click() == []
    assert len(click()) == 1


def test_wheel_scrolls_the_settings_panel(ui):
    ui.toggle_settings()
    ui.build(make_ctx(ui, size=(1280, 720)))
    panel = ui.widget("settings:panel")
    pos = (int(panel.center[0]), int(panel.center[1])) if panel is not None else (1200, 400)
    for _ in range(40):
        pygame.mouse.set_pos(pos) if pygame.display.get_init() and pygame.display.get_surface() else None
        ui.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)))
        ui.handle_event(pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=-3, flipped=False, precise_x=0.0,
                                           precise_y=-3.0, touch=False, which=0))
        ui.build(make_ctx(ui, size=(1280, 720)))
    assert 0.0 < ui.scroll_offset <= ui.scroll_max + 1e-6


def test_toasts_expire(ui):
    now = 1000.0
    ui.build(make_ctx(ui, now=now))
    base = len(ui.build(make_ctx(ui, now=now)))
    ui.notify("Calibration saved", "info", 1.0)
    shown = len(ui.build(make_ctx(ui, now=now + 0.2)))
    assert shown > base
    for i in range(1, 60):
        ui.build(make_ctx(ui, now=now + 0.2 + i * 0.1))
    assert len(ui.build(make_ctx(ui, now=now + 8.0))) <= base + 2


def test_build_is_fast(ui):
    ctx = make_ctx(ui)
    for _ in range(10):
        ui.build(ctx)
    t = time.perf_counter()
    for _ in range(50):
        ui.build(make_ctx(ui))
    ms = (time.perf_counter() - t) / 50 * 1000
    print(f"HUD build: {ms:.2f} ms")
    assert ms < 15.0


def test_wants_mouse_over_panel(ui):
    ui.toggle_settings()
    ui.build(make_ctx(ui))
    panel = ui.widget("settings:panel")
    assert panel is not None
    ui.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(int(panel.center[0]), int(panel.center[1])),
                                       rel=(0, 0), buttons=(0, 0, 0)))
    assert ui.wants_mouse()
