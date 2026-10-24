"""The application: pygame window, main loop and the wiring of all subsystems.

Per frame (main thread)::

    events -> keyboard/mouse -> ManualInput / UI actions
    tracker.latest() -> GestureEngine.update() -> BodyState, GestureSignals, events
    CalibrationController.update()          (while calibrating)
    AbilityManager.update() -> EffectSystem -> RenderList
    UIController.build() -> UIDrawList
    Renderer.render(scene, effects, ui) -> pygame.display.flip()
    (virtual camera on: a second, offscreen renderer draws the same scene without HUD
     at the call resolution when a frame is due -> VirtualCamera.submit())

Camera capture and tracking run in their own threads (see camera.py / tracking.py), so
does the virtual camera sender (virtual_camera.py).
"""

from __future__ import annotations

import ctypes
import dataclasses
import logging
import os
import sys
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pygame

from dragon_ball_powers import APP_NAME, __version__, config
from dragon_ball_powers.abilities import AbilityManager
from dragon_ball_powers.calibration import CalibrationController
from dragon_ball_powers.camera import NO_CAMERA, CameraManager, SyntheticCamera, choose_camera_index, list_cameras
from dragon_ball_powers.config import Calibration, Settings, save_calibration, save_settings, set_setting
from dragon_ball_powers.datatypes import (
    BodyState,
    CameraInfo,
    CameraState,
    GestureSignals,
    ManualInput,
    TrackerStatus,
    TrackingFrame,
    VirtualCameraState,
    VirtualCameraStatus,
)
from dragon_ball_powers.effects import EffectSystem
from dragon_ball_powers.fonts import FontManager
from dragon_ball_powers.gestures import GestureEngine
from dragon_ball_powers.interfaces import AudioSink, NullAudio, Renderer, UIAction, UIContext
from dragon_ball_powers.primitives import (
    QualityPreset,
    RenderList,
    SceneInput,
    compute_view_transform,
    get_quality,
)
from dragon_ball_powers.ui import UIController
from dragon_ball_powers.virtual_camera import VirtualCamera, device_settings

if TYPE_CHECKING:
    from dragon_ball_powers.main import AppArgs

log = logging.getLogger(__name__)

DEFAULT_SCENE_SIZE = (1280, 720)
SCREENSHOTS_DIR = config.PROJECT_ROOT / "screenshots"
ABILITY_KEYS = {pygame.K_1: "ki_charge", pygame.K_2: "kamehameha", pygame.K_3: "energy_burst",
                pygame.K_KP1: "ki_charge", pygame.K_KP2: "kamehameha", pygame.K_KP3: "energy_burst"}
AIM_KEYS = {pygame.K_LEFT: (-1.0, 0.0), pygame.K_RIGHT: (1.0, 0.0),
            pygame.K_UP: (0.0, -1.0), pygame.K_DOWN: (0.0, 1.0)}
AIM_NAMES = {(-1.0, 0.0): "left", (1.0, 0.0): "right", (0.0, -1.0): "up", (0.0, 1.0): "down"}


# ------------------------------------------------------------------ OS helpers


def _prepare_windows_process() -> None:
    """Crisp rendering on scaled displays and our own taskbar identity (Windows only)."""
    if sys.platform != "win32":
        return
    os.environ.setdefault("SDL_WINDOWS_DPI_AWARENESS", "permonitorv2")
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - older Windows / already set
        try:
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(  # type: ignore[attr-defined]
            "DragonBallPowers.App")
    except Exception:  # noqa: BLE001
        pass


def _make_icon(size: int = 64) -> pygame.Surface:
    """A small glowing ki orb used as the window icon."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r = np.hypot(xx - size / 2 + 0.5, yy - size / 2 + 0.5) / (size / 2)
    glow = np.exp(-(r * 2.2) ** 2)
    core = np.clip(1.0 - r * 2.6, 0.0, 1.0) ** 0.6
    rgb = np.stack([0.25 * glow + core, 0.7 * glow + core, 1.0 * glow + core], axis=-1)
    alpha = np.clip(glow * 1.4, 0.0, 1.0)
    rgba = np.concatenate([np.clip(rgb, 0, 1), alpha[..., None]], axis=-1)
    data = (rgba * 255).astype(np.uint8)
    return pygame.image.frombuffer(data.tobytes(), (size, size), "RGBA").copy()


class _AudioProxy:
    """AudioSink that forwards to the current engine (a NullAudio until sound is available)."""

    def __init__(self) -> None:
        self.target: AudioSink = NullAudio()

    def play(self, name: str, volume: float = 1.0) -> None:
        self.target.play(name, volume)

    def loop(self, name: str, volume: float) -> None:
        self.target.loop(name, volume)

    def stop(self, name: str, fade_ms: int = 250) -> None:
        self.target.stop(name, fade_ms)


class _PygameWindow:
    """Thin wrapper over pygame's display + ``pygame.Window`` (fullscreen without losing GL)."""

    def __init__(self) -> None:
        self.window: pygame.Window | None = None

    def attach(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                self.window = pygame.Window.from_display_module()
            except Exception as exc:  # noqa: BLE001
                log.debug("pygame.Window unavailable (%s); using toggle_fullscreen", exc)
                self.window = None

    def set_fullscreen(self, fullscreen: bool) -> bool:
        try:
            if self.window is not None:
                if fullscreen:
                    self.window.set_fullscreen(desktop=True)
                else:
                    self.window.set_windowed()
            else:
                pygame.display.toggle_fullscreen()
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not switch fullscreen mode: %s", exc)
            return False

    @staticmethod
    def size() -> tuple[int, int]:
        w, h = pygame.display.get_window_size()
        return (max(1, int(w)), max(1, int(h)))


# -------------------------------------------------------------------- smoke test


class SmokeTest:
    """Scripted keyboard input that exercises every ability through the real pipeline.

    Used by ``--smoke-test SECONDS`` (and by the automated integration test). It drives
    the same ``ManualInput`` / action paths as a human pressing keys.
    """

    def __init__(self, app: "App", duration: float) -> None:
        self.app = app
        self.duration = max(1.0, duration)
        self.t0: float | None = None
        self.done: set[int] = set()
        self.screenshot_dir = os.environ.get("DBP_SMOKE_SCREENSHOTS")
        self.reopen_mark: int | None = None     # virtual camera frames before "vcam:on"
        # (time, action) ; times are scaled to the duration (designed for >= 16 s)
        self.script: list[tuple[float, str]] = [
            (0.4, "select:kamehameha"), (0.5, "aim:right"), (0.7, "hold"), (2.4, "shot:kame_charge"), (2.6, "release"),
            (2.85, "shot:kame_beam"), (5.4, "select:ki_charge"), (5.6, "aim:right"), (5.7, "hold"),
            (6.9, "release"), (7.05, "shot:ki_blast"), (8.3, "select:energy_burst"), (8.5, "press"),
            (8.62, "shot:burst"), (9.3, "aura:on"), (9.4, "color:gold"), (10.4, "shot:aura"),
            (10.6, "aura:off"), (10.7, "color:purple"), (10.9, "select:kamehameha"),
            (11.0, "aim:auto"), (11.1, "hold"), (12.6, "release"), (12.9, "shot:frontal"),
            (13.6, "quality:low"), (14.0, "settings"), (14.3, "shot:settings"), (14.5, "settings"),
            (14.6, "debug"), (14.9, "debug"), (15.0, "quality:auto"), (15.1, "color:blue"),
        ]
        if app.args.virtual_camera:
            # Video-call output requested: switch it off and on again once (the device is
            # re-opened, the kept offscreen renderer reused) while a charge is running.
            self.script += [(11.4, "vcam:off"), (11.7, "vcam:on")]
            self.script.sort(key=lambda item: item[0])
        scale = min(1.0, self.duration / 16.0)
        self.script = [(t * scale, a) for t, a in self.script]

    def step(self, now: float) -> None:
        if self.t0 is None:
            self.t0 = now
        t = now - self.t0
        for i, (when, action) in enumerate(self.script):
            if i not in self.done and t >= when:
                self.done.add(i)
                self._do(action, now)
        if t >= self.duration:
            self.app.request_quit()

    def _do(self, action: str, now: float) -> None:
        app = self.app
        kind, _, arg = action.partition(":")
        log.info("smoke-test: %s", action)
        if kind == "select":
            app.apply_action(UIAction("select_ability", arg), now)
        elif kind == "hold":
            app.manual.trigger_held = True
            app.manual.trigger_pressed = True
        elif kind == "release":
            app.manual.trigger_held = False
            app.manual.trigger_released = True
        elif kind == "press":
            app.manual.trigger_pressed = True
        elif kind == "aim":
            app.manual.aim = {"right": (1.0, 0.0), "left": (-1.0, 0.0)}.get(arg, (0.0, 0.0))
        elif kind == "aura":
            if app.abilities.hud(now).aura_enabled != (arg == "on"):
                app.apply_action(UIAction("toggle_aura"), now)
        elif kind == "color":
            app.apply_action(UIAction("set_setting", ("effects.energy_color", arg)), now)
        elif kind == "quality":
            app.apply_action(UIAction("set_setting", ("effects.quality", arg)), now)
        elif kind == "settings":
            app.apply_action(UIAction("toggle_settings"), now)
        elif kind == "debug":
            app.apply_action(UIAction("toggle_debug"), now)
        elif kind == "vcam":
            on = arg == "on"
            if on:
                self.reopen_mark = app._vcam_submitted
            if app.settings.virtual_camera.enabled != on:
                app.apply_action(UIAction("toggle_virtual_camera"), now)
        elif kind == "shot" and self.screenshot_dir and app.using_test_pattern:
            app.save_screenshot(Path(self.screenshot_dir) / f"{arg}.png", announce=False)


# --------------------------------------------------------------------------- app


class App:
    def __init__(self, args: "AppArgs", settings: Settings) -> None:
        self.args = args
        self.settings = settings
        self.manual = ManualInput()
        self.fonts: FontManager | None = None
        self.renderer: Renderer | None = None
        self.camera: CameraManager | SyntheticCamera | None = None
        self.tracker = None
        self.gestures: GestureEngine | None = None
        self.calibration = CalibrationController()
        self.effects: EffectSystem | None = None
        self.abilities: AbilityManager | None = None
        self.ui: UIController | None = None
        self.audio = _AudioProxy()
        self._audio_engine = None
        self.vcam: VirtualCamera | None = None          # video-call output (created on first use)
        self._vcam_renderer: Renderer | None = None     # offscreen renderer for its picture
        self._vcam_state = VirtualCameraState.OFF       # last state seen (announced on change)
        self._vcam_errors = 0                           # consecutive failures rendering its picture
        self._vcam_submitted = 0                        # pictures handed to the virtual camera
        self._vcam_saved_enabled: bool | None = None    # saved choice while --virtual-camera overrides it
        self.quality: QualityPreset = get_quality("high")
        self.render_list = RenderList()
        self.using_test_pattern = bool(args.test_pattern)

        self._window = _PygameWindow()
        self._window_size = (1280, 720)
        self._fullscreen = False
        self._using_gl = False
        self._running = True
        self._exit_code = 0
        self._t0 = time.perf_counter()
        self._fps = 0.0
        self._frames = 0
        self._render_errors = 0
        self._consecutive_render_errors = 0

        self._latest_tf: TrackingFrame | None = None
        self._last_tf_id = -1
        self._body: BodyState | None = None
        self._signals = GestureSignals()
        self._scene_size = DEFAULT_SCENE_SIZE
        self._tracker_error: str | None = None

        self._cameras: list[CameraInfo] = []
        self._cameras_lock = threading.Lock()
        self._scanning = False
        self._camera_notice: tuple[str, str, float] | None = None   # from the scan thread, shown by _draw
        self._settings_dirty = False
        self._last_save = 0.0
        self._smoke = SmokeTest(self, args.smoke_test) if args.smoke_test else None

    # ------------------------------------------------------------------ lifecycle

    def run(self) -> int:
        self._apply_cli_overrides()
        try:
            self._init_display()
            self._init_subsystems()
            self._start_camera()
            if self.settings.virtual_camera.enabled:
                self._start_virtual_camera()
            log.info("Ready: renderer=%s quality=%s window=%sx%s", self.renderer.name if self.renderer else "?",
                     self.quality.name, *self._window_size)
            self._loop()
        finally:
            self._shutdown()
        if self._smoke is not None:
            ok = self._frames > 10 and self._render_errors == 0
            if self.args.virtual_camera:
                # The requested output must have worked, also after being switched off and on.
                ok = ok and self._vcam_submitted > (self._smoke.reopen_mark or 0)
            log.info("Smoke test finished: %d frames, %.1f FPS average, %d render errors, "
                     "%d virtual camera frames -> %s",
                     self._frames, self._frames / max(1e-6, time.perf_counter() - self._t0),
                     self._render_errors, self._vcam_submitted, "PASS" if ok else "FAIL")
            return 0 if ok else 1
        return self._exit_code

    def request_quit(self) -> None:
        self._running = False

    def _apply_cli_overrides(self) -> None:
        if self.args.camera is not None:
            self.settings.camera.index = max(0, int(self.args.camera))
            self.settings.camera.name_hint = ""
        if self.args.quality:
            self.settings.effects.quality = self.args.quality
        if self.args.no_audio:
            self.settings.audio.enabled = False
        if self.args.debug:
            self.settings.ui.show_debug = True
        if self.args.fullscreen is not None:
            self.settings.ui.fullscreen = self.args.fullscreen
        if self.args.virtual_camera:
            # For this run only: the saved choice is written back (see _maybe_save_settings)
            # unless the user switches the output themselves.
            self._vcam_saved_enabled = self.settings.virtual_camera.enabled
            self.settings.virtual_camera.enabled = True

    # ------------------------------------------------------------------ display

    def _init_display(self) -> None:
        _prepare_windows_process()
        pygame.mixer.pre_init(44100, -16, 2, 512)
        pygame.init()
        pygame.display.set_caption(f"{APP_NAME} {__version__}")
        try:
            pygame.display.set_icon(_make_icon())
        except pygame.error:
            pass
        self.fonts = FontManager()
        size = self._initial_window_size()
        kind = self.args.renderer
        if kind in ("auto", "gl"):
            try:
                self._create_window(size, gl=True)
                self._create_gl_renderer()
            except Exception as exc:  # noqa: BLE001 - any GL failure -> CPU fallback
                if kind == "gl":
                    raise
                log.warning("OpenGL renderer unavailable (%s); falling back to the CPU renderer", exc)
                self._create_window(size, gl=False)
                self._create_cpu_renderer()
        else:
            self._create_window(size, gl=False)
            self._create_cpu_renderer()
        if self.settings.ui.fullscreen:
            self._set_fullscreen(True)
        pygame.key.set_repeat()  # no key repeat: holding SPACE must not re-trigger

    def _initial_window_size(self) -> tuple[int, int]:
        w, h = self.settings.ui.window_width, self.settings.ui.window_height
        try:
            desk_w, desk_h = pygame.display.get_desktop_sizes()[0]
            if w > desk_w * 0.92 or h > desk_h * 0.92:
                scale = min(desk_w * 0.85 / w, desk_h * 0.85 / h)
                w, h = int(w * scale), int(h * scale)
        except (pygame.error, IndexError):
            pass
        return (max(640, w), max(360, h))

    def _create_window(self, size: tuple[int, int], gl: bool) -> None:
        flags = pygame.RESIZABLE
        if gl:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            if sys.platform == "darwin":
                pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, True)
            flags |= pygame.OPENGL | pygame.DOUBLEBUF
        try:
            pygame.display.set_mode(size, flags, vsync=1 if (gl and self.settings.ui.vsync) else 0)
        except pygame.error as exc:
            if not gl:
                raise
            log.info("VSync request failed (%s); continuing without vsync", exc)
            pygame.display.set_mode(size, flags)
        self._using_gl = gl
        self._window.attach()
        self._window_size = self._window.size()

    def _create_gl_renderer(self) -> None:
        import moderngl

        from dragon_ball_powers.renderer import GLRenderer

        ctx = moderngl.create_context(require=330)
        self.renderer = GLRenderer(ctx, self._window_size, get_quality("high"), self.fonts, self.settings.effects)
        self.quality = self._resolve_quality()
        self.renderer.set_quality(self.quality)
        info = self.renderer.info
        log.info("OpenGL renderer: %s | %s | %s", info.get("gl_vendor"), info.get("gl_renderer"),
                 info.get("gl_version"))

    def _create_cpu_renderer(self) -> None:
        from dragon_ball_powers.renderer_cpu import CpuRenderer

        if self.renderer is not None:
            self.renderer.release()
        self.renderer = CpuRenderer(self._window_size, get_quality("low"), self.fonts, self.settings.effects,
                                    target=pygame.display.get_surface())
        self.quality = self._resolve_quality()
        self.renderer.set_quality(self.quality)

    def _resolve_quality(self) -> QualityPreset:
        name = self.settings.effects.quality
        if name != "auto":
            return get_quality(name)
        if not self._using_gl:
            return get_quality("low")
        gpu = str(self.renderer.info.get("gl_renderer", "") if self.renderer else "").lower()
        if any(k in gpu for k in ("nvidia", "geforce", "rtx", "quadro", "radeon", "amd")):
            return get_quality("ultra")
        if "arc" in gpu:
            return get_quality("high")
        if "intel" in gpu or "uhd" in gpu or "iris" in gpu:
            return get_quality("medium")
        return get_quality("high")

    def _set_fullscreen(self, fullscreen: bool) -> None:
        if fullscreen == self._fullscreen:
            return
        if self._window.set_fullscreen(fullscreen):
            self._fullscreen = fullscreen
            self.settings.ui.fullscreen = fullscreen
            self._mark_dirty()
        self._on_resize(self._window.size())

    def _on_resize(self, size: tuple[int, int]) -> None:
        size = (max(1, int(size[0])), max(1, int(size[1])))
        if size == self._window_size and self.renderer is not None:
            return
        self._window_size = size
        if self.renderer is None:
            return
        if not self._using_gl:
            # The display surface may have been re-created by SDL: rebuild the CPU renderer on it.
            self._create_cpu_renderer()
        else:
            self.renderer.resize(*size)
        if not self._fullscreen:
            self.settings.ui.window_width, self.settings.ui.window_height = size
            self._mark_dirty()

    # --------------------------------------------------------------- subsystems

    def _init_subsystems(self) -> None:
        s = self.settings
        calibration = config.load_calibration()
        if calibration.valid:
            log.info("Using calibration from %s", calibration.created_at or "an earlier session")
        self.gestures = GestureEngine(s.gestures, calibration if calibration.valid else None)

        self.effects = EffectSystem(s.effects, self.quality, self._scene_size)
        self._init_audio()
        self.abilities = AbilityManager(s, self.effects, self.audio)
        self.abilities.select(s.abilities.selected, time.perf_counter())
        self.ui = UIController(s, self.fonts)

        if self.args.test_pattern:
            self.camera = SyntheticCamera(s.camera)
        else:
            self.camera = CameraManager(s.camera)
        try:
            from dragon_ball_powers.tracking import HandPoseTracker

            self.tracker = HandPoseTracker(s.tracking, config.MODELS_DIR, mask_width=self.quality.aura_resolution)
            self.tracker.start(self.camera)
        except Exception as exc:  # noqa: BLE001 - keep the app usable with keyboard controls
            log.error("Hand/body tracking unavailable: %s", exc, exc_info=True)
            self.tracker = None
            self._tracker_error = f"Tracking unavailable: {exc}"

    def _init_audio(self) -> None:
        """Create the sound engine (if enabled). Abilities always talk to ``self.audio``,
        a proxy, so sound can be switched on later without re-wiring anything."""
        if not self.settings.audio.enabled or self._audio_engine is not None:
            return
        try:
            from dragon_ball_powers.audio import AudioEngine

            engine = AudioEngine(self.settings.audio)
            if engine.start():
                self._audio_engine = engine
                self.audio.target = engine
        except Exception as exc:  # noqa: BLE001
            log.warning("Audio disabled: %s", exc)

    def _start_camera(self) -> None:
        if self.args.test_pattern:
            self.camera.start(0)
            return
        if self.settings.camera.auto_start:
            self._scan_cameras(start_after=True)
        else:
            self._scan_cameras(start_after=False)

    def _scan_cameras(self, start_after: bool) -> None:
        if self._scanning:
            return
        self._scanning = True

        def worker() -> None:
            cameras: list[CameraInfo] = []
            try:
                cameras = list_cameras(backend=self.settings.camera.backend)
            except Exception as exc:  # noqa: BLE001
                log.warning("Camera enumeration failed: %s", exc)
            status = self.camera.status if self.camera else None
            if status is not None and status.state in (CameraState.RUNNING, CameraState.STARTING) \
                    and status.index >= 0 and all(c.index != status.index for c in cameras):
                # The active camera is busy (opened by us) and may be missing from a probe.
                cameras.append(CameraInfo(status.index, status.name or f"Camera {status.index}"))
                cameras.sort(key=lambda c: c.index)
            with self._cameras_lock:
                self._cameras = cameras
            log.info("Cameras: %s", ", ".join(c.label + (" (IR)" if c.is_ir else " (virtual)" if c.is_virtual else "")
                                              for c in cameras) or "none")
            if start_after and self._running:
                index = choose_camera_index(cameras, self.settings.camera)
                if self.args.camera is not None:
                    index = self.args.camera
                if index == NO_CAMERA:
                    # Only virtual outputs (e.g. "OBS Virtual Camera", maybe our own): shown on
                    # the main thread, the camera stays off until the user picks one.
                    self._camera_notice = ("No webcam found, only virtual cameras such as 'OBS Virtual "
                                           "Camera'. Connect your webcam, then pick it in Tab → Camera.",
                                           "warning", 10.0)
                else:
                    name = next((c.name for c in cameras if c.index == index), None)
                    self.camera.start(index, name)
            self._scanning = False

        threading.Thread(target=worker, name="camera-scan", daemon=True).start()

    # ---------------------------------------------------------------- main loop

    def _loop(self) -> None:
        clock = pygame.time.Clock()
        last = time.perf_counter()
        while self._running:
            now = time.perf_counter()
            dt = min(0.25, max(0.0, now - last))
            last = now
            if dt > 0:
                self._fps = 1.0 / dt if self._fps == 0 else self._fps + (1.0 / dt - self._fps) * 0.08
            self._process_events(now)
            if self._smoke is not None:
                self._smoke.step(now)
            self._update(now, dt)
            self._draw(now)
            self._frames += 1
            self._maybe_save_settings(now)
            clock.tick(self.settings.ui.max_fps)

    def _process_events(self, now: float) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type in (pygame.VIDEORESIZE, pygame.WINDOWSIZECHANGED):
                self._on_resize(self._window.size())
            elif event.type == pygame.KEYDOWN:
                self._on_key_down(event, now)
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE and self.manual.trigger_held:
                    self.manual.trigger_held = False
                    self.manual.trigger_released = True
            elif event.type in (pygame.MOUSEMOTION, pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                                pygame.MOUSEWHEEL):
                for action in self.ui.handle_event(event):
                    self.apply_action(action, now)
            elif event.type == pygame.WINDOWFOCUSLOST and self.manual.trigger_held:
                # Never leave a charge "stuck" when the window loses the keyboard.
                self.manual.trigger_held = False
                self.manual.trigger_released = True

    def _on_key_down(self, event: pygame.event.Event, now: float) -> None:
        key = event.key
        if key == pygame.K_SPACE:
            if not self.manual.trigger_held:
                self.manual.trigger_held = True
                self.manual.trigger_pressed = True
        elif key in ABILITY_KEYS:
            self.apply_action(UIAction("select_ability", ABILITY_KEYS[key]), now)
        elif key in (pygame.K_4, pygame.K_KP4):
            self.apply_action(UIAction("toggle_aura"), now)
        elif key in (pygame.K_f, pygame.K_F11):
            self.apply_action(UIAction("toggle_fullscreen"), now)
        elif key == pygame.K_c:
            self.apply_action(UIAction("recalibrate"), now)
        elif key == pygame.K_TAB:
            self.apply_action(UIAction("toggle_settings"), now)
        elif key == pygame.K_d:
            self.apply_action(UIAction("toggle_debug"), now)
        elif key == pygame.K_v:
            self.apply_action(UIAction("toggle_virtual_camera"), now)
        elif key == pygame.K_h:
            self.settings.ui.show_hud = not self.settings.ui.show_hud
            self._mark_dirty()
        elif key in AIM_KEYS:
            self.manual.aim = AIM_KEYS[key]
            self.ui.notify(f"Keyboard aim: {AIM_NAMES[self.manual.aim]} (0 = automatic)", duration=1.5)
        elif key in (pygame.K_0, pygame.K_KP0):
            self.manual.aim = (0.0, 0.0)
            self.ui.notify("Keyboard aim: automatic", duration=1.5)
        elif key in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            step = 0.05 if key == pygame.K_RIGHTBRACKET else -0.05
            value = set_setting(self.settings, "gestures.sensitivity", self.settings.gestures.sensitivity + step)
            self._on_setting_changed("gestures.sensitivity")
            self.ui.notify(f"Gesture sensitivity {value * 100:.0f}%", duration=1.5)
        elif key == pygame.K_F12:
            self.save_screenshot()
        elif key == pygame.K_ESCAPE:
            if self.calibration.active:
                self.calibration.cancel()
                self.ui.notify("Calibration cancelled", "warning")
            elif self.ui.settings_open:
                self.ui.toggle_settings()
            elif self._fullscreen:
                self._set_fullscreen(False)
            else:
                self._running = False

    # -------------------------------------------------------------- UI actions

    def apply_action(self, action: UIAction, now: float) -> None:
        kind, value = action.kind, action.value
        if kind not in ("set_setting",):
            self.audio.play("ui_click", 0.5)
        if kind == "select_camera":
            index = int(value)
            self.settings.camera.index = index
            with self._cameras_lock:
                name = next((c.name for c in self._cameras if c.index == index), "")
            self.settings.camera.name_hint = name
            self._mark_dirty()
            if not self.using_test_pattern:
                self.camera.stop()
                self.camera.start(index, name or None)
        elif kind == "rescan_cameras":
            self._scan_cameras(start_after=False)
        elif kind == "toggle_camera":
            if self.camera.status.state in (CameraState.RUNNING, CameraState.STARTING):
                self.camera.stop()
            else:
                self.camera.start(self.settings.camera.index, self.settings.camera.name_hint or None)
        elif kind == "set_resolution":
            w, h = value
            set_setting(self.settings, "camera.width", int(w))
            set_setting(self.settings, "camera.height", int(h))
            self._on_setting_changed("camera.width")
        elif kind == "set_setting":
            dotted, new_value = value
            try:
                set_setting(self.settings, dotted, new_value)
            except KeyError:
                log.warning("UI requested unknown setting %s", dotted)
                return
            self._on_setting_changed(dotted)
        elif kind == "select_ability":
            self.abilities.select(str(value), now)
            self.settings.abilities.selected = str(value)
            self._mark_dirty()
        elif kind == "toggle_aura":
            self.settings.abilities.aura_enabled = bool(self.abilities.toggle_aura(now))
            self._mark_dirty()
        elif kind == "recalibrate":
            if self.calibration.active:
                self.calibration.cancel()
            else:
                self.calibration.start(now)
                if self.ui.settings_open:
                    self.ui.toggle_settings()
        elif kind == "toggle_fullscreen":
            self._set_fullscreen(not self._fullscreen)
        elif kind == "toggle_settings":
            self.ui.toggle_settings()
        elif kind == "toggle_debug":
            self.settings.ui.show_debug = not self.settings.ui.show_debug
            self._mark_dirty()
        elif kind == "toggle_virtual_camera":
            self._set_virtual_camera(not self.settings.virtual_camera.enabled)
        elif kind == "set_virtual_camera_resolution":
            w, h = value
            set_setting(self.settings, "virtual_camera.width", int(w))
            set_setting(self.settings, "virtual_camera.height", int(h))
            self._on_setting_changed("virtual_camera.width")
        elif kind == "reset_settings":
            self._reset_settings(now)
        elif kind == "quit":
            self._running = False
        else:
            log.warning("Unknown UI action %r", kind)

    def _on_setting_changed(self, dotted: str) -> None:
        section = dotted.partition(".")[0]
        s = self.settings
        self._mark_dirty()
        if section == "camera" and not self.using_test_pattern:
            self.camera.restart(s.camera)
        elif section == "tracking" and self.tracker is not None:
            self.tracker.update_settings(s.tracking)
        elif section == "gestures":
            self.gestures.set_settings(s.gestures)
            self.abilities.apply_settings(s)
        elif section == "effects":
            if dotted == "effects.quality":
                self._apply_quality()
            self.effects.apply_settings(s.effects)
            self.abilities.apply_settings(s)
        elif section == "abilities":
            self.abilities.apply_settings(s)
        elif section == "audio":
            self._apply_audio_settings()
        elif section == "virtual_camera":
            if dotted == "virtual_camera.enabled":
                self._set_virtual_camera(s.virtual_camera.enabled)
            else:
                self._restart_virtual_camera_if_changed()     # mirror is read every frame

    def _apply_quality(self) -> None:
        self.quality = self._resolve_quality()
        self.renderer.set_quality(self.quality)
        if self._vcam_renderer is not None:
            self._vcam_renderer.set_quality(self.quality)
        self.effects.set_quality(self.quality)
        if self.tracker is not None and hasattr(self.tracker, "set_mask_width"):
            self.tracker.set_mask_width(self.quality.aura_resolution)
        log.info("Effects quality: %s", self.quality.name)

    def _apply_audio_settings(self) -> None:
        if self._audio_engine is not None:
            self._audio_engine.apply_settings(self.settings.audio)
        elif self.settings.audio.enabled:
            self._init_audio()  # sound was off at start-up: create the engine now

    def _reset_settings(self, now: float) -> None:
        old_camera = self.settings.camera
        fresh = Settings()
        fresh.ui.fullscreen = self._fullscreen
        fresh.ui.window_width, fresh.ui.window_height = self._window_size
        # Like fullscreen, a running video-call output stays on (it may be in a call right now).
        fresh.virtual_camera.enabled = self.settings.virtual_camera.enabled
        for section in config.SECTION_TYPES:
            setattr(self.settings, section, getattr(fresh, section))
        self._restart_virtual_camera_if_changed()
        self.gestures.set_settings(self.settings.gestures)
        self.effects.apply_settings(self.settings.effects)
        self._apply_quality()
        self.abilities.apply_settings(self.settings)
        self.abilities.select(self.settings.abilities.selected, now)
        if self.tracker is not None:
            self.tracker.update_settings(self.settings.tracking)
        self._apply_audio_settings()
        if not self.using_test_pattern and old_camera != self.settings.camera:
            self.camera.restart(self.settings.camera)
        self._mark_dirty()
        self.ui.notify("Settings reset to defaults")

    # ------------------------------------------------------------------ update

    def _update(self, now: float, dt: float) -> None:
        tf = self.tracker.latest() if self.tracker is not None else None
        new_tf = None
        if tf is not None and tf.frame_id != self._last_tf_id:
            self._last_tf_id = tf.frame_id
            self._latest_tf = tf
            new_tf = tf
        if self._latest_tf is not None and now - self._latest_tf.result_time > 1.0:
            self._latest_tf = None  # stale (camera stopped / tracker stalled)

        body, signals, events = self.gestures.update(new_tf, now)
        self._body, self._signals = body, signals

        scene_size = self._current_scene_size()
        if scene_size != self._scene_size:
            self._scene_size = scene_size
            self.effects.set_scene_size(scene_size)

        if self.calibration.active:
            result = self.calibration.update(now, body, signals)
            if result is not None:
                self._on_calibrated(result)
            # Gestures are used for calibration only; keyboard controls keep working.
            signals = GestureSignals(tracking_ok=signals.tracking_ok)
            events = []

        self.abilities.update(now, dt, self._latest_tf, body, signals, events, self.manual)
        self.manual.trigger_pressed = False
        self.manual.trigger_released = False
        self.effects.update(dt, now)

    def _on_calibrated(self, calibration: Calibration) -> None:
        try:
            save_calibration(calibration)
        except OSError as exc:
            log.error("Could not save calibration: %s", exc)
        self.gestures.set_calibration(calibration)
        log.info("Calibration finished: %s", calibration)

    def _current_scene_size(self) -> tuple[int, int]:
        if self._latest_tf is not None:
            return self._latest_tf.scene_size
        frame = self.camera.latest() if self.camera else None
        if frame is not None and self.camera.status.state is CameraState.RUNNING:
            return frame.size
        return self._scene_size

    def _scene_input(self, now: float) -> SceneInput:
        status = self.camera.status if self.camera else None
        camera_live = status is not None and status.state is CameraState.RUNNING
        tf = self._latest_tf
        if camera_live and self.settings.camera.sync_video_to_tracking and tf is not None and tf.image is not None:
            return SceneInput(tf.image, tf.frame_id, tf.scene_size, now - self._t0)
        frame = self.camera.latest() if camera_live else None
        if frame is not None:
            return SceneInput(frame.image, frame.frame_id, frame.size, now - self._t0)
        return SceneInput(None, -1, self._scene_size, now - self._t0)

    # -------------------------------------------------------------------- draw

    def _draw(self, now: float) -> None:
        notice, self._camera_notice = self._camera_notice, None
        if notice is not None:
            self.ui.notify(*notice)
        scene = self._scene_input(now)
        self.render_list.clear()
        self.effects.collect(self.render_list)
        win_w, win_h = self._window_size
        view = compute_view_transform(scene.scene_size[0], scene.scene_size[1], win_w, win_h,
                                      self.settings.ui.fit_mode)
        with self._cameras_lock:
            cameras = list(self._cameras)
        tracker_status = self.tracker.status if self.tracker is not None else TrackerStatus(
            error=self._tracker_error)
        ctx = UIContext(
            now=now,
            window_size=self._window_size,
            view=view,
            settings=self.settings,
            camera_status=self.camera.status,
            tracker_status=tracker_status,
            ability=self.abilities.hud(now),
            calibration=self.calibration.hud(now),
            cameras=cameras,
            camera_scanning=self._scanning,
            body=self._body,
            signals=self._signals,
            frame=self._latest_tf,
            render_fps=self._fps,
            renderer_name=self.renderer.name,
            quality_name=self.quality.name,
            fullscreen=self._fullscreen,
            virtual_camera=self._poll_virtual_camera(),
        )
        draw_list = self.ui.build(ctx)
        try:
            self.renderer.render(scene, self.render_list, draw_list, view)
            self._consecutive_render_errors = 0
        except Exception:  # noqa: BLE001
            self._render_errors += 1
            self._consecutive_render_errors += 1
            log.exception("Rendering failed (%d in a row)", self._consecutive_render_errors)
            if self._consecutive_render_errors >= 3 and self._using_gl:
                log.error("Switching to the CPU renderer after repeated OpenGL errors")
                self._switch_to_cpu_renderer()
            elif self._consecutive_render_errors >= 30:
                raise
        pygame.display.flip()
        # After the flip: the window frame is not delayed by the call picture's read-back.
        self._render_virtual_camera(now, scene, ctx)

    def _switch_to_cpu_renderer(self) -> None:
        self._release_vcam_renderer()        # its GL objects live in the window's context
        try:
            if self.renderer is not None:
                self.renderer.release()
        except Exception:  # noqa: BLE001
            pass
        self.renderer = None
        self._create_window(self._window_size, gl=False)
        self._create_cpu_renderer()
        self._apply_quality()
        self._consecutive_render_errors = 0
        self.ui.notify("OpenGL failed - switched to the CPU renderer", "warning", 5.0)

    # -------------------------------------------------------- virtual camera

    def _set_virtual_camera(self, on: bool) -> None:
        """Switch the video-call output on or off (remembered in the settings)."""
        self.settings.virtual_camera.enabled = bool(on)
        self._vcam_saved_enabled = None           # the user's own choice is saved as it is
        self._mark_dirty()
        if on:
            self._start_virtual_camera()
            self.ui.notify("Starting the virtual camera…", duration=2.0)
        else:
            self._stop_virtual_camera()
            self.ui.notify("Virtual camera off", duration=2.0)

    def _start_virtual_camera(self) -> None:
        """(Re)open the device with the current settings; it opens in its own thread."""
        if self.vcam is None:
            self.vcam = VirtualCamera(self.settings.virtual_camera)
            self.vcam.start()
        else:
            self.vcam.start(self.settings.virtual_camera)
        self._vcam_errors = 0
        # A new attempt: its outcome must be announced even if the last one failed the same way.
        self._vcam_state = VirtualCameraState.STARTING

    def _restart_virtual_camera_if_changed(self) -> None:
        """Re-open a running output when its size, rate or driver changed (not for mirror:
        a needless re-open would interrupt the call)."""
        vcam = self.vcam
        if vcam is not None and vcam.status.active and \
                device_settings(vcam.settings) != device_settings(self.settings.virtual_camera):
            self._start_virtual_camera()

    def _virtual_camera_failed(self) -> None:
        """Remember the output as off: no automatic retry at the next start (V tries again)."""
        self.settings.virtual_camera.enabled = False
        self._vcam_saved_enabled = None
        self._mark_dirty()

    def _stop_virtual_camera(self, release_renderer: bool = False) -> None:
        if self.vcam is not None:
            self.vcam.stop()
        if release_renderer:
            self._release_vcam_renderer()
        # (otherwise kept: switching on again then needs no new shaders, i.e. no hitch)
        self._vcam_state = VirtualCameraState.OFF

    def _release_vcam_renderer(self) -> None:
        renderer, self._vcam_renderer = self._vcam_renderer, None
        if renderer is not None:
            try:
                renderer.release()
            except Exception as exc:  # noqa: BLE001
                log.debug("Releasing the virtual camera renderer failed: %s", exc)

    def _poll_virtual_camera(self) -> VirtualCameraStatus | None:
        """The status for the HUD; announces "live" and failures once."""
        if self.vcam is None:
            return None
        status = self.vcam.status
        if status.state is not self._vcam_state:
            self._vcam_state = status.state
            if status.state is VirtualCameraState.RUNNING:
                # Several cameras fed together are listed as "A / B" (virtual_camera.MultiCamera).
                device = status.device or "OBS Virtual Camera"
                names = " or ".join(f"'{name}'" for name in device.split(" / "))
                apps = "WhatsApp, Discord, Zoom, Teams…" if "Windows 11" in device else \
                    "Discord, Zoom, Teams, Google Meet…"
                self.ui.notify(f"Virtual camera live: choose {names} as the camera in {apps}", duration=7.0)
            elif status.state is VirtualCameraState.ERROR:
                self.ui.notify(status.error or "The virtual camera failed", "error", 10.0)
                self._virtual_camera_failed()
        return status

    def _render_virtual_camera(self, now: float, scene: SceneInput, ctx: UIContext) -> None:
        """Render the picture for video calls (no HUD, call resolution) when one is due."""
        vcam = self.vcam
        if vcam is None or not vcam.frame_due(now):
            return
        try:
            size = vcam.size
            renderer = self._vcam_renderer_for(size)
            view = compute_view_transform(scene.scene_size[0], scene.scene_size[1], size[0], size[1], "cover")
            # People in the call should see the scene the right way round (callouts stay readable).
            # The test pattern is never mirrored by the camera module (its text would be unreadable).
            scene_mirrored = self.settings.camera.mirror and not self.using_test_pattern
            renderer.mirror_output = scene_mirrored != self.settings.virtual_camera.mirror
            overlay = self.ui.build_overlay(dataclasses.replace(ctx, window_size=size, view=view, fullscreen=True))
            renderer.render(scene, self.render_list, overlay, view)
            vcam.submit(renderer.read_pixels())
            self._vcam_submitted += 1
            self._vcam_errors = 0
        except Exception as exc:  # noqa: BLE001 - the video-call output must never break the app
            self._vcam_errors += 1
            log.exception("Rendering the virtual camera picture failed (%d in a row)", self._vcam_errors)
            self._release_vcam_renderer()
            if self._vcam_errors >= 3:
                self._stop_virtual_camera()
                self._virtual_camera_failed()
                self.ui.notify(f"Virtual camera stopped: {exc}", "error", 8.0)

    def _vcam_renderer_for(self, size: tuple[int, int]) -> Renderer:
        """The offscreen renderer of the virtual camera picture (same backend as the window)."""
        renderer = self._vcam_renderer
        if renderer is not None and getattr(renderer, "size", None) == size:
            return renderer
        self._release_vcam_renderer()
        if self._using_gl:
            from dragon_ball_powers.renderer import GLRenderer

            # Shares the window's context: no second context, textures stay on the GPU.
            renderer = GLRenderer(self.renderer.ctx, size, self.quality, self.fonts, self.settings.effects,
                                  offscreen=True)
        else:
            from dragon_ball_powers.renderer_cpu import CpuRenderer

            renderer = CpuRenderer(size, self.quality, self.fonts, self.settings.effects)
        log.info("Virtual camera picture: %s renderer, %dx%d", renderer.name, size[0], size[1])
        self._vcam_renderer = renderer
        return renderer

    # ------------------------------------------------------------- utilities

    def save_screenshot(self, path: Path | None = None, announce: bool = True) -> Path | None:
        """Save the last rendered frame (only on explicit user request, F12)."""
        if self.renderer is None:
            return None
        try:
            pixels = self.renderer.read_pixels()
            h, w = pixels.shape[:2]
            if path is None:
                SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
                path = SCREENSHOTS_DIR / f"ki-powers-{datetime.now():%Y%m%d-%H%M%S}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            surface = pygame.image.frombuffer(np.ascontiguousarray(pixels).tobytes(), (w, h), "RGB")
            pygame.image.save(surface, str(path))
            log.info("Screenshot saved: %s", path)
            if announce:
                self.ui.notify(f"Screenshot saved: {path.name}")
            return path
        except Exception as exc:  # noqa: BLE001
            log.error("Screenshot failed: %s", exc)
            if announce:
                self.ui.notify("Screenshot failed", "error")
            return None

    def _mark_dirty(self) -> None:
        self._settings_dirty = True

    def _maybe_save_settings(self, now: float, force: bool = False) -> None:
        if not self._settings_dirty or (not force and now - self._last_save < 1.0):
            return
        if self._smoke is not None:
            self._settings_dirty = False  # the automated test never changes the user's settings
            return
        settings = self.settings
        if self._vcam_saved_enabled is not None:
            settings = settings.copy()    # --virtual-camera applies to this run only
            settings.virtual_camera.enabled = self._vcam_saved_enabled
        try:
            save_settings(settings)
            self._settings_dirty = False
            self._last_save = now
        except OSError as exc:
            log.error("Could not save settings: %s", exc)
            self._last_save = now

    def _shutdown(self) -> None:
        log.info("Shutting down")
        for name, fn in (
            ("tracker", lambda: self.tracker.stop() if self.tracker else None),
            ("camera", lambda: self.camera.stop() if self.camera else None),
            ("virtual camera", lambda: self._stop_virtual_camera(release_renderer=True)),  # before the GL context
            ("audio", lambda: self._audio_engine.close() if self._audio_engine else None),
            ("renderer", lambda: self.renderer.release() if self.renderer else None),
        ):
            try:
                fn()
            except Exception:  # noqa: BLE001 - shut down as much as possible
                log.exception("Error while stopping the %s", name)
        self._maybe_save_settings(time.perf_counter(), force=True)
        try:
            pygame.quit()
        except Exception:  # noqa: BLE001
            pass
