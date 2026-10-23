# Dragon Ball Powers – Architecture & Module Contracts

This document is the contract between the modules of the application. Every module
is implemented against the shared foundation files listed below. If you change a
public API, update this document.

## 1. Data flow

```
 ┌────────────┐  CameraFrame   ┌──────────────┐ TrackingFrame ┌──────────────┐
 │ camera.py  │ ─────────────▶ │ tracking.py  │ ────────────▶ │ gestures.py  │
 │ capture    │ (mirrored BGR) │ MediaPipe    │ (smoothed     │ features +   │
 │ thread     │                │ hands ∥ pose │  landmarks,   │ detectors    │
 └────────────┘                │ thread       │  seg. mask)   └──────┬───────┘
                               └──────────────┘    BodyState, GestureSignals,
                                                   GestureEvents      │
 keyboard (ManualInput) ───────────────────────────────────┐          ▼
                                                            ▼  ┌──────────────┐
                                    state_machine.py ◀──── abilities.py      │
                                    (per ability)          │ Ki Charge,     │
                                                           │ Kamehameha,    │
                                                           │ Energy Burst,  │
                                                           │ Power Aura,    │
                                                           │ Power Level    │
                                                           └──────┬─────────┘
                                         EffectSystem API          │ AudioSink
                                                ▼                  ▼
                                        ┌──────────────┐    ┌────────────┐
                                        │ effects.py   │    │ audio.py   │
                                        │ simulation   │    │ procedural │
                                        └──────┬───────┘    └────────────┘
                                   RenderList  │
                       (scene space) ▼         │      UIDrawList (window space)
 ┌────────────────────────────────────────┐   │   ┌──────────────┐
 │ renderer.py (OpenGL, ModernGL)  or     │◀──┴───│ ui.py  HUD,  │
 │ renderer_cpu.py (NumPy/OpenCV fallback)│       │ settings     │
 └────────────────────────────────────────┘       └──────────────┘
                     ▲
                     └── app.py main loop (pygame window, events, wiring)

 video calls (optional): same SceneInput + RenderList ──▶ second, offscreen renderer
 (call resolution, mirror_output, callout-only UI) ──▶ virtual_camera.py ──▶ pyvirtualcam
 device ("OBS Virtual Camera") ──▶ Discord / Zoom / Teams / browser calls (Meet, FaceTime links)
```

Threads: **camera capture thread** → **tracker thread** (uses two inference worker
threads so hand and pose models run in parallel; MediaPipe releases the GIL) →
**main thread** (events, gestures, abilities, effects, rendering at display rate) →
optional **virtual camera sender thread** (opens the device, sends/repeats pictures).
Only immutable snapshots (`CameraFrame`, `TrackingFrame`, status dataclasses) cross
thread boundaries; each producer guards its "latest" slot with a lock.

## 2. Foundation files (shared, owned by the integrator – do not edit)

| File | Contents |
|------|----------|
| `config.py` | `Settings` sections, load/save/validate, `Calibration`, project paths (`MODELS_DIR`, `FONTS_DIR`, `CONFIG_DIR`, `LOGS_DIR`) |
| `datatypes.py` | camera/tracking/gesture/HUD/virtual-camera status dataclasses, landmark index enums |
| `primitives.py` | `RenderList` effect primitives, palettes, quality presets, UI draw commands, `ViewTransform`, `SceneInput` |
| `interfaces.py` | `Renderer` ABC, `AudioSink` protocol + `NullAudio`, `SOUND_NAMES`, `FrameSource`, `UIAction`, `UIContext` |
| `state_machine.py` | `AttackStateMachine` (IDLE → PREPARING → CHARGING → FIRING → COOLDOWN) |
| `mathutil.py` | clamp/lerp/smoothstep/damp, 2D vector helpers |
| `fonts.py` | `FontManager` (bundled Rajdhani + Bangers fonts, system fallback, cached white RGBA text surfaces) |

## 3. Conventions

* **Scene space**: pixel coordinates of the mirrored camera frame (x right, y down).
  `scene_size = (w, h)` = camera frame size (default `(1280, 720)` without camera).
  Landmarks, gesture features and all effect primitives use scene space. The
  renderer maps scene → window with `ViewTransform` (`compute_view_transform`,
  `cover` crops to fill, `contain` letterboxes). UI uses **window pixels**.
* **Body scale (S)**: px length ≈ one shoulder width; all gesture distances and
  speeds are normalised by it (`hand_distance` in S, speeds in S/s). Effects size
  themselves from S and from `scene_h / 720` (the "scene scale").
* **Time**: `time.perf_counter()` seconds everywhere. Simulations take `dt` and `now`.
  Never use wall-clock time for logic.
* **Colours**: linear RGB floats; > 1.0 = HDR (blooms). UI colours are RGBA floats 0..1.
* **Mirroring**: the camera module mirrors frames (`camera.mirror`). Downstream code
  never mirrors again. Hand `side` = screen side. Use `PoseObservation.arm(side)`
  instead of raw LEFT/RIGHT pose indices (model labels are swapped on mirrored input).
* **Logging**: `log = logging.getLogger(__name__)`; no `print`. Errors that the user
  must see are put into status objects (`CameraStatus.error`, `TrackerStatus.error`)
  or returned, and are logged once (rate-limit repeated errors).
* **Errors**: never crash the main loop because of camera/tracking/audio failures.
  Thread top-levels catch exceptions, log them with traceback and set an error status.
* **Performance**: vectorise with NumPy in per-frame/per-particle code, no Python
  loops over particles, avoid per-frame large allocations where practical.
* **Style**: Python 3.12, `from __future__ import annotations`, type hints on public
  APIs, docstrings on classes and public methods, dataclasses for data. Moderate
  comments that explain *why*. Windows is the runtime target: use `pathlib`, no
  POSIX-only APIs, no hard-coded `/` paths.
* **Privacy**: everything runs locally. No network access at runtime. Never write
  camera images to disk unless the user explicitly asks (none of the modules do).

## 4. Development & test environment

* Runtime target: Windows 10/11, Python 3.12 venv at `<root>\.venv` (created by
  `setup.bat`), launched with `run.bat`. macOS 11+ and Linux use `setup.sh` / `run.sh` (bash 3.2
  compatible; same steps, private python-build-standalone copy in `.python/` when no
  Python 3.11/3.12 is installed). Intel Macs get jax/jaxlib 0.4.38 and moderngl 5.12
  through environment markers in requirements.txt (no newer Intel-Mac wheels).
* The code also runs on Linux (including WSL2) with the same pinned packages
  (mediapipe 0.10.21, numpy 1.26.4, opencv-contrib-python 4.11.0.86, pygame-ce 2.5.8,
  moderngl 5.13.0): `python -m venv .venv-linux && .venv-linux/bin/pip install -r requirements-dev.txt`.
* Run the tests from the project root: `python -m pytest` (or `python -m pytest tests/<file> -q`).
* OpenGL: headless contexts work on Linux/WSL through EGL
  (`moderngl.create_standalone_context(backend="egl")`; on WSL2 GPU-accelerated through
  D3D12). pygame can open GL "windows" with `SDL_VIDEODRIVER=offscreen`. Tests use the
  `gl_context` fixture (skips when unavailable).
* Without a webcam (CI, WSL) camera code is tested with fakes and tracking with the real
  models; tests that need the MediaPipe sample photos (`woman_hands.jpg`, `pose.jpg`, …)
  read them from `/tmp/dbp-samples/` and skip when they are missing. Never commit those
  images or any camera footage.
* Code must stay Windows-correct even when it is developed elsewhere (pathlib, no
  POSIX-only APIs, CRLF batch files).

## 5. Module contracts

### 5.1 camera.py — capture thread and device enumeration

```python
def list_cameras(max_devices: int = 8, backend: str = "auto") -> list[CameraInfo]
def backend_id(name: str) -> int        # "auto" -> CAP_DSHOW on Windows, CAP_ANY elsewhere
class CameraManager:                     # implements FrameSource
    def __init__(self, settings: CameraSettings) -> None
    def start(self, index: int | None = None) -> None          # non-blocking (opens in thread)
    def stop(self) -> None                                     # idempotent, joins thread, releases device
    def restart(self, settings: CameraSettings | None = None) -> None
    def latest(self) -> CameraFrame | None
    def wait_for_frame(self, after_id: int, timeout: float) -> CameraFrame | None
    @property status -> CameraStatus                          # thread-safe snapshot
class SyntheticCamera:                   # same interface; animated diagnostic test pattern
```
* Windows: DirectShow (`CAP_DSHOW`) by default; request MJPG (`CAP_PROP_FOURCC`)
  before width/height/fps; `CAP_PROP_BUFFERSIZE=1`; read back the actual size/fps.
  MSMF optional (`camera.backend`). Device names via `pygrabber` (DirectShow order =
  OpenCV DSHOW indices) when installed, else probe indices and name them "Camera N".
* Mark IR cameras (`is_ir`, name matches `\bIR\b` or "infrared"); when the configured
  index is an IR camera and a colour camera exists, prefer the colour camera.
  `name_hint` re-finds a device by name if indices changed.
* First frame must arrive within ~5 s, else `ERROR` with an actionable message
  (camera busy in another app, Windows privacy setting, unplugged). Consecutive read
  failures for ~2 s → one reopen attempt → `ERROR` ("camera disconnected").
* Frames are mirrored (`cv2.flip(img, 1)`) when `settings.mirror`.
* `SyntheticCamera` exists only for diagnostics/automated tests (`--test-pattern`).

### 5.2 tracking.py (+ filters.py) — MediaPipe hands ∥ pose

```python
class HandPoseTracker:
    def __init__(self, settings: TrackingSettings, models_dir: Path = MODELS_DIR) -> None
    def start(self, source: FrameSource) -> None
    def stop(self) -> None
    def latest(self) -> TrackingFrame | None
    def process_frame(self, frame: CameraFrame) -> TrackingFrame   # synchronous, used by tests/tools
    def update_settings(self, settings: TrackingSettings) -> None
    @property status -> TrackerStatus
class OneEuroFilter:                      # filters.py, vectorised for arrays of any shape
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0)
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray
    def reset(self) -> None
```
* MediaPipe **Tasks API** (`mediapipe.tasks.python.vision.HandLandmarker` /
  `PoseLandmarker`), `RunningMode.VIDEO`, strictly increasing integer ms timestamps.
  Models: `hand_landmarker.task`, `pose_landmarker_{lite,full,heavy}.task` in `MODELS_DIR`.
  Missing model → `TrackerStatus.error` with "run setup.bat" hint; app keeps running.
* Frame is downscaled to `tracking_width` (aspect kept), converted to RGB. Hands and
  pose run concurrently (2 worker threads). Results → scene px via full frame size.
* Hand slots: two detections → sort by x; one detection → nearest recent slot, else
  handedness label; hysteresis against swapping. Missing hands coast for
  `hand_lost_grace` s (`source="predicted"`, damped velocity extrapolation, `age`).
* One Euro smoothing in normalised coordinates per slot (reset when a hand reappears).
* `PoseObservation.model_left_is_screen_left` from shoulder x order.
* Segmentation (if enabled): first mask → resized to the aura width (quality preset,
  default 256 wide) → uint8; increment `segmentation_id`.
* Stats: per-model ms, total ms, fps (EMA), latency; inference exceptions are
  logged rate-limited, landmarkers recreated after repeated failures.

### 5.3 gestures.py (+ calibration.py) — features, detectors, calibration

```python
class GestureEngine:
    def __init__(self, settings: GestureSettings, calibration: Calibration | None = None) -> None
    def update(self, frame: TrackingFrame | None, now: float) -> tuple[BodyState | None, GestureSignals, list[GestureEvent]]
    def set_settings(self, settings: GestureSettings) -> None
    def set_calibration(self, calibration: Calibration | None) -> None
    def reset(self) -> None
class CalibrationController:              # calibration.py
    def start(self, now: float) -> None
    def cancel(self) -> None
    active: bool
    def update(self, now: float, body: BodyState | None, signals: GestureSignals) -> Calibration | None
    def hud(self, now: float) -> CalibrationHUD
```
`update(frame, now)` is called once per new tracking frame, or with `frame=None` when
no new frame arrived (so tracking timeouts still produce `TRACKING_LOST`).
See §7 for the gesture design.

### 5.4 effects.py — effect simulation (produces RenderList primitives)

```python
class EffectSystem:
    def __init__(self, settings: EffectsSettings, quality: QualityPreset, scene_size: tuple[int, int], seed: int = 0)
    def set_scene_size(self, size: tuple[int, int]) -> None
    def set_quality(self, preset: QualityPreset) -> None
    def apply_settings(self, settings: EffectsSettings) -> None      # palette, multipliers, toggles
    palette: Palette                                                 # current energy palette
    scene_scale: float                                               # scene_h / 720
    particle_count: int
    def update(self, dt: float, now: float) -> None
    def collect(self, out: RenderList) -> None                       # append primitives for this frame
    def clear(self) -> None
    # steerable effects (handles are updated by abilities every tick)
    def create_orb(self, center: Vec2, radius: float) -> OrbEffect
    def create_beam(self, origin: Vec2, direction: Vec2, width: float, power: float) -> BeamEffect
    def create_frontal_beam(self, center: Vec2, radius: float, power: float) -> FrontalBeamEffect
    def create_aura(self) -> AuraEffect
    # fire-and-forget effects
    def ki_blast(self, origin: Vec2, direction: Vec2, frontal: float, radius: float, power: float) -> None
    def burst(self, center: Vec2, radius: float, power: float) -> None
    def shockwave(self, center: Vec2, radius: float, power: float, speed: float | None = None) -> None
    def flash(self, intensity: float, center: Vec2 | None = None, radius: float = 0.0, duration: float = 0.25) -> None
    def shake(self, amplitude: float, duration: float) -> None       # amplitude: fraction of scene height
    def sparks(self, center: Vec2, count: int, speed: float, direction: Vec2 | None = None, spread: float = math.tau, power: float = 1.0) -> None
    def fizzle(self, center: Vec2, radius: float) -> None             # cancelled charge dissipates
    def set_grade(self, darken: float, desaturate: float = 0.0) -> None  # request for this frame (max wins), smoothed
    def add_light(self, position: Vec2, radius: float, intensity: float) -> None  # one frame, palette colour

class OrbEffect:
    def set_target(self, center: Vec2, radius: float, charge: float, left_hand: Vec2 | None = None, right_hand: Vec2 | None = None) -> None
    def release(self) -> None        # consumed by an attack: bright collapse
    def dissipate(self) -> None      # fizzle out (cancelled charge)
    alive: bool; center: Vec2; radius: float; charge: float
class BeamEffect:
    def set(self, origin: Vec2, direction: Vec2, power: float | None = None) -> None
    def end(self) -> None            # dissipate then die
    alive: bool; progress: float     # 0..1 extension of the beam towards the screen edge
class FrontalBeamEffect:
    def set(self, center: Vec2, power: float | None = None) -> None
    def end(self) -> None
    alive: bool
class AuraEffect:
    def set(self, intensity: float, mask: np.ndarray | None, mask_id: int, pose: PoseObservation | None, body: BodyState | None) -> None
    def end(self) -> None
    alive: bool
```
See §6 for the visual design. All emitters respect `quality.particle_budget ×
settings.particle_multiplier`. Abilities only use this public API.

### 5.5 abilities.py — abilities and the ability manager

```python
class Ability(abc.ABC):
    id: str; label: str; hotkey: str
    def __init__(self, ctx: AbilityContext) -> None
    def update(self, tick: AbilityTick) -> None
    def on_selected(self, now: float) -> None
    def on_deselected(self, now: float) -> None
    def hud(self, now: float, out: AbilityHUD) -> None
class AbilityManager:
    def __init__(self, settings: Settings, effects: EffectSystem, audio: AudioSink | None = None) -> None
    def select(self, ability_id: str, now: float) -> None
    def toggle_aura(self, now: float) -> bool
    def update(self, now: float, dt: float, frame: TrackingFrame | None, body: BodyState | None,
               signals: GestureSignals, events: list[GestureEvent], manual: ManualInput) -> None
    def hud(self, now: float) -> AbilityHUD
    def apply_settings(self, settings: Settings) -> None
    def reset(self, now: float) -> None
ABILITY_TYPES: dict[str, type[Ability]]  # registry: adding an ability = new class + entry
```
Selected ability (keys 1–3) decides what the charge gesture produces; the Energy
Burst gesture works for every selection (a charged orb explodes into the burst);
Power Aura (key 4) is an independent toggle; Power Level is always on.

### 5.6 renderer.py (OpenGL) and renderer_cpu.py (fallback)

```python
class GLRenderer(Renderer):              # renderer.py
    def __init__(self, ctx: moderngl.Context, size: tuple[int, int], quality: QualityPreset,
                 fonts: FontManager, settings: EffectsSettings | None = None, offscreen: bool = False)
    @classmethod
    def create_offscreen(cls, size, quality, fonts, settings=None) -> "GLRenderer"
class CpuRenderer(Renderer):             # renderer_cpu.py
    def __init__(self, size: tuple[int, int], quality: QualityPreset, fonts: FontManager,
                 settings: EffectsSettings | None = None, target: pygame.Surface | None = None)
def create_renderer(kind: str, size, quality, fonts, settings=None, window_has_gl: bool = True) -> Renderer
```
`render(scene, effects, ui, view)` draws a full frame: camera (graded + lit) →
aura → effects (additive HDR) → bloom → composite (shake, zoom, refraction, flash,
tone mapping) → UI. With an empty `RenderList` the camera image must appear
unchanged (identity grade). `read_pixels()` returns (H, W, 3) RGB uint8 top-down.
`mirror_output = True` mirrors the composited picture horizontally *before* the UI
pass (GL: negative u scale in the composite; CPU: `cv2.flip`), so UI text stays
readable; the virtual camera uses it to undo the camera mirroring. Several renderers
may share one ModernGL context (the app's virtual camera renderer shares the window's).

### 5.7 ui.py — HUD, settings panel, debug overlay

```python
class UIController:
    def __init__(self, settings: Settings, fonts: FontManager) -> None
    settings_open: bool
    def toggle_settings(self) -> None
    def handle_event(self, event: pygame.event.Event) -> list[UIAction]
    def wants_mouse(self) -> bool
    def notify(self, text: str, level: str = "info", duration: float = 3.0) -> None
    def build(self, ctx: UIContext) -> UIDrawList
    def build_overlay(self, ctx: UIContext) -> UIDrawList   # callout only, for the virtual camera
```
`UIContext.virtual_camera` (a `VirtualCameraStatus` or None) drives the "VCAM" chip and
the *Video calls* settings section; the UI requests `toggle_virtual_camera` and
`set_virtual_camera_resolution` actions.

### 5.8 audio.py — procedural sound

```python
class AudioEngine:                      # implements AudioSink
    def __init__(self, settings: AudioSettings) -> None
    def start(self) -> bool             # init mixer + synthesise sounds; False if unavailable
    def play(self, name: str, volume: float = 1.0) -> None
    def loop(self, name: str, volume: float) -> None
    def stop(self, name: str, fade_ms: int = 250) -> None
    def stop_all(self) -> None
    def apply_settings(self, settings: AudioSettings) -> None
    def close(self) -> None
```

### 5.9 app.py / main.py — integration (integrator)

CLI: `--camera N`, `--fullscreen/--windowed`, `--renderer auto|gl|cpu`,
`--quality`, `--test-pattern`, `--smoke-test SECONDS`, `--check`, `--list-cameras`,
`--virtual-camera`, `--debug`, `--log-level`. Logs go to `logs/dragon_ball_powers.log`
(rotating). With `--smoke-test` and `--virtual-camera` the run only passes when pictures
reached the virtual camera (the script also switches it off and on once).

### 5.10 virtual_camera.py — video-call output

```python
class VirtualCamera:
    def __init__(self, settings: VirtualCameraSettings, camera_factory=None) -> None
    def start(self, settings: VirtualCameraSettings | None = None) -> None  # non-blocking; with settings: restart
    def stop(self) -> None                       # idempotent, joins the sender, closes the device
    def frame_due(self, now: float) -> bool      # pacing for the main loop (fps, no catch-up bursts)
    def submit(self, frame: np.ndarray) -> None  # (H, W, 3) uint8 RGB, never blocks
    size: tuple[int, int]; settings: VirtualCameraSettings
    @property status -> VirtualCameraStatus      # OFF | STARTING | RUNNING | ERROR (+ actionable error text)
def probe(settings=None, camera_factory=None) -> tuple[bool, str]   # used by --check
def friendly_error(exc) -> str
```
* Device: `pyvirtualcam.Camera(width, height, fps, backend=None|name, device=None|name)`,
  imported lazily (optional dependency). Windows: `install_virtual_camera.bat` registers
  Unity Capture's DirectShow filter (pinned commit, SHA256-verified, one UAC prompt for
  regsvr32 only) as the device "Dragon Ball Powers"; OBS Studio's "OBS Virtual Camera" also
  works (one sender at a time). macOS: OBS 30+ camera extension. Linux: v4l2loopback.
* The sender thread sends black first, then the newest picture; when none arrives for 2.5
  frame intervals it repeats the last one (steady stream for call apps). Wrong sizes are
  scaled, unusable pictures skipped. Open failures → `ERROR` with a hint (install OBS /
  device busy / device name / run setup.bat / Visual C++ runtime / modprobe), logged
  without traceback. A *busy* device is retried for `OPEN_RETRY_SECONDS` (1.5 s): after a
  restart a call app holds the previous sender's OBS shared memory until its next frame
  tick. With backend `auto` a restart first re-opens the driver of the last run, so a
  briefly busy OBS camera cannot silently fall through to another driver.
* App: `V` / settings toggle; every start announces its outcome; on `ERROR` the app shows
  the hint and switches `virtual_camera.enabled` off (no retry at every start). A running
  output is re-opened only when `device_settings()` (size, fps, backend, device) change,
  never for `mirror` (read every frame). `--virtual-camera` applies to one run: the saved
  choice is written back unless the user switches the output. After `pygame.display.flip()`
  a second renderer of the window's backend draws the picture at `size` (fit mode `cover`,
  callout-only overlay, `mirror_output = (camera.mirror and not test_pattern) !=
  virtual_camera.mirror`: calls see an unmirrored picture by default, like from a real
  webcam; the test pattern is never mirrored). That renderer is kept while the output is off
  (no shader-compile hitch when switching on again).
* camera.py marks virtual camera outputs (`CameraInfo.is_virtual`: "OBS Virtual Camera",
  Unity Capture, names with "virtual") and, like IR cameras, never selects them
  automatically - the app must not capture the device it sends to. When only virtual
  outputs are listed, `choose_camera_index` returns `NO_CAMERA` (-1): the app keeps the
  camera off and asks for a webcam (a device picked by name is still opened).

## 6. Visual design bible

The look: cinematic anime energy over the real camera image. Hot white cores, saturated
colour bodies, wide soft halos, lots of small fast detail (sparks, arcs, embers),
strong but short impacts (flash, shake, shockwaves), the room darkening slightly and
the energy *lighting* the user (scene `Light`s). Never a flat circle with a blur.

* **Ki orb (charging)**: anchored at the hand midpoint. Layers: white-blue core with
  slight noise distortion → bright inner colour body with swirling animated plasma
  (fbm noise in swirled polar coords) → fresnel rim → outer halo → occasional radial
  flares at high charge. Pulses (≈3 Hz, few %). Particles spiral *into* the orb
  (gathering energy). Sparks orbit it. Lightning arcs jump between orb and both hands
  (regenerated every ~40–80 ms) and crawl over its surface at high charge.
  Shockwave + power-up pulse at charge milestones (25/50/75/100 %). Light at the orb
  (radius ≈ 6×orb radius) brightens hands and face. Radius grows with charge and with
  the gap between the hands, within `orb_min_radius..orb_max_radius` × S.
* **Kamehameha beam (lateral)**: release flash + ring + shake, the beam shoots from the
  hands to the screen edge in ~0.15 s with a bulbous turbulent head, white core,
  colour body, soft halo, energy streaming along it (noise streaks), arcs along the
  edges, rings travelling outward, particles streaming in the beam direction,
  continuous small shake. Origin follows the hands, direction follows the arms
  (smoothed). Dissipates by breaking up into wisps (0.4–0.6 s).
* **Kamehameha toward the camera (frontal)**: the beam is seen end-on: a brilliant disc
  at the hands growing to ~45 % of the screen height, swirling tunnel turbulence,
  radial speed streaks rushing outward, rings expanding, particles flying at the
  viewer (perspective growth), a big flash and stronger shake.
* **Ki blast (Ki Charge release)**: the orb flies off (lateral: across the screen with a
  trail, exploding at the edge; frontal: grows towards the camera and hits the lens –
  white flash, shake, debris).
* **Energy burst**: blinding flash, 2–3 staggered shockwave rings with refraction,
  hundreds of radial sparks, radial arcs, embers, strong shake.
* **Power aura**: flame-like aura hugging the silhouette (segmentation mask; fallback mask
  drawn from pose landmarks), rising flame tongues and embers, flickering faster with
  movement and charge; lightning around the body at very high power.
* **Grade**: darken the room up to `camera_darken` while charging/firing; vignette;
  chromatic aberration only on impacts.
* **Palettes**: `PALETTES[...]` – blue-white (default), purple, red, golden yellow.

## 7. Gesture design

Features (per tracking frame, body-scale normalised): hand centres (hand model,
fallback pose wrist/index/pinky when the hand model loses a hand – common when hands
overlap in the cupped pose), hand size (rotation-robust palm size), hand distance and
its rate, arm extension 2D/3D (world landmarks), forearm "forward" component, velocity,
approach rate `d ln(hand_size / S)/dt` (towards the camera without depth sensing),
motion summary, energy, stability. Body scale S = max(shoulder width, 0.75 × torso
length) with hand-size and frame-height fallbacks, slowly smoothed.

* **Charge pose**: both hands present, `hand_distance < together_distance` (soft,
  sensitivity-scaled), hand midpoint inside the torso region, arms *not* extended,
  hands not moving fast. Score 0..1 with hysteresis; the state machine adds hold time.
* **Thrust (Kamehameha / ki blast release)**: over `thrust_window`, combine
  (a) frontal cues: hand-size growth relative to S, 3D extension gain, forward gain,
  pose wrist-z decrease; (b) lateral cues: hand-midpoint displacement along the arm
  direction + 2D extension gain + speed. Hands must stay roughly together. Emits one
  `THRUST` event (strength, frontal share, direction) per push, with a refractory period.
* **Burst**: hands start close (≲1.2 S) and fly apart (≳2.2 S) quickly (≳3.5 S/s peak
  rate) in opposite directions → one `BURST` event, refractory ~1 s.
* **Aim**: arm direction for steering a sustained beam; `arms_extended` keeps it alive.
* Single webcam = no true depth: "towards the camera" is inferred from apparent hand
  size growth, MediaPipe's relative z and its learned 3D world landmarks. It is an
  approximation and is documented as such.

## 8. Testing

Each module ships `tests/test_<module>.py`, runnable without webcam, network or display.
GL tests use the `gl_context` fixture; real-model tests skip when models are missing.
Cross-module behaviour is covered by `tests/test_integration.py` (synthetic motion from
`tests/motion_synth.py` and real MediaPipe output on a sample photo pushed through
gestures → abilities → effects → renderer) and `tests/test_app_smoke.py` (the whole
application loop, headless, with the generated test pattern and scripted key presses).
`tests/test_virtual_camera.py` runs the app through `tests/fake_virtual_camera.py`, which
replaces every pyvirtualcam backend with a fake driver (no real device is touched).

## 9. Implementation notes (as built)

Details that go beyond the contracts above and that callers rely on.

**camera.py** – `CameraManager(settings, *, open_timeout=5.0, failure_timeout=2.0,
hang_timeout=None)`; `start(index=None, name=None)`; `restart(settings=None, index=None,
name=None)` ("same device" follows a reconnect that found the camera at a new index);
`settings` and `is_running` properties; constants `TEST_PATTERN_NAME`, `SYNTHETIC_BACKEND`.
Frame ids never reset across restarts (the tracker waits for ids greater than the last
one it saw). `latest()` returns None after `stop()` or a terminal error; `wait_for_frame`
returns None at once when a stop/error happens during the wait and after at most 50 ms
when nothing is capturing. pygrabber names are only used with the DirectShow backend
(`auto`/`dshow` on Windows) because MSMF numbers devices differently; reconnect-by-name
uses exact, case-insensitive name matching, so pass `CameraInfo.name` unchanged.

**tracking.py / filters.py** – `HandPoseTracker(settings, models_dir=MODELS_DIR, *,
mask_width=256)`, plus `set_mask_width(w)`, `reset()`, `close()`, the `running` property
and context-manager support. `inference_ms` keys: `hands`, `pose`, `total`. Error texts
for unusable model files end with "Run setup.bat to download the models."; `process_frame`
never raises for model problems, it reports them in `status.error`. Coasting displacement
is `v·(1 − e^(−6·age))/6`; predicted hands keep the last handedness score. Single-detection
slot policy: the nearest recently seen, plausibly reachable slot wins; on a genuine
re-appearance (missed ≥ 1 frame and gone > 0.1 s) a confident handedness label (≥ 0.8)
overrides it. `stop()` gives up after 3 s; the tracker thread then releases the
landmarkers and workers itself. `OneEuroFilter` also exposes `set_params()`,
`.derivative`, `.value`, `.initialized`, `.last_time`.

**effects.py** – Arrays in a `RenderList` (sprite batches, the aura mask) may be views of
internal buffers: renderers must consume them before the next `update()`. A primitive's
`time` is the age of its effect; `seed` decorrelates simultaneous effects. `set_grade(darken)`
is a 0..1 *request* that the system scales by `EffectsSettings.camera_darken` (abilities
pass 0.6·charge while charging and 0.9 while a beam fires); `add_light`/`set_grade` apply
to the next `update()`/`collect()`. `OrbEffect.dissipate()` already draws the fizzle; the
orb draws its own milestone shockwaves, light and arcs from the charge passed to
`set_target` — abilities never call `fizzle`, `add_light` or `shockwave` for that. Units:
`Ring.thickness` is the full ring width in px, `Arc.width` the core half-width in px
(≈ 1.2–2.5 px × scene scale). At most 8 live steerable handles per type (the oldest is
ended gracefully). Extras: `particle_budget`, `scene_size`, `quality`, `settings`, `stats`,
`time`, `rng`; `OrbEffect.velocity`; `BeamEffect.origin/direction/length/power`;
`FrontalBeamEffect.center/radius/power/target_radius`; `AuraEffect.intensity/mask_id`;
module-level `make_bolt()`.

**abilities.py** – Extra public API: `AbilityTick`, `AbilityContext`, `ChargeStatus`,
`CalloutBoard`, `MessageBoard`, `PowerLevel`; manager extras `hotkeys`, `selected_id`,
`aura_enabled`, `power_level`, `charge_status()`, `consume_charge()`, `tracking_lost_for()`,
`held_effects()`. `select()` and `toggle_aura()` write `settings.abilities.selected` /
`aura_enabled` into the settings object the manager holds; `select("power_aura")` toggles
the aura. A burst that consumes a charge whose pose is still held requires releasing the
pose before the next gesture charge. Orb centre fallback: hand midpoint → last hand
midpoint of this charge → chest (body centre + 0.3 S) → last centre → (0.5 w, 0.62 h).
During an attack the power value jumps to ≥ 9001 + 2500·fire_charge and the displayed
number counts up; "IT'S OVER 9000!" waits up to 1 s for the release shout to finish.
