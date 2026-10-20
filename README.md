# Dragon Ball Powers — real-life ki attacks with your webcam

Stand in front of your webcam, bring your hands together and a glowing ki orb forms
**between your real hands**. Charge it, push your arms forward and fire a
**Kamehameha** — or throw your arms apart for an **Energy Burst**. Everything runs in
real time on your PC: MediaPipe hand + body tracking, a gesture state machine and a
GPU visual-effects engine (ModernGL / OpenGL 3.3) composited over the live camera.

No prerecorded video, no cloud, no fake demo: the effects follow your tracked
movements frame by frame.

---

## Kurzanleitung (Deutsch)

1. **Einmalig einrichten:** `setup.bat` im Projektordner doppelklicken. Es sucht bzw.
   installiert Python 3.12 (nur in diesem Ordner), legt `.venv` an, installiert die
   Pakete und lädt die Tracking-Modelle (Internet nur dieses eine Mal nötig).
2. **Starten:** `run.bat` doppelklicken (oder im Terminal: `run.bat`).
3. **Spielen:** ca. 1,5–2,5 m vor die Kamera stellen, Oberkörper und Hände sichtbar.
   Hände vor der Brust zusammenführen → die Ki-Kugel entsteht zwischen den Händen.
   Halten zum Aufladen, dann **beide Arme schnell nach vorne stoßen** → Kamehameha.
   Arme ruckartig auseinanderreißen → Energy Burst. Taste **4** → Aura.
4. **Tastatur zum Testen:** `LEERTASTE` halten = aufladen, loslassen = feuern;
   `1/2/3` Fähigkeit, `4` Aura, `F` Vollbild, `C` Kalibrieren, `TAB` Einstellungen,
   `D` Debug-Overlay, `ESC` Vollbild verlassen / beenden.
5. **Probleme?** `run.bat --check` prüft Python, Pakete, Modelle, OpenGL, Kameras und
   Audio. Das Log liegt in `logs\dragon_ball_powers.log`.

---

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Starting the app](#starting-the-app)
- [Opening the project in Visual Studio Code](#opening-the-project-in-visual-studio-code)
- [Camera setup](#camera-setup)
- [Gestures](#gestures)
- [Keyboard controls](#keyboard-controls)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Privacy](#privacy)
- [Project structure](#project-structure)
- [Adding a new ability](#adding-a-new-ability)
- [Running the tests](#running-the-tests)
- [Licenses](#licenses)

## Features

**Tracking (local, real time)**
- Live, mirrored webcam preview; DirectShow capture with MJPG for low latency,
  camera selection (infrared Windows-Hello cameras are recognised and avoided).
- MediaPipe **hand landmarker** (both hands, 21 landmarks each) and **pose
  landmarker** (shoulders, elbows, wrists, hips, …) running *in parallel* in a
  background thread, plus a person **segmentation mask** for the aura.
- One-Euro filtering (low jitter at rest, low lag in fast motion), hand-slot tracking
  that survives crossing/overlapping hands, short tracking drop-outs are bridged.
- When the hand model loses a hand (typical when palms overlap in the charging pose),
  the pose model's wrist/knuckle points take over.

**Gestures** (body-size normalised, so they work near or far from the camera)
- Charging pose: both hands together in front of the torso, arms bent.
- Thrust: both arms pushed **towards the camera** (detected from hand-size growth,
  3D arm extension and depth cues) or **sideways** (2D displacement + extension).
- Energy-burst fling: hands start close together and fly apart fast.
- Motion classification (together / apart / up / down / left / right / forward /
  backward), movement energy and stability, configurable sensitivity, calibration.
- A clear attack **state machine**: idle → preparing → charging → firing → cooldown.
  One push = one attack; minimum charge time, cooldown and a pose-reset rule prevent
  repeated or accidental firing.

**Abilities** (modular: one class each, see [Adding a new ability](#adding-a-new-ability))
| Key | Ability | Gesture |
|-----|---------|---------|
| 1 | **Ki Charge** | hands together → orb; push → the orb flies off as a ki blast |
| 2 | **Kamehameha** (default) | charge with the "KA… ME… HA… ME…" chant; push → beam (towards the camera or sideways); the beam follows your arms and lasts while they stay extended |
| 3 | **Energy Burst** | fling the arms apart (works with every ability; a charged orb makes it bigger) |
| 4 | **Power Aura** (toggle) | flame-like aura around your silhouette; reacts to movement and charge (it also flares up automatically while charging) |
| – | **Power Level** | scouter-style meter: rises with charge time, stability and movement — "IT'S OVER 9000!" |

**Visual effects** (GPU, HDR, cinematic)
- Ki orb: white-hot core, swirling plasma, fresnel rim, halo, radial flares, energy
  particles spiralling in, orbiting sparks, lightning arcs to both hands, shockwave
  pulses at charge milestones, the orb's light illuminating your hands and face.
- Kamehameha beam: release flash, shockwave and screen shake; white core, coloured
  body and halo, energy streaming along the beam, turbulent bulbous head, edge arcs,
  travelling rings; end-on "coming at you" version when fired at the camera; breaks
  up into wisps when it ends.
- Energy burst: blinding flash, staggered shockwaves with refraction, hundreds of
  sparks, radial lightning, chromatic aberration, camera shake.
- Aura: flames rising from your segmentation silhouette, embers, lightning at high power.
- Bloom, additive blending, tone mapping, room darkening while charging, vignette,
  scene lighting, energy colours **blue-white** (default), **purple**, **red**,
  **golden yellow**; quality presets Low / Medium / High / Ultra (Auto picks one).
- A NumPy/OpenCV **CPU renderer** is used automatically if OpenGL 3.3 is unavailable.

**Desktop interface**
- Dark "scouter" HUD around the edges: camera / hands / body status, FPS, tracking
  latency, active ability, power level, charge progress, call-outs.
- Settings panel (Tab): camera selection and resolution, gesture sensitivity, minimum
  charge time, cooldown, quality, energy colour, bloom, shake, aura, pose model, sound,
  fit mode, debug overlay, recalibration, reset.
- Fullscreen cinematic mode (F), guided calibration (C), debug overlay with the
  tracked skeletons and live gesture scores (D), screenshots (F12).
- Procedurally synthesised sound effects (charge hum, beam roar, explosions).

## Requirements

- Windows 10 or 11, 64-bit.
- A webcam (720p or better recommended; 60 fps cameras feel even more responsive).
- A GPU with OpenGL 3.3 (any NVIDIA/AMD/Intel GPU from the last ~10 years). An RTX-class
  GPU runs the "Ultra" preset easily; integrated graphics run "Medium"/"High".
- Python **3.11 or 3.12** (64-bit). You don't need to install it yourself:
  `setup.bat` finds an existing one or downloads a private copy into `.python\`.
  (Python 3.13+ is not supported by the pinned MediaPipe version.)
- Internet access once, during setup (≈ 400 MB of Python packages, 53 MB of models).
  The app itself runs offline.

## Installation

1. Put the project folder somewhere convenient, e.g. `C:\Users\<you>\dragon-ball-powers`.
2. Double-click **`setup.bat`** (or run it in a terminal from the project folder).

`setup.bat` only touches the project folder:

| Step | What happens |
|------|--------------|
| 1/6 | finds Python 3.11/3.12 x64 (`.venv`, `py -3.12`, `py -3.11`, `python`, `.python\python.exe`); if none is found it downloads python-build-standalone 3.12, checks its SHA-256 and unpacks it into `.python\` |
| 2/6 | creates the virtual environment `.venv\` |
| 3/6 | updates pip |
| 4/6 | installs the pinned packages from `requirements.txt` |
| 5/6 | downloads and verifies the MediaPipe models into `assets\models\` (`scripts\download_models.py`) |
| 6/6 | runs the self-check `python -m dragon_ball_powers --check` |

Options: `setup.bat --dry-run` (show what would happen), `--force` (recreate `.venv`),
`--dev` (also install pytest), `--no-pause`, `--help`. Running it again is safe.
Behind a proxy: `set HTTPS_PROXY=http://proxy:port` in the same window first.

Manual alternative (if you prefer): `py -3.12 -m venv .venv`,
`.venv\Scripts\python -m pip install -r requirements.txt`,
`.venv\Scripts\python scripts\download_models.py`.

## Starting the app

Double-click **`run.bat`**, or from a terminal:

```bat
cd C:\Users\<you>\dragon-ball-powers
run.bat
```

`run.bat` checks the environment, activates `.venv`, sets `PYTHONPATH=src` and runs
`python -m dragon_ball_powers`. If the app exits with an error the window stays open so
you can read the message. All arguments are passed through:

| Option | Meaning |
|--------|---------|
| `--camera N` | use camera index N (see `--list-cameras`) |
| `--fullscreen` / `--windowed` | start mode |
| `--renderer auto\|gl\|cpu` | rendering backend (auto = OpenGL with CPU fallback) |
| `--quality auto\|low\|medium\|high\|ultra` | effects quality |
| `--no-audio` | no sound |
| `--debug` | debug logging + landmark overlay |
| `--check` | self-check of Python, packages, models (SHA-256), MediaPipe, OpenGL, cameras, audio |
| `--list-cameras` | list cameras and exit |
| `--test-pattern` | generated test image instead of the webcam (diagnostics only) |
| `--smoke-test SECONDS` | automated self-test that drives every ability, then exits |

## Opening the project in Visual Studio Code

1. Run `setup.bat --dev` once (installs pytest for the test explorer).
2. Open the folder: in a terminal `code C:\Users\<you>\dragon-ball-powers`, or in
   VS Code *File → Open Folder…*.
3. Install the recommended extensions when prompted (Python, Python Debugger).
   `.vscode/settings.json` already selects `.venv\Scripts\python.exe` and adds `src`
   to the analysis path.
4. *Run and Debug* (Ctrl+Shift+D) offers **Dragon Ball Powers**, **Dragon Ball Powers
   (CPU renderer)** and **Self-check**. Press F5 to start with the debugger.
5. *Testing* panel: the tests in `tests/` are discovered automatically.

## Camera setup

- **Selection:** the first colour camera is used by default; infrared cameras (e.g.
  "HP IR Camera" for Windows Hello) are recognised and skipped. Pick another camera in
  the settings panel (Tab → Camera) or with `run.bat --camera N` (`run.bat --list-cameras`
  shows the indices). The choice is remembered by name.
- **Resolution / FPS:** 1280×720 at 30 fps is the default and a good balance.
  1920×1080 or 60 fps work if your camera supports them (Tab → Camera).
- **Distance & framing:** stand 1.5–2.5 m away so your upper body and both hands are in
  the picture (sitting close to a laptop camera also works, but keep the hands visible).
- **Light:** light your front (not a bright window behind you). MediaPipe needs to see
  your hands clearly; avoid very dark rooms.
- **Busy camera?** Only one app can use the camera: close Teams/Zoom/OBS/browser tabs.
- **Permission:** Windows Settings → Privacy & security → Camera → allow camera access
  and "Let desktop apps access your camera".

## Gestures

The HUD hint at the bottom tells you what to do next. Turn on the debug overlay (**D**)
to see the tracked skeletons and live gesture scores.

1. **Charge (Ki Charge / Kamehameha):** bring both hands together in front of your chest
   or belly, palms facing each other as if holding a ball, elbows bent. Hold still: the
   orb appears after ~0.2 s and grows; moving the hands apart a little makes it bigger.
   The charge needs at least **0.6 s** before it can be fired (configurable).
2. **Fire:** push **both arms forward quickly** (about a quarter of a second) and keep
   them extended:
   - towards the camera → the beam comes straight at the viewer;
   - to the side (turned sideways, arms pushing left or right) → the beam crosses the screen.
   The beam follows your arms and lasts as long as your arms stay extended (longer with
   more charge). Lower your arms to stop it.
3. **Energy Burst:** start with the hands close together and throw them apart
   explosively. Works with every ability; with a charged orb the explosion is bigger.
4. **Power Aura:** press **4** (or click AURA). With "Auto aura" on it also flares up
   while you charge.
5. **New attack:** after firing, lower your hands once and charge again (one push = one attack).

**Calibration (C):** a 3-step guide (stand relaxed → hands together → one push) measures
your body size, hand distance, steadiness and push strength and adapts the thresholds.
It is optional; the defaults are normalised to body size and work for most people.
Sensitivity can also be changed with `[` / `]` or in the settings panel.

## Keyboard controls

| Key | Action |
|-----|--------|
| **Space** | hold = charge, release = fire (Ki Charge / Kamehameha); press = Energy Burst |
| **1** / **2** / **3** | select Ki Charge / Kamehameha / Energy Burst |
| **4** | toggle Power Aura |
| **← → ↑ ↓** | aim keyboard attacks; **0** = automatic aim (towards the camera) |
| **F** or **F11** | fullscreen cinematic mode |
| **C** | start / cancel calibration |
| **Tab** | settings panel |
| **D** | debug overlay (landmarks, gesture scores, state machine) |
| **H** | hide / show the HUD |
| **[** / **]** | gesture sensitivity −/+ 5 % |
| **F12** | save a screenshot to `screenshots\` |
| **Esc** | cancel calibration / close settings / leave fullscreen / quit |

Keyboard attacks use exactly the same ability, effect and rendering code as gestures.

## Configuration

Settings are stored in **`config\settings.json`** (created on first start, changed by
the settings panel). Invalid values are replaced by defaults or clamped, so editing the
file by hand is safe. Calibration results are stored in `config\calibration.json`.

| Section | Key settings (default) |
|---------|------------------------|
| `camera` | `index` (0), `name_hint`, `backend` (`auto` = DirectShow), `width`/`height` (1280×720), `fps` (30), `mirror` (true), `prefer_mjpg` (true), `sync_video_to_tracking` (true: show the analysed frame so effects line up), `auto_start` |
| `tracking` | `pose_model` (`lite`/`full`/`heavy`), `tracking_width` (640), confidences, `enable_pose`, `enable_segmentation`, `parallel_inference`, One-Euro `smoothing_min_cutoff`/`smoothing_beta`/`smoothing_d_cutoff`, `hand_lost_grace` |
| `gestures` | `sensitivity` (0.5), `prepare_time` (0.2 s), `min_charge_time` (0.6 s), `full_charge_time` (3 s), `cooldown_time` (0.8 s), `tracking_lost_grace`, `release_grace`, `together_distance`, `release_distance`, `thrust_window`, `burst_enabled`, `require_pose_reset`, `beam_min_duration`/`beam_max_duration` |
| `effects` | `quality` (`auto`), `energy_color` (`blue`/`purple`/`red`/`gold`), `bloom_strength`, `particle_multiplier`, `screen_shake`, `camera_darken`, `scene_lighting`, `show_callouts`, `motion_trails`, `orb_min_radius`/`orb_max_radius` |
| `abilities` | `selected` (`kamehameha`), `aura_enabled`, `aura_auto` |
| `ui` | `fullscreen`, `window_width`/`window_height`, `show_hud`, `show_debug`, `show_fps`, `fit_mode` (`cover`/`contain`), `vsync`, `max_fps` |
| `audio` | `enabled`, `volume` |

Delete `config\settings.json` to return to the defaults (or use *Reset to defaults*).

## Troubleshooting

| Problem | What to do |
|---------|------------|
| `run.bat` says it is not set up | run `setup.bat` first |
| "This program is blocked by group policy" during setup | your organisation's App Control policy blocks Python. Ask IT, or use a private PC |
| Camera error / black picture | close other apps using the camera, check Windows camera privacy settings, pick another camera (Tab), try `"backend": "msmf"` in `config\settings.json` |
| Hands not detected | more front light, stand closer, keep hands inside the picture; check the debug overlay (D) |
| Charge starts but firing does not trigger | push faster and keep the hands together while pushing; raise sensitivity (`]`); run the calibration (C) |
| Fires too easily | lower sensitivity (`[`), raise the minimum charge time |
| Low FPS | lower the quality (Tab → Effects), use `"pose_model": "lite"`, lower the camera resolution |
| OpenGL errors / black window | `run.bat --renderer cpu`; update your GPU driver |
| No sound | check the Windows output device; `audio.enabled` in the settings |
| Anything else | `run.bat --check`, then look at `logs\dragon_ball_powers.log` |

## Known limitations

- **No real depth.** A single webcam has no depth sensor. "Towards the camera" is
  inferred from how fast your hands grow in the image, from MediaPipe's relative z
  values and from its learned 3D pose. Strong pushes are recognised reliably, very
  slow or tiny pushes are not, and moving the whole body towards the camera is ignored
  on purpose.
- Effects are composited over the image (additive light): the orb is not occluded by
  your fingers and the beam does not cast real shadows.
- One person at a time; both hands should stay in the picture.
- Tracking runs on the CPU (MediaPipe's GPU delegate is not available on Windows);
  expect ~30–60 tracking fps depending on the CPU and the pose model.
- Sync mode shows the analysed frame, which adds about one camera frame of video delay
  in exchange for perfectly aligned effects (`camera.sync_video_to_tracking`).
- Gesture thresholds are tuned for adults standing 0.5–3 m from the camera.

## Privacy

Everything is processed locally. No images, video or landmarks leave your PC, nothing
is written to disk except the settings, the calibration, the log file and screenshots
you take yourself (F12). MediaPipe is pinned to **0.10.21** on purpose: newer
releases (0.10.30+/1.x) ship a native library that uploads usage metrics to Google;
0.10.21 does not contain that telemetry.

## Project structure

```
dragon-ball-powers\
├── run.bat                  start the app
├── setup.bat                one-time setup (Python, .venv, packages, models)
├── requirements.txt         pinned runtime packages (requirements-dev.txt adds pytest)
├── pyproject.toml           project metadata + pytest configuration
├── src\dragon_ball_powers\
│   ├── main.py              command line, logging, --check
│   ├── app.py               window, main loop, keyboard, wiring of all subsystems
│   ├── config.py            settings + calibration (JSON), project paths
│   ├── camera.py            capture thread, camera enumeration, test pattern
│   ├── tracking.py          MediaPipe hands ∥ pose thread, smoothing, hand slots
│   ├── filters.py           One-Euro filter
│   ├── gestures.py          body features, charge/thrust/burst detectors, events
│   ├── calibration.py       guided calibration
│   ├── state_machine.py     attack state machine
│   ├── abilities.py         Ki Charge, Kamehameha, Energy Burst, Power Aura, Power Level
│   ├── effects.py           particles, lightning, orb/beam/burst/aura simulation
│   ├── primitives.py        render primitives, palettes, quality presets, UI commands
│   ├── renderer.py          OpenGL renderer (ModernGL) + renderer factory
│   ├── gl_shaders.py        GLSL shaders
│   ├── renderer_cpu.py      NumPy/OpenCV fallback renderer
│   ├── ui.py                HUD, settings panel, overlays
│   ├── audio.py             procedural sound effects
│   ├── fonts.py             font loading (bundled fonts)
│   ├── datatypes.py, interfaces.py, mathutil.py
├── assets\models\           MediaPipe models (+ models.json manifest with SHA-256)
├── assets\fonts\            Bangers, Rajdhani (SIL OFL)
├── config\                  settings.json, calibration.json (generated)
├── scripts\                 download_models.py, render_preview.py
├── tests\                   pytest suite (no webcam needed)
├── docs\ARCHITECTURE.md     module contracts and design notes
└── logs\                    dragon_ball_powers.log (generated)
```

## Adding a new ability

Abilities are small classes registered in `abilities.py`; they get gesture signals,
events and keyboard input every frame and drive effects through the `EffectSystem` API
(see `docs/ARCHITECTURE.md` §5.4–5.5).

1. Subclass `Ability` (or `ChargeAbility` for charge-and-release attacks) in
   `src/dragon_ball_powers/abilities.py`, give it an `id`, `label` and `hotkey`, and
   implement `update(tick)` and `hud(now, out)`:

   ```python
   class SpiritBombAbility(Ability):
       id = "spirit_bomb"
       label = "Spirit Bomb"
       hotkey = "5"

       def __init__(self, ctx: AbilityContext) -> None:
           super().__init__(ctx)          # self.ctx: settings, effects, audio, callouts, messages
           self._orb = None

       def update(self, tick: AbilityTick) -> None:
           body = tick.body
           if body is None or body.body_center is None or not body.both_hands:
               return
           top = body.body_center[1] - body.body_scale
           raised = body.left.hand_center[1] < top and body.right.hand_center[1] < top
           if raised and self._orb is None:
               self._orb = self.ctx.effects.create_orb(body.hand_mid, 0.3 * body.body_scale)
               self.ctx.audio.play("charge_start")
           if self._orb is not None:
               self._orb.set_target(body.hand_mid, 0.5 * body.body_scale, charge=1.0,
                                    left_hand=body.left.hand_center, right_hand=body.right.hand_center)
               if not raised:
                   self._orb.dissipate()
                   self._orb = None

       def hud(self, now: float, out: AbilityHUD) -> None:
           out.message = "Raise both hands to gather energy"
   ```
2. Register it: `ABILITY_TYPES["spirit_bomb"] = SpiritBombAbility`.
3. Add the id to `ABILITY_IDS` in `config.py` (so it can be selected and saved), and a
   key in `ABILITY_KEYS` in `app.py`.
4. New gestures go into `GestureEngine` (`gestures.py`) as additional `GestureSignals`
   fields / `GestureEvent` types; new visuals into `EffectSystem` (`effects.py`) using the
   existing primitives (orbs, beams, rings, arcs, sprites, flashes, lights).
5. Add tests (see `tests/test_abilities.py` for the fake effect system).

## Running the tests

```bat
setup.bat --dev
.venv\Scripts\python -m pytest
```

The suite (653 tests) needs no webcam: camera code is tested with fakes, tracking with
the real models, gestures with a synthetic motion generator (`tests/motion_synth.py`),
the renderers offscreen, `tests/test_integration.py` pushes synthetic and real tracking
through gestures → abilities → effects → renderer, and `tests/test_app_smoke.py` runs
the whole app headlessly with the test pattern (`run.bat --test-pattern --smoke-test 16`
does the same on your PC and exits with code 0 when everything works).

Effect previews without a camera: `python scripts/render_preview.py` (writes PNGs to
`previews\`), sounds: `python -m dragon_ball_powers.audio --export previews\sounds`.

## Licenses

- Project code: MIT.
- Fonts: Bangers and Rajdhani, SIL Open Font License 1.1 (`assets/fonts/*-OFL.txt`).
- MediaPipe models: Apache License 2.0 (Google), see `assets/models/README.md`.
- Dependencies: MediaPipe (Apache 2.0), OpenCV (Apache 2.0), NumPy (BSD), pygame-ce
  (LGPL), ModernGL (MIT).

Dragon Ball is a trademark of its owners; this is an unofficial fan project.
