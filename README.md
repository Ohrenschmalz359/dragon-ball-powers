# Dragon Ball Powers — real-life ki attacks with your webcam

Stand in front of your webcam, bring your hands together and a glowing ki orb forms
**between your real hands**. Charge it, push your arms forward and fire a
**Kamehameha** — or throw your arms apart for an **Energy Burst**. Everything runs in
real time on your PC: MediaPipe hand + body tracking, a gesture state machine and a
GPU visual-effects engine (ModernGL / OpenGL 3.3) composited over the live camera.

No prerecorded video, no cloud, no fake demo: the effects follow your tracked
movements frame by frame.

---

## Quick start

1. **Set up once:** double-click `setup.bat` in the project folder. It finds or installs
   Python 3.12 (inside this folder only), creates `.venv`, installs the packages and
   downloads the tracking models (internet is needed only this once).
2. **Start:** double-click `run.bat` (or type `run.bat` in a terminal).
3. **Play:** stand about 1.5–2.5 m from the camera with your upper body and hands
   visible. Bring your hands together in front of your chest → the ki orb forms between
   your hands. Hold to charge, then **push both arms forward quickly** → Kamehameha.
   Throw your arms apart → Energy Burst. Key **4** → aura.
4. **Keyboard for testing:** hold `SPACE` = charge, release = fire; `1/2/3` ability,
   `4` aura, `V` video calls, `F` fullscreen, `C` calibrate, `TAB` settings,
   `D` debug overlay, `ESC` leave fullscreen / quit.
5. **Video calls (Discord, Zoom, Teams, Google Meet …):** double-click
   `install_virtual_camera.bat` once and confirm the administrator prompt (no OBS
   needed), then press **V** in the app ("VCAM LIVE" appears at the top) and choose
   **"Dragon Ball Powers"** as the camera in the call program. The others see you with
   the effects, without the HUD and the right way round. **WhatsApp:** the WhatsApp
   Desktop app currently does not list virtual cameras; use **WhatsApp Web**
   (web.whatsapp.com) in Edge or Chrome instead, or share the app window with the call's
   screen sharing (`H` hides the HUD). None of this works on phones. Details:
   [Video calls](#video-calls-discord-zoom-teams-google-meet-).
6. **Problems?** `run.bat --check` checks Python, packages, models, OpenGL, cameras,
   audio and the virtual camera. The log is in `logs\dragon_ball_powers.log`.

**Mac (and Linux):** open a terminal in the project folder, run `bash setup.sh` once and
`bash run.sh` to start. On the first start macOS asks whether Terminal may use the
camera: allow it, then press START CAMERA in the app. Video calls on a Mac need
[OBS Studio](https://obsproject.com) 30+ (click "Start Virtual Camera" once and allow the
system extension) — without OBS, Apple does not allow a virtual camera. Then it also
works in **FaceTime**. Details: [macOS and Linux](#macos-and-linux).

---

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Starting the app](#starting-the-app)
- [macOS and Linux](#macos-and-linux)
- [Opening the project in Visual Studio Code](#opening-the-project-in-visual-studio-code)
- [Camera setup](#camera-setup)
- [Gestures](#gestures)
- [Keyboard controls](#keyboard-controls)
- [Video calls (Discord, Zoom, Teams, Google Meet, …)](#video-calls-discord-zoom-teams-google-meet-)
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

**Video calls** (see [Video calls](#video-calls-discord-zoom-teams-google-meet-))
- Virtual camera output (V): Discord, Zoom, Teams, Google Meet and other browser calls …
  see you *with* the effects and callouts, without the HUD, the right way round.

## Requirements

- Windows 10 or 11 (64-bit), macOS 11 or newer (Apple Silicon or Intel), or Linux
  (x86-64 / ARM64) — see [macOS and Linux](#macos-and-linux).
- A webcam (720p or better recommended; 60 fps cameras feel even more responsive).
- A GPU with OpenGL 3.3 (any NVIDIA/AMD/Intel GPU from the last ~10 years). An RTX-class
  GPU runs the "Ultra" preset easily; integrated graphics run "Medium"/"High".
- Python **3.11 or 3.12** (64-bit). You don't need to install it yourself:
  `setup.bat` finds an existing one or downloads a private copy into `.python\`.
  (Python 3.13+ is not supported by the pinned MediaPipe version.)
- Internet access once, during setup (≈ 400 MB of Python packages, 53 MB of models).
  The app itself runs offline.
- Optional, for video calls: a virtual camera driver — on Windows `install_virtual_camera.bat`
  (no OBS needed), on macOS OBS Studio 30+ (see [Video calls](#video-calls-discord-zoom-teams-google-meet-)).

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
| `--virtual-camera` | switch the video-call output on for this run (see [Video calls](#video-calls-discord-zoom-teams-google-meet-)) |
| `--debug` | debug logging + landmark overlay |
| `--check` | self-check of Python, packages, models (SHA-256), MediaPipe, OpenGL, cameras, audio, virtual camera |
| `--list-cameras` | list cameras and exit |
| `--test-pattern` | generated test image instead of the webcam (diagnostics only) |
| `--smoke-test SECONDS` | automated self-test that drives every ability, then exits |

## macOS and Linux

The app runs on macOS 11+ (Apple Silicon and Intel) and on Linux with the same features.
Instead of the `.bat` files use the shell scripts, in a terminal opened in the project folder:

```sh
bash setup.sh        # once: Python 3.12 (private copy in .python if needed), .venv, packages, models
bash run.sh          # start the app; options as with run.bat, e.g. bash run.sh --fullscreen
```

- `setup.sh` uses an installed Python 3.11/3.12 (e.g. from python.org or Homebrew) or
  downloads a private, checksum-verified copy into `.python/` — nothing is installed
  system-wide. Options: `--dry-run`, `--force`, `--dev`, `--help`.
- **Camera permission (macOS):** the first start asks whether Terminal may use the camera.
  Allow it (System Settings → Privacy & Security → Camera), then press START CAMERA in the
  app or start it again.
- Cameras are listed as "Camera 0", "Camera 1" … (no device names); pick another one in
  Tab → Camera if the wrong one opens.
- Use **F** for fullscreen (macOS reserves F11).
- Video calls: see [Video calls](#video-calls-discord-zoom-teams-google-meet-) (macOS needs
  OBS Studio 30+ for the virtual camera; then it also works in FaceTime).
- Not tested on a real Mac yet — if something fails, `bash run.sh --check` and
  `logs/dragon_ball_powers.log` show the details.

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
| **V** | video-call output (virtual camera) on / off |
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

## Video calls (Discord, Zoom, Teams, Google Meet, …)

The app can be a **webcam for other programs**: your video call shows the camera picture
with the energy effects and the big callouts (without the HUD). One integration works for
the programs that accept virtual cameras — Discord, Zoom, Teams, browsers and most others.

**One-time setup** (the Python part, `pyvirtualcam`, comes with `setup.bat` / `setup.sh`;
run it again after updating an older copy of the project):

| System | Driver for the virtual camera |
|--------|-------------------------------|
| **Windows** (no OBS needed) | double-click **`install_virtual_camera.bat`** once and confirm the administrator prompt. It downloads the small Unity Capture filter (MIT license, ~330 KB, checksum-verified) into `virtualcam\` and registers it as the camera **"Dragon Ball Powers"**. `install_virtual_camera.bat --uninstall` removes it (do that before moving or deleting the folder). Alternatively OBS Studio's "OBS Virtual Camera" works too. |
| **macOS** | [OBS Studio](https://obsproject.com) 30 or newer: install it, open it, click **Start Virtual Camera** once and allow the system extension in System Settings (a restart may be needed), click Stop Virtual Camera and quit OBS. The camera is called **"OBS Virtual Camera"**; OBS does not need to run afterwards. Apple only allows virtual cameras as signed system extensions, so there is no OBS-free way on macOS. |
| **Linux** | v4l2loopback: `sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback devices=1` |

**Use it:**

1. Start the app and press **V** (or *Tab → Video calls → Virtual camera*). The HUD shows
   **VCAM LIVE** (also in fullscreen). The choice is remembered for the next start.
2. In the call program select **"Dragon Ball Powers"** (Windows driver from above) or
   **"OBS Virtual Camera"** as the camera:

   | Program | Where |
   |---------|-------|
   | Discord | User Settings → Voice & Video → Camera |
   | Zoom | Settings → Video → Camera |
   | Microsoft Teams | Settings → Devices → Camera |
   | Google Meet and other browser calls | the meeting's camera setting (Chrome, Edge, Firefox, Safari) |
   | FaceTime | macOS: in FaceTime's *Video* menu. Windows: only through a FaceTime link in Chrome or Edge, choose the camera in the browser |
   | WhatsApp | the WhatsApp Desktop app currently does not list virtual cameras: use **WhatsApp Web** (web.whatsapp.com) in a browser and pick the camera there, or share the app window with the call's screen sharing (press **H** to hide the HUD) |
   | OBS (streaming) | simply add a *Window Capture* of the app window instead |

3. The others see 1280×720 at 30 fps (1920×1080 in the settings), **not mirrored** — like
   through a normal webcam, so writing on your shirt is readable (call programs mirror only
   your own preview). The callouts are readable either way. For the mirrored look switch on
   *Tab → Mirrored picture in calls*.

Good to know:

- Only one program can send to a virtual camera at a time (e.g. OBS's own "Start Virtual
  Camera" must be off while the app uses "OBS Virtual Camera").
- The effect **sounds** play on your speakers; the others hear them only through your
  microphone (not when you wear headphones).
- **Phones** cannot use it: WhatsApp and FaceTime on a phone only use the phone's camera.
- A program does not list the camera? Restart it after installing the driver. Some Store
  apps only show real cameras: use the service in the browser, or share the app window
  with screen sharing (press **H** to hide the HUD).
- `--check` tests the virtual camera, `--virtual-camera` switches it on for this run, and
  `--test-pattern --smoke-test 16 --virtual-camera` sends a test picture through it (exit
  code 0 = it works) — with `run.bat` on Windows or `bash run.sh` on macOS / Linux.
- `virtual_camera.backend` / `virtual_camera.device` in the settings file pick a specific
  driver or device when several are installed.

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
| `virtual_camera` | `enabled` (false), `width`/`height` (1280×720), `fps` (30), `backend` (`auto`/`obs`/`unitycapture`/`v4l2loopback`), `device`, `mirror` (false) |

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
| "No virtual camera driver found" | Windows: run `install_virtual_camera.bat` once (or install OBS Studio), then press V again |
| "No virtual camera found" (macOS) | OBS Studio 30+: click Start Virtual Camera once, allow the system extension in System Settings, quit OBS, press V |
| macOS: black picture / "not authorized" camera error | System Settings → Privacy & Security → Camera: allow Terminal, then START CAMERA in the app |
| "The virtual camera is busy" | stop "Virtual Camera" in OBS if it runs, or switch the camera off and on in the call program, then press V |
| The app shows the OBS logo instead of you | "OBS Virtual Camera" was picked as its input: choose your real webcam in Tab → Camera (virtual outputs are marked VIRTUAL and never opened automatically) |
| "No webcam found, only virtual cameras" | connect your webcam, press RESCAN in Tab → Camera and pick it |
| "pyvirtualcam package is missing" | run `setup.bat` again |
| The call shows the OBS logo / a black picture | the app's virtual camera is off (press V) or the app is not running |
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
you take yourself (F12). Only while you switch the virtual camera on (V) does the app
offer its picture to other programs on your PC — like a webcam; what a call program
does with it is up to that program. MediaPipe is pinned to **0.10.21** on purpose: newer
releases (0.10.30+/1.x) ship a native library that uploads usage metrics to Google;
0.10.21 does not contain that telemetry.

## Project structure

```
dragon-ball-powers\
├── run.bat                  start the app
├── setup.bat                one-time setup (Python, .venv, packages, models)
├── install_virtual_camera.bat  optional: camera "Dragon Ball Powers" for video calls
├── setup.sh / run.sh        the same for macOS and Linux (bash setup.sh, bash run.sh)
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
│   ├── virtual_camera.py    video-call output (virtual webcam via pyvirtualcam)
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
  (LGPL), ModernGL (MIT), pyvirtualcam (GPL-2.0; only imported when the virtual camera
  is switched on — keep that in mind if you redistribute the app).
- Windows virtual camera driver: the Unity Capture filter (MIT), downloaded by
  `install_virtual_camera.bat` from github.com/schellingb/UnityCapture (not bundled).

Dragon Ball is a trademark of its owners; this is an unofficial fan project.
