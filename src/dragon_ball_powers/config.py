"""Typed application settings with validation and JSON persistence.

Settings live in ``config/settings.json`` (created on first run). Loading is
forgiving: unknown keys are ignored, missing keys fall back to defaults, wrongly
typed values are replaced by defaults and numbers are clamped into their valid
range. A hand-edited or corrupted file can therefore never crash the app.

Every field declares its constraints in ``dataclasses.field(metadata=...)``:
``{"min": x, "max": y}`` for numbers and ``{"choices": (...)}`` for strings.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, get_type_hints

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- paths

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent  # <root>/src/dragon_ball_powers -> <root>


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_DIR = _env_path("DBP_MODELS_DIR", ASSETS_DIR / "models")
FONTS_DIR = ASSETS_DIR / "fonts"
CONFIG_DIR = _env_path("DBP_CONFIG_DIR", PROJECT_ROOT / "config")
LOGS_DIR = _env_path("DBP_LOGS_DIR", PROJECT_ROOT / "logs")
#: The one-time setup script of this platform (named in error messages).
SETUP_SCRIPT = "setup.bat" if sys.platform == "win32" else "setup.sh"
SETTINGS_FILE = "settings.json"
CALIBRATION_FILE = "calibration.json"
SETTINGS_VERSION = 1

# ------------------------------------------------------------------- enumerations

ENERGY_COLORS: tuple[str, ...] = ("blue", "purple", "red", "gold")
QUALITY_LEVELS: tuple[str, ...] = ("auto", "low", "medium", "high", "ultra")
ABILITY_IDS: tuple[str, ...] = ("ki_charge", "kamehameha", "energy_burst")
POSE_MODELS: tuple[str, ...] = ("lite", "full", "heavy")
CAMERA_BACKENDS: tuple[str, ...] = ("auto", "dshow", "msmf", "any")
FIT_MODES: tuple[str, ...] = ("cover", "contain")
# pyvirtualcam backends: OBS Virtual Camera (Windows/macOS), Unity Capture (Windows), Linux.
VIRTUAL_CAMERA_BACKENDS: tuple[str, ...] = ("auto", "obs", "unitycapture", "v4l2loopback")


def _range(lo: float, hi: float) -> dict[str, float]:
    return {"min": lo, "max": hi}


def _choices(*values: str) -> dict[str, tuple[str, ...]]:
    return {"choices": tuple(values)}


# ----------------------------------------------------------------------- sections


@dataclass
class CameraSettings:
    index: int = field(default=0, metadata=_range(0, 15))
    # Preferred device name (substring). Used to find the same camera again if
    # Windows re-orders device indices. Empty = use ``index``.
    name_hint: str = ""
    backend: str = field(default="auto", metadata=_choices(*CAMERA_BACKENDS))
    width: int = field(default=1280, metadata=_range(320, 3840))
    height: int = field(default=720, metadata=_range(240, 2160))
    fps: int = field(default=30, metadata=_range(10, 120))
    mirror: bool = True
    prefer_mjpg: bool = True
    # Show the exact frame the tracker analysed so effects line up with the hands
    # (adds about one frame of video delay). False = always show the newest frame.
    sync_video_to_tracking: bool = True
    auto_start: bool = True


@dataclass
class TrackingSettings:
    pose_model: str = field(default="lite", metadata=_choices(*POSE_MODELS))
    tracking_width: int = field(default=640, metadata=_range(256, 1280))
    max_hands: int = field(default=2, metadata=_range(1, 2))
    min_hand_detection_confidence: float = field(default=0.5, metadata=_range(0.1, 0.95))
    min_hand_presence_confidence: float = field(default=0.5, metadata=_range(0.1, 0.95))
    min_tracking_confidence: float = field(default=0.5, metadata=_range(0.1, 0.95))
    min_pose_detection_confidence: float = field(default=0.5, metadata=_range(0.1, 0.95))
    min_pose_presence_confidence: float = field(default=0.5, metadata=_range(0.1, 0.95))
    enable_pose: bool = True
    enable_segmentation: bool = True
    parallel_inference: bool = True
    # One Euro filter parameters, applied to normalised [0..1] landmark coordinates.
    smoothing_min_cutoff: float = field(default=1.5, metadata=_range(0.05, 20.0))
    smoothing_beta: float = field(default=8.0, metadata=_range(0.0, 100.0))
    smoothing_d_cutoff: float = field(default=1.0, metadata=_range(0.1, 20.0))
    # Seconds a hand that disappeared keeps being predicted before it counts as lost.
    hand_lost_grace: float = field(default=0.25, metadata=_range(0.0, 2.0))


@dataclass
class GestureSettings:
    # 0 = strict (fewer accidental triggers) ... 1 = very forgiving.
    sensitivity: float = field(default=0.5, metadata=_range(0.0, 1.0))
    prepare_time: float = field(default=0.2, metadata=_range(0.0, 2.0))
    min_charge_time: float = field(default=0.6, metadata=_range(0.0, 5.0))
    full_charge_time: float = field(default=3.0, metadata=_range(0.5, 10.0))
    cooldown_time: float = field(default=0.8, metadata=_range(0.0, 5.0))
    tracking_lost_grace: float = field(default=0.6, metadata=_range(0.0, 3.0))
    release_grace: float = field(default=0.4, metadata=_range(0.0, 3.0))
    # Distances are measured in "body scales" (about one shoulder width).
    together_distance: float = field(default=1.1, metadata=_range(0.3, 3.0))
    release_distance: float = field(default=2.2, metadata=_range(0.8, 5.0))
    thrust_window: float = field(default=0.45, metadata=_range(0.15, 1.0))
    burst_enabled: bool = True
    require_pose_reset: bool = True
    beam_min_duration: float = field(default=1.0, metadata=_range(0.3, 5.0))
    beam_max_duration: float = field(default=4.5, metadata=_range(0.5, 15.0))


@dataclass
class EffectsSettings:
    quality: str = field(default="auto", metadata=_choices(*QUALITY_LEVELS))
    energy_color: str = field(default="blue", metadata=_choices(*ENERGY_COLORS))
    bloom_strength: float = field(default=1.0, metadata=_range(0.0, 3.0))
    particle_multiplier: float = field(default=1.0, metadata=_range(0.1, 3.0))
    screen_shake: float = field(default=1.0, metadata=_range(0.0, 2.0))
    # How much the room darkens while energy is charged / fired (cinematic look).
    camera_darken: float = field(default=0.45, metadata=_range(0.0, 1.0))
    # How strongly the energy lights up the camera image (hands, face, room).
    scene_lighting: float = field(default=1.0, metadata=_range(0.0, 2.0))
    show_callouts: bool = True
    motion_trails: bool = True
    # Ki ball radius limits, in body scales.
    orb_min_radius: float = field(default=0.10, metadata=_range(0.02, 1.0))
    orb_max_radius: float = field(default=0.60, metadata=_range(0.05, 2.0))


@dataclass
class AbilitySettings:
    selected: str = field(default="kamehameha", metadata=_choices(*ABILITY_IDS))
    aura_enabled: bool = False
    # Let the aura flare up automatically while charging, even when toggled off.
    aura_auto: bool = True


@dataclass
class UISettings:
    fullscreen: bool = False
    window_width: int = field(default=1280, metadata=_range(640, 7680))
    window_height: int = field(default=720, metadata=_range(360, 4320))
    show_hud: bool = True
    show_debug: bool = False
    show_fps: bool = True
    fit_mode: str = field(default="cover", metadata=_choices(*FIT_MODES))
    vsync: bool = True
    max_fps: int = field(default=120, metadata=_range(30, 360))


@dataclass
class AudioSettings:
    enabled: bool = True
    volume: float = field(default=0.7, metadata=_range(0.0, 1.0))


@dataclass
class VirtualCameraSettings:
    # Offer the picture (camera + effects + callouts, no HUD) to video-call apps as a
    # webcam. Saved, so the virtual camera starts with the app once switched on.
    enabled: bool = False
    width: int = field(default=1280, metadata=_range(320, 1920))
    height: int = field(default=720, metadata=_range(240, 1080))
    fps: int = field(default=30, metadata=_range(10, 60))
    # "auto" tries every virtual camera driver pyvirtualcam knows (OBS first).
    backend: str = field(default="auto", metadata=_choices(*VIRTUAL_CAMERA_BACKENDS))
    # Device name for drivers with several devices (Unity Capture, v4l2loopback). Empty = first free.
    device: str = ""
    # False: the call sees you the right way round, like through a normal webcam (call apps
    # mirror only your own preview). True: send the mirrored picture of the app window.
    mirror: bool = False


SECTION_TYPES: dict[str, type] = {
    "camera": CameraSettings,
    "tracking": TrackingSettings,
    "gestures": GestureSettings,
    "effects": EffectsSettings,
    "abilities": AbilitySettings,
    "ui": UISettings,
    "audio": AudioSettings,
    "virtual_camera": VirtualCameraSettings,
}


@dataclass
class Settings:
    camera: CameraSettings = field(default_factory=CameraSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    gestures: GestureSettings = field(default_factory=GestureSettings)
    effects: EffectsSettings = field(default_factory=EffectsSettings)
    abilities: AbilitySettings = field(default_factory=AbilitySettings)
    ui: UISettings = field(default_factory=UISettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    virtual_camera: VirtualCameraSettings = field(default_factory=VirtualCameraSettings)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"version": SETTINGS_VERSION}
        for name in SECTION_TYPES:
            data[name] = dataclasses.asdict(getattr(self, name))
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "Settings":
        if not isinstance(data, dict):
            log.warning("Settings root is not an object; using defaults")
            data = {}
        sections = {}
        for name, section_type in SECTION_TYPES.items():
            sections[name] = _section_from_dict(section_type, data.get(name), name)
        settings = cls(**sections)
        settings._fix_cross_field_constraints()
        return settings

    def copy(self) -> "Settings":
        return Settings.from_dict(self.to_dict())

    def _fix_cross_field_constraints(self) -> None:
        g = self.gestures
        if g.release_distance <= g.together_distance:
            g.release_distance = min(5.0, g.together_distance + 0.5)
        if g.beam_max_duration < g.beam_min_duration:
            g.beam_max_duration = g.beam_min_duration
        e = self.effects
        if e.orb_max_radius < e.orb_min_radius:
            e.orb_max_radius = e.orb_min_radius
        v = self.virtual_camera
        # Virtual cameras transport YUV 4:2:0 (NV12): both dimensions must be even.
        v.width -= v.width % 2
        v.height -= v.height % 2


# ----------------------------------------------------------------- (de)serialising


def _coerce(value: Any, typ: Any, default: Any, meta: Any, path: str) -> Any:
    """Convert ``value`` to ``typ`` and enforce field metadata; fall back to default."""
    try:
        if typ is bool:
            if isinstance(value, bool):
                result: Any = value
            elif isinstance(value, (int, float)):
                result = bool(value)
            elif isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on"):
                result = True
            elif isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off"):
                result = False
            else:
                raise TypeError(type(value).__name__)
        elif typ is int:
            if isinstance(value, bool):
                raise TypeError("bool")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("not finite")
            result = int(round(number))
        elif typ is float:
            if isinstance(value, bool):
                raise TypeError("bool")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("not finite")
        elif typ is str:
            if not isinstance(value, str):
                raise TypeError(type(value).__name__)
            result = value.strip()
        else:  # unsupported field type: keep default
            return default
    except (TypeError, ValueError) as exc:
        log.warning("Invalid setting %s=%r (%s); using default %r", path, value, exc, default)
        return default

    choices = meta.get("choices") if meta else None
    if choices is not None and result not in choices:
        lowered = result.lower() if isinstance(result, str) else result
        if lowered in choices:
            result = lowered
        else:
            log.warning("Invalid setting %s=%r (allowed: %s); using default %r",
                        path, value, ", ".join(choices), default)
            return default
    if meta and "min" in meta and result < meta["min"]:
        log.warning("Setting %s=%r below minimum %r; clamped", path, result, meta["min"])
        result = typ(meta["min"])
    if meta and "max" in meta and result > meta["max"]:
        log.warning("Setting %s=%r above maximum %r; clamped", path, result, meta["max"])
        result = typ(meta["max"])
    return result


def _section_from_dict(section_type: type, data: Any, section_name: str) -> Any:
    instance = section_type()
    if data is None:
        return instance
    if not isinstance(data, dict):
        log.warning("Settings section %r is not an object; using defaults", section_name)
        return instance
    hints = get_type_hints(section_type)
    known = {f.name: f for f in dataclasses.fields(section_type)}
    for key, value in data.items():
        f = known.get(key)
        if f is None:
            log.info("Ignoring unknown setting %s.%s", section_name, key)
            continue
        default = getattr(instance, key)
        setattr(instance, key, _coerce(value, hints[key], default, f.metadata, f"{section_name}.{key}"))
    return instance


def get_setting(settings: Settings, dotted: str) -> Any:
    """Read a setting by dotted path, e.g. ``get_setting(s, "effects.quality")``."""
    section_name, _, key = dotted.partition(".")
    section = getattr(settings, section_name, None)
    if section is None or not key or not hasattr(section, key):
        raise KeyError(dotted)
    return getattr(section, key)


def set_setting(settings: Settings, dotted: str, value: Any) -> Any:
    """Validate and assign a setting by dotted path. Returns the value actually stored."""
    section_name, _, key = dotted.partition(".")
    section_type = SECTION_TYPES.get(section_name)
    if section_type is None:
        raise KeyError(dotted)
    known = {f.name: f for f in dataclasses.fields(section_type)}
    if key not in known:
        raise KeyError(dotted)
    section = getattr(settings, section_name)
    hints = get_type_hints(section_type)
    stored = _coerce(value, hints[key], getattr(section, key), known[key].metadata, dotted)
    setattr(section, key, stored)
    settings._fix_cross_field_constraints()
    return getattr(section, key)


def field_constraints(dotted: str) -> dict[str, Any]:
    """Return ``{"type": ..., "min"/"max"/"choices": ...}`` for a setting (used by the UI)."""
    section_name, _, key = dotted.partition(".")
    section_type = SECTION_TYPES[section_name]
    known = {f.name: f for f in dataclasses.fields(section_type)}
    hints = get_type_hints(section_type)
    info: dict[str, Any] = {"type": hints[key]}
    info.update(known[key].metadata)
    return info


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)  # atomic on Windows and POSIX


def _read_json(path: Path) -> Any | None:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return None
    except OSError as exc:
        log.error("Could not read %s: %s", path, exc)
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        log.error("%s is not valid JSON (%s); using defaults", path, exc)
        backup = path.with_name(path.name + ".invalid")
        try:
            os.replace(path, backup)
            log.error("The invalid file was moved to %s", backup)
        except OSError:
            pass
        return None


def settings_path(config_dir: Path | None = None) -> Path:
    return (config_dir or CONFIG_DIR) / SETTINGS_FILE


def load_settings(path: Path | None = None) -> Settings:
    """Load settings, creating the file with defaults if it does not exist."""
    path = path or settings_path()
    data = _read_json(path)
    if data is None:
        settings = Settings()
        try:
            save_settings(settings, path)
            log.info("Created default settings file %s", path)
        except OSError as exc:
            log.error("Could not create settings file %s: %s", path, exc)
        return settings
    return Settings.from_dict(data)


def save_settings(settings: Settings, path: Path | None = None) -> None:
    _atomic_write_json(path or settings_path(), settings.to_dict())


# -------------------------------------------------------------------- calibration


@dataclass
class Calibration:
    """Per-user gesture calibration (``config/calibration.json``).

    All distances are normalised by the body scale measured during calibration so
    they stay valid when the user stands closer to or further from the camera.
    """

    valid: bool = False
    shoulder_width_px: float = 0.0     # body scale at calibration time (scene px)
    torso_length_px: float = 0.0       # shoulder-mid to hip-mid (0 if hips not visible)
    scene_height: int = 0              # camera frame height during calibration
    hand_size_ratio: float = 0.0       # mean hand size / body scale
    together_distance: float = 0.0     # measured hands-together distance (body scales)
    rest_jitter: float = 0.0           # hand speed while holding still (body scales / s)
    thrust_peak: float = 0.0           # strongest thrust score seen during calibration
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "Calibration":
        cal = cls()
        if not isinstance(data, dict):
            return cal
        hints = get_type_hints(cls)
        for f in dataclasses.fields(cls):
            if f.name in data:
                setattr(cal, f.name, _coerce(data[f.name], hints[f.name], getattr(cal, f.name),
                                             {}, f"calibration.{f.name}"))
        for name in ("shoulder_width_px", "torso_length_px", "hand_size_ratio",
                     "together_distance", "rest_jitter", "thrust_peak"):
            if getattr(cal, name) < 0:
                setattr(cal, name, 0.0)
        return cal

    @staticmethod
    def timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")


def calibration_path(config_dir: Path | None = None) -> Path:
    return (config_dir or CONFIG_DIR) / CALIBRATION_FILE


def load_calibration(path: Path | None = None) -> Calibration:
    data = _read_json(path or calibration_path())
    return Calibration.from_dict(data) if data is not None else Calibration()


def save_calibration(calibration: Calibration, path: Path | None = None) -> None:
    _atomic_write_json(path or calibration_path(), calibration.to_dict())
