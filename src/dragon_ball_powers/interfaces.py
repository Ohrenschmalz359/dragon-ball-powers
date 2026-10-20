"""Interfaces between subsystems: abstract renderer, audio sink, frame source and UI types."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from dragon_ball_powers.config import Settings
from dragon_ball_powers.datatypes import (
    AbilityHUD,
    BodyState,
    CalibrationHUD,
    CameraFrame,
    CameraInfo,
    CameraStatus,
    GestureSignals,
    TrackerStatus,
    TrackingFrame,
)
from dragon_ball_powers.primitives import QualityPreset, RenderList, SceneInput, UIDrawList, ViewTransform

# ------------------------------------------------------------------------- audio

#: Every sound the audio engine must provide. Loops are started/steered with
#: ``AudioSink.loop`` and stopped with ``AudioSink.stop``; the rest are one-shots.
SOUND_NAMES: tuple[str, ...] = (
    "charge_start",   # one-shot: energy ignites between the hands
    "charge_loop",    # loop: humming / crackling while charging (volume ~ charge)
    "beam_fire",      # one-shot: Kamehameha release blast
    "beam_loop",      # loop: roaring beam while it is sustained
    "blast_fire",     # one-shot: ki blast thrown
    "impact",         # one-shot: explosion / hit
    "burst",          # one-shot: energy burst explosion
    "aura_loop",      # loop: aura flames
    "zap",            # one-shot: short electric crackle
    "power_up",       # one-shot: charge milestone / power level jump
    "fizzle",         # one-shot: charge cancelled
    "ui_click",       # one-shot: UI button
)
LOOP_SOUNDS: frozenset[str] = frozenset({"charge_loop", "beam_loop", "aura_loop"})


@runtime_checkable
class AudioSink(Protocol):
    def play(self, name: str, volume: float = 1.0) -> None:
        """Play a one-shot sound."""

    def loop(self, name: str, volume: float) -> None:
        """Start a looping sound if needed and set its volume (0 = silent, still running)."""

    def stop(self, name: str, fade_ms: int = 250) -> None:
        """Fade out and stop a looping sound."""


class NullAudio:
    """Audio sink that does nothing (audio disabled or unavailable)."""

    def play(self, name: str, volume: float = 1.0) -> None:
        pass

    def loop(self, name: str, volume: float) -> None:
        pass

    def stop(self, name: str, fade_ms: int = 250) -> None:
        pass


# ------------------------------------------------------------------------ camera


class FrameSource(Protocol):
    def wait_for_frame(self, after_id: int, timeout: float) -> CameraFrame | None:
        """Block until a frame newer than ``after_id`` exists (newest one is returned)."""


# ---------------------------------------------------------------------- renderer


class RendererError(RuntimeError):
    """Raised when a renderer backend cannot be created or used."""


class Renderer(abc.ABC):
    """A rendering backend. ``render`` draws one complete frame into its target."""

    name: str = "renderer"

    @abc.abstractmethod
    def resize(self, width: int, height: int) -> None:
        """The output (window) size changed."""

    @abc.abstractmethod
    def set_quality(self, preset: QualityPreset) -> None:
        """Change effect resolution / bloom levels etc."""

    @abc.abstractmethod
    def render(self, scene: SceneInput, effects: RenderList, ui: UIDrawList,
               view: ViewTransform) -> None:
        """Composite the camera image, the effects and the UI into the output."""

    @abc.abstractmethod
    def read_pixels(self) -> np.ndarray:
        """Return the last rendered frame as (H, W, 3) uint8 RGB, top row first."""

    @abc.abstractmethod
    def release(self) -> None:
        """Free all resources. Safe to call more than once."""

    @property
    def info(self) -> dict[str, str]:
        return {"renderer": self.name}


# ---------------------------------------------------------------------------- UI

#: Actions the UI can request from the application (``UIAction.kind``).
UI_ACTION_KINDS: tuple[str, ...] = (
    "select_camera",     # value: int camera index
    "rescan_cameras",    # value: None
    "toggle_camera",     # value: None (start if stopped, stop if running)
    "set_resolution",    # value: (width, height)
    "set_setting",       # value: (dotted_key, new_value), e.g. ("effects.quality", "ultra")
    "select_ability",    # value: ability id
    "toggle_aura",       # value: None
    "recalibrate",       # value: None
    "toggle_fullscreen",  # value: None
    "toggle_settings",   # value: None
    "toggle_debug",      # value: None
    "reset_settings",    # value: None
    "quit",              # value: None
)


@dataclass
class UIAction:
    kind: str
    value: Any = None


@dataclass
class UIContext:
    """Snapshot of application state the UI needs to draw one frame."""

    now: float
    window_size: tuple[int, int]
    view: ViewTransform
    settings: Settings
    camera_status: CameraStatus
    tracker_status: TrackerStatus
    ability: AbilityHUD
    calibration: CalibrationHUD
    cameras: list[CameraInfo] = field(default_factory=list)
    camera_scanning: bool = False
    body: BodyState | None = None
    signals: GestureSignals | None = None
    frame: TrackingFrame | None = None
    render_fps: float = 0.0
    renderer_name: str = ""
    quality_name: str = ""
    fullscreen: bool = False
