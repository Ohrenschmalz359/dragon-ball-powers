"""HUD, settings panel, overlays and debug view (immediate-mode UI).

:meth:`UIController.build` turns a :class:`~dragon_ball_powers.interfaces.UIContext`
snapshot into a :class:`~dragon_ball_powers.primitives.UIDrawList` in **window pixels**
every frame and, while doing so, registers the interactive widgets (button rectangles,
slider tracks). The next :meth:`UIController.handle_event` hit-tests mouse input against
those widgets and returns :class:`~dragon_ball_powers.interfaces.UIAction` requests; the
app applies them and the following frame shows the result. The UI never changes settings
itself: every displayed value comes from ``ctx``, so it cannot drift from the real state.

Look: a dark anime "scouter" HUD that hugs the window edges - translucent glass panels,
thin glowing accent lines in the energy palette's UI colour, condensed Rajdhani text,
Bangers for the power number and the callouts. Everything scales with
``u = max(0.6, min(window_h / 1080, window_w / 1280))`` (the width term only matters for
narrow windows); rows switch to compact labels or stack when space runs out, and an open
settings panel pushes the HUD into the remaining area on its left.

Cost control: both renderers cache text and shapes by content (the CPU renderer even by
colour), so animated values are quantised (alpha levels, font-size steps, a 4 Hz
statistics readout) and all text measurements are cached here.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame

from dragon_ball_powers import config
from dragon_ball_powers.config import ABILITY_IDS, Settings
from dragon_ball_powers.datatypes import (
    HAND_CONNECTIONS,
    POSE_CONNECTIONS,
    CameraInfo,
    CameraState,
)
from dragon_ball_powers.fonts import FontManager
from dragon_ball_powers.interfaces import UIAction, UIContext
from dragon_ball_powers.mathutil import Vec2, clamp, lerp
from dragon_ball_powers.primitives import (
    PALETTES,
    Color3,
    Color4,
    Palette,
    UIArc,
    UICircle,
    UICommand,
    UIDrawList,
    UILine,
    UIRect,
    UIText,
    get_palette,
)
from dragon_ball_powers.state_machine import AttackPhase

log = logging.getLogger(__name__)

# ------------------------------------------------------------------- behaviour

MIN_SCALE = 0.6                 # u never drops below this (legibility)
MIN_FONT_PX = 10
HELP_SECONDS = 12.0             # the key help strip is shown this long after start-up
HELP_FADE = 1.5
RESET_CONFIRM_SECONDS = 3.0     # "reset to defaults" needs a second click within this time
TOAST_FADE_IN = 0.18
TOAST_FADE_OUT = 0.35
TOAST_MAX_VISIBLE = 4
TOAST_MAX_QUEUED = 8
TOAST_LEVELS: tuple[str, ...] = ("info", "warning", "error")
MESSAGE_FADE_IN = 0.25
MESSAGE_FADE_OUT = 0.45
STATS_REFRESH = 0.25            # s between updates of the FPS / latency readout
SCROLL_STEP = 72.0              # settings scroll per wheel notch (px at u = 1)
SETTINGS_WIDTH = 430.0          # settings panel width at u = 1
OVER_9000 = 9000
ALPHA_LEVELS = 24               # quantisation of animated alpha (renderer cache friendly)

HELP_TEXT = ("SPACE charge/fire · 1-3 abilities · 4 aura · V video calls · F fullscreen · C calibrate · "
             "TAB settings · D debug · H hide HUD · ESC exit")

# ---------------------------------------------------------------------- colours

GLASS: Color4 = (0.03, 0.04, 0.09, 0.62)
GLASS_BOTTOM: Color4 = (0.02, 0.03, 0.065, 0.70)
PANEL_FILL: Color4 = (0.025, 0.033, 0.075, 0.86)
PANEL_FILL_BOTTOM: Color4 = (0.018, 0.024, 0.055, 0.90)
PANEL_BAR: Color4 = (0.03, 0.04, 0.088, 0.97)
TEXT: Color3 = (0.93, 0.96, 1.00)
TEXT_2: Color3 = (0.72, 0.78, 0.88)
TEXT_3: Color3 = (0.55, 0.61, 0.72)
DARK: Color3 = (0.02, 0.035, 0.07)
WHITE: Color3 = (1.0, 1.0, 1.0)
GREEN: Color3 = (0.32, 0.93, 0.56)
AMBER: Color3 = (1.00, 0.74, 0.24)
RED: Color3 = (1.00, 0.36, 0.33)
GREY: Color3 = (0.55, 0.59, 0.67)
GOLD: Color3 = (1.00, 0.82, 0.30)
HOT: Color3 = (1.00, 0.40, 0.18)
LEVEL_COLORS: dict[str, Color3] = {"warning": AMBER, "error": RED}
SOURCE_COLORS: dict[str, Color3] = {"hand": (0.30, 1.00, 0.62), "predicted": (1.00, 0.72, 0.22),
                                    "pose": (1.00, 0.45, 0.85), "none": GREY}
POSE_COLOR: Color3 = (0.55, 0.72, 1.00)

# ------------------------------------------------------------------ static data

#: Ability id -> (full label, compact label). Hotkeys follow the order of ABILITY_IDS.
ABILITY_LABELS: dict[str, tuple[str, str]] = {
    "ki_charge": ("KI CHARGE", "KI"),
    "kamehameha": ("KAMEHAMEHA", "KAME"),
    "energy_burst": ("ENERGY BURST", "BURST"),
}
RESOLUTIONS: tuple[tuple[int, int], ...] = ((640, 480), (1280, 720), (1920, 1080))
CAMERA_FPS_CHOICES: tuple[int, ...] = (30, 60)
VIRTUAL_CAMERA_RESOLUTIONS: tuple[tuple[int, int], ...] = ((1280, 720), (1920, 1080))
#: Display labels (full, short) for string choices; unknown choices fall back to upper case.
CHOICE_LABELS: dict[str, dict[str, tuple[str, str]]] = {
    "effects.quality": {"auto": ("AUTO", "AUTO"), "low": ("LOW", "LOW"), "medium": ("MEDIUM", "MED"),
                        "high": ("HIGH", "HIGH"), "ultra": ("ULTRA", "ULTRA")},
    "tracking.pose_model": {"lite": ("LITE", "LITE"), "full": ("FULL", "FULL"), "heavy": ("HEAVY", "HEAVY")},
    "ui.fit_mode": {"cover": ("FILL", "FILL"), "contain": ("FIT", "FIT")},
}
PHASE_LABELS: dict[str, str] = {AttackPhase.PREPARING.value: "FOCUS...", AttackPhase.FIRING.value: "FIRE!",
                                AttackPhase.COOLDOWN.value: "COOLDOWN", AttackPhase.IDLE.value: "STANDBY"}


@dataclass(frozen=True)
class SliderSpec:
    """A settings slider: UI range (intersected with the config constraints) and display."""

    dotted: str
    label: str
    lo: float
    hi: float
    step: float
    unit: str            # "%" (value 0..1 shown as percent), "s" seconds, "x" plain multiplier


SLIDERS: dict[str, SliderSpec] = {s.dotted: s for s in (
    SliderSpec("gestures.sensitivity", "Sensitivity", 0.0, 1.0, 0.01, "%"),
    SliderSpec("gestures.min_charge_time", "Min charge time", 0.0, 3.0, 0.05, "s"),
    SliderSpec("gestures.cooldown_time", "Cooldown", 0.0, 3.0, 0.05, "s"),
    SliderSpec("effects.bloom_strength", "Bloom", 0.0, 3.0, 0.05, "x"),
    SliderSpec("effects.screen_shake", "Screen shake", 0.0, 2.0, 0.05, "x"),
    SliderSpec("effects.camera_darken", "Room darkening", 0.0, 1.0, 0.01, "%"),
    SliderSpec("audio.volume", "Volume", 0.0, 1.0, 0.01, "%"),
)}

_constraint_cache: dict[str, dict[str, Any]] = {}


def _constraints(dotted: str) -> dict[str, Any]:
    """``config.field_constraints`` (cached: it is static and uses ``get_type_hints``)."""
    info = _constraint_cache.get(dotted)
    if info is None:
        try:
            info = dict(config.field_constraints(dotted))
        except (KeyError, AttributeError, TypeError):
            log.error("Unknown setting %s in the UI definition", dotted)
            info = {}
        _constraint_cache[dotted] = info
    return info


def slider_range(spec: SliderSpec) -> tuple[float, float, bool]:
    """``(lo, hi, is_int)`` of a slider: its UI range clipped to the config limits."""
    info = _constraints(spec.dotted)
    cfg_lo = float(info.get("min", spec.lo))
    cfg_hi = float(info.get("max", spec.hi))
    lo, hi = max(spec.lo, cfg_lo), min(spec.hi, cfg_hi)
    if hi <= lo:                       # UI range outside the config range: use the config's
        lo, hi = cfg_lo, cfg_hi
    return lo, hi, info.get("type") is int


def choice_values(dotted: str, fallback: Sequence[str]) -> tuple[str, ...]:
    """Allowed string values of a setting, straight from the config metadata."""
    choices = _constraints(dotted).get("choices")
    return tuple(choices) if choices else tuple(fallback)


def snap_value(value: float, lo: float, hi: float, step: float, is_int: bool = False) -> float | int:
    """Clamp ``value`` into ``[lo, hi]`` and snap it to ``lo + k * step``."""
    v = clamp(_finite(value, lo), lo, hi)
    if step > 0.0:
        v = clamp(lo + round((v - lo) / step) * step, lo, hi)
    if is_int:
        return int(round(v))
    return round(v, 6)                 # no 0.30000000000000004 in the settings file


def format_value(value: Any, unit: str) -> str:
    v = _finite(value, 0.0)
    if unit == "%":
        return f"{int(round(v * 100.0))}%"
    if unit == "s":
        return f"{v:.2f} s"
    return f"{v:.2f}"


# ---------------------------------------------------------------------- helpers


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _rgba(rgb: Sequence[float], alpha: float) -> Color4:
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), clamp(_finite(alpha, 0.0)))


def _mix(a: Sequence[float], b: Sequence[float], t: float) -> Color3:
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t))


def _quant(x: float, levels: int = ALPHA_LEVELS) -> float:
    """Quantise 0..1 so animated colours hit the renderers' caches."""
    return round(clamp(_finite(x, 0.0)) * levels) / levels


def _clean(text: Any) -> str:
    """Single-line printable text (control characters would render as boxes)."""
    s = "" if text is None else str(text)
    if any(ord(ch) < 32 for ch in s):
        s = "".join(" " if ord(ch) < 32 else ch for ch in s)
    return s.strip()


def _ui_scale(width: int, height: int) -> float:
    return max(MIN_SCALE, min(height / 1080.0, width / 1280.0))


def ui_scale(window_size: tuple[int, int]) -> float:
    """The layout scale ``u`` for a window size (1.0 at 1920x1080)."""
    return _ui_scale(max(1, int(window_size[0])), max(1, int(window_size[1])))


Rect = tuple[float, float, float, float]          # x, y, w, h (window px)


def _inside(rect: Rect, x: float, y: float) -> bool:
    return rect[0] <= x < rect[0] + rect[2] and rect[1] <= y < rect[1] + rect[3]


def _clip_rect(rect: Rect, y0: float, y1: float) -> Rect | None:
    top, bottom = max(rect[1], y0), min(rect[1] + rect[3], y1)
    if bottom - top < 1.0:
        return None
    return (rect[0], top, rect[2], bottom - top)


def _event_pos(event: Any) -> tuple[float, float] | None:
    pos = getattr(event, "pos", None)
    try:
        x, y = float(pos[0]), float(pos[1])  # type: ignore[index]
    except (TypeError, ValueError, IndexError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


# ------------------------------------------------------------------ public types


@dataclass(frozen=True)
class SliderBinding:
    """Maps the x position on a slider track to a setting value."""

    dotted: str
    lo: float
    hi: float
    step: float
    is_int: bool
    track_x0: float
    track_x1: float
    value: float          # value shown when the widget was built

    def value_at(self, x: float) -> float | int:
        span = max(1e-6, self.track_x1 - self.track_x0)
        t = clamp((x - self.track_x0) / span)
        return snap_value(self.lo + t * (self.hi - self.lo), self.lo, self.hi, self.step, self.is_int)


@dataclass(frozen=True)
class UIWidget:
    """An interactive area registered by the last :meth:`UIController.build`.

    ``kind`` is ``"button"`` (``action`` on click), ``"slider"`` (``slider`` binding) or
    ``"blocker"`` (a panel that swallows clicks so they do not reach widgets below).
    """

    id: str
    rect: Rect
    kind: str
    action: UIAction | None = None
    slider: SliderBinding | None = None
    enabled: bool = True

    def contains(self, x: float, y: float) -> bool:
        return _inside(self.rect, x, y)

    @property
    def center(self) -> tuple[float, float]:
        return (self.rect[0] + self.rect[2] * 0.5, self.rect[1] + self.rect[3] * 0.5)


@dataclass
class _Toast:
    text: str
    level: str
    duration: float
    start: float | None = None      # set by the first build after notify() (ctx time base)


# ---------------------------------------------------------------------- painter


class _Painter:
    """Collects the draw commands of one frame and owns the text-measurement caches.

    Measuring text costs ~10 us per call through pygame; a HUD frame measures a few
    hundred strings, almost all of them identical to the previous frame, so the results
    (and derived ellipsis / wrapping) are cached until the cache is full.
    """

    _CACHE_LIMIT = 4096

    def __init__(self, fonts: FontManager) -> None:
        self.fonts = fonts
        self.cmds: list[UICommand] = []
        self.u = 1.0
        self.hair = 1.0
        self._sizes: dict[tuple[str, int, str], tuple[int, int]] = {}
        self._fits: dict[tuple[str, int, str, int], str] = {}
        self._wraps: dict[tuple[str, int, str, int, int], tuple[str, ...]] = {}

    def begin(self, u: float) -> None:
        self.cmds = []
        self.u = u
        self.hair = float(max(1, round(1.25 * u)))

    # ----------------------------------------------------------------- scaling

    def px(self, value: float) -> float:
        return value * self.u

    def fs(self, base: float) -> int:
        """Font pixel size for a size given at u = 1."""
        return max(MIN_FONT_PX, int(round(base * self.u)))

    # ------------------------------------------------------------- measurement

    def measure(self, text: str, size: int, font: str = "ui") -> tuple[int, int]:
        key = (text, size, font)
        m = self._sizes.get(key)
        if m is None:
            try:
                m = self.fonts.measure(text, size, font)
            except (pygame.error, ValueError, TypeError, UnicodeError):
                m = (int(len(text) * size * 0.55), int(size * 1.3))
            if len(self._sizes) >= self._CACHE_LIMIT:
                self._sizes.clear()
            self._sizes[key] = m
        return m

    def width(self, text: str, size: int, font: str = "ui") -> int:
        return self.measure(text, size, font)[0]

    def height(self, size: int, font: str = "ui") -> int:
        return self.measure("Ag", size, font)[1]

    def fit(self, text: str, size: int, font: str, max_w: float) -> str:
        """``text`` shortened with an ellipsis so that it fits into ``max_w`` px."""
        if self.width(text, size, font) <= max_w:
            return text
        key = (text, size, font, int(max_w))
        cached = self._fits.get(key)
        if cached is not None:
            return cached
        lo, hi = 0, len(text)             # binary search for the longest fitting prefix
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.width(text[:mid].rstrip() + "…", size, font) <= max_w:
                lo = mid
            else:
                hi = mid - 1
        result = (text[:lo].rstrip() + "…") if lo > 0 else "…"
        if len(self._fits) >= self._CACHE_LIMIT:
            self._fits.clear()
        self._fits[key] = result
        return result

    def wrap(self, text: str, size: int, font: str, max_w: float, max_lines: int = 0) -> tuple[str, ...]:
        """Greedy word wrap; over-long words are broken; the last line gets an ellipsis
        when ``max_lines`` cuts the text."""
        key = (text, size, font, int(max_w), max_lines)
        cached = self._wraps.get(key)
        if cached is not None:
            return cached
        lines: list[str] = []
        for paragraph in str(text).splitlines() or [""]:
            words = _clean(paragraph).split()
            line = ""
            for word in words:
                candidate = f"{line} {word}" if line else word
                if self.width(candidate, size, font) <= max_w:
                    line = candidate
                    continue
                if line:
                    lines.append(line)
                while self.width(word, size, font) > max_w and len(word) > 1:
                    cut = self._longest_prefix(word, size, font, max_w)
                    lines.append(word[:cut])
                    word = word[cut:]
                line = word
            if line:
                lines.append(line)
        if not lines:
            lines = [""]
        if max_lines > 0 and len(lines) > max_lines:
            rest = " ".join(lines[max_lines - 1:])
            lines = lines[:max_lines - 1] + [self.fit(rest, size, font, max_w)]
        result = tuple(lines)
        if len(self._wraps) >= self._CACHE_LIMIT:
            self._wraps.clear()
        self._wraps[key] = result
        return result

    def _longest_prefix(self, word: str, size: int, font: str, max_w: float) -> int:
        lo, hi = 1, len(word)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.width(word[:mid], size, font) <= max_w:
                lo = mid
            else:
                hi = mid - 1
        return max(1, lo)

    # -------------------------------------------------------------- primitives

    def rect(self, x: float, y: float, w: float, h: float, color: Color4, radius: float = 0.0,
             border: Color4 | None = None, border_width: float = 0.0,
             gradient_to: Color4 | None = None) -> None:
        # Whole-pixel edges keep hairline borders crisp in both renderers.
        x0, y0 = round(x), round(y)
        x1, y1 = round(x + w), round(y + h)
        if x1 <= x0 or y1 <= y0:
            return
        if color[3] <= 0.0 and (gradient_to is None or gradient_to[3] <= 0.0) and (
                border is None or border[3] <= 0.0 or border_width <= 0.0):
            return
        self.cmds.append(UIRect(float(x0), float(y0), float(x1 - x0), float(y1 - y0), color,
                                max(0.0, float(radius)), border, float(border_width) if border else 0.0,
                                gradient_to))

    def text(self, text: str, x: float, y: float, size: int, color: Color4, font: str = "ui",
             align: str = "left", valign: str = "top", glow: float = 0.0,
             glow_color: Color4 | None = None, shadow: bool = False) -> None:
        if not text or color[3] <= 0.004:
            return
        self.cmds.append(UIText(text, float(x), float(y), int(size), color, font, align, valign,
                                float(glow), glow_color, shadow))

    def line(self, points: Sequence[Vec2], color: Color4, width: float = 1.0, closed: bool = False) -> None:
        if len(points) < 2 or color[3] <= 0.004:
            return
        self.cmds.append(UILine([(float(p[0]), float(p[1])) for p in points], color, float(width), closed))

    def circle(self, x: float, y: float, radius: float, color: Color4, width: float = 0.0) -> None:
        if radius <= 0.0 or color[3] <= 0.004:
            return
        self.cmds.append(UICircle(float(x), float(y), float(radius), color, float(width)))

    def arc(self, x: float, y: float, radius: float, start: float, end: float, color: Color4,
            width: float) -> None:
        if radius <= 0.0 or color[3] <= 0.004 or abs(end - start) < 1e-3:
            return
        self.cmds.append(UIArc(float(x), float(y), float(radius), float(start), float(end), color, float(width)))

    # ---------------------------------------------------------------- compound

    def glass(self, x: float, y: float, w: float, h: float, accent: Color3, *, radius: float | None = None,
              alpha: float = 1.0, border_alpha: float = 0.34, border: Color3 | None = None,
              tab: bool = True, fill: Color4 = GLASS, fill_bottom: Color4 = GLASS_BOTTOM) -> None:
        """Translucent dark glass panel with a thin accent border and a short bright tab."""
        r = self.px(10.0) if radius is None else radius
        top = (fill[0], fill[1], fill[2], fill[3] * alpha)
        bottom = (fill_bottom[0], fill_bottom[1], fill_bottom[2], fill_bottom[3] * alpha)
        edge = border if border is not None else accent
        self.rect(x, y, w, h, top, r, _rgba(edge, border_alpha * alpha), self.hair, bottom)
        if tab and w > 4.0 * r:
            # A short bright accent segment on the top edge: the "scouter" signature.
            # (A rect, not a line: translucent lines are the slow path of the CPU renderer.)
            length = min(self.px(42.0), w * 0.3)
            self.rect(x + r, y, length, max(2.0, self.hair * 2.0), _rgba(edge, 0.95 * alpha), self.hair)

    def dot(self, x: float, y: float, radius: float, rgb: Color3, alpha: float = 1.0) -> None:
        """Status dot with a soft halo."""
        self.circle(x, y, radius * 2.1, _rgba(rgb, 0.18 * alpha))
        self.circle(x, y, radius, _rgba(rgb, alpha))


# --------------------------------------------------------------- frame context


@dataclass
class _Frame:
    """Per-build values shared by the layout helpers."""

    ctx: UIContext
    s: Settings
    pal: Palette
    accent: Color3
    now: float
    w: int
    h: int
    u: float
    x0: float              # HUD area: the window minus an open settings panel
    x1: float
    m: float               # edge margin
    fullscreen: bool
    camera_state: str      # CameraState value
    camera_live: bool      # running or starting

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) * 0.5


@dataclass(frozen=True)
class _PowerLayout:
    """Geometry of the power block (y values relative to its top)."""

    w: float
    h: float
    pad: float
    lab_size: int
    lab_y: float
    num_size: int
    num_y: float
    meter_y: float
    meter_h: float
    segments: int
    row_size: int
    row_y: float
    bar_y: float
    bar_h: float


RESET_ID = "settings:reset"


@dataclass
class _BarItem:
    """One button of the bottom ability bar / HUD button row."""

    wid: str
    label: str
    action: UIAction
    width: float
    key: str = ""             # hotkey badge (ability bar)
    active: bool = False      # selected ability / aura on / panel open
    level: float = 0.0        # 0..1 meter along the bottom edge (aura intensity)
    icon: str = ""            # "gear" | "fullscreen" | "camera" | "camera_off"


class _Column:
    """Vertical layout cursor of the scrollable settings panel."""

    def __init__(self, f: _Frame, x0: float, x1: float, vy0: float, vy1: float, y: float) -> None:
        self.f = f
        self.x0 = x0
        self.x1 = x1
        self.vy0 = vy0        # visible viewport (window y) of the scrolled content
        self.vy1 = vy1
        self.y = y            # top of the next row (window y, already scrolled)

    @property
    def w(self) -> float:
        return max(1.0, self.x1 - self.x0)

    def visible(self, h: float) -> bool:
        """Whether a row of height ``h`` at the cursor intersects the viewport."""
        return self.y + h > self.vy0 and self.y < self.vy1


def _row_width(items: Sequence[_BarItem], gap: float) -> float:
    if not items:
        return 0.0
    return sum(item.width for item in items) + gap * (len(items) - 1)


def _camera_state(status: Any) -> str:
    """The camera state as its string value ("running", "starting", "error", "stopped")."""
    state = getattr(status, "state", None)
    if isinstance(state, CameraState):
        return state.value
    text = str(getattr(state, "value", state) or "").strip().lower()
    return text if text in {s.value for s in CameraState} else CameraState.STOPPED.value


def _camera_name(ctx: UIContext) -> str:
    """Display name of the active (or configured) camera, '' when unknown."""
    status = ctx.camera_status
    name = _clean(getattr(status, "name", "") or "")
    if name:
        return name
    index = int(_finite(getattr(status, "index", -1), -1.0))
    for cam in ctx.cameras or ():
        if isinstance(cam, CameraInfo) and cam.index == index:
            return _clean(cam.name)
    return f"Camera {index}" if index >= 0 else ""


def _hands_present(body: Any) -> int:
    if body is None:
        return 0
    return sum(1 for side in ("left", "right") if getattr(getattr(body, side, None), "present", False))


def _virtual_camera_state(status: Any) -> str:
    """The virtual camera state value ("off", "starting", "running", "error")."""
    state = getattr(status, "state", None)
    return str(getattr(state, "value", state) or "off").strip().lower()


def _virtual_camera_text(status: Any) -> str:
    """One-line explanation / status of the video-call output for the settings panel."""
    state = _virtual_camera_state(status)
    if state == "running":
        device = _clean(getattr(status, "device", "")) or "the virtual camera"
        res = getattr(status, "resolution", (0, 0)) or (0, 0)
        fps = _finite(getattr(status, "fps", 0.0))
        size = f" · {int(res[0])}×{int(res[1])} @ {fps:.0f} fps" if res[0] and res[1] else ""
        return f"Live: '{device}'{size}. Choose it as the camera in your call app."
    if state == "starting":
        return "Starting the virtual camera…"
    if state == "error":
        return _clean(getattr(status, "error", "")) or "The virtual camera failed."
    if sys.platform == "darwin":
        need = "needs OBS Studio 30+, see README"
    elif sys.platform == "win32":
        need = "run install_virtual_camera.bat once, or install OBS Studio"
    else:
        need = "needs v4l2loopback"
    return ("Show the effects in Discord, Zoom, Teams, Google Meet…: switch on, then pick the virtual "
            f"camera there ({need}).")


def _phase_name(phase: Any) -> str:
    """Normalise an AttackPhase (enum or string) to its lower-case value."""
    if isinstance(phase, AttackPhase):
        return phase.value
    return str(getattr(phase, "value", phase) or "idle").strip().lower() or "idle"


def _window_points(landmarks: Any, scale: float, ox: float, oy: float) -> list[Vec2 | None]:
    """Scene-pixel landmarks (N, >=2) -> window points; None for unusable entries."""
    if landmarks is None:
        return []
    try:
        arr = np.asarray(landmarks, dtype=np.float64)
    except (TypeError, ValueError):
        return []
    if arr.ndim != 2 or arr.shape[1] < 2:
        return []
    xs = arr[:, 0] * scale + ox
    ys = arr[:, 1] * scale + oy
    ok = np.isfinite(xs) & np.isfinite(ys)
    return [(float(x), float(y)) if good else None for x, y, good in zip(xs, ys, ok)]


def _float_list(values: Any, n: int) -> list[float]:
    """``values`` as ``n`` finite floats (missing / invalid entries become 0.0)."""
    out = [0.0] * n
    if values is None:
        return out
    try:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return out
    for i in range(min(n, arr.shape[0])):
        v = float(arr[i])
        out[i] = v if math.isfinite(v) else 0.0
    return out


def _key_id(key: Any) -> str:
    """Stable widget-id fragment for a segmented-control option key."""
    if isinstance(key, (tuple, list)):
        return "x".join(_key_id(k) for k in key)
    if isinstance(key, float) and key.is_integer():
        key = int(key)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(key))


# ------------------------------------------------------------------ controller


class UIController:
    """HUD, settings panel, overlays and debug view.

    The app forwards mouse events to :meth:`handle_event` (which returns the actions to
    apply), calls :meth:`build` once per frame and :meth:`notify` for toast messages.
    ``settings`` is only a fallback: every frame reads the live values from ``ctx``.
    """

    def __init__(self, settings: Settings, fonts: FontManager) -> None:
        self.settings_open = False
        self._settings = settings
        self._fonts = fonts
        self._p = _Painter(fonts)
        self._overlay_painter = _Painter(fonts)     # virtual camera overlay (own command list)
        self._widgets: list[UIWidget] = []
        self._reg: list[UIWidget] = []
        self._pointer: tuple[float, float] | None = None
        self._hover_id: str | None = None
        self._pressed_id: str | None = None
        self._drag: SliderBinding | None = None
        self._drag_id: str | None = None
        self._drag_last: float | int | None = None
        self._scroll = 0.0
        self._scroll_max = 0.0
        self._scroll_step = SCROLL_STEP
        self._panel_rect: Rect | None = None
        self._toasts: list[_Toast] = []
        self._now: float | None = None
        self._epoch: float | None = None
        self._help_level = 1.0
        self._reset_armed_at: float | None = None
        self._msg_text: str | None = None
        self._msg_since = 0.0
        self._msg_old: str | None = None
        self._msg_old_since = 0.0
        self._stats_at = -math.inf
        self._stats: tuple[str, str, str] = ("--", "--", "-- ms")
        self._stats_fps = 0.0
        self._debug_at = -math.inf
        self._debug_cache: list[tuple[str, str, float | None]] = []
        self._error_log: dict[str, float] = {}
        self.build_errors = 0

    # ------------------------------------------------------------- public API

    def toggle_settings(self) -> None:
        """Open or close the settings panel."""
        self.settings_open = not self.settings_open
        self._reset_armed_at = None
        self._end_drag()
        self._pressed_id = None

    def notify(self, text: str, level: str = "info", duration: float = 3.0) -> None:
        """Show a toast message (``level``: info | warning | error) for ``duration`` s."""
        message = _clean(text)
        if not message:
            return
        level = level if level in TOAST_LEVELS else "info"
        duration = clamp(_finite(duration, 3.0), 0.5, 60.0)
        for toast in self._toasts:
            if toast.text == message and toast.level == level:
                # Repeated message (e.g. key pressed twice): restart it instead of stacking.
                toast.start, toast.duration = None, duration
                self._toasts.remove(toast)
                self._toasts.append(toast)
                return
        self._toasts.append(_Toast(message, level, duration))
        del self._toasts[:-TOAST_MAX_QUEUED]

    def wants_mouse(self) -> bool:
        """True when the pointer is over the settings panel or a UI widget (or a drag runs)."""
        if self._drag is not None or self._pressed_id is not None:
            return True
        return self._pointer is not None and self._hit(self._pointer) is not None

    @property
    def widgets(self) -> list[UIWidget]:
        """Interactive areas registered by the last :meth:`build` (bottom to top)."""
        return list(self._widgets)

    def widget(self, widget_id: str) -> UIWidget | None:
        """The widget with ``widget_id`` from the last build, if it exists."""
        for w in self._widgets:
            if w.id == widget_id:
                return w
        return None

    @property
    def scroll_offset(self) -> float:
        """Current scroll position of the settings panel content (px)."""
        return self._scroll

    @property
    def scroll_max(self) -> float:
        """Largest valid scroll offset measured by the last build (0 = everything fits)."""
        return self._scroll_max

    @property
    def reset_armed(self) -> bool:
        """True while "reset to defaults" waits for its confirming second click."""
        return self._reset_armed_at is not None

    # ----------------------------------------------------------------- events

    def handle_event(self, event: pygame.event.Event) -> list[UIAction]:
        """Turn a mouse event into actions, using the widgets of the last build."""
        actions: list[UIAction] = []
        try:
            etype = getattr(event, "type", None)
            if etype == pygame.MOUSEMOTION:
                pos = _event_pos(event)
                if pos is not None:
                    self._pointer = pos
                    if self._drag is not None:
                        self._drag_to(pos[0], actions)
                    self._hover_id = self._hit_id(pos)
            elif etype == pygame.MOUSEBUTTONDOWN:
                pos = _event_pos(event)
                if pos is not None:
                    self._pointer = pos
                    # Buttons 4/5 are the legacy wheel events (MOUSEWHEEL handles scrolling).
                    if getattr(event, "button", 0) == 1:
                        self._press(pos, actions)
            elif etype == pygame.MOUSEBUTTONUP:
                pos = _event_pos(event)
                if pos is not None:
                    self._pointer = pos
                if getattr(event, "button", 0) == 1:
                    self._release(pos, actions)
            elif etype == pygame.MOUSEWHEEL:
                self._wheel(event)
        except Exception:  # noqa: BLE001 - UI input must never break the main loop
            self._report("event", "UI event handling failed")
        return actions

    def _press(self, pos: tuple[float, float], actions: list[UIAction]) -> None:
        widget = self._hit(pos)
        self._pressed_id = widget.id if widget is not None else None
        if widget is None or not widget.enabled:
            return
        if widget.kind == "slider" and widget.slider is not None:
            self._drag, self._drag_id, self._drag_last = widget.slider, widget.id, None
            self._drag_to(pos[0], actions)        # clicking the track jumps to the value

    def _release(self, pos: tuple[float, float] | None, actions: list[UIAction]) -> None:
        pressed, self._pressed_id = self._pressed_id, None
        if self._drag is not None:
            if pos is not None:
                self._drag_to(pos[0], actions)
            self._end_drag()
            return
        if pressed is None or pos is None:
            return
        widget = self.widget(pressed)
        if widget is None or widget.kind != "button" or not widget.enabled or not widget.contains(*pos):
            return                                # released elsewhere: a cancelled click
        if widget.id == RESET_ID:
            self._click_reset(actions)
        elif widget.action is not None:
            actions.append(UIAction(widget.action.kind, widget.action.value))

    def _drag_to(self, x: float, actions: list[UIAction]) -> None:
        if self._drag is None:
            return
        value = self._drag.value_at(x)
        if self._drag_last is not None and value == self._drag_last:
            return                                # snapped value unchanged: no new request
        self._drag_last = value
        actions.append(UIAction("set_setting", (self._drag.dotted, value)))

    def _end_drag(self) -> None:
        self._drag = None
        self._drag_id = None
        self._drag_last = None

    def _click_reset(self, actions: list[UIAction]) -> None:
        now = self._time()
        armed = self._reset_armed_at
        if armed is not None and 0.0 <= now - armed <= RESET_CONFIRM_SECONDS:
            self._reset_armed_at = None
            actions.append(UIAction("reset_settings"))
        else:
            self._reset_armed_at = now            # first click: ask for confirmation

    def _wheel(self, event: pygame.event.Event) -> None:
        if not self.settings_open or self._panel_rect is None:
            return
        if self._pointer is not None and not _inside(self._panel_rect, *self._pointer):
            return
        dy = _finite(getattr(event, "precise_y", None), _finite(getattr(event, "y", 0.0)))
        if getattr(event, "flipped", False):
            dy = -dy                              # "natural" scrolling reports inverted values
        self._scroll = clamp(self._scroll - dy * self._scroll_step, 0.0, self._scroll_max)

    def _hit(self, pos: tuple[float, float] | None) -> UIWidget | None:
        if pos is None:
            return None
        for widget in reversed(self._widgets):    # last registered = drawn on top
            if widget.contains(pos[0], pos[1]):
                return widget
        return None

    def _hit_id(self, pos: tuple[float, float] | None) -> str | None:
        widget = self._hit(pos)
        return widget.id if widget is not None and widget.enabled else None

    def _time(self) -> float:
        """Time base of the last build (the app's ``ctx.now``)."""
        return self._now if self._now is not None else time.perf_counter()

    def _report(self, key: str, message: str) -> None:
        """Log an unexpected exception with traceback, at most every 10 s per key."""
        now = time.perf_counter()
        last = self._error_log.get(key)
        if last is None or now - last > 10.0:
            self._error_log[key] = now
            log.exception(message)
        else:
            log.debug(message, exc_info=True)

    # ------------------------------------------------------- widget registry

    def _button(self, wid: str, rect: Rect, action: UIAction | None, enabled: bool = True,
                clip: tuple[float, float] | None = None) -> tuple[bool, bool]:
        """Register a clickable area; returns ``(hovered, pressed)`` for its styling.

        ``clip`` limits the area to a vertical band (the scrolled settings viewport).
        """
        if clip is not None:
            clipped = _clip_rect(rect, clip[0], clip[1])
            if clipped is None:
                return False, False
            rect = clipped
        self._reg.append(UIWidget(wid, rect, "button", action, None, enabled))
        hovered = enabled and self._hover_id == wid
        return hovered, hovered and self._pressed_id == wid

    def _blocker(self, wid: str, rect: Rect) -> None:
        self._reg.append(UIWidget(wid, rect, "blocker"))


    # ------------------------------------------------------------------ build

    def build(self, ctx: UIContext) -> UIDrawList:
        """Draw commands for this frame (window px); registers widgets for the next events."""
        try:
            return self._build(ctx)
        except Exception:  # noqa: BLE001 - a UI bug must never take the main loop down
            self.build_errors += 1
            self._report("build", "Building the UI failed")
            self._widgets, self._reg = [], []
            self._panel_rect = None
            return UIDrawList()

    def build_overlay(self, ctx: UIContext) -> UIDrawList:
        """Draw commands for the virtual camera picture: only the big callout, laid out for
        ``ctx.window_size`` (no HUD, no widgets; the interactive state is not touched)."""
        try:
            w = max(1, int(_finite(ctx.window_size[0], 1.0)))
            h = max(1, int(_finite(ctx.window_size[1], 1.0)))
            u = _ui_scale(w, h)
            settings = ctx.settings if isinstance(ctx.settings, Settings) else self._settings
            palette = get_palette(settings.effects.energy_color)
            p = self._overlay_painter
            p.begin(u)
            f = _Frame(ctx=ctx, s=settings, pal=palette, accent=palette.ui, now=_finite(ctx.now), w=w, h=h, u=u,
                       x0=0.0, x1=float(w), m=p.px(20.0), fullscreen=True,
                       camera_state=_camera_state(ctx.camera_status), camera_live=True)
            self._build_callout(f, p)
            return UIDrawList(p.cmds)
        except Exception:  # noqa: BLE001 - the video-call picture is drawn without callout then
            self._report("overlay", "Building the virtual camera overlay failed")
            return UIDrawList()

    def _build(self, ctx: UIContext) -> UIDrawList:
        previous = self._now
        now = _finite(ctx.now, previous if previous is not None else 0.0)
        dt = 0.0 if previous is None else clamp(now - previous, 0.0, 0.5)
        self._now = now
        if self._epoch is None:
            self._epoch = now      # "start-up" = first frame (same time base as ctx.now)
        win_w = max(1, int(_finite(ctx.window_size[0], 1.0)))
        win_h = max(1, int(_finite(ctx.window_size[1], 1.0)))
        u = _ui_scale(win_w, win_h)
        settings = ctx.settings if isinstance(ctx.settings, Settings) else self._settings
        palette = get_palette(settings.effects.energy_color)
        p = self._p
        p.begin(u)
        self._reg = []
        self._scroll_step = SCROLL_STEP * u
        if self._reset_armed_at is not None and not 0.0 <= now - self._reset_armed_at <= RESET_CONFIRM_SECONDS:
            self._reset_armed_at = None
        state = _camera_state(ctx.camera_status)
        panel_w = self._panel_width(win_w, u) if self.settings_open else 0.0
        f = _Frame(ctx=ctx, s=settings, pal=palette, accent=palette.ui, now=now, w=win_w, h=win_h, u=u,
                   x0=0.0, x1=float(win_w) - panel_w, m=p.px(20.0), fullscreen=bool(ctx.fullscreen),
                   camera_state=state, camera_live=state in ("running", "starting"))

        # The help strip fades instead of popping; the bottom row slides with it.
        want_help = self.settings_open or (not f.fullscreen and settings.ui.show_hud
                                           and now - self._epoch < HELP_SECONDS)
        step = dt / 0.5
        self._help_level = clamp(self._help_level + (step if want_help else -step))

        debug = bool(settings.ui.show_debug)
        if debug:
            self._build_debug_skeleton(f)
        top = (f.x0 + f.m, f.x1 - f.m, f.m)        # left block right, right block left, bottom
        if settings.ui.show_hud:
            top = self._build_top(f)
            self._build_bottom(f)
        if self._help_level > 0.01:
            self._build_help(f, self._help_level)
        self._build_callout(f)
        if debug:
            self._build_debug_panel(f, top[2])
        self._build_camera_overlay(f)
        banner_bottom = self._build_tracker_banner(f, top)
        self._build_calibration(f)
        self._build_toasts(f, top, banner_bottom)
        if self.settings_open:
            self._build_settings(f, panel_w)
        else:
            self._panel_rect = None
        self._widgets = self._reg
        self._hover_id = self._hit_id(self._pointer)
        if self._drag_id is not None and self.widget(self._drag_id) is None:
            self._end_drag()                       # the slider is gone (panel closed)
        return UIDrawList(p.cmds)

    @staticmethod
    def _panel_width(win_w: int, u: float) -> float:
        width = max(SETTINGS_WIDTH * u, min(320.0, float(win_w)))
        return min(width, float(win_w))

    # -------------------------------------------------------------- top HUD

    def _build_top(self, f: _Frame) -> tuple[float, float, float]:
        """Title + status chips (left) and statistics (right); returns their extents."""
        left_x1, right_x0, bottom = f.x0 + f.m, f.x1 - f.m, f.m
        if f.s.ui.show_fps:
            right_x0, b = self._build_stats(f)
            bottom = max(bottom, b)
        if not f.fullscreen:
            left_x1, b = self._build_identity(f, right_x0 - self._p.px(16.0))
            bottom = max(bottom, b)
        else:
            # Cinematic mode drops the status chips, but "sent to other programs" stays visible.
            for label, value, rgb in self._status_chips(f):
                if label == "VCAM":
                    p = self._p
                    fitted, w = self._chip_size(label, value, p.px(240.0))
                    h = p.px(30.0)
                    self._chip(f, f.x0 + f.m, f.m, w, h, label, fitted, rgb)
                    left_x1, bottom = max(left_x1, f.x0 + f.m + w), max(bottom, f.m + h)
        return left_x1, right_x0, bottom

    def _build_stats(self, f: _Frame) -> tuple[float, float]:
        p, ctx = self._p, f.ctx
        if not 0.0 <= f.now - self._stats_at < STATS_REFRESH:
            # Refresh the digits a few times per second: readable, and fewer text textures.
            self._stats_at = f.now
            ts = ctx.tracker_status
            fps = max(0.0, _finite(ctx.render_fps))
            track = max(0.0, _finite(getattr(ts, "fps", 0.0)))
            latency = max(0.0, _finite(getattr(ts, "latency_ms", 0.0)))
            self._stats_fps = fps
            self._stats = (f"{fps:.0f}" if fps >= 0.5 else "--", f"{track:.0f}" if track >= 0.5 else "--",
                           f"~{latency:.0f} ms" if latency >= 0.5 else "-- ms")
        fps_s, track_s, lat_s = self._stats
        fps_rgb = RED if 0.0 < self._stats_fps < 15.0 else AMBER if 0.0 < self._stats_fps < 25.0 else TEXT
        size, small = p.fs(16.5), p.fs(13.5)
        parts = (("FPS ", TEXT_3), (fps_s, fps_rgb), (" · TRACK ", TEXT_3), (track_s, TEXT),
                 (" · ", TEXT_3), (lat_s, TEXT))
        widths = [p.width(text, size) for text, _ in parts]
        line_w = float(sum(widths))
        line_h = p.height(size)
        info = ""
        if not f.fullscreen:
            renderer, quality = _clean(ctx.renderer_name), _clean(ctx.quality_name).upper()
            info = " · ".join(x for x in (renderer, quality) if x)
        info_h = p.height(small) if info else 0
        pad_x, pad_y = p.px(12.0), p.px(6.0)
        max_w = max(p.px(80.0), (f.x1 - f.x0) * 0.5 - f.m)
        info = p.fit(info, small, "ui", max_w) if info else ""
        w = max(line_w, float(p.width(info, small)) if info else 0.0) + 2.0 * pad_x
        h = 2.0 * pad_y + line_h + (info_h - p.px(3.0) if info else 0.0)
        x, y = f.x1 - f.m - w, f.m
        p.glass(x, y, w, h, f.accent, radius=p.px(9.0), tab=False)
        tx = x + w - pad_x - line_w
        for (text, rgb), tw in zip(parts, widths):
            p.text(text, tx, y + pad_y, size, _rgba(rgb, 0.96))
            tx += tw
        if info:
            p.text(info, x + w - pad_x, y + pad_y + line_h - p.px(3.0), small, _rgba(TEXT_3, 0.92), align="right")
        return x, y + h

    def _build_identity(self, f: _Frame, max_x: float) -> tuple[float, float]:
        p = self._p
        x, y = f.x0 + f.m, f.m
        size = p.fs(23.0)
        th = p.height(size, "bold")
        bar_w = max(2.0, p.px(3.5))
        p.rect(x, y + th * 0.22, bar_w, th * 0.56, _rgba(f.accent, 1.0), radius=bar_w * 0.5)
        tx = x + bar_w + p.px(9.0)
        p.text("KI POWERS", tx, y, size, _rgba(TEXT, 0.98), "bold", glow=p.px(6.0),
               glow_color=_rgba(f.accent, 0.5), shadow=True)
        right = tx + p.width("KI POWERS", size, "bold")
        chip_h, gap = p.px(30.0), p.px(8.0)
        avail = max(p.px(140.0), max_x - x)
        cursor, cy = x, y + th + p.px(3.0)
        for label, value, rgb in self._status_chips(f):
            fitted, w = self._chip_size(label, value, min(avail, p.px(340.0)))
            if cursor > x and cursor + w > x + avail:          # wrap onto the next line
                cursor, cy = x, cy + chip_h + p.px(6.0)
            self._chip(f, cursor, cy, w, chip_h, label, fitted, rgb)
            right = max(right, cursor + w)
            cursor += w + gap
        return right, cy + chip_h

    def _status_chips(self, f: _Frame) -> list[tuple[str, str, Color3]]:
        ctx, state = f.ctx, f.camera_state
        word, cam_rgb = {"running": ("LIVE", GREEN), "starting": ("STARTING", AMBER),
                         "error": ("ERROR", RED)}.get(state, ("OFF", GREY))
        name = _camera_name(ctx)
        cam_value = f"{word} · {name}" if name and state != "stopped" else word
        body = ctx.body
        hands = _hands_present(body)
        tracked = body is not None and getattr(body, "quality", "none") in ("pose", "full")
        if state != "running":                   # nothing can be tracked: neutral, not alarming
            hands_rgb, body_rgb, body_value = GREY, GREY, "—"
        else:
            hands_rgb = GREEN if hands >= 2 else AMBER if hands == 1 else RED
            body_rgb, body_value = (GREEN, "OK") if tracked else (RED, "NONE")
        chips = [("CAMERA", cam_value, cam_rgb), ("HANDS", f"{hands}/2", hands_rgb), ("BODY", body_value, body_rgb)]
        vcam = _virtual_camera_state(ctx.virtual_camera)
        if vcam in ("running", "starting"):       # failures are reported by a toast + settings
            chips.append(("VCAM", "LIVE" if vcam == "running" else "STARTING", GREEN if vcam == "running" else AMBER))
        return chips

    def _chip_size(self, label: str, value: str, max_w: float) -> tuple[str, float]:
        p = self._p
        ls, vs = p.fs(13.0), p.fs(15.5)
        fixed = p.px(11.0) * 2 + p.px(4.0) * 2 + p.px(8.0) + p.width(label, ls, "bold") + p.px(7.0)
        fitted = p.fit(value, vs, "ui", max(p.px(24.0), max_w - fixed))
        return fitted, fixed + p.width(fitted, vs)

    def _chip(self, f: _Frame, x: float, y: float, w: float, h: float, label: str, value: str,
              rgb: Color3) -> None:
        p = self._p
        ls, vs = p.fs(13.0), p.fs(15.5)
        p.glass(x, y, w, h, f.accent, radius=h * 0.5, tab=False, border=rgb, border_alpha=0.42)
        dot_r = p.px(4.0)
        dx = x + p.px(11.0) + dot_r
        cy = y + h * 0.5
        p.dot(dx, cy, dot_r, rgb)
        lx = dx + dot_r + p.px(8.0)
        p.text(label, lx, cy, ls, _rgba(TEXT_3, 0.95), "bold", valign="middle")
        p.text(value, lx + p.width(label, ls, "bold") + p.px(7.0), cy, vs, _rgba(TEXT, 0.96), valign="middle")

    # ----------------------------------------------------------- bottom HUD

    def _help_layout(self, f: _Frame) -> tuple[int, tuple[str, ...]]:
        p = self._p
        size = p.fs(14.5)
        return size, self._pack_items(HELP_TEXT.split(" · "), size, (f.x1 - f.x0) - 2.0 * f.m)

    def _pack_items(self, items: Sequence[str], size: int, max_w: float) -> tuple[str, ...]:
        """Join items with " · " into as few lines as fit into ``max_w``."""
        p = self._p
        lines: list[str] = []
        line = ""
        for item in items:
            candidate = f"{line} · {item}" if line else item
            if not line or p.width(candidate, size) <= max_w:
                line = candidate
            else:
                lines.append(line)
                line = item
        if line:
            lines.append(line)
        return tuple(p.fit(text, size, "ui", max_w) for text in lines)

    def _help_reserve(self, f: _Frame) -> float:
        """Space kept free for the help strip below the bottom row (animated)."""
        if self._help_level <= 0.0:
            return 0.0
        size, lines = self._help_layout(f)
        return (len(lines) * self._p.height(size) + self._p.px(2.0)) * self._help_level

    def _build_help(self, f: _Frame, level: float) -> None:
        p = self._p
        size, lines = self._help_layout(f)
        lh = p.height(size)
        y = f.h - p.px(6.0) - lh * len(lines)
        color = _rgba(TEXT_2, 0.66 * _quant(level, 12))
        for i, text in enumerate(lines):
            p.text(text, f.cx, y + i * lh, size, color, align="center", shadow=True)

    def _build_bottom(self, f: _Frame) -> None:
        p = self._p
        bottom = f.h - f.m - self._help_reserve(f)
        if f.fullscreen:
            # Cinematic mode: only a compact power block (and the hint above it).
            lay = self._power_layout(f, compact=True)
            y = bottom - lay.h
            self._power_block(f, lay, f.cx - lay.w * 0.5, y)
            self._ability_message(f, y - p.px(10.0))
            return
        bar_h, gap = p.px(42.0), p.px(14.0)
        lay = self._power_layout(f, compact=False)
        left, right = f.x0 + f.m, f.x1 - f.m
        options = ((False, False), (False, True), (True, True))
        plan: tuple[int, list[_BarItem], list[_BarItem]] | None = None
        for compact_abilities, compact_buttons in options:     # everything in one row?
            ab, bt = self._ability_items(f, compact_abilities), self._hud_buttons(f, compact_buttons)
            if (left + _row_width(ab, p.px(8.0)) + gap <= f.cx - lay.w * 0.5
                    and f.cx + lay.w * 0.5 + gap <= right - _row_width(bt, p.px(8.0))):
                plan = (1, ab, bt)
                break
        if plan is None:                                       # power block above the row
            for compact_abilities, compact_buttons in options:
                ab, bt = self._ability_items(f, compact_abilities), self._hud_buttons(f, compact_buttons)
                if _row_width(ab, p.px(8.0)) + gap + _row_width(bt, p.px(8.0)) <= right - left:
                    plan = (2, ab, bt)
                    break
        if plan is None:                                       # very narrow: three rows
            plan = (3, self._ability_items(f, True), self._hud_buttons(f, True))
        rows, ab, bt = plan
        row_y = bottom - bar_h
        if rows == 1:
            ab_y, pb_y = row_y, bottom - lay.h
        elif rows == 2:
            ab_y, pb_y = row_y, row_y - gap - lay.h
        else:
            ab_y = row_y - p.px(8.0) - bar_h
            pb_y = ab_y - gap - lay.h
        self._draw_bar(f, ab, left, ab_y, bar_h, icons=False)
        self._draw_bar(f, bt, right - _row_width(bt, p.px(8.0)), row_y, bar_h, icons=True)
        self._power_block(f, lay, f.cx - lay.w * 0.5, pb_y)
        self._ability_message(f, pb_y - p.px(10.0))

    def _ability_items(self, f: _Frame, compact: bool) -> list[_BarItem]:
        p, a = self._p, f.ctx.ability
        selected = _clean(getattr(a, "selected", "")) or f.s.abilities.selected
        size, badge = p.fs(16.0), p.px(24.0)
        items: list[_BarItem] = []
        for i, aid in enumerate(ABILITY_IDS):
            full, short = ABILITY_LABELS.get(aid, (aid.replace("_", " ").upper(),) * 2)
            label = short if compact else full
            w = p.px(8.0) + badge + p.px(9.0) + p.width(label, size, "bold") + p.px(14.0)
            items.append(_BarItem(f"hud:ability:{aid}", label, UIAction("select_ability", aid), w,
                                  key=str(i + 1), active=aid == selected))
        aura_on = bool(getattr(a, "aura_enabled", False))
        label = "AURA" if compact else ("AURA ON" if aura_on else "AURA OFF")
        w = p.px(8.0) + badge + p.px(9.0) + p.width(label, size, "bold") + p.px(14.0)
        items.append(_BarItem("hud:aura", label, UIAction("toggle_aura"), w, key=str(len(ABILITY_IDS) + 1),
                              active=aura_on, level=clamp(_finite(getattr(a, "aura_intensity", 0.0)))))
        return items

    def _hud_buttons(self, f: _Frame, compact: bool) -> list[_BarItem]:
        p = self._p
        size = p.fs(15.5)
        cam_on = f.camera_live
        specs = (("hud:settings", "SETTINGS", "gear", UIAction("toggle_settings"), self.settings_open),
                 ("hud:fullscreen", "FULLSCREEN", "fullscreen", UIAction("toggle_fullscreen"), False),
                 ("hud:camera", "CAMERA ON" if cam_on else "CAMERA OFF", "camera" if cam_on else "camera_off",
                  UIAction("toggle_camera"), False))
        items: list[_BarItem] = []
        for wid, label, icon, action, active in specs:
            if compact:
                items.append(_BarItem(wid, "", action, p.px(42.0), icon=icon, active=active))
            else:
                w = p.px(12.0) + p.px(18.0) + p.px(9.0) + p.width(label, size, "bold") + p.px(14.0)
                items.append(_BarItem(wid, label, action, w, icon=icon, active=active))
        return items

    def _draw_bar(self, f: _Frame, items: list[_BarItem], x: float, y: float, h: float, icons: bool) -> None:
        p = self._p
        for item in items:
            hovered, pressed = self._button(item.wid, (x, y, item.width, h), item.action)
            self._bar_button(f, item, x, y, item.width, h, hovered, pressed, icons)
            x += item.width + p.px(8.0)

    def _bar_button(self, f: _Frame, item: _BarItem, x: float, y: float, w: float, h: float,
                    hovered: bool, pressed: bool, icons: bool) -> None:
        p, acc = self._p, f.accent
        r = p.px(9.0)
        warn = item.icon == "camera_off"
        edge = RED if warn else acc
        if pressed:
            fill, fill_b = _rgba(_mix(GLASS[:3], acc, 0.42), 0.88), _rgba(_mix(GLASS_BOTTOM[:3], acc, 0.3), 0.9)
        elif item.active:
            fill, fill_b = _rgba(_mix(GLASS[:3], acc, 0.30), 0.80), _rgba(_mix(GLASS_BOTTOM[:3], acc, 0.14), 0.84)
        elif hovered:
            fill, fill_b = _rgba(_mix(GLASS[:3], acc, 0.14), 0.74), GLASS_BOTTOM
        else:
            fill, fill_b = GLASS, GLASS_BOTTOM
        border_a = 0.95 if item.active else 0.62 if hovered else 0.45 if warn else 0.26
        if item.active:                       # soft outer glow ring around the selection
            g = p.px(3.0)
            p.rect(x - g, y - g, w + 2 * g, h + 2 * g, (0.0, 0.0, 0.0, 0.0), r + g, _rgba(acc, 0.22), g)
        p.rect(x, y, w, h, fill, r, _rgba(edge, border_a), p.hair, fill_b)
        text_rgb = TEXT if (item.active or hovered) else TEXT_2
        cy = y + h * 0.5
        if icons:
            size = p.px(18.0)
            ix = x + (w * 0.5 if not item.label else p.px(12.0) + size * 0.5)
            self._icon(item.icon, ix, cy, size * 0.5, RED if warn else (TEXT if hovered else TEXT_2))
            if item.label:
                p.text(item.label, ix + size * 0.5 + p.px(9.0), cy, p.fs(15.5), _rgba(text_rgb, 0.97), "bold",
                       valign="middle")
            return
        bs = p.px(24.0)
        bx, by = x + p.px(8.0), cy - bs * 0.5
        if item.active:
            p.rect(bx, by, bs, bs, _rgba(acc, 0.95), p.px(5.0))
        else:
            p.rect(bx, by, bs, bs, (1.0, 1.0, 1.0, 0.06), p.px(5.0), _rgba(acc, 0.35), p.hair)
        p.text(item.key, bx + bs * 0.5, cy, p.fs(14.0), _rgba(DARK if item.active else TEXT_2, 1.0), "bold",
               "center", "middle")
        p.text(item.label, bx + bs + p.px(9.0), cy, p.fs(16.0), _rgba(text_rgb, 0.97), "bold", valign="middle")
        if item.level > 0.01:                 # aura intensity meter along the bottom edge
            mw = (w - 2.0 * r) * _quant(item.level, 20)
            p.rect(x + r, y + h - p.px(5.0), mw, max(1.0, p.px(2.0)), _rgba(acc, 0.9), p.px(1.0))

    def _icon(self, name: str, cx: float, cy: float, r: float, rgb: Color3) -> None:
        """Small line icons (opaque strokes: cheapest path in both renderers)."""
        p = self._p
        col = _rgba(rgb, 1.0)
        sw = max(1.5, p.px(2.0))
        if name == "gear":
            p.circle(cx, cy, r * 0.48, col, sw)
            for k in range(8):
                ang = k * math.tau / 8.0
                c, s = math.cos(ang), math.sin(ang)
                p.line([(cx + c * r * 0.66, cy + s * r * 0.66), (cx + c * r * 0.98, cy + s * r * 0.98)], col, sw * 1.3)
        elif name == "fullscreen":
            k = r * 0.5
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                corner = (cx + sx * r, cy + sy * r * 0.85)
                p.line([(corner[0], corner[1] - sy * k), corner, (corner[0] - sx * k, corner[1])], col, sw)
        else:                                 # video camera, crossed out when off
            p.rect(cx - r, cy - r * 0.58, r * 1.3, r * 1.16, (0.0, 0.0, 0.0, 0.0), r * 0.22, col, sw)
            p.line([(cx + r * 0.42, cy - r * 0.2), (cx + r, cy - r * 0.55), (cx + r, cy + r * 0.55),
                    (cx + r * 0.42, cy + r * 0.2)], col, sw, closed=True)
            if name == "camera_off":
                p.line([(cx - r * 0.95, cy + r * 0.95), (cx + r * 0.95, cy - r * 0.95)], _rgba(RED, 1.0), sw * 1.2)

    # ---------------------------------------------------------- power block

    def _power_layout(self, f: _Frame, compact: bool) -> _PowerLayout:
        p = self._p
        if compact:
            width, pad, lab, num, meter, segments, row, bar, bottom = 300.0, 12.0, 12.0, 40.0, 6.0, 24, 13.0, 3.0, 8.0
        else:
            width, pad, lab, num, meter, segments, row, bar, bottom = 380.0, 16.0, 13.5, 56.0, 9.0, 30, 14.5, 4.0, 10.0
        w = max(p.px(160.0), min(p.px(width), (f.x1 - f.x0) - 2.0 * f.m))
        lab_size, num_size, row_size = p.fs(lab), p.fs(num), p.fs(row)
        lab_y = p.px(8.0)
        lab_h = p.height(lab_size, "bold")
        num_y = lab_y + lab_h * 0.55          # Bangers has generous internal leading at the top
        meter_y = num_y + p.height(num_size, "display") * 0.92
        meter_h = max(3.0, p.px(meter))
        row_y = meter_y + meter_h + p.px(5.0)
        bar_y = row_y + p.height(row_size, "bold") + p.px(1.0)
        bar_h = max(2.0, p.px(bar))
        return _PowerLayout(w, bar_y + bar_h + p.px(bottom), p.px(pad), lab_size, lab_y, num_size, num_y,
                            meter_y, meter_h, segments, row_size, row_y, bar_y, bar_h)

    def _power_block(self, f: _Frame, lay: _PowerLayout, x: float, y: float) -> None:
        p, a, acc = self._p, f.ctx.ability, f.accent
        level = max(0, int(_finite(getattr(a, "power_level", 0))))
        over = level > OVER_9000
        pulse = _quant(0.5 + 0.5 * math.sin(f.now * math.tau * 1.6), 8) if over else 0.0
        hot = _mix(GOLD, HOT, 0.6 * pulse)
        p.glass(x, y, lay.w, lay.h, acc, border=hot if over else acc,
                border_alpha=(0.5 + 0.35 * pulse) if over else 0.34)
        x0, x1 = x + lay.pad, x + lay.w - lay.pad
        p.text("POWER LEVEL", x0, y + lay.lab_y, lay.lab_size, _rgba(hot if over else acc, 0.92), "bold")
        label = self._selected_label(f)
        if label:
            room = (x1 - x0) - p.width("POWER LEVEL", lay.lab_size, "bold") - p.px(12.0)
            if room > p.px(30.0):
                p.text(p.fit(label, lay.lab_size, "bold", room), x1, y + lay.lab_y, lay.lab_size,
                       _rgba(TEXT_3, 0.9), "bold", align="right")
        if over:
            rgb = _mix((1.0, 0.94, 0.66), (1.0, 0.72, 0.40), pulse)
            glow = _rgba(_mix(GOLD, HOT, pulse), 0.72 + 0.2 * pulse)
        else:
            rgb, glow = _mix(TEXT, f.pal.core, 0.5), _rgba(acc, 0.62)
        p.text(f"{level:,}", x + lay.w * 0.5, y + lay.num_y, lay.num_size, _rgba(rgb, 1.0), "display", "center",
               glow=float(round(p.px(10.0))), glow_color=glow, shadow=True)

        # Segmented power meter; the top end turns gold / red once the level is over 9000.
        segments = lay.segments
        gap = float(max(1, round(p.px(3.0))))
        span = x1 - x0
        seg_w = (span - gap * (segments - 1)) / segments
        lit = clamp(_finite(getattr(a, "power_fraction", 0.0))) * segments
        my = y + lay.meter_y
        radius = min(seg_w, lay.meter_h) * 0.25
        for i in range(segments):
            fill = _quant(lit - i, 4)
            if fill <= 0.0:
                color: Color4 = (1.0, 1.0, 1.0, 0.07)
            else:
                pos = (i + 0.5) / segments
                if over and pos > 0.7:
                    seg_rgb = _mix(GOLD, HOT, clamp((pos - 0.7) / 0.3) * (0.5 + 0.5 * pulse))
                else:
                    seg_rgb = _mix(acc, f.pal.core, 0.5 * pos)
                color = _rgba(seg_rgb, 0.3 + 0.7 * fill)
            p.rect(x0 + i * (seg_w + gap), my, seg_w, lay.meter_h, color, radius)

        label, right, rgb, frac, bar_rgb, emphasis = self._phase_info(f)
        ry = y + lay.row_y
        glow_r = float(round(p.px(4.0))) if emphasis > 0.0 else 0.0
        p.text(label, x0, ry, lay.row_size, _rgba(rgb, 0.97), "bold", glow=glow_r,
               glow_color=_rgba(rgb, 0.45) if glow_r else None)
        if right:
            p.text(right, x1, ry, lay.row_size, _rgba(rgb if emphasis else TEXT_3, max(0.35, emphasis) if emphasis else 0.9),
                   "bold", align="right", glow=glow_r, glow_color=_rgba(rgb, 0.5 * emphasis) if glow_r else None)
        by = y + lay.bar_y
        p.rect(x0, by, span, lay.bar_h, (1.0, 1.0, 1.0, 0.08), lay.bar_h * 0.5)
        frac = round(clamp(frac) * 100.0) / 100.0
        if frac > 0.0:
            p.rect(x0, by, max(lay.bar_h, span * frac), lay.bar_h, _rgba(bar_rgb, 0.96), lay.bar_h * 0.5)

    def _selected_label(self, f: _Frame) -> str:
        a = f.ctx.ability
        aid = _clean(getattr(a, "selected", ""))
        if aid in ABILITY_LABELS:
            return ABILITY_LABELS[aid][0]
        return _clean(getattr(a, "selected_label", "")).upper()

    def _phase_info(self, f: _Frame) -> tuple[str, str, Color3, float, Color3, float]:
        """``(label, right label, colour, bar fraction, bar colour, emphasis 0..1)``."""
        a, acc = f.ctx.ability, f.accent
        phase = _phase_name(getattr(a, "phase", "idle"))
        charge = clamp(_finite(getattr(a, "charge", 0.0)))
        if phase == AttackPhase.CHARGING.value:
            text = f"CHARGING {int(round(charge * 100.0))}%"
            if getattr(a, "charge_ready", False):
                blink = _quant(0.5 + 0.5 * math.sin(f.now * math.tau * 2.2), 6)
                return text, "READY - PUSH!", _mix(acc, f.pal.core, 0.25), charge, _mix(acc, f.pal.core, 0.4), \
                    0.6 + 0.4 * blink
            return text, "", TEXT, charge, acc, 0.0
        if phase == AttackPhase.PREPARING.value:
            return PHASE_LABELS[phase], "", TEXT_2, charge, acc, 0.0
        if phase == AttackPhase.FIRING.value:
            return PHASE_LABELS[phase], "", _mix(f.pal.core, f.pal.inner, 0.2), 1.0, f.pal.inner, 1.0
        if phase == AttackPhase.COOLDOWN.value:
            cooldown = clamp(_finite(getattr(a, "cooldown", 0.0)))
            return PHASE_LABELS[phase], f"{int(round(cooldown * 100.0))}%", TEXT_2, cooldown, GREY, 0.0
        if phase == AttackPhase.IDLE.value:
            return PHASE_LABELS[phase], "", TEXT_3, 0.0, acc, 0.0
        return (phase.upper() or "STANDBY"), "", TEXT_3, charge, acc, 0.0

    def _ability_message(self, f: _Frame, bottom_y: float) -> None:
        """Hint / feedback text above the power block, fading in and out on change."""
        incoming = _clean(getattr(f.ctx.ability, "message", None)) or None
        if incoming != self._msg_text:
            if self._msg_text is not None:
                self._msg_old, self._msg_old_since = self._msg_text, f.now
            self._msg_text, self._msg_since = incoming, f.now
        if self._msg_text is not None:
            self._draw_message(f, self._msg_text, clamp((f.now - self._msg_since) / MESSAGE_FADE_IN), bottom_y)
        elif self._msg_old is not None:
            t = (f.now - self._msg_old_since) / MESSAGE_FADE_OUT
            if t >= 1.0 or t < 0.0:
                self._msg_old = None
            else:
                self._draw_message(f, self._msg_old, 1.0 - t, bottom_y)

    def _draw_message(self, f: _Frame, text: str, alpha: float, bottom_y: float) -> None:
        p = self._p
        alpha = _quant(alpha, 12)
        if alpha <= 0.0:
            return
        size = p.fs(17.0)
        pad_x, pad_y = p.px(14.0), p.px(4.0)
        max_w = min(p.px(760.0), (f.x1 - f.x0) - 2.0 * f.m)
        lines = p.wrap(text, size, "ui", max_w - 2.0 * pad_x, max_lines=2)
        lh = p.height(size)
        w = max(p.width(line, size) for line in lines) + 2.0 * pad_x
        h = lh * len(lines) + 2.0 * pad_y
        x, y = f.cx - w * 0.5, bottom_y - h
        p.rect(x, y, w, h, (0.02, 0.03, 0.07, 0.46 * alpha), min(h * 0.5, p.px(14.0)))
        for i, line in enumerate(lines):
            p.text(line, f.cx, y + pad_y + i * lh, size, _rgba(TEXT, 0.95 * alpha), align="center", shadow=True)

    # -------------------------------------------------------------- callout

    def _build_callout(self, f: _Frame, p: _Painter | None = None) -> None:
        """The big anime shout ("KA... ME... HA...") in the upper third."""
        a = f.ctx.ability
        p = p if p is not None else self._p
        text = _clean(getattr(a, "callout", None))
        k = _quant(getattr(a, "callout_intensity", 0.0), 10)
        if not text or k <= 0.0:
            return
        # Size steps of 4 px keep the number of distinct text / glow textures small.
        size = max(MIN_FONT_PX, int(round(120.0 * (0.84 + 0.16 * k) * f.u / 4.0)) * 4)
        max_w = (f.x1 - f.x0) - 2.0 * f.m
        width = p.width(text, size, "display")
        if width > max_w > 0:
            size = max(MIN_FONT_PX, int(size * max_w / width / 2.0) * 2)
        cy = f.h * 0.24
        halo, glow = float(max(4, round(30.0 * f.u))), float(max(3, round(12.0 * f.u)))
        p.text(text, f.cx, cy, size, _rgba(f.pal.inner, 0.9 * k), "display", "center", "middle", glow=halo,
               glow_color=_rgba(f.pal.outer, 0.75 * k), shadow=True)
        p.text(text, f.cx, cy, size, _rgba(_mix(f.pal.core, f.pal.inner, 0.15), k), "display", "center", "middle",
               glow=glow, glow_color=_rgba(f.pal.inner, 0.9 * k))

    # ------------------------------------------------------------- overlays

    def _text_button(self, f: _Frame, wid: str, label: str, x: float, y: float, w: float, h: float,
                     action: UIAction | None, style: str = "normal", enabled: bool = True,
                     size: int | None = None) -> None:
        """A labelled push button (``style``: normal | primary | danger)."""
        p, acc = self._p, f.accent
        hovered, pressed = self._button(wid, (x, y, w, h), action, enabled)
        size = size or p.fs(15.5)
        edge = RED if style == "danger" else acc
        if not enabled:
            fill, border, text_rgb = (1.0, 1.0, 1.0, 0.03), (1.0, 1.0, 1.0, 0.10), TEXT_3
        elif style == "primary":
            fill = _rgba(_mix(acc, WHITE, 0.25 if pressed else 0.12 if hovered else 0.0), 0.92)
            border, text_rgb = _rgba(_mix(acc, WHITE, 0.4), 0.95), DARK
        else:
            fill = _rgba(_mix(GLASS[:3], edge, 0.38 if pressed else 0.2 if hovered else 0.07), 0.82)
            border = _rgba(edge, 0.9 if hovered else 0.5)
            text_rgb = _mix(RED, WHITE, 0.25) if style == "danger" else TEXT
        p.rect(x, y, w, h, fill, p.px(8.0), border, p.hair)
        p.text(p.fit(label, size, "bold", w - p.px(12.0)), x + w * 0.5, y + h * 0.5, size, _rgba(text_rgb, 0.98),
               "bold", "center", "middle")

    def _button_row(self, f: _Frame, specs: Sequence[tuple[str, str, UIAction | None, str, bool]],
                    cx: float, y: float, h: float, max_w: float) -> None:
        """Centred row of text buttons: ``(id, label, action, style, enabled)``."""
        p = self._p
        size, gap = p.fs(15.5), p.px(12.0)
        widths = [max(p.px(120.0), p.width(label, size, "bold") + p.px(36.0)) for _, label, _, _, _ in specs]
        total = sum(widths) + gap * (len(specs) - 1)
        if total > max_w:                      # shrink evenly to fit
            k = max(0.2, (max_w - gap * (len(specs) - 1)) / max(1.0, sum(widths)))
            widths = [w * k for w in widths]
            total = sum(widths) + gap * (len(specs) - 1)
        x = cx - total * 0.5
        for (wid, label, action, style, enabled), w in zip(specs, widths):
            self._text_button(f, wid, label, x, y, w, h, action, style, enabled, size)
            x += w + gap

    def _build_camera_overlay(self, f: _Frame) -> None:
        state = f.camera_state
        if state == "running":
            return
        if state == "starting":
            self._spinner_pill(f, "STARTING CAMERA…", _camera_name(f.ctx))
        elif state == "error":
            error = _clean(getattr(f.ctx.camera_status, "error", None)) or "The camera stopped delivering images."
            self._camera_panel(f, "CAMERA ERROR", error, RED,
                               [("overlay:retry", "RETRY", UIAction("toggle_camera"), "primary", True),
                                ("overlay:settings", "SETTINGS", UIAction("toggle_settings"), "normal", True)])
        elif f.ctx.camera_scanning:
            self._spinner_pill(f, "LOOKING FOR CAMERAS…", "")
        else:
            self._camera_panel(f, "CAMERA IS OFF",
                               "Start the camera to control the energy with your hands. Keyboard: hold SPACE "
                               "to charge, release it to fire.", f.accent,
                               [("overlay:start", "START CAMERA", UIAction("toggle_camera"), "primary", True)])

    def _camera_panel(self, f: _Frame, title: str, text: str, rgb: Color3,
                      buttons: list[tuple[str, str, UIAction | None, str, bool]]) -> None:
        p = self._p
        w = min(p.px(640.0), (f.x1 - f.x0) - 2.0 * f.m)
        pad = p.px(24.0)
        title_size, body_size = p.fs(25.0), p.fs(18.0)
        inner = w - 2.0 * pad
        lines = p.wrap(text, body_size, "ui", inner, max_lines=8)
        name = _camera_name(f.ctx)
        th, lh, nh = p.height(title_size, "bold"), p.height(body_size), p.height(p.fs(15.0))
        bh = p.px(40.0)
        h = pad + th + (nh if name else 0.0) + p.px(10.0) + lh * len(lines) + p.px(18.0) + bh + pad
        x, y = f.cx - w * 0.5, max(f.m, f.h * 0.5 - h * 0.5)
        self._blocker("overlay:camera", (x, y, w, h))
        p.glass(x, y, w, h, f.accent, border=rgb, border_alpha=0.6, fill=(0.03, 0.035, 0.08, 0.84),
                fill_bottom=(0.02, 0.025, 0.06, 0.88))
        r = th * 0.36
        icx, icy = x + pad + r, y + pad + th * 0.5
        p.circle(icx, icy, r, _rgba(rgb, 1.0), max(1.5, p.px(2.2)))
        p.text("!", icx, icy, p.fs(17.0), _rgba(rgb, 1.0), "bold", "center", "middle")
        p.text(p.fit(title, title_size, "bold", inner - 2.0 * r - p.px(12.0)), icx + r + p.px(12.0), icy, title_size,
               _rgba(_mix(rgb, WHITE, 0.2), 1.0), "bold", valign="middle")
        ty = y + pad + th
        if name:
            p.text(p.fit(name, p.fs(15.0), "ui", inner), x + pad, ty, p.fs(15.0), _rgba(TEXT_3, 0.95))
            ty += nh
        ty += p.px(10.0)
        for i, line in enumerate(lines):
            p.text(line, x + pad, ty + i * lh, body_size, _rgba(TEXT_2, 0.97))
        self._button_row(f, buttons, f.cx, y + h - pad - bh, bh, inner)

    def _spinner_pill(self, f: _Frame, text: str, detail: str) -> None:
        p = self._p
        size, small = p.fs(18.0), p.fs(14.5)
        r = p.px(12.0)
        detail = p.fit(detail, small, "ui", p.px(360.0)) if detail else ""
        tw = max(p.width(text, size, "bold"), p.width(detail, small) if detail else 0)
        th = p.height(size, "bold") + (p.height(small) if detail else 0)
        pad = p.px(16.0)
        w = min((f.x1 - f.x0) - 2.0 * f.m, pad * 2.0 + r * 2.0 + p.px(14.0) + tw)
        h = max(2.0 * r, th) + 2.0 * pad * 0.75
        x, y = f.cx - w * 0.5, f.h * 0.5 - h * 0.5
        p.glass(x, y, w, h, f.accent, radius=min(h * 0.5, p.px(16.0)), tab=False)
        sx, sy = x + pad + r, y + h * 0.5
        p.circle(sx, sy, r, (1.0, 1.0, 1.0, 0.12), max(1.5, p.px(3.0)))
        start = round((f.now * 1.3 % 1.0) * 48.0) / 48.0 * math.tau     # 48 discrete angles
        p.arc(sx, sy, r, start, start + math.tau * 0.28, _rgba(f.accent, 1.0), max(1.5, p.px(3.0)))
        tx = sx + r + p.px(14.0)
        ty = y + h * 0.5 - th * 0.5
        p.text(text, tx, ty, size, _rgba(TEXT, 0.97), "bold")
        if detail:
            p.text(detail, tx, ty + p.height(size, "bold"), small, _rgba(TEXT_3, 0.95))

    def _build_tracker_banner(self, f: _Frame, top: tuple[float, float, float]) -> float:
        """Red banner for tracker errors at the top; returns its bottom (0 when absent)."""
        error = _clean(getattr(f.ctx.tracker_status, "error", None))
        if not error:
            return 0.0
        p = self._p
        size, title_size = p.fs(16.0), p.fs(14.0)
        left_x1, right_x0, top_bottom = top
        gap = p.px(16.0)
        half = min(f.cx - (left_x1 + gap), (right_x0 - gap) - f.cx)
        max_w = min(p.px(820.0), (f.x1 - f.x0) - 2.0 * f.m)
        if 2.0 * half >= p.px(380.0):
            max_w, y = min(max_w, 2.0 * half), f.m
        else:
            y = top_bottom + p.px(10.0)
        pad_x, pad_y, r = p.px(14.0), p.px(8.0), p.px(10.0)
        text_x0 = pad_x + 2.0 * r + p.px(12.0)
        lines = p.wrap(error, size, "ui", max_w - text_x0 - pad_x, max_lines=2)
        lh, th = p.height(size), p.height(title_size, "bold")
        w = min(max_w, text_x0 + pad_x + max(p.width("TRACKING ERROR", title_size, "bold"),
                                               max(p.width(line, size) for line in lines)))
        h = 2.0 * pad_y + th + lh * len(lines)
        x = f.cx - w * 0.5
        p.glass(x, y, w, h, f.accent, border=RED, border_alpha=0.75, tab=False, fill=(0.13, 0.02, 0.03, 0.74),
                fill_bottom=(0.08, 0.01, 0.02, 0.80))
        icx, icy = x + pad_x + r, y + h * 0.5
        p.circle(icx, icy, r, _rgba(RED, 1.0))
        p.text("!", icx, icy, p.fs(15.0), _rgba(DARK, 1.0), "bold", "center", "middle")
        p.text("TRACKING ERROR", x + text_x0, y + pad_y, title_size, _rgba(_mix(RED, WHITE, 0.25), 1.0), "bold")
        for i, line in enumerate(lines):
            p.text(line, x + text_x0, y + pad_y + th + i * lh, size, _rgba(TEXT, 0.96))
        return y + h

    def _build_calibration(self, f: _Frame) -> None:
        cal = f.ctx.calibration
        active = bool(getattr(cal, "active", False))
        result = _clean(getattr(cal, "result_message", None))
        if not active and not result:
            return
        p, acc = self._p, f.accent
        w = min(p.px(600.0), (f.x1 - f.x0) - 2.0 * f.m)
        pad = p.px(22.0)
        inner = w - 2.0 * pad
        head_size, title_size, body_size, small = p.fs(15.0), p.fs(23.0), p.fs(18.0), p.fs(13.5)
        step = max(0, int(_finite(getattr(cal, "step", 0))))
        total = max(0, int(_finite(getattr(cal, "total_steps", 0))))
        if active:
            head = f"CALIBRATION {step}/{total}" if total > 0 else "CALIBRATION"
            title_lines = p.wrap(_clean(getattr(cal, "title", "")), title_size, "bold", inner, 3)
            body = _clean(getattr(cal, "instruction", ""))
            body_lines = p.wrap(body, body_size, "ui", inner, 4) if body else ()
        else:
            head, title_lines = "CALIBRATION", ()
            body_lines = p.wrap(result, body_size, "ui", inner, 4)
        title_lines = tuple(line for line in title_lines if line)
        hh, th, lh = p.height(head_size, "bold"), p.height(title_size, "bold"), p.height(body_size)
        bar_h, bh = max(3.0, p.px(6.0)), p.px(34.0)
        h = pad + hh + p.px(8.0) + th * len(title_lines) + p.px(4.0) + lh * len(body_lines)
        if active:
            h += p.px(16.0) + bar_h + p.px(14.0) + bh
        h += pad
        x = f.cx - w * 0.5
        y = clamp(f.h * 0.32 - h * 0.5, f.m, max(f.m, f.h - f.m - h))
        self._blocker("overlay:calibration", (x, y, w, h))
        p.glass(x, y, w, h, acc, border_alpha=0.55, fill=(0.03, 0.04, 0.09, 0.80),
                fill_bottom=(0.02, 0.03, 0.065, 0.86))
        cy = y + pad
        p.text(head, x + pad, cy, head_size, _rgba(acc, 0.97), "bold")
        if active and total > 0:               # step dots: done / current / to do
            r = p.px(5.0)
            for i in range(total):
                dx = x + w - pad - r - (total - 1 - i) * r * 3.2
                if i + 1 < step:
                    p.circle(dx, cy + hh * 0.5, r, _rgba(acc, 1.0))
                elif i + 1 == step:
                    p.circle(dx, cy + hh * 0.5, r, _rgba(acc, 1.0), max(1.0, p.px(2.0)))
                else:
                    p.circle(dx, cy + hh * 0.5, r, (1.0, 1.0, 1.0, 0.22))
        cy += hh + p.px(8.0)
        for line in title_lines:
            p.text(line, x + pad, cy, title_size, _rgba(TEXT, 0.98), "bold")
            cy += th
        cy += p.px(4.0)
        for line in body_lines:
            p.text(line, x + pad, cy, body_size, _rgba(TEXT_2, 0.97))
            cy += lh
        if not active:
            return
        cy += p.px(16.0)
        progress = round(clamp(_finite(getattr(cal, "progress", 0.0))) * 100.0) / 100.0
        p.rect(x + pad, cy, inner, bar_h, (1.0, 1.0, 1.0, 0.09), bar_h * 0.5)
        if progress > 0.0:
            p.rect(x + pad, cy, max(bar_h, inner * progress), bar_h, _rgba(_mix(acc, f.pal.core, 0.25), 0.97),
                   bar_h * 0.5)
        cy += bar_h + p.px(14.0)
        p.text("ESC to cancel", x + pad, cy + bh * 0.5, small, _rgba(TEXT_3, 0.9), valign="middle")
        bw = min(p.px(150.0), inner * 0.5)
        # "recalibrate" while calibrating cancels it (same as the C key / ESC).
        self._text_button(f, "overlay:calibration_cancel", "CANCEL", x + w - pad - bw, cy, bw, bh,
                          UIAction("recalibrate"), size=p.fs(14.5))

    def _build_toasts(self, f: _Frame, top: tuple[float, float, float], banner_bottom: float) -> None:
        if not self._toasts:
            return
        now = f.now
        alive: list[_Toast] = []
        for toast in self._toasts:
            if toast.start is None or now < toast.start:
                toast.start = now                  # first frame after notify()
            if now - toast.start < toast.duration:
                alive.append(toast)
        self._toasts = alive
        if not alive:
            return
        p = self._p
        size = p.fs(17.0)
        left_x1, right_x0, top_bottom = top
        gap = p.px(16.0)
        half = min(f.cx - (left_x1 + gap), (right_x0 - gap) - f.cx)
        max_w = min(p.px(640.0), (f.x1 - f.x0) - 2.0 * f.m)
        if banner_bottom > 0.0:
            y = banner_bottom + p.px(8.0)
        elif 2.0 * half >= p.px(300.0):
            max_w, y = min(max_w, 2.0 * half), f.m
        else:
            y = top_bottom + p.px(10.0)
        pad_x, pad_y, bar_w = p.px(16.0), p.px(9.0), p.px(3.0)
        lh = p.height(size)
        for toast in reversed(alive[-TOAST_MAX_VISIBLE:]):          # newest on top
            age = now - (toast.start or now)
            appear = clamp(age / TOAST_FADE_IN)
            alpha = _quant(appear * clamp((toast.duration - age) / TOAST_FADE_OUT), 12)
            rgb = LEVEL_COLORS.get(toast.level, f.accent)
            lines = p.wrap(toast.text, size, "ui", max_w - 2.0 * pad_x - bar_w, max_lines=3)
            w = max(p.width(line, size) for line in lines) + 2.0 * pad_x + bar_w
            h = lh * len(lines) + 2.0 * pad_y
            ty = y - (1.0 - appear) * p.px(8.0)                     # slides down while appearing
            if alpha > 0.0:
                x = f.cx - w * 0.5
                p.glass(x, ty, w, h, f.accent, radius=p.px(9.0), alpha=alpha, border=rgb, border_alpha=0.6,
                        tab=False)
                p.rect(x + p.px(7.0), ty + p.px(7.0), bar_w, h - p.px(14.0), _rgba(rgb, alpha), bar_w * 0.5)
                for i, line in enumerate(lines):
                    p.text(line, x + pad_x + bar_w, ty + pad_y + i * lh, size, _rgba(TEXT, 0.97 * alpha))
            y += h + p.px(8.0)

    # ---------------------------------------------------------------- debug

    def _build_debug_skeleton(self, f: _Frame) -> None:
        """Landmarks of the tracking frame mapped scene -> window through ``ctx.view``."""
        ctx, p = f.ctx, self._p
        try:
            scale = float(ctx.view.scale)
            ox, oy = float(ctx.view.offset_x), float(ctx.view.offset_y)
        except (AttributeError, TypeError, ValueError):
            return
        if not (math.isfinite(scale) and scale > 0.0 and math.isfinite(ox) and math.isfinite(oy)):
            return
        lw, jr = max(1.0, p.px(2.0)), max(1.5, p.px(3.0))
        frame = ctx.frame
        if frame is not None:
            pose = getattr(frame, "pose", None)
            if pose is not None:
                pts = _window_points(getattr(pose, "landmarks", None), scale, ox, oy)
                vis = _float_list(getattr(pose, "visibility", None), len(pts))
                col = _rgba(POSE_COLOR, 1.0)
                used: set[int] = set()
                for a, b in POSE_CONNECTIONS:
                    if b < len(pts) and vis[a] >= 0.5 and vis[b] >= 0.5:
                        pa, pb = pts[a], pts[b]
                        if pa is not None and pb is not None:
                            p.line([pa, pb], col, lw)
                            used.update((a, b))
                for i in sorted(used):
                    pt = pts[i]
                    if pt is not None:
                        p.circle(pt[0], pt[1], jr, col)
            for hand in getattr(frame, "hands", None) or ():
                pts = _window_points(getattr(hand, "landmarks", None), scale, ox, oy)
                source = str(getattr(hand, "source", "hand"))
                col = _rgba(SOURCE_COLORS.get(source, SOURCE_COLORS["hand"]), 1.0)
                for a, b in HAND_CONNECTIONS:
                    if b < len(pts):
                        pa, pb = pts[a], pts[b]
                        if pa is not None and pb is not None:
                            p.line([pa, pb], col, lw)
                for pt in pts:
                    if pt is not None:
                        p.circle(pt[0], pt[1], jr * 0.8, col)
                if pts and pts[0] is not None:
                    tag = f"{str(getattr(hand, 'side', '?')).upper()} · {source}"
                    p.text(tag, pts[0][0], pts[0][1] + p.px(8.0), p.fs(13.0), col, "bold", "center", shadow=True)
        body = ctx.body
        if body is None:
            return

        def to_win(pt: Any) -> Vec2 | None:
            try:
                x, y = float(pt[0]) * scale + ox, float(pt[1]) * scale + oy
            except (TypeError, ValueError, IndexError):
                return None
            return (x, y) if math.isfinite(x) and math.isfinite(y) else None

        box = getattr(body, "torso_box", None)
        if box is not None:
            a, b = to_win((box[0], box[1])), to_win((box[2], box[3]))
            if a is not None and b is not None:
                p.rect(min(a[0], b[0]), min(a[1], b[1]), abs(b[0] - a[0]), abs(b[1] - a[1]), (0.0, 0.0, 0.0, 0.0),
                       0.0, _rgba(POSE_COLOR, 0.8), max(1.0, p.px(1.5)))
        for side in ("left", "right"):
            arm = getattr(body, side, None)
            if arm is None or not getattr(arm, "present", False):
                continue
            c = to_win(getattr(arm, "hand_center", None))
            if c is not None:
                rgb = SOURCE_COLORS.get(str(getattr(arm, "source", "none")), GREY)
                p.circle(c[0], c[1], p.px(8.0), _rgba(rgb, 1.0), max(1.5, p.px(2.5)))
        mid = to_win(getattr(body, "hand_mid", None))
        if mid is not None:
            k = p.px(9.0)
            white = _rgba(WHITE, 1.0)
            p.line([(mid[0] - k, mid[1]), (mid[0] + k, mid[1])], white, lw)
            p.line([(mid[0], mid[1] - k), (mid[0], mid[1] + k)], white, lw)
            sig = ctx.signals
            if sig is not None:
                d = getattr(sig, "aim_direction", (1.0, 0.0))
                length = max(p.px(40.0), _finite(getattr(body, "body_scale", 0.0)) * scale * 1.2)
                dx, dy = _finite(d[0], 1.0), _finite(d[1], 0.0)
                norm = math.hypot(dx, dy)
                if norm > 1e-6:
                    dx, dy = dx / norm, dy / norm
                    tip = (mid[0] + dx * length, mid[1] + dy * length)
                    col = _rgba(f.accent, 1.0)
                    p.line([mid, tip], col, lw)
                    hx, hy = -dy * p.px(6.0), dx * p.px(6.0)
                    back = (tip[0] - dx * p.px(10.0), tip[1] - dy * p.px(10.0))
                    p.line([(back[0] + hx, back[1] + hy), tip, (back[0] - hx, back[1] - hy)], col, lw)

    def _debug_rows(self, f: _Frame) -> list[tuple[str, str, float | None]]:
        ctx = f.ctx
        body, sig, a = ctx.body, ctx.signals, ctx.ability
        rows: list[tuple[str, str, float | None]] = []
        if body is None:
            rows.append(("quality", "no body", None))
        else:
            dist = _finite(getattr(body, "hand_distance", math.inf), math.inf)
            rows += [("quality", f"{getattr(body, 'quality', '?')} ({_hands_present(body)} hands)", None),
                     ("S (px)", f"{_finite(getattr(body, 'body_scale', 0.0)):.0f}", None),
                     ("hand dist", f"{dist:.2f} S" if math.isfinite(dist) else "—", None)]
        if sig is not None:
            aim = getattr(sig, "aim_direction", (0.0, 0.0))
            rows += [("charge pose", f"{_finite(sig.charge_pose):.2f}{' ON' if sig.charge_pose_active else ''}",
                      _finite(sig.charge_pose)),
                     ("thrust", f"{_finite(sig.thrust):.2f}", _finite(sig.thrust)),
                     ("frontal", f"{_finite(sig.thrust_frontal):.2f}", _finite(sig.thrust_frontal)),
                     ("burst", f"{_finite(sig.burst):.2f}", _finite(sig.burst)),
                     ("aim", f"{_finite(aim[0]):+.2f} {_finite(aim[1]):+.2f} f{_finite(sig.aim_frontal):.2f}",
                      _finite(sig.aim_frontal)),
                     ("arms ext", "yes" if sig.arms_extended else "no", None),
                     ("tracking", "ok" if sig.tracking_ok else "lost", None)]
        if body is not None:
            motion = getattr(body, "motion", None)
            rows += [("stability", f"{_finite(body.stability):.2f}", _finite(body.stability)),
                     ("energy", f"{_finite(body.energy):.2f}", _finite(body.energy)),
                     ("motion", str(getattr(motion, "dominant", "?")), None)]
        rows += [("phase", _phase_name(getattr(a, "phase", "idle")), None),
                 ("charge", f"{_finite(getattr(a, 'charge', 0.0)):.2f}", _finite(getattr(a, "charge", 0.0)))]
        ts = ctx.tracker_status
        if ts is not None:
            rows.append(("infer ms", f"hands {_finite(ts.hands_ms):.0f} · pose {_finite(ts.pose_ms):.0f}", None))
        return rows

    def _build_debug_panel(self, f: _Frame, top_bottom: float) -> None:
        p = self._p
        if not 0.0 <= f.now - self._debug_at < 0.1:      # 10 Hz: readable and cheap
            self._debug_at = f.now
            self._debug_cache = self._debug_rows(f)
        rows = self._debug_cache
        size, mono = p.fs(13.5), p.fs(13.0)
        rh = max(p.height(size), p.height(mono, "mono")) * 0.95
        pad = p.px(10.0)
        w = min(p.px(290.0), (f.x1 - f.x0) - 2.0 * f.m)
        x = f.x0 + f.m
        y = (top_bottom + p.px(12.0)) if f.s.ui.show_hud else f.m
        title_h = p.height(p.fs(13.0), "bold")
        room = f.h - y - p.px(260.0)                      # keep clear of the bottom HUD
        n = max(1, min(len(rows), int((room - 2.0 * pad - title_h) // max(1.0, rh))))
        h = 2.0 * pad + title_h + rh * n
        p.glass(x, y, w, h, f.accent, radius=p.px(8.0), tab=False, fill=(0.02, 0.03, 0.06, 0.70))
        p.text("DEBUG", x + pad, y + pad, p.fs(13.0), _rgba(f.accent, 0.95), "bold")
        value_x = x + pad + p.px(86.0)
        bar_w = p.px(46.0)
        for i, (label, value, bar) in enumerate(rows[:n]):
            ry = y + pad + title_h + i * rh
            p.text(label, x + pad, ry, size, _rgba(TEXT_3, 0.95))
            limit = (x + w - pad - value_x) - ((bar_w + p.px(6.0)) if bar is not None else 0.0)
            p.text(p.fit(value, mono, "mono", max(p.px(20.0), limit)), value_x, ry, mono, _rgba(TEXT, 0.97), "mono")
            if bar is not None:
                bx, by = x + w - pad - bar_w, ry + rh * 0.5 - p.px(2.0)
                p.rect(bx, by, bar_w, max(2.0, p.px(4.0)), (1.0, 1.0, 1.0, 0.1), p.px(2.0))
                q = _quant(bar, 20)
                if q > 0.0:
                    p.rect(bx, by, max(2.0, bar_w * q), max(2.0, p.px(4.0)), _rgba(f.accent, 0.95), p.px(2.0))

    # ------------------------------------------------------------- settings

    def _build_settings(self, f: _Frame, panel_w: float) -> None:
        p, acc = self._p, f.accent
        x0, height = float(f.w) - panel_w, float(f.h)
        self._panel_rect = (x0, 0.0, panel_w, height)
        head_h, foot_h = p.px(60.0), p.px(66.0)
        vy0, vy1 = head_h, max(head_h + p.px(40.0), height - foot_h)
        self._blocker("settings:panel", self._panel_rect)        # nothing below receives clicks
        p.rect(x0, 0.0, panel_w, height, PANEL_FILL, 0.0, None, 0.0, PANEL_FILL_BOTTOM)
        p.rect(x0, 0.0, p.hair, height, _rgba(acc, 0.6))
        pad = p.px(20.0)

        # Scrolled content first; the opaque header and footer are drawn over its overflow
        # (the draw list has no clipping), and widgets are clipped to the viewport.
        col = _Column(f, x0 + pad, x0 + panel_w - pad - p.px(4.0), vy0, vy1, vy0 + p.px(4.0) - self._scroll)
        self._settings_content(col)
        content_h = col.y + self._scroll - vy0 + p.px(18.0)
        view_h = vy1 - vy0
        self._scroll_max = max(0.0, content_h - view_h)
        self._scroll = clamp(self._scroll, 0.0, self._scroll_max)
        if self._scroll_max > 0.5:
            track_x, track_w = x0 + panel_w - p.px(8.0), max(2.0, p.px(3.0))
            inset = p.px(6.0)
            p.rect(track_x, vy0 + inset, track_w, view_h - 2.0 * inset, (1.0, 1.0, 1.0, 0.06), track_w * 0.5)
            thumb_h = max(p.px(32.0), (view_h - 2.0 * inset) * view_h / max(content_h, 1.0))
            thumb_y = vy0 + inset + (view_h - 2.0 * inset - thumb_h) * (self._scroll / self._scroll_max)
            p.rect(track_x, thumb_y, track_w, thumb_h, _rgba(acc, 0.55), track_w * 0.5)

        # header
        p.rect(x0 + p.hair, 0.0, panel_w - p.hair, head_h, PANEL_BAR)
        p.rect(x0 + p.hair, head_h - p.hair, panel_w - p.hair, p.hair, _rgba(acc, 0.32))
        title = p.fs(24.0)
        p.rect(x0 + pad, head_h * 0.5 - p.px(11.0), max(2.0, p.px(3.5)), p.px(22.0), _rgba(acc, 1.0), p.px(1.5))
        p.text("SETTINGS", x0 + pad + p.px(12.0), head_h * 0.5, title, _rgba(TEXT, 0.98), "bold", valign="middle",
               glow=float(round(p.px(5.0))), glow_color=_rgba(acc, 0.45))
        size = p.px(34.0)
        bx, by = x0 + panel_w - pad - size + p.px(6.0), (head_h - size) * 0.5
        hovered, pressed = self._button("settings:close_x", (bx, by, size, size), UIAction("toggle_settings"))
        if hovered:
            p.rect(bx, by, size, size, (1.0, 1.0, 1.0, 0.14 if pressed else 0.08), p.px(8.0))
        k = size * 0.19
        cx, cy = bx + size * 0.5, by + size * 0.5
        xcol = _rgba(TEXT if hovered else TEXT_2, 1.0)
        p.line([(cx - k, cy - k), (cx + k, cy + k)], xcol, max(1.5, p.px(2.2)))
        p.line([(cx - k, cy + k), (cx + k, cy - k)], xcol, max(1.5, p.px(2.2)))

        # footer: reset (two clicks) + close
        fy = height - foot_h
        p.rect(x0 + p.hair, fy, panel_w - p.hair, foot_h, PANEL_BAR)
        p.rect(x0 + p.hair, fy, panel_w - p.hair, p.hair, _rgba(acc, 0.32))
        bh, gap = p.px(38.0), p.px(10.0)
        inner = panel_w - 2.0 * pad
        reset_w = (inner - gap) * 0.62
        armed = self._reset_armed_at is not None
        self._text_button(f, RESET_ID, "CLICK AGAIN TO CONFIRM" if armed else "RESET TO DEFAULTS", x0 + pad,
                          fy + (foot_h - bh) * 0.5, reset_w, bh, UIAction("reset_settings"),
                          "danger" if armed else "normal", size=p.fs(14.5))
        self._text_button(f, "settings:close", "CLOSE", x0 + pad + reset_w + gap, fy + (foot_h - bh) * 0.5,
                          inner - reset_w - gap, bh, UIAction("toggle_settings"), "primary", size=p.fs(14.5))

    def _settings_content(self, col: _Column) -> None:
        f, s, ctx = col.f, col.f.s, col.f.ctx
        # CAMERA
        self._section(col, "CAMERA")
        status = ctx.camera_status
        index = int(_finite(getattr(status, "index", -1), -1.0))
        current = index if index >= 0 else s.camera.index
        cameras = [c for c in (ctx.cameras or ()) if isinstance(c, CameraInfo)]
        if not cameras:
            self._info_row(col, "Scanning for cameras…" if ctx.camera_scanning
                           else "No cameras found. Connect a webcam and press RESCAN.")
        for cam in cameras:
            self._radio_row(col, f"settings:camera:{cam.index}", _clean(cam.label) or f"Camera {cam.index}",
                            cam.index == current, UIAction("select_camera", int(cam.index)),
                            "IR" if cam.is_ir else "VIRTUAL" if getattr(cam, "is_virtual", False) else "")
        if f.camera_state == "running":
            res = getattr(status, "resolution", (0, 0)) or (0, 0)
            fps = _finite(getattr(status, "fps", 0.0))
            if res[0] and res[1]:
                self._info_row(col, f"Active: {int(res[0])}×{int(res[1])}" + (f" @ {fps:.0f} fps" if fps > 0 else ""))
        scanning = bool(ctx.camera_scanning)
        self._button_row_panel(col, [
            ("settings:rescan", "SCANNING..." if scanning else "RESCAN", UIAction("rescan_cameras"), "normal",
             not scanning),
            ("settings:camera_toggle", "STOP CAMERA" if f.camera_live else "START CAMERA", UIAction("toggle_camera"),
             "normal", True)])
        wc, hc = _constraints("camera.width"), _constraints("camera.height")
        resolutions = [(w, h) for w, h in RESOLUTIONS
                       if wc.get("min", 0) <= w <= wc.get("max", 1 << 30) and hc.get("min", 0) <= h <= hc.get("max", 1 << 30)]
        self._segmented_row(col, "settings:resolution", "Resolution",
                            [(f"{w}×{h}", f"{w}×{h}", (w, h), UIAction("set_resolution", (w, h))) for w, h in resolutions],
                            (s.camera.width, s.camera.height))
        fc = _constraints("camera.fps")
        self._segmented_row(col, "settings:camera.fps", "Camera FPS",
                            [(str(v), str(v), v, UIAction("set_setting", ("camera.fps", v))) for v in CAMERA_FPS_CHOICES
                             if fc.get("min", 0) <= v <= fc.get("max", 1 << 30)], s.camera.fps)
        self._setting_toggle(col, "camera.mirror", "Mirror image")
        # VIDEO CALLS (virtual camera output)
        self._section(col, "VIDEO CALLS")
        vcam = s.virtual_camera
        self._toggle_row(col, "settings:virtual_camera", "Virtual camera", bool(vcam.enabled),
                         UIAction("toggle_virtual_camera"))
        self._info_row(col, _virtual_camera_text(ctx.virtual_camera))
        self._segmented_row(col, "settings:virtual_camera.resolution", "Call resolution",
                            [(f"{w}×{h}", f"{w}×{h}", (w, h), UIAction("set_virtual_camera_resolution", (w, h)))
                             for w, h in VIRTUAL_CAMERA_RESOLUTIONS], (vcam.width, vcam.height))
        self._setting_toggle(col, "virtual_camera.mirror", "Mirrored picture in calls")
        # GESTURES
        self._section(col, "GESTURES")
        for dotted in ("gestures.sensitivity", "gestures.min_charge_time", "gestures.cooldown_time"):
            self._slider_row(col, dotted)
        self._setting_toggle(col, "gestures.burst_enabled", "Burst gesture")
        self._button_row_panel(col, [("settings:recalibrate", "RECALIBRATE", UIAction("recalibrate"), "normal", True)])
        # EFFECTS
        self._section(col, "EFFECTS")
        self._choice_row(col, "effects.quality", "Quality")
        self._swatch_row(col)
        for dotted in ("effects.bloom_strength", "effects.screen_shake", "effects.camera_darken"):
            self._slider_row(col, dotted)
        self._setting_toggle(col, "effects.show_callouts", "Callouts")
        self._setting_toggle(col, "effects.motion_trails", "Motion trails")
        # ABILITIES
        self._section(col, "ABILITIES")
        aura_on = bool(getattr(ctx.ability, "aura_enabled", s.abilities.aura_enabled))
        self._toggle_row(col, "settings:aura", "Power aura", aura_on, UIAction("toggle_aura"))
        self._setting_toggle(col, "abilities.aura_auto", "Auto aura while charging")
        # TRACKING
        self._section(col, "TRACKING")
        self._choice_row(col, "tracking.pose_model", "Pose model")
        self._setting_toggle(col, "tracking.enable_segmentation", "Body segmentation")
        # AUDIO
        self._section(col, "AUDIO")
        self._setting_toggle(col, "audio.enabled", "Sound")
        self._slider_row(col, "audio.volume")
        # DISPLAY
        self._section(col, "DISPLAY")
        self._button_row_panel(col, [("settings:fullscreen", "EXIT FULLSCREEN" if f.fullscreen else "FULLSCREEN",
                                      UIAction("toggle_fullscreen"), "normal", True)])
        self._choice_row(col, "ui.fit_mode", "Fit")
        self._setting_toggle(col, "ui.show_hud", "Show HUD")
        self._setting_toggle(col, "ui.show_fps", "Show FPS")
        self._toggle_row(col, "settings:debug", "Debug overlay", bool(s.ui.show_debug), UIAction("toggle_debug"))

    # ------------------------------------------------------- settings rows

    def _section(self, col: _Column, title: str) -> None:
        p = self._p
        first = col.y <= col.vy0 + p.px(4.0) - self._scroll + 0.5
        h = p.px(30.0 if first else 46.0)
        if col.visible(h):
            size = p.fs(14.0)
            th = p.height(size, "bold")
            ty = col.y + h - p.px(6.0) - th
            p.text(title, col.x0, ty, size, _rgba(col.f.accent, 0.97), "bold")
            lx = col.x0 + p.width(title, size, "bold") + p.px(10.0)
            if col.x1 - lx > p.px(10.0):
                p.rect(lx, ty + th * 0.55, col.x1 - lx, p.hair, _rgba(col.f.accent, 0.22))
        col.y += h

    def _info_row(self, col: _Column, text: str) -> None:
        p = self._p
        size = p.fs(14.5)
        lines = p.wrap(text, size, "ui", col.w, max_lines=3)
        lh = p.height(size)
        h = lh * len(lines) + p.px(6.0)
        if col.visible(h):
            for i, line in enumerate(lines):
                p.text(line, col.x0, col.y + p.px(2.0) + i * lh, size, _rgba(TEXT_3, 0.95))
        col.y += h

    def _row_hover(self, col: _Column, wid: str, h: float, action: UIAction | None,
                   enabled: bool = True) -> tuple[bool, bool]:
        """Register a full-width row as one button and draw its hover highlight."""
        p = self._p
        rect = (col.x0 - p.px(8.0), col.y, col.w + p.px(16.0), h)
        hovered, pressed = self._button(wid, rect, action, enabled, clip=(col.vy0, col.vy1))
        if hovered:
            p.rect(rect[0], col.y + p.px(2.0), rect[2], h - p.px(4.0),
                   (1.0, 1.0, 1.0, 0.075 if pressed else 0.045), p.px(8.0))
        return hovered, pressed

    def _toggle_row(self, col: _Column, wid: str, label: str, value: bool, action: UIAction) -> None:
        p, acc = self._p, col.f.accent
        h = p.px(40.0)
        if col.visible(h):
            self._row_hover(col, wid, h, action)
            size = p.fs(17.5)
            tw, th = p.px(42.0), p.px(22.0)
            p.text(p.fit(label, size, "ui", col.w - tw - p.px(12.0)), col.x0, col.y + h * 0.5, size,
                   _rgba(TEXT, 0.96), valign="middle")
            sx, sy = col.x1 - tw, col.y + (h - th) * 0.5
            knob = th * 0.5 - p.px(3.0)
            if value:
                p.rect(sx, sy, tw, th, _rgba(acc, 0.92), th * 0.5)
                p.circle(sx + tw - th * 0.5, sy + th * 0.5, knob, _rgba(WHITE, 1.0))
            else:
                p.rect(sx, sy, tw, th, (1.0, 1.0, 1.0, 0.10), th * 0.5, (1.0, 1.0, 1.0, 0.2), p.hair)
                p.circle(sx + th * 0.5, sy + th * 0.5, knob, _rgba(TEXT_3, 1.0))
        col.y += h

    def _setting_toggle(self, col: _Column, dotted: str, label: str) -> None:
        value = bool(config.get_setting(col.f.s, dotted))
        self._toggle_row(col, f"settings:{dotted}", label, value, UIAction("set_setting", (dotted, not value)))

    def _slider_row(self, col: _Column, dotted: str) -> None:
        p, acc = self._p, col.f.accent
        spec = SLIDERS[dotted]
        lo, hi, is_int = slider_range(spec)
        value = _finite(config.get_setting(col.f.s, dotted), lo)
        h = p.px(58.0)
        if col.visible(h):
            size = p.fs(17.5)
            value_text = format_value(value, spec.unit)
            vsize = p.fs(16.5)
            p.text(p.fit(spec.label, size, "ui", col.w - p.width(value_text, vsize, "bold") - p.px(10.0)), col.x0,
                   col.y + p.px(5.0), size, _rgba(TEXT, 0.96))
            p.text(value_text, col.x1, col.y + p.px(5.0), vsize, _rgba(_mix(acc, WHITE, 0.35), 0.98), "bold",
                   align="right")
            ty = col.y + p.px(40.0)                       # track centre line
            kr = p.px(8.5)
            tx0, tx1 = col.x0 + kr, col.x1 - kr
            wid = f"slider:{dotted}"
            binding = SliderBinding(dotted, lo, hi, spec.step, is_int, tx0, tx1, value)
            rect = _clip_rect((col.x0 - p.px(4.0), ty - p.px(15.0), col.w + p.px(8.0), p.px(30.0)), col.vy0, col.vy1)
            if rect is not None:
                self._reg.append(UIWidget(wid, rect, "slider", None, binding))
            active = self._drag_id == wid
            hot = active or self._hover_id == wid
            track_h = max(3.0, p.px(5.0))
            t = clamp((value - lo) / (hi - lo)) if hi > lo else 0.0
            kx = tx0 + t * (tx1 - tx0)
            p.rect(tx0, ty - track_h * 0.5, tx1 - tx0, track_h, (1.0, 1.0, 1.0, 0.12), track_h * 0.5)
            if kx - tx0 >= 1.0:
                p.rect(tx0, ty - track_h * 0.5, kx - tx0, track_h, _rgba(acc, 0.95), track_h * 0.5)
            if hot:
                p.circle(kx, ty, kr * 1.9, _rgba(acc, 0.26 if active else 0.15))
            p.circle(kx, ty, kr * (1.12 if hot else 1.0), _rgba(WHITE, 1.0))
            p.circle(kx, ty, kr * 0.42, _rgba(acc, 1.0))
        col.y += h

    def _segmented_row(self, col: _Column, prefix: str, label: str,
                       options: Sequence[tuple[str, str, Any, UIAction]], current: Any) -> None:
        """Label + segmented control; options are ``(label, short label, key, action)``."""
        if not options:
            return
        p, acc = self._p, col.f.accent
        size, lab_size = p.fs(14.5), p.fs(17.5)
        seg_h, seg_pad, gap = p.px(32.0), p.px(12.0), p.px(14.0)
        n = len(options)
        label_w = p.width(label, lab_size)
        full = max(p.width(o[0], size, "bold") for o in options) + 2.0 * seg_pad
        inline = label_w + gap + full * n <= col.w
        use_short = False
        if inline:
            width = min(col.w - label_w - gap, max(full * n, col.w * 0.5))
        elif full * n <= col.w:
            width = col.w
        else:
            width, use_short = col.w, True
        h = p.px(44.0) if inline else p.px(30.0) + seg_h + p.px(10.0)
        if col.visible(h):
            if inline:
                p.text(label, col.x0, col.y + h * 0.5, lab_size, _rgba(TEXT, 0.96), valign="middle")
                sx, sy = col.x1 - width, col.y + (h - seg_h) * 0.5
            else:
                p.text(label, col.x0, col.y + p.px(4.0), lab_size, _rgba(TEXT, 0.96))
                sx, sy = col.x0, col.y + p.px(30.0)
            p.rect(sx, sy, width, seg_h, (1.0, 1.0, 1.0, 0.05), p.px(8.0), (1.0, 1.0, 1.0, 0.12), p.hair)
            seg_w = width / n
            inset = p.px(3.0)
            for i, (text, short_text, key, action) in enumerate(options):
                x = sx + i * seg_w
                selected = key == current
                hovered, pressed = self._button(f"{prefix}:{_key_id(key)}", (x, sy, seg_w, seg_h), action,
                                                clip=(col.vy0, col.vy1))
                if selected:
                    p.rect(x + inset, sy + inset, seg_w - 2 * inset, seg_h - 2 * inset, _rgba(acc, 0.92), p.px(6.0))
                elif hovered:
                    p.rect(x + inset, sy + inset, seg_w - 2 * inset, seg_h - 2 * inset,
                           (1.0, 1.0, 1.0, 0.14 if pressed else 0.08), p.px(6.0))
                shown = p.fit(short_text if use_short else text, size, "bold", seg_w - p.px(6.0))
                rgb = DARK if selected else TEXT if hovered else TEXT_2
                p.text(shown, x + seg_w * 0.5, sy + seg_h * 0.5, size, _rgba(rgb, 0.98), "bold", "center", "middle")
        col.y += h

    def _choice_row(self, col: _Column, dotted: str, label: str) -> None:
        """Segmented control for a string setting; the choices come from the config."""
        names = CHOICE_LABELS.get(dotted, {})
        current = config.get_setting(col.f.s, dotted)
        options = []
        for value in choice_values(dotted, tuple(names)):
            text, short = names.get(value, (value.upper(), value.upper()[:5]))
            options.append((text, short, value, UIAction("set_setting", (dotted, value))))
        self._segmented_row(col, f"settings:{dotted}", label, options, current)

    def _radio_row(self, col: _Column, wid: str, label: str, selected: bool, action: UIAction,
                   tag: str = "") -> None:
        p, acc = self._p, col.f.accent
        h = p.px(36.0)
        if col.visible(h):
            self._row_hover(col, wid, h, action)
            r = p.px(8.0)
            cx, cy = col.x0 + r, col.y + h * 0.5
            p.circle(cx, cy, r, _rgba(acc if selected else TEXT_3, 1.0), max(1.5, p.px(2.0)))
            if selected:
                p.circle(cx, cy, r * 0.45, _rgba(acc, 1.0))
            size, tag_size = p.fs(17.0), p.fs(12.5)
            tag_w = p.width(tag, tag_size, "bold") + p.px(12.0) if tag else 0.0
            tx = cx + r + p.px(10.0)
            text = p.fit(label, size, "ui", col.x1 - tx - (tag_w + p.px(8.0) if tag else 0.0))
            p.text(text, tx, cy, size, _rgba(TEXT if selected else TEXT_2, 0.97), valign="middle")
            if tag:
                gx = tx + p.width(text, size) + p.px(8.0)
                gh = p.px(18.0)
                p.rect(gx, cy - gh * 0.5, tag_w, gh, _rgba(AMBER, 0.14), p.px(4.0), _rgba(AMBER, 0.6), p.hair)
                p.text(tag, gx + tag_w * 0.5, cy, tag_size, _rgba(AMBER, 1.0), "bold", "center", "middle")
        col.y += h

    def _button_row_panel(self, col: _Column, specs: Sequence[tuple[str, str, UIAction | None, str, bool]]) -> None:
        p = self._p
        bh, gap = p.px(34.0), p.px(10.0)
        h = bh + p.px(12.0)
        if col.visible(h):
            n = len(specs)
            bw = (col.w - gap * (n - 1)) / n
            y = col.y + p.px(6.0)
            for i, (wid, label, action, style, enabled) in enumerate(specs):
                x = col.x0 + i * (bw + gap)
                self._panel_text_button(col, wid, label, x, y, bw, bh, action, style, enabled)
        col.y += h

    def _panel_text_button(self, col: _Column, wid: str, label: str, x: float, y: float, w: float, h: float,
                           action: UIAction | None, style: str, enabled: bool) -> None:
        """``_text_button`` whose hit area is clipped to the scroll viewport."""
        clipped = _clip_rect((x, y, w, h), col.vy0, col.vy1)
        reg_len = len(self._reg)
        self._text_button(col.f, wid, label, x, y, w, h, action, style, enabled, self._p.fs(14.5))
        if len(self._reg) > reg_len:              # replace the registered rect by the clipped one
            widget = self._reg.pop()
            if clipped is not None:
                self._reg.append(UIWidget(widget.id, clipped, widget.kind, widget.action, None, widget.enabled))

    def _swatch_row(self, col: _Column) -> None:
        p = self._p
        h = p.px(46.0)
        if col.visible(h):
            current = col.f.s.effects.energy_color
            allowed = choice_values("effects.energy_color", tuple(PALETTES))
            names = [name for name in PALETTES if name in allowed]       # PALETTES order
            r, gap = p.px(12.5), p.px(12.0)
            total = len(names) * 2.0 * r + max(0, len(names) - 1) * gap
            size = p.fs(17.5)
            cy = col.y + h * 0.5
            label = "Energy colour"
            current_label = get_palette(current).label
            room = col.w - total - p.px(14.0)
            p.text(p.fit(label, size, "ui", room), col.x0, cy, size, _rgba(TEXT, 0.96), valign="middle")
            extra = room - p.width(label, size) - p.px(8.0)
            if extra > p.px(40.0):
                p.text(p.fit(current_label, p.fs(14.0), "ui", extra), col.x0 + p.width(label, size) + p.px(8.0), cy,
                       p.fs(14.0), _rgba(TEXT_3, 0.95), valign="middle")
            sx = col.x1 - total + r
            for name in names:
                pal = PALETTES[name]
                hit = (sx - r - p.px(4.0), cy - r - p.px(4.0), 2.0 * r + p.px(8.0), 2.0 * r + p.px(8.0))
                hovered, _ = self._button(f"settings:color:{name}", hit,
                                          UIAction("set_setting", ("effects.energy_color", name)),
                                          clip=(col.vy0, col.vy1))
                if name == current:
                    p.circle(sx, cy, r + p.px(4.0), _rgba(WHITE, 1.0), max(1.5, p.px(2.0)))
                elif hovered:
                    p.circle(sx, cy, r + p.px(4.0), _rgba(pal.ui, 1.0), max(1.0, p.px(1.5)))
                p.circle(sx, cy, r, _rgba(pal.inner, 1.0))
                p.circle(sx, cy, r * 0.42, _rgba(pal.core, 0.95))
                sx += 2.0 * r + gap
        col.y += h
