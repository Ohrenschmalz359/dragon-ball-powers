"""Backend-neutral render primitives.

The effects/ability layer describes *what* to draw each frame by filling a
:class:`RenderList`; a renderer backend (OpenGL or CPU) decides *how*. All
positions and sizes are in **scene pixels** (camera frame coordinates, x right,
y down). Colours are linear RGB floats; values above 1.0 are HDR and bloom.

The UI layer separately produces a :class:`UIDrawList` in **window pixels**.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from dragon_ball_powers.mathutil import Vec2

Color3 = tuple[float, float, float]
Color4 = tuple[float, float, float, float]

# ----------------------------------------------------------------------- palettes


@dataclass(frozen=True)
class Palette:
    """Energy colour scheme. ``core`` is the white-hot centre, ``outer`` the deep halo."""

    name: str
    label: str
    core: Color3
    inner: Color3
    outer: Color3
    spark: Color3
    ui: Color3          # accent colour for HUD elements


PALETTES: dict[str, Palette] = {
    "blue": Palette("blue", "Blue-white",
                    core=(1.00, 1.00, 1.00), inner=(0.55, 0.86, 1.00),
                    outer=(0.08, 0.36, 1.00), spark=(0.78, 0.94, 1.00), ui=(0.35, 0.78, 1.00)),
    "purple": Palette("purple", "Purple",
                      core=(1.00, 0.96, 1.00), inner=(0.86, 0.58, 1.00),
                      outer=(0.46, 0.10, 0.96), spark=(0.95, 0.82, 1.00), ui=(0.74, 0.46, 1.00)),
    "red": Palette("red", "Red",
                   core=(1.00, 0.97, 0.93), inner=(1.00, 0.46, 0.26),
                   outer=(0.88, 0.04, 0.04), spark=(1.00, 0.82, 0.62), ui=(1.00, 0.36, 0.30)),
    "gold": Palette("gold", "Golden yellow",
                    core=(1.00, 1.00, 0.93), inner=(1.00, 0.86, 0.32),
                    outer=(1.00, 0.50, 0.02), spark=(1.00, 0.96, 0.72), ui=(1.00, 0.80, 0.25)),
}


def get_palette(name: str) -> Palette:
    return PALETTES.get(name, PALETTES["blue"])


# ---------------------------------------------------------------- quality presets


@dataclass(frozen=True)
class QualityPreset:
    name: str
    render_scale: float          # resolution of the effect buffers relative to the window
    bloom_levels: int            # mip levels in the bloom chain
    particle_budget: int         # maximum live particles across all emitters
    arc_count: int               # lightning arcs per effect
    noise_octaves: int           # fbm octaves in energy shaders
    aura_resolution: int         # width of the segmentation mask used for the aura
    refraction: bool             # shockwave distortion of the camera image
    chromatic: bool              # chromatic aberration on impacts


QUALITY_PRESETS: dict[str, QualityPreset] = {
    "low": QualityPreset("low", 0.5, 4, 700, 2, 2, 128, False, False),
    "medium": QualityPreset("medium", 0.75, 5, 1600, 4, 3, 192, True, False),
    "high": QualityPreset("high", 1.0, 6, 3200, 6, 4, 256, True, True),
    "ultra": QualityPreset("ultra", 1.0, 7, 6400, 8, 5, 320, True, True),
}


def get_quality(name: str) -> QualityPreset:
    return QUALITY_PRESETS.get(name, QUALITY_PRESETS["high"])


# ------------------------------------------------------------- effect primitives


class SpriteShape(enum.IntEnum):
    GLOW = 0     # soft round gaussian glow
    SPARK = 1    # bright core streak, stretched along velocity
    EMBER = 2    # small hot dot with tight falloff
    SMOKE = 3    # large faint wisp (alpha-ish, still additive)


@dataclass
class SpriteBatch:
    """Struct-of-arrays batch of particles sharing one shape. All arrays have length N."""

    shape: SpriteShape
    positions: np.ndarray      # (N, 2) float32 scene px
    sizes: np.ndarray          # (N,) float32 radius in px
    colors: np.ndarray         # (N, 4) float32: rgb already multiplied by intensity, a = opacity
    velocities: np.ndarray | None = None   # (N, 2) float32 px/s (used by SPARK stretching)
    stretch: float = 0.0       # seconds of motion blur for SPARK streaks (length = |v| * stretch)

    def __len__(self) -> int:
        return int(self.positions.shape[0])


@dataclass
class Orb:
    """A ki energy sphere."""

    center: Vec2
    radius: float
    palette: "Palette"
    intensity: float = 1.0       # brightness multiplier (HDR)
    charge: float = 0.0          # 0..1 charge level (drives detail, flares, crackle)
    time: float = 0.0            # animation time in seconds
    seed: float = 0.0
    swirl: float = 1.0           # swirl speed multiplier
    instability: float = 0.0     # 0..1 surface turbulence / crackle
    flare: float = 0.0           # 0..1 radial flare spikes around the sphere
    squash: Vec2 = (1.0, 1.0)    # elliptical scale along / across ``rotation``
    rotation: float = 0.0        # radians


@dataclass
class Beam:
    """A lateral energy beam travelling across the image plane."""

    origin: Vec2
    direction: Vec2              # unit vector
    length: float                # px from origin to the beam head
    width: float                 # half-width at the origin, px
    end_width: float             # half-width at the head, px (perspective widening)
    palette: "Palette"
    intensity: float = 1.0
    time: float = 0.0
    seed: float = 0.0
    head_radius: float = 0.0     # radius of the bulging beam head (0 = none)
    turbulence: float = 1.0
    flow_speed: float = 1.0      # how fast energy streams along the beam
    fade: float = 1.0            # 0..1 overall opacity
    dissolve: float = 0.0        # 0..1 break-up into wisps while dissipating


@dataclass
class FrontalBeam:
    """An energy beam fired towards the camera, seen end-on."""

    center: Vec2
    radius: float                # radius of the beam cross-section, px
    palette: "Palette"
    intensity: float = 1.0
    time: float = 0.0
    seed: float = 0.0
    streaks: float = 1.0         # strength of radial speed lines
    fade: float = 1.0
    dissolve: float = 0.0


@dataclass
class Arc:
    """A lightning bolt / electrical arc as a polyline."""

    points: np.ndarray           # (M, 2) float32 scene px
    width: float                 # core half-width in px
    palette: "Palette"
    intensity: float = 1.0
    glow: float = 1.0            # size multiplier of the soft glow around the core


@dataclass
class Ring:
    """An expanding shockwave ring."""

    center: Vec2
    radius: float
    thickness: float
    palette: "Palette"
    intensity: float = 1.0
    refraction: float = 0.0      # px displacement applied to the camera image (if supported)
    noise: float = 0.5           # 0..1 break-up of the ring edge
    squash: Vec2 = (1.0, 1.0)    # x/y scale for perspective ellipses
    time: float = 0.0


@dataclass
class Ribbon:
    """A motion trail: a polyline with per-point width and intensity."""

    points: np.ndarray           # (M, 2) float32 scene px, oldest first
    widths: np.ndarray           # (M,) float32 half-widths in px
    intensities: np.ndarray      # (M,) float32 0..1+
    palette: "Palette"


@dataclass
class Flash:
    """A screen flash; radial around ``center`` when given, otherwise uniform."""

    intensity: float
    color: Color3 = (1.0, 1.0, 1.0)
    center: Vec2 | None = None
    radius: float = 0.0


@dataclass
class Light:
    """Coloured light that illuminates the camera image (energy lighting the user)."""

    position: Vec2
    radius: float                # falloff radius in px
    color: Color3
    intensity: float = 1.0


@dataclass
class Aura:
    """Flame-like energy aura around the user's silhouette."""

    mask: np.ndarray             # (h, w) uint8 0..255 person mask aligned to the scene
    mask_id: int                 # changes when ``mask`` content changes (avoid re-uploads)
    palette: "Palette"
    intensity: float = 1.0
    time: float = 0.0
    flame_height: float = 0.08   # flame reach as a fraction of scene height
    turbulence: float = 1.0
    flicker: float = 0.5


@dataclass
class Grade:
    """Camera image colour grading for cinematic moments.

    The defaults are the identity: a default ``Grade`` leaves the camera image untouched.
    """

    darken: float = 0.0          # 0..1 dim the room
    desaturate: float = 0.0      # 0..1
    tint: Color3 = (1.0, 1.0, 1.0)
    tint_strength: float = 0.0   # 0..1 blend towards ``tint`` multiplication
    vignette: float = 0.0        # 0..1
    chromatic: float = 0.0       # chromatic aberration amount in px (impacts)
    zoom: float = 1.0            # zoom punch (1 = none)


@dataclass
class RenderList:
    """Everything the effects layer wants drawn this frame (scene space)."""

    sprites: list[SpriteBatch] = field(default_factory=list)
    orbs: list[Orb] = field(default_factory=list)
    beams: list[Beam] = field(default_factory=list)
    frontal_beams: list[FrontalBeam] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    rings: list[Ring] = field(default_factory=list)
    ribbons: list[Ribbon] = field(default_factory=list)
    flashes: list[Flash] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    aura: Aura | None = None
    grade: Grade = field(default_factory=Grade)
    shake: Vec2 = (0.0, 0.0)     # camera shake offset in scene px
    bloom: float = 1.0           # bloom strength multiplier for this frame
    exposure: float = 1.0

    def clear(self) -> None:
        for name in ("sprites", "orbs", "beams", "frontal_beams", "arcs", "rings",
                     "ribbons", "flashes", "lights"):
            getattr(self, name).clear()
        self.aura = None
        self.grade = Grade()
        self.shake = (0.0, 0.0)
        self.bloom = 1.0
        self.exposure = 1.0

    def is_empty(self) -> bool:
        return not (self.sprites or self.orbs or self.beams or self.frontal_beams or self.arcs
                    or self.rings or self.ribbons or self.flashes or self.aura)


# --------------------------------------------------------------- view transform


@dataclass(frozen=True)
class ViewTransform:
    """Maps scene pixels to window pixels (uniform scale + offset)."""

    scale: float
    offset_x: float
    offset_y: float
    scene_w: int
    scene_h: int
    win_w: int
    win_h: int

    def scene_to_window(self, x: float, y: float) -> Vec2:
        return (x * self.scale + self.offset_x, y * self.scale + self.offset_y)

    def window_to_scene(self, x: float, y: float) -> Vec2:
        return ((x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale)

    def length_to_window(self, length: float) -> float:
        return length * self.scale


def compute_view_transform(scene_w: int, scene_h: int, win_w: int, win_h: int,
                           mode: str = "cover") -> ViewTransform:
    """``cover`` fills the window (cropping), ``contain`` letterboxes the whole scene."""
    scene_w = max(1, int(scene_w))
    scene_h = max(1, int(scene_h))
    win_w = max(1, int(win_w))
    win_h = max(1, int(win_h))
    sx, sy = win_w / scene_w, win_h / scene_h
    scale = max(sx, sy) if mode == "cover" else min(sx, sy)
    offset_x = (win_w - scene_w * scale) * 0.5
    offset_y = (win_h - scene_h * scale) * 0.5
    return ViewTransform(scale, offset_x, offset_y, scene_w, scene_h, win_w, win_h)


# ------------------------------------------------------------------- UI commands


@dataclass
class UIRect:
    x: float
    y: float
    w: float
    h: float
    color: Color4                          # fill colour (alpha 0 = no fill)
    radius: float = 0.0                    # corner radius
    border_color: Color4 | None = None
    border_width: float = 0.0
    gradient_to: Color4 | None = None      # vertical gradient: color at top -> gradient_to at bottom


@dataclass
class UIText:
    text: str
    x: float
    y: float
    size: int                              # font pixel size
    color: Color4
    font: str = "ui"                       # "ui" | "bold" | "display" | "mono"
    align: str = "left"                    # "left" | "center" | "right" (relative to x)
    valign: str = "top"                    # "top" | "middle" | "bottom" (relative to y)
    glow: float = 0.0                      # glow radius in px (0 = none)
    glow_color: Color4 | None = None
    shadow: bool = False                   # small drop shadow for legibility


@dataclass
class UILine:
    points: list[Vec2]
    color: Color4
    width: float = 1.0
    closed: bool = False


@dataclass
class UICircle:
    x: float
    y: float
    radius: float
    color: Color4
    width: float = 0.0                     # 0 = filled disc, otherwise outline width


@dataclass
class UIArc:
    """Circular arc (progress ring). Angles in radians, 0 = +x axis, clockwise on screen."""

    x: float
    y: float
    radius: float
    start: float
    end: float
    color: Color4
    width: float = 4.0


UICommand = UIRect | UIText | UILine | UICircle | UIArc


@dataclass
class UIDrawList:
    commands: list[UICommand] = field(default_factory=list)

    def add(self, command: UICommand) -> None:
        self.commands.append(command)

    def extend(self, commands: list[UICommand]) -> None:
        self.commands.extend(commands)

    def __len__(self) -> int:
        return len(self.commands)


@dataclass
class SceneInput:
    """Camera image (and metadata) the renderer composites effects onto."""

    image: np.ndarray | None               # BGR uint8 scene image, None = show placeholder
    image_id: int = -1                     # changes when ``image`` changes (skip re-uploads)
    scene_size: tuple[int, int] = (1280, 720)
    time: float = 0.0                      # seconds, for animated placeholder backgrounds
