#!/usr/bin/env python3
"""Render preview PNGs of the OpenGL renderer (look-development tool).

Every scenario is built from hand-made render primitives (no dependency on the
effect simulation), composited by :class:`GLRenderer` in a headless OpenGL context
and written as PNG:

    idle, placeholder, orb_charge_20, orb_charge_60, orb_charge_100, beam_lateral,
    beam_frontal, energy_burst, aura, ki_blast, ui

Backgrounds: a procedurally generated room with a stylised person (default), or any
image via ``--background PATH`` (it is cover-cropped to the preview size). Images
are only written to the output folder (default ``<project>/previews``, gitignored).

Examples::

    python scripts/render_preview.py
    python scripts/render_preview.py --scenarios orb_charge_100,beam_lateral --palette gold
    python scripts/render_preview.py --background my_photo.jpg --out C:\\temp\\previews --sheet
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from dragon_ball_powers.fonts import FontManager  # noqa: E402
from dragon_ball_powers.interfaces import RendererError  # noqa: E402
from dragon_ball_powers.primitives import (  # noqa: E402
    PALETTES,
    Arc,
    Aura,
    Beam,
    Flash,
    FrontalBeam,
    Grade,
    Light,
    Orb,
    Palette,
    RenderList,
    Ribbon,
    Ring,
    SceneInput,
    SpriteBatch,
    SpriteShape,
    UIArc,
    UICircle,
    UIDrawList,
    UILine,
    UIRect,
    UIText,
    compute_view_transform,
    get_palette,
    get_quality,
)
from dragon_ball_powers.renderer import GLRenderer  # noqa: E402

Vec = tuple[float, float]

# ------------------------------------------------------------------ the subject


@dataclass
class Pose:
    """Key points of the stylised person in a 1280x720 reference frame (scaled later)."""

    left_hand: Vec
    right_hand: Vec
    left_elbow: Vec
    right_elbow: Vec
    head: Vec = (640.0, 205.0)
    left_shoulder: Vec = (560.0, 305.0)
    right_shoulder: Vec = (720.0, 305.0)


POSES: dict[str, Pose] = {
    "rest": Pose((545, 585), (735, 585), (535, 450), (745, 450)),
    "charge": Pose((600, 440), (680, 440), (520, 445), (760, 445)),
    "beam": Pose((865, 372), (878, 402), (735, 360), (760, 395)),
    "frontal": Pose((612, 405), (668, 405), (545, 400), (735, 400)),
    "burst": Pose((330, 300), (950, 300), (450, 300), (830, 300)),
    "blast": Pose((540, 590), (870, 380), (530, 455), (780, 380)),
}


@dataclass
class Frame:
    """A preview scene: camera image + person mask + scale helpers."""

    image: np.ndarray
    mask: np.ndarray
    w: int
    h: int
    pose: Pose
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(7))

    @property
    def k(self) -> float:            # scene scale relative to 720p
        return self.h / 720.0

    def p(self, v: Vec) -> Vec:      # reference px -> scene px
        return (v[0] * self.w / 1280.0, v[1] * self.h / 720.0)


def make_room(w: int, h: int, seed: int = 3) -> np.ndarray:
    """A plausible dim living room (BGR uint8): wall, floor, window, shelf, lamp, sofa."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u, v = xx / w, yy / h
    horizon = 0.70
    img = np.zeros((h, w, 3), np.float32)
    wall = np.array([118, 128, 138], np.float32)          # warm grey (BGR)
    light = np.exp(-(((u - 0.82) / 0.35) ** 2 + ((v - 0.25) / 0.45) ** 2))
    img[:] = wall * (0.58 + 0.30 * (1.0 - v)[..., None] * 0.6 + 0.35 * light[..., None])
    floor = v > horizon
    planks = 0.92 + 0.08 * np.sin(u * 70.0 + np.floor((v - horizon) * 40.0) * 2.1)
    floor_col = np.array([52, 74, 102], np.float32) * (0.7 + 0.5 * (v - horizon))[..., None]
    img[floor] = (floor_col * planks[..., None])[floor]
    cv2.line(img, (0, int(horizon * h)), (w, int(horizon * h)), (40, 46, 52), max(1, h // 240), cv2.LINE_AA)
    # window with daylight on the left
    x0, y0, x1, y1 = int(0.05 * w), int(0.12 * h), int(0.25 * w), int(0.58 * h)
    sky = np.linspace(1.0, 0.82, y1 - y0, dtype=np.float32)[:, None, None] * np.array([235, 222, 205], np.float32)
    img[y0:y1, x0:x1] = sky
    cv2.rectangle(img, (x0, y0), (x1, y1), (205, 210, 215), max(2, h // 90))
    cv2.line(img, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), (205, 210, 215), max(2, h // 120))
    cv2.line(img, (x0, (y0 + y1) // 2), (x1, (y0 + y1) // 2), (205, 210, 215), max(2, h // 120))
    spill = np.exp(-(((u - 0.15) / 0.20) ** 2 + ((v - 0.75) / 0.12) ** 2)) * 40.0
    img += spill[..., None] * np.array([1.0, 0.97, 0.92], np.float32)
    # bookshelf on the right
    bx0, by0, bx1, by1 = int(0.80 * w), int(0.18 * h), int(0.97 * w), int(0.70 * h)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (30, 42, 58), -1)
    for row in range(4):
        ry = by0 + int((row + 0.15) * (by1 - by0) / 4)
        cv2.rectangle(img, (bx0, ry + int(0.2 * (by1 - by0) / 4)), (bx1, ry + int(0.24 * (by1 - by0) / 4)),
                      (22, 30, 42), -1)
        x = bx0 + 6
        while x < bx1 - 10:
            bw = int(rng.integers(max(4, w // 160), max(8, w // 70)))
            bh = int((by1 - by0) / 4 * rng.uniform(0.55, 0.8))
            col = tuple(float(c) for c in rng.integers(40, 160, 3))
            cv2.rectangle(img, (x, ry + int(0.2 * (by1 - by0) / 4) - bh), (x + bw, ry + int(0.2 * (by1 - by0) / 4)),
                          col, -1)
            x += bw + 2
    # framed picture and a sofa
    cv2.rectangle(img, (int(0.33 * w), int(0.16 * h)), (int(0.43 * w), int(0.34 * h)), (60, 72, 84), -1)
    cv2.rectangle(img, (int(0.34 * w), int(0.18 * h)), (int(0.42 * w), int(0.32 * h)), (120, 140, 96), -1)
    cv2.rectangle(img, (int(0.27 * w), int(0.55 * h)), (int(0.75 * w), int(0.80 * h)), (70, 64, 92), -1)
    cv2.rectangle(img, (int(0.25 * w), int(0.60 * h)), (int(0.30 * w), int(0.83 * h)), (62, 56, 82), -1)
    cv2.rectangle(img, (int(0.72 * w), int(0.60 * h)), (int(0.77 * w), int(0.83 * h)), (62, 56, 82), -1)
    # floor lamp glow
    lamp = np.exp(-(((u - 0.72) / 0.05) ** 2 + ((v - 0.30) / 0.06) ** 2))
    img += lamp[..., None] * np.array([120, 190, 245], np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 1.0 * h / 720.0 + 0.3)
    img += rng.normal(0.0, 3.0, img.shape).astype(np.float32)     # sensor noise
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_person(img: np.ndarray, pose: Pose, w: int, h: int) -> np.ndarray:
    """Draw a simple person into ``img`` (in place); returns the matching uint8 mask."""
    sx, sy, k = w / 1280.0, h / 720.0, h / 720.0
    mask = np.zeros((h, w), np.uint8)

    def P(v: Vec) -> tuple[int, int]:
        return (int(v[0] * sx), int(v[1] * sy))

    skin, shirt, hair, pants = (112, 140, 196), (92, 70, 54), (30, 32, 36), (58, 50, 44)
    torso = np.array([P((548, 300)), P((732, 300)), P((712, 620)), P((568, 620))], np.int32)
    arms = [(pose.left_shoulder, pose.left_elbow, pose.left_hand), (pose.right_shoulder, pose.right_elbow,
                                                                    pose.right_hand)]
    for target, value in ((img, None), (mask, 255)):
        def col(c: tuple[int, int, int]) -> tuple[int, ...] | int:
            return value if value is not None else c
        cv2.fillConvexPoly(target, np.array([P((585, 690)), P((695, 690)), P((700, 610)), P((580, 610))], np.int32),
                           col(pants), cv2.LINE_AA)
        cv2.fillConvexPoly(target, torso, col(shirt), cv2.LINE_AA)
        cv2.line(target, P((640, 240)), P((640, 300)), col(skin), int(42 * k), cv2.LINE_AA)
        cv2.ellipse(target, P(pose.head), (int(50 * k), int(62 * k)), 0, 0, 360, col(skin), -1, cv2.LINE_AA)
        cv2.ellipse(target, P((pose.head[0], pose.head[1] - 28)), (int(54 * k), int(40 * k)), 0, 180, 360,
                    col(hair), -1, cv2.LINE_AA)
        for shoulder, elbow, hand in arms:
            cv2.line(target, P(shoulder), P(elbow), col(shirt), int(46 * k), cv2.LINE_AA)
            cv2.line(target, P(elbow), P(hand), col(skin), int(34 * k), cv2.LINE_AA)
            cv2.circle(target, P(hand), int(27 * k), col(skin), -1, cv2.LINE_AA)
    return mask


def make_frame(pose_name: str, w: int, h: int, background: np.ndarray | None,
               mask_override: np.ndarray | None) -> Frame:
    pose = POSES[pose_name]
    if background is None:
        img = make_room(w, h)
        mask = draw_person(img, pose, w, h)
    else:
        img = background.copy()
        mask = mask_override if mask_override is not None else draw_person(np.zeros_like(img), pose, w, h)
    return Frame(img, mask, w, h, pose)


def cover_resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = max(w / iw, h / ih)
    resized = cv2.resize(img, (max(w, int(round(iw * s))), max(h, int(round(ih * s)))), interpolation=cv2.INTER_AREA)
    y0 = (resized.shape[0] - h) // 2
    x0 = (resized.shape[1] - w) // 2
    return np.ascontiguousarray(resized[y0:y0 + h, x0:x0 + w])


# ---------------------------------------------------------- primitive builders


def lightning(rng: np.random.Generator, a: Vec, b: Vec, detail: int = 5, jag: float = 0.22) -> np.ndarray:
    """Midpoint-displacement lightning polyline from ``a`` to ``b``."""
    pts = np.array([a, b], np.float64)
    disp = jag * math.dist(a, b)
    for _ in range(detail):
        mids = (pts[:-1] + pts[1:]) * 0.5
        d = pts[1:] - pts[:-1]
        n = np.stack([-d[:, 1], d[:, 0]], axis=1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        mids += n * rng.uniform(-1.0, 1.0, (len(mids), 1)) * disp
        out = np.empty((len(pts) + len(mids), 2))
        out[0::2] = pts
        out[1::2] = mids
        pts = out
        disp *= 0.55
    return pts.astype(np.float32)


def sprites(shape: SpriteShape, pos: np.ndarray, sizes: np.ndarray, rgb: np.ndarray, alpha: np.ndarray | float = 1.0,
            vel: np.ndarray | None = None, stretch: float = 0.0) -> SpriteBatch:
    n = len(pos)
    colors = np.empty((n, 4), np.float32)
    colors[:, :3] = rgb
    colors[:, 3] = alpha
    return SpriteBatch(shape, pos.astype(np.float32), sizes.astype(np.float32), colors,
                       None if vel is None else vel.astype(np.float32), stretch)


def ring_points(rng: np.random.Generator, c: Vec, r0: float, r1: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    ang = rng.uniform(0, math.tau, n)
    rad = np.sqrt(rng.uniform(r0 * r0, r1 * r1, n))
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    return np.asarray(c) + dirs * rad[:, None], dirs


def gather_particles(f: Frame, pal: Palette, c: Vec, radius: float, n: int) -> list[SpriteBatch]:
    """Energy spiralling into a charging orb + sparks orbiting it."""
    rng = f.rng
    pos, dirs = ring_points(rng, c, radius * 1.4, radius * 4.0, n)
    tangent = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
    vel = (-dirs * 1.0 + tangent * 0.8) * rng.uniform(120, 320, (n, 1)) * f.k
    dist = np.linalg.norm(pos - np.asarray(c), axis=1)
    bright = np.clip(1.6 - dist / (radius * 4.0), 0.25, 1.6)[:, None]
    glow = sprites(SpriteShape.GLOW, pos, rng.uniform(3, 7, n) * f.k, np.asarray(pal.inner) * 0.9 * bright, 0.9)
    m = n // 2
    epos, edirs = ring_points(rng, c, radius * 1.2, radius * 2.6, m)
    embers = sprites(SpriteShape.EMBER, epos, rng.uniform(2.0, 3.6, m) * f.k, np.asarray(pal.spark) * 2.2, 1.0)
    s = n // 3
    spos, sdirs = ring_points(rng, c, radius * 1.05, radius * 1.7, s)
    stan = np.stack([-sdirs[:, 1], sdirs[:, 0]], axis=1)
    sparks = sprites(SpriteShape.SPARK, spos, rng.uniform(1.4, 2.4, s) * f.k, np.asarray(pal.spark) * 3.0, 1.0,
                     vel=stan * rng.uniform(500, 900, (s, 1)) * f.k, stretch=0.022)
    smoke = sprites(SpriteShape.SMOKE, pos[:12], np.full(min(12, n), 34.0 * f.k), np.asarray(pal.outer) * 0.5, 0.6)
    return [glow, embers, sparks, smoke]


# ------------------------------------------------------------------ scenarios


@dataclass
class Scenario:
    pose: str
    build: Callable[[Frame, Palette, float], tuple[RenderList, UIDrawList]]
    camera: bool = True


def sc_idle(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
    return RenderList(), UIDrawList()


def sc_placeholder(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
    ui = UIDrawList()
    ui.add(UIText("NO CAMERA", f.w / 2, f.h / 2 - 20 * f.k, int(44 * f.k), (0.86, 0.92, 1.0, 0.95), "display",
                  "center", "middle", glow=8.0, glow_color=(*pal.ui, 0.7)))
    ui.add(UIText("Connect a webcam or close the app that is using it", f.w / 2, f.h / 2 + 26 * f.k, int(22 * f.k),
                  (0.75, 0.8, 0.9, 0.9), "ui", "center", "middle", shadow=True))
    return RenderList(), ui


def orb_scene(charge: float) -> Callable[[Frame, Palette, float], tuple[RenderList, UIDrawList]]:
    def build(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
        rl = RenderList()
        lh, rh = f.p(f.pose.left_hand), f.p(f.pose.right_hand)
        c = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2 - 8 * f.k)
        radius = (22 + 48 * charge) * f.k
        rl.orbs.append(Orb(c, radius, pal, intensity=0.9 + 0.5 * charge, charge=charge, time=t, seed=3.0,
                           swirl=0.8 + 0.8 * charge, instability=0.15 + 0.5 * charge * charge,
                           flare=max(0.0, (charge - 0.45) / 0.55)))
        rl.sprites.extend(gather_particles(f, pal, c, radius, int(60 + 240 * charge)))
        if charge >= 0.5:
            for hand in (lh, rh):
                for _ in range(1 if charge < 0.9 else 2):
                    rl.arcs.append(Arc(lightning(f.rng, c, hand, 5, 0.25), 1.3 * f.k, pal, 1.0 + charge, 1.0))
        if charge >= 0.95:
            for _ in range(3):   # crackle over the surface
                a0 = f.rng.uniform(0, math.tau)
                a1 = a0 + f.rng.uniform(0.8, 1.6)
                p0 = (c[0] + math.cos(a0) * radius * 1.05, c[1] + math.sin(a0) * radius * 1.05)
                p1 = (c[0] + math.cos(a1) * radius * 1.1, c[1] + math.sin(a1) * radius * 1.1)
                rl.arcs.append(Arc(lightning(f.rng, p0, p1, 4, 0.3), 1.0 * f.k, pal, 1.4, 0.8))
            rl.rings.append(Ring(c, radius * 2.6, 6 * f.k, pal, intensity=0.9, refraction=10 * f.k, noise=0.6,
                                 time=t))
        rl.lights.append(Light(c, radius * 6.0, pal.inner, 0.4 + 0.6 * charge))
        rl.grade = Grade(darken=0.45 * charge, vignette=0.25 * charge)
        rl.bloom = 1.0 + 0.3 * charge
        return rl, UIDrawList()
    return build


def sc_beam(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
    rl = RenderList()
    lh, rh = f.p(f.pose.left_hand), f.p(f.pose.right_hand)
    origin = ((lh[0] + rh[0]) / 2 + 22 * f.k, (lh[1] + rh[1]) / 2)
    d = (1.0, -0.06)
    n = math.hypot(*d)
    d = (d[0] / n, d[1] / n)
    length = f.w - origin[0] - 60 * f.k
    width = 34 * f.k
    rl.beams.append(Beam(origin, d, length, width, width * 1.35, pal, intensity=1.25, time=t, seed=5.0,
                         head_radius=width * 1.7, turbulence=1.0, flow_speed=1.2))
    perp = (-d[1], d[0])
    for side in (-1, 1):            # arcs crawling along the edges
        for j in range(2):
            u0 = f.rng.uniform(0.05, 0.55) * length
            u1 = u0 + f.rng.uniform(0.18, 0.35) * length
            off = side * width * f.rng.uniform(1.0, 1.5)
            a = (origin[0] + d[0] * u0 + perp[0] * off, origin[1] + d[1] * u0 + perp[1] * off)
            b = (origin[0] + d[0] * u1 + perp[0] * off, origin[1] + d[1] * u1 + perp[1] * off)
            rl.arcs.append(Arc(lightning(f.rng, a, b, 5, 0.12), 1.2 * f.k, pal, 1.2, 1.0))
    for i, u in enumerate((0.18, 0.45, 0.72)):
        c = (origin[0] + d[0] * length * u, origin[1] + d[1] * length * u)
        rl.rings.append(Ring(c, width * (1.9 + 0.5 * i), 4 * f.k, pal, intensity=0.7, refraction=0.0, noise=0.5,
                             squash=(0.32, 1.0), time=t + i))
    nsp = 420
    us = f.rng.uniform(0, 1, nsp) * length
    vs = f.rng.normal(0, width * 1.1, nsp)
    pos = np.stack([origin[0] + d[0] * us + perp[0] * vs, origin[1] + d[1] * us + perp[1] * vs], axis=1)
    vel = np.tile(np.array(d) * 1400 * f.k, (nsp, 1)) + f.rng.normal(0, 120 * f.k, (nsp, 2))
    rl.sprites.append(sprites(SpriteShape.SPARK, pos, f.rng.uniform(1.2, 2.2, nsp) * f.k, np.asarray(pal.spark) * 2.5,
                              1.0, vel=vel, stretch=0.02))
    rl.sprites.append(sprites(SpriteShape.EMBER, pos[:160] + f.rng.normal(0, width, (160, 2)),
                              f.rng.uniform(2, 3.5, 160) * f.k, np.asarray(pal.spark) * 2.0, 1.0))
    rl.lights.append(Light(origin, width * 9, pal.inner, 0.8))
    rl.lights.append(Light((origin[0] + d[0] * length * 0.5, origin[1]), width * 8, pal.inner, 0.5))
    rl.grade = Grade(darken=0.5, vignette=0.3)
    rl.shake = (2.0 * f.k, -1.5 * f.k)
    rl.bloom = 1.3
    return rl, UIDrawList()


def sc_frontal(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
    rl = RenderList()
    lh, rh = f.p(f.pose.left_hand), f.p(f.pose.right_hand)
    c = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2 - 6 * f.k)
    radius = 0.45 * f.h * 0.5
    rl.frontal_beams.append(FrontalBeam(c, radius, pal, intensity=1.2, time=t, seed=2.0, streaks=1.0))
    for i, rr in enumerate((1.4, 2.1, 3.0)):
        rl.rings.append(Ring(c, radius * rr, (8 + 5 * i) * f.k, pal, intensity=0.6 - 0.12 * i, refraction=0.0,
                             noise=0.55, time=t + i))
    n = 360
    pos, dirs = ring_points(f.rng, c, radius * 1.1, f.w * 0.62, n)
    dist = np.linalg.norm(pos - np.asarray(c), axis=1)
    vel = dirs * (dist * 3.0)[:, None]
    rl.sprites.append(sprites(SpriteShape.SPARK, pos, (1.5 + dist / f.w * 4.0) * f.k, np.asarray(pal.spark) * 2.2,
                              1.0, vel=vel, stretch=0.03))
    rl.lights.append(Light(c, radius * 4.0, pal.inner, 1.2))
    rl.flashes.append(Flash(0.12, pal.inner))
    rl.grade = Grade(darken=0.55, vignette=0.35)
    rl.shake = (-3.0 * f.k, 2.5 * f.k)
    rl.bloom = 1.4
    return rl, UIDrawList()


def sc_burst(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
    rl = RenderList()
    c = f.p((640, 380))
    rng = f.rng
    for i, (r, th, refr) in enumerate(((130, 12, 22), (260, 9, 16), (420, 7, 10))):
        rl.rings.append(Ring(c, r * f.k, th * f.k, pal, intensity=1.3 - 0.3 * i, refraction=refr * f.k,
                             noise=0.45, time=t + i))
    n = 700
    ang = rng.uniform(0, math.tau, n)
    rad = rng.uniform(40, 520, n) * f.k
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    pos = np.asarray(c) + dirs * rad[:, None]
    vel = dirs * rng.uniform(900, 2200, (n, 1)) * f.k
    rl.sprites.append(sprites(SpriteShape.SPARK, pos, rng.uniform(1.3, 2.6, n) * f.k, np.asarray(pal.spark) * 3.0,
                              1.0, vel=vel, stretch=0.025))
    m = 260
    epos = np.asarray(c) + dirs[:m] * rng.uniform(60, 300, (m, 1)) * f.k
    rl.sprites.append(sprites(SpriteShape.EMBER, epos, rng.uniform(2, 4, m) * f.k, np.asarray(pal.spark) * 2.4, 1.0))
    rl.sprites.append(sprites(SpriteShape.SMOKE, epos[:30], rng.uniform(40, 80, 30) * f.k,
                              np.asarray(pal.outer) * 0.35, 0.8))
    for k in range(7):
        a = k / 7 * math.tau + rng.uniform(-0.2, 0.2)
        p1 = (c[0] + math.cos(a) * 260 * f.k, c[1] + math.sin(a) * 260 * f.k)
        rl.arcs.append(Arc(lightning(rng, c, p1, 5, 0.2), 1.6 * f.k, pal, 1.5, 1.2))
    rl.flashes.append(Flash(1.2, pal.core, center=c, radius=140 * f.k))
    rl.flashes.append(Flash(0.15, (1.0, 1.0, 1.0)))
    rl.lights.append(Light(c, 380 * f.k, pal.inner, 1.0))
    rl.grade = Grade(darken=0.35, chromatic=5.0 * f.k, vignette=0.3, zoom=1.03)
    rl.shake = (6.0 * f.k, -4.0 * f.k)
    rl.bloom = 1.5
    return rl, UIDrawList()


def sc_aura(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
    rl = RenderList()
    mask = cv2.resize(f.mask, (256, max(2, int(256 * f.h / f.w))), interpolation=cv2.INTER_AREA)
    rl.aura = Aura(mask, 1, pal, intensity=1.2, time=t, flame_height=0.09, turbulence=1.0, flicker=0.6)
    rng = f.rng
    n = 220
    small = f.mask[::4, ::4]
    edge = cv2.dilate(small, np.ones((5, 5), np.uint8)) - small      # embers rise from the outline
    ys, xs = np.nonzero(edge > 128)
    pick = rng.integers(0, len(xs), n)
    pos = np.stack([xs[pick] * 4.0, ys[pick] * 4.0], axis=1) + rng.normal(0, 8 * f.k, (n, 2))
    pos[:, 1] -= rng.uniform(0, 70, n) * f.k
    rl.sprites.append(sprites(SpriteShape.EMBER, pos, rng.uniform(2, 3.8, n) * f.k, np.asarray(pal.spark) * 2.2, 1.0,
                              vel=np.tile([0.0, -160.0 * f.k], (n, 1))))
    head = f.p(f.pose.head)
    for a, b in (((-60, -20), (-95, 140)), ((60, -30), (100, 160))):
        rl.arcs.append(Arc(lightning(rng, (head[0] + a[0] * f.k, head[1] + a[1] * f.k),
                                     (head[0] + b[0] * f.k, head[1] + b[1] * f.k), 5, 0.25), 1.1 * f.k, pal, 1.3, 1.0))
    rl.lights.append(Light(f.p((640, 380)), 420 * f.k, pal.inner, 0.55))
    rl.grade = Grade(darken=0.4, vignette=0.3)
    rl.bloom = 1.2
    return rl, UIDrawList()


def sc_blast(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
    rl = RenderList()
    start = f.p(f.pose.right_hand)
    c = f.p((1040, 330))
    m = 40
    s = np.linspace(0, 1, m)
    wob = np.sin(s * 9.0) * 6 * f.k * (1 - s)
    pts = np.stack([start[0] + (c[0] - start[0]) * s, start[1] + (c[1] - start[1]) * s + wob], axis=1)
    rl.ribbons.append(Ribbon(pts.astype(np.float32), (6 + 26 * s ** 1.5).astype(np.float32) * f.k,
                             (0.3 + 0.9 * s ** 2).astype(np.float32), pal))
    radius = 34 * f.k
    rl.orbs.append(Orb(c, radius, pal, intensity=1.1, charge=0.8, time=t, seed=9.0, swirl=2.0, instability=0.45,
                       flare=0.4, squash=(1.25, 0.85), rotation=-0.25))
    rng = f.rng
    n = 160
    pos = pts[rng.integers(0, m, n)] + rng.normal(0, 10 * f.k, (n, 2))
    rl.sprites.append(sprites(SpriteShape.EMBER, pos, rng.uniform(1.8, 3.2, n) * f.k, np.asarray(pal.spark) * 2.0, 1.0))
    rl.sprites.append(sprites(SpriteShape.SMOKE, pos[:24], rng.uniform(18, 40, 24) * f.k, np.asarray(pal.outer) * 0.3,
                              0.7))
    spos = np.asarray(c) + rng.normal(0, radius * 0.6, (90, 2))
    rl.sprites.append(sprites(SpriteShape.SPARK, spos, rng.uniform(1.2, 2.0, 90) * f.k, np.asarray(pal.spark) * 2.6,
                              1.0, vel=rng.normal(0, 600 * f.k, (90, 2)) + [-500 * f.k, 120 * f.k], stretch=0.02))
    rl.lights.append(Light(c, radius * 7, pal.inner, 1.1))
    rl.grade = Grade(darken=0.3, vignette=0.2)
    return rl, UIDrawList()


def sc_ui(f: Frame, pal: Palette, t: float) -> tuple[RenderList, UIDrawList]:
    rl, _ = orb_scene(0.6)(f, pal, t)
    ui = UIDrawList()
    k = f.k
    accent = (*pal.ui, 1.0)
    # power level panel (gradient fill + border)
    ui.add(UIRect(24 * k, 22 * k, 330 * k, 112 * k, (0.05, 0.08, 0.16, 0.78), 14 * k,
                  border_color=(*pal.ui, 0.85), border_width=2 * k, gradient_to=(0.02, 0.03, 0.07, 0.62)))
    ui.add(UIText("POWER LEVEL", 44 * k, 36 * k, int(20 * k), (0.78, 0.86, 1.0, 0.9), "bold"))
    ui.add(UIText("9,001", 44 * k, 58 * k, int(54 * k), (1.0, 1.0, 1.0, 1.0), "display", glow=10.0,
                  glow_color=(*pal.ui, 0.9)))
    ui.add(UIRect(44 * k, 118 * k, 290 * k, 8 * k, (1, 1, 1, 0.12), 4 * k))
    ui.add(UIRect(44 * k, 118 * k, 290 * k * 0.62, 8 * k, accent, 4 * k, gradient_to=(1, 1, 1, 1)))
    # charge ring around the orb
    lh, rh = f.p(f.pose.left_hand), f.p(f.pose.right_hand)
    c = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2 - 8 * k)
    ui.add(UIArc(c[0], c[1], 118 * k, 0, math.tau, (1, 1, 1, 0.12), 6 * k))
    ui.add(UIArc(c[0], c[1], 118 * k, -math.pi / 2, -math.pi / 2 + math.tau * 0.6, accent, 6 * k))
    ui.add(UIText("60%", c[0], c[1] - 136 * k, int(24 * k), (1, 1, 1, 0.95), "bold", "center", "bottom", shadow=True))
    # ability chips
    for i, (label, active) in enumerate((("1 KI CHARGE", False), ("2 KAMEHAMEHA", True), ("3 BURST", False),
                                         ("4 AURA", False))):
        x = f.w / 2 + (i - 2) * 190 * k + 6 * k
        y = f.h - 64 * k
        ui.add(UIRect(x, y, 178 * k, 40 * k, (0.05, 0.08, 0.15, 0.72) if not active else (*pal.ui, 0.88), 20 * k,
                      border_color=(1, 1, 1, 0.25 if not active else 0.9), border_width=1.5 * k))
        ui.add(UIText(label, x + 89 * k, y + 20 * k, int(19 * k), (1, 1, 1, 0.95) if not active else (0.02, 0.04, 0.1, 1),
                      "bold", "center", "middle"))
    # callout with glow
    ui.add(UIText("KA... ME... HA...", f.w / 2, 120 * k, int(64 * k), (1.0, 0.97, 0.85, 1.0), "display", "center",
                  "middle", glow=14.0, glow_color=(*pal.ui, 0.85), shadow=True))
    # debug skeleton + joints
    pose = f.pose
    skel = [f.p(pose.left_hand), f.p(pose.left_elbow), f.p(pose.left_shoulder), f.p(pose.right_shoulder),
            f.p(pose.right_elbow), f.p(pose.right_hand)]
    ui.add(UILine(skel, (0.3, 1.0, 0.6, 0.8), 3 * k))
    for p in skel:
        ui.add(UICircle(p[0], p[1], 6 * k, (1, 1, 1, 0.9)))
        ui.add(UICircle(p[0], p[1], 10 * k, (0.3, 1.0, 0.6, 0.8), width=2 * k))
    ui.add(UIText("60 FPS  |  OpenGL  |  high", f.w - 20 * k, 20 * k, int(18 * k), (0.8, 0.9, 1.0, 0.85), "mono",
                  "right", "top", shadow=True))
    # notification toast
    ui.add(UIRect(f.w - 360 * k, 60 * k, 340 * k, 46 * k, (0.08, 0.1, 0.12, 0.85), 10 * k,
                  border_color=(1.0, 0.75, 0.2, 0.9), border_width=1.5 * k))
    ui.add(UIText("Calibration saved", f.w - 340 * k, 83 * k, int(20 * k), (1, 0.9, 0.7, 1), "ui", "left", "middle"))
    return rl, ui


SCENARIOS: dict[str, Scenario] = {
    "idle": Scenario("rest", sc_idle),
    "placeholder": Scenario("rest", sc_placeholder, camera=False),
    "orb_charge_20": Scenario("charge", orb_scene(0.2)),
    "orb_charge_60": Scenario("charge", orb_scene(0.6)),
    "orb_charge_100": Scenario("charge", orb_scene(1.0)),
    "beam_lateral": Scenario("beam", sc_beam),
    "beam_frontal": Scenario("frontal", sc_frontal),
    "energy_burst": Scenario("burst", sc_burst),
    "aura": Scenario("rest", sc_aura),
    "ki_blast": Scenario("blast", sc_blast),
    "ui": Scenario("charge", sc_ui),
}


# ----------------------------------------------------------------------- main


def contact_sheet(images: list[tuple[str, np.ndarray]], columns: int = 3, scale: float = 0.5) -> np.ndarray:
    tiles = []
    for name, img in images:
        tile = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(tile, name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * th, columns * tw, 3), np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile
    return sheet


def parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = (int(v) for v in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected WIDTHxHEIGHT, got {text!r}") from exc
    if w < 64 or h < 64:
        raise argparse.ArgumentTypeError("size must be at least 64x64")
    return (w, h)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "previews", help="output folder (default: previews/)")
    parser.add_argument("--background", type=Path, help="background image instead of the procedural room")
    parser.add_argument("--mask", type=Path, help="person mask image (white = person) for the aura scenario")
    parser.add_argument("--size", type=parse_size, default=(1280, 720), help="scene/window size, e.g. 1280x720")
    parser.add_argument("--quality", default="high", choices=("low", "medium", "high", "ultra"))
    parser.add_argument("--palette", default="blue", choices=tuple(PALETTES) + ("all",))
    parser.add_argument("--scenarios", default="all", help="comma separated list: " + ", ".join(SCENARIOS))
    parser.add_argument("--time", type=float, default=1.7, help="animation time in seconds")
    parser.add_argument("--sheet", action="store_true", help="also write contact_sheet.png")
    args = parser.parse_args(argv)

    names = list(SCENARIOS) if args.scenarios == "all" else [s.strip() for s in args.scenarios.split(",") if s.strip()]
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        parser.error("unknown scenario(s): " + ", ".join(unknown))
    w, h = args.size
    background = None
    if args.background is not None:
        img = cv2.imread(str(args.background), cv2.IMREAD_COLOR)
        if img is None:
            parser.error(f"cannot read background image {args.background}")
        background = cover_resize(img, w, h)
    mask = None
    if args.mask is not None:
        m = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            parser.error(f"cannot read mask image {args.mask}")
        mask = cover_resize(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), w, h)[:, :, 0]
    palettes = list(PALETTES) if args.palette == "all" else [args.palette]

    fonts = FontManager()
    try:
        renderer = GLRenderer.create_offscreen((w, h), get_quality(args.quality), fonts)
    except RendererError as exc:
        print(f"OpenGL is not available: {exc}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Renderer: {renderer.info['gl_renderer']} | {renderer.info['gl_version']}")
    view = compute_view_transform(w, h, w, h, "cover")
    written: list[tuple[str, np.ndarray]] = []
    image_id = 0
    try:
        for pal_name in palettes:
            pal = get_palette(pal_name)
            for name in names:
                sc = SCENARIOS[name]
                frame = make_frame(sc.pose, w, h, background, mask)
                rl, ui = sc.build(frame, pal, args.time)
                image_id += 1   # a new id per frame: the renderer skips uploads of unchanged ids
                scene = SceneInput(frame.image if sc.camera else None, image_id, (w, h), args.time)
                t0 = time.perf_counter()
                renderer.render(scene, rl, ui, view)
                renderer.ctx.finish()
                ms = (time.perf_counter() - t0) * 1000.0
                rgb = renderer.read_pixels()
                label = name if len(palettes) == 1 else f"{name}_{pal_name}"
                path = args.out / f"{label}.png"
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(path), bgr)
                written.append((label, bgr))
                print(f"{path}  ({ms:.1f} ms incl. first-use uploads)")
        if args.sheet and written:
            sheet_path = args.out / "contact_sheet.png"
            cv2.imwrite(str(sheet_path), contact_sheet(written))
            print(sheet_path)
    finally:
        renderer.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
