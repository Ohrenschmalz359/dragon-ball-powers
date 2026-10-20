"""CPU fallback renderer (NumPy + OpenCV compositing, pygame UI).

Used when no OpenGL 3.3 context is available. It follows the visual language of the
GPU renderer (docs/ARCHITECTURE.md section 6) in a simplified form:

* Every effect primitive is accumulated additively (linear HDR floats) in a
  low-resolution float32 BGR *effect buffer* of ``window * quality.render_scale * 0.5``
  (at least 160 px wide). Scene coordinates reach that buffer through the
  :class:`ViewTransform`, so effects stay aligned with the camera image in both fit
  modes.
* Particles are splatted vectorised (bilinear mass deposit + one Gaussian blur per
  size bucket; the wider buckets run at 1/2 and 1/4 resolution). Orbs are radial
  profile stamps cached per quantised radius and modulated by swirling polar noise.
  Beams are computed in their own axis-aligned canvas (distance-to-axis /
  along-axis coordinates) and rotated into place. Lightning, ribbons and rings are
  drawn antialiased with OpenCV into 8-bit layers whose glow is blurred. The aura is
  derived from the segmentation mask and eroded into rising flame tongues.
* A 3-level bloom pyramid, a white-hot ``1 - exp(-x)`` tone map and an additive
  composite onto the graded and lit camera image follow. Shake, zoom punch,
  chromatic aberration and screen flashes act on the composited scene; the UI is
  drawn last with pygame onto the output surface.

Per-pixel and per-particle work is vectorised; Python loops only run over
primitives (an orb, a few arcs or ring segments, UI commands).
"""

from __future__ import annotations

import logging
import math
import sys
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import pygame

from dragon_ball_powers.config import EffectsSettings
from dragon_ball_powers.fonts import FontManager
from dragon_ball_powers.interfaces import Renderer, RendererError
from dragon_ball_powers.mathutil import TAU, clamp, hash01
from dragon_ball_powers.primitives import (
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
    QualityPreset,
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
    ViewTransform,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ tuning constants

_TEX = 256                    # size of the tileable noise textures (texels)
_NOISE_CELL = 40.0            # texels per unit of the GPU's noise space (feature size match)
_SHIFT = 3                    # fixed-point bits for OpenCV drawing (1/8 px precision)
_FP = float(1 << _SHIFT)
_FP_LIMIT = 4.0e6             # clamp fixed-point drawing coordinates (keeps int32 safe)
_MIN_EFFECT_WIDTH = 160       # the effect buffer is never narrower than this
_BUCKET_EDGES = np.array([1.25, 3.5], dtype=np.float32)  # sprite sigma bucket edges (effect px)
_BUCKET_LEVELS = (1, 2, 4)    # resolution divisor per sprite bucket
_MAX_STREAK_SAMPLES = 60_000  # cap for spark streak samples per batch
_ORB_EXTENT = 3.2             # orb halo reach in radii
_ORB_DETAIL = 1.6             # plasma / turbulence is computed inside this many radii
_ORB_QUANT = math.log(1.035)  # orb stamp radius quantisation (3.5 % steps)
_ORB_CACHE_BYTES = 64 << 20
_LIGHT_DIV = 8                # the light map is window size / 8
# Scene lighting follows the GPU renderer: lit = cam * (1 + L) + 0.06 * L with a
# 1 / (1 + q^2) falloff (q = distance / radius) that fades out between 2 and 3 radii.
_LIGHT_ADD = 0.06             # additive tint per unit of light (fraction of full scale)
_LIGHT_REACH = 3.0            # light influence ends at this many radii
_BLOOM_WEIGHTS = (0.6, 0.7, 0.8)
_BLOOM_SIGMAS = (1.0, 1.4, 1.8)
_BLOOM_GAIN = 0.35
_BLOOM_THRESHOLD = 0.08       # faint haze does not bloom (keeps the room readable)
_LUMA_BGR = np.array([0.0722, 0.7152, 0.2126], dtype=np.float64)  # Rec. 709 weights, BGR order
# Tone mapping regimes (as in the GPU composite): below _TM_PALE the hue is kept, up to
# _TM_PALE_END it blends into per-channel 1 - exp(-x) (pales), from _TM_WHITE on it turns white-hot.
_TM_PALE = (0.6, 3.0)
_TM_WHITE = (3.0, 9.0)
_TM_WHITE_MAX = 0.85

# shape -> (Gaussian sigma per sprite radius, light energy per radius^2, whitening of the colour).
# The energies are the integrals of the GPU sprite profiles (GLOW exp(-4d^2), EMBER hot dot +
# glow, SMOKE soft disc, SPARK capsule head), so a particle carries the same total light in
# both renderers; "whitening" mimics the GPU's whiter hot cores (mix(rgb, max(rgb), 0.6)).
_SPRITE_SHAPES: dict[int, tuple[float, float, float]] = {
    int(SpriteShape.GLOW): (0.354, 0.785, 0.0),
    int(SpriteShape.SPARK): (0.33, 1.236, 0.40),
    int(SpriteShape.EMBER): (0.262, 0.565, 0.30),
    int(SpriteShape.SMOKE): (0.30, 0.151, 0.0),
}
_SPARK_LINE_ENERGY = 0.637    # SPARK streak light per unit length and radius (incl. the tail ramp)
_SPARK_TAIL = 0.12            # SPARK brightness at the tail end of its streak (head = 1)
_SPRITE_MIN_SIGMA = 0.5       # effect px; smaller sprites are spread wider with the same energy
# Crisp "sparkles": the brightest sparks / embers also get a window-resolution antialiased
# streak (their hot core), the rest of their light is splatted into the effect buffer.
_SPARKLE_MAX = 200            # at most this many per frame (brightest first)
_SPARKLE_MIN_PEAK = 0.35      # dimmer particles are only splatted
# shape -> (share of the light in the hot core, core peak gain relative to the sprite colour)
_SPARKLE_SHAPES: dict[int, tuple[float, float]] = {
    int(SpriteShape.SPARK): (0.65, 1.7),
    int(SpriteShape.EMBER): (0.48, 1.9),
}
_SPARKLE_LEVELS = np.array([0.22, 0.42, 0.66, 1.0], np.float32)   # brightness quantisation
_DEFAULT_PALETTE = PALETTES["blue"]
_STAGES = ("camera", "grade", "effects", "post", "present", "ui", "total")


# ------------------------------------------------------------------------ helpers


def _bgr(color: Sequence[float]) -> np.ndarray:
    """RGB colour -> BGR float32 vector (OpenCV channel order)."""
    return np.array((float(color[2]), float(color[1]), float(color[0])), dtype=np.float32)


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    """Vectorised smoothstep (``edge0 > edge1`` gives a falling edge); float32 in/out."""
    t = (x - np.float32(edge0)) * np.float32(1.0 / (edge1 - edge0))
    np.clip(t, 0.0, 1.0, out=t)
    return t * t * (np.float32(3.0) - np.float32(2.0) * t)


def _finite(*values: Any) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def _num(value: Any, default: float) -> float:
    """``float(value)`` or ``default`` when it is missing / not finite."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _palette(p: Any) -> Palette:
    return p if isinstance(p, Palette) else _DEFAULT_PALETTE


def _f32(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


def _grid(x0: float, n: int) -> np.ndarray:
    """``n`` pixel-centre coordinates starting at pixel ``x0`` (float32)."""
    return np.arange(n, dtype=np.float32) + np.float32(x0 + 0.5)


def _add_colored(region: np.ndarray, layers: Sequence[np.ndarray],
                 colors: Sequence[np.ndarray]) -> None:
    """``region (h, w, 3) += sum(layers[i] * colors[i])`` via ``cv2.transform``.

    One transform handles up to four single-channel layers, which is several times
    faster than NumPy broadcasting into a 3-channel image.
    """
    for start in range(0, len(layers), 4):
        group = [_f32(a) for a in layers[start:start + 4]]
        src = group[0] if len(group) == 1 else cv2.merge(group)
        m = np.stack([np.asarray(c, np.float32) for c in colors[start:start + 4]], axis=1)
        region += cv2.transform(src, m)


def _clip_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int
              ) -> tuple[int, int, int, int] | None:
    """Integer pixel box covering ``[x0, x1) x [y0, y1)`` clipped to ``w x h``."""
    if not _finite(x0, y0, x1, y1):
        return None
    ix0 = max(0, int(math.floor(x0)))
    iy0 = max(0, int(math.floor(y0)))
    ix1 = min(w, int(math.ceil(x1)))
    iy1 = min(h, int(math.ceil(y1)))
    if ix1 <= ix0 or iy1 <= iy0:
        return None
    return ix0, iy0, ix1, iy1


def _fp_point(x: float, y: float) -> tuple[int, int]:
    """Continuous layer coordinates -> OpenCV fixed-point pixel-index coordinates."""
    fx = min(max((x - 0.5) * _FP, -_FP_LIMIT), _FP_LIMIT)
    fy = min(max((y - 0.5) * _FP, -_FP_LIMIT), _FP_LIMIT)
    return (int(round(fx)), int(round(fy)))


def _rgba(color: Any) -> tuple[int, int, int, int]:
    """UI colour (RGBA floats 0..1, alpha optional) -> 8-bit RGBA tuple."""
    try:
        comps = [float(c) for c in color]
    except (TypeError, ValueError):
        return (255, 255, 255, 255)
    if len(comps) < 3:
        return (255, 255, 255, 255)
    if len(comps) == 3:
        comps.append(1.0)
    out = []
    for c in comps[:4]:
        c = c if math.isfinite(c) else 0.0
        out.append(int(round(min(1.0, max(0.0, c)) * 255.0)))
    return (out[0], out[1], out[2], out[3])


class _LRU:
    """Small LRU cache with an optional byte budget."""

    def __init__(self, max_items: int, max_bytes: int = 0) -> None:
        self._data: OrderedDict[Any, tuple[Any, int]] = OrderedDict()
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._bytes = 0

    def get(self, key: Any) -> Any:
        item = self._data.get(key)
        if item is None:
            return None
        self._data.move_to_end(key)
        return item[0]

    def put(self, key: Any, value: Any, nbytes: int = 0) -> None:
        old = self._data.pop(key, None)
        if old is not None:
            self._bytes -= old[1]
        self._data[key] = (value, nbytes)
        self._bytes += nbytes
        while len(self._data) > self._max_items or (
                self._max_bytes and self._bytes > self._max_bytes and len(self._data) > 1):
            _, (_, size) = self._data.popitem(last=False)
            self._bytes -= size

    def clear(self) -> None:
        self._data.clear()
        self._bytes = 0

    def __len__(self) -> int:
        return len(self._data)


class _LogLimiter:
    """Logs each message key at most once per ``interval`` seconds."""

    def __init__(self, interval: float = 10.0) -> None:
        self._interval = interval
        self._last: dict[str, float] = {}

    def __call__(self, key: str, level: int, msg: str, *args: Any, exc_info: bool = False) -> None:
        now = time.perf_counter()
        last = self._last.get(key)
        if last is not None and now - last < self._interval:
            return
        self._last[key] = now
        log.log(level, msg, *args, exc_info=exc_info)


# -------------------------------------------------------------------------- noise


def _fbm_texture(size: int, seed: int, slope: float, low: float = 1.5) -> np.ndarray:
    """Tileable fractal noise in [0, 1]: white noise shaped by a 1/f^slope spectrum.

    Filtering in the frequency domain makes the texture perfectly periodic, so it
    can be sampled with ``BORDER_WRAP`` at any offset (scrolling, rotation).
    """
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft2(rng.standard_normal((size, size)))
    fy = np.fft.fftfreq(size, d=1.0 / size)[:, None]
    fx = np.fft.rfftfreq(size, d=1.0 / size)[None, :]
    f = np.hypot(fx, fy)
    amp = np.zeros_like(f)
    band = f >= low
    amp[band] = f[band] ** -slope
    amp *= np.exp(-(f / (size * 0.3)) ** 2)   # soft high cut: no aliasing when minified
    tex = np.fft.irfft2(spectrum * amp, s=(size, size))
    tex = (tex - tex.mean()) / (4.0 * tex.std() + 1e-12) + 0.5
    return np.clip(tex, 0.0, 1.0).astype(np.float32)


def _mip_chain(tex: np.ndarray, min_size: int = 16) -> list[np.ndarray]:
    """Tileable mip levels: each level is pyrDown of a 3x3 tiling, cropped to the centre tile."""
    levels = [tex]
    while levels[-1].shape[0] // 2 >= min_size:
        prev = levels[-1]
        n = prev.shape[0] // 2
        down = cv2.pyrDown(np.tile(prev, (3, 3)))
        levels.append(np.ascontiguousarray(down[n:2 * n, n:2 * n]))
    return levels


class _NoiseBank:
    """Precomputed tileable noise textures (with mip levels) shared by all effects.

    Lookups pass their texel rate (texels per output pixel); the sampler picks the mip
    level that keeps it near one texel per pixel, so small or distant effects show
    smooth plasma instead of aliased grain.
    """

    def __init__(self) -> None:
        self.cloud = _mip_chain(_fbm_texture(_TEX, 1013, 1.25))
        self.cloud2 = _mip_chain(_fbm_texture(_TEX, 2029, 1.25))
        self.fine = _mip_chain(_fbm_texture(_TEX, 3049, 0.85, low=3.0))

    @staticmethod
    def level(mips: list[np.ndarray], rate: float) -> int:
        if not rate > 1.4:
            return 0
        return int(min(len(mips) - 1, max(0, math.ceil(math.log2(rate / 1.4)))))

    def remap(self, mips: list[np.ndarray], map_x: np.ndarray, map_y: np.ndarray, rate: float) -> np.ndarray:
        """``cv2.remap`` with wrap-around; maps are in level-0 texels."""
        lv = self.level(mips, rate)
        if lv:
            s = np.float32(1.0 / (1 << lv))
            map_x = map_x * s
            map_y = map_y * s
        return cv2.remap(mips[lv], _f32(map_x), _f32(map_y), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    @staticmethod
    def angular(mips: list[np.ndarray], theta: np.ndarray, phase: float) -> np.ndarray:
        """Angle-only noise in [0, 1]: one texture row wrapped once around the circle.

        ``theta`` is the polar angle in turns ([0, 1)); ``phase`` scrolls smoothly through
        the (vertically correlated) rows, so the pattern evolves without jumping.
        """
        tex = mips[0]
        rows, n = tex.shape
        p = (phase % 1.0) * rows
        r0 = int(p) % rows
        f = np.float32(p - math.floor(p))
        row = tex[r0] * (np.float32(1.0) - f) + tex[(r0 + 1) % rows] * f
        out = np.interp(theta, np.arange(n, dtype=np.float32) * np.float32(1.0 / n), row, period=1.0)
        return out.astype(np.float32)

    @staticmethod
    def line(mips: list[np.ndarray], x: np.ndarray, row: float) -> np.ndarray:
        """1D noise along one (wrapping) texture row; ``x`` in texels, ``row`` in [0, 1)."""
        tex = mips[0]
        rows, n = tex.shape
        r = tex[int((row % 1.0) * rows) % rows]
        out = np.interp(np.asarray(x, np.float32) % np.float32(n), np.arange(n, dtype=np.float32), r, period=n)
        return out.astype(np.float32)

    def warp(self, mips: list[np.ndarray], matrix: np.ndarray, size: tuple[int, int], rate: float) -> np.ndarray:
        """Affine lookup (``matrix`` maps output px -> level-0 texels) with wrap-around."""
        lv = self.level(mips, rate)
        m = np.asarray(matrix, np.float32) / np.float32(1 << lv)
        return cv2.warpAffine(mips[lv], m, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_WRAP)


_NOISE: _NoiseBank | None = None


def _noise_bank() -> _NoiseBank:
    global _NOISE
    if _NOISE is None:
        _NOISE = _NoiseBank()
    return _NOISE


# ------------------------------------------------------------------ effect space


@dataclass(frozen=True)
class _Space:
    """Affine map scene px -> effect-buffer px (continuous: pixel i spans [i, i + 1))."""

    ax: float
    bx: float
    ay: float
    by: float
    k: float      # scene length -> effect length
    w: int
    h: int
    w_per_px: float = 1.0   # window px per effect px (x); window = effect / k (no offset)
    h_per_px: float = 1.0

    @classmethod
    def create(cls, view: ViewTransform, win: tuple[int, int], eff: tuple[int, int]) -> "_Space":
        kx = eff[0] / max(1, win[0])
        ky = eff[1] / max(1, win[1])
        s = float(view.scale)
        return cls(s * kx, float(view.offset_x) * kx, s * ky, float(view.offset_y) * ky,
                   s * 0.5 * (kx + ky), eff[0], eff[1], 1.0 / kx, 1.0 / ky)

    def pt(self, p: Any) -> tuple[float, float]:
        return (float(p[0]) * self.ax + self.bx, float(p[1]) * self.ay + self.by)

    def pts(self, arr: Any) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
        out = np.empty_like(a)
        out[:, 0] = a[:, 0] * np.float32(self.ax) + np.float32(self.bx)
        out[:, 1] = a[:, 1] * np.float32(self.ay) + np.float32(self.by)
        return out


# --------------------------------------------------------------- sprite splatting


class _Splatter:
    """Vectorised Gaussian splatting of point masses into the effect buffer.

    Masses are deposited bilinearly (``np.add.at`` for few points, ``np.bincount``
    for many) into one padded accumulation image per size bucket; each bucket is
    blurred once with the Gaussian of its energy-weighted mean size and added to the
    effect buffer. A mass of ``peak * 2 * pi * sigma**2`` produces a blob whose centre
    brightness is ``peak``, so sprites look the same whatever bucket they land in.
    """

    def __init__(self) -> None:
        self._parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self._buffers: dict[int, np.ndarray] = {}

    def add(self, x: np.ndarray, y: np.ndarray, sigma: np.ndarray, mass: np.ndarray) -> None:
        if x.size:
            self._parts.append((_f32(x), _f32(y), _f32(sigma), _f32(mass).reshape(-1, 3)))

    @property
    def pending(self) -> int:
        return sum(p[0].size for p in self._parts)

    def clear(self) -> None:
        self._parts.clear()

    def release(self) -> None:
        self._parts.clear()
        self._buffers.clear()

    def resolve(self, effect: np.ndarray) -> int:
        """Blur and add all pending splats; returns the number of splats drawn."""
        if not self._parts:
            return 0
        x = np.concatenate([p[0] for p in self._parts])
        y = np.concatenate([p[1] for p in self._parts])
        sig = np.concatenate([p[2] for p in self._parts])
        mass = np.concatenate([p[3] for p in self._parts])
        self._parts.clear()
        bucket = np.searchsorted(_BUCKET_EDGES, sig, side="right")
        count = 0
        for b, factor in enumerate(_BUCKET_LEVELS):
            sel = np.flatnonzero(bucket == b)
            if sel.size:
                count += self._resolve_bucket(effect, b, factor, x[sel], y[sel], sig[sel], mass[sel])
        return count

    def _buffer(self, b: int, hp: int, wp: int) -> np.ndarray:
        buf = self._buffers.get(b)
        if buf is None or buf.shape[0] != hp or buf.shape[1] != wp:
            buf = np.zeros((hp, wp, 3), np.float32)
            self._buffers[b] = buf
        else:
            buf.fill(0.0)
        return buf

    def _resolve_bucket(self, effect: np.ndarray, b: int, factor: int, x: np.ndarray,
                        y: np.ndarray, sig: np.ndarray, mass: np.ndarray) -> int:
        h, w = effect.shape[:2]
        wl = max(1, -(-w // factor))
        hl = max(1, -(-h // factor))
        rx, ry = wl / w, hl / h
        weight = mass.sum(axis=1)
        wsum = float(weight.sum())
        if not wsum > 1e-9:
            return 0
        sigma_b = float(np.dot(sig, weight)) / wsum * 0.5 * (rx + ry)
        sigma_b = min(max(sigma_b, 0.55), 8.0)
        m = int(math.ceil(3.0 * sigma_b)) + 2         # padding: blobs centred off-buffer still glow in
        wp, hp = wl + 2 * m, hl + 2 * m
        px = x * np.float32(rx) + np.float32(m - 0.5)  # continuous -> pixel-index coordinates
        py = y * np.float32(ry) + np.float32(m - 0.5)
        fx0 = np.floor(px)
        fy0 = np.floor(py)
        keep = (fx0 >= 0) & (fy0 >= 0) & (fx0 < wp - 1) & (fy0 < hp - 1)
        if not keep.all():
            px, py, fx0, fy0, mass = px[keep], py[keep], fx0[keep], fy0[keep], mass[keep]
        n = int(px.size)
        if n == 0:
            return 0
        fx = px - fx0
        fy = py - fy0
        gx = np.float32(1.0) - fx
        gy = np.float32(1.0) - fy
        ix = fx0.astype(np.intp)
        iy = fy0.astype(np.intp)
        base = iy * wp + ix
        idx = np.concatenate((base, base + 1, base + wp, base + wp + 1))
        wts = np.concatenate((gx * gy, fx * gy, gx * fy, fx * fy))
        vals = np.tile(mass * np.float32(rx * ry), (4, 1))
        vals *= wts[:, None]
        acc = self._buffer(b, hp, wp)
        flat = acc.reshape(-1, 3)
        if idx.size * 16 < flat.shape[0]:
            np.add.at(flat, idx, vals)
        else:
            # one bincount over interleaved channel indices, restricted to the touched range
            lo, hi = int(idx.min()), int(idx.max()) + 1
            chan = ((idx - lo) * 3)[:, None] + np.arange(3)[None, :]
            hist = np.bincount(chan.ravel(), weights=vals.ravel(), minlength=(hi - lo) * 3)
            flat[lo:hi] += hist.reshape(-1, 3).astype(np.float32)
        # blur only the part of the bucket that holds splats
        y0 = max(int(iy.min()) - m, 0)
        y1 = min(int(iy.max()) + 2 + m, hp)
        x0 = max(int(ix.min()) - m, 0)
        x1 = min(int(ix.max()) + 2 + m, wp)
        region = acc[y0:y1, x0:x1]
        region[...] = cv2.GaussianBlur(region, (0, 0), sigma_b, borderType=cv2.BORDER_CONSTANT)
        inner = acc[m:m + hl, m:m + wl]
        if factor == 1:
            ry0, ry1 = max(y0 - m, 0), min(y1 - m, hl)
            rx0, rx1 = max(x0 - m, 0), min(x1 - m, wl)
            if ry1 > ry0 and rx1 > rx0:
                effect[ry0:ry1, rx0:rx1] += inner[ry0:ry1, rx0:rx1]
        else:
            effect += cv2.resize(inner, (w, h), interpolation=cv2.INTER_LINEAR)
        return n


# ----------------------------------------------------------------------- orb stamp


@dataclass
class _OrbStamp:
    """Static per-radius orb data (radius in effect px).

    ``halo`` covers the whole stamp (wide outer glow, already faded out at the stamp
    border); everything that moves (plasma, silhouette wobble, crackle) is evaluated per
    frame inside the centred detail region from the cached polar coordinates.
    """

    radius: float
    half: int                 # stamp is (2 * half + 1)^2 px, centre pixel [half, half]
    dhalf: int                # half size of the centred detail region
    halo: np.ndarray          # (n, n) outer halo weight (palette outer colour)
    rn: np.ndarray            # (m, m) detail region: distance / radius
    theta: np.ndarray         # (m, m) polar angle / 2 pi in [0, 1)
    twist: np.ndarray         # (m, m) spiral twist weight, 1 at the centre .. 0 at the surface
    v1: np.ndarray            # radial noise coordinates (texels) of the two plasma layers
    v2: np.ndarray

    @property
    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in (self.halo, self.rn, self.theta, self.twist, self.v1, self.v2)))


def _build_orb_stamp(radius: float) -> _OrbStamp:
    half = int(math.ceil(radius * _ORB_EXTENT)) + 2
    dhalf = min(half, int(math.ceil(radius * _ORB_DETAIL)) + 2)
    coords = np.arange(-half, half + 1, dtype=np.float32)
    n = coords.size
    xx = np.ascontiguousarray(np.broadcast_to(coords[None, :], (n, n)))
    yy = np.ascontiguousarray(np.broadcast_to(coords[:, None], (n, n)))
    inv_r = np.float32(1.0 / radius)
    rn_full = cv2.magnitude(xx, yy) * inv_r
    # outer halo of the GPU orb: exp(-2.1 * dout) * 0.55 * (1 - 0.6 * sphere), faded at the border
    dout = np.maximum(rn_full - np.float32(1.0), 0.0)
    sphere = _smoothstep(1.01, 0.93, rn_full)
    halo = np.exp(dout * np.float32(-2.1)) * np.float32(0.55) * (np.float32(1.0) - np.float32(0.6) * sphere)
    halo *= _smoothstep(_ORB_EXTENT, 0.62 * _ORB_EXTENT, rn_full)
    s, e = half - dhalf, half + dhalf + 1
    rd, ang = cv2.cartToPolar(np.ascontiguousarray(xx[s:e, s:e]), np.ascontiguousarray(yy[s:e, s:e]))
    rn = rd * inv_r
    theta = ang * np.float32(1.0 / TAU)
    # The angular offset (twist) grows towards the centre, so rotating noise forms spiral
    # arms. The radial coordinate is "projected onto the sphere" (as on the GPU) so plasma
    # features bunch up towards the rim.
    twist = np.clip(np.float32(1.0) - rn, 0.0, 1.0)
    z = np.sqrt(np.clip(np.float32(1.0) - rn * rn, 0.0, 1.0))
    proj = rn / (np.float32(0.62) + np.float32(0.6) * z)
    v1 = proj * np.float32(0.55 * _TEX)
    v2 = proj * np.float32(0.95 * _TEX)
    return _OrbStamp(float(radius), half, dhalf, _f32(halo), _f32(rn), _f32(theta),
                     _f32(twist), _f32(v1), _f32(v2))


# ------------------------------------------------------------------- line layers


class _LineCanvas:
    """Two 8-bit antialiased line layers (core + glow) over a box of the effect buffer.

    OpenCV only antialiases 8-bit images, so lines are rasterised here with
    ``LINE_AA`` (values relative to the group's brightest item) and converted to
    float when they are colourised and added to the HDR buffer.
    """

    def __init__(self, box: tuple[int, int, int, int]) -> None:
        self.x0, self.y0, self.x1, self.y1 = box
        shape = (self.y1 - self.y0, self.x1 - self.x0)
        self.core = np.zeros(shape, np.uint8)
        self.glow = np.zeros(shape, np.uint8)

    def fp(self, pts: np.ndarray) -> np.ndarray:
        """Continuous effect coordinates (N, 2) -> int32 fixed-point layer coordinates."""
        local = (pts - np.array([self.x0 + 0.5, self.y0 + 0.5], np.float32)) * np.float32(_FP)
        np.clip(local, -_FP_LIMIT, _FP_LIMIT, out=local)
        return np.round(local).astype(np.int32)

    def local(self, x: float, y: float) -> tuple[int, int]:
        return _fp_point(x - self.x0, y - self.y0)

    def composite(self, effect: np.ndarray, core_color: np.ndarray, glow_color: np.ndarray,
                  glow_sigma: float, scale: float) -> None:
        k = np.float32(scale / 255.0)
        core = self.core.astype(np.float32)
        core *= k
        glow = _soft_blur(self.glow.astype(np.float32), max(0.6, glow_sigma))
        glow *= k
        _add_colored(effect[self.y0:self.y1, self.x0:self.x1], [core, glow], [core_color, glow_color])


def _soft_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; wide kernels run at half resolution (4-6x cheaper, visually identical)."""
    h, w = img.shape[:2]
    if sigma < 3.0 or h < 8 or w < 8:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.resize(img, ((w + 1) // 2, (h + 1) // 2), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma * 0.5)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _group_by_palette(items: Sequence[Any]) -> list[tuple[Palette, list[Any]]]:
    groups: dict[int, tuple[Palette, list[Any]]] = {}
    for item in items:
        pal = _palette(getattr(item, "palette", None))
        groups.setdefault(id(pal), (pal, []))[1].append(item)
    return list(groups.values())


# ------------------------------------------------------------------------ grading


def _grade_matrix(grade: Grade) -> np.ndarray | None:
    """3x3 BGR colour matrix for darken / desaturate / tint, or None for the identity."""
    darken = clamp(_num(grade.darken, 0.0))
    desat = clamp(_num(grade.desaturate, 0.0))
    tint_strength = clamp(_num(grade.tint_strength, 0.0))
    if darken < 1e-3 and desat < 1e-3 and tint_strength < 1e-3:
        return None
    m = (1.0 - desat) * np.eye(3) + desat * np.outer(np.ones(3), _LUMA_BGR)
    try:
        tint = np.clip(_bgr(grade.tint).astype(np.float64), 0.0, 2.0)
    except (TypeError, ValueError, IndexError):
        tint = np.ones(3)
    m = np.diag(1.0 + tint_strength * (tint - 1.0)) @ m
    return (m * (1.0 - darken)).astype(np.float32)


# ----------------------------------------------------------------------- UI painter


def _surface_from_bgra(arr: np.ndarray) -> pygame.Surface:
    """(h, w, 4) uint8 BGRA -> independent SRCALPHA surface (ARGB8888, the display's layout)."""
    h, w = arr.shape[:2]
    return pygame.image.frombuffer(np.ascontiguousarray(arr).tobytes(), (w, h), "BGRA").copy()


def _bgra_image(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Float RGB (broadcastable to (h, w, 3), 0..255) + alpha (h, w) 0..1 -> uint8 BGRA."""
    h, w = alpha.shape
    out = np.empty((h, w, 4), np.uint8)
    rgb = np.broadcast_to(rgb, (h, w, 3))
    out[..., 0] = np.clip(rgb[..., 2] + 0.5, 0, 255)
    out[..., 1] = np.clip(rgb[..., 1] + 0.5, 0, 255)
    out[..., 2] = np.clip(rgb[..., 0] + 0.5, 0, 255)
    out[..., 3] = np.clip(alpha * 255.0 + 0.5, 0, 255)
    return out


class _UIPainter:
    """Draws a :class:`UIDrawList` with pygame (window pixels).

    Shapes with alpha, rounded corners or gradients are rasterised once with
    analytic antialiasing (signed distance coverage) into SRCALPHA sprites and cached
    (LRU); text is tinted / shadowed / glowed once per distinct style and cached.
    """

    def __init__(self, fonts: FontManager) -> None:
        self.fonts = fonts
        self._text = _LRU(768)
        self._shapes = _LRU(384)
        self._limiter = _LogLimiter()
        self._unknown: set[str] = set()

    def clear(self) -> None:
        self._text.clear()
        self._shapes.clear()

    def draw(self, surface: pygame.Surface, ui: UIDrawList | None) -> None:
        commands = getattr(ui, "commands", None) if ui is not None else None
        if not commands:
            return
        for cmd in commands:
            try:
                if isinstance(cmd, UIText):
                    self._draw_text(surface, cmd)
                elif isinstance(cmd, UIRect):
                    self._draw_rect(surface, cmd)
                elif isinstance(cmd, UICircle):
                    self._draw_circle(surface, cmd)
                elif isinstance(cmd, UIArc):
                    self._draw_arc(surface, cmd)
                elif isinstance(cmd, UILine):
                    self._draw_line(surface, cmd)
                else:
                    name = type(cmd).__name__
                    if name not in self._unknown:
                        self._unknown.add(name)
                        log.warning("Ignoring unknown UI command type %s", name)
            except Exception:  # noqa: BLE001 - one broken command must not kill the frame
                self._limiter(f"ui:{type(cmd).__name__}", logging.ERROR,
                              "Drawing UI command %s failed", type(cmd).__name__, exc_info=True)

    # ------------------------------------------------------------------ rects

    def _draw_rect(self, surface: pygame.Surface, cmd: UIRect) -> None:
        if not _finite(cmd.x, cmd.y, cmd.w, cmd.h):
            return
        x0, y0 = int(round(cmd.x)), int(round(cmd.y))
        w, h = int(round(cmd.x + cmd.w)) - x0, int(round(cmd.y + cmd.h)) - y0
        if w <= 0 or h <= 0:
            return
        fill = _rgba(cmd.color)
        grad = _rgba(cmd.gradient_to) if cmd.gradient_to is not None else None
        bw = int(round(_num(cmd.border_width, 0.0))) if cmd.border_color is not None else 0
        border = _rgba(cmd.border_color) if bw > 0 else None
        if border is not None and border[3] == 0:
            border, bw = None, 0
        if grad is not None and fill[3] == 0 and grad[3] == 0:
            grad = None
        if grad is None and fill[3] == 0 and border is None:
            return
        radius = int(round(max(0.0, min(_num(cmd.radius, 0.0), w / 2.0, h / 2.0))))
        if radius == 0 and grad is None and fill[3] in (0, 255) and (border is None or border[3] == 255):
            if fill[3] == 255:          # opaque square rect: draw straight onto the target
                pygame.draw.rect(surface, fill, (x0, y0, w, h))
            if border is not None:
                pygame.draw.rect(surface, border, (x0, y0, w, h), width=min(bw, (min(w, h) + 1) // 2))
            return
        key = ("rect", w, h, radius, fill, grad, border, bw)
        sprite = self._shapes.get(key)
        if sprite is None:
            sprite = self._rect_sprite(w, h, radius, fill, grad, border, bw)
            self._shapes.put(key, sprite)
        surface.blit(sprite, (x0, y0))

    @staticmethod
    def _rect_sprite(w: int, h: int, r: int, fill: tuple[int, int, int, int],
                     grad: tuple[int, int, int, int] | None,
                     border: tuple[int, int, int, int] | None, bw: int) -> pygame.Surface:
        # signed distance to the rounded rectangle at pixel centres (negative inside)
        xs = np.arange(w, dtype=np.float32) + np.float32(0.5 - w / 2.0)
        ys = np.arange(h, dtype=np.float32) + np.float32(0.5 - h / 2.0)
        qx = np.abs(xs) - np.float32(w / 2.0 - r)
        qy = np.abs(ys) - np.float32(h / 2.0 - r)
        mx = np.maximum(qx, 0.0)[None, :]
        my = np.maximum(qy, 0.0)[:, None]
        dist = np.sqrt(mx * mx + my * my) + np.minimum(np.maximum(qx[None, :], qy[:, None]), 0.0) - r
        cover = np.clip(0.5 - dist, 0.0, 1.0)
        if grad is not None:     # vertical gradient: color at the top -> gradient_to at the bottom
            t = (np.arange(h, dtype=np.float32) / max(1, h - 1))[:, None]
            top = np.asarray(fill, np.float32)
            bottom = np.asarray(grad, np.float32)
            col = top[None, :] * (1.0 - t) + bottom[None, :] * t          # (h, 4)
            fill_rgb = col[:, None, :3]
            fill_a = cover * (col[:, 3:4] / 255.0)
        else:
            fill_rgb = np.asarray(fill[:3], np.float32)
            fill_a = cover * (fill[3] / 255.0)
        if border is not None:
            inner = np.clip(0.5 - (dist + bw), 0.0, 1.0)
            ba = np.clip(cover - inner, 0.0, 1.0) * (border[3] / 255.0)
            out_a = ba + fill_a * (1.0 - ba)
            brgb = np.asarray(border[:3], np.float32)
            num = brgb * ba[..., None] + fill_rgb * (fill_a * (1.0 - ba))[..., None]
            rgb = num / np.maximum(out_a, 1e-6)[..., None]
            return _surface_from_bgra(_bgra_image(rgb, out_a))
        return _surface_from_bgra(_bgra_image(fill_rgb, fill_a))

    # -------------------------------------------------------------- circles / arcs

    def _draw_circle(self, surface: pygame.Surface, cmd: UICircle) -> None:
        if not _finite(cmd.x, cmd.y, cmd.radius):
            return
        col = _rgba(cmd.color)
        radius = round(float(cmd.radius) * 4.0) / 4.0
        width = round(max(0.0, _num(cmd.width, 0.0)) * 4.0) / 4.0
        if radius <= 0.0 or col[3] == 0:
            return
        key = ("circle", radius, width, col)
        entry = self._shapes.get(key)
        if entry is None:
            entry = self._ring_sprite(radius, width, col, None, None)
            self._shapes.put(key, entry)
        sprite, c = entry
        surface.blit(sprite, (int(round(cmd.x)) - c, int(round(cmd.y)) - c))

    def _draw_arc(self, surface: pygame.Surface, cmd: UIArc) -> None:
        if not _finite(cmd.x, cmd.y, cmd.radius, cmd.start, cmd.end):
            return
        col = _rgba(cmd.color)
        radius = round(float(cmd.radius) * 4.0) / 4.0
        width = round(max(0.5, _num(cmd.width, 4.0)) * 4.0) / 4.0
        start, end = float(cmd.start), float(cmd.end)
        if end < start:
            start, end = end, start
        if radius <= 0.0 or col[3] == 0 or end - start < 1e-3:
            return
        if end - start >= TAU - 1e-4:
            key = ("circle", radius, width, col)
            arc_range = None
        else:
            q = 4.0 * 180.0 / math.pi                      # quantise to 0.25 degree
            start_q = round((start % TAU) * q) / q
            span_q = round((end - start) * q) / q
            key = ("arc", radius, width, col, start_q, span_q)
            arc_range = (start_q, span_q)
        entry = self._shapes.get(key)
        if entry is None:
            entry = self._ring_sprite(radius, width, col, *(arc_range or (None, None)))
            self._shapes.put(key, entry)
        sprite, c = entry
        surface.blit(sprite, (int(round(cmd.x)) - c, int(round(cmd.y)) - c))

    @staticmethod
    def _ring_sprite(radius: float, width: float, col: tuple[int, int, int, int],
                     start: float | None, span: float | None) -> tuple[pygame.Surface, int]:
        """Antialiased disc (width 0), ring (stroke centred on ``radius``) or arc with round caps."""
        reach = radius + (width * 0.5 if width > 0 else 0.0) + 1.5
        c = int(math.ceil(reach))
        n = 2 * c
        coords = np.arange(n, dtype=np.float32) + np.float32(0.5 - c)
        xx = np.ascontiguousarray(np.broadcast_to(coords[None, :], (n, n)))
        yy = np.ascontiguousarray(np.broadcast_to(coords[:, None], (n, n)))
        dist, ang = cv2.cartToPolar(xx, yy)            # angle grows clockwise on screen (y down)
        if width <= 0.0:
            cover = np.clip(radius + 0.5 - dist, 0.0, 1.0)
        else:
            cover = np.clip(width * 0.5 + 0.5 - np.abs(dist - radius), 0.0, 1.0)
            if start is not None and span is not None:
                rel = np.mod(ang - np.float32(start), np.float32(TAU))
                cover *= (rel <= span).astype(np.float32)
                for a in (start, start + span):        # round caps
                    px, py = radius * math.cos(a), radius * math.sin(a)
                    d = cv2.magnitude(xx - np.float32(px), yy - np.float32(py))
                    np.maximum(cover, np.clip(width * 0.5 + 0.5 - d, 0.0, 1.0), out=cover)
        alpha = cover * (col[3] / 255.0)
        sprite = _surface_from_bgra(_bgra_image(np.asarray(col[:3], np.float32), alpha))
        return sprite, c

    # ------------------------------------------------------------------- lines

    def _draw_line(self, surface: pygame.Surface, cmd: UILine) -> None:
        pts: list[tuple[float, float]] = []
        for p in cmd.points or ():
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                pts.append((x, y))
        if len(pts) < 2:
            return
        col = _rgba(cmd.color)
        if col[3] == 0:
            return
        width = max(0.5, _num(cmd.width, 1.0))
        rgb = col[:3]
        if col[3] == 255:
            self._stroke(surface, rgb, pts, width, bool(cmd.closed))
            return
        # translucent: draw opaque into a temporary surface, then blit it with surface alpha
        pad = int(math.ceil(width)) + 2
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0, y0 = int(math.floor(min(xs))) - pad, int(math.floor(min(ys))) - pad
        x1, y1 = int(math.ceil(max(xs))) + pad, int(math.ceil(max(ys))) + pad
        clip = surface.get_rect().clip(pygame.Rect(x0, y0, x1 - x0, y1 - y0))
        if clip.width <= 0 or clip.height <= 0:
            return
        tmp = pygame.Surface((x1 - x0, y1 - y0), pygame.SRCALPHA)
        tmp.fill((rgb[0], rgb[1], rgb[2], 0))
        self._stroke(tmp, rgb, [(x - x0, y - y0) for x, y in pts], width, bool(cmd.closed))
        tmp.set_alpha(col[3])
        surface.blit(tmp, (x0, y0))

    @staticmethod
    def _stroke(surface: pygame.Surface, rgb: tuple[int, int, int], pts: list[tuple[float, float]],
                width: float, closed: bool) -> None:
        if width <= 1.5:
            pygame.draw.aalines(surface, rgb, closed, pts)
            return
        wi = max(1, int(round(width)))
        pygame.draw.lines(surface, rgb, closed, pts, wi)
        if wi >= 3:                                      # round joins / caps
            for p in pts:
                pygame.draw.circle(surface, rgb, (int(round(p[0])), int(round(p[1]))), wi // 2)

    # -------------------------------------------------------------------- text

    def _draw_text(self, surface: pygame.Surface, cmd: UIText) -> None:
        text = "" if cmd.text is None else str(cmd.text)
        if not text.strip() or not _finite(cmd.x, cmd.y):
            return
        color = _rgba(cmd.color)
        if color[3] == 0:
            return
        size = int(clamp(round(_num(cmd.size, 16.0)), 6, 400))
        glow = round(max(0.0, _num(cmd.glow, 0.0)), 1)
        gcol = _rgba(cmd.glow_color) if (glow > 0 and cmd.glow_color is not None) else color
        kind = cmd.font if isinstance(cmd.font, str) else "ui"
        key = (text, size, kind, color, glow, gcol if glow > 0 else None, bool(cmd.shadow))
        entry = self._text.get(key)
        if entry is None:
            entry = self._build_text(text, size, kind, color, glow, gcol, bool(cmd.shadow))
            self._text.put(key, entry)
        sprite, pad, tw, th = entry
        x, y = float(cmd.x), float(cmd.y)
        if cmd.align == "center":
            x -= tw / 2.0
        elif cmd.align == "right":
            x -= tw
        if cmd.valign == "middle":
            y -= th / 2.0
        elif cmd.valign == "bottom":
            y -= th
        surface.blit(sprite, (int(round(x)) - pad, int(round(y)) - pad))

    def _white_text(self, text: str, size: int, kind: str) -> pygame.Surface:
        if "\n" not in text:
            return self.fonts.render(text, size, kind)
        lines = [self.fonts.render(line if line else " ", size, kind) for line in text.split("\n")]
        step = self.fonts.line_height(size, kind)
        width = max(s.get_width() for s in lines)
        height = step * (len(lines) - 1) + lines[-1].get_height()
        out = pygame.Surface((max(1, width), max(1, height)), pygame.SRCALPHA)
        for i, line in enumerate(lines):
            out.blit(line, (0, i * step))
        return out

    def _build_text(self, text: str, size: int, kind: str, color: tuple[int, int, int, int],
                    glow: float, gcol: tuple[int, int, int, int], shadow: bool
                    ) -> tuple[pygame.Surface, int, int, int]:
        white = self._white_text(text, size, kind)
        tw, th = white.get_size()
        pad = int(math.ceil(glow * 2.0)) + 2 if glow > 0 else 0
        off = max(1, size // 18) if shadow else 0
        out = pygame.Surface((tw + 2 * pad + off, th + 2 * pad + off), pygame.SRCALPHA)
        if glow > 0 and gcol[3] > 0:
            halo = pygame.Surface(out.get_size(), pygame.SRCALPHA)
            # keep RGB = glow colour everywhere so blurring only spreads alpha (no dark fringe)
            halo.fill((gcol[0], gcol[1], gcol[2], 0))
            src = white.copy()
            src.fill(gcol, special_flags=pygame.BLEND_RGBA_MULT)
            halo.blit(src, (pad, pad), special_flags=pygame.BLEND_RGBA_MAX)
            halo = pygame.transform.gaussian_blur(halo, max(1, int(round(glow))), repeat_edge_pixels=False)
            out.blit(halo, (0, 0))
            out.blit(halo, (0, 0))                       # twice: a denser, more luminous glow
        if shadow:
            shade = white.copy()
            shade.fill((0, 0, 0, int(color[3] * 0.62)), special_flags=pygame.BLEND_RGBA_MULT)
            out.blit(shade, (pad + off, pad + off))
        tinted = white.copy()
        tinted.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
        out.blit(tinted, (pad, pad))
        return out, pad, tw, th


# ------------------------------------------------------------------------ renderer


class CpuRenderer(Renderer):
    """NumPy/OpenCV fallback renderer (see the module docstring for the pipeline).

    ``target`` is the surface to draw into (normally the pygame display surface; the
    app calls ``pygame.display.flip()``). Without a target the frame is drawn into an
    internal 32-bit surface. :meth:`read_pixels` returns the last frame including UI.
    """

    name = "CPU"

    def __init__(self, size: tuple[int, int], quality: QualityPreset, fonts: FontManager,
                 settings: EffectsSettings | None = None, target: pygame.Surface | None = None) -> None:
        self._fonts = fonts
        self._settings = settings
        self._quality = quality
        self._size = (max(1, int(size[0])), max(1, int(size[1])))
        self._target: pygame.Surface | None = None
        self._target_is_display = False
        self._released = False
        self._noise = _noise_bank()
        self._splatter = _Splatter()
        self._orb_cache = _LRU(24, _ORB_CACHE_BYTES)
        self._ui = _UIPainter(fonts)
        self._limiter = _LogLimiter()
        self._stats: dict[str, float] = {}
        self._frames = 0
        self._surface: pygame.Surface | None = None
        self._last_surface: pygame.Surface | None = None
        self._effect: np.ndarray | None = None
        self._frame: np.ndarray | None = None
        self._frame2: np.ndarray | None = None
        self._tmp: np.ndarray | None = None
        self._cam: np.ndarray | None = None
        self._cam_key: Any = None
        self._ph: np.ndarray | None = None
        self._ph_base: np.ndarray | None = None
        self._vignette: dict[tuple[int, int], np.ndarray] = {}
        self._aura_key: Any = None
        self._aura_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._space: _Space | None = None
        self._sprite_xyr: tuple[np.ndarray, np.ndarray, np.ndarray] = (np.zeros(0, np.float32),) * 3
        self._sparkles: list[tuple[np.ndarray, ...]] = []
        self._sparkle_layer: np.ndarray | None = None
        self._eff_size = (1, 1)
        self._configure()
        self.set_target(target)
        log.info("CPU renderer ready: %dx%d, effect buffer %dx%d, quality %s",
                 self._size[0], self._size[1], self._eff_size[0], self._eff_size[1],
                 getattr(quality, "name", "?"))

    # ----------------------------------------------------------------- public API

    @property
    def size(self) -> tuple[int, int]:
        """Output (window) size in pixels."""
        return self._size

    @property
    def effect_size(self) -> tuple[int, int]:
        """Resolution of the internal effect buffer."""
        return self._eff_size

    @property
    def quality(self) -> QualityPreset:
        return self._quality

    @property
    def stats(self) -> dict[str, float]:
        """Smoothed per-stage timings in ms (``camera``, ``grade``, ``effects``, ``post``,
        ``present``, ``ui``, ``total``) plus ``last_total`` and ``frames``."""
        return dict(self._stats)

    @property
    def info(self) -> dict[str, str]:
        return {
            "renderer": self.name,
            "size": f"{self._size[0]}x{self._size[1]}",
            "effect_buffer": f"{self._eff_size[0]}x{self._eff_size[1]}",
            "quality": str(getattr(self._quality, "name", "")),
        }

    def set_target(self, target: pygame.Surface | None) -> None:
        """Draw into ``target`` (e.g. a new display surface after ``set_mode``) or internally."""
        self._target = target
        self._target_is_display = False
        if target is not None:
            try:
                self._target_is_display = bool(pygame.display.get_init()) and \
                    target is pygame.display.get_surface()
            except pygame.error:
                self._target_is_display = False

    def apply_settings(self, settings: EffectsSettings | None) -> None:
        """Use new effect settings (only ``bloom_strength`` is read by the renderer)."""
        self._settings = settings

    def resize(self, width: int, height: int) -> None:
        size = (max(1, int(width)), max(1, int(height)))
        if size == self._size:
            return
        self._size = size
        self._configure()
        log.debug("CPU renderer resized to %dx%d (effect buffer %dx%d)", size[0], size[1],
                  self._eff_size[0], self._eff_size[1])

    def set_quality(self, preset: QualityPreset) -> None:
        self._quality = preset
        self._configure()
        log.debug("CPU renderer quality %s (effect buffer %dx%d)", getattr(preset, "name", "?"),
                  self._eff_size[0], self._eff_size[1])

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._splatter.release()
        self._orb_cache.clear()
        self._ui.clear()
        self._vignette.clear()
        self._effect = self._frame = self._frame2 = self._tmp = self._cam = self._ph = self._ph_base = None
        self._sparkle_layer = None
        self._sparkles.clear()
        self._aura_data = None
        self._aura_key = self._cam_key = None
        self._surface = self._last_surface = None
        self._target = None
        log.debug("CPU renderer released")

    def read_pixels(self) -> np.ndarray:
        surface = self._last_surface
        if surface is None:
            return np.zeros((self._size[1], self._size[0], 3), np.uint8)
        try:
            data = pygame.image.tobytes(surface, "RGB")
        except pygame.error as exc:
            raise RendererError(f"Could not read back the frame: {exc}") from exc
        w, h = surface.get_size()
        return np.frombuffer(data, np.uint8).reshape(h, w, 3).copy()

    def render(self, scene: SceneInput, effects: RenderList, ui: UIDrawList,
               view: ViewTransform) -> None:
        if self._released:
            raise RendererError("CpuRenderer.render() called after release()")
        t0 = time.perf_counter()
        target = self._resolve_target()
        if target is not None and target.get_size() != self._size:
            log.info("CPU renderer follows target surface size %s", target.get_size())
            self.resize(*target.get_size())
        if effects is None:
            effects = RenderList()
        if (view.win_w, view.win_h) != self._size:
            self._limiter("view-size", logging.DEBUG, "View transform is for %dx%d, output is %dx%d",
                          view.win_w, view.win_h, self._size[0], self._size[1])
        frame = self._frame_buffer()
        base = self._camera_layer(scene, view)
        t1 = time.perf_counter()
        self._grade_camera(base, frame, effects, view)
        t2 = time.perf_counter()
        overlay = self._render_effects(effects, view)
        if overlay is not None:
            self._composite(frame, overlay)
        if self._sparkles:
            self._guard("sparkles", self._draw_sparkles, frame)
            self._sparkles.clear()
        t3 = time.perf_counter()
        frame = self._post(frame, effects, view)
        t4 = time.perf_counter()
        surface = self._present(frame, target)
        t5 = time.perf_counter()
        self._ui.draw(surface, ui)
        t6 = time.perf_counter()
        self._last_surface = surface
        self._record((t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0))

    # ------------------------------------------------------------- buffers/setup

    def _configure(self) -> None:
        """(Re)derive size dependent state after a resize or quality change."""
        w, h = self._size
        scale = _num(getattr(self._quality, "render_scale", 1.0), 1.0)
        ew = max(_MIN_EFFECT_WIDTH, int(round(w * max(0.05, scale) * 0.5)))
        eh = max(1, int(round(h * ew / w)))
        self._eff_size = (ew, eh)
        self._effect = None
        self._frame = self._frame2 = self._tmp = None
        self._sparkle_layer = None
        self._cam = None
        self._cam_key = None
        self._ph = self._ph_base = None
        self._vignette.clear()
        self._aura_key = None
        self._aura_data = None
        self._splatter.release()
        if self._surface is not None and self._surface.get_size() != self._size:
            self._surface = None

    def _frame_buffer(self) -> np.ndarray:
        w, h = self._size
        if self._frame is None or self._frame.shape[:2] != (h, w):
            self._frame = np.zeros((h, w, 3), np.uint8)
        return self._frame

    def _effect_buffer(self) -> np.ndarray:
        ew, eh = self._eff_size
        if self._effect is None or self._effect.shape[:2] != (eh, ew):
            self._effect = np.zeros((eh, ew, 3), np.float32)
        else:
            self._effect.fill(0.0)
        return self._effect

    def _record(self, durations: Sequence[float]) -> None:
        self._frames += 1
        alpha = 1.0 if self._frames == 1 else 0.1
        for name, sec in zip(_STAGES, durations):
            ms = sec * 1000.0
            self._stats[name] = ms if alpha == 1.0 else self._stats.get(name, ms) * (1 - alpha) + ms * alpha
        self._stats["last_total"] = durations[-1] * 1000.0
        self._stats["frames"] = float(self._frames)

    def _guard(self, key: str, fn: Callable[..., Any], *args: Any) -> Any:
        """Run one primitive's drawing; a failure is logged (rate-limited), never raised."""
        try:
            return fn(*args)
        except Exception:  # noqa: BLE001 - a broken primitive must not kill the frame
            self._limiter(f"fx:{key}", logging.ERROR, "CPU renderer: drawing %s failed", key,
                          exc_info=True)
            return None

    # ------------------------------------------------------------ camera layer

    @staticmethod
    def _prepare_image(image: Any) -> np.ndarray | None:
        """Validate the scene image and convert it to contiguous BGR uint8."""
        if image is None:
            return None
        img = np.asarray(image)
        if img.size == 0 or img.ndim not in (2, 3):
            return None
        if img.dtype != np.uint8:
            f = img.astype(np.float32)
            if np.issubdtype(img.dtype, np.floating) and float(np.nanmax(f)) <= 1.0 + 1e-6:
                f = f * 255.0
            img = np.clip(np.nan_to_num(f), 0, 255).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[..., 0]
        if img.ndim == 2:
            return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGRA2BGR)
        if img.shape[2] != 3:
            return None
        return np.ascontiguousarray(img)

    def _camera_layer(self, scene: SceneInput | None, view: ViewTransform) -> np.ndarray:
        """Camera image mapped into the window (cover crop / contain bars); cached by image_id."""
        img = None
        if scene is not None:
            try:
                img = self._prepare_image(scene.image)
            except (TypeError, ValueError, cv2.error):
                img = None
            if img is None and scene.image is not None:
                self._limiter("bad-image", logging.WARNING, "Unusable scene image %s; showing placeholder",
                              getattr(scene.image, "shape", type(scene.image).__name__))
        if img is None:
            return self._placeholder(_num(getattr(scene, "time", 0.0), 0.0))
        image_id = int(getattr(scene, "image_id", -1))
        key = (image_id, img.shape, view, self._size) if image_id >= 0 else None
        if key is not None and key == self._cam_key and self._cam is not None:
            return self._cam
        w, h = self._size
        if self._cam is None or self._cam.shape[:2] != (h, w):
            self._cam = np.zeros((h, w, 3), np.uint8)
        out = self._cam
        ih, iw = img.shape[:2]
        sx = float(view.scale) * view.scene_w / iw          # image px -> window px
        sy = float(view.scale) * view.scene_h / ih
        ox, oy = float(view.offset_x), float(view.offset_y)
        x0, x1 = max(0, int(round(ox))), min(w, int(round(ox + iw * sx)))
        y0, y1 = max(0, int(round(oy))), min(h, int(round(oy + ih * sy)))
        if x1 <= x0 or y1 <= y0 or not _finite(sx, sy, ox, oy) or sx <= 0 or sy <= 0:
            out.fill(0)
        else:
            if y0 > 0:
                out[:y0] = 0
            if y1 < h:
                out[y1:] = 0
            if x0 > 0:
                out[y0:y1, :x0] = 0
            if x1 < w:
                out[y0:y1, x1:] = 0
            dst = out[y0:y1, x0:x1]
            dw, dh = x1 - x0, y1 - y0
            if abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6 and ox.is_integer() and oy.is_integer():
                u0, v0 = x0 - int(ox), y0 - int(oy)
                dst[...] = img[v0:v0 + dh, u0:u0 + dw]
            elif sx < 0.67 or sy < 0.67:
                # strong downscale: area filter avoids aliasing (sub-pixel crop offset is negligible)
                u0 = max(0, int(math.floor((x0 - ox) / sx)))
                u1 = min(iw, max(u0 + 1, int(math.ceil((x1 - ox) / sx))))
                v0 = max(0, int(math.floor((y0 - oy) / sy)))
                v1 = min(ih, max(v0 + 1, int(math.ceil((y1 - oy) / sy))))
                dst[...] = cv2.resize(img[v0:v1, u0:u1], (dw, dh), interpolation=cv2.INTER_AREA)
            else:
                m = np.float32([[1.0 / sx, 0.0, (x0 + 0.5 - ox) / sx - 0.5],
                                [0.0, 1.0 / sy, (y0 + 0.5 - oy) / sy - 0.5]])
                res = cv2.warpAffine(img, m, (dw, dh), dst=dst,
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_REPLICATE)
                if res is not dst:
                    dst[...] = res
        self._cam_key = key
        return out

    def _placeholder(self, t: float) -> np.ndarray:
        """Animated dark 'energy fog' background shown while there is no camera image."""
        w, h = self._size
        if self._ph_base is None or self._ph_base.shape[:2] != (h, w):
            yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
            xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
            top = np.array([26, 12, 9], np.float32)        # BGR
            bottom = np.array([10, 5, 3], np.float32)
            img = top * (1.0 - yy) + bottom * yy
            radial = np.exp(-((xx - 0.5) ** 2 / 0.16 + (yy - 0.45) ** 2 / 0.10))
            img = img + radial * np.array([46, 22, 14], np.float32)
            fog = cv2.resize(np.tile(self._noise.cloud[0], (2, 2)), (w, h), interpolation=cv2.INTER_LINEAR)
            img = img + (fog[..., None] - 0.5) * np.array([14, 8, 6], np.float32) * (0.4 + radial)
            self._ph_base = np.clip(img, 0, 255).astype(np.uint8)
            self._ph = np.empty_like(self._ph_base)
        lw, lh = max(8, w // 12), max(8, h // 12)
        gx = np.linspace(0.0, 1.0, lw, dtype=np.float32)[None, :]
        gy = np.linspace(0.0, 1.0, lh, dtype=np.float32)[:, None]
        acc = np.zeros((lh, lw, 3), np.float32)
        blobs = (((1.00, 0.55, 0.15), 0.13, 0.11, 0.0),    # BGR colours, Lissajous speeds, phase
                 ((0.95, 0.25, 0.45), 0.09, 0.14, 2.1),
                 ((0.80, 0.70, 0.20), 0.07, 0.05, 4.2))
        for col, fx, fy, ph in blobs:
            cx = 0.5 + 0.34 * math.sin(t * fx * TAU + ph)
            cy = 0.5 + 0.26 * math.sin(t * fy * TAU + ph * 1.7)
            g = np.exp(-(((gx - cx) * (w / h)) ** 2 + (gy - cy) ** 2) / 0.05)
            acc += g[..., None] * np.array(col, np.float32)
        anim = cv2.resize(np.clip(acc * 22.0, 0, 255).astype(np.uint8), (w, h),
                          interpolation=cv2.INTER_LINEAR)
        cv2.add(self._ph_base, anim, dst=self._ph)
        return self._ph

    # -------------------------------------------------------- grade / lights

    def _grade_camera(self, base: np.ndarray, frame: np.ndarray, effects: RenderList,
                      view: ViewTransform) -> None:
        """frame = lights(grade(refract(base))). ``base`` (cached camera layer) stays untouched."""
        grade = effects.grade if isinstance(effects.grade, Grade) else Grade()
        src = base
        if effects.rings and getattr(self._quality, "refraction", False):
            if self._guard("refraction", self._refract, base, frame, effects.rings, view):
                src = frame
        # A pure darken is a scalar: fold it into the light multiplier instead of a colour matrix.
        scale = 1.0 - clamp(_num(grade.darken, 0.0))
        if clamp(_num(grade.desaturate, 0.0)) >= 1e-3 or clamp(_num(grade.tint_strength, 0.0)) >= 1e-3:
            matrix = _grade_matrix(grade)
            if matrix is not None:
                cv2.transform(src, matrix, dst=frame)
                src, scale = frame, 1.0
        maps = self._guard("lights", self._light_maps, effects.lights, grade, view, scale)
        if maps is not None:
            mul, add, mul_scale = maps
            cv2.multiply(src, mul, dst=frame, scale=mul_scale)
            if add is not None:
                cv2.add(frame, add, dst=frame)
        elif scale < 1.0 - 1e-3:
            cv2.convertScaleAbs(src, dst=frame, alpha=scale)
        elif src is not frame:
            np.copyto(frame, src)

    def _light_maps(self, lights: Sequence[Light], grade: Grade, view: ViewTransform, scale: float = 1.0
                    ) -> tuple[np.ndarray, np.ndarray | None] | None:
        """Low-resolution light + vignette map -> window-size uint8 multiplier, offset, scale.

        Returns ``(mul, add, mul_scale)`` with ``frame = src * mul * mul_scale + add`` (or None
        when there is nothing to apply). ``scale`` (the darkening factor) is folded into the
        multiplier so a dimmed, lit and vignetted camera costs a single multiply.
        """
        w, h = self._size
        vig = clamp(_num(grade.vignette, 0.0))
        valid = [li for li in lights if isinstance(li, Light)
                 and _finite(li.position[0], li.position[1], li.radius, li.intensity)
                 and li.radius > 0 and li.intensity > 1e-4]
        if not valid and vig < 0.004:
            return None
        lw, lh = max(8, -(-w // _LIGHT_DIV)), max(8, -(-h // _LIGHT_DIV))
        kx, ky = lw / w, lh / h
        light = np.zeros((lh, lw, 3), np.float32)
        # light-map pixel centres in window px (lights are evaluated in window units)
        gx_all = _grid(0, lw) * np.float32(1.0 / kx)
        gy_all = _grid(0, lh) * np.float32(1.0 / ky)
        for li in valid:
            wx, wy = view.scene_to_window(float(li.position[0]), float(li.position[1]))
            r = max(1.0, float(li.radius)) * float(view.scale)
            reach = _LIGHT_REACH * r
            box = _clip_box((wx - reach) * kx, (wy - reach) * ky, (wx + reach) * kx, (wy + reach) * ky, lw, lh)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            dx = gx_all[x0:x1] - np.float32(wx)
            dy = gy_all[y0:y1] - np.float32(wy)
            d2 = dy[:, None] * dy[:, None] + dx[None, :] * dx[None, :]
            # 1 / (1 + q^2) with a smooth cut-off between 2 and 3 radii (as on the GPU)
            f = np.float32(1.0) / (np.float32(1.0) + d2 * np.float32(1.0 / (r * r)))
            f *= _smoothstep(_LIGHT_REACH * r, (_LIGHT_REACH - 1.0) * r, np.sqrt(d2))
            try:
                col = _bgr(li.color) * np.float32(min(float(li.intensity), 16.0))
            except (TypeError, ValueError, IndexError):
                col = np.full(3, min(float(li.intensity), 16.0), np.float32)
            _add_colored(light[y0:y1, x0:x1], [f], [col])
        mul = (np.float32(1.0) + light) * np.float32(scale)
        add = light * np.float32(_LIGHT_ADD * 255.0) if valid else None
        if vig >= 0.004:
            fac = np.float32(1.0) - np.float32(vig) * self._vignette_map(lw, lh, w / h)
            mul *= fac[..., None]
            if add is not None:
                add *= fac[..., None]
        # The multiplier is stored as uint8 fixed point: x/128 keeps 1/128 steps (no banding in
        # smooth vignettes); bright lights switch to x/64 or x/32 so they can reach 4x / 8x.
        peak = float(mul.max()) if mul.size else 1.0
        div = 128.0 if peak <= 255.0 / 128.0 else (64.0 if peak <= 255.0 / 64.0 else 32.0)
        mul_u8 = np.clip(mul * np.float32(div) + np.float32(0.5), 0, 255).astype(np.uint8)
        mul_up = cv2.resize(mul_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        add_up = None
        if add is not None:
            add_u8 = np.clip(add + np.float32(0.5), 0, 255).astype(np.uint8)
            if add_u8.any():
                add_up = cv2.resize(add_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        return mul_up, add_up, 1.0 / div

    def _vignette_map(self, lw: int, lh: int, aspect: float) -> np.ndarray:
        """Vignette weight per light-map pixel: ``smoothstep(0.30, 1.05, |q|)`` with ``q`` the
        aspect-corrected offset from the window centre (same shape as on the GPU)."""
        key = (lw, lh, round(aspect, 4))
        cached = self._vignette.get(key)
        if cached is None:
            gx = (_grid(0, lw) / np.float32(lw) - np.float32(0.5)) * np.float32(aspect)
            gy = _grid(0, lh) / np.float32(lh) - np.float32(0.5)
            d = np.sqrt(gx[None, :] * gx[None, :] + gy[:, None] * gy[:, None])
            cached = _smoothstep(0.30, 1.05, d)
            self._vignette[key] = cached
        return cached

    def _refract(self, base: np.ndarray, frame: np.ndarray, rings: Sequence[Ring],
                 view: ViewTransform) -> bool:
        """Shockwave refraction: displace the camera image radially around ring bands."""
        items = []
        w, h = self._size
        s = float(view.scale)
        for ring in rings:
            if not isinstance(ring, Ring) or not _finite(ring.refraction, ring.radius, ring.thickness,
                                                         ring.center[0], ring.center[1]):
                continue
            amount = float(ring.refraction) * s
            if amount < 0.3 or ring.radius <= 0:
                continue
            cx, cy = view.scene_to_window(float(ring.center[0]), float(ring.center[1]))
            sqx = max(0.05, _num(ring.squash[0], 1.0))
            sqy = max(0.05, _num(ring.squash[1], 1.0))
            thick = max(2.0, float(ring.thickness) * s)
            items.append((cx, cy, float(ring.radius) * s, thick, sqx, sqy, min(amount, 0.05 * w)))
        if not items:
            return False
        f = 4                                                 # displacement grid = window / 4
        gx0, gy0, gx1, gy1 = w, h, 0, 0
        for cx, cy, r, th, sqx, sqy, _ in items:
            ex, ey = (r + 2.5 * th) * sqx, (r + 2.5 * th) * sqy
            gx0, gy0 = min(gx0, cx - ex), min(gy0, cy - ey)
            gx1, gy1 = max(gx1, cx + ex), max(gy1, cy + ey)
        box = _clip_box(gx0 / f, gy0 / f, gx1 / f, gy1 / f, -(-w // f), -(-h // f))
        if box is None:
            return False
        bx0, by0, bx1, by1 = box
        xs = (_grid(bx0, bx1 - bx0) * f)[None, :]
        ys = (_grid(by0, by1 - by0) * f)[:, None]
        disp_x = np.zeros((by1 - by0, bx1 - bx0), np.float32)
        disp_y = np.zeros_like(disp_x)
        for cx, cy, r, th, sqx, sqy, amount in items:
            ex = (xs - np.float32(cx)) / np.float32(sqx)
            ey = (ys - np.float32(cy)) / np.float32(sqy)
            rr = np.sqrt(ex * ex + ey * ey) + np.float32(1e-3)
            prof = np.exp(-np.square((rr - np.float32(r)) / np.float32(th))) * np.float32(amount)
            disp_x += prof * ex / rr
            disp_y += prof * ey / rr
        px0, py0 = bx0 * f, by0 * f
        pw, ph = min(w, bx1 * f) - px0, min(h, by1 * f) - py0
        if pw <= 0 or ph <= 0:
            return False
        full_w, full_h = (bx1 - bx0) * f, (by1 - by0) * f
        dxu = cv2.resize(disp_x, (full_w, full_h), interpolation=cv2.INTER_LINEAR)[:ph, :pw]
        dyu = cv2.resize(disp_y, (full_w, full_h), interpolation=cv2.INTER_LINEAR)[:ph, :pw]
        map_x = (np.arange(px0, px0 + pw, dtype=np.float32)[None, :] - dxu).astype(np.float32)
        map_y = (np.arange(py0, py0 + ph, dtype=np.float32)[:, None] - dyu).astype(np.float32)
        np.copyto(frame, base)
        frame[py0:py0 + ph, px0:px0 + pw] = cv2.remap(base, map_x, map_y, cv2.INTER_LINEAR,
                                                     borderMode=cv2.BORDER_REPLICATE)
        return True

    # ---------------------------------------------------------------- effects

    def _render_effects(self, effects: RenderList, view: ViewTransform) -> np.ndarray | None:
        """Draw all primitives into the effect buffer; returns the window-size uint8 overlay."""
        radial = [f for f in effects.flashes if isinstance(f, Flash) and f.center is not None
                  and _num(f.radius, 0.0) > 0]
        if not (effects.sprites or effects.orbs or effects.beams or effects.frontal_beams
                or effects.arcs or effects.rings or effects.ribbons or radial
                or effects.aura is not None):
            return None
        effect = self._effect_buffer()
        space = _Space.create(view, self._size, self._eff_size)
        self._space = space
        if effects.aura is not None:
            self._guard("aura", self._draw_aura, effect, effects.aura, space, view)
        for beam in effects.beams:
            self._guard("beam", self._draw_beam, effect, beam, space)
        for fb in effects.frontal_beams:
            self._guard("frontal_beam", self._draw_frontal, effect, fb, space)
        for orb in effects.orbs:
            self._guard("orb", self._draw_orb, effect, orb, space)
        if effects.rings:
            self._guard("rings", self._draw_rings, effect, effects.rings, space)
        if effects.ribbons:
            self._guard("ribbons", self._draw_ribbons, effect, effects.ribbons, space)
        if effects.arcs:
            self._guard("arcs", self._draw_arcs, effect, effects.arcs, space)
        for batch in effects.sprites:
            self._guard("sprites", self._add_sprites, batch, space)
        self._guard("splats", self._splatter.resolve, effect)
        self._splatter.clear()
        for fl in radial:
            self._guard("flash", self._draw_radial_flash, effect, fl, space)
        # The effect system already folds EffectsSettings.bloom_strength into RenderList.bloom
        # (as for the GPU renderer), so it must not be applied a second time here.
        strength = min(max(_num(effects.bloom, 1.0), 0.0), 8.0)
        self._guard("bloom", self._bloom, effect, strength)
        toned = self._tonemap(effect, _num(effects.exposure, 1.0))
        w, h = self._size
        return cv2.resize(toned, (w, h), interpolation=cv2.INTER_LINEAR)

    # ---- sprites

    def _add_sprites(self, batch: SpriteBatch, space: _Space) -> None:
        """Queue one batch for splatting (vectorised over the whole batch).

        Every sprite becomes a Gaussian mass carrying the light energy of the GPU sprite
        profile, computed from its true size, so tiny particles keep their total light when
        the low-resolution buffer spreads them wider (as the GPU's minimum sprite size does).
        SPARKs with velocities become streaks: masses along ``-velocity * stretch`` with the
        GPU's brightness ramp (dim tail, bright head).
        """
        if not isinstance(batch, SpriteBatch) or len(batch) == 0:
            return
        pos = np.asarray(batch.positions, np.float32).reshape(-1, 2)
        sizes = np.asarray(batch.sizes, np.float32).reshape(-1)
        cols = np.asarray(batch.colors, np.float32)
        if cols.ndim != 2 or cols.shape[1] < 3:
            raise ValueError(f"sprite colors must be (N, 4), got {cols.shape}")
        n = min(len(pos), len(sizes), len(cols))
        if n == 0:
            return
        pos, sizes, cols = pos[:n], sizes[:n], cols[:n]
        alpha = np.clip(cols[:, 3:4], 0.0, 1.0) if cols.shape[1] >= 4 else np.ones((n, 1), np.float32)
        bgr = cols[:, 2::-1] * alpha                   # additive: rgb * opacity, BGR order
        sig_k, energy, white = _SPRITE_SHAPES.get(int(batch.shape), _SPRITE_SHAPES[int(SpriteShape.GLOW)])
        x = pos[:, 0] * np.float32(space.ax) + np.float32(space.bx)
        y = pos[:, 1] * np.float32(space.ay) + np.float32(space.by)
        rad = np.maximum(sizes, 0.0) * np.float32(space.k)            # radius in effect px
        sig = np.maximum(rad * np.float32(sig_k), np.float32(_SPRITE_MIN_SIGMA))
        reach = 4.0 * sig
        ok = (np.isfinite(x) & np.isfinite(y) & np.isfinite(rad) & np.isfinite(bgr).all(axis=1)
              & (bgr.max(axis=1) > 1e-4) & (rad > 1e-3))
        streak = (int(batch.shape) == int(SpriteShape.SPARK) and batch.velocities is not None
                  and _num(batch.stretch, 0.0) > 0)
        tail = None
        if streak:
            vel = np.asarray(batch.velocities, np.float32).reshape(-1, 2)
            if len(vel) < n:
                streak = False
            else:
                stretch = float(batch.stretch)
                tail = np.empty((n, 2), np.float32)
                tail[:, 0] = vel[:n, 0] * np.float32(stretch * space.ax)
                tail[:, 1] = vel[:n, 1] * np.float32(stretch * space.ay)
                ok &= np.isfinite(tail).all(axis=1)
                np.nan_to_num(tail, copy=False)
                reach = reach + np.abs(tail[:, 0]) + np.abs(tail[:, 1])
        ok &= (x > -reach) & (y > -reach) & (x < space.w + reach) & (y < space.h + reach)
        if not ok.all():
            x, y, rad, sig, bgr = x[ok], y[ok], rad[ok], sig[ok], bgr[ok]
            if tail is not None:
                tail = tail[ok]
        if x.size == 0:
            return
        bgr = np.maximum(bgr, 0.0)
        if white > 0.0:                                 # whiter hot cores
            bgr = bgr * np.float32(1.0 - white) + bgr.max(axis=1, keepdims=True) * np.float32(white)
        e_dot = np.float32(energy) * rad * rad
        self._sprite_xyr = (x, y, rad)
        keep = self._pick_sparkles(batch, ok, bgr, tail)
        if keep is not None:
            e_dot *= keep
        if streak and tail is not None:
            length = np.hypot(tail[:, 0], tail[:, 1])
            longest = float(length.max()) if length.size else 0.0
            if longest >= 0.75:
                samples = int(min(16, math.ceil(longest / 0.75) + 1))
                samples = max(2, min(samples, _MAX_STREAK_SAMPLES // max(1, x.size)))
                tj = np.linspace(0.0, 1.0, samples, dtype=np.float32)     # 0 = head, 1 = tail
                ramp = np.float32(_SPARK_TAIL) + np.float32(1.0 - _SPARK_TAIL) * np.square(1.0 - tj)
                ramp /= ramp.sum()
                line_e = np.float32(_SPARK_LINE_ENERGY) * rad * length
                if keep is not None:
                    line_e *= keep
                # the round caps: full brightness at the head, the tail ramp value at the end
                caps = np.zeros((x.size, samples), np.float32)
                caps[:, 0] = e_dot * np.float32(0.5)
                caps[:, -1] += e_dot * np.float32(0.5 * _SPARK_TAIL)
                wts = line_e[:, None] * ramp[None, :] + caps
                still = length < 1e-3                       # not moving: one full dot
                if still.any():
                    wts[still] = 0.0
                    wts[still, 0] = e_dot[still]
                xs = (x[:, None] - tail[:, 0:1] * tj[None, :]).ravel()
                ys = (y[:, None] - tail[:, 1:2] * tj[None, :]).ravel()
                mass = (bgr[:, None, :] * wts[:, :, None]).reshape(-1, 3)
                self._splatter.add(xs, ys, np.repeat(sig, samples), mass)
                return
        self._splatter.add(x, y, sig, bgr * e_dot[:, None])

    def _pick_sparkles(self, batch: SpriteBatch, ok: np.ndarray, bgr: np.ndarray,
                       tail: np.ndarray | None) -> np.ndarray | None:
        """Choose this batch's sparkle candidates; returns the splat energy factor per sprite.

        ``bgr`` and ``tail`` are the filtered (``ok``) sprite colours and effect-space motion
        blur vectors. Effect px map to window px by ``1 / k`` (no offset), see :class:`_Space`.
        """
        spec = _SPARKLE_SHAPES.get(int(batch.shape))
        space = self._space
        if spec is None or space is None or bgr.shape[0] == 0:
            return None
        share, gain = spec
        core = bgr * np.float32(gain)                       # HDR radiance of the hot core
        peak = core.max(axis=1)
        idx = np.flatnonzero(peak >= np.float32(_SPARKLE_MIN_PEAK))
        if idx.size == 0:
            return None
        if idx.size > _SPARKLE_MAX:
            idx = idx[np.argpartition(peak[idx], idx.size - _SPARKLE_MAX)[-_SPARKLE_MAX:]]
        x, y, rad = self._sprite_xyr
        fx, fy = space.w_per_px, space.h_per_px
        head = np.empty((idx.size, 2), np.float32)
        head[:, 0] = x[idx] * np.float32(fx)
        head[:, 1] = y[idx] * np.float32(fy)
        if tail is not None:
            end = head - tail[idx] * np.array([fx, fy], np.float32)
            dot = np.hypot(end[:, 0] - head[:, 0], end[:, 1] - head[:, 1]) < 1.0
        else:
            end = head.copy()
            dot = np.ones(idx.size, bool)
        # dots (embers, slow sparks) become a 1 px dash: an AA line needs some length to show up
        end[dot] = head[dot] + np.array([1.0, 0.0], np.float32)
        thick = np.where(rad[idx] * np.float32(fx) >= 2.6, 2, 1).astype(np.int8)
        self._sparkles.append((peak[idx], head, end, core[idx], thick, dot))
        keep = np.ones(bgr.shape[0], np.float32)
        keep[idx] = np.float32(1.0 - share)
        return keep

    # ---- orbs

    def _orb_stamp(self, radius: float) -> _OrbStamp:
        q = int(round(math.log(max(radius, 0.35)) / _ORB_QUANT))
        stamp = self._orb_cache.get(q)
        if stamp is None:
            stamp = _build_orb_stamp(math.exp(q * _ORB_QUANT))
            self._orb_cache.put(q, stamp, stamp.nbytes)
        return stamp

    def _draw_orb(self, effect: np.ndarray, orb: Orb, space: _Space) -> None:
        if not isinstance(orb, Orb):
            return
        cx, cy = space.pt(orb.center)
        radius = _num(orb.radius, 0.0) * space.k
        intensity = _num(orb.intensity, 0.0)
        if not _finite(cx, cy) or radius < 0.35 or intensity <= 1e-3:
            return
        h, w = effect.shape[:2]
        radius = min(radius, 2.0 * max(w, h))
        t = _num(orb.time, 0.0)
        seed = _num(orb.seed, 0.0)
        intensity *= 1.0 + 0.035 * math.sin(t * TAU * 3.0 + seed * 5.1)   # ~3 Hz pulse
        stamp = self._orb_stamp(radius)
        n = 2 * stamp.half + 1
        sa = max(0.05, _num(orb.squash[0], 1.0))
        sc = max(0.05, _num(orb.squash[1], 1.0))
        if abs(sa - 1.0) > 0.02 or abs(sc - 1.0) > 0.02:
            img = self._orb_pixels(stamp, 0, n, 0, n, orb, intensity)
            self._add_flares(img, stamp.half + 0.5, stamp.half + 0.5, stamp.radius, orb, intensity)
            self._warp_stamp(effect, img, cx, cy, stamp.half, sa, sc, _num(orb.rotation, 0.0))
            return
        ci, cj = int(math.floor(cy)), int(math.floor(cx))   # buffer pixel holding the centre
        y0, x0 = ci - stamp.half, cj - stamp.half
        by0, by1 = max(y0, 0), min(y0 + n, h)
        bx0, bx1 = max(x0, 0), min(x0 + n, w)
        if by1 <= by0 or bx1 <= bx0:
            return
        img = self._orb_pixels(stamp, by0 - y0, by1 - y0, bx0 - x0, bx1 - x0, orb, intensity)
        self._add_flares(img, cj + 0.5 - bx0, ci + 0.5 - by0, stamp.radius, orb, intensity)
        effect[by0:by1, bx0:bx1] += img

    def _orb_pixels(self, st: _OrbStamp, y0: int, y1: int, x0: int, x1: int, orb: Orb,
                    intensity: float) -> np.ndarray:
        """Render stamp rows y0:y1 / cols x0:x1 (stamp coordinates) as BGR HDR.

        Follows the GPU orb: a noise-wobbled, breathing silhouette; swirling plasma body
        (inner -> outer colour towards the rim) with ridge filaments; white-hot core plus
        core glow; fresnel rim; crackle sparks on the edge when unstable; tight inner halo;
        wide outer halo (static, from the stamp).
        """
        pal = _palette(orb.palette)
        ch = clamp(_num(orb.charge, 0.0))
        inst = clamp(_num(orb.instability, 0.0))
        t = _num(orb.time, 0.0) % 4096.0
        swirl = min(max(_num(orb.swirl, 1.0), 0.0), 8.0)
        seed = _num(orb.seed, 0.0) % 97.0
        # Same calibration as the GPU shader: only the core uses the full intensity, the
        # plasma body, rim and halo a compressed gain so they keep their colour.
        ib = 0.55 + 0.3 * intensity
        core_c = _bgr(pal.core) * np.float32(intensity)
        soft_core_c = _bgr(pal.core) * np.float32(ib)
        inner_c = _bgr(pal.inner) * np.float32(ib)
        outer_c = _bgr(pal.outer) * np.float32(ib)
        out = np.zeros((y1 - y0, x1 - x0, 3), np.float32)
        _add_colored(out, [st.halo[y0:y1, x0:x1]], [outer_c * np.float32(0.82)])
        s = st.half - st.dhalf
        e = s + 2 * st.dhalf + 1
        dy0, dy1 = max(y0, s), min(y1, e)
        dx0, dx1 = max(x0, s), min(x1, e)
        if dy1 <= dy0 or dx1 <= dx0:
            return out
        sub = (slice(dy0 - s, dy1 - s), slice(dx0 - s, dx1 - s))
        rn = st.rn[sub]
        theta = st.theta[sub]
        twist = st.twist[sub]
        one = np.float32(1.0)
        noise = self._noise
        # swirling plasma: two noise layers in swirled polar coordinates (seam-free angular repeats)
        spin = t * (0.9 + 1.4 * swirl) / TAU                       # turns of the whole pattern
        tw = (1.8 + 2.4 * swirl) / TAU                              # spiral twist at the centre (turns)
        u1 = theta * np.float32(3 * _TEX) + twist * np.float32(tw * 3 * _TEX) \
            + np.float32(((spin * 3.0 + seed * 0.37) % 1.0) * _TEX)
        v1 = st.v1[sub] + np.float32(((t * 0.30 + seed * 0.21) % 1.0) * _TEX)
        rate1 = max(3.0 * _TEX / (TAU * st.radius), 0.55 * _TEX / st.radius)
        n1 = noise.remap(noise.cloud, u1, v1, rate1)
        if int(_num(getattr(self._quality, "noise_octaves", 4), 4)) >= 3:
            u2 = theta * np.float32(5 * _TEX) - twist * np.float32(tw * 5 * _TEX) \
                + np.float32(((-spin * 5.0 * 0.6 + seed * 0.59) % 1.0) * _TEX)
            v2 = st.v2[sub] + np.float32(((t * 0.45 + seed * 0.13) % 1.0) * _TEX)
            rate2 = max(5.0 * _TEX / (TAU * st.radius), 0.95 * _TEX / st.radius)
            n2 = noise.remap(noise.cloud2, u2, v2, rate2)
        else:
            n2 = n1
        # the textures have more contrast than the GPU's fbm: compress to a similar spread
        pl1 = (n1 - np.float32(0.5)) * np.float32(0.6) + np.float32(0.5)
        pl2 = (n2 - np.float32(0.5)) * np.float32(0.6) + np.float32(0.5)
        plasma = _smoothstep(0.28, 0.78, pl1 * np.float32(0.62) + pl2 * np.float32(0.38))
        ridge = np.clip(one - np.abs(pl1 - pl2) * np.float32(5.0), 0.0, 1.0)
        ridge *= ridge
        ridge *= ridge
        # wobbling, breathing silhouette (angle-only noise, evolving over time)
        amp = (0.04 + 0.16 * inst) * _smoothstep(1.65, 1.25, rn)
        edge = noise.angular(noise.cloud, theta, t * 0.35 + seed * 0.11) - np.float32(0.5)
        pulse = 1.0 + 0.035 * math.sin(t * 18.85)                    # ~3 Hz breathing
        rr = rn / (np.float32(pulse) * (one + edge * amp))
        sphere = _smoothstep(1.01, 0.93, rr)
        z = np.sqrt(np.clip(one - rr * rr, 0.0, 1.0))
        fres = one - z
        fres = fres * fres * fres * sphere
        core_r = 0.26 + 0.12 * ch
        core = _smoothstep(core_r, core_r * 0.45, rr + (pl1 - np.float32(0.5)) * np.float32(0.16))
        core_glow = np.exp(rr * rr * np.float32(-1.0 / (core_r * core_r * 1.4)))
        mixf = _smoothstep(0.35, 1.0, rr) * np.float32(0.45)        # body colour: inner -> outer
        body = (np.float32(0.30) + np.float32(1.15) * plasma) * sphere * np.float32(0.85)
        rid = ridge * sphere * np.float32(0.85 * (0.25 + 0.9 * ch))
        dout = np.maximum(rr - one, 0.0)
        halo_in = np.exp(dout * np.float32(-8.5)) * (one - sphere) * np.float32(0.80)
        w_core = core * np.float32(2.6 + 0.8 * ch) + core_glow * np.float32(0.30)
        w_soft = rid * np.float32(0.7) + fres * np.float32(1.5 * 0.35)
        w_inner = body * (one - mixf) + rid * np.float32(0.3) + fres * np.float32(1.5 * 0.65) + halo_in
        w_outer = body * mixf
        if inst > 0.01:
            # crackle: bright sparks on the turbulent edge (fast angular noise, ^5)
            g = noise.angular(noise.fine, theta, t * 0.5 + seed * 0.7)
            g2 = g * g
            crackle = g2 * g2 * g * np.float32(5.0 * inst) * _smoothstep(0.86, 1.0, rr) * _smoothstep(1.18, 1.0, rr)
            w_soft += crackle * np.float32(0.55)
            w_inner += crackle * np.float32(0.45)
        _add_colored(out[dy0 - y0:dy1 - y0, dx0 - x0:dx1 - x0], [w_core, w_soft, w_inner, w_outer],
                     [core_c, soft_core_c, inner_c, outer_c])
        return out

    def _add_flares(self, img: np.ndarray, cx: float, cy: float, radius: float, orb: Orb,
                    intensity: float) -> None:
        """Radial flare rays around a highly charged orb: a few thin additive lines.

        Like the GPU's thresholded angular noise, each ray has a fixed direction and flickers
        in and out on its own clock; it is brightest at the surface and fades outwards.
        """
        flare = min(max(_num(orb.flare, 0.0), 0.0), 2.0)
        if flare < 0.02:
            return
        h, w = img.shape[:2]
        layer = np.zeros((h, w), np.uint8)
        t = _num(orb.time, 0.0) % 4096.0
        s = int(abs(_num(orb.seed, 0.0)) * 1000.0) & 0xFFFF
        thick = 1 if radius < 24.0 else 2
        drawn = 0
        for i in range(20):
            # value noise in time per ray, raised to a power: rays pop in and out
            ph = t * (1.3 + 0.9 * hash01(i + 31, s)) + 7.0 * hash01(i + 57, s)
            k = int(math.floor(ph))
            f = ph - k
            f = f * f * (3.0 - 2.0 * f)
            v = hash01(k * 32 + i, s) * (1.0 - f) + hash01((k + 1) * 32 + i, s) * f
            bright = v ** 4
            if bright < 0.04:
                continue
            ang = TAU * hash01(i, s)
            length = radius * (1.9 + 1.1 * hash01(i + 77, s)) * (0.8 + 0.4 * min(flare, 1.0))
            dx, dy = math.cos(ang), math.sin(ang)
            value = 255.0 * bright
            r0 = radius * 0.9
            for a, b, fall in ((0.0, 0.35, 1.0), (0.35, 0.7, 0.5), (0.7, 1.0, 0.2)):
                pa = _fp_point(cx + dx * (r0 + (length - r0) * a), cy + dy * (r0 + (length - r0) * a))
                pb = _fp_point(cx + dx * (r0 + (length - r0) * b), cy + dy * (r0 + (length - r0) * b))
                cv2.line(layer, pa, pb, int(round(value * fall)), thick, cv2.LINE_AA, _SHIFT)
            drawn += 1
        if not drawn:
            return
        glow = cv2.GaussianBlur(layer.astype(np.float32), (0, 0), max(0.6, radius * 0.03))
        glow *= np.float32(flare * intensity * 1.4 / 255.0)
        pal = _palette(orb.palette)
        _add_colored(img, [glow], [_bgr(pal.inner) * 0.45 + _bgr(pal.core) * 0.55])

    @staticmethod
    def _warp_stamp(effect: np.ndarray, img: np.ndarray, cx: float, cy: float, half: int,
                    sa: float, sc: float, rotation: float) -> None:
        """Add a stamp scaled by (sa along ``rotation``, sc across) around (cx, cy)."""
        h, w = effect.shape[:2]
        c, s = math.cos(rotation), math.sin(rotation)
        rot = np.array([[c, -s], [s, c]])
        inv = rot @ np.diag([1.0 / sa, 1.0 / sc]) @ rot.T     # buffer offset -> stamp offset
        ext = half * max(sa, sc) + 2.0
        box = _clip_box(cx - ext, cy - ext, cx + ext, cy + ext, w, h)
        if box is None:
            return
        x0, y0, x1, y1 = box
        offset = inv @ np.array([x0 + 0.5 - cx, y0 + 0.5 - cy]) + half
        m = np.hstack([inv, offset[:, None]]).astype(np.float32)
        warped = cv2.warpAffine(img, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        effect[y0:y1, x0:x1] += warped

    # ---- beams

    def _draw_beam(self, effect: np.ndarray, beam: Beam, space: _Space) -> None:
        """Lateral beam: rendered in its own axis-aligned canvas, then rotated into place."""
        if not isinstance(beam, Beam):
            return
        ox, oy = space.pt(beam.origin)
        dxs = _num(beam.direction[0], 0.0) * space.ax
        dys = _num(beam.direction[1], 0.0) * space.ay
        norm = math.hypot(dxs, dys)
        fade = clamp(_num(beam.fade, 1.0))
        intensity = min(_num(beam.intensity, 0.0), 64.0) * fade
        if not _finite(ox, oy) or norm < 1e-9 or intensity <= 1e-3:
            return
        dx, dy = dxs / norm, dys / norm
        length = max(0.0, _num(beam.length, 0.0) * space.k)
        w0 = max(0.5, _num(beam.width, 0.0) * space.k)
        w1 = _num(beam.end_width, 0.0) * space.k
        w1 = max(0.5, w1) if w1 > 0 else w0
        head = max(0.0, _num(beam.head_radius, 0.0) * space.k)
        h, w = effect.shape[:2]
        # canvas extents as the GPU quad: behind the origin, past the head, across the halo
        back = 3.2 * w0
        hv = max(4.6 * max(w0, w1), 2.8 * head)
        u_lo, u_hi = -back, length + max(head * 2.7, w1 * 1.6)
        corners = ((0.0, 0.0), (float(w), 0.0), (0.0, float(h)), (float(w), float(h)))
        us = [(x - ox) * dx + (y - oy) * dy for x, y in corners]
        vs = [-(x - ox) * dy + (y - oy) * dx for x, y in corners]
        u_lo, u_hi = max(u_lo, min(us) - 2.0), min(u_hi, max(us) + 2.0)
        v_lo, v_hi = max(-hv, min(vs) - 2.0), min(hv, max(vs) + 2.0)
        if u_hi - u_lo < 1.0 or v_hi - v_lo < 1.0:
            return
        # wide beams are smooth at this scale: compute their canvas at half resolution
        cf = 2.0 if min(w0, w1) >= 5.0 and (u_hi - u_lo) * (v_hi - v_lo) > 30_000 else 1.0
        cw, ch = int(math.ceil((u_hi - u_lo) / cf)), int(math.ceil((v_hi - v_lo) / cf))
        canvas = self._beam_canvas(beam, intensity, u_lo, v_lo, cw, ch, cf, length, w0, w1, head, back, hv)
        pts = [(ox + uu * dx - vv * dy, oy + uu * dy + vv * dx)
               for uu in (u_lo, u_lo + cw * cf) for vv in (v_lo, v_lo + ch * cf)]
        box = _clip_box(min(p[0] for p in pts), min(p[1] for p in pts),
                        max(p[0] for p in pts), max(p[1] for p in pts), w, h)
        if box is None:
            return
        x0, y0, x1, y1 = box
        # buffer pixel (X, Y) of the box -> canvas pixel index (inverse map for warpAffine)
        c0 = (dx * (x0 + 0.5 - ox) + dy * (y0 + 0.5 - oy) - u_lo) / cf - 0.5
        c1 = (-dy * (x0 + 0.5 - ox) + dx * (y0 + 0.5 - oy) - v_lo) / cf - 0.5
        m = np.float32([[dx / cf, dy / cf, c0], [-dy / cf, dx / cf, c1]])
        warped = cv2.warpAffine(canvas, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        effect[y0:y1, x0:x1] += warped

    def _beam_canvas(self, beam: Beam, intensity: float, u_lo: float, v_lo: float, cw: int, ch: int,
                     cf: float, length: float, w0: float, w1: float, head: float, back: float,
                     hv: float) -> np.ndarray:
        """Beam in its own frame: x = along the axis (0 = origin), y = across (0 = axis).

        Canvas pixels are ``cf`` effect px; all lengths are effect px. The profile follows the
        GPU beam shader: distance to the axis *segment* (rounded ends), a flat-ish white core,
        a coloured body with a defined tube edge and a brighter skin, an exponential halo,
        streaming streak noise, a turbulent bulb at the head, a glow at the origin and the
        noise break-up while dissolving.
        """
        pal = _palette(beam.palette)
        t = _num(beam.time, 0.0) % 4096.0
        flow = t * min(max(_num(beam.flow_speed, 1.0), 0.0), 10.0)
        turb = min(max(_num(beam.turbulence, 1.0), 0.0), 4.0)
        seed = _num(beam.seed, 0.0) % 97.0
        dissolve = clamp(_num(beam.dissolve, 0.0))
        noise = self._noise
        one = np.float32(1.0)
        L = max(length, 1.0)
        k = 1.0 / w0                                         # noise scales follow the beam size
        u = np.float32(u_lo) + (np.arange(cw, dtype=np.float32) + np.float32(0.5)) * np.float32(cf)
        v = np.float32(v_lo) + (np.arange(ch, dtype=np.float32) + np.float32(0.5)) * np.float32(cf)
        uc = np.clip(u, 0.0, L)
        width = np.maximum(np.float32(w0) + np.float32(w1 - w0) * uc * np.float32(1.0 / L), 0.5)
        # edge wobble: 1D noise along the beam, streaming with the flow
        wob = noise.line(noise.cloud, u * np.float32(k * 0.45 / _NOISE_CELL) - np.float32(flow * 3.2 / _NOISE_CELL),
                         seed * 0.013) - np.float32(0.5)
        width *= one + wob * np.float32(0.42 * turb)
        inv_w = (one / width)[None, :]
        vn = v[:, None] * inv_w
        du = np.where(u < 0.0, u, np.where(u > L, u - np.float32(L), 0.0))
        dv = np.sqrt(du[None, :] * du[None, :] + v[:, None] * v[:, None])
        dn = dv * inv_w
        # energy streaming along the beam (two noise layers)
        rate = max(0.20 * k, 1.25 / max(w0, w1)) * _NOISE_CELL * cf
        mx1 = np.broadcast_to((u * np.float32(k * 0.20) - np.float32(flow * 5.5))[None, :] * np.float32(_NOISE_CELL), (ch, cw))
        my1 = (vn * np.float32(1.25) + np.float32(seed)) * np.float32(_NOISE_CELL)
        s1 = noise.remap(noise.cloud, mx1, my1, rate)
        mx2 = np.broadcast_to((u * np.float32(k * 0.52) - np.float32(flow * 9.0))[None, :] * np.float32(_NOISE_CELL), (ch, cw))
        my2 = (vn * np.float32(2.9) - np.float32(seed)) * np.float32(_NOISE_CELL)
        s2 = noise.remap(noise.cloud2, mx2, my2, rate * 2.4)
        sv = (s1 - np.float32(0.5)) * np.float32(0.6) + np.float32(0.5)   # GPU fbm-like spread
        sv2 = (s2 - np.float32(0.5)) * np.float32(0.6) + np.float32(0.5)
        near = _smoothstep(1.7, 1.25, dn)
        streak = np.float32(0.3) + (_smoothstep(0.50, 0.88, sv * np.float32(0.62) + sv2 * np.float32(0.5))
                                    - np.float32(0.3)) * near
        a = dn * np.float32(1.0 / 0.28)
        core = np.exp(-(a * a))
        b = dn * np.float32(1.0 / 0.70)
        body = np.exp(-(b * b))
        halo = np.exp(-dv * (inv_w * np.float32(1.0 / 1.3)))
        tube = _smoothstep(1.0, 0.80, dn)
        rim = _smoothstep(0.45, 0.92, dn) * tube
        w_core = core * (np.float32(1.15) + np.float32(1.6) * streak)
        w_body = body * (np.float32(0.30) + np.float32(1.40) * streak) + tube * np.float32(0.25) + rim * np.float32(0.55)
        w_outer = halo * (np.float32(0.60) + np.float32(0.30) * streak)
        w_inner = np.zeros_like(w_core)
        # origin glow where the energy leaves the hands
        og2 = (u[None, :] * u[None, :] + v[:, None] * v[:, None]) * np.float32(1.0 / (w0 * w0))
        w_core += np.exp(og2 * np.float32(-1.4)) * np.float32(1.5)
        w_inner += np.exp(np.sqrt(og2) * np.float32(-0.85)) * np.float32(0.5)
        if head > 0.5:
            self._beam_head(u, v, L, head, flow, seed, turb, w_core, w_inner, w_outer, w_body, cf)
        if dissolve > 1e-3:
            mx3 = np.broadcast_to((u * np.float32(k * 0.16) - np.float32(flow * 1.6))[None, :] * np.float32(_NOISE_CELL), (ch, cw))
            dz = noise.remap(noise.cloud, mx3, (vn * np.float32(0.85) + np.float32(seed * 2.0)) * np.float32(_NOISE_CELL),
                             0.16 * k * _NOISE_CELL * cf)
            mx4 = np.broadcast_to((u * np.float32(k * 0.6) - np.float32(flow * 2.4))[None, :] * np.float32(_NOISE_CELL), (ch, cw))
            dz2 = noise.remap(noise.fine, mx4, vn * np.float32(1.7 * _NOISE_CELL), 0.6 * k * _NOISE_CELL * cf)
            th = dissolve * 1.15 - 0.05
            dzv = (dz - np.float32(0.5)) * np.float32(0.6) + np.float32(0.5)
            keep = _smoothstep(th - 0.10, th + 0.10, dzv * np.float32(0.8) + dz2 * np.float32(0.3))
            keep *= np.float32(1.0 - 0.35 * dissolve)
            for arr in (w_core, w_inner, w_outer, w_body):
                arr *= keep
        # fade out towards the canvas edges (as the GPU quad does)
        edge = _smoothstep(1.0, 0.72, np.abs(v) * np.float32(1.0 / max(hv, 1.0)))[:, None] \
            * _smoothstep(-back, -0.45 * back, u)[None, :]
        canvas = np.zeros((ch, cw, 3), np.float32)
        body_col = _bgr(pal.inner) * 0.75 + _bgr(pal.outer) * 0.25
        g = np.float32(intensity)
        _add_colored(canvas, [w_core * edge, w_inner * edge, w_outer * edge, w_body * edge],
                     [_bgr(pal.core) * g, _bgr(pal.inner) * g, _bgr(pal.outer) * g, body_col * g])
        return canvas

    def _beam_head(self, u: np.ndarray, v: np.ndarray, L: float, head: float, flow: float, seed: float,
                   turb: float, w_core: np.ndarray, w_inner: np.ndarray, w_outer: np.ndarray,
                   w_body: np.ndarray, cf: float) -> None:
        """Bulbous turbulent head at the beam tip (adds into the weight maps in place)."""
        c0 = int(np.searchsorted(u, L - 4.0 * head))
        c1 = int(np.searchsorted(u, L + 4.0 * head))
        if c1 <= c0:
            return
        hu = u[c0:c1] - np.float32(L)
        hr = np.sqrt(hu[None, :] * hu[None, :] + v[:, None] * v[:, None]) * np.float32(1.0 / head)
        ang = np.arctan2(v[:, None] * np.ones_like(hu)[None, :], hu[None, :]).astype(np.float32)
        ang *= np.float32(1.0 / TAU)
        noise = self._noise
        hamp = np.float32(0.38 * turb) * _smoothstep(2.2, 1.4, hr)
        hn = noise.angular(noise.cloud, ang % np.float32(1.0), -flow * 0.35 + seed * 0.071)
        hn = (hn - np.float32(0.5)) * np.float32(0.6) + np.float32(0.5)
        hrr = hr / (np.float32(1.0) + (hn - np.float32(0.5)) * hamp)
        hx = (hu[None, :] * np.float32(1.6 / head) - np.float32(flow * 4.2)) * np.float32(_NOISE_CELL)
        hy = (v[:, None] * np.float32(1.6 / head) + np.float32(seed)) * np.float32(_NOISE_CELL)
        hn2 = noise.remap(noise.cloud2, np.broadcast_to(hx, hr.shape), np.broadcast_to(hy, hr.shape),
                          1.6 / head * _NOISE_CELL * cf)
        hn2 = (hn2 - np.float32(0.5)) * np.float32(0.6) + np.float32(0.5)
        inside = (hrr < 1.05).astype(np.float32)
        hball = _smoothstep(1.02, 0.82, hrr) * inside
        hcore = _smoothstep(0.70, 0.10, hrr + (hn2 - np.float32(0.5)) * np.float32(0.30)) * inside
        hz = np.sqrt(np.clip(np.float32(1.0) - hrr * hrr, 0.0, 1.0))
        hrim = np.power(np.float32(1.0) - hz, np.float32(2.5)) * hball
        w_core[:, c0:c1] += hcore * np.float32(2.2)
        # mix(inner, outer, 0.2) * hball * (...) -> 0.8 inner + 0.2 outer
        hb = hball * (np.float32(0.45) + np.float32(1.2) * hn2)
        w_inner[:, c0:c1] += hb * np.float32(0.8) + hrim * np.float32(1.3)
        w_outer[:, c0:c1] += hb * np.float32(0.2) + np.exp(np.maximum(hrr - np.float32(1.0), 0.0) * np.float32(-2.6)) \
            * np.float32(0.8) * (hr < 4.0)

    def _draw_frontal(self, effect: np.ndarray, fb: FrontalBeam, space: _Space) -> None:
        if not isinstance(fb, FrontalBeam):
            return
        cx, cy = space.pt(fb.center)
        radius = _num(fb.radius, 0.0) * space.k
        fade = clamp(_num(fb.fade, 1.0))
        intensity = _num(fb.intensity, 0.0) * fade
        if not _finite(cx, cy) or radius < 0.5 or intensity <= 1e-3:
            return
        h, w = effect.shape[:2]
        ext = radius * 2.4 + 3.0
        box = _clip_box(cx - ext, cy - ext, cx + ext, cy + ext, w, h)
        if box is None:
            return
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        # a big disc is smooth at this scale: evaluate it on a half-resolution grid
        cf = 2 if radius >= 20.0 and bw * bh > 40_000 else 1
        gw, gh = -(-bw // cf), -(-bh // cf)
        gx = np.float32(x0 - cx) + (np.arange(gw, dtype=np.float32) + np.float32(0.5)) * np.float32(cf)
        gy = np.float32(y0 - cy) + (np.arange(gh, dtype=np.float32) + np.float32(0.5)) * np.float32(cf)
        xx = np.ascontiguousarray(np.broadcast_to(gx[None, :], (gh, gw)))
        yy = np.ascontiguousarray(np.broadcast_to(gy[:, None], (gh, gw)))
        r, ang = cv2.cartToPolar(xx, yy)
        rn = r * np.float32(1.0 / radius)
        theta = ang * np.float32(1.0 / TAU)
        lr = cv2.log(rn + np.float32(0.08))
        t = _num(fb.time, 0.0)
        seed = _num(fb.seed, 0.0)
        pal = _palette(fb.palette)
        # tunnel turbulence: polar noise swirling slowly and rushing outwards (log radius)
        u1 = theta * np.float32(4 * _TEX) + np.float32((t * 0.07 + seed * 0.29) % 1.0 * _TEX) \
            + lr * np.float32(0.35 * _TEX)
        v1 = lr * np.float32(0.55 * _TEX) - np.float32((t * 0.9) % 1.0 * _TEX)
        n1 = self._noise.remap(self._noise.cloud, u1, v1, max(4 * _TEX / (TAU * radius), 0.55 * _TEX / radius) * cf)
        # radial speed streaks: high angular frequency, fast outward motion
        u2 = theta * np.float32(8 * _TEX) + np.float32((seed * 0.71) % 1.0 * _TEX)
        v2 = lr * np.float32(0.22 * _TEX) - np.float32((t * 2.4) % 1.0 * _TEX)
        n2 = self._noise.remap(self._noise.fine, u2, v2, 8 * _TEX / (TAU * radius) * cf)
        c1 = np.clip((n1 - np.float32(0.5)) * np.float32(2.4) + np.float32(0.5), 0.0, 1.0)
        c2 = np.clip((n2 - np.float32(0.5)) * np.float32(2.6) + np.float32(0.5), 0.0, 1.0)
        rt = rn * (np.float32(1.0) + np.float32(0.2) * (n1 - np.float32(0.5)))
        core = np.exp(-np.square(rt * np.float32(1.0 / 0.22))) \
            + np.float32(0.6) * np.exp(-np.square(rn * np.float32(1.0 / 0.10)))
        body = _smoothstep(1.06, 0.55, rt) * (np.float32(0.15) + np.float32(1.4) * c1 * c1)
        body += np.exp(-np.square((rt - np.float32(0.95)) * np.float32(1.0 / 0.10))) \
            * (np.float32(0.4) + np.float32(1.0) * c1)
        halo = np.exp(-np.maximum(rn - np.float32(0.9), 0.0) * np.float32(2.4)) * np.float32(0.6)
        streaks = c2 * c2 * c2 * np.float32(1.6 * max(0.0, _num(fb.streaks, 1.0)))
        streaks *= _smoothstep(0.5, 1.0, rn) * np.exp(-np.maximum(rn - np.float32(1.0), 0.0) * np.float32(1.1))
        dissolve = clamp(_num(fb.dissolve, 0.0))
        if dissolve > 1e-3:
            keep = _smoothstep(dissolve - 0.15, dissolve + 0.12, n1 * np.float32(0.7) + n2 * np.float32(0.3)
                               + np.float32(0.1))
            core *= keep * np.float32((1.0 - dissolve) ** 2)
            body *= keep
            streaks *= keep
            halo *= np.float32(1.0 - dissolve)
        # GPU-matched calibration: white-hot centre, coloured disc (compressed gain ib).
        ib = 0.45 + 0.22 * intensity
        colors = [_bgr(pal.core) * np.float32(2.0 * (0.55 + 0.35 * intensity)),
                  _bgr(pal.inner) * np.float32(0.65 * ib),
                  _bgr(pal.outer) * np.float32(0.35 * ib),
                  (_bgr(pal.core) * 0.5 + _bgr(pal.inner) * 0.6) * np.float32(0.9 * ib)]
        if cf == 1:
            _add_colored(effect[y0:y1, x0:x1], [core, body, halo, streaks], colors)
            return
        small = np.zeros((gh, gw, 3), np.float32)
        _add_colored(small, [core, body, halo, streaks], colors)
        effect[y0:y1, x0:x1] += cv2.resize(small, (gw * cf, gh * cf), interpolation=cv2.INTER_LINEAR)[:bh, :bw]

    # ---- lines: arcs, ribbons, rings

    def _draw_arcs(self, effect: np.ndarray, arcs: Sequence[Arc], space: _Space) -> None:
        h, w = effect.shape[:2]
        for pal, group in _group_by_palette([a for a in arcs if isinstance(a, Arc)]):
            items = []
            for arc in group:
                pts = space.pts(arc.points) if arc.points is not None else np.zeros((0, 2), np.float32)
                pts = pts[np.isfinite(pts).all(axis=1)]
                inten = _num(arc.intensity, 0.0)
                if len(pts) < 2 or inten <= 1e-3:
                    continue
                items.append((pts, max(0.25, _num(arc.width, 1.0) * space.k),
                              min(max(_num(arc.glow, 1.0), 0.3), 6.0), inten))
            if not items:
                continue
            imax = max(it[3] for it in items)
            sigma = max(1.0, float(np.mean([2.2 * it[1] * it[2] for it in items])))
            margin = 3.0 * sigma + max(5.0 * it[1] * it[2] for it in items) + 2.0
            allp = np.concatenate([it[0] for it in items])
            box = _clip_box(float(allp[:, 0].min()) - margin, float(allp[:, 1].min()) - margin,
                            float(allp[:, 0].max()) + margin, float(allp[:, 1].max()) + margin, w, h)
            if box is None:
                continue
            canvas = _LineCanvas(box)
            for pts, wid, glow, inten in items:
                value = 255.0 * inten / imax
                core_w = 2.0 * wid
                p = [canvas.fp(pts)]
                # sub-pixel cores are drawn 1 px wide but dimmer, keeping their total energy
                cv2.polylines(canvas.core, p, False, int(round(value * min(1.0, core_w))),
                              max(1, int(round(core_w))), cv2.LINE_AA, _SHIFT)
                cv2.polylines(canvas.glow, p, False, int(round(value * 0.6)),
                              max(2, int(round(core_w * 2.5 * glow))), cv2.LINE_AA, _SHIFT)
            canvas.composite(effect, _bgr(pal.core) * 2.4 + _bgr(pal.inner) * 0.3,
                             _bgr(pal.inner) * 1.1 + _bgr(pal.outer) * 0.5, sigma, imax)

    def _draw_ribbons(self, effect: np.ndarray, ribbons: Sequence[Ribbon], space: _Space) -> None:
        h, w = effect.shape[:2]
        for pal, group in _group_by_palette([r for r in ribbons if isinstance(r, Ribbon)]):
            items = []
            for rib in group:
                pts = space.pts(rib.points)
                m = len(pts)
                wid = np.asarray(rib.widths, np.float32).reshape(-1)[:m] * np.float32(space.k)
                inten = np.asarray(rib.intensities, np.float32).reshape(-1)[:m]
                m = min(m, len(wid), len(inten))
                if m < 2:
                    continue
                pts, wid, inten = pts[:m], np.nan_to_num(wid[:m]), np.nan_to_num(inten[:m])
                ok = np.isfinite(pts).all(axis=1)
                if ok.sum() < 2 or float(inten.max()) <= 1e-3:
                    continue
                items.append((pts[ok], np.maximum(wid[ok], 0.25), np.maximum(inten[ok], 0.0)))
            if not items:
                continue
            imax = max(float(it[2].max()) for it in items)
            wmax = max(float(it[1].max()) for it in items)
            sigma = max(1.0, float(np.mean([float(it[1].mean()) for it in items])) * 0.7)
            margin = 3.0 * sigma + 3.0 * wmax + 2.0
            allp = np.concatenate([it[0] for it in items])
            box = _clip_box(float(allp[:, 0].min()) - margin, float(allp[:, 1].min()) - margin,
                            float(allp[:, 0].max()) + margin, float(allp[:, 1].max()) + margin, w, h)
            if box is None:
                continue
            canvas = _LineCanvas(box)
            for pts, wid, inten in items:
                p = canvas.fp(pts)
                for j in range(len(p) - 1):
                    a = 0.5 * float(inten[j] + inten[j + 1]) / imax
                    if a <= 0.004:
                        continue
                    full = float(wid[j] + wid[j + 1])           # 2 x mean half-width
                    p0 = (int(p[j, 0]), int(p[j, 1]))
                    p1 = (int(p[j + 1, 0]), int(p[j + 1, 1]))
                    core_w = full * 0.45
                    cv2.line(canvas.core, p0, p1, int(round(255 * a * min(1.0, core_w))),
                             max(1, int(round(core_w))), cv2.LINE_AA, _SHIFT)
                    cv2.line(canvas.glow, p0, p1, int(round(255 * a * 0.6)),
                             max(2, int(round(full * 1.4))), cv2.LINE_AA, _SHIFT)
            canvas.composite(effect, _bgr(pal.core) * 1.5 + _bgr(pal.inner) * 0.3,
                             _bgr(pal.inner) * 0.9 + _bgr(pal.outer) * 0.5, sigma, imax)

    def _draw_rings(self, effect: np.ndarray, rings: Sequence[Ring], space: _Space) -> None:
        h, w = effect.shape[:2]
        row_tex = self._noise.cloud[0]
        for pal, group in _group_by_palette([r for r in rings if isinstance(r, Ring)]):
            items = []
            for ring in group:
                cx, cy = space.pt(ring.center)
                radius = _num(ring.radius, 0.0) * space.k
                inten = _num(ring.intensity, 0.0)
                if not _finite(cx, cy) or radius < 0.5 or inten <= 1e-3:
                    continue
                ax = radius * max(0.05, _num(ring.squash[0], 1.0))
                ay = radius * max(0.05, _num(ring.squash[1], 1.0))
                th = max(1.0, _num(ring.thickness, 1.0) * space.k)
                items.append((cx, cy, ax, ay, th, inten, clamp(_num(ring.noise, 0.5)), _num(ring.time, 0.0)))
            if not items:
                continue
            imax = max(it[5] for it in items)
            sigma = max(1.0, float(np.mean([it[4] for it in items])) * 0.6)
            x0 = min(it[0] - it[2] - 2 * it[4] for it in items) - 3 * sigma
            y0 = min(it[1] - it[3] - 2 * it[4] for it in items) - 3 * sigma
            x1 = max(it[0] + it[2] + 2 * it[4] for it in items) + 3 * sigma
            y1 = max(it[1] + it[3] + 2 * it[4] for it in items) + 3 * sigma
            box = _clip_box(x0, y0, x1, y1, w, h)
            if box is None:
                continue
            canvas = _LineCanvas(box)
            for idx, (cx, cy, ax, ay, th, inten, brk, t) in enumerate(items):
                perimeter = TAU * math.sqrt(0.5 * (ax * ax + ay * ay))
                segs = int(min(72, max(24, perimeter / 8.0)))
                # smooth angular break-up: one periodic texture row, scrolled over time
                row = row_tex[int(hash01(idx, 7) * _TEX) % _TEX]
                pos = (np.arange(segs) * (2.0 * _TEX / segs) + t * 40.0) % _TEX
                i0 = pos.astype(np.int64)
                frac = pos - i0
                nv = row[i0 % _TEX] * (1.0 - frac) + row[(i0 + 1) % _TEX] * frac
                mod = 1.0 - brk * (1.0 - np.clip(nv * 1.8 - 0.35, 0.0, 1.0))
                values = 255.0 * inten / imax * mod
                centre = canvas.local(cx, cy)
                axes = (int(round(ax * _FP)), int(round(ay * _FP)))
                core_t = max(1, int(round(th * 0.45)))
                glow_t = max(2, int(round(th * 1.3)))
                step = 360.0 / segs
                for j in range(segs):
                    v = int(values[j])
                    if v < 2:
                        continue
                    a0 = j * step
                    cv2.ellipse(canvas.core, centre, axes, 0.0, a0, a0 + step + 0.6, v, core_t,
                                cv2.LINE_AA, _SHIFT)
                    cv2.ellipse(canvas.glow, centre, axes, 0.0, a0, a0 + step + 0.6, int(v * 0.7),
                                glow_t, cv2.LINE_AA, _SHIFT)
            canvas.composite(effect, _bgr(pal.core) * 1.4 + _bgr(pal.inner) * 0.4,
                             _bgr(pal.inner) * 0.9 + _bgr(pal.outer) * 0.6, sigma, imax)

    def _draw_radial_flash(self, effect: np.ndarray, fl: Flash, space: _Space) -> None:
        cx, cy = space.pt(fl.center)
        radius = _num(fl.radius, 0.0) * space.k
        inten = _num(fl.intensity, 0.0)
        if not _finite(cx, cy) or radius <= 0.5 or inten <= 1e-4:
            return
        h, w = effect.shape[:2]
        ext = radius * 2.0
        box = _clip_box(cx - ext, cy - ext, cx + ext, cy + ext, w, h)
        if box is None:
            return
        x0, y0, x1, y1 = box
        xx = np.ascontiguousarray(np.broadcast_to((_grid(x0, x1 - x0) - np.float32(cx))[None, :],
                                                  (y1 - y0, x1 - x0)))
        yy = np.ascontiguousarray(np.broadcast_to((_grid(y0, y1 - y0) - np.float32(cy))[:, None],
                                                  (y1 - y0, x1 - x0)))
        dn = cv2.magnitude(xx, yy) * np.float32(1.0 / radius)
        # hot gaussian centre + exponential skirt (reads as light, not as a flat disc),
        # faded to zero at the box edge
        prof = np.float32(0.6) * np.exp(-np.square(dn * np.float32(1.0 / 0.4))) \
            + np.float32(0.4) * np.exp(-dn * np.float32(2.2))
        prof *= _smoothstep(2.0, 1.4, dn)
        try:
            col = _bgr(fl.color) * np.float32(inten)
        except (TypeError, ValueError, IndexError):
            col = np.full(3, inten, np.float32)
        _add_colored(effect[y0:y1, x0:x1], [prof], [col])

    # ---- aura

    @staticmethod
    def _prepare_mask(mask: Any) -> np.ndarray | None:
        if mask is None:
            return None
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0]
        if m.ndim != 2 or m.size == 0:
            return None
        if m.dtype == np.bool_:
            return np.ascontiguousarray(m.astype(np.uint8) * 255)
        if m.dtype != np.uint8:
            f = np.nan_to_num(m.astype(np.float32))
            if float(f.max()) <= 1.0 + 1e-6:
                f = f * 255.0
            return np.clip(f, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(m)

    def _draw_aura(self, effect: np.ndarray, aura: Aura, space: _Space, view: ViewTransform) -> None:
        if not isinstance(aura, Aura):
            return
        intensity = _num(aura.intensity, 0.0)
        mask = self._prepare_mask(aura.mask)
        if mask is None or intensity <= 1e-3:
            return
        eh, ew = effect.shape[:2]
        aw = int(min(ew, max(96, int(_num(getattr(self._quality, "aura_resolution", 256), 256)))))
        ah = max(1, int(round(aw * eh / ew)))
        rx, ry = aw / ew, ah / eh
        reach = max(1.5, _num(aura.flame_height, 0.08) * view.scene_h * space.ay * ry)
        key = (int(aura.mask_id), mask.shape, aw, ah, round(space.ax, 6), round(space.bx, 3),
               round(space.ay, 6), round(space.by, 3), view.scene_w, view.scene_h, round(reach, 2))
        if key != self._aura_key or self._aura_data is None:
            self._aura_data = self._aura_fields(mask, aw, ah, rx, ry, space, view, reach)
            self._aura_key = key
        m, glow, rise = self._aura_data
        t = _num(aura.time, 0.0)
        turb = min(max(_num(aura.turbulence, 1.0), 0.0), 3.0)
        flick = clamp(_num(aura.flicker, 0.5))
        pal = _palette(aura.palette)
        # rising flame noise: stretched vertically, scrolling upwards over time
        fx, fy = _TEX / (aw * 0.45), _TEX / (ah * 1.3)
        speed = (0.55 + 0.6 * flick) * 0.9 * _TEX
        a1 = np.float32([[fx, 0.0, 37.0], [0.0, fy, (t * speed) % _TEX]])
        a2 = np.float32([[fx * 2.1, 0.0, 91.0], [0.0, fy * 1.8, (t * speed * 1.7) % _TEX]])
        n1 = self._noise.warp(self._noise.cloud, a1, (aw, ah), max(fx, fy))
        n2 = self._noise.warp(self._noise.fine, a2, (aw, ah), max(fx * 2.1, fy * 1.8))
        nn = n1 * np.float32(0.62) + n2 * np.float32(0.38)
        nn = np.clip((nn - np.float32(0.5)) * np.float32(2.2 * (0.6 + 0.4 * min(turb, 1.5)))
                     + np.float32(0.5), 0.0, 1.0)
        # noise erosion: close to the body the flame always burns, further out only where the
        # noise is high -> separate tongues that rise and flicker
        flame = _smoothstep(0.0, 0.32, rise - (np.float32(1.0) - nn) * np.float32(0.62))
        hollow = np.float32(1.0) - m                          # hug the silhouette, keep the person visible
        flame *= hollow
        hot = flame * np.clip(glow * np.float32(1.6), 0.0, 1.0)  # hotter near the body, outer colour at the tips
        edge = np.clip(glow * np.float32(2.3) - m * np.float32(1.7), 0.0, 1.0)
        edge *= (np.float32(0.55) + np.float32(0.7) * n2) * hollow
        mod = 1.0 + flick * 0.13 * (math.sin(t * 17.3) + 0.6 * math.sin(t * 29.7 + 1.3))
        inten = intensity * mod
        rgb = np.zeros((ah, aw, 3), np.float32)
        _add_colored(rgb, [flame, hot, edge],
                     [_bgr(pal.outer) * np.float32(0.55 * inten),
                      _bgr(pal.inner) * np.float32(0.45 * inten),
                      (_bgr(pal.inner) * 0.6 + _bgr(pal.core) * 0.3) * np.float32(inten)])
        if (aw, ah) == (ew, eh):
            effect += rgb
        else:
            effect += cv2.resize(rgb, (ew, eh), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _aura_fields(mask: np.ndarray, aw: int, ah: int, rx: float, ry: float, space: _Space,
                     view: ViewTransform, reach: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mask in aura-buffer space + its soft glow + the upward-smeared flame field (cached)."""
        mh, mw = mask.shape
        sx = view.scene_w / mw * space.ax * rx
        sy = view.scene_h / mh * space.ay * ry
        if sx < 0.5 or sy < 0.5:      # much larger mask than needed: area-filter it first
            nw, nh = max(1, int(round(mw * sx))), max(1, int(round(mh * sy)))
            mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_AREA)
            sx, sy = sx * mw / nw, sy * mh / nh
        tx, ty = space.bx * rx, space.by * ry
        m_aff = np.float32([[sx, 0.0, 0.5 * sx + tx - 0.5], [0.0, sy, 0.5 * sy + ty - 0.5]])
        m8 = cv2.warpAffine(mask, m_aff, (aw, ah), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
        m = m8.astype(np.float32) * np.float32(1.0 / 255.0)
        glow = cv2.GaussianBlur(m, (0, 0), max(0.8, reach * 0.33))
        rise = glow.copy()
        for frac, fall in ((0.3, 0.92), (0.6, 0.78), (0.95, 0.6)):   # flames reach upwards
            sh = int(round(reach * frac))
            if 0 < sh < ah:
                np.maximum(rise[:-sh], glow[sh:] * np.float32(fall), out=rise[:-sh])
        rise = cv2.GaussianBlur(rise, (0, 0), max(0.7, reach * 0.2))
        return m, glow, rise

    # ---- bloom / tone mapping

    @staticmethod
    def _bloom(effect: np.ndarray, strength: float) -> None:
        """3-level pyramid bloom (pyrDown + GaussianBlur + pyrUp) added onto the buffer."""
        if strength <= 1e-3:
            return
        h, w = effect.shape[:2]
        levels = []
        cur = cv2.max(cv2.subtract(effect, (_BLOOM_THRESHOLD,) * 3), (0.0, 0.0, 0.0))
        for _ in range(3):
            if cur.shape[0] < 4 or cur.shape[1] < 4:
                break
            cur = cv2.pyrDown(cur)
            levels.append(cur)
        if not levels:
            return
        acc = None
        for i in range(len(levels) - 1, -1, -1):
            lvl = cv2.GaussianBlur(levels[i], (0, 0), _BLOOM_SIGMAS[i])
            if acc is None:
                acc = lvl * np.float32(_BLOOM_WEIGHTS[i])
            else:
                acc = cv2.pyrUp(acc, dstsize=(lvl.shape[1], lvl.shape[0]))
                acc = cv2.scaleAdd(lvl, _BLOOM_WEIGHTS[i], acc)
        up = cv2.pyrUp(acc, dstsize=(w, h))
        cv2.scaleAdd(up, float(strength) * _BLOOM_GAIN, effect, dst=effect)

    @staticmethod
    def _tonemap(effect: np.ndarray, exposure: float) -> np.ndarray:
        """HDR -> 8 bit with white-hot handling (same three regimes as the GPU composite).

        ``m = max(r, g, b)``: moderate energy keeps its hue (``x * (1 - exp(-m)) / m``,
        saturated bodies and halos), brighter energy blends into per-channel
        ``1 - exp(-x)`` (pales towards white) and very bright energy goes white-hot
        (``1 - exp(-m)`` on every channel). Written as ``x * A + (1 - exp(-x)) * B + C``
        with three single-channel maps so it stays a handful of OpenCV passes.
        """
        if abs(exposure - 1.0) > 1e-4:
            effect *= np.float32(max(0.0, exposure))
        np.maximum(effect, 0.0, out=effect)
        b, g, r = cv2.split(effect)
        m = cv2.max(cv2.max(b, g), r)
        tm = cv2.exp(-m)
        np.subtract(np.float32(1.0), tm, out=tm)                     # 1 - exp(-m)
        s1 = _smoothstep(_TM_PALE[0], _TM_PALE[1], m)
        s2 = _smoothstep(_TM_WHITE[0], _TM_WHITE[1], m)
        s2 *= np.float32(_TM_WHITE_MAX)
        keep = np.float32(1.0) - s2
        coef_x = cv2.divide(tm, cv2.max(m, 1e-4))                    # hue-preserving scale
        coef_x *= np.float32(1.0) - s1
        coef_x *= keep
        coef_c = s1 * keep
        const = tm * s2
        chan = cv2.exp(-effect)
        np.subtract(np.float32(1.0), chan, out=chan)                 # 1 - exp(-x)
        out = cv2.multiply(effect, cv2.merge((coef_x, coef_x, coef_x)))
        out += cv2.multiply(chan, cv2.merge((coef_c, coef_c, coef_c)))
        out += cv2.merge((const, const, const))
        return cv2.convertScaleAbs(out, alpha=255.0)

    # ---------------------------------------------------------------- composite

    def _composite(self, frame: np.ndarray, overlay: np.ndarray) -> None:
        """``frame = cam + fx * (1 - 0.5 * cam)`` in 8 bit (the GPU renderer's soft screen blend).

        Pure addition would wash bright skin and walls out to flat white; the soft screen
        keeps the camera detail under the energy while dark areas still get the full glow.
        """
        if self._tmp is None or self._tmp.shape != frame.shape:
            self._tmp = np.empty_like(frame)
        prod = cv2.multiply(overlay, frame, dst=self._tmp, scale=1.0 / 510.0)
        cv2.subtract(overlay, prod, dst=overlay)
        cv2.add(frame, overlay, dst=frame)

    def _draw_sparkles(self, frame: np.ndarray) -> None:
        """Crisp antialiased streaks for the brightest sparks / embers (window resolution).

        Head halves are drawn brighter than tail halves (the GPU sparks fade towards the
        tail). Segments are grouped by quantised brightness and thickness so the whole set
        costs a handful of ``cv2.polylines`` calls instead of one call per particle.
        """
        peak = np.concatenate([c[0] for c in self._sparkles])
        head = np.concatenate([c[1] for c in self._sparkles])
        end = np.concatenate([c[2] for c in self._sparkles])
        core = np.concatenate([c[3] for c in self._sparkles])
        thick = np.concatenate([c[4] for c in self._sparkles])
        dot = np.concatenate([c[5] for c in self._sparkles])
        if peak.size > _SPARKLE_MAX:
            sel = np.argpartition(peak, peak.size - _SPARKLE_MAX)[-_SPARKLE_MAX:]
            head, end, core, thick, dot = head[sel], end[sel], core[sel], thick[sel], dot[sel]
        h, w = frame.shape[:2]
        # streaks: bright head half + dim tail half; dots: one dash at full core brightness
        mid = np.where(dot[:, None], end, (head + end) * np.float32(0.5))
        lo = np.minimum(head, end).min(axis=0) - 3.0
        hi = np.maximum(head, end).max(axis=0) + 3.0
        box = _clip_box(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1]), w, h)
        if box is None:
            return
        x0, y0, x1, y1 = box
        if self._sparkle_layer is None or self._sparkle_layer.shape != frame.shape:
            self._sparkle_layer = np.zeros_like(frame)
        layer = self._sparkle_layer[y0:y1, x0:x1]
        layer.fill(0)
        origin = np.array([x0, y0], np.float32)
        for seg0, seg1, k in ((head, mid, 0.8), (mid, end, 0.25)):
            kk = np.where(dot, np.float32(1.0), np.float32(k))[:, None]
            ldr = np.float32(1.0) - np.exp(-core * kk)                     # tone-mapped core colour
            lum = ldr.max(axis=1)
            level = np.searchsorted(_SPARKLE_LEVELS, lum * np.float32(0.999))
            visible = (lum >= np.float32(0.06)) & ~dot if k < 0.5 else lum >= np.float32(0.06)
            pts = np.stack([seg0 - origin, seg1 - origin], axis=1)        # (n, 2, 2) layer px
            fp = np.round(np.clip((pts - np.float32(0.5)) * np.float32(_FP), -_FP_LIMIT, _FP_LIMIT)
                          ).astype(np.int32)
            for lv in np.unique(level[visible]):
                for tk in (1, 2):
                    grp = np.flatnonzero(visible & (level == lv) & (thick == tk))
                    if grp.size == 0:
                        continue
                    col = ldr[grp].mean(axis=0) * np.float32(255.0)
                    color = (float(col[0]), float(col[1]), float(col[2]))
                    cv2.polylines(layer, list(fp[grp]), False, color, tk, cv2.LINE_AA, _SHIFT)
        roi = frame[y0:y1, x0:x1]
        res = cv2.add(roi, layer, dst=roi)
        if res is not roi:
            roi[...] = res

    def _post(self, frame: np.ndarray, effects: RenderList, view: ViewTransform) -> np.ndarray:
        """Chromatic aberration, uniform flashes, shake and zoom on the composited scene."""
        w, h = self._size
        grade = effects.grade if isinstance(effects.grade, Grade) else Grade()
        chroma = _num(grade.chromatic, 0.0) * float(view.scale)
        if chroma >= 0.5 and getattr(self._quality, "chromatic", False):
            self._guard("chromatic", self._chromatic, frame, min(chroma, 0.03 * w))
        # uniform flashes combine like stacked screen layers: keep = prod(1 - colour * intensity)
        keep = np.ones(3, np.float64)
        for fl in effects.flashes:
            if isinstance(fl, Flash) and (fl.center is None or _num(fl.radius, 0.0) <= 0):
                inten = clamp(_num(fl.intensity, 0.0))
                try:
                    col = np.clip(_bgr(fl.color).astype(np.float64), 0.0, 1.0)
                except (TypeError, ValueError, IndexError):
                    col = np.ones(3, np.float64)
                keep *= 1.0 - col * inten
        flash = np.clip(1.0 - keep, 0.0, 1.0)
        if flash.max() > 1.0 / 512.0:
            m = np.zeros((3, 4), np.float32)                 # out = in + (255 - in) * flash
            for c in range(3):
                m[c, c] = 1.0 - flash[c]
                m[c, 3] = 255.0 * flash[c]
            cv2.transform(frame, m, dst=frame)
        dx = _num(effects.shake[0], 0.0) * float(view.scale)
        dy = _num(effects.shake[1], 0.0) * float(view.scale)
        dx, dy = max(-0.25 * w, min(0.25 * w, dx)), max(-0.25 * h, min(0.25 * h, dy))
        zoom = min(2.0, max(0.5, _num(grade.zoom, 1.0)))
        if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(zoom - 1.0) < 1e-3:
            return frame
        if self._frame2 is None or self._frame2.shape != frame.shape:
            self._frame2 = np.empty_like(frame)
        m = np.float32([[zoom, 0.0, (1.0 - zoom) * w * 0.5 + dx], [0.0, zoom, (1.0 - zoom) * h * 0.5 + dy]])
        out = cv2.warpAffine(frame, m, (w, h), dst=self._frame2, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        # ping-pong the two frame buffers so neither is reallocated
        self._frame2, self._frame = frame, out
        return out

    def _chromatic(self, frame: np.ndarray, amount: float) -> None:
        """Radial chromatic aberration: red scaled outwards, blue inwards (``amount`` px at corners)."""
        h, w = frame.shape[:2]
        z = amount / (0.5 * math.hypot(w, h))
        cx, cy = w * 0.5, h * 0.5
        b, g, r = cv2.split(frame)
        mr = np.float32([[1 + z, 0, -z * cx], [0, 1 + z, -z * cy]])
        mb = np.float32([[1 - z, 0, z * cx], [0, 1 - z, z * cy]])
        flags, border = cv2.INTER_LINEAR, cv2.BORDER_REPLICATE
        r = cv2.warpAffine(r, mr, (w, h), flags=flags, borderMode=border)
        b = cv2.warpAffine(b, mb, (w, h), flags=flags, borderMode=border)
        res = cv2.merge((b, g, r), dst=frame)
        if res is not frame:
            frame[...] = res

    # ------------------------------------------------------------------ output

    def _resolve_target(self) -> pygame.Surface | None:
        if self._target is None:
            return None
        if self._target_is_display:
            # follow pygame.display.set_mode() replacing the display surface
            try:
                current = pygame.display.get_surface() if pygame.display.get_init() else None
            except pygame.error:
                current = None
            if current is not None:
                self._target = current
        return self._target

    def _present(self, frame: np.ndarray, target: pygame.Surface | None) -> pygame.Surface:
        """Write the composited frame into the target (or the internal surface)."""
        surface = target
        if surface is None:
            if self._surface is None or self._surface.get_size() != self._size:
                self._surface = pygame.Surface(self._size, 0, 32, masks=(0xFF0000, 0xFF00, 0xFF, 0))
            surface = self._surface
        if not self._write_direct(surface, frame):
            tmp = pygame.image.frombuffer(np.ascontiguousarray(frame), (frame.shape[1], frame.shape[0]), "BGR")
            surface.blit(tmp, (0, 0))
        return surface

    @staticmethod
    def _write_direct(surface: pygame.Surface, frame: np.ndarray) -> bool:
        """Convert BGR straight into a 32-bit surface's pixel memory (no intermediate copy)."""
        h, w = frame.shape[:2]
        if sys.byteorder != "little" or surface.get_bytesize() != 4 or surface.get_size() != (w, h) \
                or surface.get_pitch() != 4 * w or surface.get_locked():
            return False
        masks = tuple(surface.get_masks()[:3])
        if masks == (0xFF0000, 0xFF00, 0xFF):
            code = cv2.COLOR_BGR2BGRA            # memory order B, G, R, X
        elif masks == (0xFF, 0xFF00, 0xFF0000):
            code = cv2.COLOR_BGR2RGBA            # memory order R, G, B, X
        else:
            return False
        try:
            ref = pygame.surfarray.pixels2d(surface)
        except (pygame.error, ValueError):
            return False
        arr = None
        try:
            arr = ref.T.view(np.uint8).reshape(h, w, 4)       # (h, w, 4) view of the pixels
            res = cv2.cvtColor(frame, code, dst=arr)
            if res is not arr:
                arr[...] = res
        except (ValueError, cv2.error):
            return False
        finally:
            del arr, ref                                       # unlock the surface
        return True
