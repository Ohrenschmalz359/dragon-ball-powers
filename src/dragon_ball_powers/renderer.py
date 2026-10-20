"""OpenGL renderer (ModernGL, OpenGL 3.3 core): camera, HDR energy effects, bloom, UI.

Frame pipeline (one ``render`` call)::

    camera BGR -> RGB8 texture (re-uploaded only when image_id changes)
      |
      v
    scene pass  -> HDR RGBA16F, window size: view transform (cover/contain),
                   shockwave refraction, chromatic aberration, grade, scene lights
    aura pass   -> additive into the scene target: GPU-blurred person mask +
                   animated flame shader
    effects     -> HDR RGBA16F, window size x render_scale, additive (ONE, ONE):
                   rings, beams, frontal beams, orbs, ribbons, arcs, sprites, radial flashes
    bloom       -> 13-tap downsample chain (+ very bright scene pixels), tent upsample
    composite   -> RGBA8 output: tone-mapped effects screen-blended over the camera,
                   zoom punch + camera shake (with overscan), uniform flashes
    UI          -> premultiplied-alpha SDF shapes and cached text textures
    present     -> copy to the window (ctx.screen) unless offscreen

The composite and UI go to an internal RGBA8 target which is then copied to the
window. That costs one cheap full-screen copy but makes :meth:`read_pixels` valid
at any time (also after ``pygame.display.flip()``, e.g. for F12 screenshots).

User effect settings are *not* re-applied here: the effect system already bakes
``EffectsSettings.bloom_strength`` into ``RenderList.bloom`` and ``scene_lighting`` into
the ``Light`` intensities, so the renderer uses the RenderList values as they are
(applying them twice would square the user's setting).

Camera shake is applied exactly once, in the composite, to the camera image, the
effects and the bloom together, so energy stays locked to the hands while the whole
picture shakes.
"""

from __future__ import annotations

import atexit
import logging
import math
import sys
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import cv2
import moderngl
import numpy as np
import pygame

from dragon_ball_powers import gl_shaders
from dragon_ball_powers.config import EffectsSettings
from dragon_ball_powers.fonts import FontManager
from dragon_ball_powers.interfaces import Renderer, RendererError
from dragon_ball_powers.primitives import (
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
    compute_view_transform,
)

log = logging.getLogger(__name__)

MAX_LIGHTS = 8
MAX_REFRACTION_RINGS = 8
TEXT_CACHE_LIMIT = 400
#: Shader time is wrapped: float32 noise coordinates lose precision for large values.
TIME_WRAP = 4096.0
#: Orb quad half-size in orb radii (room for the wide halo and the flares).
ORB_EXTENT = 3.2
#: Scene pixels brighter than this (only reachable through lights/aura) feed the bloom.
SCENE_BLOOM_THRESHOLD = 1.0
SCENE_BLOOM_GAIN = 0.55
#: Overall bloom gain (the upsample chain sums all levels; this normalises it).
BLOOM_GAIN = 0.45
#: Radial flash brightness gain.
FLASH_GAIN = 1.0
#: Text glow textures: blurred alpha is boosted so a soft glow stays visible.
GLOW_BOOST = 2.4
#: Soft aura mask dilation (gain applied after the separable blur).
AURA_DILATE = 1.7
#: Arc strip half-width in core half-widths per unit of ``Arc.glow``.
ARC_STRIP_FACTOR = 5.0

_SPRITE_FLOATS = 11   # pos(2) size(1) rgba(4) vel(2) shape+stretch(2)
_STRIP_FLOATS = 10    # pos(2) v(1) core_fraction(1) core_rgb(3) glow_rgb(3)
_UI_FLOATS = 20       # five vec4 per shape instance
_SEG_PATTERN = np.array([0, 1, 2, 1, 3, 2], dtype=np.int64)

_FULLSCREEN_PROGRAMS = ("scene", "mask_blur", "bloom_down", "bloom_up", "composite", "copy")
_QUAD_PROGRAMS = ("aura", "orb", "beam", "frontal", "flash", "ui_text")
#: Segments of the ring annulus mesh (chord error < 0.2 px for a 600 px ring).
RING_SEGMENTS = 160
_SAMPLER_UNITS: dict[str, dict[str, int]] = {
    "scene": {"u_cam": 0},
    "mask_blur": {"u_src": 0},
    "aura": {"u_hard": 0, "u_soft": 1},
    "bloom_down": {"u_src": 0, "u_scene": 1},
    "bloom_up": {"u_src": 0},
    "composite": {"u_scene": 0, "u_effects": 1, "u_bloom": 2},
    "copy": {"u_src": 0},
    "ui_text": {"u_tex": 0},
}


# ------------------------------------------------------------------ small helpers


def _finite(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _wrap_time(t: Any) -> float:
    return math.fmod(_finite(t), TIME_WRAP)


def _rgb(c: Any, default: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[float, float, float]:
    try:
        return (_finite(c[0]), _finite(c[1]), _finite(c[2]))
    except (TypeError, IndexError):
        return default


def _rgba(c: Any) -> tuple[float, float, float, float]:
    try:
        a = _finite(c[3], 1.0) if len(c) > 3 else 1.0
        return (_finite(c[0]), _finite(c[1]), _finite(c[2]), _clamp(a, 0.0, 1.0))
    except (TypeError, IndexError):
        return (1.0, 1.0, 1.0, 1.0)


def _vec2(v: Any, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
    try:
        return (_finite(v[0], default[0]), _finite(v[1], default[1]))
    except (TypeError, IndexError):
        return default


def _point(v: Any) -> tuple[float, float] | None:
    """A finite 2D point, or None (missing/NaN/inf input): such primitives are skipped
    rather than drawn at a made-up position."""
    try:
        x, y = float(v[0]), float(v[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _sanitize_size(size: Sequence[int]) -> tuple[int, int]:
    try:
        w, h = int(size[0]), int(size[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise RendererError(f"Invalid render size {size!r}") from exc
    return (max(1, w), max(1, h))


def _mix3(a: Sequence[float], b: Sequence[float], t: float) -> np.ndarray:
    return np.asarray(a, np.float32) * (1.0 - t) + np.asarray(b, np.float32) * t


def _gaussian_taps(sigma: float, max_pairs: int = 16) -> tuple[float, np.ndarray, int]:
    """Weights for a separable gaussian using the bilinear "two taps per fetch" trick.

    Returns ``(centre_weight, pairs, count)`` with ``pairs[i] = (offset_texels, weight)``;
    the shader samples ``+offset`` and ``-offset`` with the same weight.
    """
    sigma = max(0.5, float(sigma))
    radius = min(int(math.ceil(sigma * 2.6)), max_pairs * 2)
    offsets = np.arange(radius + 1, dtype=np.float64)
    w = np.exp(-0.5 * (offsets / sigma) ** 2)
    w /= w[0] + 2.0 * w[1:].sum()
    pairs = np.zeros((max_pairs, 2), np.float32)
    count = 0
    i = 1
    while i <= radius and count < max_pairs:
        wa = w[i]
        wb = w[i + 1] if i + 1 <= radius else 0.0
        ws = wa + wb
        pairs[count] = ((i * wa + (i + 1) * wb) / ws if ws > 0 else float(i), ws)
        count += 1
        i += 2
    return float(w[0]), pairs, count


def strip_vertices(points: np.ndarray, half_widths: np.ndarray, core_fraction: np.ndarray,
                   core_rgb: np.ndarray, glow_rgb: np.ndarray, same: np.ndarray) -> np.ndarray:
    """Build triangle vertices for several polylines at once (vectorised, no Python loop).

    ``points`` (P, 2) window px of all polylines concatenated; ``same`` (P-1,) is True
    where points i and i+1 belong to the same polyline. Joints use miter normals
    (clamped so zig-zag lightning cannot explode). Returns (6 * segments, 10) float32
    rows: pos(2), across-coordinate v (+1 left / -1 right), core fraction, core rgb,
    glow rgb.
    """
    p = np.asarray(points, np.float32)
    n = p.shape[0]
    if n < 2 or not np.any(same):
        return np.zeros((0, _STRIP_FLOATS), np.float32)
    d = p[1:] - p[:-1]
    seg_len = np.sqrt(np.einsum("ij,ij->i", d, d))
    ok = same & (seg_len > 1e-4)
    tseg = np.zeros_like(d)
    tseg[ok] = d[ok] / seg_len[ok, None]
    prev = np.zeros_like(p)
    prev[1:] = tseg
    nxt = np.zeros_like(p)
    nxt[:-1] = tseg
    tj = prev + nxt
    tl = np.sqrt(np.einsum("ij,ij->i", tj, tj))
    has_next = np.abs(nxt).sum(axis=1) > 0
    ref = np.where(has_next[:, None], nxt, prev)
    fold = tl < 1e-3                      # end points, 180 degree folds, isolated points
    tj = np.where(fold[:, None], ref, tj / np.maximum(tl, 1e-6)[:, None])
    normal = np.stack([-tj[:, 1], tj[:, 0]], axis=1)
    cos_half = np.abs(np.einsum("ij,ij->i", tj, ref))
    miter = 1.0 / np.clip(cos_half, 0.55, 1.0)
    off = normal * (np.asarray(half_widths, np.float32) * miter)[:, None]
    verts = np.empty((n, 2, _STRIP_FLOATS), np.float32)
    verts[:, 0, 0:2] = p + off
    verts[:, 1, 0:2] = p - off
    verts[:, 0, 2] = 1.0
    verts[:, 1, 2] = -1.0
    verts[:, :, 3] = np.asarray(core_fraction, np.float32)[:, None]
    verts[:, :, 4:7] = np.asarray(core_rgb, np.float32)[:, None, :]
    verts[:, :, 7:10] = np.asarray(glow_rgb, np.float32)[:, None, :]
    seg = np.flatnonzero(same)
    idx = (2 * seg[:, None] + _SEG_PATTERN[None, :]).ravel()
    return verts.reshape(2 * n, _STRIP_FLOATS)[idx]


class _RateLimitedLog:
    """Log the same warning at most once every ``interval`` seconds."""

    def __init__(self, interval: float = 10.0) -> None:
        self._interval = interval
        self._last: dict[str, float] = {}

    def warning(self, key: str, message: str, *args: Any) -> None:
        now = time.perf_counter()
        if now - self._last.get(key, -1e9) >= self._interval:
            self._last[key] = now
            log.warning(message, *args)


# ------------------------------------------------------------------- GL wrappers


class _Prog:
    """A linked program plus the uniforms that survived linking.

    Drivers optimise unused uniforms away, so setters silently ignore names that
    are not active instead of raising.
    """

    __slots__ = ("name", "program", "_uniforms")

    def __init__(self, ctx: moderngl.Context, name: str) -> None:
        vs, fs = gl_shaders.program_sources(name)
        try:
            self.program = ctx.program(vertex_shader=vs, fragment_shader=fs)
        except moderngl.Error as exc:
            raise RendererError(f"OpenGL shader program {name!r} failed to compile or link:\n{exc}") from exc
        self.name = name
        self._uniforms: dict[str, moderngl.Uniform] = {}
        for key in self.program:
            member = self.program[key]
            if isinstance(member, moderngl.Uniform):
                self._uniforms[key] = member

    def has(self, name: str) -> bool:
        return name in self._uniforms

    def set(self, name: str, value: Any) -> None:
        uniform = self._uniforms.get(name)
        if uniform is not None:
            uniform.value = value

    def write(self, name: str, data: np.ndarray) -> None:
        """Write a uniform array (only as many elements as the linked program kept)."""
        uniform = self._uniforms.get(name)
        if uniform is not None:
            count = max(1, int(uniform.array_length))
            uniform.write(np.ascontiguousarray(data[:count], dtype=np.float32).tobytes())

    def release(self) -> None:
        self.program.release()


@dataclass
class _Target:
    """A texture with a framebuffer around it."""

    texture: moderngl.Texture
    fbo: moderngl.Framebuffer

    @property
    def size(self) -> tuple[int, int]:
        return tuple(self.texture.size)  # type: ignore[return-value]

    def release(self) -> None:
        self.fbo.release()
        self.texture.release()


def _make_target(ctx: moderngl.Context, size: tuple[int, int], components: int = 4,
                 dtype: str = "f2") -> _Target:
    tex = ctx.texture(size, components, dtype=dtype)
    try:
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        fbo = ctx.framebuffer(color_attachments=[tex])
    except Exception:
        tex.release()
        raise
    fbo.clear(0.0, 0.0, 0.0, 0.0)
    return _Target(tex, fbo)


class _DynamicBuffer:
    """Streaming vertex buffer: orphan + write every frame, grows geometrically."""

    def __init__(self, ctx: moderngl.Context, initial_bytes: int) -> None:
        self._ctx = ctx
        self.buffer = ctx.buffer(reserve=max(256, int(initial_bytes)), dynamic=True)

    def upload(self, data: np.ndarray) -> bool:
        """Upload ``data``; True when the GL buffer was re-created (VAOs must be rebuilt)."""
        nbytes = int(data.nbytes)
        recreated = False
        if nbytes > self.buffer.size:
            new_size = max(nbytes, self.buffer.size * 2)
            self.buffer.release()
            self.buffer = self._ctx.buffer(reserve=new_size, dynamic=True)
            recreated = True
        else:
            self.buffer.orphan()
        if nbytes:
            self.buffer.write(np.ascontiguousarray(data))
        return recreated

    def release(self) -> None:
        self.buffer.release()


@dataclass
class _TextTexture:
    texture: moderngl.Texture
    width: int
    height: int
    pad: int = 0          # glow textures are padded by this many px on every side


class _TextCache:
    """LRU cache of text coverage textures keyed by (text, size, font[, glow radius]).

    Evicted textures are released immediately so GPU memory stays bounded.
    """

    def __init__(self, ctx: moderngl.Context, fonts: FontManager, limit: int = TEXT_CACHE_LIMIT) -> None:
        self._ctx = ctx
        self._fonts = fonts
        self._limit = max(2, int(limit))
        self._entries: OrderedDict[tuple, _TextTexture] = OrderedDict()
        self.created = 0
        self.evicted = 0

    def __len__(self) -> int:
        return len(self._entries)

    def text(self, text: str, size: int, font: str) -> _TextTexture:
        key = ("t", text, size, font)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry
        entry = self._make(self._alpha(text, size, font), 0)
        self._insert(key, entry)
        return entry

    def glow(self, text: str, size: int, font: str, radius: float) -> _TextTexture:
        r = max(0.5, round(float(radius) * 2.0) / 2.0)
        key = ("g", text, size, font, r)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            return entry
        pad = int(math.ceil(r * 2.0)) + 2
        alpha = np.pad(self._alpha(text, size, font), pad)
        blurred = cv2.GaussianBlur(alpha, (0, 0), sigmaX=max(0.5, r * 0.55), sigmaY=max(0.5, r * 0.55))
        boosted = np.clip(blurred.astype(np.float32) * GLOW_BOOST, 0.0, 255.0).astype(np.uint8)
        entry = self._make(boosted, pad)
        self._insert(key, entry)
        return entry

    def _alpha(self, text: str, size: int, font: str) -> np.ndarray:
        surface = self._fonts.render(text, size, font)
        w, h = surface.get_size()
        raw = pygame.image.tobytes(surface, "RGBA")
        return np.frombuffer(raw, np.uint8).reshape(h, w, 4)[:, :, 3].copy()

    def _make(self, alpha: np.ndarray, pad: int) -> _TextTexture:
        h, w = alpha.shape
        tex = self._ctx.texture((w, h), 1, np.ascontiguousarray(alpha).tobytes(), alignment=1)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return _TextTexture(tex, w, h, pad)

    def _insert(self, key: tuple, entry: _TextTexture) -> None:
        self._entries[key] = entry
        self.created += 1
        while len(self._entries) > self._limit:
            _, old = self._entries.popitem(last=False)
            old.texture.release()
            self.evicted += 1

    def release(self) -> None:
        for entry in self._entries.values():
            entry.texture.release()
        self._entries.clear()


# ------------------------------------------------------------------- the renderer

#: Renderers that own a headless context. Some drivers (seen with Mesa D3D12 on WSL)
#: crash at interpreter exit when a context is still alive, so they are released at exit.
_OWNED_CONTEXT_RENDERERS: "weakref.WeakSet[GLRenderer]" = weakref.WeakSet()


@atexit.register
def _release_owned_renderers() -> None:
    for renderer in list(_OWNED_CONTEXT_RENDERERS):
        try:
            renderer.release()
        except Exception:  # noqa: BLE001 - interpreter is shutting down
            pass


class GLRenderer(Renderer):
    """OpenGL 3.3 core renderer. See the module docstring for the frame pipeline.

    ``ctx`` is a ModernGL context: ``moderngl.create_context()`` for a pygame
    OpenGL window (output goes to ``ctx.screen``), or any context with
    ``offscreen=True`` (output stays in an internal RGBA8 framebuffer; use
    :meth:`read_pixels`). Raises :class:`RendererError` if the context is too old or
    a shader fails to compile (the message contains the driver log).
    """

    name = "OpenGL"

    def __init__(self, ctx: moderngl.Context, size: tuple[int, int], quality: QualityPreset,
                 fonts: FontManager, settings: EffectsSettings | None = None, offscreen: bool = False) -> None:
        self.ctx = ctx
        self._released = False
        self._owns_ctx = False
        self._fonts = fonts
        self._settings = settings
        self._quality = quality
        self._size = _sanitize_size(size)
        self._offscreen = bool(offscreen)
        self._info: dict[str, str] = {"renderer": self.name}
        self._warn = _RateLimitedLog()
        # GL objects (all None-safe so release() works after a failed constructor)
        self._progs: dict[str, _Prog] = {}
        self._vaos: dict[str, moderngl.VertexArray] = {}
        self._quad_vbo: moderngl.Buffer | None = None
        self._sprite_buf: _DynamicBuffer | None = None
        self._sprite_vao: moderngl.VertexArray | None = None
        self._strip_buf: _DynamicBuffer | None = None
        self._strip_vao: moderngl.VertexArray | None = None
        self._ui_buf: _DynamicBuffer | None = None
        self._ui_vao: moderngl.VertexArray | None = None
        self._scene_t: _Target | None = None
        self._fx_t: _Target | None = None
        self._out_t: _Target | None = None
        self._bloom: list[_Target] = []
        self._cam_tex: moderngl.Texture | None = None        # texture of the current frame
        self._cam_texs: list[moderngl.Texture | None] = [None, None]
        self._cam_slot = 0
        self._cam_id: int | None = None
        self._ring_vbo: moderngl.Buffer | None = None
        self._mask_tex: moderngl.Texture | None = None
        self._mask_tmp: _Target | None = None
        self._mask_soft: _Target | None = None
        self._mask_id: int | None = None
        self._mask_shape: tuple[int, int] = (0, 0)
        self._mask_bbox: tuple[float, float, float, float] | None = None
        self._mask_sigma = -1.0
        self._mask_blur_dirty = True
        self._text: _TextCache | None = None
        self._fb0: moderngl.Framebuffer | None = None
        # CPU staging (re-used every frame, grown when needed)
        self._sprite_data = np.zeros((4096, _SPRITE_FLOATS), np.float32)
        self._ui_data = np.zeros((512, _UI_FLOATS), np.float32)
        self._light_a = np.zeros((MAX_LIGHTS, 4), np.float32)
        self._light_c = np.zeros((MAX_LIGHTS, 3), np.float32)
        self._ring_a = np.zeros((MAX_REFRACTION_RINGS, 4), np.float32)
        self._ring_b = np.zeros((MAX_REFRACTION_RINGS, 4), np.float32)
        # frame state
        self._fx_dirty = True
        self._bloom_dirty = True
        self._shake_env = 0.0
        self._last_frame_t: float | None = None
        self._stats = {"frames": 0, "camera_uploads": 0, "mask_uploads": 0, "mask_blurs": 0}
        self.last_cpu_ms = 0.0
        try:
            self._init_gl()
        except RendererError:
            self.release()
            raise
        except Exception as exc:  # noqa: BLE001 - any driver failure becomes a RendererError
            self.release()
            raise RendererError(f"OpenGL renderer initialisation failed: {exc}") from exc

    # ------------------------------------------------------------- construction

    @classmethod
    def create_offscreen(cls, size: tuple[int, int], quality: QualityPreset, fonts: FontManager,
                         settings: EffectsSettings | None = None) -> "GLRenderer":
        """Create a renderer with its own headless context rendering into an RGBA8 FBO."""
        ctx = _create_standalone_context()
        try:
            renderer = cls(ctx, size, quality, fonts, settings, offscreen=True)
        except Exception:
            ctx.release()
            raise
        renderer._owns_ctx = True
        _OWNED_CONTEXT_RENDERERS.add(renderer)
        return renderer

    def _init_gl(self) -> None:
        ctx = self.ctx
        if ctx.version_code < 330:
            raise RendererError(f"OpenGL 3.3 is required, the context provides {ctx.version_code / 100:.1f}")
        if not self._offscreen and ctx.screen is None:
            log.warning("The OpenGL context has no default framebuffer; rendering offscreen")
            self._offscreen = True
        info = ctx.info
        self._info = {
            "renderer": self.name,
            "gl_vendor": str(info.get("GL_VENDOR", "")),
            "gl_renderer": str(info.get("GL_RENDERER", "")),
            "gl_version": str(info.get("GL_VERSION", "")),
        }
        self._max_tex = max(1024, int(info.get("GL_MAX_TEXTURE_SIZE", 4096) or 4096))
        # Framebuffer 0 is always a valid binding (window or headless). ModernGL restores the
        # previously bound framebuffer after clear(); if that one was just deleted the driver
        # raises GL_INVALID_OPERATION, so targets are only deleted while FB 0 is bound.
        self._fb0 = ctx.screen if ctx.screen is not None else ctx.detect_framebuffer(0)
        for name in gl_shaders.PROGRAMS:
            self._progs[name] = _Prog(ctx, name)
        for name, units in _SAMPLER_UNITS.items():
            for uniform, unit in units.items():
                self._progs[name].set(uniform, unit)
        self._quad_vbo = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], np.float32).tobytes())
        for name in _FULLSCREEN_PROGRAMS:
            self._vaos[name] = ctx.vertex_array(self._progs[name].program, [])
        for name in _QUAD_PROGRAMS:
            self._vaos[name] = ctx.vertex_array(self._progs[name].program,
                                                [(self._quad_vbo, "2f", "in_corner")], skip_errors=True)
        seg = np.arange(RING_SEGMENTS + 1, dtype=np.float32) / RING_SEGMENTS
        ring = np.zeros((RING_SEGMENTS + 1, 2, 2), np.float32)
        ring[:, :, 0] = seg[:, None]
        ring[:, 1, 1] = 1.0
        self._ring_vbo = ctx.buffer(ring.tobytes())
        self._vaos["ring"] = ctx.vertex_array(self._progs["ring"].program, [(self._ring_vbo, "2f", "in_ring")],
                                              skip_errors=True)
        self._sprite_buf = _DynamicBuffer(ctx, self._sprite_data.nbytes)
        self._sprite_vao = self._make_sprite_vao()
        self._strip_buf = _DynamicBuffer(ctx, 4096 * _STRIP_FLOATS * 4)
        self._strip_vao = self._make_strip_vao()
        self._ui_buf = _DynamicBuffer(ctx, self._ui_data.nbytes)
        self._ui_vao = self._make_ui_vao()
        self._text = _TextCache(ctx, self._fonts)
        self._create_window_targets()
        self._create_effect_targets()
        self._warm_up()
        log.info("OpenGL renderer ready: %s | %s | %s (%dx%d, quality %s, %s)", self._info["gl_vendor"],
                 self._info["gl_renderer"], self._info["gl_version"], self._size[0], self._size[1],
                 self._quality.name, "offscreen" if self._offscreen else "window")

    def _warm_up(self) -> None:
        """Draw one throw-away frame that uses every program, blend mode and target format.

        Drivers (notably translation layers) build pipeline state lazily on first use,
        which would otherwise stall the frame in which, say, the first beam is fired.
        It also proves the whole pipeline works, so a broken driver fails here (and the
        factory can fall back to the CPU renderer) instead of in the middle of a session.
        """
        t0 = time.perf_counter()
        pal = Palette("warmup", "warm-up", (1.0, 1.0, 1.0), (0.5, 0.8, 1.0), (0.1, 0.3, 1.0), (0.8, 0.9, 1.0),
                      (0.4, 0.8, 1.0))
        w, h = 64, 36
        rl = RenderList()
        rl.orbs.append(Orb((32, 18), 6, pal, charge=1.0, flare=1.0, instability=1.0))
        rl.beams.append(Beam((5, 18), (1.0, 0.0), 40, 3, 4, pal, head_radius=4, dissolve=0.3))
        rl.frontal_beams.append(FrontalBeam((32, 18), 5, pal, dissolve=0.3))
        rl.rings.append(Ring((32, 18), 10, 2, pal, refraction=2.0))
        line = np.array([[5, 5], [30, 10], [59, 5]], np.float32)
        rl.arcs.append(Arc(line, 1.0, pal))
        rl.ribbons.append(Ribbon(line + 10, np.full(3, 2.0, np.float32), np.ones(3, np.float32), pal))
        for shape in SpriteShape:
            rl.sprites.append(SpriteBatch(shape, np.array([[20.0, 20.0]], np.float32), np.array([3.0], np.float32),
                                          np.ones((1, 4), np.float32), np.array([[50.0, 0.0]], np.float32), 0.02))
        rl.flashes.append(Flash(0.1, center=(32, 18), radius=8))
        rl.flashes.append(Flash(0.1))
        rl.lights.append(Light((32, 18), 10, (1.0, 1.0, 1.0)))
        rl.grade = Grade(darken=0.1, desaturate=0.1, tint_strength=0.1, vignette=0.1, chromatic=1.0, zoom=1.01)
        rl.shake = (1.0, 1.0)
        mask = np.zeros((18, 32), np.uint8)
        mask[6:18, 12:20] = 255
        rl.aura = Aura(mask, 0, pal)
        ui = UIDrawList()
        ui.add(UIRect(1, 1, 10, 6, (1, 1, 1, 0.5), 2, (1, 1, 1, 1), 1, (0, 0, 0, 0.5)))
        ui.add(UICircle(20, 5, 3, (1, 1, 1, 1), 1))
        ui.add(UIArc(30, 5, 3, 0.0, 2.0, (1, 1, 1, 1), 1))
        ui.add(UILine([(1, 30), (10, 30)], (1, 1, 1, 1), 1))
        ui.add(UIText("0", 40, 2, 8, (1, 1, 1, 1), glow=2.0, shadow=True))
        image = np.zeros((h, w, 3), np.uint8)
        view = compute_view_transform(w, h, self._size[0], self._size[1], "cover")
        offscreen = self._offscreen
        self._offscreen = True                 # never present the warm-up frame
        try:
            self.render(SceneInput(image, -1, (w, h), 0.0), rl, ui, view)
            self.ctx.finish()
        finally:
            self._offscreen = offscreen
        # Forget everything the warm-up frame left behind so it cannot alias real input.
        self._park()
        for tex in (*self._cam_texs, self._mask_tex):
            if tex is not None:
                tex.release()
        for target in (self._mask_tmp, self._mask_soft):
            if target is not None:
                target.release()
        self._cam_texs = [None, None]
        self._cam_tex = self._mask_tex = None
        self._mask_tmp = self._mask_soft = None
        self._cam_id = self._mask_id = None
        self._mask_shape = (0, 0)
        self._mask_bbox = None
        self._mask_sigma = -1.0
        assert self._text is not None
        self._text.release()
        self._text.created = self._text.evicted = 0
        assert self._out_t is not None
        self._out_t.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self._shake_env = 0.0
        self._last_frame_t = None
        self._stats = {key: 0 for key in self._stats}
        log.debug("OpenGL pipeline warm-up took %.0f ms", (time.perf_counter() - t0) * 1000.0)

    def _make_sprite_vao(self) -> moderngl.VertexArray:
        assert self._quad_vbo is not None and self._sprite_buf is not None
        return self.ctx.vertex_array(
            self._progs["sprite"].program,
            [(self._quad_vbo, "2f", "in_corner"),
             (self._sprite_buf.buffer, "2f 1f 4f 2f 2f/i", "in_pos", "in_size", "in_color", "in_vel", "in_shape")],
            skip_errors=True)

    def _make_strip_vao(self) -> moderngl.VertexArray:
        assert self._strip_buf is not None
        return self.ctx.vertex_array(
            self._progs["strip"].program,
            [(self._strip_buf.buffer, "2f 1f 1f 3f 3f", "in_pos", "in_v", "in_core", "in_core_rgb", "in_glow_rgb")],
            skip_errors=True)

    def _make_ui_vao(self) -> moderngl.VertexArray:
        assert self._quad_vbo is not None and self._ui_buf is not None
        return self.ctx.vertex_array(
            self._progs["ui_shape"].program,
            [(self._quad_vbo, "2f", "in_corner"),
             (self._ui_buf.buffer, "4f 4f 4f 4f 4f/i", "in_a", "in_b", "in_c", "in_d", "in_e")],
            skip_errors=True)

    def _park(self) -> None:
        """Bind framebuffer 0 so render targets can be deleted safely (see ``_init_gl``)."""
        if self._fb0 is not None:
            self._fb0.use()

    def _create_window_targets(self) -> None:
        self._park()
        for t in (self._scene_t, self._out_t):
            if t is not None:
                t.release()
        self._scene_t = self._out_t = None
        size = (min(self._size[0], self._max_tex), min(self._size[1], self._max_tex))
        self._scene_t = _make_target(self.ctx, size, 4, "f2")
        self._out_t = _make_target(self.ctx, size, 4, "f1")

    def _effect_size(self) -> tuple[int, int]:
        scale = _clamp(_finite(self._quality.render_scale, 1.0), 0.25, 2.0)
        w = max(1, min(self._max_tex, int(round(self._size[0] * scale))))
        h = max(1, min(self._max_tex, int(round(self._size[1] * scale))))
        return (w, h)

    def _create_effect_targets(self) -> None:
        self._park()
        if self._fx_t is not None:
            self._fx_t.release()
            self._fx_t = None
        for t in self._bloom:
            t.release()
        self._bloom = []
        size = self._effect_size()
        self._fx_t = _make_target(self.ctx, size, 4, "f2")
        levels = max(1, min(int(self._quality.bloom_levels), 10))
        w, h = size
        for _ in range(levels):
            w, h = max(1, (w + 1) // 2), max(1, (h + 1) // 2)
            self._bloom.append(_make_target(self.ctx, (w, h), 4, "f2"))
            if w <= 2 and h <= 2:
                break
        self._fx_dirty = False
        self._bloom_dirty = False

    # ------------------------------------------------------------- public API

    @property
    def info(self) -> dict[str, str]:
        return dict(self._info)

    @property
    def size(self) -> tuple[int, int]:
        """Current output (window) size in px."""
        return self._size

    @property
    def quality(self) -> QualityPreset:
        return self._quality

    @property
    def offscreen(self) -> bool:
        return self._offscreen

    @property
    def released(self) -> bool:
        return self._released

    @property
    def stats(self) -> dict[str, int]:
        """Counters for diagnostics/tests (frames, camera/mask uploads, text textures)."""
        out = dict(self._stats)
        out["text_textures"] = len(self._text) if self._text is not None else 0
        out["text_textures_created"] = self._text.created if self._text is not None else 0
        out["text_textures_evicted"] = self._text.evicted if self._text is not None else 0
        return out

    def apply_settings(self, settings: EffectsSettings | None) -> None:
        """Store new effect settings (kept for API symmetry with the CPU renderer).

        The values that affect rendering (bloom strength, scene lighting) already arrive
        baked into each ``RenderList`` from the effect system, so nothing else changes.
        """
        self._settings = settings

    def resize(self, width: int, height: int) -> None:
        """Re-create every size-dependent render target for the new window size."""
        self._check_alive()
        size = _sanitize_size((width, height))
        if size == self._size:
            return
        self._size = size
        try:
            self._create_window_targets()
            self._create_effect_targets()
        except moderngl.Error as exc:
            raise RendererError(f"Could not resize OpenGL render targets to {size}: {exc}") from exc

    def set_quality(self, preset: QualityPreset) -> None:
        """Change effect resolution, bloom levels, noise octaves, refraction/chromatic flags."""
        self._check_alive()
        old = self._quality
        self._quality = preset
        if (self._fx_t is None or old.render_scale != preset.render_scale
                or old.bloom_levels != preset.bloom_levels):
            try:
                self._create_effect_targets()
            except moderngl.Error as exc:
                raise RendererError(f"Could not apply quality preset {preset.name!r}: {exc}") from exc

    def read_pixels(self) -> np.ndarray:
        """The last rendered frame (including UI) as (H, W, 3) RGB uint8, top row first."""
        self._check_alive()
        assert self._out_t is not None
        w, h = self._out_t.size
        data = self._out_t.fbo.read(components=3, alignment=1)
        img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(img[::-1])

    def release(self) -> None:
        """Free every GL object (and the context if this renderer created it). Idempotent."""
        if self._released:
            return
        self._released = True
        try:
            self._park()
        except Exception as exc:  # noqa: BLE001 - the context may already be gone
            log.debug("Could not bind framebuffer 0 before releasing: %s", exc)
        objects: list[Any] = [self._text, self._sprite_vao, self._strip_vao, self._ui_vao,
                              self._sprite_buf, self._strip_buf, self._ui_buf, self._scene_t, self._fx_t,
                              self._out_t, self._cam_texs[0], self._cam_texs[1], self._mask_tex,
                              self._mask_tmp, self._mask_soft]
        objects.extend(self._bloom)
        objects.extend(self._vaos.values())
        objects.extend([self._quad_vbo, self._ring_vbo])
        objects.extend(self._progs.values())
        for obj in objects:
            if obj is None:
                continue
            try:
                obj.release()
            except Exception as exc:  # noqa: BLE001 - keep releasing the rest
                log.debug("Ignoring error while releasing %r: %s", obj, exc)
        self._bloom = []
        self._vaos = {}
        self._progs = {}
        self._text = None
        self._sprite_vao = self._strip_vao = self._ui_vao = None
        self._sprite_buf = self._strip_buf = self._ui_buf = None
        self._scene_t = self._fx_t = self._out_t = None
        self._cam_tex = self._mask_tex = None
        self._cam_texs = [None, None]
        self._mask_tmp = self._mask_soft = None
        self._quad_vbo = self._ring_vbo = None
        self._fb0 = None
        if self._owns_ctx:
            _OWNED_CONTEXT_RENDERERS.discard(self)
            try:
                self.ctx.release()
            except Exception as exc:  # noqa: BLE001
                log.debug("Ignoring error while releasing the GL context: %s", exc)
        log.debug("OpenGL renderer released")

    def _check_alive(self) -> None:
        if self._released:
            raise RendererError("The OpenGL renderer has been released")

    # ------------------------------------------------------------------ render

    def render(self, scene: SceneInput, effects: RenderList, ui: UIDrawList,
               view: ViewTransform) -> None:
        """Draw one complete frame (see module docstring)."""
        self._check_alive()
        t_start = time.perf_counter()
        fx = effects if effects is not None else RenderList()
        try:
            ctx = self.ctx
            ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            ctx.blend_equation = moderngl.FUNC_ADD
            has_cam = self._upload_camera(scene)
            t = _wrap_time(scene.time)
            self._scene_pass(scene, fx, view, t, has_cam)
            aura_drawn = fx.aura is not None and self._aura_pass(fx.aura, view)
            fx_drawn = self._effects_pass(fx, view)
            bloom_on = self._bloom_pass(fx_drawn or aura_drawn or bool(fx.lights))
            self._composite_pass(fx, view, bloom_on, t_start)
            self._ui_pass(ui)
            if not self._offscreen:
                self._present()
        except moderngl.Error as exc:
            raise RendererError(f"OpenGL rendering failed: {exc}") from exc
        self._stats["frames"] += 1
        self.last_cpu_ms = (time.perf_counter() - t_start) * 1000.0

    # ----------------------------------------------------------- camera upload

    def _upload_camera(self, scene: SceneInput) -> bool:
        """Upload the camera frame if it changed. Returns True when a camera texture is usable."""
        image = scene.image
        if image is None:
            return False
        image_id = int(scene.image_id) if scene.image_id is not None else -1
        if (image_id >= 0 and image_id == self._cam_id and self._cam_tex is not None
                and tuple(np.shape(image)[:2]) == (self._cam_tex.height, self._cam_tex.width)):
            return True
        img = self._prepare_image(image)
        if img is None:
            return self._cam_tex is not None
        h, w = img.shape[:2]
        # Two textures are used alternately: writing into the one the previous frame did
        # not sample avoids a pipeline stall. (Pixel-buffer uploads were measured to be
        # pathologically slow on some drivers, so plain texture writes are used.)
        self._cam_slot ^= 1
        tex = self._cam_texs[self._cam_slot]
        if tex is None or tex.size != (w, h):
            if tex is not None:
                tex.release()
            tex = self.ctx.texture((w, h), 3, img, alignment=1)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            self._cam_texs[self._cam_slot] = tex
        else:
            tex.write(img, alignment=1)
        self._cam_tex = tex
        self._cam_id = image_id
        self._stats["camera_uploads"] += 1
        return True

    def _prepare_image(self, image: np.ndarray) -> np.ndarray | None:
        """Coerce the camera image into a contiguous BGR uint8 array the GPU accepts."""
        img = np.asarray(image)
        try:
            if img.dtype != np.uint8:
                self._warn.warning("cam_dtype", "Camera image dtype %s converted to uint8", img.dtype)
                if np.issubdtype(img.dtype, np.floating) and img.size and float(np.nanmax(img)) <= 1.0:
                    img = img * 255.0
                img = np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img.ndim != 3 or img.shape[2] != 3 or img.shape[0] < 1 or img.shape[1] < 1:
                self._warn.warning("cam_shape", "Unsupported camera image shape %s; frame skipped", img.shape)
                return None
            h, w = img.shape[:2]
            if w > self._max_tex or h > self._max_tex:
                f = self._max_tex / max(w, h)
                img = cv2.resize(img, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        except (cv2.error, ValueError, TypeError) as exc:
            self._warn.warning("cam_convert", "Could not convert the camera image: %s", exc)
            return None
        return np.ascontiguousarray(img)

    # -------------------------------------------------------------- scene pass

    def _scene_pass(self, scene: SceneInput, fx: RenderList, view: ViewTransform, t: float,
                    has_cam: bool) -> None:
        assert self._scene_t is not None
        p = self._progs["scene"]
        self._scene_t.fbo.use()
        self.ctx.disable(moderngl.BLEND)
        if has_cam and self._cam_tex is not None:
            self._cam_tex.use(0)
        w, h = self._scene_t.size
        p.set("u_has_cam", 1 if has_cam else 0)
        p.set("u_win_size", (float(w), float(h)))
        p.set("u_view", (max(_finite(view.scale, 1.0), 1e-6), _finite(view.offset_x), _finite(view.offset_y)))
        p.set("u_scene_size", (float(max(1, view.scene_w)), float(max(1, view.scene_h))))
        p.set("u_time", t)
        g = fx.grade if isinstance(fx.grade, Grade) else Grade()
        p.set("u_darken", _clamp(_finite(g.darken), 0.0, 1.0))
        p.set("u_desaturate", _clamp(_finite(g.desaturate), 0.0, 1.0))
        p.set("u_tint", _rgb(g.tint))
        p.set("u_tint_strength", _clamp(_finite(g.tint_strength), 0.0, 1.0))
        p.set("u_vignette", _clamp(_finite(g.vignette), 0.0, 1.0))
        chroma = max(0.0, _finite(g.chromatic)) if self._quality.chromatic else 0.0
        p.set("u_chromatic", min(chroma, 64.0))
        n_lights = self._fill_lights(fx.lights)
        p.set("u_light_count", n_lights)
        if n_lights:
            p.write("u_light_a", self._light_a)
            p.write("u_light_c", self._light_c)
        n_rings = self._fill_refraction(fx.rings) if (has_cam and self._quality.refraction) else 0
        p.set("u_ring_count", n_rings)
        if n_rings:
            p.write("u_ring_a", self._ring_a)
            p.write("u_ring_b", self._ring_b)
        self._vaos["scene"].render(moderngl.TRIANGLES, vertices=3)

    def _fill_lights(self, lights: Iterable[Light]) -> int:
        valid: list[tuple[float, float, float, float, tuple[float, float, float]]] = []
        for light in lights:
            try:
                pos = _point(light.position)
                radius = max(1.0, _finite(light.radius, 1.0))
                intensity = min(_finite(light.intensity), 16.0)
                if pos is not None and intensity > 0.0:
                    valid.append((pos[0], pos[1], radius, intensity, _rgb(light.color)))
            except AttributeError:
                self._warn.warning("light", "Ignoring malformed Light %r", light)
        if len(valid) > MAX_LIGHTS:
            valid.sort(key=lambda v: v[3], reverse=True)   # keep the strongest
            valid = valid[:MAX_LIGHTS]
        for i, (x, y, radius, intensity, color) in enumerate(valid):
            self._light_a[i] = (x, y, radius, intensity)
            self._light_c[i] = color
        return len(valid)

    def _fill_refraction(self, rings: Iterable[Ring]) -> int:
        valid = []
        for ring in rings:
            try:
                amount = _finite(ring.refraction)
                if amount > 0.0 and _finite(ring.radius) > 0.0 and _point(ring.center) is not None:
                    valid.append(ring)
            except (AttributeError, TypeError):
                continue
        if len(valid) > MAX_REFRACTION_RINGS:
            valid.sort(key=lambda r: _finite(r.refraction), reverse=True)
            valid = valid[:MAX_REFRACTION_RINGS]
        for i, ring in enumerate(valid):
            cx, cy = _vec2(ring.center)
            sx, sy = _vec2(ring.squash, (1.0, 1.0))
            self._ring_a[i] = (cx, cy, max(0.0, _finite(ring.radius)), max(1.0, _finite(ring.thickness, 1.0)))
            self._ring_b[i] = (min(_finite(ring.refraction), 200.0), _clamp(sx, 0.05, 20.0),
                               _clamp(sy, 0.05, 20.0), 0.0)
        return len(valid)

    # --------------------------------------------------------------- aura pass

    def _aura_pass(self, aura: Aura, view: ViewTransform) -> bool:
        try:
            intensity = _finite(aura.intensity)
            if intensity <= 0.0 or not self._prepare_aura_mask(aura) or self._mask_bbox is None:
                return False
            flame = _clamp(_finite(aura.flame_height, 0.08), 0.0, 0.5)
            mh, mw = self._mask_shape
            sigma = max(1.0, flame * mh * 0.45)
            self._blur_mask(sigma)
            assert self._scene_t is not None and self._mask_tex is not None and self._mask_soft is not None
            p = self._progs["aura"]
            pal = aura.palette
            # Region that can receive flames: the mask's bounding box, grown upwards by
            # the flame reach and sideways/downwards by the blur radius.
            u0, v0, u1, v1 = self._mask_bbox
            gx = 3.0 * sigma / mw + flame * 0.5
            gy = 3.0 * sigma / mh
            u0, u1 = max(0.0, u0 - gx), min(1.0, u1 + gx)
            v0, v1 = max(0.0, v0 - gy - flame * 1.6), min(1.0, v1 + gy)
            s = max(_finite(view.scale, 1.0), 1e-6)
            sw, sh = float(max(1, view.scene_w)), float(max(1, view.scene_h))
            x0, y0 = u0 * sw * s + view.offset_x, v0 * sh * s + view.offset_y
            x1, y1 = u1 * sw * s + view.offset_x, v1 * sh * s + view.offset_y
            self._scene_t.fbo.use()
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            self._mask_tex.use(0)
            self._mask_soft.texture.use(1)
            p.set("u_time", _wrap_time(aura.time))
            p.set("u_intensity", min(intensity, 8.0))
            p.set("u_flame", flame)
            p.set("u_turbulence", _clamp(_finite(aura.turbulence, 1.0), 0.0, 4.0))
            p.set("u_flicker", _clamp(_finite(aura.flicker, 0.5), 0.0, 2.0))
            p.set("u_aspect", sw / sh)
            p.set("u_texel", (1.0 / mw, 1.0 / mh))
            p.set("u_core", _rgb(pal.core))
            p.set("u_inner", _rgb(pal.inner))
            p.set("u_outer", _rgb(pal.outer))
            p.set("u_octaves", int(_clamp(self._quality.noise_octaves, 1, 8)))
            drawn = self._draw_quad("aura", ((x0 + x1) * 0.5, (y0 + y1) * 0.5), ((x1 - x0) * 0.5, 0.0),
                                    (0.0, (y1 - y0) * 0.5), (u0, v0, u1, v1))
            self.ctx.disable(moderngl.BLEND)
            return drawn
        except (AttributeError, TypeError, ValueError) as exc:
            self._warn.warning("aura", "Skipping malformed Aura: %s", exc)
            return False

    def _prepare_aura_mask(self, aura: Aura) -> bool:
        """Upload the person mask when ``mask_id`` changed. False when there is nothing to draw."""
        mask = aura.mask
        if mask is None:
            return False
        mask_id = int(aura.mask_id)
        shape = tuple(np.shape(mask)[:2])
        if (self._mask_tex is not None and mask_id >= 0 and mask_id == self._mask_id
                and shape == self._mask_shape):
            return True
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.ndim != 2 or m.shape[0] < 2 or m.shape[1] < 2:
            self._warn.warning("aura_mask", "Unsupported aura mask shape %s", m.shape)
            return False
        if m.dtype != np.uint8:
            if m.dtype == np.bool_:
                m = m.astype(np.uint8) * 255
            else:
                mf = np.nan_to_num(m.astype(np.float32))
                if mf.size and float(mf.max()) <= 1.0:
                    mf = mf * 255.0
                m = np.clip(mf, 0, 255).astype(np.uint8)
        h, w = m.shape
        if w > self._max_tex or h > self._max_tex:
            f = self._max_tex / max(w, h)
            m = cv2.resize(m, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA)
            h, w = m.shape
        # The tracker mask is low resolution (~256 px wide): a light blur removes the
        # staircase that bilinear magnification would otherwise show along the rim.
        m = np.ascontiguousarray(cv2.GaussianBlur(m, (0, 0), 0.9))
        if self._mask_tex is None or self._mask_tex.size != (w, h):
            self._park()
            for obj in (self._mask_tex, self._mask_tmp, self._mask_soft):
                if obj is not None:
                    obj.release()
            self._mask_tex = self._mask_tmp = self._mask_soft = None
            tex = self.ctx.texture((w, h), 1, m, alignment=1)
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            tex.repeat_x = False
            tex.repeat_y = False
            self._mask_tex = tex
            self._mask_tmp = _make_target(self.ctx, (w, h), 1, "f2")
            self._mask_soft = _make_target(self.ctx, (w, h), 1, "f2")
        else:
            self._mask_tex.write(m, alignment=1)
        rows = np.flatnonzero(m.max(axis=1) > 24)
        cols = np.flatnonzero(m.max(axis=0) > 24)
        if rows.size == 0:
            self._mask_bbox = None
        else:
            self._mask_bbox = (cols[0] / w, rows[0] / h, (cols[-1] + 1) / w, (rows[-1] + 1) / h)
        self._mask_id = mask_id
        self._mask_shape = (h, w)
        self._mask_blur_dirty = True
        self._stats["mask_uploads"] += 1
        return True

    def _blur_mask(self, sigma: float) -> None:
        """Separable GPU blur of the person mask into the soft (dilated) mask."""
        if not self._mask_blur_dirty and abs(sigma - self._mask_sigma) <= 0.08 * sigma:
            return
        assert self._mask_tex is not None and self._mask_tmp is not None and self._mask_soft is not None
        h, w = self._mask_shape
        w0, pairs, count = _gaussian_taps(sigma)
        p = self._progs["mask_blur"]
        self.ctx.disable(moderngl.BLEND)
        p.set("u_w0", w0)
        p.set("u_taps", count)
        p.write("u_tap", pairs)
        self._mask_tmp.fbo.use()
        self._mask_tex.use(0)
        p.set("u_dir", (1.0 / w, 0.0))
        p.set("u_gain", 1.0)
        self._vaos["mask_blur"].render(moderngl.TRIANGLES, vertices=3)
        self._mask_soft.fbo.use()
        self._mask_tmp.texture.use(0)
        p.set("u_dir", (0.0, 1.0 / h))
        p.set("u_gain", AURA_DILATE)
        self._vaos["mask_blur"].render(moderngl.TRIANGLES, vertices=3)
        self._mask_sigma = sigma
        self._mask_blur_dirty = False
        self._stats["mask_blurs"] += 1

    # ------------------------------------------------------------ effects pass

    def _effects_pass(self, fx: RenderList, view: ViewTransform) -> bool:
        assert self._fx_t is not None
        radial = [f for f in fx.flashes if getattr(f, "center", None) is not None and _finite(f.intensity) > 0.0]
        has_sprites = any(len(b) > 0 for b in fx.sprites if isinstance(b, SpriteBatch))
        if not (fx.rings or fx.beams or fx.frontal_beams or fx.orbs or fx.ribbons or fx.arcs
                or has_sprites or radial):
            if self._fx_dirty:
                self._fx_t.fbo.use()
                self._fx_t.fbo.clear(0.0, 0.0, 0.0, 0.0)
                self._fx_dirty = False
            return False
        ctx = self.ctx
        self._fx_t.fbo.use()
        self._fx_t.fbo.clear(0.0, 0.0, 0.0, 0.0)
        self._fx_dirty = True
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        octaves = int(_clamp(self._quality.noise_octaves, 1, 8))
        for ring in fx.rings:
            self._guarded("ring", self._draw_ring, ring, view)
        for beam in fx.beams:
            self._guarded("beam", self._draw_beam, beam, view, octaves)
        for fb in fx.frontal_beams:
            self._guarded("frontal", self._draw_frontal, fb, view, octaves)
        for orb in fx.orbs:
            self._guarded("orb", self._draw_orb, orb, view, octaves)
        if fx.ribbons or fx.arcs:
            self._guarded("strip", self._draw_strips, fx.ribbons, fx.arcs, view)
        if has_sprites:
            self._guarded("sprite", self._draw_sprites, fx.sprites, view)
        for flash in radial:
            self._guarded("flash", self._draw_radial_flash, flash, view)
        ctx.disable(moderngl.BLEND)
        return True

    def _guarded(self, kind: str, fn: Any, *args: Any) -> None:
        """Draw one primitive; malformed data skips that primitive instead of the frame."""
        try:
            fn(*args)
        except (AttributeError, TypeError, ValueError, IndexError) as exc:
            self._warn.warning(f"bad_{kind}", "Skipping malformed %s primitive: %s", kind, exc)

    def _draw_quad(self, name: str, center: tuple[float, float], ax: tuple[float, float],
                   ay: tuple[float, float], local: tuple[float, float, float, float]) -> bool:
        """Render an oriented quad with ``QUAD_VS``; culled when completely off-screen."""
        w, h = self._size
        ex = abs(ax[0]) + abs(ay[0])
        ey = abs(ax[1]) + abs(ay[1])
        values = (center[0], center[1], ex, ey) + tuple(local)
        if not all(math.isfinite(v) for v in values):
            return False
        if center[0] + ex < 0.0 or center[0] - ex > w or center[1] + ey < 0.0 or center[1] - ey > h:
            return False
        p = self._progs[name]
        p.set("u_win_size", (float(w), float(h)))
        p.set("u_center", (float(center[0]), float(center[1])))
        p.set("u_axis_x", (float(ax[0]), float(ax[1])))
        p.set("u_axis_y", (float(ay[0]), float(ay[1])))
        p.set("u_local", tuple(float(v) for v in local))
        self._vaos[name].render(moderngl.TRIANGLE_STRIP, vertices=4)
        return True

    @staticmethod
    def _set_palette(p: _Prog, palette: Palette) -> None:
        p.set("u_core", _rgb(palette.core))
        p.set("u_inner", _rgb(palette.inner))
        p.set("u_outer", _rgb(palette.outer))

    def _draw_orb(self, orb: Orb, view: ViewTransform, octaves: int) -> None:
        radius = _finite(orb.radius)
        intensity = _finite(orb.intensity, 1.0)
        center = _point(orb.center)
        if radius <= 0.0 or intensity <= 0.0 or center is None:
            return
        s = view.scale
        cx, cy = view.scene_to_window(*center)
        sx, sy = _vec2(orb.squash, (1.0, 1.0))
        sx, sy = _clamp(sx, 0.1, 10.0), _clamp(sy, 0.1, 10.0)
        rot = _finite(orb.rotation)
        c, sn = math.cos(rot), math.sin(rot)
        e = radius * ORB_EXTENT * s
        p = self._progs["orb"]
        p.set("u_time", _wrap_time(orb.time))
        p.set("u_seed", math.fmod(abs(_finite(orb.seed)), 97.0))
        p.set("u_charge", _clamp(_finite(orb.charge), 0.0, 1.0))
        p.set("u_intensity", min(intensity, 64.0))
        p.set("u_swirl", _clamp(_finite(orb.swirl, 1.0), 0.0, 8.0))
        p.set("u_instability", _clamp(_finite(orb.instability), 0.0, 1.0))
        p.set("u_flare", _clamp(_finite(orb.flare), 0.0, 2.0))
        p.set("u_extent", ORB_EXTENT)
        p.set("u_octaves", octaves)
        self._set_palette(p, orb.palette)
        self._draw_quad("orb", (cx, cy), (c * e * sx, sn * e * sx), (-sn * e * sy, c * e * sy),
                        (-ORB_EXTENT, -ORB_EXTENT, ORB_EXTENT, ORB_EXTENT))

    def _draw_beam(self, beam: Beam, view: ViewTransform, octaves: int) -> None:
        fade = _clamp(_finite(beam.fade, 1.0), 0.0, 1.0)
        intensity = _finite(beam.intensity, 1.0)
        length = max(0.0, _finite(beam.length))
        origin = _point(beam.origin)
        direction = _point(beam.direction)
        if fade <= 0.0 or intensity <= 0.0 or origin is None or direction is None:
            return
        norm = math.hypot(*direction)
        if norm < 1e-9:
            return                                # no direction: nothing sensible to draw
        dx, dy = direction[0] / norm, direction[1] / norm
        w0 = max(0.5, _finite(beam.width, 1.0))
        w1 = max(0.5, _finite(beam.end_width, w0))
        head = max(0.0, _finite(beam.head_radius))
        back = 3.2 * w0
        u0 = -back
        u1 = length + max(head * 2.7, w1 * 1.6)
        hv = max(4.6 * max(w0, w1), 2.8 * head)
        s = view.scale
        ox, oy = view.scene_to_window(*origin)
        cu = (u0 + u1) * 0.5 * s
        hu = (u1 - u0) * 0.5 * s
        p = self._progs["beam"]
        p.set("u_length", max(length, 1.0))
        p.set("u_width", w0)
        p.set("u_end_width", w1)
        p.set("u_head", head)
        p.set("u_time", _wrap_time(beam.time))
        p.set("u_seed", math.fmod(abs(_finite(beam.seed)), 97.0))
        p.set("u_turbulence", _clamp(_finite(beam.turbulence, 1.0), 0.0, 4.0))
        p.set("u_flow", _clamp(_finite(beam.flow_speed, 1.0), 0.0, 10.0))
        p.set("u_fade", fade)
        p.set("u_dissolve", _clamp(_finite(beam.dissolve), 0.0, 1.0))
        p.set("u_intensity", min(intensity, 64.0))
        p.set("u_extent_v", hv)
        p.set("u_back", back)
        p.set("u_octaves", octaves)
        self._set_palette(p, beam.palette)
        self._draw_quad("beam", (ox + dx * cu, oy + dy * cu), (dx * hu, dy * hu),
                        (-dy * hv * s, dx * hv * s), (u0, -hv, u1, hv))

    def _draw_frontal(self, fb: FrontalBeam, view: ViewTransform, octaves: int) -> None:
        fade = _clamp(_finite(fb.fade, 1.0), 0.0, 1.0)
        intensity = _finite(fb.intensity, 1.0)
        radius = _finite(fb.radius)
        center = _point(fb.center)
        if fade <= 0.0 or intensity <= 0.0 or radius <= 0.0 or center is None:
            return
        r = max(radius * view.scale, 1.0)
        cx, cy = view.scene_to_window(*center)
        w, h = self._size
        p = self._progs["frontal"]
        p.set("u_time", _wrap_time(fb.time))
        p.set("u_seed", math.fmod(abs(_finite(fb.seed)), 97.0))
        p.set("u_streaks", _clamp(_finite(fb.streaks, 1.0), 0.0, 4.0))
        p.set("u_fade", fade)
        p.set("u_dissolve", _clamp(_finite(fb.dissolve), 0.0, 1.0))
        p.set("u_intensity", min(intensity, 64.0))
        p.set("u_octaves", octaves)
        self._set_palette(p, fb.palette)
        # The halo fills most of the screen: draw one window-covering quad.
        self._draw_quad("frontal", (w * 0.5, h * 0.5), (w * 0.5, 0.0), (0.0, h * 0.5),
                        (-cx / r, -cy / r, (w - cx) / r, (h - cy) / r))

    def _draw_ring(self, ring: Ring, view: ViewTransform) -> None:
        intensity = _finite(ring.intensity, 1.0)
        radius = max(0.0, _finite(ring.radius))
        center = _point(ring.center)
        if intensity <= 0.0 or center is None:
            return
        th = max(0.5, _finite(ring.thickness, 1.0))
        band = 3.2 * th                       # the profile is negligible beyond 3.2 thicknesses
        sx, sy = _vec2(ring.squash, (1.0, 1.0))
        sx, sy = _clamp(sx, 0.05, 20.0), _clamp(sy, 0.05, 20.0)
        x, y = center
        cx, cy = view.scene_to_window(x, y)
        s = view.scale
        ex, ey = (radius + band) * s * sx, (radius + band) * s * sy
        w, h = self._size
        if cx + ex < 0 or cx - ex > w or cy + ey < 0 or cy - ey > h:
            return
        p = self._progs["ring"]
        p.set("u_win_size", (float(w), float(h)))
        p.set("u_center", (cx, cy))
        p.set("u_scale_xy", (s * sx, s * sy))
        p.set("u_r_in", max(0.0, radius - band))
        p.set("u_r_out", radius + band)
        p.set("u_radius", radius)
        p.set("u_thickness", th)
        p.set("u_noise", _clamp(_finite(ring.noise, 0.5), 0.0, 1.0))
        p.set("u_time", _wrap_time(ring.time))
        p.set("u_seed", math.fmod(abs(x * 0.131 + y * 0.071), 89.0))
        p.set("u_intensity", min(intensity, 64.0))
        p.set("u_extent", band)
        self._set_palette(p, ring.palette)
        self._vaos["ring"].render(moderngl.TRIANGLE_STRIP, vertices=2 * (RING_SEGMENTS + 1))

    def _draw_radial_flash(self, flash: Flash, view: ViewTransform) -> None:
        center = _point(flash.center)
        if center is None:
            return
        radius = _finite(flash.radius)
        if radius <= 0.0:
            radius = 0.35 * view.scene_h
        ext = 3.0
        cx, cy = view.scene_to_window(*center)
        e = radius * ext * view.scale
        p = self._progs["flash"]
        p.set("u_color", _rgb(flash.color))
        p.set("u_intensity", min(_finite(flash.intensity), 64.0) * FLASH_GAIN)
        p.set("u_extent", ext)
        self._draw_quad("flash", (cx, cy), (e, 0.0), (0.0, e), (-ext, -ext, ext, ext))

    def _min_core_px(self) -> float:
        """Smallest strip core half-width (window px) that does not alias in the effect buffer."""
        return 0.7 / max(0.25, _finite(self._quality.render_scale, 1.0))

    def _strip_geometry(self, ribbons: Sequence[Ribbon], arcs: Sequence[Arc],
                        view: ViewTransform) -> np.ndarray:
        """Triangle vertices for every ribbon (first) and arc of the frame."""
        pts_parts: list[np.ndarray] = []
        hw_parts: list[np.ndarray] = []
        core_parts: list[np.ndarray] = []
        crgb_parts: list[np.ndarray] = []
        grgb_parts: list[np.ndarray] = []
        for rb in ribbons:
            pts = np.asarray(rb.points, np.float32).reshape(-1, 2)
            m = pts.shape[0]
            widths = np.asarray(rb.widths, np.float32).reshape(-1)
            inten = np.asarray(rb.intensities, np.float32).reshape(-1)
            if widths.size == 1:
                widths = np.full(m, widths[0], np.float32)
            if inten.size == 1:
                inten = np.full(m, inten[0], np.float32)
            m = min(m, widths.size, inten.size)
            if m < 2:
                continue
            pal = rb.palette
            inten = np.clip(np.nan_to_num(inten[:m]), 0.0, 64.0)
            pts_parts.append(pts[:m])
            hw_parts.append(np.maximum(np.nan_to_num(widths[:m]), 0.0))
            core_parts.append(np.full(m, 0.34, np.float32))
            crgb_parts.append(inten[:, None] * (np.asarray(pal.core, np.float32) * 1.7)[None, :])
            grgb_parts.append(inten[:, None] * (_mix3(pal.inner, pal.outer, 0.35) * 1.5)[None, :])
        for arc in arcs:
            pts = np.asarray(arc.points, np.float32).reshape(-1, 2)
            m = pts.shape[0]
            intensity = _finite(arc.intensity, 1.0)
            if m < 2 or intensity <= 0.0:
                continue
            glow = max(0.2, _finite(arc.glow, 1.0))
            width = max(0.0, _finite(arc.width, 1.0))
            pos = np.arange(m, dtype=np.float32) / float(m - 1)
            taper = np.clip(np.minimum(pos, 1.0 - pos) * 7.0, 0.45, 1.0)   # bolts thin at the ends
            pal = arc.palette
            intensity = min(intensity, 64.0)
            pts_parts.append(pts)
            hw_parts.append(width * ARC_STRIP_FACTOR * glow * taper)
            core_parts.append(np.full(m, 1.0 / ARC_STRIP_FACTOR / glow, np.float32))
            crgb_parts.append(np.tile(np.asarray(pal.core, np.float32) * (1.9 * intensity), (m, 1)))
            grgb_parts.append(np.tile(_mix3(pal.inner, pal.outer, 0.30) * (1.0 * intensity), (m, 1)))
        if not pts_parts:
            return np.zeros((0, _STRIP_FLOATS), np.float32)
        counts = np.array([a.shape[0] for a in pts_parts], np.int64)
        poly = np.repeat(np.arange(counts.size), counts)
        pts = np.concatenate(pts_parts) * np.float32(view.scale) + np.array(
            [view.offset_x, view.offset_y], np.float32)
        hw = np.concatenate(hw_parts) * np.float32(view.scale)
        core = np.concatenate(core_parts)
        crgb = np.concatenate(crgb_parts)
        grgb = np.concatenate(grgb_parts)
        # Thin strips would alias in a reduced-resolution effect buffer: widen them and
        # dim them by the same factor so their total light stays the same.
        core_px = hw * core
        widen = np.maximum(1.0, self._min_core_px() / np.maximum(core_px, 1e-3)).astype(np.float32)
        hw = hw * widen
        crgb = crgb / widen[:, None]
        grgb = grgb / widen[:, None]
        same = poly[1:] == poly[:-1]
        finite = np.isfinite(pts).all(axis=1) & np.isfinite(hw)
        if not finite.all():
            same &= finite[1:] & finite[:-1]
            pts = np.nan_to_num(pts)
            hw = np.nan_to_num(hw)
        return strip_vertices(pts, hw, core, crgb, grgb, same)

    def _draw_strips(self, ribbons: Sequence[Ribbon], arcs: Sequence[Arc], view: ViewTransform) -> None:
        verts = self._strip_geometry(ribbons, arcs, view)
        if verts.shape[0] == 0:
            return
        assert self._strip_buf is not None
        if self._strip_buf.upload(verts):
            if self._strip_vao is not None:
                self._strip_vao.release()
            self._strip_vao = self._make_strip_vao()
        p = self._progs["strip"]
        p.set("u_win_size", (float(self._size[0]), float(self._size[1])))
        assert self._strip_vao is not None
        self._strip_vao.render(moderngl.TRIANGLES, vertices=int(verts.shape[0]))

    def _pack_sprites(self, batches: Sequence[SpriteBatch]) -> int:
        """Copy all sprite batches into the instance staging array (vectorised per batch)."""
        total = 0
        for b in batches:
            try:
                total += len(b)
            except (TypeError, AttributeError):
                continue
        if total <= 0:
            return 0
        if total > self._sprite_data.shape[0]:
            self._sprite_data = np.zeros((max(total, self._sprite_data.shape[0] * 2), _SPRITE_FLOATS), np.float32)
        data = self._sprite_data
        o = 0
        for b in batches:
            try:
                n = len(b)
                if n <= 0:
                    continue
                seg = data[o:o + n]
                seg[:, 0:2] = b.positions
                seg[:, 2] = np.asarray(b.sizes, np.float32).reshape(-1)
                colors = np.asarray(b.colors, np.float32)
                if colors.ndim == 2 and colors.shape[1] == 3:
                    seg[:, 3:6] = colors
                    seg[:, 6] = 1.0
                else:
                    seg[:, 3:7] = colors
                if b.velocities is not None:
                    seg[:, 7:9] = b.velocities
                else:
                    seg[:, 7:9] = 0.0
                seg[:, 9] = float(int(b.shape))
                seg[:, 10] = _finite(b.stretch)
                o += n
            except (ValueError, TypeError, AttributeError) as exc:
                self._warn.warning("sprites", "Skipping malformed SpriteBatch: %s", exc)
        if o:
            np.nan_to_num(data[:o], copy=False)
        return o

    def _draw_sprites(self, batches: Sequence[SpriteBatch], view: ViewTransform) -> None:
        n = self._pack_sprites(batches)
        if n == 0:
            return
        assert self._sprite_buf is not None
        if self._sprite_buf.upload(self._sprite_data[:n]):
            if self._sprite_vao is not None:
                self._sprite_vao.release()
            self._sprite_vao = self._make_sprite_vao()
        p = self._progs["sprite"]
        p.set("u_win_size", (float(self._size[0]), float(self._size[1])))
        p.set("u_view", (float(view.scale), float(view.offset_x), float(view.offset_y)))
        p.set("u_min_px", 0.8 / max(0.25, _finite(self._quality.render_scale, 1.0)))
        assert self._sprite_vao is not None
        self._sprite_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)

    # -------------------------------------------------------------- bloom pass

    def _bloom_pass(self, active: bool) -> bool:
        if not self._bloom:
            return False
        if not active:
            if self._bloom_dirty:
                self._bloom[0].fbo.use()
                self._bloom[0].fbo.clear(0.0, 0.0, 0.0, 0.0)
                self._bloom_dirty = False
            return False
        assert self._fx_t is not None and self._scene_t is not None
        ctx = self.ctx
        ctx.disable(moderngl.BLEND)
        down = self._progs["bloom_down"]
        down.set("u_scene_threshold", SCENE_BLOOM_THRESHOLD)
        down.set("u_scene_gain", SCENE_BLOOM_GAIN)
        self._scene_t.texture.use(1)
        src = self._fx_t
        for i, mip in enumerate(self._bloom):
            mip.fbo.use()
            src.texture.use(0)
            sw, sh = src.size
            down.set("u_texel", (1.0 / sw, 1.0 / sh))
            down.set("u_prefilter", 1 if i == 0 else 0)
            self._vaos["bloom_down"].render(moderngl.TRIANGLES, vertices=3)
            src = mip
        up = self._progs["bloom_up"]
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE
        up.set("u_radius", 1.0)
        up.set("u_weight", 1.0)
        for i in range(len(self._bloom) - 1, 0, -1):
            small, large = self._bloom[i], self._bloom[i - 1]
            large.fbo.use()
            small.texture.use(0)
            sw, sh = small.size
            up.set("u_texel", (1.0 / sw, 1.0 / sh))
            self._vaos["bloom_up"].render(moderngl.TRIANGLES, vertices=3)
        ctx.disable(moderngl.BLEND)
        self._bloom_dirty = True
        return True

    # ---------------------------------------------------------- composite pass

    def _composite_pass(self, fx: RenderList, view: ViewTransform, bloom_on: bool, now: float) -> None:
        assert self._out_t is not None and self._scene_t is not None and self._fx_t is not None
        w, h = self._out_t.size
        # Shake: shift the whole image; an overscan zoom (following a decaying envelope of
        # the shake amplitude) keeps the borders from ever showing.
        sx, sy = _vec2(fx.shake)
        su = sx * view.scale / w
        sv = sy * view.scale / h
        dt = 0.0 if self._last_frame_t is None else _clamp(now - self._last_frame_t, 0.0, 0.25)
        self._last_frame_t = now
        self._shake_env = max(abs(su), abs(sv), self._shake_env * math.exp(-dt * 4.0))
        if self._shake_env < 1e-5:
            self._shake_env = 0.0
        env = min(self._shake_env, 0.2)
        overscan = 1.0 / (1.0 - 2.0 * env) if env > 0.0 else 1.0
        zoom = _clamp(_finite(getattr(fx.grade, "zoom", 1.0), 1.0), 0.5, 4.0) * overscan
        keep = np.ones(3, np.float64)
        for f in fx.flashes:
            if getattr(f, "center", None) is None:
                amount = _clamp(_finite(f.intensity), 0.0, 1.0)
                keep *= 1.0 - np.clip(np.asarray(_rgb(f.color)) * amount, 0.0, 1.0)
        flash = 1.0 - keep
        p = self._progs["composite"]
        self._out_t.fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self._scene_t.texture.use(0)
        self._fx_t.texture.use(1)
        self._bloom[0].texture.use(2)
        p.set("u_uv_scale", (1.0 / zoom, 1.0 / zoom))
        p.set("u_uv_offset", (-su, sv))
        p.set("u_exposure", _clamp(_finite(fx.exposure, 1.0), 0.0, 16.0))
        bloom = (_clamp(_finite(fx.bloom, 1.0), 0.0, 8.0) * BLOOM_GAIN / max(1, len(self._bloom))
                 if bloom_on else 0.0)
        p.set("u_bloom_strength", bloom)
        p.set("u_flash", (float(flash[0]), float(flash[1]), float(flash[2])))
        self._vaos["composite"].render(moderngl.TRIANGLES, vertices=3)

    # ------------------------------------------------------------------ UI pass

    def _ui_pass(self, ui: UIDrawList | None) -> None:
        commands = ui.commands if ui is not None else ()
        if not commands:
            return
        assert self._out_t is not None
        ctx = self.ctx
        self._out_t.fbo.use()
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        n = 0
        for cmd in commands:
            try:
                if isinstance(cmd, UIText):
                    if n:
                        self._flush_ui_shapes(n)
                        n = 0
                    self._draw_text(cmd)
                else:
                    n = self._add_ui_shape(cmd, n)
            except (AttributeError, TypeError, ValueError, IndexError, pygame.error) as exc:
                self._warn.warning("ui", "Skipping malformed UI command %r: %s", type(cmd).__name__, exc)
        if n:
            self._flush_ui_shapes(n)
        ctx.disable(moderngl.BLEND)

    def _ui_rows(self, n: int, extra: int) -> np.ndarray:
        if n + extra > self._ui_data.shape[0]:
            grown = np.zeros((max(n + extra, self._ui_data.shape[0] * 2), _UI_FLOATS), np.float32)
            grown[:n] = self._ui_data[:n]
            self._ui_data = grown
        return self._ui_data

    def _add_ui_shape(self, cmd: Any, n: int) -> int:
        if isinstance(cmd, UIRect):
            rows = self._ui_rows(n, 1)
            color = _rgba(cmd.color)
            bottom = _rgba(cmd.gradient_to) if cmd.gradient_to is not None else color
            border = _rgba(cmd.border_color) if cmd.border_color is not None else (0.0, 0.0, 0.0, 0.0)
            bw = max(0.0, _finite(cmd.border_width)) if cmd.border_color is not None else 0.0
            rows[n, 0:4] = (_finite(cmd.x), _finite(cmd.y), max(0.0, _finite(cmd.w)), max(0.0, _finite(cmd.h)))
            rows[n, 4:8] = (max(0.0, _finite(cmd.radius)), bw, 0.0, 0.0)
            rows[n, 8:12] = color
            rows[n, 12:16] = bottom
            rows[n, 16:20] = border
            return n + 1
        if isinstance(cmd, UICircle):
            rows = self._ui_rows(n, 1)
            color = _rgba(cmd.color)
            r = max(0.0, _finite(cmd.radius))
            width = max(0.0, _finite(cmd.width))
            x, y = _finite(cmd.x), _finite(cmd.y)
            if width <= 0.0:
                rows[n, 0:4] = (x - r, y - r, 2 * r, 2 * r)
                rows[n, 4:8] = (r, 0.0, 0.0, 0.0)
                rows[n, 8:12] = color
                rows[n, 12:16] = color
                rows[n, 16:20] = 0.0
            else:
                ro = r + width * 0.5      # outline centred on the radius
                rows[n, 0:4] = (x - ro, y - ro, 2 * ro, 2 * ro)
                rows[n, 4:8] = (ro, width, 0.0, 0.0)
                rows[n, 8:16] = 0.0
                rows[n, 16:20] = color
            return n + 1
        if isinstance(cmd, UIArc):
            rows = self._ui_rows(n, 1)
            rows[n, 0:4] = (_finite(cmd.x), _finite(cmd.y), max(0.0, _finite(cmd.radius)),
                            max(0.0, _finite(cmd.width, 4.0)))
            rows[n, 4:8] = (_finite(cmd.start), _finite(cmd.end), 1.0, 0.0)
            rows[n, 8:12] = _rgba(cmd.color)
            rows[n, 12:20] = 0.0
            return n + 1
        if isinstance(cmd, UILine):
            pts = np.asarray(cmd.points, np.float32).reshape(-1, 2)
            if pts.shape[0] < 2:
                return n
            if cmd.closed and pts.shape[0] > 2:
                pts = np.concatenate([pts, pts[:1]])
            k = pts.shape[0] - 1
            rows = self._ui_rows(n, k)
            seg = rows[n:n + k]
            seg[:, 0:2] = pts[:-1]
            seg[:, 2:4] = pts[1:]
            seg[:, 4] = max(0.0, _finite(cmd.width, 1.0))
            seg[:, 5] = 0.0
            seg[:, 6] = 2.0
            seg[:, 7] = 0.0
            seg[:, 8:12] = _rgba(cmd.color)
            seg[:, 12:20] = 0.0
            np.nan_to_num(seg, copy=False)
            return n + k
        self._warn.warning("ui_kind", "Unknown UI command type %s ignored", type(cmd).__name__)
        return n

    def _flush_ui_shapes(self, n: int) -> None:
        assert self._ui_buf is not None
        if self._ui_buf.upload(self._ui_data[:n]):
            if self._ui_vao is not None:
                self._ui_vao.release()
            self._ui_vao = self._make_ui_vao()
        p = self._progs["ui_shape"]
        p.set("u_win_size", (float(self._size[0]), float(self._size[1])))
        assert self._ui_vao is not None
        self._ui_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n)

    def _draw_text(self, cmd: UIText) -> None:
        text = str(cmd.text)
        color = _rgba(cmd.color)
        if not text or color[3] <= 0.0:
            return
        assert self._text is not None
        size = max(6, int(round(_finite(cmd.size, 16.0))))
        font = cmd.font or "ui"
        entry = self._text.text(text, size, font)
        x, y = _finite(cmd.x), _finite(cmd.y)
        if cmd.align == "center":
            x -= entry.width * 0.5
        elif cmd.align == "right":
            x -= entry.width
        if cmd.valign == "middle":
            y -= entry.height * 0.5
        elif cmd.valign == "bottom":
            y -= entry.height
        x, y = float(round(x)), float(round(y))      # texel-exact placement keeps text crisp
        p = self._progs["ui_text"]
        p.set("u_win_size", (float(self._size[0]), float(self._size[1])))
        if cmd.shadow:
            off = float(max(1, round(size / 16.0)))
            self._blit_text(entry, x + off, y + off, (0.0, 0.0, 0.0, 0.65 * color[3]))
        glow = _finite(cmd.glow)
        if glow > 0.0:
            g = self._text.glow(text, size, font, glow)
            gc = _rgba(cmd.glow_color) if cmd.glow_color is not None else (color[0], color[1], color[2],
                                                                             0.8 * color[3])
            self._blit_text(g, x - g.pad, y - g.pad, gc)
        self._blit_text(entry, x, y, color)

    def _blit_text(self, entry: _TextTexture, x: float, y: float, color: tuple[float, float, float, float]) -> None:
        p = self._progs["ui_text"]
        p.set("u_rect", (x, y, float(entry.width), float(entry.height)))
        p.set("u_color", color)
        entry.texture.use(0)
        self._vaos["ui_text"].render(moderngl.TRIANGLE_STRIP, vertices=4)

    # ----------------------------------------------------------------- present

    def _present(self) -> None:
        screen = self.ctx.screen
        if screen is None or self._out_t is None:
            return
        w, h = self._size
        screen.viewport = (0, 0, w, h)
        screen.use()
        self.ctx.disable(moderngl.BLEND)
        self._out_t.texture.use(0)
        self._vaos["copy"].render(moderngl.TRIANGLES, vertices=3)


# ----------------------------------------------------------------------- factory


def _create_standalone_context() -> moderngl.Context:
    """Headless OpenGL 3.3 context (EGL first on Linux, the platform default otherwise)."""
    errors = []
    backends: list[str | None] = ["egl", None] if sys.platform.startswith("linux") else [None]
    for backend in backends:
        try:
            if backend:
                return moderngl.create_standalone_context(require=330, backend=backend)
            return moderngl.create_standalone_context(require=330)
        except Exception as exc:  # noqa: BLE001 - try the next backend
            errors.append(f"{backend or 'default'}: {exc}")
    raise RendererError("No headless OpenGL 3.3 context available (" + "; ".join(errors) + ")")


def _create_cpu_renderer(size: tuple[int, int], quality: QualityPreset, fonts: FontManager,
                         settings: EffectsSettings | None) -> Renderer:
    try:
        from dragon_ball_powers.renderer_cpu import CpuRenderer
    except ImportError as exc:
        raise RendererError(f"The CPU renderer is not available: {exc}") from exc
    target = None
    try:
        if pygame.display.get_init():
            surface = pygame.display.get_surface()
            # An OpenGL window surface cannot be blitted to; the caller must re-create the
            # window without pygame.OPENGL before drawing with the CPU renderer.
            if surface is not None and not (surface.get_flags() & pygame.OPENGL):
                target = surface
    except pygame.error:
        target = None
    return CpuRenderer(size, quality, fonts, settings, target=target)


def create_renderer(kind: str, size: tuple[int, int], quality: QualityPreset, fonts: FontManager,
                    settings: EffectsSettings | None = None, window_has_gl: bool = True) -> Renderer:
    """Create a renderer for the current pygame window.

    * ``"gl"``: :class:`GLRenderer` on the window's OpenGL 3.3 core context; raises
      :class:`RendererError` if that is impossible.
    * ``"auto"``: like ``"gl"``, but falls back to ``CpuRenderer`` (with a logged
      warning) when OpenGL fails. When a CPU renderer is returned for a window that
      was created with ``pygame.OPENGL``, the caller must re-create the window without
      that flag (check ``isinstance(r, GLRenderer)``).
    * ``"cpu"``: ``CpuRenderer`` (NumPy/OpenCV).
    """
    choice = str(kind or "auto").strip().lower()
    if choice not in ("auto", "gl", "cpu"):
        raise RendererError(f"Unknown renderer kind {kind!r}; expected 'auto', 'gl' or 'cpu'")
    if choice == "auto" and not window_has_gl:
        log.info("Window has no OpenGL context; using the CPU renderer")
    elif choice in ("auto", "gl"):
        try:
            if not window_has_gl:
                raise RendererError("the window was created without an OpenGL context")
            ctx = moderngl.create_context(require=330)
            return GLRenderer(ctx, size, quality, fonts, settings)
        except Exception as exc:  # noqa: BLE001 - any GL failure: raise ("gl") or fall back ("auto")
            if choice == "gl":
                if isinstance(exc, RendererError):
                    raise
                raise RendererError(f"OpenGL renderer unavailable: {exc}") from exc
            log.warning("OpenGL renderer unavailable (%s); falling back to the CPU renderer", exc)
    return _create_cpu_renderer(size, quality, fonts, settings)


__all__ = ["GLRenderer", "create_renderer", "strip_vertices", "RendererError"]
