"""Tests for the OpenGL renderer (renderer.py + gl_shaders.py).

All GL tests run headless on the shared ``gl_context`` fixture (skipped when no
OpenGL 3.3 context exists). Images are synthetic; no camera, display or network.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import pytest

from dragon_ball_powers import gl_shaders
from dragon_ball_powers.fonts import FontManager
from dragon_ball_powers.interfaces import RendererError
from dragon_ball_powers.primitives import (
    Arc,
    Aura,
    Beam,
    Flash,
    FrontalBeam,
    Grade,
    Light,
    Orb,
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
from dragon_ball_powers.renderer import TEXT_CACHE_LIMIT, GLRenderer, create_renderer, strip_vertices

pytestmark = pytest.mark.gl

ROOT = Path(__file__).resolve().parents[1]
W, H = 320, 180
BLUE = get_palette("blue")
_ids = itertools.count(1)


# --------------------------------------------------------------------- helpers


@pytest.fixture(scope="module")
def fonts() -> FontManager:
    return FontManager()


@pytest.fixture
def make(gl_context, fonts, request):
    """Factory for renderers on the shared context; every renderer is released afterwards."""

    def _make(size: tuple[int, int] = (W, H), quality: str = "high") -> GLRenderer:
        r = GLRenderer(gl_context, size, get_quality(quality), fonts, offscreen=True)
        request.addfinalizer(r.release)
        return r

    return _make


def view_for(image_size: tuple[int, int], window: tuple[int, int], mode: str = "cover"):
    return compute_view_transform(image_size[0], image_size[1], window[0], window[1], mode)


def render(r: GLRenderer, image: np.ndarray | None, rl: RenderList | None = None, ui: UIDrawList | None = None,
           t: float = 1.0, image_id: int | None = None, mode: str = "cover", scene_size=None) -> np.ndarray:
    if scene_size is None:
        scene_size = (image.shape[1], image.shape[0]) if image is not None else r.size
    iid = next(_ids) if image_id is None else image_id
    scene = SceneInput(image, iid, scene_size, t)
    r.render(scene, rl if rl is not None else RenderList(), ui if ui is not None else UIDrawList(),
             view_for(scene_size, r.size, mode))
    return r.read_pixels()


def grey(v: int = 90, w: int = W, h: int = H) -> np.ndarray:
    return np.full((h, w, 3), v, np.uint8)


def lum(px: np.ndarray) -> float:
    return float(np.asarray(px, np.float64).mean())


def assert_no_gl_error(r: GLRenderer) -> None:
    assert r.ctx.error == "GL_NO_ERROR"


# ---------------------------------------------------------------- construction


def test_create_offscreen_owns_its_context(fonts):
    try:
        r = GLRenderer.create_offscreen((160, 90), get_quality("low"), fonts)
    except RendererError as exc:
        pytest.skip(f"no headless OpenGL context: {exc}")
    try:
        info = r.info
        assert r.name == "OpenGL" and info["renderer"] == "OpenGL"
        for key in ("gl_vendor", "gl_renderer", "gl_version"):
            assert isinstance(info[key], str) and info[key]
        assert r.offscreen
        out = render(r, grey(120, 160, 90))
        assert out.shape == (90, 160, 3)
        assert_no_gl_error(r)
    finally:
        r.release()
    assert r.released


def test_read_pixels_shape_dtype_and_orientation(make):
    r = make()
    img = np.zeros((H, W, 3), np.uint8)
    img[: H // 2] = (0, 0, 255)          # BGR red on top
    img[H // 2:] = (255, 0, 0)           # BGR blue at the bottom
    out = render(r, img)
    assert out.shape == (H, W, 3) and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    assert tuple(out[2, W // 2]) == (255, 0, 0)       # RGB, top row first
    assert tuple(out[H - 3, W // 2]) == (0, 0, 255)


def test_mirror_output_flips_picture_and_effects_but_not_the_ui(make):
    r = make()
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    rl = RenderList()
    rl.orbs.append(Orb((60.0, 90.0), 18.0, BLUE, intensity=1.5, charge=0.8))
    normal = render(r, img, rl, t=2.0, image_id=7)
    r.mirror_output = True
    mirrored = render(r, img, rl, t=2.0, image_id=7)
    assert np.abs(mirrored.astype(int) - normal[:, ::-1].astype(int)).max() <= 3
    assert lum(mirrored[60:120, 240:300]) > lum(mirrored[60:120, 20:80])     # the orb moved to the right
    ui = UIDrawList([UIRect(10, 10, 60, 30, (1.0, 0.0, 0.0, 1.0))])
    out = render(r, grey(30), ui=ui)
    assert out[25, 40, 0] > 200 and out[25, W - 40, 0] < 80                    # UI stays where it was drawn
    assert_no_gl_error(r)


# ------------------------------------------------------------------- identity


def test_empty_renderlist_is_identity(make):
    r = make()
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)       # noise: worst case for filtering
    out = render(r, img)
    diff = np.abs(out.astype(int) - img[:, :, ::-1].astype(int))
    assert diff.max() <= 2
    assert_no_gl_error(r)


def test_identity_with_scaling_and_letterbox(make):
    r = make((W, 240))
    img = np.empty((H * 2, W * 2, 3), np.uint8)
    img[:] = (40, 120, 200)
    out = render(r, img, mode="contain")                       # 640x360 scene in a 320x240 window
    bar = (240 - H) // 2
    assert out[: bar - 2].max() == 0 and out[-(bar - 2):].max() == 0          # black letterbox
    inside = out[bar + 2: 240 - bar - 2].reshape(-1, 3).astype(int)
    assert np.abs(inside - np.array([200, 120, 40])).max() <= 2               # unchanged colour (RGB)


def test_placeholder_without_camera_is_dark_blue_and_animated(make):
    r = make()
    a = render(r, None, t=1.0)
    b = render(r, None, t=9.0)
    assert 1.0 < lum(a) < 40.0
    assert a[..., 2].mean() > a[..., 0].mean()                 # deep blue-black, not grey
    assert np.abs(a.astype(int) - b.astype(int)).max() > 0     # subtle moving noise


# ------------------------------------------------------------------ primitives


def _orb():
    rl = RenderList()
    rl.orbs.append(Orb((160, 90), 20, BLUE, charge=0.6, time=1.0, flare=0.5, instability=0.3))
    return rl, (160, 90), (5, 5)


def _beam():
    rl = RenderList()
    rl.beams.append(Beam((40, 90), (1.0, 0.0), 200, 10, 12, BLUE, time=1.0, head_radius=16))
    return rl, (140, 90), (300, 15)


def _ring():
    rl = RenderList()
    rl.rings.append(Ring((160, 90), 50, 4, BLUE, noise=0.0))
    return rl, (210, 90), (160, 90)        # the centre of a ring stays untouched


def _arc():
    rl = RenderList()
    pts = np.stack([np.linspace(60, 260, 24), np.full(24, 90.0)], axis=1).astype(np.float32)
    rl.arcs.append(Arc(pts, 1.5, BLUE, 1.5, 1.0))
    return rl, (160, 90), (160, 40)


def _ribbon():
    rl = RenderList()
    pts = np.stack([np.linspace(60, 260, 20), np.full(20, 90.0)], axis=1).astype(np.float32)
    rl.ribbons.append(Ribbon(pts, np.full(20, 8.0, np.float32), np.ones(20, np.float32), BLUE))
    return rl, (160, 90), (160, 30)


def _sprites(shape: SpriteShape, vel=None, stretch=0.0, probe=(100, 60)):
    def build():
        rl = RenderList()
        pos = np.array([[100.0, 60.0]], np.float32)
        colors = np.array([[1.5, 2.0, 2.4, 1.0]], np.float32)
        v = None if vel is None else np.array([vel], np.float32)
        rl.sprites.append(SpriteBatch(shape, pos, np.array([8.0], np.float32), colors, v, stretch))
        return rl, probe, (250, 150)
    return build


def _radial_flash():
    rl = RenderList()
    rl.flashes.append(Flash(1.0, (1.0, 0.9, 0.8), center=(160, 90), radius=30))
    return rl, (160, 90), (5, 175)


@pytest.mark.parametrize("build", [
    _orb, _beam, _ring, _arc, _ribbon, _radial_flash,
    _sprites(SpriteShape.GLOW), _sprites(SpriteShape.EMBER), _sprites(SpriteShape.SMOKE),
    _sprites(SpriteShape.SPARK, vel=(600.0, 0.0), stretch=0.04, probe=(88, 60)),
], ids=["orb", "beam", "ring", "arc", "ribbon", "radial_flash", "glow", "ember", "smoke", "spark"])
def test_primitive_brightens_expected_region(make, build):
    r = make()
    rl, hot, far = build()
    rl.bloom = 0.0            # judge the primitive itself; bloom has its own test
    base = grey(70)
    out = render(r, base, rl)
    assert_no_gl_error(r)
    gain = lum(out[hot[1], hot[0]]) - 70
    assert gain > 4.0, f"expected a brighter pixel at {hot}, gain {gain:.1f}"
    assert abs(lum(out[far[1], far[0]]) - 70) <= 2.5, f"pixel {far} should be untouched"


def test_spark_streak_trails_behind_the_particle(make):
    r = make()
    rl = RenderList()
    rl.sprites.append(SpriteBatch(SpriteShape.SPARK, np.array([[160.0, 90.0]], np.float32),
                                  np.array([2.0], np.float32), np.array([[2.0, 2.0, 2.0, 1.0]], np.float32),
                                  np.array([[1000.0, 0.0]], np.float32), 0.04))
    rl.bloom = 0.0
    out = render(r, grey(0), rl).astype(int)
    behind = out[90, 125:155].mean()       # streak = 40 px trailing to the left
    ahead = out[90, 166:196].mean()
    assert behind > 20 and ahead < 3


def test_frontal_beam_fills_the_screen(make):
    r = make()
    rl = RenderList()
    rl.frontal_beams.append(FrontalBeam((160, 90), 30, BLUE, time=1.0))
    rl.bloom = 0.0
    out = render(r, grey(60), rl).astype(int)
    assert lum(out[90, 160]) > 230                    # brilliant disc
    assert out[90, 160 + 75, 2] > 60 + 4              # soft halo (deep blue) at 2.5 radii
    assert 58 <= out[5, 5, 2] <= 70                   # the room stays visible at the corners
    assert_no_gl_error(r)


def test_uniform_flash_whitens_everything(make):
    r = make()
    rl = RenderList()
    rl.flashes.append(Flash(1.0))
    out = render(r, grey(40), rl)
    assert out.min() >= 250


def test_hdr_cores_go_white_even_for_saturated_palettes(make):
    r = make()
    rl = RenderList()
    rl.orbs.append(Orb((160, 90), 30, get_palette("red"), intensity=2.0, charge=1.0, time=1.0))
    out = render(r, grey(20), rl)
    assert out[90, 160].min() >= 245                  # white-hot, not red


def test_bloom_spreads_light(make):
    r = make()

    def frame(bloom: float) -> np.ndarray:
        rl = RenderList()
        rl.sprites.append(SpriteBatch(SpriteShape.GLOW, np.array([[160.0, 90.0]], np.float32),
                                      np.array([10.0], np.float32), np.array([[20.0, 20.0, 20.0, 1.0]], np.float32)))
        rl.bloom = bloom
        return render(r, grey(0), rl).astype(int)

    yy, xx = np.mgrid[0:H, 0:W]
    annulus = (np.hypot(xx - 160, yy - 90) > 14) & (np.hypot(xx - 160, yy - 90) < 40)
    no_bloom, with_bloom = frame(0.0), frame(1.0)
    assert no_bloom[annulus].mean() < 1.0             # the sprite itself ends at its radius
    assert with_bloom[annulus].mean() > no_bloom[annulus].mean() + 3


# ----------------------------------------------------------------------- grade


def test_grade_darken_darkens(make):
    r = make()
    rl = RenderList()
    rl.grade = Grade(darken=0.5)
    out = render(r, grey(200), rl)
    assert abs(lum(out) - 100) <= 3


def test_grade_desaturate_tint_and_vignette(make):
    r = make()
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (30, 90, 220)                            # BGR orange
    rl = RenderList()
    rl.grade = Grade(desaturate=1.0)
    out = render(r, img, rl).astype(int)
    assert np.ptp(out[90, 160]) <= 2                  # grey
    rl.grade = Grade(tint=(0.0, 0.0, 1.0), tint_strength=1.0)
    out = render(r, grey(150), rl).astype(int)
    assert out[90, 160, 0] <= 2 and out[90, 160, 2] >= 148
    rl.grade = Grade(vignette=1.0)
    out = render(r, grey(150), rl)
    assert lum(out[2, 2]) < lum(out[90, 160]) - 30


def test_scene_light_brightens_near_the_light_only(make):
    r = make()
    rl = RenderList()
    rl.lights.append(Light((160, 90), 30, (1.0, 0.5, 0.25), 1.0))
    out = render(r, grey(100), rl).astype(int)
    c = out[90, 160]
    assert c[0] > 180 and c[0] > c[1] > c[2] > 100    # lit = cam * (1 + L) + 0.06 L, coloured
    assert abs(out[2, 2] - 100).max() <= 1            # beyond 3 radii: untouched


def test_refraction_follows_the_quality_flag(make):
    yy, xx = np.mgrid[0:H, 0:W]
    checker = ((((xx // 10) + (yy // 10)) % 2) * 200 + 30).astype(np.uint8)
    img = np.dstack([checker] * 3)
    rl = RenderList()
    rl.rings.append(Ring((160, 90), 50, 6, BLUE, intensity=0.0, refraction=8.0))   # pure distortion
    hi = render(make(quality="high"), img, rl).astype(int)
    lo = render(make(quality="low"), img, rl).astype(int)
    ref = img[:, :, ::-1].astype(int)
    band = np.abs(np.hypot(xx - 160, yy - 90) - 50) < 6
    far = np.hypot(xx - 160, yy - 90) > 90
    assert np.abs(hi - ref)[band].max() > 60          # displaced near the ring radius
    assert np.abs(hi - ref)[far].max() <= 2
    assert np.abs(lo - ref).max() <= 2                # low quality: no refraction


def test_chromatic_aberration_follows_the_quality_flag(make):
    img = np.zeros((H, W, 3), np.uint8)
    img[20:60, 20:80] = 255                           # sharp white block near the corner
    rl = RenderList()
    rl.grade = Grade(chromatic=6.0)
    hi = render(make(quality="high"), img, rl).astype(int)
    lo = render(make(quality="low"), img, rl).astype(int)
    split = np.abs(hi[..., 0] - hi[..., 2])
    assert split[15:65, 15:85].max() > 100            # red and blue separate along the edges
    assert np.abs(lo - img[:, :, ::-1].astype(int)).max() <= 2


# ------------------------------------------------------------------------ aura


def test_aura_renders_around_the_mask_and_uploads_once(make):
    r = make()
    mask = np.zeros((72, 128), np.uint8)
    mask[30:72, 48:80] = 255                          # a "person" standing in the middle
    rl = RenderList()
    rl.aura = Aura(mask, 7, BLUE, intensity=1.2, time=1.0, flame_height=0.1)
    rl.bloom = 0.0
    out = render(r, grey(40), rl).astype(int)
    above = out[int(H * 30 / 72) - 8: int(H * 30 / 72) - 2, int(W * 52 / 128): int(W * 76 / 128)]
    assert above.mean() > 40 + 10                      # flames rise above the silhouette
    assert abs(out[2:12, 2:40].mean() - 40) <= 1.5      # far away: untouched
    assert abs(out[H - 10, W // 2].mean() - 40) <= 3   # inside the body: (almost) untouched
    assert r.stats["mask_uploads"] == 1 and r.stats["mask_blurs"] == 1
    render(r, grey(40), rl)                           # same mask_id: no re-upload
    assert r.stats["mask_uploads"] == 1
    rl.aura = Aura(mask, 8, BLUE, time=1.0)
    render(r, grey(40), rl)
    assert r.stats["mask_uploads"] == 2
    assert_no_gl_error(r)


def test_empty_or_bad_aura_masks_are_ignored(make):
    r = make()
    rl = RenderList()
    rl.aura = Aura(np.zeros((36, 64), np.uint8), 1, BLUE)
    out = render(r, grey(50), rl)
    assert np.abs(out.astype(int) - 50).max() <= 2
    rl.aura = Aura(np.zeros((5,), np.uint8), 2, BLUE)          # wrong shape: skipped, no crash
    render(r, grey(50), rl)
    rl.aura = Aura(np.ones((36, 64), np.float32), 3, BLUE)     # float mask is converted
    render(r, grey(50), rl)
    assert_no_gl_error(r)


# -------------------------------------------------------------------------- UI


def test_ui_rect_fill_border_gradient_and_rounding(make):
    r = make()
    ui = UIDrawList()
    ui.add(UIRect(20, 20, 100, 60, (1.0, 0.0, 0.0, 1.0), radius=12, border_color=(0.0, 1.0, 0.0, 1.0),
                  border_width=3))
    ui.add(UIRect(150, 20, 100, 60, (0.0, 0.0, 1.0, 0.5)))
    ui.add(UIRect(20, 100, 100, 60, (1.0, 1.0, 1.0, 1.0), gradient_to=(0.0, 0.0, 0.0, 1.0)))
    out = render(r, grey(100), ui=ui).astype(int)
    assert tuple(out[50, 70]) == (255, 0, 0)                     # opaque fill
    assert out[50, 21, 1] > 200 and out[50, 21, 0] < 60         # border on the left edge
    assert abs(out[21, 21] - 100).max() <= 3                      # rounded corner stays background
    assert np.abs(out[50, 200] - np.array([50, 50, 177])).max() <= 3   # 50 % blue over grey (premultiplied)
    assert out[102, 70].mean() > out[158, 70].mean() + 150      # vertical gradient
    assert_no_gl_error(r)


def test_ui_text_shadow_glow_and_cache(make):
    r = make()
    ui = UIDrawList()
    ui.add(UIText("POWER 9001", 160, 90, 28, (1.0, 1.0, 1.0, 1.0), "bold", "center", "middle", glow=6.0,
                  glow_color=(0.2, 0.6, 1.0, 1.0), shadow=True))
    out = render(r, grey(0), ui=ui).astype(int)
    box = out[70:110, 80:240]
    assert (box.min(axis=2) > 200).sum() > 40            # white glyph pixels
    assert ((box[..., 2] > 40) & (box[..., 0] < box[..., 2])).sum() > 40   # blue glow around them
    assert out[5, 5].max() == 0
    stats = r.stats
    assert stats["text_textures"] == 2                   # glyphs + glow
    render(r, grey(0), ui=ui)
    assert r.stats["text_textures_created"] == stats["text_textures_created"]   # cached


def test_ui_text_cache_is_bounded_and_releases_old_textures(make):
    r = make()
    for start in range(0, TEXT_CACHE_LIMIT + 40, 60):
        ui = UIDrawList()
        for i in range(start, start + 60):
            ui.add(UIText(f"label {i}", 10, 10, 12, (1, 1, 1, 1)))
        render(r, None, ui=ui)
    stats = r.stats
    assert stats["text_textures"] <= TEXT_CACHE_LIMIT
    assert stats["text_textures_evicted"] >= 40
    assert_no_gl_error(r)


def test_ui_circles_arcs_and_lines(make):
    r = make()
    ui = UIDrawList()
    ui.add(UICircle(50, 50, 20, (1.0, 1.0, 0.0, 1.0)))
    ui.add(UICircle(130, 50, 20, (0.0, 1.0, 1.0, 1.0), width=4))
    ui.add(UIArc(220, 60, 30, -math.pi / 2, 0.0, (1.0, 0.0, 1.0, 1.0), width=6))   # top -> right, clockwise
    ui.add(UILine([(20, 150), (150, 150), (150, 110)], (0.0, 1.0, 0.0, 1.0), 4))
    out = render(r, grey(0), ui=ui).astype(int)
    assert tuple(out[50, 50]) == (255, 255, 0)              # filled disc
    assert out[50, 150, 1] > 200 and out[50, 130].max() == 0   # ring outline, hollow centre
    p = (220 + 30 * math.cos(-math.pi / 4), 60 + 30 * math.sin(-math.pi / 4))
    assert out[int(round(p[1])), int(round(p[0]))][0] > 200     # on the arc (upper right)
    q = (220 + 30 * math.cos(math.pi * 0.75), 60 + 30 * math.sin(math.pi * 0.75))
    assert out[int(round(q[1])), int(round(q[0]))].max() == 0   # outside the arc's span
    assert out[150, 80, 1] > 200 and out[130, 150, 1] > 200     # both line segments
    assert_no_gl_error(r)


# --------------------------------------------------------------- shake / zoom


def test_shake_moves_the_whole_image_without_showing_edges(make):
    r = make()
    img = np.full((H, W, 3), 255, np.uint8)
    img[80:100, 150:170] = 0                          # black marker at the centre
    rl = RenderList()
    rl.shake = (10.0, -6.0)
    out = render(r, img, rl)
    for edge in (out[0], out[-1], out[:, 0], out[:, -1]):
        assert edge.min() >= 250                      # overscan: no black borders
    ys, xs = np.nonzero(out.max(axis=2) < 60)
    assert 160 + 6 < xs.mean() < 160 + 14             # moved right
    assert 90 - 10 < ys.mean() < 90 - 2               # and up


def test_effects_shake_together_with_the_camera(make):
    r = make()
    img = np.zeros((H, W, 3), np.uint8)
    rl = RenderList()
    rl.sprites.append(SpriteBatch(SpriteShape.EMBER, np.array([[160.0, 90.0]], np.float32),
                                  np.array([4.0], np.float32), np.array([[3.0, 3.0, 3.0, 1.0]], np.float32)))
    rl.bloom = 0.0
    still = render(r, img, rl).astype(int).sum(axis=2)
    rl.shake = (12.0, 0.0)
    shaken = render(r, img, rl).astype(int).sum(axis=2)
    dx = np.unravel_index(shaken.argmax(), shaken.shape)[1] - np.unravel_index(still.argmax(), still.shape)[1]
    assert 9 <= dx <= 15


def test_zoom_punch_magnifies_around_the_centre(make):
    r = make()
    img = np.full((H, W, 3), 255, np.uint8)
    img[80:100, 150:170] = 0
    rl = RenderList()
    rl.grade = Grade(zoom=2.0)
    out = render(r, img, rl)
    ys, xs = np.nonzero(out.max(axis=2) < 128)
    assert 36 <= xs.max() - xs.min() + 1 <= 44        # 20 px marker -> ~40 px


# ------------------------------------------------------ resize / quality / ids


def test_resize_recreates_targets(make):
    r = make()
    render(r, grey(80))
    r.resize(400, 300)
    assert r.size == (400, 300)
    img = np.random.default_rng(3).integers(0, 256, (300, 400, 3), dtype=np.uint8)
    out = render(r, img)
    assert out.shape == (300, 400, 3)
    assert np.abs(out.astype(int) - img[:, :, ::-1].astype(int)).max() <= 2
    r.resize(400, 300)                                # same size: no-op
    with pytest.raises(RendererError):
        r.resize("bad", 3)                            # invalid sizes raise a RendererError


def test_set_quality_changes_effect_resolution_and_bloom_levels(make):
    r = make((400, 200), quality="high")
    assert r._fx_t.size == (400, 200) and len(r._bloom) == 6
    r.set_quality(get_quality("low"))
    assert r.quality.name == "low"
    assert r._fx_t.size == (200, 100) and len(r._bloom) == 4
    rl, _, _ = _orb()
    out = render(r, grey(50, 400, 200), rl, scene_size=(320, 180))
    assert out.max() > 200
    r.set_quality(get_quality("ultra"))
    assert len(r._bloom) == 7
    assert_no_gl_error(r)


def test_camera_upload_is_skipped_for_unchanged_image_id(make):
    r = make()
    a, b = grey(30), grey(220)
    out_a = render(r, a, image_id=5)
    uploads = r.stats["camera_uploads"]
    out_same_id = render(r, b, image_id=5)            # same id: the old texture is reused
    assert r.stats["camera_uploads"] == uploads
    assert np.array_equal(out_a, out_same_id)
    out_new = render(r, b, image_id=6)
    assert r.stats["camera_uploads"] == uploads + 1 and lum(out_new) > 200
    render(r, a, image_id=-1)                         # unknown id: always uploaded
    render(r, a, image_id=-1)
    assert r.stats["camera_uploads"] == uploads + 3


def test_unusual_camera_images_are_converted(make):
    r = make()
    gray = np.full((H, W), 100, np.uint8)
    assert abs(lum(render(r, gray)) - 100) <= 2
    bgra = np.zeros((H, W, 4), np.uint8)
    bgra[..., :3] = (10, 20, 200)
    assert render(r, bgra)[5, 5, 0] >= 198
    flt = np.full((H, W, 3), 0.5, np.float32)
    assert abs(lum(render(r, flt)) - 127.5) <= 3
    assert_no_gl_error(r)


def test_release_is_idempotent_and_blocks_further_use(make):
    r = make()
    render(r, grey(10))
    r.release()
    r.release()
    assert r.released
    with pytest.raises(RendererError):
        render(r, grey(10))
    with pytest.raises(RendererError):
        r.read_pixels()


# ----------------------------------------------------------------- robustness


def test_malformed_primitives_are_skipped_not_fatal(make):
    r = make()
    nan = float("nan")
    rl = RenderList()
    rl.orbs.append(Orb((nan, 20.0), 10, BLUE))                          # NaN centre
    rl.orbs.append(Orb((50, 50), -3, BLUE))                             # negative radius
    rl.beams.append(Beam((10, 10), (0.0, 0.0), 100, 5, 5, BLUE))         # zero direction
    rl.beams.append(Beam((nan, 10), (1.0, 0.0), 100, 5, 5, BLUE))        # NaN origin
    rl.frontal_beams.append(FrontalBeam((nan, nan), 30, BLUE))
    rl.arcs.append(Arc(np.zeros((1, 2), np.float32), 1, BLUE))           # a single point
    rl.ribbons.append(Ribbon(np.zeros((4, 2), np.float32), np.ones(2, np.float32), np.ones(9, np.float32), BLUE))
    rl.sprites.append(SpriteBatch(SpriteShape.GLOW, np.zeros((5, 2), np.float32), np.ones(3, np.float32),
                                  np.ones((5, 4), np.float32)))           # mismatched lengths
    rl.sprites.append(SpriteBatch(SpriteShape.GLOW, np.zeros((0, 2), np.float32), np.zeros(0, np.float32),
                                  np.zeros((0, 4), np.float32)))          # empty batch
    rl.rings.append(Ring((nan, 100), 20.0, 3.0, BLUE, refraction=5.0))
    rl.lights.append(Light((float("inf"), 0.0), 10, (1, 1, 1), 1.0))
    rl.flashes.append(Flash(nan))
    rl.flashes.append(Flash(1.0, center=(nan, 0.0), radius=20))
    rl.bloom = 0.0
    out = render(r, grey(60), rl)
    assert out.shape == (H, W, 3)
    assert np.abs(out.astype(int) - 60).max() <= 2     # nothing was drawn at a made-up position
    assert_no_gl_error(r)


def test_many_sprites_grow_the_instance_buffer(make):
    r = make()
    n = 12000
    rng = np.random.default_rng(5)
    rl = RenderList()
    rl.sprites.append(SpriteBatch(SpriteShape.EMBER, rng.uniform((0, 0), (W, H), (n, 2)).astype(np.float32),
                                  np.full(n, 2.0, np.float32), np.full((n, 4), 0.5, np.float32)))
    out = render(r, grey(0), rl)
    assert lum(out) > 5
    assert_no_gl_error(r)


def test_shader_compile_errors_raise_renderer_error(gl_context, fonts, monkeypatch):
    broken = dict(gl_shaders.PROGRAMS)
    broken["copy"] = (gl_shaders.FULLSCREEN_VS, "void main() { this is not glsl; }")
    monkeypatch.setattr(gl_shaders, "PROGRAMS", broken)
    with pytest.raises(RendererError) as info:
        GLRenderer(gl_context, (64, 64), get_quality("low"), fonts, offscreen=True)
    assert "copy" in str(info.value)


def test_strip_vertices_builds_separate_polylines():
    pts = np.array([[0, 0], [10, 0], [20, 0], [100, 50], [100, 60]], np.float32)
    same = np.array([True, True, False, True])        # polylines: 3 points + 2 points
    n = len(pts)
    verts = strip_vertices(pts, np.full(n, 2.0, np.float32), np.full(n, 0.2, np.float32),
                           np.ones((n, 3), np.float32), np.ones((n, 3), np.float32), same)
    assert verts.shape == (6 * 3, 10)                 # 3 segments, 2 triangles each
    assert set(np.unique(verts[:, 2])) == {-1.0, 1.0}
    first = verts[:12, 1]                             # horizontal line: offsets +-2 in y
    assert np.allclose(np.abs(first), 2.0)
    assert np.all(verts[12:, 0] > 90)                 # no triangle bridges the two polylines


# ---------------------------------------------------------------------- factory


def test_factory_rejects_unknown_kind(fonts):
    with pytest.raises(RendererError):
        create_renderer("vulkan", (64, 64), get_quality("low"), fonts)


def test_factory_gl_raises_when_opengl_fails(fonts, monkeypatch):
    import moderngl

    def boom(*args, **kwargs):
        raise moderngl.Error("no context")

    monkeypatch.setattr(moderngl, "create_context", boom)
    with pytest.raises(RendererError):
        create_renderer("gl", (64, 64), get_quality("low"), fonts)
    with pytest.raises(RendererError):
        create_renderer("gl", (64, 64), get_quality("low"), fonts, window_has_gl=False)


def test_factory_auto_falls_back_to_cpu_renderer(fonts, monkeypatch):
    renderer_cpu = pytest.importorskip("dragon_ball_powers.renderer_cpu")
    import moderngl

    def boom(*args, **kwargs):
        raise moderngl.Error("no context")

    monkeypatch.setattr(moderngl, "create_context", boom)
    r = create_renderer("auto", (64, 64), get_quality("low"), fonts)
    try:
        assert isinstance(r, renderer_cpu.CpuRenderer)
    finally:
        r.release()
    r = create_renderer("auto", (64, 64), get_quality("low"), fonts, window_has_gl=False)
    try:
        assert isinstance(r, renderer_cpu.CpuRenderer)
    finally:
        r.release()


_WINDOW_SCRIPT = textwrap.dedent("""
    import json, os, sys
    import numpy as np, pygame
    from dragon_ball_powers.fonts import FontManager
    from dragon_ball_powers.primitives import (Orb, RenderList, SceneInput, UIDrawList, UIRect,
                                               compute_view_transform, get_palette, get_quality)
    from dragon_ball_powers.renderer import GLRenderer, create_renderer
    pygame.init()
    for attr, val in ((pygame.GL_CONTEXT_MAJOR_VERSION, 3), (pygame.GL_CONTEXT_MINOR_VERSION, 3),
                      (pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)):
        pygame.display.gl_set_attribute(attr, val)
    try:
        pygame.display.set_mode((320, 180), pygame.OPENGL | pygame.DOUBLEBUF)
    except pygame.error as exc:
        print(json.dumps({"skip": str(exc)})); sys.exit(0)
    r = create_renderer("gl", (320, 180), get_quality("medium"), FontManager())
    img = np.full((180, 320, 3), 60, np.uint8)
    rl = RenderList(); rl.orbs.append(Orb((160, 90), 25, get_palette("blue"), time=1.0))
    ui = UIDrawList(); ui.add(UIRect(10, 10, 40, 20, (1, 0, 0, 1)))
    view = compute_view_transform(320, 180, 320, 180)
    r.render(SceneInput(img, 1, (320, 180), 1.0), rl, ui, view)
    error = r.ctx.error
    screen = np.frombuffer(r.ctx.screen.read(components=4), np.uint8).reshape(180, 320, 4)[::-1]
    _ = r.ctx.error   # reading framebuffer 0 itself flags an error on some drivers (test-only read)
    pygame.display.flip()
    px = r.read_pixels()                               # still valid after the buffer swap
    r.resize(200, 100)
    r.render(SceneInput(img, 2, (320, 180), 1.0), rl, ui, compute_view_transform(320, 180, 200, 100))
    error2 = r.ctx.error
    out = {"type": type(r).__name__, "offscreen": r.offscreen, "center": px[90, 160].tolist(),
           "rect": px[20, 30].tolist(), "screen_center": screen[90, 160, :3].tolist(), "error": error,
           "error2": error2, "resized": list(r.read_pixels().shape)}
    r.release()
    pygame.quit()
    print(json.dumps(out))
""")


@pytest.mark.slow
def test_window_renderer_in_a_pygame_opengl_window(tmp_path):
    if not sys.platform.startswith("linux"):
        pytest.skip("headless OpenGL window test uses SDL's offscreen driver (Linux)")
    env = dict(os.environ, SDL_VIDEODRIVER="offscreen", SDL_AUDIODRIVER="dummy", PYGAME_HIDE_SUPPORT_PROMPT="1",
               PYTHONPATH=str(ROOT / "src"))
    script = tmp_path / "window_check.py"
    script.write_text(_WINDOW_SCRIPT, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=180)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        if "OpenGL" in proc.stderr or "GL" in proc.stderr:
            pytest.skip(f"no OpenGL window available: {proc.stderr[-300:]}")
        pytest.fail(f"window check failed (rc={proc.returncode}): {proc.stderr[-2000:]}")
    data = json.loads(lines[-1])
    if "skip" in data:
        pytest.skip(f"no OpenGL window available: {data['skip']}")
    assert data["type"] == "GLRenderer" and data["offscreen"] is False
    assert data["error"] == "GL_NO_ERROR" and data["error2"] == "GL_NO_ERROR"
    assert min(data["center"]) > 200                  # orb core on screen
    assert data["rect"] == [255, 0, 0]                # UI drawn on top
    assert min(data["screen_center"]) > 200           # presented to the window's framebuffer
    assert data["resized"] == [100, 200, 3]


# ------------------------------------------------------------------ performance


def _heavy_scene(rng: np.random.Generator, t: float) -> RenderList:
    rl = RenderList()
    rl.orbs.append(Orb((576, 396), 60, BLUE, intensity=1.2, charge=0.8, time=t, flare=0.6, instability=0.4))
    rl.beams.append(Beam((640, 360), (1.0, -0.05), 640, 36, 48, BLUE, intensity=1.2, time=t, head_radius=60))
    n = 3000
    pos = rng.uniform((0, 0), (1280, 720), (n, 2)).astype(np.float32)
    sizes = rng.uniform(1.5, 8.0, n).astype(np.float32)
    colors = np.ones((n, 4), np.float32)
    colors[:, :3] = np.asarray(BLUE.spark) * 2.0
    vel = rng.normal(0, 600, (n, 2)).astype(np.float32)
    for i, shape in enumerate(SpriteShape):
        sl = slice(i * n // 4, (i + 1) * n // 4)
        rl.sprites.append(SpriteBatch(shape, pos[sl], sizes[sl], colors[sl], vel[sl], 0.02))
    for i in range(8):
        end = (576 + math.cos(i) * 200, 396 + math.sin(i) * 200)
        pts = np.linspace((576, 396), end, 33).astype(np.float32) + rng.normal(0, 6, (33, 2)).astype(np.float32)
        rl.arcs.append(Arc(pts, 1.5, BLUE, 1.2, 1.0))
    for i in range(4):
        rl.rings.append(Ring((576, 396), 100 + 80 * i, 8, BLUE, refraction=12, time=t))
    mask = np.zeros((144, 256), np.uint8)
    mask[40:144, 100:156] = 255
    mask[10:40, 112:144] = 255
    rl.aura = Aura(mask, 1, BLUE, time=t)
    rl.lights.append(Light((576, 396), 360, BLUE.inner, 1.0))
    rl.grade = Grade(darken=0.4, vignette=0.3)
    rl.shake = (3.0, -2.0)
    return rl


def test_performance_smoke_heavy_scene_720p(make, record_property):
    """Generous bound only (shared CI machines); the measured value is reported."""
    r = make((1280, 720), quality="high")
    rng = np.random.default_rng(1)
    cam = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    view = compute_view_transform(1280, 720, 1280, 720)
    times = []
    for i in range(36):
        rl = _heavy_scene(rng, i / 60.0)
        t0 = time.perf_counter()
        r.render(SceneInput(cam, 1000 + i, (1280, 720), i / 60.0), rl, UIDrawList(), view)
        r.ctx.finish()
        if i >= 6:                                    # skip shader warm-up frames
            times.append((time.perf_counter() - t0) * 1000.0)
    median = float(np.median(times))
    record_property("heavy_scene_720p_high_ms", round(median, 2))
    print(f"\nheavy scene 1280x720 high: median {median:.2f} ms/frame (render + ctx.finish)")
    assert_no_gl_error(r)
    assert median < 250.0
