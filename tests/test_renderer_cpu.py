"""Tests for the NumPy/OpenCV fallback renderer (no GPU, no display needed)."""

from __future__ import annotations

import time

import numpy as np
import pygame
import pytest

from dragon_ball_powers.fonts import FontManager
from dragon_ball_powers.interfaces import Renderer
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
    Ring,
    SceneInput,
    SpriteBatch,
    SpriteShape,
    UIDrawList,
    UIRect,
    UIText,
    compute_view_transform,
    get_palette,
    get_quality,
)
from dragon_ball_powers.renderer_cpu import CpuRenderer

W, H = 320, 180
BLUE = get_palette("blue")


@pytest.fixture(scope="module")
def fonts():
    pygame.init()
    return FontManager()


@pytest.fixture
def renderer(fonts):
    r = CpuRenderer((W, H), get_quality("high"), fonts)
    yield r
    r.release()


def grey(level: int = 60) -> np.ndarray:
    return np.full((H, W, 3), level, np.uint8)


def render(r: CpuRenderer, image: np.ndarray | None, rl: RenderList | None = None,
           ui: UIDrawList | None = None, mode: str = "cover") -> np.ndarray:
    size = (image.shape[1], image.shape[0]) if image is not None else (W, H)
    view = compute_view_transform(size[0], size[1], W, H, mode)
    r.render(SceneInput(image, 1, size, 1.0), rl or RenderList(), ui or UIDrawList(), view)
    return r.read_pixels()


def lum(px: np.ndarray) -> float:
    return float(np.asarray(px, np.float32).mean())


def test_is_a_renderer(renderer):
    assert isinstance(renderer, Renderer)
    assert renderer.info["renderer"] == "CPU"


def test_identity_without_effects(renderer):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    out = render(renderer, img)
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    # read_pixels is RGB, the camera image is BGR
    assert np.abs(out.astype(int) - img[:, :, ::-1].astype(int)).max() <= 2


@pytest.mark.parametrize("make", [
    lambda: Orb((160, 90), 20, BLUE, intensity=1.6, charge=0.7),
    lambda: Beam((40, 90), (1.0, 0.0), 260, 12, 16, BLUE, intensity=2.0, head_radius=18),
    lambda: FrontalBeam((160, 90), 40, BLUE, intensity=2.2),
    lambda: Ring((160, 90), 50, 8, BLUE, intensity=1.5),
    lambda: Arc(np.array([[120, 90], [140, 80], [160, 95], [200, 90]], np.float32), 1.5, BLUE, intensity=2.0),
    lambda: Flash(1.5, center=(160, 90), radius=60),
])
def test_each_primitive_brightens_its_region(renderer, make):
    prim = make()
    rl = RenderList()
    {Orb: rl.orbs, Beam: rl.beams, FrontalBeam: rl.frontal_beams, Ring: rl.rings, Arc: rl.arcs,
     Flash: rl.flashes}[type(prim)].append(prim)
    base = render(renderer, grey())
    out = render(renderer, grey(), rl)
    assert lum(out) > lum(base) + 1.0, type(prim).__name__
    assert np.isfinite(out.astype(np.float32)).all()


def test_sprites_and_lights(renderer):
    rl = RenderList()
    n = 300
    rng = np.random.default_rng(2)
    rl.sprites.append(SpriteBatch(
        SpriteShape.GLOW, rng.uniform([0, 0], [W, H], (n, 2)).astype(np.float32),
        np.full(n, 3.0, np.float32), np.tile(np.array([[0.5, 0.8, 1.0, 1.0]], np.float32), (n, 1))))
    rl.lights.append(Light((160, 90), 80, (0.5, 0.8, 1.0), 2.0))
    base = render(renderer, grey())
    out = render(renderer, grey(), rl)
    assert lum(out) > lum(base) + 2.0


def test_grade_darkens_the_room(renderer):
    rl = RenderList(grade=Grade(darken=0.5))
    out = render(renderer, grey(120), rl)
    assert lum(out) < 120 * 0.7


def test_aura_glows_around_a_mask(renderer):
    mask = np.zeros((90, 160), np.uint8)
    mask[30:80, 60:100] = 255
    rl = RenderList(aura=Aura(mask, 1, BLUE, intensity=1.2, time=1.0))
    base = render(renderer, grey(40))
    out = render(renderer, grey(40), rl)
    assert lum(out) > lum(base) + 0.5


def test_ui_text_and_rect_are_drawn(renderer):
    ui = UIDrawList([UIRect(10, 10, 100, 40, (1.0, 0.0, 0.0, 1.0), radius=6),
                     UIText("POWER", 200, 20, 24, (1.0, 1.0, 1.0, 1.0), font="bold")])
    out = render(renderer, grey(30), ui=ui)
    assert out[30, 50, 0] > 200 and out[30, 50, 1] < 80          # red rectangle
    assert out[15:45, 195:300].max() > 200                        # white text


def test_contain_mode_letterboxes(renderer):
    square = np.full((180, 180, 3), 200, np.uint8)
    out = render(renderer, square, mode="contain")
    assert out[:, :40].max() < 10 and out[:, -40:].max() < 10     # black bars left/right
    assert out[90, 160].min() > 150


def test_placeholder_when_no_camera(renderer):
    out = render(renderer, None)
    assert out.shape == (H, W, 3)
    assert out.max() > 0                                          # animated background, not pure black


def test_renders_into_a_pygame_surface(fonts):
    surface = pygame.Surface((W, H))
    r = CpuRenderer((W, H), get_quality("low"), fonts, target=surface)
    try:
        img = grey(90)
        r.render(SceneInput(img, 1, (W, H), 0.0), RenderList(), UIDrawList(), compute_view_transform(W, H, W, H))
        px = surface.get_at((W // 2, H // 2))
        assert abs(px.r - 90) <= 2 and abs(px.g - 90) <= 2
    finally:
        r.release()


def test_resize_quality_and_release(renderer):
    renderer.resize(200, 120)
    renderer.set_quality(get_quality("low"))
    view = compute_view_transform(W, H, 200, 120)
    renderer.render(SceneInput(grey(), 1, (W, H), 0.0), RenderList(), UIDrawList(), view)
    assert renderer.read_pixels().shape == (120, 200, 3)
    renderer.release()
    renderer.release()


def test_busy_scene_performance(fonts):
    r = CpuRenderer((1280, 720), get_quality("medium"), fonts)
    try:
        rl = RenderList()
        rng = np.random.default_rng(3)
        n = 800
        rl.sprites.append(SpriteBatch(
            SpriteShape.GLOW, rng.uniform([0, 0], [1280, 720], (n, 2)).astype(np.float32),
            np.full(n, 4.0, np.float32), np.tile(np.array([[0.5, 0.8, 1.0, 1.0]], np.float32), (n, 1))))
        rl.orbs.append(Orb((640, 400), 70, BLUE, intensity=1.8, charge=0.8))
        img = np.full((720, 1280, 3), 50, np.uint8)
        view = compute_view_transform(1280, 720, 1280, 720)
        r.render(SceneInput(img, 1, (1280, 720), 0.0), rl, UIDrawList(), view)   # warm-up
        t = time.perf_counter()
        for i in range(10):
            r.render(SceneInput(img, 1, (1280, 720), i * 0.03), rl, UIDrawList(), view)
        ms = (time.perf_counter() - t) / 10 * 1000
        print(f"CPU renderer 720p charging scene: {ms:.1f} ms/frame")
        assert ms < 250  # generous: other tests may load the machine
    finally:
        r.release()
