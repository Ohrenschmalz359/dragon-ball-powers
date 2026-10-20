"""Tests for the effect simulation (no GPU, camera or display needed)."""

from __future__ import annotations

import dataclasses
import math
import time
import warnings

import numpy as np
import pytest

from dragon_ball_powers.config import EffectsSettings
from dragon_ball_powers.datatypes import ArmState, BodyState, PoseObservation
from dragon_ball_powers.effects import (
    MAX_LIGHTS,
    SYNTH_MASK_ID_BASE,
    AuraEffect,
    BeamEffect,
    EffectSystem,
    FrontalBeamEffect,
    OrbEffect,
    make_bolt,
)
from dragon_ball_powers.primitives import (
    PALETTES,
    Grade,
    RenderList,
    SpriteShape,
    get_quality,
)

W, H = 1280, 720
DT = 1.0 / 60.0
LISTS = ("sprites", "orbs", "beams", "frontal_beams", "arcs", "rings", "ribbons", "flashes", "lights")


class Sim:
    """Drives an EffectSystem at 60 Hz like the app does (update, clear, collect)."""

    def __init__(self, quality: str = "high", size: tuple[int, int] = (W, H), seed: int = 7, **settings) -> None:
        self.settings = EffectsSettings(**settings)
        self.fx = EffectSystem(self.settings, get_quality(quality), size, seed=seed)
        self.now = 100.0
        self.out = RenderList()

    def step(self, n: int = 1, before=None) -> RenderList:
        for _ in range(n):
            if before is not None:
                before()
            self.now += DT
            self.fx.update(DT, self.now)
            self.out.clear()
            self.fx.collect(self.out)
        return self.out

    def frames(self, n: int, before=None):
        for _ in range(n):
            yield self.step(1, before)


def assert_finite(out: RenderList) -> None:
    """Every number reaching the renderer must be finite (and plain floats for scalars)."""

    def check(value, where: str) -> None:
        if isinstance(value, (float, np.floating)):
            assert math.isfinite(value), where
        elif isinstance(value, tuple):
            for i, v in enumerate(value):
                check(v, f"{where}[{i}]")
        elif isinstance(value, np.ndarray) and value.dtype.kind == "f":
            assert np.isfinite(value).all(), where

    for name in LISTS:
        for i, prim in enumerate(getattr(out, name)):
            for f in dataclasses.fields(prim):
                if f.name != "palette":
                    check(getattr(prim, f.name), f"{name}[{i}].{f.name}")
    if out.aura is not None:
        for f in dataclasses.fields(out.aura):
            if f.name != "palette":
                check(getattr(out.aura, f.name), f"aura.{f.name}")
    for f in dataclasses.fields(out.grade):
        check(getattr(out.grade, f.name), f"grade.{f.name}")
    check(out.shake, "shake")
    check(out.bloom, "bloom")


def sprite_count(out: RenderList, shape: SpriteShape | None = None) -> int:
    return sum(len(b) for b in out.sprites if shape is None or b.shape == shape)


def batch(out: RenderList, shape: SpriteShape):
    for b in out.sprites:
        if b.shape == shape:
            return b
    return None


def make_pose(hips_visible: bool = False) -> PoseObservation:
    lm = np.zeros((33, 3), np.float32)
    vis = np.zeros(33, np.float32)

    def put(i: int, x: float, y: float, v: float = 0.95) -> None:
        lm[i, 0] = x
        lm[i, 1] = y
        vis[i] = v

    put(0, 640, 200)                                   # nose
    put(7, 600, 195), put(8, 680, 195)                 # ears
    put(11, 540, 330), put(12, 740, 330)               # shoulders
    put(13, 500, 450), put(14, 780, 450)               # elbows
    put(15, 570, 520), put(16, 710, 520)               # wrists
    put(19, 585, 545), put(20, 695, 545)               # index fingers
    if hips_visible:
        put(23, 575, 650), put(24, 705, 650)
    return PoseObservation(landmarks=lm, visibility=vis, score=0.9)


def make_body(energy: float = 0.3) -> BodyState:
    body = BodyState(timestamp=0.0, scene_size=(W, H), quality="full", hands_detected=2, body_scale=200.0,
                     body_center=(640.0, 420.0), energy=energy)
    body.left = ArmState("left", present=True, source="hand", hand_center=(560.0, 520.0), hand_size=70.0)
    body.right = ArmState("right", present=True, source="hand", hand_center=(720.0, 520.0), hand_size=70.0)
    return body


# ----------------------------------------------------------------------- basics


def test_empty_system_is_identity():
    sim = Sim()
    fx = sim.fx
    assert fx.scene_scale == pytest.approx(1.0)
    assert fx.palette is PALETTES["blue"]
    assert fx.particle_count == 0
    assert fx.particle_budget == 3200
    out = RenderList()
    fx.collect(out)
    assert out.is_empty()
    assert out.grade == Grade()
    assert out.shake == (0.0, 0.0)
    assert out.lights == []
    out = sim.step(30)
    assert out.is_empty() and out.grade == Grade() and out.shake == (0.0, 0.0)


def test_bloom_follows_settings_and_scene_scale_follows_height():
    sim = Sim(bloom_strength=2.25, size=(1920, 1080))
    out = sim.step()
    assert out.bloom == pytest.approx(2.25)
    assert sim.fx.scene_scale == pytest.approx(1.5)
    sim.fx.apply_settings(EffectsSettings(bloom_strength=0.5))
    assert sim.step().bloom == pytest.approx(0.5)


def test_make_bolt_shape_endpoints_and_branches():
    rng = np.random.default_rng(1)
    a, b = (10.0, 20.0), (210.0, 60.0)
    bolt = make_bolt(a, b, rng, segments=16, jaggedness=0.25)
    assert bolt.dtype == np.float32 and bolt.shape == (17, 2)
    assert np.allclose(bolt[0], a) and np.allclose(bolt[-1], b)
    assert np.isfinite(bolt).all()
    # jagged, but stays close to the chord
    d = np.array(b) - np.array(a)
    n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    offsets = (bolt - np.array(a)) @ n
    assert np.abs(offsets).max() > 1.0
    assert np.abs(offsets).max() < 0.5 * np.linalg.norm(d)
    assert make_bolt(a, b, rng, segments=20).shape == (33, 2)   # rounded up to a power of two
    straight = make_bolt(a, b, rng, segments=8, jaggedness=0.0)
    assert np.allclose(np.abs((straight - np.array(a)) @ n), 0.0, atol=1e-3)
    branches: list[np.ndarray] = []
    make_bolt(a, b, rng, segments=32, jaggedness=0.25, branch_prob=1.0, branches=branches)
    assert 1 <= len(branches) <= 3
    for br in branches:
        assert br.dtype == np.float32 and br.ndim == 2 and br.shape[1] == 2 and len(br) >= 5


def test_make_bolt_is_deterministic():
    b1 = make_bolt((0, 0), (100, 0), np.random.default_rng(5), 16, 0.3)
    b2 = make_bolt((0, 0), (100, 0), np.random.default_rng(5), 16, 0.3)
    b3 = make_bolt((0, 0), (100, 0), np.random.default_rng(6), 16, 0.3)
    assert np.array_equal(b1, b2)
    assert not np.array_equal(b1, b3)


# ------------------------------------------------------------------------- orb


def test_orb_follows_target_and_draws_primitives():
    sim = Sim()
    orb = sim.fx.create_orb((400.0, 300.0), 50.0)
    assert isinstance(orb, OrbEffect) and orb.alive
    target = (640.0, 360.0)
    left, right = (530.0, 360.0), (750.0, 360.0)
    out = sim.step(20, lambda: orb.set_target(target, 60.0, 0.6, left, right))
    assert math.dist(orb.center, target) < 1.0
    assert orb.radius == pytest.approx(60.0, rel=0.1)
    assert orb.charge == pytest.approx(0.6)
    assert len(out.orbs) == 1
    prim = out.orbs[0]
    assert math.dist(prim.center, orb.center) < 1e-6
    assert prim.palette is PALETTES["blue"]
    assert prim.charge == pytest.approx(0.6)
    assert sprite_count(out, SpriteShape.GLOW) > 0
    assert sprite_count(out, SpriteShape.EMBER) > 0
    assert sprite_count(out, SpriteShape.SPARK) > 0
    sparks = batch(out, SpriteShape.SPARK)
    assert sparks.velocities is not None and sparks.velocities.shape == (len(sparks), 2)
    assert sparks.stretch == pytest.approx(0.035)
    assert out.arcs
    assert any(lt.radius == pytest.approx(6.0 * prim.radius, rel=0.01) for lt in out.lights)
    assert_finite(out)


def test_orb_primitive_follows_charge():
    sim = Sim()
    orb = sim.fx.create_orb((640.0, 360.0), 60.0)
    for charge in (0.0, 0.5, 1.0):
        out = sim.step(30, lambda c=charge: orb.set_target((640.0, 360.0), 60.0, c))
        prim = out.orbs[0]
        assert prim.instability == pytest.approx(charge * charge)
        assert prim.swirl == pytest.approx(1.0 + charge)
        assert prim.flare == pytest.approx((0.0, 0.0, 1.0)[int(charge * 2)])
        base = 1.1 + 0.9 * charge
        # +-4 % pulse plus a decaying milestone boost
        assert base * 0.95 <= prim.intensity <= base * 1.05 + 0.6


def test_orb_stays_between_fast_moving_hands():
    sim = Sim()
    orb = sim.fx.create_orb((300.0, 360.0), 50.0)
    state = {"x": 300.0}

    def move():
        state["x"] += 900.0 * DT          # hands sweep across the frame at 900 px/s
        mid = (state["x"], 360.0)
        orb.set_target(mid, 50.0, 0.5, (mid[0] - 80.0, 360.0), (mid[0] + 80.0, 360.0))

    sim.step(10, move)
    worst = 0.0
    for _ in range(40):
        sim.step(1, move)
        worst = max(worst, abs(orb.center[0] - state["x"]))
    assert worst < 40.0                  # well inside the 160 px gap between the hands


def test_orb_arcs_touch_hand_anchors():
    sim = Sim()
    orb = sim.fx.create_orb((640.0, 360.0), 60.0)
    left, right = (520.0, 360.0), (760.0, 360.0)
    hit = {"left": 0, "right": 0}
    starts_ok = True
    for out in sim.frames(60, lambda: orb.set_target((640.0, 360.0), 60.0, 0.8, left, right)):
        for arc in out.arcs:
            assert arc.points.dtype == np.float32 and arc.points.shape[1] == 2
            assert 1.2 * 0.6 <= arc.width <= 2.5 * 1.3
            end = tuple(arc.points[-1])
            if math.dist(end, left) < 0.5:
                hit["left"] += 1
                starts_ok &= abs(math.dist(tuple(arc.points[0]), orb.center) - 0.92 * out.orbs[0].radius) < 1.0
            if math.dist(end, right) < 0.5:
                hit["right"] += 1
    assert hit["left"] > 10 and hit["right"] > 10
    assert starts_ok
    # no anchors -> only surface-crawling arcs (charge > 0.5), all near the orb
    out = sim.step(10, lambda: orb.set_target((640.0, 360.0), 60.0, 0.9))
    for arc in out.arcs:
        d = np.hypot(arc.points[:, 0] - 640.0, arc.points[:, 1] - 360.0)
        assert d.max() < 2.0 * 60.0


def test_orb_arc_count_grows_with_charge():
    def hand_arcs(charge: float) -> float:
        sim = Sim()
        orb = sim.fx.create_orb((640.0, 360.0), 40.0)
        total = 0
        for out in sim.frames(60, lambda: orb.set_target((640.0, 360.0), 40.0, charge, (520.0, 360.0),
                                                         (760.0, 360.0))):
            total += sum(1 for a in out.arcs if a.points[-1][1] == pytest.approx(360.0, abs=40.0)
                         and min(abs(a.points[-1][0] - 520.0), abs(a.points[-1][0] - 760.0)) < 30.0)
        return total / 60.0

    assert hand_arcs(1.0) > 2.5 * hand_arcs(0.0)


def test_orb_milestone_pulses():
    sim = Sim()
    orb = sim.fx.create_orb((640.0, 360.0), 50.0)
    state = {"charge": 0.0}
    births = []
    prev = 0

    def ramp():
        state["charge"] = min(1.0, state["charge"] + DT / 3.0)   # full charge in 3 s
        orb.set_target((640.0, 360.0), 50.0, state["charge"], (540.0, 360.0), (740.0, 360.0))

    for out in sim.frames(200, ramp):
        if len(out.rings) > prev:
            births.append(state["charge"])
            ring = out.rings[-1]
            assert math.dist(ring.center, orb.center) < 2.0
            assert any(f.center is not None and math.dist(f.center, orb.center) < 2.0 for f in out.flashes)
        prev = len(out.rings)
    assert len(births) == 4
    for got, milestone in zip(births, (0.25, 0.5, 0.75, 1.0)):
        assert milestone - 1e-3 <= got <= milestone + 0.02


def test_orb_release_collapses_then_dies():
    sim = Sim()
    orb = sim.fx.create_orb((640.0, 360.0), 60.0)
    out = sim.step(30, lambda: orb.set_target((640.0, 360.0), 60.0, 1.0, (540.0, 360.0), (740.0, 360.0)))
    r0, i0 = out.orbs[0].radius, out.orbs[0].intensity
    orb.release()
    orb.set_target((100.0, 100.0), 10.0, 0.0)          # ignored after release
    radii, intens = [], []
    frames = 0
    while orb.alive:
        out = sim.step()
        frames += 1
        if out.orbs:
            radii.append(out.orbs[0].radius)
            intens.append(out.orbs[0].intensity)
        assert frames < 20
    assert frames <= math.ceil(0.12 / DT) + 1
    assert min(radii) < 0.8 * r0
    assert max(intens) > 1.5 * i0 / 1.05
    assert not out.orbs or sim.step().orbs == []
    assert orb.charge == pytest.approx(1.0)


def test_orb_dissipate_fizzles_then_dies():
    sim = Sim()
    orb = sim.fx.create_orb((640.0, 360.0), 60.0)
    sim.step(30, lambda: orb.set_target((640.0, 360.0), 60.0, 0.7, (540.0, 360.0), (740.0, 360.0)))
    orb.dissipate()
    frames = 0
    smoke_seen = False
    while orb.alive:
        out = sim.step()
        frames += 1
        smoke_seen |= sprite_count(out, SpriteShape.SMOKE) > 0
        assert frames < 40
    assert frames <= math.ceil(0.35 / DT) + 1
    assert smoke_seen
    out = sim.step()
    assert out.orbs == []
    assert sprite_count(out, SpriteShape.SPARK) > 0      # scattered sparks outlive the orb


def test_gathering_particles_stay_around_the_orb():
    sim = Sim()
    orb = sim.fx.create_orb((640.0, 360.0), 50.0)
    state = {"x": 640.0}

    def move():
        state["x"] = 640.0 + 200.0 * math.sin(sim.now * 3.0)
        orb.set_target((state["x"], 360.0), 50.0, 0.8)

    for out in sim.frames(120, move):
        glow = batch(out, SpriteShape.GLOW)
        if glow is None or out.orbs == []:
            continue
        c = np.array(out.orbs[0].center)
        d = np.hypot(*(glow.positions - c).T) / out.orbs[0].radius
        assert d.max() < 3.8          # spawned on the 1.8-3.6 R ring, only moving inwards
    assert glow is not None and len(glow) > 20


def test_orb_motion_trail_toggle():
    for trails in (True, False):
        sim = Sim(motion_trails=trails)
        orb = sim.fx.create_orb((200.0, 360.0), 40.0)
        state = {"x": 200.0}

        def move():
            state["x"] += 1500.0 * DT
            orb.set_target((state["x"], 360.0), 40.0, 0.5)

        ribbons = sum(len(out.ribbons) for out in sim.frames(30, move))
        if trails:
            assert ribbons > 10
            rib = sim.out.ribbons[0]
            assert rib.points.dtype == np.float32 and rib.points.shape[1] == 2
            assert rib.widths.shape == rib.intensities.shape == (rib.points.shape[0],)
            assert rib.points[0, 0] < rib.points[-1, 0]       # oldest first
            assert rib.intensities[-1] > rib.intensities[0]
        else:
            assert ribbons == 0
        assert sim.out.orbs[0].squash[0] == pytest.approx(1.25, abs=1e-6)  # stretched along the motion
        assert sim.out.orbs[0].rotation == pytest.approx(0.0, abs=0.05)


# ------------------------------------------------------------------------ beam


def test_beam_extends_past_the_scene_edge():
    sim = Sim()
    beam = sim.fx.create_beam((300.0, 360.0), (1.0, 0.0), 30.0, 1.0)
    assert isinstance(beam, BeamEffect) and beam.progress == 0.0
    progress = []
    for out in sim.frames(15, lambda: beam.set((300.0, 360.0), (1.0, 0.0), 1.0)):
        progress.append(beam.progress)
    assert progress == sorted(progress)
    assert progress[3] > 0.5                        # ease-out: fast start
    assert beam.progress == pytest.approx(1.0)      # 0.15 s extension
    prim = out.beams[0]
    assert prim.origin[0] + prim.direction[0] * prim.length > W
    assert prim.length == pytest.approx((W - 300.0) + 0.25 * W, rel=1e-3)
    assert prim.width == pytest.approx(30.0 * 1.25, rel=0.06)
    assert prim.end_width == pytest.approx(1.35 * prim.width)
    assert prim.head_radius == pytest.approx(1.6 * prim.width)
    assert prim.intensity == pytest.approx(2.6, abs=0.01)
    assert prim.flow_speed == pytest.approx(2.0, abs=0.01)
    assert prim.fade == 1.0 and prim.dissolve == 0.0
    assert_finite(out)


def test_beam_extras_rings_arcs_particles_lights_shake():
    sim = Sim()
    first = RenderList()
    beam = sim.fx.create_beam((300.0, 360.0), (1.0, 0.0), 30.0, 1.0)
    sim.fx.update(DT, sim.now)
    sim.fx.collect(first)
    assert max(f.intensity for f in first.flashes) > 1.5      # release burst
    assert first.rings and sprite_count(first, SpriteShape.SPARK) > 50
    assert math.hypot(*first.shake) > 0.0
    ring_counts, arc_counts, shakes = [], [], []
    for out in sim.frames(90, lambda: beam.set((300.0, 360.0), (1.0, 0.0), 1.0)):
        beam_rings = [r for r in out.rings if r.squash != (1.0, 1.0)]
        ring_counts.append(len(beam_rings))
        arc_counts.append(len(out.arcs))
        shakes.append(math.hypot(*out.shake))
        for r in beam_rings:
            assert r.squash[0] < r.squash[1]               # horizontal beam: ellipse narrow along x
            assert abs(r.center[1] - 360.0) < 1.0
    assert max(ring_counts) >= 1 and sum(1 for c in ring_counts if c) > 30
    assert max(arc_counts) >= 1
    assert min(shakes[-30:]) >= 0.0 and max(shakes[-30:]) > 0.0     # continuous micro shake
    assert max(shakes[-30:]) < 0.01 * H
    assert len(out.lights) >= 2
    assert sprite_count(out, SpriteShape.SPARK) > 20
    assert sprite_count(out, SpriteShape.GLOW) > 10


def test_beam_width_is_sane_for_huge_inputs():
    sim = Sim()
    beam = sim.fx.create_beam((300.0, 360.0), (1.0, 0.0), 5000.0, 1.0)
    out = sim.step(5, lambda: beam.set((300.0, 360.0), (1.0, 0.0), 1.0))
    assert out.beams[0].width <= 0.3 * H * 1.25 * 1.05


def test_beam_steering_limits_angular_speed():
    sim = Sim()
    beam = sim.fx.create_beam((640.0, 360.0), (1.0, 0.0), 30.0, 0.8)
    sim.step(10, lambda: beam.set((640.0, 360.0), (1.0, 0.0)))
    angles = [math.atan2(beam.direction[1], beam.direction[0])]
    for _ in range(60):
        sim.step(1, lambda: beam.set((640.0, 360.0), (0.0, -1.0)))   # aim straight up
        angles.append(math.atan2(beam.direction[1], beam.direction[0]))
    steps = np.abs(np.diff(angles))
    assert steps.max() <= 3.5 * DT + 1e-9
    assert steps.max() == pytest.approx(3.5 * DT, rel=1e-3)          # it really is speed-limited
    assert angles[-1] == pytest.approx(-math.pi / 2, abs=math.radians(1.0))
    # the origin is damped, not snapped
    beam.set((740.0, 360.0), (0.0, -1.0))
    sim.step()
    assert 640.0 < beam.origin[0] < 740.0


def test_beam_end_dissipates_and_dies():
    sim = Sim()
    beam = sim.fx.create_beam((300.0, 360.0), (1.0, 0.0), 30.0, 1.0)
    out = sim.step(30, lambda: beam.set((300.0, 360.0), (1.0, 0.0), 1.0))
    width0 = out.beams[0].width
    beam.end()
    beam.set((100.0, 100.0), (0.0, 1.0))                              # ignored while ending
    fades, dissolves, widths = [], [], []
    frames = 0
    wisps = False
    while beam.alive:
        out = sim.step()
        frames += 1
        if out.beams:
            fades.append(out.beams[0].fade)
            dissolves.append(out.beams[0].dissolve)
            widths.append(out.beams[0].width)
            assert out.beams[0].direction == pytest.approx((1.0, 0.0))
        wisps |= sprite_count(out, SpriteShape.SMOKE) > 0
        assert frames < 60
    assert math.ceil(0.5 / DT) - 1 <= frames <= math.ceil(0.5 / DT) + 1
    assert fades == sorted(fades, reverse=True) and fades[-1] < 0.1
    assert dissolves == sorted(dissolves) and dissolves[-1] > 0.9
    assert widths[-1] < 0.65 * width0
    assert wisps
    assert sim.step().beams == []


# ---------------------------------------------------------------- frontal beam


def test_frontal_beam_grows_to_target_and_ends():
    sim = Sim()
    first = RenderList()
    fb = sim.fx.create_frontal_beam((640.0, 360.0), 40.0, 1.0)
    assert isinstance(fb, FrontalBeamEffect)
    sim.fx.update(DT, sim.now)
    sim.fx.collect(first)
    uniform = [f for f in first.flashes if f.center is None]
    assert uniform and uniform[0].intensity > 1.0                    # big flash
    assert math.hypot(*first.shake) > 0.0
    target = 0.27 * H * (0.8 + 0.4 * 1.0)
    radii = [o.frontal_beams[0].radius for o in sim.frames(20, lambda: fb.set((640.0, 360.0), 1.0))]
    assert radii[0] < 0.6 * target
    assert radii[-1] == pytest.approx(target, rel=0.05)
    out = sim.out
    prim = out.frontal_beams[0]
    assert prim.intensity == pytest.approx(2.8, abs=0.01) and prim.streaks == 1.0
    assert sprite_count(out, SpriteShape.SPARK) > 30                 # perspective streaks
    sparks = batch(out, SpriteShape.SPARK)
    radial = np.einsum("ij,ij->i", sparks.positions - 640.0 * np.array([1.0, 0.0]) - np.array([0.0, 360.0]),
                       sparks.velocities)
    assert (radial > 0).mean() > 0.9                                  # flying outwards
    assert out.lights and out.lights[0].radius > target
    rings = []
    for o in sim.frames(30, lambda: fb.set((640.0, 360.0), 1.0)):
        rings.append(len(o.rings))
    assert max(rings) >= 2
    fb.end()
    frames = 0
    while fb.alive:
        o = sim.step()
        frames += 1
        assert frames < 60
        if o.frontal_beams:
            assert o.frontal_beams[0].dissolve > 0.0
    assert frames <= math.ceil(0.5 / DT) + 1


# ------------------------------------------------------------------- ki blasts


def test_lateral_ki_blast_explodes_at_the_edge():
    sim = Sim()
    sim.fx.ki_blast((640.0, 360.0), (1.0, 0.0), 0.2, 40.0, 1.0)
    xs = []
    exploded = None
    for i, out in enumerate(sim.frames(60)):
        if out.orbs:
            xs.append(out.orbs[0].center[0])
            assert out.orbs[0].squash == (1.25, 0.85)
        edge_flash = [f for f in out.flashes if f.center is not None and abs(f.center[0] - W) < 1.0]
        edge_ring = [r for r in out.rings if abs(r.center[0] - W) < 1.0]
        if edge_flash and edge_ring and exploded is None:
            exploded = i
            assert math.hypot(*out.shake) > 0.0
    assert xs == sorted(xs) and len(xs) >= 10
    speed = (xs[-1] - xs[0]) / ((len(xs) - 1) * DT)
    assert speed == pytest.approx(1.6 * W, rel=0.02)
    assert exploded is not None and exploded * DT < 0.45
    sim2 = Sim()                                       # the flying orb leaves a ribbon trail
    sim2.fx.ki_blast((300.0, 360.0), (1.0, 0.0), 0.0, 40.0, 1.0)
    out = sim2.step(6)
    assert out.ribbons and out.ribbons[0].points[-1][0] > out.ribbons[0].points[0][0]


def test_frontal_ki_blast_hits_the_lens():
    sim = Sim()
    sim.fx.ki_blast((500.0, 400.0), (1.0, 0.0), 0.9, 40.0, 1.0)
    radii = []
    big_flash_at = None
    for i, out in enumerate(sim.frames(60)):
        if out.orbs:
            radii.append(out.orbs[0].radius)
        for f in out.flashes:
            if f.center is None and f.intensity >= 0.8 and big_flash_at is None:   # brief white lens hit
                big_flash_at = i
                assert f.color == (1.0, 1.0, 1.0)
                assert math.hypot(*out.shake) > 0.005 * H
    assert radii == sorted(radii)
    assert radii[-1] > 0.6 * H
    assert big_flash_at is not None and big_flash_at * DT < 0.5


# ----------------------------------------------------------- bursts and impacts


def test_burst_flash_rings_sparks_arcs_shake():
    sim = Sim()
    sim.fx.burst((640.0, 360.0), 60.0, 1.0)
    out = sim.step()
    first_flash = max(f.intensity for f in out.flashes)
    assert 0.85 * 1.5 <= first_flash <= 1.5                            # 1 + 0.5 x power, 20 ms attack
    assert sprite_count(out, SpriteShape.SPARK) >= 500                 # 200 + 400 x power at "high"
    assert sprite_count(out, SpriteShape.EMBER) > 0
    # radial arcs start at 0.45 R (branches start somewhere along them)
    main_arcs = [a for a in out.arcs if abs(math.dist(tuple(a.points[0]), (640.0, 360.0)) - 0.45 * 60.0) < 1.0]
    assert 6 <= len(main_arcs) <= 12
    assert all(math.dist(tuple(a.points[-1]), (640.0, 360.0)) > 2.0 * 60.0 for a in main_arcs)
    shake0 = math.hypot(*out.shake)
    assert 0.0 < shake0 <= 0.045 * H + 1e-6                             # 0.025 + 0.02 x power
    assert out.grade.zoom > 1.0 and out.grade.zoom <= 1.03
    assert 0.0 < out.grade.chromatic <= 3.0
    assert any(lt.intensity > 2.0 for lt in out.lights)
    out = sim.step(int(0.2 / DT))
    assert len(out.rings) == 3                                          # staggered at 0, 0.08, 0.18 s
    radii = sorted(r.radius for r in out.rings)
    assert radii[0] < radii[-1]
    for r in out.rings:
        assert 0.0 < r.refraction <= 14.0
    assert sim.step(int(0.25 / DT)).arcs == []                         # arcs only last 0.2 s
    late = [math.hypot(*o.shake) for o in sim.frames(int(1.0 / DT))]
    assert max(late) < 0.25 * 0.045 * H                                # the (1 - t/T)^2 envelope decayed
    assert sim.out.shake == (0.0, 0.0)


def test_shockwave_expands_fades_and_respects_speed():
    sim = Sim()
    sim.fx.shockwave((640.0, 360.0), 40.0, 1.0)
    rings = [o.rings[0] for o in sim.frames(30) if o.rings]
    radii = [r.radius for r in rings]
    assert radii == sorted(radii)
    assert radii[-1] > 0.85 * 40.0 * 8.0 and radii[-1] <= 40.0 * 8.0
    assert rings[-1].intensity < rings[0].intensity
    assert rings[-1].thickness < rings[0].thickness
    assert rings[0].refraction == pytest.approx(8.0, rel=0.1)
    assert sim.step(10).rings == []
    slow = Sim()
    slow.fx.shockwave((640.0, 360.0), 40.0, 1.0, speed=200.0)          # (320 - 40) / 200 = 1.4 s
    alive = sum(1 for o in slow.frames(int(2.0 / DT)) if o.rings)
    assert 1.3 / DT <= alive <= 1.45 / DT


def test_sparks_direction_spread_and_count():
    sim = Sim()
    sim.fx.sparks((640.0, 360.0), 100, 800.0, direction=(0.0, -1.0), spread=0.6)
    out = sim.step()
    sp = batch(out, SpriteShape.SPARK)
    assert len(sp) == 100
    ang = np.arctan2(sp.velocities[:, 1], sp.velocities[:, 0])
    assert np.all(np.abs(ang - (-math.pi / 2)) <= 0.3 + 0.02)
    speed = np.hypot(sp.velocities[:, 0], sp.velocities[:, 1])
    assert speed.max() <= 800.0 and speed.min() > 0.3 * 800.0 * 0.9
    iso = Sim()
    iso.fx.sparks((640.0, 360.0), 200, 500.0)
    sp = batch(iso.step(), SpriteShape.SPARK)
    ang = np.arctan2(sp.velocities[:, 1], sp.velocities[:, 0])
    assert (ang > 0).mean() == pytest.approx(0.5, abs=0.15)              # isotropic


def test_flash_envelope_and_radial_flash():
    sim = Sim()
    sim.fx.flash(1.0, duration=0.25)
    sim.fx.flash(0.8, center=(100.0, 100.0), radius=200.0, duration=0.1)
    values = []
    for out in sim.frames(25):
        uniform = [f for f in out.flashes if f.center is None]
        values.append(uniform[0].intensity if uniform else 0.0)
        radial = [f for f in out.flashes if f.center is not None]
        if radial:
            assert radial[0].center == (100.0, 100.0) and radial[0].radius == 200.0
    assert max(values) <= 1.0 + 1e-9 and values[1] > 0.85
    assert values.index(max(values)) <= 2                                  # ~20 ms attack
    assert values[-1] == 0.0 and values[5] > values[10] > 0.0
    default = Sim()
    default.fx.flash(1.0, center=(10.0, 10.0))
    assert default.step().flashes[0].radius > 0.0                          # radial needs a radius


def test_fizzle_puff():
    sim = Sim()
    sim.fx.fizzle((640.0, 360.0), 40.0)
    out = sim.step()
    assert sprite_count(out, SpriteShape.SPARK) > 0
    assert sprite_count(out, SpriteShape.SMOKE) > 0
    assert sprite_count(out, SpriteShape.EMBER) > 0
    assert out.flashes and out.flashes[0].center == (640.0, 360.0)


# ------------------------------------------------------------- shake and grade


def test_shake_decays_to_zero_and_is_clamped():
    sim = Sim()
    sim.fx.shake(0.03, 0.5)
    mags = [math.hypot(*o.shake) for o in sim.frames(int(0.6 / DT))]
    assert max(mags[:6]) > 0.005 * H
    assert max(mags) <= 0.03 * H + 1e-6
    assert np.mean(mags[:10]) > np.mean(mags[15:25])
    assert sim.out.shake == (0.0, 0.0)
    many = Sim()
    for _ in range(12):
        many.fx.shake(0.1, 1.0)
    assert max(math.hypot(*o.shake) for o in many.frames(20)) <= 0.06 * H + 1e-6
    off = Sim(screen_shake=0.0)
    off.fx.shake(0.05, 0.5)
    off.fx.burst((640.0, 360.0), 50.0, 1.0)
    assert all(o.shake == (0.0, 0.0) for o in off.frames(10))
    double = Sim(screen_shake=2.0)
    double.fx.shake(0.02, 0.5)
    assert max(math.hypot(*o.shake) for o in double.frames(10)) > 0.02 * H


def test_grade_smoothing_limits_and_identity():
    sim = Sim(camera_darken=0.5)
    fx = sim.fx
    darken = [o.grade.darken for o in sim.frames(int(0.5 / DT), lambda: fx.set_grade(1.0, 1.0))]
    g = sim.out.grade
    assert darken == sorted(darken)
    assert darken[0] == pytest.approx(0.5 * (1 - math.exp(-6.0 * DT)), rel=1e-3)   # rise ~6/s
    assert darken[-1] == pytest.approx(0.5 * (1 - math.exp(-6.0 * 0.5)), rel=1e-2)
    assert g.desaturate <= 0.4 and g.desaturate > 0.3
    assert g.vignette == pytest.approx(0.15 * (1 - math.exp(-3.0)) + 0.35 * darken[-1] / 0.5, abs=0.02)
    peak = darken[-1]
    fall = [o.grade.darken for o in sim.frames(int(0.5 / DT))]
    assert fall == sorted(fall, reverse=True)
    assert fall[-1] == pytest.approx(peak * math.exp(-3.0 * 0.5), rel=0.03)        # fall ~3/s
    # max of several requests per frame wins
    sim2 = Sim()
    sim2.step(30, lambda: (sim2.fx.set_grade(0.2), sim2.fx.set_grade(0.9), sim2.fx.set_grade(0.4)))
    assert sim2.out.grade.darken == pytest.approx(0.45 * 0.9 * (1 - math.exp(-6.0 * 0.5)), rel=0.02)
    # long idle -> exactly the identity grade again
    sim.step(int(4.0 / DT))
    assert sim.out.grade == Grade()
    # no darkening wanted -> no darken/desaturation, only the vignette
    off = Sim(camera_darken=0.0)
    out = off.step(20, lambda: off.fx.set_grade(1.0, 1.0))
    assert out.grade.darken == 0.0 and out.grade.desaturate == 0.0 and out.grade.vignette > 0.0


def test_grade_is_identity_while_nothing_runs_and_vignette_while_active():
    sim = Sim()
    assert sim.step(10).grade == Grade()
    orb = sim.fx.create_orb((640.0, 360.0), 40.0)
    out = sim.step(30, lambda: orb.set_target((640.0, 360.0), 40.0, 0.2))
    assert out.grade != Grade() and out.grade.vignette > 0.1 and out.grade.darken == 0.0
    orb.dissipate()
    sim.step(int(5.0 / DT))
    assert sim.out.grade == Grade() and sim.out.is_empty()
    # a dormant aura handle (intensity 0) draws nothing and must not keep the grade active
    aura = sim.fx.create_aura()
    mask = np.full((72, 128), 255, np.uint8)
    out = sim.step(30, lambda: aura.set(0.0, mask, 5, None, None))
    assert aura.alive and out.aura is None and out.is_empty() and out.grade == Grade()


def test_stats_and_budget_properties():
    sim = Sim(quality="medium", particle_multiplier=0.5)
    fx = sim.fx
    assert fx.particle_budget == 800 and fx.quality.name == "medium"
    assert fx.settings.particle_multiplier == 0.5
    fx.burst((640.0, 360.0), 50.0, 1.0)
    sim.step()
    stats = fx.stats
    assert stats["particles"] == fx.particle_count > 0
    assert stats["budget"] == 800
    assert stats["spark"] + stats["glow"] + stats["ember"] + stats["smoke"] == stats["particles"]
    assert stats["rings"] >= 1 and stats["flashes"] >= 1 and stats["effects"] >= 1


# ---------------------------------------------------------------------- lights


def test_lights_are_limited_coloured_and_scaled():
    sim = Sim()
    for i in range(20):
        sim.fx.add_light((10.0 * i, 50.0), 100.0, 0.1 * (i + 1))
    out = sim.step()
    assert len(out.lights) == MAX_LIGHTS
    assert sorted(lt.intensity for lt in out.lights) == pytest.approx([0.1 * i for i in range(13, 21)])
    assert all(lt.color == PALETTES["blue"].inner for lt in out.lights)
    assert sim.step().lights == []                                           # one frame only
    double = Sim(scene_lighting=2.0)
    double.fx.add_light((10.0, 10.0), 50.0, 0.5)
    assert double.step().lights[0].intensity == pytest.approx(1.0)
    dark = Sim(scene_lighting=0.0)
    dark.fx.add_light((10.0, 10.0), 50.0, 0.5)
    orb = dark.fx.create_orb((640.0, 360.0), 40.0)
    assert dark.step(5, lambda: orb.set_target((640.0, 360.0), 40.0, 1.0)).lights == []
    # merged with lights already in the list, still at most 8
    sim3 = Sim()
    for i in range(10):
        sim3.fx.add_light((0.0, 0.0), 10.0, 1.0)
    sim3.fx.update(DT, 1.0)
    out = RenderList()
    from dragon_ball_powers.primitives import Light
    out.lights.extend(Light((0.0, 0.0), 5.0, (1.0, 1.0, 1.0), 5.0) for _ in range(3))
    sim3.fx.collect(out)
    assert len(out.lights) == MAX_LIGHTS and sum(lt.intensity == 5.0 for lt in out.lights) == 3


# ---------------------------------------------------------------------- budget


def test_particle_budget_is_never_exceeded():
    sim = Sim()
    fx = sim.fx
    orb = fx.create_orb((640.0, 360.0), 90.0)
    aura = fx.create_aura()
    mask = np.zeros((144, 256), np.uint8)
    mask[20:140, 80:180] = 255

    def busy():
        orb.set_target((640.0, 360.0), 90.0, 1.0, (500.0, 360.0), (780.0, 360.0))
        aura.set(1.2, mask, 1, None, None)
        fx.burst((640.0, 360.0), 60.0, 1.0)
        fx.sparks((300.0, 300.0), 400, 900.0)

    def check(out, cap):
        assert fx.particle_count <= cap
        assert sprite_count(out) == fx.particle_count
        assert fx.particle_budget == cap

    for out in sim.frames(40, busy):
        check(out, 3200)
    assert fx.particle_count >= 3000                         # saturated
    before = fx.particle_count
    fx.set_quality(get_quality("low"))
    assert fx.particle_count <= 700 < before                  # trimmed immediately
    for out in sim.frames(20, busy):
        check(out, 700)
    fx.apply_settings(EffectsSettings(particle_multiplier=3.0))
    for out in sim.frames(20, busy):
        check(out, 2100)
    fx.set_quality(get_quality("ultra"))
    for out in sim.frames(20, busy):
        check(out, 6400 * 3)
    # an impact still shows up when the budget is saturated (oldest particles are replaced)
    sat = Sim(quality="low")
    for _ in range(10):
        sat.fx.sparks((100.0, 100.0), 2000, 300.0)
        sat.step()
    assert sat.fx.particle_count == 700
    sat.fx.burst((900.0, 500.0), 50.0, 1.0)
    out = sat.step()
    sp = batch(out, SpriteShape.SPARK)
    near_burst = np.hypot(sp.positions[:, 0] - 900.0, sp.positions[:, 1] - 500.0) < 200.0
    assert near_burst.sum() >= 100
    assert sat.fx.particle_count <= 700


def test_emission_rates_scale_with_quality():
    counts = {}
    for q in ("low", "high"):
        sim = Sim(quality=q)
        orb = sim.fx.create_orb((640.0, 360.0), 60.0)
        sim.step(60, lambda: orb.set_target((640.0, 360.0), 60.0, 1.0))
        counts[q] = sim.fx.particle_count
    assert counts["low"] < 0.5 * counts["high"]


# --------------------------------------------------------------------- palette


def test_palette_switching_recolours_everything():
    sim = Sim()
    orb = sim.fx.create_orb((640.0, 360.0), 60.0)
    sim.fx.sparks((640.0, 360.0), 300, 300.0)
    out = sim.step(5, lambda: orb.set_target((640.0, 360.0), 60.0, 0.8, (540.0, 360.0), (740.0, 360.0)))
    sp = batch(out, SpriteShape.SPARK)
    assert sp.colors[:, 2].mean() >= sp.colors[:, 0].mean()                   # blue-white
    sim.fx.apply_settings(EffectsSettings(energy_color="red"))
    assert sim.fx.palette is PALETTES["red"]
    out = sim.step(1, lambda: orb.set_target((640.0, 360.0), 60.0, 0.8, (540.0, 360.0), (740.0, 360.0)))
    assert out.orbs[0].palette is PALETTES["red"]
    assert all(a.palette is PALETTES["red"] for a in out.arcs)
    assert all(lt.color == PALETTES["red"].inner for lt in out.lights)
    sp = batch(out, SpriteShape.SPARK)
    assert sp.colors[:, 0].mean() > sp.colors[:, 2].mean()                    # existing sparks recoloured
    sim.fx.apply_settings(EffectsSettings(energy_color="gold"))
    sim.fx.burst((640.0, 360.0), 50.0, 1.0)
    out = sim.step()
    assert all(r.palette is PALETTES["gold"] for r in out.rings)
    assert out.grade.tint == PALETTES["gold"].inner or out.grade.tint_strength == 0.0


# ------------------------------------------------------------------ scene size


def test_set_scene_size_rescales_running_effects():
    sim = Sim()
    fx = sim.fx
    orb = fx.create_orb((640.0, 360.0), 60.0)
    sim.step(20, lambda: orb.set_target((640.0, 360.0), 60.0, 0.5))
    beam = fx.create_beam((320.0, 360.0), (1.0, 0.0), 30.0, 1.0)
    sim.step(15)
    xs_before = batch(sim.out, SpriteShape.GLOW).positions[:, 0].copy()
    fx.set_scene_size((1920, 1080))
    assert fx.scene_scale == pytest.approx(1.5) and fx.scene_size == (1920, 1080)
    assert orb.center == pytest.approx((960.0, 540.0))
    assert orb.radius == pytest.approx(90.0, rel=0.02)
    out = RenderList()
    fx.collect(out)
    xs_after = batch(out, SpriteShape.GLOW).positions[:, 0]
    assert np.allclose(np.sort(xs_after), np.sort(xs_before) * 1.5, atol=0.05)
    out = sim.step(2)
    prim = out.beams[0]
    assert prim.origin[0] == pytest.approx(480.0, abs=1.0)
    assert prim.length == pytest.approx((1920.0 - prim.origin[0]) + 0.25 * 1920.0, rel=0.01)
    fx.set_scene_size((1920, 1080))                                           # no-op
    fx.set_scene_size((float("nan"), 5))                                      # invalid: ignored
    assert fx.scene_size == (1920, 1080)
    assert_finite(sim.step(5))


# ------------------------------------------------------------------------ aura


def test_aura_from_segmentation_mask():
    sim = Sim()
    aura = sim.fx.create_aura()
    assert isinstance(aura, AuraEffect)
    mask = np.zeros((144, 256), np.uint8)
    yy, xx = np.mgrid[0:144, 0:256]
    mask[((xx - 128) / 40.0) ** 2 + ((yy - 90) / 50.0) ** 2 <= 1.0] = 255
    body = make_body(energy=0.8)
    out = sim.step(1, lambda: aura.set(1.0, mask, 42, None, body))
    assert out.aura is not None and out.aura.intensity < 0.2                   # smoothed rise
    intens = [o.aura.intensity for o in sim.frames(36, lambda: aura.set(1.0, mask, 42, None, body))]
    assert intens == sorted(intens) and intens[-1] > 0.9
    prim = sim.out.aura
    assert prim.mask is mask and prim.mask_id == 42
    assert 0.06 <= prim.flame_height <= 0.12 and prim.flame_height > 0.11
    assert prim.palette is PALETTES["blue"]
    assert prim.flicker > 0.5 and prim.turbulence > 1.0
    em = batch(sim.out, SpriteShape.EMBER)
    assert em is not None and len(em) > 20
    # embers start on the silhouette outline (x in scene px: ellipse 88..168 mask px x 5)
    assert em.positions[:, 0].min() > 88 * 5 - 60 and em.positions[:, 0].max() < 168 * 5 + 60
    assert (em.positions[:, 1] < 140 * 5).all()
    # bool / float masks are accepted too
    out = sim.step(1, lambda: aura.set(1.0, mask > 0, 43, None, body))
    assert out.aura.mask.dtype == np.uint8 and out.aura.mask.max() == 255 and out.aura.mask_id == 43
    out = sim.step(1, lambda: aura.set(1.0, mask.astype(np.float32) / 255.0, 44, None, body))
    assert out.aura.mask.dtype == np.uint8 and out.aura.mask.max() == 255


def test_aura_accepts_full_resolution_masks():
    sim = Sim()
    aura = sim.fx.create_aura()
    mask = np.zeros((H, W), np.uint8)
    mask[200:720, 500:780] = 255
    out = sim.step(40, lambda: aura.set(1.0, mask, 77, None, None))
    assert out.aura.mask is mask and out.aura.mask_id == 77          # the renderer gets the original
    em = batch(out, SpriteShape.EMBER)
    assert em is not None and len(em) > 10
    assert em.positions[:, 0].min() > 500 - 60 and em.positions[:, 0].max() < 780 + 60


def test_aura_pose_fallback_mask():
    sim = Sim()
    aura = sim.fx.create_aura()
    pose = make_pose()
    out = sim.step(30, lambda: aura.set(1.0, None, 0, pose, None))
    prim = out.aura
    assert prim is not None
    res = get_quality("high").aura_resolution
    assert prim.mask.shape == (round(res * H / W), res) and prim.mask.dtype == np.uint8
    sx = res / W
    assert prim.mask[int(420 * sx), int(640 * sx)] > 200                      # torso (extended below the shoulders)
    assert prim.mask[int(200 * sx), int(640 * sx)] > 200                      # head
    assert prim.mask[int(480 * sx), int(510 * sx)] > 100                      # arm
    assert prim.mask[5, 5] == 0 and prim.mask[5, res - 5] == 0                # background
    first_id = prim.mask_id
    assert first_id > SYNTH_MASK_ID_BASE
    out = sim.step(1, lambda: aura.set(1.0, None, 0, pose, None))
    assert out.aura.mask_id == first_id                                        # unchanged pose: no regeneration
    moved = make_pose()
    moved.landmarks[:, 0] += 100.0
    out = sim.step(3, lambda: aura.set(1.0, None, 0, moved, None))
    assert out.aura.mask_id != first_id
    assert out.aura.mask[int(420 * sx), int(740 * sx)] > 200
    sim.fx.set_quality(get_quality("low"))                                     # aura_resolution 128
    out = sim.step(2, lambda: aura.set(1.0, None, 0, moved, None))
    assert out.aura.mask.shape == (72, 128)
    full = Sim()
    a2 = full.fx.create_aura()
    out = full.step(5, lambda: a2.set(1.0, None, 0, make_pose(hips_visible=True), None))
    assert out.aura.mask is not None and out.aura.mask[int(640 * sx), int(640 * sx)] > 200


def test_aura_hand_blob_fallback_and_stale_source():
    sim = Sim()
    aura = sim.fx.create_aura()
    body = make_body()
    out = sim.step(20, lambda: aura.set(0.8, None, 0, None, body))
    m = out.aura.mask
    sx = m.shape[1] / W
    assert m[int(520 * sx), int(560 * sx)] > 200 and m[int(520 * sx), int(720 * sx)] > 200
    assert m[int(200 * sx), int(640 * sx)] == 0
    # nothing to hug any more (user left the frame): the aura disappears, the handle lives on
    out = sim.step(int(0.7 / DT), lambda: aura.set(0.8, None, 0, None, None))
    assert out.aura is None and aura.alive


def test_aura_lightning_only_at_high_intensity():
    mask = np.zeros((144, 256), np.uint8)
    mask[30:144, 100:156] = 255

    def arcs_seen(intensity: float) -> int:
        sim = Sim()
        aura = sim.fx.create_aura()
        return sum(len(o.arcs) for o in sim.frames(90, lambda: aura.set(intensity, mask, 3, None, None)))

    assert arcs_seen(1.0) > 0
    assert arcs_seen(0.7) == 0


def test_aura_end_fades_out_and_strongest_aura_wins():
    sim = Sim()
    weak = sim.fx.create_aura()
    strong = sim.fx.create_aura()
    mask = np.full((72, 128), 255, np.uint8)
    out = sim.step(30, lambda: (weak.set(0.3, mask, 1, None, None), strong.set(1.0, mask, 2, None, None)))
    assert out.aura.mask_id == 2
    strong.end()
    values = []
    frames = 0
    while strong.alive:
        out = sim.step(1, lambda: weak.set(0.3, mask, 1, None, None))
        frames += 1
        if out.aura is not None and out.aura.mask_id == 2:
            values.append(out.aura.intensity)
        assert frames < 60
    assert frames <= math.ceil(0.4 / DT) + 1
    assert values == sorted(values, reverse=True)
    assert sim.step(1, lambda: weak.set(0.3, mask, 1, None, None)).aura.mask_id == 1


# ------------------------------------------------------------------ robustness


def test_invalid_inputs_are_ignored_and_output_stays_finite():
    sim = Sim()
    fx = sim.fx
    nan, inf = float("nan"), float("inf")
    orb = fx.create_orb((nan, 3.0), nan)
    assert orb.alive and all(math.isfinite(v) for v in orb.center) and orb.radius > 0
    beam = fx.create_beam((inf, 0.0), (0.0, 0.0), -5.0, nan)
    fb = fx.create_frontal_beam(None, inf, -1.0)
    aura = fx.create_aura()
    fx.ki_blast((nan, nan), (1.0, 0.0), 0.0, 10.0, 1.0)
    fx.ki_blast((100.0, 100.0), (nan, 0.0), nan, nan, inf)
    fx.burst((inf, 0.0), 10.0, 1.0)
    fx.burst((300.0, 300.0), nan, nan)
    fx.shockwave((1e30, -1e30), 1e30, 1e30, speed=nan)
    fx.flash(nan)
    fx.flash(1.0, center=(nan, 1.0))
    fx.shake(nan, nan)
    fx.shake(10.0, 1e9)
    fx.sparks((0.0, 0.0), nan, nan)
    fx.sparks((10.0, 10.0), 50, 1e12, direction=(0.0, 0.0), spread=nan, power=nan)
    fx.fizzle("garbage", None)
    fx.set_grade(nan, inf)
    fx.add_light((nan, 0.0), 10.0, 1.0)
    fx.add_light((0.0, 0.0), inf, nan)

    def poke():
        orb.set_target((nan, nan), inf, nan, (nan, 0.0), "x")
        beam.set((nan, 1.0), (inf, inf), nan)
        fb.set((nan, 0.0), inf)
        aura.set(nan, np.full((4, 4), np.nan, np.float32), nan, None, None)

    for out in sim.frames(40, poke):
        assert_finite(out)
    fx.update(nan, nan)
    fx.update(-1.0, 0.0)
    out = RenderList()
    fx.collect(out)
    assert_finite(out)


def test_runaway_handle_creation_is_capped():
    sim = Sim()
    orbs = [sim.fx.create_orb((100.0 + 50.0 * i, 300.0), 20.0) for i in range(12)]
    beams = [sim.fx.create_beam((300.0, 100.0 + 40.0 * i), (1.0, 0.0), 10.0, 0.5) for i in range(10)]
    assert sum(not o._finishing() for o in orbs) == 8 and sum(not b._finishing() for b in beams) == 8
    out = sim.step(40)                       # the surplus ones play their end animation, then die
    assert len(out.orbs) == 8 and len(out.beams) == 8
    assert all(o.alive for o in orbs[-8:]) and not any(o.alive for o in orbs[:4])
    assert all(b.alive for b in beams[-8:]) and not any(b.alive for b in beams[:2])
    assert_finite(out)


def test_update_clamps_dt():
    sim = Sim()
    fx = sim.fx
    fx.update(5.0, 1.0)
    assert fx.time == pytest.approx(0.1)
    fx.update(-3.0, 2.0)
    assert fx.time == pytest.approx(0.1)
    orb = fx.create_orb((100.0, 100.0), 30.0)
    orb.set_target((1100.0, 600.0), 30.0, 0.5)
    fx.update(10.0, 3.0)                       # a hitch must not teleport-and-explode anything
    out = RenderList()
    fx.collect(out)
    assert_finite(out)
    assert 100.0 < orb.center[0] < 1100.0


def test_determinism_with_seed():
    def run(seed: int):
        sim = Sim(seed=seed)
        fx = sim.fx
        orb = fx.create_orb((600.0, 350.0), 50.0)
        aura = fx.create_aura()
        mask = np.zeros((72, 128), np.uint8)
        mask[20:70, 40:90] = 255
        state = {"i": 0}

        def script():
            i = state["i"]
            state["i"] += 1
            if i < 60:
                orb.set_target((600.0 + i, 350.0), 50.0, i / 60.0, (520.0 + i, 350.0), (680.0 + i, 350.0))
            elif i == 60:
                orb.release()
                fx.ki_blast(orb.center, (1.0, 0.2), 0.1, 50.0, 1.0)
                fx.burst((400.0, 400.0), 40.0, 0.7)
            aura.set(1.0, mask, 9, None, None)

        frames = []
        for out in sim.frames(90, script):
            frames.append((
                [(b.shape, b.positions.copy(), b.colors.copy(), b.sizes.copy()) for b in out.sprites],
                [(o.center, o.radius, o.intensity) for o in out.orbs],
                [a.points.copy() for a in out.arcs],
                [(r.center, r.radius) for r in out.rings],
                out.shake, out.grade,
            ))
        return frames

    a, b, c = run(3), run(3), run(4)

    def same(x, y) -> bool:
        if isinstance(x, np.ndarray):
            return isinstance(y, np.ndarray) and x.shape == y.shape and np.array_equal(x, y)
        if isinstance(x, (list, tuple)):
            return len(x) == len(y) and all(same(i, j) for i, j in zip(x, y))
        return x == y

    assert same(a, b)
    assert not same(a, c)


def test_clear_kills_everything():
    sim = Sim()
    fx = sim.fx
    orb = fx.create_orb((640.0, 360.0), 40.0)
    beam = fx.create_beam((300.0, 360.0), (1.0, 0.0), 30.0, 1.0)
    fb = fx.create_frontal_beam((640.0, 360.0), 30.0, 1.0)
    aura = fx.create_aura()
    fx.burst((640.0, 360.0), 40.0, 1.0)
    fx.ki_blast((640.0, 360.0), (1.0, 0.0), 0.0, 40.0, 1.0)
    fx.set_grade(1.0)
    sim.step(10, lambda: (orb.set_target((640.0, 360.0), 40.0, 1.0), aura.set(1.0, np.full((8, 8), 255, np.uint8),
                                                                               1, None, None)))
    fx.clear()
    assert not any(h.alive for h in (orb, beam, fb, aura))
    assert fx.particle_count == 0
    out = RenderList()
    fx.collect(out)
    assert out.is_empty() and out.grade == Grade() and out.shake == (0.0, 0.0) and out.lights == []
    orb.set_target((1.0, 1.0), 5.0, 1.0)            # dead handles ignore calls
    beam.set((1.0, 1.0), (0.0, 1.0))
    beam.end()
    orb.release()
    orb.dissipate()
    assert sim.step(5).is_empty()


def test_collect_before_update_and_repeated_collect():
    sim = Sim()
    orb = sim.fx.create_orb((640.0, 360.0), 40.0)
    beam = sim.fx.create_beam((640.0, 360.0), (-1.0, 0.0), 20.0, 0.5)
    out = RenderList()
    sim.fx.collect(out)
    assert out.orbs and out.beams and out.beams[0].length == 0.0
    assert_finite(out)
    sim.step(5, lambda: orb.set_target((640.0, 360.0), 40.0, 0.4))
    out2 = RenderList()
    sim.fx.collect(out2)
    assert len(out2.orbs) == len(sim.out.orbs) and sprite_count(out2) == sprite_count(sim.out)
    assert beam.alive


# ------------------------------------------------------------- review regressions


def test_tiny_and_subnormal_dt_count_as_zero():
    """Velocities are displacement / dt: a near-zero step overflowed them (NaN sprite velocities)."""
    sim = Sim()
    fx = sim.fx
    orb = fx.create_orb((640.0, 360.0), 50.0)
    state = {"x": 640.0}

    def move():
        state["x"] += 15.0
        orb.set_target((state["x"], 360.0), 50.0, 0.9, (state["x"] - 100.0, 360.0), (state["x"] + 100.0, 360.0))

    sim.step(10, move)
    t0 = fx.time
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # numpy overflow / invalid-value warnings fail the test
        for tiny in (5e-324, 1e-300, 1e-9, 5e-7):
            move()
            fx.update(tiny, sim.now)
            out = RenderList()
            fx.collect(out)
            assert_finite(out)
    assert fx.time == t0                        # nothing advanced
    assert all(math.isfinite(v) for v in orb.velocity)
    assert_finite(sim.step(5, move))            # and normal steps carry on


def test_hand_blob_aura_bounds_absurd_hand_sizes():
    """A garbage hand size used to become a giant cv2 circle + blur (~4 s per mask, OOM at 1e9 px)."""
    sim = Sim()
    aura = sim.fx.create_aura()
    body = make_body()
    body.left.hand_size = 1e5
    body.right.hand_size = 1e5
    out = sim.step(3, lambda: aura.set(1.0, None, 0, None, body))
    m = out.aura.mask
    sx = m.shape[1] / W
    assert m[int(520 * sx), int(560 * sx)] == 255                 # the blobs are there ...
    assert m[0, 0] == 0 and m[0, -1] == 0                          # ... but bounded (not the whole frame)
    assert m.mean() < 200


def test_invalid_quality_and_settings_objects_are_rejected():
    sim = Sim()
    fx = sim.fx
    orb = fx.create_orb((640.0, 360.0), 50.0)

    def steer():
        orb.set_target((640.0, 360.0), 50.0, 0.8, (540.0, 360.0), (740.0, 360.0))

    sim.step(2, steer)
    fx.set_quality(None)                         # used to leave the system without a preset ...
    fx.set_quality("ultra")
    fx.apply_settings(None)                      # ... or without settings (every later frame broke)
    fx.apply_settings({"energy_color": "red"})
    assert fx.quality.name == "high" and fx.particle_budget == 3200
    assert isinstance(fx.settings, EffectsSettings) and fx.palette is PALETTES["blue"]
    out = sim.step(30, steer)
    assert orb.alive and out.orbs and out.arcs
    assert_finite(out)
    # hand-made presets are clamped instead of trusted (arc_count 10**6 = a million bolts per frame)
    wild = dataclasses.replace(get_quality("high"), arc_count=10**6, particle_budget=-5, aura_resolution=10**7)
    fx.set_quality(wild)
    assert fx.quality.arc_count == 16 and fx.quality.aura_resolution == 2048 and fx.particle_budget == 1
    out = sim.step(5, steer)
    assert orb.alive and len(out.arcs) <= 96 and fx.particle_count <= 1
    assert get_quality("medium") is not None
    fx.set_quality(get_quality("medium"))
    assert fx.quality is get_quality("medium")   # valid presets are used as they are
    fallback = EffectSystem(None, None, (W, H))
    assert fallback.quality.name == "high" and fallback.particle_budget == 3200
    assert isinstance(fallback.settings, EffectsSettings)


def test_make_bolt_sanitizes_its_arguments():
    rng = np.random.default_rng(2)
    assert make_bolt((0.0, 0.0), (100.0, 0.0), rng, segments=10**5).shape == (1025, 2)   # capped
    flat = make_bolt((0.0, 0.0), (100.0, 0.0), rng, segments=8, jaggedness=float("nan"))
    assert np.isfinite(flat).all() and np.allclose(flat[:, 1], 0.0)
    branches: list[np.ndarray] = []
    make_bolt((0.0, 0.0), (100.0, 0.0), rng, 16, 0.25, branch_prob=float("nan"), branches=branches)
    assert branches == []
    assert make_bolt((0.0, 0.0), (100.0, 0.0), rng, segments=float("nan")).shape == (17, 2)


def test_power_above_one_follows_the_spec_formulas():
    """Abilities pass power up to 1.5 (0.5 + charge); the documented formulas use it unclamped."""
    sim = Sim()
    fb = sim.fx.create_frontal_beam((640.0, 360.0), 40.0, 1.5)
    radii = [o.frontal_beams[0].radius for o in sim.frames(40, lambda: fb.set((640.0, 360.0), 1.5))]
    assert fb.target_radius == pytest.approx(0.27 * H * (0.8 + 0.4 * 1.5))
    assert radii[-1] == pytest.approx(0.27 * H * 1.4, rel=0.05)
    # shockwave: radius x (4 + 4 power), refraction ~8 power px x scene_scale
    sw = Sim()
    sw.fx.shockwave((640.0, 360.0), 40.0, 1.5)
    rings = [o.rings[0] for o in sw.frames(40) if o.rings]
    assert max(r.radius for r in rings) == pytest.approx(40.0 * (4.0 + 4.0 * 1.5), rel=0.02)
    assert rings[0].refraction == pytest.approx(8.0 * 1.5, rel=0.1)

    # burst shake amplitude 0.025 + 0.02 power: same seed -> same random phases, so the
    # unclamped shake offsets scale exactly with the amplitude
    def burst_shake(power: float) -> np.ndarray:
        s = Sim(seed=11)
        s.fx.burst((640.0, 360.0), 60.0, power)
        return np.array([math.hypot(*o.shake) for o in s.frames(30)])

    strong, normal = burst_shake(1.5), burst_shake(1.0)
    free = (strong > 0.0) & (strong < 0.06 * H * 0.999)          # below the stacking clamp
    assert free.sum() >= 10
    assert strong[free] / normal[free] == pytest.approx((0.025 + 0.02 * 1.5) / (0.025 + 0.02 * 1.0), rel=1e-6)


def test_aspect_change_maps_beams_blasts_and_synthesised_masks():
    sim = Sim()
    fx = sim.fx
    beam = fx.create_beam((300.0, 500.0), (0.8, -0.6), 30.0, 1.0)
    fx.ki_blast((200.0, 650.0), (0.8, -0.6), 0.0, 30.0, 1.0)
    out = sim.step(20, lambda: beam.set((300.0, 500.0), (0.8, -0.6), 1.0))
    rings_before = [r.center for r in out.rings if r.squash != (1.0, 1.0)]
    assert rings_before and out.orbs                                 # beam rings + the flying blast
    sx, sy = 640 / W, 480 / H
    fx.set_scene_size((640, 480))                                   # 16:9 -> 4:3
    after = RenderList()
    fx.collect(after)
    d = np.array([0.8 * sx, -0.6 * sy])
    d /= np.linalg.norm(d)
    assert after.beams[0].direction == pytest.approx(tuple(d), abs=1e-6)
    rings_after = [r.center for r in after.rings if r.squash != (1.0, 1.0)]
    assert len(rings_after) == len(rings_before)
    for (x, y), moved in zip(rings_before, rings_after):
        assert moved == pytest.approx((x * sx, y * sy), abs=1e-3)   # points on the beam stay put
    # the blast keeps flying along the mapped flight line
    origin = np.array([200.0 * sx, 650.0 * sy])
    centers = [after.orbs[0].center] + [o.orbs[0].center for o in sim.frames(3)]
    for c in centers:
        v = np.array(c) - origin
        assert abs(v[0] * d[1] - v[1] * d[0]) < 0.5
    assert centers[-1][0] > centers[0][0]
    # a mask synthesised from the pose is rebuilt with the new aspect, even for the same pose object
    sim2 = Sim()
    aura = sim2.fx.create_aura()
    pose = make_pose()
    assert sim2.step(5, lambda: aura.set(1.0, None, 0, pose, None)).aura.mask.shape == (144, 256)
    sim2.fx.set_scene_size((640, 480))
    assert sim2.step(2, lambda: aura.set(1.0, None, 0, pose, None)).aura.mask.shape == (192, 256)


def test_dead_handles_stay_harmless_and_release_their_resources():
    sim = Sim()
    fx = sim.fx
    orb_a = fx.create_orb((400.0, 360.0), 40.0)
    orb_b = fx.create_orb((800.0, 360.0), 40.0)
    beam = fx.create_beam((300.0, 360.0), (1.0, 0.0), 30.0, 1.0)
    fb = fx.create_frontal_beam((640.0, 360.0), 30.0, 1.0)
    aura = fx.create_aura()
    mask = np.zeros((72, 128), np.uint8)
    mask[20:72, 40:90] = 255

    def steer():
        orb_a.set_target((400.0, 360.0), 40.0, 0.6, (330.0, 360.0), (470.0, 360.0))
        orb_b.set_target((800.0, 360.0), 40.0, 0.6)
        beam.set((300.0, 360.0), (1.0, 0.0), 1.0)
        fb.set((640.0, 360.0), 1.0)
        aura.set(1.0, mask, 1, None, None)

    sim.step(20, steer)
    orb_a.release()
    orb_b.dissipate()
    beam.end()
    fb.end()
    aura.end()
    sim.step(int(0.6 / DT))
    assert not any(h.alive for h in (orb_a, orb_b, beam, fb, aura))

    def poke():          # an ability that still holds the handles keeps steering them
        steer()
        orb_a.release()
        orb_b.dissipate()
        beam.end()
        fb.end()
        aura.set(1.0, mask, 2, make_pose(), make_body())
        aura.end()

    for out in sim.frames(int(4.0 / DT), poke):
        assert not (out.orbs or out.beams or out.frontal_beams or out.aura or out.arcs)
    assert sim.out.is_empty() and sim.out.grade == Grade() and sim.out.shake == (0.0, 0.0)
    assert fx.stats["effects"] == 0 and fx.particle_count == 0


def test_orb_cycles_do_not_leak_attractor_slots():
    """Every orb borrows one of 32 attractor slots; a leak would stop new orbs gathering energy."""
    sim = Sim()
    fx = sim.fx
    for i in range(48):
        orb = fx.create_orb((640.0, 360.0), 40.0)
        sim.step(2, lambda: orb.set_target((640.0, 360.0), 40.0, 0.5))
        if i % 3 == 0:
            orb.release()
        elif i % 3 == 1:
            orb.dissipate()
        else:
            fx.clear()
        sim.step(int(0.4 / DT))
        assert not orb.alive
    sim.step(int(1.5 / DT))
    assert fx.particle_count == 0
    orb = fx.create_orb((640.0, 360.0), 40.0)
    out = sim.step(30, lambda: orb.set_target((640.0, 360.0), 40.0, 0.8))
    assert sprite_count(out, SpriteShape.GLOW) > 0 and sprite_count(out, SpriteShape.EMBER) > 0


def test_rapid_start_stop_before_the_first_update():
    sim = Sim()
    fx = sim.fx
    handles = []
    for _ in range(5):
        a = fx.create_orb((640.0, 360.0), 40.0)
        a.release()
        b = fx.create_orb((640.0, 360.0), 40.0)
        b.dissipate()
        b.set_target((100.0, 100.0), 10.0, 1.0)          # ignored: already fizzling
        c = fx.create_beam((300.0, 360.0), (1.0, 0.0), 30.0, 1.0)
        c.end()
        c.set((300.0, 300.0), (0.0, 1.0), 1.0)           # ignored: already ending
        d = fx.create_frontal_beam((640.0, 360.0), 30.0, 1.0)
        d.end()
        e = fx.create_aura()
        e.end()
        handles += [a, b, c, d, e]
    first = sim.step()
    assert first.orbs and first.beams and first.frontal_beams        # end animations play
    assert all(b.direction == pytest.approx((1.0, 0.0)) for b in first.beams)
    assert_finite(first)
    sim.step(math.ceil(0.5 / DT) + 1)
    assert not any(h.alive for h in handles)
    sim.step(int(4.0 / DT))
    assert sim.out.is_empty() and sim.out.grade == Grade() and fx.stats["effects"] == 0


_SATURATING_EMITTERS = {
    "burst": lambda fx: fx.burst((640.0, 360.0), 60.0, 1.5),
    "sparks": lambda fx: fx.sparks((640.0, 360.0), 5000, 900.0),
    "fizzle": lambda fx: fx.fizzle((640.0, 360.0), 50.0),
    "ki_blast_lateral": lambda fx: fx.ki_blast((640.0, 360.0), (1.0, 0.0), 0.0, 40.0, 1.5),
    "ki_blast_frontal": lambda fx: fx.ki_blast((640.0, 360.0), (1.0, 0.0), 1.0, 40.0, 1.5),
    "beam": lambda fx: fx.create_beam((300.0, 360.0), (1.0, 0.1), 40.0, 1.5),
    "frontal_beam": lambda fx: fx.create_frontal_beam((640.0, 360.0), 40.0, 1.5),
    "orb_dissipate": lambda fx: fx.create_orb((640.0, 360.0), 60.0).dissipate(),
}


@pytest.mark.parametrize("name", sorted(_SATURATING_EMITTERS))
def test_every_emitter_respects_a_saturated_budget(name):
    emitter = _SATURATING_EMITTERS[name]
    for quality, mult, cap in (("low", 1.0, 700), ("low", 3.0, 2100)):
        sim = Sim(quality=quality, particle_multiplier=mult)
        for _ in range(3):
            sim.fx.sparks((100.0, 100.0), 5000, 300.0)
            sim.step()
        assert sim.fx.particle_count == cap                         # saturated before the test
        for _ in range(25):
            emitter(sim.fx)
            out = sim.step()
            assert sim.fx.particle_count <= cap
            assert sprite_count(out) == sim.fx.particle_count
        assert_finite(out)


# ------------------------------------------------------------------- performance


def test_perf_smoke_busy_scene(capsys):
    sim = Sim(quality="high")
    fx = sim.fx
    orb = fx.create_orb((420.0, 380.0), 80.0)
    beam = fx.create_beam((420.0, 380.0), (1.0, -0.1), 40.0, 1.0)
    aura = fx.create_aura()
    mask = np.zeros((144, 256), np.uint8)
    mask[25:144, 60:200] = 255
    pose_body = make_body(energy=0.9)
    state = {"i": 0}
    times = []
    counts = []
    out = RenderList()
    for _ in range(300):
        i = state["i"]
        state["i"] += 1
        t = i * DT
        orb.set_target((420.0 + 40.0 * math.sin(t * 3.0), 380.0), 80.0, 1.0, (330.0, 380.0), (510.0, 380.0))
        beam.set((420.0, 380.0), (1.0, -0.1 + 0.2 * math.sin(t)), 1.0)
        aura.set(1.0, mask, 7, None, pose_body)
        fx.set_grade(0.8, 0.3)
        if i % 20 == 0:
            fx.burst((800.0 + 100.0 * math.sin(t), 300.0), 60.0, 1.0)
        if i % 30 == 0:
            fx.ki_blast((420.0, 380.0), (1.0, 0.3), 0.0, 40.0, 1.0)
        if i % 10 == 0:
            fx.sparks((640.0, 200.0), 120, 700.0)
        start = time.perf_counter()
        fx.update(DT, 100.0 + t)
        out.clear()
        fx.collect(out)
        times.append(time.perf_counter() - start)
        counts.append(fx.particle_count)
        assert fx.particle_count <= fx.particle_budget
    assert_finite(out)
    ms = np.array(times[30:]) * 1000.0
    with capsys.disabled():
        print(f"\n[effects perf] high: update+collect mean {ms.mean():.2f} ms, median {np.median(ms):.2f} ms, "
              f"p95 {np.percentile(ms, 95):.2f} ms, particles mean {np.mean(counts[30:]):.0f} max {max(counts)}")
    assert np.mean(counts[30:]) > 2200 and max(counts) >= 0.95 * fx.particle_budget   # really busy
    assert np.median(ms) < 30.0                     # generous bound (shared CPU); see the report
