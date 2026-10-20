"""Tests for the abilities and the ability manager (no camera, no GPU, no real effects).

``FakeEffectSystem`` implements the documented EffectSystem API (ARCHITECTURE.md 5.4)
and records every call; its handles complain (AssertionError) when they are used after
they were finished. With ``linger`` > 0 finished handles stay ``alive`` for that long
(``FakeEffectSystem.update`` time), like the end animations of the real effects.
The manager contains exceptions and logs them, so an autouse fixture fails every test
that logged an error, unless it requests ``allow_errors``.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from dragon_ball_powers.abilities import (
    ABILITY_TYPES,
    CALLOUT_BEAM,
    CALLOUT_BLAST,
    CALLOUT_OVER_9000,
    HINT_BURST,
    HINT_BURST_KEY,
    HINT_CHARGE,
    HINT_FIRE,
    HINT_FIRE_KEY,
    HINT_KEEP_CHARGING,
    HINT_NO_TRACKING,
    HINT_POSE_RESET,
    KAMEHAMEHA_CHANT,
    MSG_CHARGE_LONGER,
    PRIORITY_RELEASE,
    RELEASE_SHOUT_HOLD,
    AbilityContext,
    AbilityManager,
    CalloutBoard,
    EnergyBurstAbility,
    KamehamehaAbility,
    KiChargeAbility,
    MessageBoard,
    PowerAuraAbility,
    PowerLevel,
)
from dragon_ball_powers.config import Settings
from dragon_ball_powers.datatypes import (
    ArmState,
    BodyState,
    GestureEvent,
    GestureEventType,
    GestureSignals,
    ManualInput,
    PoseObservation,
    TrackingFrame,
)
from dragon_ball_powers.interfaces import LOOP_SOUNDS, SOUND_NAMES
from dragon_ball_powers.primitives import get_palette
from dragon_ball_powers.state_machine import AttackPhase

DT = 1.0 / 30.0
SCENE = (1280, 720)

# ------------------------------------------------------------------- fake effects


class _FakeHandle:
    """Shared end-of-life behaviour of the fake effect handles.

    Finishing (release / dissipate / end) is recorded at the effect system's time. The real
    handles stay ``alive`` while their end animation plays; with ``fx.linger`` > 0 the fakes
    do the same until ``FakeEffectSystem.update`` has advanced that far, so code relying on
    handles dying instantly fails here too.
    """

    def __init__(self, fx):
        self.fx = fx
        self.alive = True
        self.finished_time: float | None = None

    def _finish(self, what):
        if self.finished_time is not None:
            raise AssertionError(f"{what} finished twice")
        self.finished_time = self.fx.time
        self.alive = self.fx.linger > 0.0


class FakeOrb(_FakeHandle):
    def __init__(self, fx, center, radius):
        super().__init__(fx)
        self.created_center = center
        self.created_radius = radius
        self.center = center
        self.radius = radius
        self.charge = 0.0
        self.released = False
        self.dissipated = False
        self.targets: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.released or self.dissipated

    def set_target(self, center, radius, charge, left_hand=None, right_hand=None):
        if self.finished:
            raise AssertionError("set_target() on a finished orb")
        self.targets.append(dict(center=center, radius=radius, charge=charge,
                                 left_hand=left_hand, right_hand=right_hand))
        self.center, self.radius, self.charge = center, radius, charge

    def release(self):
        self._finish("orb")
        self.released = True

    def dissipate(self):
        self._finish("orb")
        self.dissipated = True


class FakeBeam(_FakeHandle):
    def __init__(self, fx, origin, direction, width, power):
        super().__init__(fx)
        self.origin, self.direction, self.width, self.power = origin, direction, width, power
        self.ended = False
        self.progress = 0.0
        self.sets: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.ended

    def set(self, origin, direction, power=None):
        if self.ended:
            raise AssertionError("set() on an ended beam")
        self.sets.append(dict(origin=origin, direction=direction, power=power))

    def end(self):
        self._finish("beam")
        self.ended = True


class FakeFrontalBeam(_FakeHandle):
    def __init__(self, fx, center, radius, power):
        super().__init__(fx)
        self.center, self.radius, self.power = center, radius, power
        self.ended = False
        self.sets: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.ended

    def set(self, center, power=None):
        if self.ended:
            raise AssertionError("set() on an ended frontal beam")
        self.sets.append(dict(center=center, power=power))

    def end(self):
        self._finish("frontal beam")
        self.ended = True


class FakeAura(_FakeHandle):
    def __init__(self, fx):
        super().__init__(fx)
        self.ended = False
        self.sets: list[dict] = []

    @property
    def finished(self) -> bool:
        return self.ended

    def set(self, intensity, mask, mask_id, pose, body):
        if self.ended:
            raise AssertionError("set() on an ended aura")
        self.sets.append(dict(intensity=intensity, mask=mask, mask_id=mask_id, pose=pose,
                              body=body))

    def end(self):
        self._finish("aura")
        self.ended = True


class FakeEffectSystem:
    """Records calls; same public API as effects.EffectSystem (ARCHITECTURE.md 5.4)."""

    def __init__(self, settings=None, quality=None, scene_size=SCENE, seed=0, linger=0.0):
        self.palette = get_palette("blue")
        self.scene_scale = scene_size[1] / 720.0
        self.particle_count = 0
        self.linger = linger
        self.time = 0.0
        self.orbs: list[FakeOrb] = []
        self.beams: list[FakeBeam] = []
        self.frontal_beams: list[FakeFrontalBeam] = []
        self.auras: list[FakeAura] = []
        self.calls: list[tuple[str, dict]] = []
        self.grades: list[float] = []

    # management
    def set_scene_size(self, size):
        self.scene_scale = size[1] / 720.0

    def set_quality(self, preset):
        pass

    def apply_settings(self, settings):
        pass

    def update(self, dt, now):
        self.time += max(0.0, dt) if math.isfinite(dt) else 0.0
        for handle in self.all_handles():
            if (handle.alive and handle.finished_time is not None
                    and self.time - handle.finished_time >= self.linger):
                handle.alive = False   # the end animation is over

    def collect(self, out):
        pass

    def clear(self):
        for handle in self.all_handles():
            handle.alive = False

    # steerable effects
    def create_orb(self, center, radius):
        orb = FakeOrb(self, center, radius)
        self.orbs.append(orb)
        return orb

    def create_beam(self, origin, direction, width, power):
        beam = FakeBeam(self, origin, direction, width, power)
        self.beams.append(beam)
        return beam

    def create_frontal_beam(self, center, radius, power):
        beam = FakeFrontalBeam(self, center, radius, power)
        self.frontal_beams.append(beam)
        return beam

    def create_aura(self):
        aura = FakeAura(self)
        self.auras.append(aura)
        return aura

    # fire-and-forget effects
    def ki_blast(self, origin, direction, frontal, radius, power):
        self.calls.append(("ki_blast", dict(origin=origin, direction=direction, frontal=frontal,
                                            radius=radius, power=power)))

    def burst(self, center, radius, power):
        self.calls.append(("burst", dict(center=center, radius=radius, power=power)))

    def shockwave(self, center, radius, power, speed=None):
        self.calls.append(("shockwave", dict(center=center, radius=radius, power=power)))

    def flash(self, intensity, center=None, radius=0.0, duration=0.25):
        self.calls.append(("flash", dict(intensity=intensity)))

    def shake(self, amplitude, duration):
        self.calls.append(("shake", dict(amplitude=amplitude, duration=duration)))

    def sparks(self, center, count, speed, direction=None, spread=math.tau, power=1.0):
        self.calls.append(("sparks", dict(center=center, count=count)))

    def fizzle(self, center, radius):
        self.calls.append(("fizzle", dict(center=center, radius=radius)))

    def set_grade(self, darken, desaturate=0.0):
        self.grades.append(darken)

    def add_light(self, position, radius, intensity):
        self.calls.append(("add_light", dict(position=position)))

    # helpers for tests
    def calls_named(self, name):
        return [kw for n, kw in self.calls if n == name]

    def all_handles(self):
        return [*self.orbs, *self.beams, *self.frontal_beams, *self.auras]


class FakeAudio:
    """AudioSink that records (time, name, volume) and tracks running loops."""

    def __init__(self, clock):
        self.clock = clock
        self.plays: list[tuple[float, str, float]] = []
        self.loop_calls: list[tuple[float, str, float]] = []
        self.stops: list[tuple[float, str]] = []
        self.active: dict[str, float] = {}

    def play(self, name, volume=1.0):
        assert name in SOUND_NAMES and name not in LOOP_SOUNDS, name
        assert 0.0 <= volume <= 1.0
        self.plays.append((self.clock.now, name, volume))

    def loop(self, name, volume):
        assert name in LOOP_SOUNDS, name
        assert 0.0 <= volume <= 1.0
        self.loop_calls.append((self.clock.now, name, volume))
        self.active[name] = volume

    def stop(self, name, fade_ms=250):
        assert name in SOUND_NAMES, name
        self.stops.append((self.clock.now, name))
        self.active.pop(name, None)

    def count(self, name):
        return sum(1 for _, n, _ in self.plays if n == name)

    def times(self, name):
        return [t for t, n, _ in self.plays if n == name]

    def loop_volumes(self, name):
        return [v for _, n, v in self.loop_calls if n == name]


# --------------------------------------------------------------- input builders


def make_arm(side, center, present=True):
    return ArmState(side, present=present, source="hand" if present else "none",
                    hand_center=center if present else None, hand_size=40.0 if present else 0.0,
                    shoulder=(center[0], center[1] - 150.0), wrist=(center[0], center[1] + 20.0))


def make_body(mid=(640.0, 420.0), gap=90.0, scale=200.0, stability=0.5, energy=0.0, both=True,
              center=(640.0, 380.0), scene=SCENE) -> BodyState:
    left = make_arm("left", (mid[0] - gap / 2, mid[1]), present=both)
    right = make_arm("right", (mid[0] + gap / 2, mid[1]), present=both)
    return BodyState(timestamp=0.0, scene_size=scene, quality="full",
                     hands_detected=2 if both else 0, body_scale=scale, body_center=center,
                     left=left, right=right, hand_mid=mid if both else None,
                     hand_gap=gap if both else 0.0,
                     hand_distance=gap / scale if both else float("inf"),
                     hands_in_front=both, energy=energy, stability=stability)


def pose_signals(**kw) -> GestureSignals:
    """Charge pose held, tracking fine."""
    return GestureSignals(charge_pose=0.9, charge_pose_active=True, tracking_ok=True, **kw)


def tracked(**kw) -> GestureSignals:
    """Tracking fine, no pose."""
    return GestureSignals(tracking_ok=True, **kw)


def extended(aim=(1.0, 0.0), **kw) -> GestureSignals:
    """Arms extended (beam sustain), tracking fine."""
    return GestureSignals(tracking_ok=True, arms_extended=True, aim_direction=aim, **kw)


LOST = GestureSignals(tracking_ok=False)


def thrust(now, frontal=0.1, direction=(1.0, 0.0), strength=0.9) -> GestureEvent:
    return GestureEvent(GestureEventType.THRUST, now, strength=strength, direction=direction,
                        frontal=frontal, position=(640.0, 420.0))


def burst_event(now, position=(600.0, 400.0)) -> GestureEvent:
    return GestureEvent(GestureEventType.BURST, now, strength=0.9, position=position)


def make_frame(frame_id=1, now=0.0, seg_id=3) -> TrackingFrame:
    pose = PoseObservation(landmarks=np.zeros((33, 3), np.float32),
                           visibility=np.ones(33, np.float32), score=0.9)
    return TrackingFrame(frame_id=frame_id, capture_time=now, result_time=now, scene_size=SCENE,
                         image=None, segmentation=np.full((144, 256), 200, np.uint8),
                         segmentation_id=seg_id, pose=pose)


PRESS = ManualInput(trigger_held=True, trigger_pressed=True)
HOLD = ManualInput(trigger_held=True)
RELEASE = ManualInput(trigger_released=True)


class Harness:
    """Drives an AbilityManager at 30 Hz with a fake clock, effects and audio."""

    def __init__(self, selected="kamehameha", settings=None, effects=None, start=100.0,
                 aura_auto=True, linger=0.0):
        self.settings = settings or Settings()
        self.settings.abilities.selected = selected
        self.settings.abilities.aura_auto = aura_auto
        self.now = start
        self.fx = effects or FakeEffectSystem(linger=linger)
        self.audio = FakeAudio(self)
        self.mgr = AbilityManager(self.settings, self.fx, self.audio)

    def step(self, body=None, signals=None, events=(), manual=None, frame=None, dt=DT):
        self.now += dt
        self.mgr.update(self.now, dt, frame, body,
                        signals if signals is not None else GestureSignals(),
                        list(events), manual if manual is not None else ManualInput())
        self.fx.update(dt, self.now)        # same order as app.py: abilities, then effects
        return self.mgr.hud(self.now)

    def run(self, seconds, **kw):
        return [self.step(**kw) for _ in range(int(round(seconds / DT)))]

    def run_while(self, predicate, limit, **kw):
        """Step until ``predicate()`` is False (at most ``limit`` seconds); returns elapsed s."""
        start = self.now
        while predicate() and self.now - start < limit:
            self.step(**kw)
        return self.now - start

    def ability(self, ability_id):
        return self.mgr.ability(ability_id)

    def phase(self, ability_id="kamehameha"):
        return self.mgr.ability(ability_id).sm.phase

    def charge(self, seconds, body):
        """Hold the charge pose for ``seconds``; returns the newest orb."""
        self.run(seconds, body=body, signals=pose_signals())
        return self.fx.orbs[-1]


@pytest.fixture
def allow_errors():
    """Request this fixture in tests that deliberately make the manager log errors."""
    return True


@pytest.fixture(autouse=True)
def _fail_on_logged_errors(request, caplog):
    caplog.set_level(logging.WARNING, logger="dragon_ball_powers")
    yield
    if "allow_errors" in request.fixturenames:
        return
    errors = [r for r in caplog.records
              if r.name.startswith("dragon_ball_powers") and r.levelno >= logging.ERROR]
    assert not errors, "errors were logged:\n" + "\n".join(r.getMessage() for r in errors)


def assert_all_finished(h: Harness):
    assert h.mgr.held_effects() == []
    unfinished = [hd for hd in h.fx.all_handles() if not hd.finished]
    assert not unfinished, unfinished


# ------------------------------------------------------------------------ registry


def test_registry_and_hotkeys():
    assert ABILITY_TYPES == {"ki_charge": KiChargeAbility, "kamehameha": KamehamehaAbility,
                             "energy_burst": EnergyBurstAbility,
                             "power_aura": PowerAuraAbility}
    for ability_id, cls in ABILITY_TYPES.items():
        assert cls.id == ability_id and cls.label
    h = Harness()
    assert h.mgr.hotkeys == {"1": "ki_charge", "2": "kamehameha", "3": "energy_burst",
                             "4": "power_aura"}
    assert h.mgr.selected_id == "kamehameha"
    hud = h.mgr.hud(h.now)
    assert hud.selected == "kamehameha" and hud.selected_label == "Kamehameha"
    assert hud.phase == "idle"


# --------------------------------------------------------------------- kamehameha


def test_kamehameha_lateral_full_cycle():
    h = Harness(aura_auto=False)
    body = make_body(stability=0.5)
    h.step(body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.PREPARING
    assert len(h.fx.orbs) == 1 and h.audio.count("charge_start") == 1
    orb = h.fx.orbs[0]
    h.run(0.25, body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.CHARGING
    assert h.audio.active.get("charge_loop") is not None
    h.run(1.0, body=body, signals=pose_signals())
    kame = h.ability("kamehameha")
    assert kame.sm.can_fire
    hud = h.mgr.hud(h.now)
    assert hud.charge_ready and hud.message == HINT_FIRE and hud.phase == "charging"

    center, radius = orb.center, orb.radius
    hud = h.step(body=body, signals=pose_signals(thrust=0.8),
                 events=[thrust(h.now, frontal=0.1, direction=(0.8, -0.6))])
    fire_time = h.now
    assert h.phase() is AttackPhase.FIRING
    assert orb.released and not orb.dissipated
    assert len(h.fx.beams) == 1 and not h.fx.frontal_beams
    beam = h.fx.beams[0]
    fc = kame.fire_charge
    assert 0.3 < fc < 0.45
    assert beam.origin == pytest.approx(center)
    assert beam.direction == pytest.approx((0.8, -0.6))
    assert beam.width == pytest.approx(radius * (0.9 + 0.6 * fc))
    assert beam.power == pytest.approx(0.5 + fc)
    assert h.audio.count("beam_fire") == 1
    assert "beam_loop" in h.audio.active and "charge_loop" not in h.audio.active
    assert hud.callout == CALLOUT_BEAM and hud.phase == "firing"

    min_d = h.settings.gestures.beam_min_duration
    max_d = min_d + (h.settings.gestures.beam_max_duration - min_d) * fc
    arms = make_body(mid=(900.0, 300.0), stability=0.5)
    h.run(min_d + 0.3, body=arms, signals=extended(aim=(0.8, -0.6)))
    assert h.phase() is AttackPhase.FIRING and not beam.ended  # sustained beyond min_d
    assert beam.sets[-1]["origin"] == pytest.approx((900.0, 300.0))  # follows the hands
    assert min(h.fx.grades[-5:]) == pytest.approx(0.9)

    lowered_at = h.now
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 3.0, body=arms, signals=tracked())
    assert beam.ended
    assert 0.35 <= h.now - lowered_at <= 0.35 + 2 * DT  # grace after lowering the arms
    assert min_d <= h.now - fire_time < max_d
    assert h.phase() is AttackPhase.COOLDOWN
    assert "beam_loop" not in h.audio.active
    assert h.mgr.hud(h.now).cooldown > 0.5
    h.run(h.settings.gestures.cooldown_time + 0.1, body=arms, signals=tracked())
    assert h.phase() is AttackPhase.IDLE
    assert h.mgr.hud(h.now).message == HINT_CHARGE
    assert_all_finished(h)


def test_kamehameha_frontal_beam_follows_hands():
    h = Harness()
    body = make_body()
    orb = h.charge(1.3, body)
    center, radius = orb.center, orb.radius
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now, frontal=0.8)])
    kame = h.ability("kamehameha")
    assert not h.fx.beams and len(h.fx.frontal_beams) == 1
    beam = h.fx.frontal_beams[0]
    assert beam.center == pytest.approx(center)
    assert beam.radius == pytest.approx(radius)
    assert beam.power == pytest.approx(0.5 + kame.fire_charge)
    assert kame.beam_frontal
    moved = make_body(mid=(700.0, 380.0))
    h.run(0.5, body=moved, signals=extended(aim_frontal=0.9))
    assert beam.sets[-1]["center"] == pytest.approx((700.0, 380.0))
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 6.0, body=moved, signals=tracked())
    assert beam.ended and h.phase() is AttackPhase.COOLDOWN
    assert_all_finished(h)


def test_frontal_beam_is_sustained_while_arms_point_at_the_camera():
    h = Harness()
    body = make_body()
    h.charge(1.5, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now, frontal=0.9)])
    fire_time = h.now
    fc = h.ability("kamehameha").fire_charge
    g = h.settings.gestures
    max_d = g.beam_min_duration + (g.beam_max_duration - g.beam_min_duration) * fc
    # arms_extended reads False (foreshortened), but the arms point at the camera.
    pointing = GestureSignals(tracking_ok=True, arms_extended=False, aim_frontal=0.8)
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 10.0, body=body, signals=pointing)
    assert h.now - fire_time == pytest.approx(max_d, abs=2 * DT)
    assert h.fx.frontal_beams[0].ended


@pytest.mark.parametrize("frontal, expect_frontal", [(0.55, True), (0.54, False)])
def test_frontal_threshold(frontal, expect_frontal):
    h = Harness()
    body = make_body()
    h.charge(1.2, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now, frontal=frontal)])
    assert len(h.fx.frontal_beams) == (1 if expect_frontal else 0)
    assert len(h.fx.beams) == (0 if expect_frontal else 1)


@pytest.mark.parametrize("key_aim, expected", [((0.0, 1.0), (0.0, 1.0)),
                                               ((0.0, 0.0), (-0.6, -0.8))])
def test_lateral_direction_fallbacks(key_aim, expected):
    """event.direction, else the arrow-key aim, else where the arms point."""
    h = Harness()
    body = make_body()
    h.charge(1.2, body)
    no_direction = thrust(h.now, frontal=0.1, direction=(0.0, 0.0))
    h.step(body=body, signals=pose_signals(aim_direction=(-3.0, -4.0)), events=[no_direction],
           manual=ManualInput(aim=key_aim))
    assert h.fx.beams[0].direction == pytest.approx(expected)


def test_beam_stops_at_max_duration_and_lasts_at_least_min_duration():
    h = Harness()
    body = make_body()
    h.charge(1.5, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    fire_time = h.now
    fc = h.ability("kamehameha").fire_charge
    g = h.settings.gestures
    max_d = g.beam_min_duration + (g.beam_max_duration - g.beam_min_duration) * fc
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 10.0, body=body, signals=extended())
    assert h.now - fire_time == pytest.approx(max_d, abs=2 * DT)

    # Arms lowered right after the release: the beam still lasts beam_min_duration.
    h.run(1.0, body=body, signals=tracked())
    h.charge(1.5, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    fire_time = h.now
    assert len(h.fx.beams) == 2
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 10.0, body=body, signals=tracked())
    assert h.now - fire_time == pytest.approx(g.beam_min_duration, abs=2 * DT)
    assert_all_finished(h)


def test_beam_direction_follows_arms_smoothly_and_power_decays():
    h = Harness()
    body = make_body()
    h.charge(1.5, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now, direction=(1.0, 0.0))])
    beam = h.fx.beams[0]
    h.step(body=body, signals=extended(aim=(0.0, -1.0)))
    first = beam.sets[-1]["direction"]
    assert first[0] > 0.7 and -0.5 < first[1] < 0.0       # turning, not jumping
    h.run(1.0, body=body, signals=extended(aim=(0.0, -1.0)))
    last = beam.sets[-1]["direction"]
    assert last[1] < -0.95 and math.hypot(*last) == pytest.approx(1.0)
    powers = [s["power"] for s in beam.sets]
    assert all(b <= a + 1e-9 for a, b in zip(powers, powers[1:]))
    assert powers[0] == pytest.approx(beam.power, rel=0.05)
    assert 0.7 * beam.power <= powers[-1] < powers[0]


def test_thrust_too_early_does_not_fire_and_says_charge_longer():
    h = Harness()
    body = make_body()
    h.run(0.4, body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.CHARGING and not h.ability("kamehameha").sm.can_fire
    hud = h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    assert h.phase() is AttackPhase.CHARGING
    assert not h.fx.beams and not h.fx.frontal_beams
    assert hud.message == MSG_CHARGE_LONGER
    assert not h.fx.orbs[0].finished                    # the charge goes on
    huds = h.run(1.6, body=body, signals=pose_signals())
    assert huds[-1].message in (HINT_FIRE, HINT_KEEP_CHARGING)


def test_thrust_while_preparing_says_charge_longer():
    h = Harness()
    body = make_body()
    h.step(body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.PREPARING
    hud = h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    assert hud.message == MSG_CHARGE_LONGER and not h.fx.beams


def test_one_thrust_gesture_is_exactly_one_attack():
    h = Harness()
    body = make_body()
    h.charge(1.2, body)
    for _ in range(5):  # a jittery detector reporting the same push several times
        h.step(body=body, signals=pose_signals(thrust=0.9), events=[thrust(h.now)])
    assert len(h.fx.beams) + len(h.fx.frontal_beams) == 1
    # The user keeps holding the charge pose: no new charge starts by itself.
    huds = h.run(4.0, body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.IDLE
    assert len(h.fx.orbs) == 1
    assert huds[-1].message == HINT_POSE_RESET
    h.step(body=body, signals=tracked())                  # pose released once
    h.run(0.4, body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.CHARGING and len(h.fx.orbs) == 2


def step_until_phase_time(h, ability_id, seconds, **kw):
    """Step until the *next* tick is the first with ``phase_time >= seconds`` (the tick in
    which the state machine leaves its current phase by timer)."""
    sm = h.ability(ability_id).sm
    while h.now + DT - sm.phase_started < seconds:
        h.step(**kw)


def test_thrust_that_fires_from_preparing_does_not_say_charge_longer():
    """Regression: with min_charge_time 0 a thrust in the tick that cascades
    PREPARING -> CHARGING -> FIRING fired the beam *and* showed "Charge longer!"."""
    settings = Settings()
    settings.gestures.min_charge_time = 0.0
    h = Harness(settings=settings)
    body = make_body()
    h.step(body=body, signals=pose_signals())
    step_until_phase_time(h, "kamehameha", settings.gestures.prepare_time, body=body,
                          signals=pose_signals())
    assert h.phase() is AttackPhase.PREPARING
    hud = h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    assert h.phase() is AttackPhase.FIRING and len(h.fx.beams) == 1
    assert hud.message is None


def test_thrust_during_cooldown_or_while_idle_does_nothing():
    h = Harness()
    body = make_body()
    h.charge(1.3, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 6.0, body=body, signals=tracked())
    assert h.phase() is AttackPhase.COOLDOWN
    hud = h.step(body=body, signals=tracked(), events=[thrust(h.now)])
    assert len(h.fx.beams) + len(h.fx.frontal_beams) == 1
    assert hud.message != MSG_CHARGE_LONGER and h.phase() is AttackPhase.COOLDOWN
    h.mgr.select("ki_charge", h.now)                      # switching during the cooldown...
    h.mgr.select("kamehameha", h.now)                     # ...and back keeps it running
    assert h.phase() is AttackPhase.COOLDOWN
    h.run(1.0, body=body, signals=tracked())
    assert h.phase() is AttackPhase.IDLE
    hud = h.step(body=body, signals=tracked(), events=[thrust(h.now)])
    assert len(h.fx.beams) + len(h.fx.frontal_beams) == 1 and len(h.fx.orbs) == 1
    assert hud.message == HINT_CHARGE                     # a push without a charge: no nagging


# ---------------------------------------------------------------------- keyboard


def test_manual_hold_and_release_fires_without_tracking():
    h = Harness()
    h.step(manual=PRESS)                                   # body None, no tracking at all
    assert h.phase() is AttackPhase.CHARGING             # manual skips preparation
    kame = h.ability("kamehameha")
    assert kame.sm.manual
    orb = h.fx.orbs[0]
    assert orb.targets[-1]["center"] == pytest.approx((0.5 * 1280, 0.62 * 720))
    assert orb.targets[-1]["radius"] == pytest.approx(0.22 * 720 * 0.10)
    huds = h.run(1.0, manual=HOLD)
    assert huds[-1].charge_ready and huds[-1].message == HINT_FIRE_KEY
    h.step(manual=RELEASE)
    fire_time = h.now
    assert h.phase() is AttackPhase.FIRING
    # No aim keys and no arm direction: the beam goes towards the camera.
    assert len(h.fx.frontal_beams) == 1 and not h.fx.beams
    assert orb.released
    fc = kame.fire_charge
    g = h.settings.gestures
    min_d = g.beam_min_duration
    max_d = min_d + (g.beam_max_duration - min_d) * fc
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 10.0)
    assert h.now - fire_time == pytest.approx((min_d + max_d) / 2, abs=2 * DT)
    assert not kame.sm.waiting_for_pose_reset
    assert_all_finished(h)


def test_manual_release_too_early_fizzles():
    h = Harness()
    h.step(manual=PRESS)
    h.run(0.3, manual=HOLD)
    orb = h.fx.orbs[0]
    hud = h.step(manual=RELEASE)
    assert h.phase() is AttackPhase.IDLE
    assert orb.dissipated and not orb.released
    assert h.audio.count("fizzle") == 1
    assert "charge_loop" not in h.audio.active
    assert hud.message == MSG_CHARGE_LONGER
    assert not h.fx.beams and not h.fx.frontal_beams
    hud = h.run(1.6)[-1]
    assert hud.message != MSG_CHARGE_LONGER           # shown for about 1.5 s


def test_manual_aim_fires_lateral_beam():
    h = Harness()
    body = make_body()
    h.step(body=body, signals=tracked(), manual=PRESS)
    h.run(1.0, body=body, signals=tracked(), manual=HOLD)
    h.step(body=body, signals=tracked(), manual=ManualInput(trigger_released=True, aim=(-1, 0)))
    assert len(h.fx.beams) == 1 and not h.fx.frontal_beams
    assert h.fx.beams[0].direction == pytest.approx((-1.0, 0.0))


def test_manual_release_with_extended_arms_uses_arm_direction():
    h = Harness()
    body = make_body()
    h.step(body=body, signals=tracked(), manual=PRESS)
    h.run(1.0, body=body, signals=tracked(), manual=HOLD)
    h.step(body=body, signals=extended(aim=(0.0, -1.0)), manual=RELEASE)
    assert len(h.fx.beams) == 1
    assert h.fx.beams[0].direction == pytest.approx((0.0, -1.0))


def test_manual_charge_orb_sits_in_front_of_the_chest_without_hands():
    h = Harness()
    body = make_body(both=False, center=(640.0, 380.0), scale=200.0)
    h.step(body=body, signals=tracked(), manual=PRESS)
    orb = h.fx.orbs[0]
    assert orb.targets[-1]["center"] == pytest.approx((640.0, 380.0 + 0.3 * 200.0))
    assert orb.targets[-1]["left_hand"] is None and orb.targets[-1]["right_hand"] is None


def test_keyboard_orb_follows_the_chest_until_hands_are_seen():
    """Regression: without hands the orb stayed where it ignited while the user moved."""
    h = Harness()
    h.step(body=make_body(both=False, center=(640.0, 380.0)), signals=tracked(), manual=PRESS)
    orb = h.fx.orbs[0]
    h.run(0.3, body=make_body(both=False, center=(400.0, 360.0)), signals=tracked(),
          manual=HOLD)
    assert orb.targets[-1]["center"] == pytest.approx((400.0, 360.0 + 0.3 * 200.0))
    h.step(body=None, signals=LOST, manual=HOLD)          # nobody in view: stays put
    assert orb.targets[-1]["center"] == pytest.approx((400.0, 360.0 + 0.3 * 200.0))
    h.step(body=make_body(mid=(420.0, 430.0)), signals=tracked(), manual=HOLD)
    assert orb.targets[-1]["center"] == pytest.approx((420.0, 430.0))   # between the hands
    # The hands vanish again: the orb stays where they were, it does not jump to the chest.
    h.step(body=make_body(both=False, center=(380.0, 360.0)), signals=tracked(), manual=HOLD)
    assert orb.targets[-1]["center"] == pytest.approx((420.0, 430.0))
    h.step(body=None, signals=LOST, manual=HOLD)
    assert orb.targets[-1]["center"] == pytest.approx((420.0, 430.0))


def test_stale_space_release_does_not_cancel_a_new_gesture_charge():
    """Regression: sm.manual stays True through FIRING / COOLDOWN of a keyboard attack. A
    Space release in the tick its cooldown ended leaked fire=True into the gesture charge
    that cascaded up in the same update (prepare_time 0) and fizzled it at once."""
    settings = Settings()
    settings.gestures.prepare_time = 0.0
    h = Harness(settings=settings)
    body = make_body()
    h.step(manual=PRESS)
    h.run(1.0, manual=HOLD)
    h.step(manual=RELEASE)
    h.run_while(lambda: h.phase() is not AttackPhase.COOLDOWN, 10.0)
    assert h.ability("kamehameha").sm.manual
    step_until_phase_time(h, "kamehameha", settings.gestures.cooldown_time, body=body,
                          signals=tracked())
    fizzles = h.audio.count("fizzle")
    hud = h.step(body=body, signals=pose_signals(), manual=ManualInput(trigger_released=True))
    assert h.phase() is AttackPhase.CHARGING              # the new gesture charge lives on
    assert h.audio.count("fizzle") == fizzles and not h.fx.orbs[-1].finished
    assert hud.message == HINT_KEEP_CHARGING


def test_space_tap_within_one_frame():
    """Pressed and released inside one frame (never seen held): no charge, but a burst."""
    tap = ManualInput(trigger_held=False, trigger_pressed=True, trigger_released=True)
    h = Harness()
    hud = h.step(manual=tap)
    assert h.phase() is AttackPhase.IDLE and not h.fx.orbs
    assert h.audio.count("fizzle") == 0 and hud.message != MSG_CHARGE_LONGER
    h = Harness("energy_burst")
    h.step(manual=tap)
    assert len(h.fx.calls_named("burst")) == 1


def test_space_held_through_the_cooldown_charges_again_when_ready():
    h = Harness("ki_charge")
    h.step(manual=PRESS)
    h.run(1.0, manual=HOLD)
    h.step(manual=RELEASE)
    h.run_while(lambda: h.phase("ki_charge") is AttackPhase.FIRING, 2.0)
    assert h.phase("ki_charge") is AttackPhase.COOLDOWN
    h.step(manual=PRESS)                                   # pressed again too early
    assert h.phase("ki_charge") is AttackPhase.COOLDOWN and len(h.fx.orbs) == 1
    h.run_while(lambda: h.phase("ki_charge") is AttackPhase.COOLDOWN, 2.0, manual=HOLD)
    ki = h.ability("ki_charge")
    assert ki.sm.phase is AttackPhase.CHARGING and ki.sm.manual and len(h.fx.orbs) == 2
    h.run(1.0, manual=HOLD)
    h.step(manual=RELEASE)
    assert len(h.fx.calls_named("ki_blast")) == 2


# ---------------------------------------------------------------------- ki charge


@pytest.mark.parametrize("frontal", [0.8, 0.1])
def test_ki_charge_throws_ki_blast(frontal):
    h = Harness("ki_charge")
    body = make_body()
    orb = h.charge(1.3, body)
    center, radius = orb.center, orb.radius
    hud = h.step(body=body, signals=pose_signals(),
                 events=[thrust(h.now, frontal=frontal, direction=(0.0, -1.0))])
    fire_time = h.now
    ki = h.ability("ki_charge")
    assert h.phase("ki_charge") is AttackPhase.FIRING
    assert orb.released
    blasts = h.fx.calls_named("ki_blast")
    assert len(blasts) == 1
    blast = blasts[0]
    assert blast["origin"] == pytest.approx(center)
    assert blast["direction"] == pytest.approx((0.0, -1.0))
    assert blast["frontal"] == pytest.approx(frontal)
    assert blast["radius"] == pytest.approx(radius)
    assert blast["power"] == pytest.approx(0.5 + ki.fire_charge)
    assert not h.fx.beams and not h.fx.frontal_beams
    assert hud.callout == CALLOUT_BLAST
    assert h.audio.times("blast_fire") == [pytest.approx(fire_time)]
    h.run_while(lambda: h.phase("ki_charge") is AttackPhase.FIRING, 2.0, body=body,
                signals=tracked())
    assert h.now - fire_time == pytest.approx(0.4, abs=2 * DT)
    assert h.phase("ki_charge") is AttackPhase.COOLDOWN
    h.run(0.3, body=body, signals=tracked())
    impacts = h.audio.times("impact")
    assert len(impacts) == 1 and impacts[0] - fire_time == pytest.approx(0.35, abs=DT + 1e-6)
    assert h.phase("kamehameha") is AttackPhase.IDLE    # only the selected ability charged
    assert_all_finished(h)


def test_ki_charge_manual_without_tracking_goes_frontal():
    h = Harness("ki_charge")
    h.step(manual=PRESS)
    h.run(1.0, manual=HOLD)
    h.step(manual=RELEASE)
    blasts = h.fx.calls_named("ki_blast")
    assert len(blasts) == 1 and blasts[0]["frontal"] == pytest.approx(1.0)


# -------------------------------------------------------------------- energy burst


def test_burst_event_triggers_burst_with_cooldown():
    h = Harness()
    body = make_body(scale=200.0)
    hud = h.step(body=body, signals=tracked(), events=[burst_event(h.now, (600.0, 400.0))])
    bursts = h.fx.calls_named("burst")
    assert len(bursts) == 1
    assert bursts[0]["center"] == pytest.approx((600.0, 400.0))
    assert bursts[0]["radius"] == pytest.approx(100.0)
    assert bursts[0]["power"] == pytest.approx(0.6)
    assert h.audio.count("burst") == 1
    assert hud.callout == "HAAAAA!!!"
    h.run(0.5, body=body, signals=tracked())
    h.step(body=body, signals=tracked(), events=[burst_event(h.now)])
    assert len(h.fx.calls_named("burst")) == 1           # own cooldown (1 s)
    h.run(0.6, body=body, signals=tracked())
    h.step(body=body, signals=tracked(), events=[burst_event(h.now)])
    assert len(h.fx.calls_named("burst")) == 2


def test_burst_center_fallbacks():
    h = Harness()
    no_position = GestureEvent(GestureEventType.BURST, 0.0, position=None)
    h.step(body=make_body(mid=(500.0, 410.0)), signals=tracked(), events=[no_position])
    h.run(1.1)
    h.step(body=make_body(both=False, center=(650.0, 360.0)), signals=tracked(),
           events=[no_position])
    h.run(1.1)
    h.step(body=None, signals=LOST, events=[no_position])
    centers = [b["center"] for b in h.fx.calls_named("burst")]
    assert centers == [pytest.approx((500.0, 410.0)), pytest.approx((650.0, 360.0)),
                       pytest.approx((640.0, 360.0))]
    radii = [b["radius"] for b in h.fx.calls_named("burst")]
    assert radii[2] == pytest.approx(0.5 * 0.22 * 720)   # no body: S from the scene height


def test_burst_during_charge_consumes_the_orb():
    h = Harness()
    body = make_body()
    orb = h.charge(1.5, body)
    charge = h.ability("kamehameha").sm.charge
    assert charge > 0.3
    h.step(body=body, signals=pose_signals(), events=[burst_event(h.now)])
    assert orb.released and not orb.dissipated
    assert h.phase() is AttackPhase.IDLE
    bursts = h.fx.calls_named("burst")
    # The charge abilities update first, so the orb charged for one more tick
    # (rate 0.7 + 0.6 * 0.5 = 1.0, full_charge_time 3 s) before the burst consumed it.
    grown = charge + DT * 1.0 / h.settings.gestures.full_charge_time
    assert len(bursts) == 1 and bursts[0]["power"] == pytest.approx(0.6 + grown)
    assert h.audio.count("fizzle") == 0
    assert "charge_loop" not in h.audio.active
    assert not h.fx.beams and not h.fx.frontal_beams
    # A burst does not need a pose reset: charging again works right away.
    h.step(body=body, signals=tracked())
    h.run(0.4, body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.CHARGING and len(h.fx.orbs) == 2


def test_burst_disabled_and_manual_burst_when_selected():
    settings = Settings()
    settings.gestures.burst_enabled = False
    h = Harness("energy_burst", settings=settings)
    body = make_body()
    hud = h.step(body=body, signals=tracked(), events=[burst_event(h.now)])
    assert not h.fx.calls_named("burst")
    assert hud.message == HINT_BURST_KEY and hud.phase == "idle"
    hud = h.step(body=body, signals=tracked(), manual=PRESS)
    bursts = h.fx.calls_named("burst")
    assert len(bursts) == 1 and bursts[0]["power"] == pytest.approx(0.6)
    assert bursts[0]["center"] == pytest.approx(body.hand_mid)
    assert hud.phase == "cooldown" and hud.cooldown == pytest.approx(1.0, abs=0.05)
    assert hud.charge == 0.0 and not hud.charge_ready
    h.run(0.5, body=body, signals=tracked(), manual=HOLD)  # holding Space does not charge
    assert len(h.fx.orbs) == 0
    hud = h.run(0.6, body=body, signals=tracked())[-1]
    assert hud.phase == "idle" and hud.cooldown == 0.0
    # Space does not trigger a burst when a charge ability is selected.
    h.mgr.select("kamehameha", h.now)
    h.step(body=body, signals=tracked(), manual=PRESS)
    assert len(h.fx.calls_named("burst")) == 1 and len(h.fx.orbs) == 1


def test_burst_gesture_works_for_every_selection():
    for selected in ("ki_charge", "kamehameha", "energy_burst"):
        h = Harness(selected)
        h.step(body=make_body(), signals=tracked(), events=[burst_event(h.now)])
        assert len(h.fx.calls_named("burst")) == 1, selected


def test_burst_requires_releasing_the_charge_pose_before_the_next_charge():
    """Regression: sm.cancel() asks for no pose reset, so a charge pose flag that is still
    set right after the arms flew apart re-ignited a fresh orb under the explosion."""
    h = Harness()
    body = make_body()
    orb = h.charge(1.5, body)
    h.step(body=body, signals=pose_signals(), events=[burst_event(h.now)])
    assert orb.released and h.phase() is AttackPhase.IDLE
    starts = h.audio.count("charge_start")
    huds = h.run(0.5, body=body, signals=pose_signals())  # detector hysteresis / lag
    assert h.phase() is AttackPhase.IDLE and len(h.fx.orbs) == 1
    assert h.audio.count("charge_start") == starts
    assert huds[-1].message == HINT_POSE_RESET
    # The keyboard is not blocked...
    h.step(body=body, signals=pose_signals(), manual=PRESS)
    assert h.phase() is AttackPhase.CHARGING and len(h.fx.orbs) == 2
    h.mgr.reset(h.now)
    # ...and after a burst, releasing the pose once re-arms the gesture.
    h.charge(1.5, body)
    h.step(body=body, signals=pose_signals(), events=[burst_event(h.now)])
    h.run(1.1, body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.IDLE
    h.step(body=body, signals=tracked())
    h.run(0.4, body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.CHARGING and len(h.fx.orbs) == 4


# ---------------------------------------------------------------------- selection


def test_selection_switch_mid_charge_dissipates_the_orb():
    h = Harness()
    body = make_body()
    orb = h.charge(1.0, body)
    h.mgr.select("ki_charge", h.now)
    assert orb.dissipated
    assert h.phase("kamehameha") is AttackPhase.IDLE
    assert "charge_loop" not in h.audio.active
    assert h.audio.count("fizzle") == 1
    assert h.settings.abilities.selected == "ki_charge"
    hud = h.step(body=body, signals=pose_signals())
    assert hud.selected == "ki_charge" and hud.selected_label == "Ki Charge"
    assert h.phase("ki_charge") is AttackPhase.PREPARING and len(h.fx.orbs) == 2
    assert h.phase("kamehameha") is AttackPhase.IDLE


def test_switching_during_a_beam_lets_it_finish():
    h = Harness()
    body = make_body()
    h.charge(1.5, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    beam = h.fx.beams[0]
    h.mgr.select("ki_charge", h.now)
    assert not beam.ended
    h.run(1.2, body=body, signals=extended())
    assert h.phase("kamehameha") is AttackPhase.FIRING and len(beam.sets) > 20
    h.run_while(lambda: h.phase("kamehameha") is AttackPhase.FIRING, 5.0, body=body,
                signals=tracked())
    assert beam.ended
    h.run(1.0, body=body, signals=tracked())
    assert h.phase("kamehameha") is AttackPhase.IDLE
    assert not h.fx.calls_named("ki_blast") and len(h.fx.orbs) == 1
    assert_all_finished(h)


def test_select_unknown_or_aura():
    h = Harness()
    h.mgr.select("does_not_exist", h.now)
    assert h.mgr.selected_id == "kamehameha"
    h.mgr.select("power_aura", h.now)                     # key 4 = aura toggle
    assert h.mgr.selected_id == "kamehameha" and h.mgr.aura_enabled
    assert h.settings.abilities.aura_enabled


# ------------------------------------------------------------------------- aura


def test_aura_toggle_creates_and_ends_the_aura():
    h = Harness(aura_auto=False)
    body = make_body(energy=0.4)
    frame = make_frame(seg_id=7)
    assert h.mgr.toggle_aura(h.now) is True
    assert h.settings.abilities.aura_enabled is True
    h.run(1.0, body=body, signals=tracked(), frame=frame)
    assert len(h.fx.auras) == 1
    aura = h.fx.auras[0]
    last = aura.sets[-1]
    assert last["mask"] is frame.segmentation and last["mask_id"] == 7
    assert last["pose"] is frame.pose and last["body"] is body
    expected = 0.45 + 0.35 * 0.4 + 0.3 * h.mgr.power_fraction
    assert last["intensity"] == pytest.approx(expected, abs=0.02)
    assert aura.sets[0]["intensity"] < aura.sets[-1]["intensity"]   # fades in
    assert h.audio.active["aura_loop"] == pytest.approx(0.6 * last["intensity"], abs=0.02)
    hud = h.mgr.hud(h.now)
    assert hud.aura_enabled and hud.aura_intensity == pytest.approx(last["intensity"])
    assert h.mgr.toggle_aura(h.now) is False
    hud = h.step(body=body, signals=tracked())
    assert aura.ended and "aura_loop" not in h.audio.active
    assert not hud.aura_enabled and hud.aura_intensity == 0.0
    assert len(h.fx.auras) == 1
    assert_all_finished(h)


def test_aura_keeps_last_mask_between_frames_and_drops_stale_data():
    h = Harness(aura_auto=False)
    h.mgr.toggle_aura(h.now)
    frame = make_frame(seg_id=2)
    h.step(body=make_body(), signals=tracked(), frame=frame)
    h.run(0.5, body=make_body(), signals=tracked(), frame=None)
    aura = h.fx.auras[0]
    assert aura.sets[-1]["mask"] is frame.segmentation  # no new frame this tick
    h.run(1.0, body=None, signals=LOST, frame=None)
    assert aura.sets[-1]["mask"] is None and aura.sets[-1]["pose"] is None
    assert not aura.ended                               # still enabled, nobody in view


def test_auto_aura_while_charging_and_firing():
    h = Harness(aura_auto=True)
    body = make_body(stability=0.5)
    h.charge(0.9, body)                                  # charge ~0.23
    assert not h.fx.auras
    h.charge(1.2, body)                                  # charge > 0.35
    assert len(h.fx.auras) == 1
    aura = h.fx.auras[0]
    intensities = [s["intensity"] for s in aura.sets]
    assert intensities[-1] > intensities[0] and max(intensities) <= 0.9
    hud = h.mgr.hud(h.now)
    assert not hud.aura_enabled and hud.aura_intensity > 0.0
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    h.run(0.8, body=body, signals=extended())
    assert aura.sets[-1]["intensity"] > 0.95             # flares to 1.0 while firing
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 6.0, body=body, signals=tracked())
    h.step(body=body, signals=tracked())
    assert aura.ended and "aura_loop" not in h.audio.active
    # A charge that fizzles also ends the automatic aura.
    h.run(1.0, body=body, signals=tracked())             # cooldown
    h.charge(2.0, body)
    assert len(h.fx.auras) == 2
    h.run(1.0, body=body, signals=tracked())
    assert h.fx.auras[1].ended and h.phase() is AttackPhase.IDLE
    assert_all_finished(h)


def test_no_auto_aura_when_disabled():
    h = Harness(aura_auto=False)
    h.charge(3.0, make_body())
    assert not h.fx.auras


# ------------------------------------------------------------------- power level


def test_power_level_rises_crosses_9000_on_full_charge_fire_and_decays():
    h = Harness(aura_auto=False)
    body = make_body(stability=0.0, energy=0.0)
    h.run(2.0, body=body, signals=tracked())
    idle_level = h.mgr.power_level
    assert 900 <= idle_level <= 1000                      # 300 + 700 tracked
    # stability 0 -> charge rate 0.7: a full charge takes 0.2 + 3.0 / 0.7 = 4.5 s
    levels = [hud.power_level for hud in h.run(5.0, body=body, signals=pose_signals())]
    assert levels[-1] > levels[0] + 4000                  # rises while charging
    assert levels[-1] < 9000 and h.ability("kamehameha").sm.charge == pytest.approx(1.0)
    assert h.audio.count("power_up") == 4                 # only the milestones so far
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    huds = h.run(1.2, body=body, signals=extended())
    assert max(hud.power_level for hud in huds) > 9000
    assert any(hud.callout == CALLOUT_OVER_9000 for hud in huds)
    assert h.audio.count("power_up") == 5
    assert huds[-1].power_fraction > 0.75
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 6.0, body=body, signals=tracked())
    peak = h.mgr.power_level
    h.run(2.0, body=body, signals=tracked())
    after_2s = h.mgr.power_level
    assert after_2s < peak                                 # decays slowly...
    assert after_2s > 0.4 * peak                           # ...half-life about 2 s
    h.run(6.0, body=body, signals=tracked())
    assert h.mgr.power_level < 3000


def test_power_level_over_9000_at_most_every_10_seconds():
    settings = Settings()
    fx = FakeEffectSystem()

    class Clock:
        now = 0.0

    audio = FakeAudio(Clock)
    callouts = CalloutBoard()
    ctx = AbilityContext(settings=settings, effects=fx, audio=audio, callouts=callouts,
                         messages=MessageBoard(), host=None)
    level = PowerLevel(ctx)
    inputs = dict(tracked=True, charge=0.0, stability=0.0, energy=0.0, aura_intensity=0.0,
                  fire_charge=1.0)
    t = 0.0
    crossings = []
    for attacking, seconds in ((True, 2.0), (False, 4.0), (True, 2.0), (False, 4.0),
                               (True, 2.0)):
        for _ in range(int(seconds / DT)):
            t += DT
            before = audio.count("power_up")
            level.update(t, DT, attacking=attacking, **inputs)
            if audio.count("power_up") > before:
                crossings.append(t)
    # Three upward crossings of 9000 (t ~0.3, ~6.3, ~12.3) but only two callouts.
    assert len(crossings) == 2 and crossings[1] - crossings[0] >= 10.0
    assert level.fraction == pytest.approx(level.value / 12000.0)
    assert 0 <= level.level <= 12000


def test_ki_blast_pushes_power_level_over_9000_during_the_attack():
    """Regression: the ki blast fires for 0.4 s; approaching the attack floor at 4/s left
    the reading around 8000, so "IT'S OVER 9000!" never happened for Ki Charge."""
    h = Harness("ki_charge", aura_auto=False)
    body = make_body(stability=0.5, energy=0.0)
    h.charge(0.9, body)                                    # just past min_charge_time
    ki = h.ability("ki_charge")
    assert ki.sm.can_fire and h.mgr.power_level < 4000
    ups = h.audio.count("power_up")
    hud = h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    fire_time = h.now
    assert hud.callout == CALLOUT_BLAST
    floor = 9001.0 + 2500.0 * ki.fire_charge
    huds = [hud]
    while h.phase("ki_charge") is AttackPhase.FIRING:
        assert h.mgr.power.value >= floor                  # pushed there during the attack
        huds.append(h.step(body=body, signals=tracked()))
    assert max(x.power_level for x in huds) > 9000         # the displayed number got there
    callout_times = []
    for _ in range(int(1.0 / DT)):
        if h.step(body=body, signals=tracked()).callout == CALLOUT_OVER_9000:
            callout_times.append(h.now - fire_time)
    assert callout_times and callout_times[0] <= 0.4 + 2 * DT + 1e-6
    assert h.audio.count("power_up") == ups + 1
    h.run(4.0, body=body, signals=tracked())
    assert h.mgr.power_level < 6000                        # and it decays afterwards


def test_release_shout_comes_before_its_over_9000():
    """The attack floor crosses 9000 within a frame or two; "IT'S OVER 9000!" must not
    swallow the "HAAAAA!!!" of the release, it follows when the shout's lock ends."""
    h = Harness(aura_auto=False)
    body = make_body(stability=0.5)
    h.charge(3.4, body)                                    # full charge
    ups = h.audio.count("power_up")
    hud = h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    fire_time = h.now
    assert hud.callout == CALLOUT_BEAM
    timeline = []
    for _ in range(int(2.5 / DT)):
        hud = h.step(body=body, signals=extended())
        timeline.append((h.now - fire_time, hud.callout, hud.power_level))
    first = next(t for t, callout, _ in timeline if callout == CALLOUT_OVER_9000)
    assert all(callout == CALLOUT_BEAM for t, callout, _ in timeline if t < first)
    assert any(level > 9000 for t, _, level in timeline if t < first)   # already crossed
    assert RELEASE_SHOUT_HOLD - 1e-6 <= first <= RELEASE_SHOUT_HOLD + 2 * DT + 1e-6
    # The scouter sound plays together with its callout, once.
    assert h.audio.count("power_up") == ups + 1
    assert h.audio.times("power_up")[-1] - fire_time == pytest.approx(first)
    assert timeline[-1][1] == CALLOUT_BEAM                 # the roar comes back afterwards


def test_over_9000_waits_for_a_blocked_callout_then_gives_up():
    class Clock:
        now = 0.0

    def make_level():
        callouts = CalloutBoard()
        audio = FakeAudio(Clock)
        ctx = AbilityContext(settings=Settings(), effects=FakeEffectSystem(), audio=audio,
                             callouts=callouts, messages=MessageBoard(), host=None)
        return PowerLevel(ctx), callouts, audio

    inputs = dict(tracked=True, charge=0.0, stability=0.0, energy=0.0, aura_intensity=0.0,
                  fire_charge=1.0, attacking=True)
    # A shout locked for 0.5 s: the callout shows right after it.
    level, callouts, audio = make_level()
    callouts.show("HA!", 0.0, priority=PRIORITY_RELEASE, hold=0.5)
    t, shown = 0.0, None
    while t < 1.5 and shown is None:
        t += DT
        level.update(t, DT, **inputs)
        if callouts.current(t)[0] == CALLOUT_OVER_9000:
            shown = t
    assert shown is not None and 0.5 <= shown <= 0.5 + 2 * DT
    assert audio.count("power_up") == 1
    # A callout blocked for longer than the patience: dropped silently, and the 10 s limit
    # is not used up by it (the next crossing is announced).
    level, callouts, audio = make_level()
    callouts.show("HA!", 0.0, priority=PRIORITY_RELEASE, duration=3.0, hold=3.0)
    t = 0.0
    for attacking, seconds in ((True, 2.0), (False, 4.0), (True, 1.0)):
        for _ in range(int(seconds / DT)):
            t += DT
            level.update(t, DT, **{**inputs, "attacking": attacking})
    assert audio.count("power_up") == 1 and callouts.current(t)[0] == CALLOUT_OVER_9000


def test_power_level_ignores_non_finite_inputs():
    level = PowerLevel()
    nan, inf = float("nan"), float("inf")
    for _ in range(30):
        level.update(1.0, DT, tracked=True, charge=nan, stability=nan, energy=inf,
                     aura_intensity=nan, attacking=False, fire_charge=nan)
    assert math.isfinite(level.value) and isinstance(level.level, int)
    assert 0.0 <= level.fraction <= 1.0
    level.update(2.0, nan, tracked=True, charge=0.5, stability=0.5, energy=0.0,
                 aura_intensity=0.0, attacking=True, fire_charge=nan)
    assert level.value >= 9001.0 and isinstance(level.level, int)
    level.display = nan                                    # even a corrupted display is safe
    assert level.level == 0


def test_power_level_target_formula():
    target = PowerLevel.target(tracked=True, charge=0.5, stability=1.0, energy=0.2,
                               aura_intensity=0.5, attacking=False, fire_charge=0.0)
    assert target == pytest.approx(300 + 700 + 8000 * 0.5 + 1500 * 0.2 + 1000 * 0.5)
    attack = PowerLevel.target(tracked=False, charge=0.0, stability=0.0, energy=0.0,
                               aura_intensity=0.0, attacking=True, fire_charge=0.4)
    assert attack == pytest.approx(9001 + 2500 * 0.4)


# ------------------------------------------------------------------------ chant


def test_kamehameha_chant_progression_and_firing_callout():
    h = Harness(aura_auto=False)
    body = make_body(stability=0.5, energy=0.0)
    seen = []
    intensities = []
    for hud in h.run(3.6, body=body, signals=pose_signals()):
        if hud.callout is not None and (not seen or seen[-1] != hud.callout):
            seen.append(hud.callout)
        if hud.callout is not None:
            intensities.append(hud.callout_intensity)
    assert seen == list(KAMEHAMEHA_CHANT)
    assert all(0.0 <= i <= 1.0 for i in intensities)
    hud = h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    assert hud.callout == CALLOUT_BEAM
    pulses = [hud.callout_intensity for hud in h.run(0.15, body=body, signals=extended())]
    assert all(0.0 < p <= 1.0 for p in pulses) and max(pulses) - min(pulses) > 0.05


def test_ki_charge_has_no_chant_and_callouts_can_be_hidden():
    h = Harness("ki_charge", aura_auto=False)
    huds = h.run(2.0, body=make_body(), signals=pose_signals())
    assert all(hud.callout is None for hud in huds)
    h.settings.effects.show_callouts = False
    h2 = Harness(settings=h.settings, aura_auto=False)
    huds = h2.run(2.0, body=make_body(), signals=pose_signals())
    assert all(hud.callout is None and hud.callout_intensity == 0.0 for hud in huds)


def test_callout_board_priority_lock_and_fade():
    board = CalloutBoard()
    assert board.show("IT'S OVER 9000!", 0.0, 1.0, priority=2, duration=2.0, hold=1.6)
    assert not board.show("HAAAAA!!!", 0.5, priority=1)   # locked by a more important one
    assert board.current(0.5)[0] == "IT'S OVER 9000!"
    assert board.show("HAAAAA!!!", 1.7, priority=1)        # lock expired
    text, intensity = board.current(1.7)
    assert text == "HAAAAA!!!" and intensity == pytest.approx(1.0)
    assert board.current(1.7 + 1.0)[1] == pytest.approx(0.5)  # fading out
    assert board.current(1.7 + 1.3) == (None, 0.0)          # gone after ~1.2 s
    messages = MessageBoard()
    messages.show("Charge longer!", 10.0)
    assert messages.current(11.0) == "Charge longer!" and messages.current(11.6) is None


def test_callout_board_one_shots_beat_refreshes_of_equal_priority():
    board = CalloutBoard()
    assert board.show("HA!", 0.0, priority=1, hold=0.4)
    assert not board.show("HAAAAA!!!", 0.1, priority=1)          # per-tick refresh waits
    assert board.current(0.1)[0] == "HA!"
    assert board.show("HAAAAA!!!", 0.2, priority=1, hold=0.5)    # another one-shot wins
    assert board.show("HAAAAA!!!", 0.3, priority=1)              # refreshing the same text
    assert not board.show("KA...", 0.4, priority=0)
    assert board.show("KA...", 0.8, priority=0)                  # lock expired
    assert board.current(0.8)[0] == "KA..."


# ---------------------------------------------------------------- orb behaviour


def test_orb_follows_hands_with_radius_from_charge_and_hand_gap():
    h = Harness(aura_auto=False)
    body = make_body(mid=(600.0, 400.0), gap=120.0, scale=200.0, stability=1.0)
    h.step(body=body, signals=pose_signals())
    orb = h.fx.orbs[0]
    assert orb.created_center == pytest.approx((600.0, 400.0))
    assert orb.created_radius == pytest.approx(0.5 * 0.10 * 200.0)
    target = orb.targets[-1]
    assert target["center"] == pytest.approx((600.0, 400.0))
    assert target["radius"] == pytest.approx(0.5 * 20.0 + 0.5 * 0.42 * 120.0)
    assert target["charge"] == 0.0
    assert target["left_hand"] == pytest.approx(body.left.hand_center)
    assert target["right_hand"] == pytest.approx(body.right.hand_center)
    h.run(3.0, body=body, signals=pose_signals())          # full charge (rate 1.3)
    assert orb.targets[-1]["charge"] == pytest.approx(1.0)
    assert orb.targets[-1]["radius"] == pytest.approx(0.5 * 120.0 + 0.5 * 0.42 * 120.0)
    wide = make_body(mid=(610.0, 400.0), gap=600.0, scale=200.0, stability=1.0)
    h.step(body=wide, signals=pose_signals())
    assert orb.targets[-1]["radius"] == pytest.approx(0.60 * 200.0)   # clamped to max
    assert orb.targets[-1]["center"] == pytest.approx((610.0, 400.0))
    hands_gone = make_body(both=False, scale=200.0, stability=1.0)
    h.step(body=hands_gone, signals=pose_signals())
    assert orb.targets[-1]["center"] == pytest.approx((610.0, 400.0))  # last known centre
    assert orb.targets[-1]["radius"] == pytest.approx(0.60 * 200.0)
    radii = [t["radius"] for t in orb.targets]
    assert all(0.10 * 200.0 - 1e-6 <= r <= 0.60 * 200.0 + 1e-6 for r in radii)


def test_charge_grade_audio_and_milestones():
    h = Harness(aura_auto=False)
    body = make_body(stability=0.5)
    h.run(3.6, body=body, signals=pose_signals())
    assert h.audio.count("charge_start") == 1
    assert h.audio.count("power_up") == 4                  # 25/50/75/100 %
    volumes = h.audio.loop_volumes("charge_loop")
    assert volumes[0] == pytest.approx(0.25, abs=0.03) and volumes[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-9 for a, b in zip(volumes, volumes[1:]))
    assert h.fx.grades[-1] == pytest.approx(0.6)            # darken = 0.6 * charge
    h.run(1.0, body=body, signals=pose_signals())
    assert h.audio.count("power_up") == 4                  # once each per charge


@pytest.mark.parametrize("path", ["fire", "pose_released", "manual_early", "burst",
                                  "deselect", "reset", "tracking_lost"])
def test_charge_loop_stops_on_every_path(path):
    h = Harness()
    body = make_body()
    if path == "manual_early":
        h.step(body=body, signals=tracked(), manual=PRESS)
        h.run(0.3, body=body, signals=tracked(), manual=HOLD)
    else:
        h.charge(1.2, body)
    assert "charge_loop" in h.audio.active
    if path == "fire":
        h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    elif path == "pose_released":
        h.run(0.6, body=body, signals=tracked())
    elif path == "manual_early":
        h.step(body=body, signals=tracked(), manual=RELEASE)
    elif path == "burst":
        h.step(body=body, signals=pose_signals(), events=[burst_event(h.now)])
    elif path == "deselect":
        h.mgr.select("energy_burst", h.now)
    elif path == "reset":
        h.mgr.reset(h.now)
    elif path == "tracking_lost":
        h.run(1.0, body=None, signals=LOST)
    assert "charge_loop" not in h.audio.active
    assert any(name == "charge_loop" for _, name in h.audio.stops)
    assert h.phase() is not AttackPhase.CHARGING
    assert h.fx.orbs[0].finished


def test_tracking_lost_during_charge_fizzles_after_the_grace():
    h = Harness()
    body = make_body()
    orb = h.charge(1.0, body)
    charge = h.ability("kamehameha").sm.charge
    h.run(0.4, body=None, signals=LOST)                    # shorter than lost_grace (0.6 s)
    assert h.phase() is AttackPhase.CHARGING and not orb.finished
    assert h.ability("kamehameha").sm.charge == pytest.approx(charge)   # frozen
    assert orb.targets[-1]["center"] == pytest.approx(body.hand_mid)    # stays in place
    h.run(0.4, body=None, signals=LOST)
    assert h.phase() is AttackPhase.IDLE
    assert orb.dissipated and h.audio.count("fizzle") == 1
    hud = h.run(2.0, body=None, signals=LOST)[-1]
    assert hud.message == HINT_NO_TRACKING


def test_short_tracking_dropout_keeps_the_charge():
    h = Harness()
    body = make_body()
    orb = h.charge(1.0, body)
    h.run(0.3, body=None, signals=LOST)
    h.run(0.5, body=body, signals=pose_signals())
    assert h.phase() is AttackPhase.CHARGING and len(h.fx.orbs) == 1 and not orb.finished


# ----------------------------------------------------------------------- HUD


def test_hud_hints():
    h = Harness()
    body = make_body()
    hud = h.step(body=body, signals=tracked())
    assert hud.message == HINT_CHARGE
    hud = h.run(1.0, body=None, signals=LOST)[-1]
    assert hud.message == HINT_CHARGE                     # short drop-out: no nagging
    hud = h.run(1.5, body=None, signals=LOST)[-1]
    assert hud.message == HINT_NO_TRACKING
    hud = h.step(body=body, signals=tracked())
    assert hud.message == HINT_CHARGE
    hud = h.run(0.4, body=body, signals=pose_signals())[-1]
    assert hud.message == HINT_KEEP_CHARGING and not hud.charge_ready
    hud = h.run(0.8, body=body, signals=pose_signals())[-1]
    assert hud.message == HINT_FIRE and hud.charge_ready and 0.0 < hud.charge < 1.0
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    hud = h.step(body=body, signals=extended())
    assert hud.message is None and hud.phase == "firing" and 0.0 < hud.charge <= 1.0
    h.mgr.select("energy_burst", h.now)
    hud = h.step(body=body, signals=extended())
    assert hud.selected == "energy_burst" and hud.message == HINT_BURST
    assert hud.phase == "idle"


# ------------------------------------------------------------ settings / reset


def test_apply_settings_updates_timings_selection_and_aura():
    h = Harness()
    new = h.settings.copy()
    new.gestures.min_charge_time = 2.0
    new.abilities.selected = "ki_charge"
    new.abilities.aura_enabled = True
    h.mgr.apply_settings(new)
    assert h.mgr.selected_id == "ki_charge" and h.mgr.aura_enabled
    ki = h.ability("ki_charge")
    assert ki.sm.config.min_charge_time == 2.0
    body = make_body()
    h.charge(1.2, body)
    hud = h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    assert h.phase("ki_charge") is AttackPhase.CHARGING and hud.message == MSG_CHARGE_LONGER
    assert not h.fx.calls_named("ki_blast")
    assert len(h.fx.auras) == 1


def test_reset_ends_effects_and_sounds():
    h = Harness()
    body = make_body()
    h.mgr.toggle_aura(h.now)
    h.charge(1.5, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    h.run(0.3, body=body, signals=extended())
    assert h.fx.beams and h.fx.auras
    h.mgr.reset(h.now)
    assert h.fx.beams[0].ended and h.fx.auras[0].ended
    assert not h.audio.active
    assert all(h.phase(a) is AttackPhase.IDLE for a in ("ki_charge", "kamehameha"))
    hud = h.mgr.hud(h.now)
    assert hud.callout is None and hud.power_level == 0
    assert hud.aura_enabled                               # the toggle survives a reset
    assert_all_finished(h)
    h.step(body=body, signals=tracked())
    assert len(h.fx.auras) == 2                           # and the aura comes back


def test_effects_cleared_externally_recreates_orb_and_ends_beam_early():
    h = Harness()
    body = make_body()
    h.charge(1.0, body)
    h.fx.clear()                                          # e.g. scene size change
    h.step(body=body, signals=pose_signals())
    assert len(h.fx.orbs) == 2 and h.fx.orbs[1].targets
    h.run(0.3, body=body, signals=pose_signals())
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    h.fx.clear()
    h.run(0.2, body=body, signals=extended())
    assert h.phase() is AttackPhase.COOLDOWN
    assert "beam_loop" not in h.audio.active
    assert h.mgr.held_effects() == []


# ----------------------------------------------------------------- robustness


def test_none_inputs_and_garbage_never_crash():
    h = Harness()
    nan = float("nan")
    for selected in ("ki_charge", "kamehameha", "energy_burst"):
        h.mgr.select(selected, h.now)
        h.mgr.update(h.now, DT, None, None, None, None, None)
        h.mgr.toggle_aura(h.now)
        for manual in (PRESS, HOLD, HOLD, RELEASE, ManualInput()):
            h.step(body=None, signals=None, manual=manual, frame=None)
        h.run(1.0, manual=HOLD)
        h.step(manual=RELEASE)
        h.step(events=[burst_event(h.now, position=None), thrust(h.now)])
        bad = make_body(mid=(nan, nan), scale=nan, stability=nan, energy=nan)
        bad.body_center = (nan, 1.0)
        bad.left.hand_center = (nan, nan)
        weird = GestureSignals(charge_pose_active=True, tracking_ok=True, thrust=nan,
                               aim_direction=(0.0, 0.0), aim_frontal=nan, arms_extended=True)
        h.run(1.0, body=bad, signals=weird)
        h.step(body=bad, signals=weird,
               events=[GestureEvent(GestureEventType.THRUST, h.now, strength=nan,
                                    direction=(nan, nan), frontal=nan, position=(nan, nan)),
                       GestureEvent(GestureEventType.BURST, h.now, position=(nan, 0.0))])
        h.run(0.5, body=bad, signals=weird)
        for dt in (-1.0, nan, 0.0, 5.0):
            h.step(body=make_body(), signals=pose_signals(), dt=dt)
        h.run(6.0)
        h.mgr.toggle_aura(h.now)
        h.run(0.5)
    for created in h.fx.orbs:
        for target in created.targets:
            assert all(math.isfinite(v) for v in target["center"])
            assert math.isfinite(target["radius"]) and target["radius"] > 0
    for call in h.fx.calls_named("burst") + h.fx.calls_named("ki_blast"):
        assert all(math.isfinite(v) for v in call.get("center", call.get("origin")))
    hud = h.mgr.hud(h.now)
    assert isinstance(hud.power_level, int)
    h.mgr.reset(h.now)
    assert_all_finished(h)


def test_malformed_frame_metadata_does_not_break_the_update():
    """The tick preamble runs outside the per-ability guards and must not raise."""
    h = Harness(aura_auto=False)
    h.mgr.toggle_aura(h.now)
    frame = make_frame(seg_id=4)
    frame.segmentation_id = None
    frame.scene_size = None
    h.step(body=make_body(), signals=tracked(), frame=frame)
    assert h.fx.auras and h.fx.auras[0].sets[-1]["mask_id"] == 0
    assert h.fx.auras[0].sets[-1]["mask"] is frame.segmentation
    inf = float("inf")
    h.step(body=make_body(scene=(inf, inf)), signals=tracked(),
           frame=make_frame(frame_id=2, seg_id=5))
    h.step(body=make_body(scene=(inf, 720)), signals=tracked())
    assert h.mgr.hud(h.now).aura_enabled and h.fx.auras[0].sets[-1]["mask_id"] == 5


def test_lingering_end_animations_are_never_touched_again():
    """Real handles stay alive while they fade out; the abilities drop them at once anyway
    and never steer or finish them a second time (the fakes raise if they do)."""
    h = Harness(linger=0.5)
    body = make_body()
    h.mgr.toggle_aura(h.now)
    for selected, frontal in (("kamehameha", 0.1), ("kamehameha", 0.9), ("ki_charge", 0.1)):
        h.mgr.select(selected, h.now)
        h.charge(1.4, body)
        h.step(body=body, signals=pose_signals(), events=[thrust(h.now, frontal=frontal)])
        h.run(0.5, body=body, signals=extended())
        h.run(3.0, body=body, signals=tracked())
    h.charge(1.0, body)
    h.step(body=body, signals=pose_signals(), events=[burst_event(h.now)])   # consumed
    h.step(body=body, signals=tracked())
    h.charge(1.0, body)
    h.mgr.select("energy_burst", h.now)                                         # dissipated
    h.mgr.toggle_aura(h.now)
    h.step(body=body, signals=tracked())
    assert h.mgr.held_effects() == []
    assert any(x.alive for x in h.fx.all_handles())      # end animations still playing...
    h.run(0.6, body=body, signals=tracked())
    assert not any(x.alive for x in h.fx.all_handles())  # ...and over now
    assert len(h.fx.beams) == 1 and len(h.fx.frontal_beams) == 1
    assert len(h.fx.calls_named("ki_blast")) == 1 and len(h.fx.calls_named("burst")) == 1
    assert_all_finished(h)


def test_effect_and_audio_failures_are_contained(allow_errors, caplog):
    class BrokenEffects(FakeEffectSystem):
        def create_beam(self, origin, direction, width, power):
            raise RuntimeError("beam shader exploded")

    class BrokenAudio(FakeAudio):
        def play(self, name, volume=1.0):
            raise OSError("mixer gone")

    h = Harness(effects=BrokenEffects())
    h.audio = BrokenAudio(h)
    h.mgr = AbilityManager(h.settings, h.fx, h.audio)
    body = make_body()
    h.charge(1.3, body)
    h.step(body=body, signals=pose_signals(), events=[thrust(h.now)])
    assert h.phase() is AttackPhase.FIRING                 # the attack itself went on
    h.run_while(lambda: h.phase() is AttackPhase.FIRING, 10.0, body=body, signals=tracked())
    assert h.phase() is AttackPhase.COOLDOWN
    assert "beam_loop" not in h.audio.active
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("kamehameha" in m for m in messages)
    assert any("Audio play" in m for m in messages)
    # Rate-limited: one traceback per problem, not one per frame.
    assert len(messages) < 10


@pytest.mark.parametrize("linger", [0.0, 0.5])
def test_random_session_keeps_handles_sounds_and_hud_consistent(linger):
    """Seeded random walk over all inputs; invariants are checked after every tick.

    With ``linger`` the fake handles stay alive during their end animation like the real
    ones: the abilities must still drop them at once and never touch them again.
    """
    rng = np.random.default_rng(20261018)
    h = Harness(linger=linger)
    frame_id = 0
    pose = space = arms = False
    tracking = True
    aim = (1.0, 0.0)
    charge_abilities = [h.ability("ki_charge"), h.ability("kamehameha")]
    aura = h.ability("power_aura")
    for _ in range(9000):
        # Persistent (Markov) inputs so that charges, beams and cooldowns really happen.
        pose ^= rng.random() < 0.03
        arms ^= rng.random() < 0.04
        space ^= rng.random() < 0.01
        tracking ^= rng.random() < (0.01 if tracking else 0.05)
        if rng.random() < 0.02:
            aim = tuple(rng.normal(size=2))
        events = []
        if rng.random() < 0.03:
            events.append(thrust(h.now, frontal=float(rng.random()), direction=aim))
        if rng.random() < 0.004:
            events.append(burst_event(h.now))
        manual = ManualInput(trigger_held=space, trigger_pressed=space and rng.random() < 0.05,
                             trigger_released=not space and rng.random() < 0.05,
                             aim=(1.0, 0.0) if rng.random() < 0.2 else (0.0, 0.0))
        roll = rng.random()
        if roll < 0.003:
            h.mgr.select(str(rng.choice(["ki_charge", "kamehameha", "energy_burst"])), h.now)
        elif roll < 0.005:
            h.mgr.toggle_aura(h.now)
        elif roll < 0.006:
            h.fx.clear()
        elif roll < 0.0065:
            h.mgr.reset(h.now)
        elif roll < 0.007:
            new = h.settings.copy()
            new.gestures.min_charge_time = float(rng.uniform(0.0, 1.5))
            new.gestures.burst_enabled = bool(rng.random() < 0.8)
            h.mgr.apply_settings(new)
            h.settings = new
        body = None
        if tracking:
            body = make_body(mid=(640.0 + 50 * rng.normal(), 420.0), gap=float(rng.uniform(0, 300)),
                             stability=float(rng.random()), energy=float(rng.random()),
                             both=rng.random() < 0.9)
        frame = None
        if tracking and rng.random() < 0.5:
            frame_id += 1
            frame = make_frame(frame_id, h.now, seg_id=frame_id)
        signals = GestureSignals(charge_pose_active=pose and tracking, tracking_ok=tracking,
                                 arms_extended=arms and tracking, aim_direction=aim,
                                 thrust=float(rng.random()) if arms else 0.0,
                                 aim_frontal=float(rng.random()))
        hud = h.step(body=body, signals=signals, events=events, manual=manual, frame=frame,
                     dt=float(rng.uniform(0.005, 0.06)))

        live = {id(x) for x in h.fx.all_handles() if x.alive and not x.finished}
        held = {id(x) for x in h.mgr.held_effects()}
        assert live == held
        assert sum(1 for o in h.fx.orbs if o.alive and not o.finished) <= 1
        for ability in charge_abilities:
            has_orb = any(x in h.fx.orbs for x in ability.held_effects())
            assert has_orb == (ability.sm.phase in (AttackPhase.PREPARING, AttackPhase.CHARGING))
        charging = any(a.sm.phase is AttackPhase.CHARGING for a in charge_abilities)
        assert ("charge_loop" in h.audio.active) == charging
        kame = h.ability("kamehameha")
        assert ("beam_loop" in h.audio.active) == (kame.sm.phase is AttackPhase.FIRING)
        assert ("aura_loop" in h.audio.active) == aura.active
        assert 0.0 <= hud.charge <= 1.0 and 0.0 <= hud.cooldown <= 1.0
        assert 0.0 <= hud.power_fraction <= 1.0 and 0.0 <= hud.callout_intensity <= 1.0
        assert 0.0 <= hud.aura_intensity <= 1.5
        assert hud.phase in {p.value for p in AttackPhase}
    # The walk really exercised the abilities.
    assert len(h.fx.beams) + len(h.fx.frontal_beams) >= 3
    assert h.fx.calls_named("ki_blast") and h.fx.calls_named("burst") and h.fx.auras
