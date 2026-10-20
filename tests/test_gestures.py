"""Tests for the gesture engine (features, detectors, events) with synthetic motion.

The sequences come from ``tests/motion_synth.py``: a physically modelled virtual person
with MediaPipe-like landmarks, tracker noise, dropouts and coasting hands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from dragon_ball_powers.config import Calibration, GestureSettings
from dragon_ball_powers.datatypes import (
    BodyState,
    GestureEvent,
    GestureEventType,
    GestureSignals,
    HandObservation,
    PoseObservation,
    TrackingFrame,
)
from dragon_ball_powers.gestures import (
    BodyFeatures,
    GestureEngine,
    compute_thresholds,
    hand_openness,
    hand_scale,
    palm_size,
)
from motion_synth import (
    MotionSynth,
    SynthConfig,
    blend_pose,
    canonical_hand,
    hand_rotation,
    offset_pose,
    pose_charge,
    pose_frontal_push,
    ticks,
)

E = GestureEventType


# ------------------------------------------------------------------ helpers


@dataclass
class Run:
    """Everything the engine produced for a frame sequence."""

    bodies: list[BodyState | None] = field(default_factory=list)
    signals: list[GestureSignals] = field(default_factory=list)
    events: list[GestureEvent] = field(default_factory=list)
    frame_ticks: list[bool] = field(default_factory=list)

    def of(self, kind: GestureEventType) -> list[GestureEvent]:
        return [e for e in self.events if e.type is kind]

    def count(self, kind: GestureEventType) -> int:
        return len(self.of(kind))

    def peak(self, name: str) -> float:
        return max((getattr(s, name) for s in self.signals), default=0.0)

    def frame_signals(self) -> list[GestureSignals]:
        return [s for s, f in zip(self.signals, self.frame_ticks) if f]


def run(frames, settings: GestureSettings | None = None, calibration: Calibration | None = None,
        display_hz: float | None = 60.0, engine: GestureEngine | None = None) -> Run:
    engine = engine or GestureEngine(settings or GestureSettings(), calibration)
    out = Run()
    for frame, now in ticks(frames, display_hz=display_hz):
        body, sig, events = engine.update(frame, now)
        out.bodies.append(body)
        out.signals.append(sig)
        out.events.extend(events)
        out.frame_ticks.append(frame is not None)
    return out


def synth(distance: float = 1.6, seed: int = 1, **kw) -> MotionSynth:
    return MotionSynth(SynthConfig(distance=distance, seed=seed, **kw))


def unit(v):
    n = math.hypot(*v)
    return (v[0] / n, v[1] / n)


SEEDS = (1, 2, 3)
DISTANCES = (1.0, 1.6, 2.5)


# ------------------------------------------------------------------ thresholds


def test_thresholds_move_monotonically_with_sensitivity():
    values = [compute_thresholds(GestureSettings(sensitivity=s)) for s in np.linspace(0.0, 1.0, 11)]
    for a, b in zip(values, values[1:]):
        assert b.together > a.together           # more forgiving "hands together"
        assert b.charge_on < a.charge_on
        assert b.charge_off < a.charge_off
        assert b.thrust < a.thrust
        assert b.burst_rate < a.burst_rate
        assert b.burst_score < a.burst_score
    mid = compute_thresholds(GestureSettings(sensitivity=0.5))
    assert mid.together == pytest.approx(1.1 * 1.05)
    assert mid.charge_on == pytest.approx(0.525) and mid.charge_off == pytest.approx(0.275)
    assert mid.thrust == pytest.approx(0.56)
    assert mid.burst_score == pytest.approx(0.625)


def test_calibration_changes_thresholds():
    settings = GestureSettings(sensitivity=0.5)
    base = compute_thresholds(settings)
    cal = Calibration(valid=True, together_distance=0.9, thrust_peak=0.5)
    t = compute_thresholds(settings, cal)
    assert t.together == pytest.approx(max(base.together, 1.6 * 0.9))
    assert t.thrust == pytest.approx(min(base.thrust, 0.7 * 0.5))
    # an invalid calibration is ignored, a tiny thrust peak cannot make thrusts fire on noise
    assert compute_thresholds(settings, Calibration(valid=False, thrust_peak=0.4)).thrust == base.thrust
    assert compute_thresholds(settings, Calibration(valid=True, thrust_peak=0.31)).thrust >= 0.25
    # no calibrated push (thrust_peak = 0) keeps the default threshold
    assert compute_thresholds(settings, Calibration(valid=True, thrust_peak=0.0)).thrust == base.thrust


# ------------------------------------------------------------------ features


def _hand_obs(side: str, centre=(640.0, 360.0), px_per_m: float = 1000.0, openness: float = 0.8,
              fingers=(0.0, -1.0, 0.0), palm=(0.0, 0.0, -1.0)) -> HandObservation:
    local = canonical_hand(openness)
    if side == "right":
        local[:, 0] = -local[:, 0]
    pts3 = local @ hand_rotation(fingers, palm).T
    xy = pts3[:, :2] * px_per_m + np.asarray(centre)
    lm = np.column_stack([xy, pts3[:, 2] * px_per_m]).astype(np.float32)
    return HandObservation(side=side, landmarks=lm, world=(pts3 - pts3.mean(axis=0)).astype(np.float32),
                           handedness=side.title(), score=0.95)


def test_palm_size_openness_and_rotation_invariant_scale():
    open_hand = _hand_obs("left", openness=1.0)
    fist = _hand_obs("left", openness=0.0)
    lm_open = open_hand.landmarks.astype(float)
    assert palm_size(lm_open) == pytest.approx(95.0, rel=0.02)          # 9.5 cm palm at 1000 px/m
    assert hand_openness(lm_open, palm_size(lm_open)) > 0.9
    assert hand_openness(fist.landmarks.astype(float), palm_size(fist.landmarks.astype(float))) < 0.1
    # turning the hand changes the 2D palm size a lot, the fitted scale stays put
    turned = _hand_obs("left", fingers=(0.0, -0.3, -0.95), palm=(1.0, 0.0, 0.0))
    lm_turned = turned.landmarks.astype(float)
    assert palm_size(lm_turned) < 0.85 * palm_size(lm_open)
    assert hand_scale(lm_open, open_hand.world) == pytest.approx(1000.0, rel=0.02)
    assert hand_scale(lm_turned, turned.world) == pytest.approx(1000.0, rel=0.02)
    assert hand_scale(lm_open, None) is None
    bad_world = np.full((21, 3), np.nan, dtype=np.float32)
    assert hand_scale(lm_open, bad_world) is None


def test_body_features_on_synthetic_person():
    s = synth(distance=1.6, noise=0.0, sway=0.0)
    frames = s.charge_pose(1.0, transition_s=0.3)
    r = run(frames, display_hz=None)
    body = r.bodies[-1]
    assert isinstance(body, BodyFeatures)
    assert body.quality in ("full", "pose")
    # S = max(shoulder width, 0.75 x torso) of the virtual skeleton (0.34 m / 0.375 m at 1.6 m)
    expected_s = 0.75 * 0.50 * s.focal / 1.6
    assert body.body_scale == pytest.approx(expected_s, rel=0.06)
    assert body.shoulder_width_px == pytest.approx(0.34 * s.focal / 1.6, rel=0.06)
    assert body.torso_length_px > 0.0
    assert body.both_hands and body.hands_in_front
    assert 0.3 < body.hand_distance < 0.8
    assert body.hand_mid is not None and body.torso_box is not None
    for arm in (body.left, body.right):
        assert arm.present and arm.source in ("hand", "pose")
        assert arm.extension_3d is not None and 0.65 < arm.extension_3d < 0.85    # bent arms
        assert arm.forward > 0.3                                                  # forearms point forward
        assert arm.speed < 0.5
    assert body.left.hand_center[0] < body.right.hand_center[0]                  # screen sides
    assert r.signals[-1].charge_pose_active and r.signals[-1].tracking_ok


@pytest.mark.parametrize("model_left_is_screen_left", [False, True])
def test_screen_sides_independent_of_pose_label_convention(model_left_is_screen_left):
    s = synth(distance=1.6, noise=0.0, sway=0.0, model_left_is_screen_left=model_left_is_screen_left)
    s.idle(0.5)
    frames = s.one_hand(1.0, side="right")
    body = run(frames, display_hz=None).bodies[-1]
    # the raised right hand is bent and forward, the left arm hangs straight
    assert body.right.extension_3d < 0.85 and body.left.extension_3d > 0.9
    assert body.right.hand_center[0] > body.left.hand_center[0]


def test_motion_summary_energy_and_stability():
    s = synth(distance=1.6, seed=5)
    s.charge_pose(2.0)
    still = run(s.frames, display_hz=None)
    body = still.bodies[-1]
    assert body.motion.dominant == "still"
    assert body.stability > 0.6 and body.energy < 0.25
    # sweep both hands quickly to the screen right
    s2 = synth(distance=1.6, seed=5)
    s2.charge_pose(1.0)
    s2.move_to(offset_pose(pose_charge(), (0.3, 0.0, 0.0), (0.3, 0.0, 0.0)), 0.4)
    r = run(s2.frames, display_hz=None)
    peak_right = max(b.motion.right for b in r.bodies if b is not None)
    assert peak_right > 1.0
    assert any(b.motion.dominant == "right" for b in r.bodies if b is not None)
    assert max(b.energy for b in r.bodies if b is not None) > body.energy + 0.1


# ------------------------------------------------------------------ charge pose


def test_charge_pose_start_end_with_hysteresis():
    s = synth(distance=1.6, seed=2)
    s.idle(1.0)
    s.charge_pose(2.0)
    s.separate_hands(0.9, 2.5, 0.5)     # slowly widen the "ball" until it is no charge pose any more
    r = run(s.frames, display_hz=None)
    th = compute_thresholds(GestureSettings())
    assert r.count(E.CHARGE_POSE_START) == 1 and r.count(E.CHARGE_POSE_END) == 1
    start, end = r.of(E.CHARGE_POSE_START)[0], r.of(E.CHARGE_POSE_END)[0]
    assert start.time < end.time
    assert start.strength > th.charge_on and end.strength < th.charge_off
    # hysteresis: while the score is between the thresholds the state never changes
    sigs = r.frame_signals()
    for prev, cur in zip(sigs, sigs[1:]):
        if th.charge_off <= cur.charge_pose <= th.charge_on:
            assert cur.charge_pose_active == prev.charge_pose_active
    in_band = [x for x in sigs if th.charge_off <= x.charge_pose <= th.charge_on and x.charge_pose_active]
    assert in_band, "the slow separation passes through the hysteresis band while active"


@pytest.mark.parametrize("where", ["chest", "hip_left", "hip_right"])
def test_charge_pose_variants_are_recognised(where):
    s = synth(distance=1.6, seed=3)
    s.idle(1.0)
    s.charge_pose(2.0, where=where)
    r = run(s.frames)
    assert r.count(E.CHARGE_POSE_START) == 1
    assert r.signals[-1].charge_pose_active


@pytest.mark.parametrize("distance", DISTANCES)
def test_overlapping_hands_keep_the_charge_pose(distance):
    s = synth(distance=distance, seed=4)
    s.idle(1.0)
    s.charge_pose(0.8)
    with s.hands_overlapping("left"):
        s.hold(3.0)
    r = run(s.frames, display_hz=None)
    tail = r.bodies[-60:]
    # the hidden hand is covered by the pose fallback (small, far hands may also drop out
    # of the hand model now and then, then the pose stands in for them as well)
    assert all(b.left.present and b.left.source == "pose" for b in tail)
    assert all(b.right.present for b in tail)
    assert sum(b.right.source == "hand" for b in tail) > 30
    assert all(sig.charge_pose_active for sig in r.signals[-60:])
    assert r.count(E.CHARGE_POSE_END) == 0
    assert r.count(E.THRUST) == 0 and r.count(E.BURST) == 0


@pytest.mark.parametrize("distance", DISTANCES)
def test_single_hand_is_no_charge_pose(distance):
    # at 1 m the hanging hand is out of the frame, further away the pose sees the hanging wrist
    s = synth(distance=distance, seed=6)
    s.idle(1.0, stance="rest")
    s.one_hand(3.0, side="right")
    r = run(s.frames)
    assert r.count(E.CHARGE_POSE_START) == 0
    assert r.peak("charge_pose") < 0.3


def test_extended_arms_after_firing_are_no_charge_pose():
    s = synth(distance=1.6, seed=7)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.frontal_thrust(0.3, 1.5)
    r = run(s.frames, display_hz=None)
    assert r.count(E.CHARGE_POSE_END) == 1
    assert not r.signals[-1].charge_pose_active
    assert r.signals[-1].charge_pose < 0.1
    assert r.signals[-1].arms_extended


# ------------------------------------------------------------------ thrust


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("seed", SEEDS)
def test_frontal_thrust_fires_exactly_once(distance, seed):
    s = synth(distance=distance, seed=seed)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.frontal_thrust(0.3, 1.0)
    s.idle(1.0)
    r = run(s.frames)
    thrusts = r.of(E.THRUST)
    assert len(thrusts) == 1, [(e.time, e.frontal) for e in thrusts]
    ev = thrusts[0]
    assert ev.frontal > 0.6
    assert 0.0 < ev.strength <= 1.0 and ev.position is not None
    assert r.count(E.BURST) == 0
    # the aim follows the arms towards the camera
    assert max(sig.aim_frontal for sig in r.signals) > 0.6


@pytest.mark.parametrize("direction", [(1.0, 0.0), (-1.0, 0.0), (1.0, -0.5)])
@pytest.mark.parametrize("seed", SEEDS)
def test_lateral_thrust_fires_once_in_the_push_direction(direction, seed):
    s = synth(distance=1.6, seed=seed)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.lateral_thrust(direction, 0.3, 1.0)
    s.idle(1.0)
    r = run(s.frames)
    thrusts = r.of(E.THRUST)
    assert len(thrusts) == 1, [(e.time, e.frontal, e.direction) for e in thrusts]
    ev = thrusts[0]
    d = unit(direction)
    assert ev.direction[0] * d[0] + ev.direction[1] * d[1] > 0.8
    assert ev.frontal < 0.5
    assert r.count(E.BURST) == 0


@pytest.mark.parametrize("where,direction", [("hip_left", (1.0, -0.3)), ("hip_right", (-1.0, -0.3))])
@pytest.mark.parametrize("distance", DISTANCES)
def test_kamehameha_from_the_hip(where, direction, distance):
    s = synth(distance=distance, seed=8)
    s.idle(1.0)
    s.charge_pose(1.5, where=where)
    s.lateral_thrust(direction, 0.3, 1.0)
    r = run(s.frames)
    thrusts = r.of(E.THRUST)
    assert len(thrusts) == 1
    d = unit(direction)
    assert thrusts[0].direction[0] * d[0] + thrusts[0].direction[1] * d[1] > 0.8
    assert thrusts[0].frontal < 0.5


def test_aim_follows_a_sustained_lateral_beam():
    s = synth(distance=1.6, seed=9)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.lateral_thrust((1.0, 0.0), 0.3, 1.5)
    r = run(s.frames, display_hz=None)
    sig = r.signals[-1]
    assert sig.arms_extended
    assert sig.aim_direction[0] > 0.8 and sig.aim_frontal < 0.5


@pytest.mark.parametrize("arms", ["keep", "swing"])
@pytest.mark.parametrize("seed", SEEDS)
def test_walking_towards_the_camera_is_no_thrust(arms, seed):
    s = synth(distance=2.8, seed=seed)
    s.idle(1.0, stance="ready" if arms == "keep" else "rest")
    if arms == "keep":
        s.charge_pose(1.0)          # hands held together while walking: the hardest case
    s.walk_towards_camera(1.2, 2.5, arms=arms)
    s.hold(1.0)
    r = run(s.frames)
    assert r.count(E.THRUST) == 0
    assert r.count(E.BURST) == 0
    assert r.peak("thrust") < 0.3


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fidgeting_with_hands_together_fires_nothing(distance, seed):
    s = synth(distance=distance, seed=seed)
    s.idle(1.0)
    s.charge_pose(0.5)
    s.random_fidget(10.0, hands_together=True)
    r = run(s.frames)
    assert r.count(E.THRUST) == 0
    assert r.count(E.BURST) == 0
    assert r.peak("thrust") < 0.5


@pytest.mark.parametrize("distance", DISTANCES)
def test_noisy_idle_fires_nothing(distance):
    s = synth(distance=distance, seed=11, noise=2.0, hand_dropout=0.05)
    s.idle(3.0)
    s.random_fidget(7.0, hands_together=False)
    r = run(s.frames)
    for kind in (E.THRUST, E.BURST, E.CHARGE_POSE_START):
        assert r.count(kind) == 0


@pytest.mark.parametrize("distance", DISTANCES)
def test_lowering_the_arms_after_charging_is_no_thrust(distance):
    for seed in SEEDS:
        s = synth(distance=distance, seed=seed)
        s.idle(1.0)
        s.charge_pose(1.5)
        s.idle(1.5, stance="rest", transition_s=0.3)
        r = run(s.frames)
        assert r.count(E.THRUST) == 0 and r.count(E.BURST) == 0


def test_one_thrust_per_push_with_refractory_and_rearm():
    s = synth(distance=1.6, seed=12)
    s.idle(1.0)
    for _ in range(2):
        s.charge_pose(1.2)
        s.frontal_thrust(0.3, 0.8)
    r = run(s.frames)
    thrusts = r.of(E.THRUST)
    assert len(thrusts) == 2
    assert thrusts[1].time - thrusts[0].time > 0.8


# ------------------------------------------------------------------ burst


@pytest.mark.parametrize("distance", DISTANCES)
@pytest.mark.parametrize("seed", SEEDS)
def test_burst_fires_exactly_once(distance, seed):
    s = synth(distance=distance, seed=seed)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.burst(0.25, 0.8)
    s.idle(1.0)
    r = run(s.frames)
    bursts = r.of(E.BURST)
    assert len(bursts) == 1
    assert r.count(E.THRUST) == 0
    ev = bursts[0]
    assert ev.strength > 0.6 and ev.position is not None
    assert abs(ev.direction[0]) > 0.8        # the hands spread horizontally


def test_burst_disabled_in_settings():
    s = synth(distance=1.6, seed=1)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.burst(0.25, 0.8)
    r = run(s.frames, settings=GestureSettings(burst_enabled=False))
    assert r.count(E.BURST) == 0
    assert r.peak("burst") > 0.6            # the score is still reported


@pytest.mark.parametrize("distance", DISTANCES)
def test_slowly_separating_hands_while_charging_is_no_burst(distance):
    for seed in SEEDS:
        s = synth(distance=distance, seed=seed)
        s.idle(1.0)
        s.charge_pose(1.0)
        s.separate_hands(0.85, 2.0, 1.0)
        r = run(s.frames)
        assert r.count(E.BURST) == 0 and r.peak("burst") < 0.2
        assert r.count(E.THRUST) == 0


@pytest.mark.parametrize("distance", DISTANCES)
def test_relaxing_the_hands_after_charging_is_no_burst(distance):
    for seed in SEEDS:
        s = synth(distance=distance, seed=seed)
        s.idle(1.0)
        s.charge_pose(1.5)
        s.idle(1.5, transition_s=0.5)       # hands drop apart into a relaxed ready stance
        r = run(s.frames)
        assert r.count(E.BURST) == 0 and r.count(E.THRUST) == 0


# ------------------------------------------------------------------ tracking


@pytest.mark.parametrize("distance", DISTANCES)
def test_dropout_reports_lost_then_found_without_spurious_events(distance):
    s = synth(distance=distance, seed=13)
    s.idle(1.0)
    s.charge_pose(1.0)
    dropout_start = s.time
    s.tracking_dropout(1.0)
    s.hold(1.5)
    s.idle(1.0)
    r = run(s.frames)
    kinds = [e.type for e in r.events if e.type in (E.TRACKING_LOST, E.TRACKING_FOUND)]
    assert kinds == [E.TRACKING_FOUND, E.TRACKING_LOST, E.TRACKING_FOUND]
    lost = r.of(E.TRACKING_LOST)[0]
    # lost hands coast for 0.25 s, TRACKING_LOST follows 0.35 s after the last hand (+ latency)
    assert 0.55 < lost.time - dropout_start < 0.8
    assert r.count(E.THRUST) == 0 and r.count(E.BURST) == 0
    assert any(not sig.tracking_ok for sig in r.signals)
    assert r.signals[-1].tracking_ok


def test_update_none_advances_timeouts():
    s = synth(distance=1.6, seed=14)
    s.idle(0.5)
    frames = s.charge_pose(1.0)
    engine = GestureEngine(GestureSettings())
    body = sig = None
    for f in frames:
        body, sig, _ = engine.update(f, f.result_time)
    assert sig.charge_pose_active and sig.tracking_ok
    last = frames[-1].result_time
    # the tracker stalls: no new frames
    b2, s2, ev = engine.update(None, last + 0.2)
    assert ev == [] and b2 is body and s2.charge_pose_active and s2.tracking_ok
    b3, s3, ev = engine.update(None, last + 0.4)
    assert [e.type for e in ev] == [E.CHARGE_POSE_END, E.TRACKING_LOST]
    assert not s3.tracking_ok and not s3.charge_pose_active and s3.charge_pose == 0.0
    assert b3 is body                                  # the last body state is still returned
    _, _, ev = engine.update(None, last + 1.0)
    assert ev == []                                    # reported once
    # tracking resumes
    later = synth(distance=1.6, seed=14, start_time=last + 1.1)
    later.pose = s.pose
    frames2 = later.hold(0.5)
    events = []
    for f in frames2:
        events += engine.update(f, f.result_time)[2]
    assert events[0].type is E.TRACKING_FOUND
    assert any(e.type is E.CHARGE_POSE_START for e in events)


def test_update_none_before_any_frame():
    engine = GestureEngine(GestureSettings())
    body, sig, ev = engine.update(None, 123.0)
    assert body is None and ev == [] and not sig.tracking_ok


def test_display_rate_none_ticks_do_not_change_detection():
    s = synth(distance=1.6, seed=15)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.frontal_thrust(0.3, 1.0)
    without = run(s.frames, display_hz=None)
    with_ticks = run(s.frames, display_hz=144.0)
    assert [e.type for e in without.events] == [e.type for e in with_ticks.events]


@pytest.mark.parametrize("seed", SEEDS)
def test_harsh_conditions_still_work(seed):
    """Twice the tracker jitter and frequent hand-model dropouts."""
    cfg = dict(noise=2.0, hand_dropout=0.1)
    s = synth(distance=1.6, seed=seed, **cfg)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.frontal_thrust(0.3, 1.0)
    s.charge_pose(1.5)
    s.burst(0.25, 0.8)
    r = run(s.frames)
    assert r.count(E.THRUST) == 1 and r.of(E.THRUST)[0].frontal > 0.6
    assert r.count(E.BURST) == 1


# ------------------------------------------------------------------ sensitivity / invariance


def _counts_by_sensitivity(frames, kind, values=(0.0, 0.25, 0.5, 0.75, 1.0)):
    return [run(frames, settings=GestureSettings(sensitivity=v), display_hz=None).count(kind) for v in values]


def _active_fraction_by_sensitivity(frames, values=(0.0, 0.25, 0.5, 0.75, 1.0)):
    out = []
    for v in values:
        sigs = run(frames, settings=GestureSettings(sensitivity=v), display_hz=None).signals
        out.append(sum(x.charge_pose_active for x in sigs) / len(sigs))
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_sensitivity_monotonicity(seed):
    # a 70 % push: not decisive enough for strict settings, accepted by forgiving ones
    s = synth(distance=1.6, seed=seed)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.move_to(blend_pose(pose_charge(), pose_frontal_push(), 0.7), 0.3, 1.0)
    thrusts = _counts_by_sensitivity(s.frames, E.THRUST)
    assert thrusts == sorted(thrusts), thrusts
    assert thrusts[0] == 0 and thrusts[-1] == 1
    # a slow, gentle burst
    s = synth(distance=1.6, seed=seed)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.burst(0.9, 0.8)
    bursts = _counts_by_sensitivity(s.frames, E.BURST)
    assert bursts == sorted(bursts), bursts
    assert bursts[0] == 0
    # a wide "ball": the hands count as together only for forgiving settings
    s = synth(distance=1.6, seed=seed)
    s.idle(1.0)
    s.charge_pose(3.0, gap=0.4)
    active = _active_fraction_by_sensitivity(s.frames)
    assert active == sorted(active), active
    assert active[0] == 0.0 and active[-1] > 0.5


def test_body_scale_invariance_near_and_far():
    results = {}
    for distance in (1.0, 2.5):
        s = synth(distance=distance, seed=16)
        s.idle(1.0)
        s.charge_pose(1.5)
        s.frontal_thrust(0.3, 1.0)
        s.charge_pose(1.5)
        s.lateral_thrust((1.0, 0.0), 0.3, 1.0)
        s.charge_pose(1.5)
        s.burst(0.25, 0.8)
        r = run(s.frames, display_hz=None)
        charge_d = [b.hand_distance for b, sig in zip(r.bodies, r.signals) if sig.charge_pose_active]
        results[distance] = (r, float(np.median(charge_d)))
    near, far = results[1.0][0], results[2.5][0]
    for r in (near, far):
        thrusts = r.of(E.THRUST)
        assert [e.frontal > 0.5 for e in thrusts] == [True, False]
        assert r.count(E.BURST) == 1
        assert r.count(E.CHARGE_POSE_START) == 3
    # the body scale differs by ~2.5x, normalised distances stay comparable
    s_near = np.median([b.body_scale for b in near.bodies])
    s_far = np.median([b.body_scale for b in far.bodies])
    assert 2.0 < s_near / s_far < 3.0
    assert results[1.0][1] == pytest.approx(results[2.5][1], rel=0.4)


@pytest.mark.parametrize("body_scale", [0.85, 1.15])
def test_smaller_and_larger_people(body_scale):
    s = synth(distance=1.8, seed=17, body_scale=body_scale)
    s.idle(1.0)
    s.charge_pose(1.5)
    s.frontal_thrust(0.3, 1.0)
    s.charge_pose(1.5)
    s.burst(0.25, 0.8)
    r = run(s.frames)
    assert r.count(E.THRUST) == 1 and r.of(E.THRUST)[0].frontal > 0.6
    assert r.count(E.BURST) == 1


# ------------------------------------------------------------------ robustness / API


def test_malformed_frames_never_raise():
    engine = GestureEngine(GestureSettings())
    good = synth(distance=1.6, seed=18).charge_pose(0.5)
    t = good[-1].capture_time
    frames = []
    nan_hand = HandObservation("left", np.full((21, 3), np.nan, dtype=np.float32))
    short_hand = HandObservation("right", np.zeros((5, 2), dtype=np.float32))
    bad_pose = PoseObservation(np.zeros((10, 3), dtype=np.float32), np.ones(10, dtype=np.float32))
    nan_pose = PoseObservation(np.full((33, 3), np.nan, dtype=np.float32), np.ones(33, dtype=np.float32),
                               world=np.full((33, 3), np.nan, dtype=np.float32))
    for i, (hands, pose) in enumerate([([nan_hand, short_hand], None), ([], bad_pose), ([], nan_pose),
                                      ([nan_hand], nan_pose), ([], None)]):
        frames.append(TrackingFrame(frame_id=1000 + i, capture_time=t + 0.033 * (i + 1),
                                    result_time=t + 0.033 * (i + 1) + 0.03, scene_size=(1280, 720), image=None,
                                    hands=hands, pose=pose))
    for f in good + frames:
        body, sig, ev = engine.update(f, f.result_time)
        assert isinstance(sig, GestureSignals)
        assert all(isinstance(e, GestureEvent) for e in ev)
    # duplicate frame, timestamps going backwards, scene size change
    engine.update(good[-1], good[-1].result_time + 1.0)
    engine.update(good[0], good[0].result_time)
    resized = TrackingFrame(frame_id=5000, capture_time=t + 2.0, result_time=t + 2.03, scene_size=(640, 360),
                            image=None, hands=[], pose=None)
    body, sig, _ = engine.update(resized, resized.result_time)
    assert body is not None and body.scene_size == (640, 360) and body.quality == "none"
    # a broken frame size (tracker bug) must not break the engine either
    hands = good[-1].hands
    broken = TrackingFrame(frame_id=5001, capture_time=t + 2.1, result_time=t + 2.13, scene_size=(0, 0),
                           image=None, hands=hands, pose=good[-1].pose)
    body, sig, _ = engine.update(broken, broken.result_time)
    assert body is not None and body.body_scale > 0.0


def test_hands_only_without_pose():
    s = synth(distance=1.4, seed=19, pose_dropout=1.0)   # the pose model never finds the person
    s.idle(1.0)
    s.charge_pose(2.0)
    r = run(s.frames, display_hz=None)
    body = r.bodies[-1]
    assert body.quality == "hands" and body.torso_box is None
    assert body.scale_source in ("hands", "held")
    assert body.both_hands and body.hands_in_front
    assert r.signals[-1].charge_pose_active and r.signals[-1].tracking_ok


def test_set_settings_calibration_and_reset():
    engine = GestureEngine(GestureSettings(sensitivity=0.2))
    t_low = engine.thresholds.thrust
    engine.set_settings(GestureSettings(sensitivity=0.9))
    assert engine.thresholds.thrust < t_low
    engine.set_calibration(Calibration(valid=True, together_distance=1.0, thrust_peak=0.4))
    assert engine.thresholds.together == pytest.approx(max(1.1 * 1.21, 1.6))
    assert engine.calibration is not None
    engine.set_calibration(Calibration(valid=False))
    assert engine.calibration is None
    frames = synth(distance=1.6, seed=20).charge_pose(1.0)
    for f in frames:
        engine.update(f, f.result_time)
    engine.reset()
    body, sig, ev = engine.update(None, frames[-1].result_time + 0.01)
    assert body is None and ev == [] and not sig.charge_pose_active


def test_engine_is_fast_enough():
    """Informational timing (generous bound; many tests run concurrently)."""
    import time

    s = synth(distance=1.6, seed=21)
    s.idle(1.0)
    s.charge_pose(1.0)
    s.random_fidget(4.0)
    engine = GestureEngine(GestureSettings())
    t0 = time.perf_counter()
    for f in s.frames:
        engine.update(f, f.result_time)
    per_frame_ms = (time.perf_counter() - t0) / len(s.frames) * 1000.0
    print(f"gesture update: {per_frame_ms:.3f} ms/frame")
    assert per_frame_ms < 10.0
