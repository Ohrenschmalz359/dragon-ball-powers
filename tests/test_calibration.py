"""Tests for the guided calibration (calibration.py) with synthetic motion and the real engine."""

from __future__ import annotations

import pytest

from dragon_ball_powers.calibration import STEP_TITLES, CalibrationController
from dragon_ball_powers.config import Calibration, GestureSettings, load_calibration, save_calibration
from dragon_ball_powers.datatypes import BodyState, CalibrationHUD, GestureEventType, GestureSignals
from dragon_ball_powers.gestures import GestureEngine, compute_thresholds
from motion_synth import MotionSynth, SynthConfig, blend_pose, pose_charge, pose_frontal_push, ticks

E = GestureEventType


def drive(frames, controller: CalibrationController, engine: GestureEngine | None = None,
          display_hz: float | None = 60.0):
    """Feed frames through the engine into the controller; collect results and HUD states."""
    engine = engine or GestureEngine(GestureSettings())
    results: list[tuple[float, Calibration]] = []
    huds: list[tuple[float, CalibrationHUD]] = []
    for frame, now in ticks(frames, display_hz=display_hz):
        body, signals, _ = engine.update(frame, now)
        if controller.active:
            cal = controller.update(now, body, signals)
            if cal is not None:
                results.append((now, cal))
        huds.append((now, controller.hud(now)))
    return results, huds


def calibration_session(distance: float = 1.6, seed: int = 1, push: bool = True) -> MotionSynth:
    s = MotionSynth(SynthConfig(distance=distance, seed=seed))
    s.idle(3.5, stance="rest")         # step 1: relaxed stance
    s.charge_pose(3.0)                 # step 2: hands together like holding a ball
    if push:
        s.frontal_thrust(0.3, 1.0)     # step 3: push
    else:
        s.hold(6.0)
    s.idle(1.5)
    return s


@pytest.mark.parametrize("distance", [1.0, 1.6, 2.5])
def test_full_calibration_flow(distance):
    s = calibration_session(distance=distance, seed=3)
    ctrl = CalibrationController()
    start = s.frames[0].result_time
    ctrl.start(start)
    assert ctrl.active and ctrl.step == 1
    results, huds = drive(s.frames, ctrl)
    assert len(results) == 1                       # returned exactly once
    done_at, cal = results[0]
    assert not ctrl.active and ctrl.last_result is cal and ctrl.failure is None
    assert cal.valid and cal.created_at
    assert cal.scene_height == 720
    # body scale at calibration time = what the engine measured (S ~ 0.375 m at this distance)
    focal = s.focal
    assert cal.shoulder_width_px == pytest.approx(0.375 * focal / distance, rel=0.12)
    assert 0.3 < cal.together_distance < 0.75      # palms ~16 cm apart while "holding a ball"
    assert cal.thrust_peak > 0.5
    assert cal.rest_jitter >= 0.0
    if distance > 1.2:                             # at 1 m the hanging hands are out of the frame
        assert 0.15 < cal.hand_size_ratio < 0.45
        assert 0.0 < cal.rest_jitter < 0.4
        assert cal.torso_length_px > 0.0
    # HUD: steps 1 -> 2 -> 3 with their titles, progress within [0, 1], result message for 3 s
    steps = []
    for _, h in huds:
        if h.active:
            assert h.total_steps == 3 and h.title == STEP_TITLES[h.step] and h.instruction
            assert 0.0 <= h.progress <= 1.0
            if not steps or steps[-1] != h.step:
                steps.append(h.step)
    assert steps == [1, 2, 3]
    after = [(t, h) for t, h in huds if t >= done_at]
    assert after[0][1].result_message and "complete" in after[0][1].result_message.lower()
    assert all(h.result_message for t, h in after if t < done_at + 2.9)
    assert all(h.result_message is None for t, h in after if t > done_at + 3.1)
    assert ctrl.update(after[-1][0] + 0.1, None, GestureSignals()) is None   # inactive afterwards


def test_progress_is_monotonic_within_each_step():
    s = calibration_session(seed=4)
    ctrl = CalibrationController()
    ctrl.start(s.frames[0].result_time)
    _, huds = drive(s.frames, ctrl)
    last: dict[int, float] = {}
    for _, h in huds:
        if h.active:
            assert h.progress >= last.get(h.step, 0.0) - 1e-9
            last[h.step] = h.progress
    assert last[1] == pytest.approx(1.0, abs=0.05) and last[2] == pytest.approx(1.0, abs=0.05)


def test_step_one_restarts_while_nobody_is_tracked():
    ctrl = CalibrationController()
    sig = GestureSignals()
    ctrl.start(0.0)
    for i in range(1, 60):                 # 2 s of nothing: the step timer keeps restarting
        ctrl.update(i / 30.0, None, sig)
    assert ctrl.step == 1 and ctrl.hud(2.0).progress == 0.0
    assert "view" in ctrl.hud(2.0).instruction.lower()
    s = MotionSynth(SynthConfig(distance=1.6, seed=5, start_time=2.0))
    s.idle(3.0)
    engine = GestureEngine(GestureSettings())
    started_tracking = None
    for frame, now in ticks(s.frames, display_hz=None):
        body, signals, _ = engine.update(frame, now)
        ctrl.update(now, body, signals)
        if started_tracking is None and body is not None and body.quality != "none":
            started_tracking = now
        if ctrl.step == 2:
            break
    assert ctrl.step == 2
    assert now - started_tracking == pytest.approx(2.5, abs=0.1)   # 2.5 s of tracked samples


def test_abort_after_ten_seconds_without_tracking():
    ctrl = CalibrationController()
    ctrl.start(100.0)
    empty = BodyState(timestamp=0.0, scene_size=(1280, 720))       # quality "none"
    t = 100.0
    while ctrl.active and t < 115.0:
        t += 1.0 / 30.0
        ctrl.update(t, empty, GestureSignals())
    assert not ctrl.active and ctrl.failure == "no_tracking"
    assert t - 100.0 == pytest.approx(10.0, abs=0.1)
    hud = ctrl.hud(t + 1.0)
    assert not hud.active and hud.result_message and "10 seconds" in hud.result_message
    assert ctrl.hud(t + 3.5).result_message is None
    assert ctrl.last_result is None


def test_cancel():
    s = calibration_session(seed=6)
    ctrl = CalibrationController()
    ctrl.start(s.frames[0].result_time)
    engine = GestureEngine(GestureSettings())
    for frame, now in ticks(s.frames[:60], display_hz=None):
        body, signals, _ = engine.update(frame, now)
        ctrl.update(now, body, signals)
    assert ctrl.active
    ctrl.cancel()
    assert not ctrl.active and ctrl.failure == "cancelled"
    assert ctrl.update(now, body, signals) is None
    assert ctrl.hud(now + 1.0).result_message == "Calibration cancelled."
    assert ctrl.hud(now + 3.2).result_message is None
    ctrl.cancel()                              # cancelling twice is harmless
    # a new start clears the old message and begins at step 1
    ctrl.start(now + 4.0)
    assert ctrl.active and ctrl.step == 1 and ctrl.hud(now + 4.0).result_message is None


def test_no_push_gives_zero_thrust_peak():
    s = calibration_session(seed=7, push=False)
    ctrl = CalibrationController()
    ctrl.start(s.frames[0].result_time)
    results, huds = drive(s.frames, ctrl)
    assert len(results) == 1
    t_done, cal = results[0]
    assert cal.valid and cal.thrust_peak == 0.0
    assert "no push" in ctrl.hud(t_done).result_message.lower()
    # step 3 waited its full 5 s
    t3 = next(t for t, h in huds if h.active and h.step == 3)
    assert t_done - t3 == pytest.approx(5.0, abs=0.2)


def test_hanging_arms_do_not_count_as_hands_together():
    s = MotionSynth(SynthConfig(distance=2.5, seed=8))
    s.idle(12.0, stance="rest")                # never brings the hands together
    ctrl = CalibrationController(together_timeout=8.0)
    ctrl.start(s.frames[0].result_time)
    _, huds = drive(s.frames, ctrl)
    assert ctrl.failure == "together_timeout" and not ctrl.active
    assert not any(h.active and h.step == 3 for _, h in huds)
    msg = next(h.result_message for _, h in huds if h.result_message)
    assert "hands" in msg.lower()


def test_calibration_changes_thresholds_and_detection():
    settings = GestureSettings(sensitivity=0.5)
    base = compute_thresholds(settings)
    # a user whose natural "ball" is wide and whose push is gentle
    cal = Calibration(valid=True, shoulder_width_px=214.0, scene_height=720, hand_size_ratio=0.27,
                      together_distance=1.0, rest_jitter=0.1, thrust_peak=0.45, created_at="now")
    tuned = compute_thresholds(settings, cal)
    assert tuned.together == pytest.approx(1.6) and tuned.together > base.together
    assert tuned.thrust == pytest.approx(0.7 * 0.45) and tuned.thrust < base.thrust

    def run(frames, calibration):
        engine = GestureEngine(settings, calibration)
        events, active = [], 0
        for frame, now in ticks(frames, display_hz=None):
            _, sig, ev = engine.update(frame, now)
            events += ev
            active += sig.charge_pose_active
        return [e.type for e in events], active

    # wide ball (palms ~45 cm apart): no charge pose by default, accepted with the calibration
    s = MotionSynth(SynthConfig(distance=1.6, seed=9))
    s.idle(1.0)
    s.charge_pose(3.0, gap=0.45)
    ev_default, active_default = run(s.frames, None)
    ev_tuned, active_tuned = run(s.frames, cal)
    assert active_default == 0 and E.CHARGE_POSE_START not in ev_default
    assert active_tuned > 30 and E.CHARGE_POSE_START in ev_tuned
    # gentle push (60 % of a full push): ignored by default, a thrust for the calibrated user
    s = MotionSynth(SynthConfig(distance=1.6, seed=10))
    s.idle(1.0)
    s.charge_pose(1.5)
    s.move_to(blend_pose(pose_charge(), pose_frontal_push(), 0.6), 0.3, 1.0)
    assert run(s.frames, None)[0].count(E.THRUST) == 0
    assert run(s.frames, cal)[0].count(E.THRUST) == 1


def test_calibration_result_round_trips_through_config(tmp_config_dir):
    s = calibration_session(seed=11)
    ctrl = CalibrationController()
    ctrl.start(s.frames[0].result_time)
    results, _ = drive(s.frames, ctrl)
    cal = results[0][1]
    path = tmp_config_dir / "calibration.json"
    save_calibration(cal, path)
    loaded = load_calibration(path)
    assert loaded == cal
    engine = GestureEngine(GestureSettings(), loaded)
    assert engine.calibration is not None and engine.thresholds.thrust <= compute_thresholds(GestureSettings()).thrust


def test_works_with_plain_body_state_objects():
    """Integrators may feed BodyState objects that are not BodyFeatures (e.g. fakes)."""
    ctrl = CalibrationController(body_seconds=0.5, together_seconds=0.5, thrust_seconds=0.5)
    ctrl.start(0.0)
    body = BodyState(timestamp=0.0, scene_size=(1280, 720), quality="full", body_scale=200.0)
    body.left.present = body.right.present = True
    body.left.source = body.right.source = "hand"
    body.left.hand_size = body.right.hand_size = 52.0
    body.left.shoulder, body.right.shoulder = (540.0, 300.0), (740.0, 300.0)
    body.left.extension_3d = body.right.extension_3d = 0.7
    body.hand_mid, body.hand_distance, body.hands_in_front = (640.0, 420.0), 0.5, True
    body.torso_box = (540.0, 300.0, 740.0, 580.0)
    result = None
    t = 0.0
    while result is None and t < 5.0:
        t += 1.0 / 30.0
        sig = GestureSignals(thrust=0.8 if 1.2 < t < 1.4 else 0.05)
        result = ctrl.update(t, body, sig)
    assert result is not None and result.valid
    assert result.shoulder_width_px == pytest.approx(200.0)
    assert result.hand_size_ratio == pytest.approx(0.26)
    assert result.together_distance == pytest.approx(0.5)
    assert result.thrust_peak == pytest.approx(0.8)
