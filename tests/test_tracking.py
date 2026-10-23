"""Tests for tracking.py: slot assignment, smoothing, coasting, the tracker thread.

Most tests use synthetic detections or fake models (no MediaPipe). Tests marked
``mediapipe`` run the real models on the MediaPipe documentation sample photos in
/tmp/dbp-samples (skipped when models or samples are missing).
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from dragon_ball_powers.config import TrackingSettings
from dragon_ball_powers.datatypes import CameraFrame, TrackingFrame
from dragon_ball_powers.tracking import (
    COAST_DAMPING,
    HAND_MODEL_FILE,
    MAX_COAST_SPEED,
    SETUP_HINT,
    HandCandidate,
    HandDetection,
    HandPoseSmoother,
    HandPoseTracker,
    HandSlotAssigner,
    ModelConfig,
    PoseDetection,
    _parse_hands,
    _parse_pose,
    next_timestamp_ms,
    pose_model_file,
    prepare_model_input,
)

W, H = 1280, 720
FPS = 30.0
DT = 1.0 / FPS
SAMPLES = Path("/tmp/dbp-samples")


@pytest.fixture(autouse=True)
def _stop_all_trackers(monkeypatch):
    """Stop every tracker a test created, even when the test fails half-way."""
    created: list[HandPoseTracker] = []
    original_init = HandPoseTracker.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    monkeypatch.setattr(HandPoseTracker, "__init__", init)
    yield
    for tracker in created:
        tracker.stop()


# --------------------------------------------------------------- synthetic data

# Unit hand shape (x right, y down), fingers pointing up; shifted so that the palm
# centre (mean of wrist + finger MCPs) is the origin.
_HAND = np.array([
    (0.0, 1.0),                                                    # wrist
    (-0.35, 0.75), (-0.6, 0.45), (-0.8, 0.2), (-0.95, 0.0),        # thumb
    (-0.35, -0.1), (-0.4, -0.55), (-0.42, -0.85), (-0.44, -1.1),   # index
    (-0.05, -0.15), (-0.05, -0.65), (-0.05, -0.95), (-0.05, -1.25),  # middle
    (0.22, -0.1), (0.25, -0.55), (0.27, -0.85), (0.29, -1.1),      # ring
    (0.45, 0.0), (0.52, -0.35), (0.56, -0.6), (0.6, -0.8),         # pinky
])
_HAND = _HAND - _HAND[[0, 5, 9, 13, 17]].mean(axis=0)


def make_hand(cx: float, cy: float, label: str = "", score: float = 0.95, size: float = 50.0,
              frame: tuple[int, int] = (W, H)) -> HandDetection:
    """A plausible hand whose palm centre is at (cx, cy) scene px."""
    w, h = frame
    px = _HAND * size + (cx, cy)
    lm = np.zeros((21, 3))
    lm[:, 0] = px[:, 0] / w
    lm[:, 1] = px[:, 1] / h
    lm[:, 2] = np.linspace(0.0, -0.05, 21)
    world = np.column_stack([_HAND * 0.04, np.zeros(21)])
    return HandDetection(lm, world, label, score)


def make_pose(l_shoulder_x: float, r_shoulder_x: float, vis: float = 0.98,
              y: float = 0.45) -> PoseDetection:
    """Pose with the model's LEFT/RIGHT shoulder at the given normalised x."""
    lm = np.zeros((33, 3))
    lm[:, 0] = 0.5
    lm[:, 1] = np.linspace(0.2, 0.95, 33)
    lm[11, :2] = (l_shoulder_x, y)
    lm[12, :2] = (r_shoulder_x, y)
    world = np.zeros((33, 3))
    world[:, 1] = np.linspace(-0.6, 0.8, 33)
    return PoseDetection(lm, np.full(33, vis), world)


def cand(x: float, y: float = 400.0, label: str = "", score: float = 0.95,
         wrist_x: float | None = None) -> HandCandidate:
    return HandCandidate((x, y), x if wrist_x is None else wrist_x, label, score)


def sides_of(assignments) -> dict[int, str]:
    return {a.index: a.side for a in assignments}


# ------------------------------------------------------------- slot assignment


def test_two_hands_sorted_by_wrist_x_without_history():
    a = HandSlotAssigner(0.55)
    # index 0 is on the right; palm centres would sort the other way: wrist x decides
    result = a.assign([cand(700, wrist_x=760), cand(720, wrist_x=600)], 0.0, W)
    assert sides_of(result) == {0: "right", 1: "left"}
    assert [r.side for r in result] == ["left", "right"]       # sorted output
    assert not any(r.continuous for r in result)                # both hands are new
    result = a.assign([cand(705, wrist_x=765), cand(722, wrist_x=602)], DT, W)
    assert sides_of(result) == {0: "right", 1: "left"}
    assert all(r.continuous for r in result)


def test_crossing_hands_keep_their_slots():
    """Both slots recent: minimum total distance (hysteresis) instead of x order."""
    a = HandSlotAssigner(0.55)
    a.assign([cand(500, 300), cand(780, 460)], 0.0, W)
    t = 0.0
    for k in range(1, 21):            # the hands pass each other (different heights) at k = 10
        t += DT
        left_hand, right_hand = cand(500 + 14 * k, 300), cand(780 - 14 * k, 460)
        result = a.assign([right_hand, left_hand], t, W)  # detector order is arbitrary
        assert sides_of(result) == {1: "left", 0: "right"}, k
        assert all(r.continuous for r in result)
    assert a.last_position("left")[0] > a.last_position("right")[0]   # crossed now


def test_touching_hands_with_noise_do_not_swap():
    rng = np.random.default_rng(2)
    a = HandSlotAssigner(0.55)
    a.assign([cand(640, 330), cand(641, 470)], 0.0, W)
    first_side = a.assign([cand(640, 330), cand(641, 470)], DT, W)
    top_side = sides_of(first_side)[0]
    for i in range(2, 120):
        jitter = rng.normal(0, 4.0, size=2)
        top = cand(640 + jitter[0], 330)
        bottom = cand(640 + jitter[1], 470)
        result = a.assign([bottom, top], i * DT, W)
        assert sides_of(result)[1] == top_side, i     # never swaps


def test_single_detection_goes_to_nearest_recent_slot():
    a = HandSlotAssigner(0.55)
    a.assign([cand(400), cand(880)], 0.0, W)
    result = a.assign([cand(870)], DT, W)
    assert sides_of(result) == {0: "right"} and result[0].continuous
    # continuous tracking: a contradicting label does not matter
    result = a.assign([cand(420, label="Right")], 2 * DT, W)
    assert sides_of(result) == {0: "left"} and result[0].continuous


def test_single_detection_without_history_uses_label_then_frame_centre():
    a = HandSlotAssigner(0.55)
    assert sides_of(a.assign([cand(1000, label="Left")], 0.0, W)) == {0: "left"}
    b = HandSlotAssigner(0.55)
    assert sides_of(b.assign([cand(1000, label="")], 0.0, W)) == {0: "right"}
    c = HandSlotAssigner(0.55)
    assert sides_of(c.assign([cand(200, label="Right", score=0.55)], 0.0, W)) == {0: "left"}
    d = HandSlotAssigner(0.55)
    assert sides_of(d.assign([cand(200, label="Right", score=0.9)], 0.0, W)) == {0: "right"}


def test_slot_memory_expires_after_recent_window():
    a = HandSlotAssigner(0.55)
    a.assign([cand(300)], 0.0, W)                          # centre rule -> left
    assert a.is_recent("left", 0.5) and not a.is_recent("left", 0.6)
    result = a.assign([cand(320, label="Right")], 0.6, W)  # memory expired: label decides
    assert sides_of(result) == {0: "right"} and not result[0].continuous


def empty_frames(assigner: HandSlotAssigner, start: float, end: float) -> None:
    """Frames without detections (the smoother calls assign() for every frame)."""
    t = start + DT
    while t < end - 1e-9:
        assert assigner.assign([], t, W) == []
        t += DT


def test_label_overrides_nearest_slot_on_reappearance():
    a = HandSlotAssigner(0.55)
    a.assign([cand(300, label="Left")], 0.0, W)
    empty_frames(a, 0.0, 0.3)
    # 0.3 s later a confident right hand shows up on the other side
    result = a.assign([cand(900, label="Right", score=0.95)], 0.3, W)
    assert sides_of(result) == {0: "right"} and not result[0].continuous
    # an unconfident label does not override the position match
    b = HandSlotAssigner(0.55)
    b.assign([cand(300, label="Left")], 0.0, W)
    empty_frames(b, 0.0, 0.3)
    result = b.assign([cand(600, label="Right", score=0.7)], 0.3, W)
    assert sides_of(result) == {0: "left"} and result[0].continuous


@pytest.mark.parametrize("fps", [30.0, 8.0, 4.0])
def test_continuously_tracked_hand_keeps_slot_despite_label_flicker(fps):
    """Regression: below 10 fps every frame gap exceeded TRACKING_GAP, so a flickering
    confident label swapped a continuously tracked hand between the slots every frame."""
    a = HandSlotAssigner(0.55)
    for i in range(12):
        label = "Left" if i % 2 == 0 else "Right"
        result = a.assign([cand(500 + 3 * i, label=label, score=0.9)], i / fps, W)
        assert sides_of(result) == {0: "left"}, (fps, i)
        assert result[0].continuous or i == 0


def test_reappearance_after_missed_frames_at_low_fps():
    """At 8 fps one missed frame (250 ms without the hand) already is a re-appearance."""
    a = HandSlotAssigner(0.55)
    a.assign([cand(500, label="Left")], 0.0, W)
    a.assign([], 0.125, W)                                    # hand not detected
    result = a.assign([cand(510, label="Right", score=0.95)], 0.25, W)
    assert sides_of(result) == {0: "right"}


def test_smoother_low_fps_label_flicker_creates_no_ghost_hand():
    s = smoother()
    for i in range(10):
        label = "Left" if i % 2 == 0 else "Right"
        hands, _ = s.update([make_hand(500 + 2 * i, 400, label, score=0.9)], None, i / 8.0, (W, H))
        assert [(h.side, h.source) for h in hands] == [("left", "hand")], i


def test_implausible_jump_is_not_continuous():
    a = HandSlotAssigner(0.55)
    a.assign([cand(100)], 0.0, W)
    result = a.assign([cand(1200, label="")], DT, W)   # 0.86 frame widths in one frame
    assert not result[0].continuous
    assert sides_of(result) == {0: "right"}             # frame-centre fallback


def test_sorted_assignment_detects_identity_swap():
    """Only one slot recent + a new hand left of it: x order wins, filters must restart."""
    a = HandSlotAssigner(0.55)
    a.assign([cand(500)], 0.0, W)                       # left slot (centre rule)
    result = a.assign([cand(505), cand(420)], DT, W)    # new hand appears left of it
    by_index = {r.index: r for r in result}
    assert by_index[1].side == "left" and not by_index[1].continuous   # someone else now
    assert by_index[0].side == "right" and not by_index[0].continuous  # right slot is new


def crossed_assigner(left_label: str, right_label: str) -> tuple[HandSlotAssigner, float]:
    """Assigner whose slots crossed: the "left" slot hand ends up at x=780."""
    a = HandSlotAssigner(0.55)
    t = 0.0
    a.assign([cand(500, 300, left_label), cand(780, 460, right_label)], t, W)
    for k in range(1, 21):
        t += DT
        a.assign([cand(500 + 14 * k, 300, left_label), cand(780 - 14 * k, 460, right_label)], t, W)
    assert a.last_position("left")[0] == pytest.approx(780)
    return a, t


def test_confident_labels_repair_crossed_assignment():
    # unlabelled crossing (e.g. the labels were unreliable): slots are crossed now
    a, t = crossed_assigner("", "")
    # the x-left hand is confidently "Left" and the x-right one "Right": x order wins
    result = a.assign([cand(770, 300, "Right"), cand(510, 460, "Left")], t + DT, W)
    assert sides_of(result) == {1: "left", 0: "right"}
    assert not any(r.continuous for r in result)        # identities changed: restart


def test_physically_crossed_arms_stay_crossed():
    # the user's left hand ("Left") moved to the screen right: labels follow the hands
    a, t = crossed_assigner("Left", "Right")
    result = a.assign([cand(770, 300, "Left"), cand(510, 460, "Right")], t + DT, W)
    assert sides_of(result) == {0: "left", 1: "right"}
    assert all(r.continuous for r in result)


def test_labels_alone_never_cross_slots():
    a = HandSlotAssigner(0.55)
    a.assign([cand(600, 300, "Left"), cand(680, 460, "Right")], 0.0, W)
    # each hand stayed near its slot but the labels flipped: positions win
    result = a.assign([cand(690, 460, "Left"), cand(590, 300, "Right")], DT, W)
    assert sides_of(result) == {1: "left", 0: "right"}


def test_more_than_two_candidates_keep_best_scores():
    a = HandSlotAssigner(0.55)
    result = a.assign([cand(300, score=0.9), cand(600, score=0.2), cand(900, score=0.8)], 0.0, W)
    assert sides_of(result) == {0: "left", 2: "right"}
    assert a.assign([], DT, W) == []


# ---------------------------------------------------------------- smoothing


def smoother(**overrides) -> HandPoseSmoother:
    return HandPoseSmoother(dataclasses.replace(TrackingSettings(), **overrides))


def test_smoother_maps_to_scene_px():
    s = smoother()
    det = make_hand(400, 300, label="Left")
    hands, pose = s.update([det], None, 1.0, (W, H))
    assert pose is None and len(hands) == 1
    hand = hands[0]
    assert hand.side == "left" and hand.source == "hand" and hand.age == 0.0
    assert hand.landmarks.dtype == np.float32 and hand.landmarks.shape == (21, 3)
    expected = det.landmarks * (W, H, W)
    np.testing.assert_allclose(hand.landmarks, expected, atol=1e-3)   # first sample passes
    assert hand.palm_center == pytest.approx((400, 300), abs=1e-3)
    assert hand.world is not None and hand.world.dtype == np.float32
    assert hand.handedness == "Left" and hand.score == pytest.approx(0.95)


def test_two_hands_emitted_left_first():
    s = smoother()
    hands, _ = s.update([make_hand(900, 350), make_hand(380, 360)], None, 0.0, (W, H))
    assert [h.side for h in hands] == ["left", "right"]
    assert hands[0].palm_center[0] < hands[1].palm_center[0]


def test_vanished_hand_coasts_then_drops():
    grace = 0.25
    s = smoother(hand_lost_grace=grace)
    v = 0.3   # frame widths per second
    t = 0.0
    for i in range(31):                        # 1 s of steady motion to the right
        t = i * DT
        hands, _ = s.update([make_hand(300 + v * W * t, 360)], None, t, (W, H))
    last_x = hands[0].palm_center[0]
    t_last = t
    xs, ages = [], []
    for k in range(1, 12):
        t = t_last + k * DT
        hands, _ = s.update([], None, t, (W, H))
        if not hands:
            break
        hand = hands[0]
        assert hand.source == "predicted" and hand.side == "left"
        xs.append(hand.palm_center[0])
        ages.append(hand.age)
        assert hand.age == pytest.approx(k * DT)
        # damped velocity extrapolation: v * (1 - exp(-6 age)) / 6
        travel = v * W * (1.0 - math.exp(-COAST_DAMPING * hand.age)) / COAST_DAMPING
        assert hand.palm_center[0] - last_x == pytest.approx(travel, rel=0.03)
    assert len(xs) == 7                         # ages 1/30 .. 7/30 <= 0.25 s, then dropped
    steps = np.diff([last_x] + xs)
    assert np.all(steps > 0) and np.all(np.diff(steps) < 0)   # keeps moving, slowing down
    assert ages == sorted(ages)


def test_zero_grace_drops_hand_immediately():
    s = smoother(hand_lost_grace=0.0)
    s.update([make_hand(500, 300)], None, 0.0, (W, H))
    hands, _ = s.update([], None, DT, (W, H))
    assert hands == []


def test_coasting_extrapolation_speed_is_capped():
    """A glitchy last detection (huge derivative) must not fling the predicted hand away."""
    s = smoother(hand_lost_grace=0.25, smoothing_min_cutoff=20.0, smoothing_beta=0.0,
                 smoothing_d_cutoff=20.0)
    s.update([make_hand(300, 360)], None, 0.0, (W, H))
    hands, _ = s.update([make_hand(650, 360)], None, DT, (W, H))   # ~8 frame widths / s
    last_x = hands[0].palm_center[0]
    hands, _ = s.update([], None, 2 * DT, (W, H))
    hand = hands[0]
    assert hand.source == "predicted" and hand.side == "left"
    travel = (1.0 - math.exp(-COAST_DAMPING * hand.age)) / COAST_DAMPING
    # pure x translation: every landmark is limited to exactly MAX_COAST_SPEED
    assert hand.palm_center[0] - last_x == pytest.approx(MAX_COAST_SPEED * travel * W, rel=1e-3)
    assert hand.palm_center[1] == pytest.approx(360, abs=1e-3)


def test_filters_reset_when_hand_reappears_after_loss():
    s = smoother(hand_lost_grace=0.25)
    for i in range(30):
        s.update([make_hand(300, 360)], None, i * DT, (W, H))
    for i in range(30, 60):                     # 1 s without the hand: lost
        s.update([], None, i * DT, (W, H))
    det = make_hand(330, 380)
    hands, _ = s.update([det], None, 60 * DT, (W, H))
    np.testing.assert_allclose(hands[0].landmarks, det.landmarks * (W, H, W), atol=1e-3)


def test_filters_continue_when_hand_returns_within_grace():
    s = smoother(hand_lost_grace=0.25)
    for i in range(30):
        s.update([make_hand(300 + 8 * i, 360)], None, i * DT, (W, H))
    s.update([], None, 30 * DT, (W, H))          # one missed frame -> coasting
    det = make_hand(300 + 8 * 31, 360)
    hands, _ = s.update([det], None, 31 * DT, (W, H))
    raw_x = det.candidate((W, H)).position[0]
    assert hands[0].source == "hand"
    assert hands[0].palm_center[0] < raw_x - 0.5    # still smoothed (lags), not restarted


def test_smoothing_reduces_jitter_of_a_still_hand():
    rng = np.random.default_rng(4)
    s = smoother()
    raw, out = [], []
    for i in range(150):
        x, y = 640 + rng.normal(0, 2.5), 360 + rng.normal(0, 2.5)
        hands, _ = s.update([make_hand(x, y)], None, i * DT, (W, H))
        raw.append(x)
        out.append(hands[0].palm_center[0])
    assert np.std(out[30:]) < np.std(raw[30:]) / 2.0


def test_apply_settings_changes_smoothing_immediately():
    s = smoother(smoothing_min_cutoff=0.1, smoothing_beta=0.0)
    s.update([make_hand(300, 360)], None, 0.0, (W, H))
    hands, _ = s.update([make_hand(400, 360)], None, DT, (W, H))
    assert hands[0].palm_center[0] < 310          # heavy smoothing
    s.apply_settings(dataclasses.replace(TrackingSettings(), smoothing_min_cutoff=20.0,
                                         smoothing_beta=0.0))
    hands, _ = s.update([make_hand(400, 360)], None, 2 * DT, (W, H))
    assert hands[0].palm_center[0] > 380          # hardly any smoothing now


def test_duplicate_hand_detection_is_dropped():
    s = smoother()
    a = make_hand(640, 360, label="Left", score=0.9)
    b = make_hand(642, 361, label="Left", score=0.8)
    hands, _ = s.update([b, a], None, 0.0, (W, H))
    assert len(hands) == 1 and hands[0].score == pytest.approx(0.9)
    # two real hands pressed together have opposite labels: both kept
    s2 = smoother()
    hands, _ = s2.update([make_hand(640, 360, "Left"), make_hand(642, 361, "Right")], None, 0.0, (W, H))
    assert len(hands) == 2


def test_frame_size_change_resets_state():
    s = smoother()
    for i in range(10):
        s.update([make_hand(300, 300)], None, i * DT, (W, H))
    hands, _ = s.update([], None, 10 * DT, (640, 480))   # other camera: no coasting
    assert hands == []


def test_time_going_backwards_resets_state():
    s = smoother()
    for i in range(10):
        s.update([make_hand(300, 300)], None, 100.0 + i * DT, (W, H))
    det = make_hand(900, 300)
    hands, _ = s.update([det], None, 5.0, (W, H))
    np.testing.assert_allclose(hands[0].landmarks, det.landmarks * (W, H, W), atol=1e-3)


def shoulders(dx: float, vis: float = 0.98) -> PoseDetection:
    """Pose whose model-LEFT shoulder is ``dx`` frame widths right of the model-RIGHT one."""
    return make_pose(0.5 + dx / 2, 0.5 - dx / 2, vis=vis)


def test_pose_orientation_with_hysteresis():
    s = smoother()
    # mirrored camera, user facing it: the model's LEFT shoulder is on screen right
    _, pose = s.update([], shoulders(0.12), 0.0, (W, H))
    assert pose is not None and pose.model_left_is_screen_left is False
    assert pose.arm("left").shoulder == 12      # screen-left arm = model RIGHT arm
    t = 0.0
    for dx in (-0.01, -0.02, -0.028):           # reversed by less than 3 %: no flip
        t += 1.0                                # (1 s apart: filters restart, no blending)
        _, pose = s.update([], shoulders(dx), t, (W, H))
        assert pose.model_left_is_screen_left is False, dx
    t += 1.0
    _, pose = s.update([], shoulders(-0.06, vis=0.2), t, (W, H))   # shoulders not visible
    assert pose.model_left_is_screen_left is False
    t += 1.0
    _, pose = s.update([], shoulders(-0.06), t, (W, H))            # clearly reversed
    assert pose.model_left_is_screen_left is True
    assert pose.arm("left").shoulder == 11
    t += 1.0
    _, pose = s.update([], shoulders(0.025), t, (W, H))            # 2.5 % back: keep
    assert pose.model_left_is_screen_left is True
    t += 1.0
    _, pose = s.update([], shoulders(0.035), t, (W, H))            # 3.5 %: flip back
    assert pose.model_left_is_screen_left is False


def test_pose_smoothing_and_reset_after_loss():
    s = smoother()
    _, pose = s.update([], make_pose(0.56, 0.44), 0.0, (W, H))
    _, pose = s.update([], make_pose(0.60, 0.48), DT, (W, H))
    assert pose.landmarks[11, 0] < 0.60 * W - 1        # smoothed
    s.update([], None, 0.2, (W, H))                     # short loss (< 0.5 s): keep state
    _, pose = s.update([], make_pose(0.70, 0.58), 0.25, (W, H))
    assert pose.landmarks[11, 0] < 0.70 * W - 1
    _, pose = s.update([], make_pose(0.30, 0.18), 1.0, (W, H))   # lost > 0.5 s: restart
    assert pose.landmarks[11, 0] == pytest.approx(0.30 * W, abs=1e-3)


def test_pose_observation_fields():
    s = smoother()
    det = make_pose(0.56, 0.44, vis=0.9)
    det.visibility[15] = 0.1
    _, pose = s.update([], det, 0.0, (W, H))
    assert pose.landmarks.dtype == np.float32 and pose.landmarks.shape == (33, 3)
    assert pose.visibility.dtype == np.float32 and pose.visibility.shape == (33,)
    assert pose.world is not None and pose.world.shape == (33, 3)
    assert pose.visible(11) and not pose.visible(15)
    assert 0.0 < pose.score <= 1.0
    np.testing.assert_allclose(pose.landmarks[:, 2], det.landmarks[:, 2] * W, atol=1e-4)


# ------------------------------------------------------------ helper functions


def test_next_timestamp_strictly_increasing():
    assert next_timestamp_ms(-1, 0.0) == 0
    assert next_timestamp_ms(-1, 12.3456) == 12345
    assert next_timestamp_ms(12345, 12.3456) == 12346      # same capture time
    assert next_timestamp_ms(12346, 1.0) == 12347          # clock went backwards
    assert next_timestamp_ms(10, float("nan")) == 11
    assert next_timestamp_ms(10, float("inf")) == 11


def test_prepare_model_input_downscales_and_converts():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[..., 0] = 255                                       # pure blue in BGR
    rgb = prepare_model_input(img, 640)
    assert rgb.shape == (360, 640, 3) and rgb.flags["C_CONTIGUOUS"]
    assert tuple(rgb[10, 10]) == (0, 0, 255)                # RGB order
    small = np.zeros((240, 320, 3), np.uint8)
    assert prepare_model_input(small, 640).shape == (240, 320, 3)   # never upscaled
    odd = np.zeros((1080, 1920, 3), np.uint8)
    assert prepare_model_input(odd, 500).shape == (281, 500, 3)     # aspect kept
    assert prepare_model_input(np.zeros((100, 200), np.uint8), 640).shape == (100, 200, 3)
    assert prepare_model_input(np.zeros((100, 200, 4), np.uint8), 640).shape == (100, 200, 3)


def test_model_config_keys():
    base = ModelConfig.from_settings(TrackingSettings())
    other = ModelConfig.from_settings(dataclasses.replace(TrackingSettings(), pose_model="full"))
    assert base.key("hands") == other.key("hands")
    assert base.key("pose") != other.key("pose")
    smooth = ModelConfig.from_settings(dataclasses.replace(TrackingSettings(), smoothing_beta=3.0))
    assert smooth == base


# ------------------------------------------------------ fakes for the tracker


class FakeHandModel:
    def __init__(self, config: ModelConfig, positions=((400, 360, "Left"), (880, 360, "Right")),
                 delay: float = 0.0, fail: bool = False) -> None:
        self.config = config
        self.positions = positions
        self.delay = delay
        self.fail = fail
        self.closed = False
        self.calls: list[tuple[tuple[int, ...], float]] = []
        self.threads: set[str] = set()
        self.spans: list[tuple[float, float]] = []

    def detect(self, rgb, capture_time):
        assert not self.closed
        start = time.perf_counter()
        self.calls.append((rgb.shape, capture_time))
        self.threads.add(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        self.spans.append((start, time.perf_counter()))
        if self.fail:
            raise RuntimeError("synthetic hand failure")
        return [make_hand(x, y, label) for x, y, label in self.positions]

    def close(self):
        self.closed = True


class FakePoseModel:
    def __init__(self, config: ModelConfig, found: bool = True, delay: float = 0.0) -> None:
        self.config = config
        self.found = found
        self.delay = delay
        self.closed = False
        self.calls: list[tuple[tuple[int, ...], float, tuple[int, int] | None]] = []
        self.threads: set[str] = set()
        self.spans: list[tuple[float, float]] = []

    def detect(self, rgb, capture_time, mask_size):
        assert not self.closed
        start = time.perf_counter()
        self.calls.append((rgb.shape, capture_time, mask_size))
        self.threads.add(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        self.spans.append((start, time.perf_counter()))
        if not self.found:
            return None, None
        mask = None if mask_size is None else np.full((mask_size[1], mask_size[0]), 200, np.uint8)
        return make_pose(0.56, 0.44), mask

    def close(self):
        self.closed = True


class FakeTracker(HandPoseTracker):
    """HandPoseTracker with injectable fake models."""

    def __init__(self, settings: TrackingSettings | None = None, hand_kwargs=None,
                 pose_kwargs=None, **kwargs) -> None:
        super().__init__(settings or TrackingSettings(), Path("no-models-needed"), **kwargs)
        self.hand_kwargs = dict(hand_kwargs or {})
        self.pose_kwargs = dict(pose_kwargs or {})
        self.hand_models: list[FakeHandModel] = []
        self.pose_models: list[FakePoseModel] = []

    def _create_hand_model(self, config):
        model = FakeHandModel(config, **self.hand_kwargs)
        self.hand_models.append(model)
        return model

    def _create_pose_model(self, config):
        model = FakePoseModel(config, **self.pose_kwargs)
        self.pose_models.append(model)
        return model


class ManualSource:
    """FrameSource the test pushes frames into."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: CameraFrame | None = None

    def push(self, frame: CameraFrame) -> None:
        with self._cond:
            self._frame = frame
            self._cond.notify_all()

    def wait_for_frame(self, after_id: int, timeout: float) -> CameraFrame | None:
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._frame is not None and self._frame.frame_id > after_id, timeout)
            return self._frame if ok else None


class TimedSource:
    """FrameSource producing a new frame (same image) every ``interval`` seconds."""

    def __init__(self, image: np.ndarray, interval: float = DT) -> None:
        self.image = image
        self.interval = interval
        self.start = time.perf_counter()

    def frame_id_at(self, t: float) -> int:
        return int(round((t - self.start) / self.interval))

    def wait_for_frame(self, after_id: int, timeout: float) -> CameraFrame | None:
        deadline = time.perf_counter() + timeout
        while True:
            fid = int((time.perf_counter() - self.start) / self.interval)
            if fid > after_id:
                return CameraFrame(fid, self.image, self.start + fid * self.interval)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return None
            time.sleep(min(remaining, self.interval / 4))


def wait_until(predicate, timeout: float = 10.0, interval: float = 0.01) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def frame(fid: int, t: float, size=(W, H)) -> CameraFrame:
    return CameraFrame(fid, np.zeros((size[1], size[0], 3), np.uint8), t)


def tracker_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "tracker" or t.name.startswith("tracker-")]


# ------------------------------------------------------ tracker with fakes


def test_process_frame_with_fake_models():
    tr = FakeTracker()
    image = np.zeros((H, W, 3), np.uint8)
    result = tr.process_frame(CameraFrame(7, image, 50.0))
    assert isinstance(result, TrackingFrame)
    assert result.frame_id == 7 and result.capture_time == 50.0
    assert result.image is image                       # frames are immutable: no copy
    assert result.scene_size == (W, H)
    assert [h.side for h in result.hands] == ["left", "right"]
    assert result.pose is not None and result.pose.model_left_is_screen_left is False
    assert set(result.inference_ms) == {"hands", "pose", "total"}
    assert result.inference_ms["total"] >= max(result.inference_ms["hands"], result.inference_ms["pose"])
    # the models saw the downscaled RGB image and the mask size follows the full frame
    assert tr.hand_models[0].calls[0][0] == (360, 640, 3)
    assert tr.pose_models[0].calls[0][2] == (256, 144)
    assert result.segmentation is not None and result.segmentation.shape == (144, 256)
    assert result.segmentation_id == 1
    assert tr.latest() is result
    status = tr.status
    assert status.error is None and status.pose_model == "lite" and not status.running
    tr.stop()
    assert tr.hand_models[0].closed and tr.pose_models[0].closed


def test_parallel_inference_uses_one_worker_per_model():
    tr = FakeTracker()
    tr.process_frame(frame(1, 1.0))
    assert tr.hand_models[0].threads == {"tracker-hands_0"}
    assert tr.pose_models[0].threads == {"tracker-pose_0"}
    tr.update_settings(dataclasses.replace(TrackingSettings(), parallel_inference=False))
    tr.process_frame(frame(2, 1.0 + DT))
    caller = threading.current_thread().name
    assert caller in tr.hand_models[0].threads and caller in tr.pose_models[0].threads
    assert len(tr.hand_models) == 1               # not a model setting: no recreation
    tr.stop()
    assert not [t for t in tracker_threads() if t.is_alive()]


def test_parallel_inference_overlaps_models():
    tr = FakeTracker(hand_kwargs={"delay": 0.1}, pose_kwargs={"delay": 0.1})
    tr.process_frame(frame(1, 1.0))
    hands, pose = tr.hand_models[0], tr.pose_models[0]
    # both models run at the same time (interval overlap; no wall-clock bound)
    (h0, h1), (p0, p1) = hands.spans[0], pose.spans[0]
    assert p0 < h1 and h0 < p1
    tr.update_settings(dataclasses.replace(TrackingSettings(), parallel_inference=False))
    tr.process_frame(frame(2, 2.0))
    (h0, h1), (p0, p1) = hands.spans[1], pose.spans[1]
    assert h1 <= p0 or p1 <= h0                   # sequential: no overlap
    tr.stop()


def test_segmentation_none_without_pose_and_mask_width():
    tr = FakeTracker(pose_kwargs={"found": False})
    result = tr.process_frame(frame(1, 1.0))
    assert result.pose is None and result.segmentation is None and result.segmentation_id == 0
    tr.stop()
    tr2 = FakeTracker(mask_width=128)
    r1 = tr2.process_frame(frame(1, 1.0, size=(640, 480)))
    assert r1.segmentation.shape == (96, 128)
    tr2.set_mask_width(200)
    r2 = tr2.process_frame(frame(2, 1.0 + DT, size=(640, 480)))
    assert r2.segmentation.shape == (150, 200) and r2.segmentation_id == 2
    tr2.update_settings(dataclasses.replace(TrackingSettings(), enable_segmentation=False))
    r3 = tr2.process_frame(frame(3, 1.0 + 2 * DT, size=(640, 480)))
    assert r3.segmentation is None and r3.segmentation_id == 2 and r3.pose is not None
    assert tr2.pose_models[-1].calls[-1][2] is None
    tr2.stop()


def test_update_settings_recreates_only_changed_models():
    tr = FakeTracker()
    tr.process_frame(frame(1, 1.0))
    tr.update_settings(dataclasses.replace(TrackingSettings(), smoothing_beta=2.0,
                                           tracking_width=320, hand_lost_grace=0.5))
    assert len(tr.hand_models) == 1 and len(tr.pose_models) == 1
    tr.process_frame(frame(2, 1.0 + DT))
    assert tr.hand_models[0].calls[-1][0] == (180, 320, 3)     # tracking width applies
    tr.update_settings(dataclasses.replace(TrackingSettings(), pose_model="full"))
    # not running: recreated synchronously
    assert len(tr.pose_models) == 2 and tr.pose_models[0].closed
    assert tr.pose_models[1].config.pose_model == "full"
    assert len(tr.hand_models) == 1 and not tr.hand_models[0].closed
    assert tr.status.pose_model == "full"
    tr.update_settings(dataclasses.replace(TrackingSettings(), pose_model="full", max_hands=1))
    assert len(tr.hand_models) == 2 and tr.hand_models[1].config.max_hands == 1
    tr.update_settings(dataclasses.replace(TrackingSettings(), enable_pose=False))
    assert tr.pose_models[-1].closed and tr.status.pose_model == ""
    result = tr.process_frame(frame(3, 1.0 + 2 * DT))
    assert result.pose is None and result.inference_ms["pose"] == 0.0
    tr.stop()


def test_update_settings_before_first_frame_is_lazy():
    tr = FakeTracker()
    tr.update_settings(dataclasses.replace(TrackingSettings(), pose_model="heavy"))
    assert tr.pose_models == [] and tr.hand_models == []
    tr.process_frame(frame(1, 1.0))
    assert tr.pose_models[0].config.pose_model == "heavy"
    tr.stop()


def test_inference_failures_are_contained_and_landmarker_recreated(caplog):
    tr = FakeTracker(hand_kwargs={"fail": True})
    with caplog.at_level("ERROR", logger="dragon_ball_powers.tracking"):
        for i in range(12):
            result = tr.process_frame(frame(i + 1, 1.0 + i * DT))
            assert result.pose is not None            # the pose model keeps working
            assert result.hands == []
    assert "Hand tracking error" in tr.status.error
    assert len(tr.hand_models) == 2 and tr.hand_models[0].closed   # recreated after 10
    reports = [r for r in caplog.records if r.getMessage().startswith("Hand inference failed:")]
    assert len(reports) == 1                          # rate-limited (once per 5 s)
    assert reports[0].exc_info is not None            # with traceback
    assert any("recreating" in r.getMessage() for r in caplog.records)
    tr.hand_kwargs["fail"] = False
    tr.hand_models[-1].fail = False
    result = tr.process_frame(frame(20, 2.0))
    assert len(result.hands) == 2 and tr.status.error is None
    tr.stop()


def test_threaded_tracker_with_fake_models():
    tr = FakeTracker()
    source = TimedSource(np.zeros((H, W, 3), np.uint8))
    tr.start(source)
    try:
        assert tr.status.running
        assert wait_until(lambda: tr.latest() is not None and len(tr.latest().hands) == 2)
        first = tr.latest()
        assert wait_until(lambda: tr.latest().frame_id > first.frame_id)
        names = {t.name for t in tracker_threads()}
        assert {"tracker", "tracker-hands_0", "tracker-pose_0"} <= names
        assert wait_until(lambda: tr.status.fps > 5.0)
        tr.start(source)                                  # same source: no-op
        assert len(tr.hand_models) == 1
    finally:
        tr.stop()
    assert not tr.status.running
    assert not [t for t in tracker_threads() if t.is_alive()]
    assert tr.hand_models[0].closed and tr.pose_models[0].closed
    # start / stop is repeatable
    tr.start(source)
    try:
        assert wait_until(lambda: tr.latest() is not None and tr.latest().hands)
    finally:
        tr.stop()
    assert len(tr.hand_models) == 2 and tr.hand_models[1].closed
    assert not [t for t in tracker_threads() if t.is_alive()]


def test_tracker_drops_stale_frames():
    tr = FakeTracker(hand_kwargs={"delay": 0.04}, pose_kwargs={"delay": 0.04})
    source = TimedSource(np.zeros((H, W, 3), np.uint8), interval=0.004)
    tr.start(source)
    try:
        assert wait_until(lambda: len(tr.hand_models) == 1 and len(tr.hand_models[0].calls) >= 8)
    finally:
        tr.stop()
    ids = [source.frame_id_at(t) for _, t in tr.hand_models[0].calls]
    assert ids == sorted(set(ids))                       # each frame at most once, in order
    assert np.mean(np.diff(ids)) >= 3.0                  # newest frame, stale ones skipped


def test_threaded_update_settings_applies_before_next_frame():
    tr = FakeTracker()
    source = ManualSource()
    tr.start(source)
    try:
        source.push(frame(1, 1.0))
        assert wait_until(lambda: tr.latest() is not None and tr.latest().frame_id == 1)
        tr.update_settings(dataclasses.replace(TrackingSettings(), pose_model="full"))
        assert tr.pose_models[-1].config.pose_model == "lite"   # thread applies it later
        source.push(frame(2, 1.0 + DT))
        assert wait_until(lambda: tr.latest().frame_id == 2)
        assert tr.pose_models[-1].config.pose_model == "full" and tr.pose_models[0].closed
        assert tr.status.pose_model == "full"
    finally:
        tr.stop()


def test_thread_survives_pipeline_errors():
    tr = FakeTracker()
    source = ManualSource()
    tr.start(source)
    try:
        bad = CameraFrame(1, np.zeros((H, W, 3), np.float32), 1.0)   # unsupported image
        source.push(bad)
        assert wait_until(lambda: tr.status.error is not None and "Tracking error" in tr.status.error)
        assert tr.status.running
        source.push(frame(2, 2.0))
        assert wait_until(lambda: tr.latest() is not None and tr.latest().frame_id == 2)
        assert tr.status.error is None
    finally:
        tr.stop()


def test_thread_accepts_restarted_frame_ids():
    tr = FakeTracker()
    source = ManualSource()
    tr.start(source)
    try:
        for fid in range(1, 4):
            source.push(frame(fid, 1.0 + fid * DT))
            assert wait_until(lambda: tr.latest() is not None and tr.latest().frame_id == fid)
        source.push(frame(0, 10.0))          # camera restarted: ids count from 0 again
        assert wait_until(lambda: tr.latest().frame_id == 0, timeout=5.0)
        source.push(frame(1, 10.0 + DT))
        assert wait_until(lambda: tr.latest().frame_id == 1 and tr.latest().capture_time > 10.0)
    finally:
        tr.stop()


class FlakySource:
    """Raises, returns garbage, then delivers real frames."""

    def __init__(self) -> None:
        self.calls = 0

    def wait_for_frame(self, after_id: int, timeout: float):
        self.calls += 1
        if self.calls <= 3:
            raise OSError("camera hiccup")
        if self.calls <= 5:
            return object()                       # not a CameraFrame
        fid = self.calls
        if fid <= after_id:
            time.sleep(timeout)
            return None
        time.sleep(0.01)
        return frame(fid, 1.0 + fid * DT)


def test_thread_survives_failing_and_malformed_source(caplog):
    tr = FakeTracker()
    with caplog.at_level("ERROR", logger="dragon_ball_powers.tracking"):
        tr.start(FlakySource())
        try:
            assert wait_until(lambda: tr.latest() is not None)
            assert tr.status.running
        finally:
            tr.stop()
    messages = [r.getMessage() for r in caplog.records]
    assert any(m.startswith("Frame source failed: camera hiccup") for m in messages)
    assert any(m.startswith("Frame source returned an invalid frame") for m in messages)
    assert not any("crashed" in m for m in messages)


def test_mediapipe_wrapper_timestamps_strictly_increase():
    from types import SimpleNamespace

    from dragon_ball_powers.tracking import _MediaPipeHandModel, _MediaPipePoseModel

    class FakeLandmarker:
        def __init__(self) -> None:
            self.timestamps: list[int] = []

        def detect_for_video(self, image, ts):
            self.timestamps.append(ts)
            return SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[],
                                   pose_landmarks=[], pose_world_landmarks=[],
                                   segmentation_masks=None)

        def close(self):
            raise ValueError("Task runner is currently not running.")   # like MediaPipe

    fake_mp = SimpleNamespace(Image=lambda image_format, data: data,
                              ImageFormat=SimpleNamespace(SRGB="srgb"))
    rgb = np.zeros((4, 4, 3), np.uint8)
    hand_lm, pose_lm = FakeLandmarker(), FakeLandmarker()
    hands, pose = _MediaPipeHandModel(hand_lm, fake_mp), _MediaPipePoseModel(pose_lm, fake_mp)
    for t in (10.0, 10.0, 10.0005, 9.0, 10.5):
        assert hands.detect(rgb, t) == []
        assert pose.detect(rgb, t, (16, 9)) == (None, None)
    for stamps in (hand_lm.timestamps, pose_lm.timestamps):
        assert stamps == [10000, 10001, 10002, 10003, 10500]
    hands.close()     # errors while closing are swallowed
    hands.close()
    pose.close()


def mp_points(rows, visibility=None, presence=None):
    """MediaPipe-like landmark objects (None scores stay None, like the real dataclasses)."""
    from types import SimpleNamespace

    n = len(rows)
    vis = [None] * n if visibility is None else list(visibility)
    pres = [None] * n if presence is None else list(presence)
    return [SimpleNamespace(x=float(r[0]), y=float(r[1]), z=float(r[2]), visibility=vis[i],
                            presence=pres[i]) for i, r in enumerate(rows)]


@pytest.mark.parametrize("mask_shape", [(90, 160), (90, 160, 1)])
@pytest.mark.parametrize("mask_size", [(64, 36), (160, 90)])   # downscaled / same size
def test_parse_pose_copies_mask_out_of_mediapipe_memory(mask_shape, mask_size):
    from types import SimpleNamespace

    rng = np.random.default_rng(1)
    xyz = rng.uniform(0.2, 0.8, size=(33, 3))
    vis = rng.uniform(0.0, 1.0, 33)
    pres = list(rng.uniform(0.0, 1.0, 33))
    pres[3] = None                                       # a missing score is ignored
    vis[7] = 1.4                                         # out-of-range scores are clipped
    pres[7] = 1.2
    owned = np.zeros(mask_shape, np.float32)             # memory "owned by MediaPipe"
    owned[20:70, 50:110] = 1.0                           # the person
    result = SimpleNamespace(
        pose_landmarks=[mp_points(xyz, vis, pres)],
        pose_world_landmarks=[mp_points(xyz - 0.5)],
        segmentation_masks=[SimpleNamespace(numpy_view=lambda: owned)])
    det, mask = _parse_pose(result, mask_size)
    owned[:] = 0.5                                       # MediaPipe reuses its buffer
    assert mask is not None and mask.dtype == np.uint8 and mask.ndim == 2
    assert mask.shape == (mask_size[1], mask_size[0])
    assert not np.shares_memory(mask, owned)
    assert mask.min() == 0 and mask.max() == 255         # 0..1 float -> 0..255, unchanged
    expected = np.fmin(vis, np.array(pres, dtype=np.float64))
    expected[3] = vis[3]
    np.testing.assert_allclose(det.visibility, np.clip(expected, 0.0, 1.0))
    np.testing.assert_allclose(det.landmarks, xyz)
    np.testing.assert_allclose(det.world, xyz - 0.5)
    # no mask without a requested size; no pose -> no mask either
    assert _parse_pose(result, None)[1] is None
    empty = SimpleNamespace(pose_landmarks=[], pose_world_landmarks=[], segmentation_masks=None)
    assert _parse_pose(empty, mask_size) == (None, None)


def test_parse_hands_tolerates_missing_world_labels_and_bad_landmarks():
    from types import SimpleNamespace

    first, second = make_hand(400, 300, "Left"), make_hand(900, 320, "Right")
    broken = first.landmarks.copy()
    broken[5, 0] = np.nan
    result = SimpleNamespace(
        hand_landmarks=[mp_points(first.landmarks), mp_points(broken), mp_points(second.landmarks)],
        hand_world_landmarks=[mp_points(first.world), mp_points(first.world)],  # 3rd missing
        handedness=[[SimpleNamespace(category_name="Left", score=0.93)],
                    [SimpleNamespace(category_name="Left", score=0.5)], []])
    dets = _parse_hands(result)
    assert len(dets) == 2                                 # the NaN hand is dropped
    a, b = dets
    np.testing.assert_allclose(a.landmarks, first.landmarks)
    np.testing.assert_allclose(a.world, first.world)
    assert (a.handedness, a.score) == ("Left", pytest.approx(0.93))
    np.testing.assert_allclose(b.landmarks, second.landmarks)   # indices stay aligned
    assert b.world is None and b.handedness == "" and b.score == 0.0


def test_error_reporter_rate_limits_with_counts(caplog):
    from dragon_ball_powers.tracking import _ErrorReporter

    reporter = _ErrorReporter(interval=0.5)
    with caplog.at_level("ERROR", logger="dragon_ball_powers.tracking"):
        for _ in range(4):
            reporter.report("k", "Thing failed", RuntimeError("boom"))
        reporter.report("other", "Other failed", RuntimeError("bang"))
        time.sleep(0.6)
        reporter.report("k", "Thing failed", RuntimeError("boom"))
    messages = [r.getMessage() for r in caplog.records]
    assert messages == ["Thing failed: boom", "Other failed: bang",
                        "Thing failed: boom (3 similar errors since the last report)"]


def test_stop_without_start_and_double_stop():
    tr = FakeTracker()
    tr.stop()
    tr.process_frame(frame(1, 1.0))
    tr.stop()
    tr.stop()
    assert tr.hand_models[0].closed
    result = tr.process_frame(frame(2, 2.0))   # usable again after stop (lazy reload)
    assert len(result.hands) == 2 and len(tr.hand_models) == 2
    tr.stop()


class SlowFirstLoadTracker(FakeTracker):
    """The first hand model takes long to create (first MediaPipe import, virus scan...)."""

    load_delay = 0.5

    def _create_hand_model(self, config):
        if not self.hand_models:
            time.sleep(self.load_delay)
        return super()._create_hand_model(config)


def alive_tracker_threads(name: str = "tracker") -> int:
    return sum(1 for t in threading.enumerate() if t.name == name and t.is_alive())


def test_stop_timeout_releases_models_when_the_thread_finishes(monkeypatch):
    """Regression: when stop() gave up waiting, the landmarkers were never closed."""
    import dragon_ball_powers.tracking as tracking

    monkeypatch.setattr(tracking, "STOP_TIMEOUT", 0.1)
    tr = SlowFirstLoadTracker()
    tr.start(ManualSource())
    time.sleep(0.05)                     # the thread is inside the slow model load
    tr.stop()                            # gives up after 0.1 s
    assert not tr.running
    assert wait_until(lambda: alive_tracker_threads() == 0)
    assert tr.hand_models and all(m.closed for m in tr.hand_models)
    assert tr.pose_models and all(m.closed for m in tr.pose_models)
    assert tr.status.pose_model == ""
    assert not [t for t in tracker_threads() if t.is_alive()]


def test_stop_timeout_then_restart_keeps_the_successors_models(monkeypatch):
    import dragon_ball_powers.tracking as tracking

    monkeypatch.setattr(tracking, "STOP_TIMEOUT", 0.1)
    tr = SlowFirstLoadTracker()
    source = TimedSource(np.zeros((H, W, 3), np.uint8))
    tr.start(source)
    time.sleep(0.05)
    tr.stop()                            # times out: the first thread is still loading
    tr.start(source)                     # the successor shares the models being loaded
    try:
        assert wait_until(lambda: alive_tracker_threads() == 1)   # the old thread exited
        assert wait_until(lambda: tr.latest() is not None and len(tr.latest().hands) == 2)
        first = tr.latest().frame_id
        assert wait_until(lambda: tr.latest().frame_id > first)
        assert len(tr.hand_models) == 1 and not tr.hand_models[0].closed
        assert len(tr.pose_models) == 1 and not tr.pose_models[0].closed
    finally:
        tr.stop()
    assert tr.hand_models[0].closed and tr.pose_models[0].closed


def test_status_fps_reads_zero_when_results_go_stale(monkeypatch):
    import dragon_ball_powers.tracking as tracking

    monkeypatch.setattr(tracking, "FPS_STALE_AFTER", 0.5)
    tr = FakeTracker()
    for i in range(4):
        tr.process_frame(frame(i + 1, 1.0 + i * DT))
        time.sleep(0.02)
    assert tr.status.fps > 0.0
    time.sleep(0.7)
    assert tr.status.fps == 0.0
    assert tr.status.total_ms > 0.0      # the timings are kept for the debug overlay


def test_missing_model_files_reported_without_crash(tmp_path):
    tr = HandPoseTracker(TrackingSettings(), models_dir=tmp_path / "does-not-exist")
    result = tr.process_frame(frame(1, 1.0))
    assert result.hands == [] and result.pose is None and result.segmentation is None
    error = tr.status.error
    assert error is not None and error.startswith("Tracking model missing: ")
    assert HAND_MODEL_FILE in error and pose_model_file("lite") in error
    assert SETUP_HINT in error
    assert tr.status.pose_model == ""
    tr.stop()


def test_missing_models_in_thread_keep_publishing_frames(tmp_path):
    tr = HandPoseTracker(TrackingSettings(), models_dir=tmp_path)
    source = TimedSource(np.zeros((240, 320, 3), np.uint8))
    tr.start(source)
    try:
        assert wait_until(lambda: tr.latest() is not None)
        assert tr.status.running and "Tracking model missing" in tr.status.error
        assert tr.latest().hands == []
    finally:
        tr.stop()
    assert not tr.status.running


def test_corrupt_model_file_reported(tmp_path):
    (tmp_path / HAND_MODEL_FILE).write_bytes(b"not a model at all" * 10)
    settings = dataclasses.replace(TrackingSettings(), enable_pose=False)
    tr = HandPoseTracker(settings, models_dir=tmp_path)
    try:
        import mediapipe  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"mediapipe not importable: {exc}")
    result = tr.process_frame(frame(1, 1.0))
    assert result.hands == []
    error = tr.status.error
    assert error is not None
    assert f"Could not load tracking model {HAND_MODEL_FILE}" in error
    # actionable: a damaged / partial download is repaired by setup.bat
    assert error.endswith(SETUP_HINT) and ".." not in error
    tr.stop()


# ------------------------------------------------------- real MediaPipe models


def require_models(models_dir: Path, *names: str) -> None:
    names = names or (HAND_MODEL_FILE, pose_model_file("lite"))
    missing = [n for n in names if not (models_dir / n).is_file()]
    if missing:
        pytest.skip(f"MediaPipe models missing: {', '.join(missing)}")
    try:
        import mediapipe  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"mediapipe not importable: {exc}")


def sample(name: str) -> np.ndarray:
    path = SAMPLES / name
    image = cv2.imread(str(path)) if path.is_file() else None
    if image is None:
        pytest.skip(f"sample image {path} not available")
    return cv2.flip(image, 1)   # the camera module delivers mirrored frames


@pytest.mark.mediapipe
def test_real_black_frame(models_dir):
    require_models(models_dir)
    with HandPoseTracker(TrackingSettings(), models_dir) as tr:
        t = time.perf_counter()
        result = tr.process_frame(CameraFrame(1, np.zeros((720, 1280, 3), np.uint8), t))
        result = tr.process_frame(CameraFrame(2, np.zeros((720, 1280, 3), np.uint8), t + DT))
        assert result.hands == [] and result.pose is None and result.segmentation is None
        assert result.scene_size == (1280, 720)
        ms = result.inference_ms
        assert set(ms) == {"hands", "pose", "total"}
        assert ms["hands"] > 0 and ms["pose"] > 0 and ms["total"] > 0
        status = tr.status
        assert status.error is None and status.pose_model == "lite"
        assert status.hands_ms > 0 and status.pose_ms > 0 and status.total_ms > 0


@pytest.mark.mediapipe
def test_real_two_hands_in_screen_slots(models_dir):
    require_models(models_dir)
    image = sample("woman_hands.jpg")
    h, w = image.shape[:2]
    with HandPoseTracker(TrackingSettings(), models_dir) as tr:
        result = tr.process_frame(CameraFrame(1, image, time.perf_counter()))
    assert result.scene_size == (w, h)
    assert [hand.side for hand in result.hands] == ["left", "right"]
    for hand in result.hands:
        assert hand.source == "hand" and hand.age == 0.0
        assert hand.landmarks.shape == (21, 3) and hand.world.shape == (21, 3)
        assert np.all(hand.landmarks[:, 0] >= -0.02 * w) and np.all(hand.landmarks[:, 0] <= 1.02 * w)
        assert np.all(hand.landmarks[:, 1] >= -0.02 * h) and np.all(hand.landmarks[:, 1] <= 1.02 * h)
        assert np.abs(hand.world).max() < 0.3       # metres, hand-centred
    left, right = result.hand("left"), result.hand("right")
    assert left.palm_center[0] < w / 2 < right.palm_center[0]
    assert (left.handedness, right.handedness) == ("Left", "Right")   # mirrored input


@pytest.mark.mediapipe
def test_real_pose_and_segmentation(models_dir):
    require_models(models_dir)
    image = sample("pose.jpg")
    h, w = image.shape[:2]
    with HandPoseTracker(TrackingSettings(), models_dir, mask_width=256) as tr:
        result = tr.process_frame(CameraFrame(1, image, time.perf_counter()))
        pose = result.pose
        assert pose is not None
        assert pose.visible(11) and pose.visible(12)
        assert pose.model_left_is_screen_left is False   # mirrored image of a person facing us
        assert np.all(pose.landmarks[[11, 12], 0] > 0) and np.all(pose.landmarks[[11, 12], 0] < w)
        seg = result.segmentation
        assert seg is not None and seg.dtype == np.uint8
        assert seg.shape == (round(256 * h / w), 256)
        assert seg.min() == 0 and seg.max() == 255
        # the person is in the middle of the image: mask is set there, not in the corner
        assert seg[seg.shape[0] // 2, seg.shape[1] // 2] > 128 and seg[2, 2] < 128
        assert result.segmentation_id == 1
        tr.set_mask_width(128)
        result = tr.process_frame(CameraFrame(2, image, time.perf_counter()))
        assert result.segmentation.shape == (round(128 * h / w), 128)
        assert result.segmentation_id == 2


@pytest.mark.mediapipe
def test_real_same_timestamp_twice_does_not_fail(models_dir):
    require_models(models_dir)
    image = sample("woman_hands.jpg")
    with HandPoseTracker(TrackingSettings(), models_dir) as tr:
        r1 = tr.process_frame(CameraFrame(1, image, 5.0))
        r2 = tr.process_frame(CameraFrame(2, image, 5.0))
        assert len(r1.hands) == 2 and len(r2.hands) == 2
        assert tr.status.error is None


@pytest.mark.mediapipe
def test_real_threaded_tracker(models_dir):
    require_models(models_dir)
    image = sample("woman_hands.jpg")
    tr = HandPoseTracker(TrackingSettings(), models_dir)
    source = TimedSource(image)
    for _ in range(2):   # start / stop twice
        tr.start(source)
        try:
            assert wait_until(lambda: tr.latest() is not None and len(tr.latest().hands) == 2,
                              timeout=20.0)
            result = tr.latest()
            assert {h.side for h in result.hands} == {"left", "right"}
            assert result.pose is not None and result.segmentation is not None
            assert tr.status.running and tr.status.error is None
            assert wait_until(lambda: tr.status.fps > 1.0, timeout=10.0)
        finally:
            tr.stop()
        assert not tr.status.running
        assert not [t for t in tracker_threads() if t.is_alive()]


@pytest.mark.mediapipe
def test_real_hands_only_when_pose_model_missing(models_dir, tmp_path):
    require_models(models_dir, HAND_MODEL_FILE)
    shutil.copy(models_dir / HAND_MODEL_FILE, tmp_path / HAND_MODEL_FILE)
    image = sample("woman_hands.jpg")
    with HandPoseTracker(TrackingSettings(), tmp_path) as tr:
        result = tr.process_frame(CameraFrame(1, image, time.perf_counter()))
        assert len(result.hands) == 2 and result.pose is None
        error = tr.status.error
        assert error.startswith("Tracking model missing: ") and pose_model_file("lite") in error
        assert HAND_MODEL_FILE not in error and tr.status.pose_model == ""


@pytest.mark.mediapipe
def test_real_pose_only_when_hand_model_missing(models_dir, tmp_path):
    require_models(models_dir, pose_model_file("lite"))
    shutil.copy(models_dir / pose_model_file("lite"), tmp_path / pose_model_file("lite"))
    image = sample("pose.jpg")
    with HandPoseTracker(TrackingSettings(), tmp_path) as tr:
        result = tr.process_frame(CameraFrame(1, image, time.perf_counter()))
        assert result.hands == [] and result.pose is not None
        assert result.segmentation is not None and result.inference_ms["hands"] == 0.0
        error = tr.status.error
        assert error == f"Tracking model missing: {HAND_MODEL_FILE}. {SETUP_HINT}"
        assert tr.status.pose_model == "lite" and tr.status.hands_ms == 0.0


@pytest.mark.mediapipe
def test_real_missing_model_recovers_when_file_appears(models_dir, tmp_path, monkeypatch):
    require_models(models_dir, HAND_MODEL_FILE)
    import dragon_ball_powers.tracking as tracking

    monkeypatch.setattr(tracking, "MISSING_MODEL_RETRY", 0.0)   # check on every frame
    image = sample("woman_hands.jpg")
    settings = dataclasses.replace(TrackingSettings(), enable_pose=False)
    with HandPoseTracker(settings, tmp_path) as tr:
        assert tr.process_frame(CameraFrame(1, image, 1.0)).hands == []
        assert HAND_MODEL_FILE in tr.status.error
        shutil.copy(models_dir / HAND_MODEL_FILE, tmp_path / HAND_MODEL_FILE)  # setup.bat ran
        result = tr.process_frame(CameraFrame(2, image, 2.0))
        assert len(result.hands) == 2 and tr.status.error is None


@pytest.mark.mediapipe
def test_real_update_settings_pose_model_full(models_dir):
    require_models(models_dir, HAND_MODEL_FILE, pose_model_file("lite"), pose_model_file("full"))
    image = sample("pose.jpg")
    with HandPoseTracker(TrackingSettings(), models_dir) as tr:
        tr.process_frame(CameraFrame(1, image, 1.0))
        assert tr.status.pose_model == "lite"
        tr.update_settings(dataclasses.replace(TrackingSettings(), pose_model="full"))
        assert tr.status.pose_model == "full"
        result = tr.process_frame(CameraFrame(2, image, 1.0 + DT))
        assert result.pose is not None and result.pose.visible(11) and result.pose.visible(12)
        assert tr.status.error is None


@pytest.mark.mediapipe
def test_real_segmentation_mask_is_a_private_copy(models_dir):
    """The published mask must not alias MediaPipe's (reused) output buffer."""
    require_models(models_dir)
    image = sample("pose.jpg")
    shifted = np.ascontiguousarray(np.roll(image, image.shape[1] // 5, axis=1))
    with HandPoseTracker(TrackingSettings(), models_dir) as tr:
        first = tr.process_frame(CameraFrame(1, image, 1.0)).segmentation
        assert first is not None
        kept = first.copy()
        later = [tr.process_frame(CameraFrame(i, shifted if i % 2 else image, 1.0 + i * DT)).segmentation
                 for i in range(2, 7)]
    assert np.array_equal(first, kept)                      # untouched by later frames
    assert any(m is not None and not np.array_equal(m, kept) for m in later)   # masks did change


@pytest.mark.mediapipe
def test_real_threaded_update_settings_switches_pose_model(models_dir):
    require_models(models_dir, HAND_MODEL_FILE, pose_model_file("lite"), pose_model_file("full"))
    image = sample("pose.jpg")
    tr = HandPoseTracker(TrackingSettings(), models_dir)
    tr.start(TimedSource(image))
    try:
        assert wait_until(lambda: tr.latest() is not None and tr.latest().pose is not None, timeout=20.0)
        assert tr.status.pose_model == "lite"
        tr.update_settings(dataclasses.replace(TrackingSettings(), pose_model="full",
                                               enable_segmentation=False))
        assert wait_until(lambda: tr.status.pose_model == "full", timeout=20.0)
        switched = tr.latest().frame_id
        assert wait_until(lambda: tr.latest().frame_id > switched + 2, timeout=20.0)
        result = tr.latest()
        assert result.pose is not None and result.segmentation is None
        assert tr.status.error is None and tr.status.running
    finally:
        tr.stop()
    assert not [t for t in tracker_threads() if t.is_alive()]
