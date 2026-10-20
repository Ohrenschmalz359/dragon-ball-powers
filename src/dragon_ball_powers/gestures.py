"""Gesture recognition: body features, gesture detectors and events (ARCHITECTURE §5.3, §7).

:class:`GestureEngine` turns smoothed tracking frames into

* a :class:`BodyState` (returned as its subclass :class:`BodyFeatures`) with per-arm
  features normalised by the body scale ``S`` (about one shoulder width in px),
* continuous :class:`GestureSignals` (charge pose, thrust, burst, aim, tracking ok),
* discrete :class:`GestureEvent` s on edges (charge pose start/end, one ``THRUST`` per
  push, one ``BURST`` per burst, tracking lost/found).

Pipeline per tracking frame
---------------------------
1. **Hands per screen side**: hand-model palm centre and rotation-robust palm size.
   When the hand model has no hand for a side but the pose sees the wrist (common when
   the hands overlap in the cupped charging pose) the pose wrist/index/pinky points
   stand in (``source="pose"``); a size ratio and centre offset between both sources is
   learned whenever both see the same hand, so the switch does not make the hand jump.
   Coasting tracker hands (``source="predicted"``) are used when the pose cannot help.
   Hand slots are matched to pose arms by wrist distance (crossed arms, slot swaps).
2. **Body scale S** = max(shoulder width, 0.75 x torso length) with the shoulder width
   de-rotated by the torso yaw from the world landmarks; hand-size and frame-height
   fallbacks, jump rejection and slow smoothing. A scale "epoch" changes whenever S has
   to jump, and no derivative or window is ever computed across an epoch change.
3. **Kinematics** from a short NumPy history: velocities, approach rate, hand distance
   rate (differences over >= 2 frames / ~66 ms, lightly smoothed, never across a
   change of the measurement source).
4. **Detectors** (scores 0..1; every threshold moves monotonically with
   ``settings.sensitivity``): charge pose with hysteresis, thrust (frontal and lateral
   cues over ``thrust_window``), burst, aim, tracking state.

Monocular depth ("towards the camera") is an approximation from several cues. Where the
literal formulas of §7 proved unreliable in the physically modelled synthetic tests
(``tests/motion_synth.py``), these deliberate refinements are used:

* Approach cues use a **rotation-invariant hand scale** (2D palm points fitted to the
  metric, camera-aligned hand world landmarks, perspective-corrected): the 2D palm size
  grows by 20-50 % when a hand merely turns (e.g. lowering the arms), which fired
  thrusts. Their ranges are rescaled for true depth growth and normalised to a user
  standing ``REFERENCE_DEPTH`` metres away (perspective makes the same push look smaller
  further away); the push speed also uses the wrists' forward speed from the metric
  world landmarks, which is far less noisy than the size rate of small, far hands.
* The frontal score needs a 3D arm-extension gain and is reduced unless the arms end
  up pointing at the camera; the wrist-depth cue uses metric world depth relative to
  the shoulder midpoint (MediaPipe's image z is exaggerated and a torso twist moves the
  own shoulder).
* The lateral cue measures the hand displacement along the arm direction, extension
  gains from the window start (2D or 3D), and de-emphasises motions straight down
  (lowering the arms).
* Bursts need clearly opposite/outward hand motion and a separation rate that real arm
  flings reach (13-22 S/s) but relaxing the hands (3.5-7 S/s) does not; extrapolated
  (coasting) hands never fire a burst.
* A straight arm hanging down vetoes the charge pose (one raised hand next to a hanging
  arm is not "hands together").
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from dragon_ball_powers.config import Calibration, GestureSettings
from dragon_ball_powers.datatypes import (
    FINGERTIP_INDICES,
    PALM_INDICES,
    ArmIndices,
    ArmState,
    BodyState,
    GestureEvent,
    GestureEventType,
    GestureSignals,
    HandObservation,
    MotionSummary,
    PoseObservation,
    TrackingFrame,
)
from dragon_ball_powers.mathutil import Vec2, clamp, damp, lerp, remap, smoothstep

log = logging.getLogger(__name__)

SIDES: tuple[str, str] = ("left", "right")

# ----------------------------------------------------------------- tunables
# Landmark usage
VISIBLE = 0.5                  # pose landmark visibility needed to use a point
POSE_HAND_VISIBLE = 0.3        # pose index/pinky visibility for the palm-centre fallback
OPENNESS_RANGE = (0.35, 1.0)   # mean fingertip distance / palm size: fist ~0.27, open ~0.9-1.05 (measured)

# Timing
HISTORY_SECONDS = 2.0          # >= 1.2 s needed by the thrust detector (max window 1.0 s)
DIFF_MIN_DT = 0.06             # derivatives difference samples >= 2 frames and ~66 ms apart ...
DIFF_MAX_DT = 0.25             # ... but never across tracking gaps longer than this
VELOCITY_RATE = 25.0           # 1/s, light smoothing of velocities (~40 ms)
APPROACH_RATE_SMOOTH = 12.0    # 1/s, smoothing of the approach rate (~80 ms)
SIZE_SMOOTH = 15.0             # 1/s, light smoothing of the log hand scale before gains (~67 ms)
DISTANCE_RATE_SMOOTH = 25.0    # 1/s, smoothing of the hand distance rate
HOLD_DECAY = 3.0               # 1/s, decay of a derivative that cannot be updated this frame
TRACKING_LOST_AFTER = 0.35     # s without any hand -> TRACKING_LOST
RECENT_HANDS = 0.3             # s, "a hand was seen recently" for tracking_ok
ENERGY_TAU = 0.3
STABILITY_TAU = 0.4

# Body scale
SCALE_RATE = 2.0               # 1/s, smoothing of S
SCALE_FAST_RATE = 6.0          # 1/s, low-lag S used to normalise hand sizes (walking robustness)
SCALE_JUMP = math.log(1.4)     # single-frame jumps larger than 40 % are ignored ...
SCALE_JUMP_CONFIRM = 0.25      # ... unless they persist this long
SCALE_POSE_HOLD = 1.5          # s to keep a pose-based S when the pose is briefly lost
HAND_SIZE_PER_S = 1.0 / 3.8    # default mean hand size / S
FRAME_SCALE = 0.22             # S = 0.22 x scene height when nothing else is known
MIN_YAW_COS = 0.6              # limit of the shoulder-width rotation correction
ASSUMED_FOCAL = 0.75           # webcam focal length / frame width (~67 deg FOV), for forward speed
REFERENCE_DEPTH = 1.6          # m: approach cues are normalised to a user standing this far away
BODY_SCALE_M = 0.36            # m: typical S (shoulder width / 0.75 torso) in world units
NOMINAL_PALM_M = 0.1           # m: converts the fitted hand scale (px/m) to a palm-like size in px
FIT_MAX_RESIDUAL = 0.3         # reject hand scale fits whose residual exceeds this fraction of the palm
RIGID_PALM = (0, 1, 5, 9, 13, 17)  # wrist, thumb CMC, finger MCPs: the rigid part of the hand
_RIGID_IDX = np.array(RIGID_PALM)
_PALM_IDX = np.array(PALM_INDICES)
_TIP_IDX = np.array(FINGERTIP_INDICES)

# Detectors
DOMINANT_MIN = 0.6             # S/s for MotionSummary.dominant
THRUST_REFRACTORY = 0.8
LATERAL_EXT_GAIN = (0.08, 0.30)  # arm extension gain range of the lateral cue (S-independent ratio)
LATERAL_REACH = (0.70, 0.88)   # final arm extension (3D, else 2D) a lateral push must reach
LATERAL_UP_PENALTY = 0.45      # share of the lateral score removed for motions straight up
# Approach cue ranges for the rotation-invariant hand scale at REFERENCE_DEPTH. The ranges
# in ARCHITECTURE §7 (0.08-0.30 log growth, 0.4-1.5 /s) assume the 2D palm size, which
# also grows when the palm turns to the camera; true depth growth of a 0.3 m push from
# 1.35 m is ~0.25, so the ranges are scaled to give the documented response for it.
APPROACH_GAIN = (0.05, 0.20)
APPROACH_SPEED = (0.3, 1.1)
# 3D arm-extension gain over the thrust window that a frontal push needs (soft gate). A
# charge pose has extension <= ~0.82 (see the charge score), a push ends near 1.0.
FRONTAL_EXT_GATE = (0.06, 0.18)
FORWARD_SPEED = (1.0, 3.5)      # S/s of the wrists towards the camera (world landmarks): speed gate
FORWARD_SPEED_SMOOTH = 15.0     # 1/s
BURST_OUTWARD_MIN = 0.35       # S each hand must move away from the start midpoint (or opposite motion)
# Peak hand-distance rate (S/s): minimum at sensitivity 0 / 1, and the score ramp. Flinging
# the arms apart peaks at 13-22 S/s; just relaxing the hands from the charge pose into a
# ready stance peaks at 3.5-7 S/s (and at close range perspective inflates the hand gap to
# > 2 S), so the 2.5-4 S/s of ARCHITECTURE §7 would turn every relaxed release into a burst.
BURST_MIN_RATE = (8.0, 5.0)
# Final hand distance (S) of a burst at sensitivity 0 / 1. A real burst ends with the arms
# flung out to the sides (3.6-4.2 S), a clap opens to 1.7-2.5 S (0.45-0.6 m swings); the
# 2.0 S of ARCHITECTURE §7's detail turned every lively clap into an explosion.
BURST_MIN_DISTANCE = (2.8, 2.2)
BURST_RATE_RAMP = (4.0, 10.0)
BURST_WINDOW = 0.45
BURST_REFRACTORY = 1.0
THRUST_MIN_THRESHOLD = 0.25    # safety floor for calibrated thrust thresholds
ARMS_EXTENDED_HOLD = 0.25      # s an "arms extended" decision survives missing data
# arms_extended: mean 3D extension of the arms that do not point at the floor (elbow
# angle >~125 deg; a push ends at 0.9-1.0, relaxed bent arms are at 0.70-0.90), the 2D
# extension only without 3D data, and arms pointing down this much never count.
ARMS_EXTENDED_3D = 0.88
ARMS_EXTENDED_2D = 0.92
ARM_POINTING_MAX_DOWN = 0.85
POSE_CORRECTION_KEEP = 3.0     # s a learned pose->hand-model correction stays valid
POSE_OFFSET_FADE = 1.0         # s over which the learned centre offset fades out


# ----------------------------------------------------------------- thresholds


@dataclass(frozen=True)
class GestureThresholds:
    """Effective detector thresholds for a sensitivity (and optional calibration)."""

    together: float        # T: hand distance (S) below which the hands count as together
    charge_on: float       # charge pose score that activates the pose
    charge_off: float      # ... and that deactivates it again (hysteresis)
    thrust: float          # thrust score that fires a THRUST event
    burst_rate: float      # minimum peak hand-distance rate (S/s) of a burst
    burst_score: float     # burst score that fires a BURST event
    burst_distance: float = 2.0  # hand distance (S) the hands must have flown apart to


def _valid_calibration(calibration: Calibration | None) -> Calibration | None:
    return calibration if calibration is not None and calibration.valid else None


def compute_thresholds(settings: GestureSettings, calibration: Calibration | None = None) -> GestureThresholds:
    """Thresholds as documented in ARCHITECTURE §7; all move monotonically with sensitivity."""
    s = float(settings.sensitivity)
    s = clamp(s) if math.isfinite(s) else 0.5   # a hand-built NaN must not disable every detector
    cal = _valid_calibration(calibration)
    together = settings.together_distance * lerp(0.85, 1.25, s)
    if cal is not None and cal.together_distance > 0.0:
        together = max(together, 1.6 * cal.together_distance)
    charge_on = lerp(0.6, 0.45, s)
    thrust = lerp(0.72, 0.40, s)
    if cal is not None and cal.thrust_peak > 0.0:
        thrust = max(THRUST_MIN_THRESHOLD, min(thrust, 0.7 * cal.thrust_peak))
    return GestureThresholds(together=together, charge_on=charge_on, charge_off=charge_on - 0.25,
                             thrust=thrust, burst_rate=lerp(BURST_MIN_RATE[0], BURST_MIN_RATE[1], s),
                             burst_score=lerp(0.8, 0.45, s),
                             burst_distance=lerp(BURST_MIN_DISTANCE[0], BURST_MIN_DISTANCE[1], s))


# ----------------------------------------------------------------- outputs


@dataclass
class BodyFeatures(BodyState):
    """:class:`BodyState` plus raw per-frame measurements (calibration, debug overlay)."""

    shoulder_width_px: float = 0.0    # yaw-corrected shoulder width this frame (0 = not visible)
    torso_length_px: float = 0.0      # shoulder midpoint to hip midpoint (0 = hips not visible)
    scale_source: str = "none"        # "pose" | "held" | "hands" | "frame" | "none"


@dataclass
class DetectorDebug:
    """Intermediate detector values of the last frame (for the debug overlay and tests)."""

    charge_parts: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # distance, position, bent, still
    frontal: float = 0.0
    lateral: float = 0.0
    frontal_cues: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)  # approach, ext3, fwd, z, speed
    lateral_cues: tuple[float, float, float] = (0.0, 0.0, 0.0)  # displacement, extension, speed
    thrust_gate: float = 0.0
    frontal_gate: float = 0.0       # 3D arm-extension gate applied to the frontal score
    lateral_prior: float = 1.0      # down/up-motion prior applied to the lateral score
    lateral_reach: float = 1.0      # final-extension / hanging-arm gate applied to the lateral score
    burst: float = 0.0


# ----------------------------------------------------------------- history


_SIDE_COLS = ("present", "family", "cx", "cy", "lnr", "lnr_raw", "lnr_seg", "ext2", "ext3", "fwd", "zrel", "zkind",
              "zspeed", "speed", "approach")
_COLUMNS = ("t", "epoch", "S", "both", "mid_x", "mid_y", "d", "d_rate", "speed_mean", "approach_mean") + tuple(
    f"{p}_{c}" for p in ("l", "r") for c in _SIDE_COLS)
_FAMILY_NONE, _FAMILY_MODEL, _FAMILY_POSE = 0.0, 1.0, 2.0


class _History:
    """Time series of per-frame features in NumPy columns (oldest row first)."""

    def __init__(self, capacity: int = 1024) -> None:
        self.col = {name: i for i, name in enumerate(_COLUMNS)}
        self._buf = np.full((capacity, len(_COLUMNS)), np.nan)
        self._start = 0
        self._end = 0

    def __len__(self) -> int:
        return self._end - self._start

    def clear(self) -> None:
        self._start = self._end = 0

    def new_row(self) -> np.ndarray:
        return np.full(len(_COLUMNS), np.nan)

    def append(self, row: np.ndarray, max_age: float) -> None:
        if self._end >= self._buf.shape[0]:
            if self._start == 0:
                # Full of rows younger than ``max_age`` (absurd frame rates or a frozen
                # timestamp source): drop the oldest quarter instead of overflowing.
                self._start = self._buf.shape[0] // 4
            n = self._end - self._start
            self._buf[:n] = self._buf[self._start:self._end].copy()
            self._start, self._end = 0, n
        self._buf[self._end] = row
        self._end += 1
        t = self._buf[self._start:self._end, 0]
        drop = int(np.searchsorted(t, row[0] - max_age, side="left"))
        if drop > 0:
            self._start += min(drop, self._end - self._start - 1)

    def rows(self) -> np.ndarray:
        return self._buf[self._start:self._end]

    def window(self, t_min: float, epoch: float) -> np.ndarray:
        """Rows with ``t >= t_min`` of the given scale epoch (a contiguous tail)."""
        rows = self.rows()
        if rows.shape[0] == 0:
            return rows
        i0 = int(np.searchsorted(rows[:, 0], t_min, side="left"))
        i1 = int(np.searchsorted(rows[:, 1], epoch, side="left"))
        return rows[max(i0, i1):]

    def reference(self, t_now: float, epoch: float) -> np.ndarray | None:
        """Newest row at least 2 frames and ``DIFF_MIN_DT`` older than ``t_now`` (same epoch).

        After a gap in the frame stream (tracker stall, dropped frames) the row two frames
        back may be older than ``DIFF_MAX_DT`` while the previous row alone already spans
        more than ``DIFF_MIN_DT``; the previous row is used then, so a push that happens
        during the gap is still measured instead of being lost.
        """
        rows = self.rows()
        n = rows.shape[0]
        if n < 1:
            return None
        if n >= 2:
            j = int(np.searchsorted(rows[:n - 1, 0], t_now - DIFF_MIN_DT, side="right")) - 1
            if j >= 0 and rows[j, 1] == epoch and t_now - rows[j, 0] <= DIFF_MAX_DT:
                return rows[j]
        row = rows[n - 1]
        if row[1] == epoch and DIFF_MIN_DT <= t_now - row[0] <= DIFF_MAX_DT:
            return row
        return None


# ----------------------------------------------------------------- body scale


class _BodyScale:
    """Body scale S with source hierarchy, jump rejection and slow smoothing."""

    _RANK = {"frame": 0, "hands": 1, "pose": 2}

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0          # smoothed S (px)
        self.fast = 0.0           # low-lag S used to normalise hand sizes
        self.base = "none"        # source of the current estimate
        self.source = "none"      # source this frame ("held" when not measured)
        self.epoch = 0            # increments whenever S had to jump (snap)
        self._last_pose_t = -math.inf
        self._last_measure_t = -math.inf
        self._jump_since: float | None = None
        self._jump_sign = 0

    def update(self, t: float, dt: float, pose_raw: float | None, hands_raw: float | None,
               scene_h: int) -> float:
        measure: float | None = None
        src = "held"
        if pose_raw is not None:
            measure, src = pose_raw, "pose"
            self._last_pose_t = t
        elif self.base == "pose" and t - self._last_pose_t < SCALE_POSE_HOLD:
            measure = None    # the pose is briefly gone: hand-based S is less reliable, keep S
        elif hands_raw is not None:
            measure, src = hands_raw, "hands"
        elif self.value > 0.0 and t - self._last_measure_t < SCALE_POSE_HOLD:
            measure = None
        else:
            measure, src = FRAME_SCALE * max(1, scene_h), "frame"
        self.source = src
        if measure is None or not math.isfinite(measure) or measure <= 1.0:
            self.source = "held" if self.value > 0.0 else "none"
            return self.value
        self._last_measure_t = t
        if self.value <= 0.0 or self._RANK[src] > self._RANK.get(self.base, -1):
            self._snap(measure, src)       # first estimate, or a better source than before
            return self.value
        self.base = src
        jump = math.log(measure / self.value)
        if abs(jump) > SCALE_JUMP:
            sign = 1 if jump > 0 else -1
            if self._jump_since is None or self._jump_sign != sign:
                self._jump_since, self._jump_sign = t, sign
            elif t - self._jump_since >= SCALE_JUMP_CONFIRM:
                self._snap(measure, src)   # persistent change (e.g. after re-entering the view)
            return self.value
        self._jump_since = None
        self.value = damp(self.value, measure, SCALE_RATE, dt)
        self.fast = damp(self.fast, measure, SCALE_FAST_RATE, dt)
        return self.value

    def _snap(self, measure: float, src: str) -> None:
        self.value = self.fast = measure
        self.base = src
        self.epoch += 1
        self._jump_since = None


# ----------------------------------------------------------------- per-side helpers


@dataclass
class _PoseHandCorrection:
    """Learned relation between the pose-based and the hand-model palm estimate of one side."""

    ratio: float = 1.0            # hand-model size / pose-based size
    offset: tuple[float, float] = (0.0, 0.0)  # (model centre - pose centre) / model size
    learned_t: float = -math.inf
    count: int = 0

    def learn(self, t: float, dt: float, model_c: Vec2, model_s: float, pose_c: Vec2, pose_s: float) -> None:
        if pose_s <= 1e-6 or model_s <= 1e-6:
            return
        ratio = model_s / pose_s
        off = ((model_c[0] - pose_c[0]) / model_s, (model_c[1] - pose_c[1]) / model_s)
        if not 0.5 <= ratio <= 5.0 or math.hypot(*off) > 1.5:
            return  # implausible: probably not the same hand
        if self.count == 0 or t - self.learned_t > POSE_CORRECTION_KEEP:
            self.ratio, self.offset = ratio, off
        else:
            k = 1.0 - math.exp(-4.0 * max(dt, 0.0))
            self.ratio += (ratio - self.ratio) * k
            self.offset = (self.offset[0] + (off[0] - self.offset[0]) * k,
                           self.offset[1] + (off[1] - self.offset[1]) * k)
        self.count += 1
        self.learned_t = t

    def valid(self, t: float) -> bool:
        return self.count > 0 and t - self.learned_t <= POSE_CORRECTION_KEEP


@dataclass
class _SideMeasure:
    """Raw measurements of one screen side for the current frame."""

    side: str
    family: float = _FAMILY_NONE
    source: str = "none"
    centre: Vec2 | None = None
    size: float = 0.0
    model_size: float = 0.0       # hand-model palm size (0 for pose fallback)
    depth_size: float = 0.0       # rotation-invariant apparent size for approach cues (px)
    depth_kind: int = 0           # 1 = fitted to world landmarks, 2 = 2D palm size
    openness: float | None = None
    hand_wrist: Vec2 | None = None
    shoulder: Vec2 | None = None
    elbow: Vec2 | None = None
    wrist: Vec2 | None = None
    ext2: float | None = None
    ext3: float | None = None
    forward: float | None = None
    zrel: float | None = None     # wrist depth relative to the shoulder midpoint (body scales, < 0 = in front)
    zkind: float = 0.0            # 1 = from metric world landmarks, 2 = from image z (px scale)
    down: float | None = None     # vertical component of the shoulder->wrist direction (hanging arm)
    pose_centre: Vec2 | None = None  # palm estimate from the pose wrist/index/pinky points
    pose_size: float = 0.0
    pose_wrist: Vec2 | None = None


def _finite(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all())


def _hand_landmarks(hand: HandObservation | None) -> np.ndarray | None:
    if hand is None:
        return None
    try:
        lm = np.asarray(hand.landmarks, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if lm.ndim != 2 or lm.shape[0] < 21 or lm.shape[1] < 2 or not _finite(lm[:21, :2]):
        return None
    return lm[:21]


def palm_size(landmarks: np.ndarray) -> float:
    """Rotation-robust palm size: max(|wrist - middle MCP|, 1.25 |index MCP - pinky MCP|)."""
    a = math.hypot(float(landmarks[0, 0] - landmarks[9, 0]), float(landmarks[0, 1] - landmarks[9, 1]))
    b = 1.25 * math.hypot(float(landmarks[5, 0] - landmarks[17, 0]), float(landmarks[5, 1] - landmarks[17, 1]))
    return max(a, b)


def hand_openness(landmarks: np.ndarray, size: float) -> float:
    """0 fist .. 1 open palm from the mean fingertip distance to the palm centre."""
    if size <= 1e-6:
        return 0.0
    centre = landmarks[_PALM_IDX, :2].sum(axis=0) / len(_PALM_IDX)
    d = landmarks[_TIP_IDX, :2] - centre
    mean_dist = float(np.sqrt((d * d).sum(axis=1)).sum()) / len(_TIP_IDX)
    return remap(mean_dist / size, OPENNESS_RANGE[0], OPENNESS_RANGE[1])


def hand_scale(landmarks: np.ndarray, world: np.ndarray | None,
               scene_size: tuple[int, int] | None = None, palm: float | None = None) -> float | None:
    """Apparent hand scale in px per metre, independent of how the hand is turned.

    MediaPipe's hand world landmarks are metric and camera-aligned (x right, y down;
    verified on sample images: the fit residual is as low as a full rotation fit and the
    scale is unchanged when the input image is rotated), so the 2D landmarks of the rigid
    palm are close to ``s * world_xy`` with ``s = focal / depth``. The least-squares ``s``
    measures movement towards the camera, whereas the 2D palm size also grows when a
    palm merely turns towards the camera.

    Away from the image centre perspective shifts palm points that are nearer to the
    camera outwards; with ``scene_size`` every point is first re-projected to the palm's
    centroid depth (two fixed-point steps using the world z, the image centre as
    principal point and the assumed webcam focal length), which removes that bias
    (up to ~18 % for hands near the frame edge at 1 m otherwise).
    Returns ``None`` when the world landmarks are missing or do not fit.
    ``palm`` (the 2D palm size, if already known) is used for the fit-quality check.
    """
    if world is None:
        return None
    try:
        w = np.asarray(world, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if w.ndim != 2 or w.shape[0] < 21 or w.shape[1] < 3:
        return None
    wp = w[_RIGID_IDX, :3]
    if not _finite(wp):
        return None
    n = float(len(_RIGID_IDX))
    y = wp[:, :2] - wp[:, :2].sum(axis=0) / n
    dz = wp[:, 2] - wp[:, 2].sum() / n
    denom = float((y * y).sum())
    if denom < 1e-8:
        return None
    pts = landmarks[_RIGID_IDX, :2]
    x = pts - pts.sum(axis=0) / n
    s = float((x * y).sum()) / denom
    if scene_size is not None and scene_size[0] > 0 and s > 0.0:
        rel = pts - (scene_size[0] * 0.5, scene_size[1] * 0.5)
        focal = ASSUMED_FOCAL * scene_size[0]
        for _ in range(2):
            corrected = rel * (1.0 + dz * (s / focal))[:, None]
            x = corrected - corrected.sum(axis=0) / n
            s = float((x * y).sum()) / denom
    if not math.isfinite(s) or s <= 0.0:
        return None
    r = x - s * y
    residual = math.sqrt(float((r * r).sum()) / n)
    if residual > FIT_MAX_RESIDUAL * max(palm if palm is not None else palm_size(landmarks), 1e-6):
        return None
    return s


class _PoseView:
    """Validated arrays of a PoseObservation (bad/missing values become invisible)."""

    def __init__(self, pose: PoseObservation) -> None:
        lm = np.asarray(pose.landmarks, dtype=np.float64)
        if lm.ndim != 2 or lm.shape[0] < 33 or lm.shape[1] < 2:
            raise ValueError(f"pose landmarks have shape {lm.shape}")
        self.xy = lm[:33, :2]
        self.z = lm[:33, 2] if lm.shape[1] >= 3 else np.zeros(33)
        vis = np.asarray(pose.visibility, dtype=np.float64).reshape(-1)
        self.vis = vis[:33] if vis.shape[0] >= 33 else np.ones(33)
        bad = ~np.isfinite(self.xy).all(axis=1) | ~np.isfinite(self.vis)
        self.vis = np.where(bad, 0.0, self.vis)
        self.z = np.where(np.isfinite(self.z), self.z, np.nan)
        world = None
        self._world_ok = np.zeros(33, dtype=bool)
        if pose.world is not None:
            w = np.asarray(pose.world, dtype=np.float64)
            if w.ndim == 2 and w.shape[0] >= 33 and w.shape[1] >= 3:
                world = w[:33, :3]
                self._world_ok = np.isfinite(world).all(axis=1) & (self.vis >= VISIBLE)
        self.world = world
        self.obs = pose

    def visible(self, i: int, threshold: float = VISIBLE) -> bool:
        return bool(self.vis[i] >= threshold)

    def point(self, i: int) -> Vec2:
        return (float(self.xy[i, 0]), float(self.xy[i, 1]))

    def wpoint(self, i: int) -> np.ndarray | None:
        """World point (metres) of a visible landmark, or None."""
        return self.world[i] if self._world_ok[i] else None


def _norm3(v: np.ndarray) -> float:
    return math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)


def _dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _mid(a: Vec2, b: Vec2) -> Vec2:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _unit(v: Vec2, fallback: Vec2 = (1.0, 0.0)) -> Vec2:
    n = math.hypot(v[0], v[1])
    return (v[0] / n, v[1] / n) if n > 1e-9 else fallback


def _nanmean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    return sum(vals) / len(vals) if vals else None


def _f(x: float | None) -> float:
    return float("nan") if x is None else float(x)


def _hang(m: _SideMeasure) -> float:
    """0..1: how much an arm hangs straight down at the side (relaxed, lowered arm)."""
    e = m.ext3 if m.ext3 is not None else m.ext2
    if e is None or m.down is None:
        return 0.0
    return smoothstep(0.88, 0.96, e) * smoothstep(0.80, 0.92, m.down)



# ----------------------------------------------------------------- engine


class GestureEngine:
    """Features, detectors and events from tracking frames (see module docstring)."""

    def __init__(self, settings: GestureSettings, calibration: Calibration | None = None) -> None:
        self._settings = settings
        self._calibration = _valid_calibration(calibration)
        self.thresholds = compute_thresholds(settings, self._calibration)
        self.debug = DetectorDebug()
        self._history = _History()
        self._scale = _BodyScale()
        self._error_count = 0
        self.reset()

    # ------------------------------------------------------------ public API

    @property
    def settings(self) -> GestureSettings:
        return self._settings

    @property
    def calibration(self) -> Calibration | None:
        return self._calibration

    def set_settings(self, settings: GestureSettings) -> None:
        """Apply new settings (thresholds change immediately, state is kept)."""
        self._settings = settings
        self.thresholds = compute_thresholds(settings, self._calibration)

    def set_calibration(self, calibration: Calibration | None) -> None:
        """Use a per-user calibration (``None`` or an invalid one = defaults)."""
        self._calibration = _valid_calibration(calibration)
        self.thresholds = compute_thresholds(self._settings, self._calibration)

    def reset(self) -> None:
        """Forget all history and state (e.g. after switching cameras)."""
        self._history.clear()
        self._scale.reset()
        self._corrections = {s: _PoseHandCorrection() for s in SIDES}
        self._velocity: dict[str, Vec2] = {s: (0.0, 0.0) for s in SIDES}
        self._approach: dict[str, float] = {s: 0.0 for s in SIDES}
        self._openness: dict[str, float] = {s: 0.0 for s in SIDES}
        self._lnr_kind: dict[str, int] = {s: 0 for s in SIDES}
        self._lnr_seg: dict[str, float] = {s: 0.0 for s in SIDES}
        self._lnr_smooth: dict[str, float | None] = {s: None for s in SIDES}
        self._zspeed: dict[str, float] = {s: 0.0 for s in SIDES}
        self._d_rate = 0.0
        self._swapped = False
        self._last_t: float | None = None
        self._last_frame_id: int | None = None
        self._scene_size: tuple[int, int] | None = None
        self._last_frame_now: float | None = None
        self._last_hands_now: float | None = None
        self._tracking_found = False
        self._body: BodyFeatures | None = None
        self._signals = GestureSignals()
        self._charge_active = False
        self._thrust_armed = True
        self._last_thrust_t = -math.inf
        self._burst_armed = True
        self._last_burst_t = -math.inf
        self._energy = 0.0
        self._stability = 0.0
        self._aim_dir: Vec2 = (1.0, 0.0)
        self._last_lateral_dir: Vec2 = (1.0, 0.0)
        self._aim_frontal = 0.0
        self._baseline_samples: deque[float] = deque(maxlen=150)
        self._baseline: float | None = None
        self._baseline_epoch = -1
        self._arms_extended_t = -math.inf
        self._world_scale = BODY_SCALE_M     # body scale in metres from the world landmarks
        self.debug = DetectorDebug()

    def update(self, frame: TrackingFrame | None,
               now: float) -> tuple[BodyState | None, GestureSignals, list[GestureEvent]]:
        """Process a new tracking frame, or advance timeouts with ``frame=None``.

        Returns ``(body, signals, events)``. ``body`` is the latest :class:`BodyFeatures`
        (the previous one for ``frame=None``) or ``None`` before the first frame.
        """
        try:
            if frame is None:
                return self._advance(now)
            return self._process(frame, now)
        except Exception:  # noqa: BLE001 - a bad frame must never crash the main loop
            self._error_count += 1
            if self._error_count <= 3 or self._error_count % 300 == 0:
                log.exception("Gesture processing failed (%d errors so far); state reset", self._error_count)
            self.reset()
            return None, GestureSignals(), []

    # ------------------------------------------------------------ no new frame

    def _advance(self, now: float) -> tuple[BodyState | None, GestureSignals, list[GestureEvent]]:
        events: list[GestureEvent] = []
        if self._last_frame_now is None:
            return None, dataclasses.replace(self._signals), events
        if now - self._last_frame_now >= TRACKING_LOST_AFTER:
            # no data at all: the tracker stalled or the camera stopped
            if self._charge_active:
                self._charge_active = False
                events.append(GestureEvent(GestureEventType.CHARGE_POSE_END, now, strength=0.0))
            if self._tracking_found:
                self._tracking_found = False
                events.append(GestureEvent(GestureEventType.TRACKING_LOST, now, strength=0.0))
            sig = self._signals
            self._signals = dataclasses.replace(sig, charge_pose=0.0, charge_pose_active=False, thrust=0.0,
                                                burst=0.0, arms_extended=False, tracking_ok=False)
        elif (self._tracking_found and self._last_hands_now is not None
              and now - self._last_hands_now >= TRACKING_LOST_AFTER):
            # frames still arrive (slowly) but none had a hand for TRACKING_LOST_AFTER:
            # report the loss now instead of waiting for the next frame
            self._tracking_found = False
            events.append(GestureEvent(GestureEventType.TRACKING_LOST, now, strength=0.0))
            self._signals = dataclasses.replace(self._signals, tracking_ok=False)
        elif self._signals.tracking_ok and self._body is not None and not self._body.both_hands:
            recent = self._last_hands_now is not None and now - self._last_hands_now < RECENT_HANDS
            if not recent:
                self._signals = dataclasses.replace(self._signals, tracking_ok=False)
        return self._body, dataclasses.replace(self._signals), events

    # ------------------------------------------------------------ frame processing

    def _process(self, frame: TrackingFrame, now: float) -> tuple[BodyState | None, GestureSignals, list[GestureEvent]]:
        t = float(frame.capture_time)
        if not math.isfinite(t):
            t = float(now)
        if self._last_t is not None:
            if frame.frame_id == self._last_frame_id and abs(t - self._last_t) < 1e-9:
                return self._advance(now)  # the same frame again
            if t <= self._last_t:
                if self._last_t - t > 0.5:
                    log.info("Tracking timestamps went backwards by %.2f s; restarting gesture history",
                             self._last_t - t)
                    self._restart_history()
                else:
                    # A repeated or slightly older capture time (broken timestamp source):
                    # advance by the time that really passed between the deliveries, so
                    # speeds are not inflated by a fake 1 ms frame interval.
                    elapsed = now - self._last_frame_now if self._last_frame_now is not None else 0.0
                    t = self._last_t + (min(elapsed, 0.5) if math.isfinite(elapsed) and elapsed > 1e-3 else 1e-3)
        scene = (max(1, int(frame.scene_size[0])), max(1, int(frame.scene_size[1])))
        if self._scene_size is not None and scene != self._scene_size:
            log.info("Scene size changed %s -> %s; restarting gesture history", self._scene_size, scene)
            self._restart_history()
        self._scene_size = scene
        dt = 0.0 if self._last_t is None else min(t - self._last_t, 0.5)
        self._last_t = t
        self._last_frame_id = frame.frame_id
        self._last_frame_now = now

        pose = None
        if frame.pose is not None:
            try:
                pose = _PoseView(frame.pose)
            except (TypeError, ValueError) as exc:
                log.debug("Ignoring malformed pose: %s", exc)
        hands = {s: frame.hand(s) for s in SIDES}
        model = {s: (_hand_landmarks(h) if h is not None else None) for s, h in hands.items()}

        arms = self._arm_mapping(pose, hands, model)
        body_geo = self._body_geometry(pose, arms)
        sides = {s: self._measure_side(s, pose, arms.get(s), hands[s], model[s]) for s in SIDES}

        # ---- body scale
        model_sizes = [m.model_size for m in sides.values() if m.family == _FAMILY_MODEL and m.model_size > 0]
        hands_raw = None
        if model_sizes:
            cal = self._calibration
            ratio = cal.hand_size_ratio if cal is not None and cal.hand_size_ratio > 0.02 else HAND_SIZE_PER_S
            hands_raw = (sum(model_sizes) / len(model_sizes)) / ratio
        S = self._scale.update(t, dt, body_geo["scale_raw"], hands_raw, scene[1])
        if S <= 0.0:
            S = FRAME_SCALE * max(1, scene[1])
        S_fast = self._scale.fast if self._scale.fast > 0 else S
        epoch = float(self._scale.epoch)

        # ---- pose fallback sizes need S; learn the pose->model correction
        for m in sides.values():
            self._finish_side(m, t, dt, S)

        events: list[GestureEvent] = []
        body = self._build_body(frame, t, dt, now, S, S_fast, epoch, pose, body_geo, sides)
        self._update_tracking_state(body, now, events)
        signals = self._detect(body, t, dt, now, S, epoch, body_geo, sides, events)
        self._body = body
        self._signals = signals
        return body, dataclasses.replace(signals), events

    def _restart_history(self) -> None:
        """Drop all kinematic state (time base or scene size changed); events state is kept."""
        self._history.clear()
        self._scale.reset()
        self._velocity = {s: (0.0, 0.0) for s in SIDES}
        self._approach = {s: 0.0 for s in SIDES}
        self._zspeed = {s: 0.0 for s in SIDES}
        self._lnr_smooth = {s: None for s in SIDES}
        self._corrections = {s: _PoseHandCorrection() for s in SIDES}
        self._d_rate = 0.0
        self._last_t = None
        self._baseline = None
        self._baseline_samples.clear()
        # Detector timestamps live in the old time base: a refractory period that ends
        # "in the future" of the new time base would block thrusts/bursts for that long.
        self._last_thrust_t = -math.inf
        self._last_burst_t = -math.inf
        self._arms_extended_t = -math.inf

    # ------------------------------------------------------------ measurement

    def _arm_mapping(self, pose: _PoseView | None, hands: dict[str, HandObservation | None],
                     model: dict[str, np.ndarray | None]) -> dict[str, ArmIndices]:
        """Pose arm per hand slot. Normally ``pose.arm(side)``; swapped when the hand slots
        clearly belong to the other arm (crossed arms, slot swap in the tracker)."""
        if pose is None:
            return {}
        base = {s: pose.obs.arm(s) for s in SIDES}
        wrists = {s: (pose.point(base[s].wrist) if pose.visible(base[s].wrist) else None) for s in SIDES}
        hand_w = {s: ((float(model[s][0, 0]), float(model[s][0, 1]))
                      if model[s] is not None and hands[s] is not None and hands[s].source == "hand" else None)
                  for s in SIDES}
        evidence: float | None = None  # > 1: swapping fits better, < 1: identity fits better
        if wrists["left"] is not None and wrists["right"] is not None:
            if hand_w["left"] is not None and hand_w["right"] is not None:
                same = _dist(hand_w["left"], wrists["left"]) + _dist(hand_w["right"], wrists["right"])
                swap = _dist(hand_w["left"], wrists["right"]) + _dist(hand_w["right"], wrists["left"])
                evidence = (same + 1.0) / (swap + 1.0)
            else:
                for s, o in (("left", "right"), ("right", "left")):
                    if hand_w[s] is not None:
                        evidence = (_dist(hand_w[s], wrists[s]) + 1.0) / (_dist(hand_w[s], wrists[o]) + 1.0)
        if evidence is not None:
            if not self._swapped and evidence > 2.5:
                self._swapped = True
            elif self._swapped and evidence < 0.4:
                self._swapped = False
        elif wrists["left"] is None and wrists["right"] is None:
            self._swapped = False
        if self._swapped:
            return {"left": base["right"], "right": base["left"]}
        return base

    def _body_geometry(self, pose: _PoseView | None, arms: dict[str, ArmIndices]) -> dict:
        geo: dict = {"scale_raw": None, "shoulders": {}, "hips": {}, "shoulder_mid": None, "hip_mid": None,
                     "shoulder_width": 0.0, "torso": 0.0, "pose_ok": False}
        if pose is None or not arms:
            return geo
        idx = {s: pose.obs.arm(s) for s in SIDES}  # body landmarks by screen side (not slot-mapped)
        shoulders = {s: pose.point(idx[s].shoulder) for s in SIDES if pose.visible(idx[s].shoulder)}
        hips = {s: pose.point(idx[s].hip) for s in SIDES if pose.visible(idx[s].hip)}
        wrists_visible = any(pose.visible(idx[s].wrist) for s in SIDES)
        geo["pose_ok"] = bool(shoulders) or wrists_visible
        geo["shoulders"], geo["hips"] = shoulders, hips
        candidates = []
        if len(shoulders) == 2:
            width = _dist(shoulders["left"], shoulders["right"])
            # Undo the foreshortening of a turned torso with the world landmarks' shoulder depth.
            wl, wr = pose.wpoint(idx["left"].shoulder), pose.wpoint(idx["right"].shoulder)
            if wl is not None and wr is not None:
                dx, dz = abs(float(wl[0] - wr[0])), float(wl[2] - wr[2])
                horiz = math.hypot(dx, dz)
                if horiz > 1e-6:
                    width /= max(dx / horiz, MIN_YAW_COS)
            geo["shoulder_width"] = width
            geo["shoulder_mid"] = _mid(shoulders["left"], shoulders["right"])
            candidates.append(width)
        if len(hips) == 2:
            geo["hip_mid"] = _mid(hips["left"], hips["right"])
            if geo["shoulder_mid"] is not None:
                torso = _dist(geo["shoulder_mid"], geo["hip_mid"])
                geo["torso"] = torso
                candidates.append(0.75 * torso)
        if candidates:
            geo["scale_raw"] = max(candidates)
        # the same body scale in metres from the world landmarks (normalises world depths)
        ws_l, ws_r = pose.wpoint(idx["left"].shoulder), pose.wpoint(idx["right"].shoulder)
        world_cands = []
        if ws_l is not None and ws_r is not None:
            world_cands.append(_norm3(ws_l - ws_r))
            wh_l, wh_r = pose.wpoint(idx["left"].hip), pose.wpoint(idx["right"].hip)
            if wh_l is not None and wh_r is not None:
                world_cands.append(0.75 * _norm3(0.5 * (ws_l + ws_r) - 0.5 * (wh_l + wh_r)))
        if world_cands and 0.15 < max(world_cands) < 0.8:
            target = max(world_cands)
            if self._world_scale <= 0.0:
                self._world_scale = target
            else:
                self._world_scale += 0.1 * (target - self._world_scale)
        return geo

    def _measure_side(self, side: str, pose: _PoseView | None, arm: ArmIndices | None,
                      hand: HandObservation | None, lm: np.ndarray | None) -> _SideMeasure:
        m = _SideMeasure(side)
        pose_wrist_visible = pose is not None and arm is not None and pose.visible(arm.wrist)
        if lm is not None and hand is not None and hand.source == "hand":
            m.family, m.source = _FAMILY_MODEL, "hand"
        elif pose_wrist_visible:
            m.family, m.source = _FAMILY_POSE, "pose"   # a real measurement beats extrapolation
        elif lm is not None and hand is not None:
            m.family, m.source = _FAMILY_MODEL, "predicted"
        if m.family == _FAMILY_MODEL and lm is not None:
            c = lm[_PALM_IDX, :2].sum(axis=0) / len(_PALM_IDX)
            m.centre = (float(c[0]), float(c[1]))
            m.model_size = m.size = palm_size(lm)
            m.hand_wrist = (float(lm[0, 0]), float(lm[0, 1]))
            m.openness = hand_openness(lm, m.size)
            fit = hand_scale(lm, hand.world, self._scene_size, m.size)
            if fit is not None:
                m.depth_size, m.depth_kind = fit * NOMINAL_PALM_M, 1
            else:
                m.depth_size, m.depth_kind = m.size, 2
        if pose is not None and arm is not None:
            if pose.visible(arm.shoulder):
                m.shoulder = pose.point(arm.shoulder)
            if pose.visible(arm.elbow):
                m.elbow = pose.point(arm.elbow)
            if pose_wrist_visible:
                m.wrist = pose.point(arm.wrist)
            elif m.hand_wrist is not None:
                m.wrist = m.hand_wrist
            if m.shoulder is not None and m.elbow is not None and m.wrist is not None:
                denom = _dist(m.shoulder, m.elbow) + _dist(m.elbow, m.wrist)
                if denom > 1e-6:
                    m.ext2 = clamp(_dist(m.shoulder, m.wrist) / denom)
                reach = _dist(m.shoulder, m.wrist)
                if reach > 1e-6:
                    m.down = (m.wrist[1] - m.shoulder[1]) / reach
            ws, we, ww = pose.wpoint(arm.shoulder), pose.wpoint(arm.elbow), pose.wpoint(arm.wrist)
            if ws is not None and ww is not None:
                sw = ww - ws
                reach3 = _norm3(sw)
                if reach3 > 1e-6:
                    m.forward = clamp(-float(sw[2]) / reach3, -1.0, 1.0)
                    m.down = float(sw[1]) / reach3
                    if we is not None:
                        denom3 = _norm3(we - ws) + _norm3(ww - we)
                        if denom3 > 1e-6:
                            m.ext3 = clamp(reach3 / denom3)
            if pose_wrist_visible and pose.visible(arm.shoulder):
                # Depth of the wrist relative to the shoulder midpoint (the own shoulder moves
                # back when the torso twists, which would look like the wrist moving forward).
                other = pose.obs.arm("right" if side == "left" else "left").shoulder
                wo = pose.wpoint(other)
                if ww is not None and ws is not None and wo is not None and self._world_scale > 0.0:
                    # metric world landmarks: MediaPipe's image z is exaggerated ~2-3x
                    m.zrel = (float(ww[2]) - 0.5 * float(ws[2] + wo[2])) / self._world_scale
                    m.zkind = 1.0
                else:
                    zs = pose.z[arm.shoulder]
                    if pose.visible(other) and math.isfinite(pose.z[other]):
                        zs = 0.5 * (zs + pose.z[other])
                    zw = pose.z[arm.wrist]
                    if math.isfinite(zw) and math.isfinite(zs):
                        m.zrel = float(zw - zs)   # px; normalised by S in _finish_side
                        m.zkind = 2.0
            if pose_wrist_visible:
                wrist = pose.point(arm.wrist)
                m.pose_wrist = wrist
                ix, pk = arm.index, arm.pinky
                if pose.visible(ix, POSE_HAND_VISIBLE) and pose.visible(pk, POSE_HAND_VISIBLE):
                    pi, pp = pose.point(ix), pose.point(pk)
                    m.pose_centre = ((wrist[0] + pi[0] + pp[0]) / 3.0, (wrist[1] + pi[1] + pp[1]) / 3.0)
                    m.pose_size = 1.1 * _dist(wrist, _mid(pi, pp))
                else:
                    m.pose_centre = wrist
                    m.pose_size = 0.0  # unknown: 0.25 S is used for a fallback hand
            if m.family == _FAMILY_POSE:
                m.centre, m.size = m.pose_centre, m.pose_size
        return m

    def _finish_side(self, m: _SideMeasure, t: float, dt: float, S: float) -> None:
        """Learn/apply the pose<->hand-model correction and finalise fallback sizes."""
        corr = self._corrections[m.side]
        if m.zrel is not None and m.zkind == 2.0:
            m.zrel = m.zrel / S
        if (m.source == "hand" and m.centre is not None and m.hand_wrist is not None and m.pose_wrist is not None
                and m.pose_centre is not None and m.pose_size > 0.0
                and _dist(m.pose_wrist, m.hand_wrist) < 1.2 * m.model_size):
            # Both models see the same hand: learn how the pose estimate relates to the hand
            # model, so a later switch to the pose fallback does not make the hand jump.
            corr.learn(t, dt, m.centre, m.model_size, m.pose_centre, m.pose_size)
        if m.family == _FAMILY_POSE and m.centre is not None:
            if m.size <= 1e-6:
                m.size = 0.25 * S
            elif corr.valid(t):
                fade = clamp(1.0 - (t - corr.learned_t) / POSE_OFFSET_FADE)
                m.size *= corr.ratio
                m.centre = (m.centre[0] + corr.offset[0] * m.size * fade,
                            m.centre[1] + corr.offset[1] * m.size * fade)
            m.size = min(max(m.size, 0.12 * S), 0.6 * S)
            m.openness = self._openness[m.side]  # hand model only: keep the last value

    # ------------------------------------------------------------ body state

    def _build_body(self, frame: TrackingFrame, t: float, dt: float, now: float, S: float, S_fast: float,
                    epoch: float, pose: _PoseView | None, geo: dict, sides: dict[str, _SideMeasure]) -> BodyFeatures:
        hist = self._history
        ref = hist.reference(t, epoch)
        c = hist.col
        row = hist.new_row()
        row[c["t"]], row[c["epoch"]], row[c["S"]] = t, epoch, S

        arm_states: dict[str, ArmState] = {}
        for side, m in sides.items():
            p = "l_" if side == "left" else "r_"
            present = m.centre is not None and m.family != _FAMILY_NONE
            st = ArmState(side=side)
            if present:
                st.present, st.source = True, m.source
                st.hand_center, st.hand_size = m.centre, float(m.size)
                st.shoulder, st.elbow, st.wrist = m.shoulder, m.elbow, m.wrist
                st.extension_2d = float(m.ext2) if m.ext2 is not None else 0.0
                st.extension_3d = m.ext3
                st.forward = float(m.forward) if m.forward is not None else 0.0
                if m.openness is not None:
                    st.openness = float(m.openness)
                    if m.source == "hand":
                        self._openness[side] = st.openness
                lnr = raw = float("nan")
                if m.family == _FAMILY_MODEL and m.depth_size > 0.0:
                    raw = math.log(m.depth_size / S_fast)
                    if m.depth_kind != self._lnr_kind[side]:
                        # fitted and 2D sizes are not comparable: start a new segment
                        self._lnr_kind[side] = m.depth_kind
                        self._lnr_seg[side] += 1.0
                        self._lnr_smooth[side] = None
                    prev = self._lnr_smooth[side]
                    lnr = raw if prev is None else damp(prev, raw, SIZE_SMOOTH, dt)
                    self._lnr_smooth[side] = lnr
                seg = self._lnr_seg[side]
                # velocity: difference over >= 2 frames, never across a source-family switch
                v_raw = None
                a_raw = None
                if ref is not None and ref[c[p + "present"]] > 0 and ref[c[p + "family"]] == m.family:
                    dtr = t - ref[c["t"]]
                    v_raw = ((m.centre[0] - ref[c[p + "cx"]]) / dtr, (m.centre[1] - ref[c[p + "cy"]]) / dtr)
                    if math.isfinite(lnr) and math.isfinite(ref[c[p + "lnr"]]) and ref[c[p + "lnr_seg"]] == seg:
                        a_raw = (lnr - ref[c[p + "lnr"]]) / dtr
                vx, vy = self._velocity[side]
                if v_raw is not None:
                    k = 1.0 - math.exp(-VELOCITY_RATE * dt) if dt > 0 else 1.0
                    vx, vy = vx + (v_raw[0] - vx) * k, vy + (v_raw[1] - vy) * k
                else:
                    vx, vy = damp(vx, 0.0, HOLD_DECAY, dt), damp(vy, 0.0, HOLD_DECAY, dt)
                self._velocity[side] = (vx, vy)
                ap = self._approach[side]
                if a_raw is not None:
                    ap = damp(ap, a_raw, APPROACH_RATE_SMOOTH, dt)
                else:
                    ap = damp(ap, 0.0, HOLD_DECAY * 2, dt)
                self._approach[side] = ap
                # forward speed of the wrist relative to the shoulders (world depth, S/s)
                zs_raw = None
                if (ref is not None and m.zrel is not None and m.zkind == 1.0 and ref[c[p + "zkind"]] == 1.0
                        and math.isfinite(ref[c[p + "zrel"]])):
                    zs_raw = -(m.zrel - ref[c[p + "zrel"]]) / (t - ref[c["t"]])
                zsp = self._zspeed[side]
                if zs_raw is not None:
                    zsp = damp(zsp, zs_raw, FORWARD_SPEED_SMOOTH, dt)
                else:
                    zsp = damp(zsp, 0.0, HOLD_DECAY * 2, dt)
                self._zspeed[side] = zsp
                st.velocity = (vx, vy)
                st.speed = math.hypot(vx, vy) / S
                st.approach_rate = ap if m.family == _FAMILY_MODEL else 0.0
                row[c[p + "present"]], row[c[p + "family"]] = 1.0, m.family
                row[c[p + "cx"]], row[c[p + "cy"]] = m.centre
                row[c[p + "lnr"]], row[c[p + "lnr_raw"]] = lnr, raw
                row[c[p + "lnr_seg"]] = seg
                row[c[p + "ext2"]], row[c[p + "ext3"]] = _f(m.ext2), _f(m.ext3)
                row[c[p + "fwd"]], row[c[p + "zrel"]] = _f(m.forward), _f(m.zrel)
                row[c[p + "zkind"]] = m.zkind
                row[c[p + "zspeed"]] = zsp if m.zkind == 1.0 else float("nan")
                row[c[p + "speed"]] = st.speed
                row[c[p + "approach"]] = st.approach_rate if m.family == _FAMILY_MODEL else float("nan")
            else:
                self._velocity[side] = (0.0, 0.0)
                self._approach[side] = 0.0
                self._openness[side] = 0.0
                self._lnr_smooth[side] = None
                self._zspeed[side] = 0.0
                row[c[p + "present"]], row[c[p + "family"]] = 0.0, _FAMILY_NONE
            arm_states[side] = st

        left, right = arm_states["left"], arm_states["right"]
        body = BodyFeatures(timestamp=t, scene_size=self._scene_size or (1280, 720))
        body.left, body.right = left, right
        body.body_scale = S
        body.hands_detected = sum(1 for h in frame.hands if h.source == "hand")
        body.shoulder_width_px = float(geo["shoulder_width"])
        body.torso_length_px = float(geo["torso"])
        body.scale_source = self._scale.source

        # quality
        n_model = sum(1 for side in SIDES if sides[side].source == "hand")
        if geo["pose_ok"]:
            body.quality = "full" if n_model >= 2 else "pose"
        elif any(h is not None for h in (frame.hand("left"), frame.hand("right"))):
            body.quality = "hands"
        else:
            body.quality = "none"

        # torso box / body centre
        shoulders, hips = geo["shoulders"], geo["hips"]
        if shoulders:
            if len(shoulders) == 2:
                sl, sr = shoulders["left"], shoulders["right"]
            else:  # one shoulder: estimate the other one S away
                side, p0 = next(iter(shoulders.items()))
                other = (p0[0] + S, p0[1]) if side == "left" else (p0[0] - S, p0[1])
                sl, sr = (p0, other) if side == "left" else (other, p0)
            sy = 0.5 * (sl[1] + sr[1])
            if len(hips) == 2:
                hy = 0.5 * (hips["left"][1] + hips["right"][1])
                xs = [sl[0], sr[0], hips["left"][0], hips["right"][0]]
            else:
                hy = sy + 1.4 * S
                xs = [sl[0], sr[0]]
            body.torso_box = (min(xs), min(sl[1], sr[1]), max(xs), max(hy, sy + 0.2 * S))
            smx = 0.5 * (sl[0] + sr[0])
            hmx = 0.5 * (hips["left"][0] + hips["right"][0]) if len(hips) == 2 else smx
            body.body_center = (0.5 * (smx + hmx), 0.5 * (sy + hy))
            geo["shoulder_mid_est"] = (smx, sy)

        # two-hand features
        if left.present and right.present:
            lc, rc = left.hand_center, right.hand_center
            body.hand_mid = _mid(lc, rc)
            body.hand_gap = _dist(lc, rc)
            body.hand_distance = body.hand_gap / S
            d_raw = None
            if (ref is not None and ref[c["both"]] > 0 and ref[c["l_family"]] == sides["left"].family
                    and ref[c["r_family"]] == sides["right"].family):
                d_raw = (body.hand_distance - ref[c["d"]]) / (t - ref[c["t"]])
            if d_raw is not None:
                self._d_rate = damp(self._d_rate, d_raw, DISTANCE_RATE_SMOOTH, dt)
            else:
                self._d_rate = damp(self._d_rate, 0.0, HOLD_DECAY, dt)
            body.hand_distance_rate = self._d_rate
            if body.torso_box is not None:
                x0, y0, x1, y1 = body.torso_box
                mx, my = body.hand_mid
                body.hands_in_front = (x0 - 1.5 * S <= mx <= x1 + 1.5 * S) and (y0 - 0.6 * S <= my <= y1 + 0.6 * S)
            else:
                body.hands_in_front = True
            row[c["both"]] = 1.0
            row[c["mid_x"]], row[c["mid_y"]] = body.hand_mid
            row[c["d"]], row[c["d_rate"]] = body.hand_distance, body.hand_distance_rate
        else:
            self._d_rate = 0.0
            row[c["both"]] = 0.0

        # motion summary, energy, stability
        present = [a for a in (left, right) if a.present]
        motion = MotionSummary()
        if present:
            vx = sum(a.velocity[0] for a in present) / len(present) / S
            vy = sum(a.velocity[1] for a in present) / len(present) / S
            motion.left, motion.right = max(0.0, -vx), max(0.0, vx)
            motion.up, motion.down = max(0.0, -vy), max(0.0, vy)
            if left.present and right.present:
                motion.apart = max(0.0, body.hand_distance_rate)
                motion.together = max(0.0, -body.hand_distance_rate)
            model_rates = [a.approach_rate for a in present if a.source in ("hand", "predicted")]
            if model_rates:
                depth_in_s = ASSUMED_FOCAL * body.scene_size[0] / S
                fwd = sum(model_rates) / len(model_rates) * depth_in_s
                motion.forward, motion.backward = max(0.0, fwd), max(0.0, -fwd)
            comps = motion.as_dict()
            name = max(comps, key=comps.get)
            motion.dominant = name if comps[name] > DOMINANT_MIN else "still"
            mean_speed = sum(a.speed for a in present) / len(present)
            self._energy = damp(self._energy, remap(mean_speed, 0.2, 4.0), 1.0 / ENERGY_TAU, dt)
            self._stability = damp(self._stability, 1.0 - remap(mean_speed, 0.1, 1.5), 1.0 / STABILITY_TAU, dt)
            row[c["speed_mean"]] = mean_speed
            rates = [a.approach_rate for a in present if a.source in ("hand", "predicted")]
            row[c["approach_mean"]] = sum(rates) / len(rates) if rates else float("nan")
        else:
            self._energy = damp(self._energy, 0.0, 1.0 / ENERGY_TAU, dt)
            self._stability = damp(self._stability, 0.0, 1.0 / STABILITY_TAU, dt)
        body.motion = motion
        body.energy = clamp(self._energy)
        body.stability = clamp(self._stability)

        hist.append(row, HISTORY_SECONDS)
        return body

    # ------------------------------------------------------------ tracking state

    def _update_tracking_state(self, body: BodyFeatures, now: float, events: list[GestureEvent]) -> None:
        if body.left.present or body.right.present:
            self._last_hands_now = now
            if not self._tracking_found:
                self._tracking_found = True
                events.append(GestureEvent(GestureEventType.TRACKING_FOUND, now, strength=1.0,
                                           position=body.hand_mid or (body.left.hand_center or body.right.hand_center)))
        elif self._tracking_found and (self._last_hands_now is None
                                       or now - self._last_hands_now >= TRACKING_LOST_AFTER):
            self._tracking_found = False
            events.append(GestureEvent(GestureEventType.TRACKING_LOST, now, strength=0.0))

    # ------------------------------------------------------------ detectors

    def _detect(self, body: BodyFeatures, t: float, dt: float, now: float, S: float, epoch: float,
                geo: dict, sides: dict[str, _SideMeasure], events: list[GestureEvent]) -> GestureSignals:
        th = self.thresholds
        dbg = DetectorDebug()
        sig = GestureSignals()
        both = body.both_hands

        # ---------------- charge pose
        charge = 0.0
        if both:
            T = th.together
            d = body.hand_distance
            s_d = 1.0 - smoothstep(0.75 * T, 1.3 * T, d)
            if body.torso_box is not None:
                x0, y0, x1, y1 = body.torso_box
                mx, my = body.hand_mid
                ox = max(0.0, (x0 - 1.5 * S) - mx, mx - (x1 + 1.5 * S))
                oy = max(0.0, (y0 - 0.6 * S) - my, my - (y1 + 0.6 * S))
                s_pos = 1.0 - smoothstep(0.0, 0.5 * S, math.hypot(ox, oy))
            else:
                s_pos = 0.85
            exts = [m.ext3 if m.ext3 is not None else m.ext2 for m in sides.values()]
            ext_mean = _nanmean(exts)
            s_bent = 1.0 if ext_mean is None else 1.0 - smoothstep(0.82, 0.95, ext_mean)
            # A straight arm hanging down is not part of a charge pose (one raised hand
            # next to a hanging arm would otherwise look like "hands together").
            s_bent *= 1.0 - max(_hang(m) for m in sides.values())
            s_still = 1.0 - smoothstep(2.5, 5.0, max(body.left.speed, body.right.speed))
            charge = clamp(s_d * s_pos * s_bent * s_still)
            dbg.charge_parts = (s_d, s_pos, s_bent, s_still)
        if not self._charge_active and charge > th.charge_on:
            self._charge_active = True
            self._baseline_samples.clear()
            events.append(GestureEvent(GestureEventType.CHARGE_POSE_START, now, strength=charge,
                                       position=body.hand_mid))
        elif self._charge_active and charge < th.charge_off:
            self._charge_active = False
            events.append(GestureEvent(GestureEventType.CHARGE_POSE_END, now, strength=charge,
                                       position=body.hand_mid))
        sig.charge_pose = charge
        sig.charge_pose_active = self._charge_active

        # normalised hand size (mean over hand-model hands) for the aim baseline
        lnr_now = _nanmean([math.log(m.depth_size / (self._scale.fast or S))
                            for m in sides.values() if m.family == _FAMILY_MODEL and m.depth_size > 0])
        if self._baseline_epoch != epoch:
            self._baseline, self._baseline_epoch = None, epoch
            self._baseline_samples.clear()
        if self._charge_active and lnr_now is not None:
            self._baseline_samples.append(lnr_now)
            self._baseline = float(np.median(np.fromiter(self._baseline_samples, dtype=float)))

        # ---------------- thrust
        fwd_now = _nanmean([m.forward for m in sides.values() if m.centre is not None])
        thrust, frontal, lateral, direction = self._thrust(body, t, S, epoch, geo, fwd_now, sides, dbg)
        sig.thrust = thrust
        sig.thrust_frontal = frontal / (frontal + lateral + 1e-6)
        sig.thrust_direction = direction
        if self._thrust_armed and thrust >= th.thrust and t - self._last_thrust_t >= THRUST_REFRACTORY:
            self._thrust_armed = False
            self._last_thrust_t = t
            share = sig.thrust_frontal
            if share < 0.5:
                self._last_lateral_dir = direction
            events.append(GestureEvent(GestureEventType.THRUST, now, strength=clamp(thrust), direction=direction,
                                       frontal=clamp(share), position=body.hand_mid))
        elif not self._thrust_armed and thrust < 0.5 * th.thrust:
            self._thrust_armed = True

        # ---------------- burst
        burst, burst_pos, burst_dir = self._burst(body, t, epoch)
        dbg.burst = burst
        sig.burst = burst
        if (self._burst_armed and self._settings.burst_enabled and burst >= th.burst_score
                and t - self._last_burst_t >= BURST_REFRACTORY):
            self._burst_armed = False
            self._last_burst_t = t
            events.append(GestureEvent(GestureEventType.BURST, now, strength=clamp(burst), direction=burst_dir,
                                       position=burst_pos))
        elif not self._burst_armed and burst < 0.5 * th.burst_score:
            self._burst_armed = True

        # ---------------- aim
        shoulder_mid = geo.get("shoulder_mid") or geo.get("shoulder_mid_est")
        if body.hand_mid is not None and shoulder_mid is not None:
            raw = _unit((body.hand_mid[0] - shoulder_mid[0], body.hand_mid[1] - shoulder_mid[1]), self._aim_dir)
        else:
            raw = self._last_lateral_dir
        k = 1.0 - math.exp(-12.0 * dt) if dt > 0 else 1.0
        self._aim_dir = _unit((self._aim_dir[0] + (raw[0] - self._aim_dir[0]) * k,
                               self._aim_dir[1] + (raw[1] - self._aim_dir[1]) * k), raw)
        sig.aim_direction = self._aim_dir
        fwd = _nanmean([m.forward for m in sides.values() if m.centre is not None])
        growth = (lnr_now - self._baseline) if (lnr_now is not None and self._baseline is not None) else None
        parts = []
        if fwd is not None:
            parts.append(clamp(fwd))
        if growth is not None:
            parts.append(remap(growth, 0.0, 0.35))
        target = sum(parts) / len(parts) if parts else 0.0
        self._aim_frontal = damp(self._aim_frontal, target, 10.0, dt) if dt > 0 else target
        sig.aim_frontal = clamp(self._aim_frontal)
        # Arms pointing at the floor (hanging, or lowered into a relaxed ready stance) are
        # straight-ish but not extended towards a target: a beam must stop when the user
        # lowers the arms, and a keyboard release must not aim at the floor. Bent arms
        # (charge pose, ready stance: 3D extension 0.70-0.90) are not extended either.
        present_sides = [m for m in sides.values() if m.centre is not None]
        pointing = [m for m in present_sides if m.down is None or m.down < ARM_POINTING_MAX_DOWN]
        ext3 = _nanmean([m.ext3 for m in pointing])
        ext2 = _nanmean([m.ext2 for m in pointing])
        extended: bool | None = None
        if present_sides and not pointing:
            extended = False
        elif ext3 is not None or ext2 is not None or growth is not None:
            # The 2D extension only decides without 3D data: forearms pointing forward or
            # down look almost straight in the image (the hip charge pose has ext2 ~1.0).
            extended = bool((ext3 is not None and ext3 > ARMS_EXTENDED_3D)
                            or (ext3 is None and ext2 is not None and ext2 > ARMS_EXTENDED_2D)
                            or (growth is not None and growth > 0.12))
        if extended:
            self._arms_extended_t = t
            sig.arms_extended = True
        else:
            sig.arms_extended = extended is None and t - self._arms_extended_t <= ARMS_EXTENDED_HOLD

        # ---------------- tracking ok
        recent = self._last_hands_now is not None and now - self._last_hands_now < RECENT_HANDS
        sig.tracking_ok = bool(both or (body.quality != "none" and recent))
        self.debug = dbg
        return sig

    def _thrust(self, body: BodyFeatures, t: float, S: float, epoch: float, geo: dict,
                fwd_now: float | None, sides: dict[str, _SideMeasure],
                dbg: DetectorDebug) -> tuple[float, float, float, Vec2]:
        """Thrust score: ``(thrust, frontal, lateral, direction)`` over the thrust window."""
        shoulder_mid = geo.get("shoulder_mid") or geo.get("shoulder_mid_est")
        fallback_dir = self._last_lateral_dir
        arm_dir: Vec2 | None = None
        if body.hand_mid is not None and shoulder_mid is not None:
            arm_dir = _unit((body.hand_mid[0] - shoulder_mid[0], body.hand_mid[1] - shoulder_mid[1]), fallback_dir)
            fallback_dir = arm_dir
        if not body.both_hands or (body.left.source == "predicted" and body.right.source == "predicted"):
            return 0.0, 0.0, 0.0, fallback_dir   # never fire on extrapolated hands only
        hist = self._history
        c = hist.col
        win = hist.window(t - self._settings.thrust_window, epoch)
        if win.shape[0] < 3:
            return 0.0, 0.0, 0.0, fallback_dir

        def gain(col: str, mode: str = "rise", seg_col: str | None = None) -> float | None:
            """Mean over arms of a window gain: ``"rise"`` = now - window min, ``"fall"`` =
            window max - now, ``"start"`` = now - first value in the window."""
            gains = []
            for p in ("l_", "r_"):
                series = win[:, c[p + col]]
                ok = np.isfinite(series)
                if seg_col is not None:
                    seg = win[:, c[p + seg_col]]
                    ok &= seg == seg[-1]
                if ok.sum() < 3 or not ok[-1]:
                    continue
                now_v = float(series[-1])
                vals = series[ok]
                if mode == "rise":
                    gains.append(now_v - float(vals.min()))
                elif mode == "rise_robust":
                    # min of a 3-sample running mean: a plain window minimum of a noisy
                    # series is biased low, which inflates the gain for small, far hands
                    base = vals.min() if vals.size < 3 else np.convolve(vals, np.full(3, 1.0 / 3.0), "valid").min()
                    gains.append(now_v - float(base))
                elif mode == "fall":
                    gains.append(float(vals.max()) - now_v)
                else:
                    gains.append(now_v - float(vals[0]))
            return sum(gains) / len(gains) if gains else None

        # Relative size growth of a push shrinks with the distance to the camera (perspective):
        # normalise the approach cues to a user standing REFERENCE_DEPTH metres away.
        s_ref = BODY_SCALE_M * ASSUMED_FOCAL * body.scene_size[0] / REFERENCE_DEPTH
        k_depth = clamp(s_ref / S, 0.6, 2.0)

        # frontal cues
        c_app = remap((gain("lnr", "rise_robust", seg_col="lnr_seg") or 0.0) * k_depth,
                      APPROACH_GAIN[0], APPROACH_GAIN[1])
        ext3_now = _nanmean([body.left.extension_3d, body.right.extension_3d])
        g_ext3 = gain("ext3")
        c_ext3 = remap(g_ext3, 0.15, 0.45) if (g_ext3 is not None and ext3_now is not None and ext3_now > 0.7) else 0.0
        c_fwd = remap(gain("fwd") or 0.0, 0.2, 0.6)
        c_z = remap(gain("zrel", "fall", seg_col="zkind") or 0.0, 0.15, 0.6)
        approach = win[:, c["approach_mean"]]
        approach = approach[np.isfinite(approach)]
        g_speed = remap(float(approach.max()) * k_depth, APPROACH_SPEED[0], APPROACH_SPEED[1]) if approach.size else 0.0
        # The approach rate of small, far hands is noisy; the wrists' forward speed from the
        # metric world landmarks measures the same push speed far more robustly.
        zs = win[:, [c["l_zspeed"], c["r_zspeed"]]]
        ok = np.isfinite(zs)
        n_ok = ok.sum(axis=1)
        if n_ok.any():
            zmean = np.where(ok, zs, 0.0).sum(axis=1)[n_ok > 0] / n_ok[n_ok > 0]
            g_speed = max(g_speed, remap(float(zmean.max()), FORWARD_SPEED[0], FORWARD_SPEED[1]))
        frontal = clamp(1.0 - (1.0 - 0.75 * c_app) * (1.0 - 0.55 * c_ext3) * (1.0 - 0.45 * c_fwd)
                        * (1.0 - 0.35 * c_z)) * (0.35 + 0.65 * g_speed)
        # A push towards the camera straightens the arms; hands that only turn, or come up
        # into the charge pose, do not. Applied only when 3D arm data exists.
        gate_ext = smoothstep(FRONTAL_EXT_GATE[0], FRONTAL_EXT_GATE[1], g_ext3) if g_ext3 is not None else 1.0
        # Frontal evidence counts fully only when the arms end up pointing towards the camera;
        # a sideways push also extends the arms and (with a torso twist) moves the wrists
        # forward, but its forearm direction stays mostly in the image plane.
        gate_fwd = lerp(0.45, 1.0, smoothstep(0.25, 0.6, fwd_now)) if fwd_now is not None else 1.0
        frontal *= gate_ext * gate_fwd

        # lateral cues: hand-midpoint displacement along the arm direction + extension gain + speed
        both_rows = win[win[:, c["both"]] > 0]
        lateral = 0.0
        disp: Vec2 = (0.0, 0.0)
        c_disp = c_ext = g_lat = 0.0
        down_prior = 1.0
        reach_gate = 1.0
        if both_rows.shape[0] >= 2:
            disp = ((body.hand_mid[0] - both_rows[0, c["mid_x"]]) / S,
                    (body.hand_mid[1] - both_rows[0, c["mid_y"]]) / S)
            along = math.hypot(*disp)
            if arm_dir is not None:
                along = max(0.0, disp[0] * arm_dir[0] + disp[1] * arm_dir[1])
            c_disp = remap(along, 0.35, 0.9)
            # Extension gains from the window start (lowering the arms makes the extension dip
            # and recover, which a window-min gain would count); the 3D gain counts as well
            # because perspective makes bent arms with forward-pointing forearms look almost
            # straight in 2D at close range.
            c_ext = max(remap(gain("ext2", "start") or 0.0, LATERAL_EXT_GAIN[0], LATERAL_EXT_GAIN[1]),
                        remap(gain("ext3", "start") or 0.0, LATERAL_EXT_GAIN[0], LATERAL_EXT_GAIN[1]))
            speeds = win[:, c["speed_mean"]]
            speeds = speeds[np.isfinite(speeds)]
            g_lat = remap(float(speeds.max()), 1.5, 4.0) if speeds.size else 0.0
            # Hands moving straight down are almost always arms being lowered, not an attack;
            # straight up is a stretch, a cheer or the Spirit Bomb pose far more often than a
            # beam fired at the ceiling (diagonal pushes are not affected).
            unit = _unit(disp)
            down_prior = ((1.0 - 0.6 * smoothstep(0.55, 0.9, unit[1]))
                          * (1.0 - LATERAL_UP_PENALTY * smoothstep(0.8, 0.97, -unit[1])))
            # A push ends with the arms extended away from the body: hands brought to the face
            # or chest move along the arm direction too, but end folded (3D extension ~0.5).
            # Arms that end hanging at the sides were lowered, however fast and straight.
            ext_now = _nanmean([m.ext3 if m.ext3 is not None else m.ext2 for m in sides.values()])
            if ext_now is not None:
                reach_gate = smoothstep(LATERAL_REACH[0], LATERAL_REACH[1], ext_now)
            reach_gate *= 1.0 - sum(_hang(m) for m in sides.values()) / len(sides)
            lateral = c_disp * max(c_ext, 0.35 * c_disp) * g_lat * down_prior * reach_gate

        g_tog = 1.0 - smoothstep(1.8, 2.6, body.hand_distance)
        thrust = clamp(max(frontal, lateral) * g_tog)
        direction = _unit(disp, fallback_dir) if math.hypot(*disp) >= 0.15 else fallback_dir
        dbg.frontal, dbg.lateral, dbg.thrust_gate = frontal, lateral, g_tog
        dbg.frontal_cues = (c_app, c_ext3, c_fwd, c_z, g_speed)
        dbg.frontal_gate = gate_ext * gate_fwd
        dbg.lateral_cues = (c_disp, c_ext, g_lat)
        dbg.lateral_prior = down_prior
        dbg.lateral_reach = reach_gate
        return thrust, frontal, lateral, direction

    def _burst(self, body: BodyFeatures, t: float, epoch: float) -> tuple[float, Vec2 | None, Vec2]:
        """Burst score: hands start close and fly apart quickly in opposite directions."""
        if not body.both_hands or "predicted" in (body.left.source, body.right.source):
            return 0.0, None, (1.0, 0.0)   # extrapolated (coasting) hands fly apart by themselves
        hist = self._history
        c = hist.col
        win = hist.window(t - BURST_WINDOW, epoch)
        rows = win[win[:, c["both"]] > 0]
        direction = _unit((body.right.hand_center[0] - body.left.hand_center[0],
                           body.right.hand_center[1] - body.left.hand_center[1]))
        if rows.shape[0] < 3:
            return 0.0, None, direction
        d = rows[:, c["d"]]
        i_min = int(np.nanargmin(d))
        d_min = float(d[i_min])
        d_now = body.hand_distance
        rates = rows[:, c["d_rate"]]
        peak = float(np.nanmax(rates)) if np.isfinite(rates).any() else 0.0
        # The mean separation rate since the closest point needs no per-frame derivative, so
        # a burst is still recognised when hand-model dropouts interrupt the rate estimate.
        span = max(t - float(rows[i_min, c["t"]]), 1.0 / 60.0)
        peak = max(peak, (d_now - d_min) / span)
        first = rows[0]
        start_mid = (float(first[c["mid_x"]]), float(first[c["mid_y"]]))
        dl = (body.left.hand_center[0] - first[c["l_cx"]], body.left.hand_center[1] - first[c["l_cy"]])
        dr = (body.right.hand_center[0] - first[c["r_cx"]], body.right.hand_center[1] - first[c["r_cy"]])
        # "Apart" = clearly opposite motion, or both hands moving outwards from where they
        # started (a hand that only drops down while the other swings over does not count).
        S = body.body_scale
        ul = _unit((first[c["l_cx"]] - start_mid[0], first[c["l_cy"]] - start_mid[1]), (-1.0, 0.0))
        ur = _unit((first[c["r_cx"]] - start_mid[0], first[c["r_cy"]] - start_mid[1]), (1.0, 0.0))
        out_l = (dl[0] * ul[0] + dl[1] * ul[1]) / S
        out_r = (dr[0] * ur[0] + dr[1] * ur[1]) / S
        nl, nr = math.hypot(*dl), math.hypot(*dr)
        cos_lr = (dl[0] * dr[0] + dl[1] * dr[1]) / (nl * nr) if nl > 1e-6 and nr > 1e-6 else 1.0
        apart = cos_lr < -0.2 or min(out_l, out_r) > BURST_OUTWARD_MIN
        if not (d_min <= 1.3 and d_now >= self.thresholds.burst_distance and peak >= self.thresholds.burst_rate
                and apart):
            return 0.0, start_mid, direction
        score = remap(d_now - d_min, 0.7, 1.6) * remap(peak, BURST_RATE_RAMP[0], BURST_RATE_RAMP[1])
        return score, start_mid, direction
