"""Synthetic, MediaPipe-like tracking data for gesture tests (no webcam needed).

:class:`MotionSynth` animates a virtual person standing in front of a virtual webcam and
emits the :class:`~dragon_ball_powers.datatypes.TrackingFrame` sequence the real tracker
(``tracking.py``) would produce: mirrored scene pixels, smoothed landmarks, coasting hands.

Model
-----
* **Camera**: pinhole, ``scene_size`` (default 1280x720), horizontal FOV ``hfov_deg``.
  Camera coordinates are metres, x right, y down, z = depth away from the camera.
* **Person**: MediaPipe-like world skeleton (shoulder landmarks 0.34 m apart, torso
  0.50 m, upper arm 0.27 m, forearm 0.25 m) scaled by ``body_scale``, standing
  ``distance`` metres from the camera. Arms are solved with two-bone IK from wrist
  targets, so 2D/3D arm extension and the forearm direction behave like real arms.
  Hands are articulated 21-point hands (finger curl = ``openness``) whose palm
  geometry matches MediaPipe (fingertip/palm-size ratio ~1.05 open, ~0.2 fist).
* **Landmarks** like MediaPipe: hand ``z`` = depth relative to the wrist in pixel
  scale, pose ``z`` = depth relative to the hip midpoint in pixel scale and
  exaggerated (``pose_z_gain``, MediaPipe's pose z is roughly 2-3x too large), world
  landmarks in metres (pose: hip-centred, hand: centred on the hand), visibility high
  inside the frame and low outside. On mirrored input MediaPipe labels the screen-left
  arm ``RIGHT_*`` (``model_left_is_screen_left=False``), which is reproduced. The pose's
  hand points (index/pinky) sit ~7 cm from the wrist, so pose-based hand sizes are
  ~30 % smaller than hand-model sizes, as measured on real MediaPipe output.
* **Tracker behaviour**: band-limited landmark jitter (``noise``), natural body sway
  (``sway``), hand-model dropouts (Markov, bursty), small or out-of-frame hands are
  not detected, overlapping palms make the hand model lose one hand (``auto_overlap``),
  lost hands coast for ``coast_s`` with ``source="predicted"`` (damped extrapolation),
  pose dropouts, forced dropouts (:meth:`MotionSynth.tracking_dropout`).

Everything is deterministic for a given ``seed``.

Usage::

    synth = MotionSynth(SynthConfig(distance=1.6, seed=3))
    synth.idle(1.0)
    synth.charge_pose(hold_s=1.5)
    synth.frontal_thrust(duration_s=0.3, hold_s=0.8)
    for frame, now in ticks(synth.frames):
        engine.update(frame, now)

Building blocks return the frames they generated and also append them to
:attr:`MotionSynth.frames`; each block starts from the pose the previous one ended in.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Iterator

import numpy as np

from dragon_ball_powers.datatypes import HandObservation, PoseObservation, TrackingFrame

Vec3 = tuple[float, float, float]
SIDES = ("left", "right")
_SIGN = {"left": -1.0, "right": 1.0}

# ------------------------------------------------------------------ body model

SHOULDER_WIDTH = 0.34     # shoulder landmark to shoulder landmark (MediaPipe world ~0.30-0.34 m)
TORSO_LENGTH = 0.50       # shoulder midpoint to hip midpoint
HIP_WIDTH = 0.22
UPPER_ARM = 0.27
FOREARM = 0.25
THIGH = 0.45
SHIN = 0.43
ARM_REACH = UPPER_ARM + FOREARM

# Head landmarks relative to the shoulder midpoint for the screen-left half (x is
# mirrored for the right half): (inner eye, eye, outer eye, ear, mouth corner).
_NOSE = (0.0, -0.22, -0.10)
_FACE_HALF = {
    "eye_inner": (-0.015, -0.255, -0.085),
    "eye": (-0.033, -0.258, -0.080),
    "eye_outer": (-0.050, -0.258, -0.070),
    "ear": (-0.075, -0.235, 0.000),
    "mouth": (-0.025, -0.190, -0.090),
}

# MediaPipe pose indices of the model's LEFT_* / RIGHT_* landmarks.
_MODEL_SIDE_INDICES = {
    "left": dict(eye_inner=1, eye=2, eye_outer=3, ear=7, mouth=9, shoulder=11, elbow=13, wrist=15,
                 pinky=17, index=19, thumb=21, hip=23, knee=25, ankle=27, heel=29, foot=31),
    "right": dict(eye_inner=4, eye=5, eye_outer=6, ear=8, mouth=10, shoulder=12, elbow=14, wrist=16,
                  pinky=18, index=20, thumb=22, hip=24, knee=26, ankle=28, heel=30, foot=32),
}

# ------------------------------------------------------------------ hand model
# Canonical screen-left hand in metres. Local axes: +x = thumb side, -y = finger
# direction, -z = palm normal (the side the palm faces). The right hand mirrors x.

_FINGERS = (  # (MCP x, MCP y), fan angle in degrees (+ = towards the thumb), segment lengths
    ((0.026, -0.090), 8.0, (0.040, 0.024, 0.020)),    # index
    ((0.004, -0.095), 0.0, (0.045, 0.028, 0.021)),    # middle
    ((-0.016, -0.089), -7.0, (0.042, 0.026, 0.020)),  # ring
    ((-0.034, -0.079), -15.0, (0.033, 0.019, 0.018)),  # pinky
)
_FLEX_DEG = (75.0, 95.0, 65.0)  # MCP, PIP, DIP flexion of a full fist
_THUMB_OPEN = np.array([(0.024, -0.024, -0.010), (0.047, -0.043, -0.018),
                        (0.062, -0.063, -0.022), (0.072, -0.083, -0.024)])
_THUMB_CLOSED = np.array([(0.024, -0.024, -0.010), (0.036, -0.040, -0.030),
                          (0.020, -0.058, -0.045), (0.004, -0.064, -0.050)])
_POSE_HAND_REACH = 0.72  # pose index/pinky points sit at ~72 % of the wrist->MCP distance


def canonical_hand(openness: float) -> np.ndarray:
    """21 hand landmarks (metres, local frame of a screen-left hand) for ``openness`` 0..1."""
    curl = 1.0 - float(np.clip(openness, 0.0, 1.0))
    pts = np.zeros((21, 3))
    pts[1:5] = _THUMB_OPEN * (1.0 - curl) + _THUMB_CLOSED * curl
    palm_side = np.array([0.0, 0.0, -1.0])
    for k, ((mx, my), fan, lengths) in enumerate(_FINGERS):
        base = 5 + 4 * k
        a = math.radians(fan)
        direction = np.array([math.sin(a), -math.cos(a), 0.0])
        p = np.array([mx, my, 0.0])
        pts[base] = p
        phi = 0.0
        for j in range(3):
            phi += math.radians(_FLEX_DEG[j]) * curl
            p = p + lengths[j] * (math.cos(phi) * direction + math.sin(phi) * palm_side)
            pts[base + 1 + j] = p
    return pts


def _unit(v: Iterable[float]) -> np.ndarray:
    a = np.asarray(tuple(v), dtype=float)
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.array([0.0, -1.0, 0.0])


def hand_rotation(fingers: Iterable[float], palm: Iterable[float]) -> np.ndarray:
    """Rotation matrix mapping the canonical hand frame to a hand whose fingers point
    along ``fingers`` and whose palm faces ``palm`` (palm is orthogonalised)."""
    f = _unit(fingers)
    p = np.asarray(tuple(palm), dtype=float)
    p = p - float(p @ f) * f
    if np.linalg.norm(p) < 1e-6:  # degenerate: pick any palm direction orthogonal to the fingers
        p = np.cross(f, [1.0, 0.0, 0.0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(f, [0.0, 0.0, 1.0])
    p = _unit(p)
    y_axis = -f
    z_axis = -p
    x_axis = np.cross(y_axis, z_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation about the vertical axis; yaw > 0 brings the screen-left shoulder forward."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def solve_arm(shoulder: np.ndarray, target: np.ndarray, pole: np.ndarray,
              upper: float, fore: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-bone IK. Returns (elbow, wrist); the wrist is clamped to the arm's reach."""
    reach = upper + fore
    v = target - shoulder
    d = float(np.linalg.norm(v))
    u = v / d if d > 1e-9 else np.array([0.0, 1.0, 0.0])
    d = float(np.clip(d, abs(upper - fore) + 1e-3, reach * 0.9995))
    wrist = shoulder + u * d
    a = (d * d + upper * upper - fore * fore) / (2.0 * d)
    h = math.sqrt(max(upper * upper - a * a, 0.0))
    p = np.asarray(pole, dtype=float)
    p = p - float(p @ u) * u
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(u, [0.0, 0.0, 1.0])
    p = p / np.linalg.norm(p)
    elbow = shoulder + u * a + p * h
    return elbow, wrist


# ------------------------------------------------------------------ poses


@dataclass(frozen=True)
class ArmTarget:
    """Target of one arm in the body frame.

    Positions are metres relative to the shoulder midpoint (camera-aligned axes: x to the
    screen right, y down, z away from the camera, so -z = towards the camera) for a
    person of ``body_scale`` 1. Wrist targets further than ``ARM_REACH`` from the
    shoulder are clamped to the reach (the arm is then fully straight).
    """

    wrist: Vec3
    fingers: Vec3 = (0.0, -1.0, 0.0)   # direction the fingers point
    palm: Vec3 = (0.0, 0.0, -1.0)      # direction the palm faces
    openness: float = 0.7              # 0 fist .. 1 open
    elbow_hint: Vec3 = (0.0, 1.0, 0.4)  # where the elbow should point (IK pole)


@dataclass(frozen=True)
class PersonPose:
    """Arm targets of both screen sides plus the torso twist (see :class:`ArmTarget`)."""

    left: ArmTarget
    right: ArmTarget
    yaw: float = 0.0                   # torso twist in radians (> 0 = turned towards the screen right)

    def arm(self, side: str) -> ArmTarget:
        return self.left if side == "left" else self.right


def _mirror(v: Vec3) -> Vec3:
    return (-v[0], v[1], v[2])


def symmetric_pose(right: ArmTarget, yaw: float = 0.0) -> PersonPose:
    """Pose whose left arm mirrors ``right`` (given for the screen-right arm)."""
    left = ArmTarget(_mirror(right.wrist), _mirror(right.fingers), _mirror(right.palm), right.openness,
                     _mirror(right.elbow_hint))
    return PersonPose(left, right, yaw)


def pose_rest() -> PersonPose:
    """Arms hanging relaxed at the sides."""
    return symmetric_pose(ArmTarget((0.215, 0.50, -0.05), fingers=(0.0, 1.0, -0.15), palm=(-1.0, 0.0, 0.0),
                                    openness=0.65, elbow_hint=(0.2, 0.4, 1.0)))


def pose_ready() -> PersonPose:
    """Relaxed ready stance: hands at waist height in front of the body, about 1.6 S apart."""
    return symmetric_pose(ArmTarget((0.30, 0.40, -0.17), fingers=(-0.2, 0.35, -0.9), palm=(-0.6, 0.6, 0.0),
                                    openness=0.6, elbow_hint=(0.5, 1.0, 0.4)))


def pose_charge(where: str = "chest", gap: float = 0.16) -> PersonPose:
    """Charging pose.

    ``where="chest"``: holding a ball in front of the chest, palms facing each other,
    palm centres about ``gap`` metres apart. ``"hip_right"``/``"hip_left"``: classic
    Kamehameha charge, cupped hands at the hip (the far arm reaches across the body).
    """
    if where == "chest":
        half = max(0.02, gap * 0.5 - 0.005)
        right = ArmTarget((half, 0.31, -0.22), fingers=(0.0, -0.55, -0.83), palm=(-1.0, 0.0, 0.0),
                          openness=0.75, elbow_hint=(1.0, 0.8, 0.3))
        return symmetric_pose(right)
    if where in ("hip_right", "hip_left"):
        # cupped hands at the waist, one above the other; elbow of the near arm pulled back
        top = ArmTarget((0.13, 0.30, -0.15), fingers=(0.25, 0.0, -0.97), palm=(0.0, 1.0, 0.0),
                        openness=0.7, elbow_hint=(-0.2, 1.0, 0.6))
        bottom = ArmTarget((0.19, 0.34, -0.12), fingers=(0.1, 0.0, -1.0), palm=(0.0, -1.0, 0.0),
                           openness=0.7, elbow_hint=(0.3, 0.6, 1.0))
        if where == "hip_right":
            return PersonPose(left=top, right=bottom, yaw=0.30)
        return PersonPose(left=ArmTarget(_mirror(bottom.wrist), _mirror(bottom.fingers), _mirror(bottom.palm),
                                         bottom.openness, _mirror(bottom.elbow_hint)),
                          right=ArmTarget(_mirror(top.wrist), _mirror(top.fingers), _mirror(top.palm),
                                          top.openness, _mirror(top.elbow_hint)),
                          yaw=-0.30)
    raise ValueError(f"unknown charge position {where!r}")


def pose_frontal_push() -> PersonPose:
    """Both arms pushed straight at the camera, palms facing it (Kamehameha release)."""
    right = ArmTarget((0.06, 0.13, -0.52), fingers=(0.35, -0.94, 0.0), palm=(0.0, 0.0, -1.0),
                      openness=1.0, elbow_hint=(0.8, 1.0, 0.0))
    return symmetric_pose(right)


def pose_lateral_push(direction: tuple[float, float] = (1.0, 0.0)) -> PersonPose:
    """Both arms pushed along a screen direction (unit (dx, dy), y down), torso twisted."""
    dx, dy = direction
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    yaw = 0.62 * dx                       # twist towards the push so both arms can reach
    # Targets lie slightly beyond the reach of both arms, so the IK straightens them.
    centre = np.array([0.56 * dx, 0.12 + 0.62 * dy, -0.22])
    spread = np.array([-dy, dx, 0.0]) * 0.055  # wrists side by side across the push direction
    palm = (dx, dy, -0.25)
    fingers = (-dy, dx, 0.0) if abs(dy) > 0.7 else (0.0, -1.0, 0.0)
    hint_r = (0.5, 1.0, 0.5) if dy > -0.5 else (1.0, 0.2, 0.3)
    flip = 1.0 if dx >= 0.0 else -1.0
    lw = centre - spread * flip
    rw = centre + spread * flip
    left = ArmTarget((float(lw[0]), float(lw[1]), float(lw[2])), fingers, palm, 1.0, _mirror(hint_r))
    right = ArmTarget((float(rw[0]), float(rw[1]), float(rw[2])), fingers, palm, 1.0, hint_r)
    return PersonPose(left, right, yaw)


def pose_burst() -> PersonPose:
    """Arms flung out to the sides and slightly up, palms open (energy burst)."""
    right = ArmTarget((0.62, -0.05, -0.14), fingers=(0.3, -0.95, 0.0), palm=(0.7, 0.0, -0.7),
                      openness=1.0, elbow_hint=(0.2, 1.0, 0.5))
    return symmetric_pose(right)


def pose_one_hand(side: str = "right") -> PersonPose:
    """One hand raised in front of the chest (palm forward), the other arm hanging."""
    rest = pose_rest()
    raised = ArmTarget((0.03, 0.22, -0.30), fingers=(0.0, -1.0, -0.2), palm=(0.0, 0.0, -1.0),
                       openness=0.9, elbow_hint=(0.5, 1.0, 0.3))
    if side == "right":
        return PersonPose(left=rest.left, right=raised)
    return PersonPose(left=ArmTarget(_mirror(raised.wrist), _mirror(raised.fingers), _mirror(raised.palm),
                                     raised.openness, _mirror(raised.elbow_hint)), right=rest.right)


def _lerp3(a: Vec3, b: Vec3, t: float) -> Vec3:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _nlerp3(a: Vec3, b: Vec3, t: float) -> Vec3:
    v = _unit(_lerp3(a, b, t))
    return (float(v[0]), float(v[1]), float(v[2]))


def blend_arm(a: ArmTarget, b: ArmTarget, t: float) -> ArmTarget:
    """Interpolate two arm targets (positions linearly, directions normalised)."""
    return ArmTarget(_lerp3(a.wrist, b.wrist, t), _nlerp3(a.fingers, b.fingers, t), _nlerp3(a.palm, b.palm, t),
                     a.openness + (b.openness - a.openness) * t, _nlerp3(a.elbow_hint, b.elbow_hint, t))


def blend_pose(a: PersonPose, b: PersonPose, t: float) -> PersonPose:
    """Interpolate two poses; ``t=0`` gives ``a``, ``t=1`` gives ``b`` (e.g. 0.7 = a 70 % push)."""
    return PersonPose(blend_arm(a.left, b.left, t), blend_arm(a.right, b.right, t), a.yaw + (b.yaw - a.yaw) * t)


def offset_pose(pose: PersonPose, left: Vec3 = (0.0, 0.0, 0.0), right: Vec3 = (0.0, 0.0, 0.0)) -> PersonPose:
    """Shift the wrist targets of both arms (metres, body frame)."""
    def shift(arm: ArmTarget, d: Vec3) -> ArmTarget:
        return replace(arm, wrist=(arm.wrist[0] + d[0], arm.wrist[1] + d[1], arm.wrist[2] + d[2]))
    return PersonPose(shift(pose.left, left), shift(pose.right, right), pose.yaw)


def min_jerk(t: float) -> float:
    """Minimum-jerk motion profile (human reaching movements), 0..1 -> 0..1."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


# ------------------------------------------------------------------ configuration


@dataclass
class SynthConfig:
    """Virtual camera, person and tracker behaviour."""

    scene_size: tuple[int, int] = (1280, 720)
    fps: float = 30.0
    hfov_deg: float = 70.0              # horizontal field of view of the virtual webcam
    distance: float = 1.6               # metres from the camera to the shoulder plane
    body_scale: float = 1.0             # 1 = average adult; scales all body dimensions
    lateral_offset: float = 0.0         # metres, > 0 = person stands right of the image centre
    shoulder_height: float = -0.10      # shoulder midpoint relative to the optical axis (< 0 = above)
    noise: float = 1.0                  # tracker jitter multiplier (0 = perfect landmarks)
    sway: float = 1.0                   # natural body micro-motion multiplier (0 = statue)
    hand_dropout: float = 0.0           # per-frame probability that the hand model loses a visible hand
    pose_dropout: float = 0.0           # per-frame probability that the pose model loses the person
    hand_min_px: float = 16.0           # palms smaller than this (scene px) are never detected
    hand_full_px: float = 34.0          # palms larger than this are detected reliably
    auto_overlap: bool = True           # overlapping palms -> the hand model reports only one hand
    coast_s: float = 0.25               # tracker hand_lost_grace: lost hands are predicted this long
    world_landmarks: bool = True
    pose_z_gain: float = 2.0            # MediaPipe pose z is exaggerated relative to x
    latency_s: float = 0.035            # result_time - capture_time
    timing_jitter_s: float = 0.002      # capture timestamp jitter (std)
    model_left_is_screen_left: bool = False
    seed: int = 0
    start_time: float = 100.0           # capture_time of the first frame

    @property
    def focal_px(self) -> float:
        return (self.scene_size[0] * 0.5) / math.tan(math.radians(self.hfov_deg) * 0.5)


class _AR1:
    """Band-limited noise: AR(1) process with standard deviation ``sigma`` and time constant ``tau``."""

    def __init__(self, rng: np.random.Generator, shape: tuple[int, ...], sigma: float, tau: float) -> None:
        self.rng = rng
        self.sigma = sigma
        self.tau = tau
        self.x = rng.normal(0.0, sigma, shape)

    def step(self, dt: float) -> np.ndarray:
        rho = math.exp(-dt / self.tau)
        self.x = rho * self.x + math.sqrt(1.0 - rho * rho) * self.rng.normal(0.0, self.sigma, self.x.shape)
        return self.x


@dataclass
class _HandTrack:
    """Tracker-side state of one hand slot (detection Markov chain + coasting)."""

    detected: bool = False
    lost_by_dropout: bool = False
    last: np.ndarray | None = None        # last detected landmarks (21, 3)
    last_world: np.ndarray | None = None
    velocity: np.ndarray | None = None    # px/s per landmark
    last_time: float = 0.0


# ------------------------------------------------------------------ synthesiser


class MotionSynth:
    """Generates TrackingFrame sequences of a virtual person (see module docstring).

    Attributes: ``frames`` (all generated frames, oldest first), ``pose`` (the current
    :class:`PersonPose`), ``distance`` (current camera distance in metres), ``time``
    (capture time of the last frame), ``focal`` (focal length in px) and ``last_truth``
    (ground truth of the last frame: 3D pose/hand points in camera coordinates).
    Every building block lasts its transition plus its hold time.
    """

    def __init__(self, config: SynthConfig | None = None, start_pose: PersonPose | None = None) -> None:
        self.config = config or SynthConfig()
        cfg = self.config
        self.rng = np.random.default_rng(cfg.seed)
        self.focal = cfg.focal_px
        self.frames: list[TrackingFrame] = []
        self.pose: PersonPose = start_pose or pose_ready()
        self.distance = float(cfg.distance)
        self.lateral = float(cfg.lateral_offset)
        self._k = 0
        self._model_side = self._side_index_map()
        # natural micro-motion: per-axis sinusoid phases/frequencies (body + each wrist)
        self._sway_freq = self.rng.uniform(0.15, 0.7, size=(3, 3, 2))
        self._sway_phase = self.rng.uniform(0.0, math.tau, size=(3, 3, 2))
        # tracker noise processes (sigma in px / metres, scaled by cfg.noise when used)
        self._n_pose = _AR1(self.rng, (33, 2), 1.4, 0.12)
        self._n_pose_z = _AR1(self.rng, (33,), 0.05, 0.15)       # in shoulder widths
        self._n_world = _AR1(self.rng, (33, 3), 0.008, 0.15)
        self._n_hand = {s: _AR1(self.rng, (2,), 0.8, 0.10) for s in SIDES}
        self._n_hand_scale = {s: _AR1(self.rng, (1,), 0.012, 0.15) for s in SIDES}
        self._tracks = {s: _HandTrack() for s in SIDES}
        self._pose_lost = False
        self._force_hands_until = -math.inf
        self._force_pose_until = -math.inf
        self._overlap_hidden: str | None = None
        self.last_truth: dict[str, object] = {}

    # ---------------------------------------------------------------- timeline

    @property
    def time(self) -> float:
        """Capture time of the most recently generated frame."""
        return self.config.start_time + self._k / self.config.fps

    @property
    def dt(self) -> float:
        return 1.0 / self.config.fps

    def _steps(self, duration_s: float) -> int:
        return max(0, int(round(duration_s * self.config.fps)))

    def _run(self, duration_s: float, pose_at: Callable[[float], PersonPose],
             root_at: Callable[[float], np.ndarray] | None = None,
             yaw_at: Callable[[float], float] | None = None) -> list[TrackingFrame]:
        """Emit frames for ``duration_s``; callbacks get the progress 0..1 of the block."""
        n = self._steps(duration_s)
        out = []
        for i in range(1, n + 1):
            u = i / n
            pose = pose_at(u)
            root = root_at(u) if root_at else np.zeros(3)
            yaw = yaw_at(u) if yaw_at else 0.0
            out.append(self._emit(pose, root, yaw))
            self.pose = pose
        return out

    # ---------------------------------------------------------------- building blocks

    def hold(self, duration_s: float) -> list[TrackingFrame]:
        """Keep the current pose (with natural micro-motion)."""
        pose = self.pose
        return self._run(duration_s, lambda u: pose)

    def move_to(self, target: PersonPose, duration_s: float, hold_s: float = 0.0,
                profile: Callable[[float], float] = min_jerk) -> list[TrackingFrame]:
        """Move from the current pose to ``target`` in ``duration_s`` (min-jerk), then hold."""
        start = self.pose
        frames = self._run(duration_s, lambda u: blend_pose(start, target, profile(u)))
        self.pose = target
        return frames + self.hold(hold_s)

    def idle(self, duration_s: float = 1.0, stance: str = "ready", transition_s: float = 0.6) -> list[TrackingFrame]:
        """Relaxed idle: ``"ready"`` (hands in front at waist height, apart) or ``"rest"`` (arms down)."""
        target = pose_ready() if stance == "ready" else pose_rest()
        return self.move_to(target, transition_s, max(0.0, duration_s - transition_s))

    def charge_pose(self, hold_s: float = 1.5, where: str = "chest", transition_s: float = 0.5,
                    gap: float = 0.16) -> list[TrackingFrame]:
        """Bring the hands together into the charging pose (``transition_s``), then hold it
        for ``hold_s``; ``where`` and ``gap`` as in :func:`pose_charge`."""
        return self.move_to(pose_charge(where, gap), transition_s, hold_s)

    def frontal_thrust(self, duration_s: float = 0.3, hold_s: float = 0.8) -> list[TrackingFrame]:
        """Push both arms straight towards the camera (palms forward), then hold."""
        return self.move_to(pose_frontal_push(), duration_s, hold_s)

    def lateral_thrust(self, direction: tuple[float, float] = (1.0, 0.0), duration_s: float = 0.3,
                       hold_s: float = 0.8) -> list[TrackingFrame]:
        """Push both arms along the screen ``direction`` (x right, y down), then hold."""
        return self.move_to(pose_lateral_push(direction), duration_s, hold_s)

    def burst(self, duration_s: float = 0.25, hold_s: float = 0.6) -> list[TrackingFrame]:
        """Fling both hands apart (arms out to the sides), then hold."""
        return self.move_to(pose_burst(), duration_s, hold_s)

    def one_hand(self, hold_s: float = 2.0, side: str = "right", transition_s: float = 0.5) -> list[TrackingFrame]:
        """Raise one hand in front of the chest while the other arm hangs down."""
        return self.move_to(pose_one_hand(side), transition_s, hold_s)

    def separate_hands(self, gap: float, duration_s: float, hold_s: float = 0.0) -> list[TrackingFrame]:
        """Move both wrists symmetrically apart (or together) to a palm gap of ``gap`` metres."""
        start = self.pose
        cur_gap = abs(start.right.wrist[0] - start.left.wrist[0]) + 0.01
        delta = 0.5 * (gap - cur_gap)
        target = offset_pose(start, left=(-delta, 0.0, 0.0), right=(delta, 0.0, 0.0))
        return self.move_to(target, duration_s, hold_s, profile=lambda u: u * u * (3.0 - 2.0 * u))

    def walk_towards_camera(self, to_distance: float, duration_s: float, arms: str = "keep") -> list[TrackingFrame]:
        """Walk to ``to_distance`` metres with gait bob/sway.

        ``arms="keep"`` holds the current arm pose (e.g. walking in the charge pose),
        ``"swing"`` lets the arms hang and swing naturally.
        """
        start_d = self.distance
        start = self.pose
        base = self.pose if arms == "keep" else pose_rest()
        step_hz = 1.9

        def pose_at(u: float) -> PersonPose:
            t = u * duration_s
            if arms != "swing":
                return base
            swing = 0.13 * math.sin(math.tau * step_hz * 0.5 * t)
            swinging = offset_pose(base, left=(0.0, -0.03 * abs(swing) / 0.13, -swing),
                                   right=(0.0, -0.03 * abs(swing) / 0.13, swing))
            return blend_pose(start, swinging, min_jerk(t / 0.4))  # let the arms drop first

        def root_at(u: float) -> np.ndarray:
            t = u * duration_s
            # the walk itself changes the camera distance used by the renderer
            self.distance = start_d + (to_distance - start_d) * (u * u * (3.0 - 2.0 * u))
            bob = 0.012 * math.cos(math.tau * step_hz * t)
            sway = 0.015 * math.sin(math.tau * step_hz * 0.5 * t)
            return np.array([sway, bob, 0.0])

        def yaw_at(u: float) -> float:
            return 0.05 * math.sin(math.tau * step_hz * 0.5 * u * duration_s)

        return self._run(duration_s, pose_at, root_at, yaw_at)

    def random_fidget(self, duration_s: float, hands_together: bool = True,
                      amplitude: float = 1.0) -> list[TrackingFrame]:
        """Restless, non-gesture movement: drifting/re-adjusting hands, turning wrists,
        opening and closing the hands. ``hands_together`` keeps a charge-like pose."""
        base = pose_charge("chest") if hands_together else pose_ready()
        start = self.pose
        rng = self.rng
        freqs = rng.uniform(0.25, 1.3, size=(2, 3, 3))
        phases = rng.uniform(0.0, math.tau, size=(2, 3, 3))
        amps = rng.uniform(0.012, 0.03, size=(2, 3, 3)) * amplitude
        # occasional quick re-adjustments of both hands to a new resting offset around the
        # base pose (mean-reverting, so the hands do not drift away over time)
        jumps: list[tuple[float, np.ndarray, np.ndarray]] = []
        t_next = rng.uniform(0.8, 2.0)
        offset = np.zeros(3)
        while t_next < duration_s:
            target = rng.normal(0.0, 0.035, 3) * amplitude
            jumps.append((t_next, offset, target))
            offset = target
            t_next += rng.uniform(1.2, 2.8)
        rot = rng.uniform(0.0, math.tau, size=4)

        def pose_at(u: float) -> PersonPose:
            t = u * duration_s
            settle = min_jerk(min(1.0, t / 0.5))
            pose = blend_pose(start, base, settle)
            shared = np.zeros(3)
            for (tj, frm, to) in jumps:
                if t >= tj:
                    shared = frm + (to - frm) * min_jerk((t - tj) / 0.25)
            arms = []
            for k, arm in enumerate((pose.left, pose.right)):
                wob = (amps[k] * np.sin(math.tau * freqs[k] * t + phases[k])).sum(axis=1)
                if hands_together:  # hands move mostly together, relative motion is small
                    common = (amps[0] * np.sin(math.tau * freqs[0] * t + phases[0])).sum(axis=1)
                    wob = common + 0.35 * wob
                w = np.asarray(arm.wrist) + wob + shared
                ang = 0.35 * amplitude * math.sin(math.tau * 0.4 * t + rot[k])
                f = np.asarray(arm.fingers)
                f = f * math.cos(ang) + np.cross([0.0, 0.0, 1.0], f) * math.sin(ang)
                openness = float(np.clip(arm.openness + 0.25 * amplitude * math.sin(math.tau * 0.3 * t + rot[k + 2]),
                                         0.0, 1.0))
                arms.append(replace(arm, wrist=tuple(w), fingers=tuple(f), openness=openness))
            return PersonPose(arms[0], arms[1], pose.yaw)

        return self._run(duration_s, pose_at)

    @contextmanager
    def hands_overlapping(self, hidden: str = "left") -> Iterator[None]:
        """While active, the hand model only reports one hand (``hidden`` side is never
        detected, as when the hands overlap in the cupped pose); the pose still sees both wrists."""
        previous = self._overlap_hidden
        self._overlap_hidden = hidden
        try:
            yield
        finally:
            self._overlap_hidden = previous

    def tracking_dropout(self, duration_s: float, hands: bool = True, pose: bool = True) -> list[TrackingFrame]:
        """The tracker loses the hands and/or the pose for ``duration_s`` while the person
        keeps the current pose (lost hands still coast for ``coast_s`` first)."""
        end = self.time + duration_s + 1e-6
        if hands:
            self._force_hands_until = end
        if pose:
            self._force_pose_until = end
        return self.hold(duration_s)

    # ---------------------------------------------------------------- rendering

    def _side_index_map(self) -> dict[str, dict[str, int]]:
        """Screen side -> MediaPipe indices of that side's landmarks."""
        if self.config.model_left_is_screen_left:
            return {"left": _MODEL_SIDE_INDICES["left"], "right": _MODEL_SIDE_INDICES["right"]}
        return {"left": _MODEL_SIDE_INDICES["right"], "right": _MODEL_SIDE_INDICES["left"]}

    def _sway(self, t: float, which: int) -> np.ndarray:
        """Smooth micro-motion (metres) for the body (0), left wrist (1) or right wrist (2)."""
        f = self._sway_freq[which]
        ph = self._sway_phase[which]
        amp = (0.004, 0.006, 0.006)[which] * self.config.sway
        return amp * np.sin(math.tau * f * t + ph).sum(axis=1)

    def _project(self, p: np.ndarray) -> np.ndarray:
        """Camera coords (N, 3) -> scene px (N, 2)."""
        w, h = self.config.scene_size
        z = np.maximum(p[:, 2], 0.05)
        return np.column_stack([w * 0.5 + self.focal * p[:, 0] / z, h * 0.5 + self.focal * p[:, 1] / z])

    def skeleton(self, pose: PersonPose, root: np.ndarray, yaw: float) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """3D landmarks in camera coordinates: (33, 3) pose points and {side: (21, 3)} hands."""
        k = self.config.body_scale
        pts = np.zeros((33, 3))
        R = _yaw_matrix(yaw)
        Rh = _yaw_matrix(yaw * 0.35)
        hip_mid = root + np.array([0.0, TORSO_LENGTH * k, 0.0])
        pts[0] = root + R @ (np.array(_NOSE) * k)
        hands: dict[str, np.ndarray] = {}
        for side in SIDES:
            sign = _SIGN[side]
            ix = self._model_side[side]
            for name, offset in _FACE_HALF.items():
                o = np.array(offset) * k
                if side == "right":
                    o[0] = -o[0]
                pts[ix[name]] = root + R @ o
            shoulder = root + R @ np.array([sign * SHOULDER_WIDTH * 0.5 * k, 0.0, 0.0])
            hip = hip_mid + Rh @ np.array([sign * HIP_WIDTH * 0.5 * k, 0.0, 0.0])
            knee = hip + np.array([0.01 * sign, THIGH * k, -0.02 * k])
            ankle = knee + np.array([0.0, SHIN * k, 0.03 * k])
            pts[ix["shoulder"]] = shoulder
            pts[ix["hip"]] = hip
            pts[ix["knee"]] = knee
            pts[ix["ankle"]] = ankle
            pts[ix["heel"]] = ankle + np.array([0.0, 0.05 * k, 0.05 * k])
            pts[ix["foot"]] = ankle + np.array([0.0, 0.07 * k, -0.14 * k])
            arm = pose.arm(side)
            target = root + np.asarray(arm.wrist) * k
            elbow, wrist = solve_arm(shoulder, target, np.asarray(arm.elbow_hint), UPPER_ARM * k, FOREARM * k)
            pts[ix["elbow"]] = elbow
            pts[ix["wrist"]] = wrist
            local = canonical_hand(arm.openness) * k
            if side == "right":
                local[:, 0] = -local[:, 0]
            hand = wrist + local @ hand_rotation(arm.fingers, arm.palm).T
            hands[side] = hand
            pts[ix["index"]] = wrist + _POSE_HAND_REACH * (hand[5] - wrist)
            pts[ix["pinky"]] = wrist + _POSE_HAND_REACH * (hand[17] - wrist)
            pts[ix["thumb"]] = 0.5 * (hand[2] + hand[3])
        return pts, hands

    def _emit(self, pose: PersonPose, root_offset: np.ndarray, yaw_offset: float) -> TrackingFrame:
        cfg = self.config
        self._k += 1
        t = self.time
        dt = self.dt
        w, h = cfg.scene_size
        rng = self.rng
        noise = cfg.noise

        # ---- virtual person (with natural micro-motion)
        sway_l, sway_r = self._sway(t, 1), self._sway(t, 2)
        pose = offset_pose(pose, tuple(sway_l), tuple(sway_r))
        root = np.array([self.lateral, cfg.shoulder_height, self.distance]) + root_offset + self._sway(t, 0)
        pts3, hands3 = self.skeleton(pose, root, pose.yaw + yaw_offset)
        self.last_truth = {"pose3d": pts3, "hands3d": hands3, "root": root, "pose": pose}

        # ---- pose observation
        hip_ix = (self._model_side["left"]["hip"], self._model_side["right"]["hip"])
        hip_mid = 0.5 * (pts3[hip_ix[0]] + pts3[hip_ix[1]])
        uv = self._project(pts3)
        sh = (self._model_side["left"]["shoulder"], self._model_side["right"]["shoulder"])
        shoulder_px = float(np.linalg.norm(uv[sh[0]] - uv[sh[1]]))
        pose_xy = uv + noise * (self._n_pose.step(dt) + rng.normal(0.0, 0.4, (33, 2)))
        z_px = cfg.pose_z_gain * (pts3[:, 2] - hip_mid[2]) * self.focal / max(hip_mid[2], 0.05)
        z_px = z_px + noise * self._n_pose_z.step(dt) * max(shoulder_px, 1.0)
        inside = (uv[:, 0] >= 0.0) & (uv[:, 0] <= w) & (uv[:, 1] >= 0.0) & (uv[:, 1] <= h)
        vis = np.where(inside, rng.uniform(0.9, 0.995, 33), rng.uniform(0.05, 0.3, 33))
        for side in SIDES:  # the pose model is less sure about its hand points
            for name in ("pinky", "index", "thumb"):
                i = self._model_side[side][name]
                if inside[i]:
                    vis[i] = rng.uniform(0.6, 0.95)
        world = None
        if cfg.world_landmarks:
            world = (pts3 - hip_mid + noise * self._n_world.step(dt)).astype(np.float32)
        else:
            self._n_world.step(dt)
        pose_ok = True
        if t < self._force_pose_until:
            pose_ok = False
        elif cfg.pose_dropout > 0.0:
            if self._pose_lost:
                self._pose_lost = rng.random() > 0.3
            else:
                self._pose_lost = rng.random() < cfg.pose_dropout
            pose_ok = not self._pose_lost
        if inside.sum() < 6:
            pose_ok = False
        pose_obs = None
        if pose_ok:
            pose_obs = PoseObservation(
                landmarks=np.column_stack([pose_xy, z_px]).astype(np.float32),
                visibility=vis.astype(np.float32), world=world, score=float(rng.uniform(0.9, 0.99)),
                model_left_is_screen_left=cfg.model_left_is_screen_left)

        # ---- hand observations
        hand_px: dict[str, np.ndarray] = {}
        palm_size: dict[str, float] = {}
        for side in SIDES:
            p3 = hands3[side]
            puv = self._project(p3)
            zrel = (p3[:, 2] - p3[0, 2]) * self.focal / max(p3[0, 2], 0.05)
            centre = puv[[0, 5, 9, 13, 17]].mean(axis=0)
            scale = 1.0 + noise * float(self._n_hand_scale[side].step(dt)[0])
            jitter = noise * self._n_hand[side].step(dt)
            pts = centre + (puv - centre) * scale + jitter + noise * rng.normal(0.0, 0.35, (21, 2))
            hand_px[side] = np.column_stack([pts, zrel * scale]).astype(np.float32)
            palm_size[side] = max(float(np.linalg.norm(puv[0] - puv[9])),
                                  1.25 * float(np.linalg.norm(puv[5] - puv[17])))

        visible = {}
        for side in SIDES:
            c = hand_px[side][[0, 5, 9, 13, 17], :2].mean(axis=0)
            size = palm_size[side]
            in_frame = (-0.2 * size <= c[0] <= w + 0.2 * size) and (-0.2 * size <= c[1] <= h + 0.2 * size)
            visible[side] = in_frame and size >= cfg.hand_min_px
        if cfg.auto_overlap and visible["left"] and visible["right"]:
            cl = hand_px["left"][[0, 5, 9, 13, 17], :2].mean(axis=0)
            cr = hand_px["right"][[0, 5, 9, 13, 17], :2].mean(axis=0)
            if np.linalg.norm(cl - cr) < 0.9 * min(palm_size.values()):
                # the hand model merges overlapping hands: keep the one closer to the camera
                far = "left" if hands3["left"][:, 2].mean() > hands3["right"][:, 2].mean() else "right"
                visible[far] = False
        if self._overlap_hidden in SIDES:
            visible[self._overlap_hidden] = False

        hands: list[HandObservation] = []
        for side in SIDES:
            track = self._tracks[side]
            detect = visible[side] and t >= self._force_hands_until
            if detect:
                # bursty dropouts + unreliable detection of small hands
                small = 1.0 - _smoothstep(cfg.hand_min_px, cfg.hand_full_px, palm_size[side])
                p_lose = min(0.9, cfg.hand_dropout + 0.35 * small)
                if track.lost_by_dropout:
                    track.lost_by_dropout = rng.random() > 0.3
                else:
                    track.lost_by_dropout = rng.random() < p_lose
                detect = not track.lost_by_dropout
            if detect:
                lm = hand_px[side]
                if track.last is not None and track.detected:
                    track.velocity = (lm - track.last) / dt
                elif not track.detected:
                    track.velocity = None
                track.detected = True
                track.last = lm.copy()
                track.last_time = t
                hand_world = None
                if cfg.world_landmarks:
                    p3 = hands3[side]
                    hand_world = (p3 - p3.mean(axis=0) + noise * rng.normal(0.0, 0.002, (21, 3))).astype(np.float32)
                track.last_world = hand_world
                hands.append(HandObservation(side=side, landmarks=lm, world=hand_world,
                                             handedness="Left" if side == "left" else "Right",
                                             score=float(rng.uniform(0.88, 0.99)), source="hand", age=0.0))
            else:
                track.detected = False
                if track.last is not None and t - track.last_time <= cfg.coast_s + 1e-9:
                    age = t - track.last_time
                    lm = track.last.copy()
                    if track.velocity is not None:
                        k = 6.0  # damped velocity extrapolation like the tracker's coasting
                        lm[:, :2] += track.velocity[:, :2] * (1.0 - math.exp(-k * age)) / k
                    hands.append(HandObservation(side=side, landmarks=lm.astype(np.float32), world=track.last_world,
                                                 handedness="Left" if side == "left" else "Right",
                                                 score=0.5, source="predicted", age=float(age)))

        capture = t + float(rng.normal(0.0, cfg.timing_jitter_s)) if cfg.timing_jitter_s > 0 else t
        frame = TrackingFrame(frame_id=self._k, capture_time=capture, result_time=capture + cfg.latency_s,
                              scene_size=(int(w), int(h)), image=None, hands=hands, pose=pose_obs,
                              inference_ms={"hands": 9.0, "pose": 12.0})
        self.frames.append(frame)
        return frame


def _smoothstep(e0: float, e1: float, x: float) -> float:
    if e1 <= e0:
        return 1.0 if x >= e0 else 0.0
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def ticks(frames: Iterable[TrackingFrame],
          display_hz: float | None = 60.0) -> Iterator[tuple[TrackingFrame | None, float]]:
    """Replay frames like the app's main loop.

    Yields ``(frame, now)`` once per frame at its ``result_time``; with ``display_hz``
    additional ``(None, now)`` ticks are yielded at display rate in between (the app
    calls ``GestureEngine.update(None, now)`` when no new tracking frame arrived).
    When several frames arrive between two display ticks only the newest is
    delivered, exactly like polling ``tracker.latest()``.
    """
    frames = list(frames)
    if not frames:
        return
    if display_hz is None:
        for f in frames:
            yield f, f.result_time
        return
    period = 1.0 / display_hz
    now = frames[0].result_time
    i = 0
    end = frames[-1].result_time + period
    while now <= end + 1e-9:
        newest = None
        while i < len(frames) and frames[i].result_time <= now + 1e-9:
            newest = frames[i]
            i += 1
        yield newest, now
        now += period
