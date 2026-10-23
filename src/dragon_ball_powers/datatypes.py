"""Shared data structures passed between the camera, tracking, gesture, ability and UI layers.

Coordinate conventions (used everywhere unless stated otherwise):

* **Scene space** = pixel coordinates of the (mirrored) camera frame, x right, y down.
  ``scene_size`` is the camera frame size ``(width, height)``. Effects, landmarks
  and gesture features all use scene space; the renderer maps it to the window.
* Landmark ``z`` values are MediaPipe's relative depth multiplied by the scene width
  (so they share the x/y pixel scale). Smaller / more negative = closer to camera.
* World landmarks are MediaPipe's metric 3D coordinates (metres), axes aligned with
  the image (x right, y down, z away from the camera; smaller z = closer).
* Hand ``side`` is the **screen side** ("left" = left half of the mirrored image,
  which is the user's own left hand when the preview is mirrored).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from dragon_ball_powers.mathutil import Vec2

# ------------------------------------------------------------------ landmark indices


class HandLandmark(enum.IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

PALM_INDICES: tuple[int, ...] = (0, 5, 9, 13, 17)  # wrist + finger MCPs
FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)


class PoseLandmark(enum.IntEnum):
    """MediaPipe pose landmark indices (labels are the model's, see PoseObservation)."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
)


@dataclass(frozen=True)
class ArmIndices:
    """Pose landmark indices of one arm (plus hip) for a given screen side."""

    shoulder: int
    elbow: int
    wrist: int
    pinky: int
    index: int
    thumb: int
    hip: int


_MODEL_LEFT_ARM = ArmIndices(11, 13, 15, 17, 19, 21, 23)
_MODEL_RIGHT_ARM = ArmIndices(12, 14, 16, 18, 20, 22, 24)

# ------------------------------------------------------------------------ camera


class CameraState(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class CameraInfo:
    """A camera device that can be opened with ``cv2.VideoCapture(index, backend)``."""

    index: int
    name: str
    backend: str = "auto"
    is_ir: bool = False  # infrared (Windows Hello) cameras are poor choices for tracking
    is_virtual: bool = False  # a virtual camera output such as "OBS Virtual Camera" (not a webcam)

    @property
    def label(self) -> str:
        return f"{self.index}: {self.name}"


@dataclass
class CameraFrame:
    frame_id: int
    image: np.ndarray  # BGR uint8, already mirrored when mirroring is enabled
    timestamp: float   # time.perf_counter() when the frame arrived

    @property
    def size(self) -> tuple[int, int]:
        return (int(self.image.shape[1]), int(self.image.shape[0]))


@dataclass
class CameraStatus:
    state: CameraState = CameraState.STOPPED
    index: int = -1
    name: str = ""
    backend: str = ""
    resolution: tuple[int, int] = (0, 0)
    fps: float = 0.0
    frames: int = 0
    error: str | None = None


# --------------------------------------------------------------- virtual camera


class VirtualCameraState(str, enum.Enum):
    OFF = "off"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class VirtualCameraStatus:
    """Snapshot of the virtual camera output for video calls (see virtual_camera.py)."""

    state: VirtualCameraState = VirtualCameraState.OFF
    device: str = ""                   # e.g. "OBS Virtual Camera"
    backend: str = ""                  # pyvirtualcam backend: "obs" | "unitycapture" | "v4l2loopback"
    resolution: tuple[int, int] = (0, 0)
    fps: float = 0.0
    frames: int = 0                    # frames sent to the device
    error: str | None = None

    @property
    def active(self) -> bool:
        """Switched on and not failed (starting or running)."""
        return self.state in (VirtualCameraState.STARTING, VirtualCameraState.RUNNING)


# ---------------------------------------------------------------------- tracking


@dataclass
class HandObservation:
    side: str                          # screen side slot: "left" | "right"
    landmarks: np.ndarray              # (21, 3) float32: x, y in scene px, z (px-scaled depth)
    world: np.ndarray | None = None    # (21, 3) float32 metres, hand-centred
    handedness: str = ""               # model label "Left"/"Right" (user's anatomical hand)
    score: float = 0.0                 # handedness / presence confidence 0..1
    source: str = "hand"               # "hand" (hand model) | "predicted" (coasting)
    age: float = 0.0                   # seconds since the hand model last saw this hand

    @property
    def wrist(self) -> Vec2:
        return (float(self.landmarks[0, 0]), float(self.landmarks[0, 1]))

    @property
    def palm_center(self) -> Vec2:
        pts = self.landmarks[list(PALM_INDICES), :2]
        c = pts.mean(axis=0)
        return (float(c[0]), float(c[1]))


@dataclass
class PoseObservation:
    landmarks: np.ndarray              # (33, 3) float32: x, y scene px, z px-scaled depth
    visibility: np.ndarray             # (33,) float32 0..1
    world: np.ndarray | None = None    # (33, 3) float32 metres, hip-centred
    # MediaPipe has no pose score: this is the mean visibility of the nose, shoulders,
    # elbows and wrists (what gesture code gates on).
    score: float = 0.0
    # True when the model's LEFT_* arm is on the left of the screen. On a mirrored
    # image the model's left/right labels are swapped relative to the user, so all
    # code should use ``arm(side)`` instead of the raw LEFT/RIGHT indices.
    model_left_is_screen_left: bool = True

    def arm(self, side: str) -> ArmIndices:
        screen_left = side == "left"
        use_model_left = screen_left == self.model_left_is_screen_left
        return _MODEL_LEFT_ARM if use_model_left else _MODEL_RIGHT_ARM

    def point(self, index: int) -> Vec2:
        return (float(self.landmarks[index, 0]), float(self.landmarks[index, 1]))

    def visible(self, index: int, threshold: float = 0.5) -> bool:
        return bool(self.visibility[index] >= threshold)


@dataclass
class TrackingFrame:
    """Everything the tracker produced for one camera frame."""

    frame_id: int
    capture_time: float                      # perf_counter() when the camera frame arrived
    result_time: float                       # perf_counter() when results were ready
    scene_size: tuple[int, int]              # (width, height) of the analysed frame
    image: np.ndarray | None                 # the analysed (mirrored) BGR frame
    hands: list[HandObservation] = field(default_factory=list)  # smoothed, max one per side
    pose: PoseObservation | None = None      # smoothed
    segmentation: np.ndarray | None = None   # (h, w) uint8 person mask, low resolution, frame-aligned
    segmentation_id: int = 0                 # changes whenever ``segmentation`` changes
    inference_ms: dict[str, float] = field(default_factory=dict)  # {"hands": .., "pose": ..}

    @property
    def latency_ms(self) -> float:
        return (self.result_time - self.capture_time) * 1000.0

    def hand(self, side: str) -> HandObservation | None:
        for h in self.hands:
            if h.side == side:
                return h
        return None


@dataclass
class TrackerStatus:
    running: bool = False
    error: str | None = None
    fps: float = 0.0
    hands_ms: float = 0.0
    pose_ms: float = 0.0
    total_ms: float = 0.0
    latency_ms: float = 0.0
    pose_model: str = ""


# ---------------------------------------------------------------------- gestures


@dataclass
class ArmState:
    """Derived per-arm features for one screen side."""

    side: str
    present: bool = False              # a usable hand position exists
    source: str = "none"               # "hand" | "pose" | "predicted" | "none"
    hand_center: Vec2 | None = None    # palm centre (scene px)
    hand_size: float = 0.0             # palm size in px (wrist to middle MCP based)
    shoulder: Vec2 | None = None
    elbow: Vec2 | None = None
    wrist: Vec2 | None = None
    extension_2d: float = 0.0          # 0 folded .. 1 fully straight (image plane)
    extension_3d: float | None = None  # same from world landmarks, None if unavailable
    forward: float = 0.0               # -1..1, 1 = forearm points straight at the camera
    velocity: Vec2 = (0.0, 0.0)        # px/s of the hand centre
    speed: float = 0.0                 # body scales per second
    approach_rate: float = 0.0         # d ln(hand_size/body_scale)/dt; > 0 moving towards camera
    openness: float = 0.0              # 0 fist .. 1 open palm


@dataclass
class MotionSummary:
    """Current hand motion, all rates in body scales per second (>= 0)."""

    together: float = 0.0
    apart: float = 0.0
    up: float = 0.0
    down: float = 0.0
    left: float = 0.0
    right: float = 0.0
    forward: float = 0.0               # towards the camera (from hand-size growth)
    backward: float = 0.0
    dominant: str = "still"            # name of the largest component above a threshold

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in ("together", "apart", "up", "down", "left", "right",
                                             "forward", "backward")}


@dataclass
class BodyState:
    timestamp: float                          # capture time of the source frame
    scene_size: tuple[int, int]
    quality: str = "none"                     # "none" | "hands" | "pose" | "full"
    hands_detected: int = 0                   # hands seen by the hand model this frame
    body_scale: float = 0.0                   # px, roughly one shoulder width
    body_center: Vec2 | None = None           # torso centre
    torso_box: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1
    left: ArmState = field(default_factory=lambda: ArmState("left"))
    right: ArmState = field(default_factory=lambda: ArmState("right"))
    hand_mid: Vec2 | None = None              # midpoint between both hand centres
    hand_gap: float = 0.0                     # px between the hand centres
    hand_distance: float = float("inf")       # body scales between the hand centres
    hand_distance_rate: float = 0.0           # body scales / s (> 0 = moving apart)
    hands_in_front: bool = False              # both hands inside the torso region
    motion: MotionSummary = field(default_factory=MotionSummary)
    energy: float = 0.0                       # 0..1 smoothed movement energy
    stability: float = 0.0                    # 0..1, 1 = hands very steady

    def arm(self, side: str) -> ArmState:
        return self.left if side == "left" else self.right

    @property
    def both_hands(self) -> bool:
        return self.left.present and self.right.present


@dataclass
class GestureSignals:
    """Continuous gesture scores, updated every tracking frame."""

    charge_pose: float = 0.0                  # 0..1 "hands together in front of body"
    charge_pose_active: bool = False          # thresholded with hysteresis + hold time
    thrust: float = 0.0                       # 0..1 forward-push score
    thrust_frontal: float = 0.0               # 0..1 share of the push aimed at the camera
    thrust_direction: Vec2 = (1.0, 0.0)       # unit screen direction of the lateral part
    burst: float = 0.0                        # 0..1 sharp outward-spread score
    arms_extended: bool = False               # both arms extended (beam may continue)
    aim_direction: Vec2 = (1.0, 0.0)          # where the arms currently point (screen)
    aim_frontal: float = 0.0                  # 0..1 how much the arms point at the camera
    tracking_ok: bool = False                 # enough tracking data for gestures


class GestureEventType(str, enum.Enum):
    CHARGE_POSE_START = "charge_pose_start"
    CHARGE_POSE_END = "charge_pose_end"
    THRUST = "thrust"
    BURST = "burst"
    TRACKING_LOST = "tracking_lost"
    TRACKING_FOUND = "tracking_found"


@dataclass
class GestureEvent:
    type: GestureEventType
    time: float
    strength: float = 1.0                     # 0..1 (how decisive the gesture was)
    direction: Vec2 = (1.0, 0.0)              # unit screen direction (thrust/burst)
    frontal: float = 0.0                      # 0..1 share aimed at the camera (thrust)
    position: Vec2 | None = None              # where it happened (hand midpoint)


# ------------------------------------------------------------------ manual input


@dataclass
class ManualInput:
    """Keyboard test controls for the current tick."""

    trigger_held: bool = False                # Space is held
    trigger_pressed: bool = False             # Space went down this tick
    trigger_released: bool = False            # Space went up this tick
    aim: Vec2 = (0.0, 0.0)                    # arrow-key aim, (0, 0) = automatic


# --------------------------------------------------------------------- HUD state


@dataclass
class AbilityHUD:
    selected: str = "kamehameha"
    selected_label: str = "Kamehameha"
    phase: str = "idle"                       # AttackPhase value of the selected ability
    charge: float = 0.0                       # 0..1
    charge_ready: bool = False                # minimum charge reached, release will fire
    power_level: int = 0                      # displayed power level number
    power_fraction: float = 0.0               # 0..1 for the meter
    aura_enabled: bool = False
    aura_intensity: float = 0.0
    cooldown: float = 0.0                     # 0..1 remaining cooldown fraction
    callout: str | None = None                # e.g. "KA... ME... HA..." chant text
    callout_intensity: float = 0.0            # 0..1 for size / glow animation
    message: str | None = None                # short hint for the user


@dataclass
class CalibrationHUD:
    active: bool = False
    step: int = 0                             # 1-based current step
    total_steps: int = 0
    title: str = ""
    instruction: str = ""
    progress: float = 0.0                     # 0..1 within the current step
    result_message: str | None = None         # shown briefly after finishing / failing
