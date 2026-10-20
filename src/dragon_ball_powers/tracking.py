"""MediaPipe hand + pose tracking: tracker thread, hand slots, smoothing and coasting.

Pipeline for every analysed camera frame (docs/ARCHITECTURE.md, section 5.2):

1. The mirrored BGR frame is downscaled to ``tracking_width`` (aspect kept, never
   upscaled) and converted to RGB.
2. The MediaPipe Tasks ``HandLandmarker`` and ``PoseLandmarker`` (VIDEO mode) run
   concurrently on two dedicated worker threads. MediaPipe releases the GIL, so a
   frame costs about max(hands, pose) instead of hands + pose.
3. :class:`HandPoseSmoother` assigns the hands to the two screen-side slots
   (:class:`HandSlotAssigner`), smooths all landmarks with One Euro filters in
   normalised coordinates, lets vanished hands coast for ``hand_lost_grace`` seconds
   and maps everything to scene pixels of the *full* frame.
4. The pose segmentation mask is downscaled to the aura resolution as uint8.
5. An immutable :class:`TrackingFrame` is published (:meth:`HandPoseTracker.latest`).

Only this module imports MediaPipe. The import is lazy (it takes about half a second
and can fail on a broken installation); every failure becomes a
``TrackerStatus.error`` message instead of an exception in the main loop.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

import cv2
import numpy as np

from dragon_ball_powers.config import MODELS_DIR, POSE_MODELS, TrackingSettings
from dragon_ball_powers.datatypes import (
    PALM_INDICES,
    CameraFrame,
    HandLandmark,
    HandObservation,
    PoseLandmark,
    PoseObservation,
    TrackerStatus,
    TrackingFrame,
)
from dragon_ball_powers.filters import OneEuroFilter
from dragon_ball_powers.interfaces import FrameSource
from dragon_ball_powers.mathutil import Vec2, v_dist

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------- constants

HAND_MODEL_FILE = "hand_landmarker.task"
SETUP_HINT = "Run setup.bat to download the models."
SIDES: tuple[str, str] = ("left", "right")

#: Velocity decay (1/s) of a coasting hand. The extrapolated displacement after
#: ``age`` seconds is ``v * (1 - exp(-k * age)) / k`` (velocity damped by exp(-k * age)).
COAST_DAMPING = 6.0
#: Cap on the extrapolation speed of a coasting landmark (normalised units per second);
#: protects against a wild last derivative (e.g. a glitchy final detection).
MAX_COAST_SPEED = 3.0
#: A slot's last position keeps steering the assignment this long after hand_lost_grace.
SLOT_MEMORY_EXTRA = 0.3
#: Pose filters restart when the pose was missing for longer than this (seconds).
POSE_RESET_AFTER = 0.5
#: The shoulders must be reversed by this fraction of the frame width before
#: ``model_left_is_screen_left`` flips (no flicker when the user turns sideways).
SHOULDER_FLIP_MARGIN = 0.03
#: Two same-handedness detections whose landmarks are on average closer than this
#: fraction of the palm length are one physical hand reported twice.
DUPLICATE_HAND_TOLERANCE = 0.3
#: Handedness labels below this score are ignored by the slot assigner.
MIN_LABEL_SCORE = 0.6
#: Labels at least this confident may override position-based slot matching.
LABEL_OVERRIDE_SCORE = 0.8
#: A slot that missed at least one frame and was not updated for longer than this (s)
#: re-appears (it is not tracked continuously).
TRACKING_GAP = 0.1
#: Reachability gate for matching a detection to a slot's last position, in frame
#: widths: JUMP_BASE + JUMP_SPEED * seconds since the slot was last seen.
JUMP_BASE = 0.25
JUMP_SPEED = 2.0

MAX_CONSECUTIVE_FAILURES = 10   # then the landmarker(s) are recreated
ERROR_LOG_INTERVAL = 5.0        # s between log lines of the same repeating error
MISSING_MODEL_RETRY = 5.0       # s between checks whether a missing model appeared
FRAME_WAIT_TIMEOUT = 0.5        # s, FrameSource.wait_for_frame timeout
STOP_TIMEOUT = 3.0              # s, join timeout of stop()
SOURCE_IDLE_RESET = 1.0         # s without frames before frame ids may restart
STATUS_SMOOTHING = 0.1          # EMA factor of the status timings
FPS_STALE_AFTER = 1.5           # s without results before status.fps reads 0
MIN_MASK_WIDTH, MAX_MASK_WIDTH = 16, 2048

_PALM = np.array(PALM_INDICES)
_WRIST = int(HandLandmark.WRIST)
_MIDDLE_MCP = int(HandLandmark.MIDDLE_MCP)
_L_SHOULDER = int(PoseLandmark.LEFT_SHOULDER)
_R_SHOULDER = int(PoseLandmark.RIGHT_SHOULDER)
# Pose score = mean visibility of the landmarks the gestures rely on.
_POSE_SCORE_INDICES = np.array([int(PoseLandmark.NOSE), _L_SHOULDER, _R_SHOULDER,
                                int(PoseLandmark.LEFT_ELBOW), int(PoseLandmark.RIGHT_ELBOW),
                                int(PoseLandmark.LEFT_WRIST), int(PoseLandmark.RIGHT_WRIST)])


def pose_model_file(name: str) -> str:
    """File name of a pose landmarker variant ("lite" | "full" | "heavy")."""
    return f"pose_landmarker_{name}.task"


# ------------------------------------------------------------------ errors / import


class ModelMissingError(FileNotFoundError):
    """A MediaPipe model file does not exist (or is empty)."""

    def __init__(self, path: Path) -> None:
        super().__init__(f"Tracking model missing: {path}")
        self.path = path
        self.file_name = path.name


class TrackingBackendError(RuntimeError):
    """MediaPipe itself cannot be used (import failed)."""


@dataclass(frozen=True)
class _MediaPipe:
    mp: Any       # the ``mediapipe`` package (Image, ImageFormat)
    tasks: Any    # mediapipe.tasks.python (BaseOptions)
    vision: Any   # mediapipe.tasks.python.vision (landmarkers, RunningMode)


_mp_lock = threading.Lock()
_mp_cache: _MediaPipe | None = None
_mp_error: str | None = None


def _load_mediapipe() -> _MediaPipe:
    """Import MediaPipe on first use (slow; may fail on a broken install)."""
    global _mp_cache, _mp_error
    with _mp_lock:
        if _mp_cache is None and _mp_error is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_tasks
                from mediapipe.tasks.python import vision
            except Exception as exc:  # ImportError, OSError ("DLL load failed"), ...
                _mp_error = f"{type(exc).__name__}: {exc}"
                log.exception("Could not import MediaPipe")
            else:
                _mp_cache = _MediaPipe(mp, mp_tasks, vision)
        if _mp_cache is None:
            raise TrackingBackendError(f"MediaPipe is not available ({_mp_error}). "
                                       "Run setup.bat to repair the installation.")
        return _mp_cache


def _read_model(path: Path) -> bytes:
    # The model is read by Python and passed as a buffer because MediaPipe cannot open
    # non-ASCII paths on Windows (e.g. umlauts in a German user name).
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise ModelMissingError(path) from None
    if not data:
        raise ModelMissingError(path)  # an aborted download: setup.bat fetches it again
    return data


# --------------------------------------------------------------- raw model output


@dataclass
class HandCandidate:
    """What the slot assigner needs to know about one detected hand (scene px)."""

    position: Vec2           # palm centre
    wrist_x: float
    handedness: str = ""     # model label "Left" / "Right"
    score: float = 0.0       # label confidence


@dataclass
class HandDetection:
    """One hand as reported by the hand model, in normalised image coordinates."""

    landmarks: np.ndarray            # (21, 3) float64: x, y in 0..1 of the frame, z in frame widths
    world: np.ndarray | None = None  # (21, 3) float64 metres, hand-centred
    handedness: str = ""             # "Left" / "Right" (user's hand on mirrored input)
    score: float = 0.0

    def candidate(self, frame_size: tuple[int, int]) -> HandCandidate:
        w, h = frame_size
        palm = self.landmarks[_PALM, :2].mean(axis=0)
        return HandCandidate(position=(float(palm[0] * w), float(palm[1] * h)),
                             wrist_x=float(self.landmarks[_WRIST, 0] * w),
                             handedness=self.handedness, score=float(self.score))


@dataclass
class PoseDetection:
    """The pose as reported by the pose model, in normalised image coordinates."""

    landmarks: np.ndarray            # (33, 3) float64: x, y in 0..1, z in frame widths
    visibility: np.ndarray           # (33,) float64: min(visibility, presence)
    world: np.ndarray | None = None  # (33, 3) float64 metres, hip-centred


class HandModel(Protocol):
    """A hand landmark model (MediaPipe or a test fake). Called by one thread at a time."""

    def detect(self, rgb: np.ndarray, capture_time: float) -> list[HandDetection]: ...

    def close(self) -> None: ...


class PoseModel(Protocol):
    """A pose landmark model; ``mask_size`` = (w, h) of the uint8 mask, None = no mask."""

    def detect(self, rgb: np.ndarray, capture_time: float, mask_size: tuple[int, int] | None
               ) -> tuple[PoseDetection | None, np.ndarray | None]: ...

    def close(self) -> None: ...


def next_timestamp_ms(last_ms: int, capture_time: float) -> int:
    """MediaPipe VIDEO-mode timestamp: ``max(last + 1, int(capture_time * 1000))``."""
    try:
        candidate = int(capture_time * 1000.0)
    except (ValueError, OverflowError):  # NaN / inf capture time
        candidate = last_ms + 1
    return max(last_ms + 1, candidate)


def prepare_model_input(image: np.ndarray, tracking_width: int) -> np.ndarray:
    """Downscale a BGR(A)/grey uint8 frame to ``tracking_width`` (never upscale) -> RGB."""
    h, w = image.shape[:2]
    target_w = max(16, int(tracking_width))
    if w > target_w:
        target_h = max(1, int(round(h * target_w / w)))
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    channels = 1 if image.ndim == 2 else image.shape[2]
    code = {1: cv2.COLOR_GRAY2RGB, 3: cv2.COLOR_BGR2RGB, 4: cv2.COLOR_BGRA2RGB}[channels]
    return np.ascontiguousarray(cv2.cvtColor(image, code))


def _check_image(image: Any) -> None:
    ok = (isinstance(image, np.ndarray) and image.dtype == np.uint8 and image.size > 0
          and (image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 3, 4))))
    if not ok:
        desc = (f"{image.dtype} {image.shape}" if isinstance(image, np.ndarray)
                else type(image).__name__)
        raise ValueError(f"Unsupported camera image ({desc}); expected uint8 BGR")


def _landmark_rows(landmarks: Sequence[Any], with_scores: bool = False) -> np.ndarray:
    # Python loop over <= 33 objects is unavoidable: MediaPipe returns dataclasses.
    # None fields become NaN (handled downstream).
    if with_scores:
        rows = [(p.x, p.y, p.z, p.visibility, p.presence) for p in landmarks]
        return np.array(rows, dtype=np.float64).reshape(-1, 5)
    return np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float64).reshape(-1, 3)


def _parse_hands(result: Any) -> list[HandDetection]:
    detections: list[HandDetection] = []
    world_all = result.hand_world_landmarks or []
    labels_all = result.handedness or []
    for i, landmarks in enumerate(result.hand_landmarks or []):
        lm = _landmark_rows(landmarks)
        if lm.shape != (21, 3) or not np.isfinite(lm).all():
            continue
        world = _landmark_rows(world_all[i]) if i < len(world_all) else None
        if world is not None and (world.shape != (21, 3) or not np.isfinite(world).all()):
            world = None
        label, score = "", 0.0
        if i < len(labels_all) and labels_all[i]:
            category = labels_all[i][0]
            label = category.category_name or ""
            score = float(category.score or 0.0)
        detections.append(HandDetection(lm, world, label, score))
    return detections


def _parse_pose(result: Any, mask_size: tuple[int, int] | None
                ) -> tuple[PoseDetection | None, np.ndarray | None]:
    poses = result.pose_landmarks or []
    if not poses:
        return None, None
    rows = _landmark_rows(poses[0], with_scores=True)
    if rows.shape != (33, 5) or not np.isfinite(rows[:, :3]).all():
        return None, None
    visibility = np.fmin(rows[:, 3], rows[:, 4])          # fmin ignores a missing (NaN) value
    visibility = np.clip(np.where(np.isnan(visibility), 1.0, visibility), 0.0, 1.0)
    world = None
    if result.pose_world_landmarks:
        world = _landmark_rows(result.pose_world_landmarks[0])
        if world.shape != (33, 3) or not np.isfinite(world).all():
            world = None
    mask = None
    if mask_size is not None and result.segmentation_masks:
        view = result.segmentation_masks[0].numpy_view()
        if view.ndim == 3:
            view = view[..., 0]
        # ``view`` points into MediaPipe-owned memory: resizing copies it out right away.
        small = cv2.resize(view, mask_size, interpolation=cv2.INTER_AREA)
        mask = np.ascontiguousarray(cv2.convertScaleAbs(small, alpha=255.0))
    return PoseDetection(rows[:, :3].copy(), visibility, world), mask


class _MediaPipeHandModel:
    """HandLandmarker (VIDEO mode) returning plain NumPy :class:`HandDetection` objects."""

    def __init__(self, landmarker: Any, mp_module: Any) -> None:
        self._landmarker = landmarker
        self._mp = mp_module
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, capture_time: float) -> list[HandDetection]:
        # Advance before the call so a failed call can never cause a repeated timestamp.
        self._last_ts = next_timestamp_ms(self._last_ts, capture_time)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return _parse_hands(self._landmarker.detect_for_video(image, self._last_ts))

    def close(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception as exc:  # closing twice raises ValueError in MediaPipe
                log.debug("Closing the hand landmarker failed: %s", exc)


class _MediaPipePoseModel:
    """PoseLandmarker (VIDEO mode) returning :class:`PoseDetection` + uint8 mask."""

    def __init__(self, landmarker: Any, mp_module: Any) -> None:
        self._landmarker = landmarker
        self._mp = mp_module
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, capture_time: float, mask_size: tuple[int, int] | None
               ) -> tuple[PoseDetection | None, np.ndarray | None]:
        self._last_ts = next_timestamp_ms(self._last_ts, capture_time)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return _parse_pose(self._landmarker.detect_for_video(image, self._last_ts), mask_size)

    def close(self) -> None:
        landmarker, self._landmarker = self._landmarker, None
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception as exc:
                log.debug("Closing the pose landmarker failed: %s", exc)


# ------------------------------------------------------------------- hand slots


@dataclass(frozen=True)
class SlotAssignment:
    """Result of :meth:`HandSlotAssigner.assign` for one candidate."""

    side: str          # "left" | "right"
    index: int         # index into the candidate list
    continuous: bool   # the slot's previous hand continues (keep its filter state)


class HandSlotAssigner:
    """Maps hand detections to the two screen-side slots ("left" = smaller x).

    Rules:

    * **Two detections, both slots seen recently** (within ``recent_window``): the
      assignment with the smaller total distance to the slots' last positions. This is
      the hysteresis that keeps touching or crossing hands from swapping every frame.
      Exception: when that assignment is crossed relative to the wrist-x order and both
      handedness labels confidently confirm the x order, the x order wins (repairs an
      earlier mix-up; physically crossed arms keep their labels, so they stay crossed).
    * **Two detections otherwise**: sorted by wrist x.
    * **One detection**: the nearest recently seen slot whose last position is
      plausibly reachable (``JUMP_BASE + JUMP_SPEED * elapsed`` frame widths). A
      confident handedness label overrides that match when the matched slot re-appears:
      it missed at least one frame and was not updated for more than ``TRACKING_GAP``
      (a continuously tracked hand keeps its slot at any frame rate). Without a
      reachable slot: the handedness label ("Left" -> "left"), finally the side of the
      frame centre.

    Call :meth:`assign` once per analysed frame, also when nothing was detected (with
    an empty list), so the assigner knows which slots missed frames.

    ``SlotAssignment.continuous`` tells the caller whether the slot's filters may keep
    their state (same hand) or must restart (new hand, identity swap, implausible jump).
    """

    def __init__(self, recent_window: float = 0.55) -> None:
        self.recent_window = float(recent_window)
        self._position: dict[str, Vec2 | None] = {side: None for side in SIDES}
        self._seen: dict[str, float | None] = {side: None for side in SIDES}
        self._previous_call: float | None = None   # ``now`` of the previous assign()

    def reset(self) -> None:
        for side in SIDES:
            self._position[side] = None
            self._seen[side] = None
        self._previous_call = None

    def last_position(self, side: str) -> Vec2 | None:
        return self._position[side]

    def last_seen(self, side: str) -> float | None:
        return self._seen[side]

    def is_recent(self, side: str, now: float) -> bool:
        seen = self._seen[side]
        return seen is not None and now - seen <= self.recent_window

    def assign(self, candidates: Sequence[HandCandidate], now: float,
               frame_width: float) -> list[SlotAssignment]:
        """Assign up to two candidates (best scores) to slots; updates the slot memory."""
        previous_call, self._previous_call = self._previous_call, float(now)
        if not candidates:
            return []
        order = list(range(len(candidates)))
        if len(order) > 2:
            order = sorted(order, key=lambda i: candidates[i].score, reverse=True)[:2]
        width = max(1.0, float(frame_width))
        recent = {side: self.is_recent(side, now) for side in SIDES}

        by_distance = False
        if len(order) == 2:
            first, second = sorted(order, key=lambda i: (candidates[i].wrist_x,
                                                         candidates[i].position[0]))
            left_index, right_index = first, second
            if recent["left"] and recent["right"]:
                left_pos, right_pos = self._position["left"], self._position["right"]
                keep = (v_dist(candidates[first].position, left_pos)
                        + v_dist(candidates[second].position, right_pos))
                swap = (v_dist(candidates[first].position, right_pos)
                        + v_dist(candidates[second].position, left_pos))
                by_distance = True
                if swap < keep:
                    if self._labels_confirm(candidates[first], candidates[second]):
                        by_distance = False   # keep the x order: the labels say so
                    else:
                        left_index, right_index = second, first
            sides = {"left": left_index, "right": right_index}
        else:
            index = order[0]
            sides = {self._side_for_single(candidates[index], now, width, recent,
                                           previous_call): index}

        result: list[SlotAssignment] = []
        for side, index in sides.items():
            position = candidates[index].position
            continuous = recent[side] and self._reachable(side, position, now, width)
            if continuous and len(sides) == 2 and not by_distance:
                # Sorted by x: the slot keeps its hand only if no other detection is
                # nearer to where its hand was (otherwise the identities swapped).
                other = sides["right" if side == "left" else "left"]
                last = self._position[side]
                continuous = v_dist(position, last) <= v_dist(candidates[other].position, last)
            result.append(SlotAssignment(side, index, continuous))
        for assignment in result:
            self._position[assignment.side] = candidates[assignment.index].position
            self._seen[assignment.side] = now
        result.sort(key=lambda a: SIDES.index(a.side))
        return result

    # ------------------------------------------------------------------ helpers

    def _reachable(self, side: str, position: Vec2, now: float, width: float) -> bool:
        last, seen = self._position[side], self._seen[side]
        if last is None or seen is None:
            return False
        elapsed = max(0.0, now - seen)
        return v_dist(position, last) <= (JUMP_BASE + JUMP_SPEED * elapsed) * width

    @staticmethod
    def _label_side(candidate: HandCandidate, min_score: float) -> str | None:
        label = candidate.handedness.strip().lower()
        if label in SIDES and candidate.score >= min_score:
            return label
        return None

    def _labels_confirm(self, x_left: HandCandidate, x_right: HandCandidate) -> bool:
        return (self._label_side(x_left, LABEL_OVERRIDE_SCORE) == "left"
                and self._label_side(x_right, LABEL_OVERRIDE_SCORE) == "right")

    def _side_for_single(self, candidate: HandCandidate, now: float, width: float,
                         recent: dict[str, bool], previous_call: float | None) -> str:
        reachable = [side for side in SIDES
                     if recent[side] and self._reachable(side, candidate.position, now, width)]
        label = self._label_side(candidate, MIN_LABEL_SCORE)
        if reachable:
            nearest = min(reachable, key=lambda s: v_dist(candidate.position, self._position[s]))
            seen = self._seen[nearest]
            # Re-appeared = missed at least one frame AND gone for > TRACKING_GAP. The time
            # gap alone is not enough: below 10 fps every frame is more than TRACKING_GAP
            # apart, and a flickering label would swap a continuously tracked hand
            # between the slots on every frame.
            reappeared = (seen is not None and previous_call is not None
                          and seen < previous_call and now - seen > TRACKING_GAP)
            if (label is not None and label != nearest and reappeared
                    and candidate.score >= LABEL_OVERRIDE_SCORE):
                return label
            return nearest
        if label is not None:
            return label
        return "left" if candidate.position[0] < width * 0.5 else "right"


# -------------------------------------------------------------------- smoothing


def _drop_duplicate_hands(detections: list[HandDetection],
                          frame_size: tuple[int, int]) -> list[HandDetection]:
    """Remove a second report of the same physical hand (same label, same landmarks)."""
    if len(detections) != 2:
        return detections
    a, b = detections
    if not a.handedness or a.handedness != b.handedness:
        return detections  # two real hands pressed together have opposite labels
    scale = np.array(frame_size, dtype=np.float64)
    pa, pb = a.landmarks[:, :2] * scale, b.landmarks[:, :2] * scale
    palm = min(np.linalg.norm(pa[_MIDDLE_MCP] - pa[_WRIST]),
               np.linalg.norm(pb[_MIDDLE_MCP] - pb[_WRIST]))
    if palm > 1e-6 and np.linalg.norm(pa - pb, axis=1).mean() < DUPLICATE_HAND_TOLERANCE * palm:
        log.debug("Dropping a duplicate %s hand detection", a.handedness)
        return [a if a.score >= b.score else b]
    return detections


class _HandTrack:
    """Smoothing and coasting state of one screen-side hand slot."""

    def __init__(self, side: str) -> None:
        self.side = side
        self.landmarks = OneEuroFilter()   # normalised (x/W, y/H, z/W)
        self.world = OneEuroFilter()       # metres
        self.last_seen: float | None = None
        self.handedness = ""
        self.score = 0.0
        self.world_value: np.ndarray | None = None

    def set_params(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.landmarks.set_params(min_cutoff, beta, d_cutoff)
        self.world.set_params(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.landmarks.reset()
        self.world.reset()
        self.last_seen = None
        self.world_value = None

    def is_lost(self, t: float, grace: float) -> bool:
        return self.last_seen is None or t - self.last_seen > grace

    def observe(self, detection: HandDetection, t: float, scale: np.ndarray,
                restart: bool) -> HandObservation:
        if restart:
            self.landmarks.reset()
            self.world.reset()
            self.world_value = None
        norm = self.landmarks(detection.landmarks, t)
        if detection.world is not None:
            self.world_value = self.world(detection.world, t).astype(np.float32)
        self.last_seen = t
        self.handedness = detection.handedness
        self.score = float(detection.score)
        return HandObservation(side=self.side, landmarks=(norm * scale).astype(np.float32),
                               world=None if self.world_value is None else self.world_value.copy(),
                               handedness=self.handedness, score=self.score,
                               source="hand", age=0.0)

    def coast(self, t: float, grace: float, scale: np.ndarray) -> HandObservation | None:
        """Damped extrapolation while the hand model does not see this hand."""
        if self.is_lost(t, grace):
            return None
        base, velocity = self.landmarks.value, self.landmarks.derivative
        if base is None or velocity is None:
            return None
        age = max(0.0, t - self.last_seen)  # type: ignore[operator]  # not lost -> seen
        speed = np.linalg.norm(velocity, axis=1, keepdims=True)
        velocity = velocity * np.minimum(1.0, MAX_COAST_SPEED / np.maximum(speed, 1e-9))
        travel = (1.0 - math.exp(-COAST_DAMPING * age)) / COAST_DAMPING
        norm = base + velocity * travel
        return HandObservation(side=self.side, landmarks=(norm * scale).astype(np.float32),
                               world=None if self.world_value is None else self.world_value.copy(),
                               handedness=self.handedness, score=self.score,
                               source="predicted", age=age)


class _PoseTrack:
    """Smoothing and screen-orientation state of the pose."""

    def __init__(self) -> None:
        self.landmarks = OneEuroFilter()
        self.world = OneEuroFilter()
        self.last_seen: float | None = None
        self.model_left_is_screen_left: bool | None = None
        self.world_value: np.ndarray | None = None

    def set_params(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.landmarks.set_params(min_cutoff, beta, d_cutoff)
        self.world.set_params(min_cutoff, beta, d_cutoff)

    def reset(self) -> None:
        self.landmarks.reset()
        self.world.reset()
        self.last_seen = None
        self.model_left_is_screen_left = None
        self.world_value = None

    def observe(self, detection: PoseDetection, t: float, scale: np.ndarray) -> PoseObservation:
        if self.last_seen is None or t - self.last_seen > POSE_RESET_AFTER:
            self.landmarks.reset()
            self.world.reset()
            self.world_value = None
        norm = self.landmarks(detection.landmarks, t)
        if detection.world is not None:
            self.world_value = self.world(detection.world, t).astype(np.float32)
        self.last_seen = t
        visibility = np.clip(np.asarray(detection.visibility, dtype=np.float32), 0.0, 1.0)
        self._update_orientation(norm, visibility)
        return PoseObservation(
            landmarks=(norm * scale).astype(np.float32),
            visibility=visibility,
            world=None if self.world_value is None else self.world_value.copy(),
            score=float(visibility[_POSE_SCORE_INDICES].mean()),
            model_left_is_screen_left=bool(self.model_left_is_screen_left))

    def _update_orientation(self, norm: np.ndarray, visibility: np.ndarray) -> None:
        # dx > 0: the model's LEFT shoulder is right of its RIGHT shoulder on screen, the
        # normal case for a user facing a mirrored camera.
        dx = float(norm[_L_SHOULDER, 0] - norm[_R_SHOULDER, 0])
        current = self.model_left_is_screen_left
        if current is None:
            self.model_left_is_screen_left = dx <= 0.0
            return
        if visibility[_L_SHOULDER] < 0.5 or visibility[_R_SHOULDER] < 0.5:
            return  # guessed shoulders are no evidence for a flip
        if current and dx > SHOULDER_FLIP_MARGIN:
            self.model_left_is_screen_left = False
        elif not current and dx < -SHOULDER_FLIP_MARGIN:
            self.model_left_is_screen_left = True


class HandPoseSmoother:
    """Turns raw per-frame detections into slot-assigned, smoothed observations.

    Pure logic (no MediaPipe, no threads): slot assignment, One Euro smoothing in
    normalised coordinates, coasting of vanished hands, pose orientation. Times are
    camera capture times in seconds. Not thread-safe (the tracker serialises calls).
    """

    def __init__(self, settings: TrackingSettings) -> None:
        self._grace = 0.25
        self._assigner = HandSlotAssigner()
        self._hands = {side: _HandTrack(side) for side in SIDES}
        self._pose = _PoseTrack()
        self._last_time: float | None = None
        self._frame_size: tuple[int, int] | None = None
        self.apply_settings(settings)

    @property
    def assigner(self) -> HandSlotAssigner:
        return self._assigner

    def apply_settings(self, settings: TrackingSettings) -> None:
        """Smoothing parameters and the coasting grace apply from the next update."""
        self._grace = max(0.0, float(settings.hand_lost_grace))
        self._assigner.recent_window = self._grace + SLOT_MEMORY_EXTRA
        params = (settings.smoothing_min_cutoff, settings.smoothing_beta,
                  settings.smoothing_d_cutoff)
        for track in self._hands.values():
            track.set_params(*params)
        self._pose.set_params(*params)

    def reset(self) -> None:
        """Forget all hands, the pose and their filter state."""
        self._assigner.reset()
        for track in self._hands.values():
            track.reset()
        self._pose.reset()
        self._last_time = None
        self._frame_size = None

    def update(self, hands: Sequence[HandDetection] | None, pose: PoseDetection | None,
               t: float, frame_size: tuple[int, int]
               ) -> tuple[list[HandObservation], PoseObservation | None]:
        """Process one frame's detections (``t`` = capture time, ``frame_size`` = (W, H))."""
        w, h = int(frame_size[0]), int(frame_size[1])
        if self._last_time is not None and t < self._last_time:
            log.debug("Tracking time went backwards (%.3f -> %.3f); resetting", self._last_time, t)
            self.reset()
        elif self._frame_size is not None and self._frame_size != (w, h):
            # Another camera / resolution: normalised coordinates of the old frames
            # describe different image content, so nothing may be blended across.
            log.debug("Frame size changed %s -> %s; resetting", self._frame_size, (w, h))
            self.reset()
        self._last_time = t
        self._frame_size = (w, h)
        scale = np.array([w, h, w], dtype=np.float64)

        detections = [d for d in (hands or ()) if d.landmarks.shape == (21, 3)]
        detections = _drop_duplicate_hands(detections, (w, h))
        candidates = [d.candidate((w, h)) for d in detections]
        assigned = {a.side: a for a in self._assigner.assign(candidates, t, float(w))}

        observations: list[HandObservation] = []
        for side in SIDES:
            track = self._hands[side]
            assignment = assigned.get(side)
            if assignment is not None:
                restart = not assignment.continuous or track.is_lost(t, self._grace)
                observations.append(track.observe(detections[assignment.index], t, scale, restart))
            else:
                predicted = track.coast(t, self._grace, scale)
                if predicted is not None:
                    observations.append(predicted)
        pose_obs = self._pose.observe(pose, t, scale) if pose is not None else None
        return observations, pose_obs


# ---------------------------------------------------------------------- tracker


@dataclass(frozen=True)
class ModelConfig:
    """The settings baked into the landmarker instances (a change means recreate)."""

    max_hands: int = 2
    min_hand_detection_confidence: float = 0.5
    min_hand_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    pose_model: str = "lite"
    min_pose_detection_confidence: float = 0.5
    min_pose_presence_confidence: float = 0.5
    enable_pose: bool = True
    enable_segmentation: bool = True

    @classmethod
    def from_settings(cls, s: TrackingSettings) -> "ModelConfig":
        return cls(max_hands=int(min(2, max(1, s.max_hands))),
                   min_hand_detection_confidence=float(s.min_hand_detection_confidence),
                   min_hand_presence_confidence=float(s.min_hand_presence_confidence),
                   min_tracking_confidence=float(s.min_tracking_confidence),
                   pose_model=s.pose_model if s.pose_model in POSE_MODELS else "lite",
                   min_pose_detection_confidence=float(s.min_pose_detection_confidence),
                   min_pose_presence_confidence=float(s.min_pose_presence_confidence),
                   enable_pose=bool(s.enable_pose),
                   enable_segmentation=bool(s.enable_segmentation))

    def key(self, kind: str) -> tuple[Any, ...]:
        """The fields that matter for one landmarker ("hands" | "pose")."""
        if kind == "hands":
            return (self.max_hands, self.min_hand_detection_confidence,
                    self.min_hand_presence_confidence, self.min_tracking_confidence)
        return (self.enable_pose, self.pose_model, self.min_pose_detection_confidence,
                self.min_pose_presence_confidence, self.min_tracking_confidence,
                self.enable_segmentation)


@dataclass
class _ModelHolder:
    """One landmarker slot of the tracker and its error bookkeeping."""

    kind: str                          # "hands" | "pose"
    label: str                         # "Hand" | "Pose" (messages)
    model: Any = None
    key: tuple[Any, ...] | None = None  # ModelConfig.key() the model was created with
    missing_file: str | None = None
    load_error: str | None = None
    infer_error: str | None = None
    failures: int = 0                  # consecutive inference failures
    last_attempt: float = -math.inf

    def clear_errors(self) -> None:
        self.missing_file = None
        self.load_error = None
        self.infer_error = None
        self.failures = 0
        self.last_attempt = -math.inf


class _ErrorReporter:
    """Logs a repeating error at most once per interval (per key), with counts."""

    def __init__(self, interval: float = ERROR_LOG_INTERVAL) -> None:
        self._interval = interval
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}

    def report(self, key: str, message: str, exc: BaseException) -> None:
        now = time.perf_counter()
        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < self._interval:
                self._suppressed[key] = self._suppressed.get(key, 0) + 1
                return
            suppressed = self._suppressed.pop(key, 0)
            self._last[key] = now
        extra = f" ({suppressed} similar errors since the last report)" if suppressed else ""
        log.error("%s: %s%s", message, exc, extra, exc_info=(type(exc), exc, exc.__traceback__))


@dataclass
class _ModelOutputs:
    hands: list[HandDetection] | None = None   # None = the hand model did not run / failed
    pose: PoseDetection | None = None
    mask: np.ndarray | None = None
    hands_ms: float = 0.0
    pose_ms: float = 0.0
    pose_ran: bool = False


@dataclass
class _Stats:
    fps: float = 0.0
    interval: float = 0.0
    hands_ms: float = 0.0
    pose_ms: float = 0.0
    total_ms: float = 0.0
    latency_ms: float = 0.0
    last_result: float | None = None

    def add(self, frame: TrackingFrame, ran_hands: bool, ran_pose: bool) -> None:
        def ema(old: float, new: float) -> float:
            return new if old <= 0.0 else old + STATUS_SMOOTHING * (new - old)

        if self.last_result is not None:
            interval = frame.result_time - self.last_result
            if 0.0 < interval < 2.0:
                self.interval = ema(self.interval, interval)
                self.fps = 1.0 / self.interval
        self.last_result = frame.result_time
        ms = frame.inference_ms
        self.hands_ms = ema(self.hands_ms, ms["hands"]) if ran_hands else 0.0
        self.pose_ms = ema(self.pose_ms, ms["pose"]) if ran_pose else 0.0
        self.total_ms = ema(self.total_ms, ms["total"])
        self.latency_ms = ema(self.latency_ms, max(0.0, frame.latency_ms))


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - start) * 1000.0


def _clamp_mask_width(width: int) -> int:
    return int(min(MAX_MASK_WIDTH, max(MIN_MASK_WIDTH, int(width))))


class HandPoseTracker:
    """MediaPipe hand + pose tracking on camera frames, in a background thread.

    ``start(source)`` launches the daemon thread "tracker", which always analyses the
    newest frame of ``source`` (stale frames are dropped) and publishes one
    :class:`TrackingFrame` per analysed frame (:meth:`latest`). :meth:`process_frame`
    runs the identical pipeline synchronously (tests, tools). All public methods are
    thread-safe. Landmarkers are created lazily (first frame) and released by
    :meth:`stop`; start/stop can be repeated.

    Missing model files or a broken MediaPipe installation never raise: they are
    reported in :attr:`status` (``error``) and tracking continues with whatever works
    (hands only, pose only, or no detections at all).
    """

    def __init__(self, settings: TrackingSettings, models_dir: Path = MODELS_DIR, *,
                 mask_width: int = 256) -> None:
        self._models_dir = Path(models_dir)
        # Lock order: _proc_lock (pipeline + model lifecycle) before _lock (shared state).
        self._lock = threading.Lock()
        self._proc_lock = threading.Lock()
        self._settings = dataclasses.replace(settings)
        self._settings_changed = False
        self._mask_width = _clamp_mask_width(mask_width)
        self._smoother = HandPoseSmoother(self._settings)
        self._hands = _ModelHolder("hands", "Hand")
        self._pose = _ModelHolder("pose", "Pose")
        self._models_initialised = False
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._errors = _ErrorReporter()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._source: FrameSource | None = None
        # published state (guarded by _lock)
        self._latest: TrackingFrame | None = None
        self._segmentation_id = 0
        self._stats = _Stats()
        self._error_text: str | None = None
        self._pipeline_error: str | None = None
        self._fatal_error: str | None = None
        self._pose_model_name = ""

    # --------------------------------------------------------------- public API

    def start(self, source: FrameSource) -> None:
        """Start the tracker thread on ``source`` (no-op if already running on it)."""
        with self._lock:
            alive = self._thread is not None and self._thread.is_alive()
            if alive and self._source is source:
                return
        if alive:
            self.stop()
        with self._lock:
            self._source = source
            self._stop_event = threading.Event()
            self._latest = None
            self._fatal_error = None
            self._pipeline_error = None
            self._thread = threading.Thread(target=self._run, args=(source, self._stop_event),
                                            name="tracker", daemon=True)
            thread = self._thread
        thread.start()

    def stop(self) -> None:
        """Stop the thread (join up to 3 s), shut down the workers, close the landmarkers.

        If the thread does not finish in time (e.g. a very slow first model load), it
        releases the workers and landmarkers itself when it exits.
        """
        with self._lock:
            thread, stop_event = self._thread, self._stop_event
            self._thread = None
            self._stop_event = None
            self._source = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(STOP_TIMEOUT)
            if thread.is_alive():
                # E.g. a very slow first model load (MediaPipe import + virus scan on
                # Windows). The thread releases the models itself when it finishes.
                log.warning("Tracker thread did not stop within %.1f s; it releases the "
                            "tracking models when it finishes", STOP_TIMEOUT)
                return
        if not self._proc_lock.acquire(timeout=STOP_TIMEOUT):
            log.warning("Tracker is busy; could not release the tracking models")
            return
        try:
            self._release_locked()
        finally:
            self._proc_lock.release()

    def close(self) -> None:
        """Same as :meth:`stop`."""
        self.stop()

    def __enter__(self) -> "HandPoseTracker":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def latest(self) -> TrackingFrame | None:
        """The newest published tracking result (or None)."""
        with self._lock:
            return self._latest

    def reset(self) -> None:
        """Forget hand slots, smoothing state and the latest result (models stay loaded)."""
        with self._proc_lock:
            self._smoother.reset()
            with self._lock:
                self._latest = None

    def process_frame(self, frame: CameraFrame) -> TrackingFrame:
        """Run the full pipeline on ``frame`` synchronously and publish the result."""
        with self._proc_lock:
            return self._process_locked(frame)

    def update_settings(self, settings: TrackingSettings) -> None:
        """Apply new tracking settings.

        Smoothing, coasting and ``tracking_width`` changes apply from the next frame.
        Model-relevant changes (pose model, confidences, max hands, pose/segmentation
        toggles) recreate the affected landmarkers: in the tracker thread before its
        next frame, or right here when the thread is not running.
        """
        new = dataclasses.replace(settings)
        with self._lock:
            self._settings = new
            self._settings_changed = True
            running = self._thread is not None and self._thread.is_alive()
        if running:
            return
        with self._proc_lock:
            current, _, changed = self._take_settings()
            if changed:
                self._smoother.apply_settings(current)
            if self._models_initialised:
                self._ensure_models_locked(current)

    def set_mask_width(self, width: int) -> None:
        """Width of the uint8 segmentation mask (aura resolution) from the next frame."""
        with self._lock:
            self._mask_width = _clamp_mask_width(width)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> TrackerStatus:
        """Thread-safe snapshot of the tracker state and timings."""
        now = time.perf_counter()
        with self._lock:
            stats = self._stats
            fresh = stats.last_result is not None and now - stats.last_result < FPS_STALE_AFTER
            return TrackerStatus(
                running=self._thread is not None and self._thread.is_alive(),
                error=self._error_text,
                fps=stats.fps if fresh else 0.0,
                hands_ms=stats.hands_ms,
                pose_ms=stats.pose_ms,
                total_ms=stats.total_ms,
                latency_ms=stats.latency_ms,
                pose_model=self._pose_model_name)

    # -------------------------------------------------------- model factories

    def _create_hand_model(self, config: ModelConfig) -> HandModel:
        """Create the hand model (overridable seam for tests)."""
        data = _read_model(self._models_dir / HAND_MODEL_FILE)
        mp = _load_mediapipe()
        options = mp.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=data),
            running_mode=mp.vision.RunningMode.VIDEO,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.min_hand_detection_confidence,
            min_hand_presence_confidence=config.min_hand_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence)
        return _MediaPipeHandModel(mp.vision.HandLandmarker.create_from_options(options), mp.mp)

    def _create_pose_model(self, config: ModelConfig) -> PoseModel:
        """Create the pose model (overridable seam for tests)."""
        data = _read_model(self._models_dir / pose_model_file(config.pose_model))
        mp = _load_mediapipe()
        options = mp.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=data),
            running_mode=mp.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.min_pose_detection_confidence,
            min_pose_presence_confidence=config.min_pose_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            output_segmentation_masks=config.enable_segmentation)
        return _MediaPipePoseModel(mp.vision.PoseLandmarker.create_from_options(options), mp.mp)

    # ---------------------------------------------------------------- pipeline

    def _take_settings(self) -> tuple[TrackingSettings, int, bool]:
        with self._lock:
            changed = self._settings_changed
            self._settings_changed = False
            return self._settings, self._mask_width, changed

    def _process_locked(self, frame: CameraFrame) -> TrackingFrame:
        settings, mask_width, changed = self._take_settings()
        if changed:
            self._smoother.apply_settings(settings)
        self._ensure_models_locked(settings)

        start = time.perf_counter()
        image = frame.image
        _check_image(image)
        h, w = int(image.shape[0]), int(image.shape[1])
        capture_time = float(frame.timestamp)
        rgb = prepare_model_input(image, settings.tracking_width)
        mask_size = None
        if settings.enable_pose and settings.enable_segmentation:
            mask_size = (mask_width, max(1, int(round(mask_width * h / w))))

        outputs = self._run_models(rgb, capture_time, mask_size, settings.parallel_inference)
        hands, pose = self._smoother.update(outputs.hands, outputs.pose, capture_time, (w, h))

        result_time = time.perf_counter()
        with self._lock:
            segmentation = None
            if pose is not None and outputs.mask is not None:
                self._segmentation_id += 1
                segmentation = outputs.mask
            result = TrackingFrame(
                frame_id=frame.frame_id, capture_time=capture_time, result_time=result_time,
                scene_size=(w, h), image=image, hands=hands, pose=pose,
                segmentation=segmentation, segmentation_id=self._segmentation_id,
                inference_ms={"hands": outputs.hands_ms, "pose": outputs.pose_ms,
                              "total": (result_time - start) * 1000.0})
            self._latest = result
            self._stats.add(result, outputs.hands is not None, outputs.pose_ran)
            self._pipeline_error = None
            self._error_text = self._compose_error_locked()
        return result

    def _run_models(self, rgb: np.ndarray, capture_time: float,
                    mask_size: tuple[int, int] | None, parallel: bool) -> _ModelOutputs:
        """Run hands and pose (concurrently if enabled); failures are contained per model."""
        jobs: list[tuple[_ModelHolder, Callable[[], Any]]] = []
        hand_model, pose_model = self._hands.model, self._pose.model
        if hand_model is not None:
            jobs.append((self._hands, lambda: hand_model.detect(rgb, capture_time)))
        if pose_model is not None:
            jobs.append((self._pose, lambda: pose_model.detect(rgb, capture_time, mask_size)))

        outcomes: list[tuple[_ModelHolder, tuple[Any, float] | None, Exception | None]] = []
        if parallel and len(jobs) > 1:
            futures = [(holder, self._submit(holder.kind, fn)) for holder, fn in jobs]
            for holder, future in futures:  # always wait for every job before going on
                try:
                    outcomes.append((holder, future.result(), None))
                except Exception as exc:
                    outcomes.append((holder, None, exc))
        else:
            for holder, fn in jobs:
                try:
                    outcomes.append((holder, _timed(fn), None))
                except Exception as exc:
                    outcomes.append((holder, None, exc))

        outputs = _ModelOutputs()
        for holder, value, exc in outcomes:
            if exc is not None or value is None:
                self._inference_failed(holder, exc or RuntimeError("no result"))
                continue
            holder.failures = 0
            holder.infer_error = None
            data, ms = value
            if holder.kind == "hands":
                outputs.hands, outputs.hands_ms = data, ms
            else:
                outputs.pose, outputs.mask = data
                outputs.pose_ms = ms
                outputs.pose_ran = True
        return outputs

    def _submit(self, kind: str, fn: Callable[[], Any]) -> Future:
        try:
            return self._executor(kind).submit(_timed, fn)
        except RuntimeError:
            # Executors refuse new work while the interpreter shuts down: run inline.
            future: Future = Future()
            try:
                future.set_result(_timed(fn))
            except Exception as exc:
                future.set_exception(exc)
            return future

    def _inference_failed(self, holder: _ModelHolder, exc: Exception) -> None:
        holder.failures += 1
        holder.infer_error = f"{holder.label} tracking error: {exc}"
        self._errors.report(holder.kind, f"{holder.label} inference failed", exc)
        if holder.failures >= MAX_CONSECUTIVE_FAILURES:
            log.error("%s inference failed %d times in a row; recreating the landmarker",
                      holder.label, holder.failures)
            self._close_holder(holder)   # reopened before the next frame
            holder.failures = 0

    def _executor(self, kind: str) -> ThreadPoolExecutor:
        # One dedicated worker per model: each landmarker always runs on "its" thread.
        executor = self._executors.get(kind)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tracker-{kind}")
            self._executors[kind] = executor
        return executor

    # ---------------------------------------------------------- model lifecycle

    def _ensure_models_locked(self, settings: TrackingSettings) -> None:
        """Create / recreate / close landmarkers so they match ``settings``."""
        config = ModelConfig.from_settings(settings)
        now = time.perf_counter()
        self._models_initialised = True
        self._ensure_one(self._hands, config, now, wanted=True)
        self._ensure_one(self._pose, config, now, wanted=config.enable_pose)
        with self._lock:
            self._pose_model_name = config.pose_model if self._pose.model is not None else ""
            self._error_text = self._compose_error_locked()

    def _ensure_one(self, holder: _ModelHolder, config: ModelConfig, now: float,
                    wanted: bool) -> None:
        key = config.key(holder.kind)
        if holder.key != key:
            if holder.model is not None:
                log.info("%s tracking settings changed; recreating the landmarker", holder.label)
            self._close_holder(holder)
            holder.clear_errors()
            holder.key = key
        if not wanted or holder.model is not None or holder.load_error is not None:
            return
        if holder.missing_file is not None and now - holder.last_attempt < MISSING_MODEL_RETRY:
            return
        holder.last_attempt = now
        file_name = (HAND_MODEL_FILE if holder.kind == "hands"
                     else pose_model_file(config.pose_model))
        started = time.perf_counter()
        try:
            if holder.kind == "hands":
                holder.model = self._create_hand_model(config)
            else:
                holder.model = self._create_pose_model(config)
        except ModelMissingError as exc:
            if holder.missing_file is None:
                log.error("Tracking model missing: %s. %s", exc.path, SETUP_HINT)
            holder.missing_file = exc.file_name
            return
        except TrackingBackendError as exc:
            holder.load_error = str(exc)
            holder.missing_file = None
            return
        except Exception as exc:
            log.exception("Could not load tracking model %s", file_name)
            # Usually a damaged / partial download: setup.bat verifies and re-fetches it.
            reason = str(exc).strip().rstrip(".") or type(exc).__name__
            holder.load_error = f"Could not load tracking model {file_name}: {reason}. {SETUP_HINT}"
            holder.missing_file = None
            return
        if holder.missing_file is not None:
            log.info("Tracking model %s found", file_name)
        holder.missing_file = None
        holder.failures = 0
        log.info("%s landmarker ready (%s, %.0f ms)", holder.label, file_name,
                 (time.perf_counter() - started) * 1000.0)

    def _close_holder(self, holder: _ModelHolder) -> None:
        model, holder.model = holder.model, None
        if model is not None:
            try:
                model.close()
            except Exception as exc:
                log.warning("Closing the %s landmarker failed: %s", holder.kind, exc)

    def _release_locked(self) -> None:
        for holder in (self._hands, self._pose):
            self._close_holder(holder)
            holder.key = None    # recreate from scratch on the next use
        executors, self._executors = self._executors, {}
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)  # idle: jobs are always awaited
        self._models_initialised = False
        with self._lock:
            self._pose_model_name = ""

    def _compose_error_locked(self) -> str | None:
        parts: list[str] = []
        holders = (self._hands, self._pose)
        missing = [h.missing_file for h in holders if h.missing_file]
        if missing:
            parts.append(f"Tracking model missing: {', '.join(missing)}. {SETUP_HINT}")
        for text in [h.load_error for h in holders] + [h.infer_error for h in holders] + [
                self._pipeline_error, self._fatal_error]:
            if text and text not in parts:
                parts.append(text)
        return " | ".join(parts) if parts else None

    # ------------------------------------------------------------------ thread

    def _run(self, source: FrameSource, stop: threading.Event) -> None:
        log.info("Tracker thread started")
        try:
            with self._proc_lock:
                self._smoother.reset()
                settings, _, changed = self._take_settings()
                if changed:
                    self._smoother.apply_settings(settings)
                self._ensure_models_locked(settings)   # load before the first frame
            self._loop(source, stop)
        except Exception as exc:  # never die silently
            log.exception("Tracker thread crashed")
            with self._lock:
                self._fatal_error = f"Tracking stopped after an internal error: {exc}"
                self._error_text = self._compose_error_locked()
        finally:
            if stop.is_set():
                self._release_if_detached()
            log.info("Tracker thread stopped")

    def _release_if_detached(self) -> None:
        """Free models and workers when a stopped thread exits after stop() gave up.

        stop() normally releases everything itself after a successful join (calling
        _release_locked twice is harmless). When its join timed out, nobody else would
        ever close the landmarkers, so the exiting thread does it, unless start() has
        meanwhile launched a successor thread that keeps using them.
        """
        try:
            with self._proc_lock:
                with self._lock:
                    if self._thread is not None:
                        return
                self._release_locked()
        except Exception:  # never let cleanup turn into an unlogged thread crash
            log.exception("Releasing the tracking models failed")

    def _loop(self, source: FrameSource, stop: threading.Event) -> None:
        last_id = -1
        last_key: tuple[int, float] | None = None
        last_frame_at = time.perf_counter()
        failures = 0
        while not stop.is_set():
            call_started = time.perf_counter()
            try:
                frame = source.wait_for_frame(last_id, FRAME_WAIT_TIMEOUT)
            except Exception as exc:
                self._errors.report("source", "Frame source failed", exc)
                stop.wait(0.1)
                continue
            if stop.is_set():
                break
            now = time.perf_counter()
            key: tuple[int, float] | None = None
            if frame is not None:
                try:
                    key = (int(frame.frame_id), float(frame.timestamp))
                except Exception as exc:
                    self._errors.report("frame", "Frame source returned an invalid frame", exc)
                    stop.wait(0.05)
                    continue
            if key is None or key == last_key:
                if key is not None:
                    last_id = key[0]           # the same frame again: source is paused
                elif last_id >= 0 and now - last_frame_at > SOURCE_IDLE_RESET:
                    # A restarted camera may count frame ids from 0 again; waiting for an
                    # id above the old one would stall tracking forever.
                    last_id = -1
                    last_frame_at = now
                if now - call_started < 0.002:
                    stop.wait(0.005)           # the source did not block: no busy loop
                continue
            last_id, last_key, last_frame_at = key[0], key, now
            try:
                with self._proc_lock:
                    self._process_locked(frame)
                failures = 0
            except Exception as exc:
                failures += 1
                self._errors.report("pipeline", "Tracking failed", exc)
                with self._proc_lock:
                    with self._lock:
                        self._pipeline_error = f"Tracking error: {exc}"
                        self._error_text = self._compose_error_locked()
                    if failures >= MAX_CONSECUTIVE_FAILURES:
                        log.error("Tracking failed %d times in a row; recreating the landmarkers",
                                  failures)
                        for holder in (self._hands, self._pose):
                            self._close_holder(holder)
                            holder.key = None
                        self._smoother.reset()
                if failures >= MAX_CONSECUTIVE_FAILURES:
                    failures = 0
                    stop.wait(0.1)
