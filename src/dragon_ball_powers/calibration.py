"""Guided per-user gesture calibration (ARCHITECTURE §5.3).

:class:`CalibrationController` walks the user through three short steps and turns the
measurements into a :class:`~dragon_ball_powers.config.Calibration`:

1. **"Stand where you will play and relax your arms"** - 2.5 s of samples: body scale,
   shoulder width, torso length, hand size relative to the body scale and the hand
   jitter at rest. The step timer restarts while nobody is tracked; after 10 s without
   tracking the calibration is aborted with a message.
2. **"Bring both hands together in front of your chest (like holding a ball)"** - waits
   until both hands are present and close, then 2 s of samples -> the user's natural
   hands-together distance (median, body scales).
3. **"Now push both arms forward quickly!"** - up to 5 s; the peak
   :attr:`GestureSignals.thrust` is recorded (< 0.3 counts as "no push": ``thrust_peak = 0``).

The controller is purely time-driven (``now`` from ``time.perf_counter()``), consumes the
outputs of :class:`~dragon_ball_powers.gestures.GestureEngine` and never touches files:
the caller saves the returned calibration (``config.save_calibration``) and applies it
(``GestureEngine.set_calibration``). :meth:`CalibrationController.hud` describes what
the UI should show, including a result message for 3 s after finishing or failing.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import dataclass, field

from dragon_ball_powers.config import Calibration
from dragon_ball_powers.datatypes import BodyState, CalibrationHUD, GestureSignals

log = logging.getLogger(__name__)

TOTAL_STEPS = 3
STEP_BODY, STEP_TOGETHER, STEP_THRUST = 1, 2, 3

STEP_TITLES: dict[int, str] = {
    STEP_BODY: "Stand where you will play and relax your arms",
    STEP_TOGETHER: "Bring both hands together in front of your chest (like holding a ball)",
    STEP_THRUST: "Now push both arms forward quickly!",
}

THRUST_MIN_PEAK = 0.3        # weaker peaks mean "no push detected": thrust_peak = 0
TOGETHER_MAX_DISTANCE = 1.5  # body scales: hands closer than this count as "together"
TOGETHER_MAX_EXTENSION = 0.92  # mean arm extension: straight (hanging) arms are not "holding a ball"
TOGETHER_ABOVE_HIPS = 0.15   # body scales the hand midpoint must be above the hip line
TOGETHER_BREAK_GRACE = 0.3   # s the hands may briefly come apart without restarting step 2
SETTLED_FRACTION = 0.6       # rest jitter uses the last 60 % of step 1 (the user settles first)
MIN_RATIO_SAMPLES = 5


@dataclass
class _Samples:
    """Measurements collected during the current calibration."""

    body_scale: list[float] = field(default_factory=list)
    shoulder_width: list[float] = field(default_factory=list)
    torso_length: list[float] = field(default_factory=list)
    hand_ratio: list[float] = field(default_factory=list)       # hand-model palm size / S
    fallback_ratio: list[float] = field(default_factory=list)   # pose-fallback palm size / S
    hand_speed: list[tuple[float, float]] = field(default_factory=list)  # (time, S/s) while relaxed
    together: list[float] = field(default_factory=list)          # hand distance (S)
    body_done: float = 0.0                                        # time step 1 completed
    scene_height: int = 0
    thrust_peak: float = 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _pose_scale(body: BodyState) -> bool:
    """True when the body scale of this frame comes from the pose (not from hand sizes)."""
    source = getattr(body, "scale_source", None)
    if source is not None:
        return source in ("pose", "held")
    return body.quality in ("full", "pose")


def _shoulder_width(body: BodyState) -> float:
    width = float(getattr(body, "shoulder_width_px", 0.0) or 0.0)
    if width > 0.0:
        return width
    ls, rs = body.left.shoulder, body.right.shoulder
    if ls is not None and rs is not None:
        return math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    return 0.0


class CalibrationController:
    """Three-step guided calibration; see the module docstring.

    Typical use in the main loop::

        if calib.active:
            result = calib.update(now, body, signals)
            if result is not None:
                save_calibration(result)
                engine.set_calibration(result)
        ui_ctx.calibration = calib.hud(now)
    """

    def __init__(self, *, body_seconds: float = 2.5, together_seconds: float = 2.0,
                 thrust_seconds: float = 5.0, lost_timeout: float = 10.0,
                 together_timeout: float = 30.0, result_seconds: float = 3.0) -> None:
        self.body_seconds = body_seconds
        self.together_seconds = together_seconds
        self.thrust_seconds = thrust_seconds
        self.lost_timeout = lost_timeout
        self.together_timeout = together_timeout
        self.result_seconds = result_seconds
        self._step = 0
        self._samples = _Samples()
        self._step_started: float | None = 0.0  # start of the current step (None: step 1 waits for tracking)
        self._progress_time = 0.0      # seconds of valid samples in the current step
        self._last_update: float | None = None
        self._untracked_since: float | None = None
        self._apart_since: float | None = None
        self._peak_time: float | None = None
        self._hint = ""
        self._result_message: str | None = None
        self._result_until: float | None = None
        self._result_pending = False   # message set by cancel(): its timer starts at the next call
        self.last_result: Calibration | None = None
        self.failure: str | None = None

    # ------------------------------------------------------------------ control

    @property
    def active(self) -> bool:
        """True while a calibration is running."""
        return self._step > 0

    @property
    def step(self) -> int:
        """Current 1-based step (0 when inactive)."""
        return self._step

    def start(self, now: float) -> None:
        """Begin (or restart) a calibration at step 1."""
        self._samples = _Samples()
        self._result_message = None
        self._result_until = None
        self._result_pending = False
        self.failure = None
        self._begin_step(STEP_BODY, now)
        log.info("Calibration started")

    def cancel(self) -> None:
        """Abort a running calibration (the HUD shows "Calibration cancelled")."""
        if not self.active:
            return
        self._step = 0
        self.failure = "cancelled"
        self._result_message = "Calibration cancelled."
        self._result_until = None
        self._result_pending = True
        log.info("Calibration cancelled")

    def update(self, now: float, body: BodyState | None, signals: GestureSignals) -> Calibration | None:
        """Feed the latest gesture outputs; returns the finished calibration exactly once."""
        self._start_pending_message(now)
        if not self.active:
            return None
        dt = 0.0 if self._last_update is None else max(0.0, min(now - self._last_update, 0.25))
        self._last_update = now
        if self._step == STEP_BODY:
            self._update_body(now, body)
        elif self._step == STEP_TOGETHER:
            self._update_together(now, dt, body)
        elif self._step == STEP_THRUST:
            return self._update_thrust(now, dt, body, signals)
        return None

    def hud(self, now: float) -> CalibrationHUD:
        """What the UI should show right now."""
        self._start_pending_message(now)
        if self.active:
            return CalibrationHUD(active=True, step=self._step, total_steps=TOTAL_STEPS,
                                  title=STEP_TITLES[self._step], instruction=self._hint,
                                  progress=self._progress(now))
        message = None
        if self._result_message is not None and self._result_until is not None and now < self._result_until:
            message = self._result_message
        return CalibrationHUD(active=False, step=0, total_steps=TOTAL_STEPS, result_message=message)

    # ------------------------------------------------------------------ steps

    def _begin_step(self, step: int, now: float) -> None:
        self._step = step
        self._step_started = now
        self._progress_time = 0.0
        self._last_update = now
        self._untracked_since = None
        self._apart_since = None
        self._peak_time = None
        self._hint = {
            STEP_BODY: "Step into view so that your upper body is visible.",
            STEP_TOGETHER: "Hold both hands close together in front of your chest.",
            STEP_THRUST: "Push both palms towards the camera like a Kamehameha.",
        }[step]

    def _tracked(self, body: BodyState | None) -> bool:
        if body is None or body.quality == "none":
            return False
        return body.left.present or body.right.present or _shoulder_width(body) > 0.0

    def _check_lost(self, now: float, tracked: bool) -> bool:
        """Track how long nobody was visible; abort after ``lost_timeout``. True = aborted."""
        if tracked:
            self._untracked_since = None
            return False
        if self._untracked_since is None:
            self._untracked_since = now
        if now - self._untracked_since >= self.lost_timeout:
            self._fail(now, "no_tracking",
                       f"Calibration stopped: nobody was visible for {self.lost_timeout:.0f} seconds. "
                       "Stand in front of the camera and try again.")
            return True
        return False

    def _update_body(self, now: float, body: BodyState | None) -> None:
        tracked = self._tracked(body)
        if self._check_lost(now, tracked):
            return
        if not tracked or body is None:
            # restart the step: the samples must come from one continuous, tracked period
            self._step_started = None
            self._progress_time = 0.0
            self._samples.body_scale.clear()
            self._samples.shoulder_width.clear()
            self._samples.torso_length.clear()
            self._samples.hand_ratio.clear()
            self._samples.fallback_ratio.clear()
            self._samples.hand_speed.clear()
            self._hint = "Step into view so that your upper body is visible."
            return
        if self._step_started is None:
            self._step_started = now          # first tracked sample of this attempt
        smp = self._samples
        S = float(body.body_scale)
        pose_scale = _pose_scale(body)
        if S > 0.0:
            smp.body_scale.append(S)
        width = _shoulder_width(body)
        if width > 0.0:
            smp.shoulder_width.append(width)
        torso = float(getattr(body, "torso_length_px", 0.0) or 0.0)
        if torso > 0.0:
            smp.torso_length.append(torso)
        smp.scene_height = int(body.scene_size[1])
        for arm in (body.left, body.right):
            if not arm.present:
                continue
            if arm.source != "predicted":       # extrapolated hands say nothing about jitter
                smp.hand_speed.append((now, float(arm.speed)))
            if S > 0.0 and pose_scale and arm.hand_size > 0.0:
                (smp.hand_ratio if arm.source == "hand" else smp.fallback_ratio).append(arm.hand_size / S)
        self._progress_time = now - self._step_started
        self._hint = ("Hold still for a moment." if (body.left.present or body.right.present)
                      else "Hold still - keep your hands where the camera can see them.")
        if self._progress_time >= self.body_seconds:
            smp.body_done = now
            self._begin_step(STEP_TOGETHER, now)

    def _update_together(self, now: float, dt: float, body: BodyState | None) -> None:
        tracked = self._tracked(body)
        if self._check_lost(now, tracked):
            return
        if self._step_started is not None and now - self._step_started >= self.together_timeout:
            self._fail(now, "together_timeout",
                       "Calibration stopped: both hands were not held together. Please try again.")
            return
        close = body is not None and self._hands_together(body)
        if not close or body is None:
            if self._progress_time > 0.0:
                if self._apart_since is None:
                    self._apart_since = now
                if now - self._apart_since > TOGETHER_BREAK_GRACE:
                    self._progress_time = 0.0       # the hold was broken: start counting again
                    self._samples.together.clear()
            if body is None or not body.both_hands:
                self._hint = "Show both hands to the camera."
            else:
                self._hint = "Bring your hands closer together in front of your chest."
            return
        self._apart_since = None
        self._samples.together.append(float(body.hand_distance))
        if not self._samples.hand_ratio and _pose_scale(body) and body.body_scale > 0.0:
            # step 1 saw no hand-model hands (arms hanging): measure the hand size here
            for arm in (body.left, body.right):
                if arm.present and arm.source == "hand" and arm.hand_size > 0.0:
                    self._samples.hand_ratio.append(arm.hand_size / body.body_scale)
        self._progress_time += dt
        self._hint = "Hold it..."
        if self._progress_time >= self.together_seconds:
            self._begin_step(STEP_THRUST, now)

    @staticmethod
    def _hands_together(body: BodyState) -> bool:
        """Both hands close together in front of the upper body with bent arms (not hanging)."""
        if not (body.both_hands and body.hands_in_front and math.isfinite(body.hand_distance)
                and body.hand_distance <= TOGETHER_MAX_DISTANCE and body.hand_mid is not None):
            return False
        if body.torso_box is not None and body.body_scale > 0.0:
            hip_y = body.torso_box[3]
            if body.hand_mid[1] > hip_y - TOGETHER_ABOVE_HIPS * body.body_scale:
                return False
        exts = [a.extension_3d if a.extension_3d is not None else a.extension_2d
                for a in (body.left, body.right) if a.extension_3d is not None or a.extension_2d > 0.0]
        if exts and sum(exts) / len(exts) >= TOGETHER_MAX_EXTENSION:
            return False
        return True

    def _update_thrust(self, now: float, dt: float, body: BodyState | None,
                       signals: GestureSignals) -> Calibration | None:
        tracked = self._tracked(body)
        if self._check_lost(now, tracked):
            return None
        if tracked:
            self._progress_time += dt       # only time the user could actually push counts
        thrust = float(signals.thrust) if math.isfinite(signals.thrust) else 0.0
        smp = self._samples
        if thrust > smp.thrust_peak:
            smp.thrust_peak = thrust
            self._peak_time = now
        detected = smp.thrust_peak >= THRUST_MIN_PEAK
        self._hint = "Great!" if detected else "Push both palms towards the camera like a Kamehameha."
        push_done = (detected and self._peak_time is not None and now - self._peak_time >= 0.3
                     and thrust < 0.5 * smp.thrust_peak)
        if push_done or self._progress_time >= self.thrust_seconds:
            return self._finish(now)
        return None

    # ------------------------------------------------------------------ results

    def _finish(self, now: float) -> Calibration:
        smp = self._samples
        S = _median(smp.body_scale)
        ratios = smp.hand_ratio if len(smp.hand_ratio) >= MIN_RATIO_SAMPLES else smp.hand_ratio + smp.fallback_ratio
        peak = smp.thrust_peak if smp.thrust_peak >= THRUST_MIN_PEAK else 0.0
        cal = Calibration(
            valid=True,
            shoulder_width_px=S,
            torso_length_px=_median(smp.torso_length),
            scene_height=int(smp.scene_height),
            hand_size_ratio=_median(ratios) if len(ratios) >= MIN_RATIO_SAMPLES else 0.0,
            together_distance=_median(smp.together),
            rest_jitter=self._rest_jitter(),
            thrust_peak=peak,
            created_at=Calibration.timestamp(),
        )
        self.last_result = cal
        self.failure = None
        self._step = 0
        if peak > 0.0:
            message = "Calibration complete - your push strength was recorded."
        else:
            message = "Calibration complete - no push detected, the default push sensitivity is kept."
        self._show_result(now, message)
        log.info("Calibration complete: S=%.0f px, shoulders=%.0f px, torso=%.0f px, hand/S=%.3f, "
                 "together=%.2f S, jitter=%.2f S/s, thrust peak=%.2f", S, _median(smp.shoulder_width),
                 cal.torso_length_px, cal.hand_size_ratio, cal.together_distance, cal.rest_jitter, cal.thrust_peak)
        return cal

    def _rest_jitter(self) -> float:
        """Median hand speed over the settled part of step 1 (0 = no hands were seen)."""
        smp = self._samples
        cut = smp.body_done - SETTLED_FRACTION * self.body_seconds
        return _median([v for t, v in smp.hand_speed if t >= cut])

    def _fail(self, now: float, reason: str, message: str) -> None:
        self._step = 0
        self.failure = reason
        self._show_result(now, message)
        log.info("Calibration aborted (%s)", reason)

    def _show_result(self, now: float, message: str) -> None:
        self._result_message = message
        self._result_until = now + self.result_seconds
        self._result_pending = False

    def _start_pending_message(self, now: float) -> None:
        if self._result_pending:
            self._result_until = now + self.result_seconds
            self._result_pending = False

    def _progress(self, now: float) -> float:
        if self._step == STEP_BODY:
            total = self.body_seconds
        elif self._step == STEP_TOGETHER:
            total = self.together_seconds
        else:
            if self._samples.thrust_peak >= THRUST_MIN_PEAK:
                return 1.0
            total = self.thrust_seconds
        return max(0.0, min(1.0, self._progress_time / total)) if total > 0 else 1.0
