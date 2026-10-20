"""Attack state machine shared by the charge-based abilities.

::

    IDLE --pose held--> PREPARING --prepare_time--> CHARGING --release gesture--> FIRING
      ^                     |                          |                           |
      |                     +--pose lost---------------+--pose lost / tracking lost |
      |                                                v                           v
      +-----------------------------------------  (fizzle)   COOLDOWN <--finish_firing()
      +------------------------cooldown_time-------------------+

Rules that keep it reliable in a normal room:

* Charging only starts after the pose was held for ``prepare_time`` (no flicker starts).
* Short tracking drop-outs (``lost_grace``) and short pose breaks (``release_grace``)
  do not cancel a charge; charge is frozen while the pose is missing.
* A release gesture fires only after ``min_charge_time`` of charging. Earlier
  releases are ignored and reported through :attr:`AttackStateMachine.fire_rejected`.
* After firing, a cooldown runs, and (``require_pose_reset``) the charging pose must
  be released once before a new charge can begin, so one gesture = one attack.

The machine is purely time-driven (no wall clock), which makes it easy to test.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from dragon_ball_powers.config import GestureSettings


class AttackPhase(str, enum.Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    CHARGING = "charging"
    FIRING = "firing"
    COOLDOWN = "cooldown"


@dataclass
class StateMachineConfig:
    prepare_time: float = 0.2
    min_charge_time: float = 0.6
    full_charge_time: float = 3.0
    cooldown_time: float = 0.8
    lost_grace: float = 0.6
    release_grace: float = 0.4
    preparing_grace: float = 0.15
    require_pose_reset: bool = True
    max_fire_time: float = 15.0
    max_dt: float = 0.1  # charge never jumps by more than this many seconds per update

    @classmethod
    def from_settings(cls, gestures: GestureSettings) -> "StateMachineConfig":
        return cls(
            prepare_time=gestures.prepare_time,
            min_charge_time=gestures.min_charge_time,
            full_charge_time=gestures.full_charge_time,
            cooldown_time=gestures.cooldown_time,
            lost_grace=gestures.tracking_lost_grace,
            release_grace=gestures.release_grace,
            require_pose_reset=gestures.require_pose_reset,
            max_fire_time=max(5.0, gestures.beam_max_duration * 2.0),
        )


@dataclass(frozen=True)
class Transition:
    previous: AttackPhase
    current: AttackPhase
    time: float
    reason: str


class AttackStateMachine:
    """See module docstring. All times are seconds from a monotonic clock."""

    def __init__(self, config: StateMachineConfig | None = None) -> None:
        self.config = config or StateMachineConfig()
        self.phase = AttackPhase.IDLE
        self.phase_started = 0.0
        self.charge = 0.0          # 0..1
        self.charge_time = 0.0     # seconds actually spent charging (pose present)
        self.fire_charge = 0.0     # charge level when the attack was released
        self.manual = False        # current sequence was started by the keyboard
        self.fire_rejected = False  # a release gesture arrived too early this update
        self._needs_pose_reset = False
        self._pose_absent_since: float | None = None
        self._last_now = 0.0

    # ------------------------------------------------------------------ queries

    def phase_time(self, now: float) -> float:
        return max(0.0, now - self.phase_started)

    @property
    def can_fire(self) -> bool:
        return self.phase is AttackPhase.CHARGING and self.charge_time >= self.config.min_charge_time

    @property
    def is_active(self) -> bool:
        """True while energy exists (preparing, charging or firing)."""
        return self.phase in (AttackPhase.PREPARING, AttackPhase.CHARGING, AttackPhase.FIRING)

    @property
    def waiting_for_pose_reset(self) -> bool:
        return self._needs_pose_reset

    def cooldown_fraction(self, now: float) -> float:
        """1.0 right after firing, falling to 0.0 when the next attack is allowed."""
        if self.phase is not AttackPhase.COOLDOWN or self.config.cooldown_time <= 0:
            return 0.0
        return max(0.0, 1.0 - self.phase_time(now) / self.config.cooldown_time)

    # ------------------------------------------------------------------ control

    def configure(self, config: StateMachineConfig) -> None:
        self.config = config

    def reset(self, now: float = 0.0) -> None:
        self.phase = AttackPhase.IDLE
        self.phase_started = now
        self.charge = 0.0
        self.charge_time = 0.0
        self.fire_charge = 0.0
        self.manual = False
        self.fire_rejected = False
        self._needs_pose_reset = False
        self._pose_absent_since = None

    def cancel(self, now: float, reason: str = "cancel") -> list[Transition]:
        """Abort a preparing/charging sequence (the energy fizzles out)."""
        if self.phase in (AttackPhase.PREPARING, AttackPhase.CHARGING):
            return [self._to_idle(now, reason)]
        return []

    def finish_firing(self, now: float, reason: str = "fired") -> list[Transition]:
        """Called by the ability when its attack animation has ended."""
        if self.phase is AttackPhase.FIRING:
            return [self._go(AttackPhase.COOLDOWN, now, reason)]
        return []

    def force_fire(self, now: float, reason: str = "manual") -> list[Transition]:
        """Fire immediately if allowed (used by keyboard release)."""
        if self.can_fire:
            return [self._fire(now, reason)]
        if self.phase is AttackPhase.CHARGING:
            self.fire_rejected = True
        return []

    def update(self, now: float, dt: float, *, pose_active: bool, tracking_ok: bool = True,
               fire: bool = False, charge_rate: float = 1.0, hold: bool = False,
               manual: bool = False) -> list[Transition]:
        """Advance the machine by one tick.

        Args:
            now: current time (s).
            dt: time since the previous update (s).
            pose_active: the charging pose is held (or the manual trigger is held).
            tracking_ok: enough tracking data exists; when False, pose loss uses
                ``lost_grace`` instead of ``release_grace``.
            fire: a release gesture (thrust) was detected during this tick.
            charge_rate: multiplier for the charge speed (e.g. stability bonus).
            hold: a release gesture is in progress - do not fizzle the charge yet.
            manual: the input comes from the keyboard trigger (skips preparation and
                the pose-reset requirement).
        """
        self.fire_rejected = False
        self._last_now = now
        dt = min(max(0.0, dt), self.config.max_dt)
        transitions: list[Transition] = []

        if not pose_active and not manual:
            self._needs_pose_reset = False  # the pose was released once: a new charge may start

        for _ in range(4):  # allow cascades such as IDLE -> PREPARING -> CHARGING
            t = self._step(now, dt, pose_active, tracking_ok, fire, charge_rate, hold, manual)
            if t is None:
                break
            transitions.append(t)
            dt = 0.0           # time is only consumed once per update
            fire = fire and t.current is not AttackPhase.FIRING
        return transitions

    # ---------------------------------------------------------------- internals

    def _step(self, now: float, dt: float, pose_active: bool, tracking_ok: bool, fire: bool,
              charge_rate: float, hold: bool, manual: bool) -> Transition | None:
        cfg = self.config
        phase = self.phase

        if phase is AttackPhase.IDLE:
            if pose_active and (manual or not self._needs_pose_reset):
                self.manual = manual
                self.charge = 0.0
                self.charge_time = 0.0
                self._pose_absent_since = None
                return self._go(AttackPhase.PREPARING, now, "manual" if manual else "pose")
            return None

        if phase is AttackPhase.PREPARING:
            if pose_active:
                self._pose_absent_since = None
                if self.manual or self.phase_time(now) >= cfg.prepare_time:
                    return self._go(AttackPhase.CHARGING, now, "prepared")
                return None
            if self._absent_for(now) > cfg.preparing_grace:
                return self._to_idle(now, "released")
            return None

        if phase is AttackPhase.CHARGING:
            if pose_active:
                self._pose_absent_since = None
                self.charge_time += dt
                if cfg.full_charge_time > 0:
                    self.charge = min(1.0, self.charge + dt * max(0.0, charge_rate) / cfg.full_charge_time)
                else:
                    self.charge = 1.0
            if fire:
                if self.charge_time >= cfg.min_charge_time:
                    return self._fire(now, "fire")
                self.fire_rejected = True
            if not pose_active and not hold:
                grace = cfg.release_grace if tracking_ok else cfg.lost_grace
                if self._absent_for(now) > grace:
                    return self._to_idle(now, "released" if tracking_ok else "lost")
            elif not pose_active and hold:
                self._absent_for(now)  # start the timer but do not expire while holding
            return None

        if phase is AttackPhase.FIRING:
            if self.phase_time(now) > cfg.max_fire_time:
                return self._go(AttackPhase.COOLDOWN, now, "timeout")
            return None

        if phase is AttackPhase.COOLDOWN:
            if self.phase_time(now) >= cfg.cooldown_time:
                return self._go(AttackPhase.IDLE, now, "ready")
            return None

        return None

    def _absent_for(self, now: float) -> float:
        if self._pose_absent_since is None:
            self._pose_absent_since = now
        return now - self._pose_absent_since

    def _fire(self, now: float, reason: str) -> Transition:
        self.fire_charge = self.charge
        self._needs_pose_reset = self.config.require_pose_reset and not self.manual
        return self._go(AttackPhase.FIRING, now, reason)

    def _to_idle(self, now: float, reason: str) -> Transition:
        self.charge = 0.0
        self.charge_time = 0.0
        self.manual = False
        return self._go(AttackPhase.IDLE, now, reason)

    def _go(self, phase: AttackPhase, now: float, reason: str) -> Transition:
        transition = Transition(self.phase, phase, now, reason)
        self.phase = phase
        self.phase_started = now
        self._pose_absent_since = None
        if phase is AttackPhase.COOLDOWN:
            self.charge = 0.0
            self.charge_time = 0.0
        if phase is AttackPhase.IDLE:
            self.manual = False
        return transition
