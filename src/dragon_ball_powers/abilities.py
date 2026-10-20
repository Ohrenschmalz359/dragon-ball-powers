"""Abilities and the ability manager.

The ability layer turns gesture signals/events and the keyboard test controls into
attacks. Every charge ability owns an :class:`~dragon_ball_powers.state_machine.AttackStateMachine`,
steers the effect handles of the ``EffectSystem`` (``effects.py``) every tick and
triggers sounds through an :class:`~dragon_ball_powers.interfaces.AudioSink`.

* **Ki Charge** (key 1) and **Kamehameha** (key 2) share :class:`ChargeAbility`: holding
  both hands together in front of the chest (or holding Space) grows a ki orb between
  the hands; pushing both arms forward (``THRUST`` event, or releasing Space) fires a
  ki blast / a sustained Kamehameha beam, sideways or towards the camera.
* **Energy Burst** (key 3): throwing the arms apart (``BURST`` event, works for every
  selection) or Space (when selected) releases an explosion. A charging orb of the
  selected charge ability is consumed by it and makes it stronger.
* **Power Aura** (key 4): an independent toggle; it also flares up automatically while
  charging or firing (``settings.abilities.aura_auto``).
* **Power Level**: always on, a scouter-like number driven by charge, movement and aura.

Only the *selected* charge ability reacts to the charge gesture. Switching abilities
dissipates a running charge; an attack that is already running (a beam) finishes.

Tick order in :meth:`AbilityManager.update`: charge abilities -> Energy Burst (may consume
the charging orb) -> Power Aura (reads the charge state) -> Power Level.

Effect handles (orb, beams, aura) are always dropped right after ``release()`` /
``dissipate()`` / ``end()`` so finished effects are never kept alive by this module.
"""

from __future__ import annotations

import abc
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np

from dragon_ball_powers.config import ABILITY_IDS, Settings
from dragon_ball_powers.datatypes import (
    AbilityHUD,
    BodyState,
    GestureEvent,
    GestureEventType,
    GestureSignals,
    ManualInput,
    PoseObservation,
    TrackingFrame,
)
from dragon_ball_powers.interfaces import SOUND_NAMES, AudioSink, NullAudio
from dragon_ball_powers.mathutil import Vec2, clamp, damp, ease_out_cubic, lerp, smoothstep
from dragon_ball_powers.state_machine import (
    AttackPhase,
    AttackStateMachine,
    StateMachineConfig,
    Transition,
)

if TYPE_CHECKING:  # effects.py is only needed for type hints
    from dragon_ball_powers.effects import (
        AuraEffect,
        BeamEffect,
        EffectSystem,
        FrontalBeamEffect,
        OrbEffect,
    )

log = logging.getLogger(__name__)

__all__ = [
    "ABILITY_TYPES",
    "Ability",
    "AbilityContext",
    "AbilityHost",
    "AbilityManager",
    "AbilityTick",
    "CalloutBoard",
    "ChargeAbility",
    "ChargeStatus",
    "EnergyBurstAbility",
    "FireAim",
    "KamehamehaAbility",
    "KiChargeAbility",
    "MessageBoard",
    "PowerAuraAbility",
    "PowerLevel",
]

# --------------------------------------------------------------------- constants

DEFAULT_SCENE_SIZE: tuple[int, int] = (1280, 720)

#: Charge levels that play a "power_up" sound (each once per charge).
CHARGE_MILESTONES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
#: A release is aimed at the camera from this frontal share on.
FRONTAL_THRESHOLD = 0.55
#: A frontal beam is also sustained while ``signals.aim_frontal`` stays above this.
FRONTAL_SUSTAIN = 0.45
#: ``signals.thrust`` above this means a push is in progress: do not fizzle the charge.
THRUST_HOLD_THRESHOLD = 0.2
#: Seconds the arms may be lowered before a sustained (tracked) beam stops.
BEAM_LOWER_GRACE = 0.35
#: How fast (1/s) a tracked lateral beam turns towards where the arms point.
BEAM_AIM_RATE = 5.0
#: Duration of the Ki Charge firing phase and the delay of its impact sound.
KI_BLAST_FIRE_TIME = 0.4
KI_BLAST_IMPACT_DELAY = 0.35
BURST_COOLDOWN = 1.0
BURST_BASE_POWER = 0.6
#: The automatic aura starts above this charge.
AURA_AUTO_CHARGE = 0.35
#: The segmentation mask / pose of the newest tracking frame stay usable this long.
FRAME_DATA_MAX_AGE = 1.0
#: A callout disappears this long after its last refresh, fading over the last part.
CALLOUT_DURATION = 1.2
CALLOUT_FADE = 0.4
#: Short feedback messages ("Charge longer!") stay this long.
MESSAGE_DURATION = 1.5
#: Show the "step into view" hint after this long without tracking.
NO_TRACKING_HINT_DELAY = 2.0

MSG_CHARGE_LONGER = "Charge longer!"
HINT_NO_TRACKING = "Step into view and raise both hands"
HINT_CHARGE = "Bring both hands together in front of your chest to charge"
HINT_BURST = "Throw your arms apart to release an Energy Burst"
HINT_BURST_KEY = "Press SPACE to release an Energy Burst"
HINT_FIRE = "Push both arms forward to fire!"
HINT_FIRE_KEY = "Release SPACE to fire!"
HINT_KEEP_CHARGING = "Keep charging..."
HINT_POSE_RESET = "Lower your hands to charge again"

KAMEHAMEHA_CHANT: tuple[str, ...] = ("KA...", "KA... ME...", "KA... ME... HA...",
                                     "KA... ME... HA... ME...")
CALLOUT_BEAM = "HAAAAA!!!"
CALLOUT_BLAST = "HA!"
CALLOUT_BURST = "HAAAAA!!!"
CALLOUT_OVER_9000 = "IT'S OVER 9000!"

# Callout priorities (see CalloutBoard): a locked callout only yields to a more important
# one, or to another one-shot of the same priority. The shout at the moment of a release
# ("HA!", "HAAAAA!!!") outranks "IT'S OVER 9000!": the attack floor pushes the reading past
# 9000 within a frame or two, and the scouter callout then waits for the shout's lock.
PRIORITY_CHANT = 0
PRIORITY_ATTACK = 1      # per-tick refreshes while an attack runs
PRIORITY_POWER = 2
PRIORITY_RELEASE = 3     # one-shot shout when an attack is released
#: How long the release shout stays locked on screen.
RELEASE_SHOUT_HOLD = 0.5

_CHARGE_PHASES = (AttackPhase.PREPARING, AttackPhase.CHARGING)

# ----------------------------------------------------------------------- helpers


def _finite(value: Any, default: float = 0.0) -> float:
    """``float(value)`` when it is a finite number, else ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _vec(value: Any) -> Vec2 | None:
    """``value`` as an ``(x, y)`` float tuple, or None when missing, malformed or not finite."""
    if value is None:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if math.isfinite(x) and math.isfinite(y):
        return (x, y)
    return None


def _unit(value: Any) -> Vec2 | None:
    """Normalised ``value``, or None for missing / zero-length / non-finite vectors."""
    v = _vec(value)
    if v is None:
        return None
    length = math.hypot(v[0], v[1])
    if length < 1e-6:
        return None
    return (v[0] / length, v[1] / length)


def _rotate_towards(current: Vec2, target: Vec2, rate: float, dt: float) -> Vec2:
    """Turn the unit vector ``current`` towards ``target`` (frame-rate independent).

    The *angle* is interpolated rather than the components, so even a reversal of the
    direction never passes through a zero-length vector.
    """
    angle = math.atan2(current[1], current[0])
    delta = (math.atan2(target[1], target[0]) - angle + math.pi) % math.tau - math.pi
    angle += delta * (1.0 - math.exp(-rate * max(0.0, dt)))
    return (math.cos(angle), math.sin(angle))


def _alive(handle: Any) -> bool:
    return handle is not None and bool(getattr(handle, "alive", True))


class _ErrorLog:
    """Rate-limited exception logging, so a failing subsystem cannot flood the log."""

    def __init__(self, interval: float = 5.0) -> None:
        self._interval = interval
        self._last: dict[str, float] = {}
        self._suppressed: dict[str, int] = {}

    def exception(self, key: str, message: str, *args: Any) -> None:
        """Log the exception being handled (call from an ``except`` block)."""
        t = time.perf_counter()  # only used to rate-limit log output, never for logic
        last = self._last.get(key)
        if last is not None and t - last < self._interval:
            self._suppressed[key] = self._suppressed.get(key, 0) + 1
            return
        self._last[key] = t
        suppressed = self._suppressed.pop(key, 0)
        if suppressed:
            message += " (%d similar errors suppressed)"
            args = (*args, suppressed)
        log.exception(message, *args)


class _SafeAudio:
    """AudioSink adapter: clamps volumes and keeps audio errors away from the main loop."""

    def __init__(self, sink: AudioSink | None) -> None:
        self.sink: AudioSink = sink if sink is not None else NullAudio()
        self._errors = _ErrorLog(10.0)
        self._unknown: set[str] = set()

    def play(self, name: str, volume: float = 1.0) -> None:
        self._call("play", name, clamp(_finite(volume)))

    def loop(self, name: str, volume: float) -> None:
        self._call("loop", name, clamp(_finite(volume)))

    def stop(self, name: str, fade_ms: int = 250) -> None:
        self._call("stop", name, max(0, int(_finite(fade_ms, 250.0))))

    def _call(self, method: str, name: str, arg: float | int) -> None:
        if name not in SOUND_NAMES and name not in self._unknown:
            self._unknown.add(name)
            log.warning("Unknown sound name %r", name)
        try:
            getattr(self.sink, method)(name, arg)
        except Exception:  # noqa: BLE001 - audio problems must never stop the app
            self._errors.exception(f"{method}:{name}", "Audio %s(%r) failed", method, name)


# ------------------------------------------------------------- callouts / messages


@dataclass
class _Callout:
    text: str
    intensity: float
    priority: int
    refreshed: float
    duration: float
    lock_until: float


class CalloutBoard:
    """The single big anime callout slot ("KA... ME...", "HA!", "IT'S OVER 9000!").

    A callout stays visible for ``duration`` seconds after its last refresh and fades out
    during the final :data:`CALLOUT_FADE` seconds. While a callout is *locked* (``hold``
    seconds after it was shown) a different text only replaces it when it is more
    important, or when it is another one-shot (``hold > 0``) of the same priority;
    per-tick refreshes (chant, beam roar) wait until the lock has expired.
    """

    def __init__(self) -> None:
        self._current: _Callout | None = None

    def show(self, text: str, now: float, intensity: float = 1.0, *, priority: int = 0,
             duration: float = CALLOUT_DURATION, hold: float = 0.0) -> bool:
        """Show or refresh ``text``. Returns False when a locked, more important callout wins."""
        cur = self._current
        visible = cur is not None and now - cur.refreshed <= cur.duration
        if cur is not None and visible:
            if cur.text != text and now < cur.lock_until and (
                    priority < cur.priority or (priority == cur.priority and hold <= 0.0)):
                return False
            if cur.text == text:
                cur.intensity = clamp(_finite(intensity))
                cur.refreshed = now
                cur.duration = max(0.05, duration)
                cur.priority = max(cur.priority, priority) if now < cur.lock_until else priority
                cur.lock_until = max(cur.lock_until, now + max(0.0, hold))
                return True
        self._current = _Callout(text, clamp(_finite(intensity)), priority, now,
                                 max(0.05, duration), now + max(0.0, hold))
        return True

    def current(self, now: float) -> tuple[str | None, float]:
        """The visible callout and its intensity (0..1, fading out), or ``(None, 0.0)``."""
        cur = self._current
        if cur is None:
            return None, 0.0
        age = max(0.0, now - cur.refreshed)
        if age > cur.duration:
            self._current = None
            return None, 0.0
        fade = clamp((cur.duration - age) / CALLOUT_FADE)
        return cur.text, clamp(cur.intensity * fade)

    def clear(self) -> None:
        self._current = None


class MessageBoard:
    """Short-lived feedback messages (e.g. "Charge longer!") that override the hints."""

    def __init__(self) -> None:
        self._text: str | None = None
        self._until = -math.inf

    def show(self, text: str, now: float, duration: float = MESSAGE_DURATION) -> None:
        self._text = text
        self._until = now + duration

    def current(self, now: float) -> str | None:
        return self._text if self._text is not None and now < self._until else None

    def clear(self) -> None:
        self._text = None
        self._until = -math.inf


# --------------------------------------------------------------- shared structures


@dataclass(frozen=True)
class ChargeStatus:
    """Aggregated state of the charge abilities (read by the aura and the power level)."""

    charging: bool = False      # a charge ability is preparing or charging
    charge: float = 0.0         # the highest of those charges, 0..1
    firing: bool = False        # a charge ability is firing its attack
    fire_charge: float = 0.0    # charge level the running attack was released with


class AbilityHost(Protocol):
    """What abilities may ask the manager (implemented by :class:`AbilityManager`)."""

    def charge_status(self) -> ChargeStatus:
        """State of all charge abilities after their update in this tick."""

    def consume_charge(self, now: float) -> float:
        """Take the charging orb of the selected charge ability; returns its charge."""

    def tracking_lost_for(self, now: float) -> float:
        """Seconds without usable tracking (0.0 while tracking is fine)."""

    @property
    def power_fraction(self) -> float:
        """Power level as 0..1 for meters."""


@dataclass
class AbilityContext:
    """Services shared by all abilities of one :class:`AbilityManager`."""

    settings: Settings
    effects: EffectSystem
    audio: AudioSink
    callouts: CalloutBoard
    messages: MessageBoard
    host: AbilityHost


@dataclass
class AbilityTick:
    """Inputs of one update; built once per tick by the manager and shared by all abilities."""

    now: float
    dt: float
    frame: TrackingFrame | None
    body: BodyState | None
    signals: GestureSignals
    events: list[GestureEvent]
    manual: ManualInput
    selected_id: str
    scene_size: tuple[int, int] = DEFAULT_SCENE_SIZE
    mask: np.ndarray | None = None          # newest person mask (may be from an earlier frame)
    mask_id: int = 0
    pose: PoseObservation | None = None     # newest pose (may be from an earlier frame)

    def is_selected(self, ability_id: str) -> bool:
        return self.selected_id == ability_id

    def strongest(self, kind: GestureEventType) -> GestureEvent | None:
        """The most decisive event of ``kind`` in this tick, or None."""
        best: GestureEvent | None = None
        for event in self.events:
            if event.type == kind and (best is None
                                       or _finite(event.strength) > _finite(best.strength)):
                best = event
        return best

    @property
    def body_scale(self) -> float:
        """S in px: the tracked body scale, else 22 % of the scene height."""
        if self.body is not None:
            s = _finite(self.body.body_scale)
            if s > 1.0:
                return s
        return 0.22 * self.scene_size[1]

    @property
    def scene_center(self) -> Vec2:
        return (0.5 * self.scene_size[0], 0.5 * self.scene_size[1])

    def hand_mid(self) -> Vec2 | None:
        """Midpoint between the hands when both hands are present."""
        body = self.body
        if body is None or not body.both_hands:
            return None
        return _vec(body.hand_mid)

    def hand_anchor(self, side: str) -> Vec2 | None:
        """Palm centre of the hand on screen ``side`` ("left"/"right"), if present."""
        body = self.body
        if body is None:
            return None
        arm = body.arm(side)
        return _vec(arm.hand_center) if arm.present else None


@dataclass(frozen=True)
class FireAim:
    """Where a released attack goes."""

    frontal: bool               # towards the camera
    frontal_share: float        # 0..1, how much of the push was aimed at the camera
    direction: Vec2             # unit screen direction of the lateral part


# -------------------------------------------------------------------- base class


class Ability(abc.ABC):
    """Base class of all abilities. Subclasses set ``id``, ``label`` and ``hotkey``."""

    id: ClassVar[str] = ""
    label: ClassVar[str] = ""
    hotkey: ClassVar[str] = ""
    #: Can be chosen with :meth:`AbilityManager.select` (the aura is a toggle instead).
    selectable: ClassVar[bool] = True
    #: Abilities update in ascending order (producers of state before its consumers).
    update_order: ClassVar[int] = 0

    def __init__(self, ctx: AbilityContext) -> None:
        self.ctx = ctx

    @abc.abstractmethod
    def update(self, tick: AbilityTick) -> None:
        """Advance the ability by one tick."""

    def on_selected(self, now: float) -> None:
        """The user selected this ability."""

    def on_deselected(self, now: float) -> None:
        """The user selected another ability."""

    @abc.abstractmethod
    def hud(self, now: float, out: AbilityHUD) -> None:
        """Fill the HUD fields this ability is responsible for."""

    def apply_settings(self, settings: Settings) -> None:
        """Settings changed (``self.ctx.settings`` already refers to ``settings``)."""

    def reset(self, now: float) -> None:
        """Return to the initial state, ending all effects and sounds."""

    def held_effects(self) -> list[Any]:
        """Effect handles this ability still references (diagnostics; empty when idle)."""
        return []


# --------------------------------------------------------------- charge abilities


class ChargeAbility(Ability):
    """Shared charge logic of Ki Charge and Kamehameha.

    Per tick (selected): the charge pose or the held Space key drives the state machine,
    a ``THRUST`` event or releasing Space fires. The orb is created when preparing starts,
    follows the hands while charging and is consumed (``release``) by the attack or
    fizzles (``dissipate``) when the charge is lost. Subclasses implement the attack in
    :meth:`_launch` / :meth:`_firing_tick` / :meth:`_end_attack`.
    """

    update_order: ClassVar[int] = 0

    def __init__(self, ctx: AbilityContext) -> None:
        super().__init__(ctx)
        self.sm = AttackStateMachine(StateMachineConfig.from_settings(ctx.settings.gestures))
        self._orb: OrbEffect | None = None
        self._orb_center: Vec2 | None = None   # last target centre of the orb
        self._hand_center: Vec2 | None = None  # last hand midpoint of this charge
        self._orb_radius = 0.0
        self._pose_seen = False                # charge pose flag of the latest selected tick
        self._pose_blocked = False             # a burst consumed the charge of the held pose
        self._milestones = 0                   # charge milestones reached in this charge
        self._charge_loop = False              # "charge_loop" sound is playing
        self._launched_this_tick = False
        self._fire_time = 0.0
        self._fire_charge = 0.0
        self._fire_tracked = False             # attack released with tracking (not keyboard)
        self._scheduled: list[tuple[float, str, float]] = []   # (time, sound, volume)

    # ----------------------------------------------------------------- queries

    @property
    def phase(self) -> AttackPhase:
        return self.sm.phase

    @property
    def charge(self) -> float:
        """Current charge 0..1 (0 unless preparing / charging)."""
        return clamp(self.sm.charge) if self.sm.phase in _CHARGE_PHASES else 0.0

    @property
    def fire_charge(self) -> float:
        """Charge the current / last attack was released with."""
        return self._fire_charge

    def firing_progress(self, now: float) -> float:
        """0..1 progress of the running attack (for the draining HUD meter)."""
        return 0.0

    def held_effects(self) -> list[Any]:
        return [self._orb] if self._orb is not None else []

    # ------------------------------------------------------------------ update

    def update(self, tick: AbilityTick) -> None:
        now = tick.now
        self._launched_this_tick = False
        self._run_scheduled(now)
        if tick.is_selected(self.id):
            self._update_selected(tick)
        else:
            # Not selected: never charge, but let a running attack and its cooldown finish.
            if self.sm.phase in _CHARGE_PHASES:
                self._cancel_charge(now, "deselected", fizzle_volume=0.6)
            self._apply(self.sm.update(now, tick.dt, pose_active=False), tick, None)

        phase = self.sm.phase
        if phase in _CHARGE_PHASES:
            self._steer_orb(tick)
            if phase is AttackPhase.CHARGING:
                self._charging_tick(tick)
        elif phase is AttackPhase.FIRING and not self._launched_this_tick:
            if self._firing_tick(tick):
                self._end_attack(now)
                self._apply(self.sm.finish_firing(now), tick, None)
        if self.sm.phase is not AttackPhase.CHARGING:
            self._stop_charge_loop()   # leaving CHARGING on any path stops the hum

    def _update_selected(self, tick: AbilityTick) -> None:
        now, sm, manual, signals, body = tick.now, self.sm, tick.manual, tick.signals, tick.body
        held = bool(manual.trigger_held)
        thrust = tick.strongest(GestureEventType.THRUST)
        # Releasing Space only concerns a charge that Space started and that is running at the
        # start of this tick. sm.manual stays True through FIRING / COOLDOWN, and a stale
        # release must not leak into a new sequence that cascades up within this update
        # (COOLDOWN -> IDLE -> PREPARING -> CHARGING with prepare_time 0 would fizzle at once).
        manual_release = (bool(manual.trigger_released) and sm.manual
                          and sm.phase is AttackPhase.CHARGING)
        thrust_while_preparing = thrust is not None and sm.phase is AttackPhase.PREPARING
        pose = bool(signals.charge_pose_active)
        self._pose_seen = pose
        if self._pose_blocked:
            if pose:
                pose = False    # an Energy Burst used this pose: it must be released once
            else:
                self._pose_blocked = False
        charge_rate = 0.7 + 0.6 * clamp(_finite(body.stability)) if body is not None else 1.0
        transitions = sm.update(
            now, tick.dt,
            pose_active=pose or held,
            tracking_ok=bool(signals.tracking_ok) or held,
            fire=thrust is not None or manual_release,
            charge_rate=charge_rate,
            hold=_finite(signals.thrust) > THRUST_HOLD_THRESHOLD,
            manual=held,
        )
        rejected = sm.fire_rejected
        fired = any(t.current is AttackPhase.FIRING for t in transitions)
        self._apply(transitions, tick, thrust)
        if manual_release and sm.phase is AttackPhase.CHARGING:
            # Space released before the minimum charge: the energy fizzles out.
            self._cancel_charge(now, "released early")
            self.ctx.messages.show(MSG_CHARGE_LONGER, now)
        elif thrust is not None and not fired and (rejected or thrust_while_preparing):
            # (A thrust in the tick that cascades PREPARING -> CHARGING -> FIRING, possible
            # with min_charge_time 0, did fire: no complaint then.)
            self.ctx.messages.show(MSG_CHARGE_LONGER, now)

    def _apply(self, transitions: list[Transition], tick: AbilityTick,
               thrust: GestureEvent | None) -> None:
        for t in transitions:
            if t.current is AttackPhase.PREPARING:
                self._on_prepare(tick)
            elif t.current is AttackPhase.FIRING:
                self._on_fire(tick, thrust)
            elif t.current is AttackPhase.COOLDOWN:
                self._end_attack(tick.now)   # idempotent; also covers the firing timeout
            elif t.current is AttackPhase.IDLE:
                if t.previous is AttackPhase.CHARGING:
                    self._fizzle(t.reason)
                elif t.previous is AttackPhase.PREPARING:
                    self._drop_orb()          # pose flicker: let the seed vanish quietly

    # ------------------------------------------------------------------ charging

    def _on_prepare(self, tick: AbilityTick) -> None:
        self._milestones = 0
        self._orb_center = None              # new charge: forget the previous position
        self._hand_center = tick.hand_mid()
        self._drop_orb()                     # defensive, there is no orb while idle
        center = self._target_center(tick)
        radius = max(1.0, 0.5 * self.ctx.settings.effects.orb_min_radius * tick.body_scale)
        self._orb_center, self._orb_radius = center, radius
        self.ctx.audio.play("charge_start", 0.85)
        self._orb = self.ctx.effects.create_orb(center, radius)
        log.debug("%s: preparing (manual=%s)", self.id, self.sm.manual)

    def _target_center(self, tick: AbilityTick) -> Vec2:
        """Between the hands; else where the hands were last seen in this charge (short
        drop-outs keep the orb in place); else in front of the chest (keyboard charge with
        the hands down: the orb follows the body instead of hanging where it ignited);
        else the last centre, else a spot below the scene centre."""
        mid = tick.hand_mid()
        if mid is not None:
            return mid
        if self._hand_center is not None:
            return self._hand_center
        body = tick.body
        if body is not None:
            center = _vec(body.body_center)
            if center is not None:
                return (center[0], center[1] + 0.3 * tick.body_scale)
        if self._orb_center is not None:
            return self._orb_center
        w, h = tick.scene_size
        return (0.5 * w, 0.62 * h)

    def _target_radius(self, tick: AbilityTick, charge: float) -> float:
        fx = self.ctx.settings.effects
        s = tick.body_scale
        lo_rel = fx.orb_min_radius
        hi_rel = max(fx.orb_min_radius, fx.orb_max_radius)
        radius = s * lerp(lo_rel, hi_rel, ease_out_cubic(charge))
        body = tick.body
        if body is not None and body.both_hands:
            # The orb also grows with the gap between the hands (it fills the "cup").
            radius = 0.5 * radius + 0.5 * 0.42 * max(0.0, _finite(body.hand_gap))
        return clamp(radius, lo_rel * s, hi_rel * s)

    def _steer_orb(self, tick: AbilityTick) -> None:
        charge = clamp(self.sm.charge)
        mid = tick.hand_mid()
        if mid is not None:
            self._hand_center = mid
        center = self._target_center(tick)
        radius = self._target_radius(tick, charge)
        self._orb_center, self._orb_radius = center, radius
        if not _alive(self._orb):
            # The effect vanished (EffectSystem.clear()) or its creation failed: recreate.
            self._orb = None
            self._orb = self.ctx.effects.create_orb(center, radius)
        if self._orb is not None:
            self._orb.set_target(center, radius, charge, left_hand=tick.hand_anchor("left"),
                                 right_hand=tick.hand_anchor("right"))

    def _charging_tick(self, tick: AbilityTick) -> None:
        charge = clamp(self.sm.charge)
        self.ctx.effects.set_grade(darken=0.6 * charge)
        self.ctx.audio.loop("charge_loop", 0.25 + 0.75 * charge)
        self._charge_loop = True
        while (self._milestones < len(CHARGE_MILESTONES)
               and charge >= CHARGE_MILESTONES[self._milestones] - 1e-6):
            self._milestones += 1
            self.ctx.audio.play("power_up", 0.45 + 0.4 * charge)
        self._charging_feedback(tick, charge)

    def _charging_feedback(self, tick: AbilityTick, charge: float) -> None:
        """Hook for per-tick feedback while charging (Kamehameha chant)."""

    # ------------------------------------------------------------------- firing

    def _on_fire(self, tick: AbilityTick, thrust: GestureEvent | None) -> None:
        self._stop_charge_loop()
        aim = self._resolve_aim(tick, thrust)
        center, radius = self._orb_geometry(tick)
        self._release_orb()
        self._fire_time = tick.now
        self._fire_charge = clamp(self.sm.fire_charge)
        self._fire_tracked = bool(tick.signals.tracking_ok) and not self.sm.manual
        self._launched_this_tick = True
        log.debug("%s: fire charge=%.2f frontal=%s dir=(%.2f, %.2f) manual=%s", self.id,
                  self._fire_charge, aim.frontal, aim.direction[0], aim.direction[1],
                  self.sm.manual)
        self._launch(tick, aim, center, radius)

    def _resolve_aim(self, tick: AbilityTick, thrust: GestureEvent | None) -> FireAim:
        signals = tick.signals
        key_aim = _unit(tick.manual.aim)
        arm_aim = _unit(signals.aim_direction)
        if thrust is not None:
            share = clamp(_finite(thrust.frontal))
            direction = _unit(thrust.direction) or key_aim or arm_aim or (1.0, 0.0)
            return FireAim(share >= FRONTAL_THRESHOLD, share, direction)
        # Keyboard release: arrow keys win, then where the extended arms point, else the
        # attack goes towards the camera (the most readable default without aiming).
        if key_aim is not None:
            return FireAim(False, 0.0, key_aim)
        if signals.tracking_ok and signals.arms_extended and arm_aim is not None:
            share = clamp(_finite(signals.aim_frontal))
            return FireAim(share >= FRONTAL_THRESHOLD, share, arm_aim)
        return FireAim(True, 1.0, arm_aim or (1.0, 0.0))

    def _orb_geometry(self, tick: AbilityTick) -> tuple[Vec2, float]:
        """Centre and radius of the orb as drawn (the effect may smooth its targets)."""
        center = self._orb_center if self._orb_center is not None else self._target_center(tick)
        radius = self._orb_radius if self._orb_radius > 0.0 else self._target_radius(tick, 0.0)
        orb = self._orb
        if orb is not None:
            drawn_center = _vec(getattr(orb, "center", None))
            drawn_radius = _finite(getattr(orb, "radius", None), -1.0)
            if drawn_center is not None:
                center = drawn_center
            if drawn_radius > 0.0:
                radius = drawn_radius
        return center, radius

    @abc.abstractmethod
    def _launch(self, tick: AbilityTick, aim: FireAim, center: Vec2, radius: float) -> None:
        """Start the attack (the orb has already been released)."""

    @abc.abstractmethod
    def _firing_tick(self, tick: AbilityTick) -> bool:
        """Steer the running attack; return True when it has ended."""

    def _end_attack(self, now: float) -> None:
        """End the attack's effects and sounds (idempotent)."""

    # ---------------------------------------------------------------- cancelling

    def _release_orb(self) -> None:
        orb, self._orb = self._orb, None   # drop the reference first, even if release() fails
        if _alive(orb):
            orb.release()

    def _drop_orb(self) -> None:
        orb, self._orb = self._orb, None
        if _alive(orb):
            orb.dissipate()

    def _fizzle(self, reason: str, volume: float = 1.0) -> None:
        self._stop_charge_loop()
        self._drop_orb()
        self.ctx.audio.play("fizzle", volume)
        log.debug("%s: charge fizzled (%s)", self.id, reason)

    def _cancel_charge(self, now: float, reason: str, fizzle_volume: float = 1.0) -> None:
        phase = self.sm.phase
        if phase not in _CHARGE_PHASES:
            return
        self.sm.cancel(now, reason)
        if phase is AttackPhase.CHARGING:
            self._fizzle(reason, fizzle_volume)
        else:
            self._drop_orb()
        self._stop_charge_loop()

    def consume_charge(self, now: float, reason: str = "burst") -> float:
        """Hand the charging orb over to another attack (the Energy Burst).

        Returns its charge (0..1), or 0.0 when nothing is charging. The orb is released
        (bright collapse) instead of fizzling, and the state machine returns to idle.
        """
        if self.sm.phase not in _CHARGE_PHASES:
            return 0.0
        charge = clamp(self.sm.charge)
        self._stop_charge_loop()
        self.sm.cancel(now, reason)
        self._release_orb()
        # sm.cancel() does not ask for a pose reset: if the charge pose flag is still set
        # (detector hysteresis right after the arms flew apart), a fresh orb would ignite
        # immediately after the explosion. One gesture = one attack, also for the burst.
        self._pose_blocked = self._pose_seen
        log.debug("%s: charge %.2f consumed (%s)", self.id, charge, reason)
        return charge

    def _stop_charge_loop(self) -> None:
        if self._charge_loop:
            self._charge_loop = False
            self.ctx.audio.stop("charge_loop")

    # -------------------------------------------------------------- scheduling

    def _schedule_sound(self, at: float, name: str, volume: float) -> None:
        self._scheduled.append((at, name, volume))

    def _run_scheduled(self, now: float) -> None:
        if not self._scheduled:
            return
        due = [item for item in self._scheduled if item[0] <= now]
        if due:
            self._scheduled = [item for item in self._scheduled if item[0] > now]
            for _, name, volume in due:
                self.ctx.audio.play(name, volume)

    # -------------------------------------------------------------- lifecycle

    def on_deselected(self, now: float) -> None:
        # A charge cannot be carried over to another ability; a running attack finishes.
        self._cancel_charge(now, "deselected", fizzle_volume=0.6)

    def apply_settings(self, settings: Settings) -> None:
        self.sm.configure(StateMachineConfig.from_settings(settings.gestures))

    def reset(self, now: float) -> None:
        self._end_attack(now)
        self._drop_orb()
        self._stop_charge_loop()
        self._scheduled.clear()
        self.sm.reset(now)
        self._milestones = 0
        self._orb_center = None
        self._hand_center = None
        self._orb_radius = 0.0
        self._pose_seen = False
        self._pose_blocked = False

    # -------------------------------------------------------------------- HUD

    def hud(self, now: float, out: AbilityHUD) -> None:
        sm = self.sm
        out.phase = sm.phase.value
        if sm.phase in _CHARGE_PHASES:
            out.charge = clamp(sm.charge)
        elif sm.phase is AttackPhase.FIRING:
            out.charge = clamp(self._fire_charge * (1.0 - self.firing_progress(now)))
        else:
            out.charge = 0.0
        out.charge_ready = sm.can_fire
        out.cooldown = clamp(sm.cooldown_fraction(now))
        out.message = self._hint(now)

    def _hint(self, now: float) -> str | None:
        sm = self.sm
        if sm.phase in _CHARGE_PHASES:
            if sm.can_fire:
                return HINT_FIRE_KEY if sm.manual else HINT_FIRE
            return HINT_KEEP_CHARGING
        if sm.phase is AttackPhase.FIRING:
            return None
        if sm.waiting_for_pose_reset or self._pose_blocked:
            return HINT_POSE_RESET
        if self.ctx.host.tracking_lost_for(now) > NO_TRACKING_HINT_DELAY:
            return HINT_NO_TRACKING
        if sm.phase is AttackPhase.IDLE:
            return HINT_CHARGE
        return None


class KiChargeAbility(ChargeAbility):
    """Charge a ki orb and throw it: across the screen, or right into the camera."""

    id: ClassVar[str] = "ki_charge"
    label: ClassVar[str] = "Ki Charge"
    hotkey: ClassVar[str] = "1"

    def _launch(self, tick: AbilityTick, aim: FireAim, center: Vec2, radius: float) -> None:
        fc = self._fire_charge
        now = tick.now
        self.ctx.audio.play("blast_fire", 0.7 + 0.3 * fc)
        # The blast needs a moment to reach the screen edge / the lens before it explodes.
        self._schedule_sound(now + KI_BLAST_IMPACT_DELAY, "impact", 0.6 + 0.4 * fc)
        self.ctx.callouts.show(CALLOUT_BLAST, now, 1.0, priority=PRIORITY_RELEASE,
                               hold=KI_BLAST_FIRE_TIME)
        self.ctx.effects.ki_blast(center, aim.direction, aim.frontal_share, radius, 0.5 + fc)

    def _firing_tick(self, tick: AbilityTick) -> bool:
        t = tick.now - self._fire_time
        if t >= KI_BLAST_FIRE_TIME:
            return True
        # Keep the room dark while the blast flies, releasing the grade towards the end.
        self.ctx.effects.set_grade(darken=0.6 * self._fire_charge
                                   * (1.0 - t / KI_BLAST_FIRE_TIME))
        return False

    def firing_progress(self, now: float) -> float:
        return clamp((now - self._fire_time) / KI_BLAST_FIRE_TIME)


class KamehamehaAbility(ChargeAbility):
    """KA... ME... HA... ME... HAAAAA!!! - a sustained beam steered by the arms."""

    id: ClassVar[str] = "kamehameha"
    label: ClassVar[str] = "Kamehameha"
    hotkey: ClassVar[str] = "2"

    def __init__(self, ctx: AbilityContext) -> None:
        super().__init__(ctx)
        self._beam: BeamEffect | None = None
        self._frontal_beam: FrontalBeamEffect | None = None
        self._beam_frontal = False
        self._beam_origin: Vec2 = (0.0, 0.0)
        self._beam_direction: Vec2 = (1.0, 0.0)
        self._beam_power = 1.0
        self._min_duration = 1.0
        self._max_duration = 1.0
        self._fixed_duration: float | None = None   # keyboard / untracked release
        self._lowered_since: float | None = None
        self._beam_loop = False

    def held_effects(self) -> list[Any]:
        handles = super().held_effects()
        handles += [h for h in (self._beam, self._frontal_beam) if h is not None]
        return handles

    @property
    def beam_frontal(self) -> bool:
        """The current / last beam was fired towards the camera."""
        return self._beam_frontal

    def _charging_feedback(self, tick: AbilityTick, charge: float) -> None:
        index = min(len(KAMEHAMEHA_CHANT) - 1, int(charge * len(KAMEHAMEHA_CHANT)))
        pulse = 0.5 + 0.5 * math.sin(tick.now * math.tau * 2.5)
        self.ctx.callouts.show(KAMEHAMEHA_CHANT[index], tick.now,
                               0.45 + 0.4 * charge + 0.15 * pulse, priority=PRIORITY_CHANT)

    def _launch(self, tick: AbilityTick, aim: FireAim, center: Vec2, radius: float) -> None:
        fc = self._fire_charge
        gestures = self.ctx.settings.gestures
        self._min_duration = gestures.beam_min_duration
        self._max_duration = self._min_duration + max(
            0.0, gestures.beam_max_duration - self._min_duration) * fc
        self._fixed_duration = (None if self._fire_tracked
                                else 0.5 * (self._min_duration + self._max_duration))
        self._lowered_since = None
        self._beam_power = 0.5 + fc
        self._beam_origin = center
        self._beam_direction = aim.direction
        self._beam_frontal = aim.frontal
        now = tick.now
        self.ctx.audio.play("beam_fire", 0.8 + 0.2 * fc)
        self.ctx.audio.loop("beam_loop", 0.6 + 0.4 * fc)
        self._beam_loop = True
        self.ctx.callouts.show(CALLOUT_BEAM, now, 1.0, priority=PRIORITY_RELEASE,
                               hold=RELEASE_SHOUT_HOLD)
        # The release flash, rings and shake are part of the beam effects themselves.
        if aim.frontal:
            self._frontal_beam = self.ctx.effects.create_frontal_beam(center, radius,
                                                                      self._beam_power)
        else:
            self._beam = self.ctx.effects.create_beam(center, aim.direction,
                                                      radius * (0.9 + 0.6 * fc),
                                                      self._beam_power)

    def _expected_duration(self) -> float:
        return self._fixed_duration if self._fixed_duration is not None else self._max_duration

    def firing_progress(self, now: float) -> float:
        duration = self._expected_duration()
        return clamp((now - self._fire_time) / duration) if duration > 0.0 else 1.0

    def _beam_done(self, tick: AbilityTick, t: float) -> bool:
        if self._fixed_duration is not None:
            return t >= self._fixed_duration
        if t >= self._max_duration:
            return True
        signals = tick.signals
        extended = bool(signals.arms_extended)
        if self._beam_frontal and signals.tracking_ok:
            # Arms pointing at the camera look short in 2D; their frontal aim counts too.
            extended = extended or _finite(signals.aim_frontal) >= FRONTAL_SUSTAIN
        # A tracking drop-out (fast motion blur during the push) is not "arms lowered"
        # until the average beam length has passed.
        mid_duration = 0.5 * (self._min_duration + self._max_duration)
        extended = extended or (not signals.tracking_ok and t < mid_duration)
        if extended:
            self._lowered_since = None
        elif self._lowered_since is None:
            self._lowered_since = tick.now
        lowered_for = 0.0 if self._lowered_since is None else tick.now - self._lowered_since
        return t >= self._min_duration and lowered_for > BEAM_LOWER_GRACE

    def _firing_tick(self, tick: AbilityTick) -> bool:
        now = tick.now
        t = now - self._fire_time
        handle = self._frontal_beam if self._beam_frontal else self._beam
        if handle is not None and not _alive(handle):
            log.debug("kamehameha: beam effect ended externally")
            return True
        if self._beam_done(tick, t):
            return True
        mid = tick.hand_mid()
        if mid is not None:
            self._beam_origin = mid
        duration = self._expected_duration()
        decay = lerp(1.0, 0.7, clamp(t / duration)) if duration > 0.0 else 0.7
        power = self._beam_power * decay
        signals = tick.signals
        if self._beam_frontal:
            if self._frontal_beam is not None:
                self._frontal_beam.set(self._beam_origin, power=power)
        else:
            if self._fire_tracked and signals.tracking_ok and signals.arms_extended:
                target = _unit(signals.aim_direction)
                if target is not None:
                    self._beam_direction = _rotate_towards(self._beam_direction, target,
                                                           BEAM_AIM_RATE, tick.dt)
            if self._beam is not None:
                self._beam.set(self._beam_origin, self._beam_direction, power=power)
        self.ctx.effects.set_grade(darken=0.9)
        self.ctx.audio.loop("beam_loop", (0.6 + 0.4 * self._fire_charge) * decay)
        self._beam_loop = True
        pulse = 0.5 + 0.5 * math.sin(now * math.tau * 6.0)
        self.ctx.callouts.show(CALLOUT_BEAM, now, 0.8 + 0.2 * pulse, priority=PRIORITY_ATTACK)
        return False

    def _end_attack(self, now: float) -> None:
        beam, self._beam = self._beam, None
        frontal, self._frontal_beam = self._frontal_beam, None
        self._lowered_since = None
        if self._beam_loop:
            self._beam_loop = False
            self.ctx.audio.stop("beam_loop", 400)
        if _alive(beam):
            beam.end()
        if _alive(frontal):
            frontal.end()


# ------------------------------------------------------------------ energy burst


class EnergyBurstAbility(Ability):
    """A blinding explosion of ki around the user (arms thrown apart, or Space)."""

    id: ClassVar[str] = "energy_burst"
    label: ClassVar[str] = "Energy Burst"
    hotkey: ClassVar[str] = "3"
    update_order: ClassVar[int] = 1

    def __init__(self, ctx: AbilityContext) -> None:
        super().__init__(ctx)
        self._ready_at = -math.inf
        self.last_power = 0.0

    def cooldown_remaining(self, now: float) -> float:
        return max(0.0, self._ready_at - now)

    def update(self, tick: AbilityTick) -> None:
        event = (tick.strongest(GestureEventType.BURST)
                 if self.ctx.settings.gestures.burst_enabled else None)
        pressed = tick.is_selected(self.id) and bool(tick.manual.trigger_pressed)
        if event is None and not pressed:
            return
        if tick.now < self._ready_at:
            log.debug("energy_burst: ignored, cooling down for %.2f s",
                      self._ready_at - tick.now)
            return
        self._release(tick, event)

    def _release(self, tick: AbilityTick, event: GestureEvent | None) -> None:
        now = tick.now
        self._ready_at = now + BURST_COOLDOWN
        # A charging orb of the selected charge ability explodes into the burst.
        consumed = clamp(_finite(self.ctx.host.consume_charge(now)))
        power = BURST_BASE_POWER + consumed
        self.last_power = power
        center = self._center(tick, event)
        radius = 0.5 * tick.body_scale
        self.ctx.audio.play("burst", 0.8 + 0.2 * consumed)
        self.ctx.callouts.show(CALLOUT_BURST, now, 1.0, priority=PRIORITY_RELEASE,
                               hold=RELEASE_SHOUT_HOLD)
        self.ctx.effects.burst(center, radius, power)
        log.debug("energy_burst: power=%.2f at (%.0f, %.0f)", power, center[0], center[1])

    @staticmethod
    def _center(tick: AbilityTick, event: GestureEvent | None) -> Vec2:
        position = _vec(event.position) if event is not None else None
        if position is not None:
            return position
        body = tick.body
        if body is not None:
            for candidate in (body.hand_mid, body.body_center):
                point = _vec(candidate)
                if point is not None:
                    return point
        return tick.scene_center

    def hud(self, now: float, out: AbilityHUD) -> None:
        remaining = self.cooldown_remaining(now)
        out.phase = "cooldown" if remaining > 0.0 else "idle"
        out.charge = 0.0
        out.charge_ready = False
        out.cooldown = clamp(remaining / BURST_COOLDOWN)
        if remaining > 0.0:
            out.message = None
        elif self.ctx.host.tracking_lost_for(now) > NO_TRACKING_HINT_DELAY:
            out.message = HINT_NO_TRACKING
        elif not self.ctx.settings.gestures.burst_enabled:
            out.message = HINT_BURST_KEY
        else:
            out.message = HINT_BURST

    def reset(self, now: float) -> None:
        self._ready_at = -math.inf
        self.last_power = 0.0


# --------------------------------------------------------------------- power aura


class PowerAuraAbility(Ability):
    """Flame aura around the silhouette: a toggle, plus automatic flare-ups while charging."""

    id: ClassVar[str] = "power_aura"
    label: ClassVar[str] = "Power Aura"
    hotkey: ClassVar[str] = "4"
    selectable: ClassVar[bool] = False
    update_order: ClassVar[int] = 2

    RISE_RATE: ClassVar[float] = 8.0   # 1/s, intensity smoothing (fade in)
    FALL_RATE: ClassVar[float] = 4.0   # 1/s

    def __init__(self, ctx: AbilityContext) -> None:
        super().__init__(ctx)
        self.enabled = bool(ctx.settings.abilities.aura_enabled)
        self._aura: AuraEffect | None = None
        self._intensity = 0.0
        self._active = False
        self._loop = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def intensity(self) -> float:
        """Current aura intensity (0 when inactive, up to 1.5)."""
        return self._intensity if self._active else 0.0

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    def held_effects(self) -> list[Any]:
        return [self._aura] if self._aura is not None else []

    def update(self, tick: AbilityTick) -> None:
        status = self.ctx.host.charge_status()
        auto = bool(self.ctx.settings.abilities.aura_auto) and (
            (status.charging and status.charge > AURA_AUTO_CHARGE) or status.firing)
        if not (self.enabled or auto):
            self._deactivate()
            return
        charge = status.charge if status.charging else (status.fire_charge if status.firing
                                                         else 0.0)
        if self.enabled:
            energy = clamp(_finite(tick.body.energy)) if tick.body is not None else 0.0
            target = min(1.5, 0.45 + 0.35 * energy + 0.5 * charge
                         + 0.3 * clamp(_finite(self.ctx.host.power_fraction)))
        else:
            target = 1.0 if status.firing else 0.9 * smoothstep(AURA_AUTO_CHARGE, 1.0, charge)
        if not self._active:
            self._active = True
            self._intensity = 0.0    # flare up from nothing
        rate = self.RISE_RATE if target > self._intensity else self.FALL_RATE
        self._intensity = damp(self._intensity, target, rate, tick.dt)
        if not _alive(self._aura):
            self._aura = None
            self._aura = self.ctx.effects.create_aura()
            log.debug("power_aura: on (enabled=%s, auto=%s)", self.enabled, auto)
        if self._aura is not None:
            self._aura.set(self._intensity, tick.mask, tick.mask_id, tick.pose, tick.body)
        self.ctx.audio.loop("aura_loop", min(1.0, 0.6 * self._intensity))
        self._loop = True

    def _deactivate(self) -> None:
        aura, self._aura = self._aura, None
        was_active = self._active
        self._active = False
        self._intensity = 0.0
        if self._loop:
            self._loop = False
            self.ctx.audio.stop("aura_loop", 500)
        if _alive(aura):
            aura.end()
        if was_active:
            log.debug("power_aura: off")

    def hud(self, now: float, out: AbilityHUD) -> None:
        out.aura_enabled = self.enabled
        out.aura_intensity = self.intensity

    def apply_settings(self, settings: Settings) -> None:
        self.enabled = bool(settings.abilities.aura_enabled)

    def reset(self, now: float) -> None:
        self._deactivate()


# -------------------------------------------------------------------- power level


class PowerLevel:
    """Scouter-style power level, always on.

    The target grows with tracking, charge (scaled by how steady the hands are),
    movement energy and the aura. The value rises quickly towards it and decays slowly.
    A released attack *pushes* the value straight to its floor above 9000 (a ki blast
    fires for only 0.4 s, a 4/s approach would never get there); the displayed number
    counts up behind the value, and crossing 9000 upwards triggers the famous callout
    (at most every 10 s). The callout waits (up to ``OVER_9000_PATIENCE``) while the
    release shout of the attack is locked on screen, then plays with its sound.
    """

    BASE: ClassVar[float] = 300.0
    TRACKED_BONUS: ClassVar[float] = 700.0
    CHARGE_GAIN: ClassVar[float] = 8000.0
    ENERGY_GAIN: ClassVar[float] = 1500.0
    AURA_GAIN: ClassVar[float] = 1000.0
    ATTACK_FLOOR: ClassVar[float] = 9001.0
    ATTACK_CHARGE_GAIN: ClassVar[float] = 2500.0
    FULL_SCALE: ClassVar[float] = 12000.0
    RISE_RATE: ClassVar[float] = 4.0
    DECAY_RATE: ClassVar[float] = math.log(2.0) / 2.0   # half-life 2 s
    DISPLAY_RATE: ClassVar[float] = 10.0
    OVER_9000: ClassVar[float] = 9000.0
    OVER_9000_INTERVAL: ClassVar[float] = 10.0
    OVER_9000_PATIENCE: ClassVar[float] = 1.0   # s a crossing may wait for a free callout

    def __init__(self, ctx: AbilityContext | None = None) -> None:
        self.ctx = ctx
        self.value = 0.0
        self.display = 0.0
        self._last_over_9000 = -math.inf
        self._pending_9000: float | None = None   # time of a crossing not announced yet

    @classmethod
    def attack_floor(cls, fire_charge: float) -> float:
        """Minimum value while an attack released with ``fire_charge`` is running."""
        return cls.ATTACK_FLOOR + cls.ATTACK_CHARGE_GAIN * clamp(_finite(fire_charge))

    @classmethod
    def target(cls, *, tracked: bool, charge: float, stability: float, energy: float,
               aura_intensity: float, attacking: bool, fire_charge: float) -> float:
        """The level the value is pulled towards for the given inputs."""
        target = cls.BASE + (cls.TRACKED_BONUS if tracked else 0.0)
        # _finite: clamp() passes NaN through, and a NaN value would make ``level`` raise.
        target += (cls.CHARGE_GAIN * clamp(_finite(charge))
                   * (0.7 + 0.3 * clamp(_finite(stability))))
        target += (cls.ENERGY_GAIN * clamp(_finite(energy))
                   + cls.AURA_GAIN * max(0.0, _finite(aura_intensity)))
        if attacking:
            target = max(target, cls.attack_floor(fire_charge))
        return target

    def update(self, now: float, dt: float, *, tracked: bool, charge: float, stability: float,
               energy: float, aura_intensity: float, attacking: bool,
               fire_charge: float) -> None:
        dt = max(0.0, _finite(dt))
        target = self.target(tracked=tracked, charge=charge, stability=stability, energy=energy,
                             aura_intensity=aura_intensity, attacking=attacking,
                             fire_charge=fire_charge)
        rate = self.RISE_RATE if target > self.value else self.DECAY_RATE
        self.value = damp(self.value, target, rate, dt)
        if attacking:
            # "Firing pushes it to >= 9001 + 2500 * fire_charge during the attack": jump
            # there at once; ``display`` still counts up behind the value.
            self.value = max(self.value, self.attack_floor(fire_charge))
        previous = self.display
        self.display = damp(self.display, self.value, self.DISPLAY_RATE, dt)
        if (previous <= self.OVER_9000 < self.display and self._pending_9000 is None
                and now - self._last_over_9000 >= self.OVER_9000_INTERVAL):
            self._pending_9000 = now
        if self._pending_9000 is not None:
            self._announce_over_9000(now)

    def _announce_over_9000(self, now: float) -> None:
        """Show "IT'S OVER 9000!" (+ sound) unless a release shout still holds the callout."""
        if self._pending_9000 is None:
            return
        if now - self._pending_9000 > self.OVER_9000_PATIENCE:
            self._pending_9000 = None              # the moment has passed
            return
        if self.ctx is not None and not self.ctx.callouts.show(
                CALLOUT_OVER_9000, now, 1.0, priority=PRIORITY_POWER, duration=2.0, hold=1.6):
            return                                 # try again next tick
        self._pending_9000 = None
        self._last_over_9000 = now
        log.debug("power level over 9000 (%.0f)", self.display)
        if self.ctx is not None:
            self.ctx.audio.play("power_up", 1.0)

    @property
    def level(self) -> int:
        """The displayed power level number."""
        return int(round(self.display)) if math.isfinite(self.display) else 0

    @property
    def fraction(self) -> float:
        """0..1 for the power meter."""
        return clamp(_finite(self.value) / self.FULL_SCALE)

    def reset(self) -> None:
        self.value = 0.0
        self.display = 0.0
        self._last_over_9000 = -math.inf
        self._pending_9000 = None


# ------------------------------------------------------------------------ registry

#: Registry of all abilities: adding an ability = a new class + an entry here.
ABILITY_TYPES: dict[str, type[Ability]] = {
    "ki_charge": KiChargeAbility,
    "kamehameha": KamehamehaAbility,
    "energy_burst": EnergyBurstAbility,
    "power_aura": PowerAuraAbility,
}


# ------------------------------------------------------------------------- manager


class AbilityManager:
    """Owns all abilities, routes gestures / keyboard input to them and builds the HUD.

    ``select`` and ``toggle_aura`` also write ``settings.abilities.selected`` /
    ``aura_enabled`` into the settings object given to the manager, so a later
    ``apply_settings`` with that object keeps the user's choice; persisting the file
    remains the app's job.
    """

    def __init__(self, settings: Settings, effects: EffectSystem,
                 audio: AudioSink | None = None) -> None:
        self.settings = settings
        self.effects = effects
        self.audio = _SafeAudio(audio)
        self.callouts = CalloutBoard()
        self.messages = MessageBoard()
        self._ctx = AbilityContext(settings=settings, effects=effects, audio=self.audio,
                                   callouts=self.callouts, messages=self.messages, host=self)
        self._abilities: dict[str, Ability] = {}
        for ability_id, ability_type in ABILITY_TYPES.items():
            ability = ability_type(self._ctx)
            if ability.id != ability_id:
                raise ValueError(f"ABILITY_TYPES[{ability_id!r}] has id {ability.id!r}")
            self._abilities[ability_id] = ability
        self._order = sorted(self._abilities.values(), key=lambda a: a.update_order)
        self.power = PowerLevel(self._ctx)
        self._selected = self._valid_selection(settings.abilities.selected)
        self._errors = _ErrorLog()
        self._last_now = 0.0
        self._lost_since: float | None = None
        self._scene_size: tuple[int, int] = DEFAULT_SCENE_SIZE
        self._frame: TrackingFrame | None = None
        self._frame_seen_at = -math.inf

    # ----------------------------------------------------------------- queries

    @property
    def selected_id(self) -> str:
        return self._selected

    @property
    def selected(self) -> Ability:
        return self._abilities[self._selected]

    @property
    def abilities(self) -> Mapping[str, Ability]:
        """Read-only view of the ability instances by id."""
        return MappingProxyType(self._abilities)

    def ability(self, ability_id: str) -> Ability:
        return self._abilities[ability_id]

    @property
    def hotkeys(self) -> dict[str, str]:
        """Hotkey -> ability id: "1" ki_charge, "2" kamehameha, "3" energy_burst, "4" aura."""
        return {a.hotkey: a.id for a in self._abilities.values() if a.hotkey}

    @property
    def aura_enabled(self) -> bool:
        aura = self._aura()
        return aura.enabled if aura is not None else False

    @property
    def power_level(self) -> int:
        return self.power.level

    @property
    def power_fraction(self) -> float:
        return self.power.fraction

    def charge_status(self) -> ChargeStatus:
        charging = firing = False
        charge = fire_charge = 0.0
        for ability in self._abilities.values():
            if not isinstance(ability, ChargeAbility):
                continue
            phase = ability.sm.phase
            if phase in _CHARGE_PHASES:
                charging = True
                charge = max(charge, clamp(ability.sm.charge))
            elif phase is AttackPhase.FIRING:
                firing = True
                fire_charge = max(fire_charge, ability.fire_charge)
        return ChargeStatus(charging, charge, firing, fire_charge)

    def consume_charge(self, now: float) -> float:
        ability = self._abilities.get(self._selected)
        if isinstance(ability, ChargeAbility):
            return ability.consume_charge(now)
        return 0.0

    def tracking_lost_for(self, now: float) -> float:
        if self._lost_since is None:
            return 0.0
        return max(0.0, now - self._lost_since)

    def held_effects(self) -> list[Any]:
        """All effect handles currently referenced by abilities (diagnostics)."""
        return [h for a in self._abilities.values() for h in a.held_effects()]

    # ----------------------------------------------------------------- control

    def select(self, ability_id: str, now: float) -> None:
        """Select the ability for the charge gesture / Space (keys 1-3).

        Switching dissipates a running charge of the previous ability; a running attack
        finishes. Selecting ``"power_aura"`` toggles the aura instead.
        """
        ability = self._abilities.get(ability_id)
        if ability is None:
            log.warning("Unknown ability %r ignored", ability_id)
            return
        if not ability.selectable:
            if isinstance(ability, PowerAuraAbility):
                self.toggle_aura(now)
            else:
                log.warning("Ability %r cannot be selected", ability_id)
            return
        if ability_id == self._selected:
            return
        previous = self._abilities[self._selected]
        self._guarded(previous.id, previous.on_deselected, now)
        self._selected = ability_id
        self._guarded(ability.id, ability.on_selected, now)
        if ability_id in ABILITY_IDS:
            self.settings.abilities.selected = ability_id
        log.info("Ability selected: %s", ability.label)

    def toggle_aura(self, now: float) -> bool:
        """Toggle the Power Aura; returns the new state (the app persists it)."""
        aura = self._aura()
        if aura is None:
            return False
        aura.set_enabled(not aura.enabled)
        self.settings.abilities.aura_enabled = aura.enabled
        log.info("Power aura %s", "enabled" if aura.enabled else "disabled")
        return aura.enabled

    def update(self, now: float, dt: float, frame: TrackingFrame | None, body: BodyState | None,
               signals: GestureSignals, events: list[GestureEvent],
               manual: ManualInput) -> None:
        """Advance all abilities by one tick (call once per rendered frame)."""
        now = _finite(now, self._last_now)
        dt = max(0.0, _finite(dt))
        self._last_now = now
        if signals is None:
            signals = GestureSignals()
        if manual is None:
            manual = ManualInput()
        event_list = [e for e in (events or ()) if isinstance(e, GestureEvent)]
        self._track_inputs(now, frame, body, signals)
        mask, mask_id, pose = self._frame_data(now)
        tick = AbilityTick(now=now, dt=dt, frame=frame, body=body, signals=signals,
                           events=event_list, manual=manual, selected_id=self._selected,
                           scene_size=self._scene_size, mask=mask, mask_id=mask_id, pose=pose)
        for ability in self._order:
            try:
                ability.update(tick)
            except Exception:  # noqa: BLE001 - one broken ability must not stop the app
                self._errors.exception(f"update:{ability.id}", "Ability %s update failed",
                                       ability.id)
        try:
            self._update_power(tick)
        except Exception:  # noqa: BLE001
            self._errors.exception("power", "Power level update failed")

    def hud(self, now: float) -> AbilityHUD:
        """HUD state for the UI."""
        out = AbilityHUD()
        selected = self._abilities[self._selected]
        out.selected = selected.id
        out.selected_label = selected.label
        targets: list[Ability] = [selected]
        aura = self._aura()
        if aura is not None:
            targets.append(aura)
        for ability in targets:
            try:
                ability.hud(now, out)
            except Exception:  # noqa: BLE001
                self._errors.exception(f"hud:{ability.id}", "Ability %s HUD failed", ability.id)
        out.power_level = self.power.level
        out.power_fraction = self.power.fraction
        if self.settings.effects.show_callouts:
            out.callout, out.callout_intensity = self.callouts.current(now)
        else:
            out.callout, out.callout_intensity = None, 0.0
        message = self.messages.current(now)
        if message is not None:
            out.message = message
        return out

    def apply_settings(self, settings: Settings) -> None:
        """Use new settings (state machine timings, burst toggle, selection, aura...)."""
        self.settings = settings
        self._ctx.settings = settings
        for ability in self._abilities.values():
            self._guarded(ability.id, ability.apply_settings, settings)
        wanted = settings.abilities.selected
        if wanted != self._selected:
            candidate = self._abilities.get(wanted)
            if candidate is not None and candidate.selectable:
                self.select(wanted, self._last_now)
            else:
                log.warning("Unknown ability %r in settings ignored", wanted)

    def reset(self, now: float) -> None:
        """End every effect and sound and return all abilities to idle."""
        for ability in self._abilities.values():
            self._guarded(ability.id, ability.reset, now)
        self.callouts.clear()
        self.messages.clear()
        self.power.reset()
        self._lost_since = None
        self._frame = None
        self._frame_seen_at = -math.inf
        self._last_now = _finite(now, self._last_now)

    # --------------------------------------------------------------- internals

    def _valid_selection(self, ability_id: str) -> str:
        ability = self._abilities.get(ability_id)
        if ability is not None and ability.selectable:
            return ability_id
        fallback = "kamehameha" if "kamehameha" in self._abilities else next(
            a.id for a in self._abilities.values() if a.selectable)
        log.warning("Unknown ability %r selected in settings; using %s", ability_id, fallback)
        return fallback

    def _aura(self) -> PowerAuraAbility | None:
        for ability in self._abilities.values():
            if isinstance(ability, PowerAuraAbility):
                return ability
        return None

    def _guarded(self, key: str, fn: Any, *args: Any) -> None:
        try:
            fn(*args)
        except Exception:  # noqa: BLE001
            self._errors.exception(f"{getattr(fn, '__name__', 'call')}:{key}",
                                   "Ability %s: %s failed", key,
                                   getattr(fn, "__name__", "call"))

    def _track_inputs(self, now: float, frame: TrackingFrame | None, body: BodyState | None,
                      signals: GestureSignals) -> None:
        # This runs outside the per-ability guards: read the inputs defensively.
        size = getattr(frame, "scene_size", None) if frame is not None else (
            getattr(body, "scene_size", None) if body is not None else None)
        if size is not None:
            try:
                w, h = int(size[0]), int(size[1])
            except (TypeError, ValueError, IndexError, OverflowError):
                w = h = 0
            if w > 0 and h > 0:
                self._scene_size = (w, h)
        if frame is not None and frame is not self._frame:
            self._frame = frame
            self._frame_seen_at = now
        if signals.tracking_ok:
            self._lost_since = None
        elif self._lost_since is None:
            self._lost_since = now

    def _frame_data(self, now: float) -> tuple[np.ndarray | None, int, PoseObservation | None]:
        """Mask and pose of the newest tracking frame, unless it is too old.

        The app passes ``frame=None`` on ticks without a new tracking result; the aura
        keeps hugging the last silhouette instead of flickering between frames.
        """
        frame = self._frame
        if frame is None or now - self._frame_seen_at > FRAME_DATA_MAX_AGE:
            return None, 0, None
        return (getattr(frame, "segmentation", None),
                int(_finite(getattr(frame, "segmentation_id", 0))),
                getattr(frame, "pose", None))

    def _update_power(self, tick: AbilityTick) -> None:
        status = self.charge_status()
        body = tick.body
        tracked = bool(tick.signals.tracking_ok) or (
            body is not None and (body.quality != "none" or body.left.present
                                  or body.right.present))
        aura = self._aura()
        self.power.update(
            tick.now, tick.dt,
            tracked=tracked,
            charge=status.charge if status.charging else 0.0,
            stability=_finite(body.stability) if body is not None else 0.0,
            energy=_finite(body.energy) if body is not None else 0.0,
            aura_intensity=aura.intensity if aura is not None else 0.0,
            attacking=status.firing,
            fire_charge=status.fire_charge,
        )
