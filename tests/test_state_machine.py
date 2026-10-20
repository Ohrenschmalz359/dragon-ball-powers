"""Tests for the attack state machine (no camera needed)."""

from __future__ import annotations

import pytest

from dragon_ball_powers.config import GestureSettings
from dragon_ball_powers.state_machine import (
    AttackPhase,
    AttackStateMachine,
    StateMachineConfig,
)

DT = 1.0 / 30.0


class Clock:
    def __init__(self) -> None:
        self.now = 0.0


def run(sm: AttackStateMachine, clock: Clock, seconds: float, **kwargs):
    """Advance in 30 Hz ticks, returning every transition that happened."""
    transitions = []
    steps = int(round(seconds / DT))
    for _ in range(steps):
        clock.now += DT
        transitions += sm.update(clock.now, DT, **kwargs)
    return transitions


def make(**overrides) -> tuple[AttackStateMachine, Clock]:
    cfg = StateMachineConfig(prepare_time=0.2, min_charge_time=0.6, full_charge_time=2.0,
                             cooldown_time=0.8, lost_grace=0.6, release_grace=0.4)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return AttackStateMachine(cfg), Clock()


def test_full_cycle():
    sm, clock = make()
    assert sm.phase is AttackPhase.IDLE
    run(sm, clock, 0.1, pose_active=True)
    assert sm.phase is AttackPhase.PREPARING
    run(sm, clock, 0.2, pose_active=True)
    assert sm.phase is AttackPhase.CHARGING
    run(sm, clock, 1.0, pose_active=True)
    assert sm.can_fire and 0.4 < sm.charge < 0.6
    clock.now += DT
    t = sm.update(clock.now, DT, pose_active=True, fire=True)
    assert [x.current for x in t] == [AttackPhase.FIRING]
    assert sm.fire_charge == pytest.approx(sm.charge)  # charge level captured at release
    assert sm.fire_charge > 0.4
    sm.finish_firing(clock.now)
    assert sm.phase is AttackPhase.COOLDOWN and sm.charge == 0.0
    assert sm.cooldown_fraction(clock.now) == pytest.approx(1.0)
    run(sm, clock, 0.9, pose_active=False)
    assert sm.phase is AttackPhase.IDLE


def test_charge_saturates_at_one():
    sm, clock = make()
    run(sm, clock, 5.0, pose_active=True)
    assert sm.phase is AttackPhase.CHARGING
    assert sm.charge == pytest.approx(1.0)


def test_fire_before_min_charge_is_rejected():
    sm, clock = make()
    run(sm, clock, 0.3, pose_active=True)
    assert sm.phase is AttackPhase.CHARGING
    clock.now += DT
    sm.update(clock.now, DT, pose_active=True, fire=True)
    assert sm.phase is AttackPhase.CHARGING
    assert sm.fire_rejected


def test_short_pose_break_does_not_cancel():
    sm, clock = make()
    run(sm, clock, 1.0, pose_active=True)
    charge = sm.charge
    run(sm, clock, 0.3, pose_active=False)       # shorter than release_grace
    assert sm.phase is AttackPhase.CHARGING
    assert sm.charge == pytest.approx(charge)    # frozen while the pose is missing
    run(sm, clock, 0.3, pose_active=True)
    assert sm.charge > charge


def test_long_pose_break_fizzles():
    sm, clock = make()
    run(sm, clock, 1.0, pose_active=True)
    t = run(sm, clock, 0.6, pose_active=False)
    assert sm.phase is AttackPhase.IDLE
    assert t[-1].reason == "released"
    assert sm.charge == 0.0


def test_tracking_loss_uses_longer_grace():
    sm, clock = make(release_grace=0.2, lost_grace=0.8)
    run(sm, clock, 1.0, pose_active=True)
    run(sm, clock, 0.5, pose_active=False, tracking_ok=False)
    assert sm.phase is AttackPhase.CHARGING      # still within lost_grace
    t = run(sm, clock, 0.5, pose_active=False, tracking_ok=False)
    assert sm.phase is AttackPhase.IDLE and t[-1].reason == "lost"


def test_hold_prevents_fizzle_during_thrust():
    sm, clock = make(release_grace=0.2)
    run(sm, clock, 1.0, pose_active=True)
    run(sm, clock, 0.5, pose_active=False, hold=True)
    assert sm.phase is AttackPhase.CHARGING
    clock.now += DT
    sm.update(clock.now, DT, pose_active=False, fire=True, hold=True)
    assert sm.phase is AttackPhase.FIRING


def test_preparing_flicker_returns_to_idle():
    sm, clock = make(prepare_time=0.5)
    run(sm, clock, 0.1, pose_active=True)
    assert sm.phase is AttackPhase.PREPARING
    run(sm, clock, 0.3, pose_active=False)
    assert sm.phase is AttackPhase.IDLE


def test_one_gesture_one_attack_requires_pose_reset():
    sm, clock = make(cooldown_time=0.2)
    run(sm, clock, 1.0, pose_active=True)
    clock.now += DT
    sm.update(clock.now, DT, pose_active=True, fire=True)
    sm.finish_firing(clock.now)
    # The user keeps the pose the whole time: no new charge starts automatically.
    run(sm, clock, 1.0, pose_active=True)
    assert sm.phase is AttackPhase.IDLE
    assert sm.waiting_for_pose_reset
    # Releasing the pose once re-arms the machine.
    run(sm, clock, 0.1, pose_active=False)
    run(sm, clock, 0.4, pose_active=True)
    assert sm.phase is AttackPhase.CHARGING


def test_repeated_fire_signals_fire_once():
    sm, clock = make()
    run(sm, clock, 1.0, pose_active=True)
    fired = 0
    for _ in range(10):
        clock.now += DT
        fired += sum(1 for t in sm.update(clock.now, DT, pose_active=True, fire=True)
                     if t.current is AttackPhase.FIRING)
    assert fired == 1


def test_manual_trigger_skips_preparation_and_reset():
    sm, clock = make()
    clock.now += DT
    t = sm.update(clock.now, DT, pose_active=True, manual=True)
    assert [x.current for x in t] == [AttackPhase.PREPARING, AttackPhase.CHARGING]
    assert sm.manual
    run(sm, clock, 0.7, pose_active=True, manual=True)
    assert sm.force_fire(clock.now)
    assert sm.phase is AttackPhase.FIRING
    assert not sm.waiting_for_pose_reset


def test_manual_force_fire_too_early_is_rejected():
    sm, clock = make()
    run(sm, clock, 0.2, pose_active=True, manual=True)
    assert sm.force_fire(clock.now) == []
    assert sm.fire_rejected
    sm.cancel(clock.now, "released")
    assert sm.phase is AttackPhase.IDLE


def test_firing_timeout_safety_net():
    sm, clock = make(max_fire_time=1.0)
    run(sm, clock, 1.0, pose_active=True)
    clock.now += DT
    sm.update(clock.now, DT, pose_active=True, fire=True)
    t = run(sm, clock, 1.2, pose_active=False)
    assert sm.phase in (AttackPhase.COOLDOWN, AttackPhase.IDLE)
    assert any(x.reason == "timeout" for x in t)


def test_large_dt_does_not_jump_charge():
    sm, clock = make()
    run(sm, clock, 0.4, pose_active=True)
    before = sm.charge
    clock.now += 2.0
    sm.update(clock.now, 2.0, pose_active=True)
    assert sm.charge - before <= 0.1 / 2.0 + 1e-6


def test_from_settings():
    g = GestureSettings(min_charge_time=1.5, cooldown_time=2.0, prepare_time=0.3)
    cfg = StateMachineConfig.from_settings(g)
    assert cfg.min_charge_time == 1.5 and cfg.cooldown_time == 2.0 and cfg.prepare_time == 0.3
