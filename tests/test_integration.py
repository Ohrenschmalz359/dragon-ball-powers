"""Integration tests across module boundaries (no webcam needed).

Synthetic tracking (tests/motion_synth.py) or real MediaPipe output on a sample photo
is pushed through the real GestureEngine -> AbilityManager -> EffectSystem chain, and
optionally through the real OpenGL renderer, checking what the user would see: the ki
orb sits between the hands, one push fires exactly one attack in the right direction,
the burst gesture explodes, the pipeline recovers after the attack.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from dragon_ball_powers.abilities import AbilityManager
from dragon_ball_powers.config import Settings
from dragon_ball_powers.datatypes import (
    BodyState,
    CameraFrame,
    GestureEventType,
    GestureSignals,
    ManualInput,
    TrackingFrame,
)
from dragon_ball_powers.effects import EffectSystem
from dragon_ball_powers.gestures import GestureEngine
from dragon_ball_powers.primitives import RenderList, SceneInput, UIDrawList, compute_view_transform, get_quality
from motion_synth import MotionSynth, SynthConfig, ticks

SAMPLES = Path("/tmp/dbp-samples")
SAVE_DIR = os.environ.get("DBP_SAVE_RENDERS")  # optional: write integration renders for inspection


class RecordingAudio:
    def __init__(self) -> None:
        self.played: list[str] = []
        self.loops: set[str] = set()

    def play(self, name: str, volume: float = 1.0) -> None:
        self.played.append(name)

    def loop(self, name: str, volume: float) -> None:
        self.loops.add(name)

    def stop(self, name: str, fade_ms: int = 250) -> None:
        self.loops.discard(name)


@dataclass
class Tick:
    now: float
    body: BodyState | None
    signals: GestureSignals
    phase: str
    charge: float
    render: RenderList
    events: list = field(default_factory=list)


class Pipeline:
    """The app's per-frame chain without window, camera thread or UI."""

    def __init__(self, ability: str = "kamehameha", scene_size: tuple[int, int] = (1280, 720),
                 settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.abilities.selected = ability
        self.engine = GestureEngine(self.settings.gestures)
        self.effects = EffectSystem(self.settings.effects, get_quality("high"), scene_size, seed=7)
        self.audio = RecordingAudio()
        self.abilities = AbilityManager(self.settings, self.effects, self.audio)
        self.manual = ManualInput()
        self.ticks: list[Tick] = []
        self._latest: TrackingFrame | None = None
        self._last_now: float | None = None

    def step(self, frame: TrackingFrame | None, now: float) -> Tick:
        if self._last_now is None:
            self.abilities.select(self.settings.abilities.selected, now)
        dt = 0.0 if self._last_now is None else now - self._last_now
        self._last_now = now
        if frame is not None:
            self._latest = frame
        body, signals, events = self.engine.update(frame, now)
        self.abilities.update(now, dt, self._latest, body, signals, events, self.manual)
        self.manual.trigger_pressed = False
        self.manual.trigger_released = False
        self.effects.update(dt, now)
        rl = RenderList()
        self.effects.collect(rl)
        hud = self.abilities.hud(now)
        tick = Tick(now, body, signals, hud.phase, hud.charge, rl, list(events))
        self.ticks.append(tick)
        return tick

    def run(self, frames: list[TrackingFrame]) -> "Pipeline":
        for frame, now in ticks(frames):
            self.step(frame, now)
        return self

    # ---------------------------------------------------------------- queries

    def phase_entries(self, phase: str) -> int:
        count, prev = 0, None
        for t in self.ticks:
            if t.phase == phase and prev != phase:
                count += 1
            prev = t.phase
        return count

    def events_of(self, kind: GestureEventType) -> list:
        return [e for t in self.ticks for e in t.events if e.type is kind]


def _kamehameha_synth(seed: int = 3, frontal: bool = True) -> MotionSynth:
    synth = MotionSynth(SynthConfig(distance=1.7, seed=seed))
    synth.idle(1.0)
    synth.charge_pose(hold_s=1.8)
    if frontal:
        synth.frontal_thrust(duration_s=0.3, hold_s=1.6)
    else:
        synth.lateral_thrust(direction=(1.0, 0.0), duration_s=0.3, hold_s=1.6)
    synth.idle(2.5)
    return synth


# ------------------------------------------------------------------- scenarios


def test_orb_stays_between_the_hands_while_charging():
    pipe = Pipeline("kamehameha").run(_kamehameha_synth().frames)
    charging = [t for t in pipe.ticks if t.phase == "charging" and t.render.orbs and t.body
                and t.body.hand_mid is not None and t.body.both_hands]
    assert len(charging) > 20, "the charge pose was not recognised"
    errors = []
    for t in charging:
        orb = max(t.render.orbs, key=lambda o: o.radius)
        mid = t.body.hand_mid
        errors.append(math.dist(orb.center, mid) / max(1.0, t.body.body_scale))
    # The orb follows the hand midpoint closely (in body scales, i.e. shoulder widths).
    assert float(np.median(errors)) < 0.08, f"median offset {np.median(errors):.3f} S"
    assert float(np.max(errors)) < 0.35, f"max offset {np.max(errors):.3f} S"
    # It grows while charging and stays within the configured limits.
    radii = [max(o.radius for o in t.render.orbs) for t in charging]
    assert radii[-1] > radii[0]
    s = Settings().effects
    for t in charging:
        r = max(o.radius for o in t.render.orbs)
        assert r <= s.orb_max_radius * t.body.body_scale * 1.35


def test_frontal_push_fires_exactly_one_frontal_kamehameha():
    pipe = Pipeline("kamehameha").run(_kamehameha_synth(frontal=True).frames)
    thrusts = pipe.events_of(GestureEventType.THRUST)
    assert len(thrusts) >= 1
    assert pipe.phase_entries("firing") == 1, "one push must fire exactly one attack"
    assert any(t.render.frontal_beams for t in pipe.ticks), "no frontal beam was drawn"
    assert pipe.audio.played.count("beam_fire") == 1
    # The attack ends and the machine returns to rest (cooldown -> idle).
    assert pipe.ticks[-1].phase in ("idle", "cooldown")
    assert not pipe.ticks[-1].render.frontal_beams


def test_lateral_push_fires_a_lateral_beam_in_the_push_direction():
    pipe = Pipeline("kamehameha").run(_kamehameha_synth(seed=5, frontal=False).frames)
    assert pipe.phase_entries("firing") == 1
    beams = [b for t in pipe.ticks for b in t.render.beams]
    assert beams, "no lateral beam was drawn"
    thrust = pipe.events_of(GestureEventType.THRUST)[0]
    direction = beams[len(beams) // 2].direction
    cos = direction[0] * thrust.direction[0] + direction[1] * thrust.direction[1]
    assert cos > 0.7, f"beam direction {direction} vs push {thrust.direction}"


def test_burst_gesture_explodes_once():
    synth = MotionSynth(SynthConfig(distance=1.7, seed=11))
    synth.idle(1.0)
    synth.charge_pose(hold_s=0.8)
    synth.burst(duration_s=0.25, hold_s=1.0)
    synth.idle(1.5)
    pipe = Pipeline("energy_burst").run(synth.frames)
    assert len(pipe.events_of(GestureEventType.BURST)) == 1
    assert pipe.audio.played.count("burst") == 1
    assert any(t.render.rings for t in pipe.ticks) and any(t.render.flashes for t in pipe.ticks)


def test_ki_blast_release():
    synth = MotionSynth(SynthConfig(distance=1.7, seed=4))
    synth.idle(1.0)
    synth.charge_pose(hold_s=1.5)
    synth.frontal_thrust(duration_s=0.3, hold_s=1.0)
    synth.idle(2.0)
    pipe = Pipeline("ki_charge").run(synth.frames)
    assert pipe.phase_entries("firing") == 1
    assert pipe.audio.played.count("blast_fire") == 1
    assert any(t.render.flashes for t in pipe.ticks)


def test_fidgeting_never_fires():
    synth = MotionSynth(SynthConfig(distance=1.7, seed=21))
    synth.idle(1.0)
    synth.random_fidget(8.0, hands_together=False)
    synth.idle(1.0)
    pipe = Pipeline("kamehameha").run(synth.frames)
    assert pipe.phase_entries("firing") == 0
    assert not pipe.events_of(GestureEventType.BURST)


def test_keyboard_charge_and_fire_without_tracking():
    pipe = Pipeline("kamehameha")
    now = 50.0
    pipe.step(None, now)
    pipe.manual.trigger_held = True
    pipe.manual.trigger_pressed = True
    for _ in range(90):  # 1.5 s at 60 Hz
        now += 1 / 60
        pipe.step(None, now)
    assert pipe.ticks[-1].phase == "charging" and pipe.ticks[-1].render.orbs
    pipe.manual.trigger_held = False
    pipe.manual.trigger_released = True
    for _ in range(30):
        now += 1 / 60
        pipe.step(None, now)
    assert pipe.phase_entries("firing") == 1
    assert any(t.render.frontal_beams or t.render.beams for t in pipe.ticks)


# --------------------------------------------------------------- with rendering


def _room_background(w: int = 1280, h: int = 720) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 40 + 30 * (yy / h) + 10 * np.sin(xx / 90.0)
    img = np.stack([base * 0.9, base, base * 1.1], axis=-1)
    img[int(h * 0.62):, :] *= 0.7  # floor
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.mark.gl
def test_rendered_orb_is_bright_between_the_hands(gl_context):
    from dragon_ball_powers.fonts import FontManager
    from dragon_ball_powers.renderer import GLRenderer

    pipe = Pipeline("kamehameha").run(_kamehameha_synth().frames)
    candidates = [t for t in pipe.ticks if t.phase == "charging" and t.charge > 0.5 and t.render.orbs
                  and t.body is not None and t.body.hand_mid is not None]
    assert candidates
    tick = candidates[len(candidates) // 2]
    renderer = GLRenderer(gl_context, (1280, 720), get_quality("high"), FontManager())
    try:
        background = _room_background()
        view = compute_view_transform(1280, 720, 1280, 720, "cover")
        renderer.render(SceneInput(background, 1, (1280, 720), tick.now), tick.render, UIDrawList(), view)
        pixels = renderer.read_pixels().astype(np.float32)
    finally:
        renderer.release()
    orb = max(tick.render.orbs, key=lambda o: o.radius)
    x, y = (int(round(v)) for v in orb.center)
    core = pixels[max(0, y - 3):y + 4, max(0, x - 3):x + 4].mean()
    corner = pixels[10:40, 10:40].mean()
    assert core > 200, f"orb core too dark ({core:.0f})"
    assert core > corner + 120
    if SAVE_DIR:
        import cv2

        Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(Path(SAVE_DIR) / "integration_orb.png"), cv2.cvtColor(pixels.astype(np.uint8),
                                                                              cv2.COLOR_RGB2BGR))


# ------------------------------------------------------------ with real tracking


@pytest.mark.mediapipe
def test_real_tracking_puts_the_orb_between_detected_hands(models_dir):
    cv2 = pytest.importorskip("cv2")
    sample = SAMPLES / "woman_hands.jpg"
    if not sample.exists() or not (models_dir / "hand_landmarker.task").exists():
        pytest.skip("sample image or models missing")
    from dragon_ball_powers.config import TrackingSettings
    from dragon_ball_powers.tracking import HandPoseTracker

    photo = cv2.imread(str(sample))
    canvas = np.zeros((720, 1280, 3), np.uint8)
    small = cv2.resize(photo, (480, 720))
    canvas[:, 400:880] = small
    mirrored = cv2.flip(canvas, 1)

    tracker = HandPoseTracker(TrackingSettings(), models_dir)
    pipe = Pipeline("kamehameha")
    pipe.manual.trigger_held = True       # keyboard charge: the orb must still sit between the real hands
    pipe.manual.trigger_pressed = True
    try:
        t0 = 200.0
        last_tick = None
        for i in range(45):
            frame = CameraFrame(i + 1, mirrored, t0 + i / 30)
            tf = tracker.process_frame(frame)
            last_tick = pipe.step(tf, t0 + i / 30 + 0.02)
    finally:
        tracker.stop()
    assert last_tick is not None and last_tick.body is not None
    body = last_tick.body
    assert body.left.present and body.right.present, "both hands should be found in the sample photo"
    assert last_tick.render.orbs
    orb = max(last_tick.render.orbs, key=lambda o: o.radius)
    mid = ((body.left.hand_center[0] + body.right.hand_center[0]) / 2,
           (body.left.hand_center[1] + body.right.hand_center[1]) / 2)
    assert math.dist(orb.center, mid) < 0.15 * body.body_scale + 8, (orb.center, mid)
