"""Tests for the procedural sound synthesis and the AudioEngine (dummy SDL audio driver)."""

from __future__ import annotations

import numpy as np
import pytest

from dragon_ball_powers import audio
from dragon_ball_powers.config import AudioSettings
from dragon_ball_powers.interfaces import LOOP_SOUNDS, SOUND_NAMES, AudioSink


@pytest.mark.parametrize("name", SOUND_NAMES)
def test_every_sound_synthesises(name):
    data = audio.synthesize(name)
    assert data.dtype == np.int16 and data.ndim == 2 and data.shape[1] == 2
    expected = audio.SOUND_DURATIONS[name]
    assert data.shape[0] == pytest.approx(expected * audio.SAMPLE_RATE, rel=0.02)
    peak = np.abs(data.astype(np.int32)).max()
    assert 0.8 * 32767 * audio.PEAK_LINEAR <= peak <= 32767
    stats = audio.analyze(name, data)
    assert np.isfinite(stats.rms_dbfs) and stats.rms_dbfs < -6.0  # not a square wave
    assert 300.0 < stats.centroid_hz < 12000.0


def test_synthesis_is_deterministic_and_seeded():
    audio.clear_cache()
    a = audio.render_sound("impact", seed=1)
    b = audio.render_sound("impact", seed=1)
    c = audio.render_sound("impact", seed=2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("name", sorted(LOOP_SOUNDS))
def test_loops_are_seamless(name):
    data = audio.synthesize(name)
    seam = audio.loop_seam(data)
    # The jump from the last to the first sample is like a normal sample step...
    assert seam.jump_over_step < 0.5
    # ...and the loudness does not change around the seam.
    assert abs(seam.rms_ratio_db) < 3.0


def test_unknown_sound_raises_in_synthesis():
    with pytest.raises(KeyError):
        audio.render_sound("no_such_sound")
    with pytest.raises(ValueError):
        audio.render_sound("zap", sample_rate=10)


def test_export_writes_wavs(tmp_path):
    import wave

    stats = audio.export_sounds(tmp_path, names=("zap", "charge_loop"))
    assert [s.name for s in stats] == ["zap", "charge_loop"]
    with wave.open(str(tmp_path / "zap.wav"), "rb") as wav:
        assert wav.getnchannels() == 2 and wav.getsampwidth() == 2 and wav.getframerate() == 44100


def test_adapt_channels():
    stereo = np.array([[1000, 3000], [-2000, 0]], np.int16)
    mono = audio._adapt_channels(stereo, 1)
    assert mono.shape == (2,) and mono[0] == 2000
    six = audio._adapt_channels(stereo, 6)
    assert six.shape == (2, 6) and (six[:, 2:] == 0).all()


# ------------------------------------------------------------------- engine


@pytest.fixture
def engine():
    eng = audio.AudioEngine(AudioSettings(enabled=True, volume=0.7))
    if not eng.start():
        pytest.skip("no audio device (even the dummy driver failed)")
    assert eng.wait_ready(20.0)
    yield eng
    eng.close()


def test_engine_implements_the_sink_protocol(engine):
    assert isinstance(engine, AudioSink)
    assert engine.available and engine.ready and engine.enabled


def test_engine_calls_never_raise(engine):
    for name in SOUND_NAMES:
        if name in LOOP_SOUNDS:
            engine.loop(name, 0.5)
            engine.stop(name, 50)
        else:
            engine.play(name, 0.5)
    engine.play("unknown_sound")
    engine.loop("unknown_sound", 1.0)
    engine.stop("unknown_sound")
    engine.loop("impact", 1.0)            # a one-shot used as a loop: warning, no crash
    engine.play("zap", float("nan"))
    engine.loop("charge_loop", -3.0)
    engine.stop_all()


def test_loop_is_not_restarted_every_frame(engine):
    engine.loop("charge_loop", 0.5)
    first = engine._loop_active["charge_loop"]
    for v in np.linspace(0.5, 0.9, 30):
        engine.loop("charge_loop", float(v))
    assert engine._loop_active["charge_loop"] == first
    # Changes smaller than VOLUME_EPSILON are not sent to the mixer (loop() runs every frame).
    target = 0.9 * 0.7 * audio.MIX_LEVELS["charge_loop"]
    assert abs(engine._loop_volume["charge_loop"] - target) <= audio.VOLUME_EPSILON


def test_disable_and_volume(engine):
    engine.apply_settings(AudioSettings(enabled=False, volume=0.3))
    assert not engine.enabled
    engine.play("zap")                    # silently ignored
    engine.loop("aura_loop", 1.0)
    assert "aura_loop" not in engine._loop_volume or engine._loop_stopping.get("aura_loop", True)
    engine.apply_settings(AudioSettings(enabled=True, volume=0.3))
    engine.loop("aura_loop", 1.0)
    assert engine._loop_volume["aura_loop"] == pytest.approx(0.3 * audio.MIX_LEVELS["aura_loop"])


def test_calls_before_start_and_after_close_are_noops():
    eng = audio.AudioEngine(AudioSettings())
    eng.play("zap")
    eng.loop("charge_loop", 1.0)
    eng.stop("charge_loop")
    eng.stop_all()
    eng.close()
    eng.close()
    eng.play("zap")
    assert not eng.available


def test_cli_stats(capsys):
    assert audio.main(["--stats"]) == 0
    out = capsys.readouterr().out
    assert "beam_fire" in out and "charge_loop" in out
