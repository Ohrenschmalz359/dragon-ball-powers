"""Procedural sound: every effect sound is synthesised with NumPy at start-up.

No audio files are shipped. :func:`synthesize` renders each name in
:data:`~dragon_ball_powers.interfaces.SOUND_NAMES` as 16-bit stereo (44.1 kHz by
default), normalised to -1 dBFS, deterministic for a given seed. The sounds aim for a
cinematic anime-energy character: punchy low end, saturated mids, crackling electric
detail, no harsh top end.

DSP toolbox (NumPy only):

* static zero-phase filters = ``rfft`` spectra multiplied by smooth Butterworth-like
  or log-Gaussian masks (:func:`_fft_filter`);
* time-varying filters (sweeps) = short-time FFT with sqrt-Hann windows at 50 %
  overlap, which reconstructs the input exactly for a unit mask, so the mask may change
  per frame without clicks (:func:`_stft_filter`);
* band-limited saw/square oscillators (PolyBLEP), sparse impulse "crackle", smoothed
  random modulators, exponential envelopes, ``tanh`` soft saturation, >= 5 ms fades.

Loops are seamless: the random/time-varying part is rendered ~150 ms longer than the
loop and its tail is equal-power cross-faded into its head (so the last sample flows
into the first exactly like two neighbouring samples of the long render), while the
tonal part uses frequencies with a whole number of cycles per loop. Everything applied
after the fold is point-wise or circular, which keeps the seam intact.

:class:`AudioEngine` implements :class:`~dragon_ball_powers.interfaces.AudioSink` on
top of ``pygame.mixer``: sounds are synthesised in a background thread (start-up is not
blocked; calls for a sound that is not ready yet are ignored), each loop owns two
reserved channels (a restart while the previous instance still fades out cross-fades
instead of cutting), one-shots share the remaining channels. Without an audio device
every method is a silent no-op.

Command line (from the project root, with ``src`` on ``PYTHONPATH``)::

    python -m dragon_ball_powers.audio --export previews/sounds   # WAV files + level table
    python -m dragon_ball_powers.audio --stats                    # table only
    python -m dragon_ball_powers.audio --play beam_fire impact    # listen (needs a device)
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
import wave
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dragon_ball_powers.config import AudioSettings
from dragon_ball_powers.interfaces import LOOP_SOUNDS, SOUND_NAMES

log = logging.getLogger(__name__)

# ----------------------------------------------------------------------- constants

#: Default synthesis rate (the engine follows the rate the mixer actually runs at).
SAMPLE_RATE = 44100
#: Every sound is normalised to this peak level.
PEAK_DBFS = -1.0
PEAK_LINEAR = 10.0 ** (PEAK_DBFS / 20.0)
#: Loop length and the cross-fade used to make the loop seamless.
LOOP_SECONDS = 2.0
LOOP_CROSSFADE_SECONDS = 0.15
#: Shortest fade applied to both ends of every one-shot (against clicks).
MIN_FADE_MS = 5.0

#: Nominal length of every sound in seconds (loops: exact loop period).
SOUND_DURATIONS: dict[str, float] = {
    "charge_start": 0.60,
    "charge_loop": LOOP_SECONDS,
    "beam_fire": 1.40,
    "beam_loop": LOOP_SECONDS,
    "blast_fire": 0.60,
    "impact": 1.00,
    "burst": 1.40,
    "aura_loop": LOOP_SECONDS,
    "zap": 0.15,
    "power_up": 0.70,
    "fizzle": 0.50,
    "ui_click": 0.03,
}

#: Mix balance applied by the engine on top of the caller's volume and the master
#: volume. Every file peaks at -1 dBFS, so these gains set the relative loudness and
#: keep typical overlaps (beam blast + beam roar + fading charge hum) out of the
#: mixer's hard clipping.
MIX_LEVELS: dict[str, float] = {
    "charge_start": 0.62,
    "charge_loop": 0.42,
    "beam_fire": 0.80,
    "beam_loop": 0.42,
    "blast_fire": 0.66,
    "impact": 0.75,
    "burst": 0.85,
    "aura_loop": 0.36,
    "zap": 0.40,
    "power_up": 0.46,
    "fizzle": 0.50,
    "ui_click": 0.32,
}

#: Mixer set-up requested by :meth:`AudioEngine.start` (if nobody initialised it yet).
MIXER_FREQUENCY = 44100
MIXER_SIZE = -16
MIXER_CHANNELS = 2
MIXER_BUFFER = 512
NUM_CHANNELS = 24
#: Volume changes smaller than this are not sent to the mixer (``loop`` runs every frame).
VOLUME_EPSILON = 0.01
#: Loops start with a short fade-in: their first sample is mid-waveform.
LOOP_FADE_IN_MS = 40
#: ``apply_settings(enabled=False)`` fades everything out this quickly.
DISABLE_FADE_MS = 120

_TAU = 2.0 * math.pi
_CLOSE_JOIN_TIMEOUT = 2.0

# ------------------------------------------------------------------ DSP: masks


def _lowpass(f: np.ndarray, fc: float | np.ndarray, order: float = 2.0) -> np.ndarray:
    """Butterworth-style low-pass magnitude (-3 dB at ``fc``)."""
    return 1.0 / np.sqrt(1.0 + (f / fc) ** (2.0 * order))


def _highpass(f: np.ndarray, fc: float | np.ndarray, order: float = 2.0) -> np.ndarray:
    """Butterworth-style high-pass magnitude; exactly 0 at DC."""
    r = (f / fc) ** order
    return r / np.sqrt(1.0 + r * r)


def _bandpass(f: np.ndarray, lo: float | np.ndarray, hi: float | np.ndarray,
              order: float = 2.0) -> np.ndarray:
    return _highpass(f, lo, order) * _lowpass(f, hi, order)


def _bump(f: np.ndarray, fc: float | np.ndarray, octaves: float | np.ndarray) -> np.ndarray:
    """Gaussian bump in log-frequency (peak 1 at ``fc``, std ``octaves``)."""
    lf = np.log2(np.maximum(f, 1.0) / fc)
    return np.exp(-0.5 * (lf / octaves) ** 2)


# --------------------------------------------------------------- DSP: filtering


def _fft_filter(x: np.ndarray, sr: int, gain: Callable[[np.ndarray], np.ndarray],
                pad_s: float = 0.0) -> np.ndarray:
    """Zero-phase filter along the last axis: ``irfft(rfft(x) * gain(freqs))``.

    ``pad_s = 0`` filters circularly, which is exactly right for a loop (it is periodic)
    and harmless for noise sources. One-shots pass a padding so that ringing does not
    wrap from the end of the buffer to its start.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    m = n + int(round(pad_s * sr))
    spec = np.fft.rfft(x, n=m, axis=-1)
    spec *= gain(np.fft.rfftfreq(m, 1.0 / sr))
    return np.fft.irfft(spec, m, axis=-1)[..., :n]


def _stft_filter(x: np.ndarray, sr: int,
                 gain: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 frame_s: float = 0.046) -> np.ndarray:
    """Time-varying zero-phase filter (short-time FFT, overlap-add).

    ``gain(freqs, times)`` gets bin frequencies with shape (1, F) and frame centre times
    in seconds with shape (T, 1) and returns gains broadcastable to (T, F). sqrt-Hann
    analysis and synthesis windows at 50 % overlap sum to exactly one, so a unit gain
    reproduces the input and smoothly changing gains produce smooth sweeps.
    """
    x = np.asarray(x, dtype=np.float64)
    lead, n = x.shape[:-1], x.shape[-1]
    frame = 1 << max(8, int(round(math.log2(max(frame_s * sr, 256.0)))))
    hop = frame // 2
    n_frames = -(-(n + hop) // hop)               # every input sample sees two frames
    padded = np.zeros(lead + ((n_frames + 1) * hop,))
    padded[..., hop:hop + n] = x
    window = np.sqrt(0.5 - 0.5 * np.cos(_TAU * np.arange(frame) / frame))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame, axis=-1)[..., ::hop, :]
    frames = frames[..., :n_frames, :]
    spec = np.fft.rfft(frames * window, axis=-1)
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)[None, :]
    times = (np.arange(n_frames) * hop / sr)[:, None]   # frame k is centred on input sample k*hop
    spec *= np.broadcast_to(gain(freqs, times), spec.shape[-2:])
    out_frames = np.fft.irfft(spec, frame, axis=-1) * window
    out = np.zeros_like(padded)
    out[..., :n_frames * hop] += out_frames[..., :hop].reshape(lead + (n_frames * hop,))
    out[..., hop:(n_frames + 1) * hop] += out_frames[..., hop:].reshape(lead + (n_frames * hop,))
    return out[..., hop:hop + n]


# ------------------------------------------------------------ DSP: generators


def _time(n: int, sr: int) -> np.ndarray:
    return np.arange(n, dtype=np.float64) / sr


def _ramp(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """Raised-cosine step from 0 (t <= start) to 1 (t >= end)."""
    u = np.clip((t - start) / max(end - start, 1e-9), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def _attack_decay(t: np.ndarray, attack: float, tau: float) -> np.ndarray:
    """``(1 - e^(-t/attack)) * e^(-t/tau)`` for t >= 0, else 0 (a punchy envelope)."""
    tp = np.maximum(t, 0.0)
    return np.where(t >= 0.0, (1.0 - np.exp(-tp / attack)) * np.exp(-tp / tau), 0.0)


def _exp_sweep(t: np.ndarray, f0: float, f1: float, duration: float,
               curve: float = 1.0) -> np.ndarray:
    """Exponential frequency glide from ``f0`` to ``f1`` over ``duration`` (then holds)."""
    u = np.clip(t / duration, 0.0, 1.0) ** curve
    return f0 * (f1 / f0) ** u


def _cycles(freq: np.ndarray, sr: int) -> np.ndarray:
    """Accumulated phase in cycles for a per-sample frequency array (starts at 0)."""
    acc = np.cumsum(freq) / sr
    return acc - acc[0]


def _saw(cycles: np.ndarray, inc: np.ndarray | float) -> np.ndarray:
    """Band-limited sawtooth via PolyBLEP; ``inc`` = phase increment in cycles/sample."""
    p = np.mod(cycles, 1.0)
    dt = np.broadcast_to(np.clip(inc, 1e-9, 0.5), p.shape)
    y = 2.0 * p - 1.0
    m = p < dt
    u = p[m] / dt[m]
    y[m] -= u + u - u * u - 1.0
    m = p > 1.0 - dt
    u = (p[m] - 1.0) / dt[m]
    y[m] -= u * u + u + u + 1.0
    return y


def _square(cycles: np.ndarray, inc: np.ndarray | float) -> np.ndarray:
    """Band-limited +-1 square wave (difference of two half-cycle shifted saws)."""
    return _saw(cycles, inc) - _saw(cycles + 0.5, inc)


def _periodic_partial(n: int, cycles: int, phase: float = 0.0, fm_cycles: int = 0,
                      fm_index: float = 0.0, fm_phase: float = 0.0) -> np.ndarray:
    """Sine with exactly ``cycles`` periods in ``n`` samples (optionally with a periodic
    vibrato of ``fm_cycles`` periods), so it loops without any seam."""
    k = np.arange(n, dtype=np.int64)
    arg = _TAU * ((k * cycles) % n) / n + phase
    if fm_index and fm_cycles:
        arg = arg + fm_index * np.sin(_TAU * ((k * fm_cycles) % n) / n + fm_phase)
    return np.sin(arg)


def _loop_cycles(freq_hz: float, n: int, sr: int) -> int:
    """Nearest whole number of cycles of ``freq_hz`` that fits a loop of ``n`` samples."""
    return max(1, int(round(freq_hz * n / sr)))


def _stereo_noise(rng: np.random.Generator, n: int, width: float = 0.35) -> np.ndarray:
    """(2, n) unit-variance white noise; ``width`` 0 = identical channels, 1 = independent."""
    common, left, right = rng.standard_normal((3, n))
    a, b = math.sqrt(1.0 - width), math.sqrt(width)
    return np.stack([a * common + b * left, a * common + b * right])


def _smooth_noise(rng: np.random.Generator, n: int, sr: int, cutoff_hz: float) -> np.ndarray:
    """Zero-mean, unit-std random modulator band-limited to ``cutoff_hz``.

    Filtered circularly, so it is periodic over ``n`` samples (useful for loops).
    """
    y = _fft_filter(rng.standard_normal(n), sr, lambda f: _lowpass(f, cutoff_hz, 2.0))
    y -= y.mean()
    std = float(y.std())
    return y / std if std > 1e-12 else y


def _crackle(rng: np.random.Generator, n: int, sr: int, density: np.ndarray | float, *,
             decay_ms: tuple[float, float] = (0.12, 1.0), amp_sigma: float = 0.7,
             amp: np.ndarray | float = 1.0, spread: float = 0.7,
             band: tuple[float, float] | None = (1500.0, 7500.0)) -> np.ndarray:
    """Electric crackle: sparse, randomly panned micro-bursts, (2, n).

    ``density`` is the event rate in events/s (scalar or per sample), ``amp`` an
    optional per-sample gain, ``spread`` 0 = centre .. 1 = hard left/right. Every event
    is a few-sample burst of decaying noise (sharp snap) with log-normal amplitude;
    ``band`` (Hz) colours the result with a band-pass.
    """
    rate = np.broadcast_to(np.asarray(density, dtype=np.float64), (n,)) / sr
    idx = np.flatnonzero(rng.random(n) < np.clip(rate, 0.0, 0.5))
    out = np.zeros((2, n))
    if idx.size:
        kernel_len = max(4, int(sr * decay_ms[1] * 6e-3))
        j = np.arange(kernel_len)
        tau = rng.uniform(decay_ms[0], decay_ms[1], idx.size) * 1e-3 * sr
        level = rng.lognormal(0.0, amp_sigma, idx.size) * np.broadcast_to(
            np.asarray(amp, dtype=np.float64), (n,))[idx]
        kernels = np.exp(-j[None, :] / tau[:, None]) * rng.standard_normal((idx.size, kernel_len))
        kernels *= level[:, None]
        pan = np.pi / 4.0 * (1.0 + spread * rng.uniform(-1.0, 1.0, idx.size))
        pos = idx[:, None] + j[None, :]
        valid = pos < n
        for ch, gain in enumerate((np.cos(pan), np.sin(pan))):
            weights = (kernels * gain[:, None])[valid]
            out[ch] = np.bincount(pos[valid], weights=weights, minlength=n)
    if band is not None:
        lo, hi = band
        out = _fft_filter(out, sr, lambda f: _bandpass(f, lo, hi, 2.0))
    return out


def _fold_loop(x: np.ndarray, n_loop: int) -> np.ndarray:
    """Equal-power cross-fade of the tail beyond ``n_loop`` samples into the head.

    ``x[..., n_loop:]`` (the cross-fade length) fades out over the first samples while
    the head fades in, so ``out[-1] -> out[0]`` equals ``x[n_loop-1] -> x[n_loop]``:
    two neighbouring samples of one continuous render. sin/cos gains keep the power of
    uncorrelated material (noise, crackle) constant through the fade.
    """
    n_fade = x.shape[-1] - n_loop
    out = np.array(x[..., :n_loop], dtype=np.float64)
    if n_fade > 0:
        theta = np.arange(n_fade) / n_fade * (np.pi / 2.0)
        out[..., :n_fade] = x[..., :n_fade] * np.sin(theta) + x[..., n_loop:] * np.cos(theta)
    return out


def _tile(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat a periodic signal (last axis) to length ``n``."""
    reps = -(-n // x.shape[-1])
    return np.tile(x, (1,) * (x.ndim - 1) + (reps,))[..., :n]


def _fade_edges(x: np.ndarray, sr: int, fade_in_ms: float, fade_out_ms: float) -> np.ndarray:
    """Raised-cosine fade-in/out of at least :data:`MIN_FADE_MS` on a one-shot."""
    n = x.shape[-1]
    for ms, head in ((fade_in_ms, True), (fade_out_ms, False)):
        k = min(n // 2, max(int(round(sr * max(ms, MIN_FADE_MS) * 1e-3)), 1))
        ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(k) + 0.5) / k)
        if head:
            x[..., :k] *= ramp
        else:
            x[..., n - k:] *= ramp[::-1]
    return x


def _peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0


def _mix(*parts: tuple[float, np.ndarray]) -> np.ndarray:
    """Weighted sum of components; mono (n,) parts go to both channels of the (2, n) result."""
    out: np.ndarray | None = None
    for gain, x in parts:
        x2 = np.broadcast_to(x, (2, x.shape[-1])) if x.ndim == 1 else x
        out = gain * x2 if out is None else out + gain * x2
    assert out is not None
    return out


def _master(x: np.ndarray, sr: int, *, loop: bool, drive: float, fade_in_ms: float = MIN_FADE_MS,
            fade_out_ms: float = 20.0, highpass_hz: float = 28.0) -> np.ndarray:
    """Final stage: DC/rumble high-pass, ``tanh`` saturation, fades (one-shots only) and
    -1 dBFS normalisation. Returns (n, 2) float32. Loops are filtered circularly and
    get no fades, so their seam stays intact."""
    x = _fft_filter(x, sr, lambda f: _highpass(f, highpass_hz, 2.0), pad_s=0.0 if loop else 0.1)
    peak = _peak(x)
    if not math.isfinite(peak):
        raise FloatingPointError("synthesis produced non-finite samples")
    if peak > 0.0:
        x /= peak
        if drive > 0.0:
            x = np.tanh(drive * x) / math.tanh(drive)
        if not loop:
            x = _fade_edges(x, sr, fade_in_ms, fade_out_ms)
        x *= PEAK_LINEAR / max(_peak(x), 1e-12)
    return np.ascontiguousarray(x.T, dtype=np.float32)


@dataclass(frozen=True)
class _Ctx:
    """Per-render context: sample rate and the sound's own deterministic RNG."""

    sr: int
    rng: np.random.Generator

    def n(self, seconds: float) -> int:
        return max(1, int(round(seconds * self.sr)))


# ----------------------------------------------------------------- recipes: charge


def _charge_start(c: _Ctx) -> np.ndarray:
    """Rising "vwoom": 70 -> 420 Hz sine/saw glide with a sub octave, an air swell whose
    band follows the pitch and crackle that ignites half-way through."""
    sr, rng = c.sr, c.rng
    n = c.n(SOUND_DURATIONS["charge_start"])
    t = _time(n, sr)
    f = _exp_sweep(t, 70.0, 420.0, 0.50, curve=1.15)
    env = _ramp(t, 0.0, 0.40) ** 1.3 * np.exp(-np.maximum(t - 0.40, 0.0) / 0.075)
    base = _cycles(f, sr)
    voices = []
    for detune, offset in ((0.996, 0.0), (1.004, 0.21)):      # L/R slightly detuned: width
        cyc = base * detune + offset
        voices.append(0.70 * np.sin(_TAU * cyc) + 0.30 * _saw(cyc, f * detune / sr)
                      + 0.45 * np.sin(np.pi * cyc))          # sub octave (35 -> 210 Hz)
    # The saw's upper harmonics are tamed by a low-pass that tracks the glide.
    tone = _stft_filter(np.stack(voices), sr,
                        lambda fr, tm: _lowpass(fr, 300.0 + 5.0 * np.interp(tm, t, f), 2.0)) * env
    air_env = _ramp(t, 0.02, 0.45) ** 1.6 * np.exp(-np.maximum(t - 0.45, 0.0) / 0.06)
    air = _stft_filter(_stereo_noise(rng, n, 0.5), sr,
                       lambda fr, tm: _bump(fr, 350.0 * (3000.0 / 350.0) ** np.clip(tm / 0.5, 0.0, 1.0), 0.9))
    crackle = _crackle(rng, n, sr, 150.0 * _ramp(t, 0.18, 0.42), amp=env, band=(2000.0, 8000.0))
    mix = _mix((1.0, tone), (1.6, air * air_env), (0.8, crackle))
    return _master(mix, sr, loop=False, drive=1.6, fade_out_ms=25.0)


def _charge_loop(c: _Ctx) -> np.ndarray:
    """Deep detuned 55/110/165 Hz hum with slow LFOs, bursts of electric crackle, a fine
    sizzle and airy band-passed wind."""
    sr, rng = c.sr, c.rng
    n = c.n(LOOP_SECONDS)
    m = n + c.n(LOOP_CROSSFADE_SECONDS)
    lfo = (1.0 + 0.20 * _periodic_partial(n, _loop_cycles(0.5, n, sr), 0.3)
           + 0.10 * _periodic_partial(n, _loop_cycles(3.0, n, sr), 1.1)
           + 0.06 * _periodic_partial(n, _loop_cycles(1.5, n, sr), 2.0))
    hum = np.zeros((2, n))
    partials = ((55.0, 1.0), (55.5, 0.5), (110.0, 0.55), (110.5, 0.25),
                (165.0, 0.30), (166.5, 0.14), (220.0, 0.14), (330.0, 0.07))
    for i, (freq, amp) in enumerate(partials):
        cycles = _loop_cycles(freq, n, sr)
        phase = rng.uniform(0.0, _TAU)
        spread = 0.35 if i % 2 else 0.0                       # detuned partners: stereo width
        hum[0] += amp * _periodic_partial(n, cycles, phase - spread)
        hum[1] += amp * _periodic_partial(n, cycles, phase + spread)
    hum *= lfo
    # Random parts are rendered longer and folded (seamless loop).
    density = 5.0 + 55.0 * np.clip(_smooth_noise(rng, m, sr, 1.2), 0.0, None) ** 1.5
    crackle = _crackle(rng, m, sr, density, decay_ms=(0.08, 0.8), band=(1800.0, 7500.0))
    sizzle = _crackle(rng, m, sr, 600.0, decay_ms=(0.05, 0.25), amp_sigma=0.4, band=(3000.0, 9000.0))
    wind = _fft_filter(_stereo_noise(rng, m, 0.6), sr, lambda f: _bump(f, 1300.0, 1.0) * _highpass(f, 400.0))
    wind *= np.clip(1.0 + 0.35 * _smooth_noise(rng, m, sr, 0.8), 0.2, None)
    texture = _fold_loop(_mix((0.9, crackle), (0.12, sizzle), (0.9, wind)), n)
    return _master(_mix((1.0, hum), (1.0, texture)), sr, loop=True, drive=1.3)


def _power_up(c: _Ctx) -> np.ndarray:
    """Rising shimmering sweep: three detuned voices gliding 260 -> 1250 Hz with an
    accelerating tremolo, glitter sparks and a high shimmer."""
    sr, rng = c.sr, c.rng
    n = c.n(SOUND_DURATIONS["power_up"])
    t = _time(n, sr)
    f = _exp_sweep(t, 260.0, 1250.0, 0.62, curve=0.9)
    env = _ramp(t, 0.0, 0.55) ** 1.2 * np.exp(-np.maximum(t - 0.56, 0.0) / 0.045)
    trem_rate = 9.0 + 19.0 * np.clip(t / 0.6, 0.0, 1.0)
    trem = 1.0 - 0.30 * (0.5 + 0.5 * np.sin(_TAU * _cycles(trem_rate, sr)))
    base = _cycles(f, sr)
    voices = np.zeros((2, n))
    for detune, left in ((0.994, 0.8), (1.0, 0.5), (1.006, 0.2)):
        cyc = base * detune + rng.uniform()
        v = np.sin(_TAU * cyc) + 0.35 * np.sin(2.0 * _TAU * cyc) + 0.15 * np.sin(3.0 * _TAU * cyc)
        voices += np.stack([math.sqrt(left) * v, math.sqrt(1.0 - left) * v])
    body = np.sin(np.pi * base)                                 # octave below for weight
    sparkle = _crackle(rng, n, sr, 90.0 * _ramp(t, 0.10, 0.50), decay_ms=(0.03, 0.15),
                       amp=env, band=(4500.0, 11000.0))
    shimmer = _fft_filter(_stereo_noise(rng, n, 0.8), sr, lambda fr: _bump(fr, 7000.0, 0.6), pad_s=0.05)
    mix = _mix((0.55, voices * trem * env), (0.30, body * env), (0.50, sparkle),
               (0.35, shimmer * trem * env))
    return _master(mix, sr, loop=False, drive=1.2, fade_out_ms=20.0)


def _fizzle(c: _Ctx) -> np.ndarray:
    """Descending sizzle: noise band falling 4.5 kHz -> 650 Hz, thinning crackle and a
    wobbling power-down tone."""
    sr, rng = c.sr, c.rng
    n = c.n(SOUND_DURATIONS["fizzle"])
    t = _time(n, sr)
    env = (1.0 - np.exp(-t / 0.008)) * (0.7 * np.exp(-t / 0.14) + 0.3 * np.clip(1.0 - t / 0.5, 0.0, 1.0))
    sizzle = _stft_filter(_stereo_noise(rng, n, 0.6), sr,
                          lambda fr, tm: _bump(fr, 4500.0 * (650.0 / 4500.0) ** np.clip(tm / 0.45, 0.0, 1.0), 0.9))
    density = 180.0 * np.exp(-t / 0.12) + 12.0 * np.clip(1.0 - t / 0.5, 0.0, 1.0)
    crackle = _crackle(rng, n, sr, density, amp=np.sqrt(env), band=(1500.0, 7000.0))
    f = _exp_sweep(t, 720.0, 140.0, 0.42) * (1.0 + 0.04 * np.sin(_TAU * 17.0 * t))
    tone = np.sin(_TAU * _cycles(f, sr)) * _attack_decay(t, 0.006, 0.11)
    mix = _mix((1.5, sizzle * env), (0.7, crackle), (0.25, tone))
    return _master(mix, sr, loop=False, drive=1.3, fade_out_ms=40.0)


def _zap(c: _Ctx) -> np.ndarray:
    """Short electric "zzt": jittered band-limited square buzz, dense crackle and hiss,
    chopped by a fast random gate."""
    sr, rng = c.sr, c.rng
    n = c.n(SOUND_DURATIONS["zap"])
    t = _time(n, sr)
    tt = t - 0.003
    env = _attack_decay(tt, 0.0012, 0.045)
    gate = np.clip(0.55 + 0.9 * _smooth_noise(rng, n, sr, 180.0), 0.0, 1.0)
    f0 = np.maximum(140.0 * (1.0 + 0.3 * _smooth_noise(rng, n, sr, 70.0)), 60.0)
    buzz = _fft_filter(_square(_cycles(f0, sr), f0 / sr), sr,
                       lambda f: _bandpass(f, 500.0, 6000.0, 2.0), pad_s=0.03)
    crackle = _crackle(rng, n, sr, 1100.0, decay_ms=(0.05, 0.4), band=(2200.0, 9500.0))
    hiss = _fft_filter(_stereo_noise(rng, n, 0.5), sr, lambda f: _bandpass(f, 3500.0, 9000.0, 2.0), pad_s=0.03)
    mix = _mix((0.8, buzz), (1.0, crackle), (0.4, hiss)) * (env * gate)
    return _master(mix, sr, loop=False, drive=1.5, fade_out_ms=15.0)


def _ui_click(c: _Ctx) -> np.ndarray:
    """Soft blip: a 1.65 -> 1.1 kHz pitch-dropping sine with a gentle 5 ms attack."""
    sr = c.sr
    n = c.n(SOUND_DURATIONS["ui_click"])
    t = _time(n, sr)
    cyc = _cycles(1100.0 + 550.0 * np.exp(-t / 0.004), sr)
    blip = (np.sin(_TAU * cyc) + 0.12 * np.sin(2.0 * _TAU * cyc)) * np.exp(-t / 0.0075)
    return _master(blip[None, :].repeat(2, axis=0), sr, loop=False, drive=0.0, fade_out_ms=8.0,
                   highpass_hz=200.0)


# ------------------------------------------------------------- recipes: attacks


def _boom(t: np.ndarray, sr: int, f_end: float, f_kick: float, kick_tau: float, attack: float,
          tau: float) -> np.ndarray:
    """Sub-bass boom with a downward pitch "kick" (punch) and a little 2nd/3rd harmonic
    so it still reads on laptop speakers."""
    f = f_end + f_kick * np.exp(-np.maximum(t, 0.0) / kick_tau)
    cyc = _cycles(f, sr)
    tone = np.sin(_TAU * cyc) + 0.28 * np.sin(2.0 * _TAU * cyc) + 0.08 * np.sin(3.0 * _TAU * cyc)
    return tone * _attack_decay(t, attack, tau)


def _crack(rng: np.random.Generator, n: int, sr: int, t: np.ndarray, tau: float = 0.005,
           low: float = 1200.0) -> np.ndarray:
    """Initial broadband crack transient (stereo)."""
    x = _stereo_noise(rng, n, 0.3) * _attack_decay(t, 0.0003, tau)
    return _fft_filter(x, sr, lambda f: _highpass(f, low, 1.0) * _lowpass(f, 10000.0, 2.0), pad_s=0.05)


def _falling_lowpass(t0: float, f_hi: float, f_lo: float, tau: float) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """STFT gain: low-pass whose cutoff falls exponentially from ``f_hi`` to ``f_lo`` after ``t0``."""
    def gain(fr: np.ndarray, tm: np.ndarray) -> np.ndarray:
        cutoff = f_lo + (f_hi - f_lo) * np.exp(-np.maximum(tm - t0, 0.0) / tau)
        return _lowpass(fr, cutoff, 2.0)
    return gain


def _beam_fire(c: _Ctx) -> np.ndarray:
    """Kamehameha release: crack, noise blast under a low-pass falling 8 kHz -> 300 Hz,
    roaring mids and an 85 -> 45 Hz sub boom, saturated."""
    sr, rng = c.sr, c.rng
    n = c.n(SOUND_DURATIONS["beam_fire"])
    t = _time(n, sr)
    t0 = 0.005                                                # onset after the anti-click fade
    tt = t - t0
    blast_env = 0.65 * _attack_decay(tt, 0.004, 0.20) + 0.35 * _attack_decay(tt, 0.004, 0.65)
    blast = _stft_filter(_stereo_noise(rng, n, 0.55) * blast_env, sr,
                         _falling_lowpass(t0, 8000.0, 300.0, 0.30))
    roar = _fft_filter(_stereo_noise(rng, n, 0.5), sr, lambda f: _bandpass(f, 150.0, 1100.0, 2.0), pad_s=0.1)
    roar *= _attack_decay(tt, 0.02, 0.5)
    sub = _boom(tt, sr, 45.0, 40.0, 0.05, 0.003, 0.42)
    mix = _mix((0.6, _crack(rng, n, sr, tt)), (1.6, blast), (1.0, roar), (1.0, sub))
    return _master(mix, sr, loop=False, drive=2.0, fade_out_ms=60.0)


def _beam_loop(c: _Ctx) -> np.ndarray:
    """Sustained beam: roaring 300-2500 Hz noise with a wandering resonance, slow swells
    and fast flutter, low rumble and a faint vibrato whine."""
    sr, rng = c.sr, c.rng
    n = c.n(LOOP_SECONDS)
    m = n + c.n(LOOP_CROSSFADE_SECONDS)
    # Periodic (exact-cycle) tonal layer.
    swell = 1.0 + 0.25 * _periodic_partial(n, _loop_cycles(1.0, n, sr), 0.4)
    rumble_tone = (0.6 * _periodic_partial(n, _loop_cycles(41.0, n, sr), rng.uniform(0, _TAU))
                   + 0.3 * _periodic_partial(n, _loop_cycles(82.0, n, sr), rng.uniform(0, _TAU))) * swell
    vib = _loop_cycles(1.5, n, sr)
    whine = np.stack([
        _periodic_partial(n, _loop_cycles(3150.0, n, sr) + dc, 0.0, vib, 9.0, ph)
        + 0.4 * _periodic_partial(n, _loop_cycles(4725.0, n, sr) + dc, 1.0, vib, 13.5, ph)
        for dc, ph in ((0, 0.0), (1, 0.8))                    # R detuned by one cycle: slow phasing
    ])
    # Random layers, rendered longer and folded. Sinusoidal modulators use whole cycles
    # per loop so the folded tail and head carry the same modulation.
    tm_long = _time(m, sr)
    wander = 2.0 ** (0.6 * np.sin(_TAU * 0.5 * tm_long + rng.uniform(0, _TAU))
                     + 0.25 * _smooth_noise(rng, m, sr, 1.0))

    def roar_gain(fr: np.ndarray, tm: np.ndarray) -> np.ndarray:
        centre = 900.0 * np.interp(tm, tm_long, wander)
        return _bandpass(fr, 300.0, 2500.0, 2.0) * (0.55 + 0.45 * _bump(fr, centre, 0.7))

    roar = _stft_filter(_stereo_noise(rng, m, 0.5), sr, roar_gain)
    roar *= np.clip((1.0 + 0.25 * np.sin(_TAU * 0.5 * tm_long) + 0.18 * _smooth_noise(rng, m, sr, 2.5))
                    * (1.0 + 0.22 * _smooth_noise(rng, m, sr, 14.0)), 0.1, None)
    rumble = _fft_filter(_stereo_noise(rng, m, 0.3), sr, lambda f: _lowpass(f, 110.0, 2.0) * _highpass(f, 30.0, 2.0))
    rumble *= np.clip(1.0 + 0.3 * _smooth_noise(rng, m, sr, 3.0), 0.2, None)
    texture = _fold_loop(_mix((1.0, roar), (1.2, rumble)), n)
    return _master(_mix((1.0, texture), (0.30, rumble_tone), (0.035, whine)), sr, loop=True, drive=1.5)


def _blast_fire(c: _Ctx) -> np.ndarray:
    """Ki blast thrown: punchy thump + click, a falling anime "pew" and a noise whoosh
    whose band drops as the blast flies away."""
    sr, rng = c.sr, c.rng
    n = c.n(SOUND_DURATIONS["blast_fire"])
    t = _time(n, sr)
    t0 = 0.005
    tt = t - t0
    thump = _boom(tt, sr, 62.0, 120.0, 0.025, 0.0015, 0.07)
    click = _crack(rng, n, sr, tt, tau=0.003, low=1500.0)
    f = _exp_sweep(tt, 1300.0, 240.0, 0.22, curve=0.8)
    pew = np.stack([np.sin(_TAU * _cycles(f * d, sr)) + 0.3 * np.sin(2.0 * _TAU * _cycles(f * d, sr))
                    for d in (0.997, 1.003)]) * _attack_decay(tt, 0.004, 0.10)
    whoosh_env = _ramp(tt, 0.0, 0.035) * np.exp(-np.maximum(tt - 0.03, 0.0) / 0.15)
    whoosh = _stft_filter(_stereo_noise(rng, n, 0.7), sr,
                          lambda fr, tm: _bump(fr, 500.0 + 2600.0 * np.exp(-np.maximum(tm - t0, 0.0) / 0.18), 1.1))
    mix = _mix((1.0, thump), (0.5, click), (0.35, pew), (1.3, whoosh * whoosh_env))
    return _master(mix, sr, loop=False, drive=1.6, fade_out_ms=30.0)


def _impact(c: _Ctx) -> np.ndarray:
    """Explosion / hit: crack, sub boom, falling-low-pass blast, then a crackling tail
    with some low debris."""
    sr, rng = c.sr, c.rng
    n = c.n(SOUND_DURATIONS["impact"])
    t = _time(n, sr)
    t0 = 0.005
    tt = t - t0
    boom = _boom(tt, sr, 38.0, 55.0, 0.06, 0.003, 0.30)
    blast_env = 0.7 * _attack_decay(tt, 0.003, 0.12) + 0.3 * _attack_decay(tt, 0.003, 0.45)
    blast = _stft_filter(_stereo_noise(rng, n, 0.6) * blast_env, sr, _falling_lowpass(t0, 5700.0, 220.0, 0.11))
    tail_density = 110.0 * np.exp(-np.maximum(tt, 0.0) / 0.30) * _ramp(tt, 0.03, 0.08) + 3.0
    tail = _crackle(rng, n, sr, tail_density, amp=np.exp(-np.maximum(tt, 0.0) / 0.45), band=(1000.0, 6500.0))
    debris = _crackle(rng, n, sr, 25.0 * np.exp(-np.maximum(tt, 0.0) / 0.25) * _ramp(tt, 0.02, 0.05),
                      decay_ms=(0.6, 2.5), amp=np.exp(-np.maximum(tt, 0.0) / 0.35), band=(250.0, 1500.0))
    mix = _mix((1.0, boom), (1.4, blast), (0.5, _crack(rng, n, sr, tt, tau=0.004, low=1000.0)),
               (0.6, tail), (0.5, debris))
    return _master(mix, sr, loop=False, drive=2.0, fade_out_ms=60.0)


def _burst(c: _Ctx) -> np.ndarray:
    """Energy burst: a 0.3 s reverse swell (rising band + rising tone, "sucking in"),
    then a massive saturated explosion with a long rumble and debris tail."""
    sr, rng = c.sr, c.rng
    n = c.n(SOUND_DURATIONS["burst"])
    t = _time(n, sr)
    tx = 0.30                                                  # moment of the explosion
    swell_env = np.where(t < tx, np.exp((t - tx) / 0.07), 0.0) * (1.0 - _ramp(t, tx - 0.004, tx))
    swell = _stft_filter(_stereo_noise(rng, n, 0.6), sr,
                         lambda fr, tm: _bump(fr, 400.0 * (5000.0 / 400.0) ** np.clip(tm / tx, 0.0, 1.0), 1.2))
    f = _exp_sweep(t, 160.0, 900.0, tx)
    cyc = _cycles(f, sr)
    rise = _fft_filter(0.6 * np.sin(_TAU * cyc) + 0.4 * _saw(cyc, f / sr), sr,
                       lambda fr: _lowpass(fr, 2500.0, 2.0), pad_s=0.05)
    tt = t - tx
    boom = _boom(tt, sr, 33.0, 70.0, 0.07, 0.003, 0.50)
    blast_env = 0.6 * _attack_decay(tt, 0.004, 0.15) + 0.4 * _attack_decay(tt, 0.004, 0.60)
    blast = _stft_filter(_stereo_noise(rng, n, 0.7) * blast_env, sr, _falling_lowpass(tx, 10000.0, 250.0, 0.22))
    rumble = _fft_filter(_stereo_noise(rng, n, 0.4), sr, lambda fr: _lowpass(fr, 200.0, 2.0), pad_s=0.1)
    rumble *= _attack_decay(tt, 0.05, 0.6)
    debris = _crackle(rng, n, sr, 140.0 * np.exp(-np.maximum(tt, 0.0) / 0.40) * _ramp(tt, 0.02, 0.06),
                      amp=np.exp(-np.maximum(tt, 0.0) / 0.6), band=(1200.0, 7000.0))
    mix = _mix((1.0, swell * swell_env), (0.35, rise * swell_env), (1.1, boom), (1.5, blast),
               (0.6, _crack(rng, n, sr, tt)), (0.6, debris), (0.8, rumble))
    return _master(mix, sr, loop=False, drive=2.4, fade_out_ms=80.0)


def _aura_loop(c: _Ctx) -> np.ndarray:
    """Aura flames: low-passed noise with a fast random flicker, a little hiss, sparse
    fire pops and a low 49 Hz hum."""
    sr, rng = c.sr, c.rng
    n = c.n(LOOP_SECONDS)
    m = n + c.n(LOOP_CROSSFADE_SECONDS)
    am = (1.0 + 0.15 * _periodic_partial(n, _loop_cycles(1.0, n, sr), 0.7)
          + 0.08 * _periodic_partial(n, _loop_cycles(2.5, n, sr), 2.1))
    hum = sum(a * _periodic_partial(n, _loop_cycles(fq, n, sr), rng.uniform(0, _TAU))
              for fq, a in ((49.0, 0.6), (98.0, 0.3), (147.0, 0.12), (196.0, 0.06))) * am
    flame = _fft_filter(_stereo_noise(rng, m, 0.7), sr, lambda f: _lowpass(f, 650.0, 2.0) * _highpass(f, 70.0, 2.0))
    flicker = np.clip(1.0 + 0.75 * _smooth_noise(rng, m, sr, 14.0) + 0.35 * _smooth_noise(rng, m, sr, 5.0), 0.08, None)
    hiss = _fft_filter(_stereo_noise(rng, m, 0.8), sr, lambda f: _bandpass(f, 1800.0, 6000.0, 2.0))
    hiss *= np.clip(1.0 + 0.9 * _smooth_noise(rng, m, sr, 9.0), 0.0, None)
    pops = _crackle(rng, m, sr, 4.0, decay_ms=(0.3, 1.5), band=(500.0, 3000.0))
    texture = _fold_loop(_mix((1.0, flame * flicker), (0.15, hiss), (0.4, pops)), n)
    return _master(_mix((0.55, hum), (1.0, texture)), sr, loop=True, drive=1.3)


_RECIPES: dict[str, Callable[[_Ctx], np.ndarray]] = {
    "charge_start": _charge_start,
    "charge_loop": _charge_loop,
    "beam_fire": _beam_fire,
    "beam_loop": _beam_loop,
    "blast_fire": _blast_fire,
    "impact": _impact,
    "burst": _burst,
    "aura_loop": _aura_loop,
    "zap": _zap,
    "power_up": _power_up,
    "fizzle": _fizzle,
    "ui_click": _ui_click,
}

# ------------------------------------------------------------ public synthesis API

_CACHE: dict[tuple[str, int, int], np.ndarray] = {}
_CACHE_LOCK = threading.Lock()


def _check_rate(sample_rate: int) -> int:
    sr = int(sample_rate)
    if not 8000 <= sr <= 384000:
        raise ValueError(f"unsupported sample rate {sample_rate!r} (8000..384000 Hz)")
    return sr


def _rng_for(name: str, seed: int) -> np.random.Generator:
    # crc32 rather than hash(): str hashes are randomised per process.
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, zlib.crc32(name.encode("utf-8"))])


def is_loop(name: str) -> bool:
    """True for the looping sounds (``charge_loop``, ``beam_loop``, ``aura_loop``)."""
    return name in LOOP_SOUNDS


def render_sound(name: str, sample_rate: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """Synthesise one sound as float32 ``(n, 2)`` samples in [-1, 1] (not cached).

    Raises ``KeyError`` for unknown names and ``ValueError`` for unusable sample rates.
    """
    recipe = _RECIPES.get(name)
    if recipe is None:
        raise KeyError(f"unknown sound {name!r}")
    return recipe(_Ctx(_check_rate(sample_rate), _rng_for(name, seed)))


def to_int16(samples: np.ndarray) -> np.ndarray:
    """Quantise float samples in [-1, 1] to C-contiguous int16."""
    return np.ascontiguousarray(np.clip(np.round(samples * 32767.0), -32767, 32767).astype(np.int16))


def synthesize(name: str, sample_rate: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """The sound as 16-bit stereo: ``(n, 2)`` int16 peaking at -1 dBFS.

    Deterministic for ``(name, sample_rate, seed)`` and cached per process; the returned
    array is read-only because it is shared.
    """
    key = (name, _check_rate(sample_rate), int(seed))
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
    if cached is not None:
        return cached
    data = to_int16(render_sound(*key))
    data.flags.writeable = False
    with _CACHE_LOCK:
        return _CACHE.setdefault(key, data)


def synthesize_all(sample_rate: int = SAMPLE_RATE, seed: int = 0) -> dict[str, np.ndarray]:
    """Every name in ``SOUND_NAMES`` -> :func:`synthesize` result."""
    return {name: synthesize(name, sample_rate, seed) for name in SOUND_NAMES}


def clear_cache() -> None:
    """Forget all cached renders (they are re-synthesised on demand)."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ------------------------------------------------------------------- analysis


def _as_float(data: np.ndarray) -> np.ndarray:
    """(n, C) samples as float64 with full scale 1.0 (int16 full scale = 32768)."""
    x = np.asarray(data)
    x = x.astype(np.float64) / 32768.0 if x.dtype.kind in "iu" else x.astype(np.float64)
    return x[:, None] if x.ndim == 1 else x


def dbfs(value: float) -> float:
    """Linear amplitude (full scale 1.0) -> dBFS; ``-inf`` for silence."""
    return 20.0 * math.log10(value) if value > 0.0 else -math.inf


def rms(data: np.ndarray) -> float:
    x = _as_float(data)
    return float(np.sqrt(np.mean(x * x))) if x.size else 0.0


def spectral_centroid(data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    """Magnitude-weighted mean frequency in Hz of the mono mix (Hann windowed)."""
    mono = _as_float(data).mean(axis=1)
    if mono.size < 2:
        return 0.0
    mag = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    total = float(mag.sum())
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    return float((freqs * mag).sum() / total) if total > 0.0 else 0.0


@dataclass(frozen=True)
class LoopSeam:
    """How the end of a loop joins its start (all amplitudes: full scale = 1)."""

    jump: float             # max over channels of |x[0] - x[-1]|
    jump_over_rms: float    # jump relative to the loop's RMS
    jump_over_step: float   # jump relative to the 99.9th percentile of |x[i+1] - x[i]|
    rms_ratio_db: float     # RMS of the +-window around the seam relative to the loop RMS


def loop_seam(data: np.ndarray, sample_rate: int = SAMPLE_RATE, window_s: float = 0.05) -> LoopSeam:
    """Measure the loop seam: a seamless loop has ``jump_over_step`` <= ~1 (the wrap is
    no bigger than ordinary sample steps) and ``rms_ratio_db`` near 0."""
    x = _as_float(data)
    jump = float(np.max(np.abs(x[0] - x[-1])))
    level = rms(x)
    step = float(np.quantile(np.abs(np.diff(x, axis=0)), 0.999)) if len(x) > 1 else 0.0
    w = max(1, min(len(x) // 2, int(round(window_s * sample_rate))))
    seam_level = rms(np.concatenate([x[-w:], x[:w]]))
    return LoopSeam(jump=jump, jump_over_rms=jump / level if level > 0 else math.inf,
                    jump_over_step=jump / step if step > 0 else math.inf,
                    rms_ratio_db=dbfs(seam_level / level) if level > 0 else 0.0)


@dataclass(frozen=True)
class SoundStats:
    """Level summary of one rendered sound (see :func:`analyze`)."""

    name: str
    loop: bool
    duration: float
    peak_dbfs: float
    rms_dbfs: float
    centroid_hz: float
    seam: LoopSeam | None


def analyze(name: str, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> SoundStats:
    x = _as_float(data)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    return SoundStats(name=name, loop=is_loop(name), duration=len(x) / sample_rate,
                      peak_dbfs=dbfs(peak), rms_dbfs=dbfs(rms(x)),
                      centroid_hz=spectral_centroid(x, sample_rate),
                      seam=loop_seam(x, sample_rate) if is_loop(name) else None)


def format_stats(stats: Sequence[SoundStats]) -> str:
    """Plain-text table (used by the command line)."""
    header = (f"{'sound':<13} {'kind':<8} {'dur s':>6} {'peak dBFS':>10} {'RMS dBFS':>9} "
              f"{'centroid Hz':>12} {'seam |d|/RMS':>13} {'seam/step':>10} {'seam RMS dB':>12}")
    lines = [header, "-" * len(header)]
    for s in stats:
        if s.seam is not None:
            seam = f"{s.seam.jump_over_rms:>13.4f} {s.seam.jump_over_step:>10.3f} {s.seam.rms_ratio_db:>+12.2f}"
        else:
            seam = f"{'-':>13} {'-':>10} {'-':>12}"
        lines.append(f"{s.name:<13} {'loop' if s.loop else 'one-shot':<8} {s.duration:>6.3f} "
                     f"{s.peak_dbfs:>10.2f} {s.rms_dbfs:>9.2f} {s.centroid_hz:>12.0f} {seam}")
    return "\n".join(lines)


def write_wav(path: Path, data: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write ``(n, C)`` int16 samples as a PCM WAV file (little-endian, as WAV requires)."""
    x = np.asarray(data)
    if x.dtype != np.int16:
        raise TypeError(f"expected int16 samples, got {x.dtype}")
    if x.ndim == 1:
        x = x[:, None]
    with wave.open(str(path), "wb") as out:
        out.setnchannels(int(x.shape[1]))
        out.setsampwidth(2)
        out.setframerate(int(sample_rate))
        out.writeframes(np.ascontiguousarray(x, dtype="<i2").tobytes())


def export_sounds(directory: Path, sample_rate: int = SAMPLE_RATE, seed: int = 0,
                  names: Sequence[str] = SOUND_NAMES) -> list[SoundStats]:
    """Write ``<directory>/<name>.wav`` for each sound and return their stats."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stats = []
    for name in names:
        data = synthesize(name, sample_rate, seed)
        write_wav(directory / f"{name}.wav", data, sample_rate)
        stats.append(analyze(name, data, sample_rate))
    return stats


# ------------------------------------------------------------------------ engine


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _adapt_channels(data: np.ndarray, channels: int) -> np.ndarray:
    """Convert a stereo ``(n, 2)`` int16 render to the mixer's channel count."""
    if channels == 2:
        return data
    if channels == 1:
        return np.ascontiguousarray(data.astype(np.int32).mean(axis=1).astype(np.int16))
    # More than two output channels: put the stereo pair on the front speakers.
    out = np.zeros((data.shape[0], channels), np.int16)
    out[:, :2] = data
    return out


class AudioEngine:
    """:class:`~dragon_ball_powers.interfaces.AudioSink` on top of ``pygame.mixer``.

    ``start()`` initialises the mixer (unless the application already did) and
    synthesises all sounds in a background thread; until a sound is ready, calls for it
    are ignored. Without an audio device ``start()`` returns False and every method is a
    silent no-op, so callers never need to check.
    """

    def __init__(self, settings: AudioSettings, *, seed: int = 0) -> None:
        self._enabled = bool(settings.enabled)
        self._master = _clamp01(settings.volume)
        self._seed = int(seed)
        self._lock = threading.RLock()
        self._sounds: dict[str, object] = {}
        self._ready = threading.Event()
        self._started = False
        self._available = False
        self._owns_mixer = False
        self._closed = False
        self._thread: threading.Thread | None = None
        self._sample_rate = SAMPLE_RATE
        self._channels = 2
        self._loop_channels: dict[str, list[object]] = {}
        self._loop_active: dict[str, int] = {}
        self._loop_volume: dict[str, float] = {}
        self._loop_stopping: dict[str, bool] = {}
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ status

    @property
    def available(self) -> bool:
        """The mixer is initialised (sounds may still be synthesising)."""
        return self._available and not self._closed

    @property
    def ready(self) -> bool:
        """All sounds are synthesised."""
        return self._ready.is_set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def wait_ready(self, timeout: float | None = None) -> bool:
        return self._ready.wait(timeout)

    # --------------------------------------------------------------- lifecycle

    def start(self) -> bool:
        """Initialise the mixer and start synthesising. Returns False without audio."""
        with self._lock:
            if self._started:
                return self.available
            self._started = True
            try:
                import pygame

                if not pygame.mixer.get_init():
                    pygame.mixer.init(frequency=MIXER_FREQUENCY, size=MIXER_SIZE,
                                      channels=MIXER_CHANNELS, buffer=MIXER_BUFFER)
                    self._owns_mixer = True
                init = pygame.mixer.get_init()
                if not init:
                    raise RuntimeError("pygame.mixer did not initialise")
                frequency, size, channels = init
                if size != -16:
                    log.warning("Mixer sample format %s is not 16-bit signed; sounds may be distorted", size)
                self._sample_rate = int(frequency)
                self._channels = int(channels)
                loops = [name for name in SOUND_NAMES if name in LOOP_SOUNDS]
                pygame.mixer.set_num_channels(max(NUM_CHANNELS, 2 * len(loops) + 8))
                pygame.mixer.set_reserved(2 * len(loops))
                for i, name in enumerate(loops):
                    self._loop_channels[name] = [pygame.mixer.Channel(2 * i), pygame.mixer.Channel(2 * i + 1)]
            except Exception as exc:  # noqa: BLE001 - no device, driver problems, ...
                log.warning("Audio unavailable (%s); continuing without sound", exc)
                self._available = False
                return False
            self._available = True
            self._thread = threading.Thread(target=self._synthesize_all, name="audio-synth", daemon=True)
            self._thread.start()
            log.info("Audio: mixer %s Hz, %d channel(s); synthesising %d sounds",
                     self._sample_rate, self._channels, len(SOUND_NAMES))
            return True

    def _synthesize_all(self) -> None:
        started = time.perf_counter()
        try:
            import pygame

            for name in SOUND_NAMES:
                if self._closed:
                    return
                data = _adapt_channels(synthesize(name, self._sample_rate, self._seed), self._channels)
                sound = pygame.mixer.Sound(buffer=np.ascontiguousarray(data).tobytes())
                with self._lock:
                    if self._closed:
                        return
                    self._sounds[name] = sound
            self._ready.set()
            log.info("Audio: %d sounds ready in %.2f s", len(self._sounds), time.perf_counter() - started)
        except Exception:  # noqa: BLE001 - never take the app down because of sound
            log.exception("Sound synthesis failed; continuing without sound")

    def close(self) -> None:
        """Stop everything; quit the mixer if this engine initialised it. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop_everything(0)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(_CLOSE_JOIN_TIMEOUT)
        with self._lock:
            self._sounds.clear()
            self._loop_channels.clear()
            self._loop_active.clear()
        if self._owns_mixer:
            try:
                import pygame

                pygame.mixer.quit()
            except Exception:  # noqa: BLE001
                pass
        self._available = False

    # -------------------------------------------------------------- settings

    def apply_settings(self, settings: AudioSettings) -> None:
        enabled = bool(settings.enabled)
        self._master = _clamp01(settings.volume)
        if self._enabled and not enabled:
            self._stop_everything(DISABLE_FADE_MS)
        self._enabled = enabled
        # Loop volumes are re-sent by the next loop() call (it runs every frame).

    def set_volume(self, volume: float) -> None:
        self._master = _clamp01(volume)

    # ------------------------------------------------------------- AudioSink

    def play(self, name: str, volume: float = 1.0) -> None:
        sound = self._sound_for(name)
        if sound is None:
            return
        try:
            import pygame

            level = _clamp01(volume) * self._master * MIX_LEVELS.get(name, 1.0)
            if level <= 0.0:
                return
            channel = pygame.mixer.find_channel(True)  # steal the oldest one-shot when all are busy
            if channel is None:
                return
            channel.set_volume(level)
            channel.play(sound)
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"play:{type(exc).__name__}", "Could not play %s: %s", name, exc)

    def loop(self, name: str, volume: float) -> None:
        if name not in LOOP_SOUNDS:
            if name in SOUND_NAMES:
                self._warn_once(f"notloop:{name}", "%s is a one-shot sound; use play()", name)
            else:
                self._sound_for(name)  # warns once about the unknown name
            return
        sound = self._sound_for(name)
        if sound is None:
            return
        level = _clamp01(volume) * self._master * MIX_LEVELS.get(name, 1.0)
        try:
            channels = self._loop_channels.get(name)
            if not channels:
                return
            active = self._loop_active.get(name)
            running = (active is not None and not self._loop_stopping.get(name, False)
                       and channels[active].get_busy())
            if running:
                if abs(level - self._loop_volume.get(name, -1.0)) > VOLUME_EPSILON:
                    channels[active].set_volume(level)
                    self._loop_volume[name] = level
                return
            # (Re)start. If the previous instance is still fading out, use the other
            # reserved channel so the two cross-fade instead of cutting.
            index = 0
            if active is not None:
                index = (active + 1) % 2 if channels[active].get_busy() else active
            channel = channels[index]
            channel.set_volume(level)
            channel.play(sound, loops=-1, fade_ms=LOOP_FADE_IN_MS)
            self._loop_active[name] = index
            self._loop_volume[name] = level
            self._loop_stopping[name] = False
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"loop:{type(exc).__name__}", "Could not loop %s: %s", name, exc)

    def stop(self, name: str, fade_ms: int = 250) -> None:
        if not self.available:
            return
        try:
            if name in LOOP_SOUNDS:
                active = self._loop_active.get(name)
                channels = self._loop_channels.get(name)
                if active is not None and channels and not self._loop_stopping.get(name, False):
                    if fade_ms > 0:
                        channels[active].fadeout(int(fade_ms))
                    else:
                        channels[active].stop()
                    self._loop_stopping[name] = True
                return
            sound = self._sounds.get(name)
            if sound is not None:
                if fade_ms > 0:
                    sound.fadeout(int(fade_ms))
                else:
                    sound.stop()
            elif name not in SOUND_NAMES:
                self._sound_for(name)
        except Exception as exc:  # noqa: BLE001
            self._warn_once(f"stop:{type(exc).__name__}", "Could not stop %s: %s", name, exc)

    def stop_all(self) -> None:
        self._stop_everything(150)

    # ---------------------------------------------------------------- helpers

    def _sound_for(self, name: str) -> object | None:
        if name not in SOUND_NAMES:
            self._warn_once(f"unknown:{name}", "Unknown sound %r ignored", name)
            return None
        if not (self._available and self._enabled and not self._closed):
            return None
        return self._sounds.get(name)

    def _stop_everything(self, fade_ms: int) -> None:
        if not self._available:
            return
        try:
            import pygame

            if not pygame.mixer.get_init():
                return
            if fade_ms > 0:
                pygame.mixer.fadeout(int(fade_ms))
            else:
                pygame.mixer.stop()
        except Exception as exc:  # noqa: BLE001
            log.debug("Stopping sounds failed: %s", exc)
        for name in list(self._loop_active):
            self._loop_stopping[name] = True

    def _warn_once(self, key: str, message: str, *args: object) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        log.warning(message, *args)


# ---------------------------------------------------------------------- CLI


def _print_table(stats: Sequence[SoundStats]) -> None:
    print(f"{'sound':14s} {'loop':5s} {'dur s':>6s} {'peak dBFS':>10s} {'RMS dBFS':>9s} "
          f"{'centroid Hz':>12s} {'seam jump/step':>15s}")
    for s in stats:
        seam = f"{s.seam.jump_over_step:.3f}" if s.seam is not None else "-"
        print(f"{s.name:14s} {'yes' if s.loop else 'no':5s} {s.duration:6.2f} {s.peak_dbfs:10.2f} "
              f"{s.rms_dbfs:9.2f} {s.centroid_hz:12.0f} {seam:>15s}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dragon_ball_powers.audio",
                                     description="Export, analyse or play the procedural sounds.")
    parser.add_argument("--export", type=Path, metavar="DIR", help="write every sound as a WAV file")
    parser.add_argument("--stats", action="store_true", help="print the level/seam table")
    parser.add_argument("--play", nargs="+", metavar="NAME", help="play sounds (needs an audio device)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not (args.export or args.stats or args.play):
        args.stats = True
    if args.export:
        _print_table(export_sounds(args.export, seed=args.seed))
        print(f"WAV files written to {args.export}")
    elif args.stats:
        _print_table([analyze(n, synthesize(n, seed=args.seed)) for n in SOUND_NAMES])
    if args.play:
        engine = AudioEngine(AudioSettings(enabled=True, volume=0.8), seed=args.seed)
        if not engine.start() or not engine.wait_ready(10.0):
            print("No audio device available.")
            return 1
        for name in args.play:
            if name in LOOP_SOUNDS:
                engine.loop(name, 1.0)
                time.sleep(SOUND_DURATIONS.get(name, 2.0) * 1.5)
                engine.stop(name, 300)
                time.sleep(0.4)
            else:
                engine.play(name)
                time.sleep(SOUND_DURATIONS.get(name, 1.0) + 0.2)
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
