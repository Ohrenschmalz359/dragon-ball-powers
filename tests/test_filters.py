"""Tests for the vectorised One Euro filter (no camera / MediaPipe needed)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dragon_ball_powers.config import TrackingSettings
from dragon_ball_powers.filters import OneEuroFilter, smoothing_factor

FPS = 30.0
DT = 1.0 / FPS


def run(f: OneEuroFilter, samples: np.ndarray, t0: float = 0.0, dt: float = DT) -> np.ndarray:
    """Feed ``samples[i]`` at ``t0 + i * dt`` and stack the outputs."""
    return np.stack([f(samples[i], t0 + i * dt) for i in range(len(samples))])


def test_smoothing_factor_formula():
    cutoff, dt = 2.0, 0.02
    tau = 1.0 / (2.0 * math.pi * cutoff)
    assert smoothing_factor(cutoff, dt) == pytest.approx(1.0 / (1.0 + tau / dt))
    # vectorised over cutoff arrays, monotonic in the cutoff
    a = smoothing_factor(np.array([0.5, 1.0, 5.0, 50.0]), DT)
    assert np.all(np.diff(a) > 0) and np.all((a > 0) & (a < 1))


def test_first_sample_passes_through_and_returns_float64_copy():
    f = OneEuroFilter(1.0, 0.0, 1.0)
    x = np.arange(63, dtype=np.float32).reshape(21, 3)
    y = f(x, 10.0)
    assert y.dtype == np.float64 and y.shape == (21, 3)
    np.testing.assert_array_equal(y, x)
    assert f.initialized and f.last_time == 10.0
    np.testing.assert_array_equal(f.derivative, np.zeros((21, 3)))
    # mutating input or output must not corrupt the filter state
    x[:] = -1.0
    y[:] = -2.0
    np.testing.assert_array_equal(f.value, np.arange(63, dtype=np.float64).reshape(21, 3))


def test_constant_signal_converges():
    f = OneEuroFilter(1.0, 0.5, 1.0)
    f(np.zeros(3), 0.0)                  # start far away from the target
    target = np.array([1.0, -2.0, 0.25])
    out = run(f, np.tile(target, (90, 1)), t0=DT)  # 3 s of a constant signal
    np.testing.assert_allclose(out[-1], target, atol=1e-3)
    # monotonic approach (exponential smoothing never overshoots a step)
    err = np.abs(out - target).max(axis=1)
    assert np.all(np.diff(err) <= 1e-12)
    # a constant signal from the start is reproduced exactly
    g = OneEuroFilter(1.5, 8.0, 1.0)
    out = run(g, np.full((30, 21, 3), 0.42))
    np.testing.assert_allclose(out, 0.42, atol=1e-12)


def test_noisy_stationary_signal_std_reduced_more_than_3x():
    rng = np.random.default_rng(7)
    noisy = 0.5 + rng.normal(0.0, 0.004, size=(300, 21, 3))
    f = OneEuroFilter(min_cutoff=0.5, beta=0.5, d_cutoff=1.0)
    out = run(f, noisy)
    ratio = noisy[60:].std() / out[60:].std()   # skip the settling phase
    assert ratio > 3.0, ratio
    assert abs(out[60:].mean() - 0.5) < 2e-3   # unbiased


def test_default_tracking_params_reduce_landmark_jitter():
    """The shipped smoothing settings still remove a good part of MediaPipe-like jitter."""
    s = TrackingSettings()
    rng = np.random.default_rng(3)
    noisy = 0.5 + rng.normal(0.0, 0.003, size=(300, 21, 3))
    f = OneEuroFilter(s.smoothing_min_cutoff, s.smoothing_beta, s.smoothing_d_cutoff)
    out = run(f, noisy)
    ratio = noisy[60:].std() / out[60:].std()
    assert ratio > 2.0, ratio   # measured about 2.5x at 30 Hz


@pytest.mark.parametrize("beta", [0.0, 1.0, 10.0])
def test_ramp_lag_is_bounded(beta: float):
    v, min_cutoff = 0.5, 1.0
    f = OneEuroFilter(min_cutoff, beta, 1.0)
    times = np.arange(90) * DT
    out = run(f, (v * times)[:, None])
    lag = v * times[-1] - out[-1, 0]
    # an exponential low-pass lags a ramp by v * tau at most (tau of min_cutoff)
    tau = 1.0 / (2.0 * math.pi * min_cutoff)
    assert 0.0 <= lag <= v * tau * 1.01


def test_ramp_lag_lower_with_higher_beta():
    v = 0.5
    times = np.arange(90) * DT
    lags = []
    for beta in (0.0, 1.0, 10.0):
        f = OneEuroFilter(1.0, beta, 1.0)
        out = run(f, (v * times)[:, None])
        lags.append(v * times[-1] - out[-1, 0])
    assert lags[0] > lags[1] > lags[2]
    assert lags[2] < lags[0] / 4.0


def test_derivative_matches_true_slope():
    slope = np.array([[0.3, -0.8, 0.05]] * 21)
    f = OneEuroFilter(1.5, 8.0, 1.0)
    times = np.arange(60) * DT
    run(f, slope[None] * times[:, None, None] + 0.2)
    np.testing.assert_allclose(f.derivative, slope, rtol=0.02, atol=1e-6)


def test_derivative_with_irregular_intervals():
    rng = np.random.default_rng(11)
    dts = rng.uniform(0.02, 0.07, size=80)
    times = np.cumsum(dts)
    f = OneEuroFilter(1.0, 2.0, 1.0)
    for t in times:
        f(np.array([2.0 * t, -1.0 * t]), t)
    np.testing.assert_allclose(f.derivative, [2.0, -1.0], rtol=0.02)


def test_dt_not_positive_returns_previous_output():
    f = OneEuroFilter(1.0, 0.0, 1.0)
    f(np.array([0.0, 0.0]), 1.0)
    y1 = f(np.array([1.0, 1.0]), 1.1)
    deriv = f.derivative
    same_time = f(np.array([50.0, 50.0]), 1.1)
    earlier = f(np.array([-50.0, -50.0]), 0.5)
    nan_time = f(np.array([-50.0, -50.0]), float("nan"))
    for y in (same_time, earlier, nan_time):
        np.testing.assert_array_equal(y, y1)
    np.testing.assert_array_equal(f.derivative, deriv)   # state untouched
    assert f.last_time == pytest.approx(1.1)


def test_infinite_timestamp_is_ignored_and_does_not_freeze_the_filter():
    """Regression: t = inf used to be accepted, after which every later sample was
    "earlier" (dt = -inf) and the output stayed frozen forever."""
    f = OneEuroFilter(1.0, 0.0, 1.0)
    f(np.array([0.0]), 0.0)
    y = f(np.array([1.0]), 0.1)
    for bad in (float("inf"), float("-inf"), float("nan")):
        np.testing.assert_array_equal(f(np.array([9.0]), bad), y)
    assert f.last_time == pytest.approx(0.1)
    out = [float(f(np.array([5.0]), 0.1 + k * DT)[0]) for k in range(1, 30)]
    assert y[0] < out[0] < out[-1] < 5.0          # filtering goes on normally
    assert out[-1] > 4.9                          # and converges to the new value


def test_reset_makes_next_sample_pass_through():
    f = OneEuroFilter(0.5, 0.0, 1.0)
    run(f, np.zeros((20, 21, 3)))
    f.reset()
    assert not f.initialized and f.value is None and f.derivative is None
    x = np.full((21, 3), 3.0)
    np.testing.assert_array_equal(f(x, 0.0), x)   # earlier timestamps are fine after reset


def test_vectorised_equals_independent_scalar_filters():
    rng = np.random.default_rng(5)
    data = np.cumsum(rng.normal(0.0, 0.02, size=(40, 21, 3)), axis=0)
    times = np.cumsum(rng.uniform(0.02, 0.05, size=40))
    vec = OneEuroFilter(1.2, 4.0, 1.0)
    out = np.stack([vec(data[i], times[i]) for i in range(40)])
    assert out.shape == (40, 21, 3)
    for j, k in [(0, 0), (8, 1), (20, 2), (12, 0)]:
        scalar = OneEuroFilter(1.2, 4.0, 1.0)
        ref = np.array([scalar(data[i, j, k], times[i]) for i in range(40)])
        np.testing.assert_allclose(out[:, j, k], ref, rtol=0, atol=1e-12)


def test_scalar_and_other_shapes():
    f = OneEuroFilter()
    assert f(1.0, 0.0).shape == ()
    g = OneEuroFilter()
    assert g(np.zeros((33, 3)), 0.0).shape == (33, 3)
    assert g(np.ones((33, 3)), DT).shape == (33, 3)


def test_shape_change_restarts():
    f = OneEuroFilter(0.5, 0.0, 1.0)
    run(f, np.zeros((10, 21, 3)))
    x = np.ones((33, 3))
    np.testing.assert_array_equal(f(x, 1.0), x)
    np.testing.assert_array_equal(f.derivative, np.zeros((33, 3)))


def test_non_finite_samples_do_not_poison_state():
    f = OneEuroFilter(1.0, 0.0, 1.0)
    f(np.array([1.0, 2.0]), 0.0)
    y = f(np.array([np.nan, 2.0]), DT)
    assert np.isfinite(y).all()
    assert y[0] == pytest.approx(1.0)
    y = f(np.array([1.0, np.inf]), 2 * DT)
    assert np.isfinite(y).all()
    assert np.isfinite(f.derivative).all()
    # a non-finite first sample passes through but does not initialise the filter
    g = OneEuroFilter()
    g(np.array([np.nan]), 0.0)
    assert not g.initialized


def test_set_params_and_validation():
    f = OneEuroFilter(1.0, 0.0, 1.0)
    f.set_params(beta=5.0)
    assert (f.min_cutoff, f.beta, f.d_cutoff) == (1.0, 5.0, 1.0)
    f.set_params(min_cutoff=2.0, d_cutoff=3.0)
    assert (f.min_cutoff, f.beta, f.d_cutoff) == (2.0, 5.0, 3.0)
    for bad in ({"min_cutoff": 0.0}, {"min_cutoff": -1.0}, {"beta": -0.1},
                {"d_cutoff": 0.0}, {"beta": float("nan")}):
        with pytest.raises(ValueError):
            f.set_params(**bad)
    with pytest.raises(ValueError):
        OneEuroFilter(min_cutoff=0.0)


def test_set_params_keeps_state_and_changes_response():
    times = np.arange(60) * DT
    ramp = (0.5 * times)[:, None]
    slow = OneEuroFilter(1.0, 0.0, 1.0)
    out_slow = run(slow, ramp)
    fast = OneEuroFilter(1.0, 0.0, 1.0)
    run(fast, ramp[:30])
    fast.set_params(min_cutoff=10.0)   # mid-stream change: no reset, just less smoothing
    out_fast = np.stack([fast(ramp[i], times[i]) for i in range(30, 60)])
    assert fast.initialized
    assert (0.5 * times[-1] - out_fast[-1, 0]) < (0.5 * times[-1] - out_slow[-1, 0]) / 3
