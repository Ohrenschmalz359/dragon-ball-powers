"""Temporal filters for noisy tracking data.

:class:`OneEuroFilter` implements the 1€ filter of Casiez, Roussel & Vogel,
*"1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive
Systems"* (CHI 2012). It is an exponential low-pass filter whose cutoff frequency
rises with the (filtered) speed of the signal: a hand held still is smoothed
strongly (no jitter), a fast movement barely (little lag).

The implementation is vectorised: one instance smooths a NumPy array of any shape
(e.g. 21 hand landmarks x 3 coordinates), every element independently, with
float64 internal state. Timestamps are seconds (``time.perf_counter()`` or camera
capture times), so irregular frame intervals are handled exactly.
"""

from __future__ import annotations

import logging
import math

import numpy as np

log = logging.getLogger(__name__)

_TWO_PI = 2.0 * math.pi


def smoothing_factor(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
    """Exponential smoothing factor ``alpha = 1 / (1 + tau / dt)``, ``tau = 1 / (2 pi cutoff)``.

    Args:
        cutoff: cutoff frequency in Hz (scalar or array, > 0).
        dt: sampling interval in seconds (> 0).
    """
    tau = 1.0 / (_TWO_PI * cutoff)
    return 1.0 / (1.0 + tau / dt)


def _checked(name: str, value: float, minimum: float, inclusive: bool) -> float:
    number = float(value)
    ok = math.isfinite(number) and (number >= minimum if inclusive else number > minimum)
    if not ok:
        relation = ">=" if inclusive else ">"
        raise ValueError(
            f"OneEuroFilter {name} must be finite and {relation} {minimum}, got {value!r}")
    return number


class OneEuroFilter:
    """Speed-adaptive low-pass filter (1€ filter), vectorised over arrays of any shape.

    Args:
        min_cutoff: cutoff frequency (Hz) at zero speed. Lower = smoother when still.
        beta: speed coefficient. Higher = less lag during fast movements. Its unit is
            the inverse of the signal's speed unit (here: normalised units per second).
        d_cutoff: cutoff frequency (Hz) of the low-pass filter applied to the derivative.

    Behaviour:
        * The first sample (and the first sample after :meth:`reset`) passes through.
        * A sample whose timestamp is not later than the previous one (``dt <= 0``) or
          not finite returns the previous output and leaves the state untouched.
        * Non-finite input elements are replaced by the previous raw value, so one
          bad sample cannot poison the state.
        * A sample with a different shape restarts the filter.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = 1.0
        self.beta = 0.0
        self.d_cutoff = 1.0
        self.set_params(min_cutoff, beta, d_cutoff)
        self._x_raw: np.ndarray | None = None   # previous raw sample (for the derivative)
        self._x_hat: np.ndarray | None = None   # previous output
        self._dx_hat: np.ndarray | None = None  # previous filtered derivative (units/s)
        self._t: float | None = None

    # ------------------------------------------------------------------ control

    def set_params(self, min_cutoff: float | None = None, beta: float | None = None,
                   d_cutoff: float | None = None) -> None:
        """Change filter parameters (``None`` keeps the current value). State is kept."""
        if min_cutoff is not None:
            self.min_cutoff = _checked("min_cutoff", min_cutoff, 0.0, inclusive=False)
        if beta is not None:
            self.beta = _checked("beta", beta, 0.0, inclusive=True)
        if d_cutoff is not None:
            self.d_cutoff = _checked("d_cutoff", d_cutoff, 0.0, inclusive=False)

    def reset(self) -> None:
        """Forget all history: the next sample passes through unchanged."""
        self._x_raw = None
        self._x_hat = None
        self._dx_hat = None
        self._t = None

    # ------------------------------------------------------------------ queries

    @property
    def initialized(self) -> bool:
        return self._x_hat is not None

    @property
    def value(self) -> np.ndarray | None:
        """The last output (copy), or ``None`` before the first sample."""
        return None if self._x_hat is None else self._x_hat.copy()

    @property
    def derivative(self) -> np.ndarray | None:
        """Filtered rate of change dx/dt in signal units per second (copy).

        Zero after the first sample, ``None`` before it.
        """
        return None if self._dx_hat is None else self._dx_hat.copy()

    @property
    def last_time(self) -> float | None:
        """Timestamp of the last accepted sample."""
        return self._t

    # ---------------------------------------------------------------- filtering

    def __call__(self, x: np.ndarray | float, t: float) -> np.ndarray:
        """Filter sample ``x`` taken at time ``t`` (seconds); returns a new float64 array."""
        sample = np.array(x, dtype=np.float64)  # always a private copy
        t = float(t)
        if self._x_hat is None or sample.shape != self._x_hat.shape:
            return self._start(sample, t)

        dt = t - self._t  # type: ignore[operator]  # _t is set whenever _x_hat is
        # Also rejects NaN / infinite timestamps: accepting t = inf would make every
        # later sample "earlier" and freeze the filter for good.
        if not (dt > 0.0 and math.isfinite(dt)):
            return self._x_hat.copy()

        finite = np.isfinite(sample)
        if not finite.all():
            sample = np.where(finite, sample, self._x_raw)

        # Casiez' reference implementation differentiates the *raw* signal; using the
        # previous filtered value instead would add the filter lag to the speed
        # estimate (about 2x too fast on a ramp at 30 Hz) and over-shoot predictions.
        dx = (sample - self._x_raw) / dt
        a_d = smoothing_factor(self.d_cutoff, dt)
        dx_hat = self._dx_hat + a_d * (dx - self._dx_hat)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = smoothing_factor(cutoff, dt)
        x_hat = self._x_hat + a * (sample - self._x_hat)

        self._x_raw = sample
        self._x_hat = x_hat
        self._dx_hat = dx_hat
        self._t = t
        return x_hat.copy()

    def _start(self, sample: np.ndarray, t: float) -> np.ndarray:
        if self._x_hat is not None:
            log.debug("OneEuroFilter input shape changed %s -> %s; restarting",
                      self._x_hat.shape, sample.shape)
        if not (math.isfinite(t) and np.isfinite(sample).all()):
            # Cannot start from garbage: pass it through but stay uninitialised.
            self.reset()
            return sample
        self._x_raw = sample
        self._x_hat = sample.copy()
        self._dx_hat = np.zeros_like(sample)
        self._t = t
        return sample.copy()
