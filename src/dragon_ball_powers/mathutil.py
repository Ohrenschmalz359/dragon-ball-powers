"""Small, dependency-free math helpers shared across modules.

2D vectors are plain ``(x, y)`` tuples in scene pixel coordinates (x right, y down).
"""

from __future__ import annotations

import math

Vec2 = tuple[float, float]

TAU = math.tau


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def inverse_lerp(a: float, b: float, x: float) -> float:
    if a == b:
        return 0.0
    return (x - a) / (b - a)


def remap(x: float, in_lo: float, in_hi: float, out_lo: float = 0.0, out_hi: float = 1.0,
          clamped: bool = True) -> float:
    t = inverse_lerp(in_lo, in_hi, x)
    if clamped:
        t = clamp(t)
    return lerp(out_lo, out_hi, t)


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0 if x < edge0 else 1.0
    t = clamp((x - edge0) / (edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def damp(current: float, target: float, rate: float, dt: float) -> float:
    """Frame-rate independent exponential approach of ``current`` towards ``target``.

    ``rate`` is in 1/s: after 1/rate seconds about 63 % of the gap is closed.
    """
    if dt <= 0.0:
        return current
    return target + (current - target) * math.exp(-rate * dt)


def damp_vec(current: Vec2, target: Vec2, rate: float, dt: float) -> Vec2:
    k = math.exp(-rate * dt) if dt > 0.0 else 1.0
    return (target[0] + (current[0] - target[0]) * k, target[1] + (current[1] - target[1]) * k)


def ease_out_cubic(t: float) -> float:
    t = clamp(t)
    return 1.0 - (1.0 - t) ** 3


def ease_in_cubic(t: float) -> float:
    t = clamp(t)
    return t * t * t


def ease_in_out_sine(t: float) -> float:
    t = clamp(t)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def ease_out_expo(t: float) -> float:
    t = clamp(t)
    return 1.0 if t >= 1.0 else 1.0 - 2.0 ** (-10.0 * t)


# ------------------------------------------------------------------ vector helpers


def v_add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def v_sub(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def v_scale(a: Vec2, s: float) -> Vec2:
    return (a[0] * s, a[1] * s)


def v_dot(a: Vec2, b: Vec2) -> float:
    return a[0] * b[0] + a[1] * b[1]


def v_cross(a: Vec2, b: Vec2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def v_len(a: Vec2) -> float:
    return math.hypot(a[0], a[1])


def v_dist(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def v_norm(a: Vec2, fallback: Vec2 = (1.0, 0.0)) -> Vec2:
    length = math.hypot(a[0], a[1])
    if length < 1e-9:
        return fallback
    return (a[0] / length, a[1] / length)


def v_lerp(a: Vec2, b: Vec2, t: float) -> Vec2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def v_mid(a: Vec2, b: Vec2) -> Vec2:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def v_perp(a: Vec2) -> Vec2:
    """Rotate 90 degrees counter-clockwise (in a y-down frame this points 'left' of a)."""
    return (-a[1], a[0])


def v_rotate(a: Vec2, angle: float) -> Vec2:
    c, s = math.cos(angle), math.sin(angle)
    return (a[0] * c - a[1] * s, a[0] * s + a[1] * c)


def v_angle(a: Vec2) -> float:
    return math.atan2(a[1], a[0])


def v_from_angle(angle: float, length: float = 1.0) -> Vec2:
    return (math.cos(angle) * length, math.sin(angle) * length)


def hash01(i: int, seed: int = 0) -> float:
    """Deterministic pseudo-random number in [0, 1) for integer inputs."""
    x = (i * 374761393 + seed * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    x ^= x >> 16
    return (x & 0xFFFFFF) / float(0x1000000)
