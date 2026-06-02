"""Heun's third-order Runge-Kutta method."""

from __future__ import annotations

import numpy as np

from .common import OdeFunction, build_grid, make_result


def solve(
    f: OdeFunction,
    t0: float,
    y0: float,
    t_end: float,
    h: float,
    problem=None,
) -> dict:
    t = build_grid(t0, t_end, h)
    w = np.zeros_like(t, dtype=float)
    w[0] = y0

    function_evaluations = 0
    for i in range(len(t) - 1):
        k1 = f(t[i], w[i])
        k2 = f(t[i] + h / 3.0, w[i] + h * k1 / 3.0)
        k3 = f(t[i] + 2.0 * h / 3.0, w[i] + 2.0 * h * k2 / 3.0)
        w[i + 1] = w[i] + h * (k1 + 3.0 * k3) / 4.0
        function_evaluations += 3

    return make_result("Heun Order 3", h, t, w, function_evaluations)
