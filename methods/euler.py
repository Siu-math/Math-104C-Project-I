"""Euler method."""

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
        w[i + 1] = w[i] + h * f(t[i], w[i])
        function_evaluations += 1

    return make_result("Euler", h, t, w, function_evaluations)
