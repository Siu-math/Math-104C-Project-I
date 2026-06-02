"""Second-order midpoint method."""

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
        k_mid = f(t[i] + h / 2.0, w[i] + h * k1 / 2.0)
        w[i + 1] = w[i] + h * k_mid
        function_evaluations += 2

    return make_result("Midpoint", h, t, w, function_evaluations)
