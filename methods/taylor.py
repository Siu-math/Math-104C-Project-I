"""Second-order Taylor method."""

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
    if problem is None or problem.taylor_second_derivative is None:
        raise ValueError("Taylor Order 2 requires problem.taylor_second_derivative.")

    t = build_grid(t0, t_end, h)
    w = np.zeros_like(t, dtype=float)
    w[0] = y0

    function_evaluations = 0
    for i in range(len(t) - 1):
        first = f(t[i], w[i])
        second = problem.taylor_second_derivative(t[i], w[i])
        w[i + 1] = w[i] + h * first + (h**2 / 2.0) * second
        function_evaluations += 1

    return make_result("Taylor Order 2", h, t, w, function_evaluations)
