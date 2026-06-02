"""Fourth-order Adams-Bashforth method with RK4 startup."""

from __future__ import annotations

from .common import (
    OdeFunction,
    bootstrap_with_rk4,
    build_grid,
    initial_function_values,
    make_result,
    require_value,
)


def solve(
    f: OdeFunction,
    t0: float,
    y0: float,
    t_end: float,
    h: float,
    problem=None,
) -> dict:
    t = build_grid(t0, t_end, h)
    if len(t) < 5:
        raise ValueError("Adams-Bashforth 4 requires at least four startup points.")

    w, function_evaluations = bootstrap_with_rk4(f, t, y0, h, start_steps=3)
    f_values, evals = initial_function_values(f, t, w, through_index=3)
    function_evaluations += evals

    for i in range(3, len(t) - 1):
        f_i = require_value(f_values, i)
        f_i_1 = require_value(f_values, i - 1)
        f_i_2 = require_value(f_values, i - 2)
        f_i_3 = require_value(f_values, i - 3)
        w[i + 1] = w[i] + h * (55.0 * f_i - 59.0 * f_i_1 + 37.0 * f_i_2 - 9.0 * f_i_3) / 24.0

        if i + 1 < len(t) - 1:
            f_values[i + 1] = f(t[i + 1], w[i + 1])
            function_evaluations += 1

    return make_result("Adams-Bashforth 4", h, t, w, function_evaluations)
