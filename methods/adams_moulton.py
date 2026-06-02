"""Third-order Adams-Moulton method with RK4 startup."""

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
        raise ValueError("Adams-Moulton 3 requires RK4 startup points.")

    w, function_evaluations = bootstrap_with_rk4(f, t, y0, h, start_steps=3)
    f_values, evals = initial_function_values(f, t, w, through_index=3)
    function_evaluations += evals

    for i in range(3, len(t) - 1):
        f_i = require_value(f_values, i)
        f_i_1 = require_value(f_values, i - 1)
        f_i_2 = require_value(f_values, i - 2)
        f_i_3 = require_value(f_values, i - 3)

        predictor = w[i] + h * (55.0 * f_i - 59.0 * f_i_1 + 37.0 * f_i_2 - 9.0 * f_i_3) / 24.0
        w[i + 1], evals = _solve_implicit_step(
            f=f,
            problem=problem,
            t_next=t[i + 1],
            w_i=w[i],
            h=h,
            f_i=f_i,
            f_i_1=f_i_1,
            f_i_2=f_i_2,
            initial_guess=predictor,
        )
        function_evaluations += evals

        if i + 1 < len(t) - 1:
            f_values[i + 1] = f(t[i + 1], w[i + 1])
            function_evaluations += 1

    return make_result("Adams-Moulton 3", h, t, w, function_evaluations)


def _solve_implicit_step(
    f: OdeFunction,
    problem,
    t_next: float,
    w_i: float,
    h: float,
    f_i: float,
    f_i_1: float,
    f_i_2: float,
    initial_guess: float,
) -> tuple[float, int]:
    if problem is not None and problem.linear_coeff is not None:
        a_next, b_next = problem.linear_coeff(t_next)
        rhs = w_i + h * (9.0 * b_next + 19.0 * f_i - 5.0 * f_i_1 + f_i_2) / 24.0
        denominator = 1.0 - 9.0 * h * a_next / 24.0
        if abs(denominator) < 1e-14:
            raise ZeroDivisionError("Adams-Moulton implicit denominator is too close to zero.")
        return rhs / denominator, 0

    constant = w_i + h * (19.0 * f_i - 5.0 * f_i_1 + f_i_2) / 24.0
    coefficient = 9.0 * h / 24.0
    current = max(float(initial_guess), 1e-14)
    evals = 0

    for _ in range(100):
        next_value = constant + coefficient * f(t_next, current)
        evals += 1
        next_value = max(float(next_value), 1e-14)
        if abs(next_value - current) <= 1e-12 * max(1.0, abs(next_value)):
            return next_value, evals
        current = next_value

    raise RuntimeError("Adams-Moulton fixed-point iteration failed to converge.")
