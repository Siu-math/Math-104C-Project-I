"""Shared utilities for numerical ODE methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


OdeFunction = Callable[[float, float], float]


def build_grid(t0: float, t_end: float, h: float) -> np.ndarray:
    """Build an equally spaced mesh and force the final point to t_end."""
    if h <= 0:
        raise ValueError("Step size h must be positive.")

    raw_steps = (t_end - t0) / h
    steps = int(round(raw_steps))
    if not np.isclose(raw_steps, steps, rtol=1e-10, atol=1e-10):
        raise ValueError(f"h={h} does not divide [{t0}, {t_end}] evenly.")

    t = t0 + h * np.arange(steps + 1, dtype=float)
    t[-1] = t_end
    return t


def make_result(
    method: str,
    h: float,
    t: np.ndarray,
    numerical: np.ndarray,
    function_evaluations: int,
) -> dict:
    return {
        "method": method,
        "h": h,
        "t": t,
        "numerical": numerical,
        "function_evaluations": function_evaluations,
    }


def rk4_step(f: OdeFunction, t_i: float, w_i: float, h: float) -> tuple[float, int]:
    k1 = f(t_i, w_i)
    k2 = f(t_i + h / 2.0, w_i + h * k1 / 2.0)
    k3 = f(t_i + h / 2.0, w_i + h * k2 / 2.0)
    k4 = f(t_i + h, w_i + h * k3)
    return w_i + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0, 4


def bootstrap_with_rk4(
    f: OdeFunction,
    t: np.ndarray,
    y0: float,
    h: float,
    start_steps: int,
) -> tuple[np.ndarray, int]:
    """Use RK4 to fill w_0 through w_start_steps."""
    if len(t) - 1 < start_steps:
        raise ValueError("Not enough mesh points for RK4 bootstrap.")

    w = np.zeros_like(t, dtype=float)
    w[0] = y0
    function_evaluations = 0
    for i in range(start_steps):
        w[i + 1], evals = rk4_step(f, t[i], w[i], h)
        function_evaluations += evals
    return w, function_evaluations


def initial_function_values(
    f: OdeFunction,
    t: np.ndarray,
    w: np.ndarray,
    through_index: int,
) -> tuple[list[float | None], int]:
    values: list[float | None] = [None] * len(t)
    evals = 0
    for i in range(through_index + 1):
        values[i] = f(t[i], w[i])
        evals += 1
    return values, evals


def require_value(values: list[float | None], index: int) -> float:
    value = values[index]
    if value is None:
        raise RuntimeError(f"Missing cached function value at index {index}.")
    return value
