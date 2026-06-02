"""Initial value problems used in the numerical experiments."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import exp, log
from typing import Optional


@dataclass(frozen=True)
class Problem:
    name: str
    f: Callable[[float, float], float]
    exact_solution: Callable[[float], float]
    t0: float
    y0: float
    t_end: float
    taylor_second_derivative: Callable[[float, float], float]
    linear_coeff: Optional[Callable[[float], tuple[float, float]]]


def problem_a() -> Problem:
    return Problem(
        name="Problem A",
        f=lambda t, y: y - t**2 + 1.0,
        exact_solution=lambda t: (t + 1.0) ** 2 - 0.5 * exp(t),
        t0=0.0,
        y0=0.5,
        t_end=2.0,
        taylor_second_derivative=lambda t, y: y - t**2 + 1.0 - 2.0 * t,
        linear_coeff=lambda t: (1.0, 1.0 - t**2),
    )


def problem_b() -> Problem:
    return Problem(
        name="Problem B",
        f=lambda _t, y: 2.0 * y,
        exact_solution=lambda t: exp(2.0 * t),
        t0=0.0,
        y0=1.0,
        t_end=2.0,
        taylor_second_derivative=lambda _t, y: 4.0 * y,
        linear_coeff=lambda _t: (2.0, 0.0),
    )


def problem_c() -> Problem:
    def f(_t: float, y: float) -> float:
        if y <= 0.0:
            raise ValueError("Gompertz model requires y > 0.")
        return -y * log(y)

    def second_derivative(t: float, y: float) -> float:
        return -(log(y) + 1.0) * f(t, y)

    return Problem(
        name="Problem C",
        f=f,
        exact_solution=lambda t: 0.1 ** exp(-t),
        t0=0.0,
        y0=0.1,
        t_end=3.0,
        taylor_second_derivative=second_derivative,
        linear_coeff=None,
    )


def get_problems() -> list[Problem]:
    return [problem_a(), problem_b(), problem_c()]
