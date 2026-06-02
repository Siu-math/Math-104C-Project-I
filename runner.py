"""Run numerical experiments and compute error metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from definitions import Problem
from methods import METHODS


H_VALUES = [0.2, 0.1, 0.05]


@dataclass(frozen=True)
class ExperimentResult:
    problem: Problem
    method: str
    h: float
    raw: dict
    detail_table: pd.DataFrame
    summary_row: dict


def run_method(problem: Problem, method_name: str, solve, h: float) -> ExperimentResult:
    raw = solve(problem.f, problem.t0, problem.y0, problem.t_end, h, problem=problem)
    t = raw["t"]
    numerical = raw["numerical"]
    exact = np.array([problem.exact_solution(t_i) for t_i in t], dtype=float)
    absolute_error = np.abs(exact - numerical)

    detail_table = pd.DataFrame(
        {
            "t_i": t,
            "exact solution": exact,
            "numerical approximation": numerical,
            "absolute error": absolute_error,
        }
    )

    summary_row = {
        "problem": problem.name,
        "method": method_name,
        "h": h,
        "final error": float(absolute_error[-1]),
        "max error": float(np.max(absolute_error)),
        "mean error": float(np.mean(absolute_error)),
        "function evaluations": int(raw["function_evaluations"]),
    }

    return ExperimentResult(problem, method_name, h, raw, detail_table, summary_row)


def run_problem(problem: Problem, h_values: list[float] | None = None) -> list[ExperimentResult]:
    h_values = H_VALUES if h_values is None else h_values
    results: list[ExperimentResult] = []

    for h in h_values:
        for method_name, solve in METHODS:
            results.append(run_method(problem, method_name, solve, h))

    return results


def run_all_experiments(
    problems: list[Problem],
    h_values: list[float] | None = None,
) -> dict[str, list[ExperimentResult]]:
    return {problem.name: run_problem(problem, h_values) for problem in problems}
