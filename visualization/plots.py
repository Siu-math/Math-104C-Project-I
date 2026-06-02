"""Plot exact/numerical solution curves and absolute error curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .tables import safe_name


LINESTYLES = {
    0.2: "--",
    0.1: "-.",
    0.05: ":",
}


def save_problem_plots(results, output_dir: Path, overview_h: float = 0.1) -> None:
    if not results:
        return

    problem = results[0].problem
    problem_key = safe_name(problem.name)
    figure_dir = output_dir / "figures" / problem_key
    figure_dir.mkdir(parents=True, exist_ok=True)

    overview_results = [result for result in results if np.isclose(result.h, overview_h)]
    _save_solution_plot(overview_results, figure_dir / f"{problem_key}_solutions_h_{_h_key(overview_h)}.png")
    _save_error_plot(overview_results, figure_dir / f"{problem_key}_absolute_errors_h_{_h_key(overview_h)}.png")
    _save_method_error_plots(results, figure_dir / "methods")


def _save_solution_plot(results, path: Path) -> None:
    problem = results[0].problem
    fig, ax = plt.subplots(figsize=(12, 7))

    longest_t = max((result.raw["t"] for result in results), key=len)
    exact = [problem.exact_solution(t_i) for t_i in longest_t]
    ax.plot(longest_t, exact, color="black", linewidth=2.5, label="exact solution")

    colors = _method_colors(results)
    for result in results:
        method = result.method
        h = result.h
        label = f"{method}, h={h:g}"
        ax.plot(
            result.raw["t"],
            result.raw["numerical"],
            color=colors[method],
            linestyle=LINESTYLES.get(h, "-"),
            linewidth=1.3,
            alpha=0.82,
            label=label,
        )

    ax.set_title(f"{problem.name}: Exact Solution vs Numerical Approximation")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_error_plot(results, path: Path) -> None:
    problem = results[0].problem
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = _method_colors(results)
    for result in results:
        method = result.method
        h = result.h
        t = result.raw["t"]
        exact = np.array([problem.exact_solution(t_i) for t_i in t], dtype=float)
        error = np.abs(exact - result.raw["numerical"])
        error_for_plot = np.where(error > 0.0, error, np.nan)
        label = f"{method}, h={h:g}"
        ax.plot(
            t,
            error_for_plot,
            color=colors[method],
            linestyle=LINESTYLES.get(h, "-"),
            linewidth=1.3,
            alpha=0.82,
            label=label,
        )

    ax.set_title(f"{problem.name}: Absolute Error Curves")
    ax.set_xlabel("t")
    ax.set_ylabel("absolute error")
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_method_error_plots(results, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    problem = results[0].problem
    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(result.method, []).append(result)

    for method, method_results in grouped.items():
        method_results = sorted(method_results, key=lambda result: result.h, reverse=True)
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(method_results)))
        for color, result in zip(colors, method_results):
            h = result.h
            t = result.raw["t"]
            exact = np.array([problem.exact_solution(t_i) for t_i in t], dtype=float)
            error = np.abs(exact - result.raw["numerical"])
            error_for_plot = np.where(error > 0.0, error, np.nan)
            ax.plot(
                t,
                error_for_plot,
                color=color,
                linestyle=LINESTYLES.get(h, "-"),
                linewidth=1.8,
                marker="o",
                markersize=3.0,
                label=f"h={h:g}",
            )

        ax.set_title(f"{problem.name}: {method} Absolute Error")
        ax.set_xlabel("t")
        ax.set_ylabel("absolute error")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        fig.savefig(figure_dir / f"{safe_name(method)}_absolute_errors.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def _method_colors(results) -> dict[str, tuple[float, float, float, float]]:
    methods = list(dict.fromkeys(result.method for result in results))
    cmap = plt.get_cmap("tab10")
    return {method: cmap(index % 10) for index, method in enumerate(methods)}


def _h_key(h: float) -> str:
    return str(h).replace(".", "p")
