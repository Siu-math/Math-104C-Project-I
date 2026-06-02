"""Entry point for Project 1 numerical experiments."""

from __future__ import annotations

from pathlib import Path

from definitions import get_problems
from runner import H_VALUES, run_all_experiments
from visualization import save_problem_plots, save_problem_tables


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    problems = get_problems()
    all_results = run_all_experiments(problems, H_VALUES)

    for problem_name, results in all_results.items():
        summary = save_problem_tables(results, OUTPUT_DIR)
        save_problem_plots(results, OUTPUT_DIR)
        print(f"{problem_name}: saved {len(results)} experiment runs.")
        print(summary[["method", "h", "max error", "mean error"]].to_string(index=False))
        print()

    print(f"Outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
