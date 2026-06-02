"""Save numerical results as CSV tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def safe_name(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def save_dataframe(df: pd.DataFrame, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_without_suffix.with_suffix(".csv"), index=False)


def save_problem_tables(results, output_dir: Path) -> pd.DataFrame:
    summary_rows = []

    for result in results:
        summary_rows.append(result.summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["problem", "method", "h"])
    problem_key = safe_name(results[0].problem.name)

    problem_dir = output_dir / "tables" / problem_key
    report_columns = ["method", "h", "max error", "mean error"]
    report_df = summary_df[report_columns]

    for method, method_df in report_df.groupby("method", sort=True):
        method_key = safe_name(method)
        save_dataframe(method_df.reset_index(drop=True), problem_dir / f"{method_key}_error_analysis")

    return summary_df
