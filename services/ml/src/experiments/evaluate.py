"""Metric computation and result-table reporting for the baseline experiments.

Each model run produces a ``ResultRow`` capturing identification (feature set,
model, target, split), basic split metadata, and the relevant metrics. Rows
are aggregated into a wide table that is written as both CSV and JSON, plus a
markdown summary that surfaces the dissertation-relevant comparison: do twin
features outperform simpler baselines for `final_grade` and `passed`?
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ResultRow:
    feature_set: str
    model: str
    target: str
    task: str  # "classification" or "regression"
    split_strategy: str
    split_metadata: dict[str, Any]
    n_train_rows: int
    n_test_rows: int
    n_train_students: int
    n_test_students: int
    metrics: dict[str, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        flat = asdict(self)
        flat.pop("metrics")
        flat.pop("split_metadata")
        flat.update({f"metric_{key}": value for key, value in self.metrics.items()})
        flat["split_metadata_json"] = json.dumps(self.split_metadata, sort_keys=True, default=str)
        return flat


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def results_to_dataframe(rows: list[ResultRow]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([row.to_record() for row in rows])


def write_result_artifacts(
    rows: list[ResultRow],
    *,
    output_dir: Path,
    run_name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    table = results_to_dataframe(rows)
    csv_path = output_dir / f"{run_name}_results.csv"
    json_path = output_dir / f"{run_name}_results.json"
    md_path = output_dir / f"{run_name}_summary.md"

    table.to_csv(csv_path, index=False)

    json_payload: dict[str, Any] = {
        "run_name": run_name,
        "rows": [
            {**asdict(row), "metrics": row.metrics, "split_metadata": row.split_metadata}
            for row in rows
        ],
    }
    if extra_metadata:
        json_payload["metadata"] = extra_metadata
    json_path.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")

    md_path.write_text(_render_markdown_summary(rows, run_name, extra_metadata), encoding="utf-8")

    return {"csv": csv_path, "json": json_path, "markdown": md_path}


def _render_markdown_summary(
    rows: list[ResultRow],
    run_name: str,
    metadata: dict[str, Any] | None,
) -> str:
    if not rows:
        return f"# {run_name}\n\nNo results were produced.\n"

    feature_set_names = sorted({row.feature_set for row in rows})
    feature_set_text = ", ".join(f"`{name}`" for name in feature_set_names)
    lines: list[str] = [
        f"# Experiment result summary: `{run_name}`",
        "",
        (
            f"This report compares the configured feature sets ({feature_set_text}) "
            "for the dissertation's current experimental targets: `final_grade` "
            "(regression) and, when configured, `passed` (classification). The "
            "teacher-facing heuristic `risk_level` is intentionally NOT used as "
            "a supervised target."
        ),
        "",
    ]

    if metadata:
        lines.extend(["## Run metadata", ""])
        for key, value in metadata.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    classification_rows = [row for row in rows if row.task == "classification"]
    regression_rows = [row for row in rows if row.task == "regression"]

    if classification_rows:
        lines.extend(_render_classification_table(classification_rows))
    if regression_rows:
        lines.extend(_render_regression_table(regression_rows))

    lines.extend(_render_feature_set_comparison(rows))
    return "\n".join(lines) + "\n"


def _render_classification_table(rows: list[ResultRow]) -> list[str]:
    header = (
        "| split | feature set | model | n_train | n_test |"
        " accuracy | precision | recall | f1 | roc_auc |"
    )
    separator = "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    row_template = (
        "| {split} | {fs} | {model} | {nt} | {nv} |"
        " {acc:.3f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {auc} |"
    )

    out = ["## Classification results — target `passed`", "", header, separator]
    for row in rows:
        out.append(
            row_template.format(
                split=row.split_strategy,
                fs=row.feature_set,
                model=row.model,
                nt=row.n_train_rows,
                nv=row.n_test_rows,
                acc=row.metrics.get("accuracy", float("nan")),
                prec=row.metrics.get("precision", float("nan")),
                rec=row.metrics.get("recall", float("nan")),
                f1=row.metrics.get("f1", float("nan")),
                auc=_format_optional(row.metrics.get("roc_auc")),
            )
        )
    out.append("")
    return out


def _render_regression_table(rows: list[ResultRow]) -> list[str]:
    out = [
        "## Regression results — target `final_grade`",
        "",
        "| split | feature set | model | n_train | n_test | MAE | RMSE | R^2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        out.append(
            "| {split} | {fs} | {model} | {nt} | {nv} | {mae:.3f} | {rmse:.3f} | {r2:.3f} |".format(
                split=row.split_strategy,
                fs=row.feature_set,
                model=row.model,
                nt=row.n_train_rows,
                nv=row.n_test_rows,
                mae=row.metrics.get("mae", float("nan")),
                rmse=row.metrics.get("rmse", float("nan")),
                r2=row.metrics.get("r2", float("nan")),
            )
        )
    out.append("")
    return out


def _render_feature_set_comparison(rows: list[ResultRow]) -> list[str]:
    if not rows:
        return []
    table = results_to_dataframe(rows)
    if table.empty:
        return []

    out = [
        "## Headline comparison: do Digital Twin features help?",
        "",
        "Best metric per (target, split, feature set), aggregated across models.",
        "",
    ]

    grouped = table.groupby(["task", "split_strategy", "feature_set"], dropna=False)
    headline: list[dict[str, Any]] = []
    for (task, split_strategy, feature_set), group in grouped:
        if task == "classification":
            metric_key = "metric_f1"
            metric_name = "f1"
        else:
            metric_key = "metric_rmse"
            metric_name = "rmse"
        if metric_key not in group.columns:
            continue
        if task == "classification":
            best = group[metric_key].max()
        else:
            best = group[metric_key].min()
        headline.append(
            {
                "task": task,
                "split": split_strategy,
                "feature_set": feature_set,
                "metric": metric_name,
                "best_value": best,
            }
        )

    if not headline:
        return out + [""]

    headline_df = pd.DataFrame(headline)
    out.append("| task | split | feature set | metric | best value |")
    out.append("| --- | --- | --- | --- | ---: |")
    for record in headline_df.to_dict(orient="records"):
        out.append(
            "| {task} | {split} | {fs} | {metric} | {value:.3f} |".format(
                task=record["task"],
                split=record["split"],
                fs=record["feature_set"],
                metric=record["metric"],
                value=record["best_value"],
            )
        )
    out.append("")
    return out


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and np.isnan(value):
        return "n/a"
    return f"{value:.3f}"
