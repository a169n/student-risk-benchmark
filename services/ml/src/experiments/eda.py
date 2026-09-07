"""Experiment-oriented EDA for the modeling dataset.

The point of this module is to inform modeling decisions, not to produce a
general statistics dump. It inspects:

1. feature distributions (numeric summaries),
2. missingness patterns (raw and conditional on the explicit indicators),
3. target distributions (`final_grade`, `passed`),
4. correlations between candidate features and `final_grade`,
5. redundancy / collinearity warnings between major features,
6. per-week target stability (mean `final_grade`, `passed`-rate by week),
7. feature behavior across `trajectory_type` if available,
8. sanity checks for the new missing-data indicator fields.

Outputs are written under ``data/artifacts/eda/`` as a markdown report and a
small set of CSV/JSON tables.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.datasets import (
    CLASSIFICATION_TARGET,
    GROUP_COLUMN,
    REGRESSION_TARGET,
    WEEK_COLUMN,
)
from src.experiments.featuresets import (
    FEATURE_SET_C_TWIN,
    FORBIDDEN_FEATURE_COLUMNS,
)

CORRELATION_RED_FLAG = 0.95
TWIN_TARGET_CORR_RED_FLAG = 0.97


@dataclass
class EDAArtifacts:
    report_path: Path
    summary_json_path: Path
    feature_summary_csv: Path
    target_by_week_csv: Path
    correlation_csv: Path


def _numeric_columns(frame: pd.DataFrame, excluded: set[str]) -> list[str]:
    numeric = []
    for column in frame.columns:
        if column in excluded:
            continue
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric.append(column)
    return numeric


def _feature_summary(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce") if frame[column].dtype != bool else frame[column].astype(int)
        n_total = int(series.shape[0])
        n_missing = int(series.isna().sum())
        records.append(
            {
                "column": column,
                "n_total": n_total,
                "n_missing": n_missing,
                "missing_pct": round(100.0 * n_missing / max(n_total, 1), 3),
                "mean": _safe(series.mean()),
                "std": _safe(series.std()),
                "min": _safe(series.min()),
                "p25": _safe(series.quantile(0.25)),
                "median": _safe(series.median()),
                "p75": _safe(series.quantile(0.75)),
                "max": _safe(series.max()),
            }
        )
    return pd.DataFrame.from_records(records)


def _safe(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return f


def _correlations_with_target(frame: pd.DataFrame, columns: list[str], target: str) -> pd.DataFrame:
    if target not in frame.columns:
        return pd.DataFrame(columns=["column", "pearson_with_final_grade"])
    target_series = pd.to_numeric(frame[target], errors="coerce")
    records: list[dict[str, Any]] = []
    for column in columns:
        if column == target:
            continue
        series = pd.to_numeric(frame[column], errors="coerce") if frame[column].dtype != bool else frame[column].astype(int)
        joined = pd.concat([series, target_series], axis=1).dropna()
        if joined.shape[0] < 5 or joined.iloc[:, 0].nunique() < 2:
            corr = float("nan")
        else:
            corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        records.append({"column": column, "pearson_with_final_grade": round(corr, 4)})
    return (
        pd.DataFrame.from_records(records)
        .sort_values("pearson_with_final_grade", ascending=False, key=lambda s: s.abs())
        .reset_index(drop=True)
    )


def _redundancy_warnings(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if len(columns) < 2:
        return []
    numeric_frame = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric_frame.corr().abs()
    warnings: list[dict[str, Any]] = []
    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            value = corr_matrix.loc[left, right]
            if pd.isna(value):
                continue
            if value >= CORRELATION_RED_FLAG:
                warnings.append(
                    {"left": left, "right": right, "abs_pearson": round(float(value), 4)}
                )
    return warnings


def _target_by_week(frame: pd.DataFrame) -> pd.DataFrame:
    if WEEK_COLUMN not in frame.columns:
        return pd.DataFrame()
    grouped = frame.groupby(WEEK_COLUMN)
    rows: list[dict[str, Any]] = []
    for week, group in grouped:
        rows.append(
            {
                "week_number": int(week),
                "n_rows": int(group.shape[0]),
                "n_students": int(group[GROUP_COLUMN].nunique()) if GROUP_COLUMN in group.columns else None,
                "mean_final_grade": _safe(group[REGRESSION_TARGET].mean()) if REGRESSION_TARGET in group.columns else float("nan"),
                "passed_rate": _safe(group[CLASSIFICATION_TARGET].astype(float).mean()) if CLASSIFICATION_TARGET in group.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("week_number").reset_index(drop=True)


def _trajectory_breakdown(
    modeling_frame: pd.DataFrame, students_frame: pd.DataFrame | None
) -> pd.DataFrame:
    if students_frame is None or "trajectory_type" not in students_frame.columns:
        return pd.DataFrame()
    merged = modeling_frame.merge(
        students_frame[["student_id", "trajectory_type"]], on="student_id", how="left"
    )
    rows: list[dict[str, Any]] = []
    for trajectory, group in merged.groupby("trajectory_type", dropna=True):
        rows.append(
            {
                "trajectory_type": str(trajectory),
                "n_rows": int(group.shape[0]),
                "n_students": int(group[GROUP_COLUMN].nunique()),
                "mean_final_grade": _safe(group[REGRESSION_TARGET].mean()),
                "passed_rate": _safe(group[CLASSIFICATION_TARGET].astype(float).mean()),
                "mean_risk_score": _safe(pd.to_numeric(group.get("risk_score"), errors="coerce").mean()) if "risk_score" in group.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_final_grade", ascending=False).reset_index(drop=True)


def _indicator_sanity(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Cross-check the explicit ``has_*_to_date`` flags against actual NaNs."""

    pairs = [
        ("avg_assignment_score_to_date", "has_assignment_score_to_date"),
        ("avg_quiz_score_to_date", "has_quiz_score_to_date"),
    ]
    out: list[dict[str, Any]] = []
    for value_col, flag_col in pairs:
        if value_col not in frame.columns or flag_col not in frame.columns:
            continue
        flag_int = frame[flag_col].astype(bool)
        value_present = pd.to_numeric(frame[value_col], errors="coerce").notna()
        agree = int((flag_int == value_present).sum())
        disagree = int((flag_int != value_present).sum())
        out.append(
            {
                "value_column": value_col,
                "flag_column": flag_col,
                "rows_in_agreement": agree,
                "rows_in_disagreement": disagree,
                "indicator_true_pct": round(100.0 * float(flag_int.mean()), 3),
            }
        )
    return out


def _twin_target_red_flags(correlations: pd.DataFrame) -> list[dict[str, Any]]:
    flagged = correlations.loc[
        correlations["pearson_with_final_grade"].abs() >= TWIN_TARGET_CORR_RED_FLAG
    ]
    return flagged.to_dict(orient="records")


def run_eda(
    modeling_frame: pd.DataFrame,
    *,
    output_dir: Path,
    correlation_top_k: int = 12,
    students_frame: pd.DataFrame | None = None,
) -> EDAArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    excluded_for_summary = set(FORBIDDEN_FEATURE_COLUMNS) | {GROUP_COLUMN, "course_id"}
    summary_columns = _numeric_columns(modeling_frame, excluded=excluded_for_summary)
    feature_summary = _feature_summary(modeling_frame, summary_columns)
    feature_summary_csv = output_dir / "feature_summary.csv"
    feature_summary.to_csv(feature_summary_csv, index=False)

    correlations = _correlations_with_target(
        modeling_frame, summary_columns, target=REGRESSION_TARGET
    )
    correlation_csv = output_dir / "correlations_with_final_grade.csv"
    correlations.to_csv(correlation_csv, index=False)

    target_by_week = _target_by_week(modeling_frame)
    target_by_week_csv = output_dir / "target_by_week.csv"
    target_by_week.to_csv(target_by_week_csv, index=False)

    twin_columns = [
        column for column in FEATURE_SET_C_TWIN.columns if column in modeling_frame.columns
    ]
    redundancy = _redundancy_warnings(modeling_frame, twin_columns)
    twin_red_flags = _twin_target_red_flags(correlations)
    indicator_sanity = _indicator_sanity(modeling_frame)
    trajectory_breakdown = _trajectory_breakdown(modeling_frame, students_frame)

    summary_payload: dict[str, Any] = {
        "n_rows": int(modeling_frame.shape[0]),
        "n_students": int(modeling_frame[GROUP_COLUMN].nunique()) if GROUP_COLUMN in modeling_frame.columns else None,
        "weeks_present": sorted(int(value) for value in modeling_frame[WEEK_COLUMN].dropna().unique().tolist()) if WEEK_COLUMN in modeling_frame.columns else [],
        "regression_target_summary": {
            "mean": _safe(modeling_frame[REGRESSION_TARGET].mean()),
            "std": _safe(modeling_frame[REGRESSION_TARGET].std()),
            "min": _safe(modeling_frame[REGRESSION_TARGET].min()),
            "max": _safe(modeling_frame[REGRESSION_TARGET].max()),
        },
        "classification_target_summary": {
            "passed_rate": _safe(modeling_frame[CLASSIFICATION_TARGET].astype(float).mean())
            if CLASSIFICATION_TARGET in modeling_frame.columns
            else float("nan")
        },
        "redundancy_warnings": redundancy,
        "indicator_sanity": indicator_sanity,
        "twin_target_red_flags": twin_red_flags,
        "trajectory_breakdown": trajectory_breakdown.to_dict(orient="records") if not trajectory_breakdown.empty else [],
    }
    summary_json_path = output_dir / "eda_summary.json"
    summary_json_path.write_text(
        json.dumps(summary_payload, indent=2, default=str), encoding="utf-8"
    )

    report_path = output_dir / "eda_report.md"
    report_path.write_text(
        _render_eda_markdown(
            modeling_frame=modeling_frame,
            feature_summary=feature_summary,
            correlations=correlations.head(correlation_top_k),
            target_by_week=target_by_week,
            redundancy_warnings=redundancy,
            indicator_sanity=indicator_sanity,
            twin_red_flags=twin_red_flags,
            trajectory_breakdown=trajectory_breakdown,
        ),
        encoding="utf-8",
    )

    return EDAArtifacts(
        report_path=report_path,
        summary_json_path=summary_json_path,
        feature_summary_csv=feature_summary_csv,
        target_by_week_csv=target_by_week_csv,
        correlation_csv=correlation_csv,
    )


def _render_eda_markdown(
    *,
    modeling_frame: pd.DataFrame,
    feature_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    target_by_week: pd.DataFrame,
    redundancy_warnings: list[dict[str, Any]],
    indicator_sanity: list[dict[str, Any]],
    twin_red_flags: list[dict[str, Any]],
    trajectory_breakdown: pd.DataFrame,
) -> str:
    lines: list[str] = [
        "# Modeling-readiness EDA report",
        "",
        (
            "This report summarizes the refined v1.3 dataset from the "
            "perspective of building leakage-safe baselines for `final_grade` "
            "(regression) and `passed` (classification). It is not a generic "
            "descriptive dump."
        ),
        "",
        "## Dataset shape",
        "",
        f"- Modeling rows: **{modeling_frame.shape[0]}**",
        f"- Distinct students: **{modeling_frame[GROUP_COLUMN].nunique()}**",
        f"- Weeks present: {sorted(modeling_frame[WEEK_COLUMN].dropna().unique().astype(int).tolist())}",
        "",
        "## Target distributions",
        "",
        f"- `final_grade` mean: **{modeling_frame[REGRESSION_TARGET].mean():.3f}**, "
        f"std: **{modeling_frame[REGRESSION_TARGET].std():.3f}**",
        f"- `passed` positive rate: **{modeling_frame[CLASSIFICATION_TARGET].astype(float).mean():.3f}**",
        "",
        "## Top correlations with `final_grade`",
        "",
        "Sorted by absolute Pearson correlation. High absolute values close to 1.0 "
        "for snapshot features hint at signal that may be too deterministic; the "
        "twin-target red flag list below makes the strong cases explicit.",
        "",
        "| column | pearson |",
        "| --- | ---: |",
    ]
    for record in correlations.to_dict(orient="records"):
        lines.append(
            "| {column} | {value:.3f} |".format(
                column=record["column"], value=record["pearson_with_final_grade"]
            )
        )

    lines.extend(
        [
            "",
            "## Per-week target stability",
            "",
            "Snapshot rows can come from any week. This table shows whether the "
            "target value seen in each week is similar enough to support "
            "training across weeks rather than per-week.",
            "",
            "| week | n_rows | n_students | mean `final_grade` | `passed` rate |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in target_by_week.to_dict(orient="records"):
        lines.append(
            "| {week} | {n_rows} | {n_students} | {mfg:.3f} | {pr:.3f} |".format(
                week=record["week_number"],
                n_rows=record["n_rows"],
                n_students=record.get("n_students") or 0,
                mfg=record["mean_final_grade"],
                pr=record["passed_rate"],
            )
        )

    if redundancy_warnings:
        lines.extend(
            [
                "",
                "## Redundancy / collinearity warnings",
                "",
                "Pairs of twin features whose absolute Pearson correlation reaches "
                f"`{CORRELATION_RED_FLAG}` or higher.",
                "",
                "| left | right | |corr| |",
                "| --- | --- | ---: |",
            ]
        )
        for warning in redundancy_warnings:
            lines.append(
                "| {left} | {right} | {value:.3f} |".format(
                    left=warning["left"],
                    right=warning["right"],
                    value=warning["abs_pearson"],
                )
            )
    else:
        lines.extend(
            [
                "",
                "## Redundancy / collinearity warnings",
                "",
                f"No twin feature pair exceeded the `{CORRELATION_RED_FLAG}` threshold.",
            ]
        )

    if twin_red_flags:
        lines.extend(
            [
                "",
                "## Twin-target red flags",
                "",
                f"Snapshot features whose absolute correlation with `final_grade` reaches `{TWIN_TARGET_CORR_RED_FLAG}` or higher are listed here. They may be too deterministic and warrant ablation in later phases.",
                "",
                "| column | pearson |",
                "| --- | ---: |",
            ]
        )
        for record in twin_red_flags:
            lines.append(
                "| {column} | {value:.3f} |".format(
                    column=record["column"], value=record["pearson_with_final_grade"]
                )
            )
    else:
        lines.extend(
            [
                "",
                "## Twin-target red flags",
                "",
                f"No snapshot feature exceeded the `{TWIN_TARGET_CORR_RED_FLAG}` correlation threshold with `final_grade`.",
            ]
        )

    if indicator_sanity:
        lines.extend(
            [
                "",
                "## Missing-data indicator sanity",
                "",
                "Cross-checks the explicit `has_*_to_date` flags against the "
                "presence of the corresponding numeric value.",
                "",
                "| value column | flag column | agree | disagree | indicator true % |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        for record in indicator_sanity:
            lines.append(
                "| {vc} | {fc} | {agree} | {disagree} | {pct:.2f} |".format(
                    vc=record["value_column"],
                    fc=record["flag_column"],
                    agree=record["rows_in_agreement"],
                    disagree=record["rows_in_disagreement"],
                    pct=record["indicator_true_pct"],
                )
            )

    if not trajectory_breakdown.empty:
        lines.extend(
            [
                "",
                "## Trajectory-type breakdown (generation-only context)",
                "",
                "Aggregates joined from the hidden `trajectory_type` field to show "
                "that the modeling targets behave differently across the four "
                "synthetic trajectories. This field is NOT a feature; it is shown "
                "only to validate the generator.",
                "",
                "| trajectory | n_rows | n_students | mean `final_grade` | `passed` rate | mean `risk_score` |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in trajectory_breakdown.to_dict(orient="records"):
            lines.append(
                "| {trajectory} | {n_rows} | {n_students} | {mfg:.3f} | {pr:.3f} | {mrs} |".format(
                    trajectory=record["trajectory_type"],
                    n_rows=record["n_rows"],
                    n_students=record["n_students"],
                    mfg=record["mean_final_grade"],
                    pr=record["passed_rate"],
                    mrs=f"{record['mean_risk_score']:.3f}" if not (record["mean_risk_score"] != record["mean_risk_score"]) else "n/a",
                )
            )

    lines.extend(
        [
            "",
            "## Feature summary",
            "",
            f"Per-feature numeric summary (only top-level descriptive stats here; the full table is in `{feature_summary.shape[0]}` row CSV).",
            "",
            "See `feature_summary.csv` next to this report for the full table.",
        ]
    )
    return "\n".join(lines) + "\n"
