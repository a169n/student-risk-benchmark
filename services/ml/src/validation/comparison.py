from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re

import pandas as pd

from src.generator.config import REPO_ROOT
from src.validation.realism import RealismAudit, RealismMetrics


TABLE_PATHS = {
    "students": ("raw", "students.csv"),
    "courses": ("raw", "courses.csv"),
    "course_topics": ("raw", "course_topics.csv"),
    "assignments": ("raw", "assignments.csv"),
    "attendance": ("raw", "attendance.csv"),
    "submissions": ("raw", "submissions.csv"),
    "weekly_activity": ("raw", "weekly_activity.csv"),
    "final_results": ("raw", "final_results.csv"),
    "student_twin_snapshots": ("processed", "student_twin_snapshots.csv"),
}


@dataclass(frozen=True)
class ComparisonInput:
    label: str
    root: Path
    metrics: RealismMetrics


@dataclass(frozen=True)
class ComparisonSummary:
    left_label: str
    right_label: str
    left_root: Path
    right_root: Path
    left_metrics: dict[str, object]
    right_metrics: dict[str, object]
    key_metrics: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["left_root"] = str(self.left_root)
        payload["right_root"] = str(self.right_root)
        return payload


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "comparison"


def _read_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_dataset_root(path: Path) -> Path:
    resolved = path.resolve()
    candidates = []
    if resolved.is_file():
        candidates.extend(
            [resolved.parent, resolved.parent.parent, resolved.parent.parent.parent]
        )
    else:
        candidates.extend([resolved, resolved.parent, resolved.parent.parent])

    for candidate in candidates:
        if (candidate / "raw").exists() and (candidate / "processed").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve a generated dataset root from path: {path}"
    )


def _load_datasets(root: Path) -> dict[str, pd.DataFrame]:
    datasets: dict[str, pd.DataFrame] = {}
    for table_name, (folder, filename) in TABLE_PATHS.items():
        csv_path = root / folder / filename
        parquet_path = csv_path.with_suffix(".parquet")
        if csv_path.exists():
            datasets[table_name] = pd.read_csv(csv_path)
        elif parquet_path.exists() and table_name == "student_twin_snapshots":
            datasets[table_name] = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"Missing dataset file for {table_name}: {csv_path}")
    return datasets


def load_comparison_input(path: str | Path, *, label: str | None = None) -> ComparisonInput:
    root = _resolve_dataset_root(Path(path))
    datasets = _load_datasets(root)
    metrics = RealismAudit().run(datasets)
    return ComparisonInput(label=label or root.name, root=root, metrics=metrics)


def _metric_row(metric: str, left: object, right: object, *, delta: object = None) -> dict[str, object]:
    return {"metric": metric, "left": left, "right": right, "delta": delta}


def _float_delta(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return round(float(right) - float(left), 4)


def build_key_metric_rows(left: RealismMetrics, right: RealismMetrics) -> list[dict[str, object]]:
    left_high = left.risk_distribution["shares"]["high"]
    right_high = right.risk_distribution["shares"]["high"]
    left_medium = left.risk_distribution["shares"]["medium"]
    right_medium = right.risk_distribution["shares"]["medium"]
    left_pass = left.passed_rate["pass_rate"]
    right_pass = right.passed_rate["pass_rate"]
    left_grade_mean = left.final_grade_distribution["mean"]
    right_grade_mean = right.final_grade_distribution["mean"]
    left_attendance = left.attendance_distribution["mean"]
    right_attendance = right.attendance_distribution["mean"]
    left_activity = left.activity_distribution["activity_score"]["mean"]
    right_activity = right.activity_distribution["activity_score"]["mean"]
    left_late = left.submission_discipline["late_rate"]
    right_late = right.submission_discipline["late_rate"]
    left_missed = left.submission_discipline["missed_rate"]
    right_missed = right.submission_discipline["missed_rate"]
    left_early_high = left.early_warning["high_risk_share_before_week_5"]
    right_early_high = right.early_warning["high_risk_share_before_week_5"]
    left_declining_final_risk = left.risk_by_trajectory["mean_risk_score"].get("declining")
    right_declining_final_risk = right.risk_by_trajectory["mean_risk_score"].get("declining")
    left_improving_final_risk = left.risk_by_trajectory["mean_risk_score"].get("improving")
    right_improving_final_risk = right.risk_by_trajectory["mean_risk_score"].get("improving")

    return [
        _metric_row(
            "student_twin_snapshots_rows",
            left.row_counts["student_twin_snapshots"],
            right.row_counts["student_twin_snapshots"],
            delta=right.row_counts["student_twin_snapshots"] - left.row_counts["student_twin_snapshots"],
        ),
        _metric_row("risk_high_share", left_high, right_high, delta=_float_delta(left_high, right_high)),
        _metric_row(
            "risk_medium_share", left_medium, right_medium, delta=_float_delta(left_medium, right_medium)
        ),
        _metric_row("pass_rate", left_pass, right_pass, delta=_float_delta(left_pass, right_pass)),
        _metric_row(
            "final_grade_mean", left_grade_mean, right_grade_mean, delta=_float_delta(left_grade_mean, right_grade_mean)
        ),
        _metric_row(
            "attendance_last_week_mean",
            left_attendance,
            right_attendance,
            delta=_float_delta(left_attendance, right_attendance),
        ),
        _metric_row(
            "activity_last_week_mean",
            left_activity,
            right_activity,
            delta=_float_delta(left_activity, right_activity),
        ),
        _metric_row("late_submission_rate", left_late, right_late, delta=_float_delta(left_late, right_late)),
        _metric_row(
            "missed_submission_rate", left_missed, right_missed, delta=_float_delta(left_missed, right_missed)
        ),
        _metric_row(
            "high_risk_share_before_week_5",
            left_early_high,
            right_early_high,
            delta=_float_delta(left_early_high, right_early_high),
        ),
        _metric_row(
            "declining_final_week_risk",
            left_declining_final_risk,
            right_declining_final_risk,
            delta=_float_delta(left_declining_final_risk, right_declining_final_risk),
        ),
        _metric_row(
            "improving_final_week_risk",
            left_improving_final_risk,
            right_improving_final_risk,
            delta=_float_delta(left_improving_final_risk, right_improving_final_risk),
        ),
        _metric_row(
            "trajectory_risk_ordering_valid",
            left.risk_by_trajectory["ordering_valid"],
            right.risk_by_trajectory["ordering_valid"],
        ),
        _metric_row(
            "trajectory_grade_ordering_valid",
            left.outcome_by_trajectory["grade_ordering_valid"],
            right.outcome_by_trajectory["grade_ordering_valid"],
        ),
        _metric_row(
            "trajectory_pass_ordering_valid",
            left.outcome_by_trajectory["pass_ordering_valid"],
            right.outcome_by_trajectory["pass_ordering_valid"],
        ),
    ]


def compare_generated_outputs(
    left: str | Path,
    right: str | Path,
    *,
    left_label: str | None = None,
    right_label: str | None = None,
) -> ComparisonSummary:
    left_input = load_comparison_input(left, label=left_label)
    right_input = load_comparison_input(right, label=right_label)
    return ComparisonSummary(
        left_label=left_input.label,
        right_label=right_input.label,
        left_root=left_input.root,
        right_root=right_input.root,
        left_metrics=left_input.metrics.to_dict(),
        right_metrics=right_input.metrics.to_dict(),
        key_metrics=build_key_metric_rows(left_input.metrics, right_input.metrics),
    )


def format_markdown_report(summary: ComparisonSummary) -> str:
    table_lines = [
        "| Metric | Left | Right | Delta |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary.key_metrics:
        table_lines.append(
            f"| `{row['metric']}` | `{row['left']}` | `{row['right']}` | `{row['delta']}` |"
        )

    return "\n".join(
        [
            "# Benchmark Comparison",
            "",
            f"- Left: `{summary.left_label}` -> `{summary.left_root}`",
            f"- Right: `{summary.right_label}` -> `{summary.right_root}`",
            "",
            "## Key Metrics",
            *table_lines,
            "",
            "## Weekly Risk Shares",
            f"- Left: `{summary.left_metrics['risk_distribution_by_week']}`",
            f"- Right: `{summary.right_metrics['risk_distribution_by_week']}`",
            "",
            "## Trajectory Risk by Week",
            f"- Left: `{summary.left_metrics['risk_score_by_trajectory_week']}`",
            f"- Right: `{summary.right_metrics['risk_score_by_trajectory_week']}`",
            "",
            "## Outcome Ordering",
            f"- Left: `{summary.left_metrics['outcome_by_trajectory']}`",
            f"- Right: `{summary.right_metrics['outcome_by_trajectory']}`",
            "",
            "## Warnings",
            f"- Left: `{summary.left_metrics['warnings']}`",
            f"- Right: `{summary.right_metrics['warnings']}`",
            "",
        ]
    )


def write_comparison_reports(summary: ComparisonSummary, output_dir: str | Path | None = None) -> Path:
    if output_dir is None:
        output_dir = (
            REPO_ROOT
            / "data"
            / "artifacts"
            / "reports"
            / "comparisons"
            / f"{_slugify(summary.left_label)}__vs__{_slugify(summary.right_label)}"
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    (output_path / "comparison_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2),
        encoding="utf-8",
    )
    (output_path / "comparison_report.md").write_text(
        format_markdown_report(summary),
        encoding="utf-8",
    )
    pd.DataFrame(summary.key_metrics).to_csv(output_path / "key_metrics.csv", index=False)
    return output_path
