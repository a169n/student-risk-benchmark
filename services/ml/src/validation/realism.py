from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd


def _series_stats(series: pd.Series) -> dict[str, float | int | None]:
    non_null = series.dropna()
    if non_null.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }

    return {
        "count": int(non_null.shape[0]),
        "mean": round(float(non_null.mean()), 4),
        "std": round(float(non_null.std(ddof=0)), 4),
        "min": round(float(non_null.min()), 4),
        "q25": round(float(non_null.quantile(0.25)), 4),
        "median": round(float(non_null.median()), 4),
        "q75": round(float(non_null.quantile(0.75)), 4),
        "max": round(float(non_null.max()), 4),
    }


def _safe_correlation(left: pd.Series, right: pd.Series) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if frame.shape[0] < 2:
        return None
    correlation = frame["left"].corr(frame["right"])
    if pd.isna(correlation):
        return None
    return round(float(correlation), 4)


def _round_or_none(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), 4)


def _nested_week_trajectory_mean(frame: pd.DataFrame, value_column: str) -> dict[str, dict[int, float]]:
    grouped = (
        frame.groupby(["week_number", "trajectory_type"], sort=True)[value_column]
        .mean()
        .unstack()
    )
    nested: dict[str, dict[int, float]] = {}
    for trajectory_type in grouped.columns:
        nested[str(trajectory_type)] = {
            int(week_number): round(float(value), 4)
            for week_number, value in grouped[trajectory_type].dropna().items()
        }
    return nested


@dataclass(frozen=True)
class RealismMetrics:
    row_counts: dict[str, int]
    risk_distribution: dict[str, object]
    final_grade_distribution: dict[str, object]
    passed_rate: dict[str, object]
    attendance_distribution: dict[str, object]
    activity_distribution: dict[str, object]
    submission_discipline: dict[str, object]
    correlations: dict[str, object]
    risk_by_trajectory: dict[str, object]
    outcome_by_trajectory: dict[str, object]
    risk_distribution_by_week: dict[int, dict[str, object]]
    risk_score_by_trajectory_week: dict[str, dict[int, float]]
    predicted_final_grade_by_trajectory_week: dict[str, dict[int, float]]
    attendance_by_week_trajectory: dict[str, dict[int, float]]
    activity_by_week_trajectory: dict[str, dict[int, float]]
    missingness_by_week: dict[int, dict[str, float]]
    early_warning: dict[str, object]
    warnings: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RealismAudit:
    def run(self, datasets: dict[str, pd.DataFrame]) -> RealismMetrics:
        students = datasets["students"][["student_id", "trajectory_type"]]
        snapshots = datasets["student_twin_snapshots"].merge(students, on="student_id", how="left")
        attendance_weekly = self._student_week_attendance(datasets, students)
        weekly_activity = datasets["weekly_activity"].merge(students, on="student_id", how="left")

        row_counts = {name: int(frame.shape[0]) for name, frame in datasets.items()}
        risk_distribution = self._check_risk_distribution(snapshots)
        final_grade_distribution = self._check_final_grade_distribution(datasets)
        passed_rate = self._check_passed_rate(datasets)
        attendance_distribution = self._check_attendance_distribution(attendance_weekly)
        activity_distribution = self._check_activity_distribution(weekly_activity)
        submission_discipline = self._check_submission_discipline(datasets)
        correlations = self._check_correlations(datasets)
        risk_by_trajectory = self._check_risk_by_trajectory(snapshots)
        outcome_by_trajectory = self._check_outcome_by_trajectory(datasets, students)
        risk_distribution_by_week = self._check_risk_distribution_by_week(snapshots)
        risk_score_by_trajectory_week = _nested_week_trajectory_mean(snapshots, "risk_score")
        predicted_final_grade_by_trajectory_week = _nested_week_trajectory_mean(
            snapshots, "predicted_final_grade"
        )
        attendance_by_week_trajectory = _nested_week_trajectory_mean(attendance_weekly, "attendance_rate")
        activity_by_week_trajectory = _nested_week_trajectory_mean(weekly_activity, "activity_score")
        missingness_by_week = self._check_missingness_by_week(snapshots)
        early_warning = self._check_early_warning(
            snapshots=snapshots,
            risk_score_by_trajectory_week=risk_score_by_trajectory_week,
            activity_by_week_trajectory=activity_by_week_trajectory,
        )

        warnings = self._build_warnings(
            risk_distribution=risk_distribution,
            risk_by_trajectory=risk_by_trajectory,
            outcome_by_trajectory=outcome_by_trajectory,
            early_warning=early_warning,
        )

        return RealismMetrics(
            row_counts=row_counts,
            risk_distribution=risk_distribution,
            final_grade_distribution=final_grade_distribution,
            passed_rate=passed_rate,
            attendance_distribution=attendance_distribution,
            activity_distribution=activity_distribution,
            submission_discipline=submission_discipline,
            correlations=correlations,
            risk_by_trajectory=risk_by_trajectory,
            outcome_by_trajectory=outcome_by_trajectory,
            risk_distribution_by_week=risk_distribution_by_week,
            risk_score_by_trajectory_week=risk_score_by_trajectory_week,
            predicted_final_grade_by_trajectory_week=predicted_final_grade_by_trajectory_week,
            attendance_by_week_trajectory=attendance_by_week_trajectory,
            activity_by_week_trajectory=activity_by_week_trajectory,
            missingness_by_week=missingness_by_week,
            early_warning=early_warning,
            warnings=warnings,
        )

    def _student_week_attendance(
        self, datasets: dict[str, pd.DataFrame], students: pd.DataFrame
    ) -> pd.DataFrame:
        attendance = datasets["attendance"].merge(students, on="student_id", how="left")
        attendance["attendance_value"] = attendance["attendance_status"].map(
            {"present": 1.0, "late": 0.85, "absent": 0.0, "excused": pd.NA}
        )
        return (
            attendance.groupby(["student_id", "trajectory_type", "week_number"], sort=True)[
                "attendance_value"
            ]
            .mean()
            .reset_index(name="attendance_rate")
        )

    def _check_risk_distribution(self, snapshots: pd.DataFrame) -> dict[str, object]:
        counts = snapshots["risk_level"].value_counts().to_dict()
        shares = snapshots["risk_level"].value_counts(normalize=True).to_dict()
        high_share = float(shares.get("high", 0.0))
        medium_share = float(shares.get("medium", 0.0))
        low_share = float(shares.get("low", 0.0))

        return {
            "counts": {level: int(counts.get(level, 0)) for level in ["low", "medium", "high"]},
            "shares": {
                "low": round(low_share, 4),
                "medium": round(medium_share, 4),
                "high": round(high_share, 4),
            },
            "target_bands": {"high": [0.08, 0.15], "medium": [0.20, 0.35]},
            "high_within_target": 0.08 <= high_share <= 0.15,
            "medium_within_target": 0.20 <= medium_share <= 0.35,
        }

    def _check_final_grade_distribution(self, datasets: dict[str, pd.DataFrame]) -> dict[str, object]:
        final_results = datasets["final_results"]
        completed = final_results.loc[final_results["completion_status"] == "completed", "final_grade"]
        stats = _series_stats(completed)
        stats["within_sanity_bounds"] = (
            stats["min"] is None
            or (
                0.0 <= float(stats["min"]) <= float(stats["max"])
                and float(stats["max"]) <= 100.0
            )
        )
        return stats

    def _check_passed_rate(self, datasets: dict[str, pd.DataFrame]) -> dict[str, object]:
        final_results = datasets["final_results"]
        completed = final_results.loc[
            final_results["completion_status"] == "completed", "passed"
        ].dropna()
        pass_rate = float(completed.astype(float).mean()) if not completed.empty else 0.0
        return {
            "completed_count": int(completed.shape[0]),
            "pass_rate": round(pass_rate, 4),
            "target_range": [0.60, 0.90],
            "within_target": 0.60 <= pass_rate <= 0.90,
        }

    def _check_attendance_distribution(self, attendance_weekly: pd.DataFrame) -> dict[str, object]:
        last_week = int(attendance_weekly["week_number"].max())
        last_week_rates = attendance_weekly.loc[
            attendance_weekly["week_number"] == last_week, "attendance_rate"
        ]
        stats = _series_stats(last_week_rates)
        stats["week_number"] = last_week
        return stats

    def _check_activity_distribution(self, weekly_activity: pd.DataFrame) -> dict[str, object]:
        last_week = int(weekly_activity["week_number"].max())
        last_week_activity = weekly_activity.loc[weekly_activity["week_number"] == last_week]
        return {
            "week_number": last_week,
            "activity_score": _series_stats(last_week_activity["activity_score"]),
            "time_on_platform_minutes": _series_stats(last_week_activity["time_on_platform_minutes"]),
        }

    def _check_submission_discipline(self, datasets: dict[str, pd.DataFrame]) -> dict[str, object]:
        submissions = datasets["submissions"]
        required_count = int(submissions.shape[0])
        late_count = int((submissions["submission_status"] == "late").sum())
        missed_count = int((submissions["submission_status"] == "missing").sum())
        submitted_attempts = submissions.loc[
            submissions["submission_status"].isin(["submitted", "late"]), "attempt_count"
        ]

        return {
            "required_count": required_count,
            "late_count": late_count,
            "missed_count": missed_count,
            "late_rate": round(late_count / required_count, 4) if required_count else 0.0,
            "missed_rate": round(missed_count / required_count, 4) if required_count else 0.0,
            "attempt_count": _series_stats(submitted_attempts),
        }

    def _check_correlations(self, datasets: dict[str, pd.DataFrame]) -> dict[str, object]:
        final_results = datasets["final_results"]
        snapshots = datasets["student_twin_snapshots"]
        completed = final_results.loc[
            final_results["completion_status"] == "completed", ["student_id", "final_grade", "passed"]
        ]
        final_week = int(snapshots["week_number"].max())
        final_snapshots = snapshots.loc[snapshots["week_number"] == final_week]
        joined = final_snapshots.merge(completed, on="student_id", how="inner")

        attendance_grade_r = _safe_correlation(joined["attendance_rate_to_date"], joined["final_grade"])
        missed_grade_r = _safe_correlation(joined["missed_assignments_to_date"], joined["final_grade"])
        activity_performance_r = _safe_correlation(
            joined["activity_score_to_date"], joined["performance_index"]
        )

        return {
            "attendance_grade_r": attendance_grade_r,
            "missed_grade_r": missed_grade_r,
            "activity_performance_r": activity_performance_r,
            "checks": {
                "attendance_grade_gt_0_3": attendance_grade_r is not None and attendance_grade_r > 0.3,
                "missed_grade_lt_minus_0_2": missed_grade_r is not None and missed_grade_r < -0.2,
                "activity_performance_positive": activity_performance_r is not None
                and activity_performance_r > 0.2,
            },
        }

    def _check_risk_by_trajectory(self, snapshots: pd.DataFrame) -> dict[str, object]:
        final_week = int(snapshots["week_number"].max())
        final_snapshots = snapshots.loc[snapshots["week_number"] == final_week]
        means = final_snapshots.groupby("trajectory_type", sort=True)["risk_score"].mean().to_dict()

        stable_high = means.get("stable_high")
        improving = means.get("improving")
        declining = means.get("declining")
        at_risk = means.get("consistently_at_risk")
        ordering_valid = all(
            value is not None for value in [stable_high, improving, declining, at_risk]
        ) and (at_risk > declining > improving > stable_high)

        return {
            "final_week": final_week,
            "mean_risk_score": {key: round(float(value), 4) for key, value in means.items()},
            "ordering_expected": [
                "consistently_at_risk",
                "declining",
                "improving",
                "stable_high",
            ],
            "ordering_valid": ordering_valid,
        }

    def _check_outcome_by_trajectory(
        self, datasets: dict[str, pd.DataFrame], students: pd.DataFrame
    ) -> dict[str, object]:
        joined = datasets["final_results"].merge(students, on="student_id", how="left")
        completed = joined.loc[joined["completion_status"] == "completed"].copy()
        final_grade_mean = completed.groupby("trajectory_type", sort=True)["final_grade"].mean().to_dict()
        pass_rate = completed.groupby("trajectory_type", sort=True)["passed"].mean().to_dict()

        stable_high_grade = final_grade_mean.get("stable_high")
        improving_grade = final_grade_mean.get("improving")
        declining_grade = final_grade_mean.get("declining")
        at_risk_grade = final_grade_mean.get("consistently_at_risk")
        grade_ordering_valid = all(
            value is not None
            for value in [stable_high_grade, improving_grade, declining_grade, at_risk_grade]
        ) and (stable_high_grade > improving_grade > declining_grade > at_risk_grade)

        stable_high_pass = pass_rate.get("stable_high")
        improving_pass = pass_rate.get("improving")
        declining_pass = pass_rate.get("declining")
        at_risk_pass = pass_rate.get("consistently_at_risk")
        pass_ordering_valid = all(
            value is not None
            for value in [stable_high_pass, improving_pass, declining_pass, at_risk_pass]
        ) and (stable_high_pass >= improving_pass > declining_pass > at_risk_pass)

        return {
            "mean_final_grade": {
                key: round(float(value), 4) for key, value in final_grade_mean.items()
            },
            "pass_rate": {key: round(float(value), 4) for key, value in pass_rate.items()},
            "grade_ordering_expected": [
                "stable_high",
                "improving",
                "declining",
                "consistently_at_risk",
            ],
            "pass_ordering_expected": [
                "stable_high",
                "improving",
                "declining",
                "consistently_at_risk",
            ],
            "grade_ordering_valid": grade_ordering_valid,
            "pass_ordering_valid": pass_ordering_valid,
        }

    def _check_risk_distribution_by_week(self, snapshots: pd.DataFrame) -> dict[int, dict[str, object]]:
        metrics: dict[int, dict[str, object]] = {}
        for week_number, frame in snapshots.groupby("week_number", sort=True):
            counts = frame["risk_level"].value_counts().to_dict()
            shares = frame["risk_level"].value_counts(normalize=True).to_dict()
            metrics[int(week_number)] = {
                "counts": {level: int(counts.get(level, 0)) for level in ["low", "medium", "high"]},
                "shares": {
                    level: round(float(shares.get(level, 0.0)), 4)
                    for level in ["low", "medium", "high"]
                },
            }
        return metrics

    def _check_missingness_by_week(self, snapshots: pd.DataFrame) -> dict[int, dict[str, float]]:
        metrics: dict[int, dict[str, float]] = {}
        for week_number, frame in snapshots.groupby("week_number", sort=True):
            metrics[int(week_number)] = {
                "assignment_score_missing_share": round(
                    float(frame["avg_assignment_score_to_date"].isna().mean()), 4
                ),
                "quiz_score_missing_share": round(
                    float(frame["avg_quiz_score_to_date"].isna().mean()), 4
                ),
                "has_assignment_score_false_share": round(
                    float((~frame["has_assignment_score_to_date"]).mean()), 4
                ),
                "has_quiz_score_false_share": round(
                    float((~frame["has_quiz_score_to_date"]).mean()), 4
                ),
            }
        return metrics

    def _check_early_warning(
        self,
        *,
        snapshots: pd.DataFrame,
        risk_score_by_trajectory_week: dict[str, dict[int, float]],
        activity_by_week_trajectory: dict[str, dict[int, float]],
    ) -> dict[str, object]:
        pre_week_5 = snapshots.loc[snapshots["week_number"] < 5]
        high_risk_share_before_week_5 = (
            float((pre_week_5["risk_level"] == "high").mean()) if not pre_week_5.empty else 0.0
        )
        weekly_high_risk_share = {
            int(week_number): round(float((frame["risk_level"] == "high").mean()), 4)
            for week_number, frame in pre_week_5.groupby("week_number", sort=True)
        }

        declining_risk = risk_score_by_trajectory_week.get("declining", {})
        improving_risk = risk_score_by_trajectory_week.get("improving", {})
        declining_activity = activity_by_week_trajectory.get("declining", {})

        declining_week1 = declining_risk.get(1)
        declining_week4 = declining_risk.get(4)
        declining_week10 = declining_risk.get(max(declining_risk.keys(), default=1))
        improving_week6 = improving_risk.get(6)
        improving_week10 = improving_risk.get(max(improving_risk.keys(), default=1))
        declining_activity_week1 = declining_activity.get(1)
        declining_activity_week4 = declining_activity.get(4)

        declining_risk_delta_1_to_4 = (
            None
            if declining_week1 is None or declining_week4 is None
            else round(float(declining_week4 - declining_week1), 4)
        )
        declining_risk_delta_1_to_10 = (
            None
            if declining_week1 is None or declining_week10 is None
            else round(float(declining_week10 - declining_week1), 4)
        )
        improving_risk_delta_6_to_10 = (
            None
            if improving_week6 is None or improving_week10 is None
            else round(float(improving_week10 - improving_week6), 4)
        )
        declining_activity_delta_1_to_4 = (
            None
            if declining_activity_week1 is None or declining_activity_week4 is None
            else round(float(declining_activity_week4 - declining_activity_week1), 4)
        )

        return {
            "weeks_considered": [1, 2, 3, 4],
            "high_risk_share_before_week_5": round(high_risk_share_before_week_5, 4),
            "weekly_high_risk_share_before_week_5": weekly_high_risk_share,
            "pre_week_5_high_risk_nonzero": high_risk_share_before_week_5 > 0.0,
            "declining_risk_delta_week1_to_week4": declining_risk_delta_1_to_4,
            "declining_risk_delta_week1_to_week10": declining_risk_delta_1_to_10,
            "declining_activity_delta_week1_to_week4": declining_activity_delta_1_to_4,
            "improving_risk_delta_week6_to_week10": improving_risk_delta_6_to_10,
            "declining_degrades_early_enough": (
                declining_risk_delta_1_to_4 is not None and declining_risk_delta_1_to_4 > 0.02
            ),
            "improving_recovers_in_late_weeks": (
                improving_risk_delta_6_to_10 is not None and improving_risk_delta_6_to_10 < -0.005
            ),
        }

    def _build_warnings(
        self,
        *,
        risk_distribution: dict[str, object],
        risk_by_trajectory: dict[str, object],
        outcome_by_trajectory: dict[str, object],
        early_warning: dict[str, object],
    ) -> list[str]:
        warnings: list[str] = []
        if not risk_distribution["high_within_target"]:
            warnings.append("High-risk share is outside the target band (8%-15%).")
        if not risk_distribution["medium_within_target"]:
            warnings.append("Medium-risk share is outside the target band (20%-35%).")
        if not risk_by_trajectory["ordering_valid"]:
            warnings.append(
                "Final-week risk ordering is weak or inverted; expected consistently_at_risk > declining > improving > stable_high."
            )
        if not outcome_by_trajectory["grade_ordering_valid"]:
            warnings.append(
                "Final-grade ordering is weak or inverted; expected stable_high > improving > declining > consistently_at_risk."
            )
        if not outcome_by_trajectory["pass_ordering_valid"]:
            warnings.append(
                "Pass-rate ordering is weak or inverted; expected stable_high >= improving > declining > consistently_at_risk."
            )
        if not early_warning["pre_week_5_high_risk_nonzero"]:
            warnings.append("No high-risk cases appear before week 5, weakening early-warning usefulness.")
        if not early_warning["declining_degrades_early_enough"]:
            warnings.append("Declining trajectories do not worsen early enough to support early-warning claims.")
        if not early_warning["improving_recovers_in_late_weeks"]:
            warnings.append("Improving trajectories do not show clear late-week recovery.")
        return warnings


def format_terminal_report(metrics: RealismMetrics) -> str:
    risk = metrics.risk_distribution
    passed = metrics.passed_rate
    grades = metrics.final_grade_distribution
    trajectory = metrics.risk_by_trajectory
    outcomes = metrics.outcome_by_trajectory
    early = metrics.early_warning

    lines = [
        "Realism Audit",
        f"- Risk shares: low={risk['shares']['low']:.2%}, medium={risk['shares']['medium']:.2%}, high={risk['shares']['high']:.2%}",
        f"- Grade mean/std: mean={grades['mean']}, std={grades['std']}",
        f"- Pass rate: {passed['pass_rate']:.2%}",
        f"- Early high-risk share before week 5: {early['high_risk_share_before_week_5']:.2%}",
        f"- Final risk ordering valid: {trajectory['ordering_valid']}",
        f"- Final grade ordering valid: {outcomes['grade_ordering_valid']}",
        f"- Pass-rate ordering valid: {outcomes['pass_ordering_valid']}",
    ]
    if metrics.warnings:
        lines.append("- Warnings:")
        lines.extend(f"  * {warning}" for warning in metrics.warnings)
    else:
        lines.append("- Warnings: none")
    return "\n".join(lines)


def format_markdown_report(metrics: RealismMetrics) -> str:
    return "\n".join(
        [
            "# Realism Audit",
            "",
            "## Row Counts",
            f"- `{metrics.row_counts}`",
            "",
            "## Global Distribution Checks",
            f"- Risk distribution: `{metrics.risk_distribution}`",
            f"- Final grade distribution: `{metrics.final_grade_distribution}`",
            f"- Passed rate: `{metrics.passed_rate}`",
            "",
            "## Behavioral Signals",
            f"- Attendance distribution: `{metrics.attendance_distribution}`",
            f"- Activity distribution: `{metrics.activity_distribution}`",
            f"- Submission discipline: `{metrics.submission_discipline}`",
            f"- Correlations: `{metrics.correlations}`",
            "",
            "## Trajectory Ordering",
            f"- Final-week risk ordering: `{metrics.risk_by_trajectory}`",
            f"- End-of-course outcome ordering: `{metrics.outcome_by_trajectory}`",
            "",
            "## Week-Aware Diagnostics",
            f"- Risk distribution by week: `{metrics.risk_distribution_by_week}`",
            f"- Mean risk by trajectory and week: `{metrics.risk_score_by_trajectory_week}`",
            f"- Mean predicted_final_grade by trajectory and week: `{metrics.predicted_final_grade_by_trajectory_week}`",
            f"- Attendance by week and trajectory: `{metrics.attendance_by_week_trajectory}`",
            f"- Activity by week and trajectory: `{metrics.activity_by_week_trajectory}`",
            f"- Missingness by week: `{metrics.missingness_by_week}`",
            f"- Early-warning diagnostics: `{metrics.early_warning}`",
            "",
            "## Warnings",
            *(["- None"] if not metrics.warnings else [f"- {warning}" for warning in metrics.warnings]),
            "",
        ]
    )


def write_reports(metrics: RealismMetrics, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "realism_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2),
        encoding="utf-8",
    )
    (report_dir / "realism_report.md").write_text(
        format_markdown_report(metrics),
        encoding="utf-8",
    )
    return report_dir
