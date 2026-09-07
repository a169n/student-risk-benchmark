"""Materialize a deterministic research demo payload from frozen artifacts.

Inputs (read-only):
- data/processed/student_twin_snapshots.csv
- data/raw/students.csv
- data/raw/final_results.csv
- data/artifacts/experiments/exp_001_baseline/baseline_v1_results.json
- data/artifacts/experiments/exp_002_twin_ablation/exp_002_twin_ablation_results.json
- data/artifacts/experiments/exp_003_mastery_validation/mastery_diagnostics.json
- data/artifacts/experiments/exp_004_xai_on_lean_twin/exp_004_xai_on_lean_twin_results.json
- data/artifacts/experiments/exp_004_xai_on_lean_twin/local_case_explanations.json
- data/artifacts/experiments/exp_004_xai_on_lean_twin/global_feature_importance.csv
- data/artifacts/experiments/exp_005_public_benchmark_oulad/exp_005_public_benchmark_oulad_results.json
- data/artifacts/experiments/exp_005_public_benchmark_oulad/experiment_metadata.json

Output:
- data/artifacts/research_demo/research_demo_payload.json

This script does not retrain any model and does not alter any experiment evidence.
It only re-shapes already-frozen artifacts into a UI-friendly contract.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


PAYLOAD_SCHEMA_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[4]

INPUTS = {
    "snapshots": REPO_ROOT / "data" / "processed" / "student_twin_snapshots.csv",
    "students": REPO_ROOT / "data" / "raw" / "students.csv",
    "final_results": REPO_ROOT / "data" / "raw" / "final_results.csv",
    "exp001": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_001_baseline"
    / "baseline_v1_results.json",
    "exp002": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_002_twin_ablation"
    / "exp_002_twin_ablation_results.json",
    "exp003": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_003_mastery_validation"
    / "mastery_diagnostics.json",
    "exp004_results": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_004_xai_on_lean_twin"
    / "exp_004_xai_on_lean_twin_results.json",
    "exp004_cases": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_004_xai_on_lean_twin"
    / "local_case_explanations.json",
    "exp004_global_csv": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_004_xai_on_lean_twin"
    / "global_feature_importance.csv",
    "exp005_results": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_005_public_benchmark_oulad"
    / "exp_005_public_benchmark_oulad_results.json",
    "exp005_metadata": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_005_public_benchmark_oulad"
    / "experiment_metadata.json",
}

OUTPUT = REPO_ROOT / "data" / "artifacts" / "research_demo" / "research_demo_payload.json"

LOW_ACTIVITY_THRESHOLD = 40.0
LOW_MASTERY_THRESHOLD = 60.0
AT_RISK_GRADE_THRESHOLD = 60.0


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def _none_if_nan(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _best_regression(
    rows: list[dict[str, Any]],
    *,
    target: str,
    split: str,
    feature_set: str,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.get("task") == "regression"
        and row.get("target") == target
        and row.get("split_strategy") == split
        and row.get("feature_set") == feature_set
    ]
    best: dict[str, Any] | None = None
    for row in candidates:
        rmse = (row.get("metrics") or {}).get("rmse")
        if rmse is None:
            rmse = row.get("metric_rmse")
        if rmse is None:
            continue
        if best is None or rmse < best["rmse"]:
            best = {
                "feature_set": feature_set,
                "model": row.get("model"),
                "rmse": float(rmse),
                "split": split,
            }
    return best


def _build_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for case in cases:
        out.append(
            {
                "studentId": case["student_id"],
                "weekNumber": case["week_number"],
                "caseType": case["case_type"],
                "actualFinalGrade": _round(case["actual_final_grade"]),
                "predictedFinalGrade": _round(case["predicted_final_grade"]),
                "predictionError": _round(case["prediction_error"]),
                "riskLevelContext": case["risk_level_context"],
                "passed": case.get("passed"),
                "masteryShare": _round(case["mastery_abs_contribution_share"], 4),
                "teacherAssessment": case["teacher_meaningfulness_assessment"],
                "topContributions": [
                    {
                        "feature": item["feature"],
                        "value": _round(item["value"], 4),
                        "contribution": _round(item["contribution"], 3),
                        "absContribution": _round(item["abs_contribution"], 3),
                        "direction": item["direction"],
                        "referenceMedian": _round(
                            item.get("reference_train_median"), 4
                        ),
                    }
                    for item in case["lean_top_contributions"]
                ],
            }
        )
    return out


def _build_xai_section(exp004: dict[str, Any]) -> dict[str, Any]:
    rows = exp004["global_explanations"]["lean_twin"]["rows"]
    return {
        "method": exp004["method_notes"]["local_explanation_method"],
        "shapUsed": exp004["method_notes"]["shap_used"],
        "topGlobalFeatures": [
            {
                "feature": row["feature"],
                "rank": row["rank"],
                "importanceShare": _round(row["importance_share"], 4),
            }
            for row in rows[:8]
        ],
        "dominance": {
            "topFeature": exp004["dominance_audit"]["top_feature"],
            "top1Share": _round(exp004["dominance_audit"]["top1_share"], 4),
            "averageLocalMasteryShare": _round(
                exp004["dominance_audit"]["average_local_mastery_share"], 4
            ),
            "outcome": exp004["dominance_audit"]["outcome"],
            "flags": exp004["dominance_audit"]["flags"],
        },
        "metrics": {
            "baselineRmse": _round(exp004["metrics"]["baseline"]["rmse"]),
            "leanRmse": _round(exp004["metrics"]["lean_twin"]["rmse"]),
            "leanDelta": _round(exp004["metrics"]["lean_delta_rmse_vs_baseline"]),
            "withoutOverallDelta": _round(
                exp004["metrics"]["without_overall_delta_rmse_vs_lean"]
            ),
        },
        "recommendation": {
            "outcome": exp004["recommendation"]["outcome"],
            "decisionText": exp004["recommendation"]["decision_text"],
            "flags": exp004["recommendation"]["flags"],
        },
    }


def _build_directions(global_csv_path: Path) -> dict[str, str]:
    if not global_csv_path.exists():
        return {}
    df = pd.read_csv(global_csv_path)
    df = df[df["feature_set"] == "B_lms_plus_mastery"]
    direction_map: dict[str, str] = {}
    for _, row in df.iterrows():
        feature = row["feature"]
        note = row.get("direction_note")
        if isinstance(note, str):
            direction_map[feature] = note
    return direction_map


def _build_experiments(
    exp001: dict[str, Any],
    exp002: dict[str, Any],
    exp003: dict[str, Any],
    exp004: dict[str, Any],
    exp005_results: dict[str, Any],
    exp005_metadata: dict[str, Any],
) -> dict[str, Any]:
    exp001_baseline = _best_regression(
        exp001["rows"], target="final_grade", split="student_group", feature_set="B_lms"
    )
    exp001_twin = _best_regression(
        exp001["rows"], target="final_grade", split="student_group", feature_set="C_twin"
    )
    exp002_baseline = _best_regression(
        exp002["rows"], target="final_grade", split="student_group", feature_set="B_lms"
    )
    exp002_lean = _best_regression(
        exp002["rows"],
        target="final_grade",
        split="student_group",
        feature_set="B_lms_plus_mastery",
    )

    oulad_grouped_baseline = _best_regression(
        exp005_results["rows"],
        target="final_weighted_score",
        split="student_group",
        feature_set="B_lms_oulad",
    )
    oulad_grouped_lean = _best_regression(
        exp005_results["rows"],
        target="final_weighted_score",
        split="student_group",
        feature_set="B_lms_plus_mastery_oulad",
    )
    oulad_temporal_baseline = _best_regression(
        exp005_results["rows"],
        target="final_weighted_score",
        split="temporal_forward",
        feature_set="B_lms_oulad",
    )
    oulad_temporal_lean = _best_regression(
        exp005_results["rows"],
        target="final_weighted_score",
        split="temporal_forward",
        feature_set="B_lms_plus_mastery_oulad",
    )

    timeline = [
        {
            "id": "exp_001",
            "title": "Baseline feature-set comparison",
            "result": (
                f"C_twin RMSE {exp001_twin['rmse']:.3f} vs B_lms {exp001_baseline['rmse']:.3f}"
                if exp001_baseline and exp001_twin
                else "Result unavailable"
            ),
            "decision": "Full Twin not justified.",
        },
        {
            "id": "exp_002",
            "title": "Twin subgroup ablation",
            "result": (
                f"B_lms_plus_mastery RMSE {exp002_lean['rmse']:.3f} vs B_lms {exp002_baseline['rmse']:.3f}"
                if exp002_lean and exp002_baseline
                else "Result unavailable"
            ),
            "decision": "Lean mastery-centered Twin candidate carried forward.",
        },
        {
            "id": "exp_003",
            "title": "Mastery validation",
            "result": (
                "Improved weeks "
                + ", ".join(
                    str(week)
                    for week in (
                        list(exp003["recommendation"]["early_weeks_improved"])
                        + list(exp003["recommendation"]["late_weeks_improved"])
                    )
                )
            ),
            "decision": "Validated with overall_mastery redundancy caveat.",
        },
        {
            "id": "exp_004",
            "title": "XAI on lean Twin",
            "result": f"Dominance audit outcome: {exp004['dominance_audit']['outcome']}",
            "decision": "Permutation + perturbation explanations retained as model-behavior signal.",
        },
        {
            "id": "exp_005",
            "title": "OULAD public benchmark",
            "result": exp005_metadata["diagnostics"]["interpretation"]["short_conclusion"],
            "decision": "External transfer evidence remains mixed.",
        },
    ]

    return {
        "timeline": timeline,
        "leanTwin": {
            "baselineRmse": _round(exp004["metrics"]["baseline"]["rmse"]),
            "leanRmse": _round(exp004["metrics"]["lean_twin"]["rmse"]),
            "leanDelta": _round(exp004["metrics"]["lean_delta_rmse_vs_baseline"]),
            "withoutOverallDelta": _round(
                exp004["metrics"]["without_overall_delta_rmse_vs_lean"]
            ),
            "earlyWeeksImproved": exp003["recommendation"]["early_weeks_improved"],
            "lateWeeksImproved": exp003["recommendation"]["late_weeks_improved"],
            "flags": exp003["recommendation"]["flags"],
        },
        "oulad": {
            "rowCounts": exp005_metadata["diagnostics"]["row_counts"],
            "weekMin": exp005_metadata["diagnostics"]["target_summary"]["week_min"],
            "weekMax": exp005_metadata["diagnostics"]["target_summary"]["week_max"],
            "outcome": exp005_metadata["diagnostics"]["interpretation"]["outcome"],
            "shortConclusion": exp005_metadata["diagnostics"]["interpretation"][
                "short_conclusion"
            ],
            "grouped": {
                "baseline": oulad_grouped_baseline,
                "lean": oulad_grouped_lean,
                "delta": _round(
                    oulad_grouped_lean["rmse"] - oulad_grouped_baseline["rmse"]
                )
                if oulad_grouped_baseline and oulad_grouped_lean
                else None,
            },
            "temporal": {
                "baseline": oulad_temporal_baseline,
                "lean": oulad_temporal_lean,
                "delta": _round(
                    oulad_temporal_lean["rmse"] - oulad_temporal_baseline["rmse"]
                )
                if oulad_temporal_baseline and oulad_temporal_lean
                else None,
            },
        },
    }


def _build_students(
    snapshots: pd.DataFrame,
    students: pd.DataFrame,
    final_results: pd.DataFrame,
    case_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    student_ids = sorted(snapshots["student_id"].unique().tolist())
    out = []
    for sid in student_ids:
        sdf = snapshots[snapshots["student_id"] == sid].sort_values("week_number")
        if sdf.empty:
            continue
        student_row = students[students["student_id"] == sid]
        final_row = final_results[final_results["student_id"] == sid]
        latest = sdf.iloc[-1]

        weekly = []
        for _, row in sdf.iterrows():
            weekly.append(
                {
                    "weekNumber": int(row["week_number"]),
                    "predictedFinalGrade": _round(_none_if_nan(row["predicted_final_grade"])),
                    "riskLevel": row["risk_level"],
                    "riskScore": _round(_none_if_nan(row["risk_score"]), 4),
                    "activityScore": _round(_none_if_nan(row["activity_score_to_date"])),
                    "assignmentAverage": _round(
                        _none_if_nan(row["avg_assignment_score_to_date"])
                    ),
                    "quizAverage": _round(_none_if_nan(row["avg_quiz_score_to_date"])),
                    "attendanceRate": _round(
                        _none_if_nan(row["attendance_rate_to_date"]), 4
                    ),
                    "overallMastery": _round(_none_if_nan(row["overall_mastery"])),
                    "currentTopicMastery": _round(
                        _none_if_nan(row["current_topic_mastery"])
                    ),
                    "engagementIndex": _round(_none_if_nan(row["engagement_index"])),
                    "performanceIndex": _round(_none_if_nan(row["performance_index"])),
                    "disciplineIndex": _round(_none_if_nan(row["discipline_index"])),
                    "scoreTrend3w": _round(_none_if_nan(row["score_trend_3w"]), 4),
                    "activityTrend3w": _round(_none_if_nan(row["activity_trend_3w"]), 4),
                    "attendanceTrend3w": _round(
                        _none_if_nan(row["attendance_trend_3w"]), 4
                    ),
                }
            )

        actual_final = None
        passed = None
        if not final_row.empty:
            actual_final = _round(_none_if_nan(final_row.iloc[0]["final_grade"]))
            passed_val = _none_if_nan(final_row.iloc[0]["passed"])
            if passed_val is not None:
                passed = bool(passed_val)

        student_label = sid
        cohort_label: str | None = None
        trajectory_label: str | None = None
        if not student_row.empty:
            srow = student_row.iloc[0]
            student_label = str(_none_if_nan(srow["student_code"]) or sid)
            cohort_label = _none_if_nan(srow["cohort_label"])
            trajectory_label = _none_if_nan(srow["trajectory_type"])

        explanation = case_lookup.get(sid)

        out.append(
            {
                "studentId": sid,
                "studentLabel": student_label,
                "cohortLabel": cohort_label,
                "trajectoryLabel": trajectory_label,
                "courseId": str(latest["course_id"]),
                "currentWeek": int(latest["week_number"]),
                "current": {
                    "predictedFinalGrade": _round(
                        _none_if_nan(latest["predicted_final_grade"])
                    ),
                    "riskLevel": latest["risk_level"],
                    "riskScore": _round(_none_if_nan(latest["risk_score"]), 4),
                    "activityScore": _round(
                        _none_if_nan(latest["activity_score_to_date"])
                    ),
                    "assignmentAverage": _round(
                        _none_if_nan(latest["avg_assignment_score_to_date"])
                    ),
                    "quizAverage": _round(_none_if_nan(latest["avg_quiz_score_to_date"])),
                    "attendanceRate": _round(
                        _none_if_nan(latest["attendance_rate_to_date"]), 4
                    ),
                    "overallMastery": _round(_none_if_nan(latest["overall_mastery"])),
                    "currentTopicMastery": _round(
                        _none_if_nan(latest["current_topic_mastery"])
                    ),
                    "engagementIndex": _round(_none_if_nan(latest["engagement_index"])),
                    "performanceIndex": _round(
                        _none_if_nan(latest["performance_index"])
                    ),
                    "disciplineIndex": _round(_none_if_nan(latest["discipline_index"])),
                    "scoreTrend3w": _round(_none_if_nan(latest["score_trend_3w"]), 4),
                },
                "actualFinalGrade": actual_final,
                "passed": passed,
                "weeklyTimeline": weekly,
                "explanation": explanation,
            }
        )
    return out


def _build_cohort(students_payload: list[dict[str, Any]]) -> dict[str, Any]:
    if not students_payload:
        return {
            "courseId": None,
            "studentCount": 0,
            "currentWeek": None,
            "meanPredictedFinalGrade": None,
            "meanActualFinalGrade": None,
            "meanOverallMastery": None,
            "meanActivityScore": None,
            "meanAttendanceRate": None,
            "riskDistribution": {"low": 0, "medium": 0, "high": 0},
            "atRiskCount": 0,
            "lowMasteryCount": 0,
            "lowActivityCount": 0,
            "topAtRisk": [],
            "topImproving": [],
        }

    course_ids = {s["courseId"] for s in students_payload}
    weeks = {s["currentWeek"] for s in students_payload}
    predicted = [
        s["current"]["predictedFinalGrade"]
        for s in students_payload
        if s["current"]["predictedFinalGrade"] is not None
    ]
    actual = [
        s["actualFinalGrade"]
        for s in students_payload
        if s["actualFinalGrade"] is not None
    ]
    mastery = [
        s["current"]["overallMastery"]
        for s in students_payload
        if s["current"]["overallMastery"] is not None
    ]
    activity = [
        s["current"]["activityScore"]
        for s in students_payload
        if s["current"]["activityScore"] is not None
    ]
    attendance = [
        s["current"]["attendanceRate"]
        for s in students_payload
        if s["current"]["attendanceRate"] is not None
    ]

    risk_distribution = {"low": 0, "medium": 0, "high": 0}
    at_risk_count = 0
    low_mastery_count = 0
    low_activity_count = 0
    for s in students_payload:
        risk = s["current"].get("riskLevel")
        if risk in risk_distribution:
            risk_distribution[risk] += 1
        if (
            s["current"]["predictedFinalGrade"] is not None
            and s["current"]["predictedFinalGrade"] < AT_RISK_GRADE_THRESHOLD
        ):
            at_risk_count += 1
        if (
            s["current"]["overallMastery"] is not None
            and s["current"]["overallMastery"] < LOW_MASTERY_THRESHOLD
        ):
            low_mastery_count += 1
        if (
            s["current"]["activityScore"] is not None
            and s["current"]["activityScore"] < LOW_ACTIVITY_THRESHOLD
        ):
            low_activity_count += 1

    sorted_by_predicted = sorted(
        [
            s
            for s in students_payload
            if s["current"]["predictedFinalGrade"] is not None
        ],
        key=lambda s: s["current"]["predictedFinalGrade"],
    )
    top_at_risk = [s["studentId"] for s in sorted_by_predicted[:5]]

    improvers = []
    for s in students_payload:
        timeline = s["weeklyTimeline"]
        first_with_pred = next(
            (w for w in timeline if w["predictedFinalGrade"] is not None), None
        )
        last_with_pred = next(
            (w for w in reversed(timeline) if w["predictedFinalGrade"] is not None),
            None,
        )
        if (
            first_with_pred
            and last_with_pred
            and first_with_pred["weekNumber"] != last_with_pred["weekNumber"]
        ):
            delta = (
                last_with_pred["predictedFinalGrade"]
                - first_with_pred["predictedFinalGrade"]
            )
            improvers.append((s["studentId"], delta))
    improvers.sort(key=lambda item: item[1], reverse=True)
    top_improving = [sid for sid, _ in improvers[:5]]

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return _round(sum(values) / len(values))

    course_id = next(iter(course_ids)) if len(course_ids) == 1 else None
    current_week = max(weeks) if weeks else None

    return {
        "courseId": course_id,
        "studentCount": len(students_payload),
        "currentWeek": current_week,
        "meanPredictedFinalGrade": _mean(predicted),
        "meanActualFinalGrade": _mean(actual),
        "meanOverallMastery": _mean(mastery),
        "meanActivityScore": _mean(activity),
        "meanAttendanceRate": _round(
            sum(attendance) / len(attendance) if attendance else None, 4
        )
        if attendance
        else None,
        "riskDistribution": risk_distribution,
        "atRiskCount": at_risk_count,
        "lowMasteryCount": low_mastery_count,
        "lowActivityCount": low_activity_count,
        "topAtRisk": top_at_risk,
        "topImproving": top_improving,
    }


def build_payload() -> dict[str, Any]:
    snapshots = pd.read_csv(INPUTS["snapshots"])
    students = pd.read_csv(INPUTS["students"])
    final_results = pd.read_csv(INPUTS["final_results"])

    exp001 = _load_json(INPUTS["exp001"])
    exp002 = _load_json(INPUTS["exp002"])
    exp003 = _load_json(INPUTS["exp003"])
    exp004 = _load_json(INPUTS["exp004_results"])
    exp004_cases = _load_json(INPUTS["exp004_cases"])
    exp005_results = _load_json(INPUTS["exp005_results"])
    exp005_metadata = _load_json(INPUTS["exp005_metadata"])

    cases_payload = _build_cases(exp004_cases)
    case_lookup = {case["studentId"]: case for case in cases_payload}

    students_payload = _build_students(snapshots, students, final_results, case_lookup)
    cohort = _build_cohort(students_payload)
    xai = _build_xai_section(exp004)
    directions = _build_directions(INPUTS["exp004_global_csv"])
    experiments = _build_experiments(
        exp001, exp002, exp003, exp004, exp005_results, exp005_metadata
    )

    payload = {
        "schemaVersion": PAYLOAD_SCHEMA_VERSION,
        "sourceArtifacts": {
            key: str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            for key, path in INPUTS.items()
        },
        "cohort": cohort,
        "students": students_payload,
        "xai": xai,
        "featureDirections": directions,
        "cases": cases_payload,
        "experiments": experiments,
        "limitations": [
            "Predictions and explanations come from a frozen research prototype.",
            "Synthetic internal evidence is not institutional validation.",
            "Full Twin representation was not justified under the current setup.",
            "overall_mastery is informative but redundant with cumulative LMS scores.",
            "XAI outputs describe model behavior, not causal effects.",
            "OULAD transfer evidence is mixed rather than confirmatory.",
        ],
    }

    return payload


def main() -> Path:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return OUTPUT


if __name__ == "__main__":
    out_path = main()
    print(f"Wrote research demo payload to {out_path}")
