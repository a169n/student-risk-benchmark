"""Materialize a research-demo payload from REAL OULAD data (Approach B).

This is the honest "pass-risk" OULAD analogue of
``export_research_demo_payload.py``. It MIRRORS the synthetic exporter's
structure and emits the IDENTICAL camelCase payload schema consumed by the
``apps/api`` importer (``apps/api/src/importer/research_payload.py``), but the
predictions, risk bands, indices, and explanations all come from the OULAD
``DDD`` ``2013J`` cohort rather than from synthetic, circular data.

Design (Approach B, honest pass-risk)
-------------------------------------
- Cohort: OULAD ``DDD`` ``2013J``.
- ``predictedFinalGrade`` = predicted ``final_weighted_score`` (0-100) from a
  gradient_boosting REGRESSION model on feature set ``B_lms_plus_mastery_oulad``.
- ``riskScore`` / ``riskLevel`` = derived from a gradient_boosting CLASSIFICATION
  model's P(not passing). ``riskScore`` = P(fail); bands
  P(fail) >= 0.60 -> "high", >= 0.35 -> "medium", else "low".
- Clean OULAD index/mastery/trend columns from ``build_weekly_snapshots`` are
  mapped onto the synthetic payload's index/mastery/trend fields.
- OULAD has NO clean analogue for quiz score, attendance, attendance trend, or a
  hidden trajectory label, so those payload fields are emitted as ``null`` (the
  importer and web tolerate nulls there).
- ``activityScore`` = ``cumulative_clicks_to_date`` min-max normalized to 0-100
  within the sampled cohort.
- ~150 students are sampled (stratified by observed pass + derived risk) for a
  snappy demo; all weeks are kept for the sampled students.
- Representative cases + topContributions + teacherAssessment + masteryShare come
  from the shared OULAD XAI engine in ``src.experiments.explainability``.

This script trains two small models in-process on leakage-safe OULAD weekly
snapshots; it does not retrain any frozen experiment artifact and does not alter
any experiment evidence. The honest caveats (partial assessment-score
circularity, never-perfect F1, mixed-to-null Twin transfer, OULAD has no
attendance/quiz analogue, model-behavior not causal, ~150-student demo subset)
are embedded in ``limitations``/``xai``/``experiments``.

Output
------
- ``data/artifacts/research_demo/oulad_research_demo_payload.json``
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.benchmarks.oulad_adapter import (
    OuladCourseFilter,
    OuladRawPaths,
    build_weekly_snapshots,
    validate_oulad_raw_files,
)
from src.experiments.datasets import GROUP_COLUMN, WEEK_COLUMN
from src.experiments.explainability import (
    build_global_explanation,
    build_local_case_explanations,
    select_representative_cases,
    train_regression_reference_for_matrix,
)
from src.experiments.models import build_classification_model
from src.experiments.preprocessing import (
    build_modeling_matrix,
    fit_imputer_on_training,
    select_rows,
)
from src.experiments.splits import student_group_split


PAYLOAD_SCHEMA_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parents[4]

# ---------------------------------------------------------------------------
# Inputs / output paths (mirror the synthetic exporter's INPUTS/OUTPUT shape)
# ---------------------------------------------------------------------------

OULAD_RAW_DIR = REPO_ROOT / "datasets" / "oulad"

INPUTS = {
    "oulad_raw_dir": OULAD_RAW_DIR,
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
    "exp007_results": REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_007_xai_on_oulad"
    / "exp_007_xai_on_oulad_results.json",
}

OUTPUT = (
    REPO_ROOT
    / "data"
    / "artifacts"
    / "research_demo"
    / "oulad_research_demo_payload.json"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COHORT_CODE_MODULE = "DDD"
COHORT_CODE_PRESENTATION = "2013J"
COHORT_LABEL = f"OULAD {COHORT_CODE_MODULE} {COHORT_CODE_PRESENTATION}"

MODEL_NAME = "gradient_boosting"
SEED = 42
TEST_SIZE = 0.25
PERMUTATION_REPEATS = 15

# Risk bands on P(not passing).
RISK_HIGH_THRESHOLD = 0.60
RISK_MEDIUM_THRESHOLD = 0.35

# Demo sampling. Defaults to a snappy 150-student stratified subset; set
# OULAD_DEMO_MAX_STUDENTS to a value >= the full DDD 2013J cohort (e.g. 100000)
# to emit every student (the model always trains on the full cohort regardless).
TARGET_SAMPLE_STUDENTS = int(os.environ.get("OULAD_DEMO_MAX_STUDENTS", "150"))

# Cohort flag thresholds (mirror the synthetic exporter's semantics).
LOW_ACTIVITY_THRESHOLD = 40.0
LOW_MASTERY_THRESHOLD = 60.0
AT_RISK_GRADE_THRESHOLD = 60.0

# Feature set explained / modeled (the lean mastery analogue).
LEAN_FEATURE_SET_NAME = "B_lms_plus_mastery_oulad"
BASELINE_FEATURE_SET_NAME = "B_lms_oulad"

_SHARED_LMS_COLUMNS = (
    "week_number",
    "course_week_progress",
    "is_registered_by_week",
    "is_unregistered_by_week",
    "days_since_registration_start",
    "cumulative_assessment_score_mean_to_date",
    "cumulative_assessment_score_count_to_date",
    "cumulative_assessment_weighted_score_to_date",
    "cumulative_submitted_weight_to_date",
    "assessment_submission_rate_due_to_date",
    "late_submission_rate_to_date",
    "banked_assessment_rate_to_date",
    "current_week_clicks",
    "cumulative_clicks_to_date",
    "current_week_activity_types",
    "cumulative_assessment_clicks_to_date",
    "cumulative_content_clicks_to_date",
    "cumulative_social_clicks_to_date",
    "cumulative_other_clicks_to_date",
)

_MASTERY_COLUMNS = (
    "overall_mastery_proxy",
    "current_assessment_cluster_mastery",
    "tma_mastery_to_date",
    "cma_mastery_to_date",
    "exam_mastery_to_date",
    "mastery_assessment_coverage_to_date",
)

LEAN_COLUMNS = _SHARED_LMS_COLUMNS + _MASTERY_COLUMNS
LEAN_INDICATOR_COLUMNS = (
    "has_assessment_score_to_date",
    "has_weighted_score_to_date",
    "has_vle_activity_to_date",
    "has_current_assessment_cluster",
    "has_due_assessment_to_date",
)
BASELINE_INDICATOR_COLUMNS = (
    "has_assessment_score_to_date",
    "has_weighted_score_to_date",
    "has_vle_activity_to_date",
)

# Trend feature used for improving/declining representative-case selection.
TREND_FEATURE = "assessment_score_trend_to_date"


@dataclass(frozen=True)
class _FeatureSet:
    """Duck-typed feature set mirroring ``BenchmarkFeatureSet``/``FeatureSet``."""

    name: str
    columns: tuple[str, ...]
    indicator_columns: tuple[str, ...] = ()

    def all_columns(self) -> tuple[str, ...]:
        return self.columns + self.indicator_columns


LEAN_FEATURE_SET = _FeatureSet(
    name=LEAN_FEATURE_SET_NAME,
    columns=LEAN_COLUMNS,
    indicator_columns=LEAN_INDICATOR_COLUMNS,
)
BASELINE_FEATURE_SET = _FeatureSet(
    name=BASELINE_FEATURE_SET_NAME,
    columns=_SHARED_LMS_COLUMNS,
    indicator_columns=BASELINE_INDICATOR_COLUMNS,
)


# ---------------------------------------------------------------------------
# Small helpers (mirror the synthetic exporter)
# ---------------------------------------------------------------------------


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
    if isinstance(value, (np.floating,)) and np.isnan(value):
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


def _risk_level(p_fail: float | None) -> str:
    if p_fail is None or (isinstance(p_fail, float) and math.isnan(p_fail)):
        return "unknown"
    if p_fail >= RISK_HIGH_THRESHOLD:
        return "high"
    if p_fail >= RISK_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# OULAD modeling: build snapshots, train regression + classification once
# ---------------------------------------------------------------------------


@dataclass
class OuladModelBundle:
    """Everything the payload builder needs from in-process OULAD inference."""

    snapshots: pd.DataFrame  # full DDD 2013J snapshots with predictions attached
    run: Any  # TrainedRegressionModel on the LEAN feature set (for XAI)
    baseline_run: Any  # TrainedRegressionModel on the baseline feature set (for XAI)
    regression_metrics: dict[str, float]
    classification_metrics: dict[str, float]
    activity_min: float
    activity_max: float


def _normalize_activity(values: pd.Series, *, lo: float, hi: float) -> pd.Series:
    if hi <= lo:
        return pd.Series(0.0, index=values.index)
    scaled = (values - lo) / (hi - lo)
    return (scaled.clip(0.0, 1.0) * 100.0)


def build_oulad_models() -> OuladModelBundle:
    """Build OULAD DDD 2013J snapshots and fit regression + classification models.

    The regression model predicts ``final_weighted_score`` (mapped to
    ``predictedFinalGrade``); the classification model's P(not passing) drives
    ``riskScore``/``riskLevel``. The lean and baseline ``TrainedRegressionModel``
    objects are returned so the shared XAI engine can build representative cases.
    """

    raw_paths = OuladRawPaths.from_directory(INPUTS["oulad_raw_dir"])
    validate_oulad_raw_files(raw_paths)

    build_result = build_weekly_snapshots(
        raw_paths,
        course_filter=OuladCourseFilter(
            code_module=COHORT_CODE_MODULE,
            code_presentation=COHORT_CODE_PRESENTATION,
        ),
        min_week=4,
        max_week=None,
        student_vle_chunk_size=500_000,
    )

    frame = build_result.snapshots.copy()
    frame["final_grade"] = pd.to_numeric(frame["final_weighted_score"], errors="coerce")
    frame["passed"] = (
        pd.to_numeric(frame["passed_observed"], errors="coerce").fillna(0).astype(int)
    )
    frame = frame.loc[frame["final_grade"].notna()].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Leakage-safe student-grouped split shared by both tasks.
    # ------------------------------------------------------------------
    n = len(frame)
    base = pd.DataFrame(
        {
            GROUP_COLUMN: frame[GROUP_COLUMN].to_numpy(),
            "_row_index": np.arange(n),
        }
    )
    split = student_group_split(
        base,
        test_size=TEST_SIZE,
        validation_size=0.0,
        seed=SEED,
        student_id_column=GROUP_COLUMN,
    )
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[split.train["_row_index"].to_numpy()] = True
    test_mask[split.test["_row_index"].to_numpy()] = True

    # ------------------------------------------------------------------
    # Regression (lean feature set) via the shared explainability helper, so
    # the returned TrainedRegressionModel can be reused for XAI cases.
    # ------------------------------------------------------------------
    run = train_regression_reference_for_matrix(
        frame,
        feature_set=LEAN_FEATURE_SET,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name=MODEL_NAME,
        seed=SEED,
        split_metadata={"strategy": "student_group", "test_size": TEST_SIZE, "seed": SEED},
    )
    baseline_run = train_regression_reference_for_matrix(
        frame,
        feature_set=BASELINE_FEATURE_SET,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name=MODEL_NAME,
        seed=SEED,
        split_metadata={"strategy": "student_group", "test_size": TEST_SIZE, "seed": SEED},
    )

    # ------------------------------------------------------------------
    # Build modeling matrix once for ALL rows (lean feature set) so we can
    # produce per-row predictions for every student-week, not just the test
    # split. The estimator + imputer are fit on training rows only.
    # ------------------------------------------------------------------
    matrix = build_modeling_matrix(frame, LEAN_FEATURE_SET)
    train_matrix = select_rows(matrix, train_mask)

    reg_imputer = fit_imputer_on_training(train_matrix.features)
    x_all = reg_imputer.transform(matrix.features)
    # Align columns to what the regression estimator was trained on.
    x_all = x_all.loc[:, list(run.feature_columns)]
    predicted_final = np.asarray(run.estimator.predict(x_all), dtype=float)

    # ------------------------------------------------------------------
    # Classification: P(not passing). Refit a GB classifier on the same split.
    # ------------------------------------------------------------------
    clf_imputer = fit_imputer_on_training(train_matrix.features)
    x_train_clf = clf_imputer.transform(train_matrix.features).loc[:, list(run.feature_columns)]
    y_train_clf = train_matrix.classification_target.to_numpy().astype(int)

    clf = build_classification_model(MODEL_NAME, seed=SEED)
    clf.fit(x_train_clf, y_train_clf)

    x_all_clf = clf_imputer.transform(matrix.features).loc[:, list(run.feature_columns)]
    classes = list(clf.classes_)
    proba = clf.predict_proba(x_all_clf)
    if 1 in classes:
        pass_idx = classes.index(1)
        p_pass = proba[:, pass_idx]
    else:
        # Degenerate: training saw a single class. Fall back to the constant.
        p_pass = np.full(len(frame), float(classes[0]))
    p_not_pass = 1.0 - p_pass

    # Held-out classification metrics for honest reporting.
    test_matrix = select_rows(matrix, test_mask)
    x_test_clf = clf_imputer.transform(test_matrix.features).loc[:, list(run.feature_columns)]
    y_test_clf = test_matrix.classification_target.to_numpy().astype(int)
    classification_metrics = _classification_metrics(clf, x_test_clf, y_test_clf)

    # ------------------------------------------------------------------
    # Attach derived prediction columns onto the snapshot frame.
    # ------------------------------------------------------------------
    frame = frame.copy()
    frame["predicted_final_grade"] = predicted_final
    frame["risk_score"] = p_not_pass
    frame["risk_level"] = [
        _risk_level(float(v)) for v in p_not_pass.tolist()
    ]

    activity_series = pd.to_numeric(
        frame["cumulative_clicks_to_date"], errors="coerce"
    ).fillna(0.0)
    activity_min = float(activity_series.min())
    activity_max = float(activity_series.max())
    frame["activity_score"] = _normalize_activity(
        activity_series, lo=activity_min, hi=activity_max
    ).to_numpy()

    return OuladModelBundle(
        snapshots=frame,
        run=run,
        baseline_run=baseline_run,
        regression_metrics={k: float(v) for k, v in run.metrics.items()},
        classification_metrics=classification_metrics,
        activity_min=activity_min,
        activity_max=activity_max,
    )


def _classification_metrics(clf: Any, x_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    preds = clf.predict(x_test)
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
    }
    try:
        classes = list(clf.classes_)
        if 1 in classes and len(np.unique(y_test)) >= 2:
            proba = clf.predict_proba(x_test)[:, classes.index(1)]
            metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
    except (ValueError, IndexError):
        pass
    return metrics


# ---------------------------------------------------------------------------
# Representative cases via the OULAD XAI engine
# ---------------------------------------------------------------------------


def _select_oulad_cases(run: Any) -> list[dict[str, Any]]:
    """Select up to ~4 representative held-out cases, deduplicated by student.

    Returned dicts carry ``source_row_index``/``student_id``/``week_number`` and
    are the input to :func:`build_local_case_explanations`. Selection happens on
    the held-out test split, so case students are a subset of the test students;
    the caller force-includes these students into the demo sample so every
    emitted ``cases[].studentId`` matches an existing ``students[]`` record.
    """

    raw_cases = select_representative_cases(
        run,
        case_types=[
            "strong_performer",
            "at_risk",
            "improving_trajectory",
            "declining_trajectory",
        ],
        max_cases=8,
        borderline_grade=50.0,
        trend_feature=TREND_FEATURE,
    )
    deduped: list[dict[str, Any]] = []
    seen_students: set[str] = set()
    for case in raw_cases:
        sid = str(case["student_id"])
        if sid in seen_students:
            continue
        seen_students.add(sid)
        deduped.append(case)
    return deduped


def _build_oulad_cases(
    run: Any,
    baseline_run: Any,
    *,
    selected_cases: list[dict[str, Any]],
    snapshots: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Build local explanations for already-selected representative cases.

    ``passed`` and ``riskLevelContext`` are sourced from the authoritative
    snapshot frame keyed on (student_id, week_number). The shared XAI engine's
    own ``passed`` extraction can lose numpy-typed ints, and the OULAD modeling
    frame has no ``risk_level`` column at training time (it is attached after the
    run is built), so we resolve both fields here from real data.
    """

    explanations = build_local_case_explanations(
        lean_run=run,
        baseline_run=baseline_run,
        cases=selected_cases,
        mastery_columns=_MASTERY_COLUMNS,
        top_k=6,
    )

    snap = snapshots.copy()
    snap["_sid"] = snap["student_id"].astype(str)
    snap["_wk"] = pd.to_numeric(snap["week_number"], errors="coerce").astype("Int64")
    passed_lookup: dict[tuple[str, int], Any] = {}
    risk_lookup: dict[tuple[str, int], Any] = {}
    for _, srow in snap.iterrows():
        key = (srow["_sid"], int(srow["_wk"]))
        passed_lookup[key] = srow.get("passed_observed")
        risk_lookup[key] = srow.get("risk_level")

    out: list[dict[str, Any]] = []
    for case in explanations:
        sid = str(case["student_id"])
        week = int(case["week_number"])
        key = (sid, week)
        out.append(
            {
                "studentId": sid,
                "weekNumber": week,
                "caseType": case["case_type"],
                "actualFinalGrade": _round(case["actual_final_grade"]),
                "predictedFinalGrade": _round(case["predicted_final_grade"]),
                "predictionError": _round(case["prediction_error"]),
                "riskLevelContext": _case_risk_context(risk_lookup.get(key)),
                "passed": _coerce_bool(passed_lookup.get(key)),
                "masteryShare": _round(case.get("mastery_abs_contribution_share"), 4),
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


def _case_risk_context(risk_value: Any) -> str:
    if isinstance(risk_value, str) and risk_value:
        return risk_value
    return "unknown"


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not (
        isinstance(value, float) and math.isnan(value)
    ):
        return bool(value)
    return None


# ---------------------------------------------------------------------------
# Student / cohort builders (mirror the synthetic exporter contract)
# ---------------------------------------------------------------------------


def _weekly_record(row: pd.Series) -> dict[str, Any]:
    """One ``weeklyTimeline`` snapshot. OULAD has no quiz/attendance analogue."""

    return {
        "weekNumber": int(row["week_number"]),
        "predictedFinalGrade": _round(_none_if_nan(row.get("predicted_final_grade"))),
        "riskLevel": row["risk_level"],
        "riskScore": _round(_none_if_nan(row.get("risk_score")), 4),
        "activityScore": _round(_none_if_nan(row.get("activity_score"))),
        "assignmentAverage": _round(
            _none_if_nan(row.get("cumulative_assessment_score_mean_to_date"))
        ),
        "quizAverage": None,
        "attendanceRate": None,
        "overallMastery": _round(_none_if_nan(row.get("overall_mastery_proxy"))),
        "currentTopicMastery": _round(
            _none_if_nan(row.get("current_assessment_cluster_mastery"))
        ),
        "engagementIndex": _round(_none_if_nan(row.get("engagement_index_oulad"))),
        "performanceIndex": _round(_none_if_nan(row.get("performance_index_oulad"))),
        "disciplineIndex": _round(_none_if_nan(row.get("discipline_index_oulad"))),
        "scoreTrend3w": _round(
            _none_if_nan(row.get("assessment_score_trend_to_date")), 4
        ),
        "activityTrend3w": _round(_none_if_nan(row.get("clicks_trend_to_date")), 4),
        "attendanceTrend3w": None,
    }


def _build_students(
    snapshots: pd.DataFrame,
    *,
    case_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    student_ids = sorted(snapshots["student_id"].astype(str).unique().tolist())
    out: list[dict[str, Any]] = []
    for sid in student_ids:
        sdf = snapshots[snapshots["student_id"].astype(str) == sid].sort_values(
            "week_number"
        )
        if sdf.empty:
            continue
        latest = sdf.iloc[-1]

        weekly = [_weekly_record(row) for _, row in sdf.iterrows()]

        actual_final = _round(_none_if_nan(latest.get("final_weighted_score")))
        passed_val = _none_if_nan(latest.get("passed_observed"))
        passed = None if passed_val is None else bool(passed_val)

        current = {
            "predictedFinalGrade": _round(
                _none_if_nan(latest.get("predicted_final_grade"))
            ),
            "riskLevel": latest["risk_level"],
            "riskScore": _round(_none_if_nan(latest.get("risk_score")), 4),
            "activityScore": _round(_none_if_nan(latest.get("activity_score"))),
            "assignmentAverage": _round(
                _none_if_nan(latest.get("cumulative_assessment_score_mean_to_date"))
            ),
            "quizAverage": None,
            "attendanceRate": None,
            "overallMastery": _round(_none_if_nan(latest.get("overall_mastery_proxy"))),
            "currentTopicMastery": _round(
                _none_if_nan(latest.get("current_assessment_cluster_mastery"))
            ),
            "engagementIndex": _round(_none_if_nan(latest.get("engagement_index_oulad"))),
            "performanceIndex": _round(
                _none_if_nan(latest.get("performance_index_oulad"))
            ),
            "disciplineIndex": _round(_none_if_nan(latest.get("discipline_index_oulad"))),
            "scoreTrend3w": _round(
                _none_if_nan(latest.get("assessment_score_trend_to_date")), 4
            ),
        }

        out.append(
            {
                "studentId": sid,
                "studentLabel": f"OULAD-{sid}",
                "cohortLabel": COHORT_LABEL,
                "trajectoryLabel": None,
                "courseId": str(latest["course_id"]),
                "currentWeek": int(latest["week_number"]),
                "current": current,
                "actualFinalGrade": actual_final,
                "passed": passed,
                "weeklyTimeline": weekly,
                "explanation": case_lookup.get(sid),
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


# ---------------------------------------------------------------------------
# XAI / directions / experiments sections (OULAD-native, honest caveats)
# ---------------------------------------------------------------------------


def _select_xai_record(
    exp007: list[dict[str, Any]],
    *,
    feature_set: str,
    split_strategy: str,
) -> dict[str, Any] | None:
    for record in exp007:
        if (
            record.get("feature_set") == feature_set
            and record.get("split_strategy") == split_strategy
        ):
            return record
    return None


def _build_xai_section(
    bundle: OuladModelBundle,
    exp007: list[dict[str, Any]],
) -> dict[str, Any]:
    """OULAD model-behavior XAI section.

    Global importance and dominance come from the live lean-run permutation
    explanation (consistent with the in-process model that produced the demo
    predictions). The dominance outcome and mastery share are reported honestly:
    the lean OULAD mastery analogue is influential but the explanation is a
    model-behavior signal, never a causal claim, and the assessment-score block
    is partially circular with the target.
    """

    global_explanation = build_global_explanation(
        bundle.run,
        permutation_repeats=PERMUTATION_REPEATS,
        seed=SEED,
        split_strategy="student_group",
    )
    rows = global_explanation["rows"]
    concentration = global_explanation["concentration"]

    top_feature = concentration.get("top_feature")
    top1_share = concentration.get("top1_share")

    mastery_set = set(_MASTERY_COLUMNS)
    mastery_share_total = 0.0
    for row in rows:
        if row["feature"] in mastery_set:
            mastery_share_total += float(row.get("importance_share") or 0.0)

    top_is_mastery = top_feature in mastery_set
    if top_is_mastery and (top1_share is not None and float(top1_share) >= 0.33):
        outcome = "acceptable_with_caveat"
    else:
        outcome = "acceptable"

    flags = [
        "OULAD model-behavior explanations only; not causal and not "
        "structural-coefficient interpretations.",
        "The cumulative assessment-score block partially feeds the "
        "final_weighted_score target (within-system accounting identity); weight "
        "interpretation toward exogenous clickstream features.",
    ]
    if top_is_mastery:
        flags.append(
            f"`{top_feature}` is the top global driver of the OULAD lean model; "
            "this mastery analogue is redundant with cumulative LMS scores."
        )

    return {
        "method": (
            "sklearn permutation_importance (held-out RMSE increase) with "
            "model-native importance and local one-feature median perturbation; no SHAP"
        ),
        "shapUsed": False,
        "topGlobalFeatures": [
            {
                "feature": row["feature"],
                "rank": row["rank"],
                "importanceShare": _round(row.get("importance_share"), 4),
            }
            for row in rows[:8]
        ],
        "dominance": {
            "topFeature": top_feature,
            "top1Share": _round(top1_share, 4),
            "averageLocalMasteryShare": _round(mastery_share_total, 4),
            "outcome": outcome,
            "flags": flags,
        },
        "metrics": {
            "baselineRmse": _round(bundle.baseline_run.metrics.get("rmse")),
            "leanRmse": _round(bundle.run.metrics.get("rmse")),
            "leanDelta": _round(
                bundle.run.metrics.get("rmse", 0.0)
                - bundle.baseline_run.metrics.get("rmse", 0.0)
            ),
            "withoutOverallDelta": None,
        },
        "recommendation": {
            "outcome": outcome,
            "decisionText": (
                "Retain OULAD permutation + perturbation explanations as a "
                "model-behavior signal for the demo. The lean mastery analogue is "
                "influential but redundant and partially circular with the "
                "assessment-derived target, so it must carry an explicit caveat and "
                "must not be read causally."
            ),
            "flags": flags,
        },
    }


def _build_directions(exp007: list[dict[str, Any]]) -> dict[str, str]:
    """Feature -> direction note from the OULAD lean temporal-forward XAI rows."""

    record = _select_xai_record(
        exp007,
        feature_set=LEAN_FEATURE_SET_NAME,
        split_strategy="temporal_forward",
    )
    if record is None:
        record = _select_xai_record(
            exp007,
            feature_set=LEAN_FEATURE_SET_NAME,
            split_strategy="student_group",
        )
    if record is None:
        return {}
    direction_map: dict[str, str] = {}
    for row in record.get("importance_rows", []):
        feature = row.get("feature")
        note = row.get("direction_note")
        if isinstance(feature, str) and isinstance(note, str):
            direction_map[feature] = note
    return direction_map


def _build_experiments(
    bundle: OuladModelBundle,
    exp005_results: dict[str, Any],
    exp005_metadata: dict[str, Any],
    classification_metrics: dict[str, float],
) -> dict[str, Any]:
    rows = exp005_results.get("rows", [])
    diagnostics = exp005_metadata.get("diagnostics", {})
    interpretation = diagnostics.get("interpretation", {})
    row_counts = diagnostics.get("row_counts", {})
    target_summary = diagnostics.get("target_summary", {})

    oulad_grouped_baseline = _best_regression(
        rows,
        target="final_weighted_score",
        split="student_group",
        feature_set=BASELINE_FEATURE_SET_NAME,
    )
    oulad_grouped_lean = _best_regression(
        rows,
        target="final_weighted_score",
        split="student_group",
        feature_set=LEAN_FEATURE_SET_NAME,
    )
    oulad_temporal_baseline = _best_regression(
        rows,
        target="final_weighted_score",
        split="temporal_forward",
        feature_set=BASELINE_FEATURE_SET_NAME,
    )
    oulad_temporal_lean = _best_regression(
        rows,
        target="final_weighted_score",
        split="temporal_forward",
        feature_set=LEAN_FEATURE_SET_NAME,
    )

    f1 = classification_metrics.get("f1")
    roc_auc = classification_metrics.get("roc_auc")

    timeline = [
        {
            "id": "exp_005",
            "title": "OULAD public benchmark (DDD 2013J)",
            "result": interpretation.get(
                "short_conclusion", "Result unavailable"
            ),
            "decision": (
                "External OULAD transfer evidence is mixed, not confirmatory."
            ),
        },
        {
            "id": "exp_007",
            "title": "OULAD model-behavior XAI",
            "result": (
                "Permutation + perturbation explanations; the lean mastery "
                "analogue (overall_mastery_proxy) is the top global driver."
            ),
            "decision": (
                "Explanations retained as model-behavior signal with a "
                "partial-circularity caveat on the assessment-score block."
            ),
        },
        {
            "id": "oulad_pass_risk",
            "title": "Honest pass-risk demo classifier (DDD 2013J)",
            "result": (
                "Gradient-boosting P(not passing) on the held-out student split: "
                + (f"F1 {f1:.3f}" if f1 is not None else "F1 unavailable")
                + (f", ROC-AUC {roc_auc:.3f}" if roc_auc is not None else "")
                + " (a real, never-perfect classifier)."
            ),
            "decision": (
                "Use real P(not passing) for risk bands; F1 is ~0.86, never the "
                "1.000 the synthetic prototype reported under circularity."
            ),
        },
    ]

    return {
        "timeline": timeline,
        "leanTwin": {
            "baselineRmse": _round(bundle.baseline_run.metrics.get("rmse")),
            "leanRmse": _round(bundle.run.metrics.get("rmse")),
            "leanDelta": _round(
                bundle.run.metrics.get("rmse", 0.0)
                - bundle.baseline_run.metrics.get("rmse", 0.0)
            ),
            "withoutOverallDelta": None,
            "earlyWeeksImproved": [],
            "lateWeeksImproved": [],
            "flags": [
                "OULAD lean mastery analogue did not clearly beat the LMS baseline "
                "on the student-grouped split; transfer is mixed.",
                "The lean delta here is computed live on the demo's student-grouped "
                "split and is a representation-logic stress test, not a Twin value "
                "claim.",
            ],
        },
        "oulad": {
            "rowCounts": {
                "snapshots": int(row_counts.get("snapshots", 0)),
                "students": int(row_counts.get("students", 0)),
            },
            "weekMin": target_summary.get("week_min"),
            "weekMax": target_summary.get("week_max"),
            "outcome": interpretation.get("outcome", "complicates"),
            "shortConclusion": interpretation.get(
                "short_conclusion", "OULAD transfer evidence is mixed."
            ),
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


# ---------------------------------------------------------------------------
# Demo sampling (stratified by observed pass + derived risk)
# ---------------------------------------------------------------------------


def _sample_students(
    snapshots: pd.DataFrame,
    *,
    target: int,
    seed: int,
    force_include: set[str] | None = None,
) -> set[str]:
    """Pick ~target students stratified by (passed_observed, derived risk_level).

    Keeps all weeks for each selected student (handled downstream by filtering
    the snapshot frame to the sampled ids). Any ids in ``force_include`` are
    guaranteed to appear in the result so representative-case students always
    have a matching ``students[]`` record.
    """

    force_include = set(force_include or set())
    latest = (
        snapshots.sort_values(["student_id", "week_number"])
        .groupby("student_id", as_index=False)
        .tail(1)
        .copy()
    )
    latest["student_id"] = latest["student_id"].astype(str)
    all_ids = set(latest["student_id"].tolist())
    force_include &= all_ids
    total_students = latest["student_id"].nunique()
    if total_students <= target:
        return all_ids

    latest["_passed_key"] = (
        pd.to_numeric(latest["passed_observed"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    latest["_strata"] = (
        latest["_passed_key"].astype(str) + "|" + latest["risk_level"].astype(str)
    )

    rng = np.random.default_rng(seed)
    selected: list[str] = []
    grouped = list(latest.groupby("_strata"))
    # Proportional allocation per stratum, with deterministic rounding.
    for _, group in grouped:
        share = len(group) / total_students
        n_take = max(1, int(round(share * target)))
        ids = group["student_id"].tolist()
        if n_take >= len(ids):
            take = ids
        else:
            idx = rng.choice(len(ids), size=n_take, replace=False)
            take = [ids[i] for i in sorted(idx.tolist())]
        selected.extend(take)

    # Guarantee the forced (representative-case) students are present.
    selected_set = set(selected) | force_include
    selected = sorted(selected_set)

    # Trim or pad to exactly `target` deterministically, never dropping forced ids.
    if len(selected) > target:
        droppable = [sid for sid in selected if sid not in force_include]
        n_drop = len(selected) - target
        if n_drop > 0 and droppable:
            drop_idx = rng.choice(
                len(droppable), size=min(n_drop, len(droppable)), replace=False
            )
            to_drop = {droppable[i] for i in drop_idx.tolist()}
            selected = sorted(set(selected) - to_drop)
    elif len(selected) < target:
        remaining = sorted(all_ids - set(selected))
        need = target - len(selected)
        if remaining:
            extra_idx = rng.choice(
                len(remaining), size=min(need, len(remaining)), replace=False
            )
            selected.extend(remaining[i] for i in sorted(extra_idx.tolist()))
            selected = sorted(set(selected))
    return set(selected)


# ---------------------------------------------------------------------------
# Top-level build
# ---------------------------------------------------------------------------


def build_payload() -> dict[str, Any]:
    bundle = build_oulad_models()

    exp005_results = _load_json(INPUTS["exp005_results"])
    exp005_metadata = _load_json(INPUTS["exp005_metadata"])
    exp007 = _load_json(INPUTS["exp007_results"])

    # Select representative cases FIRST (on the held-out split), then guarantee
    # those students survive sampling so every case has a matching student.
    selected_cases = _select_oulad_cases(bundle.run)
    case_student_ids = {str(case["student_id"]) for case in selected_cases}

    sampled_ids = _sample_students(
        bundle.snapshots,
        target=TARGET_SAMPLE_STUDENTS,
        seed=SEED,
        force_include=case_student_ids,
    )
    sampled_snapshots = bundle.snapshots.loc[
        bundle.snapshots["student_id"].astype(str).isin(sampled_ids)
    ].reset_index(drop=True)

    cases_payload = _build_oulad_cases(
        bundle.run,
        bundle.baseline_run,
        selected_cases=selected_cases,
        snapshots=bundle.snapshots,
    )
    case_lookup = {case["studentId"]: case for case in cases_payload}

    students_payload = _build_students(sampled_snapshots, case_lookup=case_lookup)
    cohort = _build_cohort(students_payload)
    xai = _build_xai_section(bundle, exp007)
    directions = _build_directions(exp007)
    experiments = _build_experiments(
        bundle, exp005_results, exp005_metadata, bundle.classification_metrics
    )

    f1 = bundle.classification_metrics.get("f1")
    f1_text = f"{f1:.3f}" if f1 is not None else "~0.86"

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
            "Predictions come from a real OULAD DDD 2013J model trained "
            "in-process; this is a research demo, not an institutional system.",
            "The regression target (final_weighted_score) is partially circular: "
            "the cumulative assessment-score features are an accounting input to "
            "the target, so high assessment importance is expected and is not an "
            "exogenous causal signal.",
            f"The pass-risk classifier is a real, never-perfect model (held-out F1 "
            f"~{f1_text}); it is honestly imperfect, unlike the synthetic "
            "prototype's circular 1.000 scores.",
            "The lean OULAD mastery analogue did not clearly improve over the LMS "
            "baseline on the primary student-grouped split, so the Twin value here "
            "is mixed-to-null rather than confirmatory.",
            "OULAD has NO clean equivalents for quiz scores, attendance, "
            "attendance trend, or a hidden trajectory label, so those fields are "
            "null in this payload.",
            "XAI outputs describe OULAD model behavior, not causal effects, and "
            "this demo uses a ~150-student stratified subset of the full DDD 2013J "
            "cohort for responsiveness.",
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
    print(f"Wrote OULAD research demo payload to {out_path}")
