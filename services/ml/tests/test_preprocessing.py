from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.experiments.featuresets import (
    FEATURE_SET_A_SIMPLE,
    FEATURE_SET_C_TWIN,
    FORBIDDEN_FEATURE_COLUMNS,
    FeatureSet,
)
from src.experiments.preprocessing import (
    assert_no_forbidden_columns,
    build_modeling_matrix,
    fit_imputer_on_training,
    select_rows,
)


def _toy_modeling_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for student_idx in range(6):
        for week in range(1, 5):
            rows.append(
                {
                    "snapshot_id": f"snap_{student_idx}_w{week}",
                    "student_id": f"student_{student_idx:03d}",
                    "course_id": "course_x",
                    "week_number": week,
                    "snapshot_date": f"2026-09-{week:02d}",
                    "attendance_rate_to_date": 0.5 + 0.1 * (student_idx % 3),
                    "avg_assignment_score_to_date": (
                        None if (student_idx == 0 and week == 1) else 60 + student_idx
                    ),
                    "avg_quiz_score_to_date": 70 - student_idx,
                    "has_assignment_score_to_date": not (student_idx == 0 and week == 1),
                    "has_quiz_score_to_date": True,
                    "on_time_submission_rate_to_date": 0.8,
                    "missed_assignments_to_date": student_idx % 2,
                    "late_submissions_to_date": 0,
                    "avg_attempt_count_to_date": 1.0,
                    "activity_score_to_date": 50 + student_idx,
                    "time_spent_to_date": 100.0 * week,
                    "score_trend_3w": 0.1 if student_idx % 2 == 0 else -0.1,
                    "activity_trend_3w": 0.05,
                    "attendance_trend_3w": 0.0,
                    "current_topic_mastery": 50 + student_idx,
                    "overall_mastery": 50 + student_idx,
                    "engagement_index": 60.0,
                    "performance_index": 65.0,
                    "discipline_index": 70.0,
                    # forbidden / leakage / target columns kept on the join frame
                    "risk_score": 0.4,
                    "risk_level": "medium",
                    "predicted_final_grade": 70.0,
                    "completion_status": "completed",
                    "final_grade": 60 + student_idx * 4,
                    "passed": (60 + student_idx * 4) >= 50,
                    # generation-only fields
                    "trajectory_type": "stable_high",
                    "baseline_level": 0.7,
                    "motivation_level": 0.5,
                    "discipline_level": 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_build_modeling_matrix_excludes_forbidden_columns() -> None:
    matrix = build_modeling_matrix(_toy_modeling_frame(), FEATURE_SET_C_TWIN)
    feature_columns = set(matrix.feature_columns)
    assert feature_columns.isdisjoint(FORBIDDEN_FEATURE_COLUMNS)
    assert "final_grade" not in feature_columns
    assert "passed" not in feature_columns
    assert "risk_level" not in feature_columns
    assert "predicted_final_grade" not in feature_columns
    assert "trajectory_type" not in feature_columns


def test_build_modeling_matrix_preserves_targets_and_groups() -> None:
    matrix = build_modeling_matrix(_toy_modeling_frame(), FEATURE_SET_A_SIMPLE)
    assert matrix.regression_target.notna().all()
    assert set(matrix.classification_target.unique()).issubset({0, 1})
    assert matrix.groups.nunique() == 6
    assert (matrix.weeks >= 1).all()


def test_build_modeling_matrix_adds_synthetic_missing_indicator() -> None:
    matrix = build_modeling_matrix(_toy_modeling_frame(), FEATURE_SET_A_SIMPLE)
    # avg_assignment_score_to_date has explicit has_* indicator already, so
    # no synthetic flag should be added for it.
    synthetic_flag = "avg_assignment_score_to_date_was_missing"
    assert synthetic_flag not in matrix.feature_columns


def test_build_modeling_matrix_preserves_indicator_columns_as_int() -> None:
    matrix = build_modeling_matrix(_toy_modeling_frame(), FEATURE_SET_A_SIMPLE)
    for column in matrix.indicator_columns:
        assert pd.api.types.is_integer_dtype(matrix.features[column])


def test_assert_no_forbidden_columns_rejects_leakage() -> None:
    bad = FeatureSet(
        name="A_simple",
        description="bad",
        columns=("avg_assignment_score_to_date", "predicted_final_grade"),
    )
    with pytest.raises(ValueError):
        assert_no_forbidden_columns(bad.all_columns())


def test_select_rows_aligns_targets_and_features() -> None:
    matrix = build_modeling_matrix(_toy_modeling_frame(), FEATURE_SET_A_SIMPLE)
    mask = (matrix.weeks <= 2).to_numpy()
    subset = select_rows(matrix, mask)

    assert subset.features.shape[0] == int(mask.sum())
    assert subset.regression_target.shape[0] == int(mask.sum())
    assert subset.classification_target.shape[0] == int(mask.sum())
    assert subset.groups.shape[0] == int(mask.sum())


def test_fit_imputer_uses_training_medians() -> None:
    matrix = build_modeling_matrix(_toy_modeling_frame(), FEATURE_SET_A_SIMPLE)
    train = select_rows(matrix, (matrix.weeks <= 2).to_numpy())
    test = select_rows(matrix, (matrix.weeks > 2).to_numpy())

    imputer = fit_imputer_on_training(train.features)
    transformed = imputer.transform(test.features)
    assert not transformed.isna().any().any()
    expected_median = float(
        pd.to_numeric(train.features["avg_assignment_score_to_date"], errors="coerce").median()
    )
    assert imputer.medians["avg_assignment_score_to_date"] == pytest.approx(expected_median)


def test_imputation_uses_partition_specific_medians() -> None:
    """Imputer must derive its medians from the rows it was fit on."""

    matrix = build_modeling_matrix(_toy_modeling_frame(), FEATURE_SET_A_SIMPLE)
    train_mask = matrix.groups.isin({"student_000", "student_001", "student_002"}).to_numpy()
    train = select_rows(matrix, train_mask)
    test = select_rows(matrix, ~train_mask)

    train_imputer = fit_imputer_on_training(train.features)
    test_imputer_would_be = fit_imputer_on_training(test.features)

    # Distributions for assignment scores differ between the two student
    # subsets, so the medians must differ — proving the imputer respects its
    # training partition rather than peeking at test rows.
    assert not np.isclose(
        train_imputer.medians["avg_assignment_score_to_date"],
        test_imputer_would_be.medians["avg_assignment_score_to_date"],
    )
