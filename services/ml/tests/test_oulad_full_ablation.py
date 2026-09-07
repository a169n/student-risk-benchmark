from pathlib import Path

import pandas as pd

from src.benchmarks.oulad_adapter import (
    OuladCourseFilter,
    OuladRawPaths,
    build_weekly_snapshots,
)
from src.generator.config import REPO_ROOT

OULAD = REPO_ROOT / "datasets" / "oulad"


def _snapshots() -> pd.DataFrame:
    raw = OuladRawPaths(
        assessments=OULAD / "assessments.csv",
        courses=OULAD / "courses.csv",
        student_info=OULAD / "studentInfo.csv",
        student_registration=OULAD / "studentRegistration.csv",
        student_vle=OULAD / "studentVle.csv",
        vle=OULAD / "vle.csv",
        student_assessment=OULAD / "studentAssessment.csv",
    )
    return build_weekly_snapshots(
        raw,
        course_filter=OuladCourseFilter("DDD", "2013J"),
        min_week=4,
        max_week=None,
    ).snapshots


def test_trend_and_index_columns_present():
    snaps = _snapshots()
    for col in [
        "assessment_score_trend_to_date",
        "clicks_trend_to_date",
        "engagement_index_oulad",
        "performance_index_oulad",
        "discipline_index_oulad",
    ]:
        assert col in snaps.columns, f"missing {col}"


def test_index_columns_are_bounded_unit_interval():
    snaps = _snapshots()
    for col in ["engagement_index_oulad", "performance_index_oulad", "discipline_index_oulad"]:
        series = snaps[col].dropna()
        assert (series >= -1e-9).all() and (series <= 1.0 + 1e-9).all(), f"{col} out of [0,1]"


def test_indices_use_fixed_denominator_on_partial_nan():
    import numpy as np
    from src.benchmarks.oulad_adapter import _add_trend_and_index_features

    df = pd.DataFrame(
        {
            "code_module": ["DDD"],
            "code_presentation": ["2013J"],
            "id_student": ["1"],
            "week_number": [4],
            "cumulative_assessment_score_mean_to_date": [np.nan],
            "cumulative_assessment_weighted_score_to_date": [np.nan],
            "current_week_clicks": [0.0],
            "assessment_submission_rate_due_to_date": [1.0],
            "late_submission_rate_to_date": [np.nan],
            "banked_assessment_rate_to_date": [np.nan],
            "has_vle_activity_to_date": [np.nan],
        }
    )
    out = _add_trend_and_index_features(df)
    # performance: both components NaN -> filled 0 -> mean([0,0]) = 0.0
    assert abs(float(out["performance_index_oulad"].iloc[0]) - 0.0) < 1e-9
    # discipline: [1.0, (1 - NaN)->0, NaN->0] -> mean = 1/3 (NOT 1.0)
    assert abs(float(out["discipline_index_oulad"].iloc[0]) - (1.0 / 3.0)) < 1e-9
    # engagement: [1.0, NaN->0] -> mean = 0.5 (NOT 1.0)
    assert abs(float(out["engagement_index_oulad"].iloc[0]) - 0.5) < 1e-9


def test_exp006_config_has_full_nested_sets():
    from src.experiments.run_public_benchmark_oulad import load_public_benchmark_config

    config, _ = load_public_benchmark_config(
        "configs/experiments/exp_006_oulad_full_ablation.yaml"
    )
    names = set(config.feature_sets)
    assert {
        "A_simple_oulad",
        "B_lms_oulad",
        "B_lms_plus_trends_oulad",
        "B_lms_plus_mastery_oulad",
        "B_lms_plus_indices_oulad",
        "C_twin_oulad",
    } <= names
    assert config.feature_set_order[0] == "A_simple_oulad"
    assert config.comparison.candidate_feature_set == "C_twin_oulad"
    assert config.comparison.primary_split == "temporal_forward"
    # C_twin_oulad must contain trend + index + mastery-extra columns, no duplicates
    cols = config.feature_sets["C_twin_oulad"].columns
    assert len(cols) == len(set(cols)), "C_twin_oulad has duplicate columns"
    for c in ["assessment_score_trend_to_date", "engagement_index_oulad", "overall_mastery_proxy"]:
        assert c in cols


def test_fixed_model_table_filters_one_model():
    import pandas as pd
    from src.experiments.run_public_benchmark_oulad import (
        _fixed_model_regression_by_feature_set,
    )

    table = pd.DataFrame(
        [
            {"task": "regression", "split_strategy": "student_group", "feature_set": "B_lms_oulad", "model": "gradient_boosting", "metric_rmse": 12.6, "metric_mae": 7.9, "metric_r2": 0.85, "n_train_rows": 1, "n_test_rows": 1},
            {"task": "regression", "split_strategy": "student_group", "feature_set": "B_lms_oulad", "model": "linear_regression", "metric_rmse": 15.0, "metric_mae": 9.0, "metric_r2": 0.70, "n_train_rows": 1, "n_test_rows": 1},
            {"task": "regression", "split_strategy": "student_group", "feature_set": "C_twin_oulad", "model": "gradient_boosting", "metric_rmse": 12.0, "metric_mae": 7.5, "metric_r2": 0.86, "n_train_rows": 1, "n_test_rows": 1},
            {"task": "regression", "split_strategy": "student_group", "feature_set": "C_twin_oulad", "model": "linear_regression", "metric_rmse": 14.0, "metric_mae": 8.5, "metric_r2": 0.75, "n_train_rows": 1, "n_test_rows": 1},
        ]
    )
    rows = _fixed_model_regression_by_feature_set(
        table,
        model="gradient_boosting",
        baseline_feature_set="B_lms_oulad",
        primary_split="student_group",
    )
    # only gradient_boosting rows kept (one per feature set)
    assert all(r["model"] == "gradient_boosting" for r in rows)
    assert len(rows) == 2
    # delta vs baseline computed against the gradient_boosting baseline RMSE (12.6)
    candidate = next(r for r in rows if r["feature_set"] == "C_twin_oulad")
    assert abs(candidate["delta_vs_baseline_rmse"] - (12.0 - 12.6)) < 1e-9
