"""TDD tests for exp_012: OULAD engagement-only PASSED classification benchmark.

These tests cover only the OULAD-specific entry point. The shared methodology
(splits, classification, permutation importance) is exercised by the KU Leuven
tests and the engagement_benchmark core; this module checks that the OULAD
modeling frame is built with the columns the shared core expects.
"""

from __future__ import annotations

from src.experiments.run_engagement_benchmark_oulad import (
    build_oulad_engagement_frame,
    load_oulad_engagement_config,
)


def test_oulad_engagement_frame_has_required_columns():
    frame, _ = build_oulad_engagement_frame(
        code_module="DDD", code_presentation="2013J", min_week=4
    )
    # core modeling columns + engagement features must exist
    from src.experiments.datasets import GROUP_COLUMN, WEEK_COLUMN

    for col in [
        GROUP_COLUMN,
        WEEK_COLUMN,
        "passed",
        "final_grade",
        "cumulative_clicks_to_date",
        "cumulative_active_days_to_date",
        "content_click_ratio_to_date",
        "cumulative_social_clicks_to_date",
    ]:
        assert col in frame.columns, f"missing {col}"
    assert set(frame["passed"].dropna().unique()) <= {0, 1, 0.0, 1.0}


def test_exp012_configs_declare_matched_engagement_sets():
    for cohort in ("ddd2013j", "bbb2013j"):
        config, _ = load_oulad_engagement_config(
            f"configs/experiments/exp_012_oulad_engagement_{cohort}.yaml"
        )
        names = set(config.feature_sets)
        assert {"A_simple_engagement_oulad", "B_engagement_oulad"} <= names
        assert not getattr(config, "regression_models", []), "exp_012 must be classification-only"
