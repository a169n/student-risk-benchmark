from __future__ import annotations

from src.experiments.featuresets import (
    FEATURE_SET_A_SIMPLE,
    FEATURE_SET_B_LMS,
    FEATURE_SET_B_LMS_PLUS_INDICES,
    FEATURE_SET_B_LMS_PLUS_MASTERY,
    FEATURE_SET_B_LMS_PLUS_TEMPORAL,
    FEATURE_SET_B_LMS_PLUS_TRENDS,
    FEATURE_SET_B_LMS_PLUS_TRENDS_MASTERY,
    FEATURE_SET_C_TWIN,
    FEATURE_SET_C_TWIN_FULL,
    FORBIDDEN_FEATURE_COLUMNS,
    FeatureSet,
    available_feature_sets,
    get_feature_set,
    validate_registry,
)


def test_feature_set_a_columns_match_spec() -> None:
    assert FEATURE_SET_A_SIMPLE.name == "A_simple"
    assert "avg_assignment_score_to_date" in FEATURE_SET_A_SIMPLE.columns
    assert "avg_quiz_score_to_date" in FEATURE_SET_A_SIMPLE.columns
    assert "attendance_rate_to_date" in FEATURE_SET_A_SIMPLE.columns
    # A is intentionally minimal: must not include richer twin fields.
    assert "score_trend_3w" not in FEATURE_SET_A_SIMPLE.all_columns()
    assert "engagement_index" not in FEATURE_SET_A_SIMPLE.all_columns()


def test_feature_set_b_extends_simple_baseline() -> None:
    simple_columns = set(FEATURE_SET_A_SIMPLE.columns)
    lms_columns = set(FEATURE_SET_B_LMS.columns)
    assert simple_columns.issubset(lms_columns)
    # B adds discipline + activity signals but not twin-only features.
    assert "on_time_submission_rate_to_date" in lms_columns
    assert "missed_assignments_to_date" in lms_columns
    assert "score_trend_3w" not in lms_columns
    assert "engagement_index" not in lms_columns


def test_feature_set_c_includes_twin_specific_features() -> None:
    twin_columns = set(FEATURE_SET_C_TWIN.columns)
    for column in (
        "score_trend_3w",
        "activity_trend_3w",
        "attendance_trend_3w",
        "current_topic_mastery",
        "overall_mastery",
        "engagement_index",
        "performance_index",
        "discipline_index",
    ):
        assert column in twin_columns, column


def test_ablation_feature_sets_extend_lms_baseline_in_small_blocks() -> None:
    lms_columns = set(FEATURE_SET_B_LMS.columns)
    assert lms_columns.issubset(set(FEATURE_SET_B_LMS_PLUS_TRENDS.columns))
    assert lms_columns.issubset(set(FEATURE_SET_B_LMS_PLUS_MASTERY.columns))
    assert lms_columns.issubset(set(FEATURE_SET_B_LMS_PLUS_INDICES.columns))
    assert lms_columns.issubset(set(FEATURE_SET_B_LMS_PLUS_TEMPORAL.columns))
    assert lms_columns.issubset(set(FEATURE_SET_B_LMS_PLUS_TRENDS_MASTERY.columns))

    assert "score_trend_3w" in FEATURE_SET_B_LMS_PLUS_TRENDS.columns
    assert "overall_mastery" in FEATURE_SET_B_LMS_PLUS_MASTERY.columns
    assert "performance_index" in FEATURE_SET_B_LMS_PLUS_INDICES.columns
    assert "week_number" in FEATURE_SET_B_LMS_PLUS_TEMPORAL.columns
    assert "discipline_index" not in FEATURE_SET_B_LMS_PLUS_TRENDS_MASTERY.columns


def test_c_twin_full_alias_matches_baseline_c_twin_columns() -> None:
    assert FEATURE_SET_C_TWIN_FULL.columns == FEATURE_SET_C_TWIN.columns
    assert FEATURE_SET_C_TWIN_FULL.indicator_columns == FEATURE_SET_C_TWIN.indicator_columns


def test_no_feature_set_contains_forbidden_columns() -> None:
    validate_registry()  # raises if anything regresses
    for feature_set in available_feature_sets():
        overlap = set(feature_set.all_columns()) & FORBIDDEN_FEATURE_COLUMNS
        assert overlap == set(), f"{feature_set.name} leaks: {overlap}"


def test_get_feature_set_returns_known_definition() -> None:
    feature_set: FeatureSet = get_feature_set("C_twin")
    assert feature_set.name == "C_twin"
    assert feature_set.indicator_columns == FEATURE_SET_C_TWIN.indicator_columns
