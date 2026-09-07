"""Explicit, reusable feature-set definitions for the baseline experiments.

The first dissertation comparison asks whether richer Digital Twin features
improve predictive usefulness compared with simpler academic and LMS-style
baselines. To make that comparison auditable, each feature set is declared as
an explicit list of column names taken from `student_twin_snapshots`.

The baseline comparison exposes three main feature sets:

- ``A_simple``: minimal academic baseline (averages and attendance only).
- ``B_lms``: stronger LMS baseline that adds behavioral and discipline signals.
- ``C_twin``: the full Digital Twin representation including trends, mastery,
  composite indices, and explicit missingness indicators.

The ablation phase adds a small set of ``B_lms_plus_*`` feature sets. These are
not new schema fields; they are disciplined regroupings of the existing
``schema_v1.2`` snapshot columns so the project can ask which parts of the Twin
representation add value beyond the LMS baseline.

Membership and provenance are documented per set so reviewers can see exactly
what each model was trained on. Feature lists must remain in sync with
``student_twin_snapshots`` as defined in
``packages/contracts/schema_versions/schema_v1.2.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.experiments.config import FeatureSetName

# Columns that must never enter a feature set because they are identifiers,
# bookkeeping fields, or directly leak the outcome.
FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        # identifiers / bookkeeping
        "snapshot_id",
        "student_id",
        "course_id",
        "snapshot_date",
        # heuristic teacher-facing label (already derived from snapshot features)
        "risk_score",
        "risk_level",
        # snapshot-level estimate of the outcome (would short-circuit modeling)
        "predicted_final_grade",
        # outcome-layer fields joined for training (must remain targets only)
        "final_grade",
        "passed",
        "completion_status",
        "completed_weeks",
        # generation-only hidden fields (must not be visible to models)
        "baseline_level",
        "motivation_level",
        "discipline_level",
        "trajectory_type",
    }
)


@dataclass(frozen=True)
class FeatureSet:
    """A named, immutable feature-set definition."""

    name: FeatureSetName
    description: str
    columns: tuple[str, ...]
    indicator_columns: tuple[str, ...] = ()

    def all_columns(self) -> tuple[str, ...]:
        return tuple(list(self.columns) + list(self.indicator_columns))


# ---------------------------------------------------------------------------
# Feature Set A — Simple academic baseline
# ---------------------------------------------------------------------------
FEATURE_SET_A_SIMPLE = FeatureSet(
    name="A_simple",
    description=(
        "Minimal academic baseline. Uses only the most basic teacher-visible "
        "performance and attendance signals available from the LMS. Intended "
        "as a deliberately weak reference point."
    ),
    columns=(
        "avg_assignment_score_to_date",
        "avg_quiz_score_to_date",
        "attendance_rate_to_date",
    ),
    indicator_columns=(
        "has_assignment_score_to_date",
        "has_quiz_score_to_date",
    ),
)


# ---------------------------------------------------------------------------
# Feature Set B — Enriched LMS baseline
# ---------------------------------------------------------------------------
FEATURE_SET_B_LMS = FeatureSet(
    name="B_lms",
    description=(
        "Stronger non-twin LMS baseline. Adds broader behavioral and "
        "submission-discipline signals that a typical LMS analytics view "
        "could plausibly expose without any digital-twin engineering."
    ),
    columns=(
        # academic
        "avg_assignment_score_to_date",
        "avg_quiz_score_to_date",
        # attendance / activity
        "attendance_rate_to_date",
        "activity_score_to_date",
        "time_spent_to_date",
        # discipline
        "on_time_submission_rate_to_date",
        "missed_assignments_to_date",
        "late_submissions_to_date",
        "avg_attempt_count_to_date",
    ),
    indicator_columns=(
        "has_assignment_score_to_date",
        "has_quiz_score_to_date",
    ),
)


# ---------------------------------------------------------------------------
# Feature Set C — Digital Twin feature set
# ---------------------------------------------------------------------------
FEATURE_SET_C_TWIN = FeatureSet(
    name="C_twin",
    description=(
        "Full Digital Twin representation. Includes the LMS baseline plus "
        "short-horizon trend features, mastery proxies, and composite "
        "engagement / performance / discipline indices."
    ),
    columns=(
        # academic
        "avg_assignment_score_to_date",
        "avg_quiz_score_to_date",
        # attendance / activity
        "attendance_rate_to_date",
        "activity_score_to_date",
        "time_spent_to_date",
        # discipline
        "on_time_submission_rate_to_date",
        "missed_assignments_to_date",
        "late_submissions_to_date",
        "avg_attempt_count_to_date",
        # trends (twin-specific temporal aggregates)
        "score_trend_3w",
        "activity_trend_3w",
        "attendance_trend_3w",
        # mastery (twin-specific)
        "current_topic_mastery",
        "overall_mastery",
        # composite indices (twin-specific)
        "engagement_index",
        "performance_index",
        "discipline_index",
        # week context (allowed because it is a structural input, not the outcome)
        "week_number",
    ),
    indicator_columns=(
        "has_assignment_score_to_date",
        "has_quiz_score_to_date",
    ),
)


TWIN_TREND_COLUMNS: tuple[str, ...] = (
    "score_trend_3w",
    "activity_trend_3w",
    "attendance_trend_3w",
)

TWIN_MASTERY_COLUMNS: tuple[str, ...] = (
    "current_topic_mastery",
    "overall_mastery",
)

TWIN_INDEX_COLUMNS: tuple[str, ...] = (
    "engagement_index",
    "performance_index",
    "discipline_index",
)

TWIN_TEMPORAL_COLUMNS: tuple[str, ...] = (
    "week_number",
)


FEATURE_SET_B_LMS_PLUS_TRENDS = FeatureSet(
    name="B_lms_plus_trends",
    description=(
        "LMS baseline plus short-horizon trend features. Tests whether recent "
        "direction of performance, activity, and attendance adds predictive "
        "value beyond cumulative LMS indicators."
    ),
    columns=FEATURE_SET_B_LMS.columns + TWIN_TREND_COLUMNS,
    indicator_columns=FEATURE_SET_B_LMS.indicator_columns,
)


FEATURE_SET_B_LMS_PLUS_MASTERY = FeatureSet(
    name="B_lms_plus_mastery",
    description=(
        "LMS baseline plus current and overall mastery proxies. Tests whether "
        "topic-level Twin state adds value beyond raw performance and activity."
    ),
    columns=FEATURE_SET_B_LMS.columns + TWIN_MASTERY_COLUMNS,
    indicator_columns=FEATURE_SET_B_LMS.indicator_columns,
)


FEATURE_SET_B_LMS_PLUS_INDICES = FeatureSet(
    name="B_lms_plus_indices",
    description=(
        "LMS baseline plus composite engagement, performance, and discipline "
        "indices. Tests whether the current index layer adds compact value or "
        "mostly duplicates the underlying LMS signals."
    ),
    columns=FEATURE_SET_B_LMS.columns + TWIN_INDEX_COLUMNS,
    indicator_columns=FEATURE_SET_B_LMS.indicator_columns,
)


FEATURE_SET_B_LMS_PLUS_TEMPORAL = FeatureSet(
    name="B_lms_plus_temporal",
    description=(
        "LMS baseline plus the snapshot week number. Tests whether coarse "
        "course-time context explains gains separately from richer trend or "
        "mastery features."
    ),
    columns=FEATURE_SET_B_LMS.columns + TWIN_TEMPORAL_COLUMNS,
    indicator_columns=FEATURE_SET_B_LMS.indicator_columns,
)


FEATURE_SET_B_LMS_PLUS_TRENDS_MASTERY = FeatureSet(
    name="B_lms_plus_trends_mastery",
    description=(
        "Compact Twin candidate combining the LMS baseline with trend and "
        "mastery blocks while excluding composite indices. Intended as a lean "
        "candidate for the later XAI phase if it is competitive with the full "
        "Twin set."
    ),
    columns=FEATURE_SET_B_LMS.columns + TWIN_TREND_COLUMNS + TWIN_MASTERY_COLUMNS,
    indicator_columns=FEATURE_SET_B_LMS.indicator_columns,
)


FEATURE_SET_C_TWIN_FULL = FeatureSet(
    name="C_twin_full",
    description=(
        "Alias for the full Digital Twin representation used in ablation "
        "reports. Kept separate from `C_twin` naming so the baseline and "
        "ablation experiment pages can be cited cleanly."
    ),
    columns=FEATURE_SET_C_TWIN.columns,
    indicator_columns=FEATURE_SET_C_TWIN.indicator_columns,
)


_REGISTRY: dict[FeatureSetName, FeatureSet] = {
    FEATURE_SET_A_SIMPLE.name: FEATURE_SET_A_SIMPLE,
    FEATURE_SET_B_LMS.name: FEATURE_SET_B_LMS,
    FEATURE_SET_B_LMS_PLUS_TRENDS.name: FEATURE_SET_B_LMS_PLUS_TRENDS,
    FEATURE_SET_B_LMS_PLUS_MASTERY.name: FEATURE_SET_B_LMS_PLUS_MASTERY,
    FEATURE_SET_B_LMS_PLUS_INDICES.name: FEATURE_SET_B_LMS_PLUS_INDICES,
    FEATURE_SET_B_LMS_PLUS_TEMPORAL.name: FEATURE_SET_B_LMS_PLUS_TEMPORAL,
    FEATURE_SET_B_LMS_PLUS_TRENDS_MASTERY.name: FEATURE_SET_B_LMS_PLUS_TRENDS_MASTERY,
    FEATURE_SET_C_TWIN.name: FEATURE_SET_C_TWIN,
    FEATURE_SET_C_TWIN_FULL.name: FEATURE_SET_C_TWIN_FULL,
}


def get_feature_set(name: FeatureSetName) -> FeatureSet:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown feature set: {name}")
    return _REGISTRY[name]


def available_feature_sets() -> tuple[FeatureSet, ...]:
    return tuple(_REGISTRY.values())


def assert_no_forbidden_columns(feature_set: FeatureSet) -> None:
    """Guard against accidental leakage when feature lists are edited."""

    overlap = set(feature_set.all_columns()) & FORBIDDEN_FEATURE_COLUMNS
    if overlap:
        raise ValueError(
            f"Feature set {feature_set.name} contains forbidden columns: {sorted(overlap)}"
        )


def validate_registry() -> None:
    """Run all guards against the registered feature sets."""

    for feature_set in _REGISTRY.values():
        assert_no_forbidden_columns(feature_set)
