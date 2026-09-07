"""Adapter for the Open University Learning Analytics Dataset (OULAD).

This module is intentionally narrow: it supports the seven canonical OULAD CSV
files already placed in the repository and converts them into weekly benchmark
snapshots for the public-transfer experiment. It does not try to force OULAD
into the synthetic schema. Instead, it exposes a compatible benchmark grain:

``1 row = 1 OULAD student-course presentation x 1 week``.

The adapter keeps target construction explicit. The continuous target is a
derived weighted assessment score from OULAD assessment weights and student
scores; it is not treated as identical to the synthetic `final_grade`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REQUIRED_OULAD_FILES: tuple[str, ...] = (
    "assessments.csv",
    "courses.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
    "studentAssessment.csv",
)

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "assessments": (
        "code_module",
        "code_presentation",
        "id_assessment",
        "assessment_type",
        "date",
        "weight",
    ),
    "courses": ("code_module", "code_presentation", "module_presentation_length"),
    "student_info": (
        "code_module",
        "code_presentation",
        "id_student",
        "final_result",
    ),
    "student_registration": (
        "code_module",
        "code_presentation",
        "id_student",
        "date_registration",
        "date_unregistration",
    ),
    "student_vle": (
        "code_module",
        "code_presentation",
        "id_student",
        "id_site",
        "date",
        "sum_click",
    ),
    "vle": (
        "id_site",
        "code_module",
        "code_presentation",
        "activity_type",
    ),
    "student_assessment": (
        "id_assessment",
        "id_student",
        "date_submitted",
        "is_banked",
        "score",
    ),
}

KEY_COLUMNS: tuple[str, ...] = ("code_module", "code_presentation", "id_student")
COURSE_COLUMNS: tuple[str, ...] = ("code_module", "code_presentation")
POSITIVE_FINAL_RESULTS: frozenset[str] = frozenset({"Pass", "Distinction"})
NEGATIVE_FINAL_RESULTS: frozenset[str] = frozenset({"Fail", "Withdrawn"})

ACTIVITY_CATEGORY_BY_TYPE: dict[str, str] = {
    "dataplus": "assessment",
    "externalquiz": "assessment",
    "questionnaire": "assessment",
    "quiz": "assessment",
    "folder": "content",
    "homepage": "content",
    "htmlactivity": "content",
    "oucontent": "content",
    "page": "content",
    "resource": "content",
    "sharedsubpage": "content",
    "subpage": "content",
    "url": "content",
    "forumng": "social",
    "oucollaborate": "social",
    "ouelluminate": "social",
    "ouwiki": "social",
}
ACTIVITY_CATEGORIES: tuple[str, ...] = ("assessment", "content", "social", "other")


@dataclass(frozen=True)
class OuladRawPaths:
    """Concrete paths for the exact OULAD CSV files used by the benchmark."""

    assessments: Path
    courses: Path
    student_info: Path
    student_registration: Path
    student_vle: Path
    vle: Path
    student_assessment: Path

    @classmethod
    def from_directory(cls, raw_dir: Path | str) -> "OuladRawPaths":
        root = Path(raw_dir)
        return cls(
            assessments=root / "assessments.csv",
            courses=root / "courses.csv",
            student_info=root / "studentInfo.csv",
            student_registration=root / "studentRegistration.csv",
            student_vle=root / "studentVle.csv",
            vle=root / "vle.csv",
            student_assessment=root / "studentAssessment.csv",
        )

    def as_dict(self) -> dict[str, Path]:
        return {
            "assessments": self.assessments,
            "courses": self.courses,
            "student_info": self.student_info,
            "student_registration": self.student_registration,
            "student_vle": self.student_vle,
            "vle": self.vle,
            "student_assessment": self.student_assessment,
        }


@dataclass(frozen=True)
class OuladCourseFilter:
    """Optional module/presentation selection.

    The benchmark runner uses this to align OULAD with the current one-course
    synthetic scope. Passing both values as ``None`` keeps all OULAD rows.
    """

    code_module: str | None = None
    code_presentation: str | None = None

    @property
    def enabled(self) -> bool:
        return self.code_module is not None or self.code_presentation is not None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "code_module": self.code_module,
            "code_presentation": self.code_presentation,
        }


@dataclass(frozen=True)
class OuladTables:
    assessments: pd.DataFrame
    courses: pd.DataFrame
    student_info: pd.DataFrame
    student_registration: pd.DataFrame
    vle: pd.DataFrame
    student_assessment: pd.DataFrame
    student_vle: pd.DataFrame | None = None


@dataclass(frozen=True)
class OuladSnapshotBuildResult:
    snapshots: pd.DataFrame
    raw_row_counts: dict[str, int]
    filtered_row_counts: dict[str, int]
    course_filter: dict[str, str | None]
    week_min: int
    week_max: int
    target_summary: dict[str, Any]


def validate_oulad_raw_files(paths: OuladRawPaths) -> None:
    """Verify the exact required OULAD CSV files are present."""

    missing = [str(path) for path in paths.as_dict().values() if not path.exists()]
    if missing:
        expected = ", ".join(REQUIRED_OULAD_FILES)
        raise FileNotFoundError(
            "Missing required local OULAD CSV file(s): "
            f"{missing}. Expected exactly these files: {expected}."
        )


def load_oulad_tables(
    paths: OuladRawPaths,
    *,
    include_student_vle: bool = True,
) -> OuladTables:
    """Load OULAD tables and normalize core identifier / numeric columns."""

    validate_oulad_raw_files(paths)
    frames: dict[str, pd.DataFrame] = {}
    for name, path in paths.as_dict().items():
        if name == "student_vle" and not include_student_vle:
            continue
        frame = pd.read_csv(path, low_memory=False)
        _validate_required_columns(frame, table_name=name)
        frames[name] = _normalize_table(frame, name)

    return OuladTables(
        assessments=frames["assessments"],
        courses=frames["courses"],
        student_info=frames["student_info"],
        student_registration=frames["student_registration"],
        vle=frames["vle"],
        student_assessment=frames["student_assessment"],
        student_vle=frames.get("student_vle"),
    )


def build_weekly_snapshots(
    paths: OuladRawPaths,
    *,
    course_filter: OuladCourseFilter | None = None,
    min_week: int = 4,
    max_week: int | None = None,
    student_vle_chunk_size: int = 500_000,
) -> OuladSnapshotBuildResult:
    """Build weekly OULAD benchmark snapshots.

    Parameters are explicit because this output is an experiment artifact, not
    a replacement data contract. `min_week` filters the final modeling table
    after cumulative features have been computed from all prior weeks.
    """

    if min_week < 1:
        raise ValueError("min_week must be >= 1")

    tables = load_oulad_tables(paths, include_student_vle=False)
    raw_counts = {
        name: _count_csv_rows(path)
        for name, path in paths.as_dict().items()
    }

    selected = _filter_tables(tables, course_filter or OuladCourseFilter())
    enrollment = _build_enrollment(selected)
    if enrollment.empty:
        raise ValueError(
            "No OULAD enrollments remain after applying the course filter: "
            f"{(course_filter or OuladCourseFilter()).as_dict()}."
        )

    weekly_index = _build_weekly_index(enrollment)
    target_frame = build_targets(selected, enrollment)
    assessment_features = build_assessment_features(selected, weekly_index)
    activity_features, selected_student_vle_rows = build_activity_features(
        paths.student_vle,
        selected.vle,
        enrollment,
        student_vle_chunk_size=student_vle_chunk_size,
    )

    target_merge = target_frame.drop(columns=["final_result"], errors="ignore")
    snapshots = weekly_index.merge(
        target_merge,
        on=list(KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    snapshots = snapshots.merge(
        assessment_features,
        on=list(KEY_COLUMNS) + ["week_number"],
        how="left",
        validate="one_to_one",
    )
    snapshots = snapshots.merge(
        activity_features,
        on=list(KEY_COLUMNS) + ["week_number"],
        how="left",
        validate="one_to_one",
    )
    snapshots = _add_trend_and_index_features(snapshots)
    snapshots = _finalize_snapshot_frame(snapshots)

    if max_week is not None:
        snapshots = snapshots.loc[snapshots["week_number"] <= max_week].copy()
    snapshots = snapshots.loc[snapshots["week_number"] >= min_week].copy()
    snapshots = snapshots.loc[snapshots["final_weighted_score"].notna()].copy()
    snapshots = snapshots.loc[snapshots["passed_observed"].notna()].copy()
    snapshots = snapshots.reset_index(drop=True)
    if snapshots.empty:
        raise ValueError("OULAD benchmark snapshots are empty after filtering.")

    filtered_counts = {
        "assessments": int(selected.assessments.shape[0]),
        "courses": int(selected.courses.shape[0]),
        "student_info": int(selected.student_info.shape[0]),
        "student_registration": int(selected.student_registration.shape[0]),
        "vle": int(selected.vle.shape[0]),
        "student_assessment": int(selected.student_assessment.shape[0]),
        "student_vle": int(selected_student_vle_rows),
        "snapshots": int(snapshots.shape[0]),
        "students": int(snapshots["student_id"].nunique()),
    }

    return OuladSnapshotBuildResult(
        snapshots=snapshots,
        raw_row_counts=raw_counts,
        filtered_row_counts=filtered_counts,
        course_filter=(course_filter or OuladCourseFilter()).as_dict(),
        week_min=int(snapshots["week_number"].min()),
        week_max=int(snapshots["week_number"].max()),
        target_summary=summarize_targets(snapshots),
    )


def build_targets(tables: OuladTables, enrollment: pd.DataFrame) -> pd.DataFrame:
    """Construct the benchmark targets from OULAD assessment and result tables.

    `final_weighted_score` is derived as:

    ``sum(score_or_zero * assessment_weight) / sum(assessment_weight)``

    over positive-weight assessments in the same module/presentation. Missing
    assessment submissions are treated as zero contribution. This produces a
    grade-like continuous benchmark target, but it is not asserted to be the
    same construct as the synthetic `final_grade`.
    """

    positive_assessments = _positive_scored_assessments(tables)
    target_basis = enrollment.loc[:, list(KEY_COLUMNS)].merge(
        positive_assessments[
            list(COURSE_COLUMNS) + ["id_assessment", "assessment_type", "weight"]
        ],
        on=list(COURSE_COLUMNS),
        how="left",
    )
    submissions = _prepare_student_assessments(tables)
    target_basis = target_basis.merge(
        submissions[["id_assessment", "id_student", "score"]],
        on=["id_assessment", "id_student"],
        how="left",
    )
    target_basis["score_for_target"] = pd.to_numeric(
        target_basis["score"], errors="coerce"
    ).fillna(0.0)
    target_basis["weighted_points"] = (
        target_basis["score_for_target"] * target_basis["weight"]
    )
    target_basis["submitted_weight"] = np.where(
        target_basis["score"].notna(), target_basis["weight"], 0.0
    )

    grouped = target_basis.groupby(list(KEY_COLUMNS), dropna=False).agg(
        target_weighted_points=("weighted_points", "sum"),
        target_total_weight=("weight", "sum"),
        target_submitted_weight=("submitted_weight", "sum"),
    )
    grouped = grouped.reset_index()
    grouped["final_weighted_score"] = np.where(
        grouped["target_total_weight"] > 0,
        grouped["target_weighted_points"] / grouped["target_total_weight"],
        np.nan,
    )
    grouped["target_weight_coverage"] = np.where(
        grouped["target_total_weight"] > 0,
        grouped["target_submitted_weight"] / grouped["target_total_weight"],
        np.nan,
    )

    labels = enrollment.loc[list(enrollment.index), list(KEY_COLUMNS) + ["final_result"]].copy()
    labels["passed_observed"] = labels["final_result"].map(_map_final_result_to_passed)
    return grouped.merge(labels, on=list(KEY_COLUMNS), how="left", validate="one_to_one")


def build_assessment_features(
    tables: OuladTables,
    weekly_index: pd.DataFrame,
) -> pd.DataFrame:
    """Create submitted-score and mastery-like weekly assessment features."""

    base_cols = list(KEY_COLUMNS) + ["week_number"]
    features = weekly_index.loc[:, base_cols].copy()
    submitted_weekly = _build_submitted_assessment_weekly(tables, weekly_index)
    mastery_weekly = _build_mastery_weekly(tables, weekly_index)
    features = features.merge(submitted_weekly, on=base_cols, how="left")
    features = features.merge(mastery_weekly, on=base_cols, how="left")
    return features


def build_activity_features(
    student_vle_path: Path,
    vle: pd.DataFrame,
    enrollment: pd.DataFrame,
    *,
    student_vle_chunk_size: int = 500_000,
) -> tuple[pd.DataFrame, int]:
    """Create weekly VLE activity features from `studentVle.csv` chunks."""

    _validate_required_columns(pd.read_csv(student_vle_path, nrows=0), "student_vle")
    weekly_index = _build_weekly_index(enrollment)
    base_cols = list(KEY_COLUMNS) + ["week_number"]
    selected_keys = enrollment.loc[:, list(KEY_COLUMNS)].drop_duplicates()
    selected_key_frame = selected_keys.assign(_selected=1)
    site_lookup = vle.loc[
        :, list(COURSE_COLUMNS) + ["id_site", "activity_type", "activity_category"]
    ].drop_duplicates()

    weekly_parts: list[pd.DataFrame] = []
    type_parts: list[pd.DataFrame] = []
    day_parts: list[pd.DataFrame] = []
    selected_rows = 0

    for chunk in pd.read_csv(
        student_vle_path,
        chunksize=student_vle_chunk_size,
        low_memory=False,
    ):
        _validate_required_columns(chunk, "student_vle")
        chunk = _normalize_table(chunk, "student_vle")
        chunk = chunk.merge(
            selected_key_frame,
            on=list(KEY_COLUMNS),
            how="inner",
        )
        if chunk.empty:
            continue
        selected_rows += int(chunk.shape[0])
        chunk = chunk.merge(
            site_lookup,
            on=list(COURSE_COLUMNS) + ["id_site"],
            how="left",
        )
        chunk["activity_category"] = chunk["activity_category"].fillna("other")
        chunk["activity_week"] = _week_from_day(chunk["date"])
        chunk = chunk.loc[chunk["activity_week"].notna()].copy()
        chunk["activity_week"] = chunk["activity_week"].astype(int)

        per_category = (
            chunk.groupby(list(KEY_COLUMNS) + ["activity_week", "activity_category"])
            .agg(clicks=("sum_click", "sum"))
            .reset_index()
        )
        weekly_parts.append(per_category)

        type_part = chunk.loc[
            :, list(KEY_COLUMNS) + ["activity_week", "activity_type"]
        ].drop_duplicates()
        type_parts.append(type_part)

        day_part = chunk.loc[
            :, list(KEY_COLUMNS) + ["activity_week", "date"]
        ].drop_duplicates()
        day_parts.append(day_part)

    if weekly_parts:
        weekly_clicks = pd.concat(weekly_parts, ignore_index=True)
        weekly_clicks = (
            weekly_clicks.groupby(list(KEY_COLUMNS) + ["activity_week", "activity_category"])
            .agg(clicks=("clicks", "sum"))
            .reset_index()
        )
        category_pivot = weekly_clicks.pivot_table(
            index=list(KEY_COLUMNS) + ["activity_week"],
            columns="activity_category",
            values="clicks",
            aggfunc="sum",
            fill_value=0.0,
        ).reset_index()
    else:
        category_pivot = pd.DataFrame(columns=list(KEY_COLUMNS) + ["activity_week"])

    for category in ACTIVITY_CATEGORIES:
        if category not in category_pivot.columns:
            category_pivot[category] = 0.0
    category_pivot["current_week_clicks"] = category_pivot[list(ACTIVITY_CATEGORIES)].sum(
        axis=1
    )
    rename_categories = {
        category: f"current_week_{category}_clicks"
        for category in ACTIVITY_CATEGORIES
    }
    category_pivot = category_pivot.rename(columns=rename_categories)
    category_pivot = category_pivot.rename(columns={"activity_week": "week_number"})

    activity = weekly_index.loc[:, base_cols].merge(
        category_pivot,
        on=base_cols,
        how="left",
    )
    current_columns = [
        "current_week_clicks",
        *[f"current_week_{category}_clicks" for category in ACTIVITY_CATEGORIES],
    ]
    for column in current_columns:
        activity[column] = pd.to_numeric(activity[column], errors="coerce").fillna(0.0)

    if type_parts:
        type_frame = pd.concat(type_parts, ignore_index=True).drop_duplicates()
        type_counts = (
            type_frame.groupby(list(KEY_COLUMNS) + ["activity_week"])
            .size()
            .rename("current_week_activity_types")
            .reset_index()
            .rename(columns={"activity_week": "week_number"})
        )
        activity = activity.merge(type_counts, on=base_cols, how="left")
    else:
        activity["current_week_activity_types"] = 0
    activity["current_week_activity_types"] = (
        pd.to_numeric(activity["current_week_activity_types"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if day_parts:
        day_frame = pd.concat(day_parts, ignore_index=True).drop_duplicates()
        day_counts = (
            day_frame.groupby(list(KEY_COLUMNS) + ["activity_week"])
            .size()
            .rename("current_week_active_days")
            .reset_index()
            .rename(columns={"activity_week": "week_number"})
        )
        activity = activity.merge(day_counts, on=base_cols, how="left")
    else:
        activity["current_week_active_days"] = 0
    activity["current_week_active_days"] = (
        pd.to_numeric(activity["current_week_active_days"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    activity = activity.sort_values(base_cols).reset_index(drop=True)
    group = activity.groupby(list(KEY_COLUMNS), sort=False)
    activity["cumulative_clicks_to_date"] = group["current_week_clicks"].cumsum()
    for category in ACTIVITY_CATEGORIES:
        current_col = f"current_week_{category}_clicks"
        cumulative_col = f"cumulative_{category}_clicks_to_date"
        activity[cumulative_col] = group[current_col].cumsum()
    activity["cumulative_active_days_to_date"] = group["current_week_active_days"].cumsum()
    activity["has_vle_activity_to_date"] = (
        activity["cumulative_clicks_to_date"] > 0
    ).astype(int)
    return activity, selected_rows


def summarize_targets(snapshots: pd.DataFrame) -> dict[str, Any]:
    """Return compact target distributions for metadata and reports."""

    student_level = snapshots.sort_values("week_number").drop_duplicates(
        list(KEY_COLUMNS), keep="last"
    )
    score = pd.to_numeric(student_level["final_weighted_score"], errors="coerce")
    return {
        "n_snapshot_rows": int(snapshots.shape[0]),
        "n_students": int(snapshots["student_id"].nunique()),
        "n_student_course_presentations": int(student_level.shape[0]),
        "week_min": int(snapshots["week_number"].min()),
        "week_max": int(snapshots["week_number"].max()),
        "final_weighted_score": {
            "count": int(score.notna().sum()),
            "mean": _safe_float(score.mean()),
            "std": _safe_float(score.std()),
            "min": _safe_float(score.min()),
            "median": _safe_float(score.median()),
            "max": _safe_float(score.max()),
        },
        "passed_observed_counts": _value_counts(student_level["passed_observed"]),
        "final_result_counts": _value_counts(student_level["final_result"]),
        "target_weight_coverage": {
            "mean": _safe_float(student_level["target_weight_coverage"].mean()),
            "min": _safe_float(student_level["target_weight_coverage"].min()),
            "median": _safe_float(student_level["target_weight_coverage"].median()),
        },
    }


def _filter_tables(tables: OuladTables, course_filter: OuladCourseFilter) -> OuladTables:
    assessments = _apply_course_filter(tables.assessments, course_filter)
    courses = _apply_course_filter(tables.courses, course_filter)
    student_info = _apply_course_filter(tables.student_info, course_filter)
    registration = _apply_course_filter(tables.student_registration, course_filter)
    vle = _apply_course_filter(tables.vle, course_filter)

    valid_assessment_ids = set(assessments["id_assessment"].dropna().unique().tolist())
    student_assessment = tables.student_assessment.loc[
        tables.student_assessment["id_assessment"].isin(valid_assessment_ids)
    ].copy()
    return OuladTables(
        assessments=assessments.reset_index(drop=True),
        courses=courses.reset_index(drop=True),
        student_info=student_info.reset_index(drop=True),
        student_registration=registration.reset_index(drop=True),
        vle=vle.reset_index(drop=True),
        student_assessment=student_assessment.reset_index(drop=True),
        student_vle=None,
    )


def _apply_course_filter(
    frame: pd.DataFrame,
    course_filter: OuladCourseFilter,
) -> pd.DataFrame:
    out = frame.copy()
    if course_filter.code_module is not None:
        out = out.loc[out["code_module"] == course_filter.code_module].copy()
    if course_filter.code_presentation is not None:
        out = out.loc[out["code_presentation"] == course_filter.code_presentation].copy()
    return out


def _build_enrollment(tables: OuladTables) -> pd.DataFrame:
    enrollment = tables.student_info.merge(
        tables.courses,
        on=list(COURSE_COLUMNS),
        how="inner",
        validate="many_to_one",
    )
    enrollment = enrollment.merge(
        tables.student_registration,
        on=list(KEY_COLUMNS),
        how="left",
        validate="one_to_one",
    )
    enrollment["student_id"] = enrollment["id_student"]
    enrollment["course_id"] = (
        enrollment["code_module"] + "_" + enrollment["code_presentation"]
    )
    enrollment["module_presentation_length"] = pd.to_numeric(
        enrollment["module_presentation_length"], errors="coerce"
    )
    enrollment["duration_weeks"] = np.ceil(
        enrollment["module_presentation_length"].fillna(0) / 7.0
    ).clip(lower=1)
    enrollment["duration_weeks"] = enrollment["duration_weeks"].astype(int)
    return enrollment.reset_index(drop=True)


def _build_weekly_index(enrollment: pd.DataFrame) -> pd.DataFrame:
    counts = enrollment["duration_weeks"].astype(int).clip(lower=1)
    repeated = enrollment.loc[enrollment.index.repeat(counts)].copy()
    repeated["week_number"] = np.concatenate(
        [np.arange(1, int(count) + 1) for count in counts]
    )
    repeated["week_end_day"] = repeated["week_number"] * 7 - 1
    repeated["course_week_progress"] = (
        repeated["week_number"] / repeated["duration_weeks"]
    ).clip(upper=1.0)
    repeated["is_registered_by_week"] = (
        repeated["date_registration"].isna()
        | (repeated["date_registration"] <= repeated["week_end_day"])
    ).astype(int)
    repeated["is_unregistered_by_week"] = (
        repeated["date_unregistration"].notna()
        & (repeated["date_unregistration"] <= repeated["week_end_day"])
    ).astype(int)
    repeated["days_since_registration_start"] = (
        repeated["week_end_day"] - repeated["date_registration"].fillna(0)
    ).clip(lower=0)
    return repeated[
        [
            *KEY_COLUMNS,
            "student_id",
            "course_id",
            "final_result",
            "duration_weeks",
            "week_number",
            "week_end_day",
            "course_week_progress",
            "is_registered_by_week",
            "is_unregistered_by_week",
            "days_since_registration_start",
        ]
    ].reset_index(drop=True)


def _build_submitted_assessment_weekly(
    tables: OuladTables,
    weekly_index: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = list(KEY_COLUMNS) + ["week_number"]
    base = weekly_index.loc[:, base_cols].copy()
    submissions = _assessment_submissions_with_definition(tables)
    if submissions.empty:
        return _empty_submitted_assessment_features(base)

    submissions["week_number"] = _week_from_day(submissions["date_submitted"])
    submissions = submissions.loc[submissions["week_number"].notna()].copy()
    submissions["week_number"] = submissions["week_number"].astype(int)
    submissions["score_value"] = pd.to_numeric(submissions["score"], errors="coerce")
    submissions = submissions.loc[submissions["score_value"].notna()].copy()
    submissions["positive_weight"] = submissions["weight"].clip(lower=0)
    submissions["weighted_points"] = submissions["score_value"] * submissions["positive_weight"]
    submissions["late_submission"] = (
        submissions["date"].notna()
        & submissions["date_submitted"].notna()
        & (submissions["date_submitted"] > submissions["date"])
    ).astype(int)
    submissions["banked_assessment"] = (
        pd.to_numeric(submissions["is_banked"], errors="coerce").fillna(0).astype(int)
    )

    weekly = submissions.groupby(base_cols, dropna=False).agg(
        assessment_score_sum=("score_value", "sum"),
        assessment_score_count=("score_value", "size"),
        assessment_weighted_points=("weighted_points", "sum"),
        assessment_submitted_weight=("positive_weight", "sum"),
        late_submission_count=("late_submission", "sum"),
        banked_assessment_count=("banked_assessment", "sum"),
    )
    out = base.merge(weekly.reset_index(), on=base_cols, how="left")
    fill_cols = [
        "assessment_score_sum",
        "assessment_score_count",
        "assessment_weighted_points",
        "assessment_submitted_weight",
        "late_submission_count",
        "banked_assessment_count",
    ]
    for column in fill_cols:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out = out.sort_values(base_cols).reset_index(drop=True)
    group = out.groupby(list(KEY_COLUMNS), sort=False)
    out["cumulative_assessment_score_count_to_date"] = group[
        "assessment_score_count"
    ].cumsum()
    out["cumulative_assessment_score_sum_to_date"] = group[
        "assessment_score_sum"
    ].cumsum()
    out["cumulative_assessment_weighted_points_to_date"] = group[
        "assessment_weighted_points"
    ].cumsum()
    out["cumulative_submitted_weight_to_date"] = group[
        "assessment_submitted_weight"
    ].cumsum()
    out["cumulative_late_submission_count_to_date"] = group[
        "late_submission_count"
    ].cumsum()
    out["cumulative_banked_assessment_count_to_date"] = group[
        "banked_assessment_count"
    ].cumsum()
    out["cumulative_assessment_score_mean_to_date"] = _safe_divide(
        out["cumulative_assessment_score_sum_to_date"],
        out["cumulative_assessment_score_count_to_date"],
    )
    out["cumulative_assessment_weighted_score_to_date"] = _safe_divide(
        out["cumulative_assessment_weighted_points_to_date"],
        out["cumulative_submitted_weight_to_date"],
    )
    out["late_submission_rate_to_date"] = _safe_divide(
        out["cumulative_late_submission_count_to_date"],
        out["cumulative_assessment_score_count_to_date"],
    )
    out["banked_assessment_rate_to_date"] = _safe_divide(
        out["cumulative_banked_assessment_count_to_date"],
        out["cumulative_assessment_score_count_to_date"],
    )
    out["has_assessment_score_to_date"] = (
        out["cumulative_assessment_score_count_to_date"] > 0
    ).astype(int)
    out["has_weighted_score_to_date"] = (
        out["cumulative_submitted_weight_to_date"] > 0
    ).astype(int)
    return out[
        base_cols
        + [
            "cumulative_assessment_score_mean_to_date",
            "cumulative_assessment_score_count_to_date",
            "cumulative_assessment_weighted_score_to_date",
            "cumulative_submitted_weight_to_date",
            "late_submission_rate_to_date",
            "banked_assessment_rate_to_date",
            "has_assessment_score_to_date",
            "has_weighted_score_to_date",
        ]
    ]


def _build_mastery_weekly(
    tables: OuladTables,
    weekly_index: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = list(KEY_COLUMNS) + ["week_number"]
    base = weekly_index.loc[:, base_cols].copy()
    due = _assessment_due_basis(tables, weekly_index)
    if due.empty:
        return _empty_mastery_features(base)

    scored_assessment_ids = _scored_assessment_ids(tables)
    positive_due = due.loc[
        (due["weight"] > 0)
        & due["date"].notna()
        & due["id_assessment"].isin(scored_assessment_ids)
    ].copy()
    if positive_due.empty:
        return _empty_mastery_features(base)

    positive_due["due_week"] = _week_from_day(positive_due["date"]).astype(int)
    positive_due["submitted_week"] = _week_from_day(positive_due["date_submitted"])
    positive_due["submitted_by_due_or_later"] = positive_due["submitted_week"].notna()
    positive_due["effective_score_week"] = np.where(
        positive_due["submitted_by_due_or_later"],
        np.maximum(
            positive_due["due_week"],
            positive_due["submitted_week"].fillna(positive_due["due_week"]).astype(int),
        ),
        np.nan,
    )
    positive_due["score_for_mastery"] = pd.to_numeric(
        positive_due["score"], errors="coerce"
    ).fillna(0.0)
    positive_due["weighted_points"] = (
        positive_due["score_for_mastery"] * positive_due["weight"]
    )

    denominators = _build_due_denominators(positive_due, base)
    numerators = _build_mastery_numerators(positive_due, base)
    out = denominators.merge(numerators, on=base_cols, how="left")

    for column in [
        "overall_mastery_points_to_date",
        "current_mastery_points",
        "tma_mastery_points_to_date",
        "cma_mastery_points_to_date",
        "exam_mastery_points_to_date",
        "submitted_due_count_to_date",
    ]:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    out["overall_mastery_proxy"] = _safe_divide(
        out["overall_mastery_points_to_date"],
        out["due_positive_weight_to_date"],
    )
    out["current_assessment_cluster_mastery"] = _safe_divide(
        out["current_mastery_points"],
        out["current_due_positive_weight"],
    )
    out["tma_mastery_to_date"] = _safe_divide(
        out["tma_mastery_points_to_date"],
        out["tma_due_weight_to_date"],
    )
    out["cma_mastery_to_date"] = _safe_divide(
        out["cma_mastery_points_to_date"],
        out["cma_due_weight_to_date"],
    )
    out["exam_mastery_to_date"] = _safe_divide(
        out["exam_mastery_points_to_date"],
        out["exam_due_weight_to_date"],
    )
    out["has_current_assessment_cluster"] = (
        out["current_due_positive_weight"] > 0
    ).astype(int)
    out["has_due_assessment_to_date"] = (
        out["due_positive_weight_to_date"] > 0
    ).astype(int)
    out["assessment_submission_rate_due_to_date"] = _safe_divide(
        out["submitted_due_count_to_date"],
        out["due_assessment_count_to_date"],
    )
    out["mastery_assessment_coverage_to_date"] = _safe_divide(
        out["due_positive_weight_to_date"],
        out["target_total_positive_weight"],
    )
    return out[
        base_cols
        + [
            "assessment_submission_rate_due_to_date",
            "overall_mastery_proxy",
            "current_assessment_cluster_mastery",
            "tma_mastery_to_date",
            "cma_mastery_to_date",
            "exam_mastery_to_date",
            "mastery_assessment_coverage_to_date",
            "has_current_assessment_cluster",
            "has_due_assessment_to_date",
        ]
    ]


def _build_due_denominators(
    positive_due: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = list(KEY_COLUMNS) + ["week_number"]
    due_by_week = positive_due.groupby(
        list(KEY_COLUMNS) + ["due_week"], dropna=False
    ).agg(
        due_positive_weight=("weight", "sum"),
        due_assessment_count=("id_assessment", "nunique"),
    )
    due_by_week = due_by_week.reset_index().rename(columns={"due_week": "week_number"})
    out = base.merge(due_by_week, on=base_cols, how="left")
    for column in [
        "due_positive_weight",
        "due_assessment_count",
    ]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out = out.sort_values(base_cols).reset_index(drop=True)
    group = out.groupby(list(KEY_COLUMNS), sort=False)
    out["due_positive_weight_to_date"] = group["due_positive_weight"].cumsum()
    out["due_assessment_count_to_date"] = group["due_assessment_count"].cumsum()
    out["current_due_positive_weight"] = out["due_positive_weight"]

    total_weight = positive_due.groupby(list(KEY_COLUMNS), dropna=False).agg(
        target_total_positive_weight=("weight", "sum")
    )
    out = out.merge(total_weight.reset_index(), on=list(KEY_COLUMNS), how="left")

    for prefix, assessment_type in [
        ("tma", "TMA"),
        ("cma", "CMA"),
        ("exam", "Exam"),
    ]:
        type_due = positive_due.loc[
            positive_due["assessment_type"] == assessment_type
        ].copy()
        if type_due.empty:
            out[f"{prefix}_due_weight_to_date"] = 0.0
            continue
        type_weekly = (
            type_due.groupby(list(KEY_COLUMNS) + ["due_week"], dropna=False)
            .agg(type_due_weight=("weight", "sum"))
            .reset_index()
            .rename(columns={"due_week": "week_number"})
        )
        out = out.merge(type_weekly, on=base_cols, how="left")
        out["type_due_weight"] = pd.to_numeric(
            out["type_due_weight"], errors="coerce"
        ).fillna(0.0)
        out[f"{prefix}_due_weight_to_date"] = out.groupby(
            list(KEY_COLUMNS), sort=False
        )["type_due_weight"].cumsum()
        out = out.drop(columns=["type_due_weight"])

    return out


def _build_mastery_numerators(
    positive_due: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    base_cols = list(KEY_COLUMNS) + ["week_number"]
    submitted = positive_due.loc[positive_due["effective_score_week"].notna()].copy()
    submitted["effective_score_week"] = submitted["effective_score_week"].astype(int)

    if submitted.empty:
        out = base.copy()
        out["overall_mastery_points_to_date"] = 0.0
        out["current_mastery_points"] = 0.0
        out["tma_mastery_points_to_date"] = 0.0
        out["cma_mastery_points_to_date"] = 0.0
        out["exam_mastery_points_to_date"] = 0.0
        out["submitted_due_count_to_date"] = 0.0
        return out

    numerator_weekly = (
        submitted.groupby(list(KEY_COLUMNS) + ["effective_score_week"], dropna=False)
        .agg(
            overall_mastery_points=("weighted_points", "sum"),
            submitted_due_count=("id_assessment", "nunique"),
        )
        .reset_index()
        .rename(columns={"effective_score_week": "week_number"})
    )
    out = base.merge(numerator_weekly, on=base_cols, how="left")
    out["overall_mastery_points"] = pd.to_numeric(
        out["overall_mastery_points"], errors="coerce"
    ).fillna(0.0)
    out = out.sort_values(base_cols).reset_index(drop=True)
    out["overall_mastery_points_to_date"] = out.groupby(
        list(KEY_COLUMNS), sort=False
    )["overall_mastery_points"].cumsum()
    out["submitted_due_count"] = pd.to_numeric(
        out["submitted_due_count"], errors="coerce"
    ).fillna(0.0)
    out["submitted_due_count_to_date"] = out.groupby(
        list(KEY_COLUMNS), sort=False
    )["submitted_due_count"].cumsum()

    current_points = (
        submitted.loc[submitted["due_week"] == submitted["effective_score_week"]]
        .groupby(list(KEY_COLUMNS) + ["due_week"], dropna=False)
        .agg(current_mastery_points=("weighted_points", "sum"))
        .reset_index()
        .rename(columns={"due_week": "week_number"})
    )
    out = out.merge(current_points, on=base_cols, how="left")
    out["current_mastery_points"] = pd.to_numeric(
        out["current_mastery_points"], errors="coerce"
    ).fillna(0.0)

    for prefix, assessment_type in [
        ("tma", "TMA"),
        ("cma", "CMA"),
        ("exam", "Exam"),
    ]:
        type_submitted = submitted.loc[
            submitted["assessment_type"] == assessment_type
        ].copy()
        if type_submitted.empty:
            out[f"{prefix}_mastery_points_to_date"] = 0.0
            continue
        type_weekly = (
            type_submitted.groupby(
                list(KEY_COLUMNS) + ["effective_score_week"], dropna=False
            )
            .agg(type_mastery_points=("weighted_points", "sum"))
            .reset_index()
            .rename(columns={"effective_score_week": "week_number"})
        )
        out = out.merge(type_weekly, on=base_cols, how="left")
        out["type_mastery_points"] = pd.to_numeric(
            out["type_mastery_points"], errors="coerce"
        ).fillna(0.0)
        out[f"{prefix}_mastery_points_to_date"] = out.groupby(
            list(KEY_COLUMNS), sort=False
        )["type_mastery_points"].cumsum()
        out = out.drop(columns=["type_mastery_points"])

    return out[
        base_cols
        + [
            "overall_mastery_points_to_date",
            "current_mastery_points",
            "tma_mastery_points_to_date",
            "cma_mastery_points_to_date",
            "exam_mastery_points_to_date",
            "submitted_due_count_to_date",
        ]
    ]


def _assessment_due_basis(
    tables: OuladTables,
    weekly_index: pd.DataFrame,
) -> pd.DataFrame:
    enrollment = weekly_index.loc[:, list(KEY_COLUMNS)].drop_duplicates()
    assessments = tables.assessments.loc[
        :, list(COURSE_COLUMNS) + ["id_assessment", "assessment_type", "date", "weight"]
    ].copy()
    due = enrollment.merge(assessments, on=list(COURSE_COLUMNS), how="left")
    submissions = _prepare_student_assessments(tables)
    return due.merge(
        submissions[
            ["id_assessment", "id_student", "date_submitted", "is_banked", "score"]
        ],
        on=["id_assessment", "id_student"],
        how="left",
    )


def _assessment_submissions_with_definition(tables: OuladTables) -> pd.DataFrame:
    submissions = _prepare_student_assessments(tables)
    return submissions.merge(
        tables.assessments[
            list(COURSE_COLUMNS) + ["id_assessment", "assessment_type", "date", "weight"]
        ],
        on="id_assessment",
        how="inner",
        validate="many_to_one",
    )


def _positive_scored_assessments(tables: OuladTables) -> pd.DataFrame:
    scored_ids = _scored_assessment_ids(tables)
    return tables.assessments.loc[
        (tables.assessments["weight"] > 0)
        & tables.assessments["id_assessment"].isin(scored_ids)
    ].copy()


def _scored_assessment_ids(tables: OuladTables) -> set[str]:
    score = pd.to_numeric(tables.student_assessment["score"], errors="coerce")
    return set(
        tables.student_assessment.loc[score.notna(), "id_assessment"]
        .dropna()
        .astype(str)
        .tolist()
    )


def _prepare_student_assessments(tables: OuladTables) -> pd.DataFrame:
    submissions = tables.student_assessment.copy()
    submissions = submissions.sort_values(["id_assessment", "id_student", "date_submitted"])
    submissions = submissions.drop_duplicates(["id_assessment", "id_student"], keep="last")
    return submissions


def _add_trend_and_index_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe trend and composite-index analogue columns.

    Trends are per-student week-over-week diffs of to-date columns (use only
    data up to week N). Indices are means of already-bounded [0,1] signals; no
    global/cross-row statistics are used, so no test-set information leaks. This
    mirrors the synthetic index intent documented in
    docs/research/deep-research-report.md (engagement/performance/discipline).

    Missing component columns (NaN on partially-observed early weeks) are
    treated as 0 via `.fillna(0.0)` before averaging, so each index always
    uses a FIXED denominator equal to the number of components (2 or 3).
    This is consistent with `_finalize_snapshot_frame`'s fill_zero treatment
    of the same input columns and is a conservative, leakage-safe choice.
    """
    out = snapshots.sort_values(list(KEY_COLUMNS) + ["week_number"]).reset_index(drop=True)
    group = out.groupby(list(KEY_COLUMNS), sort=False)

    out["assessment_score_trend_to_date"] = (
        out["cumulative_assessment_score_mean_to_date"]
        - group["cumulative_assessment_score_mean_to_date"].shift(1)
    ).fillna(0.0)
    out["clicks_trend_to_date"] = (
        out["current_week_clicks"] - group["current_week_clicks"].shift(1)
    ).fillna(0.0)

    perf_a = (out["cumulative_assessment_score_mean_to_date"] / 100.0).clip(0.0, 1.0)
    perf_b = (out["cumulative_assessment_weighted_score_to_date"] / 100.0).clip(0.0, 1.0)
    perf = pd.concat([perf_a, perf_b], axis=1).fillna(0.0)
    out["performance_index_oulad"] = perf.mean(axis=1)

    disc = pd.concat(
        [
            out["assessment_submission_rate_due_to_date"],
            1.0 - out["late_submission_rate_to_date"],
            out["banked_assessment_rate_to_date"],
        ],
        axis=1,
    ).fillna(0.0)
    out["discipline_index_oulad"] = disc.mean(axis=1)

    eng = pd.concat(
        [
            out["assessment_submission_rate_due_to_date"],
            out["has_vle_activity_to_date"].astype(float),
        ],
        axis=1,
    ).fillna(0.0)
    out["engagement_index_oulad"] = eng.mean(axis=1)

    out["content_click_ratio_to_date"] = (
        _safe_divide(
            out.get("cumulative_content_clicks_to_date", pd.Series(0.0, index=out.index)),
            out.get("cumulative_clicks_to_date", pd.Series(0.0, index=out.index)),
        )
        .fillna(0.0)
        .clip(0.0, 1.0)
    )
    return out


def _finalize_snapshot_frame(snapshots: pd.DataFrame) -> pd.DataFrame:
    fill_zero = [
        "cumulative_assessment_score_count_to_date",
        "cumulative_submitted_weight_to_date",
        "late_submission_rate_to_date",
        "banked_assessment_rate_to_date",
        "has_assessment_score_to_date",
        "has_weighted_score_to_date",
        "assessment_submission_rate_due_to_date",
        "mastery_assessment_coverage_to_date",
        "has_current_assessment_cluster",
        "has_due_assessment_to_date",
        "current_week_clicks",
        "current_week_activity_types",
        "cumulative_clicks_to_date",
        "has_vle_activity_to_date",
        "assessment_score_trend_to_date",
        "clicks_trend_to_date",
        "engagement_index_oulad",
        "performance_index_oulad",
        "discipline_index_oulad",
        "current_week_active_days",
        "cumulative_active_days_to_date",
        "content_click_ratio_to_date",
    ]
    for category in ACTIVITY_CATEGORIES:
        fill_zero.extend(
            [
                f"current_week_{category}_clicks",
                f"cumulative_{category}_clicks_to_date",
            ]
        )
    for column in fill_zero:
        if column in snapshots.columns:
            snapshots[column] = pd.to_numeric(snapshots[column], errors="coerce").fillna(0.0)

    preferred_order = [
        "student_id",
        "course_id",
        *KEY_COLUMNS,
        "week_number",
        "week_end_day",
        "duration_weeks",
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
        "cumulative_active_days_to_date",
        "content_click_ratio_to_date",
        "overall_mastery_proxy",
        "current_assessment_cluster_mastery",
        "tma_mastery_to_date",
        "cma_mastery_to_date",
        "exam_mastery_to_date",
        "mastery_assessment_coverage_to_date",
        "has_assessment_score_to_date",
        "has_weighted_score_to_date",
        "has_current_assessment_cluster",
        "has_due_assessment_to_date",
        "has_vle_activity_to_date",
        "final_weighted_score",
        "passed_observed",
        "final_result",
        "target_total_weight",
        "target_submitted_weight",
        "target_weight_coverage",
    ]
    remaining = [column for column in snapshots.columns if column not in preferred_order]
    ordered = [column for column in preferred_order if column in snapshots.columns]
    return snapshots.loc[:, ordered + remaining].copy()


def _empty_submitted_assessment_features(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for column in [
        "cumulative_assessment_score_mean_to_date",
        "cumulative_assessment_score_count_to_date",
        "cumulative_assessment_weighted_score_to_date",
        "cumulative_submitted_weight_to_date",
        "late_submission_rate_to_date",
        "banked_assessment_rate_to_date",
        "has_assessment_score_to_date",
        "has_weighted_score_to_date",
    ]:
        out[column] = 0.0
    return out


def _empty_mastery_features(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    for column in [
        "assessment_submission_rate_due_to_date",
        "overall_mastery_proxy",
        "current_assessment_cluster_mastery",
        "tma_mastery_to_date",
        "cma_mastery_to_date",
        "exam_mastery_to_date",
        "mastery_assessment_coverage_to_date",
        "has_current_assessment_cluster",
        "has_due_assessment_to_date",
    ]:
        out[column] = 0.0
    return out


def _normalize_table(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    out = frame.copy()
    for column in ["code_module", "code_presentation", "id_student", "id_assessment", "id_site"]:
        if column in out.columns:
            out[column] = out[column].astype("string").str.strip()
    numeric_by_table = {
        "assessments": ("date", "weight"),
        "courses": ("module_presentation_length",),
        "student_registration": ("date_registration", "date_unregistration"),
        "student_vle": ("date", "sum_click"),
        "student_assessment": ("date_submitted", "is_banked", "score"),
    }
    for column in numeric_by_table.get(table_name, ()):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if table_name == "vle" and "activity_type" in out.columns:
        out["activity_type"] = out["activity_type"].astype("string").str.strip()
        out["activity_category"] = (
            out["activity_type"].map(ACTIVITY_CATEGORY_BY_TYPE).fillna("other")
        )
    return out


def _validate_required_columns(frame: pd.DataFrame, table_name: str) -> None:
    missing = [
        column
        for column in REQUIRED_COLUMNS[table_name]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Missing expected OULAD columns in {table_name}: {missing}")


def _week_from_day(day: pd.Series | Iterable[float] | float) -> pd.Series:
    values = pd.to_numeric(day, errors="coerce")
    if not isinstance(values, pd.Series):
        values = pd.Series(values)
    week = np.floor(np.maximum(values.fillna(np.nan), 0) / 7.0) + 1
    week = week.where(values.notna())
    return week


def _map_final_result_to_passed(value: Any) -> float:
    if value in POSITIVE_FINAL_RESULTS:
        return 1.0
    if value in NEGATIVE_FINAL_RESULTS:
        return 0.0
    return float("nan")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.divide(denominator.where(denominator != 0))


def _safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _value_counts(series: pd.Series) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in series.value_counts(dropna=False).sort_index().items()
    }


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        # subtract the header; max protects empty test fixtures.
        return max(sum(1 for _ in handle) - 1, 0)
