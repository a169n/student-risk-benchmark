from pathlib import Path

import pandas as pd
from src.benchmarks.oulad_adapter import (
    OuladCourseFilter,
    OuladRawPaths,
    build_weekly_snapshots,
)
from src.generator.config import REPO_ROOT

OULAD = REPO_ROOT / "datasets" / "oulad"


def _snapshots():
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


def test_engagement_columns_present_bounded_and_monotone():
    snaps = _snapshots()
    for col in ["cumulative_active_days_to_date", "content_click_ratio_to_date"]:
        assert col in snaps.columns, f"missing {col}"

    ratio = pd.to_numeric(snaps["content_click_ratio_to_date"], errors="coerce")
    assert ratio.min() >= 0.0
    assert ratio.max() <= 1.0

    # KEY_COLUMNS are ("code_module", "code_presentation", "id_student")
    key = ["code_module", "code_presentation", "id_student"]
    ordered = snaps.sort_values(key + ["week_number"])
    diffs = ordered.groupby(key)["cumulative_active_days_to_date"].diff().dropna()
    assert (diffs >= -1e-9).all(), "cumulative_active_days_to_date must be monotone non-decreasing"
