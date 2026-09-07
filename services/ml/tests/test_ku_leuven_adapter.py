"""TDD tests for the KU Leuven 2026 engagement snapshot adapter.

These tests exercise the REAL data files (year 1819) located under
datasets/ku_leuven/dataset/.  The 83 MB log file is read in chunks so the
suite may take ~1-2 minutes; that is acceptable.

What this adapter is and is NOT:
  - IS: an engagement/clickstream classification adapter for binary PASSED.
  - IS NOT: a mastery or assessment-score adapter.  The KU Leuven dataset
    has no numeric intermediate grades; only categorical score buckets and
    a binary PASSED outcome.  The ``final_grade`` column is a numeric
    placeholder equal to ``passed`` and regression on it is NOT meaningful.

Tests
-----
test_snapshots_build_and_leakage_safe
    Full build + structural and leakage-safe cumulative-feature checks.
test_target_balance
    Student-level PASSED rate sanity vs. known 949/550 ~ 0.633.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path constants (resolved relative to repo root)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_DIR = _REPO_ROOT / "datasets" / "ku_leuven" / "dataset"
_COURSE_INFO = _REPO_ROOT / "datasets" / "ku_leuven" / "course_info.json"

# Skip the whole module if data is absent (allows CI without the dataset)
pytestmark = pytest.mark.skipif(
    not (_DATA_DIR / "1819_log_activity.csv").exists(),
    reason="KU Leuven dataset not present; skipping.",
)


# ---------------------------------------------------------------------------
# Shared fixture: build snapshots once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def snapshots_result():
    """Build KU Leuven weekly engagement snapshots (may take ~1-2 min)."""
    from src.benchmarks.ku_leuven_adapter import build_weekly_engagement_snapshots

    return build_weekly_engagement_snapshots(
        data_dir=_DATA_DIR,
        course_info_path=_COURSE_INFO,
        year="1819",
        min_week=2,
        chunksize=500_000,
    )


# ---------------------------------------------------------------------------
# Test 1: structural + leakage-safe cumulative check
# ---------------------------------------------------------------------------


def test_snapshots_build_and_leakage_safe(snapshots_result):
    """Verify snapshot shape, required columns, value ranges, and leakage safety."""
    result = snapshots_result
    df = result.snapshots

    # -- non-empty ----------------------------------------------------------------
    assert not df.empty, "Snapshots DataFrame is empty"

    # -- required columns ---------------------------------------------------------
    required_cols = [
        "student_id",
        "week_number",
        "passed",
        "final_grade",
        "cumulative_clicks_to_date",
        "cumulative_sessions_to_date",
        "content_click_ratio_to_date",
        "has_activity_to_date",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    # -- passed is binary 0/1 ----------------------------------------------------
    unique_passed = set(df["passed"].dropna().unique())
    assert unique_passed <= {0, 1}, (
        f"passed column contains values outside {{0, 1}}: {unique_passed}"
    )

    # -- week_number within [2, max_semester_weeks] for each course ---------------
    # Both courses: Accountancy has 13 weeks, Global economics has 15 weeks.
    # With min_week=2 all weeks must be >= 2; no week may exceed n_semester.
    assert df["week_number"].min() >= 2, (
        f"week_number minimum is {df['week_number'].min()}, expected >= 2"
    )
    # Each student row has a corresponding n_semester; verify week_number <= n_semester
    over_semester = df.loc[df["week_number"] > df["n_semester"]]
    assert over_semester.empty, (
        f"Found {len(over_semester)} rows with week_number > n_semester: "
        f"{over_semester[['student_id', 'week_number', 'n_semester']].head()}"
    )

    # -- leakage-safe cumulative: cumulative_clicks_to_date non-decreasing per student --
    # Sample up to 200 students for speed.
    sample_students = df["student_id"].unique()[:200]
    for sid in sample_students:
        sub = df.loc[df["student_id"] == sid].sort_values("week_number")
        cum_clicks = sub["cumulative_clicks_to_date"].values
        diffs = cum_clicks[1:] - cum_clicks[:-1]
        assert (diffs >= -1e-9).all(), (
            f"Student {sid}: cumulative_clicks_to_date is not non-decreasing. "
            f"Diffs: {diffs}"
        )

    # -- final_grade == passed numerically ----------------------------------------
    mismatch = df.loc[df["final_grade"] != df["passed"].astype(float)]
    assert mismatch.empty, (
        f"final_grade != passed for {len(mismatch)} rows (first 3):\n"
        f"{mismatch[['student_id', 'week_number', 'passed', 'final_grade']].head(3)}"
    )

    # -- content_click_ratio in [0, 1] --------------------------------------------
    bad_ratio = df.loc[
        df["content_click_ratio_to_date"].notna()
        & ((df["content_click_ratio_to_date"] < -1e-9) | (df["content_click_ratio_to_date"] > 1 + 1e-9))
    ]
    assert bad_ratio.empty, (
        f"content_click_ratio_to_date out of [0,1] for {len(bad_ratio)} rows"
    )

    # -- has_activity_to_date is 0/1 int -------------------------------------------
    unique_haat = set(df["has_activity_to_date"].dropna().unique())
    assert unique_haat <= {0, 1}, (
        f"has_activity_to_date values outside {{0, 1}}: {unique_haat}"
    )


# ---------------------------------------------------------------------------
# Test 2: student-level PASSED rate sanity
# ---------------------------------------------------------------------------


def test_target_balance(snapshots_result):
    """Student-level PASSED rate must lie in [0.55, 0.75].

    Known ground truth: 949 passed / 1499 total ~ 0.633 across both courses.
    """
    df = snapshots_result.snapshots

    # One row per student (last week row -> passed is constant per student anyway)
    student_level = (
        df.sort_values("week_number")
        .drop_duplicates("student_id", keep="last")
    )
    n_total = len(student_level)
    n_passed = int((student_level["passed"] == 1).sum())
    passed_rate = n_passed / n_total if n_total > 0 else 0.0

    assert 0.55 <= passed_rate <= 0.75, (
        f"Student-level PASSED rate {passed_rate:.4f} is outside expected range [0.55, 0.75]. "
        f"n_total={n_total}, n_passed={n_passed}. "
        "This may indicate a join error or wrong year/course filter."
    )


def test_expand_calendar_aliases_maps_parallel_sections_to_base_course():
    """2020-21 splits "Global economics" into sections 1 and 2; both inherit its calendar."""
    from src.benchmarks.ku_leuven_adapter import _expand_calendar_aliases

    base = {"Global economics": {"n_semester": 15}, "Accountancy": {"n_semester": 13}}
    out = _expand_calendar_aliases(
        base, ["Accountancy", "Global economics 1", "Global economics 2", "Unrelated course"]
    )
    assert out["Global economics 1"]["n_semester"] == 15
    assert out["Global economics 2"]["n_semester"] == 15
    assert out["Accountancy"]["n_semester"] == 13
    assert "Unrelated course" not in out  # unmatched names stay dropped
    assert base == {"Global economics": {"n_semester": 15}, "Accountancy": {"n_semester": 13}}
