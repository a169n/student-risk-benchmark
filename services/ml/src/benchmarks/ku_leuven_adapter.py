"""Adapter for the KU Leuven 2026 dataset (year 1819).

DATASET CITATION
----------------
Dataset: Tiukhova, E., Van Landuyt, D., Baesens, B., & Snoeck, M. (2026).
Open data, private learners: a de-identified student activity and performance
dataset for learning analytics. Scientific Data. CC-BY-4.0,
Zenodo DOI 10.5281/zenodo.17087849.

SCOPE AND HONEST FRAMING
-------------------------
This adapter produces weekly ENGAGEMENT snapshots for a binary PASSED
classification task.  It is NOT a mastery adapter.

The KU Leuven dataset has:
  - Raw click-stream logs with session and content-type information.
  - Forum post/read tables.
  - A binary outcome column PASSED (pass/fail per course attempt).
  - Score *buckets* (SCORE_CATEGORY_*), not numeric intermediate grades.

Because there are NO numeric intermediate assessment scores, the project's
mastery/assessment feature blocks CANNOT be built here.  The continuous
target column ``final_grade`` is set to ``passed.astype(float)`` purely as
a structural placeholder so the shared ``build_modeling_matrix`` function
(which requires a numeric ``final_grade`` column) can ingest the frame.
Regression on ``final_grade`` is NOT meaningful for KU Leuven; only
binary classification results are reported for this dataset.

WEEK MAPPING (leakage-safe)
-----------------------------
For each course the ``course_start`` date is read from ``course_info.json``.
Events are mapped to:

    week_number = floor((event_date - course_start).days / 7) + 1

Only weeks 1 .. n_semester (where n_semester = len(semester_weeks)) are
kept.  Pre-start activity (week < 1) and exam-period activity
(week > n_semester) are dropped.  Prediction target PASSED is post-exam,
so keeping only pre-exam engagement is correct and leakage-free.

GRAIN
-----
One row = one (course, student) x one semester week.  A complete grid
(all semester weeks, even inactive ones) is built so cumulative features
are well-defined for every week.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The dataset ships one set of files per academic year, prefixed with the year
# key used in ``course_info.json`` (e.g. "1819_log_activity.csv"). These names
# are templates; ``_year_files`` fills in the requested year.
LOG_FILE = "1819_log_activity.csv"
CONTENT_FILE = "1819_course_content.xlsx"
PARTICIPATION_FILE = "1819_course_participation.xlsx"
CONTRIBUTION_FILE = "1819_df_contribution.xlsx"
CONSUMPTION_FILE = "1819_df_consumption.xlsx"


def _year_files(year: str) -> dict[str, str]:
    """Per-year filenames. Passing "1819" reproduces the original constants."""
    return {
        "log": f"{year}_log_activity.csv",
        "content": f"{year}_course_content.xlsx",
        "participation": f"{year}_course_participation.xlsx",
        "contribution": f"{year}_df_contribution.xlsx",
        "consumption": f"{year}_df_consumption.xlsx",
    }

REQUIRED_LOG_COLUMNS: tuple[str, ...] = (
    "ACTION_ID",
    "COURSE_ID",
    "USER_ID",
    "CONTENT_ID",
    "SESSION_ID",
    "TIMESTAMP",
)

REQUIRED_PARTICIPATION_COLUMNS: tuple[str, ...] = (
    "COURSE_ID",
    "USER_ID",
    "PASSED",
)

REQUIRED_CONTENT_COLUMNS: tuple[str, ...] = (
    "COURSE_ID",
    "CONTENT_ID",
    "CONTENT_TYPE",
)

REQUIRED_CONTRIBUTION_COLUMNS: tuple[str, ...] = (
    "COURSE_ID",
    "USER_ID",
    "DTCREATED",
)

COURSE_MATERIAL_TYPE: str = "Course Material"


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KuLeuvenSnapshotBuildResult:
    """Container returned by ``build_weekly_engagement_snapshots``.

    Attributes
    ----------
    snapshots:
        Weekly engagement snapshot DataFrame.  Grain: (student_id, week_number).
        Includes identifiers, all engagement features, ``passed`` (int 0/1),
        and ``final_grade`` (float placeholder equal to ``passed``).
    row_counts:
        Dictionary of row counts from each input file and the built snapshot.
    target_summary:
        High-level statistics about the target and snapshot coverage.
    """

    snapshots: pd.DataFrame
    row_counts: dict[str, int]
    target_summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_weekly_engagement_snapshots(
    data_dir: Path | str,
    course_info_path: Path | str,
    *,
    year: str = "1819",
    min_week: int = 2,
    chunksize: int = 500_000,
) -> KuLeuvenSnapshotBuildResult:
    """Build weekly engagement snapshots for KU Leuven (year 1819, both courses).

    Parameters
    ----------
    data_dir:
        Directory containing the KU Leuven dataset files (e.g. ``…/dataset/``).
    course_info_path:
        Path to ``course_info.json`` with course calendar metadata.
    year:
        Year key in ``course_info.json``; defaults to ``"1819"``.
    min_week:
        Minimum semester week to keep in the output (default 2).  Cumulative
        features are computed from week 1 onwards; only the final filter
        respects ``min_week``.
    chunksize:
        Row chunk size for reading the 83 MB log CSV.

    Returns
    -------
    KuLeuvenSnapshotBuildResult
    """
    if min_week < 1:
        raise ValueError("min_week must be >= 1")

    data_dir = Path(data_dir)
    course_info_path = Path(course_info_path)

    # --- load course calendar --------------------------------------------------
    course_calendar = _load_course_calendar(course_info_path, year)

    # --- load small files ------------------------------------------------------
    files = _year_files(year)
    participation = _load_participation(data_dir / files["participation"])
    content_lookup = _load_content_lookup(data_dir / files["content"])
    contribution = _load_contribution(data_dir / files["contribution"])

    row_counts: dict[str, int] = {
        "participation": int(participation.shape[0]),
        "course_content": int(content_lookup.shape[0]),
        "forum_contribution": int(contribution.shape[0]),
    }

    # Parallel sections ("Global economics 1"/"2") inherit the base calendar.
    course_calendar = _expand_calendar_aliases(course_calendar, participation["COURSE_ID"])

    # --- build per-course course-material set and forum weekly aggregates ------
    # These are small; pre-build per course before the chunked log pass.
    forum_weekly = _build_forum_weekly(contribution, course_calendar)

    # --- chunked log aggregation -----------------------------------------------
    log_path = data_dir / files["log"]
    weekly_agg, log_row_count = _aggregate_log_chunked(
        log_path,
        content_lookup=content_lookup,
        course_calendar=course_calendar,
        chunksize=chunksize,
    )
    row_counts["log_activity"] = log_row_count

    # --- build complete grid (student x week) ----------------------------------
    snapshots = _build_complete_grid(
        weekly_agg=weekly_agg,
        forum_weekly=forum_weekly,
        course_calendar=course_calendar,
        participation=participation,
    )

    # --- apply min_week filter -------------------------------------------------
    snapshots = snapshots.loc[snapshots["week_number"] >= min_week].copy()
    snapshots = snapshots.reset_index(drop=True)

    if snapshots.empty:
        raise ValueError(
            f"KU Leuven snapshot frame is empty after applying min_week={min_week}. "
            "Check that the log file covers the expected semester period."
        )

    row_counts["snapshots"] = int(snapshots.shape[0])
    row_counts["students"] = int(snapshots["student_id"].nunique())

    target_summary = _build_target_summary(snapshots)

    return KuLeuvenSnapshotBuildResult(
        snapshots=snapshots,
        row_counts=row_counts,
        target_summary=target_summary,
    )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_course_calendar(
    course_info_path: Path,
    year: str,
) -> dict[str, dict[str, Any]]:
    """Parse course_info.json and return per-course calendar for ``year``.

    Returns a dict keyed by COURSE_ID (e.g. "Accountancy") with keys:
        course_start (date), n_semester (int).
    """
    with course_info_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if year not in raw:
        raise KeyError(f"Year '{year}' not found in course_info.json. Available: {list(raw)}")
    year_data = raw[year]
    calendar: dict[str, dict[str, Any]] = {}
    for course_name, info in year_data.items():
        cs = info["course_start"]
        course_start = date(cs[0], cs[1], cs[2])
        n_semester = len(info["semester_weeks"])
        calendar[course_name] = {
            "course_start": course_start,
            "n_semester": n_semester,
        }
    return calendar


def _expand_calendar_aliases(
    calendar: dict[str, dict[str, Any]],
    observed_courses: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Attach calendar entries to course names that vary from ``course_info.json``.

    In 2020-21 the dataset splits one course into parallel sections
    ("Global economics" becomes "Global economics 1" and "Global economics 2")
    while ``course_info.json`` still carries a single calendar for the base
    course. Sections share the base course's semester dates, so each observed
    course name is matched to the LONGEST calendar key that is a prefix of it.
    Unmatched course names are returned unchanged (the caller drops them), and
    exact matches are never overridden.
    """
    expanded = dict(calendar)
    for course in sorted({str(c).strip() for c in observed_courses}):
        if course in expanded:
            continue
        candidates = [key for key in calendar if course.startswith(key)]
        if not candidates:
            continue
        expanded[course] = calendar[max(candidates, key=len)]
    return expanded


def _load_participation(path: Path) -> pd.DataFrame:
    """Load course_participation, validate columns, normalise IDs."""
    df = pd.read_excel(path, engine="openpyxl")
    _check_columns(df, REQUIRED_PARTICIPATION_COLUMNS, str(path))
    df["COURSE_ID"] = df["COURSE_ID"].astype(str).str.strip()
    df["USER_ID"] = df["USER_ID"].astype(str).str.strip()
    df["PASSED"] = pd.to_numeric(df["PASSED"], errors="coerce")
    df = df.dropna(subset=["PASSED"])
    df["PASSED"] = df["PASSED"].astype(int)
    df["student_id"] = df["COURSE_ID"] + "__" + df["USER_ID"]
    return df.reset_index(drop=True)


def _load_content_lookup(path: Path) -> pd.DataFrame:
    """Load course_content and build a (COURSE_ID, CONTENT_ID) -> is_material bool lookup."""
    df = pd.read_excel(path, engine="openpyxl")
    _check_columns(df, REQUIRED_CONTENT_COLUMNS, str(path))
    df["COURSE_ID"] = df["COURSE_ID"].astype(str).str.strip()
    df["CONTENT_ID"] = df["CONTENT_ID"].astype(str).str.strip()
    df["CONTENT_TYPE"] = df["CONTENT_TYPE"].astype(str).str.strip()
    df["is_course_material"] = (df["CONTENT_TYPE"] == COURSE_MATERIAL_TYPE).astype(int)
    return df[["COURSE_ID", "CONTENT_ID", "is_course_material"]].drop_duplicates(
        subset=["COURSE_ID", "CONTENT_ID"]
    ).reset_index(drop=True)


def _load_contribution(path: Path) -> pd.DataFrame:
    """Load df_contribution (forum posts with timestamps)."""
    df = pd.read_excel(path, engine="openpyxl")
    _check_columns(df, REQUIRED_CONTRIBUTION_COLUMNS, str(path))
    df["COURSE_ID"] = df["COURSE_ID"].astype(str).str.strip()
    df["USER_ID"] = df["USER_ID"].astype(str).str.strip()
    # DTCREATED may be datetime or string
    df["DTCREATED"] = pd.to_datetime(df["DTCREATED"], errors="coerce")
    df = df.dropna(subset=["DTCREATED"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Forum weekly aggregation
# ---------------------------------------------------------------------------


def _build_forum_weekly(
    contribution: pd.DataFrame,
    course_calendar: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Return a (COURSE_ID, USER_ID, week_number) table of weekly post counts.

    Only weeks in [1 .. n_semester] are kept (pre-exam engagement).
    """
    rows = []
    for _, row in contribution.iterrows():
        course_id = row["COURSE_ID"]
        if course_id not in course_calendar:
            continue
        cal = course_calendar[course_id]
        event_date = row["DTCREATED"].date() if hasattr(row["DTCREATED"], "date") else row["DTCREATED"]
        wn = _date_to_week_number(event_date, cal["course_start"])
        if wn is None or wn < 1 or wn > cal["n_semester"]:
            continue
        rows.append({"COURSE_ID": course_id, "USER_ID": str(row["USER_ID"]).strip(), "week_number": wn})

    if not rows:
        return pd.DataFrame(columns=["COURSE_ID", "USER_ID", "week_number", "forum_posts"])

    tmp = pd.DataFrame(rows)
    agg = (
        tmp.groupby(["COURSE_ID", "USER_ID", "week_number"])
        .size()
        .rename("forum_posts")
        .reset_index()
    )
    return agg


# ---------------------------------------------------------------------------
# Chunked log aggregation
# ---------------------------------------------------------------------------


def _aggregate_log_chunked(
    log_path: Path,
    *,
    content_lookup: pd.DataFrame,
    course_calendar: dict[str, dict[str, Any]],
    chunksize: int,
) -> tuple[pd.DataFrame, int]:
    """Read log in chunks; accumulate per-(COURSE_ID, USER_ID, week_number) partial aggregates.

    Returns (weekly_agg DataFrame, total_log_rows_read).

    Per-(course, user, week) aggregates:
        clicks          - row count
        sessions        - set of SESSION_IDs  (track as nunique via partial sets)
        active_days     - set of calendar dates
        content_clicks  - clicks on CONTENT_ID mapped to "Course Material"

    Because we need nunique for sessions and dates, we collect them as
    frozen-set intermediates across chunks then finalise.
    """
    # For memory efficiency: accumulate numeric totals per key and track
    # unique SESSION_ID and date via sets stored in dicts.
    # Key: (COURSE_ID, USER_ID, week_number)

    clicks_acc: dict[tuple, int] = {}
    session_acc: dict[tuple, set] = {}
    day_acc: dict[tuple, set] = {}
    content_acc: dict[tuple, int] = {}

    # Build a flat set lookup per (COURSE_ID, CONTENT_ID) -> is_material
    material_set: set[tuple] = set(
        zip(
            content_lookup.loc[content_lookup["is_course_material"] == 1, "COURSE_ID"],
            content_lookup.loc[content_lookup["is_course_material"] == 1, "CONTENT_ID"],
        )
    )

    # Pre-compute course_start ordinals and n_semester for fast lookup in the loop
    cal_start: dict[str, date] = {k: v["course_start"] for k, v in course_calendar.items()}
    cal_nsem: dict[str, int] = {k: v["n_semester"] for k, v in course_calendar.items()}

    total_rows = 0
    for chunk in pd.read_csv(log_path, chunksize=chunksize, low_memory=False):
        _check_columns(chunk, REQUIRED_LOG_COLUMNS, LOG_FILE)
        total_rows += len(chunk)

        # Normalise IDs
        chunk["COURSE_ID"] = chunk["COURSE_ID"].astype(str).str.strip()
        chunk["USER_ID"] = chunk["USER_ID"].astype(str).str.strip()
        chunk["CONTENT_ID"] = chunk["CONTENT_ID"].astype(str).str.strip()
        chunk["SESSION_ID"] = chunk["SESSION_ID"].astype(str).str.strip()

        # Parse timestamps -> date
        ts = pd.to_datetime(chunk["TIMESTAMP"], errors="coerce")
        chunk = chunk.loc[ts.notna()].copy()
        if chunk.empty:
            continue
        chunk["ev_date"] = ts.loc[chunk.index].dt.date

        # Map each row to a week_number per its course using vectorised logic
        # Process per-course for efficiency (avoids per-row python dict lookup)
        chunk_parts = []
        for course_id, c_start in cal_start.items():
            c_chunk = chunk.loc[chunk["COURSE_ID"] == course_id].copy()
            if c_chunk.empty:
                continue
            # days delta as integer
            c_chunk["ev_wn"] = c_chunk["ev_date"].apply(
                lambda d, cs=c_start: math.floor((d - cs).days / 7) + 1
            )
            # Filter to valid semester weeks [1, n_semester]
            n_sem = cal_nsem[course_id]
            c_chunk = c_chunk.loc[
                (c_chunk["ev_wn"] >= 1) & (c_chunk["ev_wn"] <= n_sem)
            ].copy()
            if c_chunk.empty:
                continue
            # is_material flag
            c_chunk["is_mat"] = c_chunk.apply(
                lambda r: 1 if (r["COURSE_ID"], r["CONTENT_ID"]) in material_set else 0,
                axis=1,
            )
            chunk_parts.append(c_chunk)

        if not chunk_parts:
            continue

        filtered = pd.concat(chunk_parts, ignore_index=True)
        if filtered.empty:
            continue

        # Accumulate via pandas groupby aggregations for speed, then merge into dicts
        # clicks: count per key
        clicks_gb = (
            filtered.groupby(["COURSE_ID", "USER_ID", "ev_wn"])
            .size()
            .reset_index(name="n_clicks")
        )
        for row in clicks_gb.itertuples(index=False):
            key = (row.COURSE_ID, row.USER_ID, row.ev_wn)
            clicks_acc[key] = clicks_acc.get(key, 0) + row.n_clicks

        # content clicks: sum of is_mat per key
        content_gb = (
            filtered.groupby(["COURSE_ID", "USER_ID", "ev_wn"])["is_mat"]
            .sum()
            .reset_index(name="n_content")
        )
        for row in content_gb.itertuples(index=False):
            key = (row.COURSE_ID, row.USER_ID, row.ev_wn)
            content_acc[key] = content_acc.get(key, 0) + int(row.n_content)

        # sessions: collect sets across chunks
        for row in filtered[["COURSE_ID", "USER_ID", "ev_wn", "SESSION_ID"]].itertuples(index=False):
            key = (row.COURSE_ID, row.USER_ID, row.ev_wn)
            if key not in session_acc:
                session_acc[key] = set()
            session_acc[key].add(row.SESSION_ID)

        # active days: collect sets across chunks
        for row in filtered[["COURSE_ID", "USER_ID", "ev_wn", "ev_date"]].itertuples(index=False):
            key = (row.COURSE_ID, row.USER_ID, row.ev_wn)
            if key not in day_acc:
                day_acc[key] = set()
            day_acc[key].add(row.ev_date)

    # Assemble final weekly DataFrame from accumulators
    if not clicks_acc:
        weekly_agg = pd.DataFrame(
            columns=[
                "COURSE_ID",
                "USER_ID",
                "week_number",
                "current_week_clicks",
                "current_week_sessions",
                "current_week_active_days",
                "current_week_content_clicks",
            ]
        )
    else:
        records = []
        for key, clicks in clicks_acc.items():
            course_id, user_id, week_number = key
            records.append(
                {
                    "COURSE_ID": course_id,
                    "USER_ID": user_id,
                    "week_number": week_number,
                    "current_week_clicks": clicks,
                    "current_week_sessions": len(session_acc.get(key, set())),
                    "current_week_active_days": len(day_acc.get(key, set())),
                    "current_week_content_clicks": content_acc.get(key, 0),
                }
            )
        weekly_agg = pd.DataFrame(records)

    return weekly_agg, total_rows


# ---------------------------------------------------------------------------
# Complete grid + cumulative features
# ---------------------------------------------------------------------------


def _build_complete_grid(
    *,
    weekly_agg: pd.DataFrame,
    forum_weekly: pd.DataFrame,
    course_calendar: dict[str, dict[str, Any]],
    participation: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full (student_id, week_number) grid with cumulative features and targets.

    Steps:
    1.  Enumerate all (course, user) pairs from participation.
    2.  Cross with all semester weeks 1 .. n_semester for that course.
    3.  Left-join observed weekly click activity (zeros for inactive weeks).
    4.  Left-join forum weekly posts (zeros for no-post weeks).
    5.  Sort by (student_id, week_number) and compute cumulative-to-date features.
    6.  Join PASSED target.
    """

    # ---- build enrollment base (all (course, user) combos) ------------------
    enrollment_rows = []
    for course_id, cal in course_calendar.items():
        course_part = participation.loc[participation["COURSE_ID"] == course_id].copy()
        if course_part.empty:
            continue
        for _, prow in course_part.iterrows():
            for wn in range(1, cal["n_semester"] + 1):
                enrollment_rows.append(
                    {
                        "COURSE_ID": course_id,
                        "USER_ID": prow["USER_ID"],
                        "student_id": prow["student_id"],
                        "week_number": wn,
                        "n_semester": cal["n_semester"],
                        "course_start_ordinal": cal["course_start"].toordinal(),
                    }
                )

    if not enrollment_rows:
        raise ValueError(
            "No enrollment rows could be built.  Check that participation and "
            "course_calendar share COURSE_ID values."
        )

    grid = pd.DataFrame(enrollment_rows)

    # ---- join observed weekly click activity ---------------------------------
    if not weekly_agg.empty:
        weekly_agg["USER_ID"] = weekly_agg["USER_ID"].astype(str).str.strip()
        weekly_agg["COURSE_ID"] = weekly_agg["COURSE_ID"].astype(str).str.strip()
        grid = grid.merge(
            weekly_agg[
                [
                    "COURSE_ID",
                    "USER_ID",
                    "week_number",
                    "current_week_clicks",
                    "current_week_sessions",
                    "current_week_active_days",
                    "current_week_content_clicks",
                ]
            ],
            on=["COURSE_ID", "USER_ID", "week_number"],
            how="left",
        )
    else:
        for col in [
            "current_week_clicks",
            "current_week_sessions",
            "current_week_active_days",
            "current_week_content_clicks",
        ]:
            grid[col] = 0.0

    for col in [
        "current_week_clicks",
        "current_week_sessions",
        "current_week_active_days",
        "current_week_content_clicks",
    ]:
        grid[col] = pd.to_numeric(grid[col], errors="coerce").fillna(0.0)

    # ---- join forum weekly posts ---------------------------------------------
    if not forum_weekly.empty:
        forum_weekly["USER_ID"] = forum_weekly["USER_ID"].astype(str).str.strip()
        forum_weekly["COURSE_ID"] = forum_weekly["COURSE_ID"].astype(str).str.strip()
        grid = grid.merge(
            forum_weekly,
            on=["COURSE_ID", "USER_ID", "week_number"],
            how="left",
        )
    else:
        grid["forum_posts"] = 0.0

    grid["forum_posts"] = pd.to_numeric(grid["forum_posts"], errors="coerce").fillna(0.0)

    # ---- sort for cumsum -----------------------------------------------------
    grid = grid.sort_values(["student_id", "week_number"]).reset_index(drop=True)
    grp = grid.groupby("student_id", sort=False)

    # ---- cumulative features (leakage-safe: all use cumsum along sorted weeks) -
    grid["cumulative_clicks_to_date"] = grp["current_week_clicks"].cumsum()
    grid["cumulative_sessions_to_date"] = grp["current_week_sessions"].cumsum()
    grid["cumulative_active_days_to_date"] = grp["current_week_active_days"].cumsum()
    grid["cumulative_content_clicks_to_date"] = grp["current_week_content_clicks"].cumsum()
    grid["cumulative_forum_posts_to_date"] = grp["forum_posts"].cumsum()

    # avg_session_clicks: total clicks / total sessions (safe-divide)
    grid["avg_session_clicks_to_date"] = _safe_divide(
        grid["cumulative_clicks_to_date"],
        grid["cumulative_sessions_to_date"],
    )

    # content_click_ratio: content_clicks / total_clicks (safe-divide)
    grid["content_click_ratio_to_date"] = _safe_divide(
        grid["cumulative_content_clicks_to_date"],
        grid["cumulative_clicks_to_date"],
    )

    grid["has_activity_to_date"] = (grid["cumulative_clicks_to_date"] > 0).astype(int)

    # ---- time-based positional features -------------------------------------
    # days_since_course_start uses week_number*7 (end of week N is 7N days in)
    grid["days_since_course_start"] = grid["week_number"] * 7
    grid["course_week_progress"] = (grid["week_number"] / grid["n_semester"]).clip(upper=1.0)

    # course_id friendly name
    grid["course_id"] = grid["COURSE_ID"]

    # ---- join targets --------------------------------------------------------
    # Drop duplicate (student_id, PASSED) rows before building the lookup.
    # Duplicates arise from repeated participation rows with identical outcomes
    # (same student, same course, same PASSED value — e.g. re-enrolled rows).
    participation_dedup = (
        participation[["student_id", "PASSED"]]
        .drop_duplicates(subset="student_id", keep="last")
    )
    target_map = participation_dedup.set_index("student_id")["PASSED"]
    grid["passed"] = grid["student_id"].map(target_map).astype("Int64").astype(int)
    # final_grade placeholder: NOT meaningful for regression on KU Leuven.
    # It is set equal to passed so that shared pipeline code requiring a
    # numeric final_grade column can ingest this frame. Any regression metric
    # computed on this column is meaningless and should be ignored.
    grid["final_grade"] = grid["passed"].astype(float)

    # ---- drop rows with missing passed (should not happen after the above) ---
    grid = grid.dropna(subset=["passed"])

    # ---- preferred column order ---------------------------------------------
    preferred = [
        "student_id",
        "course_id",
        "COURSE_ID",
        "USER_ID",
        "week_number",
        "n_semester",
        "days_since_course_start",
        "course_week_progress",
        "current_week_clicks",
        "current_week_sessions",
        "current_week_active_days",
        "current_week_content_clicks",
        "forum_posts",
        "cumulative_clicks_to_date",
        "cumulative_sessions_to_date",
        "cumulative_active_days_to_date",
        "cumulative_content_clicks_to_date",
        "cumulative_forum_posts_to_date",
        "avg_session_clicks_to_date",
        "content_click_ratio_to_date",
        "has_activity_to_date",
        "passed",
        "final_grade",
    ]
    remaining = [c for c in grid.columns if c not in preferred]
    ordered = [c for c in preferred if c in grid.columns]
    grid = grid[ordered + remaining].copy()

    return grid.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Target summary
# ---------------------------------------------------------------------------


def _build_target_summary(snapshots: pd.DataFrame) -> dict[str, Any]:
    """Compact statistics about the snapshot for reporting."""
    # Student-level: use the last week row per student
    student_level = (
        snapshots.sort_values("week_number")
        .drop_duplicates("student_id", keep="last")
    )
    n_students = int(student_level.shape[0])
    passed_counts = student_level["passed"].value_counts().sort_index().to_dict()
    n_passed = int(passed_counts.get(1, 0))
    n_failed = int(passed_counts.get(0, 0))
    passed_rate = n_passed / n_students if n_students > 0 else float("nan")

    # Mean cumulative clicks at last week
    last_week_clicks_mean = float(student_level["cumulative_clicks_to_date"].mean())

    return {
        "n_snapshot_rows": int(snapshots.shape[0]),
        "n_students": n_students,
        "week_min": int(snapshots["week_number"].min()),
        "week_max": int(snapshots["week_number"].max()),
        "passed_counts": {str(k): v for k, v in passed_counts.items()},
        "n_passed": n_passed,
        "n_failed": n_failed,
        "passed_rate_student_level": round(passed_rate, 4),
        "mean_cumulative_clicks_at_last_week": round(last_week_clicks_mean, 2),
    }


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _date_to_week_number(event_date: date, course_start: date) -> int | None:
    """Map a calendar date to a 1-based semester week number.

    Returns None for dates before course_start.
    """
    delta = (event_date - course_start).days
    if delta < 0:
        return None
    return math.floor(delta / 7) + 1


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division; returns 0.0 where denominator is zero or NaN."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    return num.divide(den.where(den != 0)).fillna(0.0)


def _check_columns(
    df: pd.DataFrame,
    required: tuple[str, ...],
    source: str,
) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in '{source}': {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )
