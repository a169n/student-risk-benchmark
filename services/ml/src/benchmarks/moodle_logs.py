"""Shared helpers for institutions that publish a Moodle standard log export.

Two institutions in the benchmark ship the same log shape (UKZN, University of
Zambia): one row per event with a timestamp, a Moodle ``Component``, and a
student identifier, plus a separate outcome table. The cohort builder below is
the dataset-agnostic half of that pipeline; institution modules keep only their
own loading, identifier handling, and label rules.

Course calendars are not published with these exports, so the active span is
INFERRED from the log: the first to the last Monday-anchored calendar week in
which at least ``active_share`` of enrolled students were active. Weeks outside
that span (pre-start browsing, next-year revisits) are dropped.
"""

from __future__ import annotations

import pandas as pd

CONTENT_COMPONENTS = frozenset(
    {
        "File",
        "Folder",
        "URL",
        "Page",
        "Book",
        "Lesson",
        "Resource",
        "Kaltura Video Presentation",
        "Kaltura Video Resource",
        "Kaltura Media Assignment",
    }
)
SOCIAL_COMPONENTS = frozenset({"Forum", "Chat"})

WEEKLY_COLUMNS = (
    "current_clicks",
    "current_active_days",
    "current_content_clicks",
    "current_social_clicks",
)


def build_cohort_frame(
    log: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    year: int | None,
    active_share: float = 0.10,
    min_students: int = 50,
    min_minority: int = 0,
) -> pd.DataFrame | None:
    """Weekly student x week grid for one cohort, or None if too small / no active span.

    Parameters
    ----------
    log:
        Columns ``student_id``, ``ts`` (datetime), ``component``.
    labels:
        Columns ``student_id``, ``year``, ``passed``.
    year:
        Calendar year of the cohort; log rows outside it are dropped. Pass
        ``None`` for a cohort whose academic year crosses the calendar boundary
        (Oviedo runs September to June), in which case ``labels`` must already
        be restricted to that cohort and no date filter is applied.
    min_minority:
        Minimum number of students in the rarer outcome class; below this an
        AUC estimate is too unstable to be worth a row in the ladder.

    The cohort's students are those that appear BOTH in ``labels`` and in the
    log, which avoids labelling students whose site is missing from the export.
    """
    if year is None:
        enrolled = labels[["student_id", "passed"]]
    else:
        enrolled = labels.loc[labels["year"] == year, ["student_id", "passed"]]
        log = log.loc[log["ts"].dt.year == year]
    log = log.loc[log["student_id"].isin(enrolled["student_id"])]
    students = sorted(set(log["student_id"]))
    if len(students) < min_students:
        return None
    enrolled = enrolled.loc[enrolled["student_id"].isin(students)]
    if min_minority:
        positives = int(enrolled["passed"].sum())
        if min(positives, len(enrolled) - positives) < min_minority:
            return None

    day = log["ts"].dt.normalize()
    week_start = day - pd.to_timedelta(day.dt.weekday, unit="D")
    active_per_week = log.assign(week_start=week_start).groupby("week_start")["student_id"].nunique()
    active_weeks = active_per_week[active_per_week >= active_share * len(students)]
    if active_weeks.empty:
        return None
    start, end = active_weeks.index.min(), active_weeks.index.max()
    n_weeks = int((end - start).days // 7) + 1

    log = log.assign(week_number=((day - start).dt.days // 7 + 1).astype(int), day=day)
    log = log.loc[(log["week_number"] >= 1) & (log["week_number"] <= n_weeks)]
    grp = log.groupby(["student_id", "week_number"])
    weekly = pd.DataFrame(
        {
            "current_clicks": grp.size(),
            "current_active_days": grp["day"].nunique(),
            "current_content_clicks": grp["component"].apply(lambda s: s.isin(CONTENT_COMPONENTS).sum()),
            "current_social_clicks": grp["component"].apply(lambda s: s.isin(SOCIAL_COMPONENTS).sum()),
        }
    ).reset_index()

    grid = pd.MultiIndex.from_product(
        [students, range(1, n_weeks + 1)], names=["student_id", "week_number"]
    ).to_frame(index=False)
    frame = grid.merge(weekly, on=["student_id", "week_number"], how="left").fillna(0)
    frame = frame.merge(enrolled, on="student_id", how="left")
    frame["n_weeks"] = n_weeks
    frame["passed"] = frame["passed"].astype(int)
    for col in WEEKLY_COLUMNS:
        frame[col] = frame[col].astype(float)
    return frame.sort_values(["student_id", "week_number"]).reset_index(drop=True)
