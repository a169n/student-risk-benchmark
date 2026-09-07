"""Universidad de Oviedo (Moodle 2.x, AY 2014/15) adapter -> canonical weekly frames.

Dataset: Riestra-Gonzalez, Paule-Ruiz & Ortin (2021), Computers & Education
163:104108. Tables are the CSV extracts written by ``oviedo_extract`` from the
published PostgreSQL dump.

Cohort = one Moodle course. The academic year runs September to June, so it
crosses the calendar-year boundary and the shared builder is called with
``year=None``; the course's active span is still inferred from its own log.

Label
-----
``mdl_grade_grades.finalgrade`` on the course-total grade item
(``mdl_grade_items.itemtype == 'course'``), scaled by that item's ``grademax``
and thresholded at 50%. This is the gradebook course total, NOT a registrar
outcome, which differs from the other institutions and must be stated: a
student who never opens the LMS scores zero, so the label is partly entailed by
absence of activity. ``drop_zero_grades`` reproduces the withdrawal-tautology
control used for OULAD by removing students whose course total is exactly zero.

Concept mapping follows the log's ``module`` column rather than a Moodle
``Component`` string, so it is defined here rather than in ``moodle_logs``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.moodle_logs import build_cohort_frame

MONTH_FILES: tuple[str, ...] = (
    "septiembre", "octubre", "noviembre", "diciembre", "enero",
    "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
    "mdl_log",
)
CONTENT_MODULES = frozenset(
    {"resource", "folder", "url", "page", "book", "imscp", "label", "glossary", "lesson", "wiki"}
)
SOCIAL_MODULES = frozenset({"forum", "chat", "message"})
PASS_FRACTION = 0.5


def load_course_labels(tables_dir: Path | str) -> pd.DataFrame:
    """Per (courseid, student) pass label from the course-total gradebook item."""
    tables = Path(tables_dir)
    items = pd.read_csv(tables / "mdl_grade_items.csv")
    items = items.loc[items["itemtype"] == "course"].copy()
    items["grademax"] = pd.to_numeric(items["grademax"], errors="coerce")
    items = items.loc[items["grademax"] > 0]

    grades = pd.read_csv(tables / "mdl_grade_grades.csv")
    grades["finalgrade"] = pd.to_numeric(grades["finalgrade"], errors="coerce")
    merged = grades.merge(
        items[["id", "courseid", "grademax"]], left_on="itemid", right_on="id", how="inner"
    ).dropna(subset=["finalgrade"])

    merged["fraction"] = merged["finalgrade"] / merged["grademax"]
    return pd.DataFrame(
        {
            "courseid": merged["courseid"].astype(int),
            "student_id": merged["userid"].astype(str),
            "fraction": merged["fraction"].astype(float),
            "passed": (merged["fraction"] >= PASS_FRACTION).astype(float),
        }
    )


def select_courses(
    labels: pd.DataFrame,
    *,
    min_students: int = 50,
    min_minority: int = 15,
    limit: int | None = 20,
    drop_zero_grades: bool = False,
) -> list[int]:
    """Course ids passing the size/balance filter, largest minority class first.

    ``limit`` caps how many cohorts this institution contributes. The cap keeps
    the ladder balanced: Oviedo has ~90 eligible courses, and without a cap the
    mean over pairs would largely be an Oviedo statistic.
    """
    frame = labels.loc[labels["fraction"] > 0] if drop_zero_grades else labels
    stats = frame.groupby("courseid").agg(n=("student_id", "nunique"), rate=("passed", "mean"))
    stats["minority"] = (stats["n"] * np.minimum(stats["rate"], 1 - stats["rate"])).round()
    eligible = stats.loc[(stats["n"] >= min_students) & (stats["minority"] >= min_minority)]
    ordered = eligible.sort_values("minority", ascending=False).index.astype(int).tolist()
    return ordered[:limit] if limit else ordered


def load_log_for_courses(tables_dir: Path | str, courses: set[int], *, chunksize: int = 2_000_000) -> pd.DataFrame:
    """Stream the month tables, keeping only rows for the selected courses."""
    tables = Path(tables_dir)
    kept: list[pd.DataFrame] = []
    for name in MONTH_FILES:
        path = tables / f"{name}.csv"
        if not path.exists():
            continue
        for chunk in pd.read_csv(
            path, usecols=["time", "userid", "course", "module"], chunksize=chunksize
        ):
            chunk = chunk.loc[chunk["course"].isin(courses)]
            if not chunk.empty:
                kept.append(chunk)
    if not kept:
        return pd.DataFrame(columns=["courseid", "student_id", "ts", "component"])
    log = pd.concat(kept, ignore_index=True)
    return pd.DataFrame(
        {
            "courseid": log["course"].astype(int),
            "student_id": log["userid"].astype(str),
            "ts": pd.to_datetime(pd.to_numeric(log["time"], errors="coerce"), unit="s", errors="coerce"),
            "component": log["module"].astype(str).str.strip(),
        }
    ).dropna(subset=["ts"])


def _as_moodle_components(log: pd.DataFrame) -> pd.DataFrame:
    """Rewrite the ``module`` values onto the component names the shared builder counts."""
    component = np.where(
        log["component"].isin(CONTENT_MODULES),
        "File",  # any member of moodle_logs.CONTENT_COMPONENTS
        np.where(log["component"].isin(SOCIAL_MODULES), "Forum", "System"),
    )
    return log.assign(component=component)


def build_all_cohorts(
    tables_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    min_students: int = 50,
    min_minority: int = 15,
    limit: int | None = 20,
    drop_zero_grades: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build (and parquet-cache) one canonical weekly frame per selected Oviedo course."""
    labels = load_course_labels(tables_dir)
    courses = select_courses(
        labels,
        min_students=min_students,
        min_minority=min_minority,
        limit=limit,
        drop_zero_grades=drop_zero_grades,
    )
    if not courses:
        return {}
    log = _as_moodle_components(load_log_for_courses(tables_dir, set(courses)))

    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    for courseid in courses:
        cohort_id = f"oviedo_C{courseid}_1415"
        target = cache / f"{cohort_id}.parquet" if cache else None
        if target and target.exists():
            frames[cohort_id] = pd.read_parquet(target)
            continue
        course_labels = labels.loc[labels["courseid"] == courseid]
        if drop_zero_grades:
            course_labels = course_labels.loc[course_labels["fraction"] > 0]
        frame = build_cohort_frame(
            log.loc[log["courseid"] == courseid],
            course_labels[["student_id", "passed"]],
            year=None,
            min_students=min_students,
            min_minority=min_minority,
        )
        if frame is None:
            continue
        frame = frame.assign(
            institution="Oviedo", module=f"C{courseid}", cohort_id=cohort_id
        )
        if target:
            frame.to_parquet(target, index=False)
        frames[cohort_id] = frame
    return frames
