"""University of Zambia CS1 (Moodle) adapter -> canonical weekly frames.

Dataset: Phiri, L. et al. (2026), "A Multi-Source Dataset for CS1 Failure
Prediction in a Sub-Saharan African Context", Zenodo 10.5281/zenodo.21292883,
CC-BY 4.0. Files are pipe-delimited; expected under ``datasets/zambia/``.

One course (ICT 1110, Computer Systems and Architecture) taught over several
academic years, so its cohorts are academic years of the SAME course, which is
exactly the D1 rung of the transfer ladder.

Two dataset-specific notes
--------------------------
PRIVACY. The published log and examination files carry a ``StudentName`` column
holding realistic personal names next to the hashed ``StudentID``. This adapter
never reads that column, and nothing downstream should. Join and identify on
``StudentID`` only.

COHORT YEAR. ``AcademicYear`` is an enrolment code (e.g. 201801) that does not
match the calendar year of the log rows (201801 activity is timestamped 2019).
Cohorts are therefore keyed by the CALENDAR year of the log, which is what the
shared builder filters on.

Label: ``FinalExamination`` (0-100) >= ``pass_mark`` (50 by default, the
institution's pass threshold). Students without a recorded mark are dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmarks.moodle_logs import build_cohort_frame

LOGS_FILE = "logs.csv"
EXAM_FILE = "exam.csv"
SEPARATOR = "|"
COURSE_CODE = "ICT1110"


def load_labels(data_dir: Path | str, *, pass_mark: float = 50.0) -> pd.DataFrame:
    """One row per student: hashed id and binary ``passed``. Ignores StudentName."""
    exam = pd.read_csv(Path(data_dir) / EXAM_FILE, sep=SEPARATOR, usecols=["StudentID", "FinalExamination"])
    mark = pd.to_numeric(exam["FinalExamination"], errors="coerce")
    out = pd.DataFrame(
        {"student_id": exam["StudentID"].astype(str).str.strip(), "passed": (mark >= pass_mark).astype(float)}
    )
    out.loc[mark.isna(), "passed"] = float("nan")
    out = out.dropna(subset=["passed"])
    # The release records ONE final examination mark per student, with no sitting
    # year. A student active in several cohort years would otherwise carry that
    # single outcome into all of them, leaking a later result backwards; the
    # caller resolves this by assigning each student to their last active year.
    return out.groupby("student_id", as_index=False)["passed"].max()


def load_log(data_dir: Path | str) -> pd.DataFrame:
    """Parse the Moodle standard-log export into (student_id, ts, component)."""
    raw = pd.read_csv(
        Path(data_dir) / LOGS_FILE,
        sep=SEPARATOR,
        usecols=["StudentID", "Time", "Component"],
        dtype=str,
    )
    ts = pd.to_datetime(raw["Time"].str.strip(), format="%d/%m/%y, %H:%M", errors="coerce")
    if ts.isna().mean() > 0.5:
        ts = pd.to_datetime(raw["Time"], errors="coerce", dayfirst=True)
    log = pd.DataFrame(
        {
            "student_id": raw["StudentID"].astype(str).str.strip(),
            "ts": ts,
            "component": raw["Component"].astype(str).str.strip(),
        }
    )
    return log.dropna(subset=["ts"])


def build_all_cohorts(
    data_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    min_students: int = 50,
    pass_mark: float = 50.0,
) -> dict[str, pd.DataFrame]:
    """Build one cohort per calendar year of log activity."""
    log = load_log(data_dir)
    labels = load_labels(data_dir, pass_mark=pass_mark)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    # Assign each student to the LAST year they were active: the single recorded
    # examination mark belongs to their final sitting, so attaching it to earlier
    # cohorts would leak that outcome backwards in time.
    last_year = log.groupby("student_id")["ts"].max().dt.year
    log = log.assign(cohort_year=log["student_id"].map(last_year))
    log = log.loc[log["ts"].dt.year == log["cohort_year"]]

    frames: dict[str, pd.DataFrame] = {}
    for year in sorted(log["ts"].dt.year.unique()):
        cohort_id = f"zambia_{COURSE_CODE}_{year}"
        target = cache / f"{cohort_id}.parquet" if cache else None
        if target and target.exists():
            frames[cohort_id] = pd.read_parquet(target)
            continue
        frame = build_cohort_frame(
            log, labels.assign(year=year), year=int(year), min_students=min_students
        )
        if frame is None:
            continue
        frame = frame.assign(institution="Zambia", module=COURSE_CODE, cohort_id=cohort_id)
        if target:
            frame.to_parquet(target, index=False)
        frames[cohort_id] = frame
    return frames
