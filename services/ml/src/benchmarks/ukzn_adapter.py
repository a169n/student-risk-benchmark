"""UKZN (South African university, Moodle LMS) adapter -> weekly engagement frames.

Dataset: Raghavjee, Subramaniam & Govender (2026), "Anonymized Dataset of
Information Systems and Technology Students at a South African University for
Learning Analytics", Data 11(1):1. CC-BY 4.0, Zenodo 10.5281/zenodo.14810607.
Raw archive is expected to be extracted under ``datasets/ukzn/raw/``.

What this adapter produces
--------------------------
One frame per cohort = (module code, year), grain 1 row = 1 student x 1 course
week, with ONLY the four per-week engagement counters every institution can
supply (the transfer benchmark derives cumulative / recency features itself):

    student_id, week_number, n_weeks, passed,
    current_clicks, current_active_days, current_content_clicks,
    current_social_clicks

Design decisions (all stated in the paper):
- Enrollment = students that appear BOTH in the module-marks file for
  (SUBJ, YEAR) AND at least once in that course's Moodle log. This avoids
  labelling students from a campus whose Moodle site is not in the export.
- Label: ``passed`` = final subject result code (SUBJERES, falling back to
  M_ERES) starts with "P". Fail / supplementary-fail / deregistered -> 0,
  mirroring OULAD's Withdrawn -> 0 convention.
- Course calendar is NOT in the dataset. The active span is inferred from the
  log: the first and last calendar week (within the course year) in which at
  least ``active_share`` of enrolled students were active. Weeks outside the
  span (pre-start browsing, next-year revisits) are dropped.
- Component -> concept: File/Folder/URL/Page/Book/Lesson/Kaltura* = content;
  Forum/Chat = social; everything counts toward total clicks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.benchmarks import moodle_logs

MODULES_FILE = "ISTN TL Modules 2014-2021 - ANONYMIZED.xlsx"
# Component sets and the cohort builder are shared with the other Moodle
# institution (University of Zambia); see ``moodle_logs``.
CONTENT_COMPONENTS = moodle_logs.CONTENT_COMPONENTS
SOCIAL_COMPONENTS = moodle_logs.SOCIAL_COMPONENTS
build_cohort_frame = moodle_logs.build_cohort_frame

_LOG_NAME = re.compile(r"Logs\s+(?P<code>[A-Za-z0-9]+)\s+(?P<year>20\d\d)", re.IGNORECASE)


@dataclass(frozen=True)
class UkznLogFile:
    year: int
    subj: str
    path: Path

    @property
    def cohort_id(self) -> str:
        return f"ukzn_{self.subj}_{self.year}"


def list_log_files(raw_dir: Path | str) -> list[UkznLogFile]:
    """Find per-course Moodle log workbooks; normalise 2018 short codes ("102" -> "ISTN102")."""
    root = _data_root(raw_dir)
    out: list[UkznLogFile] = []
    for path in sorted(root.glob("*LMS/*Logs*ANONYMIZED*.xlsx")):
        if "Add Resources" in path.name:
            continue
        m = _LOG_NAME.search(path.name)
        if not m:
            continue
        code = m.group("code").upper()
        subj = code if code.startswith("ISTN") else f"ISTN{code}"
        out.append(UkznLogFile(year=int(m.group("year")), subj=subj, path=path))
    return out


def load_module_labels(raw_dir: Path | str) -> pd.DataFrame:
    """Return one row per (student_id, subj, year) with a binary ``passed`` label."""
    df = pd.read_excel(_data_root(raw_dir) / MODULES_FILE, sheet_name="DATA")
    result = df["SUBJERES"].where(df["SUBJERES"].notna(), df["M_ERES"])
    out = pd.DataFrame(
        {
            "student_id": df["ANONSTUDNO"].astype(str).str.strip(),
            "subj": df["SUBJ"].astype(str).str.strip().str.upper(),
            "year": pd.to_numeric(df["YEAR"], errors="coerce"),
            "passed": result.astype(str).str.strip().str.upper().str.startswith("P").astype(float),
        }
    )
    out.loc[result.isna(), "passed"] = float("nan")
    out = out.dropna(subset=["year", "passed"])
    out["year"] = out["year"].astype(int)
    # A student can have several rows (re-registrations); keep the best outcome.
    return out.groupby(["student_id", "subj", "year"], as_index=False)["passed"].max()


def load_log(path: Path | str) -> pd.DataFrame:
    """Parse a Moodle standard-log export (Sheet1) into (student_id, ts, component)."""
    raw = pd.read_excel(path, sheet_name=0, header=0)
    raw = raw.iloc[:, :5]
    raw.columns = ["time", "student_id", "context", "component", "event"]
    ts = pd.to_datetime(raw["time"].astype(str).str.strip(), format="%d/%m/%y, %H:%M", errors="coerce")
    if ts.isna().mean() > 0.5:  # tolerate exports with a different date style
        ts = pd.to_datetime(raw["time"], errors="coerce", dayfirst=True)
    log = pd.DataFrame(
        {
            "student_id": raw["student_id"].astype(str).str.strip(),
            "ts": ts,
            "component": raw["component"].astype(str).str.strip(),
        }
    )
    return log.dropna(subset=["ts"]).loc[lambda d: d["student_id"].str.startswith("STUD")]


def build_all_cohorts(
    raw_dir: Path | str,
    *,
    cache_dir: Path | str | None = None,
    min_students: int = 50,
) -> dict[str, pd.DataFrame]:
    """Build (and parquet-cache) every UKZN cohort with >= ``min_students`` labelled students."""
    labels = load_module_labels(raw_dir)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for lf in list_log_files(raw_dir):
        target = cache / f"{lf.cohort_id}.parquet" if cache else None
        if target and target.exists():
            frames[lf.cohort_id] = pd.read_parquet(target)
            continue
        subj_labels = labels.loc[labels["subj"] == lf.subj]
        frame = build_cohort_frame(load_log(lf.path), subj_labels, year=lf.year, min_students=min_students)
        if frame is None:
            continue
        if target:
            frame.to_parquet(target, index=False)
        frames[lf.cohort_id] = frame
    return frames


def _data_root(raw_dir: Path | str) -> Path:
    root = Path(raw_dir)
    nested = root / "Dataset V1 - Raw Data"
    return nested if nested.exists() else root
