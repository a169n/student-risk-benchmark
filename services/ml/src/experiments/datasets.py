"""Modeling-dataset assembly for the baseline experiments.

Loads the refined dataset artifacts produced by the synthetic generator and
joins them into a leakage-safe modeling table. The grain stays at the snapshot
level (1 row = 1 student x 1 week), and end-of-course outcomes are joined onto
each weekly row only as supervised targets.

This module deliberately avoids any feature engineering of its own. It only:

1. loads the snapshot, final-results, students, and courses tables,
2. filters snapshots to the configured week window,
3. attaches `final_grade`, `passed`, and `completion_status` from the outcome
   layer for use as supervised targets,
4. preserves grouping keys (`student_id`, `week_number`) needed for
   leakage-safe splitting downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.experiments.config import ExperimentConfig

REGRESSION_TARGET = "final_grade"
CLASSIFICATION_TARGET = "passed"
GROUP_COLUMN = "student_id"
WEEK_COLUMN = "week_number"


@dataclass(frozen=True)
class ModelingDataset:
    """The joined snapshot + outcome table used by the baseline pipeline."""

    frame: pd.DataFrame
    snapshots_path: Path
    final_results_path: Path
    snapshot_week_min: int
    snapshot_week_max: int
    completed_only: bool

    @property
    def n_rows(self) -> int:
        return int(self.frame.shape[0])

    @property
    def n_students(self) -> int:
        return int(self.frame[GROUP_COLUMN].nunique())


def _load_csv(path: Path, *, required_columns: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset artifact not found: {path}. "
            "Run the generator first (see services/ml/README.md)."
        )
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path.name}: {missing}")
    return frame


def load_modeling_dataset(
    config: ExperimentConfig,
    *,
    completed_only: bool = True,
) -> ModelingDataset:
    """Load and join the snapshot table with end-of-course outcomes.

    Snapshots without a usable `final_grade` outcome are dropped because they
    cannot contribute to the supervised baselines. The drop is reported via
    ``ModelingDataset.completed_only`` so the EDA report can flag it.
    """

    snapshots_path = config.resolve_path(config.dataset.snapshots_csv)
    final_results_path = config.resolve_path(config.dataset.final_results_csv)

    snapshots = _load_csv(
        snapshots_path,
        required_columns=("student_id", "course_id", "week_number"),
    )
    final_results = _load_csv(
        final_results_path,
        required_columns=("student_id", "course_id", "completion_status"),
    )

    week_filter = snapshots[WEEK_COLUMN] >= config.snapshot_filter.min_week
    if config.snapshot_filter.max_week is not None:
        week_filter &= snapshots[WEEK_COLUMN] <= config.snapshot_filter.max_week
    snapshots = snapshots.loc[week_filter].copy()

    outcome_columns = ["student_id", "course_id", "completion_status", "final_grade", "passed"]
    available_outcome_cols = [
        column for column in outcome_columns if column in final_results.columns
    ]
    joined = snapshots.merge(
        final_results[available_outcome_cols],
        on=["student_id", "course_id"],
        how="left",
    )

    if completed_only:
        joined = joined.loc[joined[REGRESSION_TARGET].notna()].copy()
        if CLASSIFICATION_TARGET in joined.columns:
            joined = joined.loc[joined[CLASSIFICATION_TARGET].notna()].copy()
            joined[CLASSIFICATION_TARGET] = joined[CLASSIFICATION_TARGET].astype(bool)

    if joined.empty:
        raise ValueError(
            "Modeling dataset is empty after filtering. "
            "Check snapshot_filter.min_week and the generator output."
        )

    return ModelingDataset(
        frame=joined.reset_index(drop=True),
        snapshots_path=snapshots_path,
        final_results_path=final_results_path,
        snapshot_week_min=int(joined[WEEK_COLUMN].min()),
        snapshot_week_max=int(joined[WEEK_COLUMN].max()),
        completed_only=completed_only,
    )
