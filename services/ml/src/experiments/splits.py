"""Leakage-safe snapshot splitting utilities.

Row-wise random splitting is unsafe for `student_twin_snapshots` because many
rows belong to the same student across weeks. A random row split leaks student-
specific patterns from train to test. These utilities split at the student
group level or along the temporal week axis instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    validation: pd.DataFrame | None
    metadata: dict[str, Any]


def _resolve_partition_size(total: int, size: float | int, *, name: str) -> int:
    if isinstance(size, float):
        if not 0 <= size < 1:
            raise ValueError(f"{name} must be in [0, 1) when provided as a float")
        return int(round(total * size))

    if size < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(size)


def student_group_split(
    snapshots: pd.DataFrame,
    *,
    test_size: float | int,
    validation_size: float | int = 0.0,
    seed: int = 42,
    student_id_column: str = "student_id",
) -> SplitResult:
    student_ids = np.array(sorted(snapshots[student_id_column].dropna().unique().tolist()))
    total_students = int(student_ids.shape[0])
    test_count = _resolve_partition_size(total_students, test_size, name="test_size")
    validation_count = _resolve_partition_size(
        total_students, validation_size, name="validation_size"
    )

    if test_count + validation_count >= total_students:
        raise ValueError("test_size + validation_size must leave at least one student for training")

    shuffled = student_ids.copy()
    np.random.default_rng(seed).shuffle(shuffled)

    test_ids = set(shuffled[:test_count].tolist())
    validation_ids = set(shuffled[test_count : test_count + validation_count].tolist())
    train_ids = set(shuffled[test_count + validation_count :].tolist())

    train = snapshots.loc[snapshots[student_id_column].isin(train_ids)].copy()
    test = snapshots.loc[snapshots[student_id_column].isin(test_ids)].copy()
    validation = (
        snapshots.loc[snapshots[student_id_column].isin(validation_ids)].copy()
        if validation_ids
        else None
    )

    return SplitResult(
        train=train,
        test=test,
        validation=validation,
        metadata={
            "seed": seed,
            "student_counts": {
                "train": len(train_ids),
                "test": len(test_ids),
                "validation": len(validation_ids),
            },
            "row_counts": {
                "train": int(train.shape[0]),
                "test": int(test.shape[0]),
                "validation": int(validation.shape[0]) if validation is not None else 0,
            },
        },
    )


def temporal_forward_split(
    snapshots: pd.DataFrame,
    *,
    train_weeks: int,
    week_column: str = "week_number",
) -> SplitResult:
    if train_weeks < 1:
        raise ValueError("train_weeks must be positive")

    train = snapshots.loc[snapshots[week_column] <= train_weeks].copy()
    test = snapshots.loc[snapshots[week_column] > train_weeks].copy()
    return SplitResult(
        train=train,
        test=test,
        validation=None,
        metadata={
            "train_weeks": train_weeks,
            "train_week_max": int(train[week_column].max()) if not train.empty else None,
            "test_week_min": int(test[week_column].min()) if not test.empty else None,
            "row_counts": {"train": int(train.shape[0]), "test": int(test.shape[0])},
        },
    )
