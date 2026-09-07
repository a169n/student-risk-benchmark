from __future__ import annotations

import pandas as pd

from src.experiments.splits import student_group_split, temporal_forward_split


def _sample_snapshots() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for student_id in ["student_001", "student_002", "student_003", "student_004", "student_005"]:
        for week_number in [1, 2, 3, 4]:
            rows.append(
                {
                    "snapshot_id": f"snap_{student_id}_w{week_number:02d}",
                    "student_id": student_id,
                    "week_number": week_number,
                    "risk_score": week_number / 10.0,
                }
            )
    return pd.DataFrame(rows)


def test_student_group_split_no_leakage() -> None:
    snapshots = _sample_snapshots()
    result = student_group_split(snapshots, test_size=0.4, seed=11)

    train_students = set(result.train["student_id"].unique().tolist())
    test_students = set(result.test["student_id"].unique().tolist())
    assert train_students.isdisjoint(test_students)


def test_student_group_split_with_validation() -> None:
    snapshots = _sample_snapshots()
    result = student_group_split(snapshots, test_size=0.2, validation_size=0.2, seed=7)

    train_students = set(result.train["student_id"].unique().tolist())
    test_students = set(result.test["student_id"].unique().tolist())
    validation_students = set(result.validation["student_id"].unique().tolist()) if result.validation is not None else set()

    assert train_students.isdisjoint(test_students)
    assert train_students.isdisjoint(validation_students)
    assert test_students.isdisjoint(validation_students)


def test_temporal_forward_split_week_boundaries() -> None:
    snapshots = _sample_snapshots()
    result = temporal_forward_split(snapshots, train_weeks=2)

    assert (result.train["week_number"] <= 2).all()
    assert (result.test["week_number"] > 2).all()


def test_student_group_split_deterministic() -> None:
    snapshots = _sample_snapshots()
    result_a = student_group_split(snapshots, test_size=0.4, validation_size=0.2, seed=99)
    result_b = student_group_split(snapshots, test_size=0.4, validation_size=0.2, seed=99)

    assert result_a.metadata == result_b.metadata
    assert result_a.train["snapshot_id"].tolist() == result_b.train["snapshot_id"].tolist()
    assert result_a.test["snapshot_id"].tolist() == result_b.test["snapshot_id"].tolist()
    validation_a = [] if result_a.validation is None else result_a.validation["snapshot_id"].tolist()
    validation_b = [] if result_b.validation is None else result_b.validation["snapshot_id"].tolist()
    assert validation_a == validation_b
