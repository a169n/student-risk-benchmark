from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.config import (
    DatasetSourceConfig,
    EDAConfig,
    ExperimentConfig,
    OutputConfig,
    SnapshotFilterConfig,
    SplitsConfig,
    StudentGroupSplitConfig,
    TemporalForwardSplitConfig,
)
from src.experiments.run_baselines import _build_split, run_experiments
from src.generator.config import REPO_ROOT


def _synthetic_modeling_data(num_students: int = 30, num_weeks: int = 8, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    snapshot_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    student_rows: list[dict[str, object]] = []

    for student_idx in range(num_students):
        ability = float(rng.uniform(0.05, 1.0))
        baseline = 20 + ability * 70
        student_id = f"student_{student_idx:03d}"
        student_rows.append(
            {
                "student_id": student_id,
                "student_code": f"STU-{student_idx:03d}",
                "course_id": "course_x",
                "trajectory_type": "stable_high" if ability > 0.7 else "consistently_at_risk",
                "baseline_level": ability,
                "motivation_level": ability,
                "discipline_level": ability,
            }
        )
        final_grade = float(np.clip(baseline + rng.normal(0, 5), 0, 100))
        final_rows.append(
            {
                "final_result_id": f"final_{student_id}",
                "student_id": student_id,
                "course_id": "course_x",
                "completion_status": "completed",
                "final_grade": round(final_grade, 2),
                "passed": final_grade >= 50,
                "completed_weeks": num_weeks,
            }
        )
        for week in range(1, num_weeks + 1):
            week_factor = 0.6 + 0.04 * week
            attendance_rate = float(np.clip(ability * week_factor + rng.normal(0, 0.05), 0, 1))
            avg_assignment = (
                None if (week == 1 and student_idx % 5 == 0) else float(np.clip(baseline + rng.normal(0, 4), 0, 100))
            )
            avg_quiz = float(np.clip(baseline + rng.normal(0, 4), 0, 100))
            snapshot_rows.append(
                {
                    "snapshot_id": f"snap_{student_id}_w{week:02d}",
                    "student_id": student_id,
                    "course_id": "course_x",
                    "week_number": week,
                    "snapshot_date": f"2026-09-{week:02d}",
                    "attendance_rate_to_date": round(attendance_rate, 3),
                    "avg_assignment_score_to_date": (
                        None if avg_assignment is None else round(avg_assignment, 2)
                    ),
                    "avg_quiz_score_to_date": round(avg_quiz, 2),
                    "has_assignment_score_to_date": avg_assignment is not None,
                    "has_quiz_score_to_date": True,
                    "on_time_submission_rate_to_date": round(float(np.clip(ability + rng.normal(0, 0.05), 0, 1)), 3),
                    "missed_assignments_to_date": int(max(0, round((1 - ability) * week))),
                    "late_submissions_to_date": int(max(0, round(rng.uniform(0, 1) * week))),
                    "avg_attempt_count_to_date": round(float(1.0 + rng.uniform(0, 0.5)), 2),
                    "activity_score_to_date": round(float(np.clip(baseline + rng.normal(0, 5), 0, 100)), 2),
                    "time_spent_to_date": round(float(60 * week + rng.normal(0, 10)), 2),
                    "score_trend_3w": round(float(rng.normal(0, 0.05)), 4),
                    "activity_trend_3w": round(float(rng.normal(0, 0.05)), 4),
                    "attendance_trend_3w": round(float(rng.normal(0, 0.05)), 4),
                    "current_topic_mastery": round(float(np.clip(baseline + rng.normal(0, 4), 0, 100)), 2),
                    "overall_mastery": round(float(np.clip(baseline + rng.normal(0, 4), 0, 100)), 2),
                    "engagement_index": round(float(np.clip(baseline + rng.normal(0, 5), 0, 100)), 2),
                    "performance_index": round(float(np.clip(baseline + rng.normal(0, 5), 0, 100)), 2),
                    "discipline_index": round(float(np.clip(baseline + rng.normal(0, 5), 0, 100)), 2),
                    "risk_score": round(float(np.clip(0.6 - ability * 0.5 + rng.normal(0, 0.05), 0, 1)), 4),
                    "risk_level": "low" if ability > 0.7 else "high",
                    "predicted_final_grade": round(final_grade + rng.normal(0, 3), 2),
                }
            )

    return {
        "snapshots": pd.DataFrame(snapshot_rows),
        "finals": pd.DataFrame(final_rows),
        "students": pd.DataFrame(student_rows),
    }


@contextmanager
def _temp_workspace() -> Path:
    base = REPO_ROOT / "services" / "ml" / ".tmp_test_runs" / f"experiments-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _write_synthetic_inputs(workspace: Path) -> tuple[Path, Path, Path]:
    inputs = _synthetic_modeling_data()
    snapshots_path = workspace / "snapshots.csv"
    finals_path = workspace / "finals.csv"
    students_path = workspace / "students.csv"
    inputs["snapshots"].to_csv(snapshots_path, index=False)
    inputs["finals"].to_csv(finals_path, index=False)
    inputs["students"].to_csv(students_path, index=False)
    return snapshots_path, finals_path, students_path


def _build_test_config(
    workspace: Path,
    snapshots: Path,
    finals: Path,
    students: Path,
) -> ExperimentConfig:
    return ExperimentConfig(
        name="test_run",
        seed=11,
        feature_sets=["A_simple", "C_twin"],
        classification_models=["logistic_regression", "random_forest"],
        regression_models=["linear_regression", "random_forest"],
        dataset=DatasetSourceConfig(
            snapshots_csv=snapshots,
            final_results_csv=finals,
            students_csv=students,
            courses_csv=workspace / "missing_courses.csv",
        ),
        snapshot_filter=SnapshotFilterConfig(min_week=3, max_week=None),
        splits=SplitsConfig(
            primary="student_group",
            secondary="temporal_forward",
            student_group=StudentGroupSplitConfig(test_size=0.3, validation_size=0.0, seed=11),
            temporal_forward=TemporalForwardSplitConfig(train_weeks=5, student_test_size=0.3, student_seed=11),
        ),
        eda=EDAConfig(enabled=True, correlation_top_k=5),
        outputs=OutputConfig(
            eda_dir=workspace / "eda",
            experiments_dir=workspace / "experiments",
        ),
    )


def test_run_baselines_writes_expected_artifacts() -> None:
    with _temp_workspace() as workspace:
        snapshots_path, finals_path, students_path = _write_synthetic_inputs(workspace)
        config = _build_test_config(workspace, snapshots_path, finals_path, students_path)

        payload = run_experiments(config)
        rows = payload["result_rows"]
        assert rows, "Expected at least one result row"

        feature_sets_seen = {row.feature_set for row in rows}
        assert feature_sets_seen == {"A_simple", "C_twin"}

        targets_seen = {row.target for row in rows}
        assert targets_seen == {"passed", "final_grade"}

        strategies_seen = {row.split_strategy for row in rows}
        assert strategies_seen == {"student_group", "temporal_forward"}

        # Required artifacts exist
        paths = payload["result_paths"]
        for key in ("csv", "json", "markdown"):
            assert paths[key].exists(), f"Missing artifact: {key}"

        eda = payload["eda_artifacts"]
        assert eda is not None
        assert eda.report_path.exists()
        assert eda.feature_summary_csv.exists()
        assert eda.correlation_csv.exists()
        assert eda.target_by_week_csv.exists()
        assert eda.summary_json_path.exists()

        # JSON results must round-trip
        json_payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        assert json_payload["run_name"] == "test_run"
        assert len(json_payload["rows"]) == len(rows)


def test_run_baselines_student_group_split_has_no_leakage() -> None:
    with _temp_workspace() as workspace:
        snapshots_path, finals_path, students_path = _write_synthetic_inputs(workspace)
        config = _build_test_config(workspace, snapshots_path, finals_path, students_path)

        from src.experiments.datasets import load_modeling_dataset
        from src.experiments.featuresets import FEATURE_SET_C_TWIN
        from src.experiments.preprocessing import build_modeling_matrix

        dataset = load_modeling_dataset(config)
        matrix = build_modeling_matrix(dataset.frame, FEATURE_SET_C_TWIN)

        partition = _build_split(matrix, "student_group", config)
        train_students = set(matrix.groups[partition.train_mask].unique())
        test_students = set(matrix.groups[partition.test_mask].unique())
        assert train_students.isdisjoint(test_students)


def test_run_baselines_temporal_split_separates_weeks_and_students() -> None:
    with _temp_workspace() as workspace:
        snapshots_path, finals_path, students_path = _write_synthetic_inputs(workspace)
        config = _build_test_config(workspace, snapshots_path, finals_path, students_path)

        from src.experiments.datasets import load_modeling_dataset
        from src.experiments.featuresets import FEATURE_SET_C_TWIN
        from src.experiments.preprocessing import build_modeling_matrix

        dataset = load_modeling_dataset(config)
        matrix = build_modeling_matrix(dataset.frame, FEATURE_SET_C_TWIN)

        partition = _build_split(matrix, "temporal_forward", config)
        train_weeks = matrix.weeks[partition.train_mask]
        test_weeks = matrix.weeks[partition.test_mask]
        if len(train_weeks) and len(test_weeks):
            assert train_weeks.max() <= config.splits.temporal_forward.train_weeks
            assert test_weeks.min() > config.splits.temporal_forward.train_weeks

        train_students = set(matrix.groups[partition.train_mask].unique())
        test_students = set(matrix.groups[partition.test_mask].unique())
        assert train_students.isdisjoint(test_students)


def test_run_baselines_drops_forbidden_columns() -> None:
    """Sanity: forbidden columns never reach the prepared modeling matrix."""

    with _temp_workspace() as workspace:
        snapshots_path, finals_path, students_path = _write_synthetic_inputs(workspace)
        config = _build_test_config(workspace, snapshots_path, finals_path, students_path)
        from src.experiments.datasets import load_modeling_dataset
        from src.experiments.featuresets import FEATURE_SET_C_TWIN, FORBIDDEN_FEATURE_COLUMNS
        from src.experiments.preprocessing import build_modeling_matrix

        dataset = load_modeling_dataset(config)
        matrix = build_modeling_matrix(dataset.frame, FEATURE_SET_C_TWIN)
        leak = set(matrix.feature_columns) & FORBIDDEN_FEATURE_COLUMNS
        assert leak == set(), f"Forbidden columns leaked into matrix: {leak}"


def test_load_modeling_dataset_requires_snapshots_file() -> None:
    with _temp_workspace() as workspace:
        config = _build_test_config(
            workspace,
            workspace / "missing_snapshots.csv",
            workspace / "missing_finals.csv",
            workspace / "missing_students.csv",
        )
        from src.experiments.datasets import load_modeling_dataset

        with pytest.raises(FileNotFoundError):
            load_modeling_dataset(config)
