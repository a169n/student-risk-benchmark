from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.generator.config import SERVICE_ROOT, GeneratorConfig, load_generator_config
from src.generator.pipeline import run_generation_pipeline
from src.generator.snapshots import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_LOW
from src.validation.quality import DatasetValidationError, ValidationSuite


@contextmanager
def temporary_workspace() -> Path:
    base_dir = SERVICE_ROOT / ".tmp_test_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    workspace = base_dir / f"dataset-pipeline-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(
    workspace: Path,
    *,
    seed: int = 42,
    num_students: int = 24,
    num_weeks: int = 10,
) -> tuple[GeneratorConfig, Path]:
    config, config_path = load_generator_config(
        SERVICE_ROOT / "configs" / "generator_v1.yaml",
        seed_override=seed,
        output_root=workspace / "dataset",
        num_students_override=num_students,
        num_weeks_override=num_weeks,
    )
    return config, config_path


def copy_datasets(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {name: frame.copy(deep=True) for name, frame in datasets.items()}


def test_pipeline_smoke_writes_outputs() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, num_students=12, num_weeks=6)
        result = run_generation_pipeline(config, config_path=config_path)

        for table_name in [
            "students",
            "courses",
            "course_topics",
            "assignments",
            "attendance",
            "submissions",
            "weekly_activity",
            "final_results",
        ]:
            assert (config.raw_output_dir / f"{table_name}.csv").exists()

        assert (config.processed_output_dir / "student_twin_snapshots.csv").exists()
        assert (config.processed_output_dir / "student_twin_snapshots.parquet").exists()
        assert (config.artifacts_output_dir / "reports" / "realism_metrics.json").exists()
        assert (config.artifacts_output_dir / "reports" / "realism_report.md").exists()
        assert result.summary.row_counts["student_twin_snapshots"] == config.num_students * config.num_weeks


def test_pipeline_is_deterministic_for_same_seed() -> None:
    with temporary_workspace() as workspace:
        config_a, config_path_a = make_config(workspace / "run_a", seed=77, num_students=16, num_weeks=5)
        config_b, config_path_b = make_config(workspace / "run_b", seed=77, num_students=16, num_weeks=5)

        result_a = run_generation_pipeline(config_a, config_path=config_path_a)
        result_b = run_generation_pipeline(config_b, config_path=config_path_b)

        for table_name in result_a.datasets:
            assert_frame_equal(
                result_a.datasets[table_name].reset_index(drop=True),
                result_b.datasets[table_name].reset_index(drop=True),
                check_dtype=False,
            )


def test_generator_cli_style_overrides_change_size_and_extend_topics() -> None:
    with temporary_workspace() as workspace:
        config, config_path = load_generator_config(
            SERVICE_ROOT / "configs" / "generator_v1.yaml",
            seed_override=123,
            output_root=workspace / "dataset",
            num_students_override=8,
            num_weeks_override=12,
            num_groups_override=2,
            assignments_per_week_override=1,
            sessions_per_week_override=1,
        )

        assert config.num_students == 8
        assert config.num_weeks == 12
        assert config.num_groups == 2
        assert config.assignments_per_week == 1
        assert config.sessions_per_week == 1
        assert config.course.topic_titles[10] == "Extended Practice Week 11"
        assert config.course.topic_titles[11] == "Extended Practice Week 12"

        result = run_generation_pipeline(config, config_path=config_path)

        assert result.summary.seed == 123
        assert result.summary.row_counts["students"] == 8
        assert result.summary.row_counts["course_topics"] == 12
        assert result.summary.row_counts["assignments"] == 12
        assert result.summary.row_counts["attendance"] == 96
        assert result.summary.row_counts["student_twin_snapshots"] == 96


def test_validation_fails_on_missing_required_column() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, num_students=10, num_weeks=4)
        result = run_generation_pipeline(config, config_path=config_path)
        broken = copy_datasets(result.datasets)
        broken["student_twin_snapshots"] = broken["student_twin_snapshots"].drop(columns=["risk_level"])

        with pytest.raises(DatasetValidationError):
            ValidationSuite(config.contract_path, config).assert_valid(broken)


def test_validation_fails_on_invalid_enum() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, num_students=10, num_weeks=4)
        result = run_generation_pipeline(config, config_path=config_path)
        broken = copy_datasets(result.datasets)
        broken["attendance"].loc[0, "attendance_status"] = "teleported"

        with pytest.raises(DatasetValidationError):
            ValidationSuite(config.contract_path, config).assert_valid(broken)


def test_validation_fails_on_out_of_range_score_and_fk_and_snapshot_count() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, num_students=10, num_weeks=4)
        result = run_generation_pipeline(config, config_path=config_path)
        suite = ValidationSuite(config.contract_path, config)

        invalid_score = copy_datasets(result.datasets)
        invalid_score["submissions"].loc[0, "score"] = 999.0
        with pytest.raises(DatasetValidationError):
            suite.assert_valid(invalid_score)

        invalid_fk = copy_datasets(result.datasets)
        invalid_fk["submissions"].loc[0, "assignment_id"] = "missing_assignment"
        with pytest.raises(DatasetValidationError):
            suite.assert_valid(invalid_fk)

        invalid_snapshots = copy_datasets(result.datasets)
        invalid_snapshots["student_twin_snapshots"] = invalid_snapshots["student_twin_snapshots"].iloc[:-1].copy()
        with pytest.raises(DatasetValidationError):
            suite.assert_valid(invalid_snapshots)


def test_validation_fails_on_duplicate_primary_key() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, num_students=10, num_weeks=4)
        result = run_generation_pipeline(config, config_path=config_path)
        broken = copy_datasets(result.datasets)
        broken["students"].loc[1, "student_id"] = broken["students"].loc[0, "student_id"]

        with pytest.raises(DatasetValidationError):
            ValidationSuite(config.contract_path, config).assert_valid(broken)


def test_generator_relationship_sanity() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=11, num_students=48, num_weeks=10)
        result = run_generation_pipeline(config, config_path=config_path)

        students = result.datasets["students"]
        weekly_activity = result.datasets["weekly_activity"]
        submissions = result.datasets["submissions"]
        snapshots = result.datasets["student_twin_snapshots"]

        mean_activity = weekly_activity.groupby("student_id")["activity_score"].mean().rename("mean_activity")
        on_time_rate = (
            submissions.assign(is_on_time_numeric=submissions["is_on_time"].eq(True).astype(float))
            .groupby("student_id")["is_on_time_numeric"]
            .mean()
            .rename("on_time_rate")
        )
        student_metrics = students.merge(mean_activity, on="student_id").merge(on_time_rate, on="student_id")

        high_motivation_mean = student_metrics.nlargest(10, "motivation_level")["mean_activity"].mean()
        low_motivation_mean = student_metrics.nsmallest(10, "motivation_level")["mean_activity"].mean()
        assert high_motivation_mean > low_motivation_mean

        high_discipline_on_time = student_metrics.nlargest(10, "discipline_level")["on_time_rate"].mean()
        low_discipline_on_time = student_metrics.nsmallest(10, "discipline_level")["on_time_rate"].mean()
        assert high_discipline_on_time > low_discipline_on_time

        final_week = snapshots["week_number"].max()
        final_snapshots = snapshots.loc[snapshots["week_number"] == final_week].merge(
            students[["student_id", "trajectory_type"]], on="student_id"
        )
        declining_risk = final_snapshots.loc[
            final_snapshots["trajectory_type"] == "declining", "risk_score"
        ].mean()
        stable_high_risk = final_snapshots.loc[
            final_snapshots["trajectory_type"] == "stable_high", "risk_score"
        ].mean()
        assert declining_risk > stable_high_risk

        improving = snapshots.merge(students[["student_id", "trajectory_type"]], on="student_id")
        improving = improving.loc[improving["trajectory_type"] == "improving"]
        improving_start = improving.loc[improving["week_number"] == 1, "risk_score"].mean()
        improving_end = improving.loc[improving["week_number"] == final_week, "risk_score"].mean()
        assert improving_end < improving_start

        assert (snapshots.loc[snapshots["risk_level"] == "low", "risk_score"] < RISK_THRESHOLD_LOW).all()
        assert (
            snapshots.loc[snapshots["risk_level"] == "medium", "risk_score"].between(
                RISK_THRESHOLD_LOW, RISK_THRESHOLD_HIGH, inclusive="left"
            )
        ).all()
        assert (snapshots.loc[snapshots["risk_level"] == "high", "risk_score"] >= RISK_THRESHOLD_HIGH).all()
