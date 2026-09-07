from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.generator.config import SERVICE_ROOT, GeneratorConfig, load_generator_config
from src.generator.pipeline import run_generation_pipeline


@contextmanager
def temporary_workspace() -> Path:
    base_dir = SERVICE_ROOT / ".tmp_test_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    workspace = base_dir / f"trajectory-realism-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(
    workspace: Path,
    config_name: str = "generator_v1_3_refined.yaml",
    *,
    seed: int = 42,
    num_students: int = 60,
    num_weeks: int = 10,
) -> tuple[GeneratorConfig, Path]:
    config, config_path = load_generator_config(
        SERVICE_ROOT / "configs" / config_name,
        seed_override=seed,
        output_root=workspace / "dataset",
    )
    config = config.model_copy(
        update={
            "num_students": num_students,
            "num_weeks": num_weeks,
            "course": config.course.model_copy(update={"topic_titles": config.course.topic_titles[:num_weeks]}),
        }
    )
    return config, config_path


def test_refined_final_week_trajectory_ordering() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=42, num_students=120)
        result = run_generation_pipeline(config, config_path=config_path)

        students = result.datasets["students"][["student_id", "trajectory_type"]]
        final_results = result.datasets["final_results"][["student_id", "final_grade", "passed", "completion_status"]]
        snapshots = result.datasets["student_twin_snapshots"]
        final_week = int(snapshots["week_number"].max())

        final_snapshots = snapshots.loc[snapshots["week_number"] == final_week].merge(
            students, on="student_id", how="left"
        )
        completed = final_results.merge(students, on="student_id", how="left")
        completed = completed.loc[completed["completion_status"] == "completed"]

        risk_means = final_snapshots.groupby("trajectory_type")["risk_score"].mean()
        assert risk_means["consistently_at_risk"] > risk_means["declining"] > risk_means["improving"] > risk_means["stable_high"]

        grade_means = completed.groupby("trajectory_type")["final_grade"].mean()
        assert grade_means["stable_high"] > grade_means["improving"] > grade_means["declining"] > grade_means["consistently_at_risk"]

        pass_rates = completed.groupby("trajectory_type")["passed"].mean()
        assert pass_rates["stable_high"] >= pass_rates["improving"] > pass_rates["declining"] > pass_rates["consistently_at_risk"]


def test_declining_risk_worsens_over_time() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=42, num_students=120)
        result = run_generation_pipeline(config, config_path=config_path)

        students = result.datasets["students"][["student_id", "trajectory_type"]]
        snapshots = result.datasets["student_twin_snapshots"].merge(students, on="student_id", how="left")
        declining = snapshots.loc[snapshots["trajectory_type"] == "declining"]
        weekly_risk = declining.groupby("week_number")["risk_score"].mean()

        assert weekly_risk.loc[4] > weekly_risk.loc[1]
        assert weekly_risk.loc[10] > weekly_risk.loc[4]


def test_improving_recovers_in_late_weeks() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=42, num_students=60)
        result = run_generation_pipeline(config, config_path=config_path)

        students = result.datasets["students"][["student_id", "trajectory_type"]]
        snapshots = result.datasets["student_twin_snapshots"].merge(students, on="student_id", how="left")
        improving = snapshots.loc[snapshots["trajectory_type"] == "improving"]
        weekly_risk = improving.groupby("week_number")["risk_score"].mean()
        weekly_predicted_grade = improving.groupby("week_number")["predicted_final_grade"].mean()

        assert weekly_risk.loc[10] < weekly_risk.loc[6]
        assert weekly_predicted_grade.loc[10] > weekly_predicted_grade.loc[6]


def test_pre_week_5_high_risk_share_is_non_zero_and_plausible() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=42, num_students=60)
        result = run_generation_pipeline(config, config_path=config_path)
        snapshots = result.datasets["student_twin_snapshots"]
        pre_week_5 = snapshots.loc[snapshots["week_number"] < 5]
        high_share = float((pre_week_5["risk_level"] == "high").mean())

        assert 0.01 <= high_share <= 0.20
