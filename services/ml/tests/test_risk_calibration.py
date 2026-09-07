from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from pandas.api.types import is_bool_dtype

from src.generator.config import SERVICE_ROOT, GeneratorConfig, load_generator_config
from src.generator.pipeline import run_generation_pipeline
from src.generator.snapshots import RISK_THRESHOLD_LOW


@contextmanager
def temporary_workspace() -> Path:
    base_dir = SERVICE_ROOT / ".tmp_test_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    workspace = base_dir / f"risk-calibration-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(
    workspace: Path,
    *,
    seed: int = 42,
    num_students: int = 48,
    num_weeks: int = 10,
) -> tuple[GeneratorConfig, Path]:
    config, config_path = load_generator_config(
        SERVICE_ROOT / "configs" / "generator_v1.yaml",
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


def test_risk_distribution_within_target_bands() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=24, num_students=48, num_weeks=10)
        result = run_generation_pipeline(config, config_path=config_path)
        snapshots = result.datasets["student_twin_snapshots"]
        shares = snapshots["risk_level"].value_counts(normalize=True).to_dict()

        assert 0.05 <= float(shares.get("high", 0.0)) <= 0.20
        assert 0.15 <= float(shares.get("medium", 0.0)) <= 0.40


def test_indicator_fields_present_and_correct() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=15, num_students=24, num_weeks=10)
        result = run_generation_pipeline(config, config_path=config_path)
        snapshots = result.datasets["student_twin_snapshots"]

        assert "has_assignment_score_to_date" in snapshots.columns
        assert "has_quiz_score_to_date" in snapshots.columns
        assert is_bool_dtype(snapshots["has_assignment_score_to_date"])
        assert is_bool_dtype(snapshots["has_quiz_score_to_date"])

        assert snapshots.loc[
            ~snapshots["has_assignment_score_to_date"], "avg_assignment_score_to_date"
        ].isna().all()
        assert snapshots.loc[
            snapshots["has_assignment_score_to_date"], "avg_assignment_score_to_date"
        ].notna().all()

        assert snapshots.loc[
            ~snapshots["has_quiz_score_to_date"], "avg_quiz_score_to_date"
        ].isna().all()
        assert snapshots.loc[
            snapshots["has_quiz_score_to_date"], "avg_quiz_score_to_date"
        ].notna().all()


def test_consistently_at_risk_produces_elevated_risk() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=31, num_students=48, num_weeks=10)
        result = run_generation_pipeline(config, config_path=config_path)

        students = result.datasets["students"][["student_id", "trajectory_type"]]
        snapshots = result.datasets["student_twin_snapshots"]
        final_week = int(snapshots["week_number"].max())
        final_snapshots = snapshots.loc[snapshots["week_number"] == final_week].merge(
            students, on="student_id", how="left"
        )

        at_risk_mean = final_snapshots.loc[
            final_snapshots["trajectory_type"] == "consistently_at_risk", "risk_score"
        ].mean()
        assert float(at_risk_mean) > RISK_THRESHOLD_LOW
