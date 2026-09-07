from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.generator.config import SERVICE_ROOT, GeneratorConfig, load_generator_config
from src.generator.pipeline import run_generation_pipeline
from src.validation.realism import RealismAudit, format_terminal_report, write_reports


@contextmanager
def temporary_workspace() -> Path:
    base_dir = SERVICE_ROOT / ".tmp_test_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    workspace = base_dir / f"realism-audit-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(
    workspace: Path,
    *,
    seed: int = 42,
    num_students: int = 32,
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


def test_realism_audit_completes_successfully() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=18)
        result = run_generation_pipeline(config, config_path=config_path)
        metrics = RealismAudit().run(result.datasets)
        report_text = format_terminal_report(metrics)
        report_dir = write_reports(metrics, workspace / "reports")

        assert "Realism Audit" in report_text
        assert (report_dir / "realism_metrics.json").exists()
        assert (report_dir / "realism_report.md").exists()


def test_realism_metrics_populated() -> None:
    with temporary_workspace() as workspace:
        config, config_path = make_config(workspace, seed=19)
        result = run_generation_pipeline(config, config_path=config_path)
        metrics = RealismAudit().run(result.datasets)

        assert metrics.risk_distribution
        assert metrics.final_grade_distribution
        assert metrics.passed_rate
        assert metrics.attendance_distribution
        assert metrics.activity_distribution
        assert metrics.submission_discipline
        assert metrics.correlations
        assert metrics.risk_by_trajectory
        assert metrics.outcome_by_trajectory
        assert metrics.risk_distribution_by_week
        assert metrics.risk_score_by_trajectory_week
        assert metrics.predicted_final_grade_by_trajectory_week
        assert metrics.attendance_by_week_trajectory
        assert metrics.activity_by_week_trajectory
        assert metrics.missingness_by_week
        assert metrics.early_warning is not None

        assert metrics.final_grade_distribution["count"] > 0
        assert metrics.passed_rate["completed_count"] > 0
        assert metrics.risk_by_trajectory["mean_risk_score"]
        assert 1 in metrics.risk_distribution_by_week
