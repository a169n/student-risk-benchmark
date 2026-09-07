from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

from pandas.testing import assert_frame_equal

from src.generator.config import SERVICE_ROOT, GeneratorConfig, load_generator_config
from src.generator.pipeline import run_generation_pipeline
from src.validation.comparison import compare_generated_outputs, write_comparison_reports


@contextmanager
def temporary_workspace() -> Path:
    base_dir = SERVICE_ROOT / ".tmp_test_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    workspace = base_dir / f"comparison-reports-{uuid.uuid4().hex[:8]}"
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def make_config(
    workspace: Path,
    config_name: str,
    *,
    seed: int = 42,
    num_students: int = 36,
    num_weeks: int = 10,
) -> tuple[GeneratorConfig, Path]:
    config, config_path = load_generator_config(
        SERVICE_ROOT / "configs" / config_name,
        seed_override=seed,
        output_root=workspace,
    )
    config = config.model_copy(
        update={
            "num_students": num_students,
            "num_weeks": num_weeks,
            "course": config.course.model_copy(update={"topic_titles": config.course.topic_titles[:num_weeks]}),
        }
    )
    return config, config_path


def test_comparison_report_generation() -> None:
    with temporary_workspace() as workspace:
        left_root = workspace / "baseline"
        right_root = workspace / "refined"
        left_config, left_path = make_config(left_root, "generator_v1_2_baseline.yaml", seed=42)
        right_config, right_path = make_config(right_root, "generator_v1_3_refined.yaml", seed=42)

        run_generation_pipeline(left_config, config_path=left_path)
        run_generation_pipeline(right_config, config_path=right_path)

        summary = compare_generated_outputs(left_root, right_root, left_label="v1_2", right_label="v1_3")
        report_dir = write_comparison_reports(summary, workspace / "comparison_output")

        assert report_dir.exists()
        assert (report_dir / "comparison_summary.json").exists()
        assert (report_dir / "comparison_report.md").exists()
        assert (report_dir / "key_metrics.csv").exists()
        assert len(summary.key_metrics) > 0

        metric_lookup = {row["metric"]: row for row in summary.key_metrics}
        assert metric_lookup["trajectory_risk_ordering_valid"]["left"] is False
        assert metric_lookup["trajectory_risk_ordering_valid"]["right"] is True


def test_benchmark_configs_are_reproducible() -> None:
    with temporary_workspace() as workspace:
        config_a, config_path_a = make_config(
            workspace / "run_a", "generator_v1_3_refined.yaml", seed=77, num_students=20, num_weeks=6
        )
        config_b, config_path_b = make_config(
            workspace / "run_b", "generator_v1_3_refined.yaml", seed=77, num_students=20, num_weeks=6
        )

        result_a = run_generation_pipeline(config_a, config_path=config_path_a)
        result_b = run_generation_pipeline(config_b, config_path=config_path_b)

        assert config_a.trajectory_tuning == "v1_3_refined"
        assert config_b.trajectory_tuning == "v1_3_refined"
        assert_frame_equal(
            result_a.datasets["student_twin_snapshots"].reset_index(drop=True),
            result_b.datasets["student_twin_snapshots"].reset_index(drop=True),
            check_dtype=False,
        )
