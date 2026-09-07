from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.experiments.metadata import (
    RegistryEntry,
    preserve_artifact_copy,
    resolve_experiment_artifact_dir,
    upsert_registry_entry,
    validate_experiment_id,
    write_experiment_metadata,
)
from src.experiments.run_ablation import run_ablation
from src.generator.config import REPO_ROOT


@contextmanager
def _temp_workspace() -> Path:
    base = REPO_ROOT / "services" / "ml" / ".tmp_test_runs" / f"metadata-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _synthetic_ablation_data(
    workspace: Path,
    *,
    num_students: int = 24,
    num_weeks: int = 7,
    seed: int = 23,
) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(seed)
    snapshot_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    student_rows: list[dict[str, object]] = []

    for idx in range(num_students):
        student_id = f"student_{idx:03d}"
        ability = 0.25 + 0.03 * idx
        final_grade = float(np.clip(25 + ability * 80 + rng.normal(0, 2), 0, 100))
        student_rows.append(
            {
                "student_id": student_id,
                "student_code": f"STU-{idx:03d}",
                "course_id": "course_x",
                "trajectory_type": "stable_high" if ability > 0.75 else "declining",
            }
        )
        final_rows.append(
            {
                "final_result_id": f"final_{student_id}",
                "student_id": student_id,
                "course_id": "course_x",
                "completion_status": "completed",
                "final_grade": round(final_grade, 2),
                "passed": final_grade >= 50,
            }
        )
        for week in range(1, num_weeks + 1):
            score = float(np.clip(final_grade - 4 + week + rng.normal(0, 3), 0, 100))
            snapshot_rows.append(
                {
                    "snapshot_id": f"snap_{student_id}_w{week:02d}",
                    "student_id": student_id,
                    "course_id": "course_x",
                    "week_number": week,
                    "snapshot_date": f"2026-09-{week:02d}",
                    "attendance_rate_to_date": round(float(np.clip(ability, 0, 1)), 3),
                    "avg_assignment_score_to_date": round(score, 2),
                    "avg_quiz_score_to_date": round(score + rng.normal(0, 2), 2),
                    "has_assignment_score_to_date": True,
                    "has_quiz_score_to_date": True,
                    "on_time_submission_rate_to_date": round(float(np.clip(ability, 0, 1)), 3),
                    "missed_assignments_to_date": int(max(0, round((1 - ability) * week))),
                    "late_submissions_to_date": int(max(0, round((1 - ability) * week / 2))),
                    "avg_attempt_count_to_date": 1.0,
                    "activity_score_to_date": round(score, 2),
                    "time_spent_to_date": round(45.0 * week * ability, 2),
                    "score_trend_3w": round(float(rng.normal(0, 0.05)), 4),
                    "activity_trend_3w": round(float(rng.normal(0, 0.05)), 4),
                    "attendance_trend_3w": round(float(rng.normal(0, 0.05)), 4),
                    "current_topic_mastery": round(score + rng.normal(0, 2), 2),
                    "overall_mastery": round(score + rng.normal(0, 2), 2),
                    "engagement_index": round(score, 2),
                    "performance_index": round(score, 2),
                    "discipline_index": round(score, 2),
                    "risk_score": round(float(np.clip(1 - ability, 0, 1)), 3),
                    "risk_level": "low" if ability > 0.7 else "medium",
                    "predicted_final_grade": round(final_grade, 2),
                }
            )

    snapshots_path = workspace / "snapshots.csv"
    finals_path = workspace / "finals.csv"
    students_path = workspace / "students.csv"
    courses_path = workspace / "courses.csv"
    pd.DataFrame(snapshot_rows).to_csv(snapshots_path, index=False)
    pd.DataFrame(final_rows).to_csv(finals_path, index=False)
    pd.DataFrame(student_rows).to_csv(students_path, index=False)
    pd.DataFrame(
        [
            {
                "course_id": "course_x",
                "course_code": "CSX",
                "course_name": "Synthetic Course",
                "duration_weeks": num_weeks,
                "grading_policy_pass_mark": 50,
            }
        ]
    ).to_csv(courses_path, index=False)
    return snapshots_path, finals_path, students_path, courses_path


def test_experiment_id_and_path_resolution() -> None:
    assert validate_experiment_id("exp_123_example") == "exp_123_example"
    with pytest.raises(ValueError):
        validate_experiment_id("../bad")
    with _temp_workspace() as workspace:
        path = resolve_experiment_artifact_dir("exp_123_example", root=workspace)
        assert path == (workspace / "exp_123_example").resolve()


def test_metadata_write_and_registry_upsert() -> None:
    with _temp_workspace() as workspace:
        metadata_path = write_experiment_metadata(
            {
                "experiment_id": "exp_123_example",
                "title": "Example",
                "created_at": "2026-05-02T00:00:00Z",
            },
            output_dir=workspace / "exp_123_example",
        )
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert payload["experiment_id"] == "exp_123_example"

        registry_path = workspace / "registry.md"
        upsert_registry_entry(
            registry_path,
            RegistryEntry(
                experiment_id="exp_123_example",
                title="Example",
                status="completed",
                schema_version="v1.2",
                dataset_config="generator.yaml",
                primary_target="final_grade",
                artifact_dir="data/artifacts/experiments/exp_123_example",
                doc_path="docs/experiments/exp_123_example.md",
                conclusion="Initial row.",
            ),
        )
        upsert_registry_entry(
            registry_path,
            RegistryEntry(
                experiment_id="exp_123_example",
                title="Example",
                status="completed",
                schema_version="v1.2",
                dataset_config="generator.yaml",
                primary_target="final_grade",
                artifact_dir="data/artifacts/experiments/exp_123_example",
                doc_path="docs/experiments/exp_123_example.md",
                conclusion="Updated row.",
            ),
        )
        text = registry_path.read_text(encoding="utf-8")
        registry_rows = [
            line for line in text.splitlines() if line.startswith("| exp_123_example |")
        ]
        assert len(registry_rows) == 1
        assert "Updated row." in text


def test_preserve_artifact_copy_is_idempotent_and_detects_drift() -> None:
    with _temp_workspace() as workspace:
        source = workspace / "source.json"
        destination = workspace / "preserved" / "source.json"
        source.write_text('{"ok": true}\n', encoding="utf-8")

        preserve_artifact_copy(source, destination)
        preserve_artifact_copy(source, destination)
        assert source.read_bytes() == destination.read_bytes()

        destination.write_text('{"ok": false}\n', encoding="utf-8")
        with pytest.raises(FileExistsError):
            preserve_artifact_copy(source, destination)


def test_run_ablation_writes_versioned_artifacts_and_markdown() -> None:
    with _temp_workspace() as workspace:
        snapshots, finals, students, courses = _synthetic_ablation_data(workspace)
        artifact_root = workspace / "artifacts"
        experiment_id = "exp_123_ablation"
        output_dir = artifact_root / experiment_id
        config_path = workspace / "ablation.yaml"
        config_payload = {
            "experiment": {
                "experiment_id": experiment_id,
                "title": "Test ablation",
                "schema_version": "1.2",
                "dataset_version": "test",
                "dataset_config_name": "test_generator.yaml",
                "parent_experiment": "exp_001_baseline",
                "status": "completed",
            },
            "documentation": {
                "objective": "Test runner artifact creation.",
                "hypothesis": "A tiny ablation should still produce structured outputs.",
                "limitations": ["Synthetic fixture."],
                "next_step": "Use real experiment config.",
            },
            "name": experiment_id,
            "seed": 5,
            "feature_sets": ["B_lms", "B_lms_plus_mastery", "C_twin_full"],
            "classification_models": ["logistic_regression"],
            "regression_models": ["linear_regression"],
            "dataset": {
                "snapshots_csv": str(snapshots),
                "final_results_csv": str(finals),
                "students_csv": str(students),
                "courses_csv": str(courses),
            },
            "snapshot_filter": {"min_week": 3, "max_week": None},
            "splits": {
                "primary": "student_group",
                "secondary": "temporal_forward",
                "student_group": {"test_size": 0.25, "validation_size": 0.0, "seed": 5},
                "temporal_forward": {
                    "train_weeks": 5,
                    "student_test_size": 0.25,
                    "student_seed": 5,
                },
            },
            "eda": {"enabled": False, "correlation_top_k": 5},
            "outputs": {
                "eda_dir": str(output_dir / "eda"),
                "experiments_dir": str(output_dir),
            },
        }
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

        payload = run_ablation(
            config_path,
            artifact_root=artifact_root,
            docs_dir=workspace / "docs",
        )

        assert payload["metadata_path"].exists()
        assert payload["diagnostics_path"].exists()
        assert payload["docs_path"].exists()
        assert payload["artifact_summary_path"].exists()
        assert payload["registry_path"].exists()

        metadata = json.loads(payload["metadata_path"].read_text(encoding="utf-8"))
        assert metadata["experiment_id"] == experiment_id
        assert metadata["targets"]["primary"] == "final_grade"
        assert metadata["targets"]["excluded"] == ["risk_level"]

        markdown = payload["docs_path"].read_text(encoding="utf-8")
        assert "## Lean Twin recommendation" in markdown
        assert "## Main metrics" in markdown
