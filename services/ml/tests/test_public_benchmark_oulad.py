from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.benchmarks.oulad_adapter import (
    OuladCourseFilter,
    OuladRawPaths,
    build_weekly_snapshots,
    validate_oulad_raw_files,
)
from src.experiments.run_public_benchmark_oulad import run_public_benchmark_oulad
from src.generator.config import REPO_ROOT


@contextmanager
def _temp_workspace() -> Path:
    base = REPO_ROOT / "services" / "ml" / ".tmp_test_runs" / f"oulad-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _write_sample_oulad(raw_dir: Path) -> OuladRawPaths:
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_assessment": "a1",
                "assessment_type": "TMA",
                "date": 6,
                "weight": 50,
            },
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_assessment": "a2",
                "assessment_type": "TMA",
                "date": 13,
                "weight": 50,
            },
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_assessment": "exam",
                "assessment_type": "Exam",
                "date": 20,
                "weight": 100,
            },
        ]
    ).to_csv(raw_dir / "assessments.csv", index=False)
    pd.DataFrame(
        [
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "module_presentation_length": 28,
            }
        ]
    ).to_csv(raw_dir / "courses.csv", index=False)
    pd.DataFrame(
        [
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_student": "s1",
                "gender": "F",
                "region": "Test",
                "highest_education": "A Level",
                "imd_band": "50-60%",
                "age_band": "0-35",
                "num_of_prev_attempts": 0,
                "studied_credits": 60,
                "disability": "N",
                "final_result": "Pass",
            },
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_student": "s2",
                "gender": "M",
                "region": "Test",
                "highest_education": "A Level",
                "imd_band": "50-60%",
                "age_band": "0-35",
                "num_of_prev_attempts": 0,
                "studied_credits": 60,
                "disability": "N",
                "final_result": "Fail",
            },
        ]
    ).to_csv(raw_dir / "studentInfo.csv", index=False)
    pd.DataFrame(
        [
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_student": "s1",
                "date_registration": -20,
                "date_unregistration": None,
            },
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_student": "s2",
                "date_registration": -10,
                "date_unregistration": None,
            },
        ]
    ).to_csv(raw_dir / "studentRegistration.csv", index=False)
    pd.DataFrame(
        [
            {
                "id_assessment": "a1",
                "id_student": "s1",
                "date_submitted": 5,
                "is_banked": 0,
                "score": 80,
            },
            {
                "id_assessment": "a2",
                "id_student": "s1",
                "date_submitted": 14,
                "is_banked": 0,
                "score": 60,
            },
            {
                "id_assessment": "exam",
                "id_student": "s1",
                "date_submitted": 21,
                "is_banked": 0,
                "score": 70,
            },
            {
                "id_assessment": "a1",
                "id_student": "s2",
                "date_submitted": 6,
                "is_banked": 0,
                "score": 50,
            },
        ]
    ).to_csv(raw_dir / "studentAssessment.csv", index=False)
    pd.DataFrame(
        [
            {
                "id_site": "site_content",
                "code_module": "DDD",
                "code_presentation": "2013J",
                "activity_type": "resource",
                "week_from": None,
                "week_to": None,
            },
            {
                "id_site": "site_quiz",
                "code_module": "DDD",
                "code_presentation": "2013J",
                "activity_type": "quiz",
                "week_from": None,
                "week_to": None,
            },
        ]
    ).to_csv(raw_dir / "vle.csv", index=False)
    pd.DataFrame(
        [
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_student": "s1",
                "id_site": "site_content",
                "date": -2,
                "sum_click": 3,
            },
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_student": "s1",
                "id_site": "site_quiz",
                "date": 8,
                "sum_click": 5,
            },
            {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "id_student": "s2",
                "id_site": "site_content",
                "date": 8,
                "sum_click": 2,
            },
        ]
    ).to_csv(raw_dir / "studentVle.csv", index=False)
    return OuladRawPaths.from_directory(raw_dir)


def _benchmark_config(workspace: Path, raw_dir: Path, experiment_id: str) -> Path:
    output_dir = workspace / "artifacts" / experiment_id
    config_path = workspace / "oulad_benchmark.yaml"
    payload = {
        "experiment": {
            "experiment_id": experiment_id,
            "title": "Test OULAD benchmark",
            "schema_version": "external_oulad_adapter_v1",
            "dataset_version": "OULAD fixture",
            "dataset_config_name": "sample OULAD fixture",
            "parent_experiment": "exp_004_xai_on_lean_twin",
            "status": "completed",
        },
        "documentation": {
            "objective": "Test OULAD benchmark runner.",
            "hypothesis": "The tiny fixture should produce artifacts.",
            "limitations": ["Fixture only."],
            "next_step": "Use the real OULAD files.",
        },
        "oulad": {
            "raw_dir": str(raw_dir),
            "processed_snapshots_csv": str(output_dir / "oulad_weekly_snapshots.csv"),
            "course_filter": {
                "code_module": "DDD",
                "code_presentation": "2013J",
                "rationale": "Fixture subset.",
            },
            "min_week": 1,
            "max_week": None,
            "student_vle_chunk_size": 2,
            "targets": {
                "primary": "final_weighted_score",
                "secondary": "passed_observed",
                "excluded": ["risk_level"],
                "target_construction": "Fixture weighted target.",
            },
        },
        "feature_sets": {
            "B_lms_oulad": {
                "description": "Fixture LMS baseline.",
                "columns": [
                    "week_number",
                    "course_week_progress",
                    "cumulative_assessment_score_mean_to_date",
                    "cumulative_assessment_weighted_score_to_date",
                    "cumulative_clicks_to_date",
                ],
                "indicator_columns": [
                    "has_assessment_score_to_date",
                    "has_weighted_score_to_date",
                ],
            },
            "B_lms_plus_mastery_oulad": {
                "description": "Fixture LMS plus mastery.",
                "columns": [
                    "week_number",
                    "course_week_progress",
                    "cumulative_assessment_score_mean_to_date",
                    "cumulative_assessment_weighted_score_to_date",
                    "cumulative_clicks_to_date",
                    "overall_mastery_proxy",
                    "current_assessment_cluster_mastery",
                ],
                "indicator_columns": [
                    "has_assessment_score_to_date",
                    "has_weighted_score_to_date",
                    "has_current_assessment_cluster",
                ],
            },
        },
        "feature_set_order": ["B_lms_oulad", "B_lms_plus_mastery_oulad"],
        "comparison": {
            "baseline_feature_set": "B_lms_oulad",
            "candidate_feature_set": "B_lms_plus_mastery_oulad",
            "primary_split": "student_group",
            "improvement_rmse_tolerance": 0.05,
        },
        "classification_models": [],
        "regression_models": ["linear_regression"],
        "splits": {
            "primary": "student_group",
            "secondary": "temporal_forward",
            "student_group": {"test_size": 0.5, "validation_size": 0.0, "seed": 7},
            "temporal_forward": {
                "train_weeks": 2,
                "student_test_size": 0.5,
                "student_seed": 7,
            },
        },
        "seed": 7,
        "outputs": {"experiments_dir": str(output_dir)},
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_oulad_raw_path_validation_reports_missing_files() -> None:
    with _temp_workspace() as workspace:
        raw_dir = workspace / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "assessments.csv").write_text("code_module\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError) as exc:
            validate_oulad_raw_files(OuladRawPaths.from_directory(raw_dir))

        assert "studentInfo.csv" in str(exc.value)
        assert "studentVle.csv" in str(exc.value)


def test_oulad_adapter_builds_targets_and_weekly_features() -> None:
    with _temp_workspace() as workspace:
        paths = _write_sample_oulad(workspace / "raw")

        result = build_weekly_snapshots(
            paths,
            course_filter=OuladCourseFilter("DDD", "2013J"),
            min_week=1,
            student_vle_chunk_size=2,
        )

        snapshots = result.snapshots
        last_by_student = snapshots.sort_values("week_number").drop_duplicates(
            ["code_module", "code_presentation", "id_student"],
            keep="last",
        )
        scores = dict(zip(last_by_student["student_id"], last_by_student["final_weighted_score"]))
        assert scores["s1"] == pytest.approx(70.0)
        assert scores["s2"] == pytest.approx(12.5)

        s1_week_2 = snapshots.loc[
            (snapshots["student_id"] == "s1") & (snapshots["week_number"] == 2)
        ].iloc[0]
        assert s1_week_2["cumulative_clicks_to_date"] == pytest.approx(8.0)
        assert s1_week_2["current_assessment_cluster_mastery"] == pytest.approx(0.0)
        assert "overall_mastery_proxy" in snapshots.columns
        assert result.filtered_row_counts["snapshots"] == 8


def test_public_benchmark_runner_writes_artifacts_and_protects_outputs() -> None:
    with _temp_workspace() as workspace:
        raw_dir = workspace / "raw"
        _write_sample_oulad(raw_dir)
        experiment_id = "exp_123_public_benchmark_oulad"
        config_path = _benchmark_config(workspace, raw_dir, experiment_id)

        payload = run_public_benchmark_oulad(
            config_path,
            artifact_root=workspace / "artifacts",
            docs_dir=workspace / "docs",
        )

        assert payload["metadata_path"].exists()
        assert payload["result_paths"]["csv"].exists()
        assert payload["snapshots_path"].exists()
        assert payload["mapping_path"].exists()
        assert payload["interpretation_path"].exists()

        metadata = json.loads(payload["metadata_path"].read_text(encoding="utf-8"))
        assert metadata["experiment_id"] == experiment_id
        assert metadata["targets"]["primary"] == "final_weighted_score"
        assert metadata["targets"]["excluded"] == ["risk_level"]

        with pytest.raises(FileExistsError):
            run_public_benchmark_oulad(
                config_path,
                artifact_root=workspace / "artifacts",
                docs_dir=workspace / "docs",
            )

