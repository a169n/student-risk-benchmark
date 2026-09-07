from __future__ import annotations

import json
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest
import yaml

from src.experiments.explainability import (
    build_global_explanation,
    select_representative_cases,
    train_regression_reference,
)
from src.experiments.metadata import ARTIFACT_EXPERIMENTS_DIR, resolve_experiment_artifact_dir
from src.experiments.run_xai_on_lean_twin import load_xai_config, run_xai_on_lean_twin
from src.generator.config import REPO_ROOT


@contextmanager
def _temp_workspace() -> Iterator[Path]:
    base = REPO_ROOT / "services" / "ml" / ".tmp_test_runs" / f"xai-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _synthetic_xai_frame(
    *,
    num_students: int = 36,
    num_weeks: int = 7,
    seed: int = 31,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for idx in range(num_students):
        student_id = f"student_{idx:03d}"
        ability = float(np.clip(0.15 + idx / max(num_students - 1, 1) * 0.8, 0, 1))
        trajectory = (-0.08 if idx % 5 == 0 else 0.08 if idx % 4 == 0 else 0.0)
        final_grade = float(np.clip(35 + ability * 60 + trajectory * 20, 0, 100))
        for week in range(1, num_weeks + 1):
            progress = week / num_weeks
            score = float(
                np.clip(
                    final_grade - (1 - progress) * 10 + trajectory * week * 10
                    + rng.normal(0, 2.0),
                    0,
                    100,
                )
            )
            attendance = float(np.clip(ability + rng.normal(0, 0.03), 0, 1))
            on_time = float(np.clip(ability + trajectory + rng.normal(0, 0.04), 0, 1))
            risk_level = "high" if final_grade < 50 else "medium" if final_grade < 70 else "low"
            rows.append(
                {
                    "snapshot_id": f"snap_{student_id}_w{week}",
                    "student_id": student_id,
                    "course_id": "course_x",
                    "week_number": week,
                    "snapshot_date": f"2026-02-{week:02d}",
                    "avg_assignment_score_to_date": round(score, 2),
                    "avg_quiz_score_to_date": round(score + rng.normal(0, 1.5), 2),
                    "attendance_rate_to_date": round(attendance, 3),
                    "activity_score_to_date": round(score + ability * 5, 2),
                    "time_spent_to_date": round(45 * week * max(ability, 0.1), 2),
                    "on_time_submission_rate_to_date": round(on_time, 3),
                    "missed_assignments_to_date": int(max(0, round((1 - on_time) * week))),
                    "late_submissions_to_date": int(max(0, round((1 - on_time) * week / 2))),
                    "avg_attempt_count_to_date": round(1.0 + (1 - ability) * 0.8, 2),
                    "has_assignment_score_to_date": True,
                    "has_quiz_score_to_date": True,
                    "score_trend_3w": round(trajectory + rng.normal(0, 0.015), 4),
                    "activity_trend_3w": round(trajectory + rng.normal(0, 0.015), 4),
                    "attendance_trend_3w": round(trajectory / 2 + rng.normal(0, 0.01), 4),
                    "current_topic_mastery": round(score + rng.normal(0, 2.0), 2),
                    "overall_mastery": round(score + rng.normal(0, 1.0), 2),
                    "engagement_index": round(score, 2),
                    "performance_index": round(score, 2),
                    "discipline_index": round(on_time * 100, 2),
                    "risk_score": round(float(np.clip(1 - ability, 0, 1)), 3),
                    "risk_level": risk_level,
                    "predicted_final_grade": round(final_grade, 2),
                    "final_grade": round(final_grade, 2),
                    "passed": bool(final_grade >= 50),
                }
            )
    return pd.DataFrame(rows)


def _write_xai_fixture(workspace: Path) -> tuple[Path, Path, Path, Path]:
    frame = _synthetic_xai_frame()
    snapshots = frame.drop(columns=["final_grade", "passed"]).copy()
    final_results = (
        frame[["student_id", "course_id", "final_grade", "passed"]]
        .drop_duplicates(subset=["student_id", "course_id"])
        .copy()
    )
    final_results["completion_status"] = "completed"
    final_results["final_result_id"] = [f"final_{idx}" for idx in range(len(final_results))]

    snapshots_path = workspace / "snapshots.csv"
    finals_path = workspace / "finals.csv"
    students_path = workspace / "students.csv"
    courses_path = workspace / "courses.csv"
    snapshots.to_csv(snapshots_path, index=False)
    final_results.to_csv(finals_path, index=False)
    pd.DataFrame(
        [
            {
                "student_id": sid,
                "student_code": sid.replace("_", "-").upper(),
                "course_id": "course_x",
                "trajectory_type": "synthetic_test",
            }
            for sid in final_results["student_id"].tolist()
        ]
    ).to_csv(students_path, index=False)
    pd.DataFrame(
        [
            {
                "course_id": "course_x",
                "course_code": "CSX",
                "course_name": "Synthetic XAI Course",
                "duration_weeks": int(snapshots["week_number"].max()),
                "grading_policy_pass_mark": 50,
            }
        ]
    ).to_csv(courses_path, index=False)
    return snapshots_path, finals_path, students_path, courses_path


def test_exp_004_config_and_path_resolution() -> None:
    modeling, lifecycle, resolved = load_xai_config(
        "configs/experiments/exp_004_xai_on_lean_twin.yaml"
    )
    assert lifecycle.experiment.experiment_id == "exp_004_xai_on_lean_twin"
    assert lifecycle.xai.lean_twin_feature_set == "B_lms_plus_mastery"
    assert lifecycle.xai.reference_baseline_feature_set == "B_lms"
    assert lifecycle.xai.model == "gradient_boosting"
    assert modeling.classification_models == []
    assert modeling.regression_models == ["gradient_boosting"]
    assert resolved.exists()
    assert resolve_experiment_artifact_dir("exp_004_xai_on_lean_twin").name == (
        "exp_004_xai_on_lean_twin"
    )


def test_global_importance_structure_and_case_selection_reproducibility() -> None:
    frame = _synthetic_xai_frame()
    run = train_regression_reference(
        frame,
        feature_set_name="B_lms_plus_mastery",
        model_name="gradient_boosting",
        seed=7,
        test_size=0.25,
        split_seed=7,
    )
    global_explanation = build_global_explanation(run, permutation_repeats=2, seed=7)
    rows = global_explanation["rows"]
    assert rows
    assert {"feature", "rank", "mean_rmse_increase", "importance_share"} <= set(rows[0])
    assert global_explanation["concentration"]["top_feature"] is not None
    assert global_explanation["shap"]["used"] is False

    case_types = [
        "strong_performer",
        "at_risk",
        "improving_trajectory",
        "declining_trajectory",
        "borderline_medium",
    ]
    first = select_representative_cases(
        run,
        case_types=case_types,
        max_cases=5,
        borderline_grade=50,
        trend_feature="score_trend_3w",
    )
    second = select_representative_cases(
        run,
        case_types=case_types,
        max_cases=5,
        borderline_grade=50,
        trend_feature="score_trend_3w",
    )
    assert first == second
    assert {case["case_type"] for case in first} <= set(case_types)
    assert len(first) >= 4


def test_run_xai_writes_artifacts_and_preserves_prior_runs() -> None:
    with _temp_workspace() as workspace:
        snapshots, finals, students, courses = _write_xai_fixture(workspace)
        artifact_root = workspace / "artifacts"
        experiment_id = "exp_998_xai"
        output_dir = artifact_root / experiment_id

        prior_dir = artifact_root / "exp_003_mastery_validation"
        prior_dir.mkdir(parents=True)
        prior_metadata = prior_dir / "experiment_metadata.json"
        prior_metadata.write_text(
            json.dumps({"experiment_id": "exp_003_mastery_validation"}),
            encoding="utf-8",
        )
        prior_bytes = prior_metadata.read_bytes()

        config_payload = {
            "experiment": {
                "experiment_id": experiment_id,
                "title": "XAI test",
                "schema_version": "1.2",
                "dataset_version": "test",
                "dataset_config_name": "test_generator.yaml",
                "parent_experiment": "exp_003_mastery_validation",
                "status": "completed",
            },
            "documentation": {
                "objective": "Verify XAI runner artifact creation.",
                "hypothesis": "The fixture should produce structured explanations.",
                "limitations": ["Synthetic fixture."],
                "next_step": "Use the real exp_004 config.",
            },
            "xai": {
                "primary_split": "student_group",
                "model": "gradient_boosting",
                "reference_baseline_feature_set": "B_lms",
                "lean_twin_feature_set": "B_lms_plus_mastery",
                "primary_target": "final_grade",
                "secondary_context_target": "passed",
                "mastery_columns": ["current_topic_mastery", "overall_mastery"],
                "permutation_repeats": 2,
                "top_k_global": 8,
                "top_k_local": 4,
                "local_cases": {
                    "max_cases": 5,
                    "case_types": [
                        "strong_performer",
                        "at_risk",
                        "improving_trajectory",
                        "declining_trajectory",
                        "borderline_medium",
                    ],
                    "borderline_grade": 50,
                    "trend_feature": "score_trend_3w",
                },
                "dominance": {
                    "overall_mastery_share_warn": 0.60,
                    "explanation_top1_share_warn": 0.80,
                    "local_mastery_share_warn": 0.75,
                    "drop_rmse_warn": 0.50,
                },
            },
            "name": experiment_id,
            "seed": 7,
            "feature_sets": ["B_lms", "B_lms_plus_mastery"],
            "classification_models": [],
            "regression_models": ["gradient_boosting"],
            "dataset": {
                "snapshots_csv": str(snapshots),
                "final_results_csv": str(finals),
                "students_csv": str(students),
                "courses_csv": str(courses),
            },
            "snapshot_filter": {"min_week": 3, "max_week": None},
            "splits": {
                "primary": "student_group",
                "secondary": None,
                "student_group": {"test_size": 0.25, "validation_size": 0.0, "seed": 7},
                "temporal_forward": {
                    "train_weeks": 5,
                    "student_test_size": 0.25,
                    "student_seed": 7,
                },
            },
            "eda": {"enabled": False, "correlation_top_k": 5},
            "outputs": {
                "eda_dir": str(output_dir / "eda"),
                "experiments_dir": str(output_dir),
            },
        }
        config_path = workspace / "xai_test.yaml"
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

        payload = run_xai_on_lean_twin(
            config_path,
            artifact_root=artifact_root,
            docs_dir=workspace / "docs",
        )
        paths = payload["artifact_paths"]
        required = [
            "results_json",
            "summary_markdown",
            "global_importance_csv",
            "global_importance_markdown",
            "local_cases_json",
            "local_cases_markdown",
            "recommendation",
            "metadata",
        ]
        for key in required:
            assert paths[key].exists(), key
        assert payload["docs_path"].exists()
        assert payload["registry_path"].exists()

        results = json.loads(paths["results_json"].read_text(encoding="utf-8"))
        assert results["primary_target"] == "final_grade"
        assert results["secondary_context_target"] == "passed"
        assert results["method_notes"]["shap_used"] is False
        assert results["feature_sets"]["lean_twin"] == "B_lms_plus_mastery"
        assert results["local_explanations"]

        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        assert metadata["targets"]["primary"] == "final_grade"
        assert "risk_level" in metadata["targets"]["excluded_from_supervised_training"]
        assert "passed" in metadata["targets"]["excluded_from_supervised_training"]

        importance = pd.read_csv(paths["global_importance_csv"])
        assert {"comparison_subject", "feature", "rank", "importance_share"} <= set(
            importance.columns
        )

        assert prior_metadata.read_bytes() == prior_bytes
        with pytest.raises(FileExistsError):
            run_xai_on_lean_twin(
                config_path,
                artifact_root=artifact_root,
                docs_dir=workspace / "docs",
            )


def test_existing_prior_experiment_artifacts_are_still_present() -> None:
    for experiment_id in [
        "exp_001_baseline",
        "exp_002_twin_ablation",
        "exp_003_mastery_validation",
    ]:
        metadata_path = ARTIFACT_EXPERIMENTS_DIR / experiment_id / "experiment_metadata.json"
        assert metadata_path.exists(), f"missing preserved metadata for {experiment_id}"
