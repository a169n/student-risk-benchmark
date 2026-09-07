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

from src.experiments.diagnostics import (
    compute_lms_target_correlations,
    compute_mastery_target_correlations,
    compute_redundancy_with_lms,
    derive_carry_forward_recommendation,
    run_drop_column_tests,
    run_weekly_validation,
    summarize_weekly_delta,
)
from src.experiments.featuresets import get_feature_set
from src.experiments.metadata import (
    ARTIFACT_EXPERIMENTS_DIR,
    resolve_experiment_artifact_dir,
    validate_experiment_id,
)
from src.experiments.run_mastery_validation import (
    load_mastery_validation_config,
    run_mastery_validation,
)
from src.generator.config import REPO_ROOT


@contextmanager
def _temp_workspace() -> Iterator[Path]:
    base = REPO_ROOT / "services" / "ml" / ".tmp_test_runs" / f"mastery-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _synthetic_modeling_frame(
    *,
    num_students: int = 60,
    num_weeks: int = 7,
    seed: int = 11,
) -> pd.DataFrame:
    """Build a small but realistic modeling-frame fixture for diagnostics tests."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for idx in range(num_students):
        student_id = f"student_{idx:03d}"
        ability = float(np.clip(0.2 + 0.012 * idx + rng.normal(0, 0.05), 0.0, 1.0))
        final_grade = float(np.clip(35 + ability * 60 + rng.normal(0, 1.5), 0, 100))
        for week in range(1, num_weeks + 1):
            score = float(np.clip(final_grade - 4 + week + rng.normal(0, 1.5), 0, 100))
            rows.append(
                {
                    "snapshot_id": f"snap_{student_id}_w{week}",
                    "student_id": student_id,
                    "course_id": "course_x",
                    "week_number": week,
                    "snapshot_date": f"2026-01-{week:02d}",
                    "attendance_rate_to_date": round(float(np.clip(ability, 0, 1)), 3),
                    "avg_assignment_score_to_date": round(score, 2),
                    "avg_quiz_score_to_date": round(score + rng.normal(0, 1), 2),
                    "has_assignment_score_to_date": True,
                    "has_quiz_score_to_date": True,
                    "on_time_submission_rate_to_date": round(float(np.clip(ability, 0, 1)), 3),
                    "missed_assignments_to_date": int(max(0, round((1 - ability) * week))),
                    "late_submissions_to_date": int(max(0, round((1 - ability) * week / 2))),
                    "avg_attempt_count_to_date": 1.0,
                    "activity_score_to_date": round(score, 2),
                    "time_spent_to_date": round(60.0 * week * ability, 2),
                    "score_trend_3w": round(float(rng.normal(0, 0.04)), 4),
                    "activity_trend_3w": round(float(rng.normal(0, 0.04)), 4),
                    "attendance_trend_3w": round(float(rng.normal(0, 0.04)), 4),
                    "current_topic_mastery": round(score + rng.normal(0, 1.5), 2),
                    "overall_mastery": round(score + rng.normal(0, 1.0), 2),
                    "engagement_index": round(score, 2),
                    "performance_index": round(score, 2),
                    "discipline_index": round(score, 2),
                    "risk_score": round(float(np.clip(1 - ability, 0, 1)), 3),
                    "risk_level": "low" if ability > 0.7 else "medium",
                    "predicted_final_grade": round(final_grade, 2),
                    "final_grade": round(final_grade, 2),
                    "passed": bool(final_grade >= 50),
                }
            )
    return pd.DataFrame(rows)


def _write_mastery_fixture(workspace: Path) -> tuple[Path, Path, Path, Path]:
    frame = _synthetic_modeling_frame()
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
                "trajectory_type": "stable_high",
            }
            for sid in final_results["student_id"].tolist()
        ]
    ).to_csv(students_path, index=False)
    pd.DataFrame(
        [
            {
                "course_id": "course_x",
                "course_code": "CSX",
                "course_name": "Synthetic Course",
                "duration_weeks": int(snapshots["week_number"].max()),
                "grading_policy_pass_mark": 50,
            }
        ]
    ).to_csv(courses_path, index=False)
    return snapshots_path, finals_path, students_path, courses_path


def test_experiment_id_and_artifact_dir_for_exp_003() -> None:
    assert validate_experiment_id("exp_003_mastery_validation") == "exp_003_mastery_validation"
    expected = (ARTIFACT_EXPERIMENTS_DIR / "exp_003_mastery_validation").resolve()
    assert resolve_experiment_artifact_dir("exp_003_mastery_validation") == expected


def test_load_mastery_validation_config_yields_expected_thresholds() -> None:
    modeling, lifecycle, resolved = load_mastery_validation_config(
        "configs/experiments/exp_003_mastery_validation.yaml"
    )
    assert lifecycle.experiment.experiment_id == "exp_003_mastery_validation"
    assert lifecycle.mastery_validation.candidate_feature_set == "B_lms_plus_mastery"
    assert lifecycle.mastery_validation.baseline_feature_set == "B_lms"
    assert lifecycle.mastery_validation.weekly_cutoffs == [4, 5, 6, 7, 8, 9, 10]
    assert "current_topic_mastery" in lifecycle.mastery_validation.mastery_columns
    assert resolved.exists()
    assert "B_lms_plus_mastery" in modeling.feature_sets


def test_compute_target_and_redundancy_correlations() -> None:
    frame = _synthetic_modeling_frame()
    target = compute_mastery_target_correlations(
        frame,
        mastery_columns=["current_topic_mastery", "overall_mastery"],
        weekly_cutoffs=[4, 5, 6],
    )
    assert set(target["weekly_pearson"].keys()) == {
        "current_topic_mastery",
        "overall_mastery",
    }
    assert all(
        isinstance(value, float)
        for value in target["global_pearson"].values()
        if value is not None
    )
    assert target["global_pearson"]["overall_mastery"] is not None
    assert abs(target["global_pearson"]["overall_mastery"]) > 0.5

    lms_set = get_feature_set("B_lms")
    redundancy = compute_redundancy_with_lms(
        frame,
        mastery_columns=["current_topic_mastery", "overall_mastery"],
        lms_feature_set=lms_set,
    )
    assert "overall_mastery" in redundancy
    assert "avg_assignment_score_to_date" in redundancy["overall_mastery"]

    lms_target = compute_lms_target_correlations(frame, lms_feature_set=lms_set)
    assert "avg_assignment_score_to_date" in lms_target


def test_run_weekly_validation_only_examines_requested_weeks() -> None:
    frame = _synthetic_modeling_frame()
    requested_weeks = [4, 5, 6]
    rows = run_weekly_validation(
        frame,
        feature_sets=["B_lms", "B_lms_plus_mastery"],
        weekly_cutoffs=requested_weeks,
        model_names=["linear_regression"],
        seed=7,
        test_size=0.25,
        split_seed=7,
    )
    assert rows, "weekly validation should produce at least one row on the fixture"
    weeks_seen = sorted({row["week"] for row in rows})
    assert weeks_seen == sorted(set(requested_weeks))
    feature_sets_seen = {row["feature_set"] for row in rows}
    assert {"B_lms", "B_lms_plus_mastery"} <= feature_sets_seen


def test_summarize_weekly_delta_partitions_early_and_late_improvements() -> None:
    weekly_rows = [
        {
            "week": 4,
            "feature_set": "B_lms",
            "model": "lr",
            "rmse": 3.5,
            "mae": 0,
            "r2": 0,
            "n_train_rows": 1,
            "n_test_rows": 1,
        },
        {
            "week": 4,
            "feature_set": "B_lms_plus_mastery",
            "model": "lr",
            "rmse": 3.0,
            "mae": 0,
            "r2": 0,
            "n_train_rows": 1,
            "n_test_rows": 1,
        },
        {
            "week": 8,
            "feature_set": "B_lms",
            "model": "lr",
            "rmse": 2.0,
            "mae": 0,
            "r2": 0,
            "n_train_rows": 1,
            "n_test_rows": 1,
        },
        {
            "week": 8,
            "feature_set": "B_lms_plus_mastery",
            "model": "lr",
            "rmse": 1.5,
            "mae": 0,
            "r2": 0,
            "n_train_rows": 1,
            "n_test_rows": 1,
        },
        {
            "week": 9,
            "feature_set": "B_lms",
            "model": "lr",
            "rmse": 1.0,
            "mae": 0,
            "r2": 0,
            "n_train_rows": 1,
            "n_test_rows": 1,
        },
        {
            "week": 9,
            "feature_set": "B_lms_plus_mastery",
            "model": "lr",
            "rmse": 0.99,
            "mae": 0,
            "r2": 0,
            "n_train_rows": 1,
            "n_test_rows": 1,
        },
    ]
    summary = summarize_weekly_delta(
        weekly_rows,
        baseline_feature_set="B_lms",
        candidate_feature_set="B_lms_plus_mastery",
        early_week_max=6,
        improvement_rmse_tolerance=0.05,
    )
    assert summary["early_weeks_improved"] == [4]
    assert summary["late_weeks_improved"] == [8]
    assert 9 not in summary["early_weeks_improved"]
    assert 9 not in summary["late_weeks_improved"]
    assert summary["weeks_examined"] == [4, 8, 9]


def test_drop_column_tests_report_per_feature_delta() -> None:
    frame = _synthetic_modeling_frame()
    result = run_drop_column_tests(
        frame,
        candidate_feature_set_name="B_lms_plus_mastery",
        mastery_columns=["current_topic_mastery", "overall_mastery"],
        model_name="linear_regression",
        seed=3,
        test_size=0.25,
        split_seed=3,
    )
    assert result["feature_set"] == "B_lms_plus_mastery"
    assert result["full_rmse"] is not None
    drop_columns = {item["dropped_column"] for item in result["drop_results"]}
    assert {"current_topic_mastery", "overall_mastery"} <= drop_columns
    for item in result["drop_results"]:
        if item.get("rmse") is None:
            continue
        assert item["delta_rmse_vs_full"] is not None


def test_recommendation_flags_target_proxy_when_excess_over_lms() -> None:
    target_correlations = {
        "target_column": "final_grade",
        "mastery_columns": ["overall_mastery"],
        "global_pearson": {"overall_mastery": 0.99},
        "weekly_pearson": {"overall_mastery": {}},
    }
    weekly_summary = {
        "rows": [],
        "early_weeks_improved": [],
        "late_weeks_improved": [],
        "weeks_examined": [],
        "early_week_max": 6,
        "improvement_rmse_tolerance": 0.05,
    }
    recommendation = derive_carry_forward_recommendation(
        overall_delta_rmse=-0.02,
        weekly_summary=weekly_summary,
        target_correlations=target_correlations,
        lms_target_correlations={"avg_assignment_score_to_date": 0.80},
        redundancy_summary={
            "overall_mastery": {
                "max_abs_pearson": 0.80,
                "signed_pearson": 0.80,
                "lms_column": "avg_assignment_score_to_date",
            }
        },
        drop_column={"drop_results": []},
        candidate_feature_set="B_lms_plus_mastery",
        baseline_feature_set="B_lms",
        improvement_rmse_tolerance=0.05,
        target_proxy_correlation_warn=0.95,
        redundancy_correlation_warn=0.95,
    )
    assert recommendation["outcome"] == "do_not_carry_forward"
    assert any("more target-correlated" in flag for flag in recommendation["flags"])


def test_recommendation_carries_forward_when_baseline_is_already_correlated() -> None:
    """Mastery correlation matching the LMS baseline must not flag too-target-like."""

    target_correlations = {
        "target_column": "final_grade",
        "mastery_columns": ["overall_mastery"],
        "global_pearson": {"overall_mastery": 0.984},
        "weekly_pearson": {"overall_mastery": {}},
    }
    weekly_summary = {
        "rows": [],
        "early_weeks_improved": [4, 5],
        "late_weeks_improved": [7],
        "weeks_examined": [4, 5, 6, 7],
        "early_week_max": 6,
        "improvement_rmse_tolerance": 0.05,
    }
    recommendation = derive_carry_forward_recommendation(
        overall_delta_rmse=-0.20,
        weekly_summary=weekly_summary,
        target_correlations=target_correlations,
        lms_target_correlations={
            "activity_score_to_date": 0.983,
            "avg_assignment_score_to_date": 0.982,
        },
        redundancy_summary={
            "overall_mastery": {
                "max_abs_pearson": 0.85,
                "signed_pearson": 0.85,
                "lms_column": "avg_assignment_score_to_date",
            }
        },
        drop_column={"drop_results": []},
        candidate_feature_set="B_lms_plus_mastery",
        baseline_feature_set="B_lms",
        improvement_rmse_tolerance=0.05,
        target_proxy_correlation_warn=0.95,
        redundancy_correlation_warn=0.95,
    )
    assert recommendation["outcome"] == "carry_forward"


def test_run_mastery_validation_writes_versioned_artifacts_and_does_not_touch_prior_runs(
    tmp_path: Path,
) -> None:
    with _temp_workspace() as workspace:
        snapshots, finals, students, courses = _write_mastery_fixture(workspace)
        artifact_root = workspace / "artifacts"
        experiment_id = "exp_999_mastery"
        output_dir = artifact_root / experiment_id

        # Pretend a prior experiment already exists; the new run must not touch it.
        prior_dir = artifact_root / "exp_001_baseline"
        prior_dir.mkdir(parents=True)
        prior_metadata = prior_dir / "experiment_metadata.json"
        prior_metadata.write_text(
            json.dumps({"experiment_id": "exp_001_baseline", "title": "prior"}),
            encoding="utf-8",
        )
        prior_bytes = prior_metadata.read_bytes()

        config_payload = {
            "experiment": {
                "experiment_id": experiment_id,
                "title": "Mastery validation test",
                "schema_version": "1.2",
                "dataset_version": "test",
                "dataset_config_name": "test_generator.yaml",
                "parent_experiment": "exp_002_twin_ablation",
                "status": "completed",
            },
            "documentation": {
                "objective": "Verify runner artifact creation.",
                "hypothesis": "A small mastery validation run should still produce structured outputs.",
                "limitations": ["Synthetic fixture."],
                "next_step": "Move on with real exp_003 config.",
            },
            "mastery_validation": {
                "primary_split": "student_group",
                "baseline_feature_set": "B_lms",
                "candidate_feature_set": "B_lms_plus_mastery",
                "optional_feature_sets": [],
                "mastery_columns": ["current_topic_mastery", "overall_mastery"],
                "improvement_rmse_tolerance": 0.05,
                "early_week_max": 5,
                "target_proxy_correlation_warn": 0.99,
                "redundancy_correlation_warn": 0.95,
                "weekly_cutoffs": [4, 5, 6],
            },
            "name": experiment_id,
            "seed": 7,
            "feature_sets": ["B_lms", "B_lms_plus_mastery"],
            "classification_models": [],
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
                "student_group": {"test_size": 0.25, "validation_size": 0.0, "seed": 7},
                "temporal_forward": {
                    "train_weeks": 4,
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
        config_path = workspace / "mastery_test.yaml"
        config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")

        payload = run_mastery_validation(
            config_path,
            artifact_root=artifact_root,
            docs_dir=workspace / "docs",
        )

        assert payload["metadata_path"].exists()
        assert payload["diagnostics_path"].exists()
        assert payload["weekly_csv_path"].exists()
        assert payload["docs_path"].exists()
        assert payload["artifact_summary_path"].exists()
        assert payload["registry_path"].exists()

        diagnostics = json.loads(payload["diagnostics_path"].read_text(encoding="utf-8"))
        assert diagnostics["experiment_id"] == experiment_id
        assert "current_topic_mastery" in diagnostics["redundancy_summary"]
        assert diagnostics["recommendation"]["candidate_feature_set"] == "B_lms_plus_mastery"

        weekly_csv = pd.read_csv(payload["weekly_csv_path"])
        assert set(weekly_csv["week"].unique()) <= {4, 5, 6}
        assert {"B_lms", "B_lms_plus_mastery"} <= set(weekly_csv["feature_set"].unique())

        metadata = json.loads(payload["metadata_path"].read_text(encoding="utf-8"))
        assert metadata["targets"]["primary"] == "final_grade"
        assert "passed" in metadata["targets"]["excluded"]
        assert metadata["mastery_validation"]["weekly_cutoffs"] == [4, 5, 6]

        assert prior_metadata.read_bytes() == prior_bytes, (
            "Prior experiment metadata must not be modified by a new mastery run"
        )

        # Re-running into the same directory without --overwrite must fail.
        with pytest.raises(FileExistsError):
            run_mastery_validation(
                config_path,
                artifact_root=artifact_root,
                docs_dir=workspace / "docs",
            )


def test_existing_exp_003_artifacts_are_preserved_on_disk() -> None:
    """The committed exp_003 artifacts must survive the test session intact."""

    artifact_dir = ARTIFACT_EXPERIMENTS_DIR / "exp_003_mastery_validation"
    if not artifact_dir.exists():
        pytest.skip("exp_003 has not been run in this checkout yet")
    metadata_path = artifact_dir / "experiment_metadata.json"
    diagnostics_path = artifact_dir / "mastery_diagnostics.json"
    assert metadata_path.exists()
    assert diagnostics_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["experiment_id"] == "exp_003_mastery_validation"
    assert metadata["targets"]["primary"] == "final_grade"
    assert metadata["recommendation"]["candidate_feature_set"] == "B_lms_plus_mastery"
