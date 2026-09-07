"""Tests for exp_008: controlled XAI-faithfulness probe.

Three tests:
1. test_probe_restricts_to_final_week  — frame used is week 10 only.
2. test_probe_reports_oracle_and_redundancy — results dict has required keys.
3. test_probe_text_avoids_forbidden_framing — summary markdown has no
   forbidden substrings.
"""

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_faithfulness_probe import (
    _build_split_masks,
    _run_oracle_probe,
    _run_proxy_probe,
    _run_redundancy_diagnostic,
    render_probe_markdown,
    run_faithfulness_probe,
    FaithfulnessProbeConfig,
    DocumentationConfig,
    ExperimentIdentityConfig,
    ProbeConfig,
)
from src.generator.config import REPO_ROOT


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@contextmanager
def _temp_workspace() -> Iterator[Path]:
    base = REPO_ROOT / "services" / "ml" / ".tmp_test_runs" / f"probe-{uuid.uuid4().hex[:8]}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


ORACLE_FEATURES = [
    "avg_assignment_score_to_date",
    "avg_quiz_score_to_date",
    "attendance_rate_to_date",
    "on_time_submission_rate_to_date",
]
ORACLE_WEIGHTS = {
    "avg_assignment_score_to_date": 0.55,
    "avg_quiz_score_to_date": 0.25,
    "attendance_rate_to_date": 0.10,
    "on_time_submission_rate_to_date": 0.10,
}


def _make_probe_frame(
    *,
    num_students: int = 40,
    seed: int = 77,
    week_number: int = 10,
) -> pd.DataFrame:
    """Build a small synthetic final-week frame with known oracle weights."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(num_students):
        assign = float(np.clip(30 + i * 1.5 + rng.normal(0, 3), 0, 100))
        quiz = float(np.clip(assign * 0.9 + rng.normal(0, 4), 0, 100))
        attendance = float(np.clip(0.5 + i / (2 * num_students) + rng.normal(0, 0.05), 0, 1))
        on_time = float(np.clip(0.5 + i / (2 * num_students) + rng.normal(0, 0.05), 0, 1))
        activity = float(np.clip(assign * 0.8 + rng.normal(0, 5), 0, 100))
        overall_mastery = float(np.clip(assign + rng.normal(0, 2), 0, 100))
        final_grade = float(
            np.clip(
                0.55 * assign
                + 0.25 * quiz
                + 0.10 * attendance * 100
                + 0.10 * on_time * 100,
                0,
                100,
            )
        )
        rows.append(
            {
                "student_id": f"student_{i:03d}",
                "course_id": "course_x",
                "week_number": week_number,
                "avg_assignment_score_to_date": assign,
                "avg_quiz_score_to_date": quiz,
                "attendance_rate_to_date": attendance,
                "on_time_submission_rate_to_date": on_time,
                "activity_score_to_date": activity,
                "time_spent_to_date": float(45 * week_number * max(i / num_students, 0.1)),
                "missed_assignments_to_date": int(max(0, round((1 - on_time) * week_number))),
                "late_submissions_to_date": int(
                    max(0, round((1 - on_time) * week_number / 2))
                ),
                "avg_attempt_count_to_date": round(1.0 + (1 - i / num_students) * 0.5, 2),
                "has_assignment_score_to_date": True,
                "has_quiz_score_to_date": True,
                "overall_mastery": overall_mastery,
                "final_grade": final_grade,
                "passed": bool(final_grade >= 50),
                "completion_status": "completed",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: probe restricts to final week
# ---------------------------------------------------------------------------


def test_probe_restricts_to_final_week() -> None:
    """The run_faithfulness_probe function uses only the final-week rows."""
    with _temp_workspace() as workspace:
        # Build a multi-week frame and write fixture CSVs
        frames = []
        for wk in [4, 7, 10]:
            wk_frame = _make_probe_frame(week_number=wk, num_students=30, seed=wk)
            frames.append(wk_frame)
        full_frame = pd.concat(frames, ignore_index=True)

        snap_cols = [c for c in full_frame.columns if c not in {"final_grade", "passed", "completion_status"}]
        snapshots = full_frame[snap_cols].copy()
        finals = (
            full_frame[
                ["student_id", "course_id", "final_grade", "passed", "completion_status"]
            ]
            .drop_duplicates(subset=["student_id", "course_id"])
            .copy()
        )
        finals["final_result_id"] = [f"fr_{i}" for i in range(len(finals))]

        snap_path = workspace / "snapshots.csv"
        fin_path = workspace / "finals.csv"
        stu_path = workspace / "students.csv"
        cou_path = workspace / "courses.csv"
        snapshots.to_csv(snap_path, index=False)
        finals.to_csv(fin_path, index=False)
        pd.DataFrame(
            [
                {"student_id": sid, "student_code": sid, "course_id": "course_x"}
                for sid in finals["student_id"].tolist()
            ]
        ).to_csv(stu_path, index=False)
        pd.DataFrame(
            [{"course_id": "course_x", "course_code": "CX", "course_name": "Test", "duration_weeks": 10, "grading_policy_pass_mark": 50}]
        ).to_csv(cou_path, index=False)

        artifact_dir = workspace / "artifacts"
        artifact_dir.mkdir()

        config_text = f"""
experiment:
  experiment_id: "exp_008_faithfulness_probe"
  title: "Faithfulness probe test"
  schema_version: "1.2"
  dataset_version: "v_test"
  dataset_config_name: "test.yaml"
  parent_experiment: null
  status: "completed"

documentation:
  objective: "Test"
  hypothesis: ""
  limitations: []
  next_step: ""

probe:
  oracle_weights:
    avg_assignment_score_to_date: 0.55
    avg_quiz_score_to_date: 0.25
    attendance_rate_to_date: 0.10
    on_time_submission_rate_to_date: 0.10
  oracle_features:
    - "avg_assignment_score_to_date"
    - "avg_quiz_score_to_date"
    - "attendance_rate_to_date"
    - "on_time_submission_rate_to_date"
  proxy_feature_set: "B_lms"
  redundancy_pairs:
    - ["overall_mastery", "avg_assignment_score_to_date"]
  permutation_repeats: 3
  model: "gradient_boosting"
  seed: 7
  shap_enabled: false
  final_week: null

name: "exp_008_faithfulness_probe"
seed: 7

feature_sets:
  - "B_lms"

classification_models: []

regression_models:
  - "gradient_boosting"

dataset:
  snapshots_csv: "{snap_path.as_posix()}"
  final_results_csv: "{fin_path.as_posix()}"
  students_csv: "{stu_path.as_posix()}"
  courses_csv: "{cou_path.as_posix()}"

snapshot_filter:
  min_week: 4
  max_week: null

splits:
  primary: "student_group"
  secondary: null
  student_group:
    test_size: 0.25
    validation_size: 0.0
    seed: 7
  temporal_forward:
    train_weeks: 6
    student_test_size: 0.25
    student_seed: 7

eda:
  enabled: false
  correlation_top_k: 12

outputs:
  eda_dir: "{(artifact_dir / 'eda').as_posix()}"
  experiments_dir: "{(artifact_dir / 'exp_008_faithfulness_probe').as_posix()}"
"""
        config_path = workspace / "probe_config.yaml"
        config_path.write_text(config_text, encoding="utf-8")

        docs_dir = workspace / "docs"
        docs_dir.mkdir()

        payload = run_faithfulness_probe(
            config_path,
            allow_existing=True,
            artifact_root=artifact_dir,
            docs_dir=docs_dir,
        )
        results = payload["results"]

        # The probe should use week 10 (max week in the multi-week frame)
        assert results["final_week"] == 10, f"Expected final_week=10, got {results['final_week']}"
        assert results["n_final_week_rows"] == 30, (
            f"Expected 30 rows at week 10, got {results['n_final_week_rows']}"
        )


# ---------------------------------------------------------------------------
# Test 2: results dict has required structure
# ---------------------------------------------------------------------------


def test_probe_reports_oracle_and_redundancy() -> None:
    """Results dict contains oracle Kendall tau, per-feature shares, proxy, redundancy."""
    frame = _make_probe_frame(num_students=40, seed=99)

    train_mask, test_mask = _build_split_masks(frame, test_size=0.25, seed=42)

    oracle_result = _run_oracle_probe(
        frame,
        oracle_features=ORACLE_FEATURES,
        oracle_weights=ORACLE_WEIGHTS,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name="gradient_boosting",
        seed=42,
        permutation_repeats=3,
    )
    proxy_result = _run_proxy_probe(
        frame,
        proxy_feature_set_name="B_lms",
        train_mask=train_mask,
        test_mask=test_mask,
        model_name="gradient_boosting",
        seed=42,
        permutation_repeats=3,
    )
    redundancy_result = _run_redundancy_diagnostic(
        frame,
        redundancy_pairs=[["overall_mastery", "avg_assignment_score_to_date"]],
    )

    # Oracle Kendall tau: must be a float in [-1,1] or None
    tau = oracle_result["kendall_tau"]
    assert tau is None or isinstance(tau, float), f"tau must be float or None, got {type(tau)}"
    if tau is not None:
        assert -1.0 <= tau <= 1.0, f"tau out of range: {tau}"

    # Per-feature entries
    assert "per_feature" in oracle_result
    assert len(oracle_result["per_feature"]) == len(ORACLE_FEATURES)
    for pf in oracle_result["per_feature"]:
        assert "feature" in pf
        assert "oracle_weight" in pf
        assert "importance_share" in pf

    # Proxy: activity importance must be reported
    assert "proxy_importance_share" in proxy_result
    proxy_share = proxy_result["proxy_importance_share"]
    assert proxy_share is None or isinstance(proxy_share, float), (
        f"proxy_share must be float or None, got {type(proxy_share)}"
    )

    # Redundancy: key exists and r is a float
    assert len(redundancy_result) == 1
    r_val = redundancy_result[0]["pearson_r"]
    assert isinstance(r_val, float), f"pearson_r must be float, got {type(r_val)}"
    assert r_val is not None


# ---------------------------------------------------------------------------
# Test 3: summary markdown avoids forbidden framing
# ---------------------------------------------------------------------------


def test_probe_text_avoids_forbidden_framing() -> None:
    """Rendered summary must not contain 'zero-weight' or 'infidelity'."""
    frame = _make_probe_frame(num_students=40, seed=55)
    train_mask, test_mask = _build_split_masks(frame, test_size=0.25, seed=42)

    oracle_result = _run_oracle_probe(
        frame,
        oracle_features=ORACLE_FEATURES,
        oracle_weights=ORACLE_WEIGHTS,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name="gradient_boosting",
        seed=42,
        permutation_repeats=3,
    )
    proxy_result = _run_proxy_probe(
        frame,
        proxy_feature_set_name="B_lms",
        train_mask=train_mask,
        test_mask=test_mask,
        model_name="gradient_boosting",
        seed=42,
        permutation_repeats=3,
    )
    redundancy_result = _run_redundancy_diagnostic(
        frame,
        redundancy_pairs=[["overall_mastery", "avg_assignment_score_to_date"]],
    )

    results = {
        "experiment_id": "exp_008_faithfulness_probe",
        "final_week": 10,
        "n_final_week_rows": len(frame),
        "n_students": int(frame["student_id"].nunique()),
        "oracle_probe": oracle_result,
        "proxy_probe": proxy_result,
        "redundancy": redundancy_result,
        "method_notes": {
            "shap_used": False,
            "comparison_method": "Kendall tau",
            "importance_method": "permutation importance",
            "split": "student_group",
        },
    }

    lifecycle = FaithfulnessProbeConfig(
        experiment=ExperimentIdentityConfig(
            experiment_id="exp_008_faithfulness_probe",
            title="Faithfulness probe",
        ),
        documentation=DocumentationConfig(
            objective="Test probe for forbidden framing.",
            limitations=["Synthetic data only."],
            next_step="None.",
        ),
        probe=ProbeConfig(
            oracle_weights=ORACLE_WEIGHTS,
            oracle_features=ORACLE_FEATURES,
            proxy_feature_set="B_lms",
            redundancy_pairs=[["overall_mastery", "avg_assignment_score_to_date"]],
            permutation_repeats=3,
        ),
    )

    from src.experiments.config import ExperimentConfig

    modeling_config = ExperimentConfig(
        name="exp_008_faithfulness_probe",
        seed=42,
        feature_sets=["B_lms"],
        regression_models=["gradient_boosting"],
        classification_models=[],
    )

    from pathlib import Path

    markdown = render_probe_markdown(
        lifecycle=lifecycle,
        results=results,
        modeling_config=modeling_config,
        config_path=Path("configs/experiments/exp_008_faithfulness_probe.yaml"),
        artifact_paths={},
    )

    lower_md = markdown.lower()
    assert "zero-weight" not in lower_md, "Forbidden string 'zero-weight' found in summary"
    assert "infidelity" not in lower_md, "Forbidden string 'infidelity' found in summary"
