"""TDD tests for exp_011: KU Leuven engagement-only PASSED classification benchmark.

test_config_loads
    Validates that the YAML config loads cleanly and contains the expected
    2 feature sets plus both split strategies.

test_split_masks_leakage_safe
    Constructs a small in-memory frame and verifies that:
    - student_group masks are disjoint (no row in both train and test),
    - no held-out-student leakage in temporal_forward (train contains only
      non-held-out students; test contains only held-out students),
    - temporal_forward test weeks are strictly > train_weeks for held-out students.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = (
    _REPO_ROOT
    / "services"
    / "ml"
    / "configs"
    / "experiments"
    / "exp_011_kuleuven_engagement.yaml"
)


# ---------------------------------------------------------------------------
# test_config_loads
# ---------------------------------------------------------------------------


def test_config_loads() -> None:
    """Config must load and have the expected structure."""
    from src.experiments.run_engagement_benchmark_kuleuven import (
        load_kuleuven_engagement_config,
    )

    config, resolved_path = load_kuleuven_engagement_config(_CONFIG_PATH)

    assert config.experiment_id == "exp_011_kuleuven_engagement"

    # Exactly 2 feature sets
    assert len(config.feature_sets) == 2, (
        f"Expected 2 feature sets, got {len(config.feature_sets)}: "
        f"{list(config.feature_sets.keys())}"
    )
    assert "A_simple_engagement" in config.feature_sets
    assert "B_engagement" in config.feature_sets

    # feature_set_order includes both
    assert config.feature_set_order == ["A_simple_engagement", "B_engagement"]

    # Both split strategies present
    assert config.splits.student_group is not None
    assert config.splits.temporal_forward is not None

    # Comparison fields
    assert config.comparison.baseline_feature_set == "A_simple_engagement"
    assert config.comparison.candidate_feature_set == "B_engagement"
    assert config.comparison.fixed_model == "gradient_boosting"

    # Classification only — no regression
    assert len(config.classification_models) > 0

    # Output dir matches experiment id
    output_dir = config.resolve_path(config.outputs.experiments_dir)
    assert output_dir.name == config.experiment_id, (
        f"Output dir name {output_dir.name!r} != experiment_id {config.experiment_id!r}"
    )

    assert resolved_path.exists()


# ---------------------------------------------------------------------------
# test_split_masks_leakage_safe
# ---------------------------------------------------------------------------


def _make_tiny_frame() -> pd.DataFrame:
    """20-student, 10-week synthetic frame (200 rows)."""
    students = [f"s{i:02d}" for i in range(20)]
    weeks = list(range(1, 11))
    rows = []
    for sid in students:
        for wk in weeks:
            rows.append(
                {
                    "student_id": sid,
                    "week_number": wk,
                    "final_grade": 0.7,
                    "passed": 1,
                    "cumulative_clicks_to_date": float(wk * 10),
                    "cumulative_active_days_to_date": float(wk),
                    "has_activity_to_date": 1,
                }
            )
    return pd.DataFrame(rows)


def test_split_masks_leakage_safe() -> None:
    """Verify disjointness and no student leakage across both split strategies."""
    import yaml

    from src.experiments.preprocessing import build_modeling_matrix
    from src.experiments.run_engagement_benchmark_kuleuven import (
        KuLeuvenEngagementConfig,
        build_split,
    )

    # Build a minimal config dict for the tiny frame
    raw_config: dict = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

    # Patch output dir validation — we're not running the full pipeline
    # so we need to bypass the model_validator path check. Instead we
    # instantiate the config directly without the full outputs block and
    # call build_split directly.
    from src.experiments.engagement_benchmark import (
        SplitsConfig,
        StudentGroupSplitConfig,
        TemporalForwardSplitConfig,
        _SplitPartition,
    )
    from src.experiments.preprocessing import PreparedMatrix

    # We need a real config for build_split. Use the real config from disk.
    from src.experiments.run_engagement_benchmark_kuleuven import (
        load_kuleuven_engagement_config,
    )
    config, _ = load_kuleuven_engagement_config(_CONFIG_PATH)

    frame = _make_tiny_frame()

    # Build a minimal feature set for the tiny frame
    from src.experiments.engagement_benchmark import BenchmarkFeatureSet
    tiny_fs = BenchmarkFeatureSet(
        name="A_simple_engagement",
        description="test",
        columns=("cumulative_clicks_to_date", "cumulative_active_days_to_date"),
        indicator_columns=("has_activity_to_date",),
    )
    matrix = build_modeling_matrix(frame, tiny_fs)  # type: ignore[arg-type]
    n = matrix.features.shape[0]

    # --- student_group split ---
    sg_partition = build_split(matrix, "student_group", config)

    train_mask_sg = sg_partition.train_mask
    test_mask_sg = sg_partition.test_mask

    # Disjoint
    assert not np.any(train_mask_sg & test_mask_sg), (
        "student_group: train and test masks overlap"
    )
    # Coverage: every row is in train or test (no dropped rows for student_group)
    assert np.all(train_mask_sg | test_mask_sg), (
        "student_group: some rows are in neither train nor test"
    )

    # No student appears in both train and test
    from src.experiments.datasets import GROUP_COLUMN
    groups = matrix.groups.to_numpy()
    train_students_sg = set(groups[train_mask_sg])
    test_students_sg = set(groups[test_mask_sg])
    leaking_students = train_students_sg & test_students_sg
    assert not leaking_students, (
        f"student_group: students appear in both train and test: {leaking_students}"
    )

    # --- temporal_forward split ---
    tf_partition = build_split(matrix, "temporal_forward", config)
    train_mask_tf = tf_partition.train_mask
    test_mask_tf = tf_partition.test_mask

    # Disjoint
    assert not np.any(train_mask_tf & test_mask_tf), (
        "temporal_forward: train and test masks overlap"
    )

    # Held-out students: test students must not be in train
    train_students_tf = set(groups[train_mask_tf])
    test_students_tf = set(groups[test_mask_tf])
    leaking_students_tf = train_students_tf & test_students_tf
    assert not leaking_students_tf, (
        f"temporal_forward: students appear in both train and test: {leaking_students_tf}"
    )

    # Temporal constraint: all test-set weeks > train_weeks
    weeks = matrix.weeks.to_numpy()
    train_weeks_cutoff = config.splits.temporal_forward.train_weeks
    test_weeks = weeks[test_mask_tf]
    assert np.all(test_weeks > train_weeks_cutoff), (
        f"temporal_forward: some test rows have week <= {train_weeks_cutoff}: "
        f"{np.unique(test_weeks[test_weeks <= train_weeks_cutoff])}"
    )

    # Temporal constraint: all train-set weeks <= train_weeks
    train_weeks = weeks[train_mask_tf]
    assert np.all(train_weeks <= train_weeks_cutoff), (
        f"temporal_forward: some train rows have week > {train_weeks_cutoff}: "
        f"{np.unique(train_weeks[train_weeks > train_weeks_cutoff])}"
    )
