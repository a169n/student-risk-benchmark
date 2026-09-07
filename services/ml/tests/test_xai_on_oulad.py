"""Tests for the OULAD model-behavior XAI runner (exp_007).

Covers:
- _split_masks: disjoint masks, leakage-safety for both strategies
- exp_007 config loading and structural validation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.run_xai_on_oulad import XaiOnOuladConfig, _split_masks

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = (
    Path(__file__).parent.parent
    / "configs"
    / "experiments"
    / "exp_007_xai_on_oulad.yaml"
)


def _make_tiny_frame(n_students: int = 8, n_weeks: int = 6) -> pd.DataFrame:
    """8 students x 6 weeks; weeks 1..6; student IDs are integers 101..108."""
    rows = []
    for student_id in range(101, 101 + n_students):
        for week in range(1, n_weeks + 1):
            rows.append(
                {
                    "student_id": student_id,
                    "week_number": week,
                    "final_grade": float(week * 10 + student_id % 10),
                    "passed": int((week * 10 + student_id % 10) >= 50),
                }
            )
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Test: _split_masks – student_group strategy
# ---------------------------------------------------------------------------


def test_split_masks_student_group_disjoint_and_leakage_safe() -> None:
    frame = _make_tiny_frame(n_students=8, n_weeks=6)
    student_group_params = {"test_size": 0.25, "seed": 42}
    temporal_forward_params = {
        "train_weeks": 4,
        "student_test_size": 0.25,
        "student_seed": 42,
    }

    train_mask, test_mask, meta = _split_masks(
        frame,
        "student_group",
        student_group=student_group_params,
        temporal_forward=temporal_forward_params,
    )

    # Both masks are boolean ndarrays aligned to frame length
    assert train_mask.dtype == bool
    assert test_mask.dtype == bool
    assert len(train_mask) == len(frame)
    assert len(test_mask) == len(frame)

    # Non-empty
    assert train_mask.sum() > 0, "train_mask must be non-empty"
    assert test_mask.sum() > 0, "test_mask must be non-empty"

    # Disjoint: no row can be in both train and test
    assert not np.any(train_mask & test_mask), "train and test masks must be disjoint"

    # Leakage-safety: no student in test appears in train
    train_students = set(frame.loc[train_mask, "student_id"].tolist())
    test_students = set(frame.loc[test_mask, "student_id"].tolist())
    overlap = train_students & test_students
    assert not overlap, (
        f"Student-group leakage: {overlap} students appear in both train and test"
    )

    # Metadata sanity
    assert meta["strategy"] == "student_group"
    assert meta["n_train"] == int(train_mask.sum())
    assert meta["n_test"] == int(test_mask.sum())


# ---------------------------------------------------------------------------
# Test: _split_masks – temporal_forward strategy
# ---------------------------------------------------------------------------


def test_split_masks_temporal_forward_disjoint_and_leakage_safe() -> None:
    frame = _make_tiny_frame(n_students=8, n_weeks=6)
    student_group_params = {"test_size": 0.25, "seed": 42}
    temporal_forward_params = {
        "train_weeks": 4,
        "student_test_size": 0.25,
        "student_seed": 42,
    }

    train_mask, test_mask, meta = _split_masks(
        frame,
        "temporal_forward",
        student_group=student_group_params,
        temporal_forward=temporal_forward_params,
    )

    assert train_mask.dtype == bool
    assert test_mask.dtype == bool
    assert len(train_mask) == len(frame)
    assert len(test_mask) == len(frame)

    # Non-empty
    assert train_mask.sum() > 0, "train_mask must be non-empty"
    assert test_mask.sum() > 0, "test_mask must be non-empty"

    # Disjoint
    assert not np.any(train_mask & test_mask), "train and test masks must be disjoint"

    # Leakage-safety: no held-out student appears in train
    train_students = set(frame.loc[train_mask, "student_id"].tolist())
    test_students = set(frame.loc[test_mask, "student_id"].tolist())
    overlap = train_students & test_students
    assert not overlap, (
        f"Temporal-forward leakage: {overlap} students appear in both train and test"
    )

    # Temporal constraint: all test weeks > train_weeks
    train_weeks = temporal_forward_params["train_weeks"]
    test_weeks = frame.loc[test_mask, "week_number"].tolist()
    assert all(w > train_weeks for w in test_weeks), (
        f"All test weeks must be > {train_weeks}; got {test_weeks}"
    )

    # Temporal constraint: all train weeks <= train_weeks
    tr_weeks = frame.loc[train_mask, "week_number"].tolist()
    assert all(w <= train_weeks for w in tr_weeks), (
        f"All train weeks must be <= {train_weeks}; got {tr_weeks}"
    )

    # Metadata sanity
    assert meta["strategy"] == "temporal_forward"
    assert meta["n_train"] == int(train_mask.sum())
    assert meta["n_test"] == int(test_mask.sum())
    assert meta["held_out_students"] > 0


# ---------------------------------------------------------------------------
# Test: _split_masks – both strategies tested in parametrize form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["student_group", "temporal_forward"])
def test_split_masks_disjoint_and_leakage_safe(strategy: str) -> None:
    """Parametrized version covering both strategies in a single test function."""
    frame = _make_tiny_frame(n_students=8, n_weeks=6)
    student_group_params = {"test_size": 0.25, "seed": 42}
    temporal_forward_params = {
        "train_weeks": 4,
        "student_test_size": 0.25,
        "student_seed": 42,
    }

    train_mask, test_mask, meta = _split_masks(
        frame,
        strategy,
        student_group=student_group_params,
        temporal_forward=temporal_forward_params,
    )

    assert not np.any(train_mask & test_mask), "masks must be disjoint"
    assert train_mask.sum() > 0
    assert test_mask.sum() > 0

    train_students = set(frame.loc[train_mask, "student_id"].tolist())
    test_students = set(frame.loc[test_mask, "student_id"].tolist())
    assert not (train_students & test_students), "no student may appear in both splits"

    if strategy == "temporal_forward":
        train_weeks_cutoff = temporal_forward_params["train_weeks"]
        assert all(
            w > train_weeks_cutoff
            for w in frame.loc[test_mask, "week_number"].tolist()
        )
        assert all(
            w <= train_weeks_cutoff
            for w in frame.loc[train_mask, "week_number"].tolist()
        )


# ---------------------------------------------------------------------------
# Test: exp_007 config loads and is structurally valid
# ---------------------------------------------------------------------------


def test_exp007_config_loads() -> None:
    """exp_007_xai_on_oulad.yaml must load without error and pass structural checks."""
    import yaml

    assert CONFIG_PATH.exists(), f"Config not found: {CONFIG_PATH}"
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    config = XaiOnOuladConfig.model_validate(raw)

    # Experiment identity
    assert config.experiment.experiment_id == "exp_007_xai_on_oulad"
    assert config.experiment.parent_experiment == "exp_006_oulad_full_ablation"

    # Exactly three feature sets present
    required_fs = {"B_lms_oulad", "B_lms_plus_mastery_oulad", "C_twin_oulad"}
    assert required_fs == set(config.feature_sets.keys()), (
        f"Expected feature sets {required_fs}, got {set(config.feature_sets.keys())}"
    )

    # feature_set_order contains exactly those three
    assert set(config.feature_set_order) == required_fs

    # Both splits configured
    assert config.splits.student_group.test_size == pytest.approx(0.25)
    assert config.splits.student_group.seed == 42
    assert config.splits.temporal_forward.train_weeks == 20
    assert config.splits.temporal_forward.student_test_size == pytest.approx(0.25)
    assert config.splits.temporal_forward.student_seed == 42

    # Model and seed
    assert config.model == "gradient_boosting"
    assert config.seed == 42
    assert config.permutation_repeats == 15

    # B_lms_oulad columns spot-check
    b_lms = config.feature_sets["B_lms_oulad"]
    assert "cumulative_assessment_weighted_score_to_date" in b_lms.columns
    assert "cumulative_clicks_to_date" in b_lms.columns

    # B_lms_plus_mastery_oulad must include mastery features
    mastery_fs = config.feature_sets["B_lms_plus_mastery_oulad"]
    assert "overall_mastery_proxy" in mastery_fs.columns
    assert "tma_mastery_to_date" in mastery_fs.columns

    # C_twin_oulad must include trend and index features
    c_twin = config.feature_sets["C_twin_oulad"]
    assert "assessment_score_trend_to_date" in c_twin.columns
    assert "engagement_index_oulad" in c_twin.columns
