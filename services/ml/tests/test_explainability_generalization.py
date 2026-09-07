import numpy as np
import pandas as pd

from src.experiments.explainability import (
    TrainedRegressionModel,
    build_global_explanation,
    train_regression_reference_for_matrix,
)


class _FeatureSet:
    """Minimal duck-typed feature set (mirrors BenchmarkFeatureSet shape)."""

    def __init__(self, name, columns, indicator_columns=()):
        self.name = name
        self.columns = tuple(columns)
        self.indicator_columns = tuple(indicator_columns)

    def all_columns(self):
        return tuple([*self.columns, *self.indicator_columns])


def _frame(n=60):
    rng = np.random.default_rng(0)
    students = np.repeat([f"s{i}" for i in range(n // 6)], 6)
    weeks = np.tile(np.arange(1, 7), n // 6)
    f1 = rng.normal(50, 10, size=len(students))
    f2 = rng.normal(30, 5, size=len(students))
    grade = 0.7 * f1 + 0.3 * f2 + rng.normal(0, 3, size=len(students))
    return pd.DataFrame(
        {
            "student_id": students,
            "week_number": weeks,
            "final_grade": grade,
            "passed": (grade >= 50).astype(int),
            "feat_one": f1,
            "feat_two": f2,
        }
    )


def test_train_for_matrix_with_explicit_masks_and_columns():
    frame = _frame()
    fs = _FeatureSet("custom_set", ["feat_one", "feat_two"])
    rng = np.random.default_rng(1)
    mask = rng.random(len(frame)) < 0.7
    train_mask = mask
    test_mask = ~mask

    run = train_regression_reference_for_matrix(
        frame,
        feature_set=fs,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name="gradient_boosting",
        seed=42,
    )
    assert isinstance(run, TrainedRegressionModel)
    assert run.feature_set_name == "custom_set"
    assert set(run.feature_columns) >= {"feat_one", "feat_two"}
    assert run.x_train.shape[0] == int(train_mask.sum())
    assert run.x_test.shape[0] == int(test_mask.sum())
    assert "rmse" in run.metrics


def test_build_global_explanation_split_strategy_label():
    frame = _frame()
    fs = _FeatureSet("custom_set", ["feat_one", "feat_two"])
    n = len(frame)
    train_mask = np.arange(n) % 3 != 0
    test_mask = ~train_mask
    run = train_regression_reference_for_matrix(
        frame, feature_set=fs, train_mask=train_mask, test_mask=test_mask,
        model_name="gradient_boosting", seed=42,
    )
    g = build_global_explanation(run, permutation_repeats=3, seed=42, split_strategy="temporal_forward")
    assert g["split_strategy"] == "temporal_forward"
    # default still works:
    g2 = build_global_explanation(run, permutation_repeats=3, seed=42)
    assert g2["split_strategy"] == "student_group"
