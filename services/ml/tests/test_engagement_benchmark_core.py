"""Characterization tests for the extracted dataset-agnostic engagement core.

These tests pin the public surface of ``src.experiments.engagement_benchmark``
so the OULAD runner (Task 3) can reuse the IDENTICAL methodology that produced
exp_011 for KU Leuven.
"""

from __future__ import annotations

import pandas as pd


def test_core_symbols_importable():
    from src.experiments.engagement_benchmark import (
        BenchmarkFeatureSet,
        build_split,
        compute_permutation_importance,
        _fixed_model_classification_by_feature_set,
    )

    assert BenchmarkFeatureSet(
        name="x", description="d", columns=("a",)
    ).all_columns() == ("a",)
    assert callable(build_split)
    assert callable(compute_permutation_importance)
    assert callable(_fixed_model_classification_by_feature_set)


def test_fixed_model_classification_filters_one_model():
    from src.experiments.engagement_benchmark import (
        _fixed_model_classification_by_feature_set,
    )

    # _fixed_model_classification_by_feature_set consumes a results-style frame
    # produced by results_to_dataframe: it requires the columns "task", "model",
    # "split_strategy", "feature_set", "n_train_rows", "n_test_rows", and the
    # renamed-from "metric_f1", "metric_accuracy", "metric_roc_auc" metric
    # columns. It filters rows to (task == "classification") & (model == model)
    # and groups by (split_strategy, feature_set).
    table = pd.DataFrame(
        [
            # Target model rows (should survive the filter), two feature sets.
            {
                "task": "classification",
                "model": "gradient_boosting",
                "split_strategy": "temporal_forward",
                "feature_set": "A_simple_engagement",
                "n_train_rows": 100,
                "n_test_rows": 40,
                "metric_f1": 0.70,
                "metric_accuracy": 0.72,
                "metric_roc_auc": 0.75,
            },
            {
                "task": "classification",
                "model": "gradient_boosting",
                "split_strategy": "temporal_forward",
                "feature_set": "B_engagement",
                "n_train_rows": 100,
                "n_test_rows": 40,
                "metric_f1": 0.80,
                "metric_accuracy": 0.82,
                "metric_roc_auc": 0.85,
            },
            # Other-model rows (must be filtered OUT).
            {
                "task": "classification",
                "model": "logistic_regression",
                "split_strategy": "temporal_forward",
                "feature_set": "A_simple_engagement",
                "n_train_rows": 100,
                "n_test_rows": 40,
                "metric_f1": 0.10,
                "metric_accuracy": 0.11,
                "metric_roc_auc": 0.12,
            },
            {
                "task": "classification",
                "model": "random_forest",
                "split_strategy": "temporal_forward",
                "feature_set": "B_engagement",
                "n_train_rows": 100,
                "n_test_rows": 40,
                "metric_f1": 0.20,
                "metric_accuracy": 0.21,
                "metric_roc_auc": 0.22,
            },
        ]
    )

    records = _fixed_model_classification_by_feature_set(
        table,
        model="gradient_boosting",
        baseline_feature_set="A_simple_engagement",
        primary_split="temporal_forward",
    )

    # The function must restrict output to the single requested model family.
    assert records, "expected at least one record for the fixed model"
    assert {r["model"] for r in records} == {"gradient_boosting"}
    # Two feature sets for the fixed model survive.
    assert {r["feature_set"] for r in records} == {
        "A_simple_engagement",
        "B_engagement",
    }
    # Delta is computed relative to the baseline feature set on the same split.
    candidate = next(r for r in records if r["feature_set"] == "B_engagement")
    assert candidate["delta_f1"] == round(0.80 - 0.70, 4)
    assert candidate["delta_auc"] == round(0.85 - 0.75, 4)
