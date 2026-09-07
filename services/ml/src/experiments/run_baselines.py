"""End-to-end baseline experiment runner.

Steps:
1. Load the experiment config.
2. Load the modeling dataset (snapshots joined with end-of-course outcomes).
3. Optionally run experiment-oriented EDA.
4. For each declared feature set, build a leakage-safe modeling matrix.
5. For each declared split strategy, partition the data using student-grouped
   or temporal-forward splits and fit a median imputer on the training
   partition only.
6. Train each baseline model, score it, and collect a ``ResultRow``.
7. Persist results as CSV, JSON, and markdown summary tables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.config import (
    ExperimentConfig,
    SplitStrategy,
    load_experiment_config,
)
from src.experiments.datasets import (
    GROUP_COLUMN,
    WEEK_COLUMN,
    ModelingDataset,
    load_modeling_dataset,
)
from src.experiments.eda import run_eda
from src.experiments.evaluate import (
    ResultRow,
    compute_classification_metrics,
    compute_regression_metrics,
    write_result_artifacts,
)
from src.experiments.featuresets import FeatureSet, available_feature_sets, get_feature_set
from src.experiments.models import (
    iter_classification_models,
    iter_regression_models,
)
from src.experiments.preprocessing import (
    PreparedMatrix,
    build_modeling_matrix,
    fit_imputer_on_training,
    select_rows,
)
from src.experiments.splits import (
    SplitResult,
    student_group_split,
)


@dataclass
class _SplitPartition:
    """Boolean masks aligned to the prepared matrix index."""

    train_mask: np.ndarray
    test_mask: np.ndarray
    metadata: dict[str, Any]
    strategy: SplitStrategy


def _build_split(
    matrix: PreparedMatrix,
    strategy: SplitStrategy,
    config: ExperimentConfig,
) -> _SplitPartition:
    """Compute boolean train/test masks aligned to the prepared matrix."""

    base = pd.DataFrame(
        {
            GROUP_COLUMN: matrix.groups.values,
            WEEK_COLUMN: matrix.weeks.values,
            "_row_index": np.arange(matrix.features.shape[0]),
        }
    )

    if strategy == "student_group":
        params = config.splits.student_group
        result: SplitResult = student_group_split(
            base,
            test_size=params.test_size,
            validation_size=params.validation_size,
            seed=params.seed,
            student_id_column=GROUP_COLUMN,
        )
        train_indices = result.train["_row_index"].to_numpy()
        test_indices = result.test["_row_index"].to_numpy()
        metadata = result.metadata
    elif strategy == "temporal_forward":
        params = config.splits.temporal_forward
        # First trim to held-out students so the temporal split is also free of
        # student-level leakage (a stronger guarantee than week-only splitting).
        held_out_split: SplitResult = student_group_split(
            base,
            test_size=params.student_test_size,
            seed=params.student_seed,
            student_id_column=GROUP_COLUMN,
        )
        held_out_students = set(held_out_split.test[GROUP_COLUMN].unique())
        train_pool = base.loc[~base[GROUP_COLUMN].isin(held_out_students)].copy()
        test_pool = base.loc[base[GROUP_COLUMN].isin(held_out_students)].copy()

        train_mask_pool = train_pool[WEEK_COLUMN] <= params.train_weeks
        test_mask_pool = test_pool[WEEK_COLUMN] > params.train_weeks
        train_indices = train_pool.loc[train_mask_pool, "_row_index"].to_numpy()
        test_indices = test_pool.loc[test_mask_pool, "_row_index"].to_numpy()
        metadata = {
            "train_weeks": params.train_weeks,
            "student_test_size": params.student_test_size,
            "student_seed": params.student_seed,
            "held_out_students": len(held_out_students),
            "train_rows": int(train_indices.shape[0]),
            "test_rows": int(test_indices.shape[0]),
        }
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    train_mask = np.zeros(matrix.features.shape[0], dtype=bool)
    test_mask = np.zeros(matrix.features.shape[0], dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return _SplitPartition(
        train_mask=train_mask,
        test_mask=test_mask,
        metadata=metadata,
        strategy=strategy,
    )


def _select_strategies(config: ExperimentConfig) -> list[SplitStrategy]:
    strategies: list[SplitStrategy] = [config.splits.primary]
    if config.splits.secondary and config.splits.secondary != config.splits.primary:
        strategies.append(config.splits.secondary)
    return strategies


def _train_and_score_classification(
    matrix: PreparedMatrix,
    *,
    feature_set: FeatureSet,
    partition: _SplitPartition,
    config: ExperimentConfig,
) -> list[ResultRow]:
    train_matrix = select_rows(matrix, partition.train_mask)
    test_matrix = select_rows(matrix, partition.test_mask)
    if train_matrix.features.empty or test_matrix.features.empty:
        return []

    imputer = fit_imputer_on_training(train_matrix.features)
    x_train = imputer.transform(train_matrix.features)
    x_test = imputer.transform(test_matrix.features)
    y_train = train_matrix.classification_target.to_numpy()
    y_test = test_matrix.classification_target.to_numpy()

    rows: list[ResultRow] = []
    if len(np.unique(y_train)) < 2:
        return rows

    for name, estimator in iter_classification_models(
        config.classification_models, seed=config.seed
    ):
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)
        if hasattr(estimator, "predict_proba"):
            y_proba = estimator.predict_proba(x_test)[:, 1]
        elif hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(x_test)
            y_proba = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_proba = None

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        rows.append(
            ResultRow(
                feature_set=feature_set.name,
                model=name,
                target="passed",
                task="classification",
                split_strategy=partition.strategy,
                split_metadata=partition.metadata,
                n_train_rows=int(x_train.shape[0]),
                n_test_rows=int(x_test.shape[0]),
                n_train_students=int(train_matrix.groups.nunique()),
                n_test_students=int(test_matrix.groups.nunique()),
                metrics=metrics,
            )
        )
    return rows


def _train_and_score_regression(
    matrix: PreparedMatrix,
    *,
    feature_set: FeatureSet,
    partition: _SplitPartition,
    config: ExperimentConfig,
) -> list[ResultRow]:
    train_matrix = select_rows(matrix, partition.train_mask)
    test_matrix = select_rows(matrix, partition.test_mask)
    if train_matrix.features.empty or test_matrix.features.empty:
        return []

    imputer = fit_imputer_on_training(train_matrix.features)
    x_train = imputer.transform(train_matrix.features)
    x_test = imputer.transform(test_matrix.features)
    y_train = train_matrix.regression_target.to_numpy()
    y_test = test_matrix.regression_target.to_numpy()

    rows: list[ResultRow] = []
    for name, estimator in iter_regression_models(config.regression_models, seed=config.seed):
        estimator.fit(x_train, y_train)
        y_pred = estimator.predict(x_test)
        metrics = compute_regression_metrics(y_test, y_pred)
        rows.append(
            ResultRow(
                feature_set=feature_set.name,
                model=name,
                target="final_grade",
                task="regression",
                split_strategy=partition.strategy,
                split_metadata=partition.metadata,
                n_train_rows=int(x_train.shape[0]),
                n_test_rows=int(x_test.shape[0]),
                n_train_students=int(train_matrix.groups.nunique()),
                n_test_students=int(test_matrix.groups.nunique()),
                metrics=metrics,
            )
        )
    return rows


def _maybe_load_students(config: ExperimentConfig) -> pd.DataFrame | None:
    students_path = config.resolve_path(config.dataset.students_csv)
    if not students_path.exists():
        return None
    return pd.read_csv(students_path)


def run_experiments(config: ExperimentConfig) -> dict[str, Any]:
    dataset: ModelingDataset = load_modeling_dataset(config)
    students_frame = _maybe_load_students(config)

    eda_artifacts = None
    if config.eda.enabled:
        eda_artifacts = run_eda(
            dataset.frame,
            output_dir=config.resolve_path(config.outputs.eda_dir),
            correlation_top_k=config.eda.correlation_top_k,
            students_frame=students_frame,
        )

    selected_feature_sets: list[FeatureSet] = [
        get_feature_set(name) for name in config.feature_sets
    ]
    strategies = _select_strategies(config)

    rows: list[ResultRow] = []
    for feature_set in selected_feature_sets:
        matrix = build_modeling_matrix(dataset.frame, feature_set)
        for strategy in strategies:
            partition = _build_split(matrix, strategy, config)
            if partition.train_mask.sum() == 0 or partition.test_mask.sum() == 0:
                continue
            rows.extend(
                _train_and_score_classification(
                    matrix,
                    feature_set=feature_set,
                    partition=partition,
                    config=config,
                )
            )
            rows.extend(
                _train_and_score_regression(
                    matrix,
                    feature_set=feature_set,
                    partition=partition,
                    config=config,
                )
            )

    output_dir = config.resolve_path(config.outputs.experiments_dir)
    metadata = {
        "snapshots_path": str(dataset.snapshots_path),
        "final_results_path": str(dataset.final_results_path),
        "snapshot_week_min": dataset.snapshot_week_min,
        "snapshot_week_max": dataset.snapshot_week_max,
        "n_modeling_rows": dataset.n_rows,
        "n_modeling_students": dataset.n_students,
        "feature_sets": [
            {
                "name": fs.name,
                "description": fs.description,
                "columns": list(fs.columns),
                "indicator_columns": list(fs.indicator_columns),
            }
            for fs in selected_feature_sets
        ],
        "split_strategies": list(strategies),
    }
    paths = write_result_artifacts(
        rows,
        output_dir=output_dir,
        run_name=config.name,
        extra_metadata=metadata,
    )

    return {
        "result_rows": rows,
        "result_paths": paths,
        "eda_artifacts": eda_artifacts,
        "metadata": metadata,
    }


def _print_summary(payload: dict[str, Any]) -> None:
    rows = payload["result_rows"]
    paths = payload["result_paths"]
    eda_artifacts = payload["eda_artifacts"]

    print(f"Trained models: {len(rows)}")
    print(f"Results CSV: {paths['csv']}")
    print(f"Results JSON: {paths['json']}")
    print(f"Markdown summary: {paths['markdown']}")
    if eda_artifacts is not None:
        print(f"EDA report: {eda_artifacts.report_path}")
    print(f"Metadata: {json.dumps(payload['metadata'], indent=2, default=str)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline modeling experiments.")
    parser.add_argument(
        "--config",
        default="configs/experiments_baseline.yaml",
        help="Path to the experiment YAML config.",
    )
    parser.add_argument(
        "--list-feature-sets",
        action="store_true",
        help="Print declared feature sets and exit.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.list_feature_sets:
        for fs in available_feature_sets():
            print(f"# {fs.name}")
            print(f"  description: {fs.description}")
            print(f"  columns: {list(fs.columns)}")
            print(f"  indicator_columns: {list(fs.indicator_columns)}")
        return

    config, resolved_path = load_experiment_config(args.config)
    print(f"Loaded experiment config: {resolved_path}")
    payload = run_experiments(config)
    _print_summary(payload)


if __name__ == "__main__":
    main()
