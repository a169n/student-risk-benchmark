"""Explainability helpers for controlled experiment runs.

The utilities here deliberately avoid turning XAI into a second modeling
benchmark. They fit one configured regression model with the same student-level
split and train-only imputation used by the baseline experiments, then produce
auditable fallback explanations:

1. global permutation importance on the held-out student split,
2. model-native importance when the estimator exposes it,
3. local one-feature-at-a-time perturbation against the training median.

These are not causal explanations and they are not SHAP values. They are a
small, deterministic explanation layer suitable for the current research
prototype while SHAP is not part of the project dependency contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.experiments.datasets import GROUP_COLUMN, WEEK_COLUMN
from src.experiments.evaluate import compute_regression_metrics
from src.experiments.featuresets import get_feature_set
from src.experiments.models import build_regression_model
from src.experiments.preprocessing import (
    FittedImputer,
    build_modeling_matrix,
    fit_imputer_on_training,
    select_rows,
)
from src.experiments.splits import student_group_split


@dataclass(frozen=True)
class TrainedRegressionModel:
    """A fitted reference model plus the exact held-out matrix used for XAI."""

    feature_set_name: str
    model_name: str
    estimator: Any
    imputer: FittedImputer
    feature_columns: tuple[str, ...]
    train_mask: np.ndarray
    test_mask: np.ndarray
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    predictions: np.ndarray
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame
    train_source_indices: np.ndarray
    test_source_indices: np.ndarray
    metrics: dict[str, float]
    split_metadata: dict[str, Any]
    dropped_columns: tuple[str, ...] = ()


def _assemble_trained_model(
    working_frame: pd.DataFrame,
    matrix: Any,
    *,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    feature_set_name: str,
    model_name: str,
    seed: int,
    drop_columns: tuple[str, ...],
    split_metadata: dict[str, Any],
) -> TrainedRegressionModel:
    """Shared core: select rows, impute, train, evaluate, and return a TrainedRegressionModel.

    Both ``train_regression_reference`` and ``train_regression_reference_for_matrix``
    delegate to this function after computing their respective masks and matrix.
    The ``working_frame`` must be a reset-indexed copy of the original frame.
    """

    train_matrix = select_rows(matrix, train_mask)
    test_matrix = select_rows(matrix, test_mask)
    if train_matrix.features.empty or test_matrix.features.empty:
        raise ValueError("Cannot train XAI reference model on an empty split")

    imputer = fit_imputer_on_training(train_matrix.features)
    x_train = imputer.transform(train_matrix.features)
    x_test = imputer.transform(test_matrix.features)

    dropped = drop_columns
    if dropped:
        keep_columns = [column for column in x_train.columns if column not in set(dropped)]
        if not keep_columns:
            raise ValueError("drop_columns removed every feature from the model matrix")
        x_train = x_train.loc[:, keep_columns]
        x_test = x_test.loc[:, keep_columns]

    estimator = build_regression_model(model_name, seed=seed)
    y_train = train_matrix.regression_target.to_numpy()
    y_test = test_matrix.regression_target.to_numpy()
    estimator.fit(x_train, y_train)
    predictions = estimator.predict(x_test)
    metrics = compute_regression_metrics(y_test, predictions)

    train_source_indices = np.flatnonzero(train_mask)
    test_source_indices = np.flatnonzero(test_mask)
    train_frame = working_frame.loc[train_source_indices].reset_index(drop=True)
    test_frame = working_frame.loc[test_source_indices].reset_index(drop=True)

    return TrainedRegressionModel(
        feature_set_name=feature_set_name,
        model_name=model_name,
        estimator=estimator,
        imputer=imputer,
        feature_columns=tuple(x_train.columns),
        train_mask=train_mask,
        test_mask=test_mask,
        x_train=x_train.reset_index(drop=True),
        x_test=x_test.reset_index(drop=True),
        y_train=y_train,
        y_test=y_test,
        predictions=np.asarray(predictions, dtype=float),
        train_frame=train_frame,
        test_frame=test_frame,
        train_source_indices=train_source_indices,
        test_source_indices=test_source_indices,
        metrics={key: float(value) for key, value in metrics.items()},
        split_metadata=split_metadata,
        dropped_columns=dropped,
    )


def train_regression_reference(
    frame: pd.DataFrame,
    *,
    feature_set_name: str,
    model_name: str,
    seed: int,
    test_size: float,
    split_seed: int,
    validation_size: float = 0.0,
    drop_columns: Sequence[str] | None = None,
) -> TrainedRegressionModel:
    """Train one regression model using the existing leakage-safe protocol."""

    working_frame = frame.reset_index(drop=True).copy()
    feature_set = get_feature_set(feature_set_name)
    matrix = build_modeling_matrix(working_frame, feature_set)

    split_base = pd.DataFrame(
        {
            GROUP_COLUMN: matrix.groups.values,
            "_row_index": np.arange(matrix.features.shape[0]),
        }
    )
    split = student_group_split(
        split_base,
        test_size=test_size,
        validation_size=validation_size,
        seed=split_seed,
        student_id_column=GROUP_COLUMN,
    )
    train_indices = split.train["_row_index"].to_numpy()
    test_indices = split.test["_row_index"].to_numpy()

    train_mask = np.zeros(matrix.features.shape[0], dtype=bool)
    test_mask = np.zeros(matrix.features.shape[0], dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True

    return _assemble_trained_model(
        working_frame,
        matrix,
        train_mask=train_mask,
        test_mask=test_mask,
        feature_set_name=feature_set_name,
        model_name=model_name,
        seed=seed,
        drop_columns=tuple(drop_columns or ()),
        split_metadata=split.metadata,
    )


def train_regression_reference_for_matrix(
    frame: pd.DataFrame,
    *,
    feature_set: Any,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model_name: str,
    seed: int,
    drop_columns: Sequence[str] | None = None,
    split_metadata: dict[str, Any] | None = None,
) -> TrainedRegressionModel:
    """Train a regression reference model with an explicit feature set and split masks.

    Unlike :func:`train_regression_reference`, this function accepts any duck-typed
    feature set object (not just those in the synthetic registry) and any boolean
    row masks for the train/test split.  This is the entry-point for OULAD and
    other real-dataset runners that build their own splits externally.

    Parameters
    ----------
    frame:
        Raw modeling frame; must carry ``student_id``, ``week_number``,
        ``final_grade``, and ``passed`` columns.
    feature_set:
        Any object with ``.name``, ``.columns``, and ``.indicator_columns``
        attributes (mirrors ``BenchmarkFeatureSet`` and ``FeatureSet`` shapes).
    train_mask:
        Boolean array of length ``len(frame)`` marking training rows.
    test_mask:
        Boolean array of length ``len(frame)`` marking test rows.
    model_name:
        Key passed to :func:`build_regression_model`.
    seed:
        Random seed for the estimator.
    drop_columns:
        Optional columns to exclude from the model matrix after imputation.
    split_metadata:
        Arbitrary metadata dict stored on the returned object (e.g. split
        strategy name, temporal cutoff week).  Defaults to ``{}``.
    """

    working_frame = frame.reset_index(drop=True).copy()
    matrix = build_modeling_matrix(working_frame, feature_set)
    train_mask_bool = np.asarray(train_mask, dtype=bool)
    test_mask_bool = np.asarray(test_mask, dtype=bool)

    return _assemble_trained_model(
        working_frame,
        matrix,
        train_mask=train_mask_bool,
        test_mask=test_mask_bool,
        feature_set_name=feature_set.name,
        model_name=model_name,
        seed=seed,
        drop_columns=tuple(drop_columns or ()),
        split_metadata=split_metadata or {},
    )


def build_global_explanation(
    run: TrainedRegressionModel,
    *,
    permutation_repeats: int,
    seed: int,
    split_strategy: str = "student_group",
) -> dict[str, Any]:
    """Build a global feature-ranking table for one fitted model."""

    permutation = permutation_importance(
        run.estimator,
        run.x_test,
        run.y_test,
        n_repeats=permutation_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    native_importance = _model_native_importance(run.estimator, run.feature_columns)

    rows: list[dict[str, Any]] = []
    for index, feature in enumerate(run.feature_columns):
        mean_rmse_increase = _none_if_nan(float(permutation.importances_mean[index]))
        std_rmse_increase = _none_if_nan(float(permutation.importances_std[index]))
        native_value = native_importance.get(feature)
        direction_pred = _safe_pearson(
            run.x_test[feature],
            pd.Series(run.predictions, index=run.x_test.index),
        )
        direction_target = _safe_pearson(
            run.x_test[feature],
            pd.Series(run.y_test, index=run.x_test.index),
        )
        rows.append(
            {
                "feature_set": run.feature_set_name,
                "model": run.model_name,
                "feature": feature,
                "mean_rmse_increase": mean_rmse_increase,
                "std_rmse_increase": std_rmse_increase,
                "model_native_importance": native_value,
                "direction_pearson_with_prediction": direction_pred,
                "direction_pearson_with_target": direction_target,
                "direction_note": _direction_note(direction_pred),
                "test_mean": _none_if_nan(float(run.x_test[feature].mean())),
                "train_median": _none_if_nan(float(run.x_train[feature].median())),
            }
        )

    _attach_importance_shares(rows)
    rows.sort(
        key=lambda item: (
            -float(item.get("importance_share_basis_value") or 0.0),
            -float(item.get("mean_rmse_increase") or 0.0),
            item["feature"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    return {
        "feature_set": run.feature_set_name,
        "model": run.model_name,
        "metrics": run.metrics,
        "split_strategy": split_strategy,
        "split_metadata": run.split_metadata,
        "n_train_rows": int(run.x_train.shape[0]),
        "n_test_rows": int(run.x_test.shape[0]),
        "method": {
            "primary": "sklearn permutation_importance",
            "scoring": "neg_root_mean_squared_error",
            "importance_unit": "held-out RMSE increase when feature is permuted",
            "n_repeats": int(permutation_repeats),
            "fallback": "model-native importance and local median perturbation",
        },
        "shap": {
            "used": False,
            "reason": (
                "SHAP is not a declared project dependency for schema v1.2 "
                "experiments, so this run uses documented sklearn-based "
                "fallback explainers instead of pretending SHAP support exists."
            ),
        },
        "concentration": compute_explanation_concentration(rows),
        "rows": rows,
    }


def compute_explanation_concentration(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether importance is spread out or concentrated."""

    shares = [float(row.get("importance_share") or 0.0) for row in rows]
    if not shares:
        return {
            "top_feature": None,
            "top1_share": None,
            "top3_share": None,
            "herfindahl_index": None,
            "features_with_positive_importance": 0,
        }
    sorted_rows = sorted(
        rows,
        key=lambda item: -float(item.get("importance_share") or 0.0),
    )
    top1_share = float(shares[0]) if shares else 0.0
    top3_share = float(sum(shares[:3]))
    herfindahl = float(sum(share * share for share in shares))
    return {
        "top_feature": sorted_rows[0]["feature"],
        "top1_share": top1_share,
        "top3_share": top3_share,
        "herfindahl_index": herfindahl,
        "features_with_positive_importance": int(sum(share > 0 for share in shares)),
    }


def select_representative_cases(
    run: TrainedRegressionModel,
    *,
    case_types: Sequence[str],
    max_cases: int,
    borderline_grade: float,
    trend_feature: str,
) -> list[dict[str, Any]]:
    """Select a compact deterministic set of held-out student snapshots."""

    pool = run.test_frame.copy()
    pool["_source_row_index"] = run.test_source_indices
    pool["predicted_final_grade"] = run.predictions
    pool["actual_final_grade"] = run.y_test
    pool["_student_sort"] = pool[GROUP_COLUMN].astype(str)
    pool["_week_sort"] = pd.to_numeric(pool[WEEK_COLUMN], errors="coerce").fillna(-1)

    latest = (
        pool.sort_values([GROUP_COLUMN, WEEK_COLUMN], ascending=[True, True], kind="mergesort")
        .groupby(GROUP_COLUMN, as_index=False)
        .tail(1)
        .copy()
    )

    selected: list[dict[str, Any]] = []
    used_sources: set[int] = set()
    used_students: set[str] = set()

    for case_type in case_types:
        if len(selected) >= max_cases:
            break
        if case_type == "strong_performer":
            candidates = latest.copy()
            pick = _pick_case(
                candidates,
                used_sources=used_sources,
                used_students=used_students,
                sort_by=["actual_final_grade", "predicted_final_grade", "_student_sort"],
                ascending=[False, False, True],
            )
            rule = "latest held-out snapshot with the highest actual final_grade"
        elif case_type == "at_risk":
            candidates = latest.copy()
            pick = _pick_case(
                candidates,
                used_sources=used_sources,
                used_students=used_students,
                sort_by=["actual_final_grade", "predicted_final_grade", "_student_sort"],
                ascending=[True, True, True],
            )
            rule = "latest held-out snapshot with the lowest actual final_grade"
        elif case_type == "improving_trajectory":
            if trend_feature not in pool.columns:
                continue
            candidates = pool.loc[pool[trend_feature].notna()].copy()
            pick = _pick_case(
                candidates,
                used_sources=used_sources,
                used_students=used_students,
                sort_by=[trend_feature, "_week_sort", "_student_sort"],
                ascending=[False, False, True],
            )
            rule = f"held-out snapshot with the strongest positive {trend_feature}"
        elif case_type == "declining_trajectory":
            if trend_feature not in pool.columns:
                continue
            candidates = pool.loc[pool[trend_feature].notna()].copy()
            pick = _pick_case(
                candidates,
                used_sources=used_sources,
                used_students=used_students,
                sort_by=[trend_feature, "_week_sort", "_student_sort"],
                ascending=[True, False, True],
            )
            rule = f"held-out snapshot with the strongest negative {trend_feature}"
        elif case_type == "borderline_medium":
            candidates = latest.copy()
            candidates["_borderline_distance"] = (
                candidates["predicted_final_grade"] - float(borderline_grade)
            ).abs()
            if "risk_level" in candidates.columns:
                candidates["_risk_priority"] = np.where(
                    candidates["risk_level"].astype(str).str.lower() == "medium",
                    0,
                    1,
                )
            else:
                candidates["_risk_priority"] = 1
            pick = _pick_case(
                candidates,
                used_sources=used_sources,
                used_students=used_students,
                sort_by=["_risk_priority", "_borderline_distance", "_student_sort"],
                ascending=[True, True, True],
            )
            rule = (
                "latest held-out snapshot closest to the configured borderline "
                "grade, preferring medium risk_level when available"
            )
        else:
            continue

        if pick is None:
            continue
        source = int(pick["_source_row_index"])
        student = str(pick[GROUP_COLUMN])
        used_sources.add(source)
        used_students.add(student)
        selected.append(
            {
                "case_type": case_type,
                "selection_rule": rule,
                "source_row_index": source,
                "student_id": student,
                "week_number": int(pick[WEEK_COLUMN]),
            }
        )
    return selected


def build_local_case_explanations(
    *,
    lean_run: TrainedRegressionModel,
    baseline_run: TrainedRegressionModel,
    cases: Sequence[dict[str, Any]],
    mastery_columns: Sequence[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Build local explanations for selected held-out cases."""

    lean_positions = {
        int(source): position
        for position, source in enumerate(lean_run.test_source_indices.tolist())
    }
    baseline_positions = {
        int(source): position
        for position, source in enumerate(baseline_run.test_source_indices.tolist())
    }

    explanations: list[dict[str, Any]] = []
    for case in cases:
        source = int(case["source_row_index"])
        if source not in lean_positions:
            continue
        lean_position = lean_positions[source]
        baseline_position = baseline_positions.get(source)

        lean_contributions = compute_local_perturbation_contributions(
            lean_run,
            row_position=lean_position,
        )
        baseline_contributions = (
            compute_local_perturbation_contributions(
                baseline_run,
                row_position=baseline_position,
            )
            if baseline_position is not None
            else []
        )
        lean_top = lean_contributions[:top_k]
        baseline_top = baseline_contributions[:top_k]

        mastery_abs = sum(
            abs(float(item["contribution"]))
            for item in lean_contributions
            if item["feature"] in set(mastery_columns)
        )
        total_abs = sum(abs(float(item["contribution"])) for item in lean_contributions)
        mastery_share = None if total_abs == 0 else float(mastery_abs / total_abs)
        mastery_in_top = [
            item["feature"] for item in lean_top if item["feature"] in set(mastery_columns)
        ]

        row = lean_run.test_frame.iloc[lean_position]
        lean_prediction = float(lean_run.predictions[lean_position])
        actual = float(lean_run.y_test[lean_position])
        baseline_prediction = (
            None
            if baseline_position is None
            else float(baseline_run.predictions[baseline_position])
        )

        explanations.append(
            {
                **case,
                "actual_final_grade": actual,
                "predicted_final_grade": lean_prediction,
                "prediction_error": float(lean_prediction - actual),
                "baseline_predicted_final_grade": baseline_prediction,
                "baseline_prediction_error": (
                    None if baseline_prediction is None else float(baseline_prediction - actual)
                ),
                "passed": _coerce_optional_bool(row.get("passed")),
                "risk_level_context": row.get("risk_level") if "risk_level" in row else None,
                "lean_top_contributions": lean_top,
                "baseline_top_contributions": baseline_top,
                "mastery_features_in_lean_top": mastery_in_top,
                "mastery_abs_contribution_share": mastery_share,
                "mastery_features_are_central": bool(
                    mastery_in_top or (mastery_share is not None and mastery_share >= 0.33)
                ),
                "teacher_meaningfulness_assessment": _teacher_meaningfulness(
                    lean_top,
                    mastery_share=mastery_share,
                    mastery_columns=mastery_columns,
                ),
            }
        )
    return explanations


def compute_local_perturbation_contributions(
    run: TrainedRegressionModel,
    *,
    row_position: int,
) -> list[dict[str, Any]]:
    """Approximate local feature effects by replacing one value with its median."""

    sample = run.x_test.iloc[[row_position]].copy()
    prediction = float(run.estimator.predict(sample)[0])
    contributions: list[dict[str, Any]] = []
    for feature in run.feature_columns:
        reference_value = float(run.x_train[feature].median())
        perturbed = sample.copy()
        perturbed.loc[:, feature] = reference_value
        perturbed_prediction = float(run.estimator.predict(perturbed)[0])
        contribution = float(prediction - perturbed_prediction)
        actual_value = float(sample[feature].iloc[0])
        contributions.append(
            {
                "feature": feature,
                "value": actual_value,
                "reference_train_median": reference_value,
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "direction": (
                    "raises_prediction"
                    if contribution > 0
                    else "lowers_prediction"
                    if contribution < 0
                    else "no_local_change"
                ),
            }
        )
    contributions.sort(
        key=lambda item: (-float(item["abs_contribution"]), item["feature"])
    )
    return contributions


def compare_global_explanations(
    *,
    baseline_global: dict[str, Any],
    lean_global: dict[str, Any],
    mastery_columns: Sequence[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Compare the global explanation ranking of baseline and lean Twin models."""

    baseline_rows = baseline_global.get("rows", [])
    lean_rows = lean_global.get("rows", [])
    baseline_top = [row["feature"] for row in baseline_rows[:top_k]]
    lean_top = [row["feature"] for row in lean_rows[:top_k]]
    mastery_set = set(mastery_columns)
    mastery_payload: dict[str, Any] = {}
    for feature in mastery_columns:
        record = next((row for row in lean_rows if row["feature"] == feature), None)
        mastery_payload[feature] = (
            None
            if record is None
            else {
                "rank": record.get("rank"),
                "importance_share": record.get("importance_share"),
                "mean_rmse_increase": record.get("mean_rmse_increase"),
                "direction_note": record.get("direction_note"),
            }
        )

    return {
        "baseline_top_features": baseline_top,
        "lean_top_features": lean_top,
        "top_feature_overlap": [feature for feature in lean_top if feature in baseline_top],
        "features_added_to_top_ranking_by_mastery_model": [
            feature for feature in lean_top if feature not in baseline_top
        ],
        "mastery_features_in_lean_top": [
            feature for feature in lean_top if feature in mastery_set
        ],
        "baseline_concentration": baseline_global.get("concentration", {}),
        "lean_concentration": lean_global.get("concentration", {}),
        "mastery_feature_details": mastery_payload,
    }


def audit_overall_mastery_dominance(
    *,
    lean_global: dict[str, Any],
    lean_run: TrainedRegressionModel,
    lean_without_overall_run: TrainedRegressionModel,
    local_explanations: Sequence[dict[str, Any]],
    overall_mastery_column: str,
    importance_share_warn: float,
    top1_share_warn: float,
    local_mastery_share_warn: float,
    drop_rmse_warn: float,
) -> dict[str, Any]:
    """Decide whether `overall_mastery` dominates explanations too much."""

    rows = lean_global.get("rows", [])
    overall_row = next(
        (row for row in rows if row.get("feature") == overall_mastery_column),
        None,
    )
    importance_share = (
        None if overall_row is None else overall_row.get("importance_share")
    )
    rank = None if overall_row is None else overall_row.get("rank")
    concentration = lean_global.get("concentration", {})
    top1_share = concentration.get("top1_share")
    top_feature = concentration.get("top_feature")
    drop_delta = float(
        lean_without_overall_run.metrics["rmse"] - lean_run.metrics["rmse"]
    )
    local_shares = [
        float(item["mastery_abs_contribution_share"])
        for item in local_explanations
        if item.get("mastery_abs_contribution_share") is not None
    ]
    local_avg_share = (
        None if not local_shares else float(sum(local_shares) / len(local_shares))
    )

    flags: list[str] = []
    if importance_share is not None and float(importance_share) >= importance_share_warn:
        flags.append(
            f"`{overall_mastery_column}` accounts for "
            f"{float(importance_share):.3f} of global importance share"
        )
    if top_feature == overall_mastery_column and top1_share is not None:
        flags.append(f"`{overall_mastery_column}` is the top global explanation feature")
    if top1_share is not None and float(top1_share) >= top1_share_warn:
        flags.append(
            f"top global feature accounts for {float(top1_share):.3f} of importance share"
        )
    if local_avg_share is not None and local_avg_share >= local_mastery_share_warn:
        flags.append(
            f"local mastery contribution share averages {local_avg_share:.3f}"
        )
    if drop_delta >= drop_rmse_warn:
        flags.append(
            f"removing `{overall_mastery_column}` increases RMSE by {drop_delta:+.3f}"
        )

    if (
        importance_share is not None
        and float(importance_share) >= importance_share_warn
        and local_avg_share is not None
        and local_avg_share >= local_mastery_share_warn
    ):
        outcome = "dominates_too_much"
        interpretation = (
            "`overall_mastery` dominates both global and local explanations; "
            "the lean Twin representation should be narrowed before dissertation XAI use."
        )
    elif flags:
        outcome = "acceptable_with_caveat"
        interpretation = (
            "`overall_mastery` is influential and redundant enough to require "
            "explicit caveats, but the explanation does not collapse entirely "
            "into one mastery feature."
        )
    else:
        outcome = "acceptable"
        interpretation = (
            "`overall_mastery` does not dominate the explanation under the "
            "configured thresholds."
        )

    return {
        "overall_mastery_column": overall_mastery_column,
        "rank": rank,
        "importance_share": importance_share,
        "mean_rmse_increase": (
            None if overall_row is None else overall_row.get("mean_rmse_increase")
        ),
        "lean_rmse": lean_run.metrics["rmse"],
        "rmse_without_overall_mastery": lean_without_overall_run.metrics["rmse"],
        "delta_rmse_without_overall_mastery": drop_delta,
        "top_feature": top_feature,
        "top1_share": top1_share,
        "average_local_mastery_share": local_avg_share,
        "outcome": outcome,
        "interpretation": interpretation,
        "flags": flags,
        "thresholds": {
            "importance_share_warn": importance_share_warn,
            "top1_share_warn": top1_share_warn,
            "local_mastery_share_warn": local_mastery_share_warn,
            "drop_rmse_warn": drop_rmse_warn,
        },
    }


def _pick_case(
    candidates: pd.DataFrame,
    *,
    used_sources: set[int],
    used_students: set[str],
    sort_by: Sequence[str],
    ascending: Sequence[bool],
) -> pd.Series | None:
    if candidates.empty:
        return None
    preferred = candidates.loc[
        ~candidates["_source_row_index"].astype(int).isin(used_sources)
        & ~candidates[GROUP_COLUMN].astype(str).isin(used_students)
    ].copy()
    if preferred.empty:
        preferred = candidates.loc[
            ~candidates["_source_row_index"].astype(int).isin(used_sources)
        ].copy()
    if preferred.empty:
        return None
    ordered = preferred.sort_values(
        list(sort_by),
        ascending=list(ascending),
        kind="mergesort",
    )
    return ordered.iloc[0]


def _attach_importance_shares(rows: list[dict[str, Any]]) -> None:
    positive_values = [
        max(float(row.get("mean_rmse_increase") or 0.0), 0.0) for row in rows
    ]
    if sum(positive_values) > 0:
        basis_values = positive_values
        basis_name = "positive_permutation_rmse_increase"
    else:
        native_values = [
            abs(float(row.get("model_native_importance") or 0.0)) for row in rows
        ]
        if sum(native_values) > 0:
            basis_values = native_values
            basis_name = "absolute_model_native_importance"
        else:
            basis_values = [
                abs(float(row.get("mean_rmse_increase") or 0.0)) for row in rows
            ]
            basis_name = "absolute_permutation_rmse_change"

    total = float(sum(basis_values))
    for row, value in zip(rows, basis_values, strict=True):
        row["importance_share_basis"] = basis_name
        row["importance_share_basis_value"] = float(value)
        row["importance_share"] = None if total <= 0 else float(value / total)


def _model_native_importance(
    estimator: Any,
    columns: Sequence[str],
) -> dict[str, float | None]:
    native = getattr(estimator, "feature_importances_", None)
    if native is None and hasattr(estimator, "named_steps"):
        model = estimator.named_steps.get("model")
        native = getattr(model, "feature_importances_", None)
    if native is None:
        return {column: None for column in columns}
    return {
        column: _none_if_nan(float(native[index]))
        for index, column in enumerate(columns)
    }


def _safe_pearson(x: pd.Series, y: pd.Series) -> float | None:
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.shape[0] < 3:
        return None
    if aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return None
    return _none_if_nan(float(aligned.corr().iloc[0, 1]))


def _direction_note(correlation: float | None) -> str:
    if correlation is None:
        return "direction unclear"
    if correlation >= 0.10:
        return "higher values generally align with higher predicted final_grade"
    if correlation <= -0.10:
        return "higher values generally align with lower predicted final_grade"
    return "weak or mixed direction in held-out predictions"


def _teacher_meaningfulness(
    top_contributions: Sequence[dict[str, Any]],
    *,
    mastery_share: float | None,
    mastery_columns: Sequence[str],
) -> str:
    top_features = [item["feature"] for item in top_contributions[:3]]
    mastery_set = set(mastery_columns)
    if mastery_share is not None and mastery_share >= 0.60:
        return (
            "partly teacher-meaningful, but the local explanation is led by "
            "mastery features and needs a redundancy caveat"
        )
    if any(feature in mastery_set for feature in top_features):
        return (
            "teacher-meaningful with mastery as part of the student-state story"
        )
    return "teacher-meaningful and mostly LMS-behavior/performance driven"


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _none_if_nan(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)
