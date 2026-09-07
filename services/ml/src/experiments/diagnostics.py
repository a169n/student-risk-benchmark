"""Mastery validation diagnostics.

Helpers used by ``run_mastery_validation`` to answer whether the mastery
block (``current_topic_mastery``, ``overall_mastery``) is genuinely useful
beyond the ``B_lms`` baseline or whether it is too directly aligned with the
end-of-course ``final_grade`` target.

The functions here are deliberately small. They wrap pandas / scikit-learn
calls so the runner stays declarative and the validation logic can be unit
tested without driving the full experiment pipeline.

Outputs are JSON-friendly dicts: every numeric metric is plain ``float`` and
identifiers are plain strings. ``NaN`` values produced by edge cases (constant
columns, single-class targets, fewer rows than features) are coerced to
``None`` so the diagnostics file remains valid JSON without ``NaN`` literals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.experiments.datasets import (
    GROUP_COLUMN,
    REGRESSION_TARGET,
    WEEK_COLUMN,
)
from src.experiments.evaluate import compute_regression_metrics
from src.experiments.featuresets import FeatureSet, get_feature_set
from src.experiments.models import build_regression_model
from src.experiments.preprocessing import (
    build_modeling_matrix,
    fit_imputer_on_training,
    select_rows,
)
from src.experiments.splits import student_group_split


def _none_if_nan(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def _safe_pearson(x: pd.Series, y: pd.Series) -> float | None:
    aligned = pd.concat([x, y], axis=1).dropna()
    if aligned.shape[0] < 3:
        return None
    if aligned.iloc[:, 0].nunique() < 2 or aligned.iloc[:, 1].nunique() < 2:
        return None
    return _none_if_nan(float(aligned.corr().iloc[0, 1]))


def _student_group_indices(
    matrix_groups: pd.Series,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute student-grouped train/test row indices aligned to the matrix."""

    base = pd.DataFrame(
        {
            GROUP_COLUMN: matrix_groups.values,
            "_row_index": np.arange(matrix_groups.shape[0]),
        }
    )
    split = student_group_split(
        base,
        test_size=test_size,
        seed=seed,
        student_id_column=GROUP_COLUMN,
    )
    return (
        split.train["_row_index"].to_numpy(),
        split.test["_row_index"].to_numpy(),
    )


def compute_mastery_target_correlations(
    frame: pd.DataFrame,
    *,
    mastery_columns: Sequence[str],
    target_column: str = REGRESSION_TARGET,
    week_column: str = WEEK_COLUMN,
    weekly_cutoffs: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Pearson correlations of mastery features with the regression target.

    Returns global correlations and per-week correlations so the experiment
    can flag features that look near-deterministic for the outcome only late
    in the course.
    """

    available = [column for column in mastery_columns if column in frame.columns]
    global_corrs = {
        column: _safe_pearson(frame[column], frame[target_column]) for column in available
    }
    weekly: dict[str, dict[int, float | None]] = {column: {} for column in available}
    if weekly_cutoffs:
        for week in sorted(set(weekly_cutoffs)):
            week_frame = frame.loc[frame[week_column] == week]
            for column in available:
                weekly[column][int(week)] = _safe_pearson(
                    week_frame[column], week_frame[target_column]
                )
    return {
        "target_column": target_column,
        "mastery_columns": list(available),
        "global_pearson": global_corrs,
        "weekly_pearson": {
            column: {str(week): value for week, value in weeks.items()}
            for column, weeks in weekly.items()
        },
    }


def compute_redundancy_with_lms(
    frame: pd.DataFrame,
    *,
    mastery_columns: Sequence[str],
    lms_feature_set: FeatureSet,
) -> dict[str, dict[str, float | None]]:
    """Pearson correlations between mastery features and LMS baseline features.

    A mastery feature whose maximum correlation with an LMS baseline column is
    extremely high indicates that the mastery block is a near-duplicate of an
    existing baseline signal and not a distinct Twin contribution.
    """

    lms_columns = [column for column in lms_feature_set.columns if column in frame.columns]
    table: dict[str, dict[str, float | None]] = {}
    for column in mastery_columns:
        if column not in frame.columns:
            table[column] = {}
            continue
        row: dict[str, float | None] = {}
        for lms_column in lms_columns:
            row[lms_column] = _safe_pearson(frame[column], frame[lms_column])
        table[column] = row
    return table


def compute_lms_target_correlations(
    frame: pd.DataFrame,
    *,
    lms_feature_set: FeatureSet,
    target_column: str = REGRESSION_TARGET,
) -> dict[str, float | None]:
    """Pearson correlations of LMS baseline features with the regression target.

    Used to contextualize how target-correlated mastery is relative to the
    baseline. On a synthetic dataset the strongest LMS feature can already
    correlate strongly with the outcome; the meaningful diagnostic is whether
    mastery is *more* correlated than the strongest LMS feature.
    """

    return {
        column: _safe_pearson(frame[column], frame[target_column])
        for column in lms_feature_set.columns
        if column in frame.columns
    }


def _max_abs_correlation(
    redundancy: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, float | str | None]]:
    """For each mastery column, return its strongest LMS correlate."""

    output: dict[str, dict[str, float | str | None]] = {}
    for column, lms_corrs in redundancy.items():
        best_lms: str | None = None
        best_value: float | None = None
        for lms_column, value in lms_corrs.items():
            if value is None:
                continue
            if best_value is None or abs(value) > abs(best_value):
                best_lms = lms_column
                best_value = value
        output[column] = {
            "max_abs_pearson": None if best_value is None else float(abs(best_value)),
            "signed_pearson": best_value,
            "lms_column": best_lms,
        }
    return output


@dataclass(frozen=True)
class _RegressionScore:
    rmse: float
    mae: float
    r2: float
    n_train_rows: int
    n_test_rows: int


def _score_regression(
    matrix,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_columns: Sequence[str] | None,
    model_name: str,
    seed: int,
) -> _RegressionScore | None:
    train_mask = np.zeros(matrix.features.shape[0], dtype=bool)
    test_mask = np.zeros(matrix.features.shape[0], dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    train_matrix = select_rows(matrix, train_mask)
    test_matrix = select_rows(matrix, test_mask)
    if train_matrix.features.empty or test_matrix.features.empty:
        return None

    imputer = fit_imputer_on_training(train_matrix.features)
    x_train = imputer.transform(train_matrix.features)
    x_test = imputer.transform(test_matrix.features)
    if feature_columns is not None:
        keep = [column for column in feature_columns if column in x_train.columns]
        if not keep:
            return None
        x_train = x_train.loc[:, keep]
        x_test = x_test.loc[:, keep]

    estimator = build_regression_model(model_name, seed=seed)
    estimator.fit(x_train, train_matrix.regression_target.to_numpy())
    y_pred = estimator.predict(x_test)
    y_test = test_matrix.regression_target.to_numpy()
    metrics = compute_regression_metrics(y_test, y_pred)
    return _RegressionScore(
        rmse=float(metrics["rmse"]),
        mae=float(metrics["mae"]),
        r2=float(metrics["r2"]),
        n_train_rows=int(x_train.shape[0]),
        n_test_rows=int(x_test.shape[0]),
    )


def run_drop_column_tests(
    frame: pd.DataFrame,
    *,
    candidate_feature_set_name: str,
    mastery_columns: Sequence[str],
    model_name: str,
    seed: int,
    test_size: float,
    split_seed: int,
) -> dict[str, Any]:
    """Train the candidate model with and without each mastery feature.

    The reported delta is the increase in test-set RMSE that results from
    dropping the feature. A large positive delta means the candidate model
    relies heavily on that single feature; a near-zero delta means the
    feature is redundant given the rest of the candidate set.
    """

    candidate_set = get_feature_set(candidate_feature_set_name)
    matrix = build_modeling_matrix(frame, candidate_set)
    train_idx, test_idx = _student_group_indices(
        matrix.groups, test_size=test_size, seed=split_seed
    )
    full = _score_regression(
        matrix,
        train_idx=train_idx,
        test_idx=test_idx,
        feature_columns=None,
        model_name=model_name,
        seed=seed,
    )
    if full is None:
        return {
            "feature_set": candidate_feature_set_name,
            "model": model_name,
            "full_rmse": None,
            "drop_results": [],
        }

    drop_results: list[dict[str, Any]] = []
    for column in mastery_columns:
        if column not in matrix.feature_columns:
            drop_results.append(
                {
                    "dropped_column": column,
                    "rmse": None,
                    "delta_rmse_vs_full": None,
                    "skipped_reason": "column not present in feature set",
                }
            )
            continue
        kept_columns = [
            existing for existing in matrix.feature_columns if existing != column
        ]
        scored = _score_regression(
            matrix,
            train_idx=train_idx,
            test_idx=test_idx,
            feature_columns=kept_columns,
            model_name=model_name,
            seed=seed,
        )
        if scored is None:
            drop_results.append(
                {
                    "dropped_column": column,
                    "rmse": None,
                    "delta_rmse_vs_full": None,
                    "skipped_reason": "scoring failed",
                }
            )
            continue
        drop_results.append(
            {
                "dropped_column": column,
                "rmse": scored.rmse,
                "mae": scored.mae,
                "r2": scored.r2,
                "delta_rmse_vs_full": float(scored.rmse - full.rmse),
            }
        )
    return {
        "feature_set": candidate_feature_set_name,
        "model": model_name,
        "split_strategy": "student_group",
        "full_rmse": full.rmse,
        "full_mae": full.mae,
        "full_r2": full.r2,
        "n_train_rows": full.n_train_rows,
        "n_test_rows": full.n_test_rows,
        "drop_results": drop_results,
    }


def compute_permutation_importance(
    frame: pd.DataFrame,
    *,
    candidate_feature_set_name: str,
    model_name: str,
    seed: int,
    test_size: float,
    split_seed: int,
    n_repeats: int = 10,
) -> dict[str, Any]:
    """Permutation importance for the candidate model on the held-out split.

    Reported as the mean RMSE increase induced by shuffling each feature in
    the test partition. Useful as a complementary view to the drop-column
    test: drop-column re-trains, permutation only re-evaluates.
    """

    candidate_set = get_feature_set(candidate_feature_set_name)
    matrix = build_modeling_matrix(frame, candidate_set)
    train_idx, test_idx = _student_group_indices(
        matrix.groups, test_size=test_size, seed=split_seed
    )
    train_mask = np.zeros(matrix.features.shape[0], dtype=bool)
    test_mask = np.zeros(matrix.features.shape[0], dtype=bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    train_matrix = select_rows(matrix, train_mask)
    test_matrix = select_rows(matrix, test_mask)
    if train_matrix.features.empty or test_matrix.features.empty:
        return {
            "feature_set": candidate_feature_set_name,
            "model": model_name,
            "importances": [],
        }

    imputer = fit_imputer_on_training(train_matrix.features)
    x_train = imputer.transform(train_matrix.features)
    x_test = imputer.transform(test_matrix.features)
    estimator = build_regression_model(model_name, seed=seed)
    estimator.fit(x_train, train_matrix.regression_target.to_numpy())
    result = permutation_importance(
        estimator,
        x_test,
        test_matrix.regression_target.to_numpy(),
        n_repeats=n_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    importances: list[dict[str, Any]] = []
    for index, column in enumerate(x_test.columns):
        importances.append(
            {
                "feature": column,
                "mean_rmse_increase": _none_if_nan(float(-result.importances_mean[index])),
                "std_rmse_increase": _none_if_nan(float(result.importances_std[index])),
            }
        )
    importances.sort(
        key=lambda record: (
            -(record["mean_rmse_increase"] or float("-inf")),
            record["feature"],
        )
    )
    return {
        "feature_set": candidate_feature_set_name,
        "model": model_name,
        "importances": importances,
    }


def run_weekly_validation(
    frame: pd.DataFrame,
    *,
    feature_sets: Iterable[str],
    weekly_cutoffs: Sequence[int],
    model_names: Iterable[str],
    seed: int,
    test_size: float,
    split_seed: int,
) -> list[dict[str, Any]]:
    """For each (feature set, model, week), score on snapshots restricted to that week.

    The week-only restriction answers whether mastery becomes useful early in
    the course or only after most of the course has elapsed. The split is
    still student-grouped so it does not leak student identity across train
    and test partitions.
    """

    feature_set_objects = [get_feature_set(name) for name in feature_sets]
    rows: list[dict[str, Any]] = []
    for cutoff in sorted(set(weekly_cutoffs)):
        week_frame = frame.loc[frame[WEEK_COLUMN] == cutoff].copy()
        if week_frame.empty:
            continue
        for feature_set in feature_set_objects:
            matrix = build_modeling_matrix(week_frame, feature_set)
            if matrix.features.shape[0] < 4:
                continue
            train_idx, test_idx = _student_group_indices(
                matrix.groups, test_size=test_size, seed=split_seed
            )
            if train_idx.size == 0 or test_idx.size == 0:
                continue
            for model_name in model_names:
                scored = _score_regression(
                    matrix,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    feature_columns=None,
                    model_name=model_name,
                    seed=seed,
                )
                if scored is None:
                    continue
                rows.append(
                    {
                        "week": int(cutoff),
                        "feature_set": feature_set.name,
                        "model": model_name,
                        "rmse": scored.rmse,
                        "mae": scored.mae,
                        "r2": scored.r2,
                        "n_train_rows": scored.n_train_rows,
                        "n_test_rows": scored.n_test_rows,
                    }
                )
    return rows


def summarize_weekly_delta(
    weekly_rows: list[dict[str, Any]],
    *,
    baseline_feature_set: str,
    candidate_feature_set: str,
    early_week_max: int,
    improvement_rmse_tolerance: float,
) -> dict[str, Any]:
    """Aggregate the week-aware records into a baseline-vs-candidate delta table.

    For every week where both feature sets produced a score, pick the best
    model per feature set, then report ``delta_rmse_vs_baseline`` as
    ``candidate_rmse - baseline_rmse``. Negative values mean the candidate
    improved on the baseline.
    """

    if not weekly_rows:
        return {
            "rows": [],
            "early_weeks_improved": [],
            "late_weeks_improved": [],
            "weeks_examined": [],
        }
    table = pd.DataFrame(weekly_rows)
    relevant = table.loc[
        table["feature_set"].isin([baseline_feature_set, candidate_feature_set])
    ].copy()
    if relevant.empty:
        return {
            "rows": [],
            "early_weeks_improved": [],
            "late_weeks_improved": [],
            "weeks_examined": [],
        }

    best = (
        relevant.sort_values(["rmse", "mae"], ascending=[True, True])
        .groupby(["week", "feature_set"], sort=True)
        .first()
        .reset_index()
    )
    pivot = best.pivot(index="week", columns="feature_set", values="rmse")
    rows: list[dict[str, Any]] = []
    early_improved: list[int] = []
    late_improved: list[int] = []
    for week, scores in pivot.iterrows():
        baseline_rmse = scores.get(baseline_feature_set)
        candidate_rmse = scores.get(candidate_feature_set)
        if pd.isna(baseline_rmse) or pd.isna(candidate_rmse):
            continue
        delta = float(candidate_rmse - baseline_rmse)
        improved = delta < -improvement_rmse_tolerance
        record = {
            "week": int(week),
            "baseline_feature_set": baseline_feature_set,
            "candidate_feature_set": candidate_feature_set,
            "baseline_rmse": float(baseline_rmse),
            "candidate_rmse": float(candidate_rmse),
            "delta_rmse_vs_baseline": delta,
            "candidate_improves_baseline": bool(improved),
        }
        rows.append(record)
        if improved:
            if int(week) <= early_week_max:
                early_improved.append(int(week))
            else:
                late_improved.append(int(week))
    return {
        "rows": rows,
        "early_weeks_improved": sorted(early_improved),
        "late_weeks_improved": sorted(late_improved),
        "weeks_examined": [int(week) for week in pivot.index.tolist()],
        "early_week_max": int(early_week_max),
        "improvement_rmse_tolerance": float(improvement_rmse_tolerance),
    }


def derive_carry_forward_recommendation(
    *,
    overall_delta_rmse: float | None,
    weekly_summary: dict[str, Any],
    target_correlations: dict[str, Any],
    lms_target_correlations: dict[str, float | None] | None,
    redundancy_summary: dict[str, dict[str, float | str | None]],
    drop_column: dict[str, Any],
    candidate_feature_set: str,
    baseline_feature_set: str,
    improvement_rmse_tolerance: float,
    target_proxy_correlation_warn: float,
    redundancy_correlation_warn: float,
    target_proxy_excess_margin: float = 0.02,
) -> dict[str, Any]:
    """Produce one explicit decision based on the diagnostics."""

    mastery_max_target = _max_abs_target_correlation(target_correlations)
    lms_max_target: float | None = None
    lms_max_target_column: str | None = None
    if lms_target_correlations:
        for column, value in lms_target_correlations.items():
            if value is None:
                continue
            magnitude = abs(float(value))
            if lms_max_target is None or magnitude > lms_max_target:
                lms_max_target = magnitude
                lms_max_target_column = column

    flags: list[str] = []
    too_target_like = False
    if mastery_max_target is not None and mastery_max_target >= target_proxy_correlation_warn:
        absolute_excess = (
            mastery_max_target
            if lms_max_target is None
            else mastery_max_target - lms_max_target
        )
        if absolute_excess >= target_proxy_excess_margin:
            too_target_like = True
            flags.append(
                "mastery is more target-correlated than the LMS baseline "
                f"(mastery max |r|={mastery_max_target:.3f}, "
                f"LMS max |r|={(lms_max_target if lms_max_target is not None else float('nan')):.3f}, "
                f"excess >= {target_proxy_excess_margin}); thresholds: "
                f"|r| >= {target_proxy_correlation_warn}"
            )

    redundant_columns: list[str] = []
    for column, payload in redundancy_summary.items():
        max_abs = payload.get("max_abs_pearson")
        if max_abs is None:
            continue
        if float(max_abs) >= redundancy_correlation_warn:
            redundant_columns.append(column)
            flags.append(
                f"`{column}` is highly redundant with LMS feature "
                f"`{payload.get('lms_column')}` (|r|={float(max_abs):.3f})"
            )

    overall_improves = (
        overall_delta_rmse is not None
        and overall_delta_rmse < -improvement_rmse_tolerance
    )
    early_weeks = weekly_summary.get("early_weeks_improved", []) or []
    late_weeks = weekly_summary.get("late_weeks_improved", []) or []
    early_support = bool(early_weeks)

    drop_results = drop_column.get("drop_results", []) or []
    dominant_drop: tuple[str, float] | None = None
    for record in drop_results:
        delta = record.get("delta_rmse_vs_full")
        if delta is None:
            continue
        if dominant_drop is None or float(delta) > dominant_drop[1]:
            dominant_drop = (record["dropped_column"], float(delta))
    drop_dominated = (
        dominant_drop is not None
        and dominant_drop[1] > max(0.10, 2.0 * improvement_rmse_tolerance)
    )
    if drop_dominated and dominant_drop is not None:
        flags.append(
            f"dropping `{dominant_drop[0]}` increases RMSE by {dominant_drop[1]:+.3f}, "
            "indicating the candidate model leans heavily on a single mastery feature"
        )

    if too_target_like:
        outcome = "do_not_carry_forward"
        decision_text = (
            f"Mastery is too target-like in the current synthetic data. Do not "
            f"carry `{candidate_feature_set}` into XAI without a generator or "
            "feature revision that reduces direct alignment with `final_grade`."
        )
    elif overall_improves and early_support:
        outcome = "carry_forward"
        decision_text = (
            f"Carry `{candidate_feature_set}` into XAI as the lean Twin "
            f"candidate. It improves on `{baseline_feature_set}` overall and in "
            "at least one early-week cutoff, supporting the early-warning story."
        )
    elif overall_improves and not early_support and late_weeks:
        outcome = "carry_forward_with_caveat"
        decision_text = (
            f"Carry `{candidate_feature_set}` into XAI but flag that the gains "
            "concentrate in late weeks; mastery is not yet supporting the "
            "early-warning narrative."
        )
    elif redundant_columns and not overall_improves:
        outcome = "narrow_or_drop"
        decision_text = (
            "Mastery does not improve over the LMS baseline and at least one "
            "mastery column is highly redundant with an LMS feature. Drop the "
            "redundant column or do not carry mastery forward without revising "
            "the generator."
        )
    else:
        outcome = "inconclusive"
        decision_text = (
            f"The diagnostics do not clearly support `{candidate_feature_set}` "
            f"over `{baseline_feature_set}`. Treat the carry-forward decision "
            "as inconclusive on the current synthetic dataset."
        )

    return {
        "outcome": outcome,
        "decision_text": decision_text,
        "flags": flags,
        "candidate_feature_set": candidate_feature_set,
        "baseline_feature_set": baseline_feature_set,
        "overall_delta_rmse_vs_baseline": (
            None if overall_delta_rmse is None else float(overall_delta_rmse)
        ),
        "early_weeks_improved": list(early_weeks),
        "late_weeks_improved": list(late_weeks),
        "redundant_mastery_columns": redundant_columns,
        "max_target_correlation": mastery_max_target,
        "max_lms_target_correlation": lms_max_target,
        "max_lms_target_correlation_column": lms_max_target_column,
        "dominant_drop_column": (
            None
            if dominant_drop is None
            else {"column": dominant_drop[0], "rmse_increase": dominant_drop[1]}
        ),
        "thresholds": {
            "improvement_rmse_tolerance": improvement_rmse_tolerance,
            "target_proxy_correlation_warn": target_proxy_correlation_warn,
            "target_proxy_excess_margin": target_proxy_excess_margin,
            "redundancy_correlation_warn": redundancy_correlation_warn,
        },
    }


def _max_abs_target_correlation(target_correlations: dict[str, Any]) -> float | None:
    values = [
        abs(float(value))
        for value in target_correlations.get("global_pearson", {}).values()
        if value is not None
    ]
    return max(values) if values else None


def build_lineage_documentation() -> dict[str, Any]:
    """Document the source-field lineage of mastery features.

    The lineage is read from the generator code at
    ``services/ml/src/generator/snapshots.py`` and recorded here so the
    experiment markdown and metadata can show whether mastery is constructed
    only from information available up to the snapshot week.
    """

    return {
        "current_topic_mastery": {
            "source_fields": [
                "submissions.normalized_score (filtered to current topic and week<=snapshot week)",
                "fallback: avg_quiz_score_to_date / avg_assignment_score_to_date",
            ],
            "uses_future_information": False,
            "uses_final_outcome_fields": False,
            "notes": (
                "Computed from cumulative submission scores filtered to "
                "`week_number <= snapshot week`. Falls back to other "
                "snapshot-time averages when the current topic has no "
                "submissions yet."
            ),
        },
        "overall_mastery": {
            "source_fields": [
                "submissions.normalized_score (filtered to week<=snapshot week, grouped per topic)",
                "fallback: current_topic_mastery",
            ],
            "uses_future_information": False,
            "uses_final_outcome_fields": False,
            "notes": (
                "Mean of per-topic mean scores using only submissions up to "
                "the snapshot week. Does not read `final_results.final_grade` "
                "or any post-course field."
            ),
        },
        "verification": (
            "Lineage is documented from the generator code, not enforced by "
            "an automated lineage check. See "
            "`services/ml/src/generator/snapshots.py` for the construction."
        ),
    }
