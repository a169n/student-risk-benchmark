"""Leakage-safe preprocessing for the baseline experiments.

For each feature set the modeling matrix is assembled by:

1. Selecting the configured feature columns plus their explicit missingness
   indicators (`has_assignment_score_to_date`, `has_quiz_score_to_date`).
2. Adding a synthetic missingness indicator for any other numeric feature with
   missing values (this preserves honest "no signal yet" semantics for early
   weeks rather than silently imputing the mean).
3. Filling the remaining numeric NaNs with column medians computed on the
   training rows only — never on the test rows. The fitted medians are
   re-applied when transforming new partitions.
4. Casting boolean indicators to ``int`` so they can flow into linear models
   without dtype warnings.

The module also exposes the targets and grouping keys used by the splitting
utilities so the rest of the pipeline can stay declarative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.experiments.datasets import (
    CLASSIFICATION_TARGET,
    GROUP_COLUMN,
    REGRESSION_TARGET,
    WEEK_COLUMN,
)
from src.experiments.featuresets import FORBIDDEN_FEATURE_COLUMNS, FeatureSet


@dataclass(frozen=True)
class PreparedMatrix:
    """A modeling matrix prepared for a single feature set."""

    features: pd.DataFrame
    regression_target: pd.Series
    classification_target: pd.Series
    groups: pd.Series
    weeks: pd.Series
    feature_columns: tuple[str, ...]
    indicator_columns: tuple[str, ...]


@dataclass
class FittedImputer:
    """Median-impute numeric features using statistics fit on a training split."""

    medians: dict[str, float] = field(default_factory=dict)
    added_indicator_columns: tuple[str, ...] = ()

    def fit(self, frame: pd.DataFrame, columns: tuple[str, ...]) -> "FittedImputer":
        medians: dict[str, float] = {}
        for column in columns:
            if column not in frame.columns:
                continue
            series = pd.to_numeric(frame[column], errors="coerce")
            median_value = float(series.median()) if series.notna().any() else 0.0
            medians[column] = median_value
        self.medians = medians
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column, median_value in self.medians.items():
            if column not in out.columns:
                out[column] = median_value
                continue
            numeric = pd.to_numeric(out[column], errors="coerce")
            out[column] = numeric.fillna(median_value)
        return out


def _ensure_columns_exist(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Modeling frame is missing required columns: {missing}")


def _coerce_indicator_columns(
    frame: pd.DataFrame, indicator_columns: tuple[str, ...]
) -> pd.DataFrame:
    out = frame.copy()
    truthy_map = {True: 1, False: 0, "True": 1, "False": 0}
    for column in indicator_columns:
        if column not in out.columns:
            continue
        if out[column].dtype == bool:
            out[column] = out[column].astype(int)
        else:
            out[column] = out[column].map(truthy_map).fillna(0).astype(int)
    return out


def _add_missingness_indicators(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    explicit_indicators: tuple[str, ...],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Add ``<feature>_was_missing`` flags for numeric features with NaNs.

    Explicit schema-level indicators (``has_*_to_date``) are not duplicated.
    """

    out = frame.copy()
    added: list[str] = []
    explicit_targets = {
        column.replace("has_", "").replace("_to_date", "")
        for column in explicit_indicators
    }
    for column in feature_columns:
        if column not in out.columns:
            continue
        numeric = pd.to_numeric(out[column], errors="coerce")
        if not numeric.isna().any():
            continue
        if any(target in column for target in explicit_targets):
            continue
        flag_name = f"{column}_was_missing"
        if flag_name in out.columns:
            continue
        out[flag_name] = numeric.isna().astype(int)
        added.append(flag_name)
    return out, tuple(added)


def assert_no_forbidden_columns(columns: tuple[str, ...]) -> None:
    overlap = set(columns) & FORBIDDEN_FEATURE_COLUMNS
    if overlap:
        raise ValueError(
            f"Refusing to build modeling matrix with forbidden columns: {sorted(overlap)}"
        )


def build_modeling_matrix(modeling_frame: pd.DataFrame, feature_set: FeatureSet) -> PreparedMatrix:
    """Build a leakage-safe modeling matrix for a single feature set.

    Imputation is intentionally NOT applied here. It must be fit on the
    training partition only, see :func:`fit_imputer_on_training`. Indicator
    columns and dtype coercions are applied because they are deterministic and
    do not depend on the split.
    """

    base_columns = tuple(feature_set.columns)
    indicator_columns = tuple(feature_set.indicator_columns)
    assert_no_forbidden_columns(base_columns + indicator_columns)

    required_grouping = (GROUP_COLUMN, WEEK_COLUMN, REGRESSION_TARGET, CLASSIFICATION_TARGET)
    _ensure_columns_exist(modeling_frame, required_grouping)

    available_base = tuple(column for column in base_columns if column in modeling_frame.columns)
    available_indicators = tuple(
        column for column in indicator_columns if column in modeling_frame.columns
    )
    if not available_base:
        raise ValueError(f"None of the requested feature columns exist for {feature_set.name}")

    grouping_cols = tuple(
        col
        for col in required_grouping
        if col not in available_base and col not in available_indicators
    )
    selected_columns: list[str] = []
    seen: set[str] = set()
    for column in list(available_base) + list(available_indicators) + list(grouping_cols):
        if column in seen:
            continue
        seen.add(column)
        selected_columns.append(column)
    working = modeling_frame.loc[:, selected_columns].copy()

    working = _coerce_indicator_columns(working, available_indicators)
    working, synthetic_indicators = _add_missingness_indicators(
        working, available_base, available_indicators
    )

    feature_columns = (
        tuple(available_base) + tuple(available_indicators) + tuple(synthetic_indicators)
    )

    features = working.loc[:, list(feature_columns)].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    regression_target = pd.to_numeric(working[REGRESSION_TARGET], errors="coerce")
    classification_target = working[CLASSIFICATION_TARGET].astype(bool).astype(int)
    groups = working[GROUP_COLUMN]
    weeks = working[WEEK_COLUMN].astype(int)

    return PreparedMatrix(
        features=features,
        regression_target=regression_target,
        classification_target=classification_target,
        groups=groups,
        weeks=weeks,
        feature_columns=feature_columns,
        indicator_columns=tuple(available_indicators) + tuple(synthetic_indicators),
    )


def fit_imputer_on_training(features: pd.DataFrame) -> FittedImputer:
    """Fit a median imputer on a training partition of the feature matrix."""

    columns = tuple(features.columns)
    return FittedImputer().fit(features, columns)


def select_rows(matrix: PreparedMatrix, mask: np.ndarray | pd.Series) -> PreparedMatrix:
    """Return a sub-matrix using a boolean row mask, preserving alignment."""

    boolean_mask = np.asarray(mask).astype(bool)
    if boolean_mask.shape[0] != matrix.features.shape[0]:
        raise ValueError("Row mask length does not match the prepared matrix")

    return PreparedMatrix(
        features=matrix.features.loc[boolean_mask].reset_index(drop=True),
        regression_target=matrix.regression_target.loc[boolean_mask].reset_index(drop=True),
        classification_target=matrix.classification_target.loc[boolean_mask].reset_index(drop=True),
        groups=matrix.groups.loc[boolean_mask].reset_index(drop=True),
        weeks=matrix.weeks.loc[boolean_mask].reset_index(drop=True),
        feature_columns=matrix.feature_columns,
        indicator_columns=matrix.indicator_columns,
    )
