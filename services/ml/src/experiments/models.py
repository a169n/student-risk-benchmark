"""Baseline model factories for the dissertation experiments.

Hyperparameters are intentionally modest. The first round of experiments is
about comparing feature sets and model families, not chasing leaderboard
scores. Logistic regression and linear regression include a `StandardScaler`
because they are scale-sensitive; the tree-based families do not need it.
"""

from __future__ import annotations

from typing import Iterable

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.config import ClassificationModelName, RegressionModelName


def _scaled(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


def build_classification_model(name: ClassificationModelName, *, seed: int):
    if name == "logistic_regression":
        return _scaled(
            LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                C=1.0,
                random_state=seed,
            )
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        )
    raise ValueError(f"Unknown classification model: {name}")


def build_regression_model(name: RegressionModelName, *, seed: int):
    if name == "linear_regression":
        # Ridge with very weak regularization stays close to plain OLS but
        # behaves better in the presence of correlated twin features.
        return _scaled(Ridge(alpha=1.0, random_state=seed))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=seed,
        )
    raise ValueError(f"Unknown regression model: {name}")


def iter_classification_models(
    names: Iterable[ClassificationModelName], *, seed: int
):
    for name in names:
        yield name, build_classification_model(name, seed=seed)


def iter_regression_models(
    names: Iterable[RegressionModelName], *, seed: int
):
    for name in names:
        yield name, build_regression_model(name, seed=seed)
