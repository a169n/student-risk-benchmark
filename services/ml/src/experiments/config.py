"""Experiment configuration loading.

Defines the YAML-backed configuration used by `run_baselines.py` and friends so
that paths, splits, model lists, and feature-set selections are not hardcoded
inside scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

from src.generator.config import REPO_ROOT, SERVICE_ROOT

SplitStrategy = Literal["student_group", "temporal_forward"]
FeatureSetName = Literal[
    "A_simple",
    "B_lms",
    "B_lms_plus_trends",
    "B_lms_plus_mastery",
    "B_lms_plus_indices",
    "B_lms_plus_temporal",
    "B_lms_plus_trends_mastery",
    "C_twin",
    "C_twin_full",
]
ClassificationModelName = Literal["logistic_regression", "random_forest", "gradient_boosting"]
RegressionModelName = Literal["linear_regression", "random_forest", "gradient_boosting"]


class DatasetSourceConfig(BaseModel):
    """Where the modeling pipeline reads its inputs from."""

    snapshots_csv: Path = Path("data/processed/student_twin_snapshots.csv")
    final_results_csv: Path = Path("data/raw/final_results.csv")
    students_csv: Path = Path("data/raw/students.csv")
    courses_csv: Path = Path("data/raw/courses.csv")


class StudentGroupSplitConfig(BaseModel):
    test_size: float = 0.25
    validation_size: float = 0.0
    seed: int = 42


class TemporalForwardSplitConfig(BaseModel):
    train_weeks: int = 6
    student_test_size: float = 0.25
    student_seed: int = 42


class SplitsConfig(BaseModel):
    primary: SplitStrategy = "student_group"
    secondary: SplitStrategy | None = "temporal_forward"
    student_group: StudentGroupSplitConfig = Field(default_factory=StudentGroupSplitConfig)
    temporal_forward: TemporalForwardSplitConfig = Field(default_factory=TemporalForwardSplitConfig)


class SnapshotFilterConfig(BaseModel):
    """Which weekly snapshots to use for training."""

    min_week: int = 4
    max_week: int | None = None


class EDAConfig(BaseModel):
    enabled: bool = True
    correlation_top_k: int = 12


class OutputConfig(BaseModel):
    eda_dir: Path = Path("data/artifacts/eda")
    experiments_dir: Path = Path("data/artifacts/experiments/baselines")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    name: str = "baseline_v1"
    seed: int = 42
    feature_sets: list[FeatureSetName] = Field(
        default_factory=lambda: ["A_simple", "B_lms", "C_twin"]
    )
    classification_models: list[ClassificationModelName] = Field(
        default_factory=lambda: ["logistic_regression", "random_forest", "gradient_boosting"]
    )
    regression_models: list[RegressionModelName] = Field(
        default_factory=lambda: ["linear_regression", "random_forest", "gradient_boosting"]
    )
    dataset: DatasetSourceConfig = Field(default_factory=DatasetSourceConfig)
    snapshot_filter: SnapshotFilterConfig = Field(default_factory=SnapshotFilterConfig)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    eda: EDAConfig = Field(default_factory=EDAConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def _validate(self) -> "ExperimentConfig":
        if not self.feature_sets:
            raise ValueError("feature_sets must not be empty")
        if not self.classification_models and not self.regression_models:
            raise ValueError("at least one model list must be non-empty")
        if self.snapshot_filter.min_week < 1:
            raise ValueError("snapshot_filter.min_week must be >= 1")
        return self

    def resolve_path(self, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (REPO_ROOT / path).resolve()


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend([Path.cwd() / path, SERVICE_ROOT / path, REPO_ROOT / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve experiment config path: {config_path}")


def load_experiment_config(config_path: str | Path) -> tuple[ExperimentConfig, Path]:
    resolved = _resolve_config_path(config_path)
    raw_payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw_payload), resolved
