"""Versioned public OULAD benchmark runner.

`exp_005_public_benchmark_oulad` is an external representation-transfer
benchmark. It asks whether the lean Twin logic carried forward from the
synthetic experiments can be approximated on OULAD weekly snapshots and whether
that analogue improves over a strong OULAD LMS-style baseline.

This runner is deliberately separate from the synthetic experiment pipeline:
OULAD uses different raw tables, a derived weighted-assessment target, and
benchmark-specific feature names. The supervised target is never `risk_level`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from src.benchmarks.oulad_adapter import (
    OuladCourseFilter,
    OuladRawPaths,
    OuladSnapshotBuildResult,
    build_weekly_snapshots,
    validate_oulad_raw_files,
)
from src.experiments.config import ClassificationModelName, RegressionModelName
from src.experiments.datasets import GROUP_COLUMN, WEEK_COLUMN
from src.experiments.evaluate import (
    ResultRow,
    compute_classification_metrics,
    compute_regression_metrics,
    results_to_dataframe,
)
from src.experiments.metadata import (
    DOCS_EXPERIMENTS_DIR,
    RegistryEntry,
    ensure_new_experiment_dir,
    relative_repo_path,
    resolve_experiment_artifact_dir,
    upsert_registry_entry,
    utc_now_iso,
    validate_experiment_id,
    write_experiment_metadata,
)
from src.experiments.models import iter_classification_models, iter_regression_models
from src.experiments.preprocessing import (
    PreparedMatrix,
    build_modeling_matrix,
    fit_imputer_on_training,
    select_rows,
)
from src.experiments.splits import SplitResult, student_group_split
from src.generator.config import REPO_ROOT, SERVICE_ROOT


SplitStrategy = Literal["student_group", "temporal_forward"]


@dataclass(frozen=True)
class BenchmarkFeatureSet:
    name: str
    description: str
    columns: tuple[str, ...]
    indicator_columns: tuple[str, ...] = ()

    def all_columns(self) -> tuple[str, ...]:
        return tuple([*self.columns, *self.indicator_columns])


@dataclass(frozen=True)
class _SplitPartition:
    train_mask: np.ndarray
    test_mask: np.ndarray
    metadata: dict[str, Any]
    strategy: SplitStrategy


class ExperimentIdentityConfig(BaseModel):
    experiment_id: str
    title: str
    schema_version: str = "external_oulad_adapter_v1"
    dataset_version: str = "OULAD"
    dataset_config_name: str = "local OULAD CSV files"
    parent_experiment: str | None = "exp_004_xai_on_lean_twin"
    status: str = "completed"


class DocumentationConfig(BaseModel):
    objective: str
    hypothesis: str
    limitations: list[str] = Field(default_factory=list)
    next_step: str


class OuladFilesConfig(BaseModel):
    assessments: Path = Path("assessments.csv")
    courses: Path = Path("courses.csv")
    student_info: Path = Path("studentInfo.csv")
    student_registration: Path = Path("studentRegistration.csv")
    student_vle: Path = Path("studentVle.csv")
    vle: Path = Path("vle.csv")
    student_assessment: Path = Path("studentAssessment.csv")


class CourseFilterConfig(BaseModel):
    code_module: str | None = None
    code_presentation: str | None = None
    rationale: str | None = None


class TargetConfig(BaseModel):
    primary: str = "final_weighted_score"
    secondary: str = "passed_observed"
    excluded: list[str] = Field(default_factory=lambda: ["risk_level"])
    target_construction: str


class OuladBenchmarkConfig(BaseModel):
    raw_dir: Path = Path("datasets/oulad")
    files: OuladFilesConfig = Field(default_factory=OuladFilesConfig)
    processed_snapshots_csv: Path
    course_filter: CourseFilterConfig = Field(default_factory=CourseFilterConfig)
    min_week: int = 4
    max_week: int | None = None
    student_vle_chunk_size: int = 500_000
    targets: TargetConfig

    @model_validator(mode="after")
    def _validate(self) -> "OuladBenchmarkConfig":
        if self.min_week < 1:
            raise ValueError("oulad.min_week must be >= 1")
        if self.max_week is not None and self.max_week < self.min_week:
            raise ValueError("oulad.max_week must be >= min_week")
        return self


class FeatureSetConfig(BaseModel):
    description: str
    columns: list[str]
    indicator_columns: list[str] = Field(default_factory=list)


class ComparisonConfig(BaseModel):
    baseline_feature_set: str = "B_lms_oulad"
    candidate_feature_set: str = "B_lms_plus_mastery_oulad"
    primary_split: SplitStrategy = "student_group"
    improvement_rmse_tolerance: float = 0.05
    fixed_model: RegressionModelName = "gradient_boosting"


class StudentGroupSplitConfig(BaseModel):
    test_size: float = 0.25
    validation_size: float = 0.0
    seed: int = 42


class TemporalForwardSplitConfig(BaseModel):
    train_weeks: int = 20
    student_test_size: float = 0.25
    student_seed: int = 42


class SplitsConfig(BaseModel):
    primary: SplitStrategy = "student_group"
    secondary: SplitStrategy | None = "temporal_forward"
    student_group: StudentGroupSplitConfig = Field(default_factory=StudentGroupSplitConfig)
    temporal_forward: TemporalForwardSplitConfig = Field(
        default_factory=TemporalForwardSplitConfig
    )


class OutputsConfig(BaseModel):
    experiments_dir: Path


class PublicBenchmarkOuladConfig(BaseModel):
    experiment: ExperimentIdentityConfig
    documentation: DocumentationConfig
    oulad: OuladBenchmarkConfig
    feature_sets: dict[str, FeatureSetConfig]
    feature_set_order: list[str]
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    classification_models: list[ClassificationModelName] = Field(default_factory=list)
    regression_models: list[RegressionModelName] = Field(default_factory=list)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    seed: int = 42
    outputs: OutputsConfig

    @model_validator(mode="after")
    def _validate(self) -> "PublicBenchmarkOuladConfig":
        validate_experiment_id(self.experiment.experiment_id)
        missing = [name for name in self.feature_set_order if name not in self.feature_sets]
        if missing:
            raise ValueError(f"feature_set_order contains unknown feature sets: {missing}")
        if self.comparison.baseline_feature_set not in self.feature_sets:
            raise ValueError("comparison.baseline_feature_set is not configured")
        if self.comparison.candidate_feature_set not in self.feature_sets:
            raise ValueError("comparison.candidate_feature_set is not configured")
        if not self.regression_models and not self.classification_models:
            raise ValueError("At least one model family must be configured")
        return self

    def resolve_path(self, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (REPO_ROOT / path).resolve()

    def resolve_raw_paths(self) -> OuladRawPaths:
        raw_dir = self.resolve_path(self.oulad.raw_dir)

        def resolve_file(value: Path) -> Path:
            path = Path(value)
            if path.is_absolute():
                return path
            return raw_dir / path

        files = self.oulad.files
        return OuladRawPaths(
            assessments=resolve_file(files.assessments),
            courses=resolve_file(files.courses),
            student_info=resolve_file(files.student_info),
            student_registration=resolve_file(files.student_registration),
            student_vle=resolve_file(files.student_vle),
            vle=resolve_file(files.vle),
            student_assessment=resolve_file(files.student_assessment),
        )

    def build_feature_sets(self) -> list[BenchmarkFeatureSet]:
        return [
            BenchmarkFeatureSet(
                name=name,
                description=self.feature_sets[name].description,
                columns=tuple(self.feature_sets[name].columns),
                indicator_columns=tuple(self.feature_sets[name].indicator_columns),
            )
            for name in self.feature_set_order
        ]


def load_public_benchmark_config(
    config_path: str | Path,
) -> tuple[PublicBenchmarkOuladConfig, Path]:
    resolved = _resolve_config_path(config_path)
    raw_payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return PublicBenchmarkOuladConfig.model_validate(raw_payload), resolved


def run_public_benchmark_oulad(
    config_path: str | Path,
    *,
    allow_existing: bool = False,
    artifact_root: Path | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    config, resolved_config_path = load_public_benchmark_config(config_path)
    experiment_id = config.experiment.experiment_id
    output_dir = config.resolve_path(config.outputs.experiments_dir)
    expected_dir = resolve_experiment_artifact_dir(experiment_id, root=artifact_root)
    if output_dir != expected_dir:
        raise ValueError(
            "OULAD benchmark output directory must match the experiment ID. "
            f"Expected {expected_dir}, got {output_dir}."
        )

    ensure_new_experiment_dir(output_dir, allow_existing=allow_existing)
    raw_paths = config.resolve_raw_paths()
    validate_oulad_raw_files(raw_paths)

    processed_snapshots_path = config.resolve_path(config.oulad.processed_snapshots_csv)
    if not processed_snapshots_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError(
            "OULAD processed snapshot output must live inside the experiment "
            f"artifact directory. Got {processed_snapshots_path}."
        )

    build_result = build_weekly_snapshots(
        raw_paths,
        course_filter=OuladCourseFilter(
            code_module=config.oulad.course_filter.code_module,
            code_presentation=config.oulad.course_filter.code_presentation,
        ),
        min_week=config.oulad.min_week,
        max_week=config.oulad.max_week,
        student_vle_chunk_size=config.oulad.student_vle_chunk_size,
    )
    processed_snapshots_path.parent.mkdir(parents=True, exist_ok=True)
    build_result.snapshots.to_csv(processed_snapshots_path, index=False)

    rows = run_benchmark_models(build_result.snapshots, config)
    result_paths = write_benchmark_result_artifacts(
        rows,
        output_dir=output_dir,
        run_name=experiment_id,
        config=config,
        build_result=build_result,
        snapshots_path=processed_snapshots_path,
    )

    diagnostics = build_benchmark_diagnostics(rows, config, build_result)
    mapping_path = output_dir / "public_benchmark_mapping_summary.md"
    mapping_path.write_text(
        render_mapping_artifact(config, build_result),
        encoding="utf-8",
    )
    interpretation_path = output_dir / "public_vs_synthetic_interpretation.md"
    interpretation_path.write_text(
        render_public_vs_synthetic_interpretation(config, diagnostics),
        encoding="utf-8",
    )

    artifact_summary_path = result_paths["markdown"]
    summary_markdown = render_experiment_markdown(
        config=config,
        diagnostics=diagnostics,
        build_result=build_result,
        result_paths=result_paths,
        config_path=resolved_config_path,
        snapshots_path=processed_snapshots_path,
        mapping_path=mapping_path,
        interpretation_path=interpretation_path,
    )
    artifact_summary_path.write_text(summary_markdown, encoding="utf-8")

    experiment_docs_dir = docs_dir or DOCS_EXPERIMENTS_DIR
    docs_path = experiment_docs_dir / f"{experiment_id}.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(summary_markdown, encoding="utf-8")

    metadata = build_experiment_metadata(
        config=config,
        diagnostics=diagnostics,
        build_result=build_result,
        result_paths=result_paths,
        config_path=resolved_config_path,
        output_dir=output_dir,
        docs_path=docs_path,
        snapshots_path=processed_snapshots_path,
        mapping_path=mapping_path,
        interpretation_path=interpretation_path,
    )
    metadata_path = write_experiment_metadata(metadata, output_dir=output_dir)

    registry_path = experiment_docs_dir / "registry.md"
    upsert_registry_entry(
        registry_path,
        RegistryEntry(
            experiment_id=experiment_id,
            title=config.experiment.title,
            status=config.experiment.status,
            schema_version=config.experiment.schema_version,
            dataset_config=_registry_dataset_label(config),
            primary_target=config.oulad.targets.primary,
            artifact_dir=_markdown_relative_path(registry_path.parent, output_dir),
            doc_path=_markdown_relative_path(registry_path.parent, docs_path),
            conclusion=diagnostics["interpretation"]["short_conclusion"],
        ),
    )

    return {
        "result_rows": rows,
        "result_paths": result_paths,
        "diagnostics": diagnostics,
        "metadata_path": metadata_path,
        "docs_path": docs_path,
        "registry_path": registry_path,
        "snapshots_path": processed_snapshots_path,
        "mapping_path": mapping_path,
        "interpretation_path": interpretation_path,
    }


def run_benchmark_models(
    snapshots: pd.DataFrame,
    config: PublicBenchmarkOuladConfig,
) -> list[ResultRow]:
    modeling_frame = snapshots.copy()
    modeling_frame["final_grade"] = pd.to_numeric(
        modeling_frame[config.oulad.targets.primary], errors="coerce"
    )
    modeling_frame["passed"] = (
        pd.to_numeric(modeling_frame[config.oulad.targets.secondary], errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )
    modeling_frame = modeling_frame.loc[modeling_frame["final_grade"].notna()].copy()

    feature_sets = config.build_feature_sets()
    _validate_feature_columns(modeling_frame, feature_sets)

    rows: list[ResultRow] = []
    for feature_set in feature_sets:
        matrix = build_modeling_matrix(modeling_frame, feature_set)  # type: ignore[arg-type]
        for strategy in _select_strategies(config):
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
    return rows


def build_benchmark_diagnostics(
    rows: list[ResultRow],
    config: PublicBenchmarkOuladConfig,
    build_result: OuladSnapshotBuildResult,
) -> dict[str, Any]:
    table = results_to_dataframe(rows)
    regression_summary = _best_regression_by_feature_set(
        table,
        baseline_feature_set=config.comparison.baseline_feature_set,
        primary_split=config.comparison.primary_split,
    )
    fixed_model_summary = _fixed_model_regression_by_feature_set(
        table,
        model=config.comparison.fixed_model,
        baseline_feature_set=config.comparison.baseline_feature_set,
        primary_split=config.comparison.primary_split,
    )
    classification_summary = _best_classification_by_feature_set(table)
    interpretation = _interpret_transfer_result(regression_summary, config)
    return {
        "experiment_id": config.experiment.experiment_id,
        "primary_target": config.oulad.targets.primary,
        "secondary_target": config.oulad.targets.secondary,
        "primary_split": config.comparison.primary_split,
        "row_counts": build_result.filtered_row_counts,
        "target_summary": build_result.target_summary,
        "regression_best_by_feature_set": regression_summary,
        "regression_fixed_model_by_feature_set": fixed_model_summary,
        "fixed_model": config.comparison.fixed_model,
        "classification_best_by_feature_set": classification_summary,
        "interpretation": interpretation,
    }


def build_experiment_metadata(
    *,
    config: PublicBenchmarkOuladConfig,
    diagnostics: dict[str, Any],
    build_result: OuladSnapshotBuildResult,
    result_paths: dict[str, Path],
    config_path: Path,
    output_dir: Path,
    docs_path: Path,
    snapshots_path: Path,
    mapping_path: Path,
    interpretation_path: Path,
) -> dict[str, Any]:
    raw_paths = config.resolve_raw_paths()
    return {
        "experiment_id": config.experiment.experiment_id,
        "title": config.experiment.title,
        "created_at": utc_now_iso(),
        "schema_version": config.experiment.schema_version,
        "dataset": {
            "dataset_version": config.experiment.dataset_version,
            "dataset_config_name": config.experiment.dataset_config_name,
            "source": "Open University Learning Analytics Dataset (OULAD)",
            "raw_files": {
                name: relative_repo_path(path)
                for name, path in raw_paths.as_dict().items()
            },
            "raw_row_counts": build_result.raw_row_counts,
            "course_filter": build_result.course_filter,
            "course_filter_rationale": config.oulad.course_filter.rationale,
            "processed_snapshots_path": relative_repo_path(snapshots_path),
        },
        "targets": {
            "primary": config.oulad.targets.primary,
            "secondary": config.oulad.targets.secondary,
            "excluded": list(config.oulad.targets.excluded),
            "target_construction": config.oulad.targets.target_construction,
        },
        "split_strategies": {
            "primary": config.splits.primary,
            "secondary": config.splits.secondary,
            "student_group": config.splits.student_group.model_dump(),
            "temporal_forward": config.splits.temporal_forward.model_dump(),
        },
        "feature_sets": [
            {
                "name": feature_set.name,
                "description": feature_set.description,
                "columns": list(feature_set.columns),
                "indicator_columns": list(feature_set.indicator_columns),
            }
            for feature_set in config.build_feature_sets()
        ],
        "models": {
            "regression": list(config.regression_models),
            "classification": list(config.classification_models),
        },
        "seed": config.seed,
        "config_path": relative_repo_path(config_path),
        "output_directory": relative_repo_path(output_dir),
        "status": config.experiment.status,
        "parent_experiment": config.experiment.parent_experiment,
        "artifacts": {
            "results_csv": relative_repo_path(result_paths["csv"]),
            "results_json": relative_repo_path(result_paths["json"]),
            "results_markdown": relative_repo_path(result_paths["markdown"]),
            "processed_snapshots_csv": relative_repo_path(snapshots_path),
            "mapping_summary": relative_repo_path(mapping_path),
            "public_vs_synthetic_interpretation": relative_repo_path(
                interpretation_path
            ),
            "documentation": relative_repo_path(docs_path),
        },
        "diagnostics": diagnostics,
    }


def write_benchmark_result_artifacts(
    rows: list[ResultRow],
    *,
    output_dir: Path,
    run_name: str,
    config: PublicBenchmarkOuladConfig,
    build_result: OuladSnapshotBuildResult,
    snapshots_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_results.csv"
    json_path = output_dir / f"{run_name}_results.json"
    markdown_path = output_dir / f"{run_name}_summary.md"

    table = results_to_dataframe(rows)
    table.to_csv(csv_path, index=False)
    json_payload = {
        "run_name": run_name,
        "rows": [
            {
                **row.to_record(),
                "metrics": row.metrics,
                "split_metadata": row.split_metadata,
            }
            for row in rows
        ],
        "metadata": {
            "primary_target": config.oulad.targets.primary,
            "secondary_target": config.oulad.targets.secondary,
            "processed_snapshots_path": relative_repo_path(snapshots_path),
            "row_counts": build_result.filtered_row_counts,
            "target_summary": build_result.target_summary,
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")
    markdown_path.write_text(
        f"# {run_name}\n\nResults are written after diagnostics are rendered.\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


def render_experiment_markdown(
    *,
    config: PublicBenchmarkOuladConfig,
    diagnostics: dict[str, Any],
    build_result: OuladSnapshotBuildResult,
    result_paths: dict[str, Path],
    config_path: Path,
    snapshots_path: Path,
    mapping_path: Path,
    interpretation_path: Path,
) -> str:
    exp = config.experiment
    interp = diagnostics["interpretation"]
    lines: list[str] = [
        f"# {exp.experiment_id}: {exp.title}",
        "",
        "## Objective",
        "",
        config.documentation.objective,
        "",
        "## Hypothesis",
        "",
        config.documentation.hypothesis,
        "",
        "## Dataset / config used",
        "",
        "- Public benchmark dataset: Open University Learning Analytics Dataset (OULAD)",
        f"- Experiment config: `{relative_repo_path(config_path)}`",
        f"- Raw directory: `{relative_repo_path(config.resolve_path(config.oulad.raw_dir))}`",
        f"- Processed benchmark snapshots: `{relative_repo_path(snapshots_path)}`",
        f"- Output directory: `{relative_repo_path(config.resolve_path(config.outputs.experiments_dir))}`",
        f"- Course filter: `{build_result.course_filter}`",
        f"- Course-filter rationale: {config.oulad.course_filter.rationale}",
        f"- Snapshot weeks used: `{build_result.week_min}..{build_result.week_max}`",
        "",
        "## Exact OULAD files used",
        "",
    ]
    for name, path in config.resolve_raw_paths().as_dict().items():
        lines.append(f"- `{Path(path).name}` from `{relative_repo_path(path)}`")
    lines.extend(
        [
            "",
            "## Targets",
            "",
            f"- Primary regression target: `{config.oulad.targets.primary}`",
            f"- Secondary classification target: `{config.oulad.targets.secondary}`",
            f"- Excluded supervised target(s): `{config.oulad.targets.excluded}`",
            "",
            config.oulad.targets.target_construction,
            "",
            "Target distribution is computed at the student-course-presentation level.",
            "",
            "| target summary | value |",
            "| --- | ---: |",
        ]
    )
    target_score = build_result.target_summary["final_weighted_score"]
    for key in ["count", "mean", "std", "min", "median", "max"]:
        lines.append(f"| final_weighted_score {key} | {_format_optional(target_score[key])} |")
    lines.append(
        "| passed_observed counts | "
        f"{build_result.target_summary['passed_observed_counts']} |"
    )

    lines.extend(
        [
            "",
            "## Feature sets",
            "",
        ]
    )
    for feature_set in config.build_feature_sets():
        lines.extend(
            [
                f"### `{feature_set.name}`",
                "",
                feature_set.description,
                "",
                f"- Columns: {', '.join(f'`{column}`' for column in feature_set.columns)}",
            ]
        )
        if feature_set.indicator_columns:
            lines.append(
                "- Indicator columns: "
                + ", ".join(f"`{column}`" for column in feature_set.indicator_columns)
            )
        lines.append("")

    lines.extend(
        [
            "## Split strategies and models",
            "",
            f"- Primary split: `{config.splits.primary}`",
            f"- Secondary split: `{config.splits.secondary}`",
            (
                "- Student-group split: "
                f"`test_size={config.splits.student_group.test_size}`, "
                f"`seed={config.splits.student_group.seed}`"
            ),
            (
                "- Temporal-forward split: "
                f"`train_weeks={config.splits.temporal_forward.train_weeks}`, "
                f"`student_test_size={config.splits.temporal_forward.student_test_size}`, "
                f"`student_seed={config.splits.temporal_forward.student_seed}`"
            ),
            "- Regression models: "
            + ", ".join(f"`{name}`" for name in config.regression_models),
            "- Classification models: "
            + (
                ", ".join(f"`{name}`" for name in config.classification_models)
                if config.classification_models
                else "`none`"
            ),
            "",
            "## Row counts",
            "",
            "| item | count |",
            "| --- | ---: |",
        ]
    )
    for key, value in build_result.filtered_row_counts.items():
        lines.append(f"| {key} | {value} |")

    lines.extend(
        [
            "",
            "## Headline regression results",
            "",
            "| split | feature set | best model | RMSE | MAE | R^2 | delta vs baseline |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in diagnostics["regression_best_by_feature_set"]:
        lines.append(
            "| {split} | {fs} | {model} | {rmse:.3f} | {mae:.3f} | {r2:.3f} | {delta} |".format(
                split=record["split_strategy"],
                fs=record["feature_set"],
                model=record["model"],
                rmse=record["rmse"],
                mae=record["mae"],
                r2=record["r2"],
                delta=_format_delta(record.get("delta_vs_baseline_rmse")),
            )
        )

    if diagnostics["regression_fixed_model_by_feature_set"]:
        lines.extend(
            [
                "",
                f"## Fixed-model regression results (model = `{diagnostics['fixed_model']}`)",
                "",
                "Same model across all feature sets and both splits, to neutralize "
                "best-model-per-cell (model-flip) artifacts. "
                f"Headline split: `{diagnostics['primary_split']}`.",
                "",
                "| split | feature set | model | RMSE | MAE | R^2 | delta vs baseline |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in diagnostics["regression_fixed_model_by_feature_set"]:
            lines.append(
                "| {split} | {fs} | {model} | {rmse:.3f} | {mae:.3f} | {r2:.3f} | {delta} |".format(
                    split=record["split_strategy"],
                    fs=record["feature_set"],
                    model=record["model"],
                    rmse=record["rmse"],
                    mae=record["mae"],
                    r2=record["r2"],
                    delta=_format_delta(record.get("delta_vs_baseline_rmse")),
                )
            )

    if diagnostics["classification_best_by_feature_set"]:
        lines.extend(
            [
                "",
                "## Secondary classification results",
                "",
                "| split | feature set | best model | F1 | accuracy | ROC AUC |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for record in diagnostics["classification_best_by_feature_set"]:
            lines.append(
                "| {split} | {fs} | {model} | {f1:.3f} | {acc:.3f} | {auc} |".format(
                    split=record["split_strategy"],
                    fs=record["feature_set"],
                    model=record["model"],
                    f1=record["f1"],
                    acc=record["accuracy"],
                    auc=_format_optional(record.get("roc_auc")),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            interp["interpretation"],
            "",
            f"- Outcome: `{interp['outcome']}`",
            f"- Primary delta RMSE: `{_format_delta(interp.get('primary_delta_rmse'))}`",
            f"- Short conclusion: {interp['short_conclusion']}",
            "",
            "This is a representation-transfer stress test, not a claim that one "
            "dataset is better than another and not a claim of full external validity.",
            "",
            "## Artifact paths",
            "",
            f"- Results CSV: `{relative_repo_path(result_paths['csv'])}`",
            f"- Results JSON: `{relative_repo_path(result_paths['json'])}`",
            f"- Summary Markdown: `{relative_repo_path(result_paths['markdown'])}`",
            f"- Processed snapshots CSV: `{relative_repo_path(snapshots_path)}`",
            f"- Mapping artifact: `{relative_repo_path(mapping_path)}`",
            f"- Public-vs-synthetic interpretation: `{relative_repo_path(interpretation_path)}`",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in config.documentation.limitations:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "## Next step",
            "",
            config.documentation.next_step,
        ]
    )
    return "\n".join(lines) + "\n"


def render_mapping_artifact(
    config: PublicBenchmarkOuladConfig,
    build_result: OuladSnapshotBuildResult,
) -> str:
    lines = [
        "# Public Benchmark Mapping Summary",
        "",
        "This artifact summarizes the OULAD-to-benchmark mapping used by "
        f"`{config.experiment.experiment_id}`.",
        "",
        "## Scope",
        "",
        f"- Course filter: `{build_result.course_filter}`",
        f"- Snapshot grain: `1 row = 1 student-course presentation x 1 week`",
        f"- Weeks used after filtering: `{build_result.week_min}..{build_result.week_max}`",
        "",
        "## Mapping",
        "",
        "- LMS baseline analogue: cumulative assessment scores, submission timing, "
        "VLE click intensity, VLE activity categories, course-week progression, "
        "and registration status signals.",
        "- Lean mastery analogue: due-to-date mastery proxies that treat scheduled "
        "positive-weight assessments as the local content structure available in "
        "OULAD.",
        "- No OULAD field is treated as the synthetic `risk_level`; risk is not a "
        "supervised target in this benchmark.",
        "",
        "## Feature Sets",
        "",
    ]
    for feature_set in config.build_feature_sets():
        lines.append(f"### `{feature_set.name}`")
        lines.append("")
        lines.append(feature_set.description)
        lines.append("")
        lines.append(", ".join(f"`{column}`" for column in feature_set.all_columns()))
        lines.append("")
    return "\n".join(lines)


def render_public_vs_synthetic_interpretation(
    config: PublicBenchmarkOuladConfig,
    diagnostics: dict[str, Any],
) -> str:
    interp = diagnostics["interpretation"]
    return "\n".join(
        [
            "# Public Benchmark vs Synthetic Interpretation",
            "",
            "This document compares the direction of the OULAD representation-transfer "
            "benchmark with the earlier synthetic experiment line. It does not "
            "compare dataset quality.",
            "",
            "## Synthetic carry-forward state",
            "",
            "- The full Twin representation was not justified by `exp_001`/`exp_002`.",
            "- `B_lms_plus_mastery` was carried forward as the lean Twin candidate.",
            "- `exp_004` found the lean candidate interpretable with an "
            "`overall_mastery` redundancy caveat.",
            "",
            "## OULAD benchmark result",
            "",
            interp["interpretation"],
            "",
            "## External validity boundary",
            "",
            "The benchmark is a public-dataset stress test of representation logic. "
            "It can support, weaken, or complicate the synthetic finding, but it "
            "does not establish full external validity or institutional deployment "
            "readiness.",
            "",
            f"Outcome: `{interp['outcome']}`.",
            f"Short conclusion: {interp['short_conclusion']}",
            "",
        ]
    )


def _build_split(
    matrix: PreparedMatrix,
    strategy: SplitStrategy,
    config: PublicBenchmarkOuladConfig,
) -> _SplitPartition:
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
        held_out_split: SplitResult = student_group_split(
            base,
            test_size=params.student_test_size,
            seed=params.student_seed,
            student_id_column=GROUP_COLUMN,
        )
        held_out_students = set(held_out_split.test[GROUP_COLUMN].unique())
        train_pool = base.loc[~base[GROUP_COLUMN].isin(held_out_students)].copy()
        test_pool = base.loc[base[GROUP_COLUMN].isin(held_out_students)].copy()
        train_indices = train_pool.loc[
            train_pool[WEEK_COLUMN] <= params.train_weeks, "_row_index"
        ].to_numpy()
        test_indices = test_pool.loc[
            test_pool[WEEK_COLUMN] > params.train_weeks, "_row_index"
        ].to_numpy()
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


def _train_and_score_classification(
    matrix: PreparedMatrix,
    *,
    feature_set: BenchmarkFeatureSet,
    partition: _SplitPartition,
    config: PublicBenchmarkOuladConfig,
) -> list[ResultRow]:
    train_matrix = select_rows(matrix, partition.train_mask)
    test_matrix = select_rows(matrix, partition.test_mask)
    if train_matrix.features.empty or test_matrix.features.empty:
        return []
    if not config.classification_models:
        return []

    imputer = fit_imputer_on_training(train_matrix.features)
    x_train = imputer.transform(train_matrix.features)
    x_test = imputer.transform(test_matrix.features)
    y_train = train_matrix.classification_target.to_numpy()
    y_test = test_matrix.classification_target.to_numpy()
    if len(np.unique(y_train)) < 2:
        return []

    rows: list[ResultRow] = []
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
        rows.append(
            ResultRow(
                feature_set=feature_set.name,
                model=name,
                target=config.oulad.targets.secondary,
                task="classification",
                split_strategy=partition.strategy,
                split_metadata=partition.metadata,
                n_train_rows=int(x_train.shape[0]),
                n_test_rows=int(x_test.shape[0]),
                n_train_students=int(train_matrix.groups.nunique()),
                n_test_students=int(test_matrix.groups.nunique()),
                metrics=compute_classification_metrics(y_test, y_pred, y_proba),
            )
        )
    return rows


def _train_and_score_regression(
    matrix: PreparedMatrix,
    *,
    feature_set: BenchmarkFeatureSet,
    partition: _SplitPartition,
    config: PublicBenchmarkOuladConfig,
) -> list[ResultRow]:
    train_matrix = select_rows(matrix, partition.train_mask)
    test_matrix = select_rows(matrix, partition.test_mask)
    if train_matrix.features.empty or test_matrix.features.empty:
        return []
    if not config.regression_models:
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
        rows.append(
            ResultRow(
                feature_set=feature_set.name,
                model=name,
                target=config.oulad.targets.primary,
                task="regression",
                split_strategy=partition.strategy,
                split_metadata=partition.metadata,
                n_train_rows=int(x_train.shape[0]),
                n_test_rows=int(x_test.shape[0]),
                n_train_students=int(train_matrix.groups.nunique()),
                n_test_students=int(test_matrix.groups.nunique()),
                metrics=compute_regression_metrics(y_test, y_pred),
            )
        )
    return rows


def _regression_by_feature_set(
    table: pd.DataFrame,
    *,
    baseline_feature_set: str,
    primary_split: str,
    model: str | None = None,
) -> list[dict[str, Any]]:
    if table.empty or "task" not in table.columns:
        return []
    regression = table.loc[table["task"] == "regression"].copy()
    if regression.empty:
        return []
    if model is not None:
        regression = regression.loc[regression["model"] == model].copy()
        if regression.empty:
            return []
    regression = regression.rename(
        columns={"metric_rmse": "rmse", "metric_mae": "mae", "metric_r2": "r2"}
    )
    records: list[dict[str, Any]] = []
    for (split_strategy, feature_set), group in regression.groupby(
        ["split_strategy", "feature_set"], sort=False
    ):
        best = group.sort_values(["rmse", "mae"], ascending=[True, True]).iloc[0]
        records.append(
            {
                "split_strategy": split_strategy,
                "feature_set": feature_set,
                "model": best["model"],
                "rmse": float(best["rmse"]),
                "mae": float(best["mae"]),
                "r2": float(best["r2"]),
                "n_train_rows": int(best["n_train_rows"]),
                "n_test_rows": int(best["n_test_rows"]),
            }
        )
    baseline_by_split = {
        record["split_strategy"]: record["rmse"]
        for record in records
        if record["feature_set"] == baseline_feature_set
    }
    for record in records:
        baseline = baseline_by_split.get(record["split_strategy"])
        record["delta_vs_baseline_rmse"] = (
            None if baseline is None else float(record["rmse"] - baseline)
        )
    return sorted(
        records,
        key=lambda item: (
            0 if item["split_strategy"] == primary_split else 1,
            item["split_strategy"],
            item["rmse"],
        ),
    )


def _best_regression_by_feature_set(
    table: pd.DataFrame,
    *,
    baseline_feature_set: str,
    primary_split: str,
) -> list[dict[str, Any]]:
    return _regression_by_feature_set(
        table,
        baseline_feature_set=baseline_feature_set,
        primary_split=primary_split,
        model=None,
    )


def _fixed_model_regression_by_feature_set(
    table: pd.DataFrame,
    *,
    model: str,
    baseline_feature_set: str,
    primary_split: str,
) -> list[dict[str, Any]]:
    return _regression_by_feature_set(
        table,
        baseline_feature_set=baseline_feature_set,
        primary_split=primary_split,
        model=model,
    )


def _best_classification_by_feature_set(table: pd.DataFrame) -> list[dict[str, Any]]:
    if table.empty or "task" not in table.columns:
        return []
    classification = table.loc[table["task"] == "classification"].copy()
    if classification.empty:
        return []
    classification = classification.rename(
        columns={
            "metric_f1": "f1",
            "metric_accuracy": "accuracy",
            "metric_roc_auc": "roc_auc",
        }
    )
    records: list[dict[str, Any]] = []
    for (split_strategy, feature_set), group in classification.groupby(
        ["split_strategy", "feature_set"], sort=False
    ):
        best = group.sort_values(["f1", "accuracy"], ascending=[False, False]).iloc[0]
        records.append(
            {
                "split_strategy": split_strategy,
                "feature_set": feature_set,
                "model": best["model"],
                "f1": float(best["f1"]),
                "accuracy": float(best["accuracy"]),
                "roc_auc": float(best["roc_auc"]),
            }
        )
    return sorted(records, key=lambda item: (item["split_strategy"], item["feature_set"]))


def _interpret_transfer_result(
    regression_summary: list[dict[str, Any]],
    config: PublicBenchmarkOuladConfig,
) -> dict[str, Any]:
    comparison = config.comparison
    primary_candidate = next(
        (
            record
            for record in regression_summary
            if record["split_strategy"] == comparison.primary_split
            and record["feature_set"] == comparison.candidate_feature_set
        ),
        None,
    )
    primary_delta = (
        None
        if primary_candidate is None
        else primary_candidate.get("delta_vs_baseline_rmse")
    )
    secondary_deltas = [
        record.get("delta_vs_baseline_rmse")
        for record in regression_summary
        if record["feature_set"] == comparison.candidate_feature_set
        and record["split_strategy"] != comparison.primary_split
        and record.get("delta_vs_baseline_rmse") is not None
    ]
    secondary_improves = any(
        delta < -comparison.improvement_rmse_tolerance
        for delta in secondary_deltas
    )
    tolerance = comparison.improvement_rmse_tolerance
    if primary_delta is None:
        outcome = "inconclusive"
        short = "OULAD benchmark did not produce a primary candidate regression record."
        interpretation = (
            "The public benchmark cannot be interpreted because the primary "
            "candidate comparison is missing."
        )
    elif primary_delta < -tolerance:
        outcome = "supports_with_caveats"
        short = (
            f"`{comparison.candidate_feature_set}` improved over "
            f"`{comparison.baseline_feature_set}` on OULAD."
        )
        interpretation = (
            "The OULAD benchmark is directionally consistent with the synthetic "
            "carry-forward decision: the lean mastery analogue improves over the "
            "OULAD LMS baseline on the primary grouped split. This supports the "
            "representation-transfer logic with caveats, not full external validity."
        )
    elif abs(primary_delta) <= tolerance:
        outcome = "complicates"
        short = (
            f"`{comparison.candidate_feature_set}` was approximately level with "
            f"`{comparison.baseline_feature_set}` on OULAD."
        )
        interpretation = (
            "The OULAD benchmark complicates the synthetic finding: the mastery "
            "analogue is approximately level with the LMS baseline under the "
            "configured tolerance. This suggests that the representation may be "
            "context-sensitive rather than universally advantageous."
        )
    elif secondary_improves:
        outcome = "complicates"
        short = (
            f"`{comparison.candidate_feature_set}` did not improve the primary "
            "OULAD grouped split, but improved a secondary temporal-forward split."
        )
        interpretation = (
            "The OULAD benchmark complicates the synthetic carry-forward claim: "
            "the mastery analogue is slightly worse than the LMS baseline on the "
            "primary student-grouped split, but improves the secondary "
            "temporal-forward split. This mixed result suggests transfer "
            "sensitivity rather than clear public-benchmark confirmation or "
            "rejection."
        )
    else:
        outcome = "weakens"
        short = (
            f"`{comparison.candidate_feature_set}` did not improve over "
            f"`{comparison.baseline_feature_set}` on OULAD."
        )
        interpretation = (
            "The OULAD benchmark weakens the synthetic carry-forward claim: the "
            "mastery analogue performs worse than the LMS baseline on the primary "
            "grouped split. The result should be read as transfer stress, not as a "
            "dataset-quality comparison."
        )
    return {
        "outcome": outcome,
        "primary_delta_rmse": primary_delta,
        "improvement_rmse_tolerance": tolerance,
        "short_conclusion": short,
        "interpretation": interpretation,
    }


def _validate_feature_columns(
    frame: pd.DataFrame,
    feature_sets: list[BenchmarkFeatureSet],
) -> None:
    missing_by_set: dict[str, list[str]] = {}
    for feature_set in feature_sets:
        missing = [
            column
            for column in feature_set.all_columns()
            if column not in frame.columns
        ]
        if missing:
            missing_by_set[feature_set.name] = missing
    if missing_by_set:
        raise ValueError(f"OULAD benchmark feature columns are missing: {missing_by_set}")


def _select_strategies(config: PublicBenchmarkOuladConfig) -> list[SplitStrategy]:
    strategies: list[SplitStrategy] = [config.splits.primary]
    if config.splits.secondary and config.splits.secondary != config.splits.primary:
        strategies.append(config.splits.secondary)
    return strategies


def _registry_dataset_label(config: PublicBenchmarkOuladConfig) -> str:
    filt = config.oulad.course_filter
    if filt.code_module and filt.code_presentation:
        return f"OULAD {filt.code_module} {filt.code_presentation}"
    if filt.code_module:
        return f"OULAD {filt.code_module}"
    if filt.code_presentation:
        return f"OULAD {filt.code_presentation}"
    return "OULAD all module-presentations"


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
    raise FileNotFoundError(f"Could not resolve OULAD benchmark config path: {config_path}")


def _format_delta(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:+.3f}"


def _format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.3f}"


def _markdown_relative_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the versioned OULAD public benchmark experiment."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_005_public_benchmark_oulad.yaml",
        help="Path to the OULAD benchmark YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rerunning into an existing experiment directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_public_benchmark_oulad(args.config, allow_existing=args.overwrite)
    print(f"Experiment metadata: {payload['metadata_path']}")
    print(f"Experiment docs: {payload['docs_path']}")
    print(f"Processed snapshots: {payload['snapshots_path']}")
    print(f"Results CSV: {payload['result_paths']['csv']}")


if __name__ == "__main__":
    main()
