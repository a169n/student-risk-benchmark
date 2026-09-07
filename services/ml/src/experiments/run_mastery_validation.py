"""Versioned mastery-validation experiment runner.

Validates whether the mastery block (`current_topic_mastery`,
`overall_mastery`) is genuinely useful beyond the `B_lms` baseline or whether
it is too directly aligned with `final_grade` to be trusted as the main
carry-forward representation.

The runner reuses the baseline modeling machinery for the headline
feature-set comparison and adds focused mastery diagnostics on top:

1. mastery vs. target correlations, globally and per week,
2. mastery vs. LMS-feature redundancy,
3. week-cutoff evaluation of `B_lms` and `B_lms_plus_mastery`,
4. drop-column tests for each mastery column,
5. permutation importance for the best regression model,
6. an explicit carry-forward recommendation.

The primary supervised target stays `final_grade`. `passed` is intentionally
ignored here because earlier experiments already showed it is near-trivial on
the current synthetic data; this experiment focuses on whether mastery helps
the grade-level story.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src.experiments.config import ExperimentConfig, load_experiment_config
from src.experiments.datasets import load_modeling_dataset
from src.experiments.diagnostics import (
    build_lineage_documentation,
    compute_lms_target_correlations,
    compute_mastery_target_correlations,
    compute_permutation_importance,
    compute_redundancy_with_lms,
    derive_carry_forward_recommendation,
    run_drop_column_tests,
    run_weekly_validation,
    summarize_weekly_delta,
)
from src.experiments.evaluate import ResultRow, results_to_dataframe
from src.experiments.featuresets import get_feature_set
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
from src.experiments.run_baselines import run_experiments


class ExperimentIdentityConfig(BaseModel):
    experiment_id: str
    title: str
    schema_version: str = "1.2"
    dataset_version: str = "v1_3_refined"
    dataset_config_name: str = "generator_v1_3_refined.yaml"
    parent_experiment: str | None = None
    status: str = "completed"


class DocumentationConfig(BaseModel):
    objective: str
    hypothesis: str
    limitations: list[str] = Field(default_factory=list)
    next_step: str


class MasteryValidationConfig(BaseModel):
    primary_split: str = "student_group"
    baseline_feature_set: str = "B_lms"
    candidate_feature_set: str = "B_lms_plus_mastery"
    optional_feature_sets: list[str] = Field(default_factory=list)
    mastery_columns: list[str] = Field(
        default_factory=lambda: ["current_topic_mastery", "overall_mastery"]
    )
    improvement_rmse_tolerance: float = 0.05
    early_week_max: int = 6
    target_proxy_correlation_warn: float = 0.95
    redundancy_correlation_warn: float = 0.95
    weekly_cutoffs: list[int] = Field(
        default_factory=lambda: [4, 5, 6, 7, 8, 9, 10]
    )


class MasteryValidationLifecycleConfig(BaseModel):
    experiment: ExperimentIdentityConfig
    documentation: DocumentationConfig
    mastery_validation: MasteryValidationConfig = Field(
        default_factory=MasteryValidationConfig
    )


def load_mastery_validation_config(
    config_path: str | Path,
) -> tuple[ExperimentConfig, MasteryValidationLifecycleConfig, Path]:
    modeling_config, resolved_path = load_experiment_config(config_path)
    raw_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    lifecycle = MasteryValidationLifecycleConfig.model_validate(raw_payload)
    validate_experiment_id(lifecycle.experiment.experiment_id)
    return modeling_config, lifecycle, resolved_path


def run_mastery_validation(
    config_path: str | Path,
    *,
    allow_existing: bool = False,
    artifact_root: Path | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    modeling_config, lifecycle, resolved_config_path = load_mastery_validation_config(
        config_path
    )
    experiment_id = lifecycle.experiment.experiment_id
    output_dir = modeling_config.resolve_path(modeling_config.outputs.experiments_dir)
    expected_dir = resolve_experiment_artifact_dir(experiment_id, root=artifact_root)
    if output_dir != expected_dir:
        raise ValueError(
            "Mastery-validation output directory must match the experiment ID. "
            f"Expected {expected_dir}, got {output_dir}."
        )

    ensure_new_experiment_dir(output_dir, allow_existing=allow_existing)

    payload = run_experiments(modeling_config)
    rows: list[ResultRow] = payload["result_rows"]

    dataset = load_modeling_dataset(modeling_config)
    diagnostics = build_mastery_diagnostics(
        dataset_frame=dataset.frame,
        rows=rows,
        modeling_config=modeling_config,
        lifecycle=lifecycle,
    )

    diagnostics_path = output_dir / "mastery_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    weekly_csv_path = output_dir / "mastery_weekly_validation.csv"
    _write_weekly_csv(weekly_csv_path, diagnostics["weekly_validation"]["rows"])

    artifact_summary_path = output_dir / "mastery_carry_forward_recommendation.md"
    summary_markdown = render_mastery_markdown(
        modeling_config=modeling_config,
        lifecycle=lifecycle,
        diagnostics=diagnostics,
        result_paths=payload["result_paths"],
        config_path=resolved_config_path,
        artifact_summary_path=artifact_summary_path,
        diagnostics_path=diagnostics_path,
        weekly_csv_path=weekly_csv_path,
    )
    artifact_summary_path.write_text(summary_markdown, encoding="utf-8")

    experiment_docs_dir = docs_dir or DOCS_EXPERIMENTS_DIR
    docs_path = experiment_docs_dir / f"{experiment_id}.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(summary_markdown, encoding="utf-8")

    metadata = build_experiment_metadata(
        modeling_config=modeling_config,
        lifecycle=lifecycle,
        diagnostics=diagnostics,
        result_paths=payload["result_paths"],
        config_path=resolved_config_path,
        output_dir=output_dir,
        docs_path=docs_path,
        diagnostics_path=diagnostics_path,
        artifact_summary_path=artifact_summary_path,
        weekly_csv_path=weekly_csv_path,
    )
    metadata_path = write_experiment_metadata(metadata, output_dir=output_dir)

    registry_path = experiment_docs_dir / "registry.md"
    upsert_registry_entry(
        registry_path,
        RegistryEntry(
            experiment_id=experiment_id,
            title=lifecycle.experiment.title,
            status=lifecycle.experiment.status,
            schema_version=f"v{lifecycle.experiment.schema_version}",
            dataset_config=lifecycle.experiment.dataset_config_name,
            primary_target="final_grade",
            artifact_dir=_markdown_relative_path(registry_path.parent, output_dir),
            doc_path=_markdown_relative_path(registry_path.parent, docs_path),
            conclusion=diagnostics["recommendation"]["decision_text"],
        ),
    )

    return {
        **payload,
        "diagnostics": diagnostics,
        "diagnostics_path": diagnostics_path,
        "metadata_path": metadata_path,
        "docs_path": docs_path,
        "artifact_summary_path": artifact_summary_path,
        "weekly_csv_path": weekly_csv_path,
        "registry_path": registry_path,
    }


def build_mastery_diagnostics(
    *,
    dataset_frame: pd.DataFrame,
    rows: list[ResultRow],
    modeling_config: ExperimentConfig,
    lifecycle: MasteryValidationLifecycleConfig,
) -> dict[str, Any]:
    mv = lifecycle.mastery_validation
    mastery_columns = list(mv.mastery_columns)

    headline_summary = _best_regression_by_feature_set(
        rows,
        primary_split=mv.primary_split,
        baseline_feature_set=mv.baseline_feature_set,
    )
    overall_delta_rmse = _candidate_overall_delta(
        headline_summary,
        primary_split=mv.primary_split,
        candidate_feature_set=mv.candidate_feature_set,
    )

    target_correlations = compute_mastery_target_correlations(
        dataset_frame,
        mastery_columns=mastery_columns,
        weekly_cutoffs=mv.weekly_cutoffs,
    )
    redundancy = compute_redundancy_with_lms(
        dataset_frame,
        mastery_columns=mastery_columns,
        lms_feature_set=get_feature_set(mv.baseline_feature_set),
    )
    redundancy_summary = _summarize_redundancy(redundancy)
    lms_target_corrs = compute_lms_target_correlations(
        dataset_frame,
        lms_feature_set=get_feature_set(mv.baseline_feature_set),
    )

    weekly_rows = run_weekly_validation(
        dataset_frame,
        feature_sets=[mv.baseline_feature_set, mv.candidate_feature_set],
        weekly_cutoffs=mv.weekly_cutoffs,
        model_names=modeling_config.regression_models,
        seed=modeling_config.seed,
        test_size=modeling_config.splits.student_group.test_size,
        split_seed=modeling_config.splits.student_group.seed,
    )
    weekly_summary = summarize_weekly_delta(
        weekly_rows,
        baseline_feature_set=mv.baseline_feature_set,
        candidate_feature_set=mv.candidate_feature_set,
        early_week_max=mv.early_week_max,
        improvement_rmse_tolerance=mv.improvement_rmse_tolerance,
    )

    best_regression_model = _best_regression_model_name(
        rows,
        feature_set=mv.candidate_feature_set,
        split_strategy=mv.primary_split,
        fallback=modeling_config.regression_models[0]
        if modeling_config.regression_models
        else "gradient_boosting",
    )
    drop_column = run_drop_column_tests(
        dataset_frame,
        candidate_feature_set_name=mv.candidate_feature_set,
        mastery_columns=mastery_columns,
        model_name=best_regression_model,
        seed=modeling_config.seed,
        test_size=modeling_config.splits.student_group.test_size,
        split_seed=modeling_config.splits.student_group.seed,
    )
    permutation = compute_permutation_importance(
        dataset_frame,
        candidate_feature_set_name=mv.candidate_feature_set,
        model_name=best_regression_model,
        seed=modeling_config.seed,
        test_size=modeling_config.splits.student_group.test_size,
        split_seed=modeling_config.splits.student_group.seed,
    )

    recommendation = derive_carry_forward_recommendation(
        overall_delta_rmse=overall_delta_rmse,
        weekly_summary=weekly_summary,
        target_correlations=target_correlations,
        lms_target_correlations=lms_target_corrs,
        redundancy_summary=redundancy_summary,
        drop_column=drop_column,
        candidate_feature_set=mv.candidate_feature_set,
        baseline_feature_set=mv.baseline_feature_set,
        improvement_rmse_tolerance=mv.improvement_rmse_tolerance,
        target_proxy_correlation_warn=mv.target_proxy_correlation_warn,
        redundancy_correlation_warn=mv.redundancy_correlation_warn,
    )

    return {
        "experiment_id": lifecycle.experiment.experiment_id,
        "primary_target": "final_grade",
        "secondary_target": None,
        "primary_split": mv.primary_split,
        "headline_regression": headline_summary,
        "overall_delta_rmse_vs_baseline": overall_delta_rmse,
        "target_correlations": target_correlations,
        "lms_target_correlations": lms_target_corrs,
        "redundancy_with_lms": redundancy,
        "redundancy_summary": redundancy_summary,
        "weekly_validation": {
            "rows": weekly_rows,
            "summary": weekly_summary,
        },
        "drop_column_tests": drop_column,
        "permutation_importance": permutation,
        "best_regression_model": best_regression_model,
        "lineage": build_lineage_documentation(),
        "recommendation": recommendation,
        "thresholds": {
            "improvement_rmse_tolerance": mv.improvement_rmse_tolerance,
            "target_proxy_correlation_warn": mv.target_proxy_correlation_warn,
            "redundancy_correlation_warn": mv.redundancy_correlation_warn,
            "early_week_max": mv.early_week_max,
        },
    }


def build_experiment_metadata(
    *,
    modeling_config: ExperimentConfig,
    lifecycle: MasteryValidationLifecycleConfig,
    diagnostics: dict[str, Any],
    result_paths: dict[str, Path],
    config_path: Path,
    output_dir: Path,
    docs_path: Path,
    diagnostics_path: Path,
    artifact_summary_path: Path,
    weekly_csv_path: Path,
) -> dict[str, Any]:
    dataset = modeling_config.dataset
    return {
        "experiment_id": lifecycle.experiment.experiment_id,
        "title": lifecycle.experiment.title,
        "created_at": utc_now_iso(),
        "dataset": {
            "snapshots_path": str(modeling_config.resolve_path(dataset.snapshots_csv)),
            "final_results_path": str(modeling_config.resolve_path(dataset.final_results_csv)),
            "students_path": str(modeling_config.resolve_path(dataset.students_csv)),
            "courses_path": str(modeling_config.resolve_path(dataset.courses_csv)),
            "dataset_version": lifecycle.experiment.dataset_version,
            "dataset_config_name": lifecycle.experiment.dataset_config_name,
        },
        "schema_version": lifecycle.experiment.schema_version,
        "targets": {
            "primary": "final_grade",
            "secondary": None,
            "excluded": ["risk_level", "passed"],
        },
        "split_strategies": {
            "primary": modeling_config.splits.primary,
            "secondary": modeling_config.splits.secondary,
            "student_group": modeling_config.splits.student_group.model_dump(),
            "temporal_forward": modeling_config.splits.temporal_forward.model_dump(),
        },
        "feature_sets": [
            {
                "name": feature_set.name,
                "description": feature_set.description,
                "columns": list(feature_set.columns),
                "indicator_columns": list(feature_set.indicator_columns),
            }
            for feature_set in [
                get_feature_set(name) for name in modeling_config.feature_sets
            ]
        ],
        "mastery_validation": lifecycle.mastery_validation.model_dump(),
        "models": {
            "regression": list(modeling_config.regression_models),
            "classification": list(modeling_config.classification_models),
        },
        "seed": modeling_config.seed,
        "config_path": relative_repo_path(config_path),
        "output_directory": relative_repo_path(output_dir),
        "status": lifecycle.experiment.status,
        "parent_experiment": lifecycle.experiment.parent_experiment,
        "artifacts": {
            "results_csv": relative_repo_path(result_paths["csv"]),
            "results_json": relative_repo_path(result_paths["json"]),
            "results_markdown": relative_repo_path(result_paths["markdown"]),
            "diagnostics_json": relative_repo_path(diagnostics_path),
            "weekly_validation_csv": relative_repo_path(weekly_csv_path),
            "carry_forward_recommendation": relative_repo_path(artifact_summary_path),
            "documentation": relative_repo_path(docs_path),
        },
        "lineage_summary": diagnostics["lineage"],
        "recommendation": diagnostics["recommendation"],
    }


def render_mastery_markdown(
    *,
    modeling_config: ExperimentConfig,
    lifecycle: MasteryValidationLifecycleConfig,
    diagnostics: dict[str, Any],
    result_paths: dict[str, Path],
    config_path: Path,
    artifact_summary_path: Path,
    diagnostics_path: Path,
    weekly_csv_path: Path,
) -> str:
    exp = lifecycle.experiment
    mv = lifecycle.mastery_validation
    recommendation = diagnostics["recommendation"]
    dataset = modeling_config.dataset
    snapshots_path = relative_repo_path(modeling_config.resolve_path(dataset.snapshots_csv))
    final_results_path = relative_repo_path(
        modeling_config.resolve_path(dataset.final_results_csv)
    )
    output_dir = relative_repo_path(
        modeling_config.resolve_path(modeling_config.outputs.experiments_dir)
    )

    lines: list[str] = [
        f"# {exp.experiment_id}: {exp.title}",
        "",
        "## Objective",
        "",
        lifecycle.documentation.objective,
        "",
        "## Hypothesis",
        "",
        lifecycle.documentation.hypothesis,
        "",
        "## Dataset / config used",
        "",
        f"- Schema version: `v{exp.schema_version}`",
        f"- Dataset version/config: `{exp.dataset_version}` / `{exp.dataset_config_name}`",
        f"- Experiment config: `{relative_repo_path(config_path)}`",
        f"- Snapshot table: `{snapshots_path}`",
        f"- Final results table: `{final_results_path}`",
        f"- Output directory: `{output_dir}`",
        "",
        "## Compared feature sets",
        "",
        f"- Baseline: `{mv.baseline_feature_set}`",
        f"- Candidate: `{mv.candidate_feature_set}`",
    ]
    if mv.optional_feature_sets:
        joined = ", ".join(f"`{name}`" for name in mv.optional_feature_sets)
        lines.append(f"- Optional context sets: {joined}")
    lines.extend(
        [
            "",
            "## Split strategies",
            "",
            f"- Primary: `{modeling_config.splits.primary}`",
            f"- Secondary: `{modeling_config.splits.secondary}`",
            f"- Seed: `{modeling_config.seed}`",
            "",
            "## Week-aware protocol",
            "",
            (
                "Per-week regression evaluation restricts snapshots to a single "
                "week and applies the student-group split (test_size="
                f"{modeling_config.splits.student_group.test_size}, seed="
                f"{modeling_config.splits.student_group.seed}). The candidate "
                f"feature set is `{mv.candidate_feature_set}`; the baseline is "
                f"`{mv.baseline_feature_set}`. Improvement at a week is "
                f"`candidate_rmse - baseline_rmse < -{mv.improvement_rmse_tolerance}`. "
                f"Weeks examined: {sorted(set(mv.weekly_cutoffs))}. Early-week "
                f"window: weeks <= {mv.early_week_max}."
            ),
            "",
            "## Diagnostic checks",
            "",
            (
                "1. mastery vs `final_grade` correlations, globally and per "
                "week,"
            ),
            (
                "2. mastery vs LMS-baseline-feature redundancy (max |Pearson r|),"
            ),
            (
                "3. drop-column re-training of the candidate model,"
            ),
            (
                "4. permutation importance for the same model on the held-out "
                "split."
            ),
            "",
            "## Headline regression results",
            "",
            (
                "Primary regression on the full snapshot range (week >= "
                f"{modeling_config.snapshot_filter.min_week}). Lower RMSE is "
                "better. `delta_vs_B_lms_rmse` is computed within the same "
                "split after picking the best model per feature set."
            ),
            "",
            "| split | feature set | best model | RMSE | MAE | R^2 | delta vs `B_lms` |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in diagnostics["headline_regression"]:
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

    lines.extend(
        [
            "",
            "## Mastery vs `final_grade` correlations",
            "",
            "| feature | global Pearson r | strongest weekly |r| | weakest weekly |r| |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for column in diagnostics["target_correlations"]["mastery_columns"]:
        global_value = diagnostics["target_correlations"]["global_pearson"].get(column)
        weekly_values = [
            float(value)
            for value in diagnostics["target_correlations"]["weekly_pearson"]
            .get(column, {})
            .values()
            if value is not None
        ]
        max_abs = max((abs(v) for v in weekly_values), default=None)
        min_abs = min((abs(v) for v in weekly_values), default=None)
        lines.append(
            "| {col} | {g} | {hi} | {lo} |".format(
                col=column,
                g=_format_optional(global_value),
                hi=_format_optional(max_abs),
                lo=_format_optional(min_abs),
            )
        )

    lines.extend(
        [
            "",
            "## Mastery redundancy with LMS baseline features",
            "",
            "| mastery feature | strongest LMS correlate | |Pearson r| |",
            "| --- | --- | ---: |",
        ]
    )
    for column, payload in diagnostics["redundancy_summary"].items():
        lines.append(
            "| {col} | {lms} | {value} |".format(
                col=column,
                lms=str(payload.get("lms_column") or "n/a"),
                value=_format_optional(payload.get("max_abs_pearson")),
            )
        )

    lms_target_corrs = diagnostics.get("lms_target_correlations") or {}
    if lms_target_corrs:
        lines.extend(
            [
                "",
                "## Context: LMS baseline target correlations",
                "",
                (
                    "Reported so the mastery vs target correlation can be "
                    "compared to what `B_lms` already provides on this dataset."
                ),
                "",
                "| LMS feature | Pearson r vs `final_grade` |",
                "| --- | ---: |",
            ]
        )
        for column, value in lms_target_corrs.items():
            lines.append(
                "| {col} | {val} |".format(
                    col=column,
                    val=_format_optional(value),
                )
            )

    weekly_summary = diagnostics["weekly_validation"]["summary"]
    lines.extend(
        [
            "",
            "## Week-aware delta of `B_lms_plus_mastery` vs `B_lms`",
            "",
            "Negative `delta_rmse` means mastery improved on the LMS baseline at that week.",
            "",
            "| week | baseline RMSE | candidate RMSE | delta RMSE | candidate improves baseline? |",
            "| ---: | ---: | ---: | ---: | :---: |",
        ]
    )
    for record in weekly_summary["rows"]:
        lines.append(
            "| {week} | {b:.3f} | {c:.3f} | {d:+.3f} | {flag} |".format(
                week=record["week"],
                b=record["baseline_rmse"],
                c=record["candidate_rmse"],
                d=record["delta_rmse_vs_baseline"],
                flag="yes" if record["candidate_improves_baseline"] else "no",
            )
        )

    drop_column = diagnostics["drop_column_tests"]
    lines.extend(
        [
            "",
            "## Drop-column tests on `B_lms_plus_mastery`",
            "",
            (
                f"Best regression model for the candidate set: "
                f"`{drop_column.get('model', 'n/a')}`. "
                f"Full RMSE on student-group split: "
                f"`{_format_optional(drop_column.get('full_rmse'))}`."
            ),
            "",
            "| dropped feature | new RMSE | delta vs full RMSE |",
            "| --- | ---: | ---: |",
        ]
    )
    for record in drop_column.get("drop_results", []):
        lines.append(
            "| {col} | {rmse} | {delta} |".format(
                col=record["dropped_column"],
                rmse=_format_optional(record.get("rmse")),
                delta=_format_delta(record.get("delta_rmse_vs_full")),
            )
        )

    permutation = diagnostics["permutation_importance"]
    if permutation.get("importances"):
        lines.extend(
            [
                "",
                "## Permutation importance (top features)",
                "",
                "| feature | mean RMSE increase | std |",
                "| --- | ---: | ---: |",
            ]
        )
        for record in permutation["importances"][:10]:
            lines.append(
                "| {feat} | {mean} | {std} |".format(
                    feat=record["feature"],
                    mean=_format_optional(record.get("mean_rmse_increase")),
                    std=_format_optional(record.get("std_rmse_increase")),
                )
            )

    lines.extend(
        [
            "",
            "## Temporal legitimacy of mastery features",
            "",
        ]
    )
    lineage = diagnostics["lineage"]
    for column in mv.mastery_columns:
        info = lineage.get(column)
        if not info:
            continue
        joined_sources = ", ".join(info.get("source_fields", []))
        lines.extend(
            [
                f"### `{column}`",
                "",
                f"- Source fields: {joined_sources}",
                f"- Uses future information: `{info.get('uses_future_information', False)}`",
                (
                    "- Uses end-of-course outcome fields: "
                    f"`{info.get('uses_final_outcome_fields', False)}`"
                ),
                f"- Notes: {info.get('notes', '')}",
                "",
            ]
        )
    lines.append(f"_Verification:_ {lineage.get('verification', '')}")

    lines.extend(
        [
            "",
            "## Main findings",
            "",
            "- Headline overall delta `candidate - baseline` (primary split): "
            f"{_format_delta(diagnostics['overall_delta_rmse_vs_baseline'])}",
            "- Weeks where mastery improved baseline (early): "
            f"{weekly_summary.get('early_weeks_improved', [])}",
            "- Weeks where mastery improved baseline (late): "
            f"{weekly_summary.get('late_weeks_improved', [])}",
            "- Max absolute mastery vs `final_grade` Pearson r: "
            f"{_format_optional(recommendation.get('max_target_correlation'))}",
            "- Highly redundant mastery columns: "
            f"{recommendation.get('redundant_mastery_columns', [])}",
            "",
            "## Interpretation",
            "",
            recommendation["decision_text"],
            "",
        ]
    )
    if recommendation["flags"]:
        lines.append("### Flags")
        lines.append("")
        for flag in recommendation["flags"]:
            lines.append(f"- {flag}")
        lines.append("")

    lines.extend(
        [
            "## Decision",
            "",
            f"- Outcome: `{recommendation['outcome']}`",
            f"- Carry-forward feature set: `{recommendation['candidate_feature_set']}`"
            f" if outcome is `carry_forward`/`carry_forward_with_caveat`,"
            f" otherwise revisit before XAI.",
            f"- Reference baseline: `{recommendation['baseline_feature_set']}`",
            "",
            "## Artifact paths",
            "",
            f"- Results JSON: `{relative_repo_path(result_paths['json'])}`",
            f"- Results CSV: `{relative_repo_path(result_paths['csv'])}`",
            f"- Runner summary: `{relative_repo_path(result_paths['markdown'])}`",
            f"- Diagnostics JSON: `{relative_repo_path(diagnostics_path)}`",
            f"- Weekly validation CSV: `{relative_repo_path(weekly_csv_path)}`",
            f"- Carry-forward recommendation: `{relative_repo_path(artifact_summary_path)}`",
            "",
            "## Limitations",
            "",
        ]
    )
    for limitation in lifecycle.documentation.limitations:
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "## Next step",
            "",
            lifecycle.documentation.next_step,
        ]
    )
    return "\n".join(lines) + "\n"


def _best_regression_by_feature_set(
    rows: list[ResultRow],
    *,
    primary_split: str,
    baseline_feature_set: str,
) -> list[dict[str, Any]]:
    table = results_to_dataframe(rows)
    if table.empty:
        return []
    regression = table.loc[table["task"] == "regression"].copy()
    if regression.empty:
        return []
    regression = regression.rename(
        columns={
            "metric_rmse": "rmse",
            "metric_mae": "mae",
            "metric_r2": "r2",
        }
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
    baseline_rmse_by_split = {
        record["split_strategy"]: record["rmse"]
        for record in records
        if record["feature_set"] == baseline_feature_set
    }
    for record in records:
        baseline_rmse = baseline_rmse_by_split.get(record["split_strategy"])
        record["delta_vs_baseline_rmse"] = (
            None if baseline_rmse is None else float(record["rmse"] - baseline_rmse)
        )
    return sorted(
        records,
        key=lambda item: (
            0 if item["split_strategy"] == primary_split else 1,
            item["split_strategy"],
            item["rmse"],
        ),
    )


def _candidate_overall_delta(
    headline: list[dict[str, Any]],
    *,
    primary_split: str,
    candidate_feature_set: str,
) -> float | None:
    for record in headline:
        if (
            record["split_strategy"] == primary_split
            and record["feature_set"] == candidate_feature_set
        ):
            return record.get("delta_vs_baseline_rmse")
    return None


def _summarize_redundancy(
    redundancy: dict[str, dict[str, float | None]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for column, lms_corrs in redundancy.items():
        best_lms: str | None = None
        best_value: float | None = None
        for lms_column, value in lms_corrs.items():
            if value is None:
                continue
            if best_value is None or abs(value) > abs(best_value):
                best_lms = lms_column
                best_value = value
        summary[column] = {
            "max_abs_pearson": None if best_value is None else float(abs(best_value)),
            "signed_pearson": best_value,
            "lms_column": best_lms,
        }
    return summary


def _best_regression_model_name(
    rows: list[ResultRow],
    *,
    feature_set: str,
    split_strategy: str,
    fallback: str,
) -> str:
    candidates = [
        row
        for row in rows
        if row.task == "regression"
        and row.feature_set == feature_set
        and row.split_strategy == split_strategy
    ]
    if not candidates:
        return fallback
    best = min(candidates, key=lambda row: row.metrics.get("rmse", float("inf")))
    return best.model


def _write_weekly_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            "week,feature_set,model,rmse,mae,r2,n_train_rows,n_test_rows\n",
            encoding="utf-8",
        )
        return path
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and (value != value):
        return "n/a"
    return f"{value:.3f}"


def _markdown_relative_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the versioned mastery-validation experiment."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_003_mastery_validation.yaml",
        help="Path to the mastery-validation YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rerunning into an existing experiment directory with metadata.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_mastery_validation(args.config, allow_existing=args.overwrite)
    print(f"Experiment metadata: {payload['metadata_path']}")
    print(f"Experiment docs: {payload['docs_path']}")
    print(f"Diagnostics: {payload['diagnostics_path']}")
    print(f"Weekly validation CSV: {payload['weekly_csv_path']}")


if __name__ == "__main__":
    main()
