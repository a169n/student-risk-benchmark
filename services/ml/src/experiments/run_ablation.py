"""Versioned Twin subgroup ablation runner.

This runner reuses the baseline modeling machinery but wraps it in the new
experiment lifecycle conventions:

1. load a versioned YAML config,
2. reserve an experiment-specific artifact directory,
3. run the configured feature-set ablation,
4. write result artifacts, diagnostics, metadata, Markdown summary, and
   registry entry.

The primary supervised target remains `final_grade`. `passed` can be logged as
secondary context when classification models are configured.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src.experiments.config import ExperimentConfig, load_experiment_config
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


class RecommendationConfig(BaseModel):
    primary_split: str = "student_group"
    baseline_feature_set: str = "B_lms"
    full_feature_set: str = "C_twin_full"
    lean_rmse_tolerance: float = 0.05


class DocumentationConfig(BaseModel):
    hypothesis: str
    objective: str
    limitations: list[str] = Field(default_factory=list)
    next_step: str


class AblationLifecycleConfig(BaseModel):
    experiment: ExperimentIdentityConfig
    recommendation: RecommendationConfig = Field(default_factory=RecommendationConfig)
    documentation: DocumentationConfig


def load_ablation_config(
    config_path: str | Path,
) -> tuple[ExperimentConfig, AblationLifecycleConfig, Path]:
    modeling_config, resolved_path = load_experiment_config(config_path)
    raw_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    lifecycle = AblationLifecycleConfig.model_validate(raw_payload)
    validate_experiment_id(lifecycle.experiment.experiment_id)
    return modeling_config, lifecycle, resolved_path


def run_ablation(
    config_path: str | Path,
    *,
    allow_existing: bool = False,
    artifact_root: Path | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    modeling_config, lifecycle, resolved_config_path = load_ablation_config(config_path)
    experiment_id = lifecycle.experiment.experiment_id
    output_dir = modeling_config.resolve_path(modeling_config.outputs.experiments_dir)
    expected_dir = resolve_experiment_artifact_dir(experiment_id, root=artifact_root)
    if output_dir != expected_dir:
        raise ValueError(
            "Ablation output directory must match the experiment ID. "
            f"Expected {expected_dir}, got {output_dir}."
        )

    ensure_new_experiment_dir(output_dir, allow_existing=allow_existing)
    payload = run_experiments(modeling_config)
    rows: list[ResultRow] = payload["result_rows"]
    diagnostics = build_ablation_diagnostics(rows, modeling_config, lifecycle)

    diagnostics_path = output_dir / "ablation_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )

    artifact_summary_path = output_dir / "lean_twin_recommendation.md"
    summary_markdown = render_ablation_markdown(
        modeling_config=modeling_config,
        lifecycle=lifecycle,
        diagnostics=diagnostics,
        result_paths=payload["result_paths"],
        config_path=resolved_config_path,
        artifact_summary_path=artifact_summary_path,
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
            conclusion=diagnostics["recommendation"]["short_conclusion"],
        ),
    )

    return {
        **payload,
        "diagnostics": diagnostics,
        "diagnostics_path": diagnostics_path,
        "metadata_path": metadata_path,
        "docs_path": docs_path,
        "artifact_summary_path": artifact_summary_path,
        "registry_path": registry_path,
    }


def build_ablation_diagnostics(
    rows: list[ResultRow],
    modeling_config: ExperimentConfig,
    lifecycle: AblationLifecycleConfig,
) -> dict[str, Any]:
    table = results_to_dataframe(rows)
    feature_counts = {
        name: len(get_feature_set(name).all_columns()) for name in modeling_config.feature_sets
    }
    regression_summary = _best_regression_by_feature_set(
        table,
        primary_split=lifecycle.recommendation.primary_split,
        baseline_feature_set=lifecycle.recommendation.baseline_feature_set,
    )
    classification_summary = _best_classification_by_feature_set(table)

    recommendation = _recommend_lean_set(
        regression_summary,
        feature_counts=feature_counts,
        lifecycle=lifecycle,
    )
    return {
        "experiment_id": lifecycle.experiment.experiment_id,
        "primary_target": "final_grade",
        "secondary_target": "passed" if modeling_config.classification_models else None,
        "primary_split": lifecycle.recommendation.primary_split,
        "feature_counts": feature_counts,
        "regression_best_by_feature_set": regression_summary,
        "classification_best_by_feature_set": classification_summary,
        "recommendation": recommendation,
    }


def build_experiment_metadata(
    *,
    modeling_config: ExperimentConfig,
    lifecycle: AblationLifecycleConfig,
    diagnostics: dict[str, Any],
    result_paths: dict[str, Path],
    config_path: Path,
    output_dir: Path,
    docs_path: Path,
    diagnostics_path: Path,
    artifact_summary_path: Path,
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
            "secondary": "passed" if modeling_config.classification_models else None,
            "excluded": ["risk_level"],
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
            for feature_set in [get_feature_set(name) for name in modeling_config.feature_sets]
        ],
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
            "lean_twin_recommendation": relative_repo_path(artifact_summary_path),
            "documentation": relative_repo_path(docs_path),
        },
        "short_conclusion": diagnostics["recommendation"]["short_conclusion"],
    }


def render_ablation_markdown(
    *,
    modeling_config: ExperimentConfig,
    lifecycle: AblationLifecycleConfig,
    diagnostics: dict[str, Any],
    result_paths: dict[str, Path],
    config_path: Path,
    artifact_summary_path: Path,
) -> str:
    exp = lifecycle.experiment
    recommendation = diagnostics["recommendation"]
    dataset = modeling_config.dataset
    outputs = modeling_config.outputs
    snapshots_path = relative_repo_path(modeling_config.resolve_path(dataset.snapshots_csv))
    final_results_path = relative_repo_path(
        modeling_config.resolve_path(dataset.final_results_csv)
    )
    output_dir = relative_repo_path(modeling_config.resolve_path(outputs.experiments_dir))
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
        "## Experimental setup",
        "",
        "- Primary target: `final_grade`",
        "- Secondary context target: `passed`",
        "- Explicitly excluded as supervised target: `risk_level`",
        f"- Primary split: `{modeling_config.splits.primary}`",
        f"- Secondary split: `{modeling_config.splits.secondary}`",
        f"- Seed: `{modeling_config.seed}`",
        "",
        "## Feature sets",
        "",
        "| feature set | columns | intent |",
        "| --- | ---: | --- |",
    ]
    for name in modeling_config.feature_sets:
        feature_set = get_feature_set(name)
        lines.append(
            "| {name} | {count} | {description} |".format(
                name=feature_set.name,
                count=len(feature_set.all_columns()),
                description=feature_set.description,
            )
        )

    lines.extend(
        [
            "",
            "## Models",
            "",
            f"- Regression: `{', '.join(modeling_config.regression_models)}`",
            f"- Classification: `{', '.join(modeling_config.classification_models)}`",
            "",
            "## Main metrics",
            "",
            "Primary regression summary. Lower RMSE is better. `delta_vs_B_lms_rmse` "
            "is computed within the same split, after selecting the best model for "
            "each feature set.",
            "",
            "| split | feature set | best model | features | RMSE | MAE | R^2 | delta vs `B_lms` |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in diagnostics["regression_best_by_feature_set"]:
        row_template = (
            "| {split} | {feature_set} | {model} | {count} | {rmse:.3f} | "
            "{mae:.3f} | {r2:.3f} | {delta} |"
        )
        lines.append(
            row_template.format(
                split=record["split_strategy"],
                feature_set=record["feature_set"],
                model=record["model"],
                count=diagnostics["feature_counts"][record["feature_set"]],
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
                "Secondary classification context for `passed`. This is reported to "
                "show whether pass/fail behavior changes, but it is not the main "
                "selection criterion for the ablation.",
                "",
                "| split | feature set | best model | F1 | accuracy | roc_auc |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for record in diagnostics["classification_best_by_feature_set"]:
            lines.append(
                "| {split} | {feature_set} | {model} | {f1:.3f} | {accuracy:.3f} | {auc} |".format(
                    split=record["split_strategy"],
                    feature_set=record["feature_set"],
                    model=record["model"],
                    f1=record["f1"],
                    accuracy=record["accuracy"],
                    auc=_format_optional(record.get("roc_auc")),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            recommendation["interpretation"],
            "",
            "## Lean Twin recommendation",
            "",
            f"- Best full Twin set: `{recommendation['best_full_set']}`",
            f"- Best lean set: `{recommendation['best_lean_set']}`",
            f"- Carry forward: `{recommendation['carry_forward_feature_set']}`",
            f"- Full `C_twin` justified now: `{recommendation['full_twin_justified']}`",
            f"- Short conclusion: {recommendation['short_conclusion']}",
            "",
            "## Artifact paths",
            "",
            f"- Results JSON: `{relative_repo_path(result_paths['json'])}`",
            f"- Results CSV: `{relative_repo_path(result_paths['csv'])}`",
            f"- Runner summary: `{relative_repo_path(result_paths['markdown'])}`",
            f"- Lean recommendation artifact: `{relative_repo_path(artifact_summary_path)}`",
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
    table: pd.DataFrame,
    *,
    primary_split: str,
    baseline_feature_set: str,
) -> list[dict[str, Any]]:
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


def _best_classification_by_feature_set(table: pd.DataFrame) -> list[dict[str, Any]]:
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


def _recommend_lean_set(
    regression_summary: list[dict[str, Any]],
    *,
    feature_counts: dict[str, int],
    lifecycle: AblationLifecycleConfig,
) -> dict[str, Any]:
    primary_split = lifecycle.recommendation.primary_split
    full_feature_set = lifecycle.recommendation.full_feature_set
    tolerance = lifecycle.recommendation.lean_rmse_tolerance
    primary_records = [
        record for record in regression_summary if record["split_strategy"] == primary_split
    ]
    if not primary_records:
        return {
            "best_full_set": full_feature_set,
            "best_lean_set": lifecycle.recommendation.baseline_feature_set,
            "carry_forward_feature_set": lifecycle.recommendation.baseline_feature_set,
            "full_twin_justified": False,
            "short_conclusion": "No primary-split regression records were available.",
            "interpretation": "The ablation did not produce usable primary-split records.",
        }

    best_overall = min(primary_records, key=lambda record: record["rmse"])
    full_records = [
        record for record in primary_records if record["feature_set"] == full_feature_set
    ]
    best_full = full_records[0] if full_records else best_overall
    lean_candidates = [
        record for record in primary_records if record["feature_set"] != full_feature_set
    ]
    best_rmse = best_overall["rmse"]
    competitive = [
        record
        for record in lean_candidates
        if record["rmse"] <= best_rmse + tolerance
    ]
    if competitive:
        best_lean = min(
            competitive,
            key=lambda record: (feature_counts[record["feature_set"]], record["rmse"]),
        )
    else:
        best_lean = min(lean_candidates, key=lambda record: record["rmse"])

    full_twin_justified = best_full["rmse"] + tolerance < best_lean["rmse"]
    carry_forward = full_feature_set if full_twin_justified else best_lean["feature_set"]
    baseline_delta = best_lean.get("delta_vs_baseline_rmse")

    if full_twin_justified:
        interpretation = (
            "The full Twin set materially outperforms the best lean candidate on "
            f"the primary split by more than the configured RMSE tolerance ({tolerance})."
        )
    elif baseline_delta is not None and baseline_delta < -tolerance:
        interpretation = (
            f"`{best_lean['feature_set']}` improves on `"
            f"{lifecycle.recommendation.baseline_feature_set}` enough to justify carrying "
            "that compact subset forward. The full Twin set is not currently justified."
        )
    else:
        interpretation = (
            "The ablation reinforces the baseline finding: richer Twin blocks do "
            "not yet add a robust improvement over the LMS baseline on the primary "
            "student-group split. This points to redundancy in the current Twin "
            "layer rather than a need for a larger model."
        )

    short_conclusion = (
        f"Carry forward `{carry_forward}`; full Twin justified: {full_twin_justified}."
    )
    return {
        "best_full_set": best_full["feature_set"],
        "best_full_rmse": best_full["rmse"],
        "best_lean_set": best_lean["feature_set"],
        "best_lean_rmse": best_lean["rmse"],
        "carry_forward_feature_set": carry_forward,
        "full_twin_justified": full_twin_justified,
        "lean_rmse_tolerance": tolerance,
        "short_conclusion": short_conclusion,
        "interpretation": interpretation,
    }


def _format_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}"


def _format_optional(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.3f}"


def _markdown_relative_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a versioned Twin ablation experiment.")
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_002_twin_ablation.yaml",
        help="Path to the versioned ablation experiment YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rerunning into an existing experiment directory with metadata.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_ablation(args.config, allow_existing=args.overwrite)
    print(f"Experiment metadata: {payload['metadata_path']}")
    print(f"Experiment docs: {payload['docs_path']}")
    print(f"Diagnostics: {payload['diagnostics_path']}")


if __name__ == "__main__":
    main()
