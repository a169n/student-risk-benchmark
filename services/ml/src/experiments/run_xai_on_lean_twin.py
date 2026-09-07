"""Versioned XAI experiment for the lean Twin candidate.

This runner implements ``exp_004_xai_on_lean_twin``. It keeps the modeling
scope intentionally narrow:

- primary target: ``final_grade``
- secondary context only: ``passed``
- split: student-grouped holdout
- reference model: gradient boosting regressor by default
- compared feature sets: ``B_lms`` and ``B_lms_plus_mastery``

The goal is not another model sweep. The goal is to decide whether the
validated lean Twin candidate can be explained in a teacher-meaningful way, and
whether ``overall_mastery`` overwhelms the explanation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src.experiments.config import ExperimentConfig, load_experiment_config
from src.experiments.datasets import REGRESSION_TARGET, WEEK_COLUMN, load_modeling_dataset
from src.experiments.diagnostics import (
    compute_mastery_target_correlations,
    compute_redundancy_with_lms,
)
from src.experiments.explainability import (
    audit_overall_mastery_dominance,
    build_global_explanation,
    build_local_case_explanations,
    compare_global_explanations,
    select_representative_cases,
    train_regression_reference,
)
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


class ExperimentIdentityConfig(BaseModel):
    experiment_id: str
    title: str
    schema_version: str = "1.2"
    dataset_version: str = "v1_3_refined"
    dataset_config_name: str = "generator_v1_3_refined.yaml"
    parent_experiment: str | None = "exp_003_mastery_validation"
    status: str = "completed"


class DocumentationConfig(BaseModel):
    objective: str
    hypothesis: str
    limitations: list[str] = Field(default_factory=list)
    next_step: str


class LocalCaseConfig(BaseModel):
    max_cases: int = 5
    case_types: list[str] = Field(
        default_factory=lambda: [
            "strong_performer",
            "at_risk",
            "improving_trajectory",
            "declining_trajectory",
            "borderline_medium",
        ]
    )
    borderline_grade: float = 50.0
    trend_feature: str = "score_trend_3w"


class DominanceConfig(BaseModel):
    overall_mastery_share_warn: float = 0.45
    explanation_top1_share_warn: float = 0.55
    local_mastery_share_warn: float = 0.60
    drop_rmse_warn: float = 0.25


class XaiConfig(BaseModel):
    primary_split: str = "student_group"
    model: str = "gradient_boosting"
    reference_baseline_feature_set: str = "B_lms"
    lean_twin_feature_set: str = "B_lms_plus_mastery"
    primary_target: str = "final_grade"
    secondary_context_target: str = "passed"
    mastery_columns: list[str] = Field(
        default_factory=lambda: ["current_topic_mastery", "overall_mastery"]
    )
    permutation_repeats: int = 15
    top_k_global: int = 12
    top_k_local: int = 6
    local_cases: LocalCaseConfig = Field(default_factory=LocalCaseConfig)
    dominance: DominanceConfig = Field(default_factory=DominanceConfig)
    shap_enabled: bool = False
    shap_fallback_reason: str = (
        "SHAP is not part of the current project dependency contract; use "
        "documented sklearn permutation and local perturbation fallbacks."
    )


class XaiLifecycleConfig(BaseModel):
    experiment: ExperimentIdentityConfig
    documentation: DocumentationConfig
    xai: XaiConfig = Field(default_factory=XaiConfig)


def load_xai_config(
    config_path: str | Path,
) -> tuple[ExperimentConfig, XaiLifecycleConfig, Path]:
    modeling_config, resolved_path = load_experiment_config(config_path)
    raw_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    lifecycle = XaiLifecycleConfig.model_validate(raw_payload)
    validate_experiment_id(lifecycle.experiment.experiment_id)
    if lifecycle.xai.primary_target != REGRESSION_TARGET:
        raise ValueError("exp_004 must use final_grade as the primary target")
    if lifecycle.xai.primary_split != "student_group":
        raise ValueError("exp_004 is controlled to the student_group split")
    if modeling_config.classification_models:
        raise ValueError("exp_004 must not train classification models")
    if modeling_config.feature_sets != [
        lifecycle.xai.reference_baseline_feature_set,
        lifecycle.xai.lean_twin_feature_set,
    ]:
        raise ValueError(
            "exp_004 feature_sets must be exactly baseline then lean Twin candidate"
        )
    if modeling_config.regression_models != [lifecycle.xai.model]:
        raise ValueError("exp_004 must use exactly one configured regression model")
    return modeling_config, lifecycle, resolved_path


def run_xai_on_lean_twin(
    config_path: str | Path,
    *,
    allow_existing: bool = False,
    artifact_root: Path | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    modeling_config, lifecycle, resolved_config_path = load_xai_config(config_path)
    experiment_id = lifecycle.experiment.experiment_id
    output_dir = modeling_config.resolve_path(modeling_config.outputs.experiments_dir)
    expected_dir = resolve_experiment_artifact_dir(experiment_id, root=artifact_root)
    if output_dir != expected_dir:
        raise ValueError(
            "XAI output directory must match the experiment ID. "
            f"Expected {expected_dir}, got {output_dir}."
        )

    ensure_new_experiment_dir(output_dir, allow_existing=allow_existing)

    dataset = load_modeling_dataset(modeling_config)
    xai = lifecycle.xai
    split_config = modeling_config.splits.student_group

    baseline_run = train_regression_reference(
        dataset.frame,
        feature_set_name=xai.reference_baseline_feature_set,
        model_name=xai.model,
        seed=modeling_config.seed,
        test_size=split_config.test_size,
        validation_size=split_config.validation_size,
        split_seed=split_config.seed,
    )
    lean_run = train_regression_reference(
        dataset.frame,
        feature_set_name=xai.lean_twin_feature_set,
        model_name=xai.model,
        seed=modeling_config.seed,
        test_size=split_config.test_size,
        validation_size=split_config.validation_size,
        split_seed=split_config.seed,
    )
    lean_without_overall_run = train_regression_reference(
        dataset.frame,
        feature_set_name=xai.lean_twin_feature_set,
        model_name=xai.model,
        seed=modeling_config.seed,
        test_size=split_config.test_size,
        validation_size=split_config.validation_size,
        split_seed=split_config.seed,
        drop_columns=("overall_mastery",),
    )

    baseline_global = build_global_explanation(
        baseline_run,
        permutation_repeats=xai.permutation_repeats,
        seed=modeling_config.seed,
    )
    lean_global = build_global_explanation(
        lean_run,
        permutation_repeats=xai.permutation_repeats,
        seed=modeling_config.seed,
    )
    comparison = compare_global_explanations(
        baseline_global=baseline_global,
        lean_global=lean_global,
        mastery_columns=xai.mastery_columns,
    )

    selected_cases = select_representative_cases(
        lean_run,
        case_types=xai.local_cases.case_types,
        max_cases=xai.local_cases.max_cases,
        borderline_grade=xai.local_cases.borderline_grade,
        trend_feature=xai.local_cases.trend_feature,
    )
    local_explanations = build_local_case_explanations(
        lean_run=lean_run,
        baseline_run=baseline_run,
        cases=selected_cases,
        mastery_columns=xai.mastery_columns,
        top_k=xai.top_k_local,
    )

    dominance = audit_overall_mastery_dominance(
        lean_global=lean_global,
        lean_run=lean_run,
        lean_without_overall_run=lean_without_overall_run,
        local_explanations=local_explanations,
        overall_mastery_column="overall_mastery",
        importance_share_warn=xai.dominance.overall_mastery_share_warn,
        top1_share_warn=xai.dominance.explanation_top1_share_warn,
        local_mastery_share_warn=xai.dominance.local_mastery_share_warn,
        drop_rmse_warn=xai.dominance.drop_rmse_warn,
    )

    target_correlations = compute_mastery_target_correlations(
        dataset.frame,
        mastery_columns=xai.mastery_columns,
        weekly_cutoffs=sorted(dataset.frame[WEEK_COLUMN].dropna().unique().tolist()),
    )
    redundancy = compute_redundancy_with_lms(
        dataset.frame,
        mastery_columns=xai.mastery_columns,
        lms_feature_set=get_feature_set(xai.reference_baseline_feature_set),
    )
    redundancy_summary = _summarize_redundancy(redundancy)

    recommendation = derive_xai_recommendation(
        baseline_rmse=baseline_run.metrics["rmse"],
        lean_rmse=lean_run.metrics["rmse"],
        dominance=dominance,
        redundancy_summary=redundancy_summary,
        xai=xai,
    )

    results = {
        "experiment_id": experiment_id,
        "primary_target": xai.primary_target,
        "secondary_context_target": xai.secondary_context_target,
        "split_strategy": xai.primary_split,
        "model": xai.model,
        "feature_sets": {
            "baseline": xai.reference_baseline_feature_set,
            "lean_twin": xai.lean_twin_feature_set,
            "lean_twin_without_overall_mastery": {
                "feature_set": xai.lean_twin_feature_set,
                "dropped_columns": ["overall_mastery"],
            },
        },
        "metrics": {
            "baseline": baseline_run.metrics,
            "lean_twin": lean_run.metrics,
            "lean_twin_without_overall_mastery": lean_without_overall_run.metrics,
            "lean_delta_rmse_vs_baseline": float(
                lean_run.metrics["rmse"] - baseline_run.metrics["rmse"]
            ),
            "without_overall_delta_rmse_vs_lean": float(
                lean_without_overall_run.metrics["rmse"] - lean_run.metrics["rmse"]
            ),
        },
        "global_explanations": {
            "baseline": baseline_global,
            "lean_twin": lean_global,
            "comparison": comparison,
        },
        "local_explanations": local_explanations,
        "dominance_audit": dominance,
        "target_correlations": target_correlations,
        "redundancy_with_lms": redundancy,
        "redundancy_summary": redundancy_summary,
        "recommendation": recommendation,
        "method_notes": {
            "shap_used": False,
            "shap_fallback_reason": xai.shap_fallback_reason,
            "local_explanation_method": (
                "one-feature-at-a-time replacement with the training median; "
                "contributions are directional perturbation effects, not "
                "additive causal attributions"
            ),
        },
    }

    artifact_paths = write_xai_artifacts(
        output_dir=output_dir,
        experiment_id=experiment_id,
        results=results,
        modeling_config=modeling_config,
        lifecycle=lifecycle,
        config_path=resolved_config_path,
    )

    experiment_docs_dir = docs_dir or DOCS_EXPERIMENTS_DIR
    docs_path = experiment_docs_dir / f"{experiment_id}.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    summary_markdown = render_xai_markdown(
        modeling_config=modeling_config,
        lifecycle=lifecycle,
        results=results,
        config_path=resolved_config_path,
        artifact_paths=artifact_paths,
    )
    docs_path.write_text(summary_markdown, encoding="utf-8")
    artifact_paths["docs"] = docs_path
    artifact_paths["summary_markdown"].write_text(summary_markdown, encoding="utf-8")

    recommendation_markdown = render_recommendation_markdown(
        lifecycle=lifecycle,
        results=results,
    )
    artifact_paths["recommendation"].write_text(
        recommendation_markdown,
        encoding="utf-8",
    )

    metadata = build_experiment_metadata(
        modeling_config=modeling_config,
        lifecycle=lifecycle,
        results=results,
        config_path=resolved_config_path,
        output_dir=output_dir,
        artifact_paths=artifact_paths,
        docs_path=docs_path,
    )
    metadata_path = write_experiment_metadata(metadata, output_dir=output_dir)
    artifact_paths["metadata"] = metadata_path

    registry_path = experiment_docs_dir / "registry.md"
    upsert_registry_entry(
        registry_path,
        RegistryEntry(
            experiment_id=experiment_id,
            title=lifecycle.experiment.title,
            status=lifecycle.experiment.status,
            schema_version=f"v{lifecycle.experiment.schema_version}",
            dataset_config=lifecycle.experiment.dataset_config_name,
            primary_target=xai.primary_target,
            artifact_dir=_markdown_relative_path(registry_path.parent, output_dir),
            doc_path=_markdown_relative_path(registry_path.parent, docs_path),
            conclusion=recommendation["decision_text"],
        ),
    )
    artifact_paths["registry"] = registry_path

    return {
        "results": results,
        "artifact_paths": artifact_paths,
        "metadata_path": metadata_path,
        "docs_path": docs_path,
        "registry_path": registry_path,
    }


def write_xai_artifacts(
    *,
    output_dir: Path,
    experiment_id: str,
    results: dict[str, Any],
    modeling_config: ExperimentConfig,
    lifecycle: XaiLifecycleConfig,
    config_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "results_json": output_dir / f"{experiment_id}_results.json",
        "summary_markdown": output_dir / f"{experiment_id}_summary.md",
        "global_importance_csv": output_dir / "global_feature_importance.csv",
        "global_importance_markdown": output_dir / "global_feature_importance.md",
        "local_cases_json": output_dir / "local_case_explanations.json",
        "local_cases_markdown": output_dir / "local_case_explanations.md",
        "recommendation": output_dir / "xai_carry_forward_recommendation.md",
    }
    paths["results_json"].write_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    importance_rows = []
    for label in ["baseline", "lean_twin"]:
        for row in results["global_explanations"][label]["rows"]:
            importance_rows.append({"comparison_subject": label, **row})
    importance_frame = pd.DataFrame(importance_rows)
    importance_frame.to_csv(paths["global_importance_csv"], index=False)
    paths["global_importance_markdown"].write_text(
        render_global_importance_markdown(results),
        encoding="utf-8",
    )

    paths["local_cases_json"].write_text(
        json.dumps(_json_safe(results["local_explanations"]), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    paths["local_cases_markdown"].write_text(
        render_local_cases_markdown(results["local_explanations"]),
        encoding="utf-8",
    )
    paths["summary_markdown"].write_text(
        render_xai_markdown(
            modeling_config=modeling_config,
            lifecycle=lifecycle,
            results=results,
            config_path=config_path,
            artifact_paths=paths,
        ),
        encoding="utf-8",
    )
    return paths


def build_experiment_metadata(
    *,
    modeling_config: ExperimentConfig,
    lifecycle: XaiLifecycleConfig,
    results: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    artifact_paths: dict[str, Path],
    docs_path: Path,
) -> dict[str, Any]:
    dataset = modeling_config.dataset
    xai = lifecycle.xai
    return {
        "experiment_id": lifecycle.experiment.experiment_id,
        "title": lifecycle.experiment.title,
        "created_at": utc_now_iso(),
        "schema_version": lifecycle.experiment.schema_version,
        "dataset": {
            "snapshots_path": str(modeling_config.resolve_path(dataset.snapshots_csv)),
            "final_results_path": str(modeling_config.resolve_path(dataset.final_results_csv)),
            "students_path": str(modeling_config.resolve_path(dataset.students_csv)),
            "courses_path": str(modeling_config.resolve_path(dataset.courses_csv)),
            "dataset_version": lifecycle.experiment.dataset_version,
            "dataset_config_name": lifecycle.experiment.dataset_config_name,
        },
        "targets": {
            "primary": xai.primary_target,
            "secondary_context": xai.secondary_context_target,
            "excluded_from_supervised_training": ["risk_level", "passed"],
        },
        "split_strategies": {
            "primary": modeling_config.splits.primary,
            "secondary": modeling_config.splits.secondary,
            "student_group": modeling_config.splits.student_group.model_dump(),
        },
        "feature_sets": {
            "baseline": xai.reference_baseline_feature_set,
            "lean_twin": xai.lean_twin_feature_set,
        },
        "model": xai.model,
        "seed": modeling_config.seed,
        "xai_config": xai.model_dump(),
        "metrics": results["metrics"],
        "dominance_audit": results["dominance_audit"],
        "recommendation": results["recommendation"],
        "config_path": relative_repo_path(config_path),
        "output_directory": relative_repo_path(output_dir),
        "status": lifecycle.experiment.status,
        "parent_experiment": lifecycle.experiment.parent_experiment,
        "artifacts": {
            key: relative_repo_path(value)
            for key, value in artifact_paths.items()
            if key not in {"registry", "docs"}
        }
        | {"documentation": relative_repo_path(docs_path)},
    }


def derive_xai_recommendation(
    *,
    baseline_rmse: float,
    lean_rmse: float,
    dominance: dict[str, Any],
    redundancy_summary: dict[str, dict[str, Any]],
    xai: XaiConfig,
) -> dict[str, Any]:
    delta = float(lean_rmse - baseline_rmse)
    redundant_mastery = [
        column
        for column, payload in redundancy_summary.items()
        if (payload.get("max_abs_pearson") or 0.0) >= 0.95
    ]
    flags = list(dominance.get("flags", []))
    if redundant_mastery:
        flags.append(
            "mastery redundancy remains high for: " + ", ".join(redundant_mastery)
        )

    if dominance["outcome"] == "dominates_too_much":
        outcome = "do_not_use_as_xai_reference"
        decision = (
            "`B_lms_plus_mastery` is not yet suitable as the dissertation XAI "
            "reference because `overall_mastery` dominates the explanation."
        )
    elif delta <= 0 and dominance["outcome"] in {"acceptable", "acceptable_with_caveat"}:
        outcome = "carry_forward_with_caveat"
        decision = (
            "Carry `B_lms_plus_mastery` forward for dissertation XAI with an "
            "`overall_mastery` redundancy caveat; explanations remain "
            "teacher-meaningful and do not collapse into one feature."
        )
    elif dominance["outcome"] in {"acceptable", "acceptable_with_caveat"}:
        outcome = "xai_context_only"
        decision = (
            "`B_lms_plus_mastery` is interpretable, but it does not improve RMSE "
            "over `B_lms` in this controlled XAI run; use as context rather "
            "than the main reference."
        )
    else:
        outcome = "inconclusive"
        decision = (
            "The current XAI diagnostics are inconclusive; do not broaden the "
            "modeling scope before revisiting the lean feature definition."
        )

    return {
        "outcome": outcome,
        "decision_text": decision,
        "baseline_feature_set": xai.reference_baseline_feature_set,
        "lean_twin_feature_set": xai.lean_twin_feature_set,
        "model": xai.model,
        "lean_delta_rmse_vs_baseline": delta,
        "dominance_outcome": dominance["outcome"],
        "redundant_mastery_columns": redundant_mastery,
        "flags": flags,
    }


def render_xai_markdown(
    *,
    modeling_config: ExperimentConfig,
    lifecycle: XaiLifecycleConfig,
    results: dict[str, Any],
    config_path: Path,
    artifact_paths: dict[str, Path],
) -> str:
    exp = lifecycle.experiment
    xai = lifecycle.xai
    metrics = results["metrics"]
    dominance = results["dominance_audit"]
    recommendation = results["recommendation"]
    output_dir = modeling_config.resolve_path(modeling_config.outputs.experiments_dir)
    snapshots_path = relative_repo_path(
        modeling_config.resolve_path(modeling_config.dataset.snapshots_csv)
    )
    final_results_path = relative_repo_path(
        modeling_config.resolve_path(modeling_config.dataset.final_results_csv)
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
        f"- Output directory: `{relative_repo_path(output_dir)}`",
        "",
        "## Reference model",
        "",
        f"- Split: `{xai.primary_split}`",
        f"- Model: `{xai.model}`",
        f"- Primary target: `{xai.primary_target}`",
        f"- Secondary context only: `{xai.secondary_context_target}`",
        f"- Baseline feature set: `{xai.reference_baseline_feature_set}`",
        f"- Lean Twin feature set: `{xai.lean_twin_feature_set}`",
        "",
        "## Explainability methods",
        "",
        (
            "Global explanations use held-out permutation importance with "
            "`neg_root_mean_squared_error`, plus model-native tree importance "
            "where available. Local explanations use one-feature-at-a-time "
            "replacement with the training median. These are directional "
            "model-behavior explanations, not causal claims."
        ),
        "",
        (
            f"SHAP used: `False`. Fallback reason: {xai.shap_fallback_reason}"
        ),
        "",
        "## Model behavior",
        "",
        "| feature set | RMSE | MAE | R^2 |",
        "| --- | ---: | ---: | ---: |",
        _metric_row(xai.reference_baseline_feature_set, metrics["baseline"]),
        _metric_row(xai.lean_twin_feature_set, metrics["lean_twin"]),
        _metric_row(
            "`B_lms_plus_mastery` without `overall_mastery`",
            metrics["lean_twin_without_overall_mastery"],
        ),
        "",
        (
            "Lean Twin delta vs baseline RMSE: "
            f"`{metrics['lean_delta_rmse_vs_baseline']:+.3f}`. "
            "Without-`overall_mastery` delta vs lean RMSE: "
            f"`{metrics['without_overall_delta_rmse_vs_lean']:+.3f}`."
        ),
        "",
        "## Global findings",
        "",
        *render_top_global_lines(results, top_k=xai.top_k_global),
        "",
        "## Baseline vs lean Twin explanation comparison",
        "",
        _comparison_text(results["global_explanations"]["comparison"]),
        "",
        "## `overall_mastery` dominance audit",
        "",
        f"- Outcome: `{dominance['outcome']}`",
        f"- Rank: `{dominance.get('rank')}`",
        f"- Global importance share: `{_format_optional(dominance.get('importance_share'))}`",
        (
            "- RMSE increase when removed: "
            f"`{dominance['delta_rmse_without_overall_mastery']:+.3f}`"
        ),
        (
            "- Average local mastery contribution share: "
            f"`{_format_optional(dominance.get('average_local_mastery_share'))}`"
        ),
        f"- Interpretation: {dominance['interpretation']}",
        "",
    ]
    if dominance["flags"]:
        lines.append("Flags:")
        lines.extend(f"- {flag}" for flag in dominance["flags"])
        lines.append("")

    lines.extend(
        [
            "## Local case findings",
            "",
            *render_local_case_summary_lines(results["local_explanations"]),
            "",
            "## Interpretation",
            "",
            recommendation["decision_text"],
            "",
            "## Decision / next step",
            "",
            f"- Outcome: `{recommendation['outcome']}`",
            f"- Carry-forward feature set: `{recommendation['lean_twin_feature_set']}`",
            f"- Reference baseline: `{recommendation['baseline_feature_set']}`",
            "",
            "## Artifact paths",
            "",
        ]
    )
    for label, path in artifact_paths.items():
        if label in {"registry", "docs"}:
            continue
        lines.append(f"- {label}: `{relative_repo_path(path)}`")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in lifecycle.documentation.limitations)
    lines.extend(["", "## Next step", "", lifecycle.documentation.next_step])
    return "\n".join(lines) + "\n"


def render_top_global_lines(results: dict[str, Any], *, top_k: int = 8) -> list[str]:
    lines = [
        "Top held-out permutation-importance features are shown below. Positive values",
        "mean RMSE increases when the feature is permuted.",
        "",
        "| feature set | rank | feature | RMSE increase | share | direction note |",
        "| --- | ---: | --- | ---: | ---: | --- |",
    ]
    for label in ["baseline", "lean_twin"]:
        rows = results["global_explanations"][label]["rows"][:top_k]
        for row in rows:
            lines.append(
                "| {fs} | {rank} | {feature} | {increase} | {share} | {note} |".format(
                    fs=row["feature_set"],
                    rank=row["rank"],
                    feature=row["feature"],
                    increase=_format_optional(row.get("mean_rmse_increase")),
                    share=_format_optional(row.get("importance_share")),
                    note=row.get("direction_note", ""),
                )
            )
    return lines


def render_global_importance_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# Global feature importance",
        "",
        "Importance is computed on the held-out student-group split. Positive",
        "`mean_rmse_increase` values indicate that permuting the feature hurt",
        "the model. Shares are normalized over positive permutation importance",
        "when available.",
        "",
        *render_top_global_lines(results),
    ]
    return "\n".join(lines) + "\n"


def render_local_cases_markdown(local_explanations: list[dict[str, Any]]) -> str:
    lines = [
        "# Local case explanations",
        "",
        "Local contributions are one-feature perturbation effects against the",
        "training median. They are directional model-behavior summaries, not",
        "causal attributions.",
        "",
    ]
    for case in local_explanations:
        lines.extend(
            [
                f"## {case['case_type']} - {case['student_id']} week {case['week_number']}",
                "",
                f"- Selection rule: {case['selection_rule']}",
                f"- Predicted final_grade: `{case['predicted_final_grade']:.3f}`",
                f"- Actual final_grade: `{case['actual_final_grade']:.3f}`",
                f"- Prediction error: `{case['prediction_error']:+.3f}`",
                (
                    "- Baseline predicted final_grade: "
                    f"`{_format_optional(case.get('baseline_predicted_final_grade'))}`"
                ),
                (
                    "- Mastery contribution share: "
                    f"`{_format_optional(case.get('mastery_abs_contribution_share'))}`"
                ),
                f"- Assessment: {case['teacher_meaningfulness_assessment']}",
                "",
                "| rank | lean feature | contribution | direction |",
                "| ---: | --- | ---: | --- |",
            ]
        )
        for rank, item in enumerate(case["lean_top_contributions"], start=1):
            lines.append(
                f"| {rank} | {item['feature']} | {item['contribution']:+.3f} | "
                f"{item['direction']} |"
            )
        lines.extend(["", "Baseline top contributors:", ""])
        lines.extend(["| rank | baseline feature | contribution | direction |"])
        lines.extend(["| ---: | --- | ---: | --- |"])
        for rank, item in enumerate(case["baseline_top_contributions"], start=1):
            lines.append(
                f"| {rank} | {item['feature']} | {item['contribution']:+.3f} | "
                f"{item['direction']} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_recommendation_markdown(
    *,
    lifecycle: XaiLifecycleConfig,
    results: dict[str, Any],
) -> str:
    recommendation = results["recommendation"]
    dominance = results["dominance_audit"]
    lines = [
        f"# {lifecycle.experiment.experiment_id}: XAI carry-forward recommendation",
        "",
        f"Outcome: `{recommendation['outcome']}`",
        "",
        recommendation["decision_text"],
        "",
        "## Evidence",
        "",
        (
            "- Lean Twin RMSE delta vs `B_lms`: "
            f"`{recommendation['lean_delta_rmse_vs_baseline']:+.3f}`"
        ),
        f"- Dominance audit outcome: `{dominance['outcome']}`",
        (
            "- `overall_mastery` importance share: "
            f"`{_format_optional(dominance.get('importance_share'))}`"
        ),
        (
            "- `overall_mastery` removal RMSE delta: "
            f"`{dominance['delta_rmse_without_overall_mastery']:+.3f}`"
        ),
        "",
        "## Caveats",
        "",
    ]
    if recommendation["flags"]:
        lines.extend(f"- {flag}" for flag in recommendation["flags"])
    else:
        lines.append("- No configured dominance flags were triggered.")
    return "\n".join(lines) + "\n"


def render_local_case_summary_lines(local_explanations: list[dict[str, Any]]) -> list[str]:
    if not local_explanations:
        return ["No local cases were selected."]
    lines = [
        "| case | student | week | predicted | actual | mastery central? | assessment |",
        "| --- | --- | ---: | ---: | ---: | :---: | --- |",
    ]
    for case in local_explanations:
        lines.append(
            "| {case_type} | {student} | {week} | {pred:.3f} | {actual:.3f} | "
            "{central} | {assessment} |".format(
                case_type=case["case_type"],
                student=case["student_id"],
                week=case["week_number"],
                pred=case["predicted_final_grade"],
                actual=case["actual_final_grade"],
                central="yes" if case["mastery_features_are_central"] else "no",
                assessment=case["teacher_meaningfulness_assessment"],
            )
        )
    return lines


def _comparison_text(comparison: dict[str, Any]) -> str:
    return (
        "Baseline top features: "
        f"{comparison['baseline_top_features']}. Lean Twin top features: "
        f"{comparison['lean_top_features']}. Mastery features entering the lean "
        f"top ranking: {comparison['mastery_features_in_lean_top']}."
    )


def _metric_row(label: str, metrics: dict[str, float]) -> str:
    return "| {label} | {rmse:.3f} | {mae:.3f} | {r2:.3f} |".format(
        label=label,
        rmse=metrics["rmse"],
        mae=metrics["mae"],
        r2=metrics["r2"],
    )


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


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and value != value:
        return "n/a"
    return f"{float(value):.3f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return relative_repo_path(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    return value


def _markdown_relative_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the versioned XAI-on-lean-Twin experiment."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_004_xai_on_lean_twin.yaml",
        help="Path to the XAI experiment YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rerunning into an existing experiment directory with metadata.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_xai_on_lean_twin(args.config, allow_existing=args.overwrite)
    paths = payload["artifact_paths"]
    print(f"Experiment metadata: {payload['metadata_path']}")
    print(f"Experiment docs: {payload['docs_path']}")
    print(f"Results JSON: {paths['results_json']}")
    print(f"Global importance CSV: {paths['global_importance_csv']}")
    print(f"Local cases JSON: {paths['local_cases_json']}")


if __name__ == "__main__":
    main()
