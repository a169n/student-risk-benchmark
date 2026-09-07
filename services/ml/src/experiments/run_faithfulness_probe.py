"""Controlled XAI-faithfulness probe: exp_008_faithfulness_probe.

This is a METHODS APPENDIX experiment. The synthetic generator's ``final_grade``
is a known closed-form weighted mean:

    final_grade = 0.55 * avg_assignment_score_to_date
                + 0.25 * avg_quiz_score_to_date
                + 0.10 * attendance_rate_to_date  (scaled 0-100)
                + 0.10 * on_time_submission_rate_to_date  (scaled 0-100)

We probe whether permutation / native importance recovers the ORDERING of these
oracle weights at the final course week (week 10), where cumulative-to-date
features approximate the formula's full-course inputs.

Two probes run:
- ORACLE probe: feature set = exactly the 4 oracle features.
- PROXY probe: feature set = B_lms (includes activity_score_to_date as a
  latent-driven proxy that enters base_score via +0.04, submissions.py:83).

A REDUNDANCY diagnostic reports the Pearson r between ``overall_mastery`` and
``avg_assignment_score_to_date`` on the final-week frame.

Comparison uses Kendall tau on importance ORDER (not magnitude).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from src.experiments.config import ExperimentConfig, load_experiment_config
from src.experiments.datasets import REGRESSION_TARGET, WEEK_COLUMN, load_modeling_dataset
from src.experiments.explainability import (
    build_global_explanation,
    train_regression_reference_for_matrix,
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
from src.experiments.splits import student_group_split
from src.experiments.stability import kendall_tau


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


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
    hypothesis: str = ""
    limitations: list[str] = Field(default_factory=list)
    next_step: str = ""


class ProbeConfig(BaseModel):
    oracle_weights: dict[str, float]
    oracle_features: list[str]
    proxy_feature_set: str = "B_lms"
    redundancy_pairs: list[list[str]] = Field(default_factory=list)
    permutation_repeats: int = 15
    model: str = "gradient_boosting"
    seed: int = 42
    shap_enabled: bool = False
    final_week: int | None = None


class FaithfulnessProbeConfig(BaseModel):
    experiment: ExperimentIdentityConfig
    documentation: DocumentationConfig
    probe: ProbeConfig


# ---------------------------------------------------------------------------
# Duck-typed feature set for probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SimpleFeatureSet:
    """A minimal duck-typed feature set for use with train_regression_reference_for_matrix."""

    name: str
    columns: tuple[str, ...]
    indicator_columns: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Core probe logic
# ---------------------------------------------------------------------------


def _load_probe_config(
    config_path: str | Path,
) -> tuple[ExperimentConfig, FaithfulnessProbeConfig, Path]:
    modeling_config, resolved_path = load_experiment_config(config_path)
    raw_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
    probe_lifecycle = FaithfulnessProbeConfig.model_validate(raw_payload)
    validate_experiment_id(probe_lifecycle.experiment.experiment_id)
    return modeling_config, probe_lifecycle, resolved_path


def _build_split_masks(
    frame: pd.DataFrame,
    *,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean train/test masks for a final-week frame (one row per student)."""
    n = len(frame)
    split_base = frame[["student_id"]].copy().reset_index(drop=True)
    split_base["_row"] = range(n)
    split = student_group_split(
        split_base,
        test_size=test_size,
        seed=seed,
        student_id_column="student_id",
    )
    train_indices = split.train["_row"].to_numpy()
    test_indices = split.test["_row"].to_numpy()
    train_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)
    train_mask[train_indices] = True
    test_mask[test_indices] = True
    return train_mask, test_mask


def _extract_importance_order(global_explanation: dict[str, Any]) -> list[str]:
    """Return feature names sorted best-first by permutation importance."""
    rows = sorted(
        global_explanation["rows"],
        key=lambda r: -(float(r.get("importance_share_basis_value") or 0.0)),
    )
    return [r["feature"] for r in rows]


def _run_oracle_probe(
    frame: pd.DataFrame,
    *,
    oracle_features: list[str],
    oracle_weights: dict[str, float],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model_name: str,
    seed: int,
    permutation_repeats: int,
) -> dict[str, Any]:
    feature_set = _SimpleFeatureSet(
        name="oracle_4",
        columns=tuple(oracle_features),
        indicator_columns=(),
    )
    run = train_regression_reference_for_matrix(
        frame,
        feature_set=feature_set,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name=model_name,
        seed=seed,
        split_metadata={"strategy": "student_group", "probe": "oracle"},
    )
    global_explanation = build_global_explanation(
        run,
        permutation_repeats=permutation_repeats,
        seed=seed,
        split_strategy="student_group",
    )

    # Oracle-weight order (descending weight; tie-break alphabetically)
    oracle_order = [
        f
        for f, _ in sorted(
            oracle_weights.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
    ]

    # Importance order from the model
    importance_order = _extract_importance_order(global_explanation)

    tau = kendall_tau(importance_order, oracle_order)

    # Per-feature table
    rows_by_feature = {r["feature"]: r for r in global_explanation["rows"]}
    per_feature = []
    for feat in oracle_features:
        row = rows_by_feature.get(feat, {})
        per_feature.append(
            {
                "feature": feat,
                "oracle_weight": oracle_weights.get(feat),
                "oracle_rank": oracle_order.index(feat) + 1
                if feat in oracle_order
                else None,
                "importance_share": row.get("importance_share"),
                "importance_rank": row.get("rank"),
                "mean_rmse_increase": row.get("mean_rmse_increase"),
            }
        )

    return {
        "oracle_order": oracle_order,
        "importance_order": importance_order,
        "kendall_tau": tau,
        "per_feature": per_feature,
        "model_metrics": run.metrics,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "note": (
            "Ties at oracle_weight=0.10 (attendance_rate/on_time) make exact "
            "tail ordering ambiguous. Permutation importance reflects "
            "weight x feature-spread, not weight alone; partial recovery is "
            "expected and is the methodological point."
        ),
    }


def _run_proxy_probe(
    frame: pd.DataFrame,
    *,
    proxy_feature_set_name: str,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    model_name: str,
    seed: int,
    permutation_repeats: int,
    proxy_feature: str = "activity_score_to_date",
) -> dict[str, Any]:
    feature_set = get_feature_set(proxy_feature_set_name)
    run = train_regression_reference_for_matrix(
        frame,
        feature_set=feature_set,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name=model_name,
        seed=seed,
        split_metadata={"strategy": "student_group", "probe": "proxy"},
    )
    global_explanation = build_global_explanation(
        run,
        permutation_repeats=permutation_repeats,
        seed=seed,
        split_strategy="student_group",
    )

    rows_by_feature = {r["feature"]: r for r in global_explanation["rows"]}
    proxy_row = rows_by_feature.get(proxy_feature, {})
    return {
        "feature_set": proxy_feature_set_name,
        "proxy_feature": proxy_feature,
        "proxy_importance_share": proxy_row.get("importance_share"),
        "proxy_importance_rank": proxy_row.get("rank"),
        "proxy_mean_rmse_increase": proxy_row.get("mean_rmse_increase"),
        "top_features": [r["feature"] for r in global_explanation["rows"][:5]],
        "model_metrics": run.metrics,
        "framing": (
            f"{proxy_feature} enters base_score at +0.04 (submissions.py:83) "
            "as a latent-driven proxy under correlation. Importance attributed "
            "to it reflects proxy-correlation importance, not a direct "
            "formula coefficient."
        ),
    }


def _run_redundancy_diagnostic(
    frame: pd.DataFrame,
    *,
    redundancy_pairs: list[list[str]],
) -> list[dict[str, Any]]:
    results = []
    for pair in redundancy_pairs:
        if len(pair) != 2:
            continue
        col_a, col_b = pair
        if col_a not in frame.columns or col_b not in frame.columns:
            results.append(
                {
                    "feature_a": col_a,
                    "feature_b": col_b,
                    "pearson_r": None,
                    "note": "One or both columns missing from frame.",
                }
            )
            continue
        valid = frame[[col_a, col_b]].dropna()
        if len(valid) < 2:
            r_val = None
        else:
            r_val = float(valid[col_a].corr(valid[col_b]))
        results.append(
            {
                "feature_a": col_a,
                "feature_b": col_b,
                "pearson_r": r_val,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_faithfulness_probe(
    config_path: str | Path,
    *,
    allow_existing: bool = False,
    artifact_root: Path | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the exp_008 faithfulness probe and write artifacts."""

    modeling_config, lifecycle, resolved_config_path = _load_probe_config(config_path)
    experiment_id = lifecycle.experiment.experiment_id
    probe = lifecycle.probe

    output_dir = modeling_config.resolve_path(modeling_config.outputs.experiments_dir)
    expected_dir = resolve_experiment_artifact_dir(experiment_id, root=artifact_root)
    if output_dir != expected_dir:
        raise ValueError(
            "Probe output directory must match the experiment ID. "
            f"Expected {expected_dir}, got {output_dir}."
        )

    ensure_new_experiment_dir(output_dir, allow_existing=allow_existing)

    # Load dataset and filter to final week
    dataset = load_modeling_dataset(modeling_config)
    full_frame = dataset.frame

    final_week = probe.final_week if probe.final_week is not None else int(
        full_frame[WEEK_COLUMN].max()
    )
    frame = full_frame[full_frame[WEEK_COLUMN] == final_week].reset_index(drop=True)

    if frame.empty:
        raise ValueError(f"No rows found for week_number == {final_week}")

    # Build train/test masks
    split_config = modeling_config.splits.student_group
    train_mask, test_mask = _build_split_masks(
        frame,
        test_size=split_config.test_size,
        seed=probe.seed,
    )

    # Oracle probe
    oracle_result = _run_oracle_probe(
        frame,
        oracle_features=probe.oracle_features,
        oracle_weights=probe.oracle_weights,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name=probe.model,
        seed=probe.seed,
        permutation_repeats=probe.permutation_repeats,
    )

    # Proxy probe
    proxy_result = _run_proxy_probe(
        frame,
        proxy_feature_set_name=probe.proxy_feature_set,
        train_mask=train_mask,
        test_mask=test_mask,
        model_name=probe.model,
        seed=probe.seed,
        permutation_repeats=probe.permutation_repeats,
    )

    # Redundancy diagnostic
    redundancy_result = _run_redundancy_diagnostic(frame, redundancy_pairs=probe.redundancy_pairs)

    results = {
        "experiment_id": experiment_id,
        "final_week": final_week,
        "n_final_week_rows": len(frame),
        "n_students": int(frame["student_id"].nunique()),
        "oracle_probe": oracle_result,
        "proxy_probe": proxy_result,
        "redundancy": redundancy_result,
        "method_notes": {
            "shap_used": False,
            "comparison_method": "Kendall tau on importance rank order vs oracle weight order",
            "importance_method": "sklearn permutation importance (neg_root_mean_squared_error)",
            "split": "student_group",
        },
    }

    # Write artifacts
    artifact_paths = _write_probe_artifacts(
        output_dir=output_dir,
        experiment_id=experiment_id,
        results=results,
        modeling_config=modeling_config,
        lifecycle=lifecycle,
        config_path=resolved_config_path,
    )

    # Docs
    experiment_docs_dir = docs_dir or DOCS_EXPERIMENTS_DIR
    docs_path = experiment_docs_dir / f"{experiment_id}.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    summary_markdown = render_probe_markdown(
        lifecycle=lifecycle,
        results=results,
        modeling_config=modeling_config,
        config_path=resolved_config_path,
        artifact_paths=artifact_paths,
    )
    docs_path.write_text(summary_markdown, encoding="utf-8")
    artifact_paths["docs"] = docs_path
    artifact_paths["summary_markdown"].write_text(summary_markdown, encoding="utf-8")

    # Metadata
    metadata = _build_probe_metadata(
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

    # Registry
    registry_path = experiment_docs_dir / "registry.md"
    tau_val = oracle_result.get("kendall_tau")
    conclusion = (
        f"Oracle-ordering Kendall tau = {tau_val:.3f}" if tau_val is not None else "tau=n/a"
    )
    upsert_registry_entry(
        registry_path,
        RegistryEntry(
            experiment_id=experiment_id,
            title=lifecycle.experiment.title,
            status=lifecycle.experiment.status,
            schema_version=f"v{lifecycle.experiment.schema_version}",
            dataset_config=lifecycle.experiment.dataset_config_name,
            primary_target=REGRESSION_TARGET,
            artifact_dir=_markdown_relative_path(registry_path.parent, output_dir),
            doc_path=_markdown_relative_path(registry_path.parent, docs_path),
            conclusion=conclusion,
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


# ---------------------------------------------------------------------------
# Artifact writers and renderers
# ---------------------------------------------------------------------------


def _write_probe_artifacts(
    *,
    output_dir: Path,
    experiment_id: str,
    results: dict[str, Any],
    modeling_config: ExperimentConfig,
    lifecycle: FaithfulnessProbeConfig,
    config_path: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {
        "results_json": output_dir / f"{experiment_id}_results.json",
        "summary_markdown": output_dir / f"{experiment_id}_summary.md",
    }
    paths["results_json"].write_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    # summary written by caller after docs_path is assembled
    return paths


def _build_probe_metadata(
    *,
    modeling_config: ExperimentConfig,
    lifecycle: FaithfulnessProbeConfig,
    results: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    artifact_paths: dict[str, Path],
    docs_path: Path,
) -> dict[str, Any]:
    dataset = modeling_config.dataset
    probe = lifecycle.probe
    return {
        "experiment_id": lifecycle.experiment.experiment_id,
        "title": lifecycle.experiment.title,
        "created_at": utc_now_iso(),
        "schema_version": lifecycle.experiment.schema_version,
        "dataset": {
            "snapshots_path": str(modeling_config.resolve_path(dataset.snapshots_csv)),
            "final_results_path": str(modeling_config.resolve_path(dataset.final_results_csv)),
            "dataset_version": lifecycle.experiment.dataset_version,
            "dataset_config_name": lifecycle.experiment.dataset_config_name,
        },
        "probe": probe.model_dump(),
        "final_week": results["final_week"],
        "n_final_week_rows": results["n_final_week_rows"],
        "oracle_kendall_tau": results["oracle_probe"].get("kendall_tau"),
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


def render_probe_markdown(
    *,
    lifecycle: FaithfulnessProbeConfig,
    results: dict[str, Any],
    modeling_config: ExperimentConfig,
    config_path: Path,
    artifact_paths: dict[str, Path],
) -> str:
    exp = lifecycle.experiment
    oracle = results["oracle_probe"]
    proxy = results["proxy_probe"]
    redundancy = results["redundancy"]
    tau = oracle.get("kendall_tau")
    tau_str = f"{tau:.4f}" if tau is not None else "n/a"

    lines: list[str] = [
        f"# {exp.experiment_id}: {exp.title}",
        "",
        "> **Methods appendix.** This is a controlled probe on *known* ground truth.",
        "> It is never primary evidence about learning, and recovering the importance",
        "> ordering of a formula we designed is a controlled instrument, not a claim",
        "> about real student outcomes.",
        "",
        "## Objective",
        "",
        lifecycle.documentation.objective,
        "",
        "## Dataset / config",
        "",
        f"- Schema version: `v{exp.schema_version}`",
        f"- Dataset version: `{exp.dataset_version}` / `{exp.dataset_config_name}`",
        f"- Config: `{relative_repo_path(config_path)}`",
        f"- Final week used: `{results['final_week']}` "
        f"({results['n_final_week_rows']} rows, {results['n_students']} students)",
        "",
        "## Oracle-recovery probe",
        "",
        (
            "Feature set: exactly the 4 oracle features. "
            "Comparison: importance rank order vs oracle-weight rank order via Kendall tau."
        ),
        "",
        f"**Oracle-ordering Kendall tau: `{tau_str}`**",
        "",
        "| feature | oracle weight | oracle rank | importance share | importance rank |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for pf in oracle["per_feature"]:
        ow = pf.get("oracle_weight")
        imp_share = pf.get("importance_share")
        lines.append(
            "| {feat} | {ow} | {or_} | {imp} | {ir} |".format(
                feat=pf["feature"],
                ow=f"{ow:.2f}" if ow is not None else "n/a",
                or_=pf.get("oracle_rank", "n/a"),
                imp=f"{imp_share:.3f}" if imp_share is not None else "n/a",
                ir=pf.get("importance_rank", "n/a"),
            )
        )

    lines.extend(
        [
            "",
            f"Oracle weight order (desc): `{oracle['oracle_order']}`",
            f"Importance order (best-first): `{oracle['importance_order']}`",
            "",
            f"*Note:* {oracle['note']}",
            "",
            "## Proxy demonstration (B_lms feature set)",
            "",
            (
                f"`{proxy['proxy_feature']}` importance share: "
                f"`{_fmt(proxy.get('proxy_importance_share'))}` "
                f"(rank {proxy.get('proxy_importance_rank', 'n/a')} "
                f"of {len(get_feature_set(proxy['feature_set']).columns)} features)."
            ),
            "",
            proxy["framing"],
            "",
            "## Redundancy diagnostic",
            "",
            "| feature A | feature B | Pearson r |",
            "| --- | --- | ---: |",
        ]
    )
    for rd in redundancy:
        r_val = rd.get("pearson_r")
        lines.append(
            f"| {rd['feature_a']} | {rd['feature_b']} | "
            f"{f'{r_val:.4f}' if r_val is not None else 'n/a'} |"
        )

    lines.extend(
        [
            "",
            "## Explicit caveats",
            "",
            (
                "- This probe uses a synthetic dataset where `final_grade` is a deterministic "
                "function of the 4 oracle features. Recovering the importance ordering is "
                "a controlled instrument; it cannot be generalized to claims about real learning."
            ),
            (
                "- Permutation importance reflects weight multiplied by feature spread "
                "(variance × model sensitivity), not oracle weight alone. Partial or "
                "imperfect rank recovery is expected and is the methodological point."
            ),
            (
                "- Ties in oracle weight (both attendance_rate and on_time at 0.10) "
                "make exact tail ordering ambiguous."
            ),
            "- SHAP: `false` (not a declared project dependency).",
            "",
            "## Limitations",
            "",
        ]
    )
    for lim in lifecycle.documentation.limitations:
        lines.append(f"- {lim}")

    lines.extend(["", "## Next step", "", lifecycle.documentation.next_step, ""])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and value != value):
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the controlled XAI-faithfulness probe (exp_008)."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_008_faithfulness_probe.yaml",
        help="Path to the probe YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rerunning into an existing experiment directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_faithfulness_probe(args.config, allow_existing=args.overwrite)
    paths = payload["artifact_paths"]
    results = payload["results"]
    tau = results["oracle_probe"].get("kendall_tau")
    print(f"Experiment: {results['experiment_id']}")
    print(f"Final week: {results['final_week']}  rows: {results['n_final_week_rows']}")
    print(f"Oracle-ordering Kendall tau: {tau:.4f}" if tau is not None else "Oracle tau: n/a")
    print(f"Docs: {payload['docs_path']}")
    print(f"Results JSON: {paths['results_json']}")
    print(f"Registry: {payload['registry_path']}")


if __name__ == "__main__":
    main()
