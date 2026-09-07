"""OULAD model-behavior XAI runner.

`exp_007_xai_on_oulad` characterizes which features drive the OULAD regression
model and how concentrated the explanations are.  It uses permutation importance,
model-native importance, and local one-feature-at-a-time perturbation — no SHAP.

Key design constraints
----------------------
- Both splits (student_group, temporal_forward) are run; headline = temporal_forward.
- Feature sets explained: B_lms_oulad, B_lms_plus_mastery_oulad, C_twin_oulad.
- These are model-BEHAVIOR explanations, never structural-coefficient or causal claims.
- The OULAD assessment-score partial circularity (cumulative_assessment_weighted_score_to_date
  feeds the target) is disclosed; interpretation is weighted toward exogenous clickstream
  features.
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
from pydantic import BaseModel, Field, model_validator

from src.benchmarks.oulad_adapter import (
    OuladCourseFilter,
    OuladRawPaths,
    build_weekly_snapshots,
    validate_oulad_raw_files,
)
from src.experiments.explainability import (
    build_global_explanation,
    compute_local_perturbation_contributions,
    train_regression_reference_for_matrix,
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
from src.experiments.run_public_benchmark_oulad import BenchmarkFeatureSet
from src.experiments.splits import student_group_split
from src.generator.config import REPO_ROOT, SERVICE_ROOT


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class XaiExperimentIdentityConfig(BaseModel):
    experiment_id: str
    title: str
    schema_version: str = "external_oulad_adapter_v1"
    dataset_version: str = "OULAD"
    dataset_config_name: str = "local OULAD CSV files"
    parent_experiment: str | None = None
    status: str = "completed"


class XaiDocumentationConfig(BaseModel):
    objective: str
    hypothesis: str
    limitations: list[str] = Field(default_factory=list)
    next_step: str = ""


class XaiOuladFilesConfig(BaseModel):
    assessments: Path = Path("assessments.csv")
    courses: Path = Path("courses.csv")
    student_info: Path = Path("studentInfo.csv")
    student_registration: Path = Path("studentRegistration.csv")
    student_vle: Path = Path("studentVle.csv")
    vle: Path = Path("vle.csv")
    student_assessment: Path = Path("studentAssessment.csv")


class XaiCourseFilterConfig(BaseModel):
    code_module: str | None = None
    code_presentation: str | None = None
    rationale: str | None = None


class XaiOuladConfig(BaseModel):
    raw_dir: Path = Path("datasets/oulad")
    files: XaiOuladFilesConfig = Field(default_factory=XaiOuladFilesConfig)
    snapshots_csv: Path = Path("snapshots.csv")
    course_filter: XaiCourseFilterConfig = Field(default_factory=XaiCourseFilterConfig)
    min_week: int = 4
    max_week: int | None = None
    student_vle_chunk_size: int = 500_000

    @model_validator(mode="after")
    def _validate(self) -> "XaiOuladConfig":
        if self.min_week < 1:
            raise ValueError("oulad.min_week must be >= 1")
        if self.max_week is not None and self.max_week < self.min_week:
            raise ValueError("oulad.max_week must be >= min_week")
        return self


class XaiFeatureSetConfig(BaseModel):
    description: str
    columns: list[str]
    indicator_columns: list[str] = Field(default_factory=list)


class XaiStudentGroupSplitConfig(BaseModel):
    test_size: float = 0.25
    seed: int = 42


class XaiTemporalForwardSplitConfig(BaseModel):
    train_weeks: int = 20
    student_test_size: float = 0.25
    student_seed: int = 42


class XaiSplitsConfig(BaseModel):
    student_group: XaiStudentGroupSplitConfig = Field(
        default_factory=XaiStudentGroupSplitConfig
    )
    temporal_forward: XaiTemporalForwardSplitConfig = Field(
        default_factory=XaiTemporalForwardSplitConfig
    )


class XaiOutputsConfig(BaseModel):
    experiments_dir: Path


class XaiOnOuladConfig(BaseModel):
    experiment: XaiExperimentIdentityConfig
    documentation: XaiDocumentationConfig
    oulad: XaiOuladConfig
    feature_sets: dict[str, XaiFeatureSetConfig]
    feature_set_order: list[str]
    splits: XaiSplitsConfig = Field(default_factory=XaiSplitsConfig)
    model: str = "gradient_boosting"
    permutation_repeats: int = 15
    seed: int = 42
    outputs: XaiOutputsConfig

    @model_validator(mode="after")
    def _validate(self) -> "XaiOnOuladConfig":
        validate_experiment_id(self.experiment.experiment_id)
        missing = [
            name for name in self.feature_set_order if name not in self.feature_sets
        ]
        if missing:
            raise ValueError(
                f"feature_set_order contains unknown feature sets: {missing}"
            )
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


# ---------------------------------------------------------------------------
# Mask helper
# ---------------------------------------------------------------------------


def _split_masks(
    frame: pd.DataFrame,
    strategy: str,
    *,
    student_group: dict[str, Any],
    temporal_forward: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return (train_mask, test_mask, split_metadata) aligned to frame rows.

    student_group: hold out a set of students (test); train on the rest.
    temporal_forward: hold out a set of students AND require test week > train_weeks;
    train on non-held-out students with week <= train_weeks.  Mirrors the OULAD
    benchmark _build_split logic.  Leakage-safe: no held-out student appears in train.
    """
    n = len(frame)
    sid = frame["student_id"].to_numpy()
    week = pd.to_numeric(frame["week_number"], errors="coerce").to_numpy()
    base = pd.DataFrame({"student_id": sid, "_row": np.arange(n)})

    if strategy == "student_group":
        split = student_group_split(
            base,
            test_size=student_group["test_size"],
            seed=student_group["seed"],
            student_id_column="student_id",
        )
        test_students = set(split.test["student_id"].tolist())
        test_mask = np.array([s in test_students for s in sid])
        train_mask = ~test_mask
        meta: dict[str, Any] = {
            "strategy": "student_group",
            **student_group,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }

    elif strategy == "temporal_forward":
        split = student_group_split(
            base,
            test_size=temporal_forward["student_test_size"],
            seed=temporal_forward["student_seed"],
            student_id_column="student_id",
        )
        held = set(split.test["student_id"].tolist())
        tw = temporal_forward["train_weeks"]
        is_held = np.array([s in held for s in sid])
        train_mask = (~is_held) & (week <= tw)
        test_mask = is_held & (week > tw)
        meta = {
            "strategy": "temporal_forward",
            **temporal_forward,
            "held_out_students": len(held),
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
        }

    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    return train_mask, test_mask, meta


# ---------------------------------------------------------------------------
# Local explanation helper
# ---------------------------------------------------------------------------


def _local_explanations_for_run(
    run: Any,
    *,
    top_k: int = 6,
    n_quantile_cases: int = 5,
) -> list[dict[str, Any]]:
    """Pick up to n_quantile_cases held-out rows spanning predicted-grade quantiles."""
    preds = run.predictions
    if len(preds) == 0:
        return []

    quantile_positions_raw = np.linspace(0, len(preds) - 1, n_quantile_cases)
    sorted_positions = np.argsort(preds)
    row_positions_raw = sorted_positions[np.round(quantile_positions_raw).astype(int)]
    # deduplicate preserving order
    seen: set[int] = set()
    row_positions: list[int] = []
    for pos in row_positions_raw.tolist():
        if pos not in seen:
            seen.add(pos)
            row_positions.append(int(pos))

    local_records: list[dict[str, Any]] = []
    for pos in row_positions:
        contributions = compute_local_perturbation_contributions(run, row_position=pos)
        top_contribs = contributions[:top_k]
        local_records.append(
            {
                "row_position": pos,
                "predicted_grade": float(preds[pos]),
                "actual_grade": float(run.y_test[pos]),
                "top_contributions": top_contribs,
            }
        )
    return local_records


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_xai_on_oulad(
    config_path: str | Path,
    *,
    allow_existing: bool = False,
    artifact_root: Path | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the OULAD XAI experiment and write all artifacts."""

    config, resolved_config_path = _load_config(config_path)
    experiment_id = config.experiment.experiment_id
    output_dir = config.resolve_path(config.outputs.experiments_dir)
    expected_dir = resolve_experiment_artifact_dir(experiment_id, root=artifact_root)
    if output_dir != expected_dir:
        raise ValueError(
            "XAI output directory must match the experiment ID. "
            f"Expected {expected_dir}, got {output_dir}."
        )

    ensure_new_experiment_dir(output_dir, allow_existing=allow_existing)

    raw_paths = config.resolve_raw_paths()
    validate_oulad_raw_files(raw_paths)

    snapshots_path = config.resolve_path(config.oulad.snapshots_csv)
    if not snapshots_path.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError(
            "oulad.snapshots_csv must live inside the experiment artifact directory. "
            f"Got {snapshots_path}."
        )

    # ------------------------------------------------------------------
    # 1. Build OULAD snapshots
    # ------------------------------------------------------------------
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

    frame = build_result.snapshots.copy()
    frame["final_grade"] = pd.to_numeric(
        frame["final_weighted_score"], errors="coerce"
    )
    frame["passed"] = (
        pd.to_numeric(frame["passed_observed"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    frame = frame.loc[frame["final_grade"].notna()].reset_index(drop=True)

    snapshots_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(snapshots_path, index=False)

    # ------------------------------------------------------------------
    # 2. Validate feature columns
    # ------------------------------------------------------------------
    feature_sets = config.build_feature_sets()
    _validate_feature_columns(frame, feature_sets)

    # ------------------------------------------------------------------
    # 3. Run XAI per (feature_set, split_strategy)
    # ------------------------------------------------------------------
    splits_cfg = config.splits
    student_group_params = {
        "test_size": splits_cfg.student_group.test_size,
        "seed": splits_cfg.student_group.seed,
    }
    temporal_forward_params = {
        "train_weeks": splits_cfg.temporal_forward.train_weeks,
        "student_test_size": splits_cfg.temporal_forward.student_test_size,
        "student_seed": splits_cfg.temporal_forward.student_seed,
    }

    all_records: list[dict[str, Any]] = []
    importance_rows_flat: list[dict[str, Any]] = []

    for fs in feature_sets:
        for strategy in ["student_group", "temporal_forward"]:
            train_mask, test_mask, meta = _split_masks(
                frame,
                strategy,
                student_group=student_group_params,
                temporal_forward=temporal_forward_params,
            )

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                print(
                    f"  [skip] {fs.name} / {strategy}: "
                    f"train={train_mask.sum()} test={test_mask.sum()}"
                )
                continue

            print(
                f"  [run] {fs.name} / {strategy}: "
                f"train={train_mask.sum()} test={test_mask.sum()}"
            )

            run = train_regression_reference_for_matrix(
                frame,
                feature_set=fs,
                train_mask=train_mask,
                test_mask=test_mask,
                model_name=config.model,
                seed=config.seed,
                split_metadata=meta,
            )

            g = build_global_explanation(
                run,
                permutation_repeats=config.permutation_repeats,
                seed=config.seed,
                split_strategy=strategy,
            )

            local = _local_explanations_for_run(run, top_k=6, n_quantile_cases=5)

            record: dict[str, Any] = {
                "feature_set": fs.name,
                "split_strategy": strategy,
                "metrics": run.metrics,
                "concentration": g["concentration"],
                "importance_rows": g["rows"],
                "local": local,
                "n_train_rows": int(train_mask.sum()),
                "n_test_rows": int(test_mask.sum()),
            }
            all_records.append(record)

            for row in g["rows"]:
                importance_rows_flat.append(
                    {
                        "feature_set": fs.name,
                        "split_strategy": strategy,
                        **{
                            k: v
                            for k, v in row.items()
                            if k
                            not in {
                                "feature_set",
                            }
                        },
                    }
                )

    # ------------------------------------------------------------------
    # 4. Write artifacts
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    results_json_path = output_dir / f"{experiment_id}_results.json"
    results_json_path.write_text(
        json.dumps(_to_jsonable(all_records), indent=2), encoding="utf-8"
    )

    importance_csv_path = output_dir / "global_feature_importance.csv"
    if importance_rows_flat:
        pd.DataFrame(importance_rows_flat).to_csv(importance_csv_path, index=False)
    else:
        pd.DataFrame().to_csv(importance_csv_path, index=False)

    summary_md = _render_summary(
        config=config,
        records=all_records,
        build_result=build_result,
        config_path=resolved_config_path,
        snapshots_path=snapshots_path,
    )
    summary_path = output_dir / f"{experiment_id}_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    experiment_docs_dir = docs_dir or DOCS_EXPERIMENTS_DIR
    docs_path = experiment_docs_dir / f"{experiment_id}.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(summary_md, encoding="utf-8")

    metadata = _build_metadata(
        config=config,
        records=all_records,
        build_result=build_result,
        config_path=resolved_config_path,
        output_dir=output_dir,
        docs_path=docs_path,
        snapshots_path=snapshots_path,
        results_json_path=results_json_path,
        importance_csv_path=importance_csv_path,
        summary_path=summary_path,
    )
    metadata_path = write_experiment_metadata(metadata, output_dir=output_dir)

    registry_path = experiment_docs_dir / "registry.md"
    short_conclusion = _short_conclusion(
        all_records,
        cohort_label=(
            f"{config.oulad.course_filter.code_module or ''} "
            f"{config.oulad.course_filter.code_presentation or ''}".strip()
        ),
    )
    upsert_registry_entry(
        registry_path,
        RegistryEntry(
            experiment_id=experiment_id,
            title=config.experiment.title,
            status=config.experiment.status,
            schema_version=config.experiment.schema_version,
            dataset_config=(
                f"OULAD {config.oulad.course_filter.code_module or ''} "
                f"{config.oulad.course_filter.code_presentation or ''}".strip()
            ),
            primary_target="final_weighted_score",
            artifact_dir=_markdown_relative_path(registry_path.parent, output_dir),
            doc_path=_markdown_relative_path(registry_path.parent, docs_path),
            conclusion=short_conclusion,
        ),
    )

    return {
        "records": all_records,
        "metadata_path": metadata_path,
        "docs_path": docs_path,
        "registry_path": registry_path,
        "results_json_path": results_json_path,
        "importance_csv_path": importance_csv_path,
        "summary_path": summary_path,
        "snapshots_path": snapshots_path,
    }


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------


def _render_summary(
    *,
    config: XaiOnOuladConfig,
    records: list[dict[str, Any]],
    build_result: Any,
    config_path: Path,
    snapshots_path: Path,
) -> str:
    exp = config.experiment
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
        "## XAI method note",
        "",
        (
            "These are **model-behavior explanations**, not causal explanations and not "
            "structural-coefficient interpretations.  No SHAP is used.  The explanation "
            "layer consists of:"
        ),
        "",
        "- **Permutation importance** (held-out RMSE increase when a feature is permuted, "
        f"{config.permutation_repeats} repeats).",
        "- **Model-native importance** (gradient-boosting feature importances).",
        "- **Local one-feature perturbation** (replace one feature with its training median; "
        "measure prediction change).",
        "",
        "Findings describe what the fitted model relies on, not what causes student outcomes.",
        "",
        "## OULAD assessment-score partial circularity",
        "",
        (
            "`cumulative_assessment_weighted_score_to_date` partially feeds the regression "
            "target `final_weighted_score`.  Any high importance for this feature is "
            "**expected and does not reflect an exogenous causal signal** — it is a "
            "within-system accounting identity.  Interpretation should be weighted toward "
            "**exogenous clickstream features** (`cumulative_*_clicks_to_date`, "
            "`current_week_clicks`, `assessment_submission_rate_due_to_date`) which are "
            "not part of the target construction and are genuinely behavioural signals."
        ),
        "",
        "## Dataset / config used",
        "",
        f"- Experiment config: `{relative_repo_path(config_path)}`",
        f"- Raw directory: `{relative_repo_path(config.resolve_path(config.oulad.raw_dir))}`",
        f"- Processed snapshots: `{relative_repo_path(snapshots_path)}`",
        f"- Output directory: `{relative_repo_path(config.resolve_path(config.outputs.experiments_dir))}`",
        f"- Course filter: `{build_result.course_filter}`",
        f"- Weeks: `{build_result.week_min}..{build_result.week_max}`",
        "",
        "## Feature sets explained",
        "",
    ]
    for fs in config.build_feature_sets():
        lines.extend(
            [
                f"### `{fs.name}`",
                "",
                fs.description,
                "",
                "Columns: " + ", ".join(f"`{c}`" for c in fs.columns),
            ]
        )
        if fs.indicator_columns:
            lines.append(
                "Indicator columns: "
                + ", ".join(f"`{c}`" for c in fs.indicator_columns)
            )
        lines.append("")

    lines.extend(
        [
            "## Splits",
            "",
            f"- student_group: test_size={config.splits.student_group.test_size}, seed={config.splits.student_group.seed}",
            f"- temporal_forward (headline): train_weeks={config.splits.temporal_forward.train_weeks}, student_test_size={config.splits.temporal_forward.student_test_size}, student_seed={config.splits.temporal_forward.student_seed}",
            "",
            "## Results: global feature importance",
            "",
        ]
    )

    for record in records:
        fs_name = record["feature_set"]
        strategy = record["split_strategy"]
        metrics = record["metrics"]
        conc = record["concentration"]
        rows = record["importance_rows"]

        lines.extend(
            [
                f"### {fs_name} / {strategy}",
                "",
                "| metric | value |",
                "| --- | ---: |",
                f"| RMSE | {metrics.get('rmse', float('nan')):.4f} |",
                f"| MAE | {metrics.get('mae', float('nan')):.4f} |",
                f"| R² | {metrics.get('r2', float('nan')):.4f} |",
                f"| train rows | {record['n_train_rows']} |",
                f"| test rows | {record['n_test_rows']} |",
                "",
                "**Concentration**",
                "",
                f"- Top feature: `{conc.get('top_feature')}`",
                f"- Top-1 share: {_fmt(conc.get('top1_share'))}",
                f"- Top-3 share: {_fmt(conc.get('top3_share'))}",
                f"- Herfindahl index: {_fmt(conc.get('herfindahl_index'))}",
                f"- Features with positive importance: {conc.get('features_with_positive_importance')}",
                "",
                "**Top-10 features by importance share**",
                "",
                "| rank | feature | importance_share | mean_rmse_increase | native_importance | direction |",
                "| ---: | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for r in rows[:10]:
            lines.append(
                "| {rank} | `{feat}` | {share} | {rmse} | {native} | {dir} |".format(
                    rank=r.get("rank", ""),
                    feat=r["feature"],
                    share=_fmt(r.get("importance_share")),
                    rmse=_fmt(r.get("mean_rmse_increase")),
                    native=_fmt(r.get("model_native_importance")),
                    dir=r.get("direction_note", ""),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Limitations",
            "",
        ]
    )
    for limitation in config.documentation.limitations:
        lines.append(f"- {limitation}")

    if config.documentation.next_step:
        lines.extend(
            [
                "",
                "## Next step",
                "",
                config.documentation.next_step,
            ]
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------


def _build_metadata(
    *,
    config: XaiOnOuladConfig,
    records: list[dict[str, Any]],
    build_result: Any,
    config_path: Path,
    output_dir: Path,
    docs_path: Path,
    snapshots_path: Path,
    results_json_path: Path,
    importance_csv_path: Path,
    summary_path: Path,
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
        "xai_method": {
            "shap": False,
            "permutation_importance": True,
            "model_native_importance": True,
            "local_perturbation": True,
            "permutation_repeats": config.permutation_repeats,
            "note": (
                "Model-behavior explanations only; no causal claims; "
                "no structural-coefficient interpretation."
            ),
        },
        "split_strategies": {
            "student_group": config.splits.student_group.model_dump(),
            "temporal_forward": config.splits.temporal_forward.model_dump(),
            "headline": "temporal_forward",
        },
        "feature_sets": [
            {
                "name": fs.name,
                "description": fs.description,
                "columns": list(fs.columns),
                "indicator_columns": list(fs.indicator_columns),
            }
            for fs in config.build_feature_sets()
        ],
        "model": config.model,
        "seed": config.seed,
        "config_path": relative_repo_path(config_path),
        "output_directory": relative_repo_path(output_dir),
        "status": config.experiment.status,
        "parent_experiment": config.experiment.parent_experiment,
        "artifacts": {
            "results_json": relative_repo_path(results_json_path),
            "global_feature_importance_csv": relative_repo_path(importance_csv_path),
            "summary_markdown": relative_repo_path(summary_path),
            "processed_snapshots_csv": relative_repo_path(snapshots_path),
            "documentation": relative_repo_path(docs_path),
        },
        "records_summary": [
            {
                "feature_set": r["feature_set"],
                "split_strategy": r["split_strategy"],
                "metrics": r["metrics"],
                "concentration": r["concentration"],
            }
            for r in records
        ],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_feature_columns(
    frame: pd.DataFrame,
    feature_sets: list[BenchmarkFeatureSet],
) -> None:
    missing_by_set: dict[str, list[str]] = {}
    for fs in feature_sets:
        missing = [c for c in fs.all_columns() if c not in frame.columns]
        if missing:
            missing_by_set[fs.name] = missing
    if missing_by_set:
        raise ValueError(f"XAI feature columns missing in frame: {missing_by_set}")


def _short_conclusion(records: list[dict[str, Any]], cohort_label: str) -> str:
    """One-line summary of the XAI result for the registry."""
    tf_records = [r for r in records if r["split_strategy"] == "temporal_forward"]
    if not tf_records:
        return "XAI runner completed; no temporal_forward records."
    top_features = [
        r["concentration"].get("top_feature") for r in tf_records if r.get("concentration")
    ]
    unique_tops = list(dict.fromkeys(f for f in top_features if f))
    if not unique_tops:
        return "XAI runner completed; concentration data unavailable."
    return (
        f"Model-behavior explanations for OULAD {cohort_label}; "
        f"top driver(s) on temporal_forward: {', '.join(unique_tops[:3])}."
    )


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        if isinstance(value, float) and (value != value):  # NaN check
            return "n/a"
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, float):
        if value != value:  # NaN
            return None
        return value
    return value


def _markdown_relative_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()


def _load_config(
    config_path: str | Path,
) -> tuple[XaiOnOuladConfig, Path]:
    resolved = _resolve_config_path(config_path)
    raw_payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return XaiOnOuladConfig.model_validate(raw_payload), resolved


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
    raise FileNotFoundError(
        f"Could not resolve XAI config path: {config_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OULAD model-behavior XAI experiment (exp_007)."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_007_xai_on_oulad.yaml",
        help="Path to the XAI YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rerunning into an existing experiment directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_xai_on_oulad(args.config, allow_existing=args.overwrite)
    print(f"Experiment metadata: {payload['metadata_path']}")
    print(f"Experiment docs:     {payload['docs_path']}")
    print(f"Snapshots:           {payload['snapshots_path']}")
    print(f"Results JSON:        {payload['results_json_path']}")
    print(f"Importance CSV:      {payload['importance_csv_path']}")
    print(f"Summary:             {payload['summary_path']}")


if __name__ == "__main__":
    main()
