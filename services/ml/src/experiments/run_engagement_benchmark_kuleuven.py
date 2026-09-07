"""KU Leuven engagement-only PASSED classification benchmark runner.

exp_011_kuleuven_engagement — THIRD institution, engagement-only.

SCOPE AND HONEST FRAMING
-------------------------
KU Leuven (year 1819) has clickstream logs and a binary PASSED outcome, but
NO numeric intermediate assessment scores and NO meaningful continuous grade.
Only binary PASSED classification is reported. Regression is NOT applicable.
The mastery/Twin feature ablation from OULAD experiments CANNOT be reproduced
here. The ``final_grade`` column in the snapshot frame is set to
``float(passed)`` purely as a structural placeholder for the shared pipeline
(which requires a numeric ``final_grade`` column). Any regression metric
computed on that column would be meaningless and is intentionally omitted.

This runner mirrors the structure of ``run_public_benchmark_oulad.py`` but
removes all regression concerns and adds permutation importance computation
for the candidate feature set (B_engagement) on both split strategies.

The dataset-agnostic methodology (config dataclasses, split builder,
classification + permutation importance, diagnostics, artifact writers, and the
markdown renderer) lives in ``src.experiments.engagement_benchmark`` so the
OULAD runner can reuse the IDENTICAL pipeline. This module keeps only the
KU-Leuven-specific wiring: the KU config models, config loading, snapshot
building (via the KU Leuven adapter), experiment-metadata assembly, and the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from src.benchmarks.ku_leuven_adapter import (
    KuLeuvenSnapshotBuildResult,
    build_weekly_engagement_snapshots,
)
from src.experiments.config import ClassificationModelName
from src.experiments.engagement_benchmark import (
    BenchmarkFeatureSet,
    ComparisonConfig,
    DocumentationConfig,
    FeatureSetConfig,
    OutputsConfig,
    SplitsConfig,
    SplitStrategy,
    _markdown_relative_path,
    _train_and_score_classification,
    _validate_feature_columns,
    _write_importance_csv,
    _write_result_artifacts,
    build_benchmark_diagnostics,
    build_split,
    compute_permutation_importance,
    render_experiment_markdown,
)
from src.experiments.evaluate import ResultRow
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
from src.experiments.preprocessing import build_modeling_matrix
from src.generator.config import REPO_ROOT, SERVICE_ROOT


# ---------------------------------------------------------------------------
# Pydantic config (KU-Leuven-specific)
# ---------------------------------------------------------------------------


class KuLeuvenDataConfig(BaseModel):
    data_dir: Path = Path("datasets/ku_leuven/dataset")
    course_info_path: Path = Path("datasets/ku_leuven/course_info.json")
    year: str = "1819"
    min_week: int = 2


class KuLeuvenEngagementConfig(BaseModel):
    experiment_id: str
    title: str
    schema_version: str = "ku_leuven_engagement_v1"
    dataset_version: str = "KU Leuven 2026"
    dataset_config_name: str = "local KU Leuven dataset files"
    parent_experiment: str | None = "exp_010_xai_on_oulad_bbb2013j"
    status: str = "completed"
    documentation: DocumentationConfig
    data: KuLeuvenDataConfig = Field(default_factory=KuLeuvenDataConfig)
    feature_sets: dict[str, FeatureSetConfig]
    feature_set_order: list[str]
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    classification_models: list[ClassificationModelName] = Field(default_factory=list)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    seed: int = 42
    outputs: OutputsConfig

    @model_validator(mode="after")
    def _validate(self) -> "KuLeuvenEngagementConfig":
        validate_experiment_id(self.experiment_id)
        missing = [name for name in self.feature_set_order if name not in self.feature_sets]
        if missing:
            raise ValueError(f"feature_set_order contains unknown feature sets: {missing}")
        if self.comparison.baseline_feature_set not in self.feature_sets:
            raise ValueError("comparison.baseline_feature_set is not configured")
        if self.comparison.candidate_feature_set not in self.feature_sets:
            raise ValueError("comparison.candidate_feature_set is not configured")
        if not self.classification_models:
            raise ValueError("classification_models must not be empty")
        # Enforce output dir matches experiment id
        expected_dir = resolve_experiment_artifact_dir(self.experiment_id)
        resolved_out = self.resolve_path(self.outputs.experiments_dir)
        if resolved_out != expected_dir:
            raise ValueError(
                "Output directory must match the experiment ID. "
                f"Expected {expected_dir}, got {resolved_out}."
            )
        return self

    def resolve_path(self, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (REPO_ROOT / path).resolve()

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
# Config loading
# ---------------------------------------------------------------------------


def load_kuleuven_engagement_config(
    config_path: str | Path,
) -> tuple[KuLeuvenEngagementConfig, Path]:
    resolved = _resolve_config_path(config_path)
    raw_payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return KuLeuvenEngagementConfig.model_validate(raw_payload), resolved


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------


def run_kuleuven_engagement_benchmark(
    config_path: str | Path,
    *,
    allow_existing: bool = False,
    artifact_root: Path | None = None,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    config, resolved_config_path = load_kuleuven_engagement_config(config_path)
    experiment_id = config.experiment_id
    output_dir = config.resolve_path(config.outputs.experiments_dir)

    ensure_new_experiment_dir(output_dir, allow_existing=allow_existing)

    # Build snapshots from KU Leuven raw data
    data_dir = config.resolve_path(config.data.data_dir)
    course_info_path = config.resolve_path(config.data.course_info_path)
    build_result = build_weekly_engagement_snapshots(
        data_dir=data_dir,
        course_info_path=course_info_path,
        year=config.data.year,
        min_week=config.data.min_week,
    )

    # The snapshot frame already has final_grade (placeholder) + passed + student_id + week_number
    modeling_frame = build_result.snapshots.copy()

    feature_sets = config.build_feature_sets()
    _validate_feature_columns(modeling_frame, feature_sets)

    strategies: list[SplitStrategy] = ["student_group", "temporal_forward"]

    # --- main model loop ---
    rows: list[ResultRow] = []
    for feature_set in feature_sets:
        matrix = build_modeling_matrix(modeling_frame, feature_set)  # type: ignore[arg-type]
        for strategy in strategies:
            partition = build_split(matrix, strategy, config)
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

    # --- permutation importance on candidate feature set ---
    candidate_fs = next(
        (fs for fs in feature_sets if fs.name == config.comparison.candidate_feature_set),
        None,
    )
    importance_by_split: dict[str, dict[str, Any]] = {}
    if candidate_fs is not None:
        candidate_matrix = build_modeling_matrix(modeling_frame, candidate_fs)  # type: ignore[arg-type]
        for strategy in strategies:
            partition = build_split(candidate_matrix, strategy, config)
            if partition.train_mask.sum() == 0 or partition.test_mask.sum() == 0:
                continue
            importance_by_split[strategy] = compute_permutation_importance(
                candidate_matrix,
                partition=partition,
                config=config,
            )

    # --- write artifacts ---
    result_paths = _write_result_artifacts(
        rows,
        output_dir=output_dir,
        run_name=experiment_id,
        config=config,
        build_result=build_result,
    )
    diagnostics = build_benchmark_diagnostics(rows, config, build_result, importance_by_split)

    # Write permutation importance CSV
    importance_csv_path = output_dir / "engagement_importance.csv"
    _write_importance_csv(importance_by_split, importance_csv_path)

    # Write summary markdown
    summary_md = render_experiment_markdown(
        config=config,
        diagnostics=diagnostics,
        build_result=build_result,
        result_paths=result_paths,
        config_path=resolved_config_path,
        importance_csv_path=importance_csv_path,
    )
    artifact_summary_path = result_paths["markdown"]
    artifact_summary_path.write_text(summary_md, encoding="utf-8")

    experiment_docs_dir = docs_dir or DOCS_EXPERIMENTS_DIR
    docs_path = experiment_docs_dir / f"{experiment_id}.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(summary_md, encoding="utf-8")

    metadata = _build_experiment_metadata(
        config=config,
        diagnostics=diagnostics,
        build_result=build_result,
        result_paths=result_paths,
        config_path=resolved_config_path,
        output_dir=output_dir,
        docs_path=docs_path,
        importance_csv_path=importance_csv_path,
    )
    metadata_path = write_experiment_metadata(metadata, output_dir=output_dir)

    registry_path = experiment_docs_dir / "registry.md"
    upsert_registry_entry(
        registry_path,
        RegistryEntry(
            experiment_id=experiment_id,
            title=config.title,
            status=config.status,
            schema_version=config.schema_version,
            dataset_config=config.dataset_config_name,
            primary_target="passed (classification only)",
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
        "importance_csv_path": importance_csv_path,
    }


# ---------------------------------------------------------------------------
# Experiment metadata (KU-Leuven-specific)
# ---------------------------------------------------------------------------


def _build_experiment_metadata(
    *,
    config: KuLeuvenEngagementConfig,
    diagnostics: dict[str, Any],
    build_result: KuLeuvenSnapshotBuildResult,
    result_paths: dict[str, Path],
    config_path: Path,
    output_dir: Path,
    docs_path: Path,
    importance_csv_path: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "title": config.title,
        "created_at": utc_now_iso(),
        "schema_version": config.schema_version,
        "dataset": {
            "dataset_version": config.dataset_version,
            "dataset_config_name": config.dataset_config_name,
            "source": "KU Leuven 2026 dataset (year 1819)",
            "data_dir": relative_repo_path(config.resolve_path(config.data.data_dir)),
            "course_info_path": relative_repo_path(
                config.resolve_path(config.data.course_info_path)
            ),
            "row_counts": build_result.row_counts,
        },
        "targets": {
            "primary": "passed",
            "note": (
                "Classification only. final_grade is a placeholder = float(passed). "
                "Regression NOT applicable."
            ),
        },
        "split_strategies": {
            "student_group": config.splits.student_group.model_dump(),
            "temporal_forward": config.splits.temporal_forward.model_dump(),
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
        "models": {
            "classification": list(config.classification_models),
            "regression": "NOT APPLICABLE — final_grade is a placeholder",
        },
        "seed": config.seed,
        "config_path": relative_repo_path(config_path),
        "output_directory": relative_repo_path(output_dir),
        "status": config.status,
        "parent_experiment": config.parent_experiment,
        "artifacts": {
            "results_csv": relative_repo_path(result_paths["csv"]),
            "results_json": relative_repo_path(result_paths["json"]),
            "results_markdown": relative_repo_path(result_paths["markdown"]),
            "engagement_importance_csv": relative_repo_path(importance_csv_path),
            "documentation": relative_repo_path(docs_path),
        },
        "diagnostics": diagnostics,
    }


# ---------------------------------------------------------------------------
# Private utilities (KU-Leuven-specific)
# ---------------------------------------------------------------------------


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
        f"Could not resolve KU Leuven engagement config path: {config_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the KU Leuven engagement-only PASSED classification benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_011_kuleuven_engagement.yaml",
        help="Path to the KU Leuven engagement YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rerunning into an existing experiment directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_kuleuven_engagement_benchmark(args.config, allow_existing=args.overwrite)
    print(f"Experiment metadata: {payload['metadata_path']}")
    print(f"Experiment docs: {payload['docs_path']}")
    print(f"Results CSV: {payload['result_paths']['csv']}")
    print(f"Importance CSV: {payload['importance_csv_path']}")


if __name__ == "__main__":
    main()
