"""OULAD engagement-only PASSED classification benchmark runner.

exp_012_oulad_engagement — matched engagement-only design on OULAD.

SCOPE AND HONEST FRAMING
-------------------------
This runner reuses the IDENTICAL classification + permutation-importance
methodology as the KU Leuven engagement benchmark (exp_011), so the two
institutions can be compared under a matched, engagement-only design.

- Classification only. Regression is intentionally omitted: this is an
  engagement-only matched comparison, and OULAD's assessment-derived score
  is deliberately not used here. The ``final_grade`` column is set to
  ``float(passed)`` purely as a structural placeholder for the shared
  pipeline (which requires a numeric ``final_grade`` column). Any regression
  metric computed on that column would be meaningless and is omitted.
- ``cumulative_social_clicks_to_date`` (forum / social CLICKS) is the OULAD
  analog of KU Leuven forum POSTS. The two are different units, so the
  cross-institution comparison relies on permutation-importance RANKS, not on
  raw importance magnitudes.

The dataset-agnostic methodology lives in
``src.experiments.engagement_benchmark`` and is shared verbatim with the KU
Leuven runner. This module keeps only the OULAD-specific wiring: the OULAD
config models, config loading, snapshot building (via the OULAD adapter), the
OULAD ``InstitutionContext``, experiment-metadata assembly, and the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from src.benchmarks import oulad_adapter
from src.experiments.config import ClassificationModelName
from src.experiments.datasets import GROUP_COLUMN, WEEK_COLUMN
from src.experiments.engagement_benchmark import (
    BenchmarkFeatureSet,
    ComparisonConfig,
    DocumentationConfig,
    FeatureSetConfig,
    InstitutionContext,
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
# OULAD InstitutionContext
# ---------------------------------------------------------------------------


_OULAD_CITATION: str = (
    "Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning "
    "Analytics dataset. Scientific Data, 4, 170171. "
    "https://doi.org/10.1038/sdata.2017.171"
)


def _oulad_institution_context(
    *,
    code_module: str,
    code_presentation: str,
    config_path: Path,
    raw_dir: Path,
) -> InstitutionContext:
    """Build the OULAD-specific markdown / JSON literals.

    Mirrors the KU Leuven framing but states the matched engagement-only design
    and the forum-CLICKS-vs-POSTS rank-comparison caveat explicitly. No KU
    Leuven text leaks into the OULAD output.
    """

    banner_lines: tuple[str, ...] = (
        "> **OULAD — ENGAGEMENT-ONLY — CLASSIFICATION ONLY (PASSED)**",
        ">",
        "> Matched engagement-only design for cross-institution comparison with the",
        "> KU Leuven engagement benchmark. Only binary PASSED classification is",
        "> reported. Regression is intentionally omitted (engagement-only matched",
        "> design); final_grade = float(passed) is a structural placeholder.",
        "> `cumulative_social_clicks_to_date` (forum/social CLICKS) is the OULAD",
        "> analog of KU Leuven forum POSTS — different units, so the cross-institution",
        "> comparison uses permutation-importance RANKS, not magnitudes.",
    )
    dataset_identity_lines: tuple[str, ...] = (
        f"- Institution: OULAD ({code_module} {code_presentation})",
        f"- Citation: {_OULAD_CITATION}",
        f"- Experiment config: `{relative_repo_path(config_path)}`",
        f"- Raw dir: `{relative_repo_path(raw_dir)}`",
        f"- Course filter: code_module={code_module}, code_presentation={code_presentation}",
    )
    json_note: str = (
        "OULAD engagement-only benchmark (matched design with KU Leuven exp_011). "
        "Classification (PASSED) only. Regression intentionally omitted "
        "(engagement-only matched design). final_grade is a placeholder equal to "
        "float(passed). cumulative_social_clicks_to_date (forum/social clicks) is "
        "the OULAD analog of KU Leuven forum posts; cross-institution comparison "
        "uses ranks, not magnitudes."
    )
    return InstitutionContext(
        banner_lines=banner_lines,
        dataset_identity_lines=dataset_identity_lines,
        json_note=json_note,
    )


# ---------------------------------------------------------------------------
# Modeling frame builder
# ---------------------------------------------------------------------------


def build_oulad_engagement_frame(
    code_module: str,
    code_presentation: str,
    min_week: int,
    max_week: int | None = None,
    raw_dir: str | Path = "datasets/oulad",
) -> tuple[pd.DataFrame, oulad_adapter.OuladSnapshotBuildResult]:
    """Build the OULAD modeling frame the shared engagement core expects.

    Produces weekly OULAD snapshots via the OULAD adapter, then maps them onto
    the column contract used by ``build_modeling_matrix``:

    - ``GROUP_COLUMN`` (``student_id``) — student identifier,
    - ``WEEK_COLUMN`` (``week_number``) — week index,
    - ``passed`` — from the snapshot's ``passed_observed``,
    - ``final_grade`` — ``float(passed)`` structural placeholder, mirroring the
      KU Leuven engagement-only convention (no real grade is used).

    Rows with a NaN ``passed`` label are dropped.

    Returns both the processed modeling ``frame`` and the adapter's
    ``OuladSnapshotBuildResult`` from the single ``build_weekly_snapshots`` pass,
    so callers can capture row counts / target summary without re-running the
    chunked VLE pass.
    """

    resolved_raw_dir = _resolve_raw_dir(raw_dir)
    paths = oulad_adapter.OuladRawPaths.from_directory(resolved_raw_dir)
    course_filter = oulad_adapter.OuladCourseFilter(
        code_module=code_module,
        code_presentation=code_presentation,
    )
    build_result = oulad_adapter.build_weekly_snapshots(
        paths,
        course_filter=course_filter,
        min_week=min_week,
        max_week=max_week,
    )

    frame = build_result.snapshots.copy()

    # GROUP_COLUMN: the snapshot frame already carries a `student_id` alias.
    if GROUP_COLUMN not in frame.columns:
        frame[GROUP_COLUMN] = frame["id_student"]
    # WEEK_COLUMN == week_number (already present in OULAD snapshots).
    if WEEK_COLUMN not in frame.columns:
        frame[WEEK_COLUMN] = frame["week_number"]

    frame["passed"] = frame["passed_observed"]
    frame = frame.loc[frame["passed"].notna()].copy()
    frame["final_grade"] = frame["passed"].astype(float)

    return frame.reset_index(drop=True), build_result


# ---------------------------------------------------------------------------
# Pydantic config (OULAD-specific)
# ---------------------------------------------------------------------------


class OuladCourseFilterConfig(BaseModel):
    code_module: str
    code_presentation: str


class OuladDataConfig(BaseModel):
    raw_dir: Path = Path("datasets/oulad")
    files: dict[str, str] = Field(default_factory=dict)
    course_filter: OuladCourseFilterConfig
    min_week: int = 4
    max_week: int | None = None


class OuladEngagementConfig(BaseModel):
    experiment_id: str
    title: str
    schema_version: str = "oulad_engagement_v1"
    dataset_version: str = "OULAD"
    dataset_config_name: str = "local OULAD CSV files"
    parent_experiment: str | None = "exp_011_kuleuven_engagement"
    status: str = "completed"
    documentation: DocumentationConfig
    oulad: OuladDataConfig
    feature_sets: dict[str, FeatureSetConfig]
    feature_set_order: list[str]
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    classification_models: list[ClassificationModelName] = Field(default_factory=list)
    splits: SplitsConfig = Field(default_factory=SplitsConfig)
    seed: int = 42
    outputs: OutputsConfig

    @model_validator(mode="after")
    def _validate(self) -> "OuladEngagementConfig":
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
# Build-result wrapper (adapts OULAD adapter output to the shared writers)
# ---------------------------------------------------------------------------


class _OuladBuildResult:
    """Expose the ``snapshots`` / ``row_counts`` / ``target_summary`` surface the
    shared writers and markdown renderer consume.

    The OULAD adapter returns ``filtered_row_counts`` (with ``snapshots`` and
    ``students`` keys) and a ``target_summary`` keyed differently from the KU
    Leuven one. This wrapper re-keys those into the exact fields the shared core
    reads (``row_counts['snapshots'/'students']`` and ``target_summary`` keys
    ``week_min``, ``week_max``, ``passed_rate_student_level``, ``n_passed``,
    ``n_failed``).
    """

    def __init__(self, frame: pd.DataFrame, adapter_result: Any) -> None:
        self.snapshots = frame
        # Report the modeling-frame row count (post NaN-drop) as the snapshot
        # count so reported rows == rows that actually go into training.
        self.row_counts = {
            **adapter_result.filtered_row_counts,
            "snapshots": int(frame.shape[0]),
        }

        student_level = frame.sort_values(WEEK_COLUMN).drop_duplicates(
            GROUP_COLUMN, keep="last"
        )
        n_students = int(student_level.shape[0])
        passed = pd.to_numeric(student_level["passed"], errors="coerce")
        n_passed = int((passed == 1).sum())
        n_failed = int((passed == 0).sum())
        passed_rate = n_passed / n_students if n_students > 0 else float("nan")

        self.target_summary = {
            **adapter_result.target_summary,
            "week_min": int(frame[WEEK_COLUMN].min()),
            "week_max": int(frame[WEEK_COLUMN].max()),
            "n_passed": n_passed,
            "n_failed": n_failed,
            "passed_rate_student_level": round(passed_rate, 4),
        }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_oulad_engagement_config(
    config_path: str | Path,
) -> tuple[OuladEngagementConfig, Path]:
    resolved = _resolve_config_path(config_path)
    raw_payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    return OuladEngagementConfig.model_validate(raw_payload), resolved


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------


def run_oulad_engagement_benchmark(
    config_path: str | Path,
    *,
    allow_existing: bool = False,
    docs_dir: Path | None = None,
) -> dict[str, Any]:
    config, resolved_config_path = load_oulad_engagement_config(config_path)
    experiment_id = config.experiment_id
    output_dir = config.resolve_path(config.outputs.experiments_dir)

    ensure_new_experiment_dir(output_dir, allow_existing=allow_existing)

    # Build OULAD modeling frame via the shared snapshot builder. A single
    # adapter pass yields both the processed frame and the adapter build result
    # (row counts / target summary) for reporting.
    raw_dir = config.resolve_path(config.oulad.raw_dir)
    modeling_frame, adapter_result = build_oulad_engagement_frame(
        code_module=config.oulad.course_filter.code_module,
        code_presentation=config.oulad.course_filter.code_presentation,
        min_week=config.oulad.min_week,
        max_week=config.oulad.max_week,
        raw_dir=raw_dir,
    )
    build_result = _OuladBuildResult(modeling_frame, adapter_result)

    institution_context = _oulad_institution_context(
        code_module=config.oulad.course_filter.code_module,
        code_presentation=config.oulad.course_filter.code_presentation,
        config_path=resolved_config_path,
        raw_dir=raw_dir,
    )

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
        institution_context=institution_context,
    )
    diagnostics = build_benchmark_diagnostics(
        rows,
        config,
        build_result,
        importance_by_split,
        interpretation_subject="OULAD",
    )

    importance_csv_path = output_dir / "engagement_importance.csv"
    _write_importance_csv(importance_by_split, importance_csv_path)

    summary_md = render_experiment_markdown(
        config=config,
        diagnostics=diagnostics,
        build_result=build_result,
        result_paths=result_paths,
        config_path=resolved_config_path,
        importance_csv_path=importance_csv_path,
        institution_context=institution_context,
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
        raw_dir=raw_dir,
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
# Experiment metadata (OULAD-specific)
# ---------------------------------------------------------------------------


def _build_experiment_metadata(
    *,
    config: OuladEngagementConfig,
    diagnostics: dict[str, Any],
    build_result: _OuladBuildResult,
    result_paths: dict[str, Path],
    config_path: Path,
    output_dir: Path,
    docs_path: Path,
    importance_csv_path: Path,
    raw_dir: Path,
) -> dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "title": config.title,
        "created_at": utc_now_iso(),
        "schema_version": config.schema_version,
        "dataset": {
            "dataset_version": config.dataset_version,
            "dataset_config_name": config.dataset_config_name,
            "source": (
                f"OULAD ({config.oulad.course_filter.code_module} "
                f"{config.oulad.course_filter.code_presentation})"
            ),
            "raw_dir": relative_repo_path(raw_dir),
            "course_filter": {
                "code_module": config.oulad.course_filter.code_module,
                "code_presentation": config.oulad.course_filter.code_presentation,
            },
            "row_counts": build_result.row_counts,
        },
        "targets": {
            "primary": "passed",
            "note": (
                "Classification only. final_grade is a placeholder = float(passed). "
                "Regression intentionally omitted (engagement-only matched design). "
                "cumulative_social_clicks_to_date (forum/social clicks) is the OULAD "
                "analog of KU Leuven forum posts; cross-institution comparison uses "
                "ranks, not magnitudes."
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
# Private utilities (OULAD-specific)
# ---------------------------------------------------------------------------


def _resolve_raw_dir(raw_dir: str | Path) -> Path:
    path = Path(raw_dir)
    if path.is_absolute():
        return path
    for candidate in (Path.cwd() / path, REPO_ROOT / path, SERVICE_ROOT / path):
        if candidate.exists():
            return candidate.resolve()
    # Fall back to the repo-root-relative path even if it does not yet exist so
    # the adapter raises a clear FileNotFoundError naming the expected location.
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
    raise FileNotFoundError(
        f"Could not resolve OULAD engagement config path: {config_path}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the OULAD engagement-only PASSED classification benchmark."
    )
    parser.add_argument(
        "--config",
        default="configs/experiments/exp_012_oulad_engagement.yaml",
        help="Path to the OULAD engagement YAML config.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow rerunning into an existing experiment directory.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_oulad_engagement_benchmark(args.config, allow_existing=args.overwrite)
    print(f"Experiment metadata: {payload['metadata_path']}")
    print(f"Experiment docs: {payload['docs_path']}")
    print(f"Results CSV: {payload['result_paths']['csv']}")
    print(f"Importance CSV: {payload['importance_csv_path']}")


if __name__ == "__main__":
    main()
