"""Dataset-agnostic core for engagement-only PASSED classification benchmarks.

This module holds the methodology shared between institution-specific runners
(KU Leuven exp_011, OULAD exp_012, ...): feature-set / config dataclasses, the
split builder, classification training + permutation importance, the
best-per-cell and fixed-model diagnostics, the artifact writers, and the
markdown renderer.

A function lives here if it would be byte-identical for any institution. The
institution-specific entry points (config loading, snapshot building, the data
citation, and ``_build_experiment_metadata``) stay in their own runner modules
and call into this core.

The bodies in this module are extracted VERBATIM from
``run_engagement_benchmark_kuleuven.py`` to preserve exp_011 output exactly.
Config/build-result parameters are typed against the KU Leuven classes for
documentation only; under ``from __future__ import annotations`` those hints are
never evaluated, so any structurally-compatible config/build-result works.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.inspection import permutation_importance

from src.experiments.config import ClassificationModelName
from src.experiments.datasets import GROUP_COLUMN, WEEK_COLUMN
from src.experiments.evaluate import (
    ResultRow,
    compute_classification_metrics,
    results_to_dataframe,
)
from src.experiments.metadata import relative_repo_path
from src.experiments.models import build_classification_model, iter_classification_models
from src.experiments.preprocessing import (
    PreparedMatrix,
    fit_imputer_on_training,
    select_rows,
)
from src.experiments.splits import SplitResult, student_group_split

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.benchmarks.ku_leuven_adapter import KuLeuvenSnapshotBuildResult
    from src.experiments.run_engagement_benchmark_kuleuven import (
        KuLeuvenEngagementConfig,
    )


SplitStrategy = Literal["student_group", "temporal_forward"]


# ---------------------------------------------------------------------------
# Dataclasses mirroring run_public_benchmark_oulad
# ---------------------------------------------------------------------------


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


@dataclass(frozen=True)
class InstitutionContext:
    """Institution-specific literal strings substituted into the shared writers.

    The dataset-agnostic core (``render_experiment_markdown`` and
    ``_write_result_artifacts``) renders a banner block, a dataset-identity
    block, and a JSON ``note`` whose wording is institution-specific (institution
    name, dataset citation, data-source paths, and the regression-not-applicable
    note). Runners pass an ``InstitutionContext`` to swap that wording without
    leaking another institution's name into their output.

    Fields
    ------
    banner_lines:
        The markdown blockquote banner shown directly under the title. Each
        element is one already-formatted markdown line (e.g. ``"> ..."``).
    dataset_identity_lines:
        The institution-identity bullet lines of the ``## Dataset`` section
        (institution, citation, config path, data-source paths). These come
        before the shared snapshot/week/passed-rate statistics, which are
        rendered from ``build_result`` and are identical across institutions.
    json_note:
        The ``note`` string embedded in the results JSON payload.
    """

    banner_lines: tuple[str, ...]
    dataset_identity_lines: tuple[str, ...]
    json_note: str


_KULEUVEN_BANNER_LINES: tuple[str, ...] = (
    "> **THIRD INSTITUTION — ENGAGEMENT-ONLY — CLASSIFICATION ONLY (PASSED)**",
    ">",
    "> KU Leuven has NO numeric intermediate assessment scores and NO meaningful",
    "> continuous grade. Only binary PASSED classification is reported.",
    "> Regression is NOT applicable (final_grade = float(passed) placeholder).",
    "> The mastery/Twin feature ablation from OULAD experiments CANNOT be",
    "> reproduced here.",
)


_KULEUVEN_JSON_NOTE: str = (
    "KU Leuven engagement-only benchmark. "
    "Classification (PASSED) only. Regression NOT applicable. "
    "final_grade is a placeholder equal to float(passed)."
)


def _kuleuven_dataset_identity_lines(
    config: KuLeuvenEngagementConfig,
    config_path: Path,
) -> tuple[str, ...]:
    """Reconstruct the exact KU Leuven ``## Dataset`` identity bullet lines.

    Used as the ``None``-default fallback so the KU runner's markdown stays
    byte-identical with no changes to its call site.
    """

    return (
        "- Institution: KU Leuven (year 1819) — THIRD institution",
        "- Citation: Tiukhova, E., Van Landuyt, D., Baesens, B., & Snoeck, M. (2026). "
        "Open data, private learners: a de-identified student activity and performance "
        "dataset for learning analytics. Scientific Data. CC-BY-4.0, "
        "Zenodo DOI 10.5281/zenodo.17087849.",
        f"- Experiment config: `{relative_repo_path(config_path)}`",
        f"- Data dir: `{relative_repo_path(config.resolve_path(config.data.data_dir))}`",
        f"- Course info: `{relative_repo_path(config.resolve_path(config.data.course_info_path))}`",
    )


# ---------------------------------------------------------------------------
# Pydantic config
# ---------------------------------------------------------------------------


class DocumentationConfig(BaseModel):
    objective: str
    hypothesis: str
    limitations: list[str] = Field(default_factory=list)
    next_step: str


class FeatureSetConfig(BaseModel):
    description: str
    columns: list[str]
    indicator_columns: list[str] = Field(default_factory=list)


class ComparisonConfig(BaseModel):
    baseline_feature_set: str = "A_simple_engagement"
    candidate_feature_set: str = "B_engagement"
    primary_split: SplitStrategy = "temporal_forward"
    fixed_model: ClassificationModelName = "gradient_boosting"


class StudentGroupSplitConfig(BaseModel):
    test_size: float = 0.25
    seed: int = 42


class TemporalForwardSplitConfig(BaseModel):
    train_weeks: int = 8
    student_test_size: float = 0.25
    student_seed: int = 42


class SplitsConfig(BaseModel):
    student_group: StudentGroupSplitConfig = Field(default_factory=StudentGroupSplitConfig)
    temporal_forward: TemporalForwardSplitConfig = Field(
        default_factory=TemporalForwardSplitConfig
    )


class OutputsConfig(BaseModel):
    experiments_dir: Path


# ---------------------------------------------------------------------------
# Split builder (mirrors _build_split from run_public_benchmark_oulad)
# ---------------------------------------------------------------------------


def build_split(
    matrix: PreparedMatrix,
    strategy: SplitStrategy,
    config: KuLeuvenEngagementConfig,
) -> _SplitPartition:
    """Build train/test masks for a given split strategy.

    Public function so tests can call it directly without running the full pipeline.
    """
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


# ---------------------------------------------------------------------------
# Training and scoring
# ---------------------------------------------------------------------------


def _train_and_score_classification(
    matrix: PreparedMatrix,
    *,
    feature_set: BenchmarkFeatureSet,
    partition: _SplitPartition,
    config: KuLeuvenEngagementConfig,
) -> list[ResultRow]:
    train_matrix = select_rows(matrix, partition.train_mask)
    test_matrix = select_rows(matrix, partition.test_mask)
    if train_matrix.features.empty or test_matrix.features.empty:
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
                target="passed",
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


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


def compute_permutation_importance(
    matrix: PreparedMatrix,
    *,
    partition: _SplitPartition,
    config: KuLeuvenEngagementConfig,
) -> dict[str, Any]:
    """Train the fixed model on train split and compute permutation importance on test.

    Returns a dict with per-feature mean importance, ranks, and top-k shares.
    """
    train_matrix = select_rows(matrix, partition.train_mask)
    test_matrix = select_rows(matrix, partition.test_mask)

    if train_matrix.features.empty or test_matrix.features.empty:
        return {"error": "empty split", "features": []}

    imputer = fit_imputer_on_training(train_matrix.features)
    x_train = imputer.transform(train_matrix.features)
    x_test = imputer.transform(test_matrix.features)
    y_train = train_matrix.classification_target.to_numpy()
    y_test = test_matrix.classification_target.to_numpy()

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return {"error": "single class in train or test", "features": []}

    estimator = build_classification_model(config.comparison.fixed_model, seed=config.seed)
    estimator.fit(x_train, y_train)

    result = permutation_importance(
        estimator,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=15,
        random_state=config.seed,
        n_jobs=1,
    )

    feature_names = list(matrix.feature_columns)
    importances_mean = result.importances_mean
    importances_std = result.importances_std

    # Rank best-first (highest importance = rank 1)
    ranked_indices = np.argsort(importances_mean)[::-1]

    # Top-k importance share (normalise over positive importances only)
    positive_sum = float(np.sum(importances_mean[importances_mean > 0]))

    feature_records = []
    for rank, idx in enumerate(ranked_indices, start=1):
        name = feature_names[idx]
        mean_imp = float(importances_mean[idx])
        std_imp = float(importances_std[idx])
        share = mean_imp / positive_sum if positive_sum > 0 and mean_imp > 0 else 0.0
        feature_records.append(
            {
                "feature": name,
                "mean_importance": round(mean_imp, 6),
                "std_importance": round(std_imp, 6),
                "rank": rank,
                "share_of_positive": round(share, 4),
            }
        )

    # Top-1 and top-3 share
    top1_share = feature_records[0]["share_of_positive"] if feature_records else 0.0
    top3_share = sum(r["share_of_positive"] for r in feature_records[:3])

    return {
        "model": config.comparison.fixed_model,
        "scoring": "roc_auc",
        "n_repeats": 15,
        "n_features": len(feature_names),
        "positive_importance_sum": round(positive_sum, 6),
        "top1_share": round(top1_share, 4),
        "top3_share": round(top3_share, 4),
        "features": feature_records,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _best_classification_by_feature_set(
    table: pd.DataFrame,
    *,
    primary_split: str,
) -> list[dict[str, Any]]:
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
                "roc_auc": float(best["roc_auc"]) if not pd.isna(best["roc_auc"]) else None,
                "n_train_rows": int(best["n_train_rows"]),
                "n_test_rows": int(best["n_test_rows"]),
            }
        )
    return sorted(
        records,
        key=lambda item: (
            0 if item["split_strategy"] == primary_split else 1,
            item["split_strategy"],
            item["feature_set"],
        ),
    )


def _fixed_model_classification_by_feature_set(
    table: pd.DataFrame,
    *,
    model: str,
    baseline_feature_set: str,
    primary_split: str,
) -> list[dict[str, Any]]:
    if table.empty or "task" not in table.columns:
        return []
    classification = table.loc[
        (table["task"] == "classification") & (table["model"] == model)
    ].copy()
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
        row = group.iloc[0]
        records.append(
            {
                "split_strategy": split_strategy,
                "feature_set": feature_set,
                "model": row["model"],
                "f1": float(row["f1"]),
                "accuracy": float(row["accuracy"]),
                "roc_auc": float(row["roc_auc"]) if not pd.isna(row["roc_auc"]) else None,
                "n_train_rows": int(row["n_train_rows"]),
                "n_test_rows": int(row["n_test_rows"]),
            }
        )
    # Add delta vs baseline
    baseline_by_split: dict[str, dict[str, float]] = {}
    for record in records:
        if record["feature_set"] == baseline_feature_set:
            baseline_by_split[record["split_strategy"]] = {
                "f1": record["f1"],
                "roc_auc": record["roc_auc"] if record["roc_auc"] is not None else float("nan"),
            }
    for record in records:
        baseline = baseline_by_split.get(record["split_strategy"])
        if baseline is None:
            record["delta_f1"] = None
            record["delta_auc"] = None
        else:
            record["delta_f1"] = round(record["f1"] - baseline["f1"], 4)
            roc_auc = record["roc_auc"]
            if roc_auc is None:
                record["delta_auc"] = None
            else:
                record["delta_auc"] = round(roc_auc - baseline["roc_auc"], 4)

    return sorted(
        records,
        key=lambda item: (
            0 if item["split_strategy"] == primary_split else 1,
            item["split_strategy"],
            item["feature_set"],
        ),
    )


def build_benchmark_diagnostics(
    rows: list[ResultRow],
    config: KuLeuvenEngagementConfig,
    build_result: KuLeuvenSnapshotBuildResult,
    importance_by_split: dict[str, dict[str, Any]],
    interpretation_subject: str = "KU Leuven",
) -> dict[str, Any]:
    table = results_to_dataframe(rows)
    best_classification = _best_classification_by_feature_set(
        table,
        primary_split=config.comparison.primary_split,
    )
    fixed_model_classification = _fixed_model_classification_by_feature_set(
        table,
        model=config.comparison.fixed_model,
        baseline_feature_set=config.comparison.baseline_feature_set,
        primary_split=config.comparison.primary_split,
    )
    interpretation = _interpret_engagement_result(
        fixed_model_classification,
        config,
        interpretation_subject=interpretation_subject,
    )
    return {
        "experiment_id": config.experiment_id,
        "primary_target": "passed",
        "primary_split": config.comparison.primary_split,
        "row_counts": build_result.row_counts,
        "target_summary": build_result.target_summary,
        "classification_best_by_feature_set": best_classification,
        "classification_fixed_model_by_feature_set": fixed_model_classification,
        "fixed_model": config.comparison.fixed_model,
        "permutation_importance": importance_by_split,
        "interpretation": interpretation,
    }


def _interpret_engagement_result(
    fixed_model_summary: list[dict[str, Any]],
    config: KuLeuvenEngagementConfig,
    interpretation_subject: str = "KU Leuven",
) -> dict[str, Any]:
    primary_candidate = next(
        (
            record
            for record in fixed_model_summary
            if record["split_strategy"] == config.comparison.primary_split
            and record["feature_set"] == config.comparison.candidate_feature_set
        ),
        None,
    )
    primary_delta_f1 = (
        None if primary_candidate is None else primary_candidate.get("delta_f1")
    )

    if primary_delta_f1 is None:
        outcome = "inconclusive"
        short = "Primary candidate classification record not found."
        interpretation = (
            "The engagement benchmark cannot be interpreted because the primary "
            "candidate comparison is missing."
        )
    elif primary_delta_f1 > 0.02:
        outcome = "richer_engagement_helps"
        short = (
            f"`{config.comparison.candidate_feature_set}` improved F1 by "
            f"{primary_delta_f1:+.3f} over `{config.comparison.baseline_feature_set}` "
            "on the primary split."
        )
        interpretation = (
            "Richer engagement features modestly improve PASSED classification on "
            f"{interpretation_subject}. The additional session, content-type, forum, "
            "and temporal features provide signal beyond raw click volume and active "
            "days."
        )
    elif abs(primary_delta_f1) <= 0.02:
        outcome = "engagement_richness_neutral"
        short = (
            f"`{config.comparison.candidate_feature_set}` was approximately level "
            f"with `{config.comparison.baseline_feature_set}` (delta F1 = "
            f"{primary_delta_f1:+.3f})."
        )
        interpretation = (
            "Richer engagement features provide negligible additional F1 lift over "
            f"the minimal baseline on {interpretation_subject}. The dominant PASSED "
            "signal is already captured by cumulative clicks and active days; the "
            "additional features do not materially shift classification accuracy."
        )
    else:
        outcome = "richer_engagement_hurts"
        short = (
            f"`{config.comparison.candidate_feature_set}` was worse than "
            f"`{config.comparison.baseline_feature_set}` (delta F1 = "
            f"{primary_delta_f1:+.3f})."
        )
        interpretation = (
            "Richer engagement features slightly degraded PASSED classification "
            f"compared with the minimal baseline on {interpretation_subject}. This "
            "may reflect correlation noise or overfitting in the richer set."
        )
    return {
        "outcome": outcome,
        "primary_delta_f1": primary_delta_f1,
        "short_conclusion": short,
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------


def _write_result_artifacts(
    rows: list[ResultRow],
    *,
    output_dir: Path,
    run_name: str,
    config: KuLeuvenEngagementConfig,
    build_result: KuLeuvenSnapshotBuildResult,
    institution_context: InstitutionContext | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{run_name}_results.csv"
    json_path = output_dir / f"{run_name}_results.json"
    markdown_path = output_dir / f"{run_name}_summary.md"

    json_note = (
        _KULEUVEN_JSON_NOTE
        if institution_context is None
        else institution_context.json_note
    )

    table = results_to_dataframe(rows)
    if not table.empty:
        table.to_csv(csv_path, index=False)
    json_payload = {
        "run_name": run_name,
        "note": json_note,
        "rows": [
            {
                **row.to_record(),
                "metrics": row.metrics,
                "split_metadata": row.split_metadata,
            }
            for row in rows
        ],
        "metadata": {
            "target": "passed",
            "row_counts": build_result.row_counts,
            "target_summary": build_result.target_summary,
        },
    }
    json_path.write_text(json.dumps(json_payload, indent=2, default=str), encoding="utf-8")
    # markdown written later with full diagnostics
    markdown_path.write_text(
        f"# {run_name}\n\nResults placeholder — will be overwritten with full summary.\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "json": json_path, "markdown": markdown_path}


def _write_importance_csv(
    importance_by_split: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    rows = []
    for split_strategy, result in importance_by_split.items():
        if "error" in result or not result.get("features"):
            continue
        for record in result["features"]:
            rows.append(
                {
                    "split_strategy": split_strategy,
                    "feature": record["feature"],
                    "mean_importance": record["mean_importance"],
                    "std_importance": record["std_importance"],
                    "rank": record["rank"],
                    "share_of_positive": record["share_of_positive"],
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "split_strategy",
                "feature",
                "mean_importance",
                "std_importance",
                "rank",
                "share_of_positive",
            ]
        ).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_experiment_markdown(
    *,
    config: KuLeuvenEngagementConfig,
    diagnostics: dict[str, Any],
    build_result: KuLeuvenSnapshotBuildResult,
    result_paths: dict[str, Path],
    config_path: Path,
    importance_csv_path: Path,
    institution_context: InstitutionContext | None = None,
) -> str:
    interp = diagnostics["interpretation"]
    if institution_context is None:
        banner_lines = _KULEUVEN_BANNER_LINES
        dataset_identity_lines = _kuleuven_dataset_identity_lines(config, config_path)
    else:
        banner_lines = institution_context.banner_lines
        dataset_identity_lines = institution_context.dataset_identity_lines
    lines: list[str] = [
        f"# {config.experiment_id}: {config.title}",
        "",
        *banner_lines,
        "",
        "## Objective",
        "",
        config.documentation.objective,
        "",
        "## Hypothesis",
        "",
        config.documentation.hypothesis,
        "",
        "## Dataset",
        "",
        *dataset_identity_lines,
        f"- Snapshot rows: {build_result.row_counts.get('snapshots', 'n/a')}",
        f"- Students: {build_result.row_counts.get('students', 'n/a')}",
        f"- Weeks: {build_result.target_summary.get('week_min', '?')}–{build_result.target_summary.get('week_max', '?')}",
        f"- PASSED rate (student-level): {build_result.target_summary.get('passed_rate_student_level', 'n/a')}",
        f"- n_passed: {build_result.target_summary.get('n_passed', 'n/a')}, n_failed: {build_result.target_summary.get('n_failed', 'n/a')}",
        "",
        "## Feature sets",
        "",
    ]
    for fs in config.build_feature_sets():
        lines.extend(
            [
                f"### `{fs.name}`",
                "",
                fs.description,
                "",
                f"- Columns: {', '.join(f'`{c}`' for c in fs.columns)}",
            ]
        )
        if fs.indicator_columns:
            lines.append(
                "- Indicator columns: "
                + ", ".join(f"`{c}`" for c in fs.indicator_columns)
            )
        lines.append("")

    lines.extend(
        [
            "## Split strategies",
            "",
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
            "- Classification models: "
            + ", ".join(f"`{m}`" for m in config.classification_models),
            "",
            "## Headline classification results (best model per cell)",
            "",
            "| split | feature set | best model | F1 | accuracy | ROC-AUC |",
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

    if diagnostics["classification_fixed_model_by_feature_set"]:
        lines.extend(
            [
                "",
                f"## Fixed-model classification results (model = `{diagnostics['fixed_model']}`)",
                "",
                "Same model across all feature sets and both splits to neutralize "
                "model-flip artifacts. Delta is relative to "
                f"`{config.comparison.baseline_feature_set}`.",
                "",
                "| split | feature set | F1 | accuracy | ROC-AUC | delta F1 | delta AUC |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in diagnostics["classification_fixed_model_by_feature_set"]:
            lines.append(
                "| {split} | {fs} | {f1:.3f} | {acc:.3f} | {auc} | {df1} | {dauc} |".format(
                    split=record["split_strategy"],
                    fs=record["feature_set"],
                    f1=record["f1"],
                    acc=record["accuracy"],
                    auc=_format_optional(record.get("roc_auc")),
                    df1=_format_delta(record.get("delta_f1")),
                    dauc=_format_delta(record.get("delta_auc")),
                )
            )

    # Permutation importance section
    lines.extend(
        [
            "",
            "## Permutation importance (candidate feature set: "
            f"`{config.comparison.candidate_feature_set}`)",
            "",
            "Model: `{model}`, scoring: ROC-AUC, n_repeats=15.".format(
                model=config.comparison.fixed_model
            ),
            "Shares are normalised over positive importances only.",
            "",
        ]
    )
    for strategy in ["student_group", "temporal_forward"]:
        imp = diagnostics["permutation_importance"].get(strategy, {})
        lines.append(f"### Split: `{strategy}`")
        lines.append("")
        if "error" in imp or not imp.get("features"):
            lines.append(f"_Not available: {imp.get('error', 'no data')}_")
            lines.append("")
            continue
        lines.append(
            f"- Total positive importance sum: {imp.get('positive_importance_sum', 'n/a')}"
        )
        lines.append(f"- Top-1 feature share: {imp.get('top1_share', 'n/a'):.1%}")
        lines.append(f"- Top-3 feature share: {imp.get('top3_share', 'n/a'):.1%}")
        lines.append("")
        lines.append("| rank | feature | mean importance | std | share of positive |")
        lines.append("| ---: | --- | ---: | ---: | ---: |")
        for record in imp["features"][:10]:  # show top 10
            lines.append(
                "| {rank} | {feat} | {mean:.4f} | {std:.4f} | {share:.1%} |".format(
                    rank=record["rank"],
                    feat=record["feature"],
                    mean=record["mean_importance"],
                    std=record["std_importance"],
                    share=record["share_of_positive"],
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            interp["interpretation"],
            "",
            f"- Outcome: `{interp['outcome']}`",
            f"- Primary delta F1: `{_format_delta(interp.get('primary_delta_f1'))}`",
            f"- Short conclusion: {interp['short_conclusion']}",
            "",
            "## Artifact paths",
            "",
            f"- Results JSON: `{relative_repo_path(result_paths['json'])}`",
            f"- Results CSV: `{relative_repo_path(result_paths['csv'])}`",
            f"- Importance CSV: `{relative_repo_path(importance_csv_path)}`",
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


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _validate_feature_columns(
    frame: pd.DataFrame,
    feature_sets: list[BenchmarkFeatureSet],
) -> None:
    missing_by_set: dict[str, list[str]] = {}
    for feature_set in feature_sets:
        missing = [
            column for column in feature_set.all_columns() if column not in frame.columns
        ]
        if missing:
            missing_by_set[feature_set.name] = missing
    if missing_by_set:
        raise ValueError(f"Feature columns are missing in modeling frame: {missing_by_set}")


def _format_delta(value: float | None) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:+.3f}"


def _format_optional(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{float(value):.3f}"


def _markdown_relative_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target.resolve(), from_dir.resolve())).as_posix()
