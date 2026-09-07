"""exp_015 — feature-richness control for the transfer ladder (reviewer objection).

Objection this answers: "your model transfers badly because your seven shared
features are too crude; a properly specified model would transfer."

Design: stay inside OULAD (where rich features exist) and run the SAME ladder
with three nested feature sets:

    S  shared-7        the cross-institution schema used in exp_014
    E  rich engagement + click categories, ratios, activity breadth, progress
    A  E + assessment  + intermediate assessment scores, submission behaviour
                         (the strongest known predictors; unavailable at the
                          other two institutions)

If the transfer gap were caused by feature poverty, the D2 gap (other course,
same institution) should shrink from S to A. Reported per feature set:
within-cohort AUC (D0), same-course-other-year (D1), other-course (D2), and the
RELATIVE drop D0 -> D2, which is the quantity the objection is about.

Usage (from services/ml):
    uv run python -m src.experiments.run_feature_richness_control
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.benchmarks import oulad_adapter as oa
from src.experiments import transfer_benchmark as tb
from src.experiments.models import build_classification_model

REPO = Path(__file__).resolve().parents[4]
CACHE = REPO / "datasets" / "cache"
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_015_feature_richness"

MODULES = ("BBB", "DDD", "FFF")
PRESENTATIONS = ("2013B", "2013J", "2014B", "2014J")

# Columns pulled from the OULAD adapter snapshots (superset of all three sets).
_RICH_COLUMNS = (
    "id_student",
    "week_number",
    "duration_weeks",
    "passed_observed",
    "current_week_clicks",
    "current_week_active_days",
    "current_week_content_clicks",
    "current_week_social_clicks",
    "current_week_assessment_clicks",
    "current_week_activity_types",
    "cumulative_clicks_to_date",
    "cumulative_active_days_to_date",
    "cumulative_content_clicks_to_date",
    "cumulative_social_clicks_to_date",
    "cumulative_assessment_clicks_to_date",
    "content_click_ratio_to_date",
    "course_week_progress",
    "cumulative_assessment_score_mean_to_date",
    "cumulative_assessment_score_count_to_date",
    "cumulative_assessment_weighted_score_to_date",
    "assessment_submission_rate_due_to_date",
    "assessment_score_trend_to_date",
    "has_assessment_score_to_date",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "S_shared7": tb.CANON,
    "E_rich_engagement": tb.CANON
    + (
        "cumulative_assessment_clicks_to_date",
        "current_week_activity_types",
        "content_click_ratio_to_date",
        "course_week_progress",
    ),
    "A_engagement_plus_assessment": tb.CANON
    + (
        "cumulative_assessment_clicks_to_date",
        "current_week_activity_types",
        "content_click_ratio_to_date",
        "course_week_progress",
        "cumulative_assessment_score_mean_to_date",
        "cumulative_assessment_score_count_to_date",
        "cumulative_assessment_weighted_score_to_date",
        "assessment_submission_rate_due_to_date",
        "assessment_score_trend_to_date",
        "has_assessment_score_to_date",
    ),
}


def build_rich_cache() -> None:
    """Cache OULAD snapshots keeping the rich column superset (skips existing files)."""
    paths = oa.OuladRawPaths.from_directory(REPO / "datasets" / "oulad")
    CACHE.mkdir(parents=True, exist_ok=True)
    for mod in MODULES:
        for pres in PRESENTATIONS:
            target = CACHE / f"ouladrich_{mod}_{pres}.parquet"
            if target.exists():
                continue
            result = oa.build_weekly_snapshots(
                paths, course_filter=oa.OuladCourseFilter(mod, pres), min_week=1
            )
            snapshots = result.snapshots
            keep = [c for c in _RICH_COLUMNS if c in snapshots.columns]
            snapshots[keep].to_parquet(target, index=False)
            print(f"cached {target.name} ({len(snapshots)} rows, {len(keep)} cols)", flush=True)


def load_rich_cohorts(fraction: float) -> dict[str, pd.DataFrame]:
    """One cutoff row per student per cohort, carrying every available rich column."""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(CACHE.glob("ouladrich_*.parquet")):
        _, mod, pres = path.stem.split("_")
        raw = pd.read_parquet(path)
        raw = raw.loc[raw["passed_observed"].notna()].copy()
        frame = raw.rename(
            columns={
                "id_student": "student_id",
                "duration_weeks": "n_weeks",
                "current_week_clicks": "current_clicks",
                "current_week_active_days": "current_active_days",
                "current_week_content_clicks": "current_content_clicks",
                "current_week_social_clicks": "current_social_clicks",
            }
        )
        frame["student_id"] = frame["student_id"].astype(str)
        frame["passed"] = frame["passed_observed"].astype(int)
        canon = tb.canonical_from_weekly(frame)  # adds the shared-7 columns
        rows = tb.cutoff_rows(canon, fraction)
        rows["module"] = mod
        out[f"oulad_{mod}_{pres}"] = rows
    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def run_control(fraction: float, seed: int, model: str) -> pd.DataFrame:
    cohorts = load_rich_cohorts(fraction)
    rows: list[dict] = []
    for set_name, columns in FEATURE_SETS.items():
        available = [c for c in columns if c in next(iter(cohorts.values())).columns]
        missing = [c for c in columns if c not in available]
        if missing:
            print(f"  [{set_name}] missing columns skipped: {missing}", flush=True)
        prepared = {
            cid: (
                df[available].astype(float).fillna(0.0).to_numpy(),
                df["passed"].to_numpy(dtype=int),
                str(df["module"].iloc[0]),
            )
            for cid, df in cohorts.items()
        }
        fitted = {
            cid: build_classification_model(model, seed=seed).fit(X, y)
            for cid, (X, y, _) in prepared.items()
        }
        for src, (_, _, src_mod) in prepared.items():
            for tgt, (Xt, yt, tgt_mod) in prepared.items():
                if src == tgt:
                    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                    p = cross_val_predict(
                        build_classification_model(model, seed=seed), Xt, yt, cv=cv, method="predict_proba"
                    )[:, 1]
                    distance = "D0_within_cohort"
                else:
                    p = fitted[src].predict_proba(Xt)[:, 1]
                    distance = "D1_same_module" if src_mod == tgt_mod else "D2_other_module"
                rows.append(
                    {
                        "feature_set": set_name,
                        "n_features": len(available),
                        "source": src,
                        "target": tgt,
                        "distance": distance,
                        "model": model,
                        "auc": _auc(yt, p),
                    }
                )
    return pd.DataFrame(rows)


def summarise(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per feature set: mean AUC by distance, plus the absolute and relative D0 -> D2 drop."""
    per_target = pairs.groupby(["feature_set", "n_features", "distance", "target"])["auc"].mean().reset_index()
    table = per_target.groupby(["feature_set", "n_features", "distance"])["auc"].mean().unstack()
    table["drop_D0_to_D2"] = table["D0_within_cohort"] - table["D2_other_module"]
    table["relative_drop_pct"] = 100 * table["drop_D0_to_D2"] / (table["D0_within_cohort"] - 0.5)
    return table.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="gradient_boosting")
    args = ap.parse_args()

    build_rich_cache()
    pairs = run_control(args.fraction, args.seed, args.model)
    summary = summarise(pairs)
    out = OUT / f"f{int(round(args.fraction * 100)):02d}"
    tb.write_outputs(out, pairs=pairs, summary=summary)
    print(summary.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
