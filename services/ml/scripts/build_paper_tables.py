"""Render the SIDe 2026 paper tables from exp_014 / exp_015 artifacts.

Usage (from services/ml):
    uv run python scripts/build_paper_tables.py --run f33

Writes markdown to docs/dissertation/side2026_tables.md and prints it.

Design note on the aggregation: mean AUC per distance class is NOT comparable
across classes, because each class is a mean over a different set of target
cohorts (only OULAD/UKZN/KU cohorts with a sibling year contribute to D1, every
cohort contributes to D3). Every table here therefore aggregates PER TARGET
first and, for the ladder, restricts to targets that have all four distance
classes, so the columns describe the same cohorts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
EXP14 = REPO / "data" / "artifacts" / "experiments" / "exp_014_transfer_ladder"
EXP15 = REPO / "data" / "artifacts" / "experiments" / "exp_015_feature_richness"
OUT = REPO / "docs" / "dissertation" / "side2026_tables.md"

DIST = ["D0_within_cohort", "D1_same_module", "D2_other_module", "D3_other_institution"]
DIST_SHORT = {
    "D0_within_cohort": "D0 within cohort",
    "D1_same_module": "D1 same course, other year",
    "D2_other_module": "D2 other course, same inst.",
    "D3_other_institution": "D3 other institution",
}
MODEL_SHORT = {
    "gradient_boosting": "Gradient boosting",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([header, rule, *body])


def table_cohorts(cohorts: pd.DataFrame) -> str:
    g = cohorts.groupby("institution").agg(
        Cohorts=("cohort_id", "size"),
        Courses=("module", "nunique"),
        Students=("n_students", "sum"),
        **{"Students / cohort (min-max)": ("n_students", lambda s: f"{s.min()}-{s.max()}")},
        **{"Pass rate (min-max)": ("pass_rate", lambda s: f"{s.min():.2f}-{s.max():.2f}")},
    )
    g = g.reset_index().rename(columns={"institution": "Institution"})
    total = pd.DataFrame(
        [{
            "Institution": "**Total**",
            "Cohorts": g.Cohorts.sum(),
            "Courses": g.Courses.sum(),
            "Students": g.Students.sum(),
            "Students / cohort (min-max)": "",
            "Pass rate (min-max)": "",
        }]
    )
    return _md(pd.concat([g, total], ignore_index=True))


def table_ladder(pairs: pd.DataFrame) -> tuple[str, str]:
    """Paired ladder: only targets that appear at all four distances."""
    complete = pairs.groupby("target")["distance"].nunique()
    keep = set(complete[complete == len(DIST)].index)
    p = pairs[pairs["target"].isin(keep)]
    rows = []
    for (model, rep), g in p.groupby(["model", "representation"]):
        rec = {"Model": MODEL_SHORT.get(model, model), "Representation": rep}
        for d in DIST:
            sub = g[g["distance"] == d]
            rec[DIST_SHORT[d]] = sub.groupby("target")["auc"].mean().mean() if len(sub) else np.nan
        rec["D0-D3 loss"] = rec[DIST_SHORT["D0_within_cohort"]] - rec[DIST_SHORT["D3_other_institution"]]
        rows.append(rec)
    tbl = pd.DataFrame(rows).sort_values(["Model", "Representation"])
    note = f"Paired over the {len(keep)} target cohorts present at all four distances."
    return _md(tbl), note


def table_threshold(pairs: pd.DataFrame) -> str:
    """F1 on the FAILING class - the class an early-warning system alerts on.

    Paired over the same target cohorts as the AUC ladder, so the two tables are
    directly comparable.
    """
    complete = pairs.groupby("target")["distance"].nunique()
    pairs = pairs[pairs["target"].isin(set(complete[complete == len(DIST)].index))]
    rows = []
    for (model, rep), g in pairs.groupby(["model", "representation"]):
        rec = {"Model": MODEL_SHORT.get(model, model), "Representation": rep}
        for d in DIST:
            sub = g[g["distance"] == d]
            rec[DIST_SHORT[d]] = sub.groupby("target")["f1_fail"].mean().mean() if len(sub) else np.nan
        rows.append(rec)
    return _md(pd.DataFrame(rows).sort_values(["Model", "Representation"]))


def table_explanations(pairs: pd.DataFrame) -> str:
    """Agreement between the transferred model's ranking and the local model's."""
    rows = []
    for (model, rep), g in pairs.groupby(["model", "representation"]):
        rec = {"Model": MODEL_SHORT.get(model, model), "Representation": rep}
        for d in DIST[1:]:  # D0 is 1.0 by construction
            sub = g[g["distance"] == d]
            tau = pd.to_numeric(sub["explanation_tau"], errors="coerce")
            rec[DIST_SHORT[d] + " tau"] = tau.mean()
            rec[DIST_SHORT[d] + " J@3"] = sub["explanation_jaccard_top3"].mean()
        rows.append(rec)
    return _md(pd.DataFrame(rows).sort_values(["Model", "Representation"]))


def table_pooled(pooled: pd.DataFrame) -> str:
    g = pooled.groupby(["model", "representation", "pool", "target_institution"])["auc"].mean().reset_index()
    t = g.pivot_table(
        index=["model", "representation", "pool"], columns="target_institution", values="auc"
    ).reset_index()
    t["model"] = t["model"].map(lambda m: MODEL_SHORT.get(m, m))
    return _md(t.rename(columns={"model": "Model", "representation": "Representation", "pool": "Source pool"}))


def table_fewshot(few: pd.DataFrame) -> str:
    g = few.groupby(["target_institution", "k"])["auc"].agg(["mean", "std"]).reset_index()
    t = g.pivot(index="k", columns="target_institution", values="mean").reset_index()
    return _md(t.rename(columns={"k": "Labelled target students (k)"}))


def table_richness() -> str | None:
    f = EXP15 / "f33" / "summary.csv"
    if not f.exists():
        return None
    s = pd.read_csv(f)
    s = s.rename(
        columns={
            "feature_set": "Feature set",
            "n_features": "Features",
            "D0_within_cohort": "D0 within cohort",
            "D1_same_module": "D1 same course",
            "D2_other_module": "D2 other course",
            "drop_D0_to_D2": "D0-D2 loss",
        }
    )
    cols = ["Feature set", "Features", "D0 within cohort", "D1 same course", "D2 other course", "D0-D2 loss"]
    return _md(s[[c for c in cols if c in s.columns]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="f33")
    args = ap.parse_args()
    d = EXP14 / args.run
    pairs = pd.read_csv(d / "pairs.csv")
    cohorts = pd.read_csv(d / "cohorts.csv")

    ladder, note = table_ladder(pairs)
    parts = [
        "# SIDe 2026 - result tables",
        f"\nGenerated from `{d.relative_to(REPO)}`.\n",
        "## Table I. Benchmark composition\n",
        table_cohorts(cohorts),
        "\n## Table II. Transfer ladder, ROC-AUC\n",
        f"_{note}_\n",
        ladder,
        "\n## Table III. F1 at a fixed 0.5 threshold\n",
        table_threshold(pairs),
        "\n## Table IV. Explanation agreement with the local model\n",
        "_Kendall tau and top-3 Jaccard between the transferred model's permutation-importance "
        "ranking on the target and the ranking of a model trained on that target._\n",
        table_explanations(pairs),
        "\n## Table V. Pooled sources\n",
        table_pooled(pd.read_csv(d / "pooled.csv")),
    ]
    if (d / "fewshot.csv").exists():
        parts += ["\n## Table VI. Few-shot calibration on the target cohort\n", table_fewshot(pd.read_csv(d / "fewshot.csv"))]
    rich = table_richness()
    if rich:
        parts += ["\n## Table VII. Feature-richness control (OULAD only, gradient boosting)\n", rich]

    text = "\n".join(parts) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
