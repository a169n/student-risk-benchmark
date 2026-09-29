"""Confidence intervals for every explanation-agreement number the paper quotes.

A reviewer asked for intervals or tests on the explanation-stability differences,
above all on the estimator effect in Table II. Each agreement figure is a mean over
clustered observations (three seeds per cohort, or per course-year pair), so the
interval is a percentile bootstrap that resamples the clusters, not the rows. The
estimator spread is a paired difference on the same 78 course-year pairs, so it
gets a paired bootstrap and a Wilcoxon signed-rank test as well.

Writes data/artifacts/experiments/exp_024_estimator_conditional/f33/intervals.csv,
which the text, Table II and the agreement figure all read from.

Usage (from services/ml):
    uv run python scripts/estimator_intervals.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
ART = REPO / "data" / "artifacts" / "experiments"
OUT = ART / "exp_024_estimator_conditional" / "f33" / "intervals.csv"
N_BOOT = 10_000
SEED = 0


def cluster_ci(frame: pd.DataFrame, cluster: str, value: str, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile interval of the pooled mean, resampling whole clusters.

    Pooling the resampled rows (rather than averaging cluster means) keeps the
    point estimate identical to the plain mean the paper already quotes, so the
    interval brackets the number in the text rather than a neighbour of it."""
    groups = [g[value].to_numpy() for _, g in frame.groupby(cluster)]
    n = len(groups)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = np.concatenate([groups[k] for k in rng.integers(0, n, n)]).mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    def add(label: str, frame: pd.DataFrame, *, cluster: str, value: str = "tau", p_value: float = float("nan")) -> None:
        lo, hi = cluster_ci(frame, cluster, value, rng)
        rows.append(
            {
                "label": label,
                "tau": float(frame[value].mean()),
                "ci_low": lo,
                "ci_high": hi,
                "n_clusters": int(frame[cluster].nunique()),
                "cluster": cluster,
                "p_value": p_value,
            }
        )

    # Table II: next year's students. Cluster = course-year pair (three seeds each).
    cond = pd.read_csv(ART / "exp_024_estimator_conditional" / "f33" / "pairs.csv")
    cond["pair"] = cond["source"] + ">" + cond["target"]
    for est in ("shap", "permutation", "drop_column"):
        add(f"{est}: next year", cond[cond.estimator == est], cluster="pair")
    per_pair = cond.groupby(["pair", "estimator"])["tau"].mean().unstack()
    diff = (per_pair["shap"] - per_pair["permutation"]).rename("tau").reset_index()
    add(
        "spread: shap minus permutation, next year",
        diff,
        cluster="pair",
        p_value=float(stats.wilcoxon(per_pair["shap"], per_pair["permutation"]).pvalue),
    )

    # Table II: same cohort, another split. Per-cohort self-agreement is written by
    # exp_022 only since the revision; without it the cost-of-one-year interval is
    # skipped and the table keeps the point estimate alone.
    health_path = ART / "exp_022_explainer_agreement" / "f33" / "estimator_health_by_cohort.csv"
    if health_path.exists():
        health = pd.read_csv(health_path)
        for est in ("shap", "permutation", "drop_column"):
            add(
                f"{est}: same cohort, another split",
                health[health.estimator == est].rename(columns={"self_agreement_tau": "tau"}),
                cluster="cohort_id",
            )
        # Cost of one year per course-year pair and estimator: the target cohort's
        # own self-agreement minus its agreement with last year's model.
        ref = health.pivot(index="cohort_id", columns="estimator", values="self_agreement_tau")
        cost = per_pair.reset_index()
        cost["target"] = cost["pair"].str.split(">").str[1]
        for est in ("shap", "permutation", "drop_column"):
            # The reference the pairs are read against is the self-agreement of
            # the cohorts in those pairs, not of all 63: differencing a 63-cohort
            # mean from a 78-pair mean mixes populations, which is what the
            # first submission did.
            cost[f"ref_{est}"] = cost["target"].map(ref[est])
            add(f"{est}: same cohort, another split, pair targets", cost, cluster="pair", value=f"ref_{est}")
            cost[f"cost_{est}"] = cost[f"ref_{est}"] - cost[est]
            add(f"{est}: cost of one year", cost, cluster="pair", value=f"cost_{est}")
        cost["cost_diff"] = cost["cost_shap"] - cost["cost_permutation"]
        add(
            "cost difference: shap minus permutation",
            cost,
            cluster="pair",
            value="cost_diff",
            p_value=float(stats.wilcoxon(cost["cost_shap"], cost["cost_permutation"]).pvalue),
        )

    # Section V-B: agreement between estimators. Cluster = cohort.
    agree = pd.read_csv(ART / "exp_022_explainer_agreement" / "f33" / "pairs.csv")
    for pair in sorted(agree["pair"].unique()):
        add(f"explainer: {pair}", agree[agree.pair == pair], cluster="cohort_id")

    # Floor and ceiling. Cluster = cohort.
    ceil = pd.read_csv(ART / "exp_018_explanation_ceiling" / "f33" / "per_cohort.csv")
    add("floor: same model, reseeded", ceil, cluster="cohort_id", value="floor_tau")
    add("ceiling: disjoint halves, ladder-matched", ceil, cluster="cohort_id", value="ceiling_full_tau")

    # Ladder rungs on clean pairs. Cluster = target cohort.
    pairs = pd.read_csv(ART / "exp_014_transfer_ladder" / "f33" / "pairs.csv")
    overlap = pd.read_csv(ART / "exp_017_contamination" / "f33" / "pair_overlap.csv")
    gbm = pairs[(pairs.model == "gradient_boosting") & (pairs.representation == "raw")].merge(
        overlap[["source", "target", "is_clean"]], on=["source", "target"], how="left"
    )
    gbm["is_clean"] = gbm["is_clean"].astype("object").fillna(True).astype(bool)
    clean = gbm[gbm.is_clean]
    for distance, label in (
        ("D1_same_module", "D1: same course, next year"),
        ("D2_other_module", "D2: another course, same institution"),
        ("D3_other_institution", "D3: another institution"),
    ):
        add(label, clean[clean.distance == distance], cluster="target", value="explanation_tau")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.round(4).to_string(index=False))
    print("\nwritten to", OUT)


if __name__ == "__main__":
    main()
