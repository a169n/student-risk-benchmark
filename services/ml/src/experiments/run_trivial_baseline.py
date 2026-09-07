"""exp_021 — does the model beat a rule the institution already has?

Every number in the transfer ladder is compared against chance, or against
another model. Neither is the relevant comparison. An institution that owns its
LMS can already sort its students by how many days each has logged in, with no
model, no training data, no source cohort and no labels. If a trained model does
not beat that sort, the model is not the thing producing the value.

This experiment supplies the missing baseline at three levels:

    rule            rank students by a single raw counter at the cutoff
    local model     the cohort's own cross-validated model (ladder D0)
    transferred     a model trained at another institution (ladder D3)

and repeats it at all three cutoffs. It also asks the follow-up question the
result forces: is the failure a property of machine learning here, or of the
deliberately impoverished seven-feature schema that had to be common to five
platforms? The OULAD-only arm answers that by comparing the rule against the
richer feature sets of exp_015, which include intermediate assessment scores.

Usage (from services/ml):
    uv run python -m src.experiments.run_trivial_baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.experiments import transfer_benchmark as tb
from src.experiments.run_transfer_ladder import (
    load_ku,
    load_oulad,
    load_oviedo,
    load_ukzn,
    load_zambia,
)

REPO = Path(__file__).resolve().parents[4]
EXP14 = REPO / "data" / "artifacts" / "experiments" / "exp_014_transfer_ladder"
EXP15 = REPO / "data" / "artifacts" / "experiments" / "exp_015_feature_richness"
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_021_trivial_baseline"

# Single counters an institution can sort on without fitting anything.
RULES = ("cum_active_days", "cum_clicks", "active_weeks", "weeks_since_active")
FLAG_RATE = 0.20


def _recall_at_budget(fail: np.ndarray, risk: np.ndarray) -> float:
    n_flag = max(1, int(round(FLAG_RATE * len(risk))))
    flagged = np.zeros(len(risk), dtype=int)
    flagged[np.argsort(-risk)[:n_flag]] = 1
    return float((flagged & fail).sum() / fail.sum()) if fail.sum() else float("nan")


def rule_scores(cohorts: list[tb.Cohort]) -> pd.DataFrame:
    """Every zero-training rule on every cohort."""
    rows = []
    for cohort in cohorts:
        fail = 1 - cohort.y
        for rule in RULES:
            # More activity means less risk, except for the recency counter,
            # where a longer gap since the last visit means more risk.
            sign = 1.0 if rule == "weeks_since_active" else -1.0
            risk = sign * cohort.X_raw[rule].to_numpy(dtype=float)
            rows.append(
                {
                    "cohort_id": cohort.cohort_id,
                    "institution": cohort.institution,
                    "rule": rule,
                    "n_students": cohort.n,
                    "auc": float(roc_auc_score(cohort.y, -risk)),
                    "recall_at_flag20": _recall_at_budget(fail, risk),
                }
            )
    return pd.DataFrame(rows)


def compare_with_models(rules: pd.DataFrame, run: str) -> pd.DataFrame:
    """Join the best rule per cohort against the ladder's local and transferred models."""
    pairs = pd.read_csv(EXP14 / run / "pairs.csv")
    gbm = pairs[(pairs["model"] == "gradient_boosting") & (pairs["representation"] == "raw")]
    local = gbm[gbm["distance"] == "D0_within_cohort"].groupby("target")[
        ["auc", "recall_at_flag20"]
    ].mean()
    transferred = gbm[gbm["distance"] == "D3_other_institution"].groupby("target")[
        ["auc", "recall_at_flag20"]
    ].mean()

    best = rules.loc[rules.groupby("cohort_id")["auc"].idxmax()].set_index("cohort_id")
    days = rules[rules["rule"] == "cum_active_days"].set_index("cohort_id")

    frame = pd.DataFrame(
        {
            "institution": days["institution"],
            "n_students": days["n_students"],
            "rule_days_auc": days["auc"],
            "rule_days_recall": days["recall_at_flag20"],
            "rule_best_auc": best["auc"],
            "rule_best_name": best["rule"],
            "local_auc": local["auc"],
            "local_recall": local["recall_at_flag20"],
            "transferred_auc": transferred["auc"],
            "transferred_recall": transferred["recall_at_flag20"],
        }
    ).dropna(subset=["local_auc"])
    frame["local_gain_over_rule"] = frame["local_auc"] - frame["rule_days_auc"]
    frame["transferred_gain_over_rule"] = frame["transferred_auc"] - frame["rule_days_auc"]
    return frame.reset_index().rename(columns={"index": "cohort_id"})


def richness_arm() -> pd.DataFrame | None:
    """Does a richer feature set beat the rule where richer data exists (OULAD)?"""
    path = EXP15 / "f33" / "summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)[["feature_set", "n_features", "D0_within_cohort"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.25,0.33,0.50")
    args = ap.parse_args()

    weekly = {**load_oulad(), **load_ku(), **load_ukzn(), **load_zambia(), **load_oviedo()}
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in weekly.items()}

    all_rules, all_cmp = [], []
    for fraction in [float(x) for x in args.fractions.split(",")]:
        run = f"f{int(round(fraction * 100)):02d}"
        if not (EXP14 / run / "pairs.csv").exists():
            print(f"skip {run}: no ladder artifact")
            continue
        cohorts = tb.make_cohorts(canon, fraction)
        rules = rule_scores(cohorts).assign(cutoff=run)
        comparison = compare_with_models(rules, run).assign(cutoff=run)
        all_rules.append(rules)
        all_cmp.append(comparison)

        print(f"\n=== cutoff {run}: {len(cohorts)} cohorts ===")
        print(
            comparison[
                ["rule_days_auc", "local_auc", "transferred_auc", "rule_days_recall",
                 "local_recall", "transferred_recall"]
            ].mean().round(4).to_string()
        )
        beats_local = (comparison["local_gain_over_rule"] < 0).sum()
        beats_tr = (comparison["transferred_gain_over_rule"] < 0).sum()
        print(f"  rule beats the local model in {beats_local}/{len(comparison)} cohorts")
        print(f"  rule beats the transferred model in {beats_tr}/{len(comparison)} cohorts")

    rules_all = pd.concat(all_rules, ignore_index=True)
    cmp_all = pd.concat(all_cmp, ignore_index=True)

    by_institution = (
        cmp_all[cmp_all["cutoff"] == "f33"]
        .groupby("institution")[["rule_days_auc", "local_auc", "local_gain_over_rule"]]
        .mean()
        .reset_index()
    )
    print("\n=== per institution at the 1/3 cutoff ===")
    print(by_institution.round(3).to_string(index=False))

    richness = richness_arm()
    if richness is not None:
        oulad_rule = cmp_all[(cmp_all.cutoff == "f33") & (cmp_all.institution == "OULAD")][
            "rule_days_auc"
        ].mean()
        richness = richness.assign(rule_days_auc=oulad_rule)
        richness["gain_over_rule"] = richness["D0_within_cohort"] - oulad_rule
        print("\n=== OULAD only: does richer data beat the rule? ===")
        print(richness.round(3).to_string(index=False))

    frames = {"rules": rules_all, "comparison": cmp_all, "by_institution": by_institution}
    if richness is not None:
        frames["richness_vs_rule"] = richness
    tb.write_outputs(OUT, **frames)


if __name__ == "__main__":
    main()
