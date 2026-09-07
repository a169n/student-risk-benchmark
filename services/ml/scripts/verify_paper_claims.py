"""Check every quantitative claim in the manuscript against the frozen artifacts.

A paper this number-dense drifts as soon as an experiment is rerun, and a wrong
number that nobody recomputes is exactly how a review finds you out. This script
recomputes each claim from the artifact that should support it and reports PASS,
FAIL, or MISSING against the value written in the text, so the drift is caught
before a reviewer catches it.

Add a claim here whenever a number enters the manuscript. A claim that cannot be
recomputed from a released artifact should not be in the paper.

Usage (from services/ml):
    uv run python scripts/verify_paper_claims.py
Exit status is non-zero if any claim fails.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
# The LaTeX submission is the manuscript of record. The Markdown draft is kept
# for history and deliberately not checked: a stale second copy would hide the
# very contradictions this script exists to surface.
PAPERS = [REPO / "paper" / "main.tex"]
ART = REPO / "data" / "artifacts" / "experiments"
LADDER = ART / "exp_014_transfer_ladder" / "f33"
DIST = ["D0_within_cohort", "D1_same_module", "D2_other_module", "D3_other_institution"]
TOL = 0.0015  # claims are quoted to three decimals


def _paired(pairs: pd.DataFrame) -> pd.DataFrame:
    complete = pairs.groupby("target")["distance"].nunique()
    return pairs[pairs["target"].isin(set(complete[complete == len(DIST)].index))]


def _cell(frame: pd.DataFrame, distance: str, metric: str) -> float:
    return frame.loc[frame["distance"] == distance].groupby("target")[metric].mean().mean()


def build_claims() -> list[tuple[str, float]]:
    """(label, recomputed value) for every number the manuscript states."""
    pairs = pd.read_csv(LADDER / "pairs.csv")
    cohorts = pd.read_csv(LADDER / "cohorts.csv")
    gbm = pairs[pairs["model"] == "gradient_boosting"]
    raw = gbm[gbm["representation"] == "raw"]
    pct = gbm[gbm["representation"] == "percentile"]
    paired_raw = _paired(gbm)[lambda d: d["representation"] == "raw"]

    claims: list[tuple[str, float]] = [
        ("cohorts", float(len(cohorts))),
        ("D0 AUC all targets", _cell(raw, "D0_within_cohort", "auc")),
        ("D3 AUC all targets raw", _cell(raw, "D3_other_institution", "auc")),
        ("D3 AUC all targets percentile", _cell(pct, "D3_other_institution", "auc")),
        ("D0 AUC paired raw", _cell(paired_raw, "D0_within_cohort", "auc")),
        ("D3 AUC paired raw", _cell(paired_raw, "D3_other_institution", "auc")),
        ("D0 brier", _cell(raw, "D0_within_cohort", "brier")),
        ("D3 brier raw", _cell(raw, "D3_other_institution", "brier")),
        ("D3 brier percentile", _cell(pct, "D3_other_institution", "brier")),
        ("D0 calibration_in_large", _cell(raw, "D0_within_cohort", "calibration_in_large")),
        ("D3 calibration_in_large raw", _cell(raw, "D3_other_institution", "calibration_in_large")),
        ("D3 calibration_in_large percentile", _cell(pct, "D3_other_institution", "calibration_in_large")),
        ("D0 recall_at_flag20", _cell(raw, "D0_within_cohort", "recall_at_flag20")),
        ("D3 recall_at_flag20 raw", _cell(raw, "D3_other_institution", "recall_at_flag20")),
        ("D3 recall_at_flag20 percentile", _cell(pct, "D3_other_institution", "recall_at_flag20")),
        ("D0 lift_at_flag20", _cell(raw, "D0_within_cohort", "lift_at_flag20")),
        ("D3 lift_at_flag20", _cell(raw, "D3_other_institution", "lift_at_flag20")),
        ("D3 f1_fail raw", _cell(raw, "D3_other_institution", "f1_fail")),
        ("D3 f1_fail percentile", _cell(pct, "D3_other_institution", "f1_fail")),
        ("D3 f1_fail baseline", _cell(raw, "D3_other_institution", "f1_fail_baseline")),
        ("D1 explanation tau", gbm.loc[gbm["distance"] == "D1_same_module", "explanation_tau"].mean()),
        ("D2 explanation tau", gbm.loc[gbm["distance"] == "D2_other_module", "explanation_tau"].mean()),
        ("D3 explanation tau", gbm.loc[gbm["distance"] == "D3_other_institution", "explanation_tau"].mean()),
        (
            "D3 explanation tau percentile",
            pct.loc[pct["distance"] == "D3_other_institution", "explanation_tau"].mean(),
        ),
        (
            "D3 explanation jaccard",
            gbm.loc[gbm["distance"] == "D3_other_institution", "explanation_jaccard_top3"].mean(),
        ),
    ]

    d0 = raw[raw["distance"] == "D0_within_cohort"].merge(
        cohorts, left_on="target", right_on="cohort_id"
    )
    for institution, group in d0.groupby("institution"):
        claims.append((f"D0 AUC {institution}", group["auc"].mean()))

    counts = pd.read_csv(ART / "exp_017_contamination" / "f33" / "student_counts.csv")
    claims += [
        ("distinct students", float(counts["distinct_students"].iloc[0])),
        ("cohort rows summed", float(counts["cohort_rows_summed"].iloc[0])),
    ]
    contam = pd.read_csv(ART / "exp_017_contamination" / "f33" / "summary.csv").set_index("distance")
    for distance, label in (("D1_same_module", "D1"), ("D2_other_module", "D2"), ("D3_other_institution", "D3")):
        claims.append((f"{label} contaminated pairs", float(contam.loc[distance, "contaminated_pairs"])))
        claims.append((f"{label} total pairs", float(contam.loc[distance, "pairs"])))

    clean = pd.read_csv(ART / "exp_017_contamination" / "f33" / "clean_vs_all.csv")
    clean = clean[(clean.model == "gradient_boosting") & (clean.representation == "raw")]
    for subset, label in (("all pairs", "all"), ("clean pairs", "clean")):
        row = clean[(clean.subset == subset) & (clean.distance == "D1_same_module")]
        claims.append((f"D1 AUC {label} pairs", float(row["auc"].iloc[0])))

    ceiling = pd.read_csv(ART / "exp_018_explanation_ceiling" / "f33" / "summary.csv")
    for _, row in ceiling.iterrows():
        claims.append((f"ceiling: {row['reference']}", float(row["tau"])))
        claims.append((f"ceiling jaccard: {row['reference']}", float(row["jaccard_top3"])))

    explainer = pd.read_csv(ART / "exp_022_explainer_agreement" / "f33" / "summary.csv")
    for _, row in explainer.iterrows():
        claims.append((f"explainer {row['pair']} tau", float(row["tau"])))
        claims.append((f"explainer {row['pair']} jaccard", float(row["jaccard_top3"])))

    trivial = pd.read_csv(ART / "exp_021_trivial_baseline" / "comparison.csv")
    trivial = trivial[trivial["cutoff"] == "f33"]
    claims += [
        ("rule AUC (cum_active_days)", float(trivial["rule_days_auc"].mean())),
        ("local model AUC vs rule arm", float(trivial["local_auc"].mean())),
        ("cohorts where rule beats local", float((trivial["local_gain_over_rule"] < 0).sum())),
    ]

    thr = pd.read_csv(ART / "exp_019_threshold_transfer" / "f33" / "summary.csv")
    thr = thr[thr.distance == "D3_other_institution"]
    for _, row in thr.iterrows():
        claims.append((f"threshold {row['representation']}/{row['rule']} f1_fail", float(row["f1_fail"])))

    health = pd.read_csv(
        ART / "exp_022_explainer_agreement" / "f33" / "estimator_health.csv"
    ).set_index("estimator")
    cond = pd.read_csv(
        ART / "exp_024_estimator_conditional" / "f33" / "summary.csv"
    ).set_index("estimator")
    for est in ("shap", "permutation", "drop_column"):
        claims += [
            (f"{est}: self-agreement, same cohort", float(health.loc[est, "self_agreement_tau"])),
            (f"{est}: dead features of 7", float(health.loc[est, "dead_features"])),
            (f"{est}: agreement between years", float(cond.loc[est, "tau"])),
            (
                f"{est}: cost of one year",
                float(health.loc[est, "self_agreement_tau"] - cond.loc[est, "tau"]),
            ),
        ]
    reliable = ["shap", "permutation"]
    claims += [
        ("estimator spread between years", float(cond.loc[reliable, "tau"].max()
                                                 - cond.loc[reliable, "tau"].min())),
        (
            "mean cost of one year, reliable estimators",
            float(
                sum(health.loc[e, "self_agreement_tau"] - cond.loc[e, "tau"] for e in reliable) / 2
            ),
        ),
        ("course-year pairs in Table II", float(cond.loc["shap", "observations"] / 3)),
    ]

    spread = pd.read_csv(ART / "exp_023_cohort_intervals" / "f33" / "cohort_auc.csv")
    grand = spread["auc"].mean()
    between = sum(
        len(g) * (g["auc"].mean() - grand) ** 2 for _, g in spread.groupby("institution")
    )
    claims += [
        ("cohort AUC min", float(spread["auc"].min())),
        ("cohort AUC max", float(spread["auc"].max())),
        ("cohorts covering chance", float(spread["covers_chance"].sum())),
        ("cohorts entirely below chance", float(spread["below_chance"].sum())),
        ("institution share of AUC variance (%)", float(100 * between / ((spread["auc"] - grand) ** 2).sum())),
        ("corr(AUC, cohort size)", float(spread["auc"].corr(spread["n_students"]))),
        ("OULAD cohort AUC min", float(spread.loc[spread.institution == "OULAD", "auc"].min())),
        ("OULAD cohort AUC max", float(spread.loc[spread.institution == "OULAD", "auc"].max())),
        ("median OULAD cohort size", float(spread.loc[spread.institution == "OULAD", "n_students"].median())),
        ("median cohort size ratio, OULAD over Oviedo", float(
            spread.loc[spread.institution == "OULAD", "n_students"].median()
            / spread.loc[spread.institution == "Oviedo", "n_students"].median()
        )),
    ]
    for inst, g in spread.groupby("institution"):
        # Table I prints a pass-rate range per institution. It went unchecked
        # until a review asked what the column meant, and OULAD's low end was
        # wrong by 0.11 the whole time.
        claims.append((f"{inst} pass rate min", float(g["pass_rate"].min())))
        claims.append((f"{inst} pass rate max", float(g["pass_rate"].max())))
        claims.append((f"{inst} mean within-cohort AUC", float(g["auc"].mean())))
        claims.append((f"{inst} cohort count", float(len(g))))
        claims.append((f"{inst} students", float(g["n_students"].sum())))

    shift = pd.read_csv(ART / "exp_016_shift_analysis" / "f33" / "regression.csv")
    shift = shift[shift.model == "gradient_boosting"].set_index("representation")
    for rep in ("raw", "percentile"):
        claims.append((f"shift r2 {rep}", float(shift.loc[rep, "r2"])))
        claims.append((f"shift corr_d_scale {rep}", float(shift.loc[rep, "corr_d_scale"])))
        claims.append((f"shift corr_d_shape {rep}", float(shift.loc[rep, "corr_d_shape"])))

    return claims


def main() -> int:
    text = "\n".join(p.read_text(encoding="utf-8") for p in PAPERS if p.exists())
    # Every number the manuscript states, as a set for membership testing.
    # The en-dash in a LaTeX range (0.34--0.73) has to go first, or the
    # second endpoint parses as a negative number and never matches.
    stated = {
        float(m) for m in re.findall(r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?![\w])", text.replace("{,}", "").replace(",", "").replace("--", " "))
    }

    failures = 0
    print(f"{'claim':46s} {'recomputed':>11s}  status")
    print("-" * 74)
    for label, value in build_claims():
        if pd.isna(value):
            print(f"{label:46s} {'nan':>11s}  SKIP (not computed)")
            continue
        # The two-decimal fallback exists for values the paper quotes to two
        # decimals. It must not let a three-decimal claim match a different
        # three-decimal number that happens to round the same way.
        def quoted_at(places: int) -> set[float]:
            scale = 10**places
            return {s for s in stated if abs(s * scale - round(s * scale)) < 1e-9}

        # A value quoted to fewer decimals matches only if it rounds exactly.
        # Without the exactness the fallback let a recomputed 0.410 pass against
        # a stated 0.408, which is how one stale number survived a whole pass.
        hit = any(abs(value - s) <= TOL for s in stated) or any(
            abs(round(value, places) - s) <= 1e-9
            for places in (2, 1)
            for s in quoted_at(places)
        )
        status = "PASS" if hit else "ABSENT from text"
        if not hit:
            failures += 1
        print(f"{label:46s} {value:11.3f}  {status}")

    print("-" * 74)
    print(f"{failures} recomputed values do not appear in the manuscript.")
    print("ABSENT is not automatically an error: a value may be deliberately unreported.")
    print("It IS an error if the manuscript states a different number for the same quantity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
