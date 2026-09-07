"""exp_017 — how much of each target cohort was already in the source cohort?

A transfer benchmark silently assumes that source and target contain different
students. Inside one institution that assumption fails: the same undergraduates
appear in several courses of the same semester and in consecutive years of the
same course. A "cross-course" evaluation that shares 90 % of its target with its
training set is not measuring transfer, it is measuring memorisation, and it
inflates the near rungs of the ladder relative to the cross-institution rung.

This audit computes, for every ordered pair of cohorts, the share of the TARGET
cohort's students that also appear in the SOURCE cohort. The output is joined
onto ``pairs.csv`` by (source, target) so any result can be recomputed on
uncontaminated pairs only.

Usage (from services/ml):
    uv run python -m src.experiments.run_contamination_audit --fraction 0.33
"""

from __future__ import annotations

import argparse
import itertools
from collections import Counter
from pathlib import Path

import pandas as pd

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
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_017_contamination"

CLEAN_THRESHOLD = 0.01  # a pair is "clean" when under 1 % of the target was seen in training


def cohort_student_sets(fraction: float) -> tuple[dict[str, set[str]], list[tb.Cohort]]:
    weekly = {**load_oulad(), **load_ku(), **load_ukzn(), **load_zambia(), **load_oviedo()}
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in weekly.items()}
    cohorts = tb.make_cohorts(canon, fraction)
    ids = {
        c.cohort_id: set(canon[c.cohort_id]["student_id"].astype(str).unique()) for c in cohorts
    }
    return ids, cohorts


def pair_overlap(ids: dict[str, set[str]], cohorts: list[tb.Cohort]) -> pd.DataFrame:
    rows = []
    for source, target in itertools.product(cohorts, cohorts):
        if source.cohort_id == target.cohort_id:
            continue
        shared = len(ids[source.cohort_id] & ids[target.cohort_id])
        rows.append(
            {
                "source": source.cohort_id,
                "target": target.cohort_id,
                "distance": tb.distance_class(source, target),
                "shared_students": shared,
                "target_size": len(ids[target.cohort_id]),
                "contamination": shared / len(ids[target.cohort_id]),
            }
        )
    frame = pd.DataFrame(rows)
    frame["is_clean"] = frame["contamination"] < CLEAN_THRESHOLD
    return frame


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("distance")
        .agg(
            pairs=("contamination", "size"),
            contaminated_pairs=("contamination", lambda s: int((s > 0).sum())),
            clean_pairs=("is_clean", "sum"),
            mean_contamination=("contamination", "mean"),
            max_contamination=("contamination", "max"),
        )
        .reset_index()
    )


def recompute_clean(pairs_path: Path, overlap: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Ladder means on all pairs versus on uncontaminated pairs only."""
    pairs = pd.read_csv(pairs_path).merge(overlap, on=["source", "target", "distance"], how="left")
    pairs["is_clean"] = pairs["is_clean"].fillna(True)  # D0 rows have no overlap entry
    rows = []
    for (model, rep), g in pairs.groupby(["model", "representation"]):
        for subset, sel in (("all pairs", g), ("clean pairs", g[g["is_clean"]])):
            for distance, h in sel.groupby("distance"):
                rec = {
                    "model": model,
                    "representation": rep,
                    "subset": subset,
                    "distance": distance,
                    "n_pairs": len(h),
                }
                for metric in metrics:
                    if metric in h.columns:
                        rec[metric] = h.groupby("target")[metric].mean().mean()
                rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.33)
    ap.add_argument("--run", default="f33")
    args = ap.parse_args()

    ids, cohorts = cohort_student_sets(args.fraction)
    overlap = pair_overlap(ids, cohorts)
    summary = summarise(overlap)

    distinct = len(set().union(*ids.values()))
    rows = sum(len(v) for v in ids.values())
    # Per-student cohort membership, so the share of STUDENTS who recur can be
    # reported without being mistaken for the share of rows that are repeats.
    appearances = Counter(sid for members in ids.values() for sid in members)
    multi = sum(1 for n in appearances.values() if n > 1)
    counts = pd.DataFrame(
        [
            {
                "cohort_rows_summed": rows,
                "distinct_students": distinct,
                "duplication_rate": 1 - distinct / rows,
                "students_in_multiple_cohorts": multi,
                "share_of_students_recurring": multi / distinct,
                "max_cohorts_per_student": max(appearances.values()),
            }
        ]
    )

    out = OUT / args.run
    frames = {"pair_overlap": overlap, "summary": summary, "student_counts": counts}
    pairs_path = EXP14 / args.run / "pairs.csv"
    if pairs_path.exists():
        frames["clean_vs_all"] = recompute_clean(
            pairs_path, overlap, ["auc", "f1_fail", "recall_at_flag20", "brier"]
        )
    tb.write_outputs(out, **frames)

    print(summary.round(3).to_string(index=False))
    print("\nstudent counts:")
    print(counts.round(3).to_string(index=False))
    if "clean_vs_all" in frames:
        c = frames["clean_vs_all"]
        c = c[(c.model == "gradient_boosting") & (c.representation == "raw")]
        print("\ngradient boosting / raw, all pairs vs clean pairs:")
        print(c.pivot(index="distance", columns="subset", values="auc").round(3).to_string())


if __name__ == "__main__":
    main()
