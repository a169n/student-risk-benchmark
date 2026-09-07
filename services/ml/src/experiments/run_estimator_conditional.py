"""exp_024 — ask one question three times, changing only the importance estimator.

The question is one the literature does ask: how much does retraining on the
next year's students change a model's feature ranking? Table II of the paper
answers it under SHAP, under permutation importance and under drop-column, on
the same course-year pairs and the same fitted models, and the answers differ by
more than the effect being measured. Those numbers were computed by hand once;
this releases them, because a paper whose point is that unreported choices move
results should not itself carry a number nobody can recompute.

Protocol, deliberately the same as exp_022 so the two tables sit on one scale:
each cohort is split two-thirds / one-third, the model is fitted on the larger
part and every estimator is evaluated on the held-out third. It is not the
ladder's protocol --- the ladder scores a source model on the whole target
cohort --- which is why the permutation figure here does not equal the ladder's
D1 rung, and the paper says so rather than quietly presenting them as the same
measurement.

Usage (from services/ml):
    uv run python -m src.experiments.run_estimator_conditional --fraction 0.33
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.experiments import transfer_benchmark as tb
from src.experiments.run_explainer_agreement import EVAL_CAP, rankings
from src.experiments.run_transfer_ladder import (
    load_ku,
    load_oulad,
    load_oviedo,
    load_ukzn,
    load_zambia,
)
from src.experiments.stability import jaccard_topk, kendall_tau

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_024_estimator_conditional"
ESTIMATORS = ("shap", "permutation", "drop_column")
TOP_K = 3


def cohort_rankings(cohort: tb.Cohort, seed: int) -> dict[str, list[str]] | None:
    X, y = cohort.X("raw"), cohort.y
    try:
        tr, ev = train_test_split(np.arange(len(y)), test_size=1 / 3, stratify=y, random_state=seed)
    except ValueError:
        return None
    if min(len(set(y[tr])), len(set(y[ev]))) < 2:
        return None
    if len(ev) > EVAL_CAP:
        ev = ev[:EVAL_CAP]
    return rankings(X[tr], y[tr], X[ev], y[ev], seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.33)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--run", default="f33")
    args = ap.parse_args()

    weekly = {**load_oulad(), **load_ku(), **load_ukzn(), **load_zambia(), **load_oviedo()}
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in weekly.items()}
    cohorts = tb.make_cohorts(canon, args.fraction)
    print(f"cohorts: {len(cohorts)}", flush=True)

    # One ranking per (cohort, estimator, split). Pairs are then compared within
    # a split, so a pair's agreement never mixes two different held-out thirds.
    ranks: dict[int, dict[str, dict[str, list[str]]]] = {s: {} for s in range(args.seeds)}
    for i, c in enumerate(cohorts, 1):
        for seed in range(args.seeds):
            r = cohort_rankings(c, seed)
            if r is not None:
                ranks[seed][c.cohort_id] = r
        print(f"  rankings {i}/{len(cohorts)} {c.cohort_id}", flush=True)

    rows = []
    for s in cohorts:
        for t in cohorts:
            if tb.distance_class(s, t) != "D1_same_module":
                continue
            for seed in range(args.seeds):
                r = ranks[seed]
                if s.cohort_id not in r or t.cohort_id not in r:
                    continue
                for est in ESTIMATORS:
                    a, b = r[s.cohort_id][est], r[t.cohort_id][est]
                    rows.append(
                        {
                            "source": s.cohort_id,
                            "target": t.cohort_id,
                            "institution": t.institution,
                            "seed": seed,
                            "estimator": est,
                            "tau": kendall_tau(a, b),
                            "jaccard_top3": jaccard_topk(a, b, TOP_K),
                        }
                    )

    pairs = pd.DataFrame(rows)
    if pairs.empty:
        raise SystemExit("no same-course year pairs; nothing to report")
    summary = (
        pairs.groupby("estimator")
        .agg(observations=("tau", "size"), tau=("tau", "mean"), tau_sd=("tau", "std"),
             jaccard_top3=("jaccard_top3", "mean"))
        .reset_index()
        .sort_values("tau", ascending=False)
    )

    reliable = summary[summary.estimator != "drop_column"]
    spread = float(reliable["tau"].max() - reliable["tau"].min())

    out = OUT / args.run
    tb.write_outputs(out, pairs=pairs, summary=summary)
    tb.dump_json(
        out / "run_metadata.json",
        {
            "experiment_id": "exp_024_estimator_conditional",
            "fraction": args.fraction,
            "seeds": args.seeds,
            "n_pairs_per_estimator": int(len(pairs) / len(ESTIMATORS) / args.seeds),
            "estimators": list(ESTIMATORS),
            "spread_between_reliable_estimators": spread,
            "note": "drop_column is degenerate on this collinear feature set (exp_022)",
        },
    )
    print(summary.round(3).to_string(index=False), flush=True)
    print(f"\nspread between the two reliable estimators: {spread:.3f}", flush=True)
    print("written to", out, flush=True)


if __name__ == "__main__":
    main()
