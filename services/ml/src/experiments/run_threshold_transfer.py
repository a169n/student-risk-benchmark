"""exp_019 — is the cohort-relative representation better than simply moving the threshold?

The ladder shows that a transferred model keeps much of its ranking ability but
loses its calibration, and that a within-cohort percentile representation
restores calibration-in-the-large. That raises an obvious competing explanation:
the percentile transform forces every cohort's feature marginals to Uniform[0,1],
so the transferred model reproduces its SOURCE's alert rate on the target. If
that is all it does, a one-line threshold move achieves the same thing without
touching the features, and the representation is not the contribution.

This experiment compares four label-free deployment rules on the same fitted
models, over the same ordered cohort pairs:

    raw + fixed 0.5          the naive rule the ladder reports
    raw + source-rate        flag the share of the target that the SOURCE cohort
                             had failing (uses only source labels, never target)
    percentile + fixed 0.5   the paper's proposed transform
    percentile + source-rate both together

Only gradient boosting is run: the question is about the decision rule, not the
model family. Metrics are the ones a deployment cares about — F1 on the failing
class against the trivial "alert everyone" rule, and calibration-in-the-large.

Usage (from services/ml):
    uv run python -m src.experiments.run_threshold_transfer --fraction 0.33
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.experiments import transfer_benchmark as tb
from src.experiments.models import build_classification_model
from src.experiments.run_transfer_ladder import (
    load_ku,
    load_oulad,
    load_oviedo,
    load_ukzn,
    load_zambia,
)

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_019_threshold_transfer"
MODEL = "gradient_boosting"


def _rule_scores(fail: np.ndarray, risk: np.ndarray, flagged: np.ndarray) -> dict[str, float]:
    prevalence = float(fail.mean())
    n_flag = int(flagged.sum())
    caught = int((flagged & fail).sum())
    return {
        "f1_fail": float(f1_score(fail, flagged, zero_division=0)),
        "f1_fail_baseline": float(2 * prevalence / (1 + prevalence)) if prevalence else 0.0,
        "alert_rate": float(flagged.mean()),
        "recall": float(caught / fail.sum()) if fail.sum() else float("nan"),
        "precision": float(caught / n_flag) if n_flag else float("nan"),
        "calibration_in_large": float(risk.mean() - prevalence),
    }


def evaluate_pair(source: tb.Cohort, target: tb.Cohort, models: dict, seed: int) -> list[dict]:
    """Four deployment rules on one ordered pair."""
    fail_t = 1 - target.y
    source_fail_rate = float((1 - source.y).mean())
    rows = []
    for representation in ("raw", "percentile"):
        risk = 1.0 - models[(source.cohort_id, representation)].predict_proba(
            target.X(representation)
        )[:, 1]

        fixed = (risk > 0.5).astype(int)

        # Source-rate rule: flag exactly the share of students the source cohort
        # had failing. Uses no target label, only the source's own base rate.
        n_flag = max(1, int(round(source_fail_rate * len(risk))))
        by_rate = np.zeros(len(risk), dtype=int)
        by_rate[np.argsort(-risk)[:n_flag]] = 1

        for rule, flagged in (("fixed_0.5", fixed), ("source_rate", by_rate)):
            rows.append(
                {
                    "source": source.cohort_id,
                    "target": target.cohort_id,
                    "target_institution": target.institution,
                    "distance": tb.distance_class(source, target),
                    "representation": representation,
                    "rule": rule,
                    **_rule_scores(fail_t, risk, flagged),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run", default="f33")
    args = ap.parse_args()

    weekly = {**load_oulad(), **load_ku(), **load_ukzn(), **load_zambia(), **load_oviedo()}
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in weekly.items()}
    cohorts = tb.make_cohorts(canon, args.fraction)
    print(f"cohorts: {len(cohorts)}", flush=True)

    models = {
        (c.cohort_id, rep): build_classification_model(MODEL, seed=args.seed).fit(c.X(rep), c.y)
        for c in cohorts
        for rep in ("raw", "percentile")
    }
    print(f"fitted {len(models)} source models", flush=True)

    rows = [
        row
        for source, target in itertools.product(cohorts, cohorts)
        if source.cohort_id != target.cohort_id
        for row in evaluate_pair(source, target, models, args.seed)
    ]
    frame = pd.DataFrame(rows)

    summary = (
        frame.groupby(["distance", "representation", "rule"])
        .apply(
            lambda g: pd.Series(
                {
                    "n_pairs": len(g),
                    "f1_fail": g.groupby("target")["f1_fail"].mean().mean(),
                    "f1_fail_baseline": g.groupby("target")["f1_fail_baseline"].mean().mean(),
                    "alert_rate": g.groupby("target")["alert_rate"].mean().mean(),
                    "recall": g.groupby("target")["recall"].mean().mean(),
                    "precision": g.groupby("target")["precision"].mean().mean(),
                    "beats_baseline_share": (g["f1_fail"] > g["f1_fail_baseline"]).mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    tb.write_outputs(OUT / args.run, pairs=frame, summary=summary)
    print("\n=== D3 (other institution) ===")
    print(summary[summary.distance == "D3_other_institution"].round(3).to_string(index=False))
    print("\n=== all distances, F1 on the failing class ===")
    print(
        summary.pivot_table(
            index="distance", columns=["representation", "rule"], values="f1_fail"
        ).round(3).to_string()
    )


if __name__ == "__main__":
    main()
