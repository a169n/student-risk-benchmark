"""exp_018 — what agreement between two feature rankings is even attainable?

The transfer ladder reports that a transferred model's permutation-importance
ranking barely agrees with a locally trained model's. That number is
uninterpretable on its own, because two rankings of seven correlated features
never agree perfectly even under ideal conditions. This experiment measures the
three reference points the claim needs:

    floor      two rankings of the SAME fitted model on the SAME data, differing
               only in the permutation seed. Anything at this level is estimator
               noise.
    ceiling    two models trained on DISJOINT HALVES of the same cohort, both
               ranked on a held-out third of that cohort. Same institution, same
               course, same year, same everything except the training sample.
               This is the best agreement a ranking comparison can produce.
    random     the analytic expectation for two independent random rankings.

Both models in the ceiling condition are scored OUT of sample, which fixes an
asymmetry in the ladder: there the local model was ranked on data it had been
fitted on while the transferred model had not seen the target at all, so part of
the gap was in-sample versus out-of-sample rather than local versus foreign.

Usage (from services/ml):
    uv run python -m src.experiments.run_explanation_ceiling --fraction 0.33
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.experiments import transfer_benchmark as tb
from src.experiments.models import build_classification_model
from src.experiments.run_transfer_ladder import (
    load_ku,
    load_oulad,
    load_oviedo,
    load_ukzn,
    load_zambia,
)
from src.experiments.stability import jaccard_topk, kendall_tau

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_018_explanation_ceiling"

MODEL = "gradient_boosting"
TOP_K = 3


def random_jaccard_topk(n_features: int, k: int) -> float:
    """E[Jaccard of the top-k of two independent uniformly random rankings]."""
    # |A n B| is hypergeometric(n, k, k); Jaccard = i / (2k - i).
    total = 0.0
    for i in range(0, k + 1):
        p = (
            math.comb(k, i)
            * math.comb(n_features - k, k - i)
            / math.comb(n_features, k)
        )
        total += p * (i / (2 * k - i) if 2 * k - i else 1.0)
    return total


def _fit_and_rank(X_train, y_train, X_eval, y_eval, seed: int) -> list[str]:
    model = build_classification_model(MODEL, seed=seed).fit(X_train, y_train)
    return tb._ranking(model, X_eval, y_eval, seed)


def cohort_reference_points(cohort: tb.Cohort, representation: str, seed: int) -> dict | None:
    """Floor and ceiling for one cohort, both evaluated out of sample."""
    X, y = cohort.X(representation), cohort.y
    try:
        rest_idx, eval_idx = train_test_split(
            np.arange(len(y)), test_size=1 / 3, stratify=y, random_state=seed
        )
        half_a, half_b = train_test_split(
            rest_idx, test_size=0.5, stratify=y[rest_idx], random_state=seed
        )
    except ValueError:  # a class too small to stratify three ways
        return None
    if min(len(set(y[half_a])), len(set(y[half_b])), len(set(y[eval_idx]))) < 2:
        return None

    X_eval, y_eval = X[eval_idx], y[eval_idx]
    rank_a = _fit_and_rank(X[half_a], y[half_a], X_eval, y_eval, seed)
    rank_b = _fit_and_rank(X[half_b], y[half_b], X_eval, y_eval, seed)

    # Second ceiling, matched to the ladder's evaluation set: the ladder ranks
    # every model on the FULL target cohort, so a ceiling scored on a held-out
    # third is measured on less data and is pessimistic by comparison. Here both
    # half-trained models rank on the full cohort, exactly as the ladder does.
    rank_a_full = _fit_and_rank(X[half_a], y[half_a], X, y, seed)
    rank_b_full = _fit_and_rank(X[half_b], y[half_b], X, y, seed)

    # Floor: one fitted model, two permutation seeds, same evaluation data.
    model = build_classification_model(MODEL, seed=seed).fit(X[half_a], y[half_a])
    floor_a = tb._ranking(model, X_eval, y_eval, seed)
    floor_b = tb._ranking(model, X_eval, y_eval, seed + 1000)

    return {
        "cohort_id": cohort.cohort_id,
        "institution": cohort.institution,
        "representation": representation,
        "seed": seed,
        "n_students": cohort.n,
        "floor_tau": kendall_tau(floor_a, floor_b),
        "floor_jaccard": jaccard_topk(floor_a, floor_b, TOP_K),
        "ceiling_tau": kendall_tau(rank_a, rank_b),
        "ceiling_jaccard": jaccard_topk(rank_a, rank_b, TOP_K),
        "ceiling_full_tau": kendall_tau(rank_a_full, rank_b_full),
        "ceiling_full_jaccard": jaccard_topk(rank_a_full, rank_b_full, TOP_K),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.33)
    ap.add_argument("--representation", default="raw")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--run", default="f33")
    args = ap.parse_args()

    weekly = {**load_oulad(), **load_ku(), **load_ukzn(), **load_zambia(), **load_oviedo()}
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in weekly.items()}
    cohorts = tb.make_cohorts(canon, args.fraction)

    rows = [
        rec
        for cohort, seed in itertools.product(cohorts, range(args.seeds))
        if (rec := cohort_reference_points(cohort, args.representation, seed)) is not None
    ]
    frame = pd.DataFrame(rows)

    summary = pd.DataFrame(
        [
            {
                "reference": "floor (same model, reseeded)",
                "tau": frame["floor_tau"].mean(),
                "tau_sd": frame["floor_tau"].std(),
                "jaccard_top3": frame["floor_jaccard"].mean(),
            },
            {
                "reference": "ceiling (disjoint halves, same cohort)",
                "tau": frame["ceiling_tau"].mean(),
                "tau_sd": frame["ceiling_tau"].std(),
                "jaccard_top3": frame["ceiling_jaccard"].mean(),
            },
            {
                "reference": "ceiling, ladder-matched evaluation set",
                "tau": frame["ceiling_full_tau"].mean(),
                "tau_sd": frame["ceiling_full_tau"].std(),
                "jaccard_top3": frame["ceiling_full_jaccard"].mean(),
            },
            {
                "reference": "random rankings (analytic)",
                "tau": 0.0,
                "tau_sd": float("nan"),
                "jaccard_top3": random_jaccard_topk(len(tb.CANON), TOP_K),
            },
        ]
    )

    tb.write_outputs(OUT / args.run, per_cohort=frame, summary=summary)
    print(summary.round(3).to_string(index=False))
    print(f"\ncohorts x seeds usable: {len(frame)}")
    print("\nceiling by institution:")
    print(frame.groupby("institution")[["ceiling_tau", "ceiling_full_tau", "ceiling_full_jaccard"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
