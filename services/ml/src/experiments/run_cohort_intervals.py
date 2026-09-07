"""exp_023: within-cohort AUC and its bootstrap interval, one row per cohort.

The paper says predictability belongs to the institution, and supports it with a
range over cohorts and a count of cohorts whose interval covers chance. Those two
numbers were computed ad hoc; the ladder only releases the mean per distance
class, so neither could be recomputed from a released artifact. This writes the
per-cohort table they come from.

The point estimate is the same procedure the ladder calls D0 --- 5-fold
cross-validated gradient boosting on the raw representation, one out-of-fold
probability per student --- so the numbers here and in pairs.csv agree by
construction rather than by coincidence. The interval resamples students within
the cohort and re-scores the same out-of-fold probabilities, which prices the
sampling of students but not the refitting; it is therefore a lower bound on the
uncertainty, and the paper says so.

Usage (from services/ml):
    uv run python -m src.experiments.run_cohort_intervals --fraction 0.33
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.experiments import run_transfer_ladder as ladder
from src.experiments import transfer_benchmark as tb
from src.experiments.models import build_classification_model

REPO = Path(__file__).resolve().parents[4]
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_023_cohort_intervals"

MODEL = "gradient_boosting"
REPRESENTATION = "raw"


def bootstrap_auc(
    y: np.ndarray, p: np.ndarray, *, n_boot: int, seed: int
) -> tuple[float, float]:
    """Percentile interval over students. Resamples with both classes present are
    the only ones AUC is defined on, so degenerate draws are discarded rather
    than scored as 0.5, which would pull the interval toward chance."""
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        draws.append(roc_auc_score(yb, p[idx]))
    if not draws:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.33)
    ap.add_argument("--min-students", type=int, default=50)
    ap.add_argument("--min-minority", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=400)
    args = ap.parse_args()

    weekly = {
        **ladder.load_oulad(),
        **ladder.load_ku(),
        **ladder.load_ukzn(),
        **ladder.load_zambia(),
        **ladder.load_oviedo(),
    }
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in weekly.items()}
    cohorts = tb.make_cohorts(
        canon, args.fraction, min_students=args.min_students, min_minority=args.min_minority
    )
    print(f"cohorts at fraction {args.fraction}: {len(cohorts)}", flush=True)

    rows = []
    for c in cohorts:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
        p = cross_val_predict(
            build_classification_model(MODEL, seed=args.seed),
            c.X(REPRESENTATION),
            c.y,
            cv=cv,
            method="predict_proba",
        )[:, 1]
        auc = float(roc_auc_score(c.y, p))
        lo, hi = bootstrap_auc(c.y, p, n_boot=args.n_boot, seed=args.seed)
        rows.append(
            {
                "cohort_id": c.cohort_id,
                "institution": c.institution,
                "module": c.module,
                "n_students": c.n,
                "pass_rate": float(c.y.mean()),
                "auc": auc,
                "auc_ci_low": lo,
                "auc_ci_high": hi,
                "covers_chance": bool(lo <= 0.5 <= hi),
                "below_chance": bool(hi < 0.5),
            }
        )
        print(f"{c.cohort_id:28s} {auc:.3f}  [{lo:.3f}, {hi:.3f}]", flush=True)

    d = pd.DataFrame(rows).sort_values("auc").reset_index(drop=True)
    out = OUT / f"f{int(round(args.fraction * 100)):02d}"
    tb.write_outputs(out, cohort_auc=d)
    tb.dump_json(
        out / "run_metadata.json",
        {
            "experiment_id": "exp_023_cohort_intervals",
            "fraction": args.fraction,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "model": MODEL,
            "representation": REPRESENTATION,
            "n_cohorts": len(d),
            "auc_min": float(d["auc"].min()),
            "auc_max": float(d["auc"].max()),
            "n_covering_chance": int(d["covers_chance"].sum()),
            "n_below_chance": int(d["below_chance"].sum()),
        },
    )
    print(
        f"\nrange {d['auc'].min():.3f}-{d['auc'].max():.3f}; "
        f"{int(d['covers_chance'].sum())} of {len(d)} cover 0.5; "
        f"{int(d['below_chance'].sum())} entirely below",
        flush=True,
    )
    print("written to", out, flush=True)


if __name__ == "__main__":
    main()
