"""exp_016 — does measured distribution shift, rather than the institution label,
explain the transfer loss?

Motivation: the ladder has only five institutions, so "D3 is worse" rests on few
institution pairs. This analysis replaces the categorical distance with
CONTINUOUS, measurable shift between source and target cohorts, turning five
institutions into thousands of ordered pairs with real predictors:

    d_prevalence   |pass rate difference|
    d_scale        |log10 ratio of median cumulative clicks|  (platform click scale)
    d_length       |course length difference| in weeks
    d_shape        mean Kolmogorov-Smirnov distance over the seven features,
                   computed on RAW values (the quantity a relative
                   representation is designed to remove)

Each is regressed against the per-pair AUC loss relative to the target's own
within-cohort score. If loss tracks shift, the claim generalises past the five
institutions we happen to have.

NOTE on an earlier misreading: a first version of this analysis was run over
three institutions only, and its output was used to claim that the percentile
representation works by removing sensitivity to click scale. It does not.
``corr_d_scale`` RISES slightly under percentiles; what falls is ``d_shape``.
Read the signs off the regression table rather than assuming the mechanism.

Usage (from services/ml):
    uv run python -m src.experiments.run_shift_analysis --run f33
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression

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
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_016_shift_analysis"

SHIFT_COLUMNS = ["d_prevalence", "d_scale", "d_length", "d_shape"]


def _all_weekly() -> dict:
    """Every institution. Loading a subset here silently drops them from the mechanism analysis."""
    return {**load_oulad(), **load_ku(), **load_ukzn(), **load_zambia(), **load_oviedo()}


def build_cohorts(fraction: float) -> list[tb.Cohort]:
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in _all_weekly().items()}
    return tb.make_cohorts(canon, fraction)


def pair_shift(cohorts: list[tb.Cohort], n_weeks: dict[str, int]) -> pd.DataFrame:
    """Continuous shift descriptors for every ordered (source, target) pair."""
    rows = []
    for s, t in itertools.product(cohorts, cohorts):
        if s.cohort_id == t.cohort_id:
            continue
        s_scale = max(float(np.median(s.X_raw["cum_clicks"])), 1.0)
        t_scale = max(float(np.median(t.X_raw["cum_clicks"])), 1.0)
        ks = [
            ks_2samp(s.X_raw[c].to_numpy(), t.X_raw[c].to_numpy()).statistic
            for c in tb.CANON
        ]
        rows.append(
            {
                "source": s.cohort_id,
                "target": t.cohort_id,
                "d_prevalence": abs(float(s.y.mean() - t.y.mean())),
                "d_scale": abs(float(np.log10(s_scale / t_scale))),
                "d_length": abs(n_weeks[s.cohort_id] - n_weeks[t.cohort_id]),
                "d_shape": float(np.mean(ks)),
            }
        )
    return pd.DataFrame(rows)


def attach_loss(pairs: pd.DataFrame, shift: pd.DataFrame) -> pd.DataFrame:
    """AUC loss = target's own within-cohort AUC minus the transferred AUC."""
    local = (
        pairs[pairs["distance"] == "D0_within_cohort"]
        .set_index(["target", "representation", "model"])["auc"]
        .rename("auc_local")
    )
    transferred = pairs[pairs["distance"] != "D0_within_cohort"].merge(
        local, left_on=["target", "representation", "model"], right_index=True, how="inner"
    )
    transferred["auc_loss"] = transferred["auc_local"] - transferred["auc"]
    return transferred.merge(shift, on=["source", "target"], how="inner")


def regress(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (model, representation): standardised coefficients of loss on shift, plus R^2."""
    rows = []
    for (model, rep), g in frame.groupby(["model", "representation"]):
        X = g[SHIFT_COLUMNS].to_numpy(dtype=float)
        y = g["auc_loss"].to_numpy(dtype=float)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        Xz = (X - X.mean(axis=0)) / sd
        fit = LinearRegression().fit(Xz, y)
        rec = {"model": model, "representation": rep, "n_pairs": len(g), "r2": float(fit.score(Xz, y))}
        rec.update({f"beta_{c}": float(b) for c, b in zip(SHIFT_COLUMNS, fit.coef_)})
        rec.update(
            {f"corr_{c}": float(np.corrcoef(g[c], g["auc_loss"])[0, 1]) for c in SHIFT_COLUMNS}
        )
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["model", "representation"])


def binned_loss(frame: pd.DataFrame, column: str, model: str, bins: int = 5) -> pd.DataFrame:
    """Mean loss by quintile of one shift measure, per representation (for a figure)."""
    g = frame[frame["model"] == model].copy()
    g["bin"] = pd.qcut(g[column], bins, labels=False, duplicates="drop")
    out = g.groupby(["representation", "bin"]).agg(
        shift_mid=(column, "mean"), auc_loss=("auc_loss", "mean"), n=("auc_loss", "size")
    )
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="f33")
    ap.add_argument("--fraction", type=float, default=0.33)
    args = ap.parse_args()

    pairs = pd.read_csv(EXP14 / args.run / "pairs.csv")
    cohorts = build_cohorts(args.fraction)
    weekly = _all_weekly()
    n_weeks = {cid: int(f["n_weeks"].iloc[0]) for cid, f in weekly.items()}

    shift = pair_shift(cohorts, n_weeks)
    frame = attach_loss(pairs, shift)
    coefs = regress(frame)
    binned = pd.concat(
        [binned_loss(frame, c, "gradient_boosting").assign(shift_measure=c) for c in SHIFT_COLUMNS],
        ignore_index=True,
    )

    out = OUT / args.run
    tb.write_outputs(out, pair_shift=shift, loss_vs_shift=frame, regression=coefs, binned=binned)
    print(coefs.round(3).to_string(index=False))
    print("\nMean shift by distance class (gradient boosting, raw):")
    g = frame[(frame.model == "gradient_boosting") & (frame.representation == "raw")]
    print(g.groupby("distance")[SHIFT_COLUMNS + ["auc_loss"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
