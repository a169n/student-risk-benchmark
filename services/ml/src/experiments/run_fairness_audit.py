"""exp_020 — who is missed when an early-warning model is transferred?

The benchmark's shared feature schema carries no demographics, which is a
modelling choice, not a data limitation: OULAD publishes gender, region, index
of multiple deprivation, age band and disability, and UKZN publishes gender,
race, school quintile and means-tested-aid flags. A click-volume risk score in a
low-income context is partly a device-and-connectivity score, so a transferred
model can fail unevenly even when its aggregate numbers look stable.

This audit asks one question with a fixed answer format: at a realistic 20 %
flag budget, what share of each group's failing students does the model catch,
locally and after transfer, and how wide is the gap between the best-served and
worst-served group?

Sensitive attributes are used ONLY to evaluate. They never enter a feature
vector, and the models audited here are exactly the models the ladder reports.

Usage (from services/ml):
    uv run python -m src.experiments.run_fairness_audit --fraction 0.33
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

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
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_020_fairness"
MODEL = "gradient_boosting"
FLAG_RATE = 0.20
MIN_GROUP = 25  # below this a per-group recall is too noisy to report

OULAD_ATTRS = ("gender", "imd_band", "disability", "age_band", "highest_education")
UKZN_ATTRS = ("GENDER", "RACE", "QUINTILE", "NSFASBURSARYYN")


def load_oulad_attributes() -> pd.DataFrame:
    info = pd.read_csv(REPO / "datasets" / "oulad" / "studentInfo.csv")
    frame = info[["id_student", *OULAD_ATTRS]].copy()
    frame["student_id"] = frame["id_student"].astype(str)
    frame["institution"] = "OULAD"
    return frame.drop(columns=["id_student"]).drop_duplicates("student_id")


def load_ukzn_attributes() -> pd.DataFrame:
    path = (
        REPO / "datasets" / "ukzn" / "raw" / "Dataset V1 - Raw Data"
        / "ISTN TL Modules 2014-2021 - ANONYMIZED.xlsx"
    )
    df = pd.read_excel(path, sheet_name="DATA", usecols=["ANONSTUDNO", *UKZN_ATTRS])
    df["student_id"] = df["ANONSTUDNO"].astype(str).str.strip()
    df["institution"] = "UKZN"
    return df.drop(columns=["ANONSTUDNO"]).drop_duplicates("student_id")


def recall_at_budget(fail: np.ndarray, risk: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    """Per-group recall of failing students inside a cohort-wide 20 % flag budget."""
    n_flag = max(1, int(round(FLAG_RATE * len(risk))))
    flagged = np.zeros(len(risk), dtype=int)
    flagged[np.argsort(-risk)[:n_flag]] = 1
    frame = pd.DataFrame({"group": groups.to_numpy(), "fail": fail, "flagged": flagged})
    out = (
        frame[frame["fail"] == 1]
        .groupby("group")
        .agg(n_failing=("fail", "size"), recall=("flagged", "mean"))
        .reset_index()
    )
    return out[out["n_failing"] >= MIN_GROUP]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--representation", default="raw")
    ap.add_argument("--run", default="f33")
    args = ap.parse_args()

    weekly = {**load_oulad(), **load_ku(), **load_ukzn(), **load_zambia(), **load_oviedo()}
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in weekly.items()}
    cohorts = tb.make_cohorts(canon, args.fraction)
    by_id = {c.cohort_id: c for c in cohorts}

    attrs = pd.concat([load_oulad_attributes(), load_ukzn_attributes()], ignore_index=True)
    lookup = attrs.set_index("student_id")

    # Student ids in cohort order, so a cohort's rows line up with its X and y.
    students = {
        cid: tb.cutoff_rows(canon[cid], args.fraction)["student_id"].astype(str).reset_index(drop=True)
        for cid in by_id
    }

    audited = [c for c in cohorts if c.institution in ("OULAD", "UKZN")]
    print(f"cohorts with published attributes: {len(audited)} of {len(cohorts)}", flush=True)

    models = {
        c.cohort_id: build_classification_model(MODEL, seed=args.seed).fit(
            c.X(args.representation), c.y
        )
        for c in cohorts
    }

    rows: list[dict] = []
    for target in audited:
        ids = students[target.cohort_id]
        present = ids[ids.isin(lookup.index)]
        if len(present) < MIN_GROUP * 2:
            continue
        mask = ids.isin(lookup.index).to_numpy()
        fail = (1 - target.y)[mask]
        attr_block = lookup.loc[present]

        sources = [("local", target)] + [
            ("transferred", s) for s in cohorts if s.institution != target.institution
        ]
        for kind, source in sources:
            risk = 1.0 - models[source.cohort_id].predict_proba(
                target.X(args.representation)
            )[:, 1]
            risk = risk[mask]
            for attribute in OULAD_ATTRS + UKZN_ATTRS:
                if attribute not in attr_block.columns:
                    continue
                values = attr_block[attribute].astype(str).replace({"nan": None}).dropna()
                if values.nunique() < 2:
                    continue
                sub = recall_at_budget(fail, risk, values.reindex(present).astype(str))
                if len(sub) < 2:
                    continue
                rows.append(
                    {
                        "target": target.cohort_id,
                        "institution": target.institution,
                        "source_kind": kind,
                        "source": source.cohort_id,
                        "attribute": attribute,
                        "groups": len(sub),
                        "recall_best": sub["recall"].max(),
                        "recall_worst": sub["recall"].min(),
                        "recall_gap": sub["recall"].max() - sub["recall"].min(),
                        "worst_group": sub.loc[sub["recall"].idxmin(), "group"],
                    }
                )

    frame = pd.DataFrame(rows)
    if frame.empty:
        print("no auditable cohort/attribute combination met the size floor")
        return

    summary = (
        frame.groupby(["institution", "attribute", "source_kind"])
        .agg(
            observations=("recall_gap", "size"),
            mean_gap=("recall_gap", "mean"),
            max_gap=("recall_gap", "max"),
            mean_worst=("recall_worst", "mean"),
        )
        .reset_index()
    )
    tb.write_outputs(OUT / args.run, per_source=frame, summary=summary)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
