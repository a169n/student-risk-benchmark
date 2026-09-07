"""exp_014 runner: build canonical cohort frames for three institutions and run the transfer ladder.

Usage (from services/ml):
    uv run python -m src.experiments.run_transfer_ladder --fraction 0.33

Inputs are the parquet caches written by the adapter scripts:
    datasets/cache/oulad_<MOD>_<PRES>.parquet     (OULAD weekly snapshots, min_week=1)
    datasets/cache/ku_leuven_1819.parquet          (KU Leuven weekly engagement snapshots, min_week=1)
    datasets/ukzn/cache/ukzn_<SUBJ>_<YEAR>.parquet (UKZN weekly frames from ukzn_adapter)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.experiments import transfer_benchmark as tb

REPO = Path(__file__).resolve().parents[4]
CACHE = REPO / "datasets" / "cache"
UKZN_CACHE = REPO / "datasets" / "ukzn" / "cache"
OUT = REPO / "data" / "artifacts" / "experiments" / "exp_014_transfer_ladder"

_WEEKLY = ["student_id", "week_number", "n_weeks", "passed", "current_clicks", "current_active_days", "current_content_clicks", "current_social_clicks"]


def load_oulad() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(CACHE.glob("oulad_*.parquet")):
        _, mod, pres = path.stem.split("_")
        s = pd.read_parquet(path)
        s = s.loc[s["passed_observed"].notna()]
        f = pd.DataFrame(
            {
                "student_id": s["id_student"].astype(str),
                "week_number": s["week_number"].astype(int),
                "n_weeks": s["duration_weeks"].astype(int),
                "passed": s["passed_observed"].astype(int),
                "current_clicks": s["current_week_clicks"].astype(float),
                "current_active_days": s["current_week_active_days"].astype(float),
                "current_content_clicks": s["current_week_content_clicks"].astype(float),
                "current_social_clicks": s["current_week_social_clicks"].astype(float),
            }
        )
        frames[f"oulad_{mod}_{pres}"] = f.assign(institution="OULAD", module=mod, cohort_id=f"oulad_{mod}_{pres}")
    return frames


def load_ku() -> dict[str, pd.DataFrame]:
    """One cohort per (course, academic year); the module key is the course, so
    the same course in another year forms a D1 pair."""
    frames = {}
    for path in sorted(CACHE.glob("ku_leuven_*.parquet")):
        year = path.stem.split("_")[-1]
        s = pd.read_parquet(path)
        for course, g in s.groupby("course_id"):
            code = "".join(ch for ch in str(course) if ch.isalnum())[:12]
            cid = f"ku_{code}_{year}"
            f = pd.DataFrame(
                {
                    "student_id": g["student_id"].astype(str),
                    "week_number": g["week_number"].astype(int),
                    "n_weeks": int(g["week_number"].max()),
                    "passed": g["passed"].astype(int),
                    "current_clicks": g["current_week_clicks"].astype(float),
                    "current_active_days": g["current_week_active_days"].astype(float),
                    "current_content_clicks": g["current_week_content_clicks"].astype(float),
                    "current_social_clicks": g["forum_posts"].astype(float),
                }
            )
            frames[cid] = f.assign(institution="KU Leuven", module=code, cohort_id=cid)
    return frames


def load_ukzn() -> dict[str, pd.DataFrame]:
    frames = {}
    for path in sorted(UKZN_CACHE.glob("ukzn_*.parquet")):
        _, subj, year = path.stem.split("_")
        f = pd.read_parquet(path)[_WEEKLY]
        frames[path.stem] = f.assign(institution="UKZN", module=subj, cohort_id=path.stem)
    return frames


def load_zambia() -> dict[str, pd.DataFrame]:
    """University of Zambia CS1: one cohort per calendar year of the same course."""
    frames = {}
    for path in sorted(CACHE.glob("zambia_*.parquet")):
        f = pd.read_parquet(path)
        frames[path.stem] = f[_WEEKLY].assign(
            institution="Zambia", module="ICT1110", cohort_id=path.stem
        )
    return frames


def load_oviedo() -> dict[str, pd.DataFrame]:
    """Universidad de Oviedo: one cohort per Moodle course, academic year 2014/15."""
    frames = {}
    for path in sorted(CACHE.glob("oviedo_*.parquet")):
        module = path.stem.split("_")[1]
        f = pd.read_parquet(path)
        frames[path.stem] = f[_WEEKLY].assign(
            institution="Oviedo", module=module, cohort_id=path.stem
        )
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.33)
    ap.add_argument("--min-students", type=int, default=50)
    ap.add_argument("--min-minority", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-fewshot", action="store_true")
    args = ap.parse_args()

    weekly = {
        **load_oulad(),
        **load_ku(),
        **load_ukzn(),
        **load_zambia(),
        **load_oviedo(),
    }
    canon = {cid: tb.canonical_from_weekly(f) for cid, f in weekly.items()}
    cohorts = tb.make_cohorts(canon, args.fraction, min_students=args.min_students, min_minority=args.min_minority)
    print(f"cohorts at fraction {args.fraction}: {len(cohorts)}", flush=True)

    out = OUT / f"f{int(round(args.fraction * 100)):02d}"
    summary = tb.cohort_summary(cohorts)
    print(summary.to_string(), flush=True)
    pairs = tb.run_ladder(cohorts, seed=args.seed)
    print("ladder done", len(pairs), flush=True)
    agg = tb.aggregate(pairs, seed=args.seed)
    # Write the ladder before the pooled / few-shot stages: those take as long
    # again, and a failure there must not cost the main result.
    tb.write_outputs(out, cohorts=summary, pairs=pairs, aggregate=agg)
    print("ladder artifacts written", flush=True)

    pooled = tb.run_pooled(cohorts, seed=args.seed)
    print("pooled done", len(pooled), flush=True)
    frames = {"pooled": pooled}
    if not args.skip_fewshot:
        frames["fewshot"] = tb.run_fewshot(cohorts)
        print("fewshot done", flush=True)
    tb.write_outputs(out, **frames)
    tb.dump_json(
        out / "run_metadata.json",
        {
            "experiment_id": "exp_014_transfer_ladder",
            "fraction": args.fraction,
            "min_students": args.min_students,
            "min_minority": args.min_minority,
            "seed": args.seed,
            "n_cohorts": len(cohorts),
            "institutions": sorted({c.institution for c in cohorts}),
            "features": list(tb.CANON),
            "representations": list(tb.REPRESENTATIONS),
            "models": list(tb.MODELS),
        },
    )
    print(agg.to_string(), flush=True)


if __name__ == "__main__":
    main()
