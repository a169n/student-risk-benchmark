"""Rebuild the parquet caches the transfer-ladder runners read, from the raw downloads.

The runners in ``src/experiments/`` (exp_014 onwards) read weekly frames from
``datasets/cache/`` and ``datasets/ukzn/cache/``, which are not in git. UKZN,
Zambia and Oviedo already had a cache writer in their adapters; OULAD and
KU Leuven did not, so their column contract is taken from
``run_transfer_ladder.load_oulad`` / ``load_ku``. Existing files are skipped.

Rebuilt this way on 24 Sep 2026, the 63 cohorts at the 1/3 cutoff match the
frozen ``exp_014_transfer_ladder/f33/cohorts.csv`` exactly (45,158 rows).

Raw data layout: see ``datasets/PROVENANCE.md``. KU Leuven also needs the three
``<year>_log_activity.csv`` files from ``dataset_full.zip``.

Usage (from services/ml; about 15 minutes, OULAD dominates):
    uv run python -m src.benchmarks.build_caches [oulad] [ku] [ukzn] [zambia] [oviedo]
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.benchmarks import ku_leuven_adapter as ku
from src.benchmarks import oulad_adapter as oa
from src.benchmarks import oviedo_adapter, ukzn_adapter, zambia_adapter

REPO = Path(__file__).resolve().parents[4]
DS = REPO / "datasets"
CACHE = DS / "cache"


def _oulad_one(presentation: tuple[str, str]) -> str:
    mod, pres = presentation
    target = CACHE / f"oulad_{mod}_{pres}.parquet"
    if not target.exists():
        try:
            result = oa.build_weekly_snapshots(
                oa.OuladRawPaths.from_directory(DS / "oulad"),
                course_filter=oa.OuladCourseFilter(mod, pres),
                min_week=1,
            )
        except ValueError as exc:  # GGG: no final weighted score, so no labelled rows
            return f"skipped {mod}_{pres}: {exc}"
        result.snapshots.to_parquet(target, index=False)
    return f"cached {target.name}"


def oulad() -> None:
    courses = pd.read_csv(DS / "oulad" / "courses.csv")
    presentations = list(zip(courses["code_module"], courses["code_presentation"], strict=True))
    with ProcessPoolExecutor(6) as pool:
        for line in pool.map(_oulad_one, presentations):
            print(line, flush=True)


def kuleuven() -> None:
    for year in ("1819", "1920", "2021"):
        target = CACHE / f"ku_leuven_{year}.parquet"
        if target.exists():
            continue
        result = ku.build_weekly_engagement_snapshots(
            DS / "ku_leuven" / "dataset",
            DS / "ku_leuven" / "course_info.json",
            year=year,
            min_week=1,
        )
        result.snapshots.to_parquet(target, index=False)
        print(f"cached {target.name}", result.row_counts, flush=True)


def ukzn() -> None:
    frames = ukzn_adapter.build_all_cohorts(DS / "ukzn" / "raw", cache_dir=DS / "ukzn" / "cache")
    print("ukzn", len(frames), "cohorts", flush=True)


def zambia() -> None:
    frames = zambia_adapter.build_all_cohorts(DS / "zambia", cache_dir=CACHE)
    print("zambia", len(frames), "cohorts", flush=True)


def oviedo() -> None:
    tables = DS / "_candidates" / "oviedo" / "tables"
    frames = oviedo_adapter.build_all_cohorts(tables, cache_dir=CACHE)
    print("oviedo", len(frames), "cohorts", flush=True)


STEPS = {"oulad": oulad, "ku": kuleuven, "ukzn": ukzn, "zambia": zambia, "oviedo": oviedo}


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or STEPS:
        STEPS[name]()
