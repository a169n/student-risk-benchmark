"""Stream-extract the tables we need from the Universidad de Oviedo Moodle dump.

Dataset: Riestra-Gonzalez, Paule-Ruiz & Ortin (2021), "Massive LMS log data
analysis for the early prediction of course-agnostic student performance",
Computers & Education 163:104108. Anonymised full Moodle 2.x database for
academic year 2014/15, released as a PostgreSQL plain dump (781 MB zipped,
~7 GB expanded) alongside the paper's GitHub repository (Apache-2.0).

Why a custom extractor: the dump is a plain-SQL ``pg_dump`` containing 123
tables, and only four matter here. Rather than restoring ~7 GB into a database,
this streams the compressed file once, watches for ``COPY <table> (...) FROM
stdin;`` blocks, and writes only the requested columns of the requested tables
to CSV. Nothing is written to disk uncompressed except those extracts.

IMPORTANT — the published feature CSVs in that repository are NOT used. They
include grade-derived predictors (``ACCOMPLISH_MANDATORY_GRADE`` and similar)
for a grade-derived label, which is the circularity failure mode this project
already documented. Everything here is rebuilt from raw ``mdl_log`` clicks.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from pathlib import Path

# The dump stores the activity log partitioned into twelve Spanish month tables
# that share the ``mdl_log`` schema; ``mdl_log`` itself may be empty.
LOG_COLUMNS: tuple[str, ...] = ("time", "userid", "course", "module", "action")
MONTH_TABLES: tuple[str, ...] = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)

# table -> columns we keep (everything else in the row is discarded)
WANTED: dict[str, tuple[str, ...]] = {
    **{month: LOG_COLUMNS for month in MONTH_TABLES},
    "mdl_log": LOG_COLUMNS,
    "mdl_course": ("id", "shortname", "fullname", "startdate", "category"),
    "mdl_grade_items": ("id", "courseid", "itemtype", "itemname", "grademax"),
    "mdl_grade_grades": ("itemid", "userid", "finalgrade", "rawgrade"),
    "mdl_user_enrolments": ("id", "enrolid", "userid", "status"),
    "mdl_enrol": ("id", "courseid", "status"),
}

_COPY = re.compile(r"^COPY\s+(?:public\.)?\"?(?P<table>\w+)\"?\s*\((?P<cols>[^)]*)\)\s+FROM\s+stdin;", re.IGNORECASE)


def _open_stream(path: Path) -> io.TextIOWrapper:
    """Text stream over the dump, transparently unzipping a single-member archive."""
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        members = [n for n in archive.namelist() if not n.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"expected one member in {path.name}, found {members}")
        raw = archive.open(members[0])
    else:
        raw = path.open("rb")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")


def extract(dump_path: Path | str, out_dir: Path | str, *, progress_every: int = 5_000_000) -> dict[str, int]:
    """Write ``<out_dir>/<table>.csv`` for every table in WANTED. Returns row counts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    lines_seen = 0

    with _open_stream(Path(dump_path)) as stream:
        writer = None
        handle = None
        keep_idx: list[int] = []
        table = ""
        for line in stream:
            lines_seen += 1
            if progress_every and lines_seen % progress_every == 0:
                print(f"  ... {lines_seen:,} lines, tables done: {sorted(counts)}", flush=True)

            if writer is None:
                match = _COPY.match(line)
                if not match:
                    continue
                table = match.group("table")
                if table not in WANTED or table in counts:
                    continue
                cols = [c.strip().strip('"') for c in match.group("cols").split(",")]
                wanted = WANTED[table]
                missing = [c for c in wanted if c not in cols]
                if missing:
                    print(f"  !! {table}: columns absent from dump, skipping {missing}", flush=True)
                keep = [c for c in wanted if c in cols]
                keep_idx = [cols.index(c) for c in keep]
                handle = (out_dir / f"{table}.csv").open("w", newline="", encoding="utf-8")
                writer = csv.writer(handle)
                writer.writerow(keep)
                counts[table] = 0
                print(f"  -> {table}: {len(keep)} of {len(cols)} columns", flush=True)
                continue

            if line.startswith("\\."):
                handle.close()
                print(f"  == {table}: {counts[table]:,} rows", flush=True)
                writer = None
                handle = None
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) <= max(keep_idx):
                continue
            writer.writerow([fields[i] for i in keep_idx])
            counts[table] += 1

        if handle is not None:
            handle.close()

    missing_tables = [t for t in WANTED if t not in counts]
    if missing_tables:
        print(f"WARNING: never saw COPY blocks for {missing_tables}", flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="../../datasets/_candidates/oviedo/moodle_pg.sql.zip")
    ap.add_argument("--out", default="../../datasets/_candidates/oviedo/tables")
    args = ap.parse_args()
    counts = extract(args.dump, args.out)
    print("\nrow counts:")
    for table, n in sorted(counts.items()):
        print(f"  {table:24s} {n:>12,}")


if __name__ == "__main__":
    main()
