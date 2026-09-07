"""GdP MOOC (edX, "Gestion de Projet", MOCAH / LIP6 Sorbonne) -> canonical weekly frame.

Source: Fairness of MOOC Completion Predictions, AIED 2024
(doi 10.1007/978-3-031-64302-6_27); feature tables released at
https://github.com/Mocahteam/GdPMOOC_Fairness (BSD-3-Clause).

Shape of the release: per-student event-type counts for FOUR cumulative course
phases (``week`` = S1..S4) plus a binary ``success`` label, joined on
``username_anon``. 1,007 students, 383 pass / 624 fail.

Schema coverage — IMPORTANT
---------------------------
The release has no daily granularity, so ``current_active_days`` cannot be
derived and this institution supports only the SIX-feature reduced schema
(everything in ``transfer_benchmark.CANON`` except ``cum_active_days``).
``current_active_days`` is emitted as NaN so that any accidental use of the full
schema fails loudly rather than silently scoring zeros.

Time granularity is four phases rather than calendar weeks; the benchmark's
cutoff is relative (a fraction of course length), so phases slot in directly,
but the coarseness is a stated limitation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_FILE = "df_features2.csv"  # per-phase deltas (not the cumulative twin)
LABELS_FILE = "df_labels.csv"

# edX event-type counters grouped onto the canonical concepts.
_SOCIAL_PREFIX = "edx.forum."
_CONTENT_SUFFIXES = ("_video_count", "video_hide_cc_menu_count", "video_show_cc_menu_count")
_CONTENT_EXTRA = ("show_transcript_count", "hide_transcript_count", "page_close_count",
                  "seq_goto_count", "seq_next_count", "seq_prev_count")


def _classify(columns: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Split counter columns into (all, content, social)."""
    counters = [c for c in columns if c.endswith("_count")]
    social = [c for c in counters if c.startswith(_SOCIAL_PREFIX)]
    content = [
        c
        for c in counters
        if c not in social and (c.endswith(_CONTENT_SUFFIXES) or c in _CONTENT_EXTRA)
    ]
    return counters, content, social


def build_cohort_frame(repo_dir: Path | str, *, cohort_id: str = "gdp_mooc_2015") -> pd.DataFrame:
    """Return the canonical weekly frame for the single GdP cohort."""
    root = Path(repo_dir)
    features = pd.read_csv(root / "features" / FEATURES_FILE)
    labels = pd.read_csv(root / "features" / LABELS_FILE)

    counters, content, social = _classify(list(features.columns))
    if not counters or not social or not content:
        raise ValueError("GdP feature table does not look like the expected edX counter export.")

    phase = features["week"].astype(str).str.extract(r"(\d+)")[0].astype(int)
    n_weeks = int(phase.max())

    frame = pd.DataFrame(
        {
            "student_id": features["username_anon"].astype(str),
            "week_number": phase,
            "n_weeks": n_weeks,
            "current_clicks": features[counters].sum(axis=1).astype(float),
            "current_active_days": np.nan,  # not derivable: no daily granularity
            "current_content_clicks": features[content].sum(axis=1).astype(float),
            "current_social_clicks": features[social].sum(axis=1).astype(float),
        }
    )
    frame = frame.merge(
        labels.assign(student_id=lambda d: d["username_anon"].astype(str))[["student_id", "success"]],
        on="student_id",
        how="inner",
    ).rename(columns={"success": "passed"})
    frame["passed"] = frame["passed"].astype(int)

    return frame.assign(
        institution="GdP MOOC", module="GestionDeProjet", cohort_id=cohort_id
    ).sort_values(["student_id", "week_number"]).reset_index(drop=True)
