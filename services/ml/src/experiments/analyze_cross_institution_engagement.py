"""Cross-institution engagement-only analysis: OULAD (DDD, BBB) vs KU Leuven.

Aggregates the three matched engagement-only benchmark runs (exp_011 KU Leuven,
exp_012 OULAD DDD 2013J, exp_012 OULAD BBB 2013J) into two views, entirely from
the already-computed artifacts (NO model re-training, NO XAI re-computation):

  Part A -- B-vs-A engagement delta table.  For each institution x split, the
    change in the FIXED-MODEL (gradient_boosting) classification F1 / ROC-AUC
    when moving from the simple-engagement baseline feature set
    (``A_simple_engagement`` / ``A_simple_engagement_oulad``) to the richer
    engagement candidate set (``B_engagement`` / ``B_engagement_oulad``).

  Part B -- importance-rank-stability across institutions.  Each institution's
    permutation-importance ranking (from ``engagement_importance.csv``) is mapped
    onto a shared set of canonical engagement concepts (dropping concepts with no
    cross-institution analog), then compared pairwise via Kendall tau and
    top-k Jaccard (reusing ``src.experiments.stability``).

Methodology mirrors ``analyze_cross_cohort_stability`` but adds a concept-
alignment step because OULAD and KU Leuven use different raw feature names and
KU Leuven carries session-level features with no OULAD analog.

Fixed-model metrics come from the results-JSON ``rows`` list (one record per
``(feature_set, model, split_strategy)``, with ``metric_f1`` / ``metric_roc_auc``).
Importance rankings come from the sibling ``engagement_importance.csv``
(columns ``split_strategy, feature, ..., rank``; rows already best-first by rank).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from itertools import combinations
from typing import Any

from src.experiments.stability import jaccard_topk, kendall_tau

# ---------------------------------------------------------------------------
# Concept-alignment map (canonical concept -> per-institution raw feature name)
# ---------------------------------------------------------------------------

CONCEPT_ALIGNMENT: dict[str, dict[str, str]] = {
    "total_clicks": {
        "oulad": "cumulative_clicks_to_date",
        "ku_leuven": "cumulative_clicks_to_date",
    },
    "current_clicks": {
        "oulad": "current_week_clicks",
        "ku_leuven": "current_week_clicks",
    },
    "content_clicks": {
        "oulad": "cumulative_content_clicks_to_date",
        "ku_leuven": "cumulative_content_clicks_to_date",
    },
    "forum_engagement": {
        "oulad": "cumulative_social_clicks_to_date",
        "ku_leuven": "cumulative_forum_posts_to_date",
    },
    "content_ratio": {
        "oulad": "content_click_ratio_to_date",
        "ku_leuven": "content_click_ratio_to_date",
    },
    "active_days": {
        "oulad": "cumulative_active_days_to_date",
        "ku_leuven": "cumulative_active_days_to_date",
    },
    "week": {
        "oulad": "week_number",
        "ku_leuven": "week_number",
    },
}

# KU-only concepts with no OULAD analog -> dropped from Part B.  Listed here only
# for documentation / the Caveats section; they have no CONCEPT_ALIGNMENT entry.
KU_ONLY_FEATURES: tuple[str, ...] = (
    "cumulative_sessions_to_date",
    "avg_session_clicks_to_date",
    "days_since_course_start",
)

# Reverse lookup per institution: raw feature name -> canonical concept.
_RAW_TO_CONCEPT: dict[str, dict[str, str]] = {}
for _concept, _by_inst in CONCEPT_ALIGNMENT.items():
    for _inst, _raw in _by_inst.items():
        _RAW_TO_CONCEPT.setdefault(_inst, {})[_raw] = _concept

# The model held fixed for the Part A delta comparison.
FIXED_MODEL = "gradient_boosting"
STABLE_TAU_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_ARTIFACTS = _REPO_ROOT / "data" / "artifacts" / "experiments"

_DEFAULT_OULAD_DDD = (
    _ARTIFACTS
    / "exp_012_oulad_engagement_ddd2013j"
    / "exp_012_oulad_engagement_ddd2013j_results.json"
)
_DEFAULT_OULAD_BBB = (
    _ARTIFACTS
    / "exp_012_oulad_engagement_bbb2013j"
    / "exp_012_oulad_engagement_bbb2013j_results.json"
)
_DEFAULT_KU_LEUVEN = (
    _ARTIFACTS
    / "exp_011_kuleuven_engagement"
    / "exp_011_kuleuven_engagement_results.json"
)
_DEFAULT_OUTPUT = (
    _ARTIFACTS / "exp_012_cross_institution_engagement" / "cross_institution_engagement"
)

# Per-institution feature-set keys (baseline / candidate) for Part A.
_BASELINE_KEY = {"oulad": "A_simple_engagement_oulad", "ku_leuven": "A_simple_engagement"}
_CANDIDATE_KEY = {"oulad": "B_engagement_oulad", "ku_leuven": "B_engagement"}


# ---------------------------------------------------------------------------
# Concept alignment
# ---------------------------------------------------------------------------

def align_ranking_to_concepts(ranking: list[str], institution: str) -> list[str]:
    """Map a raw best-first feature ranking onto canonical concepts.

    - Raw feature names with no canonical concept for ``institution`` (KU-only /
      unmapped) are DROPPED.
    - Order is preserved (best-first).
    - Duplicate concepts are deduped, keeping the first occurrence.
    """
    lookup = _RAW_TO_CONCEPT.get(institution, {})
    aligned: list[str] = []
    seen: set[str] = set()
    for raw in ranking:
        concept = lookup.get(raw)
        if concept is None:
            continue
        if concept in seen:
            continue
        seen.add(concept)
        aligned.append(concept)
    return aligned


# ---------------------------------------------------------------------------
# Part A: B-vs-A fixed-model engagement delta table
# ---------------------------------------------------------------------------

def build_engagement_delta_table(
    per_institution: dict[str, dict[str, dict[str, dict[str, float]]]],
    *,
    baseline_key_by_inst: dict[str, str],
    candidate_key_by_inst: dict[str, str],
) -> list[dict[str, Any]]:
    """Compute candidate-minus-baseline fixed-model deltas per institution x split.

    Parameters
    ----------
    per_institution:
        ``{institution_label: {split: {feature_set_key: {"f1": .., "roc_auc": ..}}}}``
        holding the FIXED-MODEL classification metrics.
    baseline_key_by_inst, candidate_key_by_inst:
        ``{institution_label: feature_set_key}`` naming the A (baseline) and
        B (candidate) feature sets for each institution.

    Returns
    -------
    list of row dicts: ``institution, split, a_f1, b_f1, delta_f1,
    a_roc_auc, b_roc_auc, delta_roc_auc``.
    """
    table: list[dict[str, Any]] = []
    for institution in sorted(per_institution):
        base_key = baseline_key_by_inst[institution]
        cand_key = candidate_key_by_inst[institution]
        by_split = per_institution[institution]
        for split in sorted(by_split):
            cell = by_split[split]
            base = cell.get(base_key)
            cand = cell.get(cand_key)
            if base is None or cand is None:
                continue
            a_f1 = base.get("f1")
            b_f1 = cand.get("f1")
            a_roc = base.get("roc_auc")
            b_roc = cand.get("roc_auc")
            delta_f1 = (b_f1 - a_f1) if (a_f1 is not None and b_f1 is not None) else None
            delta_roc = (
                (b_roc - a_roc) if (a_roc is not None and b_roc is not None) else None
            )
            table.append(
                {
                    "institution": institution,
                    "split": split,
                    "baseline_feature_set": base_key,
                    "candidate_feature_set": cand_key,
                    "a_f1": a_f1,
                    "b_f1": b_f1,
                    "delta_f1": delta_f1,
                    "a_roc_auc": a_roc,
                    "b_roc_auc": b_roc,
                    "delta_roc_auc": delta_roc,
                }
            )
    return table


# ---------------------------------------------------------------------------
# Part B: cross-institution importance-rank stability
# ---------------------------------------------------------------------------

def build_cross_institution_importance(
    rankings_by_split: dict[str, dict[str, list[str]]],
    *,
    institution_by_label: dict[str, str],
    top_k: int = 5,
    stable_tau_threshold: float = STABLE_TAU_THRESHOLD,
) -> dict[str, Any]:
    """Pairwise rank-stability of importance rankings across institutions.

    Parameters
    ----------
    rankings_by_split:
        ``{split: {institution_label: raw_best_first_ranking}}``.
    institution_by_label:
        ``{institution_label: "oulad" | "ku_leuven"}`` selecting the alignment
        mapping for each label's raw feature names.
    top_k:
        Top-k cutoff for the Jaccard overlap (computed over aligned concepts).
    stable_tau_threshold:
        Minimum tau for a pair to count as "stable".

    Returns
    -------
    dict with ``per_split`` (per split: aligned rankings, pairwise tau/Jaccard,
    mean tau), ``top_k``, ``mean_kendall_tau``, ``all_stable`` and ``verdict``.

    Verdict is ``drivers_transfer_across_institutions`` iff every non-None tau is
    >= ``stable_tau_threshold`` (and at least one tau was computable); otherwise
    ``drivers_partly_institution_specific``.  Pairs with <2 shared aligned
    concepts yield tau=None (per stability.kendall_tau); Jaccard is still
    reported for those pairs.
    """
    per_split: dict[str, Any] = {}
    all_valid_taus: list[float] = []

    for split in sorted(rankings_by_split):
        by_label = rankings_by_split[split]
        aligned: dict[str, list[str]] = {
            label: align_ranking_to_concepts(
                raw, institution_by_label[label]
            )
            for label, raw in by_label.items()
        }
        labels = sorted(aligned)
        pairs: list[dict[str, Any]] = []
        split_taus: list[float] = []
        for label_a, label_b in combinations(labels, 2):
            rank_a = aligned[label_a]
            rank_b = aligned[label_b]
            tau = kendall_tau(rank_a, rank_b)
            jac = jaccard_topk(rank_a, rank_b, k=top_k)
            shared = len(set(rank_a) & set(rank_b))
            if tau is not None:
                split_taus.append(tau)
                all_valid_taus.append(tau)
            pairs.append(
                {
                    "pair": [label_a, label_b],
                    "kendall_tau": tau,
                    "jaccard_topk": jac,
                    "n_shared_concepts": shared,
                    "top_k": top_k,
                }
            )
        per_split[split] = {
            "aligned_rankings": aligned,
            "pairs": pairs,
            "mean_kendall_tau": (
                sum(split_taus) / len(split_taus) if split_taus else None
            ),
        }

    mean_tau = (
        sum(all_valid_taus) / len(all_valid_taus) if all_valid_taus else None
    )
    all_stable = bool(
        all_valid_taus
        and all(t >= stable_tau_threshold for t in all_valid_taus)
    )
    verdict = (
        "drivers_transfer_across_institutions"
        if all_stable
        else "drivers_partly_institution_specific"
    )
    return {
        "per_split": per_split,
        "top_k": top_k,
        "stable_tau_threshold": stable_tau_threshold,
        "mean_kendall_tau": mean_tau,
        "all_stable": all_stable,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Artifact loaders
# ---------------------------------------------------------------------------

def load_fixed_model_metrics(
    results_path: pathlib.Path,
    *,
    fixed_model: str = FIXED_MODEL,
) -> dict[str, dict[str, dict[str, float]]]:
    """Extract fixed-model classification metrics from a results JSON.

    Returns ``{split: {feature_set: {"f1": .., "roc_auc": ..}}}`` for rows whose
    ``model == fixed_model`` (and ``task`` is classification when present).
    """
    with results_path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if "rows" not in payload:
        raise KeyError(
            f"Expected key 'rows' in {results_path}; got keys: {list(payload.keys())}"
        )
    rows = payload["rows"]
    out: dict[str, dict[str, dict[str, float]]] = {}
    for rec in rows:
        if rec.get("model") != fixed_model:
            continue
        task = rec.get("task")
        if task is not None and task != "classification":
            continue
        split = rec["split_strategy"]
        fs = rec["feature_set"]
        out.setdefault(split, {})[fs] = {
            "f1": rec.get("metric_f1"),
            "roc_auc": rec.get("metric_roc_auc"),
        }
    return out


def load_importance_rankings(
    importance_csv_path: pathlib.Path,
) -> dict[str, list[str]]:
    """Read ``engagement_importance.csv`` into ``{split: [feature, ...best-first]}``.

    The CSV is sorted within each ``split_strategy`` by ascending ``rank``
    (rank 1 = most important); rows are read and ordered by that rank.
    """
    by_split: dict[str, list[tuple[int, str]]] = {}
    with importance_csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            split = row["split_strategy"]
            feature = row["feature"]
            rank = int(row["rank"])
            by_split.setdefault(split, []).append((rank, feature))
    if not by_split:
        raise ValueError(
            f"No importance rows found in {importance_csv_path}; "
            "check the file is non-empty and has 'split_strategy', 'feature', 'rank' columns."
        )
    return {
        split: [feat for _rank, feat in sorted(rows, key=lambda t: t[0])]
        for split, rows in by_split.items()
    }


def _importance_csv_for(results_path: pathlib.Path) -> pathlib.Path:
    """Locate the sibling engagement_importance.csv for a results JSON."""
    return results_path.parent / "engagement_importance.csv"


def load_institution_results(
    results_path: pathlib.Path,
    institution: str,
    *,
    importance_csv_path: pathlib.Path | None = None,
) -> dict[str, Any]:
    """Load both fixed-model metrics and importance rankings for one institution.

    Returns ``{"institution": <"oulad"|"ku_leuven">,
    "metrics": {split: {feature_set: {f1, roc_auc}}},
    "rankings": {split: [raw_feature, ...best-first]}}``.
    """
    metrics = load_fixed_model_metrics(results_path)
    csv_path = importance_csv_path or _importance_csv_for(results_path)
    rankings = load_importance_rankings(csv_path)
    return {
        "institution": institution,
        "metrics": metrics,
        "rankings": rankings,
    }


# ---------------------------------------------------------------------------
# Combined report builder
# ---------------------------------------------------------------------------

def build_report(
    loaded: dict[str, dict[str, Any]],
    *,
    top_k: int = 5,
    stable_tau_threshold: float = STABLE_TAU_THRESHOLD,
) -> dict[str, Any]:
    """Assemble the combined Part A + Part B report from loaded institutions.

    Parameters
    ----------
    loaded:
        ``{institution_label: load_institution_results(...) dict}``.
    """
    institution_by_label = {
        label: data["institution"] for label, data in loaded.items()
    }

    # Part A
    per_institution_metrics = {
        label: data["metrics"] for label, data in loaded.items()
    }
    baseline_key_by_inst = {
        label: _BASELINE_KEY[data["institution"]] for label, data in loaded.items()
    }
    candidate_key_by_inst = {
        label: _CANDIDATE_KEY[data["institution"]] for label, data in loaded.items()
    }
    delta_table = build_engagement_delta_table(
        per_institution_metrics,
        baseline_key_by_inst=baseline_key_by_inst,
        candidate_key_by_inst=candidate_key_by_inst,
    )

    # Part B: assemble {split: {label: raw_ranking}} over splits common to all.
    splits = sorted(
        set.intersection(
            *(set(data["rankings"].keys()) for data in loaded.values())
        )
        if loaded
        else set()
    )
    rankings_by_split: dict[str, dict[str, list[str]]] = {
        split: {label: loaded[label]["rankings"][split] for label in loaded}
        for split in splits
    }
    importance_report = build_cross_institution_importance(
        rankings_by_split,
        institution_by_label=institution_by_label,
        top_k=top_k,
        stable_tau_threshold=stable_tau_threshold,
    )

    return {
        "institutions": institution_by_label,
        "fixed_model": FIXED_MODEL,
        "baseline_key_by_inst": baseline_key_by_inst,
        "candidate_key_by_inst": candidate_key_by_inst,
        "part_a_delta_table": delta_table,
        "part_b_importance_stability": importance_report,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _fmt(value: float | None, places: int = 4) -> str:
    return f"{value:.{places}f}" if value is not None else "N/A"


def render_cross_institution_markdown(report: dict) -> str:
    """Render the combined report as a Markdown document (Part A + Part B + caveats)."""
    institutions = report["institutions"]
    fixed_model = report["fixed_model"]
    importance = report["part_b_importance_stability"]
    top_k = importance["top_k"]

    lines: list[str] = [
        "# Cross-Institution Engagement-Only Comparison (exp_012)",
        "",
        "Matched engagement-only PASSED comparison across three institution cohorts: "
        + ", ".join(f"{label} ({inst})" for label, inst in sorted(institutions.items()))
        + ".",
        "",
        f"**Fixed model** = `{fixed_model}`  |  **top_k** = {top_k}  |  "
        f"**stable_tau_threshold** = {importance['stable_tau_threshold']}",
        "",
        "## Part A -- Engagement delta (candidate B minus baseline A), fixed model",
        "",
        "Change in fixed-model classification metrics when moving from the simple-"
        "engagement baseline feature set to the richer engagement candidate set.",
        "",
        "| Institution | Split | A F1 | B F1 | delta F1 | A ROC-AUC | B ROC-AUC | delta ROC-AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["part_a_delta_table"]:
        lines.append(
            "| {inst} | {split} | {a_f1} | {b_f1} | {d_f1} | {a_roc} | {b_roc} | {d_roc} |".format(
                inst=row["institution"],
                split=row["split"],
                a_f1=_fmt(row["a_f1"]),
                b_f1=_fmt(row["b_f1"]),
                d_f1=_fmt(row["delta_f1"]),
                a_roc=_fmt(row["a_roc_auc"]),
                b_roc=_fmt(row["b_roc_auc"]),
                d_roc=_fmt(row["delta_roc_auc"]),
            )
        )

    lines += [
        "",
        "## Part B -- Importance-rank stability across institutions (shared concepts)",
        "",
        "Permutation-importance rankings aligned onto a shared engagement-concept "
        "set, then compared pairwise. Kendall tau over aligned concepts; "
        f"Jaccard over top-{top_k} concepts.",
        "",
        "| Split | Pair | Kendall tau | Jaccard@{k} | n_shared_concepts |".format(k=top_k),
        "|---|---|---|---|---|",
    ]
    for split in sorted(importance["per_split"]):
        block = importance["per_split"][split]
        for pair in block["pairs"]:
            a, b = pair["pair"]
            lines.append(
                "| {split} | {a} vs {b} | {tau} | {jac} | {n} |".format(
                    split=split,
                    a=a,
                    b=b,
                    tau=_fmt(pair["kendall_tau"]),
                    jac=_fmt(pair["jaccard_topk"]),
                    n=pair["n_shared_concepts"],
                )
            )

    lines += [
        "",
        f"**Mean Kendall tau (all pairs, all splits)**: {_fmt(importance['mean_kendall_tau'])}",
        "",
        "## Verdict",
        "",
        importance["verdict"],
        "",
        "## Caveats",
        "",
        "- **Forum/social proxy mismatch.** The `forum_engagement` concept maps to "
        "social CLICKS in OULAD (`cumulative_social_clicks_to_date`) but to forum "
        "POSTS in KU Leuven (`cumulative_forum_posts_to_date`). These are different "
        "units of behaviour; the cross-institution comparison uses RANKS, not "
        "magnitudes, precisely to avoid comparing incommensurable quantities.",
        "",
        "- **Excluded KU-only concepts.** KU Leuven carries session-level features "
        "with no OULAD analog (`cumulative_sessions_to_date`, "
        "`avg_session_clicks_to_date`, `days_since_course_start`); these are dropped "
        "from Part B's shared-concept alignment, so the stability comparison covers "
        "only the concepts present in both institutions' feature sets.",
        "",
        "- **Both splits reported.** Both `student_group` and `temporal_forward` "
        "splits are shown for completeness, but each per-cohort headline conclusion "
        "keys off only its primary split; cross-split disagreement within a cohort "
        "is expected and is not, by itself, evidence of cross-institution instability.",
        "",
        "- **Model behaviour, not causality.** Importances are permutation-importance "
        "rankings of a gradient-boosting model's learned input-output mapping. High "
        "cross-institution tau means the learned reliance pattern transfers, not that "
        "the same causal factors drive student outcomes across institutions.",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-institution engagement-only analysis: aggregate the matched "
            "exp_011 (KU Leuven) + exp_012 (OULAD DDD, BBB) benchmark runs into a "
            "B-vs-A engagement delta table and a shared-concept importance-rank "
            "stability comparison."
        )
    )
    parser.add_argument(
        "--oulad-ddd",
        type=pathlib.Path,
        default=_DEFAULT_OULAD_DDD,
        help="Path to OULAD DDD exp_012 results JSON.",
    )
    parser.add_argument(
        "--oulad-bbb",
        type=pathlib.Path,
        default=_DEFAULT_OULAD_BBB,
        help="Path to OULAD BBB exp_012 results JSON.",
    )
    parser.add_argument(
        "--ku-leuven",
        type=pathlib.Path,
        default=_DEFAULT_KU_LEUVEN,
        help="Path to KU Leuven exp_011 results JSON.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Output path prefix (without extension). "
            "Writes <output>.json and <output>.md."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k cutoff for Jaccard overlap over aligned concepts (default: 5).",
    )
    args = parser.parse_args()

    loaded = {
        "OULAD_DDD": load_institution_results(args.oulad_ddd, "oulad"),
        "OULAD_BBB": load_institution_results(args.oulad_bbb, "oulad"),
        "KU_Leuven": load_institution_results(args.ku_leuven, "ku_leuven"),
    }

    report = build_report(loaded, top_k=args.top_k)

    output_path: pathlib.Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote: {json_path}")

    md_path = output_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(render_cross_institution_markdown(report))
    print(f"Wrote: {md_path}")

    # Summary to stdout
    imp = report["part_b_importance_stability"]
    print("\n--- Cross-Institution Engagement Summary ---")
    print("Part A (delta F1 / delta ROC-AUC, fixed model):")
    for row in report["part_a_delta_table"]:
        print(
            f"  {row['institution']} / {row['split']}: "
            f"dF1={_fmt(row['delta_f1'])}, dROC={_fmt(row['delta_roc_auc'])}"
        )
    print(f"Part B mean Kendall tau = {_fmt(imp['mean_kendall_tau'])}")
    print(f"Verdict: {imp['verdict']}")


if __name__ == "__main__":
    main()
