"""Cross-cohort explanation-stability analysis: DDD 2013J vs BBB 2013J.

Compares feature-importance rankings between two OULAD courses for the same
(feature_set, split_strategy) combination, using the already-computed
per-experiment results JSON files.  No model re-training or XAI re-computation
is performed; all data come from the artifacts.

Methodology mirrors analyze_xai_stability (within-cohort, cross-split) but
now holds the split fixed and varies the cohort.  Because both courses use the
same OULAD adapter and feature columns, the ranking universes are identical and
Kendall tau is well-defined over the full shared feature list.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

from src.experiments.stability import jaccard_topk, kendall_tau

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_DEFAULT_RESULTS_A = (
    _REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_007_xai_on_oulad"
    / "exp_007_xai_on_oulad_results.json"
)
_DEFAULT_RESULTS_B = (
    _REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_010_xai_on_oulad_bbb2013j"
    / "exp_010_xai_on_oulad_bbb2013j_results.json"
)
_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_010_xai_on_oulad_bbb2013j"
    / "cross_cohort_stability"
)


# ---------------------------------------------------------------------------
# Extraction helper (mirrors analyze_xai_stability._extract_rankings)
# ---------------------------------------------------------------------------

def _extract_rankings(results: list[dict]) -> dict[str, dict[str, list[str]]]:
    """Return {feature_set: {split_strategy: [feature, ...best-first]}}."""
    index: dict[str, dict[str, list[str]]] = {}
    for rec in results:
        fs = rec["feature_set"]
        sp = rec["split_strategy"]
        rows = rec["importance_rows"]  # already sorted best-first by rank
        features = [r["feature"] for r in rows]
        index.setdefault(fs, {})[sp] = features
    return index


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def build_cross_cohort_stability(
    results_a: list[dict] | dict,
    results_b: list[dict] | dict,
    *,
    cohort_a_label: str = "DDD_2013J",
    cohort_b_label: str = "BBB_2013J",
    top_k: int = 5,
    stable_tau_threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute cross-cohort explanation stability for matched (feature_set, split) cells.

    Parameters
    ----------
    results_a, results_b:
        Raw contents of the per-cohort results JSON (list of records, each
        with ``feature_set``, ``split_strategy``, ``importance_rows``).
    cohort_a_label, cohort_b_label:
        Human-readable cohort names included in the report.
    top_k:
        Number of top features used for Jaccard overlap.
    stable_tau_threshold:
        Minimum Kendall tau to consider a cell's cross-cohort rankings stable.

    Returns
    -------
    dict with keys: cohort_a, cohort_b, top_k, stable_tau_threshold,
    per_cell, mean_kendall_tau, all_stable, verdict.
    """
    # Support both list and dict-wrapper inputs (mirrors analyze_xai_stability)
    def _normalise(raw: list[dict] | dict) -> list[dict]:
        if isinstance(raw, dict):
            return raw.get("records", raw.get("data", []))
        return raw

    records_a = _normalise(results_a)
    records_b = _normalise(results_b)

    index_a = _extract_rankings(records_a)
    index_b = _extract_rankings(records_b)

    # Collect all (feature_set, split) keys present in BOTH cohorts
    per_cell: list[dict[str, Any]] = []
    valid_taus: list[float] = []

    all_keys_a: set[tuple[str, str]] = {
        (fs, sp)
        for fs, split_map in index_a.items()
        for sp in split_map
    }
    all_keys_b: set[tuple[str, str]] = {
        (fs, sp)
        for fs, split_map in index_b.items()
        for sp in split_map
    }
    shared_keys = sorted(all_keys_a & all_keys_b)

    for fs, sp in shared_keys:
        ranking_a = index_a[fs][sp]
        ranking_b = index_b[fs][sp]

        tau = kendall_tau(ranking_a, ranking_b)
        jac = jaccard_topk(ranking_a, ranking_b, k=top_k)
        n_features = len(ranking_a)

        if tau is not None:
            valid_taus.append(tau)

        per_cell.append(
            {
                "feature_set": fs,
                "split": sp,
                "kendall_tau": tau,
                "jaccard_topk": jac,
                "n_features": n_features,
            }
        )

    mean_tau = (sum(valid_taus) / len(valid_taus)) if valid_taus else None

    all_stable = bool(
        valid_taus
        and all(t >= stable_tau_threshold for t in valid_taus)
        and all(c["kendall_tau"] is not None for c in per_cell)
    )

    if all_stable:
        verdict = "explanations_transfer_across_courses"
    else:
        verdict = "explanations_partly_course_specific"

    return {
        "cohort_a": cohort_a_label,
        "cohort_b": cohort_b_label,
        "top_k": top_k,
        "stable_tau_threshold": stable_tau_threshold,
        "per_cell": per_cell,
        "mean_kendall_tau": mean_tau,
        "all_stable": all_stable,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_cross_cohort_markdown(report: dict) -> str:
    """Render the cross-cohort stability report as a Markdown document."""
    cohort_a = report["cohort_a"]
    cohort_b = report["cohort_b"]
    top_k = report["top_k"]
    threshold = report["stable_tau_threshold"]

    lines: list[str] = [
        f"# Cross-Cohort Explanation Stability: {cohort_a} vs {cohort_b}",
        "",
        f"**Cohorts compared**: {cohort_a} (exp_007) vs {cohort_b} (exp_010)  |  "
        f"**top_k** = {top_k}  |  "
        f"**stable_tau_threshold** = {threshold}  |  "
        f"**all_stable** = {report['all_stable']}",
        "",
        f"| Feature Set | Split | Kendall tau ({cohort_a} vs {cohort_b}) | Jaccard@{top_k} | n_features |",
        f"|---|---|---|---|---|",
    ]

    for cell in report["per_cell"]:
        fs = cell["feature_set"]
        sp = cell["split"]
        tau = cell["kendall_tau"]
        jac = cell["jaccard_topk"]
        n = cell["n_features"]
        tau_str = f"{tau:.4f}" if tau is not None else "N/A"
        jac_str = f"{jac:.4f}" if jac is not None else "N/A"
        lines.append(f"| {fs} | {sp} | {tau_str} | {jac_str} | {n} |")

    mean_tau = report["mean_kendall_tau"]
    mean_str = f"{mean_tau:.4f}" if mean_tau is not None else "N/A"
    lines += [
        "",
        f"**Mean Kendall tau**: {mean_str}",
        "",
        "## Verdict",
        "",
        report["verdict"],
        "",
        "## Interpretation",
        "",
        (
            "These metrics compare how consistently the gradient-boosting model "
            "ranks features across the two OULAD courses — "
            f"{cohort_a} (module DDD, presentation 2013J) and "
            f"{cohort_b} (module BBB, presentation 2013J) — for each "
            "(feature_set, split_strategy) combination.  Both courses are "
            "processed through the same adapter and share the same feature "
            "columns, so the Kendall tau is computed over the full, identical "
            "feature universe (no alignment step needed)."
        ),
        "",
        (
            "**Ties to exp_009 ablation findings.** "
            "In exp_009's ablation, adding the mastery block "
            "(B_lms_plus_mastery_oulad vs B_lms_oulad) improved predictive "
            "performance on BBB 2013J but produced no consistent gain on "
            "DDD 2013J.  If that asymmetry is reflected in the importances, "
            "we would expect mastery features to rank higher relative to LMS "
            "features on BBB than on DDD — which would depress the "
            "B_lms_plus_mastery_oulad cross-cohort tau.  Conversely, stable "
            "tau on that feature set would suggest that, despite the "
            "performance difference, the model still leans on the same "
            "features in both courses; the mastery block's predictive benefit "
            "on BBB may then come from features already present in the LMS set "
            "acting more strongly, rather than the mastery features themselves "
            "rising in relative rank."
        ),
        "",
        (
            "**Model-behavior caveat.** "
            "These are permutation-importance rankings of a gradient-boosting "
            "model's learned input-output mapping, not causal effect estimates.  "
            "A feature ranked #1 is one whose removal most disrupts the model's "
            "predictions; that disruption reflects the model's reliance on the "
            "feature within a particular cohort and evaluation regime, not an "
            "invariant causal mechanism.  Cross-cohort tau measures whether "
            "that learned reliance pattern transfers — high tau means the two "
            "course populations produce similar model behaviour, not that the "
            "same causal factors drive student outcomes in both courses.  "
            "Combined with the mixed-to-null predictive R² findings from "
            "exp_006/exp_009, any interpretation of these importances as "
            "actionable levers should be made with caution."
        ),
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-cohort explanation-stability analysis: "
            "compare feature-importance rankings between two OULAD courses."
        )
    )
    parser.add_argument(
        "--results-a",
        type=pathlib.Path,
        default=_DEFAULT_RESULTS_A,
        help="Path to cohort A results JSON (default: DDD exp_007)",
    )
    parser.add_argument(
        "--results-b",
        type=pathlib.Path,
        default=_DEFAULT_RESULTS_B,
        help="Path to cohort B results JSON (default: BBB exp_010)",
    )
    parser.add_argument(
        "--label-a",
        default="DDD_2013J",
        help="Human-readable label for cohort A (default: DDD_2013J)",
    )
    parser.add_argument(
        "--label-b",
        default="BBB_2013J",
        help="Human-readable label for cohort B (default: BBB_2013J)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT,
        help=(
            "Output path prefix (without extension).  "
            "Writes <output>.json and <output>.md  "
            "(default: .../exp_010_xai_on_oulad_bbb2013j/cross_cohort_stability)"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top features for Jaccard overlap (default: 5)",
    )
    parser.add_argument(
        "--stable-tau-threshold",
        type=float,
        default=0.9,
        help="Minimum Kendall tau to count as stable (default: 0.9)",
    )
    args = parser.parse_args()

    print(f"Loading cohort A ({args.label_a}) from: {args.results_a}")
    with args.results_a.open(encoding="utf-8") as fh:
        raw_a = json.load(fh)

    print(f"Loading cohort B ({args.label_b}) from: {args.results_b}")
    with args.results_b.open(encoding="utf-8") as fh:
        raw_b = json.load(fh)

    report = build_cross_cohort_stability(
        raw_a,
        raw_b,
        cohort_a_label=args.label_a,
        cohort_b_label=args.label_b,
        top_k=args.top_k,
        stable_tau_threshold=args.stable_tau_threshold,
    )

    output_path: pathlib.Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    json_path = output_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote: {json_path}")

    md_path = output_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(render_cross_cohort_markdown(report))
    print(f"Wrote: {md_path}")

    # Print summary to stdout
    print(f"\n--- Cross-Cohort Stability Summary: {args.label_a} vs {args.label_b} ---")
    for cell in report["per_cell"]:
        tau = cell["kendall_tau"]
        jac = cell["jaccard_topk"]
        tau_str = f"{tau:.4f}" if tau is not None else "N/A"
        jac_str = f"{jac:.4f}" if jac is not None else "N/A"
        print(
            f"  {cell['feature_set']} / {cell['split']}: "
            f"tau={tau_str}, jaccard@{report['top_k']}={jac_str}, "
            f"n={cell['n_features']}"
        )
    mean_tau = report["mean_kendall_tau"]
    mean_str = f"{mean_tau:.4f}" if mean_tau is not None else "N/A"
    print(f"  mean_kendall_tau={mean_str}")
    print(f"  all_stable={report['all_stable']}")
    print(f"  verdict={report['verdict']}")


if __name__ == "__main__":
    main()
