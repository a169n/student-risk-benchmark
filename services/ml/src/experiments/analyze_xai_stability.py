"""Explanation-stability analysis over exp_007 OULAD XAI results.

Compares feature-importance RANKINGS between student_group and
temporal_forward evaluation splits, per feature set, using the
already-computed exp_007 results JSON.  No model re-training or
XAI re-computation is performed; all data come from the artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

from src.experiments.stability import jaccard_topk, kendall_tau

_DEFAULT_RESULTS = (
    pathlib.Path(__file__).resolve().parents[4]
    / "data"
    / "artifacts"
    / "experiments"
    / "exp_007_xai_on_oulad"
    / "exp_007_xai_on_oulad_results.json"
)
_DEFAULT_OUTPUT_DIR = _DEFAULT_RESULTS.parent


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


def build_xai_stability(
    results: list[dict] | dict,
    *,
    top_k: int = 5,
    stable_tau_threshold: float = 0.9,
) -> dict[str, Any]:
    """Compute cross-split explanation stability for each feature set.

    Parameters
    ----------
    results:
        The raw contents of exp_007_xai_on_oulad_results.json (list of
        per-(feature_set, split_strategy) records).
    top_k:
        Number of top features used for Jaccard overlap.
    stable_tau_threshold:
        Minimum Kendall tau to consider a feature set's rankings stable.

    Returns
    -------
    dict with keys: top_k, stable_tau_threshold, per_feature_set,
    mean_kendall_tau, all_stable, verdict.
    """
    # Support both the real list format and a dict wrapper for tests
    if isinstance(results, dict):
        records: list[dict] = results.get("records", results.get("data", []))
    else:
        records = results

    ranking_index = _extract_rankings(records)

    per_feature_set: list[dict[str, Any]] = []
    valid_taus: list[float] = []

    for fs, split_map in ranking_index.items():
        sg_ranking = split_map.get("student_group")
        tf_ranking = split_map.get("temporal_forward")

        if sg_ranking is None or tf_ranking is None:
            # One split is missing — record tau as None
            tau = None
            jac = None
            n_features = len(sg_ranking or tf_ranking or [])
        else:
            tau = kendall_tau(sg_ranking, tf_ranking)
            jac = jaccard_topk(sg_ranking, tf_ranking, k=top_k)
            n_features = len(sg_ranking)

        if tau is not None:
            valid_taus.append(tau)

        per_feature_set.append(
            {
                "feature_set": fs,
                "kendall_tau": tau,
                "jaccard_topk": jac,
                "n_features": n_features,
            }
        )

    mean_tau = (sum(valid_taus) / len(valid_taus)) if valid_taus else None

    all_stable = bool(
        valid_taus
        and all(t >= stable_tau_threshold for t in valid_taus)
        and all(e["kendall_tau"] is not None for e in per_feature_set)
    )

    if all_stable:
        verdict = (
            "highly_stable_confirmation: Feature-importance rankings are "
            "consistent across student_group and temporal_forward evaluation "
            "splits (all Kendall tau >= {thresh:.2f}).  Stability is therefore "
            "a confirmation paragraph rather than a standalone XAI contribution "
            "— the rankings are regime-robust, meaning the model's 'explanation' "
            "does not change depending on how the test set is constructed.  "
            "Given exp_006's mixed-to-null predictive finding, this stability "
            "tells us the model explains the same (possibly confounded) signal "
            "across both regimes.".format(thresh=stable_tau_threshold)
        )
    else:
        verdict = (
            "regime_sensitive_finding: Feature-importance rankings differ "
            "meaningfully between student_group and temporal_forward evaluation "
            "splits (at least one Kendall tau < {thresh:.2f}).  This is itself "
            "a reportable result: which features the model 'relies on' depends "
            "on the evaluation scenario (split strategy).  Combined with "
            "exp_006's mixed-to-null predictive finding, regime-sensitive "
            "explanations further caution against interpreting these importances "
            "as stable causal signals — they reflect model behavior under a "
            "specific evaluation regime, not an invariant mechanism.".format(
                thresh=stable_tau_threshold
            )
        )

    return {
        "top_k": top_k,
        "stable_tau_threshold": stable_tau_threshold,
        "per_feature_set": per_feature_set,
        "mean_kendall_tau": mean_tau,
        "all_stable": all_stable,
        "verdict": verdict,
    }


def render_stability_markdown(report: dict) -> str:
    """Render the stability report as a Markdown section."""
    top_k = report["top_k"]
    lines: list[str] = [
        "# Explanation Stability: student_group vs temporal_forward",
        "",
        f"**top_k** = {top_k}  |  "
        f"**stable_tau_threshold** = {report['stable_tau_threshold']}  |  "
        f"**all_stable** = {report['all_stable']}",
        "",
        f"| Feature Set | Kendall tau (SG vs TF) | Jaccard@{top_k} | n_features |",
        f"|---|---|---|---|",
    ]

    for entry in report["per_feature_set"]:
        fs = entry["feature_set"]
        tau = entry["kendall_tau"]
        jac = entry["jaccard_topk"]
        n = entry["n_features"]
        tau_str = f"{tau:.4f}" if tau is not None else "N/A"
        jac_str = f"{jac:.4f}" if jac is not None else "N/A"
        lines.append(f"| {fs} | {tau_str} | {jac_str} | {n} |")

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
            "ranks features across two evaluation regimes — student-grouped "
            "(random student split) and temporal-forward (train on earlier "
            "cohorts, test on later ones) — within the OULAD public benchmark. "
            "High Kendall tau indicates that the same features dominate "
            "regardless of which split is used; low tau indicates that the "
            f"top-{top_k} importance hierarchy is regime-sensitive. "
            "Critically, these are model-behavior importance rankings, not "
            "causal mechanisms: a feature ranked #1 by permutation importance "
            "is one whose removal most disrupts the model's learned "
            "input-output mapping, which may itself be a proxy or artifact. "
            "Linking to exp_006: the mixed-to-null predictive R² across "
            "feature sets and splits (particularly weaker temporal-forward "
            "performance) means that stable importance rankings, if found, "
            "confirm stability of an already-questionable signal, not "
            "confirmation of causal relevance. Regime-sensitive importances "
            "compound this concern: not only does predictive power vary by "
            "evaluation regime, but so does the model's reliance on individual "
            "features, limiting the interpretive value of any single importance "
            "ranking without anchoring it to a specific evaluation context."
        ),
    ]

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explanation-stability analysis over exp_007 OULAD XAI results."
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=_DEFAULT_RESULTS,
        help="Path to exp_007_xai_on_oulad_results.json",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where explanation_stability.json and .md are written",
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

    results_path: pathlib.Path = args.results
    output_dir: pathlib.Path = args.output_dir

    print(f"Loading results from: {results_path}")
    with results_path.open() as fh:
        raw = json.load(fh)

    report = build_xai_stability(
        raw,
        top_k=args.top_k,
        stable_tau_threshold=args.stable_tau_threshold,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "explanation_stability.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote: {json_path}")

    md_path = output_dir / "explanation_stability.md"
    with md_path.open("w", encoding="utf-8") as fh:
        fh.write(render_stability_markdown(report))
    print(f"Wrote: {md_path}")

    # Print summary to stdout
    print("\n--- Stability Summary ---")
    for entry in report["per_feature_set"]:
        tau = entry["kendall_tau"]
        jac = entry["jaccard_topk"]
        tau_str = f"{tau:.4f}" if tau is not None else "N/A"
        jac_str = f"{jac:.4f}" if jac is not None else "N/A"
        print(
            f"  {entry['feature_set']}: tau={tau_str}, jaccard@{report['top_k']}={jac_str}, "
            f"n={entry['n_features']}"
        )
    mean_tau = report["mean_kendall_tau"]
    mean_str = f"{mean_tau:.4f}" if mean_tau is not None else "N/A"
    print(f"  mean_kendall_tau={mean_str}")
    print(f"  all_stable={report['all_stable']}")
    print(f"  verdict={report['verdict'][:80]}...")


if __name__ == "__main__":
    main()
