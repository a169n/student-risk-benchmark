"""Explanation-stability metrics: rank agreement of feature-importance orderings.

These quantify whether the importance RANKING of a model's features is stable
across evaluation regimes (e.g. student-grouped vs temporal-forward split, or
across cohorts). High agreement => the explanation is regime-robust; low
agreement => the 'why' depends on the evaluation scenario, which is itself a
reportable finding. This extends explanation-stability analysis (Tiukhova et
al. 2024) to the public-benchmark transfer setting.

Rankings are passed as ordered lists of feature names, best (most important)
first. kendall_tau uses tau-b over the features shared by both rankings;
jaccard_topk measures top-k membership overlap.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

from scipy.stats import kendalltau


def jaccard_topk(features_a: Sequence[str], features_b: Sequence[str], k: int) -> float:
    a = set(list(features_a)[:k])
    b = set(list(features_b)[:k])
    union = a | b
    if not union:  # both rankings empty -> trivially identical
        return 1.0
    return len(a & b) / len(union)


def kendall_tau(ranking_a: Sequence[str], ranking_b: Sequence[str]) -> float | None:
    set_b = set(ranking_b)
    shared = [f for f in ranking_a if f in set_b]
    if len(shared) < 2:
        return None
    pos_a = {f: i for i, f in enumerate(ranking_a)}
    pos_b = {f: i for i, f in enumerate(ranking_b)}
    a_ranks = [pos_a[f] for f in shared]
    b_ranks = [pos_b[f] for f in shared]
    tau, _ = kendalltau(a_ranks, b_ranks)
    if tau is None:
        return None
    tau = float(tau)
    if tau != tau:  # NaN guard (e.g. constant input)
        return None
    return tau


def stability_report(
    importance_tables: Mapping[str, Sequence[str]],
    *,
    top_k: int = 5,
) -> dict:
    labels = list(importance_tables.keys())
    pairs: list[dict] = []
    taus: list[float] = []
    for label_a, label_b in combinations(labels, 2):
        tau = kendall_tau(importance_tables[label_a], importance_tables[label_b])
        jac = jaccard_topk(importance_tables[label_a], importance_tables[label_b], k=top_k)
        if tau is not None:
            taus.append(tau)
        pairs.append(
            {
                "pair": [label_a, label_b],
                "kendall_tau": tau,
                "jaccard_topk": jac,
                "top_k": top_k,
            }
        )
    return {
        "labels": labels,
        "top_k": top_k,
        "pairs": pairs,
        "mean_kendall_tau": (sum(taus) / len(taus)) if taus else None,
    }
