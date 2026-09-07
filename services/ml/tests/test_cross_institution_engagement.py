"""TDD tests for analyze_cross_institution_engagement.

Verifies the three behaviours that drive exp_012's matched engagement-only
cross-institution comparison:

  1. Concept alignment renames raw per-institution feature names to canonical
     concepts, DROPS unmapped (KU-only) features, preserves order, and dedupes.
  2. Kendall tau over aligned concepts == 1.0 for identical rankings.
  3. The B-vs-A delta table computes delta = candidate.f1 - baseline.f1 from the
     fixed-model classification metrics.
"""

from __future__ import annotations

from src.experiments.analyze_cross_institution_engagement import (
    CONCEPT_ALIGNMENT,
    align_ranking_to_concepts,
    build_cross_institution_importance,
    build_engagement_delta_table,
)


def test_align_ranking_to_concepts_renames_and_drops_unmapped():
    ranking = [
        "cumulative_active_days_to_date",
        "cumulative_clicks_to_date",
        "cumulative_sessions_to_date",  # KU-only -> dropped
    ]
    aligned = align_ranking_to_concepts(ranking, institution="ku_leuven")
    assert aligned == ["active_days", "total_clicks"]


def test_align_ranking_dedupes_preserving_first_occurrence():
    # Two raw names map to the same concept only via different institutions,
    # but within one institution duplicate raw names should collapse to one
    # concept, keeping the first occurrence's order.
    ranking = [
        "cumulative_clicks_to_date",
        "cumulative_active_days_to_date",
        "cumulative_clicks_to_date",  # duplicate -> dropped
    ]
    aligned = align_ranking_to_concepts(ranking, institution="oulad")
    assert aligned == ["total_clicks", "active_days"]


def test_importance_stability_identical_rankings_tau_one():
    # The function accepts raw rankings keyed by institution; alignment renames
    # them to canonical concepts. Identical raw rankings -> tau 1.0 on every pair.
    raw_rankings = {
        "OULAD_DDD": [
            "cumulative_active_days_to_date",
            "cumulative_clicks_to_date",
            "cumulative_content_clicks_to_date",
            "week_number",
        ],
        "KU_Leuven": [
            "cumulative_active_days_to_date",
            "cumulative_clicks_to_date",
            "cumulative_content_clicks_to_date",
            "week_number",
        ],
    }
    institution_by_label = {"OULAD_DDD": "oulad", "KU_Leuven": "ku_leuven"}
    report = build_cross_institution_importance(
        {"temporal_forward": raw_rankings},
        institution_by_label=institution_by_label,
        top_k=3,
    )
    split_report = report["per_split"]["temporal_forward"]
    pair = split_report["pairs"][0]
    assert pair["kendall_tau"] == 1.0
    # Identical rankings -> every tau = 1.0 >= 0.9 -> drivers transfer.
    assert report["verdict"] == "drivers_transfer_across_institutions"
    # Alignment must rename raw feature names to canonical concepts only: every
    # feature appearing in the aligned rankings must be a key of CONCEPT_ALIGNMENT.
    canonical = set(CONCEPT_ALIGNMENT)
    for aligned in split_report["aligned_rankings"].values():
        assert aligned, "expected non-empty aligned ranking"
        assert set(aligned) <= canonical


def test_delta_table_computes_b_minus_a():
    per_institution = {
        "OULAD_DDD": {
            "temporal_forward": {
                "A_simple_engagement_oulad": {"f1": 0.70, "roc_auc": 0.90},
                "B_engagement_oulad": {"f1": 0.68, "roc_auc": 0.91},
            },
        },
    }
    table = build_engagement_delta_table(
        per_institution,
        baseline_key_by_inst={"OULAD_DDD": "A_simple_engagement_oulad"},
        candidate_key_by_inst={"OULAD_DDD": "B_engagement_oulad"},
    )
    row = [
        r
        for r in table
        if r["institution"] == "OULAD_DDD" and r["split"] == "temporal_forward"
    ][0]
    assert abs(row["delta_f1"] - (-0.02)) < 1e-9
    assert abs(row["delta_roc_auc"] - 0.01) < 1e-9
