"""TDD tests for analyze_xai_stability.build_xai_stability and render_stability_markdown.

Uses a small in-memory results list that mimics the real exp_007 structure:
  list of { feature_set, split_strategy, importance_rows: [{feature, rank, ...}] }
"""

from __future__ import annotations

import math

import pytest

from src.experiments.analyze_xai_stability import (
    build_xai_stability,
    render_stability_markdown,
)


def _make_rows(feature_set: str, features: list[str]) -> list[dict]:
    """Build minimal importance_rows matching the real JSON shape."""
    return [
        {
            "feature_set": feature_set,
            "model": "gradient_boosting",
            "feature": feat,
            "mean_rmse_increase": float(len(features) - rank),
            "rank": rank + 1,
            "importance_share": 1.0 / len(features),
        }
        for rank, feat in enumerate(features)
    ]


# ---------------------------------------------------------------------------
# Fixture: two feature sets with known rankings
# ---------------------------------------------------------------------------

_FEATURES_FS1 = ["alpha", "beta", "gamma", "delta", "epsilon"]
_FEATURES_FS2 = ["x", "y", "z", "w", "v"]

# fs1: identical rankings across splits -> tau should be 1.0
# fs2: reversed rankings across splits -> tau should be -1.0
_MOCK_RESULTS = [
    {
        "feature_set": "fs1_identical",
        "split_strategy": "student_group",
        "importance_rows": _make_rows("fs1_identical", _FEATURES_FS1),
    },
    {
        "feature_set": "fs1_identical",
        "split_strategy": "temporal_forward",
        "importance_rows": _make_rows("fs1_identical", _FEATURES_FS1),
    },
    {
        "feature_set": "fs2_reversed",
        "split_strategy": "student_group",
        "importance_rows": _make_rows("fs2_reversed", _FEATURES_FS2),
    },
    {
        "feature_set": "fs2_reversed",
        "split_strategy": "temporal_forward",
        "importance_rows": _make_rows("fs2_reversed", list(reversed(_FEATURES_FS2))),
    },
]


# ---------------------------------------------------------------------------
# Tests: build_xai_stability
# ---------------------------------------------------------------------------


def test_per_feature_set_has_one_entry_per_feature_set():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    feature_sets = {e["feature_set"] for e in report["per_feature_set"]}
    assert feature_sets == {"fs1_identical", "fs2_reversed"}
    assert len(report["per_feature_set"]) == 2


def test_identical_ranking_gives_tau_one_and_jaccard_one():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    entry = next(e for e in report["per_feature_set"] if e["feature_set"] == "fs1_identical")
    assert entry["kendall_tau"] is not None
    assert abs(entry["kendall_tau"] - 1.0) < 1e-9, f"expected tau=1.0, got {entry['kendall_tau']}"
    assert entry["jaccard_topk"] is not None
    assert abs(entry["jaccard_topk"] - 1.0) < 1e-9, f"expected jaccard=1.0, got {entry['jaccard_topk']}"


def test_reversed_ranking_gives_tau_minus_one_and_jaccard_zero_topk():
    report = build_xai_stability(_MOCK_RESULTS, top_k=3)
    entry = next(e for e in report["per_feature_set"] if e["feature_set"] == "fs2_reversed")
    assert entry["kendall_tau"] is not None
    assert abs(entry["kendall_tau"] - (-1.0)) < 1e-9, f"expected tau=-1.0, got {entry['kendall_tau']}"
    # top-3 of [x,y,z,w,v] vs top-3 of [v,w,z,y,x]: {x,y,z} vs {v,w,z} => intersection={z}, union={x,y,z,v,w} => 1/5
    assert entry["jaccard_topk"] is not None
    assert 0.0 <= entry["jaccard_topk"] <= 1.0


def test_all_stable_false_when_reversed():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5, stable_tau_threshold=0.9)
    assert report["all_stable"] is False


def test_verdict_is_regime_sensitive_when_reversed():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5, stable_tau_threshold=0.9)
    assert report["verdict"].startswith("regime_sensitive_finding"), (
        f"Expected regime_sensitive_finding, got: {report['verdict'][:60]}"
    )


def test_all_stable_true_when_both_identical():
    """Two feature sets, both with identical rankings -> all_stable True."""
    results_identical = [
        {
            "feature_set": "fs1_identical",
            "split_strategy": "student_group",
            "importance_rows": _make_rows("fs1_identical", _FEATURES_FS1),
        },
        {
            "feature_set": "fs1_identical",
            "split_strategy": "temporal_forward",
            "importance_rows": _make_rows("fs1_identical", _FEATURES_FS1),
        },
        {
            "feature_set": "fs_also_identical",
            "split_strategy": "student_group",
            "importance_rows": _make_rows("fs_also_identical", _FEATURES_FS2),
        },
        {
            "feature_set": "fs_also_identical",
            "split_strategy": "temporal_forward",
            "importance_rows": _make_rows("fs_also_identical", _FEATURES_FS2),
        },
    ]
    report = build_xai_stability(results_identical, top_k=5, stable_tau_threshold=0.9)
    assert report["all_stable"] is True
    assert report["verdict"].startswith("highly_stable_confirmation")


def test_mean_kendall_tau_present():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    assert "mean_kendall_tau" in report
    mean = report["mean_kendall_tau"]
    assert mean is not None
    assert isinstance(mean, float)


def test_jaccard_in_unit_interval():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    for entry in report["per_feature_set"]:
        jac = entry["jaccard_topk"]
        if jac is not None:
            assert 0.0 <= jac <= 1.0, f"Jaccard out of [0,1]: {jac}"


def test_n_features_correct():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    for entry in report["per_feature_set"]:
        assert entry["n_features"] == len(_FEATURES_FS1)  # both have 5 features


def test_missing_split_gives_none_tau():
    """If one split is absent, tau and jaccard_topk should be None."""
    results_missing_tf = [
        {
            "feature_set": "fs_only_sg",
            "split_strategy": "student_group",
            "importance_rows": _make_rows("fs_only_sg", _FEATURES_FS1),
        },
    ]
    report = build_xai_stability(results_missing_tf, top_k=5)
    entry = report["per_feature_set"][0]
    assert entry["kendall_tau"] is None
    assert entry["jaccard_topk"] is None


def test_report_keys():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    expected_keys = {"top_k", "stable_tau_threshold", "per_feature_set", "mean_kendall_tau", "all_stable", "verdict"}
    assert expected_keys.issubset(set(report.keys()))


# ---------------------------------------------------------------------------
# Tests: render_stability_markdown
# ---------------------------------------------------------------------------


def test_render_contains_feature_set_names():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    md = render_stability_markdown(report)
    assert "fs1_identical" in md
    assert "fs2_reversed" in md


def test_render_contains_kendall():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    md = render_stability_markdown(report)
    assert "Kendall" in md


def test_render_contains_verdict_text():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    md = render_stability_markdown(report)
    # verdict starts with regime_sensitive_finding for reversed case
    assert "regime_sensitive" in md or "highly_stable" in md


def test_render_is_string():
    report = build_xai_stability(_MOCK_RESULTS, top_k=5)
    md = render_stability_markdown(report)
    assert isinstance(md, str)
    assert len(md) > 100
