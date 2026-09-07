"""TDD tests for analyze_cross_cohort_stability.

Uses small in-memory results_a / results_b (2 feature sets x 2 splits each,
importance_rows with a few features) to verify:
  - per_cell has one entry per (feature_set, split) present in both cohorts.
  - identical rankings across cohorts -> kendall_tau == 1.0.
  - reversed rankings -> kendall_tau == -1.0, all_stable == False,
    verdict == "explanations_partly_course_specific".
  - render returns a string containing "Kendall" and both cohort labels.
"""

from __future__ import annotations

import pytest

from src.experiments.analyze_cross_cohort_stability import (
    build_cross_cohort_stability,
    render_cross_cohort_markdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _make_record(feature_set: str, split: str, features: list[str]) -> dict:
    return {
        "feature_set": feature_set,
        "split_strategy": split,
        "importance_rows": _make_rows(feature_set, features),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FEATURES_FS1 = ["alpha", "beta", "gamma", "delta", "epsilon"]
_FEATURES_FS2 = ["x", "y", "z", "w", "v"]

_SPLITS = ["student_group", "temporal_forward"]


def _make_cohort_identical():
    """Both feature sets, both splits with _FEATURES_FS1 / _FEATURES_FS2."""
    records = []
    for sp in _SPLITS:
        records.append(_make_record("fs_lms", sp, _FEATURES_FS1))
        records.append(_make_record("fs_twin", sp, _FEATURES_FS2))
    return records


def _make_cohort_reversed():
    """fs_lms is reversed relative to _make_cohort_identical; fs_twin is identical."""
    records = []
    for sp in _SPLITS:
        records.append(_make_record("fs_lms", sp, list(reversed(_FEATURES_FS1))))
        records.append(_make_record("fs_twin", sp, _FEATURES_FS2))
    return records


# Base cohort A: canonical order
_RESULTS_A = _make_cohort_identical()

# Cohort B identical -> all taus == 1.0
_RESULTS_B_IDENTICAL = _make_cohort_identical()

# Cohort B with fs_lms reversed -> fs_lms tau == -1.0
_RESULTS_B_REVERSED_FS_LMS = _make_cohort_reversed()


# ---------------------------------------------------------------------------
# Tests: structure
# ---------------------------------------------------------------------------

def test_per_cell_count_matches_shared_combinations():
    """per_cell should have one entry per (feature_set, split) in both cohorts."""
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    # 2 feature sets x 2 splits = 4 cells
    assert len(report["per_cell"]) == 4


def test_per_cell_keys_present():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    for cell in report["per_cell"]:
        assert "feature_set" in cell
        assert "split" in cell
        assert "kendall_tau" in cell
        assert "jaccard_topk" in cell
        assert "n_features" in cell


def test_report_top_level_keys():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    expected = {"cohort_a", "cohort_b", "top_k", "stable_tau_threshold",
                "per_cell", "mean_kendall_tau", "all_stable", "verdict"}
    assert expected.issubset(set(report.keys()))


def test_cohort_labels_stored():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD_2013J", cohort_b_label="BBB_2013J",
    )
    assert report["cohort_a"] == "DDD_2013J"
    assert report["cohort_b"] == "BBB_2013J"


def test_only_shared_cells_included():
    """If cohort B is missing one feature_set entirely, that key must be absent."""
    results_b_partial = [
        _make_record("fs_lms", "student_group", _FEATURES_FS1),
        _make_record("fs_lms", "temporal_forward", _FEATURES_FS1),
        # fs_twin absent in B
    ]
    report = build_cross_cohort_stability(
        _RESULTS_A, results_b_partial,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    feature_sets_in_cells = {c["feature_set"] for c in report["per_cell"]}
    assert "fs_twin" not in feature_sets_in_cells
    assert "fs_lms" in feature_sets_in_cells


# ---------------------------------------------------------------------------
# Tests: metric correctness
# ---------------------------------------------------------------------------

def test_identical_rankings_give_tau_one():
    """Identical feature orderings in both cohorts -> kendall_tau == 1.0."""
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    for cell in report["per_cell"]:
        tau = cell["kendall_tau"]
        assert tau is not None
        assert abs(tau - 1.0) < 1e-9, (
            f"Expected tau=1.0 for identical rankings, got {tau} "
            f"(feature_set={cell['feature_set']}, split={cell['split']})"
        )


def test_identical_rankings_give_jaccard_one():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD", cohort_b_label="BBB",
        top_k=5,
    )
    for cell in report["per_cell"]:
        jac = cell["jaccard_topk"]
        assert jac is not None
        assert abs(jac - 1.0) < 1e-9, (
            f"Expected jaccard=1.0 for identical rankings, got {jac}"
        )


def test_reversed_fs_lms_gives_tau_minus_one():
    """fs_lms reversed in cohort B -> kendall_tau == -1.0 for those cells."""
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_REVERSED_FS_LMS,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    lms_cells = [c for c in report["per_cell"] if c["feature_set"] == "fs_lms"]
    assert len(lms_cells) == 2  # one per split
    for cell in lms_cells:
        tau = cell["kendall_tau"]
        assert tau is not None
        assert abs(tau - (-1.0)) < 1e-9, (
            f"Expected tau=-1.0 for reversed ranking, got {tau}"
        )


def test_reversed_gives_all_stable_false():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_REVERSED_FS_LMS,
        cohort_a_label="DDD", cohort_b_label="BBB",
        stable_tau_threshold=0.9,
    )
    assert report["all_stable"] is False


def test_reversed_gives_partly_course_specific_verdict():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_REVERSED_FS_LMS,
        cohort_a_label="DDD", cohort_b_label="BBB",
        stable_tau_threshold=0.9,
    )
    assert report["verdict"] == "explanations_partly_course_specific", (
        f"Expected 'explanations_partly_course_specific', got: {report['verdict']}"
    )


def test_identical_gives_transfer_verdict():
    """All taus == 1.0 -> verdict == 'explanations_transfer_across_courses'."""
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD", cohort_b_label="BBB",
        stable_tau_threshold=0.9,
    )
    assert report["all_stable"] is True
    assert report["verdict"] == "explanations_transfer_across_courses", (
        f"Expected 'explanations_transfer_across_courses', got: {report['verdict']}"
    )


def test_mean_kendall_tau_is_float():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    mean = report["mean_kendall_tau"]
    assert mean is not None
    assert isinstance(mean, float)


def test_mean_tau_approximately_correct_for_mixed():
    """With 4 cells (2 x 2 splits): fs_lms tau=-1, fs_twin tau=1 -> mean=0."""
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_REVERSED_FS_LMS,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    # 2 cells with tau=-1 (fs_lms), 2 cells with tau=1 (fs_twin) -> mean=0.0
    mean = report["mean_kendall_tau"]
    assert mean is not None
    assert abs(mean - 0.0) < 1e-9, f"Expected mean tau=0.0, got {mean}"


def test_jaccard_in_unit_interval():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_REVERSED_FS_LMS,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    for cell in report["per_cell"]:
        jac = cell["jaccard_topk"]
        if jac is not None:
            assert 0.0 <= jac <= 1.0, f"Jaccard out of [0,1]: {jac}"


def test_n_features_populated():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD", cohort_b_label="BBB",
    )
    for cell in report["per_cell"]:
        assert cell["n_features"] > 0


# ---------------------------------------------------------------------------
# Tests: render_cross_cohort_markdown
# ---------------------------------------------------------------------------

def test_render_is_string():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD_2013J", cohort_b_label="BBB_2013J",
    )
    md = render_cross_cohort_markdown(report)
    assert isinstance(md, str)
    assert len(md) > 100


def test_render_contains_kendall():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD_2013J", cohort_b_label="BBB_2013J",
    )
    md = render_cross_cohort_markdown(report)
    assert "Kendall" in md


def test_render_contains_cohort_a_label():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD_2013J", cohort_b_label="BBB_2013J",
    )
    md = render_cross_cohort_markdown(report)
    assert "DDD_2013J" in md


def test_render_contains_cohort_b_label():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD_2013J", cohort_b_label="BBB_2013J",
    )
    md = render_cross_cohort_markdown(report)
    assert "BBB_2013J" in md


def test_render_contains_feature_set_names():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD_2013J", cohort_b_label="BBB_2013J",
    )
    md = render_cross_cohort_markdown(report)
    assert "fs_lms" in md
    assert "fs_twin" in md


def test_render_contains_verdict():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_REVERSED_FS_LMS,
        cohort_a_label="DDD_2013J", cohort_b_label="BBB_2013J",
    )
    md = render_cross_cohort_markdown(report)
    assert "explanations_partly_course_specific" in md or "explanations_transfer" in md


def test_render_contains_interpretation_section():
    report = build_cross_cohort_stability(
        _RESULTS_A, _RESULTS_B_IDENTICAL,
        cohort_a_label="DDD_2013J", cohort_b_label="BBB_2013J",
    )
    md = render_cross_cohort_markdown(report)
    assert "Interpretation" in md
    assert "causal" in md  # honest caveat must be present
