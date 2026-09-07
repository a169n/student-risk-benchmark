import math

from src.experiments.stability import jaccard_topk, kendall_tau, stability_report


def test_jaccard_identical_and_disjoint():
    assert jaccard_topk(["a", "b", "c"], ["a", "b", "c"], k=3) == 1.0
    assert jaccard_topk(["a", "b", "c"], ["x", "y", "z"], k=3) == 0.0
    # top-2 overlap of {a,b} vs {a,c} = |{a}| / |{a,b,c}| = 1/3
    assert abs(jaccard_topk(["a", "b", "c"], ["a", "c", "d"], k=2) - (1 / 3)) < 1e-9
    # both empty -> defined as 1.0
    assert jaccard_topk([], [], k=3) == 1.0


def test_kendall_identical_reversed_partial():
    assert abs(kendall_tau(["a", "b", "c", "d"], ["a", "b", "c", "d"]) - 1.0) < 1e-9
    assert abs(kendall_tau(["a", "b", "c", "d"], ["d", "c", "b", "a"]) - (-1.0)) < 1e-9
    # fewer than 2 shared features -> None
    assert kendall_tau(["a"], ["a"]) is None
    assert kendall_tau(["a", "b"], ["x", "y"]) is None


def test_kendall_uses_shared_features_only():
    # ranking_b is missing 'd'; shared = [a,b,c] in same order -> tau 1.0
    assert abs(kendall_tau(["a", "b", "c", "d"], ["a", "b", "c"]) - 1.0) < 1e-9


def test_stability_report_pairwise():
    tables = {
        "student_group": ["a", "b", "c", "d"],
        "temporal_forward": ["a", "c", "b", "d"],
    }
    report = stability_report(tables, top_k=2)
    # one unordered pair
    assert len(report["pairs"]) == 1
    pair = report["pairs"][0]
    assert set(pair["pair"]) == {"student_group", "temporal_forward"}
    assert pair["kendall_tau"] is not None
    assert 0.0 <= pair["jaccard_topk"] <= 1.0
    assert "mean_kendall_tau" in report


def test_stability_report_fewer_than_two_labels():
    report = stability_report({"only_one": ["a", "b", "c"]}, top_k=2)
    assert report["pairs"] == []
    assert report["mean_kendall_tau"] is None


def test_kendall_tau_none_when_no_shared_features():
    # zero shared features -> None (not an exception)
    assert kendall_tau(["a", "b", "c"], ["x", "y", "z"]) is None
