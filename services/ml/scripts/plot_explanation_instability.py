"""Figure 1: every explanation-agreement comparison in the paper, on one axis.

An agreement number between two feature rankings means nothing without bounds,
so the figure carries all of them together: the noise floor, the attainable
ceiling, the three estimator pairs, the three cohort separations, and the
analytic random baseline. The two bars involving the drop-column estimator are
starred because that estimator is degenerate on this collinear feature set; the
caption, not the plot, carries the explanation.

Usage (from services/ml):
    uv run python scripts/plot_explanation_instability.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
ART = REPO / "data" / "artifacts" / "experiments"
FIG = REPO / "paper" / "figures"

# Colour encodes WHAT was changed between the two rankings being compared.
FAMILY = {
    "reference": "#8a8a8a",
    "explainer": "#b7791f",
    "sample": "#1f5f7a",
    "institution": "#7a3b2e",
}
LABEL = {
    "reference": "reference point",
    "explainer": "explanation method changed",
    "sample": "training sample changed",
    "institution": "institution changed",
}

plt.rcParams.update(
    {
        # Times matches the IEEE body text; fonttype 42 embeds real TrueType
        # outlines, which the conference requires ("embedded fonts") and which
        # matplotlib's default Type 3 export does not give.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
    }
)


def collect() -> pd.DataFrame:
    ceiling = pd.read_csv(
        ART / "exp_018_explanation_ceiling" / "f33" / "summary.csv"
    ).set_index("reference")
    explainer = pd.read_csv(
        ART / "exp_022_explainer_agreement" / "f33" / "summary.csv"
    ).set_index("pair")
    pairs = pd.read_csv(ART / "exp_014_transfer_ladder" / "f33" / "pairs.csv")
    overlap = pd.read_csv(ART / "exp_017_contamination" / "f33" / "pair_overlap.csv")

    gbm = pairs[(pairs.model == "gradient_boosting") & (pairs.representation == "raw")].merge(
        overlap[["source", "target", "is_clean"]], on=["source", "target"], how="left"
    )
    gbm["is_clean"] = gbm["is_clean"].astype("object").fillna(True).astype(bool)
    clean = gbm[gbm.is_clean]

    def rung(distance: str) -> float:
        return float(clean.loc[clean.distance == distance, "explanation_tau"].mean())

    rows = [
        (
            "Noise floor: same model, only the seed differs",
            ceiling.loc["floor (same model, reseeded)", "tau"],
            "reference",
        ),
        (
            "Ceiling: two models, halves of one cohort",
            ceiling.loc["ceiling, ladder-matched evaluation set", "tau"],
            "sample",
        ),
        ("Permutation vs SHAP", explainer.loc["permutation vs shap", "tau"], "explainer"),
        (
            "Permutation vs drop-column *",
            explainer.loc["drop_column vs permutation", "tau"],
            "explainer",
        ),
        ("Same course, next year's students", rung("D1_same_module"), "sample"),
        ("SHAP vs drop-column *", explainer.loc["drop_column vs shap", "tau"], "explainer"),
        ("Another course, same institution", rung("D2_other_module"), "institution"),
        ("Another institution", rung("D3_other_institution"), "institution"),
    ]
    frame = pd.DataFrame(rows, columns=["label", "tau", "family"])
    # 95 % cluster-bootstrap intervals, written by scripts/estimator_intervals.py.
    ci = pd.read_csv(ART / "exp_024_estimator_conditional" / "f33" / "intervals.csv").set_index("label")
    key = {
        "Noise floor: same model, only the seed differs": "floor: same model, reseeded",
        "Ceiling: two models, halves of one cohort": "ceiling: disjoint halves, ladder-matched",
        "Permutation vs SHAP": "explainer: permutation vs shap",
        "Permutation vs drop-column *": "explainer: drop_column vs permutation",
        "SHAP vs drop-column *": "explainer: drop_column vs shap",
        "Same course, next year's students": "D1: same course, next year",
        "Another course, same institution": "D2: another course, same institution",
        "Another institution": "D3: another institution",
    }
    frame["lo"] = [ci.loc[key[l], "ci_low"] if l in key else np.nan for l in frame.label]
    frame["hi"] = [ci.loc[key[l], "ci_high"] if l in key else np.nan for l in frame.label]
    return frame.sort_values("tau")


def main() -> None:
    d = collect()
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 2.6), dpi=220)

    ys = np.arange(len(d))
    ax.barh(ys, d["tau"], height=0.62, color=[FAMILY[f] for f in d["family"]])
    has = d["lo"].notna()
    ax.errorbar(
        d.loc[has, "tau"], ys[has.to_numpy()],
        xerr=[d.loc[has, "tau"] - d.loc[has, "lo"], d.loc[has, "hi"] - d.loc[has, "tau"]],
        fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2,
    )
    for y, (tau, hi, family) in enumerate(zip(d["tau"], d["hi"], d["family"])):
        x = (hi if not np.isnan(hi) else tau) + 0.012
        ax.text(x, y, f"{tau:.3f}", va="center", fontsize=9, color=FAMILY[family])

    ax.set_yticks(ys)
    ax.set_yticklabels(d["label"], fontsize=9)
    ax.set_xlabel("agreement between the two feature rankings (Kendall τ), 95 % cluster-bootstrap interval", fontsize=9)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.set_xlim(0, 0.82)
    ax.axvline(0.0, color="#c8c8c8", lw=0.8)
    # Dotted guides at the two permutation-scale bounds, so a bar's position
    # relative to the floor and the ceiling can be read without the caption.
    ref = d.set_index("label")["tau"]
    for label, name in (
        ("Noise floor: same model, only the seed differs", "floor"),
        ("Ceiling: two models, halves of one cohort", "ceiling"),
    ):
        ax.axvline(ref[label], color="#8a8a8a", lw=0.7, ls=":")
        ax.text(ref[label] + 0.005, len(d) - 0.35, name, fontsize=8, color="#6a6a6a", va="bottom")
    ax.set_ylim(-0.6, len(d) - 0.2)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY[k])
        for k in ("explainer", "sample", "institution", "reference")
    ]
    ax.legend(
        handles,
        [LABEL[k] for k in ("explainer", "sample", "institution", "reference")],
        frameon=False,
        fontsize=8.5,
        loc="lower right",
    )
    fig.tight_layout()
    for ext in (".png", ".svg", ".pdf"):
        fig.savefig(FIG / f"fig1_explanation_instability{ext}", bbox_inches="tight")
    plt.close(fig)
    print(d.to_string(index=False))
    print("\nwritten to", FIG / "fig1_explanation_instability.pdf")


if __name__ == "__main__":
    main()
