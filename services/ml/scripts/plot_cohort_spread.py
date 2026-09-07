"""Figure 2: within-cohort AUC for all 63 cohorts, with intervals, by institution.

Section V-A claims predictability is a property of the institution rather than of
the task. A table of five institution means cannot show that, because the claim
is about the spread: the reader has to see that cohorts from one institution sit
together and that the whole range crosses chance. One row per cohort, sorted, is
the only honest way to show it, and it is also the figure that makes a reader
distrust any single-cohort headline, which is the paper's point.

Reads exp_023, which is the artifact the paper's range and interval counts come
from, so the figure and the text cannot drift apart.

Usage (from services/ml):
    uv run python scripts/plot_cohort_spread.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "data" / "artifacts" / "experiments" / "exp_023_cohort_intervals" / "f33" / "cohort_auc.csv"
FIG = REPO / "docs" / "dissertation" / "figures" / "side2026"

COLOUR = {
    "OULAD": "#1f5f7a",
    "UKZN": "#7a3b2e",
    "KU Leuven": "#b7791f",
    "Oviedo": "#4a6741",
    "Zambia": "#6b4a7a",
}

plt.rcParams.update(
    {
        # fonttype 42 embeds TrueType outlines; the conference requires embedded
        # fonts and matplotlib's default Type 3 export does not provide them.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
    }
)


def main() -> None:
    d = pd.read_csv(SRC).sort_values("auc").reset_index(drop=True)
    FIG.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3.4, 2.9), dpi=220)
    ys = np.arange(len(d))
    for y, r in zip(ys, d.itertuples()):
        c = COLOUR.get(r.institution, "#555555")
        ax.plot([r.auc_ci_low, r.auc_ci_high], [y, y], color=c, lw=1.0, alpha=0.55)
        ax.plot(r.auc, y, "o", color=c, ms=2.6)

    ax.axvline(0.5, color="#333333", lw=0.7, ls="--")
    ax.text(0.515, len(d) - 2.0, "chance", fontsize=6.5, color="#333333")
    ax.set_ylim(-1, len(d))
    ax.set_yticks([])
    ax.set_xlabel("within-cohort ROC-AUC", fontsize=8)
    ax.set_ylabel(f"{len(d)} cohorts, sorted", fontsize=8)

    order = [i for i in COLOUR if i in set(d.institution)]
    ax.legend(
        [plt.Line2D([], [], color=COLOUR[i], marker="o", ms=3, lw=1) for i in order],
        list(order),
        frameon=False,
        fontsize=6.8,
        loc="lower right",
        handlelength=1.2,
    )
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig2_cohort_spread{ext}", bbox_inches="tight")
    plt.close(fig)

    print(
        f"{len(d)} cohorts, {d.auc.min():.3f}-{d.auc.max():.3f}, "
        f"{int(d.covers_chance.sum())} cover chance, {int(d.below_chance.sum())} below"
    )
    print("written to", FIG / "fig2_cohort_spread.pdf")


if __name__ == "__main__":
    main()
