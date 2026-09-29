"""Fig. 2: the 63 cohorts — size against pass rate, and within-cohort AUC with intervals.

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
FIG = REPO / "paper" / "figures"

# Shape as well as colour: five hues alone are hard to tell apart in print.
MARKER = {"OULAD": "o", "UKZN": "s", "KU Leuven": "^", "Oviedo": "D", "Zambia": "v"}
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

    # Two panels spanning both columns: a reviewer found the single 3.4 in panel
    # too dense and asked for the cohort distribution itself, not only its AUC.
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(7.4, 4.5), dpi=220, gridspec_kw={"width_ratios": [1, 1.5]}
    )
    order = [i for i in COLOUR if i in set(d.institution)]

    # (a) what the cohorts are: size against pass rate, one dot per cohort.
    for inst in order:
        g = d[d.institution == inst]
        a.scatter(g.n_students, g.pass_rate, s=24, color=COLOUR[inst], marker=MARKER[inst], alpha=0.9, lw=0)
    a.set_xscale("log")
    a.set_xlabel("students in cohort (log scale)", fontsize=9)
    a.set_ylabel("pass rate", fontsize=9)
    a.set_ylim(0.2, 1.0)
    a.tick_params(labelsize=8)
    a.text(0.02, 0.97, "(a)", transform=a.transAxes, fontsize=9, va="top")

    # (b) what they yield: within-cohort AUC, sorted, with intervals.
    ys = np.arange(len(d))
    for y, r in zip(ys, d.itertuples()):
        c = COLOUR.get(r.institution, "#555555")
        b.plot([r.auc_ci_low, r.auc_ci_high], [y, y], color=c, lw=0.9, alpha=0.45)
        b.plot(r.auc, y, MARKER.get(r.institution, "o"), color=c, ms=3.4)
    b.axvline(0.5, color="#333333", lw=0.7, ls="--")
    b.text(0.51, len(d) - 1.5, "chance", fontsize=8, color="#333333")
    b.set_ylim(-1, len(d))
    b.set_yticks([])
    b.set_xlabel("within-cohort ROC-AUC, 95 % bootstrap interval", fontsize=9)
    b.set_ylabel(f"{len(d)} cohorts, sorted by AUC", fontsize=9)
    b.tick_params(labelsize=8)
    b.text(0.02, 0.97, "(b)", transform=b.transAxes, fontsize=9, va="top")
    b.legend(
        [plt.Line2D([], [], color=COLOUR[i], marker=MARKER[i], ms=4, lw=1.1) for i in order],
        list(order),
        frameon=False,
        fontsize=8,
        loc="lower right",
        handlelength=1.2,
    )
    fig.tight_layout(w_pad=1.5)
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
