"""Fig. 1: the benchmark pipeline, from five releases to the reference points.

A reviewer asked for a figure of the pipeline. A vertical flow, one box per
stage and the operation written on the arrow between them, fits a single column
and reads top to bottom like the Method section. Drawn with matplotlib rather
than TikZ so that it ships as PDF and PNG like the other figures and the .docx
builder needs no new case.

Usage (from services/ml):
    uv run python scripts/plot_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "paper" / "figures"

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8,
    }
)

# (box text, text on the arrow leading to the next box). Data stages are blue,
# evaluation stages ochre.
STAGES = [
    ("5 public releases: OU VLE, Toledo, three Moodle sites", "one adapter per release"),
    ("4 weekly counters per student:\nclicks, active days, content clicks, social clicks",
     "cut at week L/3 into 7 leakage-safe features"),
    ("63 cohorts, 35,529 distinct students\none course, one period, one institution each",
     "gradient boosting per source cohort;\nlogistic regression, random forest as checks"),
    ("3,969 source–target pairs\nD0 within cohort, D1 same course next year,\nD2 other course, D3 other institution",
     "permutation, SHAP and drop-column\non the same fitted models"),
    ("agreement between feature rankings (Kendall τ)\nread against noise floor, ceiling, random baseline", None),
]
STYLE = [("#e8eef2", "#1f5f7a")] * 3 + [("#f3ecdf", "#b7791f")] * 2


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.1), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_h, gap = 0.135, 0.078
    y_top = 0.995
    for i, ((text, arrow_text), (fill, edge)) in enumerate(zip(STAGES, STYLE)):
        y = y_top - i * (box_h + gap) - box_h
        ax.add_patch(
            FancyBboxPatch(
                (0.01, y), 0.98, box_h, boxstyle="round,pad=0.005,rounding_size=0.02",
                fc=fill, ec=edge, lw=0.9,
            )
        )
        ax.text(0.5, y + box_h / 2, text, ha="center", va="center", fontsize=6.4, linespacing=1.25)
        if arrow_text:
            ax.add_patch(
                FancyArrowPatch(
                    (0.5, y - 0.004), (0.5, y - gap + 0.004),
                    arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#444444",
                )
            )
            ax.text(0.53, y - gap / 2, arrow_text, ha="left", va="center", fontsize=6.0, linespacing=1.15,
                    color="#444444", style="italic")

    fig.tight_layout(pad=0.05)
    for ext in (".png", ".pdf"):
        fig.savefig(FIG / f"fig0_pipeline{ext}", bbox_inches="tight")
    plt.close(fig)
    print("written to", FIG / "fig0_pipeline.pdf")


if __name__ == "__main__":
    main()
