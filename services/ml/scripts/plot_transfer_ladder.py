"""Figures for exp_014 (transfer ladder). Reads the CSVs written by run_transfer_ladder.py.

Usage (from services/ml):
    uv run python scripts/plot_transfer_ladder.py --run f33
Writes PNG + SVG into docs/dissertation/figures/side2026/.

All ladder figures are computed from ``pairs.csv`` with the SAME pairing rule as
the paper tables: aggregate per target cohort, then keep only targets present at
all four distances. The pre-aggregated ``aggregate.csv`` must not be plotted -
its four distance classes average over different sets of target cohorts, which
makes D1 appear better than D0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
FIG = REPO / "docs" / "dissertation" / "figures" / "side2026"

DIST = ["D0_within_cohort", "D1_same_module", "D2_other_module", "D3_other_institution"]
DIST_LABEL = [
    "D0\nwithin\ncohort",
    "D1\nsame course\nother year",
    "D2\nother course\nsame inst.",
    "D3\nother\ninstitution",
]
REP_STYLE = {"raw": ("#7a7a7a", "o"), "zscore": ("#1f5f7a", "s"), "percentile": ("#b7791f", "D")}
MODEL_TITLE = {
    "gradient_boosting": "Gradient boosting",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
    }
)


def paired(pairs: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Restrict to targets present at all four distances (see module docstring)."""
    complete = pairs.groupby("target")["distance"].nunique()
    keep = set(complete[complete == len(DIST)].index)
    return pairs[pairs["target"].isin(keep)], len(keep)


def _curve(frame: pd.DataFrame, metric: str, n_boot: int = 1000, seed: int = 0):
    """Per-distance mean over target cohorts, with a bootstrap CI over targets."""
    rng = np.random.default_rng(seed)
    means, los, his = [], [], []
    for d in DIST:
        per_target = frame.loc[frame["distance"] == d].groupby("target")[metric].mean().dropna().to_numpy()
        if len(per_target) == 0:
            means.append(np.nan)
            los.append(np.nan)
            his.append(np.nan)
            continue
        boots = np.array([rng.choice(per_target, len(per_target), replace=True).mean() for _ in range(n_boot)])
        means.append(per_target.mean())
        los.append(np.percentile(boots, 2.5))
        his.append(np.percentile(boots, 97.5))
    return np.array(means), np.array(los), np.array(his)


def _panel(ax, frame: pd.DataFrame, metric: str, ylabel: str, reference: float | None = None) -> None:
    xs = np.arange(len(DIST))
    for i, (rep, (color, marker)) in enumerate(REP_STYLE.items()):
        m, lo, hi = _curve(frame[frame["representation"] == rep], metric)
        ax.errorbar(
            xs + (i - 1) * 0.09,
            m,
            yerr=[m - lo, hi - m],
            fmt=marker + "-",
            color=color,
            ms=3.5,
            lw=1,
            capsize=2,
            elinewidth=0.8,
            label=rep,
        )
    if reference is not None:
        ax.axhline(reference, color="#c8c8c8", lw=0.7, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(DIST_LABEL, fontsize=6.5)
    ax.set_xlim(-0.45, len(DIST) - 0.55)
    ax.set_ylabel(ylabel, fontsize=8)
    # Panels carry different metrics on different scales; a shared limit hid one
    # of them entirely in an earlier version.
    ax.margins(y=0.28)


def fig_ladder(pairs: pd.DataFrame, model: str, out: Path) -> None:
    """Two panels for one model family: discrimination (AUC) and a fixed threshold (F1)."""
    frame, n_targets = paired(pairs)
    frame = frame[frame["model"] == model]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.5), dpi=220)
    _panel(axes[0], frame, "auc", "ROC-AUC", reference=0.5)
    _panel(axes[1], frame, "recall_at_flag20", "recall of failures at a 20% flag budget", reference=0.20)
    axes[0].set_title("(a) discrimination", fontsize=8, loc="left")
    axes[1].set_title("(b) decision quality at a fixed flag budget", fontsize=8, loc="left")
    axes[1].legend(frameon=False, title="representation", fontsize=7, title_fontsize=7, loc="best")
    fig.suptitle(
        f"{MODEL_TITLE.get(model, model)} - {n_targets} target cohorts present at all four distances",
        fontsize=8,
        y=1.02,
    )
    fig.tight_layout()
    for ext in (".png", ".svg"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def fig_explanations(pairs: pd.DataFrame, out: Path) -> None:
    """Explanation agreement by distance - the claim the AUC panels cannot show."""
    frame, _ = paired(pairs)
    frame = frame[frame["model"] == "gradient_boosting"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=220)
    xs = np.arange(len(DIST))
    for metric, label, color, marker in (
        ("explanation_tau", "Kendall tau", "#7a3b2e", "o"),
        ("explanation_jaccard_top3", "top-3 Jaccard", "#1f5f7a", "s"),
    ):
        m, lo, hi = _curve(frame, metric)
        ax.errorbar(xs, m, yerr=[m - lo, hi - m], fmt=marker + "-", color=color, ms=3.5, lw=1, capsize=2, label=label)
    ax.axhline(0.0, color="#c8c8c8", lw=0.7, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(DIST_LABEL, fontsize=6.5)
    ax.set_xlim(-0.45, len(DIST) - 0.55)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("agreement with the local model", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in (".png", ".svg"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def fig_fewshot(few: pd.DataFrame, out: Path) -> None:
    """AUC against the number of labelled target students, per target institution."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=220)
    colors = {
        "OULAD": "#1f5f7a",
        "KU Leuven": "#b7791f",
        "UKZN": "#7a3b2e",
        "Oviedo": "#4a7a3b",
        "Zambia": "#6b4a7a",
    }
    for inst, g in few.groupby("target_institution"):
        curve = g.groupby("k")["auc"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            curve["k"],
            curve["mean"],
            yerr=curve["std"],
            fmt="o-",
            ms=3.5,
            lw=1,
            capsize=2,
            color=colors.get(inst, "k"),
            label=inst,
        )
    ax.set_xlabel("labelled students from the target cohort (k)", fontsize=8)
    ax.set_ylabel("ROC-AUC on the remaining students", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    for ext in (".png", ".svg"):
        fig.savefig(out.with_suffix(ext), bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(pairs: pd.DataFrame, representation: str, model: str, out: Path) -> None:
    """Source x target AUC matrix, cohorts grouped by institution (supplementary)."""
    g = pairs[(pairs["representation"] == representation) & (pairs["model"] == model)]
    order = (
        g[["target", "target_institution"]]
        .drop_duplicates()
        .sort_values(["target_institution", "target"])["target"]
        .tolist()
    )
    mat = g.pivot_table(index="source", columns="target", values="auc").reindex(index=order, columns=order)
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=200)
    im = ax.imshow(mat.to_numpy(), vmin=0.4, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=4.5)
    ax.set_yticklabels(order, fontsize=4.5)
    ax.set_xlabel("target cohort")
    ax.set_ylabel("source cohort")
    fig.colorbar(im, ax=ax, fraction=0.03, label="ROC-AUC")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="f33")
    args = ap.parse_args()
    d = REPO / "data" / "artifacts" / "experiments" / "exp_014_transfer_ladder" / args.run
    FIG.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(d / "pairs.csv")

    for model, slug in (("gradient_boosting", "gbm"), ("logistic_regression", "lr"), ("random_forest", "rf")):
        if (pairs["model"] == model).any():
            fig_ladder(pairs, model, FIG / f"fig2_ladder_{slug}_{args.run}")
    fig_explanations(pairs, FIG / f"fig3_explanations_{args.run}")
    for rep in ("raw", "percentile"):
        fig_heatmap(pairs, rep, "gradient_boosting", FIG / f"figS_heatmap_{rep}_{args.run}")
    if (d / "fewshot.csv").exists():
        fig_fewshot(pd.read_csv(d / "fewshot.csv"), FIG / f"fig4_fewshot_{args.run}")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
