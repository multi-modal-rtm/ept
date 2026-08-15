"""Figure 4: per-seed margins for H1 (A1-A2) and H2 (A1-A0), showing the
item-paired bootstrap CI vs. the seed-paired CI side by side -- the same
five points diverge sharply for H1 (one outlier seed dominates the
between-seed variance) and agree for H2 (no outlier). Reads only
outputs/meld_phase4_statistics.json and outputs/meld_phase4_posthoc_analysis.json,
both already on disk; no data touched.
"""
import json
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = "/home/devops/ept"
OUT_PATH = os.path.join(REPO_ROOT, "paper", "figures", "fig4_seed_margin_divergence.pdf")
SEEDS = [42, 1337, 2024, 7, 31337]
EFFECT_FLOOR = 0.04


def load():
    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase4_statistics.json")) as f:
        stats = json.load(f)
    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase4_posthoc_analysis.json")) as f:
        posthoc = json.load(f)
    return stats, posthoc


def panel(ax, title, item_result, seed_result, mark_outlier_idx=None):
    margins = np.array(seed_result["per_seed_margin"])
    x = np.arange(len(SEEDS))
    colors = ["#1f77b4"] * len(SEEDS)
    if mark_outlier_idx is not None:
        colors[mark_outlier_idx] = "#d62728"
    ax.scatter(x, margins, c=colors, s=18, zorder=5, edgecolors="black", linewidths=0.4)

    item_lo, item_hi, item_mean = item_result["ci_low"], item_result["ci_high"], item_result["mean_margin"]
    seed_lo, seed_hi = seed_result["t_ci_low"], seed_result["t_ci_high"]

    off = 0.28
    ax.errorbar([-off], [item_mean], yerr=[[item_mean - item_lo], [item_hi - item_mean]],
                fmt="s", color="#2ca02c", markersize=4, capsize=3, capthick=0.8, lw=0.8,
                label="item-paired bootstrap")
    ax.errorbar([len(SEEDS) - 1 + off], [seed_result["mean"]],
                yerr=[[seed_result["mean"] - seed_lo], [seed_hi - seed_result["mean"]]],
                fmt="^", color="#9467bd", markersize=4, capsize=3, capthick=0.8, lw=0.8,
                label="seed-paired t")

    ax.axhline(0, color="black", lw=0.5, ls="-", zorder=1)
    ax.axhline(EFFECT_FLOOR, color="gray", lw=0.6, ls="--", zorder=1)
    ax.text(len(SEEDS) - 1 + off + 0.15, EFFECT_FLOOR, "floor=0.04", fontsize=4.5, va="center", color="gray")

    ax.set_xticks(list(range(len(SEEDS))))
    ax.set_xticklabels([str(s) for s in SEEDS], fontsize=4.3, rotation=90)
    ax.set_xlim(-0.8, len(SEEDS) - 1 + 0.8)
    ax.set_title(title, fontsize=6.5)
    ax.tick_params(axis="y", labelsize=5)
    ax.set_ylabel("macro-F1 margin", fontsize=5.5)


def main():
    stats, posthoc = load()
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9), sharey=False)

    panel(axes[0], "H1: A1 - A2", stats["h1_persistence"], posthoc["seed_paired_h1"], mark_outlier_idx=4)
    panel(axes[1], "H2: A1 - A0", stats["h2_entity_structure"], posthoc["seed_paired_h2"])

    axes[0].set_xlabel("seed", fontsize=5.5)
    axes[1].set_xlabel("seed", fontsize=5.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=5, frameon=False,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Per-seed margins: item-paired bootstrap vs. seed-paired CI", fontsize=6.5, y=0.995)
    fig.text(0.5, -0.10,
              "Red point = seed 31337 (A2 training collapse). Item bootstrap holds the 5 trained\n"
              "models fixed and resamples test items; seed-paired treats the 5 seeds as the sample.",
              ha="center", va="top", fontsize=4.5, style="italic")
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
