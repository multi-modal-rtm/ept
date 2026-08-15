"""Figure 3: accuracy (test macro-F1) vs. total model GFLOPs/clip Pareto,
E=1,S=2 and the trivial-feature probe both marked. Reads only
results/summary.csv, results/efficiency.csv, results/efficiency_budget_sweep.csv,
and outputs/meld_phase4_budget_sweep_summary.json -- all already on disk.
GFLOPs axis is backbone+attention-stack total (results/efficiency.csv's
total_model_gflops_per_clip), NOT including detection+clustering (reported
separately per docs/PLAN.md Sec.6 -- see results/efficiency.csv's own
detection_clustering_ms_per_clip column).
"""
import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

REPO_ROOT = "/home/devops/ept"
OUT_PATH = os.path.join(REPO_ROOT, "paper", "figures", "fig3_pareto_gflops.pdf")
TRIVIAL_PROBE_MACRO_F1 = 0.3648
TRIVIAL_PROBE_GFLOPS = 44.6072 * 8  # 8-frame mean-pooled backbone pass, no attention stack at all


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    summary = {r["condition"]: r for r in load_csv(os.path.join(REPO_ROOT, "results", "summary.csv"))}
    efficiency = {r["condition"]: r for r in load_csv(os.path.join(REPO_ROOT, "results", "efficiency.csv"))}
    budget_eff = load_csv(os.path.join(REPO_ROOT, "results", "efficiency_budget_sweep.csv"))

    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase4_budget_sweep_summary.json")) as f:
        budget_acc_raw = json.load(f)
    by_point = defaultdict(list)
    for r in budget_acc_raw["results"]:
        by_point[(int(r["e"]), int(r["s"]))].append(r["test_macro_f1"])
    budget_acc = {k: sum(v) / len(v) for k, v in by_point.items()}

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    conditions = ["A0", "A1", "A2", "A3", "A4", "A5", "mask_only"]
    # Manual per-condition label offsets (points): A1/A3/A4 sit at nearly
    # identical GFLOPs (backbone cost is what dominates, and it's the same
    # E=6,S=4 for all of them) with close-together accuracy too, so default
    # placement collides -- staggered by hand instead of relying on autolayout.
    label_offset = {
        "A0": (4, -2), "A1": (6, 4), "A2": (6, -8), "A3": (-16, 6),
        "A4": (6, 4), "A5": (-16, -8), "mask_only": (6, 0),
    }
    for c in conditions:
        x = float(efficiency[c]["total_model_gflops_per_clip"])
        y = float(summary[c]["macro_f1_mean"])
        yerr = float(summary[c]["macro_f1_std"])
        marker = "*" if c == "A1" else "o"
        ax.errorbar(x, y, yerr=yerr, fmt=marker, ms=7 if c == "A1" else 4.5,
                    color="#1f77b4" if c != "mask_only" else "#7f7f7f",
                    ecolor="gray", elinewidth=0.5, capsize=1.5, zorder=5)
        ax.annotate(c, (x, y), fontsize=4.8, xytext=label_offset[c], textcoords="offset points")

    bx, by_ = [], []
    e1s2 = None
    for r in budget_eff:
        e, s = int(r["e"]), int(r["s"])
        x = float(r["total_model_gflops_per_clip"])
        y = budget_acc[(e, s)]
        bx.append(x)
        by_.append(y)
        if e == 1 and s == 2:
            e1s2 = (x, y)
    ax.scatter(bx, by_, marker="x", s=12, color="#ff7f0e", zorder=4, label="budget sweep (A1 arch.)")

    if e1s2:
        ax.scatter(*e1s2, marker="D", s=32, facecolors="none", edgecolors="#d62728",
                   linewidths=1.1, zorder=6)
        ax.annotate("E=1,S=2", e1s2, fontsize=4.8, color="#d62728",
                    xytext=(4, -8), textcoords="offset points")

    ax.axhline(TRIVIAL_PROBE_MACRO_F1, color="black", lw=0.6, ls=":", zorder=1)
    ax.scatter([TRIVIAL_PROBE_GFLOPS], [TRIVIAL_PROBE_MACRO_F1], marker="P", s=34,
               color="black", zorder=6)
    ax.annotate("trivial probe", (TRIVIAL_PROBE_GFLOPS, TRIVIAL_PROBE_MACRO_F1), fontsize=4.8,
                xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel("total model GFLOPs / clip (backbone + attention stack, log scale)", fontsize=6)
    ax.set_ylabel("test macro-F1 (mean $\\pm$ std, 5 seeds)", fontsize=6)
    ax.tick_params(labelsize=5.5)
    ax.set_title("Accuracy vs. compute (detection cost excluded -- see text)", fontsize=6.5)

    handles = [
        plt.Line2D([], [], marker="*", ls="", color="#1f77b4", ms=7, label="A1 (primary)"),
        plt.Line2D([], [], marker="o", ls="", color="#1f77b4", ms=4.5, label="A0,A2-A5"),
        plt.Line2D([], [], marker="o", ls="", color="#7f7f7f", ms=4.5, label="mask-only"),
        plt.Line2D([], [], marker="x", ls="", color="#ff7f0e", ms=4.5, label="budget sweep"),
        plt.Line2D([], [], marker="P", ls="", color="black", ms=5.5, label="trivial probe"),
    ]
    ax.legend(handles=handles, fontsize=4.5, loc="lower right", frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
