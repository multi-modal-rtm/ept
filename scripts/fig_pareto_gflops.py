"""Figure 3 (regenerated): accuracy vs. LATENCY, two curves on shared axes --
model-only (backbone+attention stack GPU latency) and end-to-end-with-
detection (+ single-stream detection+clustering, 6276.9ms, or 0 for A0/the
trivial probe, neither of which needs a detector). GFLOPs alone can't be the
shared axis here since detection's cost was only ever measured in wall-clock
time, not FLOPs -- switching to latency (ms) is what lets both curves share
one axis honestly. The divergence between the two curves -- not either curve
alone -- is the figure's point: model-only ranks A1 as cheaper than A0;
end-to-end reverses that ranking entirely, by two orders of magnitude.

Reads only results/summary.csv, results/efficiency_breakdown.csv, and
outputs/meld_phase4_budget_sweep_summary.json/meld_phase6_e2e_breakdown.json
-- all already on disk; no data touched.
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


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    summary = {r["condition"]: r for r in load_csv(os.path.join(REPO_ROOT, "results", "summary.csv"))}
    eff = {r["condition"]: r for r in load_csv(os.path.join(REPO_ROOT, "results", "efficiency_breakdown.csv"))}

    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase4_budget_sweep_summary.json")) as f:
        budget_acc_raw = json.load(f)
    by_point = defaultdict(list)
    for r in budget_acc_raw["results"]:
        by_point[(int(r["e"]), int(r["s"]))].append(r["test_macro_f1"])
    budget_acc = {k: sum(v) / len(v) for k, v in by_point.items()}

    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_efficiency_combined.json")) as f:
        eff_combined = json.load(f)
    budget_latency = {(r["e"], r["s"]): r["gpu_end_to_end_latency_ms"] for r in eff_combined["budget_sweep_rows"]}
    single_stream_det_ms = eff_combined["detection_clustering_latency"]["total_ms_mean"]

    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_trivial_probe_latency.json")) as f:
        trivial_latency_ms = json.load(f)["latency_ms"]

    fig, ax = plt.subplots(figsize=(3.3, 2.7))

    conditions = ["A0", "A1", "A2", "A3", "A4", "A5", "mask_only"]
    label_offset = {
        "A0": (4, -2), "A1": (6, 4), "A2": (6, -8), "A3": (-16, 6),
        "A4": (6, 4), "A5": (-16, -8), "mask_only": (6, 0),
    }
    for c in conditions:
        model_x = float(eff[c]["model_only_gpu_latency_ms"])
        e2e_x = float(eff[c]["e2e_single_stream_ms"])
        y = float(summary[c]["macro_f1_mean"])
        yerr = float(summary[c]["macro_f1_std"])
        marker = "*" if c == "A1" else "o"
        ms = 7 if c == "A1" else 4.5
        color = "#1f77b4" if c != "mask_only" else "#7f7f7f"
        ax.errorbar(model_x, y, yerr=yerr, fmt=marker, ms=ms, color=color,
                    ecolor="gray", elinewidth=0.5, capsize=1.5, zorder=5)
        ax.errorbar(e2e_x, y, yerr=yerr, fmt=marker, ms=ms, mfc="none", color=color,
                    ecolor="gray", elinewidth=0.5, capsize=1.5, zorder=5)
        ax.plot([model_x, e2e_x], [y, y], color=color, lw=0.5, ls=":", zorder=2, alpha=0.6)
        ax.annotate(c, (e2e_x, y), fontsize=4.8, xytext=label_offset[c], textcoords="offset points")

    bx_model, bx_e2e, by_ = [], [], []
    e1s2 = None
    for (e, s), acc in budget_acc.items():
        model_x = budget_latency[(e, s)]
        e2e_x = model_x + single_stream_det_ms
        bx_model.append(model_x)
        bx_e2e.append(e2e_x)
        by_.append(acc)
        if e == 1 and s == 2:
            e1s2 = (model_x, e2e_x, acc)
    ax.scatter(bx_model, by_, marker="x", s=10, color="#ff7f0e", zorder=4, label="budget sweep (model-only)")
    ax.scatter(bx_e2e, by_, marker="x", s=10, color="#ffbb78", zorder=4, label="budget sweep (end-to-end)")

    if e1s2:
        mx, ex, y = e1s2
        ax.scatter([mx], [y], marker="D", s=30, facecolors="none", edgecolors="#d62728", linewidths=1.1, zorder=6)
        ax.scatter([ex], [y], marker="D", s=30, facecolors="#d62728", edgecolors="#d62728", linewidths=1.1, zorder=6)
        ax.annotate("E=1,S=2", (ex, y), fontsize=4.8, color="#d62728", xytext=(4, -9), textcoords="offset points")

    ax.axhline(TRIVIAL_PROBE_MACRO_F1, color="black", lw=0.6, ls=":", zorder=1)
    ax.scatter([trivial_latency_ms], [TRIVIAL_PROBE_MACRO_F1], marker="P", s=34, color="black", zorder=6)
    ax.annotate("trivial probe\n(no detector needed)", (trivial_latency_ms, TRIVIAL_PROBE_MACRO_F1),
                fontsize=4.3, xytext=(4, 4), textcoords="offset points")

    ax.set_xscale("log")
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] * 2.2)
    ax.set_xlabel("GPU latency / clip, ms (log scale)", fontsize=6)
    ax.set_ylabel("test macro-F1 (mean $\\pm$ std, 5 seeds)", fontsize=6)
    ax.tick_params(labelsize=5.5)
    ax.set_title("Model-only vs. end-to-end latency: the ranking reverses", fontsize=6.3)

    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="#1f77b4", mec="#1f77b4", ms=4.5, label="model-only (filled)"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#1f77b4", ms=4.5, label="end-to-end (open)"),
        plt.Line2D([], [], marker="*", ls="", color="#1f77b4", ms=7, label="A1 (primary)"),
        plt.Line2D([], [], marker="P", ls="", color="black", ms=5.5, label="trivial probe"),
    ]
    ax.legend(handles=handles, fontsize=4.2, loc="lower right", frameon=True, framealpha=0.9)

    fig.text(0.5, -0.02,
              "Dotted segments join each condition's model-only and end-to-end latency.\n"
              "A1 end-to-end (detection included) costs ~161x its own model-only latency.",
              ha="center", va="top", fontsize=4.3, style="italic")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
