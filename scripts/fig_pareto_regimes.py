"""Figure 3 (regenerated again): one panel per internally-consistent
hardware regime -- no panel ever mixes GPU and CPU numbers for model vs
detection. Reads outputs/meld_phase6_regimes.json + per-condition latency
already on disk (results/efficiency.csv, results/summary.csv); no data
touched, no retraining.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_ROOT = "/home/devops/ept"
OUT_PATH = os.path.join(REPO_ROOT, "paper", "figures", "fig3_pareto_gflops.pdf")
TRIVIAL_PROBE_MACRO_F1 = 0.3648
CONDITIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "mask_only"]
LABEL_OFFSET = {
    "A0": (4, -2), "A1": (5, 4), "A2": (5, -8), "A3": (-15, 6),
    "A4": (5, 4), "A5": (-15, -8), "mask_only": (5, 0),
}


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_json(name):
    with open(os.path.join(REPO_ROOT, "outputs", name)) as f:
        return json.load(f)


def panel(ax, title, cond_latency_ms, summary, trivial_latency_ms=None, e1s2=None, unavailable_reason=None):
    if unavailable_reason:
        ax.text(0.5, 0.5, "not available:\n" + unavailable_reason, ha="center", va="center",
                fontsize=4.6, wrap=True, transform=ax.transAxes, color="#8c1f1f")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=6.2)
        return

    for c in CONDITIONS:
        if c not in cond_latency_ms:
            continue
        x = cond_latency_ms[c]
        y = float(summary[c]["macro_f1_mean"])
        yerr = float(summary[c]["macro_f1_std"])
        marker = "*" if c == "A1" else "o"
        ms = 6.5 if c == "A1" else 4
        color = "#1f77b4" if c != "mask_only" else "#7f7f7f"
        ax.errorbar(x, y, yerr=yerr, fmt=marker, ms=ms, color=color,
                    ecolor="gray", elinewidth=0.4, capsize=1.2, zorder=5)
        ax.annotate(c, (x, y), fontsize=4.2, xytext=LABEL_OFFSET[c], textcoords="offset points")

    if trivial_latency_ms is not None:
        ax.axhline(TRIVIAL_PROBE_MACRO_F1, color="black", lw=0.5, ls=":", zorder=1)
        ax.scatter([trivial_latency_ms], [TRIVIAL_PROBE_MACRO_F1], marker="P", s=26, color="black", zorder=6)
        ax.annotate("trivial", (trivial_latency_ms, TRIVIAL_PROBE_MACRO_F1), fontsize=4.0,
                    xytext=(3, 3), textcoords="offset points")

    if e1s2 is not None:
        ex, ey = e1s2
        ax.scatter([ex], [ey], marker="D", s=24, facecolors="#d62728", edgecolors="#d62728", zorder=6)
        ax.annotate("E=1,S=2", (ex, ey), fontsize=4.0, color="#d62728",
                    xytext=(3, -8), textcoords="offset points")

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation(base=10))
    ax.tick_params(labelsize=4.8)
    ax.set_title(title, fontsize=6.2)


def main():
    regimes = load_json("meld_phase6_regimes.json")["regimes"]
    eff = {r["condition"]: r for r in load_csv(os.path.join(REPO_ROOT, "results", "efficiency.csv"))}
    summary = {r["condition"]: r for r in load_csv(os.path.join(REPO_ROOT, "results", "summary.csv"))}
    eff_combined = load_json("meld_phase6_efficiency_combined.json")
    budget_by_es = {(r["e"], r["s"]): r for r in eff_combined["budget_sweep_rows"]}
    trivial_lat = load_json("meld_phase6_trivial_probe_latency.json")["latency_ms"]
    single_stream_det = eff_combined["detection_clustering_latency"]["total_ms_mean"]

    import json as _json
    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase4_budget_sweep_summary.json")) as f:
        budget_acc_raw = _json.load(f)
    from collections import defaultdict
    by_point = defaultdict(list)
    for r in budget_acc_raw["results"]:
        by_point[(int(r["e"]), int(r["s"]))].append(r["test_macro_f1"])
    e1s2_acc = sum(by_point[(1, 2)]) / len(by_point[(1, 2)])

    fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.4))

    # Panel A: GPU-only
    a_reg = regimes.get("A_gpu_only", {})
    if a_reg.get("available"):
        det_ms = a_reg["detection_gpu_ms_per_clip"]
        cond_lat = {c: float(eff[c]["gpu_end_to_end_latency_ms"]) + (0.0 if c == "A0" else det_ms)
                    for c in CONDITIONS}
        e1s2_lat = budget_by_es[(1, 2)]["gpu_end_to_end_latency_ms"] + det_ms
        panel(axes[0, 0], f"A: GPU-only (det. {det_ms:.1f}ms)", cond_lat, summary,
              trivial_latency_ms=trivial_lat, e1s2=(e1s2_lat, e1s2_acc))
    else:
        panel(axes[0, 0], "A: GPU-only", {}, summary, unavailable_reason=a_reg.get("reason", "not attempted"))

    # Panel B1: CPU-only, threads=1
    b1 = regimes.get("B_cpu_only_threads1")
    if b1:
        cond_lat = {c: float(eff[c]["cpu_end_to_end_ms_threads1"]) + (0.0 if c == "A0" else b1["detection_cpu_ms_per_clip"])
                    for c in CONDITIONS}
        panel(axes[0, 1], "B1: CPU-only, 1 thread", cond_lat, summary, trivial_latency_ms=None)
    else:
        panel(axes[0, 1], "B1: CPU-only, 1 thread", {}, summary, unavailable_reason="not computed")

    # Panel B2: CPU-only, threads=60
    b2 = regimes.get("B_cpu_only_threads60")
    if b2:
        cond_lat = {c: float(eff[c]["cpu_end_to_end_ms_threads60"]) + (0.0 if c == "A0" else b2["detection_cpu_ms_per_clip"])
                    for c in CONDITIONS}
        panel(axes[1, 0], "B2: CPU-only, 60 threads*", cond_lat, summary, trivial_latency_ms=None)
    else:
        panel(axes[1, 0], "B2: CPU-only, 60 threads", {}, summary, unavailable_reason="not computed")

    # Panel C: realistic deployment
    c_reg = regimes.get("C_realistic_deployment")
    if c_reg:
        # Same 60-way-throughput detection cost applied to every non-A0 condition (A0 needs no
        # detector regardless of regime), matching how A1's own C-regime total was derived.
        det_delta = c_reg["A1_e2e_ms"] - float(eff["A1"]["gpu_end_to_end_latency_ms"])
        cond_lat = {c: float(eff[c]["gpu_end_to_end_latency_ms"]) + (0.0 if c == "A0" else det_delta)
                    for c in CONDITIONS}
        e1s2_lat = budget_by_es[(1, 2)]["gpu_end_to_end_latency_ms"] + det_delta
        panel(axes[1, 1], "C: realistic (GPU model +\nCPU 60x-throughput det.)", cond_lat, summary,
              trivial_latency_ms=trivial_lat, e1s2=(e1s2_lat, e1s2_acc))
    else:
        panel(axes[1, 1], "C: realistic deployment", {}, summary, unavailable_reason="not computed")

    fig.supxlabel("latency / clip, ms (log scale)", fontsize=6)
    fig.supylabel("test macro-F1 (mean, 5 seeds)", fontsize=6)
    fig.suptitle("A0/A1 cost ranking by hardware regime (never mixed within a panel)", fontsize=6.8)
    fig.text(0.5, -0.01,
              "*B2 model: intra-op 60-thread call; B2 detection: 60-way multi-process throughput --\n"
              "different parallelism mechanisms, both spending a 60-CPU-thread budget (see outputs/meld_phase6_regimes.json).",
              ha="center", va="top", fontsize=3.8, style="italic")
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.94])
    fig.savefig(OUT_PATH, bbox_inches="tight")
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
