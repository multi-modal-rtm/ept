"""Writes results/summary.csv: one row per condition, mean +/- std over 5
seeds, macro-F1 + accuracy + per-class F1, plus standing rows for mask-only
(already one of the 7 conditions) and the trivial-feature probe. Reads only
the metrics.json files scripts/meld_phase4_test_eval.py already wrote --
no data touched by this script.
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

REPO_ROOT = "/home/devops/ept"
SEEDS = [42, 1337, 2024, 7, 31337]
CONDITIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "mask_only"]
LOCKED_RECIPE = {
    "A0": "r08", "A1": "r07", "A2": "r03", "A3": "r04",
    "A4": "r03", "A5": "r07", "mask_only": "r05",
}


def load_condition_metrics(condition):
    recipe_name = LOCKED_RECIPE[condition]
    rows = []
    for seed in SEEDS:
        run_id = f"meld_test_{condition}_{recipe_name}_seed{seed}"
        path = os.path.join(REPO_ROOT, "outputs", run_id, "metrics.json")
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def mean_std(values):
    return float(np.mean(values)), float(np.std(values))


def summarize_condition(condition):
    rows = load_condition_metrics(condition)
    macro_f1 = [r["test_macro_f1"] for r in rows]
    accuracy = [r["test_accuracy"] for r in rows]
    f1_neg = [r["test_per_class_f1"]["negative"] for r in rows]
    f1_neu = [r["test_per_class_f1"]["neutral"] for r in rows]
    f1_pos = [r["test_per_class_f1"]["positive"] for r in rows]
    m_f1, s_f1 = mean_std(macro_f1)
    m_acc, s_acc = mean_std(accuracy)
    m_neg, s_neg = mean_std(f1_neg)
    m_neu, s_neu = mean_std(f1_neu)
    m_pos, s_pos = mean_std(f1_pos)
    return {
        "condition": condition, "recipe": LOCKED_RECIPE[condition], "n_seeds": len(rows),
        "macro_f1_mean": m_f1, "macro_f1_std": s_f1,
        "accuracy_mean": m_acc, "accuracy_std": s_acc,
        "f1_negative_mean": m_neg, "f1_negative_std": s_neg,
        "f1_neutral_mean": m_neu, "f1_neutral_std": s_neu,
        "f1_positive_mean": m_pos, "f1_positive_std": s_pos,
    }


def summarize_trivial_probe():
    path = os.path.join(REPO_ROOT, "outputs", "meld_test_trivial_probe.json")
    with open(path) as f:
        tp = json.load(f)
    rows = tp["per_seed"]
    macro_f1 = [r["test_macro_f1"] for r in rows]
    accuracy = [r["test_accuracy"] for r in rows]
    f1_neg = [r["test_per_class_f1"][0] for r in rows]
    f1_neu = [r["test_per_class_f1"][1] for r in rows]
    f1_pos = [r["test_per_class_f1"][2] for r in rows]
    m_f1, s_f1 = mean_std(macro_f1)
    m_acc, s_acc = mean_std(accuracy)
    m_neg, s_neg = mean_std(f1_neg)
    m_neu, s_neu = mean_std(f1_neu)
    m_pos, s_pos = mean_std(f1_pos)
    return {
        "condition": "trivial_feature_probe", "recipe": "n/a (not searched)", "n_seeds": len(rows),
        "macro_f1_mean": m_f1, "macro_f1_std": s_f1,
        "accuracy_mean": m_acc, "accuracy_std": s_acc,
        "f1_negative_mean": m_neg, "f1_negative_std": s_neg,
        "f1_neutral_mean": m_neu, "f1_neutral_std": s_neu,
        "f1_positive_mean": m_pos, "f1_positive_std": s_pos,
    }


def main():
    fieldnames = ["condition", "recipe", "n_seeds",
                  "macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std",
                  "f1_negative_mean", "f1_negative_std", "f1_neutral_mean", "f1_neutral_std",
                  "f1_positive_mean", "f1_positive_std"]
    rows = [summarize_condition(c) for c in CONDITIONS]
    rows.append(summarize_trivial_probe())

    out_path = os.path.join(REPO_ROOT, "results", "summary.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"saved -> {out_path}")
    for r in rows:
        print(f"  {r['condition']}: macro_f1={r['macro_f1_mean']:.4f}+/-{r['macro_f1_std']:.4f} "
              f"acc={r['accuracy_mean']:.4f}+/-{r['accuracy_std']:.4f}")


if __name__ == "__main__":
    main()
