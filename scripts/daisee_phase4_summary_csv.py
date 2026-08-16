"""Writes results/summary_daisee.csv: one row per condition, mean+/-std over
5 seeds. primary_macro_f1_classes1to3 is the primary metric (class 0
excluded from this average per docs/DECISION_RULES_DAISEE.md's 2026-08-16
amendment -- see the class0_* columns for its own count and raw F1,
reported unconditionally, never silently dropped). Reads only
outputs/daisee_phase4_test_summary.json, already on disk -- no data touched.
"""
import csv
import json
import os

import numpy as np

REPO_ROOT = "/home/devops/ept"
CONDITIONS = ["A0", "A1"]


def main():
    with open(os.path.join(REPO_ROOT, "outputs", "daisee_phase4_test_summary.json")) as f:
        d = json.load(f)
    by_cond = {}
    for r in d["results"]:
        by_cond.setdefault(r["condition"], []).append(r)

    fieldnames = [
        "condition", "recipe", "n_seeds",
        "primary_macro_f1_classes1to3_mean", "primary_macro_f1_classes1to3_std",
        "NOTE_class0_excluded_from_primary",
        "macro_f1_4class_secondary_mean", "macro_f1_4class_secondary_std",
        "accuracy_mean", "accuracy_std",
        "class0_raw_f1_mean", "class0_raw_f1_std", "class0_n_test_items",
        "class1_f1_mean", "class1_f1_std",
        "class2_f1_mean", "class2_f1_std",
        "class3_f1_mean", "class3_f1_std",
    ]
    rows = []
    for cond in CONDITIONS:
        r = by_cond[cond]
        recipe = r[0]["recipe_id"]

        def ms(key):
            vals = [x[key] for x in r]
            return float(np.mean(vals)), float(np.std(vals))

        def ms_class(name):
            vals = [x["per_class_f1"][name] for x in r]
            return float(np.mean(vals)), float(np.std(vals))

        p_m, p_s = ms("primary_macro_f1_classes1to3")
        f4_m, f4_s = ms("macro_f1_4class_secondary")
        acc_m, acc_s = ms("accuracy")
        c0_m, c0_s = ms("class0_raw_f1")
        c1_m, c1_s = ms_class("1_low")
        c2_m, c2_s = ms_class("2_high")
        c3_m, c3_s = ms_class("3_veryhigh")

        rows.append({
            "condition": cond, "recipe": recipe, "n_seeds": len(r),
            "primary_macro_f1_classes1to3_mean": p_m, "primary_macro_f1_classes1to3_std": p_s,
            "NOTE_class0_excluded_from_primary": "class 0 (4 test clips) is excluded from the primary "
                                                  "macro-F1 average; see class0_raw_f1_mean/class0_n_test_items",
            "macro_f1_4class_secondary_mean": f4_m, "macro_f1_4class_secondary_std": f4_s,
            "accuracy_mean": acc_m, "accuracy_std": acc_s,
            "class0_raw_f1_mean": c0_m, "class0_raw_f1_std": c0_s, "class0_n_test_items": 4,
            "class1_f1_mean": c1_m, "class1_f1_std": c1_s,
            "class2_f1_mean": c2_m, "class2_f1_std": c2_s,
            "class3_f1_mean": c3_m, "class3_f1_std": c3_s,
        })

    out_path = os.path.join(REPO_ROOT, "results", "summary_daisee.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"saved -> {out_path}")
    for row in rows:
        print(f"  {row['condition']}: primary(1-3)={row['primary_macro_f1_classes1to3_mean']:.4f}+/-"
              f"{row['primary_macro_f1_classes1to3_std']:.4f} "
              f"4class={row['macro_f1_4class_secondary_mean']:.4f} acc={row['accuracy_mean']:.4f} "
              f"class0_f1={row['class0_raw_f1_mean']:.4f} (n=4, excluded from primary)")


if __name__ == "__main__":
    main()
