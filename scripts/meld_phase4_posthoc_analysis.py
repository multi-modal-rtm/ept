"""POST HOC sensitivity/diagnostic analysis of the Phase 4 MELD test results.
Every number in this file is computed from data already on disk before this
script ran -- no test evaluation is repeated. The pre-registered result
(5-seed mean margin, item-paired bootstrap, docs/DECISION_RULES.md's decision
rule) is NOT revised by anything here; this is diagnostic re-analysis, run
and labeled as such.

1. Leave-one-seed-out sensitivity for H1/H2.
2. Seed-paired statistics (paired t-test, Wilcoxon signed-rank) alongside the
   item-paired bootstrap, to show what each is/isn't sensitive to.
3. (partial -- collapse scan across all conditions/seeds using train_history
   already saved; the dev-diagnostic re-run for A2 seed 31337 is a separate
   script, since it required one new DEV-only training run, never touching
   test.)
4. Recipe confound: A1/A2 dev scores under both r03 and r07, read from the
   EXISTING Phase 3 search outputs (single seed=42, already on disk).
5. Budget-sweep-vs-trivial-probe: seed-level (not item-level, since budget
   sweep runs never saved per-item predictions) one-sample comparison against
   the trivial probe's fixed point estimate.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy import stats as scipy_stats

from meld_phase4_statistics import SEEDS, paired_bootstrap_margin

REPO_ROOT = "/home/devops/ept"
LOCKED_RECIPE = {
    "A0": "r08", "A1": "r07", "A2": "r03", "A3": "r04",
    "A4": "r03", "A5": "r07", "mask_only": "r05",
}
TRIVIAL_PROBE = 0.3648  # results/summary.csv, 5-seed mean (std 0.0000)
E_GRID = [1, 2, 4, 6, 8]
S_GRID = [2, 4]
BUDGET_RECIPE = "r07"


# ---------- 1. Leave-one-seed-out ----------

def leave_one_out(condition_a, condition_b):
    rows = []
    for drop in SEEDS:
        remaining = [s for s in SEEDS if s != drop]
        r = paired_bootstrap_margin(condition_a, condition_b, n_bootstrap=10000, seeds=remaining)
        rows.append({"dropped_seed": drop, "remaining_seeds": remaining,
                      "mean_margin": r["mean_margin"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    return rows


# ---------- 2. Seed-paired statistics ----------

def seed_paired_stats(condition_a, condition_b):
    r = paired_bootstrap_margin(condition_a, condition_b, n_bootstrap=1)  # only need per_seed_margin
    margins = np.array(r["per_seed_margin"])
    n = len(margins)
    mean = margins.mean()
    sem = margins.std(ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    ci_low, ci_high = mean - t_crit * sem, mean + t_crit * sem

    t_stat, t_p = scipy_stats.ttest_1samp(margins, popmean=0.0)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(margins)
    except ValueError as e:
        w_stat, w_p = None, None  # e.g. all-identical-sign edge cases with n=5

    return {
        "per_seed_margin": margins.tolist(), "n": n, "mean": float(mean),
        "t_ci_low": float(ci_low), "t_ci_high": float(ci_high),
        "paired_t_stat": float(t_stat), "paired_t_p": float(t_p),
        "wilcoxon_stat": (float(w_stat) if w_stat is not None else None),
        "wilcoxon_p": (float(w_p) if w_p is not None else None),
        "item_bootstrap_mean": r["mean_margin"],  # sanity cross-check, ignore r's n_bootstrap=1 CI
    }


# ---------- 4. Recipe confound ----------

def recipe_confound():
    out = {}
    for condition in ("A1", "A2"):
        out[condition] = {}
        for recipe_id in ("r03", "r07"):
            run_id = f"meld_{condition}_{recipe_id}_seed42"
            path = os.path.join(REPO_ROOT, "outputs", run_id, "metrics.json")
            with open(path) as f:
                m = json.load(f)
            out[condition][recipe_id] = {
                "best_val_macro_f1": m["best_val_macro_f1"], "best_epoch": m["best_epoch"],
                "final_val_macro_f1": m["final_val_macro_f1"],
            }
    return out


# ---------- 5. Budget sweep vs trivial probe ----------

def budget_vs_trivial_probe():
    results = []
    for e in E_GRID:
        for s in S_GRID:
            macro_f1s = []
            for seed in SEEDS:
                run_id = f"meld_test_budget_e{e}_s{s}_{BUDGET_RECIPE}_seed{seed}"
                path = os.path.join(REPO_ROOT, "outputs", run_id, "metrics.json")
                with open(path) as f:
                    m = json.load(f)
                macro_f1s.append(m["test_macro_f1"])
            margins = np.array(macro_f1s) - TRIVIAL_PROBE
            n = len(margins)
            mean_margin = margins.mean()
            sem = margins.std(ddof=1) / np.sqrt(n)
            t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
            ci_low, ci_high = mean_margin - t_crit * sem, mean_margin + t_crit * sem
            t_stat, t_p = scipy_stats.ttest_1samp(macro_f1s, popmean=TRIVIAL_PROBE)
            results.append({
                "e": e, "s": s, "tokens": e * s, "mean_macro_f1": float(np.mean(macro_f1s)),
                "mean_margin_vs_probe": float(mean_margin), "ci_low": float(ci_low), "ci_high": float(ci_high),
                "excludes_probe": bool(ci_low > 0), "t_stat": float(t_stat), "t_p": float(t_p),
            })
    return results


def main():
    print("=" * 70)
    print("1. LEAVE-ONE-SEED-OUT SENSITIVITY (post hoc)")
    print("=" * 70)
    loo_h1 = leave_one_out("A1", "A2")
    loo_h2 = leave_one_out("A1", "A0")
    print("\nH1 (A1-A2):")
    for r in loo_h1:
        print(f"  drop seed {r['dropped_seed']:>6}: margin={r['mean_margin']:.4f} "
              f"95% CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print("\nH2 (A1-A0):")
    for r in loo_h2:
        print(f"  drop seed {r['dropped_seed']:>6}: margin={r['mean_margin']:.4f} "
              f"95% CI=[{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
    print("\nPre-registered result is the 5-seed number and it stands. "
          "This is sensitivity analysis, not a revision.")

    print("\n" + "=" * 70)
    print("2. SEED-PAIRED STATISTICS (post hoc)")
    print("=" * 70)
    sp_h1 = seed_paired_stats("A1", "A2")
    sp_h2 = seed_paired_stats("A1", "A0")
    print("\nH1 (A1-A2):")
    print(json.dumps(sp_h1, indent=2))
    print("\nH2 (A1-A0):")
    print(json.dumps(sp_h2, indent=2))

    print("\n" + "=" * 70)
    print("4. RECIPE CONFOUND (existing Phase 3 dev-search results, seed=42 only)")
    print("=" * 70)
    rc = recipe_confound()
    print(json.dumps(rc, indent=2))

    print("\n" + "=" * 70)
    print("5. BUDGET SWEEP vs TRIVIAL PROBE (seed-level, n=5 per cell)")
    print("=" * 70)
    bv = budget_vs_trivial_probe()
    for r in bv:
        print(f"  E={r['e']} S={r['s']} tokens={r['tokens']:2d}: mean_f1={r['mean_macro_f1']:.4f} "
              f"margin_vs_probe={r['mean_margin_vs_probe']:.4f} 95%CI=[{r['ci_low']:.4f},{r['ci_high']:.4f}] "
              f"excludes_probe={r['excludes_probe']} p={r['t_p']:.4g}")
    excludes = [r for r in bv if r["excludes_probe"]]
    if excludes:
        smallest = min(excludes, key=lambda r: r["tokens"])
        print(f"\nSmallest (E,S) by token count whose CI excludes the trivial probe: "
              f"E={smallest['e']}, S={smallest['s']} (tokens={smallest['tokens']})")
    else:
        print("\nNo (E,S) cell's CI excludes the trivial probe.")

    out = {"leave_one_out_h1": loo_h1, "leave_one_out_h2": loo_h2,
           "seed_paired_h1": sp_h1, "seed_paired_h2": sp_h2,
           "recipe_confound": rc, "budget_vs_trivial_probe": bv,
           "smallest_excluding_probe": (min(excludes, key=lambda r: r["tokens"]) if excludes else None)}
    out_path = os.path.join(REPO_ROOT, "outputs", "meld_phase4_posthoc_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
