"""DAiSEE Phase 4 statistics: H2 (A1-A0) margin on the amended primary metric
(macro-F1, classes 1-3 only, docs/DECISION_RULES_DAISEE.md 2026-08-16
amendment), item-paired bootstrap (10,000 resamples) PLUS a seed-paired test
(paired t-test + Wilcoxon), computed together as first-class parts of this
gate, not as later sensitivity analysis. Reads only the predictions.json
files scripts/daisee_phase4_test_eval.py already wrote -- touches no data
itself.

Bootstrap design: identical to scripts/meld_phase4_statistics.py's -- each
resample draws test-item indices WITH REPLACEMENT (same resampled index set
applied to every seed and both conditions, since it's the same 1784 test
items throughout), macro-F1(classes 1-3) recomputed per seed on the
resampled items, then averaged across the 5 seeds. Both item- and seed-level
variance flow into the CI.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import f1_score

REPO_ROOT = "/home/devops/ept"
SEEDS = [42, 1337, 2024, 7, 31337]
LOCKED_RECIPE = {"A0": "r08", "A1": "r06"}
EFFECT_FLOOR = 0.03
N_BOOTSTRAP = 10000


def primary_metric(y, p):
    return f1_score(y, p, labels=[1, 2, 3], average="macro")


def load_predictions(condition, seed):
    recipe_name = LOCKED_RECIPE[condition]
    run_id = f"daisee_test_{condition}_{recipe_name}_seed{seed}"
    path = os.path.join(REPO_ROOT, "outputs", run_id, "predictions.json")
    with open(path) as f:
        return json.load(f)


def aligned_arrays(condition_a, condition_b, seed):
    pa = load_predictions(condition_a, seed)
    pb = load_predictions(condition_b, seed)
    idx_b = {cid: i for i, cid in enumerate(pb["clip_ids"])}
    assert set(pa["clip_ids"]) == set(pb["clip_ids"]), (
        f"{condition_a} seed{seed} and {condition_b} seed{seed} test sets differ"
    )
    order_b = [idx_b[cid] for cid in pa["clip_ids"]]
    labels_a = np.array(pa["labels"])
    labels_b = np.array(pb["labels"])[order_b]
    assert np.array_equal(labels_a, labels_b), (
        f"label mismatch between {condition_a} and {condition_b} at seed {seed}"
    )
    preds_a = np.array(pa["preds"])
    preds_b = np.array(pb["preds"])[order_b]
    return pa["clip_ids"], labels_a, preds_a, preds_b


def paired_bootstrap_margin(condition_a, condition_b, n_bootstrap=N_BOOTSTRAP, seed_for_resample=0):
    per_seed = []
    for seed in SEEDS:
        clip_ids, labels, preds_a, preds_b = aligned_arrays(condition_a, condition_b, seed)
        per_seed.append({"seed": seed, "labels": labels, "preds_a": preds_a, "preds_b": preds_b})

    n_items = len(per_seed[0]["labels"])
    for ps in per_seed:
        assert len(ps["labels"]) == n_items

    metric_a = [primary_metric(ps["labels"], ps["preds_a"]) for ps in per_seed]
    metric_b = [primary_metric(ps["labels"], ps["preds_b"]) for ps in per_seed]
    per_seed_margin = [a - b for a, b in zip(metric_a, metric_b)]
    mean_margin = float(np.mean(per_seed_margin))

    rng = np.random.RandomState(seed_for_resample)
    boot_margins = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        resample_idx = rng.randint(0, n_items, size=n_items)
        seed_margins_b = np.empty(len(SEEDS), dtype=np.float64)
        for si, ps in enumerate(per_seed):
            y = ps["labels"][resample_idx]
            pa = ps["preds_a"][resample_idx]
            pb = ps["preds_b"][resample_idx]
            seed_margins_b[si] = primary_metric(y, pa) - primary_metric(y, pb)
        boot_margins[b] = seed_margins_b.mean()

    ci_low, ci_high = np.percentile(boot_margins, [2.5, 97.5])
    return {
        "condition_a": condition_a, "condition_b": condition_b,
        "n_items": n_items, "n_bootstrap": n_bootstrap,
        "per_seed_metric_a": metric_a, "per_seed_metric_b": metric_b,
        "per_seed_margin": per_seed_margin, "mean_margin": mean_margin,
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def seed_paired_stats(per_seed_margin):
    margins = np.array(per_seed_margin)
    n = len(margins)
    mean = margins.mean()
    sem = margins.std(ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
    ci_low, ci_high = mean - t_crit * sem, mean + t_crit * sem
    t_stat, t_p = scipy_stats.ttest_1samp(margins, popmean=0.0)
    try:
        w_stat, w_p = scipy_stats.wilcoxon(margins)
    except ValueError:
        w_stat, w_p = None, None
    return {
        "per_seed_margin": margins.tolist(), "n": n, "mean": float(mean),
        "t_ci_low": float(ci_low), "t_ci_high": float(ci_high),
        "paired_t_stat": float(t_stat), "paired_t_p": float(t_p),
        "wilcoxon_stat": (float(w_stat) if w_stat is not None else None),
        "wilcoxon_p": (float(w_p) if w_p is not None else None),
    }


def apply_decision_rule(h2_result):
    """docs/DECISION_RULES_DAISEE.md's frozen 3-branch table, applied exactly,
    against the amended primary metric (classes 1-3 macro-F1). "Supported" =
    paired requirement: point estimate clears EFFECT_FLOOR=0.03 AND the
    item-paired bootstrap CI excludes zero (the pre-registered CI method for
    the decision rule itself; the seed-paired test is reported alongside,
    not substituted into this criterion)."""
    m = h2_result["mean_margin"]
    h2_supported = (m >= EFFECT_FLOOR) and h2_result["ci_excludes_zero"]

    if h2_supported:
        branch = "a"
        description = ("H2 supported: entity-localized tokenization helps over a fixed grid, "
                        "replicating MELD's H2 direction.")
    elif abs(m) < EFFECT_FLOOR:
        branch = "b"
        description = "Null / underpowered -- reported as such, not as evidence of absence."
    elif m < -EFFECT_FLOOR:
        branch = "c"
        description = "Reversed: entity tokens hurt relative to the grid on this dataset."
    else:
        branch = None
        description = ("None of the three pre-specified branches fired cleanly. Reported as such, "
                        "not forced into the nearest branch.")

    return {"effect_floor": EFFECT_FLOOR, "h2_supported": h2_supported,
            "mean_margin_A1_A0": m, "branch": branch, "description": description}


def main():
    print("=== H2: A1 - A0, primary metric (macro-F1, classes 1-3) ===", flush=True)
    h2 = paired_bootstrap_margin("A1", "A0")
    print(f"mean_margin={h2['mean_margin']:.4f} 95% CI=[{h2['ci_low']:.4f}, {h2['ci_high']:.4f}] "
          f"excludes_zero={h2['ci_excludes_zero']}", flush=True)
    print(f"per_seed_metric A1={[f'{x:.4f}' for x in h2['per_seed_metric_a']]}", flush=True)
    print(f"per_seed_metric A0={[f'{x:.4f}' for x in h2['per_seed_metric_b']]}", flush=True)

    print("\n=== Seed-paired test (same margins, treated as the sample) ===", flush=True)
    sp = seed_paired_stats(h2["per_seed_margin"])
    print(json.dumps(sp, indent=2), flush=True)

    decision = apply_decision_rule(h2)
    print("\n=== DECISION RULE ===", flush=True)
    print(json.dumps(decision, indent=2), flush=True)

    out = {"h2_item_paired_bootstrap": h2, "h2_seed_paired": sp, "decision": decision}
    out_path = os.path.join(REPO_ROOT, "outputs", "daisee_phase4_statistics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
