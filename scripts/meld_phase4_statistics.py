"""Phase 4 statistics: A1-A2 and A1-A0 margins, paired bootstrap over TEST
ITEMS (docs/DECISION_RULES.md's "paired requirement"). Reads only the
predictions.json files scripts/meld_phase4_test_eval.py already wrote --
this script touches no data itself, so it can be re-run freely without
touching test again.

Bootstrap design: each of 10,000 resamples draws a set of test-item indices
WITH REPLACEMENT (same resampled index set applied to every seed and both
conditions being compared, since it's the same 2610 test items throughout --
this is what makes it item-PAIRED, not seed-paired: resampling seeds instead
of items would only capture seed variance, exactly the "seed-paired only"
failure mode the task calls out). Within each resample, macro-F1 is
recomputed per seed on the resampled items, then averaged across the 5
seeds -- so both item-level and seed-level variance flow into the resulting
CI width, matching the same two-source-of-variance logic already used for
the blind EFFECT_FLOOR bootstrap (docs/DECISION_RULES.md, 2026-08-15).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from sklearn.metrics import f1_score

REPO_ROOT = "/home/devops/ept"
SEEDS = [42, 1337, 2024, 7, 31337]
LOCKED_RECIPE = {
    "A0": "r08", "A1": "r07", "A2": "r03", "A3": "r04",
    "A4": "r03", "A5": "r07", "mask_only": "r05",
}
EFFECT_FLOOR = 0.04
N_BOOTSTRAP = 10000


def load_predictions(condition, seed):
    recipe_name = LOCKED_RECIPE[condition]
    run_id = f"meld_test_{condition}_{recipe_name}_seed{seed}"
    path = os.path.join(REPO_ROOT, "outputs", run_id, "predictions.json")
    with open(path) as f:
        return json.load(f)


def aligned_arrays(condition_a, condition_b, seed):
    """Join two conditions' predictions for one seed by clip_id (not by
    trusting index order across separately-run processes), then return
    labels + both conditions' preds in a common, verified-aligned order."""
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
        f"label mismatch between {condition_a} and {condition_b} at seed {seed} -- "
        f"same clip_id must have the same true label regardless of condition"
    )
    preds_a = np.array(pa["preds"])
    preds_b = np.array(pb["preds"])[order_b]
    return pa["clip_ids"], labels_a, preds_a, preds_b


def paired_bootstrap_margin(condition_a, condition_b, n_bootstrap=N_BOOTSTRAP, seed_for_resample=0,
                             clip_id_filter=None, seeds=None):
    """Returns (mean_margin_across_seeds, ci_low, ci_high, per_seed_macro_f1_a,
    per_seed_macro_f1_b, per_seed_margin). clip_id_filter, if given, restricts
    to that subset of clip_ids (used for the purity-tercile stratification).
    seeds, if given, overrides the full SEEDS list (used for leave-one-seed-out
    sensitivity -- post hoc, re-reads the same predictions.json files, does not
    retrain or re-touch test)."""
    seeds = seeds if seeds is not None else SEEDS
    per_seed = []
    for seed in seeds:
        clip_ids, labels, preds_a, preds_b = aligned_arrays(condition_a, condition_b, seed)
        if clip_id_filter is not None:
            keep = np.array([cid in clip_id_filter for cid in clip_ids])
            labels, preds_a, preds_b = labels[keep], preds_a[keep], preds_b[keep]
            clip_ids = [c for c, k in zip(clip_ids, keep) if k]
        per_seed.append({"seed": seed, "clip_ids": clip_ids, "labels": labels,
                          "preds_a": preds_a, "preds_b": preds_b})

    n_items = len(per_seed[0]["labels"])
    for ps in per_seed:
        assert len(ps["labels"]) == n_items, "item count differs across seeds -- should be impossible"

    macro_f1_a = [f1_score(ps["labels"], ps["preds_a"], average="macro") for ps in per_seed]
    macro_f1_b = [f1_score(ps["labels"], ps["preds_b"], average="macro") for ps in per_seed]
    per_seed_margin = [a - b for a, b in zip(macro_f1_a, macro_f1_b)]
    mean_margin = float(np.mean(per_seed_margin))

    rng = np.random.RandomState(seed_for_resample)
    boot_margins = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        resample_idx = rng.randint(0, n_items, size=n_items)
        seed_margins_b = np.empty(len(seeds), dtype=np.float64)
        for si, ps in enumerate(per_seed):
            y = ps["labels"][resample_idx]
            pa = ps["preds_a"][resample_idx]
            pb = ps["preds_b"][resample_idx]
            f1_a = f1_score(y, pa, average="macro")
            f1_b = f1_score(y, pb, average="macro")
            seed_margins_b[si] = f1_a - f1_b
        boot_margins[b] = seed_margins_b.mean()

    ci_low, ci_high = np.percentile(boot_margins, [2.5, 97.5])
    return {
        "condition_a": condition_a, "condition_b": condition_b,
        "n_items": n_items, "n_bootstrap": n_bootstrap,
        "per_seed_macro_f1_a": macro_f1_a, "per_seed_macro_f1_b": macro_f1_b,
        "per_seed_margin": per_seed_margin, "mean_margin": mean_margin,
        "ci_low": float(ci_low), "ci_high": float(ci_high),
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
    }


def apply_decision_rule(h1_result, h2_result):
    """docs/DECISION_RULES.md's decision table, applied exactly, EFFECT_FLOOR=0.04
    substituted for the table's original 0.02. "Supported" = paired requirement:
    point estimate clears the floor AND the CI excludes zero."""
    m12 = h1_result["mean_margin"]  # A1 - A2
    m10 = h2_result["mean_margin"]  # A1 - A0
    h1_supported = (m12 >= EFFECT_FLOOR) and h1_result["ci_excludes_zero"]
    h2_supported = (m10 >= EFFECT_FLOOR) and h2_result["ci_excludes_zero"]

    if h1_supported:
        branch = "a"
        description = "H1 supported: persistence is the mechanism. Headline comparison: A1 vs A2."
    elif abs(m12) < EFFECT_FLOOR and h2_supported:
        branch = "b"
        description = ("abs(A1-A2) < EFFECT_FLOOR and H2 supported: boundary paper -- persistence is "
                        "not what helps; entity-localized cropping and token budget are. "
                        "Headline: A1/A2 vs A0.")
    elif m12 < -EFFECT_FLOOR:
        branch = "c"
        description = "A1 < A2 - EFFECT_FLOOR: hypothesis refuted. Pivot to the efficiency Pareto result."
    else:
        branch = None
        description = ("None of the three pre-specified branches fired. The frozen decision table "
                        "does not name an outcome for this result -- reported as such, not forced "
                        "into the nearest branch.")

    return {"effect_floor": EFFECT_FLOOR, "h1_supported": h1_supported, "h2_supported": h2_supported,
            "mean_margin_A1_A2": m12, "mean_margin_A1_A0": m10, "branch": branch,
            "description": description}


def main():
    print("=== H1: A1 - A2 (persistence) ===", flush=True)
    h1 = paired_bootstrap_margin("A1", "A2")
    print(f"mean_margin={h1['mean_margin']:.4f} 95% CI=[{h1['ci_low']:.4f}, {h1['ci_high']:.4f}] "
          f"excludes_zero={h1['ci_excludes_zero']}", flush=True)
    print(f"per_seed_macro_f1 A1={[f'{x:.4f}' for x in h1['per_seed_macro_f1_a']]}", flush=True)
    print(f"per_seed_macro_f1 A2={[f'{x:.4f}' for x in h1['per_seed_macro_f1_b']]}", flush=True)

    print("\n=== H2: A1 - A0 (entity structure) ===", flush=True)
    h2 = paired_bootstrap_margin("A1", "A0")
    print(f"mean_margin={h2['mean_margin']:.4f} 95% CI=[{h2['ci_low']:.4f}, {h2['ci_high']:.4f}] "
          f"excludes_zero={h2['ci_excludes_zero']}", flush=True)
    print(f"per_seed_macro_f1 A1={[f'{x:.4f}' for x in h2['per_seed_macro_f1_a']]}", flush=True)
    print(f"per_seed_macro_f1 A0={[f'{x:.4f}' for x in h2['per_seed_macro_f1_b']]}", flush=True)

    decision = apply_decision_rule(h1, h2)
    print("\n=== DECISION RULE ===", flush=True)
    print(json.dumps(decision, indent=2), flush=True)

    out = {"h1_persistence": h1, "h2_entity_structure": h2, "decision": decision}
    out_path = os.path.join(REPO_ROOT, "outputs", "meld_phase4_statistics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
