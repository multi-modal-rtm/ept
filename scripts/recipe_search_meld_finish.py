"""Finish the MELD recipe search after the mask_only-geometry crash: rerun only
the mask_only condition's 8 grid points (A0-A5 already completed and their
metrics.json files are on disk — not rerun, to avoid ~35 min of wasted repeat
compute), then rebuild the aggregate summary by scanning all 56 run
directories. Same run_id/results_dir scheme as recipe_search_meld.py so the
mask_only outputs land in the exact same tree.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import yaml

from ept.model.ept_former import assert_no_backbone_params
from ept.tokenization.extract_features_meld import MELD_PRIMARY_E_MAX, S as MELD_S
from ept.train.dataset_meld import MELDDataset
from ept.train.train import build_model, run_epoch, to_device

REPO_ROOT = "/home/devops/ept"
CONFIGS_ROOT = os.path.join(REPO_ROOT, "configs")
CONDITIONS = ["a0", "a1", "a2", "a3", "a4", "a5", "mask_only"]
RECIPES = ["r01", "r02", "r03", "r04", "r05", "r06", "r07", "r08"]
SEARCH_SEED = 42
TRIVIAL_PROBE_FLOOR = 0.4155


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run_mask_only():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recipes = {r: load_yaml(os.path.join(CONFIGS_ROOT, "recipe", f"{r}.yaml")) for r in RECIPES}
    cond = load_yaml(os.path.join(CONFIGS_ROOT, "condition", "mask_only.yaml"))
    condition = cond["name"]
    assert condition == "mask_only"

    print(f"=== condition {condition} — loading dataset ===")
    torch.manual_seed(SEARCH_SEED)
    np.random.seed(SEARCH_SEED)
    train_ds = to_device(MELDDataset("train", condition, run_seed=SEARCH_SEED), device)
    dev_ds = to_device(MELDDataset("dev", condition, run_seed=SEARCH_SEED), device)

    for recipe_name in RECIPES:
        recipe = recipes[recipe_name]
        run_id = f"meld_{condition}_{recipe_name}_seed{SEARCH_SEED}"
        results_dir = os.path.join(REPO_ROOT, "outputs", run_id)
        os.makedirs(results_dir, exist_ok=True)

        torch.manual_seed(SEARCH_SEED)
        model = build_model(
            condition, recipe["dropout"], e_max=MELD_PRIMARY_E_MAX, s=MELD_S
        ).to(device)
        assert_no_backbone_params(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
        )

        history = []
        t0 = time.time()
        best_dev_f1, best_epoch = -1.0, -1
        for epoch in range(recipe["epochs"]):
            train_loss, train_f1 = run_epoch(
                model, train_ds, condition, recipe["batch_size"], device, optimizer, shuffle=True
            )
            dev_loss, dev_f1 = run_epoch(
                model, dev_ds, condition, recipe["batch_size"], device, optimizer=None, shuffle=False
            )
            history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                             "val_loss": dev_loss, "val_macro_f1": dev_f1})
            if dev_f1 > best_dev_f1:
                best_dev_f1, best_epoch = dev_f1, epoch
        wall_clock = time.time() - t0

        metrics = {
            "run_id": run_id, "condition": condition, "seed": SEARCH_SEED,
            "recipe": recipe, "split_evaluated": "dev",
            "n_train": len(train_ds), "n_dev": len(dev_ds),
            "trivial_probe_floor": TRIVIAL_PROBE_FLOOR,
            "history": history,
            "best_val_macro_f1": best_dev_f1, "best_epoch": best_epoch,
            "final_val_macro_f1": history[-1]["val_macro_f1"],
            "wall_clock_seconds": wall_clock,
        }
        with open(os.path.join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        with open(os.path.join(results_dir, "config.yaml"), "w") as f:
            yaml.dump({"condition": cond, "recipe": recipe, "seed": SEARCH_SEED, "run_id": run_id}, f)

        print(f"  {run_id}: best_dev_macro_f1={best_dev_f1:.4f} @ epoch {best_epoch} "
              f"(floor {TRIVIAL_PROBE_FLOOR}: {'PASS' if best_dev_f1 > TRIVIAL_PROBE_FLOOR else 'FAIL'}) "
              f"({wall_clock:.1f}s)", flush=True)


def rebuild_summary():
    all_results = []
    n_dev_evals = 0
    for cond_file in CONDITIONS:
        cond = load_yaml(os.path.join(CONFIGS_ROOT, "condition", f"{cond_file}.yaml"))
        condition = cond["name"]
        for recipe_name in RECIPES:
            run_id = f"meld_{condition}_{recipe_name}_seed{SEARCH_SEED}"
            metrics_path = os.path.join(REPO_ROOT, "outputs", run_id, "metrics.json")
            with open(metrics_path) as f:
                m = json.load(f)
            n_dev_evals += len(m["history"])
            all_results.append({
                "condition": condition, "recipe": recipe_name,
                "best_dev_macro_f1": m["best_val_macro_f1"], "best_epoch": m["best_epoch"],
                "beats_trivial_floor": m["best_val_macro_f1"] > TRIVIAL_PROBE_FLOOR,
                "wall_clock_seconds": m["wall_clock_seconds"],
            })

    summary_path = os.path.join(REPO_ROOT, "outputs", "meld_phase3_search_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_conditions": len(CONDITIONS), "n_recipes": len(RECIPES),
            "search_seed": SEARCH_SEED, "n_runs": len(all_results),
            "n_dev_evaluations": n_dev_evals,
            "trivial_probe_floor": TRIVIAL_PROBE_FLOOR,
            "results": all_results,
        }, f, indent=2)
    print(f"rebuilt summary ({len(all_results)} runs, {n_dev_evals} dev evaluations) -> {summary_path}",
          flush=True)


if __name__ == "__main__":
    run_mask_only()
    rebuild_summary()
