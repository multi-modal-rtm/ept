"""DAiSEE Phase 3: run the frozen grid (docs/SEARCH_GRID_DAISEE.md) identically
for A0 and A1, on val only. Mirrors scripts/recipe_search_meld.py's in-process
structure (each condition's dataset loaded once, reused across its 8 recipe
points). Records per-class F1 at every epoch, not just macro-F1, so the
class-imbalance risk pre-registered in docs/DECISION_RULES_DAISEE.md (class 0:
4 test clips) can be previewed against real val numbers before Phase 4.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from ept.model.ept_former import EPTFormer, MeanPoolMLP, assert_no_backbone_params
from ept.train.dataset_daisee import DAISEE_PRIMARY_E_MAX, DAiSEEDataset, S as DAISEE_S

REPO_ROOT = "/home/devops/ept"
CONFIGS_ROOT = os.path.join(REPO_ROOT, "configs")
CONDITIONS = ["A0", "A1"]
RECIPES = ["r01", "r02", "r03", "r04", "r05", "r06", "r07", "r08"]
SEARCH_SEED = 42
NUM_CLASSES = 4  # DAiSEE Engagement, docs/DECISION_RULES_DAISEE.md
CLASS_NAMES = ["0_verylow", "1_low", "2_high", "3_veryhigh"]  # DAiSEE Engagement levels 0-3


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(condition, dropout, e_max, s):
    if condition == "A0":
        return EPTFormer(dropout=dropout, use_temporal=True, use_social=True, s_max=s,
                          num_classes=NUM_CLASSES)
    if condition == "A1":
        return EPTFormer(dropout=dropout, use_temporal=True, use_social=True, s_max=s,
                          num_classes=NUM_CLASSES)
    raise ValueError(f"unsupported condition for this H2-only track: {condition!r}")


def to_device(ds, device):
    ds.features = ds.features.to(device)
    ds.masks = ds.masks.to(device)
    ds.labels = ds.labels.to(device)
    return ds


def run_epoch(model, ds, batch_size, device, optimizer=None, shuffle=False):
    training = optimizer is not None
    model.train(training)
    n = len(ds)
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)

    all_logits, all_labels = [], []
    total_loss = 0.0
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        feat, mask, label = ds.features[idx], ds.masks[idx], ds.labels[idx]
        with torch.set_grad_enabled(training):
            logits = model(feat, mask)
            loss = nn.functional.cross_entropy(logits, label)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * label.size(0)
        all_logits.append(logits.detach())
        all_labels.append(label)
    logits = torch.cat(all_logits).cpu()
    labels = torch.cat(all_labels).cpu()
    preds = logits.argmax(dim=-1)
    y, p = labels.numpy(), preds.numpy()
    macro_f1 = f1_score(y, p, average="macro")
    per_class_f1 = f1_score(y, p, average=None, labels=[0, 1, 2, 3], zero_division=0).tolist()
    return total_loss / n, macro_f1, per_class_f1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recipes = {r: load_yaml(os.path.join(CONFIGS_ROOT, "recipe", f"{r}.yaml")) for r in RECIPES}

    all_results = []
    grand_t0 = time.time()
    n_val_evals = 0

    for condition in CONDITIONS:
        print(f"\n=== condition {condition} — loading dataset ===", flush=True)
        torch.manual_seed(SEARCH_SEED)
        np.random.seed(SEARCH_SEED)
        train_ds = to_device(DAiSEEDataset("train", condition, run_seed=SEARCH_SEED), device)
        val_ds = to_device(DAiSEEDataset("val", condition, run_seed=SEARCH_SEED), device)

        for recipe_name in RECIPES:
            recipe = recipes[recipe_name]
            run_id = f"daisee_{condition}_{recipe_name}_seed{SEARCH_SEED}"
            results_dir = os.path.join(REPO_ROOT, "outputs", run_id)
            os.makedirs(results_dir, exist_ok=True)

            torch.manual_seed(SEARCH_SEED)
            model = build_model(condition, recipe["dropout"], DAISEE_PRIMARY_E_MAX, DAISEE_S).to(device)
            assert_no_backbone_params(model)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
            )

            history = []
            t0 = time.time()
            best_val_f1, best_epoch, best_per_class_f1 = -1.0, -1, None
            for epoch in range(recipe["epochs"]):
                train_loss, train_f1, _ = run_epoch(
                    model, train_ds, recipe["batch_size"], device, optimizer, shuffle=True
                )
                val_loss, val_f1, val_per_class_f1 = run_epoch(
                    model, val_ds, recipe["batch_size"], device, optimizer=None, shuffle=False
                )
                n_val_evals += 1
                history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                                 "val_loss": val_loss, "val_macro_f1": val_f1,
                                 "val_per_class_f1": dict(zip(CLASS_NAMES, val_per_class_f1))})
                if val_f1 > best_val_f1:
                    best_val_f1, best_epoch, best_per_class_f1 = val_f1, epoch, val_per_class_f1
            wall_clock = time.time() - t0

            metrics = {
                "run_id": run_id, "dataset": "daisee", "condition": condition, "seed": SEARCH_SEED,
                "recipe": recipe, "split_evaluated": "val",
                "n_train": len(train_ds), "n_val": len(val_ds),
                "history": history,
                "best_val_macro_f1": best_val_f1, "best_epoch": best_epoch,
                "best_val_per_class_f1": dict(zip(CLASS_NAMES, best_per_class_f1)),
                "final_val_macro_f1": history[-1]["val_macro_f1"],
                "wall_clock_seconds": wall_clock,
            }
            with open(os.path.join(results_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)
            with open(os.path.join(results_dir, "config.yaml"), "w") as f:
                yaml.dump({"condition": condition, "recipe": recipe, "seed": SEARCH_SEED, "run_id": run_id}, f)

            all_results.append({
                "condition": condition, "recipe": recipe_name,
                "best_val_macro_f1": best_val_f1, "best_epoch": best_epoch,
                "best_val_per_class_f1": dict(zip(CLASS_NAMES, best_per_class_f1)),
                "wall_clock_seconds": wall_clock,
            })
            print(f"  {run_id}: best_val_macro_f1={best_val_f1:.4f} @ epoch {best_epoch} "
                  f"per_class={[f'{x:.3f}' for x in best_per_class_f1]} ({wall_clock:.1f}s)", flush=True)

    total_wall = time.time() - grand_t0
    print(f"\n=== TOTAL: {len(all_results)} runs, {n_val_evals} val evaluations, {total_wall:.1f}s ===",
          flush=True)

    summary_path = os.path.join(REPO_ROOT, "outputs", "daisee_phase3_search_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "n_conditions": len(CONDITIONS), "n_recipes": len(RECIPES),
            "search_seed": SEARCH_SEED, "n_runs": len(all_results),
            "n_val_evaluations": n_val_evals, "total_wall_clock_seconds": total_wall,
            "results": all_results,
        }, f, indent=2)
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
