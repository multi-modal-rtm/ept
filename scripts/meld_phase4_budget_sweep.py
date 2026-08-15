"""Phase 4 token-budget sweep: E in {1,2,4,6,8} x S in {2,4}, condition=A1
(primary), A1's locked recipe (r07, docs/LOCKED_RECIPE_MELD.md) unchanged,
5 seeds. Part of the same one-shot test-touch as
scripts/meld_phase4_test_eval.py -- trains on train only, evaluates on test
exactly once per (E,S,seed) point, no test-based selection. Supplementary
Pareto context, not part of the paired-bootstrap decision-rule analysis
(docs/DECISION_RULES.md's H1/H2 statistics are computed from the main
matrix's A0/A1/A2 predictions only).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from ept.model.ept_former import assert_no_backbone_params
from ept.train.dataset_meld_budget import MELDBudgetDataset
from ept.train.train import build_model, forward_pass, run_epoch, to_device

REPO_ROOT = "/home/devops/ept"
CONFIGS_ROOT = os.path.join(REPO_ROOT, "configs")
SEEDS = [42, 1337, 2024, 7, 31337]
E_GRID = [1, 2, 4, 6, 8]
S_GRID = [2, 4]
RECIPE_ID = "r07"  # A1's locked recipe, docs/LOCKED_RECIPE_MELD.md, unchanged
CONDITION = "A1"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def eval_full(model, ds, condition, batch_size, device):
    model.eval()
    n = len(ds)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n), device=device)
            feat, mask, label = ds.features[idx], ds.masks[idx], ds.labels[idx]
            logits = forward_pass(model, condition, feat, mask)
            all_logits.append(logits)
            all_labels.append(label)
    logits = torch.cat(all_logits).cpu()
    labels = torch.cat(all_labels).cpu()
    preds = logits.argmax(dim=-1)
    y, p = labels.numpy(), preds.numpy()
    macro_f1 = f1_score(y, p, average="macro")
    accuracy = accuracy_score(y, p)
    return macro_f1, accuracy


def run_point(e, s, seed, recipe, device):
    run_id = f"meld_test_budget_e{e}_s{s}_{RECIPE_ID}_seed{seed}"
    results_dir = os.path.join(REPO_ROOT, "outputs", run_id)
    os.makedirs(results_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = to_device(MELDBudgetDataset("train", e, s, run_seed=seed), device)
    test_ds = to_device(MELDBudgetDataset("test", e, s, run_seed=seed), device)

    torch.manual_seed(seed)
    model = build_model(CONDITION, recipe["dropout"], e_max=e, s=s).to(device)
    assert_no_backbone_params(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )

    t0 = time.time()
    for epoch in range(recipe["epochs"]):
        run_epoch(model, train_ds, CONDITION, recipe["batch_size"], device, optimizer, shuffle=True)
    train_wall = time.time() - t0

    test_macro_f1, test_accuracy = eval_full(model, test_ds, CONDITION, recipe["batch_size"], device)
    wall_clock = time.time() - t0

    metrics = {
        "run_id": run_id, "dataset": "meld", "condition": CONDITION, "e": e, "s": s, "seed": seed,
        "recipe_id": RECIPE_ID, "recipe": recipe,
        "n_train": len(train_ds), "n_test": len(test_ds),
        "test_macro_f1": test_macro_f1, "test_accuracy": test_accuracy,
        "wall_clock_seconds": wall_clock,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  {run_id}: test_macro_f1={test_macro_f1:.4f} test_acc={test_accuracy:.4f} ({wall_clock:.1f}s)",
          flush=True)
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recipe = load_yaml(os.path.join(CONFIGS_ROOT, "recipe", f"{RECIPE_ID}.yaml"))

    all_results = []
    for e in E_GRID:
        for s in S_GRID:
            for seed in SEEDS:
                all_results.append(run_point(e, s, seed, recipe, device))

    summary_path = os.path.join(REPO_ROOT, "outputs", "meld_phase4_budget_sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"e_grid": E_GRID, "s_grid": S_GRID, "seeds": SEEDS,
                    "recipe_id": RECIPE_ID, "condition": CONDITION,
                    "results": all_results}, f, indent=2)
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
