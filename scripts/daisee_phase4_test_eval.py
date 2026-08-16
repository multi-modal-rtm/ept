"""DAiSEE Phase 4: the ONE-SHOT test touch. Test split touched exactly once
per (condition, seed) -- training uses only DAiSEEDataset("train", ...);
evaluation uses DAiSEETestDataset exactly once, after all training epochs
complete, never during training, never more than once per (condition, seed).
No test-based epoch selection anywhere in this file -- the locked recipe's
epoch count (docs/LOCKED_RECIPE_DAISEE.md, read-only) is trained to
completion and the FINAL epoch's model is what gets evaluated on test.

Primary metric, per docs/DECISION_RULES_DAISEE.md's 2026-08-16 amendment:
macro-F1 over classes 1-3 only (class 0 excluded from the average, still
included in every other class's confusion counts). 4-class macro-F1,
accuracy, and full per-class F1 (class 0 included) are computed and saved
unconditionally as secondary numbers, and per-item predictions are saved for
the paired-bootstrap analysis.
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
from sklearn.metrics import accuracy_score, f1_score

from ept.model.ept_former import EPTFormer, assert_no_backbone_params
from ept.train.dataset_daisee import DAISEE_PRIMARY_E_MAX, DAiSEEDataset, S as DAISEE_S
from ept.train.dataset_daisee_test import DAiSEETestDataset

REPO_ROOT = "/home/devops/ept"
CONFIGS_ROOT = os.path.join(REPO_ROOT, "configs")
SEEDS = [42, 1337, 2024, 7, 31337]
CONDITIONS = ["A0", "A1"]
NUM_CLASSES = 4
CLASS_NAMES = ["0_verylow", "1_low", "2_high", "3_veryhigh"]
# docs/LOCKED_RECIPE_DAISEE.md -- read-only, reproduced here verbatim.
LOCKED_RECIPE = {"A0": "r08", "A1": "r06"}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(condition, dropout, e_max, s):
    return EPTFormer(dropout=dropout, use_temporal=True, use_social=True, s_max=s,
                      num_classes=NUM_CLASSES)


def to_device(ds, device):
    ds.features = ds.features.to(device)
    ds.masks = ds.masks.to(device)
    ds.labels = ds.labels.to(device)
    return ds


def run_train_epoch(model, ds, batch_size, device, optimizer, shuffle=True):
    model.train()
    n = len(ds)
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    total_loss = 0.0
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        feat, mask, label = ds.features[idx], ds.masks[idx], ds.labels[idx]
        logits = model(feat, mask)
        loss = nn.functional.cross_entropy(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.size(0)
    return total_loss / n


def eval_full(model, ds, batch_size, device):
    model.eval()
    n = len(ds)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n), device=device)
            feat, mask, label = ds.features[idx], ds.masks[idx], ds.labels[idx]
            logits = model(feat, mask)
            all_logits.append(logits)
            all_labels.append(label)
    logits = torch.cat(all_logits).cpu()
    labels = torch.cat(all_labels).cpu()
    preds = logits.argmax(dim=-1)
    y, p = labels.numpy(), preds.numpy()

    primary_macro_f1_1to3 = f1_score(y, p, labels=[1, 2, 3], average="macro")
    macro_f1_4class = f1_score(y, p, labels=[0, 1, 2, 3], average="macro")
    accuracy = accuracy_score(y, p)
    per_class_f1 = f1_score(y, p, labels=[0, 1, 2, 3], average=None, zero_division=0).tolist()
    class0_count = int((y == 0).sum())

    return {
        "primary_macro_f1_classes1to3": primary_macro_f1_1to3,
        "macro_f1_4class_secondary": macro_f1_4class,
        "accuracy": accuracy,
        "per_class_f1": dict(zip(CLASS_NAMES, per_class_f1)),
        "class0_raw_f1": per_class_f1[0],
        "class0_n_test_items": class0_count,
    }, p.tolist(), y.tolist()


def run_condition_seed(condition, seed, device):
    recipe_name = LOCKED_RECIPE[condition]
    recipe = load_yaml(os.path.join(CONFIGS_ROOT, "recipe", f"{recipe_name}.yaml"))
    run_id = f"daisee_test_{condition}_{recipe_name}_seed{seed}"
    results_dir = os.path.join(REPO_ROOT, "outputs", run_id)
    os.makedirs(results_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = DAiSEEDataset("train", condition, run_seed=seed)
    test_ds = DAiSEETestDataset(condition)
    clip_ids = list(test_ds.clip_ids)
    train_ds = to_device(train_ds, device)
    test_ds = to_device(test_ds, device)

    torch.manual_seed(seed)
    model = build_model(condition, recipe["dropout"], DAISEE_PRIMARY_E_MAX, DAISEE_S).to(device)
    assert_no_backbone_params(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )

    t0 = time.time()
    train_losses = []
    for epoch in range(recipe["epochs"]):
        loss = run_train_epoch(model, train_ds, recipe["batch_size"], device, optimizer, shuffle=True)
        train_losses.append(loss)
    train_wall = time.time() - t0

    # THE single test touch for this (condition, seed).
    test_metrics, preds, labels = eval_full(model, test_ds, recipe["batch_size"], device)
    wall_clock = time.time() - t0

    metrics = {
        "run_id": run_id, "dataset": "daisee", "condition": condition, "seed": seed,
        "recipe_id": recipe_name, "recipe": recipe,
        "train_split": "train", "eval_split": "test",
        "n_train": len(train_ds), "n_test": len(test_ds),
        "final_train_loss": train_losses[-1],
        **test_metrics,
        "train_wall_seconds": train_wall, "wall_clock_seconds": wall_clock,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "predictions.json"), "w") as f:
        json.dump({"run_id": run_id, "condition": condition, "seed": seed,
                    "clip_ids": clip_ids, "labels": labels, "preds": preds}, f)

    print(f"  {run_id}: primary(1-3)={test_metrics['primary_macro_f1_classes1to3']:.4f} "
          f"4class={test_metrics['macro_f1_4class_secondary']:.4f} acc={test_metrics['accuracy']:.4f} "
          f"class0_f1={test_metrics['class0_raw_f1']:.4f} (n=4) ({wall_clock:.1f}s)", flush=True)
    return metrics


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_metrics = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            all_metrics.append(run_condition_seed(condition, seed, device))

    summary_path = os.path.join(REPO_ROOT, "outputs", "daisee_phase4_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"conditions": CONDITIONS, "seeds": SEEDS, "locked_recipe": LOCKED_RECIPE,
                    "results": all_metrics}, f, indent=2)
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
