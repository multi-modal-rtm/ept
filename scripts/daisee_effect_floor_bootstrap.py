"""Compute the DAiSEE H2-track effect floor, per the procedure fixed in
docs/DECISION_RULES_DAISEE.md (committed before this script ran). Blind to
condition ranking throughout: only dispersion (standard errors) is ever
inspected or recorded -- no comparison between A0/A1's point estimates
appears anywhere in this script's output. Mirrors
scripts/meld_effect_floor_bootstrap.py's exact procedure, restricted to the
two conditions this track actually compares (no A2-A5/mask-only -- H2 only).

Fixed, reasonable recipe (NOT tuned for this purpose, NOT a new locked
recipe): reuses r07 (lr=1e-4, wd=0.0, dropout=0.0, epochs=50) -- the same
placeholder MELD's own floor bootstrap used, for the same reason (*some*
fixed recipe is needed to produce trained models to bootstrap from).
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from ept.model.ept_former import EPTFormer, assert_no_backbone_params
from ept.train.dataset_daisee import DAISEE_PRIMARY_E_MAX, S
from ept.train.dataset_daisee import DAiSEEDataset

CONDITIONS = ["A0", "A1"]
SEEDS = [42, 1337, 2024]
RECIPE = {"lr": 1e-4, "weight_decay": 0.0, "dropout": 0.0, "epochs": 50, "batch_size": 32}
N_BOOTSTRAP = 500
NUM_CLASSES = 4  # DAiSEE Engagement, docs/DECISION_RULES_DAISEE.md
OUT_DIR = "/home/devops/ept/outputs/daisee_effect_floor"


def to_device(ds, device):
    ds.features = ds.features.to(device)
    ds.masks = ds.masks.to(device)
    ds.labels = ds.labels.to(device)
    return ds


def build_model(condition, dropout, e_max, s):
    return EPTFormer(dropout=dropout, use_temporal=True, use_social=True, s_max=s,
                      num_classes=NUM_CLASSES)


def run_epoch(model, ds, batch_size, device, optimizer=None, shuffle=False):
    training = optimizer is not None
    model.train(training)
    n = len(ds)
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
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


def eval_logits(model, ds):
    model.eval()
    with torch.no_grad():
        logits = model(ds.features, ds.masks)
    return logits.cpu().numpy(), ds.labels.cpu().numpy()


def bootstrap_item_se(logits, labels, n_boot=N_BOOTSTRAP, seed=0):
    preds = logits.argmax(axis=-1)
    n = len(labels)
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        scores.append(f1_score(labels[idx], preds[idx], average="macro"))
    return float(np.std(scores)), scores


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    e_max, s = DAISEE_PRIMARY_E_MAX, S

    seed_level_values = {}
    item_level_se = {}
    log_lines = []

    for condition in CONDITIONS:
        print(f"=== {condition} ===", flush=True)
        train_ds = to_device(DAiSEEDataset("train", condition, run_seed=42), device)
        val_ds = to_device(DAiSEEDataset("val", condition, run_seed=42), device)

        seed_scores = []
        for seed_i, seed in enumerate(SEEDS):
            torch.manual_seed(seed)
            model = build_model(condition, RECIPE["dropout"], e_max, s).to(device)
            assert_no_backbone_params(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                                           weight_decay=RECIPE["weight_decay"])
            for epoch in range(RECIPE["epochs"]):
                run_epoch(model, train_ds, RECIPE["batch_size"], device, optimizer, shuffle=True)
            logits, labels = eval_logits(model, val_ds)
            preds = logits.argmax(axis=-1)
            f1 = f1_score(labels, preds, average="macro")
            seed_scores.append(f1)
            if seed_i == 0:
                se_item, _ = bootstrap_item_se(logits, labels)
                item_level_se[condition] = se_item
        seed_level_values[condition] = seed_scores
        line = (f"  seed scores: {[f'{x:.4f}' for x in seed_scores]}, "
                f"item-level SE (seed42 model, {N_BOOTSTRAP} boot): {item_level_se[condition]:.4f}")
        print(line, flush=True)
        log_lines.append(f"=== {condition} ===\n{line}")

    seed_level_var = {c: float(np.var(v, ddof=1)) for c, v in seed_level_values.items()}
    pooled_var_seed = float(np.mean(list(seed_level_var.values())))
    pooled_var_item = float(np.mean([se ** 2 for se in item_level_se.values()]))
    pooled_SE = math.sqrt(pooled_var_seed + pooled_var_item)
    effect_floor = max(0.02, 2 * pooled_SE)
    effect_floor_rounded = math.ceil(effect_floor * 100) / 100
    underpowered = effect_floor_rounded > 0.02

    summary = (f"\npooled_var_seed={pooled_var_seed:.6f} pooled_var_item={pooled_var_item:.6f}\n"
               f"pooled_SE={pooled_SE:.4f}\n"
               f"EFFECT_FLOOR = max(0.02, 2*{pooled_SE:.4f}) = {effect_floor:.4f} "
               f"-> rounded up to {effect_floor_rounded:.2f}\n"
               f"underpowered for the original 0.02 target: {underpowered}")
    print(summary, flush=True)
    log_lines.append(summary)

    report = {
        "conditions": CONDITIONS, "seeds": SEEDS, "recipe": RECIPE, "n_bootstrap": N_BOOTSTRAP,
        "seed_level_values": seed_level_values, "seed_level_var": seed_level_var,
        "item_level_se": item_level_se,
        "pooled_var_seed": pooled_var_seed, "pooled_var_item": pooled_var_item,
        "pooled_SE": pooled_SE, "effect_floor_raw": effect_floor,
        "effect_floor_rounded": effect_floor_rounded, "underpowered_for_0.02": underpowered,
    }
    with open(os.path.join(OUT_DIR, "effect_floor_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUT_DIR, "run.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nsaved -> {OUT_DIR}/effect_floor_report.json")


if __name__ == "__main__":
    main()
