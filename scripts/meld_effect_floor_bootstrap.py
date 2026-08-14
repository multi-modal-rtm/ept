"""Compute the effect floor per the procedure fixed in docs/DECISION_RULES.md's
2026-08-14 "Primary dataset changed" amendment, committed BEFORE this script was
run. Blind to condition ranking throughout: only dispersion (standard errors) is
ever inspected or recorded — no comparison between conditions' point estimates
appears anywhere in this script's output.

Procedure (verbatim from the committed amendment):
1. Estimate seed-level and item-level variance of macro-F1 on MELD dev via
   bootstrap, using only conditions already implemented (A0-A5, mask-only).
2. EFFECT_FLOOR = max(0.02, 2 x pooled_SE), rounded UP to two decimals.
3. (Paired requirement -- recorded in the amendment, not computed here; it
   applies at Phase-4 test time, not to this floor-setting step.)
4. If pooled_SE implies 0.02 can't be detected, say so.

Fixed, reasonable recipe (NOT tuned for this purpose, NOT a new locked recipe):
r07 from docs/SEARCH_GRID.md (lr=1e-4, wd=0.0, dropout=0.0, epochs=50) -- the
single most frequently-best point across the OUC-CGE search, used here only
because a *some* fixed recipe is needed to produce trained models to bootstrap
from, not because it has been validated for MELD's geometry.
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
from torch.utils.data import Dataset

from ept.model.ept_former import EPTFormer, MeanPoolMLP, MaskOnlyMLP, assert_no_backbone_params
from ept.tokenization.extract_features_meld import (
    MELD_PRIMARY_E_MAX, S, FEATURES_ROOT, TRACKS_ROOT,
)

import glob

CONDITIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "mask_only"]
SEEDS = [42, 1337, 2024]
RECIPE = {"lr": 1e-4, "weight_decay": 0.0, "dropout": 0.0, "epochs": 50, "batch_size": 32}
N_BOOTSTRAP = 500
OUT_DIR = "/home/devops/ept/outputs/meld_effect_floor"

FEATURES_GRID_ROOT = "/home/devops/ept/cache/features_grid/meld"


def deterministic_seed(run_seed, clip_id):
    import hashlib
    h = hashlib.sha256(f"{run_seed}:{clip_id}".encode()).hexdigest()
    return int(h[:8], 16)


class MELDDataset(Dataset):
    def __init__(self, split, condition, run_seed=42, e_max=MELD_PRIMARY_E_MAX):
        assert split in ("train", "dev")
        from ept.tokenization.mask_ops import shuffle_entities_per_segment
        tracks_dir = os.path.join(TRACKS_ROOT, split)
        clip_ids, labels = [], []
        for fp in sorted(glob.glob(os.path.join(tracks_dir, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            clip_ids.append(clip["clip_id"])
            labels.append(clip["label"])
        self.clip_ids = clip_ids
        self.labels = torch.tensor(labels, dtype=torch.long)

        feats, masks = [], []
        for cid in clip_ids:
            if condition == "A0":
                feat = np.load(os.path.join(FEATURES_GRID_ROOT, split, f"{cid}.npy"))
                mask = np.ones(feat.shape[:2], dtype=bool)
            else:
                feat_full = np.load(os.path.join(FEATURES_ROOT, split, f"{cid}.npy"))
                mask_full = np.load(os.path.join(FEATURES_ROOT, split, f"{cid}_mask.npy"))
                feat = feat_full[:e_max]
                mask = mask_full[:e_max]
                if condition == "A2":
                    seed = deterministic_seed(run_seed, cid)
                    feat = shuffle_entities_per_segment(feat, seed)
                    mask = shuffle_entities_per_segment(mask, seed)
            feats.append(feat.astype(np.float32))
            masks.append(mask)
        self.features = torch.from_numpy(np.stack(feats))
        self.masks = torch.from_numpy(np.stack(masks))

    def __len__(self):
        return len(self.labels)


def to_device(ds, device):
    ds.features = ds.features.to(device)
    ds.masks = ds.masks.to(device)
    ds.labels = ds.labels.to(device)
    return ds


def build_model(condition, dropout, e_max, s):
    if condition == "A0":
        return EPTFormer(dropout=dropout, use_temporal=True, use_social=True, s_max=s)
    if condition in ("A1", "A2"):
        return EPTFormer(dropout=dropout, use_temporal=True, use_social=True, s_max=s)
    if condition == "A3":
        return EPTFormer(dropout=dropout, use_temporal=True, use_social=False, s_max=s)
    if condition == "A4":
        return EPTFormer(dropout=dropout, use_temporal=False, use_social=True, s_max=s)
    if condition == "A5":
        return MeanPoolMLP(dropout=dropout)
    if condition == "mask_only":
        return MaskOnlyMLP(e_max=e_max, s=s, dropout=dropout)
    raise ValueError(condition)


def forward_pass(model, condition, feat, mask):
    if condition == "mask_only":
        return model(mask)
    return model(feat, mask)


def run_epoch(model, ds, condition, batch_size, device, optimizer=None, shuffle=False):
    training = optimizer is not None
    model.train(training)
    n = len(ds)
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    for i in range(0, n, batch_size):
        idx = order[i:i + batch_size]
        feat, mask, label = ds.features[idx], ds.masks[idx], ds.labels[idx]
        with torch.set_grad_enabled(training):
            logits = forward_pass(model, condition, feat, mask)
            loss = nn.functional.cross_entropy(logits, label)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def eval_logits(model, ds, condition):
    model.eval()
    with torch.no_grad():
        logits = forward_pass(model, condition, ds.features, ds.masks)
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
    e_max, s = MELD_PRIMARY_E_MAX, S

    seed_level_values = {}   # condition -> [macro_f1 per seed]
    item_level_se = {}       # condition -> se (from seed=42's model)

    for condition in CONDITIONS:
        print(f"=== {condition} ===")
        train_ds = to_device(MELDDataset("train", condition, run_seed=42, e_max=e_max), device)
        dev_ds = to_device(MELDDataset("dev", condition, run_seed=42, e_max=e_max), device)

        seed_scores = []
        for seed_i, seed in enumerate(SEEDS):
            torch.manual_seed(seed)
            model = build_model(condition, RECIPE["dropout"], e_max, s).to(device)
            assert_no_backbone_params(model)
            optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE["lr"],
                                           weight_decay=RECIPE["weight_decay"])
            for epoch in range(RECIPE["epochs"]):
                run_epoch(model, train_ds, condition, RECIPE["batch_size"], device, optimizer, shuffle=True)
            logits, labels = eval_logits(model, dev_ds, condition)
            preds = logits.argmax(axis=-1)
            f1 = f1_score(labels, preds, average="macro")
            seed_scores.append(f1)
            if seed_i == 0:
                # item-level bootstrap on the first seed's trained model only
                se_item, _ = bootstrap_item_se(logits, labels)
                item_level_se[condition] = se_item
        seed_level_values[condition] = seed_scores
        print(f"  seed scores: {[f'{x:.4f}' for x in seed_scores]}, "
              f"item-level SE (seed42 model, {N_BOOTSTRAP} boot): {item_level_se[condition]:.4f}")

    seed_level_var = {c: float(np.var(v, ddof=1)) for c, v in seed_level_values.items()}
    pooled_var_seed = float(np.mean(list(seed_level_var.values())))
    pooled_var_item = float(np.mean([se ** 2 for se in item_level_se.values()]))
    pooled_SE = math.sqrt(pooled_var_seed + pooled_var_item)
    effect_floor = max(0.02, 2 * pooled_SE)
    effect_floor_rounded = math.ceil(effect_floor * 100) / 100  # round UP to 2 decimals

    underpowered = effect_floor_rounded > 0.02

    print(f"\npooled_var_seed={pooled_var_seed:.6f} pooled_var_item={pooled_var_item:.6f}")
    print(f"pooled_SE={pooled_SE:.4f}")
    print(f"EFFECT_FLOOR = max(0.02, 2*{pooled_SE:.4f}) = {effect_floor:.4f} -> "
          f"rounded up to {effect_floor_rounded:.2f}")
    print(f"underpowered for the original 0.02 target: {underpowered}")

    report = {
        "conditions": CONDITIONS, "seeds": SEEDS, "recipe": RECIPE,
        "n_bootstrap": N_BOOTSTRAP,
        "seed_level_values": seed_level_values,
        "seed_level_variance_per_condition": seed_level_var,
        "item_level_se_per_condition": item_level_se,
        "pooled_var_seed": pooled_var_seed, "pooled_var_item": pooled_var_item,
        "pooled_SE": pooled_SE, "effect_floor_raw": effect_floor,
        "effect_floor_rounded": effect_floor_rounded,
        "underpowered_for_0.02": underpowered,
    }
    with open(os.path.join(OUT_DIR, "effect_floor_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR}/effect_floor_report.json")


if __name__ == "__main__":
    main()
