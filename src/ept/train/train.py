"""Phase 3/4 shared training loop, per .claude/skills/project-conventions.md and
CLAUDE.md non-negotiables: every run writes outputs/<run_id>/metrics.json + a
resolved-config snapshot, gets its own results_dir, and Phase 3 evaluates on
VAL ONLY — this file has no code path that can load the test split (see
src/ept/train/dataset.py, which asserts split in {train, val}).
"""
import json
import os
import time

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import f1_score

from ept.model.ept_former import EPTFormer, MeanPoolMLP, MaskOnlyMLP, assert_no_backbone_params
from ept.train.dataset import OUCCGEDataset

REPO_ROOT = "/home/devops/ept"


def build_model(condition, dropout):
    if condition == "A0":
        return EPTFormer(dropout=dropout, use_temporal=True, use_social=True)
    if condition in ("A1", "A2"):
        return EPTFormer(dropout=dropout, use_temporal=True, use_social=True)
    if condition == "A3":
        return EPTFormer(dropout=dropout, use_temporal=True, use_social=False)
    if condition == "A4":
        return EPTFormer(dropout=dropout, use_temporal=False, use_social=True)
    if condition == "A5":
        return MeanPoolMLP(dropout=dropout)
    if condition == "mask_only":
        return MaskOnlyMLP(dropout=dropout)
    raise ValueError(f"unknown condition {condition!r}")


def forward_pass(model, condition, feat, mask):
    if condition == "mask_only":
        return model(mask)
    return model(feat, mask)


def to_device(ds, device):
    """Whole dataset resident on GPU once per run — small (~1GB total), and this
    avoids per-batch host->device transfer + DataLoader collation overhead, which
    dominated wall-clock for this model (many small MultiheadAttention calls):
    measured 427s -> 198s from preloading alone, further reduced by this."""
    ds.features = ds.features.to(device)
    ds.masks = ds.masks.to(device)
    ds.labels = ds.labels.to(device)
    return ds


def run_epoch(model, ds, condition, batch_size, device, optimizer=None, shuffle=False):
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
            logits = forward_pass(model, condition, feat, mask)
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
    macro_f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    return total_loss / n, macro_f1


@hydra.main(config_path="../../../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    condition = cfg.condition.name
    train_ds = to_device(OUCCGEDataset("train", condition, run_seed=cfg.seed), device)
    val_ds = to_device(OUCCGEDataset("val", condition, run_seed=cfg.seed), device)

    model = build_model(condition, cfg.recipe.dropout).to(device)
    assert_no_backbone_params(model)  # trivially true (no backbone ref); explicit per CLAUDE.md
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.recipe.lr, weight_decay=cfg.recipe.weight_decay
    )

    results_dir = os.path.join(REPO_ROOT, "outputs", cfg.run_id)
    os.makedirs(results_dir, exist_ok=True)

    history = []
    t0 = time.time()
    best_val_f1 = -1.0
    best_epoch = -1
    for epoch in range(cfg.recipe.epochs):
        train_loss, train_f1 = run_epoch(
            model, train_ds, condition, cfg.recipe.batch_size, device, optimizer, shuffle=True
        )
        val_loss, val_f1 = run_epoch(
            model, val_ds, condition, cfg.recipe.batch_size, device, optimizer=None, shuffle=False
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                         "val_loss": val_loss, "val_macro_f1": val_f1})
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
    wall_clock = time.time() - t0

    with open(os.path.join(results_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    metrics = {
        "run_id": cfg.run_id, "condition": condition, "seed": cfg.seed,
        "recipe": OmegaConf.to_container(cfg.recipe, resolve=True),
        "split_evaluated": "val",
        "n_train": len(train_ds), "n_val": len(val_ds),
        "history": history,
        "best_val_macro_f1": best_val_f1, "best_epoch": best_epoch,
        "final_val_macro_f1": history[-1]["val_macro_f1"],
        "wall_clock_seconds": wall_clock,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{cfg.run_id}] best_val_macro_f1={best_val_f1:.4f} @ epoch {best_epoch} "
          f"({wall_clock:.1f}s)")


if __name__ == "__main__":
    main()
