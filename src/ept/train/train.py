"""Phase 3/4 shared training loop, per .claude/skills/project-conventions.md and
CLAUDE.md non-negotiables: every run writes outputs/<run_id>/metrics.json + a
resolved-config snapshot, gets its own results_dir, and Phase 3/pre-Phase-4
evaluates on the dataset's own dev/val split only — this file has no code path
that can load a test split (both OUCCGEDataset and MELDDataset assert their
split argument against a train/{val,dev} allowlist that excludes test).

Dataset is selected by `cfg.dataset` (configs/train.yaml declares it `???` --
mandatory, no default) rather than a hardcoded class reference: the locked
recipe this script eventually runs feeds a one-shot test evaluation, and a
silently-wrong dataset there would be exactly the kind of mistake that isn't
caught by anything short of actually reading the code (see logs/GATES.md,
2026-08-15 provenance-audit gate). assert_dataset_matches_config is the
construction-time guard against that; tests/test_dataset_selection.py locks it.
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
from ept.tokenization.extract_features_meld import MELD_PRIMARY_E_MAX, S as MELD_S
from ept.train.dataset import OUCCGEDataset
from ept.train.dataset_meld import MELDDataset

REPO_ROOT = "/home/devops/ept"

# Registry, not an if/elif chain baked into main() -- adding a dataset means adding
# one entry here, not hunting for every place a class name was written by hand.
DATASET_CLASSES = {"ouccge": OUCCGEDataset, "meld": MELDDataset}
EVAL_SPLIT_NAME = {"ouccge": "val", "meld": "dev"}
DATASET_GEOMETRY = {
    "ouccge": {"e_max": 8, "s": 8},
    "meld": {"e_max": MELD_PRIMARY_E_MAX, "s": MELD_S},
}


def assert_dataset_matches_config(dataset_name, ds):
    """Construction-time check that the Dataset object actually built is the
    one `dataset_name` (from config) named -- not a hardcoded default or a
    copy-paste of the wrong class. Two independent checks, not one: the
    class's own declared identity (DATASET_NAME), AND that the config name
    literally appears in the cache path features were read from, so the two
    can't silently drift apart if one is ever edited without the other."""
    assert ds.DATASET_NAME == dataset_name, (
        f"config declares dataset={dataset_name!r} but the constructed dataset "
        f"object identifies itself as {ds.DATASET_NAME!r} -- this would silently "
        f"train/evaluate on the wrong dataset's cache"
    )
    assert dataset_name in ds.features_root, (
        f"dataset={dataset_name!r} does not appear in the cache path "
        f"{ds.features_root!r} features were actually loaded from"
    )


def build_model(condition, dropout, e_max=8, s=8):
    """e_max/s must match the dataset's actual geometry (OUC-CGE: 8x8, MELD:
    6x4). mask_only's input layer is E*S-shaped and needs the exact value.
    EPTFormer's segment-position table is only ever sliced to the first `s`
    rows (verified: a sinusoidal, non-learned buffer -- values depend on
    position index alone, not on the table's allocated size, so passing a
    larger-than-needed s_max here is provably harmless) but is still passed
    explicitly rather than left on its own 8-default, so a future geometry
    with S > 8 fails loudly here instead of silently relying on headroom that
    happens to still be big enough."""
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

    dataset_name = cfg.dataset  # mandatory in configs/train.yaml (dataset: ???); no
    # in-code default, so an invocation that forgets to pass dataset=... fails at
    # Hydra config resolution, not by silently picking one.
    assert dataset_name in DATASET_CLASSES, (
        f"unknown dataset {dataset_name!r} in config, expected one of {sorted(DATASET_CLASSES)}"
    )
    DatasetClass = DATASET_CLASSES[dataset_name]
    eval_split = EVAL_SPLIT_NAME[dataset_name]
    geometry = DATASET_GEOMETRY[dataset_name]

    condition = cfg.condition.name
    train_ds = DatasetClass("train", condition, run_seed=cfg.seed)
    eval_ds = DatasetClass(eval_split, condition, run_seed=cfg.seed)
    assert_dataset_matches_config(dataset_name, train_ds)
    assert_dataset_matches_config(dataset_name, eval_ds)
    train_ds = to_device(train_ds, device)
    eval_ds = to_device(eval_ds, device)

    model = build_model(
        condition, cfg.recipe.dropout, e_max=geometry["e_max"], s=geometry["s"]
    ).to(device)
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
            model, eval_ds, condition, cfg.recipe.batch_size, device, optimizer=None, shuffle=False
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
        "run_id": cfg.run_id, "dataset": dataset_name, "condition": condition, "seed": cfg.seed,
        "recipe": OmegaConf.to_container(cfg.recipe, resolve=True),
        "split_evaluated": eval_split,
        "n_train": len(train_ds), "n_val": len(eval_ds),
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
