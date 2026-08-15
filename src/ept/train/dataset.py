"""OUC-CGE dataset for Phase 3 recipe search / Phase 4 matrix training. Only
`train`/`val` are ever loadable here — test lives in a separate, deliberately
absent code path (src/ept/eval/, not built yet; test is a phase-4 gate event).
"""
import glob
import hashlib
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ept.tokenization.extract_features import LOCKED_PRIMARY_E_MAX
from ept.tokenization.mask_ops import shuffle_entities_per_segment

FEATURES_ROOT = "/home/devops/ept/cache/features"
FEATURES_GRID_ROOT = "/home/devops/ept/cache/features_grid"
TRACKS_ROOT = "/home/devops/ept/cache/tracks"

CONDITIONS = ("A0", "A1", "A2", "A3", "A4", "A5", "mask_only")


def deterministic_seed(run_seed, clip_id):
    """A2's per-sample seed, derived from the run seed (docs/DECISION_RULES.md):
    same run_seed always gives the same shuffle for a given clip (reproducible),
    different clips get different shuffles (not one global permutation)."""
    h = hashlib.sha256(f"{run_seed}:{clip_id}".encode()).hexdigest()
    return int(h[:8], 16)


class OUCCGEDataset(Dataset):
    """Preloads every clip's features/mask into RAM once at construction (the
    whole train+val cache is ~1GB, easily fits) — the first implementation
    re-read two .npy files from disk on every single __getitem__ call, every
    epoch (6160 train clips x 2 files x up to 50 epochs), measured at 427s for
    a single 50-epoch run. Preloading turns __getitem__ into pure tensor
    indexing; the fixed one-time load cost is paid once per process instead of
    once per epoch."""

    DATASET_NAME = "ouccge"

    def __init__(self, split, condition, run_seed=42):
        assert split in ("train", "val"), (
            f"only train/val are loadable from this class — got split={split!r}. "
            "Test is a Phase-4 gate event, not something a Dataset class should make easy to reach."
        )
        assert condition in CONDITIONS, f"unknown condition {condition!r}, expected one of {CONDITIONS}"
        self.split = split
        self.condition = condition
        self.run_seed = run_seed
        self.e_max = LOCKED_PRIMARY_E_MAX  # 8 — the locked primary condition, always,
        # regardless of the cache's E_max=16 (docs/CLAUDE.md "Feature cache E_max").
        # Recorded so a caller can verify, at construction time, that the dataset it got
        # is actually the one its config asked for (src/ept/train/train.py's
        # assert_dataset_matches_config) instead of trusting a hardcoded default.
        self.features_root = os.path.join(FEATURES_ROOT, "ouccge")

        tracks_dir = os.path.join(TRACKS_ROOT, split)
        clip_ids, labels = [], []
        for fp in sorted(glob.glob(os.path.join(tracks_dir, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            clip_ids.append(clip["clip_id"])
            labels.append(clip["label"])
        self.labels = torch.tensor(labels, dtype=torch.long)

        feats, masks = [], []
        for clip_id in clip_ids:
            if condition == "A0":
                feat = np.load(os.path.join(FEATURES_GRID_ROOT, "ouccge", split, f"{clip_id}.npy"))
                mask = np.ones(feat.shape[:2], dtype=bool)
            else:
                feat_full = np.load(os.path.join(FEATURES_ROOT, "ouccge", split, f"{clip_id}.npy"))
                mask_full = np.load(os.path.join(FEATURES_ROOT, "ouccge", split, f"{clip_id}_mask.npy"))
                feat = feat_full[: self.e_max]
                mask = mask_full[: self.e_max]
                if condition == "A2":
                    seed = deterministic_seed(run_seed, clip_id)
                    feat = shuffle_entities_per_segment(feat, seed)
                    mask = shuffle_entities_per_segment(mask, seed)
            feats.append(feat.astype(np.float32))
            masks.append(mask)

        self.features = torch.from_numpy(np.stack(feats))  # [N, E, S, D]
        self.masks = torch.from_numpy(np.stack(masks))  # [N, E, S]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx], self.labels[idx]
