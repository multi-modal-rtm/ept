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

        tracks_dir = os.path.join(TRACKS_ROOT, split)
        self.clip_ids = []
        self.labels = []
        for fp in sorted(glob.glob(os.path.join(tracks_dir, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            self.clip_ids.append(clip["clip_id"])
            self.labels.append(clip["label"])

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        label = self.labels[idx]

        if self.condition == "A0":
            feat = np.load(os.path.join(FEATURES_GRID_ROOT, "ouccge", self.split, f"{clip_id}.npy"))
            mask = np.ones(feat.shape[:2], dtype=bool)
        else:
            feat_full = np.load(os.path.join(FEATURES_ROOT, "ouccge", self.split, f"{clip_id}.npy"))
            mask_full = np.load(os.path.join(FEATURES_ROOT, "ouccge", self.split, f"{clip_id}_mask.npy"))
            feat = feat_full[: self.e_max]
            mask = mask_full[: self.e_max]
            if self.condition == "A2":
                seed = deterministic_seed(self.run_seed, clip_id)
                feat = shuffle_entities_per_segment(feat, seed)
                mask = shuffle_entities_per_segment(mask, seed)

        feat_t = torch.from_numpy(np.ascontiguousarray(feat).astype(np.float32))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask))
        return feat_t, mask_t, label
