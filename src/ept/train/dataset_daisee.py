"""DAiSEE dataset for the H2-replication track (docs/DECISION_RULES_DAISEE.md).
Mirrors src/ept/train/dataset_meld.py's structure (same preload-to-RAM
rationale, same A2-shuffle mechanics available though not used by this
track's H2-only scope), differing in: label field (`labels["Engagement"]`,
4-class, not a flat `label` key), locked E_max=1 (not 6 or 8 -- DAiSEE is
single-subject; CLAUDE.md's existing "E=1 by construction" note), and split
names (train/val, not train/dev -- DAiSEE's official split naming). Only
train/val are ever loadable here; test is this track's own later, separate
gate event, same discipline as every other dataset in this project.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ept.tokenization.mask_ops import shuffle_entities_per_segment
from ept.train.dataset import CONDITIONS, deterministic_seed

DAISEE_PRIMARY_E_MAX = 1  # locked, docs/DECISION_RULES_DAISEE.md
S = 8  # DAiSEE's existing cache convention (extract_features.py's module-level S=8,T=32)
DATASET_NAME = "daisee"

FEATURES_ROOT = "/home/devops/ept/cache/features/daisee"
FEATURES_GRID_ROOT = "/home/devops/ept/cache/features_grid/daisee"
TRACKS_ROOT = "/home/devops/ept/cache/tracks/daisee"


class DAiSEEDataset(Dataset):
    DATASET_NAME = "daisee"

    def __init__(self, split, condition, run_seed=42):
        assert split in ("train", "val"), (
            f"only train/val are loadable from this class — got split={split!r}. "
            "Test is this track's own separate, later gate event."
        )
        assert condition in CONDITIONS, f"unknown condition {condition!r}, expected one of {CONDITIONS}"
        self.split = split
        self.condition = condition
        self.run_seed = run_seed
        self.e_max = DAISEE_PRIMARY_E_MAX
        self.features_root = FEATURES_ROOT

        tracks_dir = os.path.join(TRACKS_ROOT, split)
        clip_ids, labels = [], []
        for fp in sorted(glob.glob(os.path.join(tracks_dir, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            clip_ids.append(clip["clip_id"])
            labels.append(clip["labels"]["Engagement"])
        self.clip_ids = clip_ids
        self.labels = torch.tensor(labels, dtype=torch.long)

        feats, masks = [], []
        for clip_id in clip_ids:
            if condition == "A0":
                feat = np.load(os.path.join(FEATURES_GRID_ROOT, split, f"{clip_id}.npy"))
                mask = np.ones(feat.shape[:2], dtype=bool)
            else:
                feat_full = np.load(os.path.join(FEATURES_ROOT, split, f"{clip_id}.npy"))
                mask_full = np.load(os.path.join(FEATURES_ROOT, split, f"{clip_id}_mask.npy"))
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
