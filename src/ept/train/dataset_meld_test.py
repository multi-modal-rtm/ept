"""MELD TEST split loader — Phase 4, the one-shot test evaluation
(docs/DECISION_RULES.md). Deliberately its OWN class, separate from
MELDDataset (src/ept/train/dataset_meld.py, which only allows train/dev and
is relied on by every dev-touching script in this project). Test-split
access is isolated to this one file on purpose: "the test split is touched
once" needs to be auditable by grep, not just true by convention — nothing
in the recipe-search, calibration, or dev-training code paths imports this
module, and this module is only ever imported by scripts/meld_phase4_*.py.

Same preload-to-RAM / A2-shuffle logic as MELDDataset, differing only in
which split is hardcoded, and that clip_ids are retained on the instance
(needed so the paired bootstrap can join predictions across conditions and
against the purity terciles by clip_id, not by trusting index order alone).
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ept.tokenization.extract_features_meld import MELD_PRIMARY_E_MAX
from ept.tokenization.mask_ops import shuffle_entities_per_segment
from ept.train.dataset import CONDITIONS, deterministic_seed
from ept.train.dataset_meld import FEATURES_GRID_ROOT, FEATURES_ROOT, TRACKS_ROOT

DATASET_NAME = "meld"


class MELDTestDataset(Dataset):
    DATASET_NAME = "meld"

    def __init__(self, condition, run_seed=42):
        assert condition in CONDITIONS, f"unknown condition {condition!r}, expected one of {CONDITIONS}"
        self.split = "test"
        self.condition = condition
        self.run_seed = run_seed
        self.e_max = MELD_PRIMARY_E_MAX
        self.features_root = FEATURES_ROOT

        tracks_dir = os.path.join(TRACKS_ROOT, "test")
        clip_ids, labels = [], []
        for fp in sorted(glob.glob(os.path.join(tracks_dir, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            clip_ids.append(clip["clip_id"])
            labels.append(clip["label"])
        self.clip_ids = clip_ids
        self.labels = torch.tensor(labels, dtype=torch.long)

        feats, masks = [], []
        for clip_id in clip_ids:
            if condition == "A0":
                feat = np.load(os.path.join(FEATURES_GRID_ROOT, "test", f"{clip_id}.npy"))
                mask = np.ones(feat.shape[:2], dtype=bool)
            else:
                feat_full = np.load(os.path.join(FEATURES_ROOT, "test", f"{clip_id}.npy"))
                mask_full = np.load(os.path.join(FEATURES_ROOT, "test", f"{clip_id}_mask.npy"))
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
