"""DAiSEE TEST split loader — Phase 4, the one-shot test evaluation
(docs/DECISION_RULES_DAISEE.md). Deliberately its OWN class, separate from
DAiSEEDataset (src/ept/train/dataset_daisee.py, train/val only, relied on by
Phase 3) so test-split access stays isolated to one grep-able file -- "the
test split is touched once" needs to be auditable, not just true by
convention. Nothing in the Phase 3 search path imports this module; only
scripts/daisee_phase4_test_eval.py does.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ept.train.dataset import CONDITIONS
from ept.train.dataset_daisee import DAISEE_PRIMARY_E_MAX, FEATURES_GRID_ROOT, FEATURES_ROOT, S, TRACKS_ROOT


class DAiSEETestDataset(Dataset):
    DATASET_NAME = "daisee"

    def __init__(self, condition):
        assert condition in ("A0", "A1"), f"H2-only track: expected A0 or A1, got {condition!r}"
        self.split = "test"
        self.condition = condition
        self.e_max = DAISEE_PRIMARY_E_MAX
        self.features_root = FEATURES_ROOT

        tracks_dir = os.path.join(TRACKS_ROOT, "test")
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
                feat = np.load(os.path.join(FEATURES_GRID_ROOT, "test", f"{clip_id}.npy"))
                mask = np.ones(feat.shape[:2], dtype=bool)
            else:
                feat_full = np.load(os.path.join(FEATURES_ROOT, "test", f"{clip_id}.npy"))
                mask_full = np.load(os.path.join(FEATURES_ROOT, "test", f"{clip_id}_mask.npy"))
                feat = feat_full[: self.e_max]
                mask = mask_full[: self.e_max]
            feats.append(feat.astype(np.float32))
            masks.append(mask)

        self.features = torch.from_numpy(np.stack(feats))  # [N, E, S, D]
        self.masks = torch.from_numpy(np.stack(masks))  # [N, E, S]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx], self.labels[idx]
