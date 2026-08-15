"""MELD dataset for Phase 3-for-MELD recipe search. Mirrors
src/ept/train/dataset.py's OUCCGEDataset exactly (same preload-to-RAM
rationale, same condition set, same A2 per-sample shuffle), differing only in
root paths, E_max (primary E=6, sliced from the E_max=8 cache per
MELD_CACHE_E_MAX/MELD_PRIMARY_E_MAX in extract_features_meld.py), and which
splits are loadable. Only `train`/`dev` are ever loadable here — test lives in
a separate, deliberately absent code path; test is a phase-gate event, not
something a Dataset class should make easy to reach.
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

FEATURES_ROOT = "/home/devops/ept/cache/features/meld"
FEATURES_GRID_ROOT = "/home/devops/ept/cache/features_grid/meld"
TRACKS_ROOT = "/home/devops/ept/cache/tracks/meld"


class MELDDataset(Dataset):
    DATASET_NAME = "meld"

    def __init__(self, split, condition, run_seed=42):
        assert split in ("train", "dev"), (
            f"only train/dev are loadable from this class — got split={split!r}. "
            "Test is a phase-gate event, not something a Dataset class should make easy to reach."
        )
        assert condition in CONDITIONS, f"unknown condition {condition!r}, expected one of {CONDITIONS}"
        self.split = split
        self.condition = condition
        self.run_seed = run_seed
        self.e_max = MELD_PRIMARY_E_MAX  # 6 — primary condition, regardless of the
        # cache's E_max=8 (mirrors CLAUDE.md "Feature cache E_max" for OUC-CGE).
        # Recorded so a caller can verify, at construction time, that the dataset it got
        # is actually the one its config asked for (src/ept/train/train.py's
        # assert_dataset_matches_config) instead of trusting a hardcoded default.
        self.features_root = FEATURES_ROOT

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
