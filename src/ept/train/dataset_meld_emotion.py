"""MELD dataset variant for the 7-class emotion secondary calibration endpoint
(docs/DECISION_RULES.md, 2026-08-15 amendment) — SECONDARY, not part of the
branch decision. Identical to MELDDataset (src/ept/train/dataset_meld.py) in
every respect (same cache, same E/S geometry, same A2 shuffle) except which
label is loaded: 7-class emotion instead of 3-class sentiment. Only
train/dev are loadable, matching this endpoint's "dev only" instruction.
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
from ept.train.emotion_labels import load_emotion_labels


class MELDEmotionDataset(Dataset):
    def __init__(self, split, condition, run_seed=42):
        assert split in ("train", "dev"), (
            f"only train/dev are loadable from this class — got split={split!r}. "
            "This is the secondary emotion calibration endpoint's own 'dev only' "
            "instruction, not just the usual test-is-a-gate-event rule."
        )
        assert condition in CONDITIONS, f"unknown condition {condition!r}, expected one of {CONDITIONS}"
        self.split = split
        self.condition = condition
        self.run_seed = run_seed
        self.e_max = MELD_PRIMARY_E_MAX

        emotion_by_id = load_emotion_labels(split)

        tracks_dir = os.path.join(TRACKS_ROOT, split)
        clip_ids, labels = [], []
        for fp in sorted(glob.glob(os.path.join(tracks_dir, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            clip_ids.append(clip["clip_id"])
            labels.append(emotion_by_id[clip["clip_id"]])
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
