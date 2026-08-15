"""Token-budget sweep dataset (docs/PLAN.md Sec.5 / DECISION_RULES.md
token-budget-sweep amendment): E in {1,2,4,6,8} x S in {2,4}, condition
fixed to A1's architecture (full temporal+social attention, entity tokens,
no shuffle) -- the sweep asks "how much does the token budget cost/buy for
the primary condition," not a re-run of the A0-A5 comparison.

E sweep is a plain slice of the E_max=8 cache -- no approximation.

S sweep is derived from the existing S=4 cache by merging adjacent segment
pairs, NOT a fresh extraction: "secondary sweep from the SAME cache"
(docs/PLAN.md Sec.5). Per extract_features_meld.py's segment_of(), S=4 at
T=16 maps frames {0-3,4-7,8-11,12-15} to segments {0,1,2,3}; S=2 at T=16
would group the SAME physical frames into {0-7}->seg0, {8-15}->seg1 -- i.e.
merging original segments {0,1}->new 0 and {2,3}->new 1 reproduces a fresh
S=2 extraction's frame groupings exactly. The one real approximation: the
cache stores each segment's mean feature, not its raw per-segment frame
count, so merging two PRESENT segments here is an unweighted average of
their means, not a frame-count-weighted one (a fresh extraction would weight
by how many frames actually landed in each half-segment). Documented, not
hidden. If only one of the pair is present, that one is used as-is (no
[ABSENT]-filler contamination); if neither is present, the merged segment
stays absent.

Loads train AND test (unlike MELDDataset, which is train/dev only) --
Phase 4's budget sweep is part of the one-shot test evaluation, same as the
main matrix.
"""
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from ept.tokenization.extract_features_meld import MELD_CACHE_E_MAX
from ept.train.dataset import deterministic_seed
from ept.train.dataset_meld import FEATURES_ROOT, TRACKS_ROOT
from ept.tokenization.mask_ops import shuffle_entities_per_segment  # noqa: F401 (not used; A1 never shuffles)

DATASET_NAME = "meld"


def _merge_to_s2(feat4, mask4):
    """feat4: [E,4,D] float32, mask4: [E,4] bool -> feat2: [E,2,D], mask2: [E,2]."""
    E, _, D = feat4.shape
    feat2 = np.zeros((E, 2, D), dtype=feat4.dtype)
    mask2 = np.zeros((E, 2), dtype=bool)
    for new_seg, (a, b) in enumerate([(0, 1), (2, 3)]):
        ma, mb = mask4[:, a], mask4[:, b]
        both = ma & mb
        only_a = ma & ~mb
        only_b = ~ma & mb
        feat2[both, new_seg] = (feat4[both, a] + feat4[both, b]) / 2.0
        feat2[only_a, new_seg] = feat4[only_a, a]
        feat2[only_b, new_seg] = feat4[only_b, b]
        mask2[:, new_seg] = ma | mb
    return feat2, mask2


class MELDBudgetDataset(Dataset):
    DATASET_NAME = "meld"

    def __init__(self, split, e, s, run_seed=42):
        assert split in ("train", "test"), f"budget sweep only trains on train, evaluates on test, got {split!r}"
        assert s in (2, 4), f"s must be 2 or 4 (derived from the S=4 cache), got {s!r}"
        assert 1 <= e <= MELD_CACHE_E_MAX, f"e must be in [1,{MELD_CACHE_E_MAX}], got {e!r}"
        self.split = split
        self.e = e
        self.s = s
        self.run_seed = run_seed
        self.features_root = FEATURES_ROOT

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
        for clip_id in clip_ids:
            feat_full = np.load(os.path.join(FEATURES_ROOT, split, f"{clip_id}.npy")).astype(np.float32)
            mask_full = np.load(os.path.join(FEATURES_ROOT, split, f"{clip_id}_mask.npy"))
            feat4 = feat_full[:e]
            mask4 = mask_full[:e]
            if s == 4:
                feat, mask = feat4, mask4
            else:
                feat, mask = _merge_to_s2(feat4, mask4)
            feats.append(feat)
            masks.append(mask)

        self.features = torch.from_numpy(np.stack(feats))  # [N, e, s, D]
        self.masks = torch.from_numpy(np.stack(masks))  # [N, e, s]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.masks[idx], self.labels[idx]
