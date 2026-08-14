"""Locks the property the E_max=16 extension depends on: extracting with
e_max=16 and slicing to [:8] must be bitwise identical to extracting with
e_max=8 directly. If this ever breaks, the pre-registered primary condition
(A1, E=8) silently stops being reproducible from the extended cache — see
CLAUDE.md and docs/DECISION_RULES.md's 2026-08-14 token-budget-sweep amendment.

Requires CUDA + the real tracking cache (skips otherwise, e.g. in a CI box
without a GPU) — this is deliberately an integration-level test of the actual
pipeline, not a synthetic unit test, because the risk being guarded against
(GPU kernel batch-composition sensitivity affecting fp16 numerics) can only be
observed by actually running the model.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch

from ept.tokenization.extract_features import (
    decode_needed_frames,
    extract_entity_features,
    load_model,
)

TRACKS_ROOT = "/home/devops/ept/cache/tracks"
DATA_ROOT = "/home/devops/data/OUC-CGE"
N_CLIPS = 10


def _sample_clips():
    files = sorted(glob.glob(os.path.join(TRACKS_ROOT, "train", "*.json")))
    picks = []
    for fp in files:
        with open(fp) as f:
            clip = json.load(f)
        video_path = os.path.join(DATA_ROOT, clip["rel_path"].replace("videos/", ""))
        picks.append((clip, video_path))
        if len(picks) >= N_CLIPS:
            break
    return picks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_e8_slice_of_e16_is_bitwise_identical_to_e8():
    if not os.path.isdir(os.path.join(TRACKS_ROOT, "train")):
        pytest.skip("no tracking cache available")

    model = load_model()
    clips = _sample_clips()
    assert len(clips) == N_CLIPS, f"expected {N_CLIPS} sample clips, found {len(clips)}"

    mismatches = []
    for clip, video_path in clips:
        frames = decode_needed_frames(video_path, clip)
        feat8, mask8, scores8 = extract_entity_features(model, frames, clip, e_max=8)
        feat16, mask16, scores16 = extract_entity_features(model, frames, clip, e_max=16)

        feat_ok = np.array_equal(feat16[:8], feat8)
        mask_ok = np.array_equal(mask16[:8], mask8)
        scores_ok = np.array_equal(scores16[:8], scores8)
        if not (feat_ok and mask_ok and scores_ok):
            mismatches.append({
                "clip_id": clip["clip_id"],
                "feat_ok": feat_ok, "mask_ok": mask_ok, "scores_ok": scores_ok,
                "max_feat_abs_diff": float(np.abs(
                    feat16[:8].astype(np.float32) - feat8.astype(np.float32)
                ).max()) if not feat_ok else 0.0,
            })

    assert not mismatches, (
        f"e_max=16 sliced to [:8] is NOT identical to e_max=8 for "
        f"{len(mismatches)}/{N_CLIPS} clips: {mismatches}"
    )


if __name__ == "__main__":
    test_e8_slice_of_e16_is_bitwise_identical_to_e8()
    print("slice-equivalence test PASSED")
