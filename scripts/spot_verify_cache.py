"""Phase 2 gate: spot-verify N clips by reloading the saved cache and comparing
against a fresh, independent forward pass (not the same in-memory result —
this checks for save/load corruption, not just algorithmic correctness, which
tests/test_e_max_slice_equivalence.py already covers separately).
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ept.tokenization.extract_features import (
    LOCKED_PRIMARY_E_MAX,
    OUCCGE_E_MAX,
    OUCCGE_DATA_ROOT,
    DAISEE_DATA_ROOT,
    decode_needed_frames,
    extract_entity_features,
    extract_grid_features,
    load_model,
)

N_SPOT = 5


def main():
    with open("/home/devops/ept/cache/MANIFEST.json") as f:
        manifest = json.load(f)
    entries = manifest["entries"]

    rng = random.Random(42)
    picks = rng.sample(entries, N_SPOT)

    model = load_model()
    all_ok = True
    for entry in picks:
        dataset, split, clip_id, e_max = entry["dataset"], entry["split"], entry["clip_id"], entry["e_max"]
        tracks_dir = ("/home/devops/ept/cache/tracks/daisee/" + split if dataset == "daisee"
                      else "/home/devops/ept/cache/tracks/" + split)
        with open(os.path.join(tracks_dir, f"{clip_id}.json")) as f:
            clip = json.load(f)
        data_root = DAISEE_DATA_ROOT if dataset == "daisee" else OUCCGE_DATA_ROOT
        video_path = os.path.join(
            data_root,
            clip["rel_path"] if dataset == "daisee" else clip["rel_path"].replace("videos/", ""),
        )

        cached_feat = np.load(entry["entity_path"])
        cached_mask = np.load(entry["mask_path"])
        cached_scores = np.load(entry["scores_path"])
        cached_grid = np.load(entry["grid_path"])

        frames = decode_needed_frames(video_path, clip)
        fresh_feat, fresh_mask, fresh_scores = extract_entity_features(model, frames, clip, e_max)
        fresh_grid = extract_grid_features(model, frames, clip)

        ok_feat = np.array_equal(cached_feat, fresh_feat)
        ok_mask = np.array_equal(cached_mask, fresh_mask)
        ok_scores = np.array_equal(cached_scores, fresh_scores)
        ok_grid = np.array_equal(cached_grid, fresh_grid)
        ok = ok_feat and ok_mask and ok_scores and ok_grid
        all_ok = all_ok and ok
        print(f"{dataset}/{split}/{clip_id} (e_max={e_max}): "
              f"feat={'OK' if ok_feat else 'MISMATCH'} mask={'OK' if ok_mask else 'MISMATCH'} "
              f"scores={'OK' if ok_scores else 'MISMATCH'} grid={'OK' if ok_grid else 'MISMATCH'}")

    print()
    if all_ok:
        print(f"SPOT-VERIFY PASSED: all {N_SPOT} clips exactly match a fresh forward pass")
    else:
        print(f"SPOT-VERIFY FAILED: mismatch found among {N_SPOT} clips")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
