"""Driver for MELD feature extraction (Phase-2 equivalent). Same
multiprocessing-GPU-sharing pattern as extract_features.py, adapted for MELD's
geometry (E_max=8 cached, S=4, T=16).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import torch

from ept.tokenization.extract_features import load_model, sha256_of_file
from ept.tokenization.extract_features_meld import (
    iter_meld_clips, process_clip, MELD_CACHE_E_MAX, FEATURES_ROOT, FEATURES_GRID_ROOT,
)

_MODEL = None


def _init_worker():
    global _MODEL
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _MODEL = load_model()


def _process_clip_mp(args):
    split, clip_id, video_path, clip = args
    return process_clip(_MODEL, split, clip_id, video_path, clip, e_max=MELD_CACHE_E_MAX)


def main():
    from multiprocessing import Pool

    all_clips = list(iter_meld_clips())
    print(f"total clips: {len(all_clips)}")

    n_workers = 8
    t0 = time.time()
    n_done = 0
    entries = []
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        for entry in pool.imap_unordered(_process_clip_mp, all_clips, chunksize=4):
            entries.append(entry)
            n_done += 1
            if n_done % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {n_done} clips done, {elapsed:.1f}s elapsed, {elapsed/n_done:.3f}s/clip amortized")

    wall_clock = time.time() - t0
    print(f"MELD extraction wall clock: {wall_clock:.1f}s for {n_done} clips "
          f"({wall_clock/n_done:.3f}s/clip amortized, {n_workers} workers)")

    manifest_path = "/home/devops/ept/cache/MELD_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump({"n_clips": n_done, "wall_clock_seconds": wall_clock, "n_workers": n_workers,
                    "e_max_cached": MELD_CACHE_E_MAX, "entries": entries}, f)
    print(f"saved -> {manifest_path}")


if __name__ == "__main__":
    main()
