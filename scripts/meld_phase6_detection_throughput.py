"""Measures detection+clustering throughput at 60-way parallelism (clips/sec,
amortized ms/clip) to report ALONGSIDE the existing single-stream latency
(6276.9ms/clip, measured single-clip single-thread in
scripts/meld_phase6_efficiency.py) -- not a replacement for it. Also times
video-frame-decode separately from SCRFD detection itself, to ground the
E=1 "detection could stop at the largest face" estimate in a real
measurement of where the 6276.9ms actually goes, rather than a guess.

No test evaluation, no training, no labels touched -- this profiles
wall-clock cost of the tokenization pipeline's front end only.
"""
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from ept.tokenization.detect_cluster_meld import (
    LABEL_CSVS, MELD_ROOT, SPLIT_DIRS, T, _init_worker, cluster_identities, detect_faces,
    extract_frames,
)

REPO_ROOT = "/home/devops/ept"
CLUSTER_THRESHOLD = 0.55
N_CLIPS_THROUGHPUT = 600
N_CLIPS_DECODE_SPLIT = 100
N_WORKERS = 60


def _manifest(n_clips):
    manifest = []
    import csv
    with open(os.path.join(MELD_ROOT, "labels", LABEL_CSVS["test"])) as f:
        for row in csv.DictReader(f):
            dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
            fname = f"dia{dia}_utt{utt}.mp4"
            path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS["test"], fname)
            if os.path.exists(path):
                manifest.append(path)
            if len(manifest) >= n_clips:
                break
    return manifest


def _process_one(path):
    t0 = time.time()
    records = detect_faces(path)
    t1 = time.time()
    cluster_identities(records, CLUSTER_THRESHOLD)
    t2 = time.time()
    return t1 - t0, t2 - t1


def measure_throughput():
    manifest = _manifest(N_CLIPS_THROUGHPUT)
    print(f"throughput benchmark: {len(manifest)} clips, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(_process_one, manifest, chunksize=2)
    wall = time.time() - t0
    clips_per_sec = len(manifest) / wall
    amortized_ms_per_clip = wall / len(manifest) * 1000
    return {
        "n_clips": len(manifest), "n_workers": N_WORKERS, "wall_clock_seconds": wall,
        "clips_per_sec": clips_per_sec, "amortized_ms_per_clip": amortized_ms_per_clip,
    }


def _decode_only(path):
    t0 = time.time()
    extract_frames(path, T)
    return time.time() - t0


def _decode_and_detect(path):
    t0 = time.time()
    frames = extract_frames(path, T)
    t1 = time.time()
    from ept.tokenization.detect_cluster_meld import _APP
    for frame_pos, (frame_idx, frame) in enumerate(frames):
        _APP.get(frame)
    t2 = time.time()
    return t1 - t0, t2 - t1


def measure_decode_vs_detect_split():
    """Single-thread, mirrors the single-stream latency benchmark's
    conditions exactly (same _init_worker, same T=16 frames/clip)."""
    _init_worker()
    manifest = _manifest(N_CLIPS_DECODE_SPLIT)
    decode_times, detect_times = [], []
    for path in manifest:
        d, det = _decode_and_detect(path)
        decode_times.append(d)
        detect_times.append(det)
    return {
        "n_clips": len(manifest),
        "decode_ms_mean": float(np.mean(decode_times) * 1000),
        "decode_ms_std": float(np.std(decode_times) * 1000),
        "scrfd_detect_ms_mean": float(np.mean(detect_times) * 1000),
        "scrfd_detect_ms_std": float(np.std(detect_times) * 1000),
        "scrfd_ms_per_frame_mean": float(np.mean(detect_times) * 1000 / T),
    }


def main():
    print("=== decode vs SCRFD-detect split (single-thread, T=16 frames/clip) ===", flush=True)
    split = measure_decode_vs_detect_split()
    print(json.dumps(split, indent=2), flush=True)

    print("\n=== 60-way parallel throughput ===", flush=True)
    tp = measure_throughput()
    print(json.dumps(tp, indent=2), flush=True)

    out = {"decode_vs_detect_split": split, "throughput_60way": tp,
           "single_stream_reference_ms": 6276.9}  # scripts/meld_phase6_efficiency.py, already on disk
    out_path = os.path.join(REPO_ROOT, "outputs", "meld_phase6_detection_throughput.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
