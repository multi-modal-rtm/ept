"""Phase 2: detection + tracking over DAiSEE, reusing detect_track.py's core
(same detector/tracker/threading fixes). DAiSEE is the E=1 control (PLAN.md §4) —
tracking still matters for consistent entity cropping even with a single subject.

Align by ClipID from the label CSVs, never by directory listing (file counts
exceed label counts here too, same shape as the MELD quirk documented in
CLAUDE.md): Train 5482 files/5358 labels, Validation 1720/1429, Test 1866/1784.
All labeled ClipIDs resolve to a file on disk (checked; mixed .avi/.mp4 extensions
in the same split, some label rows were briefly thought "missing" before that mix
was accounted for).
"""
import csv
import glob
import os
import time
from multiprocessing import Pool

from ept.tokenization.detect_track import (
    T,
    _init_worker,
    detect_and_track,
)
import json

DATA_ROOT = "/home/devops/data/DAiSEE"
CACHE_ROOT = "/home/devops/ept/cache/tracks/daisee"
SPLIT_DIRS = {"train": "Train", "val": "Validation", "test": "Test"}
LABEL_CSVS = {"train": "TrainLabels.csv", "val": "ValidationLabels.csv", "test": "TestLabels.csv"}
LABEL_FIELDS = ["Boredom", "Engagement", "Confusion", "Frustration "]


def build_file_index(split_dir):
    files = glob.glob(os.path.join(DATA_ROOT, "DataSet", split_dir, "*", "*", "*.avi")) + \
            glob.glob(os.path.join(DATA_ROOT, "DataSet", split_dir, "*", "*", "*.mp4"))
    return {os.path.basename(f): f for f in files}


def load_manifest():
    manifest = []
    for split, split_dir in SPLIT_DIRS.items():
        index = build_file_index(split_dir)
        csv_path = os.path.join(DATA_ROOT, "Labels", LABEL_CSVS[split])
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                clip_id_raw = row["ClipID"]
                full_path = index.get(clip_id_raw)
                if full_path is None:
                    raise ValueError(f"labeled clip not found on disk: {clip_id_raw} ({split})")
                manifest.append({
                    "clip_id_raw": clip_id_raw,
                    "full_path": full_path,
                    "split": split,
                    "labels": {k.strip(): int(row[k]) for k in LABEL_FIELDS},
                })
    return manifest


def clip_id_for(clip_id_raw):
    return os.path.splitext(clip_id_raw)[0]


def _process_clip(entry):
    cid = clip_id_for(entry["clip_id_raw"])
    split = entry["split"]
    out_dir = os.path.join(CACHE_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cid}.json")

    t0 = time.time()
    records = detect_and_track(entry["full_path"], T)
    elapsed = time.time() - t0

    payload = {
        "clip_id": cid,
        "rel_path": os.path.relpath(entry["full_path"], DATA_ROOT),
        "split": split,
        "labels": entry["labels"],
        "t": T,
        "n_frames_grabbed": len(records),
        "detector": "yolo11n-person",
        "tracker": "supervision.ByteTrack",
        "wall_clock_seconds": elapsed,
        "frames": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return {"clip_id": cid, "split": split, "elapsed": elapsed, "n_frames": len(records)}


def main():
    n_workers = 20
    manifest = load_manifest()
    print(f"manifest size: {len(manifest)} clips across "
          f"{[(s, sum(1 for m in manifest if m['split']==s)) for s in ['train','val','test']]}")

    t0 = time.time()
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        results = pool.map(_process_clip, manifest, chunksize=4)
    wall_clock = time.time() - t0

    print(f"detect_track (DAiSEE) wall clock: {wall_clock:.1f}s for {len(manifest)} clips "
          f"({n_workers} workers)")
    os.makedirs(CACHE_ROOT, exist_ok=True)
    with open(os.path.join(CACHE_ROOT, "run_summary.json"), "w") as f:
        json.dump(
            {"n_clips": len(manifest), "n_workers": n_workers,
             "wall_clock_seconds": wall_clock, "results": results},
            f,
        )
    print(f"saved -> {CACHE_ROOT}/run_summary.json")


if __name__ == "__main__":
    main()
