"""Phase 1 step 2: person detection + ByteTrack ID association over OUC-CGE.

For a video path, samples T=32 frames uniformly and returns, per frame, a list of
(track_id, bbox, confidence). YOLO person-class detection (chosen over SCRFD/buffalo_l
after the Phase 1 detector bake-off — see outputs/phase1_detector_bakeoff/REPORT.md:
SCRFD goes fully blind whenever heads are turned away, a plausibly label-correlated
failure mode; YOLO degrades gracefully). supervision ByteTrack for ID association,
reset per clip.
"""
import os

# Must be set before numpy/cv2/torch are imported anywhere in this process. With
# the default "fork" multiprocessing start method, worker processes inherit the
# PARENT's already-initialized native thread pools (BLAS, OpenCV's parallel_for,
# torch's interop pool) — setting these env vars inside the pool initializer,
# after those libraries are already imported at module level, is too late and
# was measured to still oversubscribe badly (20 workers: 6.45s/clip amortized,
# worse than 8 workers' 0.72s/clip). Setting them here, before any of those
# imports below, fixes it at the source.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import csv
import json
import time
from multiprocessing import Pool

import cv2
import numpy as np

DATA_ROOT = "/home/devops/data/OUC-CGE"
CACHE_ROOT = "/home/devops/ept/cache/tracks"
T = 32
YOLO_MODEL = "yolo11n.pt"
PERSON_CLASS = 0

# Two originally-documented bad files + three found during the Phase 1 bake-off
# (see CLAUDE.md "Hard-won facts"). All three additions are train-split only.
EXCLUDE = {
    "videos/low/view2572.mp4",
    "videos/low/view2531.mp4",
    "videos/low/view61.mp4",
    "videos/low/view1579.mp4",
    "videos/high/view1325.mp4",
}
# Non-clip placeholder row present in test.csv.
EXCLUDE_ROWS = {"videos/anonymity/a.mp4"}

CLASS_NAMES = {0: "low", 1: "mid", 2: "high"}
SPLIT_CSVS = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}


def load_manifest():
    """All (rel_path, label, split) triples across train/val/test, excluding bad
    files and the test.csv placeholder row."""
    manifest = []
    for split, csv_name in SPLIT_CSVS.items():
        path = os.path.join(DATA_ROOT, csv_name)
        with open(path) as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if len(row) != 2:
                    if len(row) == 1 and row[0] not in EXCLUDE_ROWS:
                        raise ValueError(f"unexpected single-column row: {row}")
                    continue
                rel_path, label = row[0], int(row[1])
                if rel_path in EXCLUDE:
                    continue
                manifest.append(
                    {
                        "rel_path": rel_path,
                        "label": label,
                        "class_name": CLASS_NAMES[label],
                        "split": split,
                    }
                )
    return manifest


def clip_id_for(rel_path):
    stem = os.path.splitext(os.path.basename(rel_path))[0]
    class_name = rel_path.split("/")[1]  # videos/<class>/<file>
    return f"{class_name}_{stem}"


def sample_frame_indices(n_frames, t):
    if n_frames <= 0:
        return []
    return np.linspace(0, n_frames - 1, t).astype(int).tolist()


def extract_frames(full_path, t):
    cap = cv2.VideoCapture(full_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(n_frames, t)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames


_YOLO = None


def _init_worker():
    global _YOLO
    # Thread-limiting env vars are set at module import time (see top of file) —
    # too late to set here under fork-based multiprocessing. These runtime API
    # calls are a separate mechanism (not env-var-driven) and do still need to
    # happen per-worker.
    import torch
    torch.set_num_threads(1)
    # torch's INTEROP thread pool (distinct from intra-op) defaults to one thread
    # per core and is not covered by set_num_threads. Measured: a single worker at
    # interop=60 (default) hit 614% CPU on one image. Must be set before first use.
    torch.set_num_interop_threads(1)
    # OpenCV's own parallel_for defaults to all-cores-per-process for decode/resize
    # ops, same oversubscription failure mode as onnxruntime in the step-1 bake-off
    # (see CLAUDE.md hard-won facts) but for cv2 instead. Cap it explicitly.
    cv2.setNumThreads(1)
    from ultralytics import YOLO

    _YOLO = YOLO(YOLO_MODEL)


def detect_and_track(full_path, t=T):
    """Core entry point: video path in, per-frame (track_id, bbox, confidence) out.

    Returns a list of dicts, one per successfully-grabbed sampled frame:
        {"frame_pos": int (0..t-1 index into the sample grid),
         "frame_idx": int (original video frame index),
         "detections": [{"track_id": int, "bbox": [x1,y1,x2,y2], "confidence": float}, ...]}
    """
    import supervision as sv

    frames = extract_frames(full_path, t)
    tracker = sv.ByteTrack()  # fresh tracker state per clip
    import torch

    records = []
    for frame_pos, (frame_idx, frame) in enumerate(frames):
        # ultralytics silently resets torch's intra-op thread count to
        # min(8, ncpu) on first predict() (measured: 1 -> 8, matching the 20
        # workers x 8 threads = 160 runnable threads seen in vmstat during the
        # oversubscription incident). Force it back before every call.
        torch.set_num_threads(1)
        result = _YOLO.predict(frame, classes=[PERSON_CLASS], verbose=False, device="cpu")[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        dets_out = []
        for i in range(len(detections)):
            bbox = detections.xyxy[i].tolist()
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            conf = float(detections.confidence[i]) if detections.confidence is not None else None
            dets_out.append({"track_id": track_id, "bbox": bbox, "confidence": conf})
        records.append({"frame_pos": frame_pos, "frame_idx": int(frame_idx), "detections": dets_out})
    return records


def _process_clip(entry):
    rel_path, split = entry["rel_path"], entry["split"]
    full_path = os.path.join(DATA_ROOT, rel_path.replace("videos/", ""))
    cid = clip_id_for(rel_path)
    out_dir = os.path.join(CACHE_ROOT, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{cid}.json")

    t0 = time.time()
    records = detect_and_track(full_path, T)
    elapsed = time.time() - t0

    payload = {
        "clip_id": cid,
        "rel_path": rel_path,
        "split": split,
        "label": entry["label"],
        "class_name": entry["class_name"],
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
          f"{[ (s, sum(1 for m in manifest if m['split']==s)) for s in ['train','val','test'] ]}")

    t0 = time.time()
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        results = pool.map(_process_clip, manifest, chunksize=4)
    wall_clock = time.time() - t0

    print(f"detect_track wall clock: {wall_clock:.1f}s for {len(manifest)} clips "
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
