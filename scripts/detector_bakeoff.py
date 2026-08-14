"""Phase 1 step 1: SCRFD vs YOLO person detector bake-off on a 200-clip stratified
OUC-CGE subsample, broken down by class. Selects the detector for the full
detect_track.py run in step 2. Not part of the shipped pipeline.
"""
import csv
import json
import os
import random
import time
from multiprocessing import Pool

import cv2
import numpy as np

DATA_ROOT = "/home/devops/data/OUC-CGE"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
BAD_FILES = {"videos/low/view2572.mp4", "videos/low/view2531.mp4"}
CLASS_NAMES = {0: "low", 1: "mid", 2: "high"}
T = 32
N_SAMPLE = 200
SEED = 42
OUT_DIR = "/home/devops/ept/outputs/phase1_detector_bakeoff"


def load_rows():
    rows = []
    with open(TRAIN_CSV) as f:
        reader = csv.reader(f, delimiter=" ")
        for r in reader:
            if len(r) != 2:
                continue
            path, label = r[0], int(r[1])
            if path in BAD_FILES:
                continue
            rows.append((path, label))
    return rows


def is_readable(full_path, t, min_ok_frac=0.9):
    """Quick decode check: can we grab at least min_ok_frac of the T requested
    frames? Cheap relative to running detectors — no inference, just decode."""
    cap = cv2.VideoCapture(full_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(n_frames, t)
    ok_count = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, _ = cap.read()
        if ok:
            ok_count += 1
    cap.release()
    return ok_count >= min_ok_frac * t, ok_count, n_frames


def stratified_sample(rows, n_total, seed):
    rng = random.Random(seed)
    by_class = {0: [], 1: [], 2: []}
    for path, label in rows:
        by_class[label].append(path)
    totals = {c: len(v) for c, v in by_class.items()}
    grand_total = sum(totals.values())
    quotas = {c: round(n_total * totals[c] / grand_total) for c in totals}
    # fix rounding drift
    drift = n_total - sum(quotas.values())
    if drift != 0:
        biggest = max(quotas, key=lambda c: totals[c])
        quotas[biggest] += drift

    unreadable = []
    sample = []
    for c, quota in quotas.items():
        pool = by_class[c][:]
        rng.shuffle(pool)
        accepted = []
        pool_iter = iter(pool)
        while len(accepted) < quota:
            try:
                candidate = next(pool_iter)
            except StopIteration:
                raise RuntimeError(f"ran out of candidates for class {c}")
            full = os.path.join(DATA_ROOT, candidate.replace("videos/", ""))
            ok, ok_count, n_frames = is_readable(full, T)
            if ok:
                accepted.append(candidate)
            else:
                unreadable.append(
                    {"path": candidate, "label": c, "ok_frames": ok_count,
                     "reported_n_frames": n_frames}
                )
        sample.extend((p, c) for p in accepted)
    rng.shuffle(sample)
    return sample, quotas, totals, unreadable


def sample_frame_indices(n_frames, t):
    if n_frames <= 0:
        return []
    idxs = np.linspace(0, n_frames - 1, t).astype(int)
    return idxs.tolist()


def extract_frames(path, t):
    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(n_frames, t)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


_SCRFD = None
_YOLO = None


def _init_worker():
    global _SCRFD, _YOLO
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)

    # onnxruntime.InferenceSession defaults intra/inter-op threads to "use all
    # cores" when no SessionOptions is passed, and insightface doesn't expose
    # that control. Under multiprocessing this oversubscribes badly (measured:
    # load average >500 on a 60-core box with 48 workers). Force 1 thread/session.
    import onnxruntime as ort
    _orig_init = ort.InferenceSession.__init__

    def _patched_init(self, *args, sess_options=None, **kwargs):
        if sess_options is None:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
        return _orig_init(self, *args, sess_options=sess_options, **kwargs)

    ort.InferenceSession.__init__ = _patched_init

    from insightface.app import FaceAnalysis
    from ultralytics import YOLO

    _SCRFD = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection"],
    )
    _SCRFD.prepare(ctx_id=-1, det_size=(640, 640))
    _YOLO = YOLO("yolo11n.pt")


def _process_clip(args):
    rel_path, label = args
    full_path = os.path.join(DATA_ROOT, rel_path.replace("videos/", ""))
    frames = extract_frames(full_path, T)
    scrfd_counts = []
    yolo_counts = []
    for f in frames:
        faces = _SCRFD.get(f)
        scrfd_counts.append(len(faces))
        res = _YOLO.predict(f, classes=[0], verbose=False, device="cpu")
        yolo_counts.append(len(res[0].boxes))
    return {
        "path": rel_path,
        "label": label,
        "class_name": CLASS_NAMES[label],
        "n_frames_grabbed": len(frames),
        "scrfd_counts": scrfd_counts,
        "yolo_counts": yolo_counts,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows()
    sample, quotas, totals, unreadable = stratified_sample(rows, N_SAMPLE, SEED)
    print(f"population totals: {totals}")
    print(f"sample quotas (n={len(sample)}): {quotas}")
    print(f"unreadable clips hit during sampling (backfilled): {len(unreadable)}")
    for u in unreadable:
        print(f"  {u}")

    n_workers = 20
    t0 = time.time()
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        results = pool.map(_process_clip, sample, chunksize=1)
    wall_clock = time.time() - t0
    print(f"bake-off wall clock: {wall_clock:.1f}s for {len(sample)} clips "
          f"({n_workers} workers)")

    with open(os.path.join(OUT_DIR, "raw_results.json"), "w") as f:
        json.dump(
            {"sample_quotas": quotas, "population_totals": totals,
             "unreadable_clips_backfilled": unreadable,
             "wall_clock_seconds": wall_clock, "n_workers": n_workers,
             "results": results},
            f,
        )
    print(f"saved -> {OUT_DIR}/raw_results.json")


if __name__ == "__main__":
    main()
