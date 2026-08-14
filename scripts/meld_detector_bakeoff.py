"""MELD tokenization step: brief detector bake-off (face vs person), per the
same logic as the Phase 1 OUC-CGE bake-off but shorter — MELD is close-framed
television, hypothesis is faces are large/frontal enough that SCRFD won't have
the pose-correlated blind spot found on OUC-CGE. Report both, choose on evidence.
"""
import csv
import os
import random
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import cv2
import numpy as np

MELD_ROOT = "/home/devops/socialarcnet-v2/data/meld/raw"
SPLIT_DIRS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}
LABEL_CSVS = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
BAD_CLIPS = {
    "data/meld/raw/MELD.Raw/MELD.Raw/dev_splits_complete/dia110_utt7.mp4",
    "data/meld/raw/MELD.Raw/MELD.Raw/train_splits/dia125_utt3.mp4",
}
T = 16
N_SAMPLE = 100


def build_manifest():
    manifest = []
    for split, csv_name in LABEL_CSVS.items():
        with open(os.path.join(MELD_ROOT, "labels", csv_name)) as f:
            for row in csv.DictReader(f):
                dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
                fname = f"dia{dia}_utt{utt}.mp4"
                path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS[split], fname)
                rel = f"data/meld/raw/MELD.Raw/MELD.Raw/{SPLIT_DIRS[split]}/{fname}"
                if rel in BAD_CLIPS or not os.path.exists(path):
                    continue
                manifest.append({"path": path, "split": split, "sentiment": row["Sentiment"]})
    return manifest


def extract_frames(path, t):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(n - 1, 0), t).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from insightface.app import FaceAnalysis
    from ultralytics import YOLO
    import torch

    torch.set_num_threads(1)
    scrfd = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    scrfd.prepare(ctx_id=-1, det_size=(640, 640))
    yolo = YOLO("yolo11n.pt")

    manifest = build_manifest()
    rng = random.Random(42)
    sample = rng.sample(manifest, N_SAMPLE)
    print(f"sampled {len(sample)} MELD clips for bake-off")

    scrfd_counts, yolo_counts = [], []
    t0 = time.time()
    for m in sample:
        frames = extract_frames(m["path"], T)
        for f in frames:
            torch.set_num_threads(1)
            faces = scrfd.get(f)
            scrfd_counts.append(len(faces))
            res = yolo.predict(f, classes=[0], verbose=False, device="cpu")
            yolo_counts.append(len(res[0].boxes))
    elapsed = time.time() - t0

    scrfd_counts = np.array(scrfd_counts)
    yolo_counts = np.array(yolo_counts)
    print(f"\n{len(scrfd_counts)} frames, {elapsed:.1f}s")
    print(f"SCRFD: mean={scrfd_counts.mean():.3f} pct_zero={100*(scrfd_counts==0).mean():.2f}% "
          f"var={scrfd_counts.var():.3f}")
    print(f"YOLO:  mean={yolo_counts.mean():.3f} pct_zero={100*(yolo_counts==0).mean():.2f}% "
          f"var={yolo_counts.var():.3f}")


if __name__ == "__main__":
    main()
