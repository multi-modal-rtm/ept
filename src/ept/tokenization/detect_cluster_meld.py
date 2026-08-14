"""MELD tokenization (Phase-1 equivalent, promoted-primary dataset): shot cuts
break IoU tracking, so identity recovery is face-embedding clustering
(agglomerative, cosine), not ByteTrack. Detector: SCRFD (chosen over YOLO
person-detection after a brief bake-off — scripts/meld_detector_bakeoff.py:
SCRFD zero-detection rate 0.94% vs YOLO 0.88%, near-parity confirming MELD's
close-framed/frontal-face hypothesis, unlike OUC-CGE's severe pose-correlated
failure; also, only face detection produces crops suitable for ArcFace-style
identity embeddings, which the clustering step requires).

Clip alignment: (Dialogue_ID, Utterance_ID) from the label CSV, never directory
listing. Excludes data/meld/bad_clips.txt.
"""
import csv
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import cv2
import numpy as np

MELD_ROOT = "/home/devops/socialarcnet-v2/data/meld/raw"
SPLIT_DIRS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}
LABEL_CSVS = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
CACHE_ROOT = "/home/devops/ept/cache/tracks/meld"
T = 16  # MELD clips are ~3s vs OUC-CGE's longer clips; 16 frames keeps ~4 frames/segment at S=4
SENTIMENT_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}

BAD_CLIPS_REL = {
    "data/meld/raw/MELD.Raw/MELD.Raw/dev_splits_complete/dia110_utt7.mp4",
    "data/meld/raw/MELD.Raw/MELD.Raw/train_splits/dia125_utt3.mp4",
}


def load_manifest():
    manifest = []
    for split, csv_name in LABEL_CSVS.items():
        with open(os.path.join(MELD_ROOT, "labels", csv_name)) as f:
            for row in csv.DictReader(f):
                dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
                fname = f"dia{dia}_utt{utt}.mp4"
                path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS[split], fname)
                rel = f"data/meld/raw/MELD.Raw/MELD.Raw/{SPLIT_DIRS[split]}/{fname}"
                if rel in BAD_CLIPS_REL or not os.path.exists(path):
                    continue
                manifest.append({
                    "clip_id": f"dia{dia}_utt{utt}", "split": split, "path": path,
                    "sentiment": SENTIMENT_TO_INT[row["Sentiment"]],
                    "speaker": row["Speaker"], "dialogue_id": dia, "utterance_id": utt,
                })
    return manifest


def sample_frame_indices(n_frames, t):
    if n_frames <= 0:
        return []
    return np.linspace(0, n_frames - 1, t).astype(int).tolist()


def extract_frames(path, t):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_frame_indices(n, t)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    return frames


_APP = None


def _init_worker():
    global _APP
    import torch
    torch.set_num_threads(1)
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
    _APP = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"],
                         allowed_modules=["detection", "recognition"])
    _APP.prepare(ctx_id=-1, det_size=(640, 640))


def detect_faces(video_path, t=T):
    """[{frame_pos, frame_idx, detections: [{bbox, confidence, embedding}]}]"""
    frames = extract_frames(video_path, t)
    records = []
    for frame_pos, (frame_idx, frame) in enumerate(frames):
        faces = _APP.get(frame)
        dets = [{
            "bbox": face.bbox.tolist(), "confidence": float(face.det_score),
            "embedding": face.embedding.tolist(),
        } for face in faces]
        records.append({"frame_pos": frame_pos, "frame_idx": int(frame_idx), "detections": dets})
    return records


def cluster_identities(records, threshold):
    """Agglomerative clustering (cosine distance) over all face embeddings in a
    clip; assigns a cluster/track_id to each detection in place. Returns records
    (same structure as detect_faces, with 'track_id' added per detection)."""
    from sklearn.cluster import AgglomerativeClustering

    all_embeddings = []
    index_map = []  # (frame_list_idx, det_idx)
    for fi, fr in enumerate(records):
        for di, det in enumerate(fr["detections"]):
            all_embeddings.append(det["embedding"])
            index_map.append((fi, di))

    if len(all_embeddings) == 0:
        return records
    if len(all_embeddings) == 1:
        records[index_map[0][0]]["detections"][index_map[0][1]]["track_id"] = 0
        return records

    emb = np.array(all_embeddings)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norm > 0, norm, 1.0)
    # cosine distance = 1 - cosine similarity; AgglomerativeClustering wants a
    # distance_threshold with metric='cosine', linkage='average'
    clust = AgglomerativeClustering(
        n_clusters=None, distance_threshold=1 - threshold, metric="cosine", linkage="average"
    )
    labels = clust.fit_predict(emb)
    for (fi, di), label in zip(index_map, labels):
        records[fi]["detections"][di]["track_id"] = int(label)
    return records


def _process_clip(args, threshold):
    entry = args
    records = detect_faces(entry["path"], T)
    records = cluster_identities(records, threshold)
    # strip embeddings before saving (large; not needed downstream, identity is
    # now encoded as track_id) but keep bbox/confidence/track_id like OUC-CGE format
    for fr in records:
        for det in fr["detections"]:
            det.pop("embedding", None)

    out_dir = os.path.join(CACHE_ROOT, entry["split"])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{entry['clip_id']}.json")
    payload = {
        "clip_id": entry["clip_id"], "split": entry["split"], "label": entry["sentiment"],
        "speaker": entry["speaker"], "dialogue_id": entry["dialogue_id"],
        "utterance_id": entry["utterance_id"], "rel_path": os.path.relpath(entry["path"], MELD_ROOT),
        "t": T, "n_frames_grabbed": len(records),
        "detector": "scrfd-buffalo_l", "identity_method": "agglomerative-cosine",
        "cluster_threshold": threshold,
        "frames": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return {"clip_id": entry["clip_id"], "split": entry["split"], "n_frames": len(records)}


def _process_clip_mp(args_and_threshold):
    entry, threshold = args_and_threshold
    return _process_clip(entry, threshold)


def run_full(manifest, threshold, n_workers=20, log_path=None):
    tagged = [(e, threshold) for e in manifest]
    t0 = time.time()
    results = []
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_clip_mp, tagged, chunksize=4)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(tagged)} done, {time.time()-t0:.1f}s elapsed")
    wall_clock = time.time() - t0
    print(f"MELD tokenization: {len(results)} clips, {wall_clock:.1f}s, {n_workers} workers")
    if log_path:
        with open(log_path, "w") as f:
            json.dump({"n_clips": len(results), "wall_clock_seconds": wall_clock,
                        "n_workers": n_workers, "threshold": threshold}, f)
    return results, wall_clock
