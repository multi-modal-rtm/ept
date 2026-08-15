"""Compute per-clip clustering purity (mean silhouette score, cosine distance)
for MELD's TEST split only, per the pre-specified secondary analysis
(docs/DECISION_RULES.md, 2026-08-15). Detection-only pass (embeddings were not
retained in the production tracking cache); no labels are touched — this is an
unsupervised clustering-quality measurement, not a test evaluation.
"""
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
    MELD_ROOT, SPLIT_DIRS, LABEL_CSVS, BAD_CLIPS_REL, SENTIMENT_TO_INT,
    detect_faces, cluster_identities, _init_worker,
)

THRESHOLD = 0.55
OUT_DIR = "/home/devops/ept/outputs/meld_purity"


def load_test_manifest():
    import csv
    manifest = []
    with open(os.path.join(MELD_ROOT, "labels", LABEL_CSVS["test"])) as f:
        for row in csv.DictReader(f):
            dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
            fname = f"dia{dia}_utt{utt}.mp4"
            path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS["test"], fname)
            rel = f"data/meld/raw/MELD.Raw/MELD.Raw/{SPLIT_DIRS['test']}/{fname}"
            if rel in BAD_CLIPS_REL or not os.path.exists(path):
                continue
            manifest.append({"clip_id": f"dia{dia}_utt{utt}", "path": path})
    return manifest


def clip_purity(path):
    from sklearn.metrics import silhouette_score

    records = detect_faces(path)
    n_det = sum(len(fr["detections"]) for fr in records)
    if n_det < 2:
        return 1.0, n_det

    records = cluster_identities(records, THRESHOLD)
    embs, labels = [], []
    for fr in records:
        for det in fr["detections"]:
            embs.append(det["embedding"])
            labels.append(det["track_id"])
    if len(set(labels)) < 2:
        return 1.0, n_det

    emb = np.array(embs)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.where(norm > 0, norm, 1.0)
    try:
        score = silhouette_score(emb, labels, metric="cosine")
    except ValueError:
        return 1.0, n_det
    # silhouette in [-1,1]; rescale to [0,1] "purity"-style score for readability
    purity = (score + 1) / 2
    return float(purity), n_det


def _process(entry):
    purity, n_det = clip_purity(entry["path"])
    return {"clip_id": entry["clip_id"], "purity": purity, "n_detections": n_det}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    manifest = load_test_manifest()
    print(f"test manifest: {len(manifest)} clips")

    t0 = time.time()
    results = []
    with Pool(processes=30, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap_unordered(_process, manifest, chunksize=4)):
            results.append(r)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(manifest)} done, {time.time()-t0:.1f}s elapsed")
    wall = time.time() - t0
    print(f"purity pass: {len(results)} clips, {wall:.1f}s")

    purities = np.array([r["purity"] for r in results])
    p33, p67 = np.percentile(purities, [33.333, 66.667])
    print(f"purity distribution: mean={purities.mean():.4f} median={np.median(purities):.4f} "
          f"p33={p33:.4f} p67={p67:.4f}")
    print(f"tercile counts: low(<{p33:.4f})={int((purities<p33).sum())} "
          f"mid={int(((purities>=p33)&(purities<p67)).sum())} "
          f"high(>={p67:.4f})={int((purities>=p67).sum())}")

    with open(os.path.join(OUT_DIR, "test_purity.json"), "w") as f:
        json.dump({
            "n_clips": len(results), "wall_clock_seconds": wall,
            "threshold": THRESHOLD,
            "purity_mean": float(purities.mean()), "purity_median": float(np.median(purities)),
            "tercile_p33": float(p33), "tercile_p67": float(p67),
            "results": results,
        }, f, indent=2)
    print(f"saved -> {OUT_DIR}/test_purity.json")


if __name__ == "__main__":
    main()
