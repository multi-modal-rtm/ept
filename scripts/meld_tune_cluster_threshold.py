"""Tune the agglomerative clustering cosine threshold on DEV ONLY. Reference
embeddings for the 6 recurring FRIENDS characters (identity IS inferable for
these via the Speaker column) are built from TRAIN clips only, to keep dev used
purely for evaluation/tuning, not also for building the ground-truth references
it's then scored against.
"""
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from ept.tokenization.detect_cluster_meld import (
    MELD_ROOT, SPLIT_DIRS, LABEL_CSVS, BAD_CLIPS_REL, SENTIMENT_TO_INT,
    detect_faces, cluster_identities, _init_worker,
)

MAIN_CHARACTERS = ["Ross", "Monica", "Chandler", "Joey", "Phoebe", "Rachel"]
N_REF_CLIPS_PER_CHAR = 15
N_TUNE_CLIPS_PER_CHAR = 15
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
OUT_DIR = "/home/devops/ept/outputs/meld_cluster_tuning"


def load_manifest_by_split_and_speaker():
    manifest = {"train": [], "dev": []}
    for split in ["train", "dev"]:
        with open(os.path.join(MELD_ROOT, "labels", LABEL_CSVS[split])) as f:
            for row in csv.DictReader(f):
                if row["Speaker"] not in MAIN_CHARACTERS:
                    continue
                dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
                fname = f"dia{dia}_utt{utt}.mp4"
                path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS[split], fname)
                rel = f"data/meld/raw/MELD.Raw/MELD.Raw/{SPLIT_DIRS[split]}/{fname}"
                if rel in BAD_CLIPS_REL or not os.path.exists(path):
                    continue
                manifest[split].append({"path": path, "speaker": row["Speaker"], "split": split})
    return manifest


def face_embeddings_for_clip(path, max_faces_per_frame=1):
    """Detection-only pass; returns list of raw embeddings (not clustered) for
    the highest-confidence face per frame (keeps reference-building simple and
    robust to background extras)."""
    records = detect_faces(path)
    embs = []
    for fr in records:
        if not fr["detections"]:
            continue
        best = max(fr["detections"], key=lambda d: d["confidence"])
        embs.append(best["embedding"])
    return embs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    _init_worker()
    manifest = load_manifest_by_split_and_speaker()
    rng = random.Random(42)

    print("=== building reference embeddings from TRAIN ===")
    references = {}
    for char in MAIN_CHARACTERS:
        clips = [m for m in manifest["train"] if m["speaker"] == char]
        picks = rng.sample(clips, min(N_REF_CLIPS_PER_CHAR, len(clips)))
        all_embs = []
        for p in picks:
            all_embs.extend(face_embeddings_for_clip(p["path"]))
        if not all_embs:
            print(f"  WARNING: no faces found for {char}, skipping")
            continue
        emb = np.array(all_embs)
        norm = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norm > 0, norm, 1.0)
        ref = emb.mean(axis=0)
        ref = ref / np.linalg.norm(ref)
        references[char] = ref
        print(f"  {char}: {len(picks)} clips, {len(all_embs)} face samples")

    print("\n=== sweeping threshold on DEV ===")
    dev_picks = {}
    for char in references:
        clips = [m for m in manifest["dev"] if m["speaker"] == char]
        dev_picks[char] = rng.sample(clips, min(N_TUNE_CLIPS_PER_CHAR, len(clips)))

    # Detect once per clip, re-cluster per threshold (avoid re-running detection).
    dev_records_cache = {}
    for char, clips in dev_picks.items():
        for p in clips:
            dev_records_cache[p["path"]] = (detect_faces(p["path"]), char)

    results = {}
    for threshold in THRESHOLDS:
        correct, total = 0, 0
        for path, (records, true_char) in dev_records_cache.items():
            records_copy = json.loads(json.dumps(records))  # deep copy, cheap here
            clustered = cluster_identities(records_copy, threshold)
            # find the cluster with the most detections (most "on-screen" identity)
            counts = {}
            embs_by_cluster = {}
            for fr in clustered:
                for det in fr["detections"]:
                    tid = det["track_id"]
                    counts[tid] = counts.get(tid, 0) + 1
                    embs_by_cluster.setdefault(tid, []).append(det["embedding"])
            if not counts:
                total += 1
                continue
            top_cluster = max(counts, key=counts.get)
            cluster_emb = np.array(embs_by_cluster[top_cluster]).mean(axis=0)
            cluster_emb = cluster_emb / np.linalg.norm(cluster_emb)
            sims = {char: float(cluster_emb @ ref) for char, ref in references.items()}
            pred_char = max(sims, key=sims.get)
            correct += int(pred_char == true_char)
            total += 1
        acc = correct / total if total else 0.0
        results[threshold] = {"accuracy": acc, "n": total}
        print(f"  threshold={threshold}: accuracy={acc:.4f} (n={total})")

    best_threshold = max(results, key=lambda t: results[t]["accuracy"])
    print(f"\nbest threshold: {best_threshold} (accuracy={results[best_threshold]['accuracy']:.4f})")

    with open(os.path.join(OUT_DIR, "threshold_tuning_report.json"), "w") as f:
        json.dump({"results": {str(k): v for k, v in results.items()},
                    "best_threshold": best_threshold,
                    "characters": list(references.keys())}, f, indent=2)
    print(f"saved -> {OUT_DIR}/threshold_tuning_report.json")


if __name__ == "__main__":
    main()
