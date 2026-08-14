"""Phase 1 step 4: mask-only control (docs/DECISION_RULES.md amendment, 2026-08-14).

Presence masks [E_max=8, S=8] built from the tracking cache (top-E tracks by mean
confidence x frame coverage, matching the locked entity-selection rule). Trains a
logistic regression and a small MLP on the flattened mask alone (no visual features),
3 seeds, VAL split only (never touches test). Reports macro-F1 vs majority-class
baseline. Determines how much of any later EPT gain could be presence-pattern (i.e.
detection-artifact) rather than visual content.
"""
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

CACHE_ROOT = "/home/devops/ept/cache/tracks"
OUT_DIR = "/home/devops/ept/outputs/phase1_mask_only_control"
E_MAX = 8
S = 8
T = 32
SEEDS = [42, 1337, 2024]


def load_val_clips():
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, "val", "*.json")))
    clips = []
    for fp in files:
        with open(fp) as f:
            clips.append(json.load(f))
    return clips


def build_mask(clip):
    """[E_max, S] boolean presence mask via top-E selection by
    mean_confidence * frame_coverage (the locked entity-selection rule)."""
    n_total = clip["n_frames_grabbed"]
    seg_len = T / S  # frames per segment, mapped from frame_pos (0..31)

    track_frames = {}  # track_id -> list of (frame_pos, confidence)
    for frame in clip["frames"]:
        for det in frame["detections"]:
            track_frames.setdefault(det["track_id"], []).append(
                (frame["frame_pos"], det["confidence"] or 0.0)
            )

    scored = []
    for tid, entries in track_frames.items():
        coverage = len(entries) / n_total if n_total else 0.0
        mean_conf = np.mean([c for _, c in entries])
        scored.append((mean_conf * coverage, tid, entries))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:E_MAX]

    mask = np.zeros((E_MAX, S), dtype=bool)
    for e, (_, tid, entries) in enumerate(top):
        for frame_pos, _ in entries:
            s = min(int(frame_pos // seg_len), S - 1)
            mask[e, s] = True
    return mask.astype(np.float32).flatten()


def run_one_seed(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    results = {}

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    results["majority_baseline"] = f1_score(y_te, dummy.predict(X_te), average="macro")

    logreg = LogisticRegression(max_iter=2000, random_state=seed)
    logreg.fit(X_tr, y_tr)
    results["logreg"] = f1_score(y_te, logreg.predict(X_te), average="macro")

    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=seed,
                         early_stopping=True)
    mlp.fit(X_tr, y_tr)
    results["mlp"] = f1_score(y_te, mlp.predict(X_te), average="macro")

    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = load_val_clips()
    print(f"loaded {len(clips)} val clips")

    X = np.stack([build_mask(c) for c in clips])
    y = np.array([c["label"] for c in clips])
    print(f"X shape: {X.shape}, class counts: {np.bincount(y)}")

    per_seed = {"majority_baseline": [], "logreg": [], "mlp": []}
    for seed in SEEDS:
        r = run_one_seed(X, y, seed)
        for k, v in r.items():
            per_seed[k].append(v)
        print(f"seed={seed} " + " ".join(f"{k}={v:.4f}" for k, v in r.items()))

    summary = {}
    for k, vals in per_seed.items():
        summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "per_seed": vals}
        print(f"{k}: mean={summary[k]['mean']:.4f} std={summary[k]['std']:.4f}")

    with open(os.path.join(OUT_DIR, "mask_only_control_report.json"), "w") as f:
        json.dump({"n_clips": len(clips), "seeds": SEEDS, "summary": summary}, f, indent=2)
    print(f"saved -> {OUT_DIR}/mask_only_control_report.json")


if __name__ == "__main__":
    main()
