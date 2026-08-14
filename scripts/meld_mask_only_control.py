"""MELD mask-only control (Phase 2 equivalent), with permutation null as in
Phase 2 (post-audit methodology: majority-class alone is a misleadingly weak
baseline for macro-F1, per the OUC-CGE mask-only correction). MELD's presence
mask reflects who is on screen, which in edited television tracks who is
speaking — plausibly informative, per the task's own framing. Report it
plainly whatever it says.
"""
import glob
import json
import os

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from ept.tokenization.extract_features_meld import MELD_PRIMARY_E_MAX, S, TRACKS_ROOT

OUT_DIR = "/home/devops/ept/outputs/meld_mask_only_control"
SEED = 42
N_PERMUTATIONS = 20
SEEDS = [42, 1337, 2024]


def load_dev_clips():
    files = sorted(glob.glob(os.path.join(TRACKS_ROOT, "dev", "*.json")))
    clips = []
    for fp in files:
        with open(fp) as f:
            clips.append(json.load(f))
    return clips


def build_mask(clip, e_max=MELD_PRIMARY_E_MAX):
    n_total = clip["n_frames_grabbed"]
    track_frames = {}
    for frame in clip["frames"]:
        for det in frame["detections"]:
            tid = det.get("track_id")
            if tid is None:
                continue
            track_frames.setdefault(tid, []).append((frame["frame_pos"], det["confidence"] or 0.0))
    scored = []
    for tid, entries in track_frames.items():
        coverage = len(entries) / n_total if n_total else 0.0
        mean_conf = np.mean([c for _, c in entries])
        scored.append((mean_conf * coverage, tid, entries))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:e_max]
    from ept.tokenization.extract_features_meld import T
    seg_len = T / S
    mask = np.zeros((e_max, S), dtype=bool)
    for e, (_, tid, entries) in enumerate(top):
        for frame_pos, _ in entries:
            s = min(int(frame_pos // seg_len), S - 1)
            mask[e, s] = True
    return mask


def macro_f1_logreg(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    return f1_score(y_te, clf.predict(X_te), average="macro")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = load_dev_clips()
    masks = [build_mask(c) for c in clips]
    y = np.array([c["label"] for c in clips])
    X = np.array([m.astype(np.float32).flatten() for m in masks])
    print(f"loaded {len(clips)} MELD dev clips, class counts: {np.bincount(y)}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    majority_f1 = f1_score(y_te, dummy.predict(X_te), average="macro")
    K = len(np.unique(y))
    stratified_random_closed_form = 1.0 / K
    print(f"majority-class macro-F1: {majority_f1:.4f}")
    print(f"stratified-random macro-F1 (closed-form, K={K}): {stratified_random_closed_form:.4f}")

    seed_scores = [macro_f1_logreg(X, y, s) for s in SEEDS]
    real_3seed_mean = float(np.mean(seed_scores))
    print(f"real logreg macro-F1 per seed: {seed_scores}, 3-seed mean={real_3seed_mean:.4f}")

    null_scores = []
    rng = np.random.RandomState(SEED)
    for i in range(N_PERMUTATIONS):
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)
        perm_scores = [macro_f1_logreg(X, y_shuffled, s) for s in SEEDS]
        null_scores.append(float(np.mean(perm_scores)))
    null_scores = np.array(null_scores)
    n_ge = int((null_scores >= real_3seed_mean).sum())
    p_value = (n_ge + 1) / (N_PERMUTATIONS + 1)
    print(f"permutation null ({N_PERMUTATIONS} perms, 3-seed mean each): "
          f"mean={null_scores.mean():.4f} std={null_scores.std():.4f}, "
          f"{n_ge}/{N_PERMUTATIONS} nulls >= real, p-value={p_value:.4f}")

    report = {
        "n_clips": len(clips), "n_train": len(y_tr), "n_test": len(y_te),
        "majority_baseline": float(majority_f1),
        "stratified_random_closed_form": stratified_random_closed_form,
        "real_logreg_per_seed": seed_scores, "real_logreg_3seed_mean": real_3seed_mean,
        "permutation_null_scores": null_scores.tolist(),
        "permutation_null_mean": float(null_scores.mean()),
        "permutation_null_std": float(null_scores.std()),
        "p_value": p_value,
    }
    with open(os.path.join(OUT_DIR, "meld_mask_only_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {OUT_DIR}/meld_mask_only_report.json")


if __name__ == "__main__":
    main()
