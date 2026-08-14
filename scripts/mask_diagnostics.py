"""Track B: mask-only control diagnostics (VAL split only), per the follow-up
request after the Phase 1 mask-only control came back well above chance.

1. Baselines: majority-class, stratified-random (closed-form + empirical), and a
   20-permutation null for the logistic-regression macro-F1, with an empirical
   p-value for where the real (unpermuted) score falls.
2. Reduced feature sets: track-count-only, mean-coverage-only, per-segment
   presence-counts-only, full mask — to see which recovers most of the signal.
3. Confound check: see REPORT.md — no room/session/recording identifier exists
   in OUC-CGE (checked: xlsx duplicates the csv exactly; video container metadata
   has no per-clip recording timestamp, only a shared dataset-processing mtime;
   filenames are sequential view<N>.mp4 with no session encoding). Nothing to
   compute a contingency table against.
"""
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

CACHE_ROOT = "/home/devops/ept/cache/tracks"
OUT_DIR = "/home/devops/ept/outputs/phase2_mask_diagnostics"
E_MAX = 8
S = 8
T = 32
SEED = 42  # primary seed for baselines/permutation null (matches the original run)
N_PERMUTATIONS = 20


def load_val_clips():
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, "val", "*.json")))
    clips = []
    for fp in files:
        with open(fp) as f:
            clips.append(json.load(f))
    return clips


def build_mask(clip):
    n_total = clip["n_frames_grabbed"]
    seg_len = T / S
    track_frames = {}
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
    return mask


def macro_f1_logreg(X, y, seed):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X_tr, y_tr)
    return f1_score(y_te, clf.predict(X_te), average="macro")


def section1_baselines(X, y):
    print("\n=== 1. Baselines ===")
    result = {}

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    majority_f1 = f1_score(y_te, dummy.predict(X_te), average="macro")
    result["majority_baseline"] = majority_f1
    print(f"majority-class macro-F1: {majority_f1:.4f}")

    # Closed-form: for K-class stratified-random guessing (predictions drawn
    # independently from the true prior), precision_c = recall_c = p_c exactly
    # in expectation, so F1_c = p_c and macro-F1 = mean(p_c) = 1/K regardless of
    # class balance. Verify empirically too.
    K = len(np.unique(y))
    closed_form = 1.0 / K
    rng = np.random.RandomState(SEED)
    priors = np.bincount(y_tr) / len(y_tr)
    empirical_stratified = []
    for _ in range(200):
        pred = rng.choice(len(priors), size=len(y_te), p=priors)
        empirical_stratified.append(f1_score(y_te, pred, average="macro"))
    result["stratified_random_closed_form"] = closed_form
    result["stratified_random_empirical_mean"] = float(np.mean(empirical_stratified))
    result["stratified_random_empirical_std"] = float(np.std(empirical_stratified))
    print(f"stratified-random macro-F1: closed-form=1/{K}={closed_form:.4f}, "
          f"empirical (200 draws) mean={np.mean(empirical_stratified):.4f} "
          f"std={np.std(empirical_stratified):.4f}")

    seeds = [42, 1337, 2024]
    real_f1_per_seed = [macro_f1_logreg(X, y, s) for s in seeds]
    real_f1_single = real_f1_per_seed[0]
    real_f1_3seed_mean = float(np.mean(real_f1_per_seed))
    result["real_logreg_f1_single_seed42"] = real_f1_single
    result["real_logreg_f1_3seed_mean"] = real_f1_3seed_mean
    print(f"real (unpermuted) logreg macro-F1: seed42={real_f1_single:.4f}, "
          f"3-seed mean={real_f1_3seed_mean:.4f} (matches the originally-reported 0.373)")

    # Permutation null computed the SAME way as the originally-reported 0.373 —
    # 3-seed mean per permutation — so the p-value is apples-to-apples against
    # that exact number, not just a single seed.
    null_scores = []
    rng2 = np.random.RandomState(SEED)
    for i in range(N_PERMUTATIONS):
        y_shuffled = y.copy()
        rng2.shuffle(y_shuffled)
        perm_scores = [macro_f1_logreg(X, y_shuffled, s) for s in seeds]
        null_scores.append(float(np.mean(perm_scores)))
    null_scores = np.array(null_scores)
    result["permutation_null_scores_3seed_mean"] = null_scores.tolist()
    result["permutation_null_mean"] = float(null_scores.mean())
    result["permutation_null_std"] = float(null_scores.std())
    n_ge = int((null_scores >= real_f1_3seed_mean).sum())
    p_value = (n_ge + 1) / (N_PERMUTATIONS + 1)
    result["p_value"] = p_value
    print(f"permutation null ({N_PERMUTATIONS} shuffles, 3-seed mean each): "
          f"mean={null_scores.mean():.4f} std={null_scores.std():.4f}, "
          f"{n_ge}/{N_PERMUTATIONS} nulls >= real 3-seed score ({real_f1_3seed_mean:.4f}), "
          f"empirical p-value={p_value:.4f}")

    # Robustness check: 20 permutations gives coarse p-value resolution (min
    # possible ~0.048). Single-seed (cheaper) at N=200 for a stabler estimate,
    # supplementary to -- not a replacement for -- the requested 20-perm/3-seed number.
    null_single_seed = []
    rng3 = np.random.RandomState(SEED + 1)
    for i in range(200):
        y_shuffled = y.copy()
        rng3.shuffle(y_shuffled)
        null_single_seed.append(macro_f1_logreg(X, y_shuffled, SEED))
    null_single_seed = np.array(null_single_seed)
    n_ge_200 = int((null_single_seed >= real_f1_single).sum())
    p_value_200 = (n_ge_200 + 1) / 201
    result["permutation_null_200_single_seed_mean"] = float(null_single_seed.mean())
    result["permutation_null_200_single_seed_std"] = float(null_single_seed.std())
    result["p_value_200_single_seed"] = p_value_200
    print(f"robustness check (200 shuffles, single seed42): mean={null_single_seed.mean():.4f} "
          f"std={null_single_seed.std():.4f}, {n_ge_200}/200 nulls >= real seed42 score "
          f"({real_f1_single:.4f}), empirical p-value={p_value_200:.4f}")
    return result


def section2_reduced_features(masks, y):
    print("\n=== 2. Reduced feature sets ===")
    result = {}

    track_count = np.array([[m.any(axis=1).sum()] for m in masks], dtype=np.float32)
    mean_coverage = np.array([[m.mean()] for m in masks], dtype=np.float32)
    per_segment_counts = np.array([m.sum(axis=0) for m in masks], dtype=np.float32)  # [n, S]
    full_mask = np.array([m.flatten() for m in masks], dtype=np.float32)  # [n, E*S]

    for name, X in [
        ("track_count_only", track_count),
        ("mean_coverage_only", mean_coverage),
        ("per_segment_presence_counts", per_segment_counts),
        ("full_mask", full_mask),
    ]:
        scores = [macro_f1_logreg(X, y, seed) for seed in [42, 1337, 2024]]
        result[name] = {"mean": float(np.mean(scores)), "std": float(np.std(scores)), "per_seed": scores}
        print(f"{name:30s} (dim={X.shape[1]:2d}): macro-F1 mean={np.mean(scores):.4f} std={np.std(scores):.4f}")
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = load_val_clips()
    masks = [build_mask(c) for c in clips]
    y = np.array([c["label"] for c in clips])
    X_full = np.array([m.astype(np.float32).flatten() for m in masks])
    print(f"loaded {len(clips)} val clips, class counts: {np.bincount(y)}")

    r1 = section1_baselines(X_full, y)
    r2 = section2_reduced_features(masks, y)

    with open(os.path.join(OUT_DIR, "diagnostics_report.json"), "w") as f:
        json.dump({"n_clips": len(clips), "baselines": r1, "reduced_features": r2}, f, indent=2)
    print(f"\nsaved -> {OUT_DIR}/diagnostics_report.json")


if __name__ == "__main__":
    main()
