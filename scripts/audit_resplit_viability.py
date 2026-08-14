"""Phase 3.5 Track 1: can OUC-CGE be re-split by recording? AUDIT/DESIGN ONLY —
no test evaluation, no LOCKED_RECIPE.md change. Builds a similarity graph over
ALL 7700 clips (ignoring the original train/val/test assignment — that's the
thing being replaced), finds connected components at several cosine thresholds,
checks class purity within components, and if viable, constructs a group-disjoint
split and re-runs the trivial probe on it.
"""
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

FEATURES_ROOT = "/home/devops/ept/cache/features"
TRACKS_ROOT = "/home/devops/ept/cache/tracks"
OUT_DIR = "/home/devops/ept/outputs/phase3_5_audit"
THRESHOLDS = [0.95, 0.98, 0.99]
SPLIT_THRESHOLD = 0.98
TARGET_FRACS = {"train": 0.8, "val": 0.1, "test": 0.1}


def load_all_clips():
    """All 7700 OUC-CGE clips across the original train/val/test dirs, combined
    (the audit ignores the original split -- that's what's being replaced)."""
    clip_ids, embeddings, labels, orig_splits = [], [], [], []
    for split in ["train", "val", "test"]:
        ent_dir = os.path.join(FEATURES_ROOT, "ouccge", split)
        tracks_dir = os.path.join(TRACKS_ROOT, split)
        labels_by_clip = {}
        for fp in glob.glob(os.path.join(tracks_dir, "*.json")):
            with open(fp) as f:
                clip = json.load(f)
            labels_by_clip[clip["clip_id"]] = clip["label"]

        for fp in sorted(glob.glob(os.path.join(ent_dir, "*.npy"))):
            if fp.endswith("_mask.npy") or fp.endswith("_scores.npy"):
                continue
            clip_id = os.path.splitext(os.path.basename(fp))[0]
            if clip_id not in labels_by_clip:
                continue
            feat = np.load(fp)
            mask = np.load(os.path.join(ent_dir, f"{clip_id}_mask.npy"))
            m = mask[..., None].astype(np.float32)
            summed = (feat.astype(np.float32) * m).sum(axis=(0, 1))
            count = max(m.sum(), 1.0)
            emb = summed / count
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            clip_ids.append(clip_id)
            embeddings.append(emb.astype(np.float32))
            labels.append(labels_by_clip[clip_id])
            orig_splits.append(split)
    return clip_ids, np.stack(embeddings), np.array(labels), orig_splits


def component_analysis(sim, threshold, labels):
    n = sim.shape[0]
    adj = (sim >= threshold)
    np.fill_diagonal(adj, False)
    adj_sparse = csr_matrix(adj)
    n_components, comp_labels = connected_components(adj_sparse, directed=False)

    comp_members = defaultdict(list)
    for i, c in enumerate(comp_labels):
        comp_members[c].append(i)

    sizes = np.array([len(v) for v in comp_members.values()])
    n_over_1pct = int((sizes >= 0.01 * n).sum())

    entropies = []
    pure_clips = 0
    for c, members in comp_members.items():
        lbls = [labels[i] for i in members]
        counts = Counter(lbls)
        probs = np.array(list(counts.values())) / len(lbls)
        ent = float(-(probs * np.log2(probs)).sum()) if len(probs) > 1 else 0.0
        entropies.append((ent, len(members)))
        if len(counts) == 1:
            pure_clips += len(members)

    weighted_entropy = sum(e * n for e, n in entropies) / n
    pct_pure = 100 * pure_clips / n

    return {
        "threshold": threshold, "n_components": int(n_components),
        "size_mean": float(sizes.mean()), "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()), "size_p90": float(np.percentile(sizes, 90)),
        "n_components_over_1pct_corpus": n_over_1pct,
        "pct_clips_in_class_pure_components": float(pct_pure),
        "weighted_mean_entropy_bits": float(weighted_entropy),
        "size_histogram": {str(k): int(v) for k, v in Counter(sizes.tolist()).items()},
        "comp_labels": comp_labels,  # kept in-process only, not JSON-serialized below
    }


def greedy_group_stratified_split(comp_labels, labels, seed=42):
    """Assign whole connected components to train/val/test, stratified to
    preserve class balance, sized ~80/10/10. Greedy: process components largest
    -> smallest; assign each to whichever split is currently furthest below its
    target share OF THAT COMPONENT'S MAJORITY CLASS (falls back to overall
    target share once a class is exhausted in a split)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    comp_to_members = defaultdict(list)
    for i, c in enumerate(comp_labels):
        comp_to_members[c].append(i)

    classes = sorted(set(labels.tolist()))
    class_totals = Counter(labels.tolist())
    target_class_counts = {
        split: {cls: TARGET_FRACS[split] * class_totals[cls] for cls in classes}
        for split in TARGET_FRACS
    }
    current_class_counts = {split: Counter() for split in TARGET_FRACS}

    comps = list(comp_to_members.items())
    order = sorted(comps, key=lambda kv: -len(kv[1]))
    # break ties among equal-size components randomly for fairness
    rng.shuffle(order)
    order.sort(key=lambda kv: -len(kv[1]))

    assignment = {}
    for comp_id, members in order:
        comp_lbls = Counter(labels[i] for i in members)
        majority_cls = comp_lbls.most_common(1)[0][0]
        best_split, best_deficit = None, -1e18
        for split in TARGET_FRACS:
            deficit = (target_class_counts[split][majority_cls]
                       - current_class_counts[split][majority_cls])
            if deficit > best_deficit:
                best_deficit, best_split = deficit, split
        assignment[comp_id] = best_split
        for cls, cnt in comp_lbls.items():
            current_class_counts[best_split][cls] += cnt

    new_split = [assignment[c] for c in comp_labels]
    return new_split, current_class_counts


def trivial_probe_on_split(embeddings, labels, new_split):
    new_split = np.array(new_split)
    X_train, y_train = embeddings[new_split == "train"], labels[new_split == "train"]
    X_val, y_val = embeddings[new_split == "val"], labels[new_split == "val"]
    clf = LogisticRegression(max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)
    val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
    return val_f1, len(y_train), len(y_val)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading all 7700 clip embeddings (ignoring original split)...")
    clip_ids, embeddings, labels, orig_splits = load_all_clips()
    n = len(clip_ids)
    print(f"n={n}")

    print("computing full pairwise similarity matrix...")
    sim = embeddings @ embeddings.T

    report = {"n_clips": n}
    comp_labels_by_thresh = {}
    for t in THRESHOLDS:
        print(f"\n=== threshold {t} ===")
        res = component_analysis(sim, t, labels)
        comp_labels_by_thresh[t] = res.pop("comp_labels")
        report[f"threshold_{t}"] = res
        print(f"  n_components={res['n_components']} size_mean={res['size_mean']:.1f} "
              f"size_median={res['size_median']:.1f} size_max={res['size_max']} "
              f"size_p90={res['size_p90']:.1f}")
        print(f"  components holding >1% of corpus: {res['n_components_over_1pct_corpus']}")
        print(f"  pct clips in class-PURE components: {res['pct_clips_in_class_pure_components']:.2f}%")
        print(f"  weighted mean entropy (bits, max=log2(3)=1.585): {res['weighted_mean_entropy_bits']:.4f}")

    # Viability + re-split at SPLIT_THRESHOLD
    comp_labels = comp_labels_by_thresh[SPLIT_THRESHOLD]
    n_components_at_split_thresh = report[f"threshold_{SPLIT_THRESHOLD}"]["n_components"]
    print(f"\n=== effective sample size (independent recordings) at threshold {SPLIT_THRESHOLD}: "
          f"{n_components_at_split_thresh} ===")

    print(f"\n=== constructing group-disjoint split at threshold {SPLIT_THRESHOLD} ===")
    new_split, final_counts = greedy_group_stratified_split(comp_labels, labels)
    new_split_arr = np.array(new_split)
    for s in TARGET_FRACS:
        n_s = int((new_split_arr == s).sum())
        print(f"  {s}: {n_s} clips ({100*n_s/n:.1f}%), class counts: {dict(final_counts[s])}")

    val_f1, n_train, n_val = trivial_probe_on_split(embeddings, labels, new_split)
    print(f"\n=== trivial probe on GROUP-DISJOINT val (threshold {SPLIT_THRESHOLD}) ===")
    print(f"  n_train={n_train} n_val={n_val} val_macro_f1={val_f1:.4f}")
    print(f"  (DAiSEE reference: 0.2177; original leaky OUC-CGE val: 0.9748)")

    # Diagnostic: does residual near-duplication (just below the 0.98 grouping
    # cutoff) explain why this isn't near the DAiSEE reference? Nearest-neighbor
    # similarity from new-val to new-train, same method as the original audit.
    train_idx = np.where(new_split_arr == "train")[0]
    val_idx = np.where(new_split_arr == "val")[0]
    sim_v2tr = sim[np.ix_(val_idx, train_idx)]
    best_sim = sim_v2tr.max(axis=1)
    print(f"\n=== residual val->train similarity AFTER group-disjoint resplit ===")
    print(f"  mean={best_sim.mean():.4f} median={np.median(best_sim):.4f} "
          f"p90={np.percentile(best_sim,90):.4f} max={best_sim.max():.4f}")
    for thresh in [0.90, 0.95, 0.98]:
        print(f"    pct >= {thresh}: {100*(best_sim>=thresh).mean():.2f}%")
    report["resplit_residual_val_to_train_similarity"] = {
        "mean": float(best_sim.mean()), "median": float(np.median(best_sim)),
        "p90": float(np.percentile(best_sim, 90)), "max": float(best_sim.max()),
        "pct_ge_0.90": float(100*(best_sim>=0.90).mean()),
        "pct_ge_0.95": float(100*(best_sim>=0.95).mean()),
        "pct_ge_0.98": float(100*(best_sim>=0.98).mean()),
    }

    report["split_threshold"] = SPLIT_THRESHOLD
    report["n_independent_recordings_at_split_threshold"] = n_components_at_split_thresh
    report["resplit_class_counts"] = {
        s: {str(k): int(v) for k, v in final_counts[s].items()} for s in TARGET_FRACS
    }
    report["resplit_n_train"] = n_train
    report["resplit_n_val"] = n_val
    report["resplit_trivial_probe_val_macro_f1"] = float(val_f1)

    with open(os.path.join(OUT_DIR, "resplit_viability_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR}/resplit_viability_report.json")


if __name__ == "__main__":
    main()
