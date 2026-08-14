"""Phase 3.5 data audit — near-duplicate / leakage check across OUC-CGE splits.
AUDIT ONLY: no model fit on test, no hyperparameter touched. Uses only the
already-extracted frozen DINOv2 feature cache (Phase 2) — computing a clip-level
embedding and cosine similarity does not evaluate or select anything.

Clip-level embedding = masked mean over all (entity, segment) positions in the
cached entity feature tensor (E=16 cache for OUC-CGE, E=8 for DAiSEE — whatever
each dataset's cache actually holds; this is a visual-fingerprint diagnostic, not
tied to the locked E=8 primary condition).
"""
import glob
import json
import os

import numpy as np

FEATURES_ROOT = "/home/devops/ept/cache/features"
OUT_DIR = "/home/devops/ept/outputs/phase3_5_audit"


def clip_level_embeddings(dataset, split):
    ent_dir = os.path.join(FEATURES_ROOT, dataset, split)
    clip_ids, embeddings = [], []
    for fp in sorted(glob.glob(os.path.join(ent_dir, "*.npy"))):
        if fp.endswith("_mask.npy") or fp.endswith("_scores.npy"):
            continue
        clip_id = os.path.splitext(os.path.basename(fp))[0]
        feat = np.load(fp)  # [E, S, D]
        mask = np.load(os.path.join(ent_dir, f"{clip_id}_mask.npy"))  # [E, S]
        m = mask[..., None].astype(np.float32)
        summed = (feat.astype(np.float32) * m).sum(axis=(0, 1))
        count = max(m.sum(), 1.0)
        emb = summed / count
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        clip_ids.append(clip_id)
        embeddings.append(emb.astype(np.float32))
    return clip_ids, np.stack(embeddings)


def nearest_neighbor_similarities(query_ids, query_emb, ref_ids, ref_emb, exclude_self=False):
    """For each query, cosine similarity to its nearest neighbor in ref (both
    L2-normalized already, so cosine sim = dot product)."""
    sims = query_emb @ ref_emb.T  # [Nq, Nr]
    if exclude_self:
        for i, qid in enumerate(query_ids):
            for j, rid in enumerate(ref_ids):
                if qid == rid:
                    sims[i, j] = -1.0
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(query_ids)), best_idx]
    best_ref_id = [ref_ids[j] for j in best_idx]
    return best_sim, best_ref_id


def summarize(name, sims):
    sims = np.array(sims)
    print(f"{name}: n={len(sims)} mean={sims.mean():.4f} median={np.median(sims):.4f} "
          f"p90={np.percentile(sims,90):.4f} p99={np.percentile(sims,99):.4f} max={sims.max():.4f}")
    for thresh in [0.95, 0.98, 0.99]:
        pct = 100 * (sims >= thresh).mean()
        print(f"    pct >= {thresh}: {pct:.2f}%")
    return {
        "n": int(len(sims)), "mean": float(sims.mean()), "median": float(np.median(sims)),
        "p90": float(np.percentile(sims, 90)), "p99": float(np.percentile(sims, 99)),
        "max": float(sims.max()),
        "pct_ge_0.95": float(100 * (sims >= 0.95).mean()),
        "pct_ge_0.98": float(100 * (sims >= 0.98).mean()),
        "pct_ge_0.99": float(100 * (sims >= 0.99).mean()),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading embeddings...")
    train_ids, train_emb = clip_level_embeddings("ouccge", "train")
    val_ids, val_emb = clip_level_embeddings("ouccge", "val")
    test_ids, test_emb = clip_level_embeddings("ouccge", "test")
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    report = {}

    print("\n=== val -> train nearest neighbor ===")
    sim_v2tr, nn_v2tr = nearest_neighbor_similarities(val_ids, val_emb, train_ids, train_emb)
    report["val_to_train"] = summarize("val->train", sim_v2tr)

    print("\n=== test -> train nearest neighbor ===")
    sim_te2tr, nn_te2tr = nearest_neighbor_similarities(test_ids, test_emb, train_ids, train_emb)
    report["test_to_train"] = summarize("test->train", sim_te2tr)

    print("\n=== val -> test nearest neighbor ===")
    sim_v2te, nn_v2te = nearest_neighbor_similarities(val_ids, val_emb, test_ids, test_emb)
    report["val_to_test"] = summarize("val->test", sim_v2te)

    # Save full pair lists for step 2 (top-12 pair rendering) and step 3.
    pairs = []
    for qid, sim, rid in zip(val_ids, sim_v2tr, nn_v2tr):
        pairs.append({"pair_type": "val_to_train", "query": qid, "match": rid, "sim": float(sim)})
    for qid, sim, rid in zip(test_ids, sim_te2tr, nn_te2tr):
        pairs.append({"pair_type": "test_to_train", "query": qid, "match": rid, "sim": float(sim)})
    for qid, sim, rid in zip(val_ids, sim_v2te, nn_v2te):
        pairs.append({"pair_type": "val_to_test", "query": qid, "match": rid, "sim": float(sim)})

    with open(os.path.join(OUT_DIR, "near_duplicate_report.json"), "w") as f:
        json.dump({"summary": report, "all_pairs": pairs}, f)
    print(f"\nsaved -> {OUT_DIR}/near_duplicate_report.json")


if __name__ == "__main__":
    main()
