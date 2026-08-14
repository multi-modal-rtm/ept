"""Reusable core for the dataset admission test (CLAUDE.md: no dataset enters
this project without passing it). Operates on already-computed
(clip_ids, embeddings[N,D] L2-normalized, labels, splits) — dataset-specific
adapters in dataset_admission.py build that tuple, this module only audits it.
"""
import os

import cv2
import numpy as np


def nearest_neighbor_similarities(query_idx, ref_idx, sim, exclude_diag=False):
    sub = sim[np.ix_(query_idx, ref_idx)].copy()
    if exclude_diag:
        for qi, q in enumerate(query_idx):
            for ri, r in enumerate(ref_idx):
                if q == r:
                    sub[qi, ri] = -1.0
    best_idx_local = sub.argmax(axis=1)
    best_sim = sub[np.arange(len(query_idx)), best_idx_local]
    best_ref_global = ref_idx[best_idx_local]
    return best_sim, best_ref_global


def summarize_similarity(sims):
    sims = np.asarray(sims)
    out = {
        "n": int(len(sims)), "mean": float(sims.mean()), "median": float(np.median(sims)),
        "p90": float(np.percentile(sims, 90)), "p99": float(np.percentile(sims, 99)),
        "max": float(sims.max()),
    }
    for t in [0.95, 0.98, 0.99]:
        out[f"pct_ge_{t}"] = float(100 * (sims >= t).mean())
    return out


def cross_split_audit(clip_ids, embeddings, splits, split_pairs):
    """split_pairs: list of (query_split, ref_split) tuples, e.g. [("val","train"),
    ("test","train"), ("val","test")]. Returns {pair_name: {"summary":..., "pairs": [...]}}
    `embeddings` are L2-normalized HERE (locally, for cosine similarity only) —
    the caller's copy stays raw, so trivial_probe() below matches the original
    Phase 3.5 audit's methodology (unnormalized masked-mean features) exactly,
    not a different, unintentionally-varied probe."""
    splits_arr = np.array(splits)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms > 0, norms, 1.0)
    sim = normed @ normed.T
    report = {}
    for q_split, r_split in split_pairs:
        q_idx = np.where(splits_arr == q_split)[0]
        r_idx = np.where(splits_arr == r_split)[0]
        exclude_diag = q_split == r_split
        best_sim, best_ref_global = nearest_neighbor_similarities(q_idx, r_idx, sim, exclude_diag)
        pair_name = f"{q_split}_to_{r_split}"
        pairs = [
            {"pair_type": pair_name, "query": clip_ids[q], "match": clip_ids[r], "sim": float(s)}
            for q, r, s in zip(q_idx, best_ref_global, best_sim)
        ]
        report[pair_name] = {"summary": summarize_similarity(best_sim), "pairs": pairs}
    return report, sim


def top_n_pairs_across(report, n=12):
    all_pairs = []
    for pair_name, d in report.items():
        all_pairs.extend(d["pairs"])
    return sorted(all_pairs, key=lambda p: -p["sim"])[:n]


def render_pair_thumbnails(pairs, video_path_fn, label_fn, out_dir, height=480):
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for i, p in enumerate(pairs):
        qid, mid, sim, ptype = p["query"], p["match"], p["sim"], p["pair_type"]
        q_path = video_path_fn(qid)
        m_path = video_path_fn(mid)
        q_frame = _middle_frame(q_path)
        m_frame = _middle_frame(m_path)
        if q_frame is None or m_frame is None:
            print(f"skip {qid}/{mid}: frame grab failed")
            continue

        def resize(f):
            scale = height / f.shape[0]
            return cv2.resize(f, (int(f.shape[1] * scale), height))

        combo = cv2.hconcat([resize(q_frame), resize(m_frame)])
        label = (f"{ptype}  sim={sim:.6f}   QUERY={qid} ({label_fn(qid)})   |   "
                 f"MATCH={mid} ({label_fn(mid)})")
        canvas = cv2.copyMakeBorder(combo, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        out_path = os.path.join(out_dir, f"pair_{i:02d}_{qid}_vs_{mid}.png")
        cv2.imwrite(out_path, canvas)
        saved.append(out_path)
        print(f"saved {out_path}")
    return saved


def _middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(n // 2, 0))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def trivial_probe(embeddings, labels, splits, train_split="train", val_split="val"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    splits_arr = np.array(splits)
    labels_arr = np.array(labels)
    X_train = embeddings[splits_arr == train_split]
    y_train = labels_arr[splits_arr == train_split]
    X_val = embeddings[splits_arr == val_split]
    y_val = labels_arr[splits_arr == val_split]
    clf = LogisticRegression(max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)
    val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
    return {"n_train": int(len(y_train)), "n_val": int(len(y_val)), "val_macro_f1": float(val_f1)}
