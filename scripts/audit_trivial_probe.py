"""Phase 3.5 data audit — trivial-feature probe: logistic regression on the
mean-pooled clip embedding alone (no temporal axis, no entity axis, no
attention). Train on TRAIN, evaluate on VAL only. AUDIT ONLY: this is not a
model in the EPT sense and is never touched by/touches test; it exists purely
to characterize the split, and its selection (there is none -- no hyperparameter
search here) has zero bearing on docs/LOCKED_RECIPE.md.
"""
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

FEATURES_ROOT = "/home/devops/ept/cache/features"
TRACKS_ROOT = "/home/devops/ept/cache/tracks"
DAISEE_TRACKS_ROOT = "/home/devops/ept/cache/tracks/daisee"
OUT_DIR = "/home/devops/ept/outputs/phase3_5_audit"


def clip_level_embeddings_and_labels(dataset, split, label_field="label", label_key=None):
    ent_dir = os.path.join(FEATURES_ROOT, dataset, split)
    tracks_dir = (os.path.join(DAISEE_TRACKS_ROOT, split) if dataset == "daisee"
                  else os.path.join(TRACKS_ROOT, split))

    labels_by_clip = {}
    for fp in glob.glob(os.path.join(tracks_dir, "*.json")):
        with open(fp) as f:
            clip = json.load(f)
        cid = clip["clip_id"]
        if dataset == "daisee":
            labels_by_clip[cid] = clip["labels"][label_key]
        else:
            labels_by_clip[cid] = clip[label_field]

    clip_ids, embeddings, labels = [], [], []
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
        clip_ids.append(clip_id)
        embeddings.append(emb.astype(np.float32))
        labels.append(labels_by_clip[clip_id])
    return clip_ids, np.stack(embeddings), np.array(labels)


def run_probe(name, dataset, label_field="label", label_key=None):
    _, X_train, y_train = clip_level_embeddings_and_labels(dataset, "train", label_field, label_key)
    _, X_val, y_val = clip_level_embeddings_and_labels(dataset, "val", label_field, label_key)

    clf = LogisticRegression(max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)
    val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")

    print(f"{name}: n_train={len(y_train)} n_val={len(y_val)} classes={sorted(set(y_train.tolist()))} "
          f"val_macro_f1={val_f1:.4f}")
    return {"name": name, "n_train": int(len(y_train)), "n_val": int(len(y_val)),
            "val_macro_f1": float(val_f1)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = {}
    results["ouccge"] = run_probe("OUC-CGE (3-class engagement)", "ouccge")
    results["daisee_engagement"] = run_probe(
        "DAiSEE (4-class Engagement, subject-disjoint)", "daisee", label_key="Engagement"
    )

    with open(os.path.join(OUT_DIR, "trivial_probe_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT_DIR}/trivial_probe_report.json")


if __name__ == "__main__":
    main()
