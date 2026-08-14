"""Dataset admission test — CLAUDE.md: no dataset enters this project without
passing this. Cross-split near-duplicate analysis + trivial-feature probe,
compared against a subject-disjoint reference (DAiSEE). One command:

    python3 scripts/dataset_admission.py --dataset ouccge
    python3 scripts/dataset_admission.py --dataset daisee
    python3 scripts/dataset_admission.py --dataset meld --n-sample 1500

AUDIT ONLY. No model is fit on any dataset's test split by this script; the
trivial probe here trains on train/evaluates on val, exactly like every other
admission check in this project.
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import cv2
import numpy as np

from dataset_admission_lib import cross_split_audit, render_pair_thumbnails, top_n_pairs_across, trivial_probe

FEATURES_ROOT = "/home/devops/ept/cache/features"
TRACKS_ROOT = "/home/devops/ept/cache/tracks"
OUT_ROOT = "/home/devops/ept/outputs/dataset_admission"

MELD_ROOT = "/home/devops/socialarcnet-v2/data/meld/raw"
MELD_SPLIT_DIRS = {
    "train": "train_splits", "val": "dev_splits_complete", "test": "output_repeated_splits_test",
}
MELD_LABEL_CSVS = {"train": "train_sent_emo.csv", "val": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}
SENTIMENT_TO_INT = {"negative": 0, "neutral": 1, "positive": 2}


def _masked_mean(feat, mask):
    """Raw (unnormalized) masked mean, matching the original Phase 3.5 audit's
    trivial-probe methodology. cross_split_audit() normalizes its own local copy
    for cosine similarity; this function's output feeds trivial_probe() as-is."""
    m = mask[..., None].astype(np.float32)
    summed = (feat.astype(np.float32) * m).sum(axis=(0, 1))
    count = max(m.sum(), 1.0)
    return summed / count


def load_ouccge():
    clip_ids, embeddings, labels, splits = [], [], [], []
    video_paths = {}
    for split in ["train", "val", "test"]:
        ent_dir = os.path.join(FEATURES_ROOT, "ouccge", split)
        tracks_dir = os.path.join(TRACKS_ROOT, split)
        labels_by_clip, path_by_clip = {}, {}
        for fp in glob.glob(os.path.join(tracks_dir, "*.json")):
            with open(fp) as f:
                clip = json.load(f)
            labels_by_clip[clip["clip_id"]] = clip["label"]
            path_by_clip[clip["clip_id"]] = os.path.join(
                "/home/devops/data/OUC-CGE", clip["rel_path"].replace("videos/", "")
            )
        for fp in sorted(glob.glob(os.path.join(ent_dir, "*.npy"))):
            if fp.endswith("_mask.npy") or fp.endswith("_scores.npy"):
                continue
            cid = os.path.splitext(os.path.basename(fp))[0]
            if cid not in labels_by_clip:
                continue
            feat = np.load(fp)
            mask = np.load(os.path.join(ent_dir, f"{cid}_mask.npy"))
            clip_ids.append(cid)
            embeddings.append(_masked_mean(feat, mask))
            labels.append(labels_by_clip[cid])
            splits.append(split)
            video_paths[cid] = path_by_clip[cid]
    return clip_ids, np.stack(embeddings), labels, splits, video_paths


def load_daisee():
    clip_ids, embeddings, labels, splits = [], [], [], []
    video_paths = {}
    for split in ["train", "val", "test"]:
        ent_dir = os.path.join(FEATURES_ROOT, "daisee", split)
        tracks_dir = os.path.join(TRACKS_ROOT, "daisee", split)
        labels_by_clip, path_by_clip = {}, {}
        for fp in glob.glob(os.path.join(tracks_dir, "*.json")):
            with open(fp) as f:
                clip = json.load(f)
            labels_by_clip[clip["clip_id"]] = clip["labels"]["Engagement"]
            path_by_clip[clip["clip_id"]] = os.path.join(
                "/home/devops/data/DAiSEE", clip["rel_path"]
            )
        for fp in sorted(glob.glob(os.path.join(ent_dir, "*.npy"))):
            if fp.endswith("_mask.npy") or fp.endswith("_scores.npy"):
                continue
            cid = os.path.splitext(os.path.basename(fp))[0]
            if cid not in labels_by_clip:
                continue
            feat = np.load(fp)
            mask = np.load(os.path.join(ent_dir, f"{cid}_mask.npy"))
            clip_ids.append(cid)
            embeddings.append(_masked_mean(feat, mask))
            labels.append(labels_by_clip[cid])
            splits.append(split)
            video_paths[cid] = path_by_clip[cid]
    return clip_ids, np.stack(embeddings), labels, splits, video_paths


def _meld_manifest():
    manifest = []
    for split, csv_name in MELD_LABEL_CSVS.items():
        with open(os.path.join(MELD_ROOT, "labels", csv_name)) as f:
            for row in csv.DictReader(f):
                dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
                fname = f"dia{dia}_utt{utt}.mp4"
                path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", MELD_SPLIT_DIRS[split], fname)
                if not os.path.exists(path):
                    continue
                manifest.append({
                    "clip_id": f"{split}_dia{dia}_utt{utt}", "split": split, "path": path,
                    "sentiment": SENTIMENT_TO_INT[row["Sentiment"]],
                })
    return manifest


def _stratified_sample_meld(manifest, n_total, seed=42):
    rng = random.Random(seed)
    by_split = {}
    for m in manifest:
        by_split.setdefault(m["split"], []).append(m)
    split_totals = {s: len(v) for s, v in by_split.items()}
    grand_total = sum(split_totals.values())

    sample = []
    for split, items in by_split.items():
        quota = round(n_total * split_totals[split] / grand_total)
        by_sent = {}
        for it in items:
            by_sent.setdefault(it["sentiment"], []).append(it)
        sent_totals = {s: len(v) for s, v in by_sent.items()}
        split_grand = sum(sent_totals.values())
        for sent, sent_items in by_sent.items():
            sent_quota = round(quota * sent_totals[sent] / split_grand)
            sent_quota = min(sent_quota, len(sent_items))
            sample.extend(rng.sample(sent_items, sent_quota))
    return sample


def load_meld(n_sample=1500, t_frames=8, crop_size=224, seed=42):
    from ept.tokenization.extract_features import load_model, embed_batch, preprocess_bgr_crop

    manifest = _meld_manifest()
    print(f"MELD manifest: {len(manifest)} clips with video on disk")
    sample = _stratified_sample_meld(manifest, n_sample, seed)
    print(f"stratified sample: {len(sample)} clips")
    print(f"  by split: {Counter(m['split'] for m in sample)}")
    print(f"  by sentiment: {Counter(m['sentiment'] for m in sample)}")

    model = load_model()
    clip_ids, embeddings, labels, splits = [], [], [], []
    video_paths = {}
    for i, m in enumerate(sample):
        cap = cv2.VideoCapture(m["path"])
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idxs = np.linspace(0, max(n_frames - 1, 0), t_frames).astype(int)
        crops = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                crops.append(preprocess_bgr_crop(frame))
        cap.release()
        if not crops:
            continue
        emb = embed_batch(model, crops).astype(np.float32).mean(axis=0)  # raw, unnormalized

        clip_ids.append(m["clip_id"])
        embeddings.append(emb)
        labels.append(m["sentiment"])
        splits.append(m["split"])
        video_paths[m["clip_id"]] = m["path"]
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sample)} embedded")

    return clip_ids, np.stack(embeddings), labels, splits, video_paths


ADAPTERS = {"ouccge": load_ouccge, "daisee": load_daisee, "meld": load_meld}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ADAPTERS))
    ap.add_argument("--n-sample", type=int, default=1500, help="MELD only")
    ap.add_argument("--n-pairs", type=int, default=12)
    args = ap.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== loading {args.dataset} ===")
    if args.dataset == "meld":
        clip_ids, embeddings, labels, splits, video_paths = load_meld(n_sample=args.n_sample)
    else:
        clip_ids, embeddings, labels, splits, video_paths = ADAPTERS[args.dataset]()
    print(f"n={len(clip_ids)}  splits={Counter(splits)}  labels={Counter(labels)}")

    label_by_id = dict(zip(clip_ids, labels))
    split_by_id = dict(zip(clip_ids, splits))

    print("\n=== cross-split near-duplicate audit ===")
    pairs_to_check = [("val", "train"), ("test", "train"), ("val", "test")]
    pairs_to_check = [(q, r) for q, r in pairs_to_check
                       if q in set(splits) and r in set(splits)]
    audit_report, _ = cross_split_audit(clip_ids, embeddings, splits, pairs_to_check)
    for name, d in audit_report.items():
        s = d["summary"]
        print(f"{name}: mean={s['mean']:.4f} median={s['median']:.4f} "
              f"pct>=0.95={s['pct_ge_0.95']:.2f}% pct>=0.98={s['pct_ge_0.98']:.2f}% "
              f"pct>=0.99={s['pct_ge_0.99']:.2f}%")

    print(f"\n=== rendering top {args.n_pairs} pairs ===")
    top_pairs = top_n_pairs_across(audit_report, n=args.n_pairs)
    render_pair_thumbnails(
        top_pairs,
        video_path_fn=lambda cid: video_paths[cid],
        label_fn=lambda cid: f"{split_by_id[cid]}, label={label_by_id[cid]}",
        out_dir=os.path.join(out_dir, "pairs"),
    )

    print("\n=== trivial-feature probe (train -> val) ===")
    probe_result = None
    if "train" in splits and "val" in splits:
        probe_result = trivial_probe(embeddings, labels, splits)
        print(f"val_macro_f1={probe_result['val_macro_f1']:.4f} "
              f"(n_train={probe_result['n_train']}, n_val={probe_result['n_val']})")

    report = {
        "dataset": args.dataset, "n_clips": len(clip_ids),
        "split_counts": dict(Counter(splits)), "label_counts": dict(Counter(labels)),
        "near_duplicate_audit": {k: v["summary"] for k, v in audit_report.items()},
        "trivial_probe": probe_result,
    }
    with open(os.path.join(out_dir, "admission_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {out_dir}/admission_report.json")


if __name__ == "__main__":
    main()
