"""MELD Phase 3 secondary calibration endpoint (docs/DECISION_RULES.md,
2026-08-15 amendment): 7-class emotion, A0/A1/A2 + trivial probe, dev only,
reusing the already-locked recipe per condition (docs/LOCKED_RECIPE_MELD.md) —
no new search. NOT part of H1/H2 or the branch decision; purpose is checking
this pipeline's numbers land in a sane range next to published video-only
MELD EMOTION baselines (A6 found none for sentiment, but did find some for
emotion). No test split touched anywhere in this script.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import yaml

from ept.model.ept_former import EPTFormer, assert_no_backbone_params
from ept.tokenization.extract_features_meld import MELD_PRIMARY_E_MAX, S as MELD_S
from ept.train.dataset_meld_emotion import MELDEmotionDataset
from ept.train.emotion_labels import NUM_EMOTION_CLASSES
from ept.train.train import forward_pass, run_epoch, to_device

REPO_ROOT = "/home/devops/ept"
CONFIGS_ROOT = os.path.join(REPO_ROOT, "configs")
SEED = 42
# Locked per docs/LOCKED_RECIPE_MELD.md -- read-only, not re-searched here.
LOCKED_RECIPE = {"A0": "r08", "A1": "r07", "A2": "r03"}
TRIVIAL_PROBE_FLOOR_SENTIMENT = 0.4155  # for reference only; not the metric this endpoint reports


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def eval_both_f1(model, ds, condition, batch_size, device):
    """Most published MELD emotion baselines report weighted-F1 (the metric
    the original MELD paper used), not macro-F1 (this project's own metric
    throughout, chosen for class-imbalance sensitivity). Computing both here
    rather than only macro-F1 avoids an apples-to-oranges comparison against
    the literature in the calibration verdict."""
    import torch as _torch
    from sklearn.metrics import f1_score

    model.eval()
    n = len(ds)
    all_logits, all_labels = [], []
    with _torch.no_grad():
        for i in range(0, n, batch_size):
            idx = _torch.arange(i, min(i + batch_size, n), device=device)
            feat, mask, label = ds.features[idx], ds.masks[idx], ds.labels[idx]
            logits = forward_pass(model, condition, feat, mask)
            all_logits.append(logits)
            all_labels.append(label)
    logits = _torch.cat(all_logits).cpu()
    labels = _torch.cat(all_labels).cpu()
    preds = logits.argmax(dim=-1)
    macro_f1 = f1_score(labels.numpy(), preds.numpy(), average="macro")
    weighted_f1 = f1_score(labels.numpy(), preds.numpy(), average="weighted")
    return float(macro_f1), float(weighted_f1)


def run_condition(condition, device):
    recipe_name = LOCKED_RECIPE[condition]
    recipe = load_yaml(os.path.join(CONFIGS_ROOT, "recipe", f"{recipe_name}.yaml"))
    run_id = f"meld_emotion_{condition}_{recipe_name}_seed{SEED}"
    results_dir = os.path.join(REPO_ROOT, "outputs", run_id)
    os.makedirs(results_dir, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_ds = to_device(MELDEmotionDataset("train", condition, run_seed=SEED), device)
    dev_ds = to_device(MELDEmotionDataset("dev", condition, run_seed=SEED), device)

    torch.manual_seed(SEED)
    model = EPTFormer(
        dropout=recipe["dropout"], use_temporal=True, use_social=True,
        s_max=MELD_S, num_classes=NUM_EMOTION_CLASSES,
    ).to(device)
    assert_no_backbone_params(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )

    history = []
    t0 = time.time()
    best_dev_f1, best_epoch = -1.0, -1
    best_state = None
    for epoch in range(recipe["epochs"]):
        train_loss, train_f1 = run_epoch(
            model, train_ds, condition, recipe["batch_size"], device, optimizer, shuffle=True
        )
        dev_loss, dev_f1 = run_epoch(
            model, dev_ds, condition, recipe["batch_size"], device, optimizer=None, shuffle=False
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                         "val_loss": dev_loss, "val_macro_f1": dev_f1})
        if dev_f1 > best_dev_f1:
            best_dev_f1, best_epoch = dev_f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    wall_clock = time.time() - t0

    model.load_state_dict(best_state)
    best_macro_f1_check, best_weighted_f1 = eval_both_f1(
        model, dev_ds, condition, recipe["batch_size"], device
    )
    assert abs(best_macro_f1_check - best_dev_f1) < 1e-6, (
        f"best-epoch reload mismatch: {best_macro_f1_check} vs {best_dev_f1}"
    )

    metrics = {
        "run_id": run_id, "condition": condition, "seed": SEED, "task": "meld_emotion_7class",
        "recipe_id": recipe_name, "recipe": recipe, "split_evaluated": "dev",
        "n_train": len(train_ds), "n_dev": len(dev_ds),
        "history": history,
        "best_dev_macro_f1": best_dev_f1, "best_dev_weighted_f1": best_weighted_f1,
        "best_epoch": best_epoch,
        "final_dev_macro_f1": history[-1]["val_macro_f1"],
        "wall_clock_seconds": wall_clock,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  {run_id}: best_dev_macro_f1={best_dev_f1:.4f} best_dev_weighted_f1={best_weighted_f1:.4f} "
          f"@ epoch {best_epoch} ({wall_clock:.1f}s)", flush=True)
    return metrics


def trivial_probe_emotion(n_sample=1500, t_frames=8, seed=42):
    """Same methodology as scripts/dataset_admission_lib.py's trivial_probe:
    mean-pooled clip embedding (CLS ++ mean-patch, frozen DINOv2, raw frames,
    no temporal/entity/attention structure), logistic regression. Restricted
    to train+dev only -- no test split sampled, matching this endpoint's
    dev-only scope (unlike the original admission-test adapter, which also
    samples test for its cross-split near-duplicate audit; that audit is not
    needed here)."""
    import random

    import cv2
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    from ept.tokenization.detect_cluster_meld import LABEL_CSVS, MELD_ROOT, SPLIT_DIRS
    from ept.tokenization.extract_features import embed_batch, load_model, preprocess_bgr_crop
    from ept.train.emotion_labels import EMOTION_TO_INT

    import csv as csv_mod

    manifest = []
    for split in ("train", "dev"):
        with open(os.path.join(MELD_ROOT, "labels", LABEL_CSVS[split])) as f:
            for row in csv_mod.DictReader(f):
                dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
                fname = f"dia{dia}_utt{utt}.mp4"
                path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS[split], fname)
                if not os.path.exists(path):
                    continue
                manifest.append({"clip_id": f"dia{dia}_utt{utt}", "split": split, "path": path,
                                  "emotion": EMOTION_TO_INT[row["Emotion"]]})
    print(f"emotion trivial-probe manifest: {len(manifest)} clips (train+dev only)")

    rng = random.Random(seed)
    by_split = {}
    for m in manifest:
        by_split.setdefault(m["split"], []).append(m)
    split_totals = {s: len(v) for s, v in by_split.items()}
    grand_total = sum(split_totals.values())
    sample = []
    for split, items in by_split.items():
        quota = round(n_sample * split_totals[split] / grand_total)
        by_emo = {}
        for it in items:
            by_emo.setdefault(it["emotion"], []).append(it)
        emo_totals = {e: len(v) for e, v in by_emo.items()}
        split_grand = sum(emo_totals.values())
        for emo, emo_items in by_emo.items():
            emo_quota = min(round(quota * emo_totals[emo] / split_grand), len(emo_items))
            sample.extend(rng.sample(emo_items, emo_quota))
    print(f"stratified sample: {len(sample)} clips")

    model = load_model()
    embeddings, labels, splits = [], [], []
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
        emb = embed_batch(model, crops).astype(np.float32).mean(axis=0)
        embeddings.append(emb)
        labels.append(m["emotion"])
        splits.append(m["split"])
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(sample)} embedded", flush=True)

    embeddings = np.stack(embeddings)
    splits_arr = np.array(splits)
    labels_arr = np.array(labels)
    X_train, y_train = embeddings[splits_arr == "train"], labels_arr[splits_arr == "train"]
    X_dev, y_dev = embeddings[splits_arr == "dev"], labels_arr[splits_arr == "dev"]
    clf = LogisticRegression(max_iter=5000, random_state=seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_dev)
    dev_macro_f1 = f1_score(y_dev, preds, average="macro")
    dev_weighted_f1 = f1_score(y_dev, preds, average="weighted")
    result = {"n_train": int(len(y_train)), "n_dev": int(len(y_dev)),
              "dev_macro_f1": float(dev_macro_f1), "dev_weighted_f1": float(dev_weighted_f1)}
    print(f"trivial-probe (emotion, dev): macro_f1={dev_macro_f1:.4f} weighted_f1={dev_weighted_f1:.4f} "
          f"(n_train={result['n_train']}, n_dev={result['n_dev']})", flush=True)
    return result


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}
    for condition in ["A0", "A1", "A2"]:
        print(f"=== condition {condition} (7-class emotion, locked recipe {LOCKED_RECIPE[condition]}) ===",
              flush=True)
        all_results[condition] = run_condition(condition, device)

    print("\n=== trivial-feature probe (7-class emotion, dev) ===", flush=True)
    probe_result = trivial_probe_emotion()
    all_results["trivial_probe"] = probe_result

    summary_path = os.path.join(REPO_ROOT, "outputs", "meld_emotion_calibration_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved -> {summary_path}")


if __name__ == "__main__":
    main()
