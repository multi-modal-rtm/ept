"""Phase 4: MELD test evaluation. THE test split is touched exactly ONCE by
this script's `run_condition_seed` -- training uses only `MELDDataset("train",
...)`; the model is evaluated on `MELDTestDataset` exactly once, after all
training epochs complete, never during training and never more than once per
(condition, seed). There is deliberately no per-epoch test evaluation and no
test-based best-epoch selection anywhere in this file -- that would be a
hidden second form of test-touching (test-set model selection) even if it
never appears in a table. The locked recipe's `epochs` field (already fixed
in docs/LOCKED_RECIPE_MELD.md, chosen on dev, read-only) is trained to
completion and the FINAL epoch's model is what gets evaluated on test.

Runs: A0-A5, mask-only, x 5 seeds {42,1337,2024,7,31337} = 35 runs. Plus the
trivial-feature probe (full train->full test, not a subsample, since this is
the final reportable number). Per-run outputs: metrics.json (macro-F1,
accuracy, per-class F1) AND predictions.json (clip_ids, true labels,
predicted labels) -- the latter is what the paired bootstrap
(scripts/meld_phase4_statistics.py) reads; nothing here computes the
bootstrap itself, keeping "touch test" and "analyze what we saved" as
separate, replayable steps.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from ept.model.ept_former import assert_no_backbone_params
from ept.tokenization.extract_features_meld import MELD_PRIMARY_E_MAX, S as MELD_S
from ept.train.dataset_meld import MELDDataset
from ept.train.dataset_meld_test import MELDTestDataset
from ept.train.train import build_model, forward_pass, run_epoch, to_device

REPO_ROOT = "/home/devops/ept"
CONFIGS_ROOT = os.path.join(REPO_ROOT, "configs")
SEEDS = [42, 1337, 2024, 7, 31337]
CONDITIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "mask_only"]
# docs/LOCKED_RECIPE_MELD.md -- read-only, reproduced here verbatim, not re-derived.
LOCKED_RECIPE = {
    "A0": "r08", "A1": "r07", "A2": "r03", "A3": "r04",
    "A4": "r03", "A5": "r07", "mask_only": "r05",
}
TRIVIAL_PROBE_FLOOR = 0.4155


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def eval_full(model, ds, condition, batch_size, device):
    model.eval()
    n = len(ds)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n), device=device)
            feat, mask, label = ds.features[idx], ds.masks[idx], ds.labels[idx]
            logits = forward_pass(model, condition, feat, mask)
            all_logits.append(logits)
            all_labels.append(label)
    logits = torch.cat(all_logits).cpu()
    labels = torch.cat(all_labels).cpu()
    preds = logits.argmax(dim=-1)
    y, p = labels.numpy(), preds.numpy()
    macro_f1 = f1_score(y, p, average="macro")
    accuracy = accuracy_score(y, p)
    per_class_f1 = f1_score(y, p, average=None, labels=[0, 1, 2]).tolist()
    return macro_f1, accuracy, per_class_f1, p.tolist(), y.tolist()


def resolved_config(condition, seed):
    """Config-only, zero data touched -- the pre-launch eyeball checkpoint."""
    recipe_name = LOCKED_RECIPE[condition]
    recipe = load_yaml(os.path.join(CONFIGS_ROOT, "recipe", f"{recipe_name}.yaml"))
    return {
        "dataset": "meld", "condition": condition, "recipe_id": recipe_name,
        "recipe": recipe, "seed": seed,
        "e_max": MELD_PRIMARY_E_MAX, "s_max": MELD_S,
        "run_id": f"meld_test_{condition}_{recipe_name}_seed{seed}",
        "train_split": "train", "eval_split": "test",
    }


def run_condition_seed(condition, seed, device):
    recipe_name = LOCKED_RECIPE[condition]
    recipe = load_yaml(os.path.join(CONFIGS_ROOT, "recipe", f"{recipe_name}.yaml"))
    run_id = f"meld_test_{condition}_{recipe_name}_seed{seed}"
    results_dir = os.path.join(REPO_ROOT, "outputs", run_id)
    os.makedirs(results_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_ds = MELDDataset("train", condition, run_seed=seed)
    test_ds = MELDTestDataset(condition, run_seed=seed)
    clip_ids = list(test_ds.clip_ids)
    train_ds = to_device(train_ds, device)
    test_ds = to_device(test_ds, device)

    torch.manual_seed(seed)
    model = build_model(
        condition, recipe["dropout"], e_max=MELD_PRIMARY_E_MAX, s=MELD_S
    ).to(device)
    assert_no_backbone_params(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe["lr"], weight_decay=recipe["weight_decay"]
    )

    train_history = []
    t0 = time.time()
    for epoch in range(recipe["epochs"]):
        train_loss, train_f1 = run_epoch(
            model, train_ds, condition, recipe["batch_size"], device, optimizer, shuffle=True
        )
        train_history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1})
    train_wall = time.time() - t0

    # THE single test touch for this (condition, seed).
    test_macro_f1, test_accuracy, per_class_f1, preds, labels = eval_full(
        model, test_ds, condition, recipe["batch_size"], device
    )
    wall_clock = time.time() - t0

    metrics = {
        "run_id": run_id, "dataset": "meld", "condition": condition, "seed": seed,
        "recipe_id": recipe_name, "recipe": recipe,
        "train_split": "train", "eval_split": "test",
        "n_train": len(train_ds), "n_test": len(test_ds),
        "train_history": train_history,
        "test_macro_f1": test_macro_f1, "test_accuracy": test_accuracy,
        "test_per_class_f1": {"negative": per_class_f1[0], "neutral": per_class_f1[1],
                               "positive": per_class_f1[2]},
        "train_wall_seconds": train_wall, "wall_clock_seconds": wall_clock,
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(results_dir, "predictions.json"), "w") as f:
        json.dump({"run_id": run_id, "condition": condition, "seed": seed,
                    "clip_ids": clip_ids, "labels": labels, "preds": preds}, f)

    print(f"  {run_id}: test_macro_f1={test_macro_f1:.4f} test_acc={test_accuracy:.4f} "
          f"per_class_f1={[f'{x:.3f}' for x in per_class_f1]} ({wall_clock:.1f}s)", flush=True)
    return metrics


def trivial_probe_test():
    """Full train -> full test (not a subsample -- this is the final
    reportable standing row). Embeddings are deterministic (frozen DINOv2, no
    dropout); computed ONCE and reused across 5 LogisticRegression fits with
    different random_state, since only the solver's random_state varies
    across 'seeds' here, not the embeddings themselves."""
    import csv
    import random

    import cv2

    from ept.tokenization.detect_cluster_meld import LABEL_CSVS, MELD_ROOT, SPLIT_DIRS
    from ept.tokenization.extract_features import embed_batch, load_model, preprocess_bgr_crop

    def manifest_for(split):
        m = []
        with open(os.path.join(MELD_ROOT, "labels", LABEL_CSVS[split])) as f:
            for row in csv.DictReader(f):
                dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
                fname = f"dia{dia}_utt{utt}.mp4"
                path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS[split], fname)
                if not os.path.exists(path):
                    continue
                sent = {"negative": 0, "neutral": 1, "positive": 2}[row["Sentiment"]]
                m.append({"clip_id": f"dia{dia}_utt{utt}", "path": path, "label": sent})
        return m

    train_manifest = manifest_for("train")
    test_manifest = manifest_for("test")
    print(f"trivial probe: {len(train_manifest)} train clips, {len(test_manifest)} test clips",
          flush=True)

    model = load_model()

    def embed_all(manifest, t_frames=8):
        embs, labels, clip_ids = [], [], []
        for i, m in enumerate(manifest):
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
            embs.append(emb)
            labels.append(m["label"])
            clip_ids.append(m["clip_id"])
            if (i + 1) % 1000 == 0:
                print(f"    {i+1}/{len(manifest)} embedded", flush=True)
        return np.stack(embs), np.array(labels), clip_ids

    X_train, y_train, _ = embed_all(train_manifest)
    X_test, y_test, test_clip_ids = embed_all(test_manifest)

    results = []
    for seed in SEEDS:
        clf = LogisticRegression(max_iter=5000, random_state=seed)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        macro_f1 = f1_score(y_test, preds, average="macro")
        accuracy = accuracy_score(y_test, preds)
        per_class_f1 = f1_score(y_test, preds, average=None, labels=[0, 1, 2]).tolist()
        results.append({"seed": seed, "test_macro_f1": macro_f1, "test_accuracy": accuracy,
                         "test_per_class_f1": per_class_f1})
        if seed == SEEDS[0]:
            with open(os.path.join(REPO_ROOT, "outputs", "meld_test_trivial_probe_predictions.json"), "w") as f:
                json.dump({"clip_ids": test_clip_ids, "labels": y_test.tolist(),
                            "preds": preds.tolist()}, f)
        print(f"  trivial-probe seed={seed}: test_macro_f1={macro_f1:.4f} test_acc={accuracy:.4f}",
              flush=True)

    macro_fs = [r["test_macro_f1"] for r in results]
    summary = {
        "n_train": len(train_manifest), "n_test": len(test_manifest),
        "per_seed": results,
        "mean_test_macro_f1": float(np.mean(macro_fs)), "std_test_macro_f1": float(np.std(macro_fs)),
    }
    with open(os.path.join(REPO_ROOT, "outputs", "meld_test_trivial_probe.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"trivial probe: mean_test_macro_f1={summary['mean_test_macro_f1']:.4f} "
          f"+/- {summary['std_test_macro_f1']:.4f}", flush=True)
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-config-only", action="store_true",
                     help="Print resolved config for condition=A1, seed=42 and exit. No data touched.")
    args = ap.parse_args()

    if args.print_config_only:
        cfg = resolved_config("A1", 42)
        print(json.dumps(cfg, indent=2))
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_metrics = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            all_metrics.append(run_condition_seed(condition, seed, device))

    trivial = trivial_probe_test()

    summary_path = os.path.join(REPO_ROOT, "outputs", "meld_phase4_test_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"conditions": CONDITIONS, "seeds": SEEDS, "locked_recipe": LOCKED_RECIPE,
                    "results": all_metrics, "trivial_probe": trivial}, f, indent=2)
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
