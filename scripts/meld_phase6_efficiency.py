"""Phase 6 efficiency measurement (docs/PLAN.md Sec.6). No MELD test
re-evaluation anywhere in this file -- it profiles compute cost (FLOPs,
latency, memory), not accuracy, and never loads labels.

Reports, per condition (A0-A5, mask-only) and every budget-sweep cell
(E in {1,2,4,6,8} x S in {2,4}):
  - token count (E*S, or 8*S for A0's fixed 8-region grid)
  - backbone GFLOPs/clip = (measured average raw-crop count/clip) x
    (thop-measured per-crop DINOv2 GFLOPs) -- NOT estimated from E*S, since
    the backbone runs once per detected face-crop-frame BEFORE segment
    pooling, and that count depends on real per-clip detection density, not
    just the token budget. Measured from the actual cached track JSONs.
  - attention-stack GFLOPs/clip (thop, the EPTFormer/MeanPoolMLP/MaskOnlyMLP
    forward pass only) -- this is the piece "GFLOPs (ptflops/fvcore)" in
    PLAN.md Sec.6 refers to; thop substituted since it's what's installed
    (fvcore/ptflops are not, and installing them risks re-triggering the
    opencv-headless/opencv-python conflict documented in CLAUDE.md).
  - total model GFLOPs/clip = backbone + attention-stack. This is what
    varies across the budget sweep -- entity tokenization does NOT reduce
    detection cost (every frame still gets face-detected regardless of E),
    only backbone+attention-stack cost, per PLAN.md's explicit warning.

Detection+clustering cost is measured SEPARATELY, single-clip, single-thread
wall-clock latency -- not folded into the GFLOPs table, not amortized across
parallel workers (that would report throughput, not the per-clip cost a
deployed single-stream system actually pays).
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cv2
import numpy as np
import torch
from thop import profile

from ept.model.ept_former import EPTFormer, MeanPoolMLP, MaskOnlyMLP
from ept.tokenization.extract_features import CROP_SIZE, GRID_COLS, GRID_ROWS, load_model
from ept.tokenization.extract_features_meld import MELD_CACHE_E_MAX, S as MELD_S, T as MELD_T
from ept.tokenization.detect_cluster_meld import (
    LABEL_CSVS, MELD_ROOT, SPLIT_DIRS, _init_worker, cluster_identities, detect_faces,
)

REPO_ROOT = "/home/devops/ept"
TRACKS_ROOT = "/home/devops/ept/cache/tracks/meld"
CONDITIONS = ["A0", "A1", "A2", "A3", "A4", "A5", "mask_only"]
LOCKED_RECIPE_E_S = {
    "A0": (8, MELD_S),  # fixed 8-region grid, always
    "A1": (6, MELD_S), "A2": (6, MELD_S), "A3": (6, MELD_S), "A4": (6, MELD_S),
    "A5": (6, MELD_S), "mask_only": (6, MELD_S),
}
BUDGET_E_GRID = [1, 2, 4, 6, 8]
BUDGET_S_GRID = [2, 4]
CLUSTER_THRESHOLD = 0.55
N_CLIPS_FOR_CROP_COUNT = 500  # sample of test clips, deterministic (sorted, first N)
N_CLIPS_FOR_DETECTION_LATENCY = 30


def select_top_e_frame_counts(clip, e_max):
    """Mirrors extract_features_meld.py's select_top_e + the crop-emitting
    loop in extract_entity_features exactly, but only COUNTS crops (one per
    (entity, frame-detection) pair actually cropped) instead of running the
    backbone -- this is what determines backbone forward-pass count, not S."""
    n_total = clip["n_frames_grabbed"]
    track_frames = {}
    for frame in clip["frames"]:
        for det in frame["detections"]:
            tid = det.get("track_id")
            if tid is None:
                continue
            track_frames.setdefault(tid, []).append((frame["frame_pos"], det["bbox"], det["confidence"] or 0.0))
    scored = []
    for tid, entries in track_frames.items():
        coverage = len(entries) / n_total if n_total else 0.0
        mean_conf = float(np.mean([c for _, _, c in entries]))
        scored.append((mean_conf * coverage, tid, entries))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_e = scored[:e_max]
    return sum(len(entries) for _, _, entries in top_e)


def measure_avg_entity_crops_per_clip(e_max, n_clips=N_CLIPS_FOR_CROP_COUNT):
    files = sorted(glob.glob(os.path.join(TRACKS_ROOT, "test", "*.json")))[:n_clips]
    counts = []
    for fp in files:
        with open(fp) as f:
            clip = json.load(f)
        counts.append(select_top_e_frame_counts(clip, e_max))
    return float(np.mean(counts)), float(np.std(counts)), len(counts)


def measure_avg_grid_crops_per_clip(n_clips=N_CLIPS_FOR_CROP_COUNT):
    """Grid path crops every frame at GRID_ROWS*GRID_COLS regions, always --
    no detection/entity-count dependence, so this should be ~constant at
    GRID_ROWS*GRID_COLS*n_frames_grabbed."""
    files = sorted(glob.glob(os.path.join(TRACKS_ROOT, "test", "*.json")))[:n_clips]
    counts = []
    for fp in files:
        with open(fp) as f:
            clip = json.load(f)
        n_frames_present = sum(1 for fr in clip["frames"])  # extract_grid_features iterates clip["frames"]
        counts.append(GRID_ROWS * GRID_COLS * n_frames_present)
    return float(np.mean(counts)), float(np.std(counts)), len(counts)


def profile_backbone_per_crop():
    model = load_model()
    x = torch.randn(1, 3, CROP_SIZE, CROP_SIZE).to("cuda", dtype=torch.float16)
    macs, params = profile(model, inputs=(x,), verbose=False)
    return 2 * macs / 1e9, int(params)  # GFLOPs, params


def build_model_for_profile(condition, e, s):
    if condition == "A5":
        return MeanPoolMLP(dropout=0.0)
    if condition == "mask_only":
        return MaskOnlyMLP(e_max=e, s=s, dropout=0.0)
    use_temporal = condition != "A4"
    use_social = condition != "A3"
    return EPTFormer(dropout=0.0, use_temporal=use_temporal, use_social=use_social, s_max=s)


def profile_attention_stack(condition, e, s):
    model = build_model_for_profile(condition, e, s)
    if condition == "mask_only":
        mask = torch.ones(1, e, s, dtype=torch.bool)
        macs, params = profile(model, inputs=(mask,), verbose=False)
    else:
        feat = torch.randn(1, e, s, 1536)
        mask = torch.ones(1, e, s, dtype=torch.bool)
        macs, params = profile(model, inputs=(feat, mask), verbose=False)
    return 2 * macs / 1e9, int(params)


def measure_detection_clustering_latency(n_clips=N_CLIPS_FOR_DETECTION_LATENCY):
    """Single-clip, single-process wall-clock latency -- NOT amortized across
    parallel workers (that measures throughput, a different, more flattering
    number than what a single deployed stream actually experiences)."""
    _init_worker()
    manifest = []
    with open(os.path.join(MELD_ROOT, "labels", LABEL_CSVS["test"])) as f:
        import csv
        for row in csv.DictReader(f):
            dia, utt = row["Dialogue_ID"], row["Utterance_ID"]
            fname = f"dia{dia}_utt{utt}.mp4"
            path = os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS["test"], fname)
            if os.path.exists(path):
                manifest.append(path)
            if len(manifest) >= n_clips:
                break

    times_detect, times_cluster, times_total = [], [], []
    for path in manifest:
        t0 = time.time()
        records = detect_faces(path)
        t1 = time.time()
        cluster_identities(records, CLUSTER_THRESHOLD)
        t2 = time.time()
        times_detect.append(t1 - t0)
        times_cluster.append(t2 - t1)
        times_total.append(t2 - t0)
    return {
        "n_clips": len(manifest),
        "detect_ms_mean": float(np.mean(times_detect) * 1000), "detect_ms_std": float(np.std(times_detect) * 1000),
        "cluster_ms_mean": float(np.mean(times_cluster) * 1000), "cluster_ms_std": float(np.std(times_cluster) * 1000),
        "total_ms_mean": float(np.mean(times_total) * 1000), "total_ms_std": float(np.std(times_total) * 1000),
    }


def measure_latency_gpu(condition, e, s, n_reps=100, warmup=20):
    model = build_model_for_profile(condition, e, s).to("cuda").half().eval()
    if condition == "mask_only":
        inp = (torch.ones(1, e, s, dtype=torch.bool, device="cuda"),)
    else:
        feat = torch.randn(1, e, s, 1536, device="cuda", dtype=torch.float16)
        mask = torch.ones(1, e, s, dtype=torch.bool, device="cuda")
        inp = (feat, mask)

    with torch.no_grad():
        for _ in range(warmup):
            model(*inp)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(n_reps):
            model(*inp)
        torch.cuda.synchronize()
        t1 = time.time()
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {"latency_ms_mean": (t1 - t0) / n_reps * 1000, "peak_mem_mb": peak_mem_mb}


def measure_latency_cpu(condition, e, s, n_threads, n_reps=30, warmup=5):
    torch.set_num_threads(n_threads)
    model = build_model_for_profile(condition, e, s).eval()
    if condition == "mask_only":
        inp = (torch.ones(1, e, s, dtype=torch.bool),)
    else:
        feat = torch.randn(1, e, s, 1536)
        mask = torch.ones(1, e, s, dtype=torch.bool)
        inp = (feat, mask)

    with torch.no_grad():
        for _ in range(warmup):
            model(*inp)
        t0 = time.time()
        for _ in range(n_reps):
            model(*inp)
        t1 = time.time()
    return (t1 - t0) / n_reps * 1000


def main():
    print("=== backbone per-crop GFLOPs ===", flush=True)
    backbone_gflops_per_crop, backbone_params = profile_backbone_per_crop()
    print(f"backbone: {backbone_gflops_per_crop:.4f} GFLOPs/crop, {backbone_params:,} params", flush=True)

    print("\n=== detection+clustering latency (single-clip, single-thread) ===", flush=True)
    det_latency = measure_detection_clustering_latency()
    print(json.dumps(det_latency, indent=2), flush=True)

    print("\n=== per-condition efficiency ===", flush=True)
    condition_rows = []
    for condition in CONDITIONS:
        e, s = LOCKED_RECIPE_E_S[condition]
        if condition == "A0":
            avg_crops, std_crops, n = measure_avg_grid_crops_per_clip()
            tokens = GRID_ROWS * GRID_COLS * s  # matched-budget token count (8 regions x S segments)
        else:
            avg_crops, std_crops, n = measure_avg_entity_crops_per_clip(e)
            tokens = e * s
        att_gflops, att_params = profile_attention_stack(condition, e, s)
        backbone_gflops = avg_crops * backbone_gflops_per_crop
        total_gflops = backbone_gflops + att_gflops
        gpu = measure_latency_gpu(condition, e, s)
        cpu_latencies = {t: measure_latency_cpu(condition, e, s, t) for t in (1, 8, 60)}
        row = {
            "condition": condition, "e": e, "s": s, "tokens": tokens,
            "avg_raw_crops_per_clip": avg_crops, "std_raw_crops_per_clip": std_crops, "n_clips_sampled": n,
            "backbone_gflops_per_clip": backbone_gflops, "attention_stack_gflops": att_gflops,
            "attention_stack_params": att_params, "total_model_gflops_per_clip": total_gflops,
            "gpu_batch1_fp16_latency_ms": gpu["latency_ms_mean"], "gpu_peak_mem_mb": gpu["peak_mem_mb"],
            "cpu_latency_ms": cpu_latencies,
        }
        condition_rows.append(row)
        print(f"  {condition}: tokens={tokens} avg_crops={avg_crops:.1f} "
              f"backbone_gflops={backbone_gflops:.2f} attn_gflops={att_gflops:.4f} "
              f"total_gflops={total_gflops:.2f} gpu_lat={gpu['latency_ms_mean']:.3f}ms "
              f"gpu_mem={gpu['peak_mem_mb']:.1f}MB cpu1={cpu_latencies[1]:.2f}ms "
              f"cpu8={cpu_latencies[8]:.2f}ms cpu60={cpu_latencies[60]:.2f}ms", flush=True)

    print("\n=== budget-sweep cell efficiency (condition=A1 architecture) ===", flush=True)
    budget_rows = []
    for e in BUDGET_E_GRID:
        for s in BUDGET_S_GRID:
            avg_crops, std_crops, n = measure_avg_entity_crops_per_clip(e)
            att_gflops, att_params = profile_attention_stack("A1", e, s)
            backbone_gflops = avg_crops * backbone_gflops_per_crop
            total_gflops = backbone_gflops + att_gflops
            gpu = measure_latency_gpu("A1", e, s)
            row = {
                "e": e, "s": s, "tokens": e * s,
                "avg_raw_crops_per_clip": avg_crops, "std_raw_crops_per_clip": std_crops,
                "backbone_gflops_per_clip": backbone_gflops, "attention_stack_gflops": att_gflops,
                "total_model_gflops_per_clip": total_gflops,
                "gpu_batch1_fp16_latency_ms": gpu["latency_ms_mean"], "gpu_peak_mem_mb": gpu["peak_mem_mb"],
            }
            budget_rows.append(row)
            print(f"  E={e} S={s}: tokens={e*s} avg_crops={avg_crops:.1f} "
                  f"total_gflops={total_gflops:.2f} gpu_lat={gpu['latency_ms_mean']:.3f}ms", flush=True)

    out = {
        "backbone_gflops_per_crop": backbone_gflops_per_crop, "backbone_params": backbone_params,
        "detection_clustering_latency": det_latency,
        "condition_rows": condition_rows, "budget_sweep_rows": budget_rows,
    }
    out_path = os.path.join(REPO_ROOT, "outputs", "meld_phase6_efficiency.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
