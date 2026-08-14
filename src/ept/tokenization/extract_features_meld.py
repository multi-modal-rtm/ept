"""MELD feature cache (Phase-2 equivalent, promoted-primary dataset). Same
layout, same manifest/checksum/spot-verify/tar-then-rclone discipline as
extract_features.py (CLAUDE.md hard-won fact: tar before rclone).

Geometry: MELD clips are ~3s (vs OUC-CGE's longer clips), so S=4 not S=8.
E_max=8 cached (matching the OUC-CGE cache convention of caching headroom above
the locked primary), **primary E=6** — see the speaker-count distribution this
module reports for the justification (run scripts/meld_speaker_distribution.py).
T=16 (src/ept/tokenization/detect_cluster_meld.py) — 4 frames/segment at S=4,
matching OUC-CGE's 4 frames/segment at T=32/S=8.

Grid baseline: same 2x4=8-region tiling as OUC-CGE, same backbone, same
crop-then-CLS+mean-patch path, now pooled over S=4 segments instead of 8.
"""
import glob
import json
import os

import cv2
import numpy as np

from ept.tokenization.extract_features import (
    D, GRID_COLS, GRID_ROWS, embed_batch, preprocess_bgr_crop, sha256_of_file,
)

MELD_CACHE_E_MAX = 8
MELD_PRIMARY_E_MAX = 6
S = 4
T = 16

MELD_ROOT = "/home/devops/socialarcnet-v2/data/meld/raw"
TRACKS_ROOT = "/home/devops/ept/cache/tracks/meld"
FEATURES_ROOT = "/home/devops/ept/cache/features/meld"
FEATURES_GRID_ROOT = "/home/devops/ept/cache/features_grid/meld"
SPLIT_DIRS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}


def video_path_for(clip):
    return os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS[clip["split"]],
                         f"{clip['clip_id']}.mp4")


def iter_meld_clips():
    for split in ["train", "dev", "test"]:
        for fp in sorted(glob.glob(os.path.join(TRACKS_ROOT, split, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            yield split, clip["clip_id"], video_path_for(clip), clip


def segment_of(frame_pos, s=S, t=T):
    return min(int(frame_pos // (t / s)), s - 1)


def select_top_e(clip, e_max):
    """Top-E clusters by mean_confidence x frame_coverage — same selection rule
    as OUC-CGE, applied to cluster (track_id) identities instead of ByteTrack
    tracks."""
    n_total = clip["n_frames_grabbed"]
    track_frames = {}
    for frame in clip["frames"]:
        for det in frame["detections"]:
            tid = det.get("track_id")
            if tid is None:
                continue
            track_frames.setdefault(tid, []).append(
                (frame["frame_pos"], det["bbox"], det["confidence"] or 0.0)
            )
    scored = []
    for tid, entries in track_frames.items():
        coverage = len(entries) / n_total if n_total else 0.0
        mean_conf = float(np.mean([c for _, _, c in entries]))
        scored.append((mean_conf * coverage, tid, entries))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:e_max]


def decode_needed_frames(video_path, clip):
    cap = cv2.VideoCapture(video_path)
    frames = {}
    for fr in clip["frames"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame_idx"])
        ok, frame = cap.read()
        if ok:
            frames[fr["frame_pos"]] = frame
    cap.release()
    return frames


def extract_entity_features(model, frames, clip, e_max):
    top_e = select_top_e(clip, e_max)
    scores = np.zeros(e_max, dtype=np.float32)
    crops, targets = [], []
    for e, (score, tid, entries) in enumerate(top_e):
        scores[e] = score
        for frame_pos, bbox, _ in entries:
            frame = frames.get(frame_pos)
            if frame is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            crops.append(preprocess_bgr_crop(crop))
            targets.append((e, segment_of(frame_pos)))

    embeddings = embed_batch(model, crops)
    feat_sum = np.zeros((e_max, S, D), dtype=np.float32)
    feat_cnt = np.zeros((e_max, S), dtype=np.int32)
    for (e, s), emb in zip(targets, embeddings):
        feat_sum[e, s] += emb
        feat_cnt[e, s] += 1
    mask = feat_cnt > 0
    feat = np.where(mask[..., None], feat_sum / np.maximum(feat_cnt, 1)[..., None], 0.0)
    return feat.astype(np.float16), mask, scores


def extract_grid_features(model, frames, clip):
    h_bin = w_bin = None
    crops, targets = [], []
    for fr in clip["frames"]:
        frame = frames.get(fr["frame_pos"])
        if frame is None:
            continue
        H, W = frame.shape[:2]
        if h_bin is None:
            h_bin, w_bin = H // GRID_ROWS, W // GRID_COLS
        s = segment_of(fr["frame_pos"])
        region = 0
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                y1, y2 = r * h_bin, (r + 1) * h_bin if r < GRID_ROWS - 1 else H
                x1, x2 = c * w_bin, (c + 1) * w_bin if c < GRID_COLS - 1 else W
                crop = frame[y1:y2, x1:x2]
                crops.append(preprocess_bgr_crop(crop))
                targets.append((region, s))
                region += 1

    embeddings = embed_batch(model, crops)
    feat_sum = np.zeros((GRID_ROWS * GRID_COLS, S, D), dtype=np.float32)
    feat_cnt = np.zeros((GRID_ROWS * GRID_COLS, S), dtype=np.int32)
    for (region, s), emb in zip(targets, embeddings):
        feat_sum[region, s] += emb
        feat_cnt[region, s] += 1
    feat = feat_sum / np.maximum(feat_cnt, 1)[..., None]
    return feat.astype(np.float16)


def process_clip(model, split, clip_id, video_path, clip, e_max=MELD_CACHE_E_MAX):
    ent_dir = os.path.join(FEATURES_ROOT, split)
    grid_dir = os.path.join(FEATURES_GRID_ROOT, split)
    os.makedirs(ent_dir, exist_ok=True)
    os.makedirs(grid_dir, exist_ok=True)

    ent_path = os.path.join(ent_dir, f"{clip_id}.npy")
    mask_path = os.path.join(ent_dir, f"{clip_id}_mask.npy")
    scores_path = os.path.join(ent_dir, f"{clip_id}_scores.npy")
    grid_path = os.path.join(grid_dir, f"{clip_id}.npy")

    frames = decode_needed_frames(video_path, clip)
    feat, mask, scores = extract_entity_features(model, frames, clip, e_max)
    np.save(ent_path, feat)
    np.save(mask_path, mask)
    np.save(scores_path, scores)

    grid_feat = extract_grid_features(model, frames, clip)
    np.save(grid_path, grid_feat)

    return {
        "dataset": "meld", "split": split, "clip_id": clip_id, "e_max": e_max,
        "entity_path": ent_path, "mask_path": mask_path,
        "scores_path": scores_path, "grid_path": grid_path,
        "entity_sha256": sha256_of_file(ent_path),
        "mask_sha256": sha256_of_file(mask_path),
        "scores_sha256": sha256_of_file(scores_path),
        "grid_sha256": sha256_of_file(grid_path),
        "n_entity_present": int(mask.sum()),
    }
