"""Phase 2: frozen DINOv2 feature cache, per docs/PLAN.md §3.1/§3.3 and
.claude/skills/ept-data.md.

For every clip: crop each of the top-E selected tracks per frame to 224x224,
batch through frozen DINOv2-with-registers-base (fp16, no grad). Representation =
CLS-token concatenated with mean-of-patch-tokens (excluding the 4 register tokens),
D=1536. Mean-pool within each of S=8 segments. Writes
cache/features/<dataset>/<split>/<clip_id>.npy [E, S, D] float16, a sibling
<clip_id>_mask.npy boolean presence mask [E, S], and <clip_id>_scores.npy float32
[E] holding the (mean_confidence x frame_coverage) selection score for each slot
(0.0 for padding slots — see extract_entity_features docstring) so the selection
ordering is reproducible and auditable, not just implicit in array position.

**E is 16 for OUC-CGE, 8 for DAiSEE.** OUC-CGE's cache was extended from the
original E_max=8 after finding 19% of clips have real simultaneous crowding beyond
8 people (outputs/phase2_truncation/REPORT.md) — this is the token-budget SWEEP
ceiling (docs/DECISION_RULES.md amendment). **The pre-registered PRIMARY condition
(A1) is unchanged at E=8**: train it by slicing the OUC-CGE cache to [:8], which is
asserted bitwise-identical to a from-scratch E_max=8 extraction in
tests/test_e_max_slice_equivalence.py. See CLAUDE.md — this is exactly the kind of
detail that silently becomes a wrong result if missed.

A0 grid baseline: the SAME backbone, SAME crop-then-CLS+mean-patch processing path,
applied to a fixed 2x4 (rows x cols) non-overlapping spatial tiling of each frame
instead of tracked entity boxes — 8 "regions" x S=8 segments = 64 tokens, identical
dtype/shape/pooling to the entity cache, so accuracy differences are attributable to
tokenization/attention topology and not to any difference in how the backbone was
used. This tiling choice is a Phase 2 implementation decision (PLAN.md doesn't pin
the exact grid geometry) — see the writeup for other options and why this one.
Written to cache/features_grid/<dataset>/<split>/<clip_id>.npy [8, S, D] (mask is
always all-True — grid regions always "exist" — so no mask file for grid).
"""
import glob
import hashlib
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2
import numpy as np
import torch
from transformers import AutoModel

# LOCKED_PRIMARY_E_MAX=8 is the pre-registered primary condition (A1) and DAiSEE's
# E_max (DAiSEE is E=1 by construction; PLAN.md's 8-slot budget was never the
# constraint there). OUCCGE_E_MAX=16 is the extended token-budget sweep ceiling
# (docs/DECISION_RULES.md amendment, added after the truncation finding that 19%
# of OUC-CGE clips exceed 8 simultaneous tracked people). Anyone training the
# primary A1 condition slices the OUC-CGE cache to [:8] — see CLAUDE.md.
LOCKED_PRIMARY_E_MAX = 8
OUCCGE_E_MAX = 16
S = 8
T = 32
D = 1536  # 768 (CLS) + 768 (mean patch)
GRID_ROWS, GRID_COLS = 2, 4  # 8 spatial regions
CROP_SIZE = 224
MODEL_NAME = "facebook/dinov2-with-registers-base"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model():
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in model.parameters())
    return model


def preprocess_bgr_crop(bgr_crop):
    """cv2 BGR uint8 crop (any size) -> normalized CHW float32, resized to 224x224."""
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.transpose(2, 0, 1)  # CHW


@torch.no_grad()
def embed_batch(model, crops_chw, batch_size=256):
    """List of CHW float32 arrays -> [n, D] float16 numpy (CLS ++ mean-patch)."""
    if len(crops_chw) == 0:
        return np.zeros((0, D), dtype=np.float16)
    out = []
    n_reg = model.config.num_register_tokens
    for i in range(0, len(crops_chw), batch_size):
        chunk = crops_chw[i:i + batch_size]
        x = torch.from_numpy(np.stack(chunk)).to("cuda", dtype=torch.float16)
        hidden = model(x, interpolate_pos_encoding=True).last_hidden_state
        cls = hidden[:, 0]
        patches = hidden[:, 1 + n_reg:]
        rep = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        out.append(rep.to(torch.float16).cpu().numpy())
    return np.concatenate(out, axis=0)


def select_top_e(clip, e_max):
    """Top-E tracks by mean_confidence x frame_coverage — the locked selection
    rule (docs/DECISION_RULES.md), identical code path for every condition and
    independent of e_max itself (only the slice point differs), which is exactly
    what the slice-equivalence test in tests/test_e_max_slice_equivalence.py
    checks empirically rather than just assuming from this comment."""
    n_total = clip["n_frames_grabbed"]
    track_frames = {}
    for frame in clip["frames"]:
        for det in frame["detections"]:
            track_frames.setdefault(det["track_id"], []).append(
                (frame["frame_pos"], det["bbox"], det["confidence"] or 0.0)
            )
    scored = []
    for tid, entries in track_frames.items():
        coverage = len(entries) / n_total if n_total else 0.0
        mean_conf = float(np.mean([c for _, _, c in entries]))
        scored.append((mean_conf * coverage, tid, entries))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:e_max]


def segment_of(frame_pos, s=S, t=T):
    return min(int(frame_pos // (t / s)), s - 1)


def decode_needed_frames(video_path, clip):
    """Decode each of the clip's sampled frame_pos values EXACTLY ONCE and return
    {frame_pos: frame}. Both entity crops (multiple per frame, one per present
    track) and grid crops (8 per frame, always) are then derived from this single
    decode pass — critical for throughput: seeking+decoding per (entity, frame)
    pair independently (the first implementation) meant up to 8x redundant seeks
    for the same frame, measured at ~5s/clip; sharing one decode brought this to
    ~0.5s/clip amortized (~10x)."""
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
    """[e_max, S, D] float16 + [e_max, S] bool mask + [e_max] float32 selection
    score (0.0 for padding slots where fewer than e_max tracks exist — always
    check mask.any(axis=1) for validity, not the score array, exactly the same
    "never silently zero-fill without an explicit validity signal" rule as the
    presence mask itself). `frames`: {frame_pos: frame}, pre-decoded once per clip."""
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
    """[8, S, D] float16, matched-budget grid baseline. Always-present -> no mask
    needed (all True). `frames`: {frame_pos: frame}, pre-decoded once per clip."""
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


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


OUCCGE_DATA_ROOT = "/home/devops/data/OUC-CGE"
DAISEE_DATA_ROOT = "/home/devops/data/DAiSEE"
TRACKS_ROOT = "/home/devops/ept/cache/tracks"
FEATURES_ROOT = "/home/devops/ept/cache/features"
FEATURES_GRID_ROOT = "/home/devops/ept/cache/features_grid"


def iter_ouccge_clips(e_max=OUCCGE_E_MAX):
    for split in ["train", "val", "test"]:
        for fp in sorted(glob.glob(os.path.join(TRACKS_ROOT, split, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            video_path = os.path.join(OUCCGE_DATA_ROOT, clip["rel_path"].replace("videos/", ""))
            yield "ouccge", split, clip["clip_id"], video_path, clip, e_max


def iter_daisee_clips(e_max=LOCKED_PRIMARY_E_MAX):
    for split in ["train", "val", "test"]:
        for fp in sorted(glob.glob(os.path.join(TRACKS_ROOT, "daisee", split, "*.json"))):
            with open(fp) as f:
                clip = json.load(f)
            video_path = os.path.join(DAISEE_DATA_ROOT, clip["rel_path"])
            yield "daisee", split, clip["clip_id"], video_path, clip, e_max


def process_clip(model, dataset, split, clip_id, video_path, clip, e_max):
    ent_dir = os.path.join(FEATURES_ROOT, dataset, split)
    grid_dir = os.path.join(FEATURES_GRID_ROOT, dataset, split)
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
        "dataset": dataset, "split": split, "clip_id": clip_id, "e_max": e_max,
        "entity_path": ent_path, "mask_path": mask_path,
        "scores_path": scores_path, "grid_path": grid_path,
        "entity_sha256": sha256_of_file(ent_path),
        "mask_sha256": sha256_of_file(mask_path),
        "scores_sha256": sha256_of_file(scores_path),
        "grid_sha256": sha256_of_file(grid_path),
        "n_entity_present": int(mask.sum()),
    }


_MODEL = None


def _init_worker():
    global _MODEL
    # Same threading fixes as detect_track.py: cap OpenCV/torch intra-op threads
    # so N worker processes don't oversubscribe the CPU side of the pipeline
    # (each worker also does GPU forward passes, but decode/crop is what's
    # parallelized here — see CLAUDE.md hard-won facts for why this matters).
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _MODEL = load_model()


def _process_clip_mp(args):
    dataset, split, clip_id, video_path, clip, e_max = args
    return process_clip(_MODEL, dataset, split, clip_id, video_path, clip, e_max)


def run_extraction(all_clips, n_workers=8, log_path=None):
    from multiprocessing import Pool

    print(f"total clips to process: {len(all_clips)}")
    manifest_entries = []
    t0 = time.time()
    n_done = 0
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        for entry in pool.imap_unordered(_process_clip_mp, all_clips, chunksize=4):
            manifest_entries.append(entry)
            n_done += 1
            if n_done % 200 == 0:
                elapsed = time.time() - t0
                msg = f"  {n_done} clips done, {elapsed:.1f}s elapsed, {elapsed/n_done:.3f}s/clip amortized"
                print(msg)

    wall_clock = time.time() - t0
    print(f"extract_features wall clock: {wall_clock:.1f}s for {n_done} clips "
          f"({wall_clock/n_done:.3f}s/clip amortized, {n_workers} workers)")
    if log_path:
        with open(log_path, "w") as f:
            json.dump({"n_clips": n_done, "wall_clock_seconds": wall_clock,
                        "n_workers": n_workers, "entries": manifest_entries}, f)
    return manifest_entries, wall_clock


def build_manifest():
    """Scan whatever's currently in cache/features(_grid)/ and (re)write
    cache/MANIFEST.json with checksums — decoupled from any single extraction
    run so OUC-CGE and DAiSEE can be extracted separately (e.g. while DAiSEE
    tracking is still finishing) and merged here once both are complete."""
    entries = []
    for dataset in ["ouccge", "daisee"]:
        dataset_e_max = OUCCGE_E_MAX if dataset == "ouccge" else LOCKED_PRIMARY_E_MAX
        for split in ["train", "val", "test"]:
            ent_dir = os.path.join(FEATURES_ROOT, dataset, split)
            grid_dir = os.path.join(FEATURES_GRID_ROOT, dataset, split)
            if not os.path.isdir(ent_dir):
                continue
            for ent_path in sorted(glob.glob(os.path.join(ent_dir, "*.npy"))):
                if ent_path.endswith("_mask.npy") or ent_path.endswith("_scores.npy"):
                    continue
                clip_id = os.path.splitext(os.path.basename(ent_path))[0]
                mask_path = os.path.join(ent_dir, f"{clip_id}_mask.npy")
                scores_path = os.path.join(ent_dir, f"{clip_id}_scores.npy")
                grid_path = os.path.join(grid_dir, f"{clip_id}.npy")
                if not (os.path.exists(mask_path) and os.path.exists(grid_path)
                        and os.path.exists(scores_path)):
                    continue
                mask = np.load(mask_path)
                entries.append({
                    "dataset": dataset, "split": split, "clip_id": clip_id, "e_max": dataset_e_max,
                    "entity_path": ent_path, "mask_path": mask_path,
                    "scores_path": scores_path, "grid_path": grid_path,
                    "entity_sha256": sha256_of_file(ent_path),
                    "mask_sha256": sha256_of_file(mask_path),
                    "scores_sha256": sha256_of_file(scores_path),
                    "grid_sha256": sha256_of_file(grid_path),
                    "n_entity_present": int(mask.sum()),
                })
    manifest_path = "/home/devops/ept/cache/MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "n_clips": len(entries),
            "locked_primary_e_max": LOCKED_PRIMARY_E_MAX, "ouccge_e_max": OUCCGE_E_MAX,
            "s": S, "t": T, "d": D,
            "grid_rows": GRID_ROWS, "grid_cols": GRID_COLS,
            "model": MODEL_NAME,
            "entries": entries,
        }, f)
    print(f"MANIFEST.json: {len(entries)} clips -> {manifest_path}")
    return entries


if __name__ == "__main__":
    ouccge_clips = list(iter_ouccge_clips())  # e_max=OUCCGE_E_MAX=16
    run_extraction(ouccge_clips, n_workers=8,
                    log_path="/home/devops/ept/outputs/phase2_extract_features/ouccge_run.json")

    daisee_clips = list(iter_daisee_clips())  # e_max=LOCKED_PRIMARY_E_MAX=8
    run_extraction(daisee_clips, n_workers=8,
                    log_path="/home/devops/ept/outputs/phase2_extract_features/daisee_run.json")

    build_manifest()
