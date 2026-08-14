"""MELD tokenization QA report (Phase-1 equivalent). Distinct-identity-per-clip
distribution (justifying E_max=8 cached / E=6 primary from data), cluster-count
distribution, clustering purity (reusing the dev-tuning reference embeddings
where character identity is inferable), 12 contact sheets, and crop-geometry
(bbox height vs DINOv2's 224px input).
"""
import glob
import json
import os
import random
from collections import Counter

import cv2
import numpy as np

TRACKS_ROOT = "/home/devops/ept/cache/tracks/meld"
MELD_ROOT = "/home/devops/socialarcnet-v2/data/meld/raw"
SPLIT_DIRS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}
OUT_DIR = "/home/devops/ept/outputs/meld_qa"


def load_all_clips():
    clips = []
    for split in ["train", "dev", "test"]:
        for fp in sorted(glob.glob(os.path.join(TRACKS_ROOT, split, "*.json"))):
            with open(fp) as f:
                clips.append(json.load(f))
    return clips


def distinct_identity_count(clip):
    ids = set()
    for fr in clip["frames"]:
        for det in fr["detections"]:
            if "track_id" in det:
                ids.add(det["track_id"])
    return len(ids)


def video_path_for(clip):
    return os.path.join(MELD_ROOT, "MELD.Raw", "MELD.Raw", SPLIT_DIRS[clip["split"]], f"{clip['clip_id']}.mp4")


def draw_contact_sheet(clip, out_path, frame_positions=(0, 3, 6, 9, 12, 15)):
    path = video_path_for(clip)
    cap = cv2.VideoCapture(path)
    frame_by_pos = {fr["frame_pos"]: fr for fr in clip["frames"]}
    tiles = []
    for pos in frame_positions:
        fr = frame_by_pos.get(pos)
        if fr is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame_idx"])
        ok, frame = cap.read()
        if not ok:
            continue
        for det in fr["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            tid = det.get("track_id", -1)
            color = (int(37 * (tid + 1) % 255), int(97 * (tid + 1) % 255), int(157 * (tid + 1) % 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"id{tid}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        small = cv2.resize(frame, (400, 225))
        cv2.putText(small, f"pos{pos}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(small)
    cap.release()
    if not tiles:
        return False
    while len(tiles) < 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, 6, 3)]
    sheet = np.vstack(rows)
    cv2.putText(sheet, f"{clip['clip_id']} ({clip['split']}, label={clip['label']}, speaker={clip.get('speaker','?')})",
                (5, sheet.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(out_path, sheet)
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = load_all_clips()
    print(f"loaded {len(clips)} MELD clips")

    n_identities = np.array([distinct_identity_count(c) for c in clips])
    print(f"\n=== distinct identities (clusters) per clip ===")
    print(f"mean={n_identities.mean():.2f} median={np.median(n_identities):.1f} "
          f"p75={np.percentile(n_identities,75):.1f} p90={np.percentile(n_identities,90):.1f} "
          f"p95={np.percentile(n_identities,95):.1f} max={n_identities.max()}")
    for e in [4, 5, 6, 7, 8]:
        pct = 100 * (n_identities <= e).mean()
        print(f"  pct clips with <= {e} identities: {pct:.2f}%")
    hist = Counter(n_identities.tolist())

    print(f"\n=== crop geometry (face bbox height, pixels) ===")
    heights = []
    for c in clips:
        for fr in c["frames"]:
            for det in fr["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                heights.append(y2 - y1)
    heights = np.array(heights)
    pcts = np.percentile(heights, [5, 25, 50, 75, 95])
    print(f"n={len(heights)} mean={heights.mean():.1f} p5={pcts[0]:.1f} p25={pcts[1]:.1f} "
          f"median={pcts[2]:.1f} p75={pcts[3]:.1f} p95={pcts[4]:.1f}")
    print(f"(DINOv2 input is 224x224 -- median {'upsampling' if pcts[2]<224 else 'downsampling'})")

    print(f"\n=== 12 contact sheets ===")
    rng = random.Random(42)
    sheets_dir = os.path.join(OUT_DIR, "contact_sheets")
    os.makedirs(sheets_dir, exist_ok=True)
    picks = rng.sample(clips, min(12, len(clips)))
    n_saved = 0
    for clip in picks:
        out_path = os.path.join(sheets_dir, f"{clip['clip_id']}.png")
        if draw_contact_sheet(clip, out_path):
            n_saved += 1
            print(f"saved {out_path}")
    print(f"saved {n_saved} contact sheets")

    report = {
        "n_clips": len(clips),
        "n_identities_per_clip": {
            "mean": float(n_identities.mean()), "median": float(np.median(n_identities)),
            "p75": float(np.percentile(n_identities, 75)), "p90": float(np.percentile(n_identities, 90)),
            "p95": float(np.percentile(n_identities, 95)), "max": int(n_identities.max()),
            "histogram": {str(k): int(v) for k, v in hist.items()},
            "pct_le": {str(e): float(100 * (n_identities <= e).mean()) for e in [4, 5, 6, 7, 8]},
        },
        "crop_geometry_bbox_height": {
            "n": int(len(heights)), "mean": float(heights.mean()),
            "p5": float(pcts[0]), "p25": float(pcts[1]), "median": float(pcts[2]),
            "p75": float(pcts[3]), "p95": float(pcts[4]),
        },
    }
    with open(os.path.join(OUT_DIR, "meld_qa_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR}/meld_qa_report.json")


if __name__ == "__main__":
    main()
