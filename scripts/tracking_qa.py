"""Phase 1 step 3: tracking QA report over the full detect_track cache.

- tracks-per-clip distribution, overall and by class
- frame-coverage-per-track distribution
- % clips with >=2 tracks at >=50% coverage (the Phase 1 gate), overall and by class
- ID-switch proxy: mean distinct-track-id count per clip (fragmentation magnitude)
- 12 contact sheets (4/class) with boxes + track IDs drawn, sampled across frame positions
"""
import glob
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np

CACHE_ROOT = "/home/devops/ept/cache/tracks"
DATA_ROOT = "/home/devops/data/OUC-CGE"
OUT_DIR = "/home/devops/ept/outputs/phase1_tracking_qa"
COVERAGE_GATE = 0.5
MIN_TRACKS_GATE = 2
CONTACT_SHEET_FRAME_POS = [0, 4, 8, 12, 16, 20, 24, 28]  # 8 of the 32 sampled frames


def load_clips():
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, "*", "*.json")))
    files = [f for f in files if not f.endswith("run_summary.json")]
    clips = []
    for fp in files:
        with open(fp) as f:
            clips.append(json.load(f))
    return clips


def per_track_coverage(clip):
    """track_id -> (n_frames_present, coverage_fraction)."""
    n_total = clip["n_frames_grabbed"]
    presence = defaultdict(int)
    for frame in clip["frames"]:
        for det in frame["detections"]:
            presence[det["track_id"]] += 1
    return {tid: (n, n / n_total if n_total else 0.0) for tid, n in presence.items()}


def gate_pass(clip):
    cov = per_track_coverage(clip)
    n_qualifying = sum(1 for _, c in cov.values() if c >= COVERAGE_GATE)
    return n_qualifying >= MIN_TRACKS_GATE, len(cov), cov


def draw_contact_sheet(clip, out_path):
    full_path = os.path.join(DATA_ROOT, clip["rel_path"].replace("videos/", ""))
    cap = cv2.VideoCapture(full_path)
    frame_by_pos = {fr["frame_pos"]: fr for fr in clip["frames"]}
    tiles = []
    for pos in CONTACT_SHEET_FRAME_POS:
        fr = frame_by_pos.get(pos)
        if fr is None:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr["frame_idx"])
        ok, frame = cap.read()
        if not ok:
            continue
        for det in fr["detections"]:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            tid = det["track_id"]
            color = (
                int(37 * tid % 255), int(97 * tid % 255), int(157 * tid % 255)
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"id{tid}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        small = cv2.resize(frame, (480, 270))
        cv2.putText(small, f"pos{pos}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        tiles.append(small)
    cap.release()
    if not tiles:
        return False
    while len(tiles) < 8:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, 8, 4)]
    sheet = np.vstack(rows)
    cv2.putText(sheet, f"{clip['clip_id']} ({clip['class_name']}, {clip['split']})",
                (5, sheet.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2)
    cv2.imwrite(out_path, sheet)
    return True


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = load_clips()
    print(f"loaded {len(clips)} clips")

    by_class = defaultdict(list)
    for c in clips:
        by_class[c["class_name"]].append(c)

    report = {}

    # tracks-per-clip and fragment proxy (= distinct track ids per clip)
    for scope_name, scope_clips in [("overall", clips)] + [(c, by_class[c]) for c in ["low", "mid", "high"]]:
        n_tracks_per_clip = []
        coverage_all = []
        gate_flags = []
        for clip in scope_clips:
            passed, n_tracks, cov = gate_pass(clip)
            n_tracks_per_clip.append(n_tracks)
            coverage_all.extend(c for _, c in cov.values())
            gate_flags.append(passed)
        n_tracks_per_clip = np.array(n_tracks_per_clip)
        coverage_all = np.array(coverage_all)
        report[scope_name] = {
            "n_clips": len(scope_clips),
            "tracks_per_clip_mean": float(n_tracks_per_clip.mean()),
            "tracks_per_clip_median": float(np.median(n_tracks_per_clip)),
            "tracks_per_clip_p90": float(np.percentile(n_tracks_per_clip, 90)),
            "frame_coverage_per_track_mean": float(coverage_all.mean()) if len(coverage_all) else None,
            "frame_coverage_per_track_median": float(np.median(coverage_all)) if len(coverage_all) else None,
            "gate_pass_pct": float(100 * np.mean(gate_flags)),
            "fragment_proxy_mean_distinct_tracks_per_clip": float(n_tracks_per_clip.mean()),
        }
        print(f"[{scope_name}] n={len(scope_clips)} "
              f"tracks/clip mean={report[scope_name]['tracks_per_clip_mean']:.2f} "
              f"gate_pass={report[scope_name]['gate_pass_pct']:.1f}%")

    with open(os.path.join(OUT_DIR, "qa_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {OUT_DIR}/qa_report.json")

    # contact sheets: 4 per class, sampled with a fixed seed
    rng = random.Random(42)
    sheets_dir = os.path.join(OUT_DIR, "contact_sheets")
    os.makedirs(sheets_dir, exist_ok=True)
    n_saved = 0
    for cls in ["low", "mid", "high"]:
        candidates = [c for c in by_class[cls] if len(c["frames"]) > 0]
        picks = rng.sample(candidates, min(4, len(candidates)))
        for clip in picks:
            out_path = os.path.join(sheets_dir, f"{clip['clip_id']}.png")
            if draw_contact_sheet(clip, out_path):
                n_saved += 1
                print(f"saved contact sheet -> {out_path}")
    print(f"saved {n_saved} contact sheets -> {sheets_dir}")


if __name__ == "__main__":
    main()
