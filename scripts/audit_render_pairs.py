"""Phase 3.5 data audit — render the top-N highest-similarity cross-split pairs
as side-by-side frame thumbnails. AUDIT ONLY, no model fit, no eval.
"""
import json
import os

import cv2

DATA_ROOT = "/home/devops/data/OUC-CGE"
TRACKS_ROOT = "/home/devops/ept/cache/tracks"
OUT_DIR = "/home/devops/ept/outputs/phase3_5_audit/pairs"
N_PAIRS = 12


def find_split_for_clip(clip_id):
    for split in ["train", "val", "test"]:
        path = os.path.join(TRACKS_ROOT, split, f"{clip_id}.json")
        if os.path.exists(path):
            return split, path
    raise FileNotFoundError(clip_id)


def middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def clip_video_path(clip_id):
    split, track_path = find_split_for_clip(clip_id)
    with open(track_path) as f:
        clip = json.load(f)
    return os.path.join(DATA_ROOT, clip["rel_path"].replace("videos/", "")), split, clip


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open("/home/devops/ept/outputs/phase3_5_audit/near_duplicate_report.json") as f:
        d = json.load(f)
    pairs = sorted(d["all_pairs"], key=lambda p: -p["sim"])[:N_PAIRS]

    for i, p in enumerate(pairs):
        qid, mid, sim, ptype = p["query"], p["match"], p["sim"], p["pair_type"]
        q_path, q_split, q_clip = clip_video_path(qid)
        m_path, m_split, m_clip = clip_video_path(mid)
        q_frame = middle_frame(q_path)
        m_frame = middle_frame(m_path)
        if q_frame is None or m_frame is None:
            print(f"skip {qid}/{mid}: frame grab failed")
            continue

        h = 480
        def resize(f):
            scale = h / f.shape[0]
            return cv2.resize(f, (int(f.shape[1] * scale), h))
        q_r, m_r = resize(q_frame), resize(m_frame)
        combo = cv2.hconcat([q_r, m_r])
        label = (f"{ptype}  sim={sim:.6f}   "
                 f"QUERY={qid} ({q_split}, label={q_clip['label']})   |   "
                 f"MATCH={mid} ({m_split}, label={m_clip['label']})")
        banner = 40
        canvas = cv2.copyMakeBorder(combo, banner, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        out_path = os.path.join(OUT_DIR, f"pair_{i:02d}_{qid}_vs_{mid}.png")
        cv2.imwrite(out_path, canvas)
        print(f"saved {out_path}  labels: query={q_clip['label']} match={m_clip['label']}")


if __name__ == "__main__":
    main()
