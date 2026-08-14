"""Phase 1 step 2: bbox height distribution from the full detect_track cache,
overall and by class. Informs whether DINOv2's 224x224 input is upsampling
heavily from the source crop resolution.
"""
import glob
import json
import os

import numpy as np

CACHE_ROOT = "/home/devops/ept/cache/tracks"
OUT_DIR = "/home/devops/ept/outputs/phase1_crop_geometry"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, "*", "*.json")))
    files = [f for f in files if not f.endswith("run_summary.json")]
    print(f"found {len(files)} clip track files")

    heights_by_class = {"low": [], "mid": [], "high": []}
    heights_all = []

    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        cls = d["class_name"]
        for frame in d["frames"]:
            for det in frame["detections"]:
                x1, y1, x2, y2 = det["bbox"]
                h = y2 - y1
                heights_by_class[cls].append(h)
                heights_all.append(h)

    def summarize(arr, label):
        arr = np.array(arr)
        pcts = np.percentile(arr, [5, 25, 50, 75, 95])
        print(f"{label:6s} n={len(arr):8d} mean={arr.mean():7.1f} "
              f"p5={pcts[0]:6.1f} p25={pcts[1]:6.1f} median={pcts[2]:6.1f} "
              f"p75={pcts[3]:6.1f} p95={pcts[4]:6.1f}")
        return {
            "n": int(len(arr)), "mean": float(arr.mean()),
            "p5": float(pcts[0]), "p25": float(pcts[1]), "median": float(pcts[2]),
            "p75": float(pcts[3]), "p95": float(pcts[4]),
        }

    summary = {"overall": summarize(heights_all, "ALL")}
    for cls in ["low", "mid", "high"]:
        summary[cls] = summarize(heights_by_class[cls], cls)

    with open(os.path.join(OUT_DIR, "bbox_height_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {OUT_DIR}/bbox_height_summary.json")


if __name__ == "__main__":
    main()
