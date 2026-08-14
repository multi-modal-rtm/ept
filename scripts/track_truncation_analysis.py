"""Phase 2 addition: distinct-track-ID distribution and E_max=8 truncation analysis
on the OUC-CGE tracking cache. For clips exceeding the cap, classifies dropped
(rank>8) tracks as fragment-like vs whole-person-like using two signals: frame
coverage (fragments are brief) and simultaneous co-occurrence with >=2 already-
selected tracks (evidence of a genuinely separate, simultaneously-present person
that the cap truncates).
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

CACHE_ROOT = "/home/devops/ept/cache/tracks"
OUT_DIR = "/home/devops/ept/outputs/phase2_truncation"
E_MAX = 8
FRAGMENT_COVERAGE_THRESHOLD = 3 / 32  # <~10% of frames = classic ID-switch signature
CO_OCCURRENCE_MIN = 2


def load_ouccge_clips():
    files = sorted(glob.glob(os.path.join(CACHE_ROOT, "*", "*.json")))
    files = [f for f in files if not f.endswith("run_summary.json")]
    clips = []
    for fp in files:
        with open(fp) as f:
            clips.append(json.load(f))
    return clips


def track_stats(clip):
    """track_id -> {frame_positions: set, mean_conf: float, coverage: float}"""
    n_total = clip["n_frames_grabbed"]
    per_track = defaultdict(list)
    for frame in clip["frames"]:
        for det in frame["detections"]:
            per_track[det["track_id"]].append((frame["frame_pos"], det["confidence"] or 0.0))
    stats = {}
    for tid, entries in per_track.items():
        positions = {p for p, _ in entries}
        confs = [c for _, c in entries]
        coverage = len(positions) / n_total if n_total else 0.0
        stats[tid] = {
            "frame_positions": positions,
            "mean_conf": float(np.mean(confs)),
            "coverage": coverage,
            "score": float(np.mean(confs)) * coverage,
        }
    return stats


def classify_dropped(stats, retained_ids, dropped_ids):
    results = []
    for tid in dropped_ids:
        s = stats[tid]
        is_brief = s["coverage"] < FRAGMENT_COVERAGE_THRESHOLD
        # simultaneous co-occurrence: frames where >=2 retained tracks are ALSO present
        co_occurring_frames = 0
        for pos in s["frame_positions"]:
            n_retained_present = sum(
                1 for rid in retained_ids if pos in stats[rid]["frame_positions"]
            )
            if n_retained_present >= CO_OCCURRENCE_MIN:
                co_occurring_frames += 1
        co_occurs = co_occurring_frames >= max(1, len(s["frame_positions"]) // 2)
        if is_brief and not co_occurs:
            label = "fragment-like"
        elif not is_brief and co_occurs:
            label = "whole-person-like"
        else:
            label = "ambiguous"
        results.append({
            "track_id": tid, "coverage": s["coverage"], "mean_conf": s["mean_conf"],
            "score": s["score"], "co_occurring_frames": co_occurring_frames,
            "n_frames_present": len(s["frame_positions"]), "label": label,
        })
    return results


def max_simultaneous(clip):
    """Peak number of distinct track IDs present in any single sampled frame —
    the direct measure of real simultaneous crowding, as opposed to cumulative
    distinct-ID count over the clip (which conflates walk-throughs / re-entries /
    ID switches with genuinely-simultaneous people)."""
    peak = 0
    for frame in clip["frames"]:
        n = len(frame["detections"])
        if n > peak:
            peak = n
    return peak


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    clips = load_ouccge_clips()
    print(f"loaded {len(clips)} OUC-CGE clips")

    n_tracks = []
    max_simul = []
    per_clip_n = []
    for clip in clips:
        stats = track_stats(clip)
        n = len(stats)
        n_tracks.append(n)
        peak = max_simultaneous(clip)
        max_simul.append(peak)
        per_clip_n.append((clip["clip_id"], clip["class_name"], clip["split"], n, peak, stats))

    n_tracks = np.array(n_tracks)
    max_simul = np.array(max_simul)
    n_exceed = int((n_tracks > E_MAX).sum())
    pct_exceed = 100 * n_exceed / len(n_tracks)
    n_exceed_simul = int((max_simul > E_MAX).sum())
    pct_exceed_simul = 100 * n_exceed_simul / len(max_simul)

    print(f"distinct track IDs per clip (cumulative over 32 frames): mean={n_tracks.mean():.2f} "
          f"median={np.median(n_tracks):.1f} p90={np.percentile(n_tracks, 90):.1f} max={n_tracks.max()}")
    print(f"clips with CUMULATIVE distinct IDs > E_max={E_MAX}: {n_exceed} / {len(n_tracks)} ({pct_exceed:.1f}%)")
    print(f"peak SIMULTANEOUS tracks in one frame: mean={max_simul.mean():.2f} "
          f"median={np.median(max_simul):.1f} p90={np.percentile(max_simul, 90):.1f} max={max_simul.max()}")
    print(f"clips with peak SIMULTANEOUS tracks > E_max={E_MAX}: {n_exceed_simul} / {len(max_simul)} "
          f"({pct_exceed_simul:.1f}%)  <-- the real truncation-of-real-people number")

    hist = {int(k): int(v) for k, v in zip(*np.unique(n_tracks, return_counts=True))}
    simul_hist = {int(k): int(v) for k, v in zip(*np.unique(max_simul, return_counts=True))}

    # Top-20 clips by CUMULATIVE track count (the ones that most exceed the cap on
    # that measure) -- peak-simultaneous is reported alongside each to show whether
    # the excess is real simultaneous crowding or accumulated walk-throughs/fragments.
    per_clip_n.sort(key=lambda x: x[3], reverse=True)
    top20 = per_clip_n[:20]

    top20_report = []
    all_dropped_labels = []
    n_top20_simul_exceeds = 0
    for clip_id, cls, split, n, peak, stats in top20:
        ranked = sorted(stats.items(), key=lambda kv: kv[1]["score"], reverse=True)
        retained_ids = [tid for tid, _ in ranked[:E_MAX]]
        dropped_ids = [tid for tid, _ in ranked[E_MAX:]]
        classified = classify_dropped(stats, retained_ids, dropped_ids)
        for c in classified:
            all_dropped_labels.append(c["label"])
        if peak > E_MAX:
            n_top20_simul_exceeds += 1
        top20_report.append({
            "clip_id": clip_id, "class_name": cls, "split": split,
            "n_tracks_cumulative": n, "peak_simultaneous": peak,
            "n_dropped": len(dropped_ids), "dropped": classified,
        })
        labels_str = ", ".join(f"{c['track_id']}:{c['label']}(cov={c['coverage']:.2f})" for c in classified)
        print(f"  {clip_id} ({cls}): {n} cumulative, peak_simultaneous={peak} "
              f"({'EXCEEDS' if peak > E_MAX else 'within'} cap), {len(dropped_ids)} dropped -> {labels_str}")

    print(f"\nof the top-20 cumulative-exceeding clips, {n_top20_simul_exceeds}/20 also have "
          f"peak simultaneous presence > E_max (i.e. genuinely need more than 8 slots at once "
          f"at some single frame, not just accumulated over the clip)")

    from collections import Counter
    label_counts = Counter(all_dropped_labels)
    print(f"\nacross top-20 clips, dropped-track classification: {dict(label_counts)}")

    with open(os.path.join(OUT_DIR, "truncation_report.json"), "w") as f:
        json.dump({
            "n_clips": len(clips),
            "cumulative_track_count_histogram": hist,
            "cumulative_mean": float(n_tracks.mean()), "cumulative_median": float(np.median(n_tracks)),
            "cumulative_p90": float(np.percentile(n_tracks, 90)), "cumulative_max": int(n_tracks.max()),
            "n_exceeding_e_max_cumulative": n_exceed, "pct_exceeding_e_max_cumulative": pct_exceed,
            "peak_simultaneous_histogram": simul_hist,
            "peak_simultaneous_mean": float(max_simul.mean()), "peak_simultaneous_median": float(np.median(max_simul)),
            "peak_simultaneous_p90": float(np.percentile(max_simul, 90)), "peak_simultaneous_max": int(max_simul.max()),
            "n_exceeding_e_max_simultaneous": n_exceed_simul, "pct_exceeding_e_max_simultaneous": pct_exceed_simul,
            "top20_clips": top20_report,
            "top20_n_also_exceed_simultaneous": n_top20_simul_exceeds,
            "dropped_label_counts": dict(label_counts),
        }, f, indent=2)
    print(f"saved -> {OUT_DIR}/truncation_report.json")


if __name__ == "__main__":
    main()
