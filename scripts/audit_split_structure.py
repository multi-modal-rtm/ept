"""Phase 3.5 data audit — split vs view-number structure. AUDIT ONLY."""
import csv
import json
import os
from collections import Counter

import numpy as np

DATA_ROOT = "/home/devops/data/OUC-CGE"
OUT_DIR = "/home/devops/ept/outputs/phase3_5_audit"


def load_rows():
    rows = []  # (class_name, view_num, split)
    for split, fname in [("train", "train.csv"), ("val", "val.csv"), ("test", "test.csv")]:
        path = os.path.join(DATA_ROOT, fname)
        with open(path) as f:
            reader = csv.reader(f, delimiter=" ")
            for row in reader:
                if len(row) != 2:
                    continue
                rel_path, label = row[0], int(label_str := row[1])
                parts = rel_path.split("/")  # videos/<class>/view<N>.mp4
                cls = parts[1]
                stem = os.path.splitext(parts[2])[0]  # viewN
                num = int(stem.replace("view", ""))
                rows.append((cls, num, split))
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_rows()
    print(f"total rows: {len(rows)}")

    by_class = {}
    for cls, num, split in rows:
        by_class.setdefault(cls, []).append((num, split))

    report = {}
    for cls, entries in by_class.items():
        entries.sort()
        nums = [n for n, s in entries]
        splits = [s for n, s in entries]

        # adjacency: how often does view N and view N+1 (consecutive integers,
        # not necessarily consecutive in the sorted list -- check literal N+1)
        num_to_split = {n: s for n, s in entries}
        adjacent_total = 0
        adjacent_diff_split = 0
        for n in nums:
            if (n + 1) in num_to_split:
                adjacent_total += 1
                if num_to_split[n] != num_to_split[n + 1]:
                    adjacent_diff_split += 1

        # run-length: how many contiguous same-split runs when scanning view
        # numbers in sorted order (fewer, longer runs = "blocked"; many short
        # runs = "interleaved")
        runs = 1
        for i in range(1, len(splits)):
            if splits[i] != splits[i - 1]:
                runs += 1

        report[cls] = {
            "n_clips": len(entries),
            "view_num_min": min(nums), "view_num_max": max(nums),
            "n_adjacent_pairs (N,N+1 both exist)": adjacent_total,
            "n_adjacent_pairs_different_split": adjacent_diff_split,
            "pct_adjacent_pairs_different_split": (
                100 * adjacent_diff_split / adjacent_total if adjacent_total else None
            ),
            "n_contiguous_same_split_runs": runs,
            "split_counts": dict(Counter(splits)),
        }
        print(f"\n=== class {cls} ===")
        print(f"  n_clips={len(entries)} view_num range=[{min(nums)},{max(nums)}]")
        print(f"  adjacent (N,N+1) pairs existing: {adjacent_total}, "
              f"of which different-split: {adjacent_diff_split} "
              f"({report[cls]['pct_adjacent_pairs_different_split']:.1f}%)")
        print(f"  contiguous same-split runs when scanning sorted view numbers: {runs} "
              f"(vs {len(entries)} clips total -> runs/clips={runs/len(entries):.3f}; "
              f"1.0 would mean every clip is its own run i.e. fully interleaved/random, "
              f"near-0 would mean a few big blocks)")

    # A small illustrative slice of the actual sequence for eyeballing
    print("\n=== sample: first 40 (view_num, split) pairs for class 'low' ===")
    low_sorted = sorted(by_class["low"])
    for n, s in low_sorted[:40]:
        print(f"  view{n}: {s}")

    with open(os.path.join(OUT_DIR, "split_structure_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nsaved -> {OUT_DIR}/split_structure_report.json")


if __name__ == "__main__":
    main()
