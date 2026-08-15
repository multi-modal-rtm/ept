# Phase 1 step 2 — Detection + tracking, full OUC-CGE (YOLO person-class)

## 0. Housekeeping

- `CLAUDE.md` exclusion list updated with the 3 bake-off-discovered corrupt clips.
- All three (`low/view61.mp4`, `low/view1579.mp4`, `high/view1325.mp4`) are **train-split
  only** — confirmed absent from `val.csv` and `test.csv`. **The published 771-clip test
  protocol is unaffected; no correction needed.**
- Two new hard-won facts added to `CLAUDE.md` (onnxruntime + this run's threading bugs).

## 1. Detection + tracking wall clock

**1918.9s (32.0 min) for 7700 clips, 20 workers — 0.249s/clip amortized.**

Getting here took three rounds of debugging genuine CPU oversubscription (not the
onnxruntime issue from step 1 — that library isn't used here):
1. `ultralytics` YOLO silently resets torch's intra-op thread count to `min(8, ncpu)`
   internally on the **first** `.predict()` call, overriding any prior
   `torch.set_num_threads(1)`. Fixed by re-forcing it before every predict call.
2. Under the default `fork` multiprocessing start method, worker processes inherit
   whatever native thread pools (BLAS/OpenCV) the **parent** process already
   initialized at module-import time. Setting `OMP_NUM_THREADS`-style env vars inside
   the pool initializer was too late; moved to the first lines of the file.
3. Net effect measured on an 80-clip/20-worker sample: 6.45s/clip → 6.00s/clip →
   **0.32s/clip** across the three fixes (~19x). Both are now `CLAUDE.md` hard-won facts.

## 2. Crop geometry (YOLO person bbox height, pixels)

| Scope | n boxes | mean | p5 | p25 | median | p75 | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 1,551,658 | 404.3 | 184.1 | 263.0 | 374.6 | 520.1 | 728.8 |
| low  | 597,789 | 383.4 | 166.1 | 253.4 | 349.0 | 494.1 | 689.1 |
| mid  | 425,700 | 418.8 | 206.9 | 274.3 | 405.2 | 529.5 | 717.7 |
| high | 528,169 | 416.4 | 187.4 | 265.9 | 380.9 | 533.3 | 776.1 |

**Median crop height (~375–405px) is comfortably above DINOv2's 224×224 input** — the
large majority of crops are being **downsampled**, not upsampled/interpolated from a
small source. Only the bottom ~5% of boxes (worst case: `low` class, p5=166px) would need
mild upsampling (~1.35×). This does not argue for cropping from higher-resolution source
frames; the geometry is healthy as designed.

## 3. Tracking QA report

**Gate (≥2 tracks at ≥50% coverage): 98.5% overall — passes clearly (≥80% threshold).**

| Scope | n clips | tracks/clip mean | tracks/clip median | frame-coverage/track mean | frame-coverage/track median | gate pass % |
|---|---:|---:|---:|---:|---:|---:|
| overall | 7700 | 8.23 | 8.0 | 0.767 | 1.0 | 98.5% |
| low  | 3024 | 8.17 | 8.0 | 0.758 | 0.969 | 100.0% |
| mid  | 2041 | 8.09 | 8.0 | 0.806 | 1.0 | 100.0% |
| high | 2635 | 8.41 | 8.0 | 0.748 | 0.969 | 95.7% |

**ID-switch / fragmentation proxy** (mean distinct track-IDs per clip — by construction
identical to "tracks/clip mean" above, since that *is* the fragmentation magnitude
being asked for): **8.23 overall**, ranging 8.09 (mid) to 8.41 (high). Given the bake-off
found ~7 people visible per frame in typical clips, a mean of ~8.2 distinct track IDs per
32-frame clip indicates **mild fragmentation** (roughly 1 extra ID beyond the visible
headcount on average) rather than severe ID churn — crowded desk rows do fragment
somewhat, magnitude is small.

12 contact sheets (4/class) saved to `outputs/phase1_tracking_qa/contact_sheets/` —
visually, tracked boxes are stable and consistent across the 8 sampled time points per
sheet (same color/ID persists through motion, gesture, occlusion).

## 4. Mask-only control (VAL split only, 3 seeds)

| Model | seed 42 | seed 1337 | seed 2024 | mean ± std |
|---|---:|---:|---:|---:|
| Majority-class baseline | 0.1899 | 0.1899 | 0.1899 | 0.1899 ± 0.0000 |
| Logistic regression | 0.3573 | 0.3811 | 0.3819 | **0.3734 ± 0.0114** |
| Small MLP (16 hidden) | 0.3346 | 0.2750 | 0.2758 | 0.2951 ± 0.0279 |

**This is NOT near chance.** Logistic regression on the presence mask alone (no visual
features) scores macro-F1 **0.373, essentially double the 0.190 majority baseline.** The
entity presence pattern — which entities are trackable in which of the 8 segments —
carries real, non-trivial information about the engagement label on its own.

**This is the finding flagged for discussion before proceeding.** It means detection/
tracking success or failure is itself correlated with engagement class (plausibly:
disengaged students are more often turned away, occluded, or absent from frame — exactly
the mechanism the bake-off's manual recall check surfaced for SCRFD, and evidently not
fully eliminated by switching to YOLO). Any later EPT accuracy gain over baselines will
need to be shown to survive **beyond** what this mask alone already predicts, or the paper's
central claim is vulnerable to a "it's just detection artifact" objection. Concretely this
argues for reporting the mask-only number as a floor alongside every future result table
(as the amendment to `DECISION_RULES.md` already commits to), and for treating A2
(identity-shuffle control) results with extra scrutiny since shuffling identity doesn't
change presence-mask structure.
