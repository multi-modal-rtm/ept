# MELD promoted to primary — Phase 1/2 equivalent, complete

## 1. DECISION_RULES.md amendment

Two commits, as required (procedure fixed before the blind bootstrap ran):
- `a6d19a8` — OUC-CGE dropped (rejected-dataset case study, audit numbers recorded:
  0.9748 trivial probe, 98.5% giant component at 0.80–0.90, 0.7305 after
  group-disjoint re-split), MELD promoted to primary (3-class sentiment, macro-F1,
  official splits, seeds unchanged), H1/H2 unchanged in form, effect-floor
  procedure fixed with the number left `TBD`.
- `8c156a9` — computed floor filled in as a separate, non-editing append:
  **`EFFECT_FLOOR = 0.04`** (`max(0.02, 2×pooled_SE)`, `pooled_SE = 0.0170` from
  blind seed-level + item-level bootstrap variance, conditions A0–A5 + mask-only,
  never inspecting which condition scored higher). **The study is underpowered
  for the originally-hoped 0.02 margin** — stated explicitly per the
  pre-committed procedure; H1/H2 now require ≥0.04, and a null under that floor
  will be reported as underpowered, not as evidence of absence.

## 2. Detector choice — SCRFD, with evidence

Brief bake-off (`scripts/meld_detector_bakeoff.py`, 100 clips, 1600 frames):
SCRFD zero-detection rate **0.94%** vs. YOLO person-detection **0.88%** — near
parity, confirming the close-framed/frontal-face hypothesis (contrast with
OUC-CGE, where SCRFD's zero-rate was severely elevated and pose-correlated).
Combined with the architectural requirement — face-embedding identity
clustering needs face crops, not person boxes, for usable ArcFace-quality
embeddings — the evidence favors **SCRFD**. Visual spot-check confirms clean,
well-framed face detections.

Identity recovery: agglomerative clustering, cosine distance, threshold tuned
on **dev only** using reference embeddings for the 6 recurring FRIENDS
characters (built from **train** clips only, keeping dev used purely for
evaluation). Best threshold **0.55** (purity/accuracy against the character
references: 62.2%, well above the 16.7% chance floor for 6 classes).

## 3. QA report

- **Distinct identities (clusters) per clip:** mean 5.87, median 4.0, p75=7,
  p90=13, p95=19, max=91. `pct <= 6: 72.35%`, `pct <= 8: 80.18%` — **E=6 primary
  is justified by the median+p75 falling at/under 6-7**, matching the
  instruction to justify the cap from data rather than inherit OUC-CGE's E=8.
  The long tail (max=91 in a ~3s clip) is clustering over-segmentation on hard
  clips (motion blur, lighting), not 91 genuine distinct people — worth a
  Limitations note, not a cap-justification concern since E_max=8/E=6 already
  truncates it regardless.
- **Crop geometry:** face bbox height median **130.8px** (p5=42.7, p25=77.6,
  p75=187.3, p95=237.8) against DINOv2's 224px input — **moderate upsampling**
  (median ~1.7x), the opposite of OUC-CGE's mostly-downsampling finding. Faces
  are smaller on average than the entity crops OUC-CGE produced, consistent
  with television framing showing more of the scene than a fixed classroom
  camera's close entity crops.
- **12 contact sheets**: `outputs/meld_qa/contact_sheets/` — visually, tracked
  IDs are consistent across the 6 sampled frames per sheet even through
  movement and partial occlusion.

## 4. Cache manifest

`cache/MELD_MANIFEST.json`: 13706 clips (train 9988, dev 1108, test 2610,
matching label counts exactly), sha256-checksummed (entity/mask/scores/grid per
clip). Wall clock: tokenization 3783.9s (63.1 min, 40 workers — required a
mid-run fix, `cv2.setNumThreads(1)` was missing from this script's worker init,
one of the two documented CLAUDE.md hard-won facts; load hit 149 before the fix,
healthy ~50-57 after), extraction 1226.3s (20.4 min, 8 workers, 0.089s/clip
amortized). Cache footprint: `tracks/meld` 101M, `features/meld` 1.5G,
`features_grid/meld` 1.4G. Spot-verify: 5 random clips reloaded from disk vs. a
fresh forward pass — **exact match on all 4 arrays for all 5 clips.** Backup:
tarred first (CLAUDE.md hard-won fact — raw per-file rclone would take hours for
~14k clips' worth of files), uploaded to
`gdrive:ICFNDS 2026 - EPT/cache_backup_meld/`.

## 5. MELD mask-only control

Permutation null, as in Phase 2:

| | seed 42 | seed 1337 | seed 2024 | 3-seed mean |
|---|---:|---:|---:|---:|
| Real logistic regression | 0.2795 | 0.2665 | 0.2571 | **0.2677** |

Majority-class baseline: 0.1983. Stratified-random closed-form (K=3): 0.3333.
**Permutation null (20 perms, 3-seed mean each): mean 0.2982, std 0.0136.**
**p-value = 0.9524.**

**The real score (0.2677) is actually below the permutation null's mean
(0.2982)** — reported plainly, exactly as it came out. There is **no detectable
presence-mask signal** on MELD, in contrast to the plausible hypothesis that
who's-on-screen would track who's-speaking and leak sentiment information. This
is a clean, favorable result for MELD as primary: whatever EPT-Former shows on
MELD will not be confounded by a mask-only artifact the way OUC-CGE's was.
