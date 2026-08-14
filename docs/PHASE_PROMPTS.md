# Staged Claude Code prompts

One prompt per phase. Each ends in a stop-and-show checkpoint. Do not chain.

---

## P0 — Environment and freeze (Aug 14–15)

```
Read CLAUDE.md and docs/PLAN.md. Do not write model code yet.

1. Report free disk on the partition holding /home/devops (df -h), and the size of
   ~/.cache/huggingface. We need headroom for a feature cache before Phase 2.
2. Verify the DINOv2 snapshot loads offline:
   AutoModel.from_pretrained('facebook/dinov2-with-registers-base'), print param count
   and output dim.
3. Install and smoke-test insightface (buffalo_l) and supervision. Report whether
   buffalo_l weights downloaded successfully.
4. Scaffold the repo layout from CLAUDE.md. Commit docs/DECISION_RULES.md as the
   FIRST commit that touches docs/. Print the commit hash.

STOP. Report disk numbers, the DINOv2 output dim, detector status, and the
DECISION_RULES commit hash.
```

**Gate:** decision rules committed; weights resolve offline; disk headroom >= 20 GB.

---

## P1 — Detection and tracking (Aug 16–18)

```
Build src/ept/tokenization/detect_track.py.

Input: a video path. Output: for each of T=32 uniformly sampled frames, a list of
(track_id, bbox, confidence).

- SCRFD face detection per sampled frame; supervision ByteTrack for ID association.
- Persist per-clip track records as JSON under cache/tracks/<split>/<clip_id>.json.
- Exclude the two known-bad OUC-CGE files listed in CLAUDE.md.

Then run it over ALL OUC-CGE clips and produce a tracking QA report:
  - distribution of tracks-per-clip
  - distribution of frame coverage per track
  - % of clips with >=2 tracks at >=50% coverage   <-- the gate number
  - 12 sampled clips rendered with boxes and IDs drawn, saved as PNG contact sheets

STOP. Show the QA report and the contact sheets. Do not extract features yet.
```

**Gate:** >=80% of clips with >=2 tracks at >=50% coverage. If below, invoke the
region-persistent fallback in docs/PLAN.md §8 and say so explicitly — do not proceed silently.

---

## P2 — Feature cache (Aug 19–20)

```
Build src/ept/tokenization/extract_features.py.

For every clip: crop each track per frame to 224x224, batch through frozen DINOv2
(fp16, no grad), representation = CLS concat mean-of-patch-tokens. Mean-pool within
each of S=8 segments. Write cache/features/<split>/<clip_id>.npy of shape
[E_max, S, D] plus a boolean presence mask [E_max, S].

Also cache the GRID baseline features (A0) from the SAME backbone: 64 spatial-temporal
grid tokens per clip, same dtype and layout.

Run over OUC-CGE and DAiSEE. Then:
  - checksum the cache, write cache/MANIFEST.json
  - spot-verify 5 clips by reloading and comparing against a fresh forward pass
  - rclone the cache to Google Drive

STOP. Report cache size, clip counts per split, and the spot-verify result.
```

**Gate:** manifest written, spot-verify exact, rclone backup confirmed.

---

## P3 — Model and recipe lock (Aug 21–23)

```
Implement src/ept/model/ept_former.py per docs/PLAN.md §3.2 and the constraints in
docs/DECISION_RULES.md (note: NO per-slot identity embedding; slot order randomized
per sample in every condition).

Implement all seven conditions A0–A6 as Hydra configs sharing one training loop.

Search the optimizer recipe on the VALIDATION split only (lr, weight decay, dropout,
epochs, batch size). Report the search grid and the chosen values. Write
docs/LOCKED_RECIPE.md and commit it.

STOP. Show the val curves and the locked recipe. Do NOT touch the test split.
```

**Gate:** `LOCKED_RECIPE.md` committed; zero test evaluations so far.

---

## P4 — Full matrix (Aug 24–26)

```
Run A0–A6 x seeds {42,1337,2024} on OUC-CGE with the locked recipe, plus the
E x S token-budget sweep. Each run gets its own outputs/<run_id>/ with a config
snapshot and metrics.json.

Then, ONCE, evaluate the locked configs on test and produce results/summary.csv with
mean +/- std per condition.

Apply the decision rule in docs/DECISION_RULES.md exactly as written and state which
branch fired.

STOP. Show summary.csv and the branch.
```

**Gate:** decision rule fired and recorded in `logs/GATES.md`.

---

## P5 — MELD extension (conditional, Aug 24–27)

```
ONLY if Phase 4 was on schedule as of Aug 24.

MELD is edited television: shot cuts break IoU tracking. Replace ByteTrack with
per-clip face-embedding clustering (agglomerative, cosine, tuned on dev only) to
recover identity across cuts.

Align clips to labels by (dialogue_id, utterance_id) from the label CSV — NOT by
directory listing; file counts exceed label counts. Exclude data/meld/bad_clips.txt.

Primary endpoint on MELD is 3-class sentiment. Report 7-class emotion as secondary.
Run A0/A1/A2 x 3 seeds only.

STOP. Show the MELD summary table and the clustering purity statistics.
```

---

## P6 — Efficiency and figures (Aug 27–28)

```
Measure, per docs/PLAN.md §6:
  - token count and GFLOPs of the attention stack for every condition
  - detection+tracking cost per clip, as a SEPARATE line item (do not fold it in)
  - latency: RTX 5090 batch-1 fp16, and CPU-only at threads in {1,8,60}
  - peak memory

Figures: (1) architecture, (2) tracking contact sheet, (3) accuracy-vs-GFLOPs Pareto,
(4) attention maps for A1 vs A2. Vector PDF, single-column width.

STOP. Show the efficiency table and all four figures.
```
