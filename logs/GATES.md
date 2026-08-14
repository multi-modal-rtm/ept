# EPT — Phase Gate Log

## Phase 0 — Bootstrap

- **Date:** 2026-08-14T10:47:15Z
- **Gate:** Decision rules committed before any run touches code or data.
- **Evidence:** `docs/DECISION_RULES.md` frozen at commit `b65063f03f978dd87bd55f1864ba6fd53ff6a885`
  ("Freeze pre-registration: hypotheses and decision rule (EPT / ICFNDS 2026)"), committed alone,
  before `src/`, `configs/`, or any other repo content existed.
- **Result:** PASS — pre-registration frozen before any run.

## Phase 1 — Detection + tracking, tracking QA

- **Date:** 2026-08-14T13:10:16Z
- **Gate:** ≥80% of clips yield ≥2 tracks with ≥50% frame coverage (`docs/PLAN.md` §7;
  else → §8 fallback to region-persistent tokens).
- **Evidence:** Detector switched to YOLO person-class (from SCRFD/buffalo_l) after the
  step-1 bake-off found SCRFD's failure mode pose-correlated and label-plausible —
  see `outputs/phase1_detector_bakeoff/REPORT.md`. Full detection+tracking run over all
  7700 non-excluded OUC-CGE clips (YOLO person + supervision ByteTrack), 1918.9s wall
  clock, 20 workers. Tracking QA: **98.5% of clips pass the gate overall** (low 100.0%,
  mid 100.0%, high 95.7%) — full numbers in `outputs/phase1_step2_summary/REPORT.md`.
- **Result:** PASS — clearly above the 80% threshold in aggregate and per class.
- **Flagged alongside, not a gate criterion but material to interpreting any later
  result:** the pre-registered mask-only control (`docs/DECISION_RULES.md` amendment,
  2026-08-14) came back well above the majority-class baseline (macro-F1 0.373 vs.
  0.190, logistic regression, VAL split, 3 seeds) — entity presence pattern alone
  carries real signal about the engagement label. Reported for discussion, not
  blocking the gate.
- **Correction (2026-08-14, same day):** majority-class (0.190) was the wrong baseline
  for this comparison — see `outputs/phase2_mask_diagnostics/REPORT.md`. Against the
  proper permutation null (0.322 ± 0.022, computed the same 3-seed way as 0.373), the
  result is **not statistically significant** (empirical p≈0.095–0.11). Diagnosis:
  the 8-dim per-segment presence-count vector alone recovers 0.325 of the 0.373 —
  mechanism is aggregate group-presence dynamics, not per-entity identity. No
  room/session identifier exists in OUC-CGE to check for a session confound. Mask-only
  stays a permanent row in every results table per the amendment; the number to carry
  forward is 0.373 alongside the null (0.322±0.022, p≈0.10), not 0.373 vs. 0.190.

## Phase 2 — Frozen DINOv2 feature cache

- **Date:** 2026-08-14T15:40:20Z
- **Gate:** manifest written, spot-verify exact, rclone backup confirmed (`docs/PLAN.md` §7).
- **Evidence:**
  - Mid-phase revision: OUC-CGE cache extended to E_max=16 (token-budget sweep
    `E ∈ {1,2,4,8,16}`, `docs/DECISION_RULES.md` amendment) after the truncation
    finding that 19% of clips exceed 8 simultaneous tracked people, dropped tracks
    predominantly whole persons (159/180). **Primary condition (A1) stays locked at
    E=8** — `cache[:8]` asserted bitwise-identical to a from-scratch E_max=8
    extraction on 10 real clips (`tests/test_e_max_slice_equivalence.py`, passing).
    DAiSEE unaffected (E=8 throughout; E=1 by construction).
  - Full extraction: OUC-CGE 7700/7700 clips (2128.0s, 8 workers), DAiSEE 8571/8571
    clips (896.6s, 8 workers). 16271 total clips, matching manifest exactly.
  - `cache/MANIFEST.json`: 16271 entries, sha256 checksummed (entity/mask/scores/grid
    per clip). Cache footprint: 8.0G local (`features` 4.6G, `features_grid` 3.1G,
    `tracks` 0.3G).
  - Spot-verify: 5 random clips (mixed dataset/split/e_max) reloaded from disk and
    compared against an independent fresh forward pass — exact match on all 4
    arrays (entity features, mask, scores, grid) for all 5 clips.
  - rclone backup: raw per-file copy was abandoned after measuring an 8h50m ETA
    (Drive per-file API overhead dominates for ~16k small files); switched to
    tarring each cache subdirectory first. 4 files, 4.107 GiB, confirmed present on
    `gdrive:ICFNDS 2026 - EPT/cache_backup/` with byte-identical sizes to local.
- **Result:** PASS.
- **Also flagged (Track A, requested addition):** E_max=8 truncation is real —
  peak-simultaneous-tracks exceeds 8 in 19.0% of OUC-CGE clips (not just cumulative
  ID count); among the 20 most-affected clips, dropped tracks are overwhelmingly
  high-coverage ("whole-person-like," 159/180 classified, 0 fragment-like). Belongs
  in Limitations regardless of the E_max=16 sweep addition — see
  `outputs/phase2_truncation/truncation_report.json`.
