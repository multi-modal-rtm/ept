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

## Phase 3 — Model implementation and recipe lock

- **Date:** 2026-08-14T18:58:31Z
- **Gate:** `docs/LOCKED_RECIPE.md` committed; **zero test evaluations** (`docs/PLAN.md` §7).
- **Evidence:**
  - A6 redefined as a non-feature-matched reference row (`docs/PLAN.md` §5, commit
    `c307e8f`) with verified BPAVTforSGER numbers: TimeSformer fine-tuned 0.9917,
    TimeSformer video-only linear-probe 0.2648, VideoMAE fine-tuned 0.9875, VideoMAE
    linear-probe **not available** (reported as a gap, not fabricated).
  - `src/ept/model/ept_former.py`: EPTFormer (A0/A1/A2/A3/A4 share one class via
    `use_temporal`/`use_social` flags), `MeanPoolMLP` (A5), `MaskOnlyMLP`
    (mask-only). All four non-negotiables checked: no per-slot identity embedding
    (asserted by name at construction **and** proven behaviorally —
    `tests/test_ept_former.py`, entity-permutation invariance holds exactly for
    every attention configuration); presence mask respected via
    `key_padding_mask` + `[ABSENT]` embedding; backbone never referenced (this
    module consumes cached features only); A2 shuffle reuses
    `src/ept/tokenization/mask_ops.py`, not reimplemented.
  - Grid frozen **before any run**: `docs/SEARCH_GRID.md`, commit `507b8c7` — 8
    hand-picked (lr, weight_decay, dropout, epochs) points, `batch_size=32` fixed,
    identical grid for all 7 conditions (A0–A5 + mask-only), single seed (42) for
    search.
  - Full search: 56 runs, **2100 val evaluations** (exact — sum of epochs across
    all runs), 47.9 minutes wall clock. Every run has its own
    `outputs/<condition>_<recipe>_seed42/{config.yaml,metrics.json}`.
  - `docs/LOCKED_RECIPE.md` committed (`ddfc32f`) with the per-condition selected
    recipe and val macro-F1. Hyperparameters are now read-only.
  - **Zero test evaluations**: structurally enforced, not just observed —
    `OUCCGEDataset.__init__` asserts `split in ("train", "val")`; grepped the
    entire training/search code path for any `"test"` split reference — none
    found; no `metrics.json` from this phase has `split_evaluated: "test"`.
- **Result:** PASS.
- **Flagged for discussion, not blocking the gate:** all of A0–A5 land in a
  narrow, very high val macro-F1 band (0.971–0.992) — echoes A6's fine-tuned
  reference numbers, suggesting near-saturation from visual content alone at
  this val-split size, for every tokenization scheme including A0 (no entity
  structure) and A2 (identity-shuffled). This could compress H1/H2's required
  ≥0.02 margins once locked configs run on test in Phase 4 — watch, don't
  pre-empt. Also: mask-only's locked-recipe number (0.430, tuned MLP) is
  higher than Phase 2's reported number (0.373, untuned logistic regression) —
  different estimator capacity, not a different phenomenon; both carry forward
  with provenance stated, not conflated.

## Phase 3.5 data audit (NOT an evaluation)

- **Date:** 2026-08-14T19:11:31Z
- **Trigger:** Phase 3's flagged observation (A0–A5 all landing in a narrow,
  very high val band, 0.971–0.992) warranted investigation before Phase 4.
- **Constraints honored:** no model fit on test; no test result used to change
  any hyperparameter; `docs/LOCKED_RECIPE.md` untouched, zero recipe changes.
  Full writeup: `outputs/phase3_5_audit/REPORT.md`.
- **Findings:**
  1. **Near-duplicate analysis** (clip-level masked-mean DINOv2 embedding,
     cosine similarity): val→train 98.57% of clips have a train neighbor at
     similarity ≥0.95 (44.99% ≥0.99); test→train 99.09% ≥0.95 (47.34% ≥0.99).
     Elevated roughly equally for both split pairs — not localized to val alone.
  2. **12 highest-similarity cross-split pairs, rendered and visually inspected**
     (`outputs/phase3_5_audit/pairs/`): every one at similarity ≥0.9999996,
     identical labels, and visually the **same room, same students, same table,
     same objects, same moment** — adjacent temporal segments of one continuous
     recording split across train/val/test independently. One train clip
     (`low_view2603`) is the nearest neighbor for 8 different val/test clips.
  3. **Split structure:** view numbers are sequential within each class and
     **not blocked by split** — adjacent view numbers (N, N+1) land in
     different splits 33.1–35.1% of the time across all three classes, starting
     within the first ~40 clips of the `low` class. Consistent with a clip-level
     (not session-level) split of continuous recordings.
  4. **Trivial-feature probe** (logistic regression on mean-pooled clip
     embedding, no temporal/entity/attention structure): **OUC-CGE val
     macro-F1 = 0.9748** — within noise of the full locked-recipe EPT-Former
     (0.9767).
  5. **DAiSEE cross-check**, identical probe, subject-disjoint split by
     construction: **val macro-F1 = 0.2177.** Same code, same feature
     extraction, 0.97 vs. 0.22 — **localizes the anomaly to OUC-CGE's provided
     split, not to this project's pipeline, features, or probe.**
- **Conclusion:** OUC-CGE's train/val/test assignment does not separate
  recording sessions; every Phase 3 number computed correctly given the data as
  split, but the split itself is confounded with near-duplicate leakage. This is
  a data-integrity finding requiring a project-level decision (re-split by
  session? treat OUC-CGE results as unreliable pending a fix? both?) before
  Phase 4 can proceed meaningfully. **Not logged as a phase gate PASS/FAIL** —
  this audit doesn't have a go/no-go criterion; it surfaces a finding for
  discussion.

## Phase 3.5 data audit, continued — resplit viability + MELD admission (NOT an evaluation)

- **Date:** 2026-08-14T19:50:19Z
- **Constraints honored:** no test evaluation; `docs/LOCKED_RECIPE.md` untouched,
  zero recipe changes. Full writeup: `outputs/phase3_5_audit/TRACK1_TRACK2_REPORT.md`.
- **Track 1 — can OUC-CGE be re-split by recording?** Connected-components at
  0.95/0.98/0.99 give 126/1560/4905 components respectively, but a broader sweep
  (0.80–0.90) shows a **single component holding 98.5% of the entire 7700-clip
  corpus** — a continuum with no natural gap, not discrete recording clusters.
  Constructed a group-disjoint 80/10/10 split at threshold 0.98 (stratified by
  component); trivial probe on that val: **macro-F1 = 0.7305** — a real drop from
  the leaky 0.9748 but nowhere near the DAiSEE target (0.2177). Diagnostic:
  95.2% of the new val's clips still have a train neighbor at similarity ≥0.90
  even after grouping. **Verdict: NOT VIABLE** — the confound is structural
  (small number of fixed physical setups reused across the corpus, no
  session/recording metadata exists to group by instead — confirmed absent in
  Phase 2), not a threshold-calibration problem. Effective sample size (true
  independent recordings) is not reliably recoverable from visual similarity
  alone but is almost certainly far below 7700, plausibly in the tens.
- **Track 2 — MELD admission test** (1498-clip stratified sample, lightweight
  embeddings, screen only): near-duplicate similarity mean 0.70–0.75 vs. OUC-CGE's
  0.97–0.98, **zero pairs ≥0.99** (OUC-CGE: 45–47%), max cross-split similarity
  0.96 (barely above OUC-CGE's *mean*). 12 highest-similarity pairs visually
  confirmed as recurring standing sets/cast in different moments, not same-moment
  duplication — the documented, acceptable MELD property, not the OUC-CGE
  failure mode. Trivial probe: **macro-F1 = 0.4155**, modestly above the 0.333
  stratified-random floor, far below any degenerate/solved threshold. **Verdict:
  PASSES.**
- **CLAUDE.md updated**: new non-negotiable #8, no dataset enters this project
  without passing `scripts/dataset_admission.py` (near-duplicate audit + trivial
  probe vs. the DAiSEE subject-disjoint reference), codified as one command,
  validated to reproduce both established reference numbers exactly (OUC-CGE
  0.9748, DAiSEE 0.2177) before being run on MELD.
- **Recommendation:** MELD should become primary, or OUC-CGE should be demoted,
  pending a project-level decision — this changes `docs/DECISION_RULES.md`'s
  named primary dataset and is not something to resolve unilaterally. Existing
  OUC-CGE Phase 3 numbers should be treated as unreliable and not publishable
  as-is. Not logged as a phase gate PASS/FAIL for the same reason as above.
