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

## MELD promoted to primary — Phase 1/2 equivalent (tokenization + feature cache)

- **Date:** 2026-08-15T05:05:03Z
- **Decision (user):** MELD is primary; OUC-CGE becomes a rejected-dataset case study.
  No test evaluation in this work. Full writeup: `outputs/meld_phase1_2_summary/REPORT.md`.
- **`docs/DECISION_RULES.md` amended in two commits**, procedure fixed before the
  blind bootstrap ran: `a6d19a8` (OUC-CGE rejection + MELD promotion + effect-floor
  procedure, number left TBD) then `8c156a9` (computed floor filled in as a
  non-editing append). **`EFFECT_FLOOR = 0.04`** — the study is underpowered for
  the originally-hoped 0.02 margin; stated explicitly per the pre-committed rule.
- **Detector: SCRFD**, evidence-based (bake-off: SCRFD zero-rate 0.94% vs. YOLO
  0.88%, near parity — unlike OUC-CGE's severe pose-correlated SCRFD failure —
  plus the architectural requirement that identity clustering needs face crops).
  Identity recovery: agglomerative/cosine clustering, threshold 0.55 tuned on dev
  only (purity 62.2% vs. 6-class chance 16.7%).
- **Tokenization**: 13706 clips (train 9988/dev 1108/test 2610, matching labels
  exactly), 3783.9s/63.1min, 40 workers — required a mid-run fix (missing
  `cv2.setNumThreads(1)`, load hit 149 before the fix).
- **Feature cache**: E_max=8 cached, **primary E=6** (justified by data: median 4,
  p75=7 distinct identities/clip, `pct<=6=72.35%`), S=4, T=16, grid baseline from
  the same backbone/pass. 1226.3s/20.4min, 8 workers. `cache/MELD_MANIFEST.json`
  checksummed; spot-verify 5/5 exact; tarred-then-uploaded to
  `gdrive:ICFNDS 2026 - EPT/cache_backup_meld/` (1.561 GiB, confirmed).
- **Mask-only control**: real macro-F1 0.2677 (3-seed mean) is **below** the
  permutation-null mean (0.2982, p=0.9524) — no detectable presence-mask signal,
  contrary to the plausible speaking-tracks-onscreen hypothesis. Reported plainly.
- **Result:** PASS (tokenization + cache gate, mirroring Phase 2's manifest/
  checksum/spot-verify/backup criteria).

## MELD Phase 3 — purity-stratified prep, A6 literature search, recipe lock

- **Date:** 2026-08-15T07:09:48Z
- **Task (user):** "Phase 3 for MELD. No test evaluation." Two pre-specified
  additions (amend `docs/DECISION_RULES.md` before any run): purity-stratified
  secondary analysis (test-split terciles), and seeds extended 3→5 with
  `EFFECT_FLOOR` frozen at 0.04 (not recomputed). Frozen search grid, dev-only
  recipe selection, trivial-floor (0.4155) standing row, A6 as a genuinely
  sourced literature reference (not estimated).
- **`docs/DECISION_RULES.md` amended, two commits**, procedure fixed before each
  number was computed: `fbee42e` (purity-stratified analysis pre-specified +
  seeds extended, terciles left TBD) then `0102944` (purity distribution
  computed as a non-editing append: mean 0.8493, terciles at p33=0.8008/
  p67=0.9067, exact thirds — 870/870/870 of 2610 test clips). Detection-only
  pass, no labels touched (`scripts/meld_test_purity.py`, 767.1s, 30 workers).
- **A6 literature search** (`5423b5f`, amends `docs/PLAN.md` §5): no citable
  video-only **3-class sentiment** MELD baseline exists in the published
  literature — MELD's own paper explicitly excludes video baselines ("video-
  based speaker identification and localization is an open problem"), and every
  downstream video-only ablation located is for the *7-class emotion* task
  instead. Reported as a gap (mirroring the missing VideoMAE linear-probe row),
  not filled with a task-mismatched number.
- **Frozen search grid**: `docs/SEARCH_GRID_MELD.md` (`c87d939`), OUC-CGE's
  8-point grid reused verbatim, kept at 8 points per instruction (noting the
  stated premise that MELD dev is smaller than OUC-CGE val is factually
  backwards — MELD dev is 1108, OUC-CGE val was 769 — without changing the
  actionable instruction).
- **Recipe search**: 56 runs, 2100 dev evaluations, dev split only. One bug
  found and fixed mid-run (`44122ca`): `MaskOnlyMLP`'s input layer is
  E×S-shaped and was hardcoded to OUC-CGE's 8×8, crashing on MELD's E=6,S=4
  geometry after all 48 A0–A5 runs had already completed; fixed by
  parameterizing `build_model()`, then rerunning only the 8 crashed mask_only
  points rather than repeating completed work.
- **Locked recipe**: `docs/LOCKED_RECIPE_MELD.md` (`6b7fae0`). A0–A5 all clear
  the 0.4155 trivial-feature floor at their best grid point (dev band
  0.493–0.512); mask-only clears it at none of its 8 points (best 0.2491),
  consistent with the earlier no-presence-signal finding. Hyperparameters are
  now read-only.
- **Zero test evaluations**: `src/ept/train/dataset_meld.py` asserts
  `split in ("train", "dev")` at construction — there is no code path in this
  phase's dataset/training/search scripts that can load MELD's test split for
  anything label-bearing. The one test-split touch this phase makes
  (`meld_test_purity.py`) is unsupervised and detection-only, exactly as
  pre-specified.
- **Result:** PASS (recipe-lock gate). Phase 4 (locked-config × 5-seed run,
  test touched exactly once) not yet started — separate, later gate event.

## Pre-Phase-4 prep: hardcode audit + emotion calibration endpoint

- **Date:** 2026-08-15T08:07:37Z
- **Task (user):** Two tasks before Phase 4, neither touching test. (1) Grep
  `src/` for OUC-CGE-geometry literals that survived into MELD without crashing.
  (2) Pre-register and run a 7-class MELD emotion secondary calibration
  endpoint, since A6's search found no same-task sentiment baseline but did
  find same-task emotion ones.
- **Hardcode sweep**: grepped `src/ept/` for `8`/`16`/`64`/`4` in shape/slice/
  reshape/view/dimension contexts. Two real findings:
  - `MaskOnlyMLP(e_max=8, s=8)` — the crash already found and fixed in the
    prior gate (`44122ca`). Confirmed no other instance of this pattern
    crashed silently instead.
  - `EPTFormer(s_max=8)` default — **not** overridden anywhere in the MELD
    recipe-search path (`build_model()` only forwarded `e_max`/`s` to
    `MaskOnlyMLP`). Did not crash and did not corrupt results: verified by
    direct computation that `SinusoidalSegmentEmbedding`'s sliced output is
    bitwise-identical regardless of unused table headroom (it's a fixed,
    non-learned buffer indexed by absolute position, not relative to table
    size) — confirmed `torch.equal(pe_maxs8[:4], pe_maxs4[:4])` is `True`.
    Also verified a future `S > s_max` fails loudly (`RuntimeError` on the
    `.view()` reshape) rather than silently truncating. Hardened anyway
    (`f2571d0`): `build_model()` now forwards `s_max` explicitly for every
    `EPTFormer`-based condition, so this stops being "safe by an unexamined
    invariant" and starts being "correct by construction." All 13 tests still
    pass after the change; the already-locked recipe table (`6b7fae0`) is
    numerically unaffected by construction, not just by assumption.
  - Also flagged, not fixed (out of scope for "neither touches test," and
    Phase 4 hasn't started): `train.py`'s hydra `main()` is still hardcoded to
    `OUCCGEDataset` — a MELD-equivalent entry point needs to exist before
    Phase 4's locked-config run reuses this path, or it will silently train on
    the wrong dataset rather than crash.
- **7-class emotion calibration** (`f2571d0` amendment, `e6a1291` computed
  numbers): A0/A1/A2 + trivial probe, dev only, locked recipe reused unchanged
  (no new search). Results (macro-F1 / weighted-F1): A0 0.1732/0.2990, A1
  0.1958/0.3235, A2 0.1719/0.3372, trivial probe 0.1748/0.2890, majority-class
  0.0850/0.2518, stratified-random macro-F1 (K=7) 0.1429. Published video-only
  MELD-emotion weighted-F1 range located and cited (`docs/PLAN.md` §5):
  25.18%–61.4% (61.4% an unverified-metric outlier; 37.9–41.8% the more
  comparable feature-based cluster). **Verdict: calibrated, not alarming.** A1
  clears every internal floor (majority, stratified-random, trivial probe) by
  a comfortable margin; it sits modestly below the closest-matched published
  cluster, which is expected (untuned, sentiment-locked recipe on a harder,
  more imbalanced task never searched for it), not evidence of a pipeline bug.
- **Result:** PASS. No blocker for Phase 4 found; one action item recorded
  (MELD training entry point still needs building before Phase 4, see above).

## Track 1 — Phase 6: efficiency + figures

- **Date:** 2026-08-15T16:33:10Z
- **Commits:** `dfec96a` (efficiency measurement: GFLOPs, latency, memory,
  detection cost — `results/efficiency.csv`, `results/efficiency_budget_sweep.csv`),
  `3f58bdf` (4 vector-PDF figures, `paper/figures/`).
- **Key finding**: detection+clustering (6.28s/clip, single-clip single-thread)
  is 2-3 orders of magnitude larger than the model's own end-to-end compute
  (9.5–39ms GPU depending on condition) — reported as its own line item per
  `docs/PLAN.md` §6's explicit warning, never folded into model GFLOPs/latency.
  A0 (grid) pays zero detection cost but ~4.5x more backbone GFLOPs than the
  entity conditions (crops every spatial region every frame unconditionally
  vs. only actually-detected top-E entity frames).
- **Tooling note**: `thop` substituted for the `ptflops`/`fvcore` PLAN.md
  suggested — neither is installed, and installing them risked re-triggering
  the opencv-headless/opencv-python conflict (`CLAUDE.md` hard-won fact).
- **Figures**: architecture schematic (cross-checked against `ept_former.py`'s
  actual `forward()`, not memory), OUC-CGE leakage (two visually distinct
  cosine-sim-1.000000 cross-split pairs), accuracy-vs-GFLOPs Pareto (detection
  cost explicitly excluded, labeled as such), per-seed margin divergence
  (item- vs seed-paired CIs, seed 31337 marked).
- **Result:** PASS. Track 1 complete in full.

## Phase 4 — MELD test evaluation (the one-shot test touch)

- **Date:** 2026-08-15T15:03:40Z
- **Commits (code committed before any test run):** `8747720` (main matrix +
  budget sweep scripts + `MELDTestDataset`/`MELDBudgetDataset`, verified via
  unit tests + dev-split smoke tests + shape checks across all 10 (E,S)
  points, all on train/dev only, before touching test), `3ee4b05` (paired
  bootstrap, purity stratification, summary.csv generator — these scripts
  read only already-written `predictions.json`/`metrics.json`, verified
  against synthetic data with known answers in all three branch directions
  before real predictions existed), `4de324b` (`results/summary.csv`).
- **Runs:** A0–A5, mask-only × 5 seeds {42,1337,2024,7,31337} = 35 runs,
  locked recipe (`docs/LOCKED_RECIPE_MELD.md`) unchanged, train-to-completion
  then a single final-epoch test evaluation per (condition, seed) — no
  per-epoch test touching, no test-based epoch selection. Plus the
  trivial-feature probe (full train → full test, not a subsample). Plus the
  token-budget sweep (E∈{1,2,4,6,8} × S∈{2,4}, A1, r07 unchanged, 5 seeds =
  50 runs; S=2 derived from the cached S=4 features by documented segment
  merging, not a fresh extraction). **90 total (condition,seed) test-eval
  points, one test touch each.**
- **Pre-launch checkpoint:** `--print-config-only` for A1/seed42 confirmed
  `dataset=meld, e_max=6, s_max=4` before any run launched.
- **H1 (A1−A2, persistence):** mean margin `0.0340`, 95% CI `[0.0203, 0.0480]`
  (excludes zero), 10,000-resample item-paired bootstrap. **Below
  `EFFECT_FLOOR=0.04`** — not supported.
- **H2 (A1−A0, entity structure):** mean margin `0.0315`, 95% CI
  `[0.0159, 0.0470]` (excludes zero). **Below `EFFECT_FLOOR=0.04`** — not
  supported.
- **Decision rule: no branch fired.** Neither H1 nor H2 clears the
  pre-registered floor despite both margins being positive with
  zero-excluding CIs — exactly the "underpowered for the 0.04 margin"
  scenario the 2026-08-15 effect-floor amendment named in advance as a
  possible outcome and pre-committed to reporting as underpowered, not as
  evidence of absence. Applied exactly as frozen; not softened, not argued
  with. Full numbers: `outputs/meld_phase4_statistics.json`.
- **Purity-stratified secondary (does not alter the branch):** margins
  low/mid/high purity tercile = `0.0377 / 0.0264 / 0.0389` (n=870 each), all
  three CIs exclude zero, **pattern is not monotone** (dips at mid) — per the
  pre-specification, a non-monotone pattern is evidence against "identity
  recovery quality is the limiting factor," sharpening rather than resolving
  the pooled underpowered result. `outputs/meld_phase4_purity_stratified.json`.
- **Budget sweep:** `outputs/meld_phase4_budget_sweep_summary.json`. E=6,S=4
  point (`0.4040±0.0077`) matches A1's main-matrix result exactly — internal
  consistency check passes (same recipe/geometry computed two independent
  ways).
- **Anomaly, reported not excluded:** A2 seed=31337 collapsed to
  near-majority-class prediction (macro-F1 `0.2166`, zero F1 on
  negative/positive) — a genuine training-instability outlier, included as-is
  in all means/std/bootstrap per `CLAUDE.md` non-negotiable #6 (no
  cherry-picking across seeds).
- **Result:** Test split touched exactly once per (condition, seed, E, S)
  point; no test-based model/epoch selection anywhere in this phase.
  `results/summary.csv` written. **Decision rule outcome: no branch fires —
  study underpowered for H1/H2 at `EFFECT_FLOOR=0.04`, both margins positive
  and significant but below the pre-registered bar.**
