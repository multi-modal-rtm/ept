# Pre-registration — EPT

**Committed:** _(fill on commit, before any test evaluation)_

This document is frozen once committed. Amendments append below with a date and reason; nothing above
the amendment line is ever edited.

## Primary endpoint

Test macro-F1 on **OUC-CGE**, mean over seeds {42, 1337, 2024}, 771-clip video-only protocol.

## Hypotheses

- **H1 (persistence).** `A1 (EPT) − A2 (identity-shuffled EPT) ≥ +0.02` macro-F1.
- **H2 (entity structure).** `A1 (EPT) − A0 (grid tokens, matched 64-token budget) ≥ +0.02` macro-F1.

## Decision rule — fires exactly once, on locked test numbers

| Outcome | Branch |
|---|---|
| H1 supported | **(a)** Persistence is the mechanism. Headline comparison: A1 vs A2. Title 1. |
| abs(A1 − A2) < 0.02 and H2 supported | **(b)** Boundary paper: persistence is *not* what helps; entity-localized cropping and token budget are. Headline: A1/A2 vs A0. Title 2. |
| A1 < A2 − 0.02 | **(c)** Hypothesis refuted. Paper pivots to the efficiency Pareto result; the negative finding is the contribution. Title 2. |

## Controls fixed in advance

- **No learned per-slot identity embedding** — it would leak slot ordering and confound A1 vs A2.
- **Slot assignment order is randomized per sample in every condition, including A2.** A1-vs-A2
  therefore isolates temporal identity consistency and nothing else.
- Entity selection: top-`E` tracks by (mean detection confidence x frame coverage), `E_max = 8`,
  identical selection code in all conditions.
- `A2` shuffles entity-to-slot assignment **independently per segment**, using a per-sample seed
  derived from the run seed, so the shuffle is reproducible.

## Locked architecture and protocol

`T = 32`, `S = 8`, `E_max = 8`, `d = 384`, `L = 4`, 6 heads, mlp_ratio 4, pre-norm.
Optimizer recipe: locked at end of Phase 3 into `docs/LOCKED_RECIPE.md`; unchanged thereafter.

## What would invalidate the study

- Tracking QA gate failure (<80% of clips with >=2 tracks at >=50% coverage): the entity abstraction
  is not recoverable from this data. Fall back to region-persistent tokens and **relabel the claim
  in the title and abstract** — do not quietly keep calling it entity-persistent.
- Any condition trained on features other than the shared cache.

## Amendments

### 2026-08-14 — Mask-only control

Mask-only control (added 2026-08-14, before any test evaluation): a classifier trained on the
entity presence mask `[E,S]` alone, with no visual features. Rationale: face detection failure is
plausibly correlated with the engagement label (disengaged students turn away), so the presence
pattern may leak label information. Reported alongside A0–A6.

### 2026-08-14 — Extended token-budget sweep

Token-budget sweep extended to E in {1,2,4,8,16} (added 2026-08-14, before any test evaluation).
Rationale: 19% of OUC-CGE clips exceed 8 simultaneous tracked people, and dropped tracks are
predominantly whole persons (159/180), so the original sweep ceiling coincided with systematic
truncation. The primary condition is UNCHANGED at E=8; slice-equivalence to the original pipeline
is asserted in tests.

### 2026-08-14 — Primary dataset changed: OUC-CGE rejected, MELD promoted

**OUC-CGE dropped as primary** (added 2026-08-14, before any test evaluation on either dataset).
Phase 3.5 data-audit findings (`logs/GATES.md`, `outputs/phase3_5_audit/`): a trivial-feature
probe (no temporal/entity/attention structure — mean-pooled clip embedding, logistic regression)
scores macro-F1 **0.9748** on OUC-CGE's provided val split, within noise of the full locked-recipe
EPT-Former (0.9767). A single connected component holds **98.5% of the entire 7700-clip corpus**
at cosine-similarity thresholds 0.80–0.90 (masked-mean DINOv2 clip embeddings), consistent with a
small number of fixed physical recording setups reused across nearly the whole dataset, not
thousands of independent sessions. A group-disjoint re-split at threshold 0.98 reduced but did not
close the leak (trivial probe **0.7305** on the re-split val, vs. a DAiSEE subject-disjoint
reference of 0.2177) — residual near-duplicate similarity (95.2% of the re-split val's clips at
similarity ≥0.90 to a train clip) shows the confound is structural, not a threshold-calibration
artifact, and no session/recording metadata exists in the released dataset to group by instead.
**OUC-CGE is retained in the project as a rejected-dataset case study** (methodology and finding,
not a results-table entry) — its Phase 3 numbers are not to be reported as EPT results anywhere.

**MELD promoted to primary.** `docs/PLAN.md` §4's description of MELD as a "conditional
generalization set" is superseded — MELD is now primary, not conditional, and Phase 5's original
gate (only-if-Phase-4-on-schedule) no longer applies. Passed the dataset admission test
(`scripts/dataset_admission.py --dataset meld`, `CLAUDE.md` non-negotiable #8): near-duplicate
similarity mean 0.70–0.75 (vs. OUC-CGE's 0.97–0.98), zero pairs ≥0.99 (OUC-CGE: 45–47%), trivial
probe macro-F1 **0.4155** (modestly above the 0.333 stratified-random floor, far below any
degenerate/solved threshold). **Primary endpoint: test macro-F1 on MELD, 3-class sentiment** (not
7-class emotion), mean over seeds `{42, 1337, 2024}` (unchanged), MELD's official train/dev/test
splits (dev plays the role val played for OUC-CGE).

**H1/H2 unchanged in form, re-targeted to MELD:**
- H1 (persistence). `A1 (EPT) − A2 (identity-shuffled EPT) ≥ EFFECT_FLOOR` macro-F1.
- H2 (entity structure). `A1 (EPT) − A0 (grid tokens, matched budget) ≥ EFFECT_FLOOR` macro-F1.

**Effect-floor procedure — fixed here, before the number is computed.** The estimate below runs
immediately after this commit, blind to which condition scores higher:

1. Estimate seed-level and item-level variance of macro-F1 on MELD **dev** via bootstrap, using
   only conditions already implemented (A0–A5, mask-only). Only dispersion (standard errors) is
   recorded; which condition scores higher is never inspected or used to set the floor.
2. `EFFECT_FLOOR = max(0.02, 2 × pooled_SE)`, rounded **up** to two decimals.
3. **Paired requirement.** A decision-rule branch fires only if BOTH (a) the point-estimate margin
   clears `EFFECT_FLOOR`, AND (b) a paired bootstrap over test items gives a 95% CI on the margin
   that excludes zero.
4. If the pooled SE implies the study cannot detect a 0.02 margin at reasonable power, this
   amendment states so explicitly, and the paper will report the study as **underpowered** for
   that comparison rather than reporting a null result as evidence of absence.

**`EFFECT_FLOOR = TBD`** — filled in by an immediately-following, separately dated commit below,
computed by the procedure above, run once, not tuned. This commit's text above is not edited by
that follow-up; the number is appended, not backfilled into this section.

### 2026-08-15 — Effect floor computed (immediate, same-session follow-up to the above)

Bootstrap run per the procedure fixed in the amendment above, in the same working session, blind
to condition ranking throughout (only dispersion was ever inspected or recorded). Fixed recipe used
to produce trained models to bootstrap from (not a new locked recipe, not tuned for this purpose):
`lr=1e-4, weight_decay=0.0, dropout=0.0, epochs=50, batch_size=32` (`docs/SEARCH_GRID.md`'s r07, the
single most frequently-best point across the OUC-CGE search).

- Seed-level: 3 seeds `{42, 1337, 2024}` per condition (A0–A5, mask-only), pooled variance across
  conditions: `0.000075`.
- Item-level: 500-resample bootstrap over MELD dev items on each condition's seed-42 model, pooled
  variance across conditions: `0.000216`.
- `pooled_SE = sqrt(0.000075 + 0.000216) = 0.0170`.
- `EFFECT_FLOOR = max(0.02, 2 x 0.0170) = max(0.02, 0.0341) = 0.0341`, rounded **up** to two
  decimals: **`EFFECT_FLOOR = 0.04`**.

**The study is underpowered for the originally-hoped-for 0.02 margin** — the natural noise floor at
this dev-set size (1108 clips) already exceeds it. Per the procedure fixed above, this is stated
plainly: H1 and H2 will be evaluated against `EFFECT_FLOOR = 0.04`, not 0.02, and if neither
hypothesis clears 0.04, the paper reports the comparison as underpowered rather than treating a
null result under a floor the study could never have cleared as evidence of absence.

**H1/H2, with the computed floor substituted:**
- H1 (persistence). `A1 (EPT) − A2 (identity-shuffled EPT) ≥ 0.04` macro-F1.
- H2 (entity structure). `A1 (EPT) − A0 (grid tokens, matched budget) ≥ 0.04` macro-F1.

Full report: `outputs/meld_effect_floor/effect_floor_report.json`.

### 2026-08-15 — Purity-stratified secondary analysis (pre-specified, before any run)

**Pre-specified secondary analysis, fixed before purity is computed for any clip.** Rationale: the
clustering threshold tuned in Phase 1 (0.55) achieves 62.2% purity against the 6-character reference
check — well above chance (16.7%) but far from perfect. This means a null on the pooled test set is
**ambiguous between two different findings**: "persistence does not help" vs. "identity was not
recovered well enough to tell." Stratifying by purity distinguishes them — a monotone trend is
evidence even when the pooled A1−A2 difference misses `EFFECT_FLOOR`.

**Purity metric** (computable without any character-identity ground truth, so it applies to every
clip, not just the 6 recurring characters used for threshold tuning): per-clip **mean silhouette
score** (cosine distance) of the agglomerative cluster assignment over that clip's detected face
embeddings, at the locked threshold (0.55). Clips with fewer than 2 detections or only 1 resulting
cluster get purity `= 1.0` by convention — there is no cross-identity ambiguity possible in a
discovered structure with only one identity, so it is trivially "pure." (Embeddings were not
retained in the production tracking cache to keep it lean — computed via a fresh, detection-only
pass over the **test split only**, since that is the only split this pre-registered analysis
stratifies; train/dev do not need this metric for this plan.)

**Pre-specified analysis, secondary, does NOT affect the primary branch decision (§ Decision rule)**:
split test clips into purity **terciles** (bottom/middle/top third by this score). Report `A1 − A2`
macro-F1 within each tercile. **Prediction, stated in advance**: if persistence carries real signal,
the `A1 − A2` gap grows **monotonically** with purity tercile (low-purity tercile closest to null,
high-purity tercile showing the clearest gap). A monotone trend across the three terciles is treated
as corroborating evidence for H1 even if the pooled (non-stratified) difference does not clear
`EFFECT_FLOOR = 0.04`; a flat or non-monotone pattern across terciles is treated as evidence against
the "identity wasn't recovered well enough" explanation, sharpening a pooled null into a genuine
finding that persistence does not help, rather than leaving it ambiguous.

**Purity distribution and exact tercile cutpoints = TBD** — filled in by an immediately-following,
separately dated commit below, computed once the test-split purity pass runs, not tuned. This
commit's text above is not edited by that follow-up.

### 2026-08-15 — Purity distribution computed (immediate, same-session follow-up to the above)

Detection-only pass over all 2610 MELD test clips (`scripts/meld_test_purity.py`), 767.1s wall clock,
30 workers. No labels touched.

- `purity_mean = 0.8493`, `purity_median = 0.8608`.
- Tercile cutpoints: `p33 = 0.8008`, `p67 = 0.9067`.
- Tercile counts: low (`purity < 0.8008`) = 870, mid = 870, high (`purity >= 0.9067`) = 870 — exact
  thirds of 2610, confirming the percentile split is clean with no ties landing on a cutpoint.

Full results (per-clip purity, detection counts): `outputs/meld_purity/test_purity.json`.

### 2026-08-15 — Seeds extended 3 -> 5

Seeds extended from `{42, 1337, 2024}` to `{42, 1337, 2024, 7, 31337}` for all future reported
numbers (mean ± std, `CLAUDE.md` non-negotiable #6). **`EFFECT_FLOOR` stays at `0.04` — it is
explicitly NOT recomputed.** Recomputing the floor now, after seeing that 2 more seeds are being
added, would lower the seed-level variance component using information gathered after the original
blind bootstrap — exactly what the blind procedure (record dispersion only, never inspect ranking,
freeze before running) existed to prevent. The added seeds increase the precision of future reported
means; they cannot retroactively lower the bar a result has to clear.

### 2026-08-15 — 7-class MELD emotion added as a secondary calibration endpoint

**Secondary calibration endpoint, fixed before any run.** `docs/PLAN.md`'s A6 search
(`5423b5f`) found no citable published video-only baseline for MELD's 3-class sentiment task —
MELD's own paper excludes video baselines entirely, and every located downstream video-only
ablation is for the 7-class emotion task instead. Rather than leave A6 with nothing to compare
against, 7-class emotion is added as a **secondary, calibration-only** endpoint: its sole purpose
is checking this pipeline's numbers land in a sane range next to published video-only MELD emotion
baselines, since a same-task published reference is available there and is not for sentiment.

**Explicitly not part of the branch decision (§ Decision rule) and does not touch H1/H2.** No new
recipe search — reuses the already-locked recipe per condition from `docs/LOCKED_RECIPE_MELD.md`
(A0: r08, A1: r07, A2: r03) unchanged, since this endpoint asks "is the pipeline sane," not "what's
the best recipe for emotion." Scope: **A0, A1, A2 + the trivial-feature probe, dev split only, no
test evaluation** — sufficient to compare against a literature range; the full A0–A5 + mask-only
matrix is not needed for a calibration check.

**Labels**: MELD's standard 7-class taxonomy (anger, disgust, fear, joy, neutral, sadness,
surprise), read directly from the same raw label CSVs already used for tokenization
(`src/ept/train/emotion_labels.py`), joined against the existing cache by clip_id — no new
tokenization or feature-extraction pass, since emotion is a label swap over the same cached
features, not a different input.

**Verdict criterion, stated in advance**: if our dev macro-F1 (or weighted-F1, whichever the
cited literature reports, both will be computed and labeled) lands within or above the published
video-only range, the pipeline is calibrated and this is reported as supporting evidence that the
tokenization/training pipeline works as intended, not as a new hypothesis test. If it lands **far**
below the published range, that is treated as a pipeline problem to diagnose **before** the Phase 4
test run, not folded into the results as if it were informative about H1/H2 — this endpoint exists
precisely to catch that failure mode before the one-shot test evaluation, not to explain it after.
