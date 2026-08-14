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
