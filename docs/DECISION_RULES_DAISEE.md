# Pre-registration — DAiSEE H2 replication

**Committed:** _(fill on commit, before any test evaluation)_

This document is frozen once committed. Amendments append below with a date and reason;
nothing above the amendment line is ever edited. Separate file from `docs/DECISION_RULES.md`
(MELD, primary) by design — this is a replication track with a hard external deadline
(complete by 2026-08-24 or dropped), not a revision of the MELD study.

## Scope: H2 only

DAiSEE's cache is `E_max=8` (`cache/features/daisee/`, tokenized/extracted and gated already —
`logs/GATES.md`, DAiSEE 8571/8571 clips) but the **primary condition here is locked at `E=1`**,
carried forward from `CLAUDE.md`'s existing note ("DAiSEE is E=1 by construction; the cap was
never binding there") — DAiSEE clips are single-subject webcam recordings; the detector's raw
output has multiple candidate person-tracks per clip (background/reflection artifacts, brief ID
switches), but only the single top-confidence×coverage track corresponds to the actual annotated
subject. Truncating to `E=1` therefore discards noise, not signal.

**With `E=1` there is no second entity slot to shuffle, so A2 (identity-shuffled EPT) is
undefined here and is not run. H1 (persistence) is consequently not tested and not restated —
a single-slot sequence has no cross-entity identity to persist or scramble.** This track tests
**H2 only**: `A1 (EPT) − A0 (grid tokens, matched token budget) ≥ EFFECT_FLOOR`.

Rationale for testing H2 specifically as a replication target: MELD's Phase 4 found H2's margin
(0.0315, 95% CI [0.016, 0.047], excluding zero) positive and significant but short of
`EFFECT_FLOOR=0.04` — an underpowered-but-suggestive result for "entity-localized tokens help
over a fixed spatial grid, independent of persistence." DAiSEE tests whether that same
entity-vs-grid gap replicates on a structurally different dataset (single-subject webcam feed vs.
multi-party TV dialogue) rather than being an artifact of MELD's specific data.

## Primary endpoint

Test macro-F1 on DAiSEE, mean over seeds `{42, 1337, 2024, 7, 31337}`, DAiSEE's official
train/val/test splits (val plays the role dev played for MELD). **4-class Engagement**
(`clip["labels"]["Engagement"]`, levels 0-3), matching the label already used for this
project's own DAiSEE admission-test reference number (`CLAUDE.md` non-negotiable #8,
trivial probe macro-F1 0.2177) — not re-defined or re-binned for this track.

**Known, foreseeable class-imbalance risk, stated in advance:** Engagement's class distribution
is heavily skewed (train: {0:34, 1:213, 2:2617, 3:2494}; val: {0:23, 1:143, 2:813, 3:450}; test:
{0:4, 1:84, 2:882, 3:814}). Class 0 has only **4 test clips**. Macro-F1 will be highly sensitive
to this class specifically — a model that never predicts class 0 scores exactly 0 F1 on it
regardless of overall quality, and 4 test items give that one class enormous single-item
leverage on the aggregate metric. This is flagged here, before any run, as a known interpretive
hazard for the eventual test numbers, not discovered after the fact.

## Hypothesis

- **H2 (entity structure).** `A1 (EPT) − A0 (grid tokens, matched token budget) ≥ EFFECT_FLOOR` macro-F1.
- H1 is explicitly not tested (see Scope above).

## Decision rule — fires once, on locked test numbers

Simplified from `docs/DECISION_RULES.md`'s three-branch table (which distinguishes persistence
from entity-structure effects via A1 vs A2) since A2 does not exist here:

| Outcome | Branch |
|---|---|
| H2 supported (margin ≥ EFFECT_FLOOR AND paired-bootstrap 95% CI excludes zero) | **(a)** Entity-localized tokenization helps over a fixed grid, replicating MELD's H2 direction. |
| abs(margin) < EFFECT_FLOOR, or CI does not exclude zero | **(b)** Null / underpowered — reported as such, not as evidence of absence (same discipline as MELD's actual H1/H2 outcome). |
| A1 < A0 − EFFECT_FLOOR | **(c)** Reversed: entity tokens hurt relative to the grid on this dataset — a genuine boundary finding, reported plainly. |

**Paired requirement**, same as `docs/DECISION_RULES.md`: a branch fires only if BOTH (a) the
point-estimate margin clears `EFFECT_FLOOR` in the relevant direction, AND (b) a paired
bootstrap over test items (10,000 resamples, same item-paired design as
`scripts/meld_phase4_statistics.py`) gives a 95% CI that excludes zero.

## Effect-floor procedure — fixed here, before the number is computed

Same blind procedure as MELD's (`docs/DECISION_RULES.md`, 2026-08-15 amendment), adapted to
two conditions instead of seven:

1. Estimate seed-level and item-level variance of macro-F1 on DAiSEE **val**, via bootstrap,
   for **A0 and A1 only** (no A2-A5/mask-only here — H2 is the only comparison this track makes).
   Only dispersion (standard errors) is recorded; which condition scores higher is never
   inspected or used to set the floor.
2. `EFFECT_FLOOR = max(0.02, 2 × pooled_SE)`, rounded **up** to two decimals.
3. If the pooled SE implies the study cannot detect a 0.02 margin at reasonable power, this
   amendment states so explicitly (matching MELD's precedent), and underpowered is reported as
   underpowered, not as a null result standing in for evidence of absence.

**`EFFECT_FLOOR = TBD`** — filled in by an immediately-following, separately dated commit below,
computed by the procedure above, run once, not tuned. This commit's text above is not edited by
that follow-up; the number is appended, not backfilled into this section.

## Locked architecture and protocol

Same architecture as MELD (`EPTFormer`/`MeanPoolMLP`, unchanged, D=384, L=4, 6 heads,
mlp_ratio 4, pre-norm — matched-feature discipline, `CLAUDE.md` non-negotiable #3). Geometry:
`E=1` (A1, primary), `S=8`, `T=32` (DAiSEE's existing cache convention, `cache/features/daisee/`
and `cache/features_grid/daisee/`, both already extracted and gated). A0's grid baseline uses
the existing `features_grid/daisee` cache (same 2×4=8-region tiling as OUC-CGE/MELD, `S=8`
segments — 64-token budget). **`NUM_CLASSES=4`** here (not the project's usual 3), passed
explicitly to `EPTFormer`/`MeanPoolMLP` per-call, per the same pattern already used for MELD's
7-class emotion secondary endpoint (`src/ept/train/emotion_labels.py`) — the shared model code's
own `NUM_CLASSES=3` module-level default is never changed, only overridden at construction.

Optimizer recipe: locked at the end of this track's Phase 3 (fresh 8-point dev grid search,
`docs/SEARCH_GRID_DAISEE.md`, not yet run) into `docs/LOCKED_RECIPE_DAISEE.md`; unchanged
thereafter. **Not the same locked recipe as MELD's** — a fresh search, since dataset/geometry
differ.

## What would invalidate the study

- Any condition trained on features other than the shared cache (`cache/features/daisee/`,
  `cache/features_grid/daisee/`).
- Test split touched more than once, or touched before Phase 3's grid search and effect-floor
  procedure are both complete and committed.
- Missing the 2026-08-24 deadline: per the task that opened this track, incomplete work is
  **dropped**, not carried over as a silently-abandoned partial result.

## Gates for this track (mirrors `docs/DECISION_RULES.md`'s discipline, own log entries)

1. This pre-registration, committed before any run (this commit).
2. Effect-floor computed (blind bootstrap, A0/A1 only) — immediately following, separate commit.
3. Phase 3 equivalent: frozen 8-point grid (`docs/SEARCH_GRID_DAISEE.md`), dev-only search,
   locked recipe (`docs/LOCKED_RECIPE_DAISEE.md`) — separate, later gate.
4. Phase 4 equivalent: 5-seed test evaluation, touched once, decision rule applied exactly as
   frozen — separate, later gate.
