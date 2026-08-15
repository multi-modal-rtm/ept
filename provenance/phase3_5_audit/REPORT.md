# Phase 3.5 — Data audit (NOT an evaluation)

No model fit on test. No hyperparameter touched. `docs/LOCKED_RECIPE.md` unchanged.
All computations use the already-extracted frozen DINOv2 feature cache from Phase 2
— nothing was re-extracted, no label information from test flowed into any
parameter anywhere in this audit.

## 1. Near-duplicate analysis

Clip-level embedding = masked mean over all (entity, segment) positions in the
cached entity feature tensor, L2-normalized. Cosine similarity to nearest neighbor
in the comparison split.

| Comparison | n | mean | median | p90 | p99 | max | %≥0.95 | %≥0.98 | %≥0.99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val → train | 769 | 0.974 | 0.989 | 0.999 | 1.000 | 1.000 | **98.57%** | 84.01% | 44.99% |
| test → train | 771 | 0.981 | 0.990 | 0.999 | 1.000 | 1.000 | **99.09%** | 87.03% | 47.34% |
| val → test | 769 | 0.960 | 0.975 | 0.991 | 1.000 | 1.000 | 91.16% | 34.20% | 10.92% |

**Both train-val and train-test are severely elevated, roughly equally** — this is
not localized to one split pair; it's systemic across all three. (val→test is
somewhat lower, consistent with train being the largest pool and thus most likely
to contain a near-duplicate of any given clip, not with train-test being clean.)

## 2. The pairs — decisive

12 highest-similarity cross-split pairs rendered as side-by-side middle-frame
thumbnails (`outputs/phase3_5_audit/pairs/`). Every one of the top 12 has
**cosine similarity ≥ 0.9999996** (floating-point-indistinguishable from 1.0) and
**identical engagement labels** between query and match. Visual inspection:
**same room, same table, same students, same objects on the table, same poses** —
e.g. `high_view802` (val) vs. `high_view775` (train), and separately
`high_view791` (test) vs. `high_view773` (train), are visibly the *same frame*
down to hand position and laptop screen content. `low_view2603` (train) is the
nearest-neighbor match for **five different** val/test clips
(view2598, 2620, 2624, 2626, 2629, 2632, 2636, 2640 all resolve to it or a
near-identical neighbor). This is not "similar classroom footage" — these are
adjacent temporal segments of the same continuous recording session, chopped into
separate numbered "clips" and then scattered across splits independently.

## 3. Split structure

View numbers are sequential within each class (`view1..view2636` for `high`, etc.)
and **not blocked by split** — adjacent view numbers land in different splits
33.1–35.1% of the time across all three classes:

| Class | n clips | adjacent (N,N+1) pairs | different-split | % |
|---|---:|---:|---:|---:|
| high | 2636 | 2635 | 873 | 33.1% |
| low | 3028 | 3027 | 1061 | 35.1% |
| mid | 2041 | 2040 | 678 | 33.2% |

Sample of the first 40 sequential `low` view numbers: train dominates but test/val
clips are interspersed as early as view23, view26, view28, view38 — i.e. within
the first 40 sequentially-numbered clips of the entire class. Combined with §2's
visual evidence (adjacent-numbered clips being literally the same recorded
moment), this is consistent with **the original dataset split being drawn at the
individual-clip level from continuous recordings, without session/recording-level
grouping** — exactly the mechanism the audit was designed to test for.

## 4. Trivial-feature probe

Logistic regression on the mean-pooled clip embedding alone (no temporal axis, no
entity axis, no attention, no architecture at all) — trained on train, evaluated
on val:

**OUC-CGE: val macro-F1 = 0.9748.**

This is within noise of the full EPT-Former's locked-recipe result (0.9767) and
every other Phase 3 condition (0.971–0.992), using nothing but a linear classifier
on a globally-pooled feature vector. A linear probe with zero temporal/entity/
attention structure should not be competitive with a purpose-built architecture
unless the task, as split, is not actually testing what it's supposed to test.

## 5. DAiSEE cross-check — localizes the anomaly

Identical trivial probe, identical code path, DAiSEE's `Engagement` label
(4-class), **subject-disjoint split by construction**:

**DAiSEE: val macro-F1 = 0.2177.**

Same features, same probe, same amount of code — **0.97 vs. 0.22.** DAiSEE's
subject-disjoint split makes the trivial probe fail exactly as it should (below
even the ~0.25 chance floor for stratified-random guessing on 4 balanced-ish
classes, consistent with a genuinely hard, non-leaky classification problem).
**This localizes the anomaly to OUC-CGE's split, not to the feature cache, not to
the trivial probe, not to the EPT pipeline.**

## Bottom line

The near-saturated Phase 3 val numbers (flagged for discussion, not yet a
diagnosed problem, in the Phase 3 report) are now diagnosed: **OUC-CGE's
train/val/test split does not separate recording sessions.** Adjacent clips from
the same continuous classroom recording — same students, same room, same moment
— were assigned to different splits independently, so val and test both contain
near-duplicates of train clips at a rate too high to be coincidental (§1–§3), and
a model with zero architecture can exploit this to near-ceiling performance (§4),
while the identical method fails appropriately on a dataset that doesn't have
this problem (§5). Every Phase 3 result (A0–A5, mask-only, the recipe search) was
computed correctly given the data as split — the split itself is the confound.
This is a data-integrity finding about OUC-CGE's provided train/val/test
assignment, not a bug introduced anywhere in this project's own pipeline.
