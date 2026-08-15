# Two parallel screens: can OUC-CGE be fixed, and does MELD pass admission?

AUDIT/DESIGN ONLY. No test evaluation anywhere in this task. `docs/LOCKED_RECIPE.md`
untouched. Full data behind this report: `outputs/phase3_5_audit/resplit_viability_report.json`,
`outputs/dataset_admission/{ouccge,daisee,meld}/admission_report.json`.

## TRACK 1 — Can OUC-CGE be re-split by recording?

### 1. Connected components at three thresholds (all 7700 clips, ignoring the original split)

| Threshold | Components | Size mean | Size median | Size max | >1% of corpus | % clips in class-pure components | Weighted entropy (bits, max 1.585) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.95 | 126 | 61.1 | 1.0 | 6440 (83.6%) | 2 | 2.29% | 1.527 |
| 0.98 | 1560 | 4.9 | 1.0 | 576 | 9 | **85.82%** | 0.137 |
| 0.99 | 4905 | 1.6 | 1.0 | 72 | 0 | 100.00% | 0.000 |

**A broader sweep is decisive and wasn't in the original ask, but changes the
verdict:** at every threshold from 0.80 to 0.90, a single component holds
**98.5% of the entire corpus** (7587/7700 clips). The component count barely
moves (114–115) until ~0.92, then declines smoothly through 0.96–0.99 with no
natural break. This is a **continuum, not discrete clusters** — the similarity
manifold chains almost the whole dataset together at any threshold below ~0.92,
which is the classic signature of connected-components/single-linkage clustering
failing on data with no real gaps.

### 2. Viability question — are components class-pure?

At 0.98, yes, mostly (85.8% of clips sit in single-class components, weighted
entropy 0.137 bits out of a possible 1.585) — **but this doesn't mean the
components correspond to true independent recordings.** See §3.

### 3. Construct the group-disjoint split at 0.98, re-run the trivial probe

Stratified ~80/10/10 by component (`StratifiedGroupKFold`-style greedy
allocation, majority-class-per-component): train 6160 (class counts
low=2419/mid=1633/high=2108), val 771 (303/204/264), test 769 (302/204/263).

**Trivial probe on this group-disjoint val: macro-F1 = 0.7305.**

Original leaky val: 0.9748. DAiSEE reference: 0.2177. **This lands in neither
target zone** — a real drop (0.97 → 0.73) but nowhere near "closed."

**Diagnostic — why:** nearest-neighbor similarity from the new val to the new
train, computed *after* the group-disjoint split: mean 0.923, **95.2% of val
clips still have a train neighbor at similarity ≥0.90, 90.4% ≥0.95.** The 0.98
grouping threshold removes literal near-duplicate segments (the biggest single
contributor to the original leak) but leaves a large amount of residual,
still-very-high similarity between components — consistent with §1's finding
that the whole corpus lives on a smooth, densely-connected similarity manifold.
Different recording sessions that reuse the same physical room/camera/lighting
setup are apparently about as similar to each other as adjacent segments of one
session are — there is no similarity threshold where "same session" cleanly
separates from "different session, same setup."

### 4. Effective sample size — independent recordings, not clips

**Not reliably recoverable from visual similarity alone**, and that is itself
the finding. The 0.98-threshold component count (1560) is not trustworthy as an
estimate of true session count — §3 shows components at that threshold do not
function as independent units. The 0.80–0.90 range (114–115 components, one of
which alone eats 98.5% of the corpus) is closer to the truth of "how many
visually-distinct physical setups exist," consistent with OUC-CGE's own
description as a **fixed-camera** dataset (`docs/PLAN.md` §4) — but even that
number overstates independence, since the giant component's dominance suggests
the *effective* number of independent, exchangeable recording units is small,
plausibly in the tens, not the thousands implied by the 7700 clip count.
**Whatever the exact number, it is far below 7700 — every claim OUC-CGE
supports needs to be evaluated against that, not against clip count.**

### Track 1 verdict: **NOT VIABLE as a pure visual-similarity re-split.**

Grouping by embedding similarity partially closes the leak (0.97→0.73) but
cannot close it, because the underlying confound isn't "duplicate segments
occasionally split apart" — it's a dataset built from a small number of
physical setups with no natural similarity gap between "same session" and
"different session in the same room." No threshold fixes that. A real fix needs
actual session/recording metadata, which Phase 2's confound check already
established does not exist in the released dataset (no filename, directory, or
video-metadata identifier). **Recommend abandoning this track** rather than
tuning the threshold further — the residual-similarity diagnostic shows the
problem is structural, not a threshold-calibration issue.

---

## TRACK 2 — MELD admission test (1500-clip stratified sample, before any full extraction)

1498 clips sampled (stratified by split × sentiment; train 1092/val 120/test 286),
lightweight embeddings (8 frames/clip, whole-frame, no detection/tracking —
appropriate for a screen, not the full Phase-2-style pipeline).

### Near-duplicate audit

| Comparison | mean | median | max | %≥0.95 | %≥0.98 | %≥0.99 |
|---|---:|---:|---:|---:|---:|---:|
| val → train | 0.750 | 0.758 | 0.913 | 0.00% | 0.00% | 0.00% |
| test → train | 0.744 | 0.759 | **0.961** | 0.35% | 0.00% | 0.00% |
| val → test | 0.708 | 0.738 | 0.878 | 0.00% | 0.00% | 0.00% |

Contrast with OUC-CGE: mean similarity there was 0.97–0.98; here it's 0.70–0.75.
OUC-CGE had 45–47% of pairs above 0.99; MELD has **zero**. The single highest
MELD pair (0.961, test↔train) is the only one that even clears OUC-CGE's *mean*.

### Top-12 pairs — visually inspected

Every rendered pair is confirmed **same standing set and cast, different
moment**: e.g. `test_dia159_utt1` vs. `train_dia700_utt11` — both Ross and
Rachel in Monica's living room, but different scene, different framing, different
gesture; `test_dia72_utt2` vs. `train_dia362_utt7` — both a hospital-room scene
with Ross and Rachel, different dialogue beat entirely; `test_dia196_utt10` vs.
`train_dia596_utt7` — Monica's kitchen, same core cast, different blocking and
outfits. **None of the 12 are the same moment.** This is the documented,
acceptable property of MELD (one TV show, standing sets, recurring cast) — not
the OUC-CGE failure mode (literally the same frame down to hand position).

### Trivial-feature probe

**MELD: val macro-F1 = 0.4155** (n_train=1092, n_val=120). Stratified-random
floor for 3 classes is 0.333 (closed-form, Phase 2). Modestly above chance —
consistent with some real set/context correlation with sentiment (e.g. hospital
scenes skew negative) — **far below both OUC-CGE's leaky ceiling (0.97) and any
degenerate/solved-by-scene-recognition threshold.**

### Track 2 verdict: **MELD PASSES the admission test.**

---

## Recommendation

**MELD should become primary, or at minimum OUC-CGE should be demoted pending a
real fix.** OUC-CGE fails its own admission test outright (§ Track 1) and the
failure is structural, not repairable by re-splitting on the data as released.
MELD passes cleanly on every measure in this screen. This doesn't resolve
itself — it's a call about which dataset anchors the paper's primary claim, and
it changes the pre-registration (`docs/DECISION_RULES.md` names OUC-CGE as
primary). Recommend: treat existing OUC-CGE numbers (Phase 3's search results,
and by extension any future OUC-CGE test evaluation) as **unreliable and not
publishable as-is**, run the full MELD admission-passed pipeline as the
candidate new primary, and decide the pre-registration amendment before Phase 4
resumes on either dataset.
