# Track B — Mask-only diagnostics (VAL split, revisiting the Phase 1 finding)

## 1. Redone baselines — the permutation null changes the picture

| Baseline | macro-F1 |
|---|---:|
| Majority-class | 0.190 |
| Stratified-random (closed-form, K=3 classes) | **0.333** = 1/K, exactly, regardless of class balance |
| Stratified-random (empirical, 200 draws) | 0.337 ± 0.032 |
| Permutation null (20 shuffles, 3-seed mean each — matches how 0.373 was computed) | **0.322 ± 0.022** |
| Permutation null (200 shuffles, robustness check, single seed) | 0.320 ± 0.030 |
| **Real logistic regression (3-seed mean, as originally reported)** | **0.373** |

**Empirical p-value: 0.095 (20 perms, matched to the exact 0.373 figure) / 0.110 (200-perm robustness check).** Neither clears the conventional 0.05 threshold.

**Closed-form derivation for the stratified-random baseline:** for K-class stratified
random guessing (predictions drawn independently from the same prior as the true
labels), precision_c = recall_c = p_c in expectation for every class c, so F1_c = p_c,
and macro-F1 = mean(p_c) = 1/K — **exactly 0.333 for 3 classes, regardless of the
actual class skew.** This is markedly higher than the majority-class baseline (0.190)
and much closer to the real logistic-regression score.

**Reading:** majority-class macro-F1 is a systematically weak baseline for 3-class
macro-F1 — a classifier that ignores the two minority classes entirely already scores
near-zero on 2 of 3 classes by construction, so *any* classifier that hits all three
classes with non-trivial rates will clear majority-class by a wide margin whether or
not it has learned anything real. Comparing 0.373 against 0.190 (the Phase 1 framing)
overstated the finding. Comparing it against the proper null — a permutation test that
destroys any real label relationship while preserving the mask's feature distribution
— **the result is not statistically significant** at conventional thresholds, though it
sits in the "suggestive, want more data" range (p≈0.10, ~2.3 std above the null mean)
rather than clearly null.

## 2. What the mask encodes — reduced feature sets

| Feature set | dim | macro-F1 (3-seed mean ± std) |
|---|---:|---:|
| track count only (# entities with any presence) | 1 | 0.196 ± 0.008 |
| mean coverage only (overall presence density) | 1 | 0.196 ± 0.008 |
| **per-segment presence counts** (how many of the ≤8 entities present, per segment) | 8 | **0.325 ± 0.014** |
| full mask (entity × segment presence) | 64 | 0.373 ± 0.011 |

The two single-scalar features carry essentially no signal beyond majority-class —
plausibly because track count saturates near E_max=8 for most clips (little variance
to exploit; see Track A's truncation finding) and mean coverage alone doesn't linearly
separate the classes well enough for logistic regression to beat an intercept-only fit.

**The 8-dimensional per-segment presence-count vector alone recovers 0.325 of the
0.373 full-mask score** — the large majority of whatever signal exists. **This means the
mechanism, if real, is aggregate group-presence dynamics over time** ("how many
trackable people are visible at each of the 8 time segments") **rather than which
specific entity slot is occupied when.** The full 64-dim mask's entity-identity
structure adds comparatively little (0.373 vs 0.325) on top of that aggregate signal.

## 3. Confound check — no recording/session/room identifier exists

Checked three places:
- `train.xlsx`/`val.xlsx`/`test.xlsx` — byte-for-byte the same single
  `"path label"` column as the `.csv` files. No extra metadata columns.
- Video container metadata (`ffprobe -show_entries format_tags`) — only generic
  encoder tags (`Lavf58.47.100`, `isom` brand). No per-clip recording timestamp.
- Filesystem mtimes cluster within minutes of each other on the same day
  (2025-06-22), consistent with a single bulk dataset-preparation/copy pass, not
  original recording times — not usable as a session proxy.
- Filenames are sequential `view<N>.mp4` with no visible session/room encoding.

**No identifier exists to check a room/session confound against.** This is a genuine
gap, not a negative result to bury: if OUC-CGE's engagement labels are session-level
(e.g., a whole recording session tends toward one label, and "number of trackable
students" is a property of which table/camera-angle the session used), the mask-only
signal could be entirely a room/session artifact rather than a real engagement
correlate — and there is currently no way to test this from what ships with the
dataset. **Worth a Limitations sentence exactly as flagged:** the absence of any
session identifier means this confound cannot be ruled out or in.

## 4. A1/A2 mask-invariance — now asserted in code, not assumed

`src/ept/tokenization/mask_ops.py::shuffle_entities_per_segment` implements the A2
shuffle (independent per-segment permutation of the entity axis). `tests/test_mask_ops.py`
— 4 tests, run via `pytest`, all passing:
- invariance holds on 50 real val-split masks × 3 seeds (150 checks)
- invariance holds on 200 synthetic random masks at varying densities
- invariance holds when derived from a shuffled `[E,S,D]` feature tensor (what A2
  actually shuffles at model time), not just a raw boolean mask
- **guard test**: confirms the shuffle actually moves entities around (not a
  vacuous no-op that would make the invariance check meaningless)

```
tests/test_mask_ops.py::test_a2_shuffle_preserves_per_segment_presence_count_on_real_masks PASSED
tests/test_mask_ops.py::test_a2_shuffle_preserves_presence_count_synthetic PASSED
tests/test_mask_ops.py::test_a2_shuffle_preserves_presence_count_on_feature_tensor PASSED
tests/test_mask_ops.py::test_a2_shuffle_is_not_a_no_op PASSED
4 passed in 0.15s
```

## Bottom line for reporting

Per instruction, mask-only stays a **permanent row in every results table** — that
doesn't change. What changes is the interpretation to carry forward: report macro-F1
0.373 alongside the **permutation null (0.322 ± 0.022, p≈0.10)**, not just the
majority-class baseline (0.190), so readers aren't misled into thinking this is a
clearly-significant detection artifact. It's a real number worth watching — comparable
in magnitude to plausible seed-to-seed noise — and the per-segment-count diagnosis
plus the missing-session-identifier gap are both worth a line in Limitations regardless
of significance, per the original request.
