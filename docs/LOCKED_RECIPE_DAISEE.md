# Locked recipe — DAiSEE H2 track

Per-condition selection from the frozen grid (`docs/SEARCH_GRID_DAISEE.md`), val split
only, single seed (42) per grid point. **16 runs, 600 val evaluations** (5 recipes × 30
epochs + 3 recipes × 50 epochs = 300 epochs/condition × 2 conditions = 600, exact).
Selection criterion: best single-epoch val macro-F1 anywhere in the run
(`best_val_macro_f1` in each run's `metrics.json`).

**Committed once. After this commit, hyperparameters are read-only** — any change needs
a dated amendment, same discipline as `docs/DECISION_RULES_DAISEE.md`.

## Selected recipe per condition

| Condition | Recipe | lr | weight_decay | dropout | epochs | Val macro-F1 | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 (grid) | r08 | 1e-4 | 0.01 | 0.1 | 30 | 0.2978 | 25 |
| A1 (EPT, primary, E=1) | r06 | 3e-4 | 0.01 | 0.0 | 50 | 0.3438 | 24 |

`batch_size=32` throughout (fixed, not swept). Full per-recipe results for both
conditions: `outputs/daisee_phase3_search_summary.json`.

## Per-class F1 at the selected point (val, class order 0=very-low, 1=low, 2=high, 3=very-high)

| Condition | class 0 | class 1 | class 2 | class 3 |
|---|---:|---:|---:|---:|
| A0 (r08) | 0.129 | 0.066 | 0.571 | 0.425 |
| A1 (r06) | 0.150 | 0.120 | 0.649 | 0.456 |

## Class-imbalance risk: previewed on val, plainly stated in advance

`docs/DECISION_RULES_DAISEE.md` flagged before any run that class 0 has only 4 test
clips and would give macro-F1 "enormous single-item leverage." The val results now
in hand sharpen that from a headcount concern into a demonstrated one: **class 0's F1
is 0.000 in 9 of the 16 grid×condition runs**
%% outputs/daisee_phase3_search_summary.json: computed directly, best_val_per_class_f1['0_verylow']==0.0 count
, and never exceeds 0.150 in any run, despite val having 23 class-0 clips to
calibrate against — over five times the 4 clips test will have. Class 1 shows the
same pattern, slightly less severely (0.000 in 6 of 16 runs).

**We state plainly, before Phase 4 runs, not after:** a class this hard to predict
with 23 validation examples is not going to become reliably predictable with 4 test
examples, and macro-F1's equal per-class weighting means Phase 4's headline number
will be substantially determined by whether the model happens to get 0, 1, 2, 3, or 4
of those 4 class-0 test clips right — a near-discrete, high-variance quantity, not a
continuous performance measure. **Phase 4 will report accuracy and per-class F1
alongside macro-F1, and will not treat macro-F1 alone as sufficient evidence for or
against H2** if class 0's test behavior is the deciding factor in whether the
pre-registered margin clears `EFFECT_FLOOR=0.03` — consistent with
`docs/DECISION_RULES_DAISEE.md`'s existing commitment to report the study as
underpowered where warranted rather than force a metric to carry more than it can
support.

## What this table does not do

This is a val-split model-selection result, not a reportable finding. Per
`docs/DECISION_RULES_DAISEE.md`, H2 (`A1 − A0 ≥ 0.03`) is decided on **test**, with
these locked recipes, mean over 5 seeds — a separate, later, explicitly-logged gate
event. The val gap above (A1 0.3438 vs A0 0.2978, a 0.0460 margin) is not the test
result and is not treated as one; it is flagged here only because reading it as if it
already answered H2 would be exactly the wrong-split mistake this document's
structure exists to prevent.
