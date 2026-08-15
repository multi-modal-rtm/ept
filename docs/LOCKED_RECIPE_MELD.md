# Locked recipe — MELD Phase 3

Per-condition selection from the frozen grid (`docs/SEARCH_GRID_MELD.md`), dev
split only, single seed (42) per grid point. **56 runs, 2100 dev evaluations
total** (5 recipes × 30 epochs + 3 recipes × 50 epochs = 300 epochs/condition
× 7 conditions = 2100, exact — identical to OUC-CGE's search by construction,
since it's the same grid and condition count). Selection criterion: best
single-epoch dev macro-F1 anywhere in the run (`best_val_macro_f1` — field
name kept for schema parity with `train.py`'s `metrics.json` — in each run's
`metrics.json`).

**Committed once. After this commit, hyperparameters are read-only** — any
change needs a dated amendment, same discipline as `docs/DECISION_RULES.md`.

Trivial-feature probe floor (`0.4155`, `docs/DECISION_RULES.md` 2026-08-14):
reported as a standing row below, not folded into the table, since it is not
a condition — it did not go through this recipe search at all.

## Selected recipe per condition

| Condition | Recipe | lr | weight_decay | dropout | epochs | Dev macro-F1 | Best epoch | Beats trivial floor |
|---|---|---:|---:|---:|---:|---:|---:|:---:|
| A0 (grid) | r08 | 1e-4 | 0.01 | 0.1 | 30 | 0.4346 | 2 | yes |
| A1 (EPT, primary) | r07 | 1e-4 | 0.0 | 0.0 | 50 | 0.5099 | 8 | yes |
| A2 (shuffled) | r03 | 3e-4 | 0.0 | 0.0 | 30 | 0.5030 | 8 | yes |
| A3 (temporal-only) | r04 | 3e-4 | 0.01 | 0.1 | 30 | 0.5117 | 17 | yes |
| A4 (social-only) | r03 | 3e-4 | 0.0 | 0.0 | 30 | 0.4991 | 8 | yes |
| A5 (mean-pool+MLP) | r07 | 1e-4 | 0.0 | 0.0 | 50 | 0.4932 | 3 | yes |
| mask-only | r05 | 1e-3 | 0.0 | 0.1 | 50 | 0.2491 | 47 | **no** |
| **trivial-feature probe** (standing row, not searched) | — | — | — | — | — | **0.4155** | — | — |

`batch_size=32` throughout (fixed, not swept — see `docs/SEARCH_GRID_MELD.md`).

## Pattern across the search

Every attention-based/pooled condition (A0–A5) clears the trivial-feature
floor at its best grid point; **mask-only does not clear it at any of its 8
points** (range 0.1997–0.2491, best still below 0.4155) — consistent with the
Phase-2-equivalent finding (`outputs/meld_phase1_2_summary/REPORT.md`) that
MELD's presence mask carries no detectable sentiment signal on its own. That
earlier result used a fixed sklearn logistic-regression baseline (3-seed mean
0.2677); this search's tuned-MLP selection (0.2491) is a different, slightly
weaker estimator on the same underlying non-signal, not a contradictory
finding — both numbers should be carried forward with their provenance
stated, not averaged.

Unlike OUC-CGE's search (where all of A0–A5 landed in a narrow high band,
0.971–0.992, and only mask-only separated from the pack), **MELD's A0–A5
recipes cluster in a much lower, tighter band (0.493–0.512)** — the task is
far from saturated at this feature/architecture combination, which is exactly
the more informative regime for H1/H2 to actually discriminate in, rather
than testing a near-ceiling effect where every condition already wins.

No single recipe id dominates across conditions (r03, r04, r07, r08 each win
at least one condition) — the grid's spread was doing real work here, not
just picking the same point every time.

## What this table does not do

This is a dev-split model-selection result, not a reportable finding. Per
`docs/DECISION_RULES.md`, H1 (`A1 − A2 ≥ 0.04`) and H2 (`A1 − A0 ≥ 0.04`) are
decided on **test**, with these locked recipes, mean over 5 seeds — a
separate, later, explicitly-logged gate event. Reading the dev numbers above
as if they already answered H1/H2 (A1's 0.5099 vs A2's 0.5030, a 0.0069 gap,
far under the 0.04 floor) would be exactly the single-seed, wrong-split
mistake this document's structure exists to prevent — flagged here only to
name the trap, not because dev is being treated as evidence.
