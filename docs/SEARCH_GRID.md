# Phase 3 — Recipe search grid

Frozen **before any run**, per the Phase 3 instruction that fairness matters more
than convenience here: tuning on A1 and transferring the recipe to the other
conditions would bias the comparison toward A1. Instead, the **identical** grid
below runs for **every** condition (A0, A1, A2, A3, A4, A5, mask_only) on the
**val split only** — test is never touched in this phase.

## Grid

Not a full factorial (3 lr × 2 wd × 2 dropout × 2 epochs = 24 would exceed the
12-point budget) — a hand-picked sparse grid covering the extremes and a few
interior points, 8 points total. `batch_size=32` is fixed (not swept) across
every point and every condition.

| id | lr | weight_decay | dropout | epochs |
|---|---:|---:|---:|---:|
| r01 | 1e-3 | 0.0  | 0.0 | 30 |
| r02 | 1e-3 | 0.01 | 0.1 | 30 |
| r03 | 3e-4 | 0.0  | 0.0 | 30 |
| r04 | 3e-4 | 0.01 | 0.1 | 30 |
| r05 | 1e-3 | 0.0  | 0.1 | 50 |
| r06 | 3e-4 | 0.01 | 0.0 | 50 |
| r07 | 1e-4 | 0.0  | 0.0 | 50 |
| r08 | 1e-4 | 0.01 | 0.1 | 30 |

## Search protocol

- 7 conditions × 8 grid points × **1 seed (42)** per point = **56 val evaluations**.
  Seeds {42, 1337, 2024} are for the *final* reported numbers (Phase 4, on the
  locked recipe); search itself is single-seed to keep this phase's val-touch
  count bounded and legible — 769 val clips is small enough that val overfitting
  is a live risk (explicitly the reason this number is reported at all), and
  running 3 seeds × 56 points would triple that exposure for a stage whose only
  job is picking one operating point per condition, not producing a reportable
  result.
- Selection criterion per condition: **best single-epoch val macro-F1** across
  all epochs of all 8 grid points (`best_val_macro_f1` in each run's `metrics.json`),
  i.e. early-stopping-by-selection within the grid search itself.
- Every one of the 56 runs gets its own `outputs/<condition>_<recipe_id>_seed42/`
  with `config.yaml` + `metrics.json` (per-epoch train/val loss and macro-F1).
- Frozen at commit (this file, committed alone, before `docs/LOCKED_RECIPE.md`
  or any training run).
