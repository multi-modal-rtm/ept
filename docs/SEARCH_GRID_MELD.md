# MELD Phase 3 — Recipe search grid

Frozen **before any run**, same discipline as `docs/SEARCH_GRID.md` (OUC-CGE): the
**identical** grid below runs for **every** condition (A0, A1, A2, A3, A4, A5, mask_only)
on the **dev split only** — test is never touched in this phase. Per the task
instruction, the grid stays at 8 points and is not expanded, even though MELD's dev
split (1108 clips) is not smaller than OUC-CGE's val split (769 clips) — the premise
that motivated keeping the grid small still holds for the same underlying reason
(bounding the number of dev-touches at a stage whose only job is picking one
operating point per condition, not producing a reportable result), independent of
which split happens to be larger.

## Grid

Identical to `docs/SEARCH_GRID.md`'s 8 points — reused verbatim, not re-tuned for
MELD, per the same "one frozen grid for every condition" discipline. `batch_size=32`
fixed (not swept) across every point and every condition.

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

- 7 conditions × 8 grid points × **1 seed (42)** per point = **56 dev evaluations**.
  Seeds `{42, 1337, 2024, 7, 31337}` (extended per the 2026-08-15 `DECISION_RULES.md`
  amendment) are for the *final* reported numbers only; search itself stays
  single-seed for the same reason as OUC-CGE's search — bounding dev-touch count at
  a stage that only picks one operating point per condition.
- Selection criterion per condition: **best single-epoch dev macro-F1** across all
  epochs of all 8 grid points (`best_dev_macro_f1` in each run's `metrics.json`).
- Every one of the 56 runs gets its own `outputs/<run_id>/` with `config.yaml` +
  `metrics.json` (per-epoch train/dev loss and macro-F1).
- The trivial-feature probe floor (`0.4155`, `DECISION_RULES.md` 2026-08-14) is
  reported as a standing row alongside these results — any condition/recipe that
  fails to beat it is flagged, not silently reported as if it were informative.
- Frozen at commit (this file, committed alone, before `docs/LOCKED_RECIPE_MELD.md`
  or any training run).
