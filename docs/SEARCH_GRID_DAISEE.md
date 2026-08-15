# DAiSEE H2 track — Recipe search grid

Frozen **before any run**, same discipline as `docs/SEARCH_GRID.md` (OUC-CGE) and
`docs/SEARCH_GRID_MELD.md`: the **identical** grid below runs for **both** conditions this
track uses (A0, A1 — A2/A3/A4/A5/mask_only are out of scope per
`docs/DECISION_RULES_DAISEE.md`'s H2-only scope) on the **val split only** — test is never
touched in this phase.

## Grid

Reused verbatim from `docs/SEARCH_GRID.md` / `docs/SEARCH_GRID_MELD.md`, not re-tuned for
DAiSEE — same rationale both prior tracks used: a fixed, already-validated sparse grid applied
identically across datasets avoids inflating the effective search budget with a fresh
per-dataset hyperparameter hunt, and keeps the discipline ("one frozen grid, same for every
condition") the whole point of this file. `batch_size=32` fixed (not swept).

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

- 2 conditions (A0, A1) × 8 grid points × **1 seed (42)** per point = **16 runs, 600 val
  evaluations** (5 recipes × 30 epochs + 3 recipes × 50 epochs = 300 epochs/condition × 2
  conditions = 600, exact). Seeds `{42, 1337, 2024, 7, 31337}` are for the *final* Phase-4 test
  numbers only; search stays single-seed for the same reason as every prior track — bounding
  val-touch count at a stage that only picks one operating point per condition.
- Selection criterion per condition: **best single-epoch val macro-F1** across all epochs of all
  8 grid points (`best_val_macro_f1` in each run's `metrics.json`).
- Per-class F1 is also recorded at every epoch (not just macro-F1), so the class-imbalance risk
  `docs/DECISION_RULES_DAISEE.md` flags in advance for the eventual test evaluation (class 0: 4
  test clips) can be previewed against real val numbers before Phase 4 runs, not discovered
  after.
- Every run gets its own `outputs/<run_id>/` with `config.yaml` + `metrics.json`.
- `E=1` (A1), `S=8`, `NUM_CLASSES=4` throughout, per `docs/DECISION_RULES_DAISEE.md`'s locked
  architecture section — unchanged by this file.
- Frozen at commit (this file, committed alone, before `docs/LOCKED_RECIPE_DAISEE.md` or any
  training run).
