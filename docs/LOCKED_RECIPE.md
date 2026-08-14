# Locked recipe — Phase 3

Per-condition selection from the frozen grid (`docs/SEARCH_GRID.md`), val split
only, single seed (42) per grid point. **56 runs, 2100 val evaluations total**
(sum of epochs across all runs — 5 recipes × 30 epochs + 3 recipes × 50 epochs =
300 epochs/condition × 7 conditions = 2100, exact). Selection criterion: best
single-epoch val macro-F1 anywhere in the run (`best_val_macro_f1` in each run's
`metrics.json`).

**Committed once. After this commit, hyperparameters are read-only** — any change
needs a dated amendment, same discipline as `docs/DECISION_RULES.md`.

## Selected recipe per condition

| Condition | Recipe | lr | weight_decay | dropout | epochs | Val macro-F1 | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 (grid) | r07 | 1e-4 | 0.0 | 0.0 | 50 | 0.9918 | 18 |
| A1 (EPT, primary) | r07 | 1e-4 | 0.0 | 0.0 | 50 | 0.9767 | 25 |
| A2 (shuffled) | r07 | 1e-4 | 0.0 | 0.0 | 50 | 0.9762 | 28 |
| A3 (temporal-only) | r08 | 1e-4 | 0.01 | 0.1 | 30 | 0.9710 | 24 |
| A4 (social-only) | r07 | 1e-4 | 0.0 | 0.0 | 50 | 0.9861 | 40 |
| A5 (mean-pool+MLP) | r06 | 3e-4 | 0.01 | 0.0 | 50 | 0.9873 | 45 |
| mask-only | r02 | 1e-3 | 0.01 | 0.1 | 30 | 0.4301 | 28 |

`batch_size=32` throughout (fixed, not swept — see `docs/SEARCH_GRID.md`).

## Pattern across the search

Low learning rate (1e-4, r06–r08) with more epochs (50) consistently won for
every attention-based condition (A0–A4) — high-lr/high-regularization points
(r02, r05) were unstable or slow to converge for these (e.g. A1/r05 best epoch 33
at val_f1 0.56, well below its r07 result of 0.977). A5 and mask-only, being
much shallower (no attention stack), were far more robust to the choice across
the whole grid (A5 ranges 0.972–0.987 across all 8 points; mask-only 0.380–0.430).

## Note for discussion, not a gate criterion

**All of A0–A5 land in a narrow, very high band (0.971–0.992) on val.** This
echoes the BPAVTforSGER fine-tuned reference numbers (A6: TimeSformer 0.992,
VideoMAE 0.988, `docs/PLAN.md` §5) — the task appears close to saturated from
visual content alone at this val-split size, for every tokenization scheme
tried, including A0 (grid, no entity structure) and A2 (identity-shuffled).
Whether this compresses H1/H2's margins (both require ≥0.02 macro-F1 gaps) is a
live risk to watch once locked configs run on **test** in Phase 4 — flagging
now, not attempting to fix by re-opening the grid (that would violate the search
budget this document exists to freeze).

**mask-only's locked-recipe number (0.430) is meaningfully higher than Phase 2's
reported 0.373** (3-seed mean, plain logistic regression, no hyperparameter
search). The standing-row model here is a tuned MLP (hidden=32) selected from 8
grid points, single seed — a different, more capable estimator than Phase 2's
sklearn baseline, not a different underlying phenomenon. Both numbers should be
carried forward with their provenance stated; do not average or conflate them.
Phase 4's 3-seed run on this locked recipe will be the number that actually
enters the results table.
