# Skill: ept-model

Architecture and training.

- Block order: temporal attention over `s` at fixed `e`, then social attention over `e` at
  fixed `s`, then FFN. Pre-norm, d=384, L=4, 6 heads, mlp_ratio 4.
- Additive embeddings: sinusoidal segment position, `[ABSENT]` flag. **No per-slot identity
  embedding** — it leaks slot ordering and destroys the A1-vs-A2 contrast.
- Slot assignment order is randomized per sample in every condition, A2 included.
- A2 shuffles entity-to-slot assignment independently per segment, seeded from the run seed.
- Readout: learned `[CLS]` cross-attending over valid tokens only (respect the mask).
- The backbone is frozen and never appears in the optimizer's parameter groups. Assert this
  at construction time.
- Recipe search happens on validation only and stops at the Phase 3 gate. After
  `docs/LOCKED_RECIPE.md` is committed, hyperparameters are read-only.
