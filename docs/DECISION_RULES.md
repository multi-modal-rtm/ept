# Pre-registration — EPT

**Committed:** _(fill on commit, before any test evaluation)_

This document is frozen once committed. Amendments append below with a date and reason; nothing above
the amendment line is ever edited.

## Primary endpoint

Test macro-F1 on **OUC-CGE**, mean over seeds {42, 1337, 2024}, 771-clip video-only protocol.

## Hypotheses

- **H1 (persistence).** `A1 (EPT) − A2 (identity-shuffled EPT) ≥ +0.02` macro-F1.
- **H2 (entity structure).** `A1 (EPT) − A0 (grid tokens, matched 64-token budget) ≥ +0.02` macro-F1.

## Decision rule — fires exactly once, on locked test numbers

| Outcome | Branch |
|---|---|
| H1 supported | **(a)** Persistence is the mechanism. Headline comparison: A1 vs A2. Title 1. |
| abs(A1 − A2) < 0.02 and H2 supported | **(b)** Boundary paper: persistence is *not* what helps; entity-localized cropping and token budget are. Headline: A1/A2 vs A0. Title 2. |
| A1 < A2 − 0.02 | **(c)** Hypothesis refuted. Paper pivots to the efficiency Pareto result; the negative finding is the contribution. Title 2. |

## Controls fixed in advance

- **No learned per-slot identity embedding** — it would leak slot ordering and confound A1 vs A2.
- **Slot assignment order is randomized per sample in every condition, including A2.** A1-vs-A2
  therefore isolates temporal identity consistency and nothing else.
- Entity selection: top-`E` tracks by (mean detection confidence x frame coverage), `E_max = 8`,
  identical selection code in all conditions.
- `A2` shuffles entity-to-slot assignment **independently per segment**, using a per-sample seed
  derived from the run seed, so the shuffle is reproducible.

## Locked architecture and protocol

`T = 32`, `S = 8`, `E_max = 8`, `d = 384`, `L = 4`, 6 heads, mlp_ratio 4, pre-norm.
Optimizer recipe: locked at end of Phase 3 into `docs/LOCKED_RECIPE.md`; unchanged thereafter.

## What would invalidate the study

- Tracking QA gate failure (<80% of clips with >=2 tracks at >=50% coverage): the entity abstraction
  is not recoverable from this data. Fall back to region-persistent tokens and **relabel the claim
  in the title and abstract** — do not quietly keep calling it entity-persistent.
- Any condition trained on features other than the shared cache.

## Amendments

_(none)_
