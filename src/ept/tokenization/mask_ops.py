"""Shared entity-slot operations. Currently: the A2 condition's shuffle
(docs/DECISION_RULES.md: "A2 shuffles entity-to-slot assignment independently
per segment, using a per-sample seed derived from the run seed").
"""
import numpy as np


def shuffle_entities_per_segment(x, seed):
    """Independently permute the entity axis (0) within each segment (axis 1).
    Works for a presence mask [E, S] or a feature tensor [E, S, D] — the entity
    axis is permuted per-segment in both cases, same code path."""
    rng = np.random.RandomState(seed)
    e_max, s = x.shape[0], x.shape[1]
    out = np.empty_like(x)
    for seg in range(s):
        perm = rng.permutation(e_max)
        out[:, seg, ...] = x[perm, seg, ...]
    return out
