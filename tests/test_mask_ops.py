"""A1/A2 mask-invariance: the A2 shuffle permutes WHICH entity occupies each
slot within a segment, but must never change HOW MANY entities are present in
that segment (per-segment presence count) — that invariant is what lets A1-vs-A2
isolate temporal identity consistency and nothing else (docs/DECISION_RULES.md).
Asserted here, not assumed.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from ept.tokenization.mask_ops import shuffle_entities_per_segment


def test_a2_shuffle_preserves_per_segment_presence_count_on_real_masks():
    """Load real presence masks from the tracking cache (if available) and
    verify the invariant holds on actual data, not just synthetic arrays."""
    import glob
    import json

    cache = "/home/devops/ept/cache/tracks/val"
    files = sorted(glob.glob(os.path.join(cache, "*.json")))[:50]
    if not files:
        pytest.skip("no tracking cache available to test against")

    e_max, s, t = 8, 8, 32
    for fp in files:
        with open(fp) as f:
            clip = json.load(f)
        n_total = clip["n_frames_grabbed"]
        track_frames = {}
        for frame in clip["frames"]:
            for det in frame["detections"]:
                track_frames.setdefault(det["track_id"], []).append(
                    (frame["frame_pos"], det["confidence"] or 0.0)
                )
        scored = []
        for tid, entries in track_frames.items():
            coverage = len(entries) / n_total if n_total else 0.0
            mean_conf = np.mean([c for _, c in entries])
            scored.append((mean_conf * coverage, tid, entries))
        scored.sort(key=lambda x: x[0], reverse=True)
        mask = np.zeros((e_max, s), dtype=bool)
        for e, (_, tid, entries) in enumerate(scored[:e_max]):
            for frame_pos, _ in entries:
                seg = min(int(frame_pos // (t / s)), s - 1)
                mask[e, seg] = True

        before = mask.sum(axis=0)
        for seed in [42, 1337, 2024]:
            shuffled = shuffle_entities_per_segment(mask, seed)
            after = shuffled.sum(axis=0)
            assert np.array_equal(before, after), (
                f"{clip['clip_id']} seed={seed}: per-segment presence count changed "
                f"under A2 shuffle: before={before}, after={after}"
            )
            # entity axis must actually be a permutation, not e.g. accidentally zeroed
            assert shuffled.sum() == mask.sum()


def test_a2_shuffle_preserves_presence_count_synthetic():
    """Synthetic random masks, many trials, to cover cases the real cache might
    not (e.g. entities present in every segment, or none)."""
    rng = np.random.RandomState(0)
    e_max, s = 8, 8
    for trial in range(200):
        mask = rng.random((e_max, s)) < rng.uniform(0.1, 0.9)
        before = mask.sum(axis=0)
        shuffled = shuffle_entities_per_segment(mask, seed=trial)
        after = shuffled.sum(axis=0)
        assert np.array_equal(before, after)


def test_a2_shuffle_preserves_presence_count_on_feature_tensor():
    """Same invariant on a [E, S, D] feature tensor (what A2 actually shuffles
    at model time) — mask presence must still line up when derived from it."""
    rng = np.random.RandomState(1)
    e_max, s, d = 8, 8, 16
    presence = rng.random((e_max, s)) < 0.6
    feat = rng.standard_normal((e_max, s, d)) * presence[..., None]

    shuffled_feat = shuffle_entities_per_segment(feat, seed=7)
    before_mask = (feat != 0).any(axis=-1)
    after_mask = (shuffled_feat != 0).any(axis=-1)
    assert np.array_equal(before_mask.sum(axis=0), after_mask.sum(axis=0))


def test_a2_shuffle_is_not_a_no_op():
    """Guard against a vacuous invariant: the shuffle must actually move entities
    around (for E_max > 1 and enough distinct assignments), or the count-invariance
    test above would trivially pass without testing anything."""
    rng = np.random.RandomState(2)
    e_max, s = 8, 8
    mask = rng.random((e_max, s)) < 0.5
    changed = False
    for seed in range(20):
        shuffled = shuffle_entities_per_segment(mask, seed=seed)
        if not np.array_equal(mask, shuffled):
            changed = True
            break
    assert changed, "shuffle never changed the mask across 20 seeds — suspicious"


if __name__ == "__main__":
    test_a2_shuffle_preserves_per_segment_presence_count_on_real_masks()
    test_a2_shuffle_preserves_presence_count_synthetic()
    test_a2_shuffle_preserves_presence_count_on_feature_tensor()
    test_a2_shuffle_is_not_a_no_op()
    print("all mask_ops tests PASSED")
