"""Behavioral proof of "no per-slot identity embedding": permuting the entity
axis of the input (with the mask permuted correspondingly) must leave EPTFormer's
output logits exactly unchanged. If the model could distinguish slot 0 from slot
3 by anything other than content, this would fail — that's precisely the leak
docs/DECISION_RULES.md forbids.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from ept.model.ept_former import EPTFormer, MeanPoolMLP, MaskOnlyMLP, assert_no_backbone_params


def test_entity_permutation_invariance_full():
    torch.manual_seed(0)
    model = EPTFormer(input_dim=32, d=16, n_blocks=2, n_heads=2, s_max=8).eval()
    B, E, S, Din = 4, 8, 8, 32
    feat = torch.randn(B, E, S, Din)
    mask = torch.rand(B, E, S) < 0.6

    with torch.no_grad():
        out1 = model(feat, mask)
        perm = torch.randperm(E)
        out2 = model(feat[:, perm], mask[:, perm])

    assert torch.allclose(out1, out2, atol=1e-5), (
        f"entity permutation changed output — max diff {(out1 - out2).abs().max().item()}"
    )


def test_entity_permutation_invariance_temporal_only():
    torch.manual_seed(1)
    model = EPTFormer(input_dim=32, d=16, n_blocks=2, n_heads=2, s_max=8,
                       use_temporal=True, use_social=False).eval()
    B, E, S, Din = 3, 8, 8, 32
    feat = torch.randn(B, E, S, Din)
    mask = torch.rand(B, E, S) < 0.6
    with torch.no_grad():
        out1 = model(feat, mask)
        perm = torch.randperm(E)
        out2 = model(feat[:, perm], mask[:, perm])
    assert torch.allclose(out1, out2, atol=1e-5)


def test_entity_permutation_invariance_social_only():
    torch.manual_seed(2)
    model = EPTFormer(input_dim=32, d=16, n_blocks=2, n_heads=2, s_max=8,
                       use_temporal=False, use_social=True).eval()
    B, E, S, Din = 3, 8, 8, 32
    feat = torch.randn(B, E, S, Din)
    mask = torch.rand(B, E, S) < 0.6
    with torch.no_grad():
        out1 = model(feat, mask)
        perm = torch.randperm(E)
        out2 = model(feat[:, perm], mask[:, perm])
    assert torch.allclose(out1, out2, atol=1e-5)


def test_meanpoolmlp_entity_permutation_invariance():
    torch.manual_seed(3)
    model = MeanPoolMLP(input_dim=32, d=16).eval()
    B, E, S, Din = 4, 8, 8, 32
    feat = torch.randn(B, E, S, Din)
    mask = torch.rand(B, E, S) < 0.6
    with torch.no_grad():
        out1 = model(feat, mask)
        perm = torch.randperm(E)
        out2 = model(feat[:, perm], mask[:, perm])
    assert torch.allclose(out1, out2, atol=1e-5)


def test_forward_shapes():
    model = EPTFormer(input_dim=1536, d=384, n_blocks=4, n_heads=6, s_max=8)
    B, E, S = 5, 8, 8
    feat = torch.randn(B, E, S, 1536)
    mask = torch.rand(B, E, S) < 0.7
    out = model(feat, mask)
    assert out.shape == (5, 3)

    mp = MeanPoolMLP(input_dim=1536, d=384)
    out2 = mp(feat, mask)
    assert out2.shape == (5, 3)

    mo = MaskOnlyMLP(e_max=8, s=8)
    out3 = mo(mask)
    assert out3.shape == (5, 3)


def test_grid_always_present_no_absent_used():
    """A0 grid tokens are always-present; feeding an all-True mask should still
    work cleanly (no key_padding_mask row fully masked)."""
    model = EPTFormer(input_dim=1536, d=384, n_blocks=4, n_heads=6, s_max=8)
    B, E, S = 3, 8, 8
    feat = torch.randn(B, E, S, 1536)
    mask = torch.ones(B, E, S, dtype=torch.bool)
    out = model(feat, mask)
    assert out.shape == (3, 3)
    assert torch.isfinite(out).all()


def test_fully_absent_entity_row_no_nan():
    """An entity slot with zero valid segments (can happen for padding slots when
    fewer than E_max real tracks exist) must not produce NaN."""
    model = EPTFormer(input_dim=1536, d=384, n_blocks=4, n_heads=6, s_max=8)
    B, E, S = 2, 8, 8
    feat = torch.randn(B, E, S, 1536)
    mask = torch.ones(B, E, S, dtype=torch.bool)
    mask[:, -1, :] = False  # last entity slot entirely absent (padding)
    out = model(feat, mask)
    assert torch.isfinite(out).all()


def test_assert_no_backbone_params_passes_for_all_models():
    assert_no_backbone_params(EPTFormer(input_dim=32, d=16, n_blocks=1, n_heads=2))
    assert_no_backbone_params(MeanPoolMLP(input_dim=32, d=16))
    assert_no_backbone_params(MaskOnlyMLP())


if __name__ == "__main__":
    test_entity_permutation_invariance_full()
    test_entity_permutation_invariance_temporal_only()
    test_entity_permutation_invariance_social_only()
    test_meanpoolmlp_entity_permutation_invariance()
    test_forward_shapes()
    test_grid_always_present_no_absent_used()
    test_fully_absent_entity_row_no_nan()
    test_assert_no_backbone_params_passes_for_all_models()
    print("all ept_former tests PASSED")
