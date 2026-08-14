"""EPT-Former, per docs/PLAN.md §3.2 and .claude/skills/ept-model.md.

Non-negotiables, each checked below (construction-time assertion or, where an
assertion can't prove a structural property, a comment pointing at the test that
does):
  - NO per-slot identity embedding: this file contains no nn.Parameter/nn.Embedding
    indexed by entity-slot position. `_assert_no_identity_embedding` checks by name;
    the stronger, actually-conclusive check is
    tests/test_ept_former.py::test_entity_permutation_invariance, which verifies
    permuting the entity axis of the input leaves logits EXACTLY unchanged — the
    behavioral consequence of having no way to distinguish slot 0 from slot 3.
  - Slot order is randomized per sample in every condition, A2 included: enforced by
    the *data loader* (src/ept/train/dataset.py applies mask_ops.shuffle_entities_per_segment
    for A2), not by this model — the model is architecturally identical for A1 and A2,
    which is the point.
  - Attention respects the presence mask; [ABSENT] embedding for absent pairs:
    `key_padding_mask` throughout; absent (e,s) positions get `self.absent_embedding`
    substituted for their (zero-filled) cached feature before any attention.
  - Backbone parameters never appear in any optimizer parameter group: this module
    never imports or references the DINOv2 backbone at all — it consumes cached
    `.npy` features. `assert_no_backbone_params` is a defensive check callable from
    the training loop before constructing the optimizer.
  - A1 slices the E=16 cache to [:8]: a data-loading concern (src/ept/train/dataset.py),
    not this file's.
"""
import math

import torch
import torch.nn as nn

D_MODEL = 384
N_HEADS = 6
MLP_RATIO = 4
N_BLOCKS = 4
NUM_CLASSES = 3
INPUT_DIM = 1536  # DINOv2 CLS ++ mean-patch, D from cache/MANIFEST.json


def _assert_no_identity_embedding(module):
    for name, _ in module.named_parameters():
        lname = name.lower()
        assert "identity" not in lname and "slot_embed" not in lname, (
            f"found a parameter that looks like a per-slot identity embedding: {name} "
            "— this is explicitly forbidden (docs/DECISION_RULES.md: it would leak "
            "slot ordering and confound A1 vs A2)"
        )


def assert_no_backbone_params(model):
    """Callable from the training loop before optimizer construction. Trivially
    true here (this module never references the backbone) but kept as an explicit,
    named check rather than an implicit assumption."""
    for name, _ in model.named_parameters():
        lname = name.lower()
        assert "dinov2" not in lname and "backbone" not in lname, (
            f"backbone parameter found in model, would leak into optimizer: {name}"
        )


class SinusoidalSegmentEmbedding(nn.Module):
    def __init__(self, d, max_s=32):
        super().__init__()
        pe = torch.zeros(max_s, d)
        position = torch.arange(0, max_s).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)  # [max_s, d]

    def forward(self, s_len):
        return self.pe[:s_len]


def _key_padding_mask_from_presence(presence):
    """presence: bool, True=present. Returns key_padding_mask for
    nn.MultiheadAttention (True=ignore), with an escape hatch for rows that are
    entirely absent (would otherwise make every key masked -> NaN softmax); those
    rows' output is unused downstream (they stay [ABSENT] and get masked out again
    at the readout), so unmasking them fully is safe."""
    key_padding_mask = ~presence
    all_masked = key_padding_mask.all(dim=1)
    if all_masked.any():
        key_padding_mask = key_padding_mask.clone()
        key_padding_mask[all_masked] = False
    return key_padding_mask


class TemporalSocialBlock(nn.Module):
    """temporal attention over s at fixed e -> social attention over e at fixed s -> FFN."""

    def __init__(self, d=D_MODEL, n_heads=N_HEADS, mlp_ratio=MLP_RATIO, dropout=0.0,
                 use_temporal=True, use_social=True):
        super().__init__()
        self.use_temporal = use_temporal
        self.use_social = use_social
        if use_temporal:
            self.norm_t = nn.LayerNorm(d)
            self.attn_t = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        if use_social:
            self.norm_s = nn.LayerNorm(d)
            self.attn_s = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * mlp_ratio, d), nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        # x: [B, E, S, d], mask: [B, E, S] bool (True = present)
        B, E, S, d = x.shape
        if self.use_temporal:
            xt = x.reshape(B * E, S, d)
            kpm = _key_padding_mask_from_presence(mask.reshape(B * E, S))
            normed = self.norm_t(xt)
            attn_out, _ = self.attn_t(normed, normed, normed, key_padding_mask=kpm, need_weights=False)
            x = (xt + attn_out).reshape(B, E, S, d)
        if self.use_social:
            xs = x.transpose(1, 2).reshape(B * S, E, d)
            kpm = _key_padding_mask_from_presence(mask.transpose(1, 2).reshape(B * S, E))
            normed = self.norm_s(xs)
            attn_out, _ = self.attn_s(normed, normed, normed, key_padding_mask=kpm, need_weights=False)
            x = (xs + attn_out).reshape(B, S, E, d).transpose(1, 2)
        x = x + self.ffn(self.norm_ffn(x))
        return x


class EPTFormer(nn.Module):
    """A0 (grid tokens), A1 (full EPT), A3 (temporal-only), A4 (social-only) all
    instantiate this same class — they differ only in `use_temporal`/`use_social`
    and, for A0 vs A1/A3/A4, the token SOURCE (grid vs entity crops), which is a
    data-loading concern. A2 also instantiates this class unchanged; only the
    input ordering differs (shuffled by the data loader)."""

    def __init__(self, input_dim=INPUT_DIM, d=D_MODEL, n_blocks=N_BLOCKS, n_heads=N_HEADS,
                 mlp_ratio=MLP_RATIO, num_classes=NUM_CLASSES, dropout=0.0, s_max=8,
                 use_temporal=True, use_social=True):
        super().__init__()
        assert use_temporal or use_social, "at least one of temporal/social attention must be active"
        self.input_proj = nn.Linear(input_dim, d)
        self.absent_embedding = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.absent_embedding, std=0.02)
        self.segment_pos = SinusoidalSegmentEmbedding(d, max_s=s_max)
        self.blocks = nn.ModuleList([
            TemporalSocialBlock(d, n_heads, mlp_ratio, dropout, use_temporal, use_social)
            for _ in range(n_blocks)
        ])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.cls_token, std=0.02)
        self.cls_norm_q = nn.LayerNorm(d)
        self.cls_norm_kv = nn.LayerNorm(d)
        self.cls_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.readout_norm = nn.LayerNorm(d)
        self.classifier = nn.Linear(d, num_classes)

        _assert_no_identity_embedding(self)
        assert_no_backbone_params(self)

    def forward(self, features, mask):
        """features: [B, E, S, input_dim] float (zero-filled at absent positions,
        per the cache convention — see extract_features.py). mask: [B, E, S] bool,
        True = present."""
        B, E, S, _ = features.shape
        x = self.input_proj(features)
        x = torch.where(
            (~mask).unsqueeze(-1),
            self.absent_embedding.view(1, 1, 1, -1).expand_as(x),
            x,
        )
        x = x + self.segment_pos(S).view(1, 1, S, -1)

        for block in self.blocks:
            x = block(x, mask)

        tokens = x.reshape(B, E * S, -1)
        kpm = _key_padding_mask_from_presence(mask.reshape(B, E * S))
        cls = self.cls_token.expand(B, -1, -1)
        q = self.cls_norm_q(cls)
        kv = self.cls_norm_kv(tokens)
        out, _ = self.cls_attn(q, kv, kv, key_padding_mask=kpm, need_weights=False)
        out = self.readout_norm(out.squeeze(1))
        return self.classifier(out)


class MeanPoolMLP(nn.Module):
    """A5: is attention needed at all? Mean-pool valid (e,s) tokens, then a plain
    MLP — no attention, no positional/[ABSENT] embeddings (pooling already discards
    position; an absent slot contributes nothing rather than a learned filler,
    since it's excluded from the mean rather than attended over)."""

    def __init__(self, input_dim=INPUT_DIM, d=D_MODEL, num_classes=NUM_CLASSES,
                 mlp_ratio=MLP_RATIO, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d * mlp_ratio, num_classes),
        )
        _assert_no_identity_embedding(self)
        assert_no_backbone_params(self)

    def forward(self, features, mask):
        x = self.input_proj(features)  # [B,E,S,d]
        mask_f = mask.unsqueeze(-1).to(x.dtype)
        summed = (x * mask_f).sum(dim=(1, 2))
        count = mask_f.sum(dim=(1, 2)).clamp(min=1.0)
        pooled = summed / count
        return self.mlp(pooled)


class MaskOnlyMLP(nn.Module):
    """The pre-registered mask-only control (docs/DECISION_RULES.md, 2026-08-14
    amendment) as a standing row in the same training loop / metrics.json /
    results_dir infrastructure as every other condition, rather than the
    standalone sklearn script used for the Phase 2 diagnostics. Input: the
    presence mask alone, flattened, no visual features."""

    def __init__(self, e_max=8, s=8, hidden=32, num_classes=NUM_CLASSES, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(e_max * s, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )
        _assert_no_identity_embedding(self)
        assert_no_backbone_params(self)

    def forward(self, mask):
        x = mask.flatten(start_dim=1).to(self.mlp[0].weight.dtype)
        return self.mlp(x)
