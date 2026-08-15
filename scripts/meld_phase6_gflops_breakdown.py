"""Re-analysis only: decomposes total per-clip compute into two correctly-
named categories, per the corrected framing (both are attention-based --
the distinction is what's being attended over, not "attention vs no
attention"):

  - "encoder_gflops": the per-crop DINOv2 pass. DINOv2 is itself a Vision
    Transformer -- this is attention too, just applied to a much larger
    token count (~257 patch tokens per 224x224 crop) and run once per
    detected/gridded crop.
  - "fusion_attention_gflops": EPT-Former's own temporal/social/CLS-readout
    multi-head attention, over the small already-encoded entity-token set
    (E*S <= 32 tokens total).
  - "head_gflops": EPT-Former's non-attention compute (input_proj, FFN,
    norms, classifier).

The claim this decomposition supports is that COST LIVES IN ENCODING, NOT
FUSION -- not that "attention is cheap." Fusion attention is cheap because
it operates over a handful of tokens; encoder attention is expensive
because it operates over hundreds of patch tokens per crop, repeated once
per crop. Conflating the two into one "attention" bucket obscured this.

No model is retrained; no test data is touched -- pure re-profiling of the
already-locked architecture (outputs/meld_phase6_efficiency_combined.json,
from the committed Phase 6 efficiency run) via thop's per-submodule MAC
breakdown (ret_layer_info=True). thop DOES have a registered hook for
nn.MultiheadAttention (verified: thop.profile.register_hooks), so the
earlier combined total was not silently missing the attention computation
-- this script only re-attributes an already-correct total, it does not
correct a prior undercount.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from thop import profile

from ept.model.ept_former import EPTFormer, MeanPoolMLP, MaskOnlyMLP
from ept.tokenization.extract_features_meld import S as MELD_S

REPO_ROOT = "/home/devops/ept"

LOCKED_RECIPE_E_S = {
    "A0": (8, MELD_S), "A1": (6, MELD_S), "A2": (6, MELD_S), "A3": (6, MELD_S),
    "A4": (6, MELD_S), "A5": (6, MELD_S), "mask_only": (6, MELD_S),
}
BUDGET_E_GRID = [1, 2, 4, 6, 8]
BUDGET_S_GRID = [2, 4]

FUSION_ATTN_LEAF_NAMES = {"attn_t", "attn_s", "cls_attn"}


def build(condition, e, s):
    if condition == "A5":
        return MeanPoolMLP(dropout=0.0), False
    if condition == "mask_only":
        return MaskOnlyMLP(e_max=e, s=s, dropout=0.0), False
    use_temporal = condition != "A4"
    use_social = condition != "A3"
    return EPTFormer(dropout=0.0, use_temporal=use_temporal, use_social=use_social, s_max=s), True


def sum_fusion_attention_macs(tree, path=""):
    """Walks thop's ret_layer_info tree; returns (fusion_attention_macs,
    total_macs) -- total_macs re-derived by summing all leaves, as a
    self-check against the top-level number thop itself reports."""
    attn_total = 0.0
    leaf_total = 0.0
    for name, (macs, params, sub) in tree.items():
        is_attn_leaf = name in FUSION_ATTN_LEAF_NAMES
        if sub:  # has children -- descend; thop's non-leaf MACs already equal children's sum
            child_attn, child_total = sum_fusion_attention_macs(sub, path + "/" + name)
            if is_attn_leaf:
                # attn_t/attn_s/cls_attn are themselves leaves w.r.t. attention
                # accounting even though thop lists a (zero-MAC) out_proj child;
                # count the parent's own MACs, not double-count via children.
                attn_total += macs
                leaf_total += macs
            else:
                attn_total += child_attn
                leaf_total += child_total
        else:
            leaf_total += macs
            if is_attn_leaf:
                attn_total += macs
    return attn_total, leaf_total


def profile_condition(condition, e, s):
    model, has_attn = build(condition, e, s)
    if condition == "mask_only":
        mask = torch.ones(1, e, s, dtype=torch.bool)
        macs, params, tree = profile(model, inputs=(mask,), ret_layer_info=True, verbose=False)
    else:
        feat = torch.randn(1, e, s, 1536)
        mask = torch.ones(1, e, s, dtype=torch.bool)
        macs, params, tree = profile(model, inputs=(feat, mask), ret_layer_info=True, verbose=False)

    attn_macs, leaf_total_macs = sum_fusion_attention_macs(tree)
    assert abs(leaf_total_macs - macs) / max(macs, 1) < 1e-6, (
        f"{condition} e={e} s={s}: leaf-summed MACs {leaf_total_macs} != thop top-level MACs {macs}"
    )
    head_macs = macs - attn_macs
    return {
        "fusion_attention_gflops": 2 * attn_macs / 1e9,
        "head_gflops": 2 * head_macs / 1e9,
        "fusion_stack_total_gflops": 2 * macs / 1e9,
        "params": int(params),
    }


def main():
    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_efficiency_combined.json")) as f:
        eff = json.load(f)
    encoder_gflops_per_crop = eff["backbone_gflops_per_crop"]
    cond_by_name = {r["condition"]: r for r in eff["condition_rows"]}
    budget_by_es = {(r["e"], r["s"]): r for r in eff["budget_sweep_rows"]}

    print("=== per-condition GFLOPs breakdown: encoder (DINOv2) / fusion attention / head ===")
    condition_rows = []
    for condition, (e, s) in LOCKED_RECIPE_E_S.items():
        d = profile_condition(condition, e, s)
        encoder = cond_by_name[condition]["backbone_gflops_per_clip"]
        total = encoder + d["fusion_stack_total_gflops"]
        attn_pct_of_total = 100 * d["fusion_attention_gflops"] / total if total > 0 else 0.0
        attn_pct_of_fusion_stack = (100 * d["fusion_attention_gflops"] / d["fusion_stack_total_gflops"]
                                     if d["fusion_stack_total_gflops"] > 0 else 0.0)
        row = {
            "condition": condition, "e": e, "s": s,
            "encoder_gflops": encoder, "fusion_attention_gflops": d["fusion_attention_gflops"],
            "head_gflops": d["head_gflops"], "total_gflops": total,
            "fusion_attention_pct_of_grand_total": attn_pct_of_total,
            "fusion_attention_pct_of_fusion_stack": attn_pct_of_fusion_stack,
        }
        condition_rows.append(row)
        print(f"  {condition}: encoder={encoder:.1f} fusion_attn={d['fusion_attention_gflops']:.4f} "
              f"head={d['head_gflops']:.4f} total={total:.1f} "
              f"fusion_attn%_of_total={attn_pct_of_total:.4f}% "
              f"fusion_attn%_of_fusion_stack={attn_pct_of_fusion_stack:.2f}%")

    print("\n=== budget-sweep GFLOPs breakdown (A1 architecture) ===")
    budget_rows = []
    for e in BUDGET_E_GRID:
        for s in BUDGET_S_GRID:
            d = profile_condition("A1", e, s)
            encoder = budget_by_es[(e, s)]["avg_raw_crops_per_clip"] * encoder_gflops_per_crop
            total = encoder + d["fusion_stack_total_gflops"]
            attn_pct = 100 * d["fusion_attention_gflops"] / total if total > 0 else 0.0
            row = {"e": e, "s": s, "encoder_gflops": encoder, "fusion_attention_gflops": d["fusion_attention_gflops"],
                   "head_gflops": d["head_gflops"], "total_gflops": total,
                   "fusion_attention_pct_of_grand_total": attn_pct}
            budget_rows.append(row)
            print(f"  E={e} S={s}: encoder={encoder:.1f} fusion_attn={d['fusion_attention_gflops']:.4f} "
                  f"head={d['head_gflops']:.4f} total={total:.1f} fusion_attn%={attn_pct:.4f}%")

    # --- encoder-reduction finding: a positive result, not just a cost table ---
    a0 = next(r for r in condition_rows if r["condition"] == "A0")
    a1 = next(r for r in condition_rows if r["condition"] == "A1")
    encoder_reduction_ratio = a0["encoder_gflops"] / a1["encoder_gflops"]
    print(f"\n=== encoder-reduction finding ===")
    print(f"A0 (grid) encoder: {a0['encoder_gflops']:.1f} GFLOPs/clip")
    print(f"A1 (entity crop) encoder: {a1['encoder_gflops']:.1f} GFLOPs/clip")
    print(f"reduction: {encoder_reduction_ratio:.2f}x")
    print("(paired with the macro-F1 improvement -- see results/summary.csv and "
          "scripts/meld_phase6_e2e_breakdown.json for the combined statement)")

    out = {"encoder_gflops_per_crop": encoder_gflops_per_crop,
           "condition_rows": condition_rows, "budget_sweep_rows": budget_rows,
           "encoder_reduction_a0_to_a1": encoder_reduction_ratio}
    out_path = os.path.join(REPO_ROOT, "outputs", "meld_phase6_gflops_breakdown.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
