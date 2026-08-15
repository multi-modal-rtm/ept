"""Re-analysis only: decomposes the already-measured attention-stack GFLOPs
(outputs/meld_phase6_efficiency_combined.json, from the committed Phase 6
efficiency run) into attention-proper (temporal/social/CLS-readout MHA)
vs. head (input_proj + FFN + norms + classifier) using thop's per-submodule
MAC breakdown (ret_layer_info=True). No model is retrained; no test data is
touched; this only profiles the existing architecture at each condition's
already-locked geometry.

thop DOES have a registered hook for nn.MultiheadAttention (verified: it's
in thop.profile.register_hooks), so the earlier attention_stack_gflops
totals were not silently missing the attention computation -- this script
only re-attributes that already-correct total across two sub-categories,
it does not correct a prior undercount.
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

ATTN_LEAF_NAMES = {"attn_t", "attn_s", "cls_attn"}


def build(condition, e, s):
    if condition == "A5":
        return MeanPoolMLP(dropout=0.0), False
    if condition == "mask_only":
        return MaskOnlyMLP(e_max=e, s=s, dropout=0.0), False
    use_temporal = condition != "A4"
    use_social = condition != "A3"
    return EPTFormer(dropout=0.0, use_temporal=use_temporal, use_social=use_social, s_max=s), True


def sum_attention_macs(tree, path=""):
    """Walks thop's ret_layer_info tree; returns (attention_macs, total_macs)
    -- total_macs re-derived by summing all leaves, as a self-check against
    the top-level number thop itself reports."""
    attn_total = 0.0
    leaf_total = 0.0
    for name, (macs, params, sub) in tree.items():
        is_attn_leaf = name in ATTN_LEAF_NAMES
        if sub:  # has children -- descend; thop's non-leaf MACs already equal children's sum
            child_attn, child_total = sum_attention_macs(sub, path + "/" + name)
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

    attn_macs, leaf_total_macs = sum_attention_macs(tree)
    assert abs(leaf_total_macs - macs) / max(macs, 1) < 1e-6, (
        f"{condition} e={e} s={s}: leaf-summed MACs {leaf_total_macs} != thop top-level MACs {macs}"
    )
    head_macs = macs - attn_macs
    return {
        "attention_gflops": 2 * attn_macs / 1e9,
        "head_gflops": 2 * head_macs / 1e9,
        "attention_stack_total_gflops": 2 * macs / 1e9,
        "params": int(params),
    }


def main():
    with open(os.path.join(REPO_ROOT, "outputs", "meld_phase6_efficiency_combined.json")) as f:
        eff = json.load(f)
    backbone_gflops_per_crop = eff["backbone_gflops_per_crop"]
    cond_by_name = {r["condition"]: r for r in eff["condition_rows"]}
    budget_by_es = {(r["e"], r["s"]): r for r in eff["budget_sweep_rows"]}

    print("=== per-condition GFLOPs breakdown: backbone / attention / head ===")
    condition_rows = []
    for condition, (e, s) in LOCKED_RECIPE_E_S.items():
        d = profile_condition(condition, e, s)
        backbone = cond_by_name[condition]["backbone_gflops_per_clip"]
        total = backbone + d["attention_stack_total_gflops"]
        attn_pct_of_total = 100 * d["attention_gflops"] / total if total > 0 else 0.0
        attn_pct_of_model = (100 * d["attention_gflops"] / d["attention_stack_total_gflops"]
                              if d["attention_stack_total_gflops"] > 0 else 0.0)
        row = {
            "condition": condition, "e": e, "s": s,
            "backbone_gflops": backbone, "attention_gflops": d["attention_gflops"],
            "head_gflops": d["head_gflops"], "total_gflops": total,
            "attention_pct_of_grand_total": attn_pct_of_total,
            "attention_pct_of_model_only": attn_pct_of_model,
        }
        condition_rows.append(row)
        print(f"  {condition}: backbone={backbone:.1f} attn={d['attention_gflops']:.4f} "
              f"head={d['head_gflops']:.4f} total={total:.1f} "
              f"attn%_of_total={attn_pct_of_total:.4f}% attn%_of_model={attn_pct_of_model:.2f}%")

    print("\n=== budget-sweep GFLOPs breakdown (A1 architecture) ===")
    budget_rows = []
    for e in BUDGET_E_GRID:
        for s in BUDGET_S_GRID:
            d = profile_condition("A1", e, s)
            backbone = budget_by_es[(e, s)]["avg_raw_crops_per_clip"] * backbone_gflops_per_crop
            total = backbone + d["attention_stack_total_gflops"]
            attn_pct = 100 * d["attention_gflops"] / total if total > 0 else 0.0
            row = {"e": e, "s": s, "backbone_gflops": backbone, "attention_gflops": d["attention_gflops"],
                   "head_gflops": d["head_gflops"], "total_gflops": total, "attention_pct_of_grand_total": attn_pct}
            budget_rows.append(row)
            print(f"  E={e} S={s}: backbone={backbone:.1f} attn={d['attention_gflops']:.4f} "
                  f"head={d['head_gflops']:.4f} total={total:.1f} attn%={attn_pct:.4f}%")

    out = {"backbone_gflops_per_crop": backbone_gflops_per_crop,
           "condition_rows": condition_rows, "budget_sweep_rows": budget_rows}
    out_path = os.path.join(REPO_ROOT, "outputs", "meld_phase6_gflops_breakdown.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
