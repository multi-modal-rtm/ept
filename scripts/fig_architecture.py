"""Figure 1: EPT-Former architecture schematic. Pure diagram, no data --
describes src/ept/model/ept_former.py's actual forward pass structure
exactly (input_proj -> [ABSENT] substitution -> segment positional embedding
-> L x TemporalSocialBlock -> CLS cross-attention readout -> classifier),
cross-checked against the module docstring and forward() method, not drawn
from memory of an earlier, possibly-stale description.
"""
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = "/home/devops/ept"
OUT_PATH = os.path.join(REPO_ROOT, "paper", "figures", "fig1_architecture.pdf")

GAP = 0.35  # vertical gap between a box's bottom and the next arrow's start


def box(ax, cursor_top, w, h, text, fc="#e8eef7", ec="#1f4e8c", fontsize=5.2, ls="-"):
    """Places a box with its TOP at cursor_top, centered at x=5. Returns new
    cursor (the box's bottom)."""
    x = 5 - w / 2
    y = cursor_top - h
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                        fc=fc, ec=ec, lw=0.7, ls=ls, zorder=3)
    ax.add_patch(p)
    ax.text(5, y + h / 2, text, ha="center", va="center", fontsize=fontsize, zorder=4)
    return y  # new cursor = box bottom


def arrow(ax, cursor_top):
    """Draws a short downward arrow starting at cursor_top, returns the new
    cursor (arrow's tip, i.e. cursor_top - GAP)."""
    y1 = cursor_top - GAP
    ax.annotate("", xy=(5, y1), xytext=(5, cursor_top),
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#333333", shrinkA=0, shrinkB=0))
    return y1


def main():
    fig, ax = plt.subplots(figsize=(3.3, 5.0))
    ax.set_xlim(0, 10)

    cursor = 24.5
    ax.text(5, cursor + 0.55, "EPT-Former", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.text(5, cursor + 0.10, "(A0-A4 share this class; A3: social off, A4: temporal off)",
            ha="center", va="bottom", fontsize=4.6, style="italic")

    cursor = box(ax, cursor, 8.6, 1.05,
                 "video: E entity tracks x S segments\n(top-E by conf.$\\times$coverage; A0 = fixed grid)",
                 fc="#f5f5f5", ec="#555555", fontsize=4.8)
    cursor = arrow(ax, cursor)
    cursor = box(ax, cursor, 8.6, 0.95, "frozen DINOv2 backbone (per crop)\nCLS $\\oplus$ mean-patch, D=1536",
                 fontsize=5)
    cursor = arrow(ax, cursor)
    cursor = box(ax, cursor, 8.6, 0.95, "cached features [E,S,D] + presence mask [E,S]",
                 fc="#f5f5f5", ec="#555555", fontsize=4.8)
    cursor = arrow(ax, cursor)
    cursor = box(ax, cursor, 8.6, 0.85, "input_proj: Linear(1536 $\\to$ 384)", fontsize=5)
    cursor = arrow(ax, cursor)
    cursor = box(ax, cursor, 8.6, 0.95,
                 "absent (e,s) $\\to$ learned [ABSENT] embedding\n(A2 only: shuffle entity$\\to$slot per segment)",
                 fontsize=4.5)
    cursor = arrow(ax, cursor)
    cursor = box(ax, cursor, 8.6, 0.85, "+ sinusoidal segment position embedding", fontsize=5)
    cursor = arrow(ax, cursor)

    # --- L=4 repeated block ---
    block_pad = 0.20
    inner_w = 7.9
    row_h = 0.75
    row_gap = 0.30
    block_h = 3 * row_h + 2 * row_gap + 2 * block_pad
    block_top = cursor
    block_bottom = block_top - block_h
    block = FancyBboxPatch((5 - (inner_w + 0.5) / 2, block_bottom), inner_w + 0.5, block_h,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            fc="none", ec="#8c1f1f", lw=1.0, ls="--", zorder=2)
    ax.add_patch(block)
    ax.text(5 + (inner_w + 0.5) / 2 - 0.15, block_top - 0.05, "$\\times L{=}4$",
            fontsize=5.5, color="#8c1f1f", ha="right", va="top")

    row_cursor = block_top - block_pad
    row_cursor = box(ax, row_cursor, inner_w, row_h,
                      "temporal attn. (over S, per E,\nLayerNorm+MHA+resid.)",
                      fc="#fdeeee", ec="#8c1f1f", fontsize=4.4)
    row_cursor -= row_gap
    ax.annotate("", xy=(5, row_cursor), xytext=(5, row_cursor + row_gap),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#8c1f1f", shrinkA=0, shrinkB=0))
    row_cursor = box(ax, row_cursor, inner_w, row_h,
                      "social attn. (over E, per S,\nLayerNorm+MHA+resid.)",
                      fc="#fdeeee", ec="#8c1f1f", fontsize=4.4)
    row_cursor -= row_gap
    ax.annotate("", xy=(5, row_cursor), xytext=(5, row_cursor + row_gap),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#8c1f1f", shrinkA=0, shrinkB=0))
    row_cursor = box(ax, row_cursor, inner_w, row_h,
                      "FFN (LayerNorm,\nLinear-GELU-Linear, resid.)",
                      fc="#fdeeee", ec="#8c1f1f", fontsize=4.4)

    cursor = block_bottom
    cursor = arrow(ax, cursor)
    cursor = box(ax, cursor, 8.6, 1.05,
                 "flatten to E$\\times$S tokens; learned CLS query\ncross-attends (key_padding_mask=presence)",
                 fontsize=4.5)
    cursor = arrow(ax, cursor)
    cursor = box(ax, cursor, 8.6, 0.85, "readout LayerNorm $\\to$ Linear(384 $\\to$ 3)", fontsize=5)
    cursor = arrow(ax, cursor)
    cursor = box(ax, cursor, 5.5, 0.85, "sentiment logits", fc="#eaf7ea", ec="#1f6b1f", fontsize=5.2)

    ax.set_ylim(cursor - 0.3, 25.3)
    ax.axis("off")
    fig.tight_layout(pad=0.15)
    fig.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
