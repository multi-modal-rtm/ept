"""Figure 2: OUC-CGE cross-split leakage, top pair thumbnails side by side.
Source images already exist (outputs/dataset_admission/ouccge/pairs/,
rendered by scripts/dataset_admission_lib.py's render_pair_thumbnails during
the Phase 3.5 audit) -- this script only re-composes two of the clearest,
highest-similarity, visually-distinct examples into a clean vector-PDF
figure with proper labels, replacing the raw cv2-drawn overlay text. No new
computation; no data touched beyond reading two existing PNGs.
"""
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = "/home/devops/ept"
PAIRS_DIR = os.path.join(REPO_ROOT, "outputs", "dataset_admission", "ouccge", "pairs")
OUT_PATH = os.path.join(REPO_ROOT, "paper", "figures", "fig2_ouccge_leakage.pdf")

# Two visually distinct examples, both val_to_train, both cosine similarity 1.000000
# exactly (masked-mean DINOv2 clip embedding) -- picked to show the pattern is not
# tied to one physical scene or one label.
EXAMPLES = [
    {"file": "pair_00_high_view1076_vs_high_view1073.png",
     "query": "high/view1076 (val, label=high)", "match": "high/view1073 (train, label=high)"},
    {"file": "pair_02_low_view2598_vs_low_view2603.png",
     "query": "low/view2598 (val, label=low)", "match": "low/view2603 (train, label=low)"},
]


def crop_frames(img):
    """Source PNGs have a black label bar (cv2.putText) at the top -- crop
    it off so this figure's own matplotlib-drawn labels are the only text."""
    bar_h = 42
    content = img[bar_h:, :]
    mid = content.shape[1] // 2
    return content[:, :mid], content[:, mid:]


def main():
    fig_w = 3.3  # ACM sigconf single-column width, inches
    panel_w = fig_w / 2 * 0.94
    frame_aspect = 478 / 853  # h/w, measured from the source crop
    panel_h = panel_w * frame_aspect
    title_h, suptitle_h, caption_h, row_gap = 0.14, 0.22, 0.42, 0.10
    n = len(EXAMPLES)
    fig_h = suptitle_h + n * (title_h + panel_h) + (n - 1) * row_gap + caption_h

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        n, 2, left=0.09, right=0.99,
        top=1 - suptitle_h / fig_h, bottom=caption_h / fig_h,
        hspace=row_gap / panel_h, wspace=0.04,
    )
    for row, ex in enumerate(EXAMPLES):
        img = mpimg.imread(os.path.join(PAIRS_DIR, ex["file"]))
        left, right = crop_frames(img)
        for col, (frame, label) in enumerate([(left, ex["query"]), (right, ex["match"])]):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(frame)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
            ax.set_title(label, fontsize=5.5, pad=2)
            if col == 0:
                ax.set_ylabel("cos. sim. = 1.000000", fontsize=6, labelpad=3)

    fig.suptitle("OUC-CGE: identical frames split across train/val", fontsize=7, y=0.995)
    fig.text(0.5, 0.012,
              "Masked-mean DINOv2 embedding, cosine similarity. Both pairs score exactly 1.000000 --\n"
              "not near-duplicates, the same physical moment assigned to different splits.",
              ha="center", va="bottom", fontsize=5, style="italic")
    fig.savefig(OUT_PATH)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
