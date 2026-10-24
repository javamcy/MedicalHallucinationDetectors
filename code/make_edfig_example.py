#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Data Fig. 5: one item carried through the protocol ladder.

Source of every number below: results/remote_artefacts/cache/rows_vqa_rad_<level>.json
and trained_vqa_rad_<level>.npz, item pid=1705 (VQA-RAD gid vqa_rad_142).
Reproduce with code/recompute_tables.py --example.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.image import imread

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "nmi_analysis_sn", "figures")
IMG = os.path.join(HERE, "figures_src", "vqa_rad_142.png")

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 6.5,
    "axes.linewidth": 0.7, "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
    "pdf.fonttype": 42, "savefig.dpi": 500, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
INK, WARM, GREY = "#12284c", "#b4553a", "#999999"

# level, constraint applied to the negative, negative answer, grounded score, hallucinated score
ROWS = [
    ("L0", "none",           "one",              0.318, 0.983),
    ("L1", "none",           "subarachnoid",     0.627, 1.000),
    ("L2", "length \u00b12",  "axial",            0.976, 1.000),
    ("L3", "format",         "yes",              0.998, 0.000),
    ("L4", "length + format","yes",              0.963, 0.009),
]

def edfig_example():
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 2.55, 1.55], wspace=0.10,
                          left=0.02, right=0.98, top=0.86, bottom=0.16)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(imread(IMG), cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("VQA-RAD CT slice", fontsize=6.5, pad=3)
    ax.text(-0.02, 1.13, "a", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")

    tx = fig.add_subplot(gs[0, 1]); tx.axis("off")
    tx.text(0.00, 1.02, "negative built at each rung", fontsize=6.4, color="#333333",
            transform=tx.transAxes, va="bottom")
    tx.text(0.62, 1.02, "constraint", fontsize=6.4, color="#333333", transform=tx.transAxes, va="bottom")
    for k, (lv, cons, neg, _, _) in enumerate(ROWS):
        y = 0.86 - k * 0.19
        tx.text(0.00, y, lv, fontsize=6.6, transform=tx.transAxes, va="center", fontweight="bold")
        tx.text(0.13, y, "\u201c%s\u201d" % neg, fontsize=6.6, transform=tx.transAxes, va="center",
                style="italic")
        tx.text(0.62, y, cons, fontsize=6.2, transform=tx.transAxes, va="center", color="#555555")
    tx.text(0.13, -0.10, "question: has the small bowel perforated?   expert answer: \u201cno\u201d",
            fontsize=6.4, transform=tx.transAxes, va="top", color="#333333")

    sc = fig.add_subplot(gs[0, 2])
    ys = [4 - k for k in range(len(ROWS))]
    for (lv, cons, neg, g, h), y in zip(ROWS, ys):
        sc.plot([g, h], [y, y], color=GREY, lw=1.0, zorder=1)
        sc.scatter([g], [y], s=22, color=INK, zorder=3)
        sc.scatter([h], [y], s=22, facecolor="white", edgecolor=WARM, lw=1.0, zorder=3)
        ok = h > g
        sc.text(1.06, y, "flagged" if ok else "missed", fontsize=5.9, va="center",
                color="#333333" if ok else WARM, transform=sc.get_yaxis_transform())
    sc.set_yticks(ys); sc.set_yticklabels([r[0] for r in ROWS]); sc.set_ylim(-0.7, 4.7)
    sc.set_xlim(-0.05, 1.05); sc.set_xlabel("detector's hallucination probability", labelpad=1.5)
    for s in ("top", "right", "left"): sc.spines[s].set_visible(False)
    sc.tick_params(axis="y", length=0)
    sc.set_title("score of the two members of the pair", fontsize=6.4, pad=3)
    sc.text(-0.02, 1.13, "b", transform=sc.transAxes, fontsize=8.5, fontweight="bold", va="top")
    sc.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.4, color=INK, label="grounded answer"),
                       Line2D([], [], marker="o", ls="", ms=3.4, mfc="white", mec=WARM, mew=1.0,
                              label="matched hallucination")],
              loc="lower left", bbox_to_anchor=(-0.02, -0.30), ncol=2, frameon=False,
              fontsize=5.9, handletextpad=0.3, columnspacing=1.0, borderpad=0.1)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "edfig5." + ext))
    plt.close(fig)
    print("[edfig5] example written")


if __name__ == "__main__":
    edfig_example()
