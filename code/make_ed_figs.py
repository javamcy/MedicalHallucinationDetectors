#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended Data figures for the NMI submission."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "remote_out")
FIG = os.path.join(HERE, "nmi_submission", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.0, "axes.titlesize": 7.5, "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 400, "savefig.bbox": "tight",
})
DS = [("vqa_rad", "VQA-RAD"), ("path_vqa", "Path-VQA"), ("omni", "OmniMed")]
LV = ["L0_leaky", "L1_imgsplit", "L2_len", "L3_type", "L4_fair", "L5_fair_row", "L6_native"]
SHORT = {"L0_leaky": "L0", "L1_imgsplit": "L1", "L2_len": "L2", "L3_type": "L3",
         "L4_fair": "L4", "L5_fair_row": "L5", "L6_native": "L6"}
CH = "#999999"


def load(n):
    p = os.path.join(RES, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def edfig_residual():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15), sharey=True)
    series = [("tfidf_answer_char", "answer char $n$-gram", "#1b3b6f", "o"),
              ("tfidf_answer_word", "answer bag-of-words", "#2e7d32", "s"),
              ("tfidf_qa_word", "question+answer words", "#c1440e", "^"),
              ("lexical_only", "20 text features", "#6a1b9a", "D")]
    for ax, (ds, nm) in zip(axes, DS):
        d = load("e4_residual_%s.json" % ds)
        if not d:
            ax.set_axis_off(); continue
        lvs = [l for l in LV if l in d]
        x = np.arange(len(lvs))
        for key, lab, col, mk in series:
            ys = []
            for l in lvs:
                v = d[l][key]
                ys.append(max(v, 1 - v))
            ax.plot(x, ys, mk + "-", ms=3.0, lw=1.1, color=col, label=lab, mfc="white", mew=0.9)
        ax.axhline(0.5, ls=":", lw=0.8, color=CH)
        ax.set_xticks(x); ax.set_xticklabels([SHORT[l] for l in lvs])
        ax.set_title(nm.split("-")[0], pad=3)
        ax.set_xlabel("protocol rung")
        ax.set_ylim(0.44, 1.02)
    axes[0].set_ylabel("AUROC of a text-only classifier")
    axes[0].legend(frameon=False, handlelength=1.5, loc="upper left")
    for ax, s in zip(axes, "abc"):
        ax.text(-0.16, 1.08, s, transform=ax.transAxes, fontsize=9, fontweight="bold")
    fig.savefig(os.path.join(FIG, "edfig2.pdf")); plt.close(fig)
    print("[edfig2] ok")


def edfig_practice():
    # counts over the 15 coded detector papers
    crit = [
        ("Negatives matched to\ntheir correct counterpart", [2, 1, 5, 7]),
        ("Answer length\nequalised",                        [0, 5, 10, 0]),
        ("Image-level split",                               [7, 0, 0, 8]),
        ("Language-only\ncontrol reported",                 [2, 2, 11, 0]),
        ("Confidence interval on\nthe headline metric",      [3, 0, 12, 0]),
        ("Headline result on\n<1,000 items",                [5, 0, 10, 0]),
    ]
    labels = [c[0] for c in crit]
    keys = ["satisfied", "partial", "explicitly absent", "not stated / not applicable"]
    cols = ["#1b3b6f", "#8fb3d9", "#c1440e", "#d9d9d9"]
    M = np.array([c[1] for c in crit], float)
    M = 100.0 * M / M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 2.5))
    left = np.zeros(len(crit))
    y = np.arange(len(crit))[::-1]
    for j, (k, c) in enumerate(zip(keys, cols)):
        ax.barh(y, M[:, j], left=left, color=c, label=k, height=0.62, lw=0)
        for yi, (l, w) in enumerate(zip(left, M[:, j])):
            if w >= 8:
                ax.text(l + w / 2, y[yi], "%d" % round(w * 15 / 100), ha="center", va="center",
                        fontsize=5.8, color="white" if j in (0, 2) else "black")
        left += M[:, j]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.1)
    ax.set_xlim(0, 100); ax.set_xlabel("share of the 15 coded detector papers (%)")
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.savefig(os.path.join(FIG, "edfig3.pdf")); plt.close(fig)
    print("[edfig3] ok")


if __name__ == "__main__":
    edfig_residual()
    edfig_practice()


# ---------------------------------------------------------------------------
def edfig_value():
    """Image contribution across the five benchmarks (the POPE boundary)."""
    D = [("VQA-RAD",  0.513, 0.6818, 0.6645),
         ("Path-VQA", 0.697, 0.9060, 0.8474),
         ("OmniMed",  0.769, 0.9658, 0.9263),
         ("SLAKE",    0.826, 0.9831, 0.8995),
         ("POPE",     0.656, 0.9196, 0.7641)]
    D = sorted(D, key=lambda r: r[2] - r[3])
    names = [r[0] for r in D]
    oracle = np.array([r[1] for r in D]); full = np.array([r[2] for r in D]); noi = np.array([r[3] for r in D])
    cost = full - noi
    y = np.arange(len(D))[::-1]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.55), gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    a = ax[0]
    for yi, f_, n_, c in zip(y, full, noi, cost):
        a.plot([n_, f_], [yi, yi], color="#B9B9B9", lw=2.2, zorder=1, solid_capstyle="round")
    a.scatter(noi, y, s=26, color="#E07B39", zorder=3, label="no-image detector")
    a.scatter(full, y, s=26, color="#2E5E8E", zorder=3, label="full detector")
    for yi, f_, n_, c, nm in zip(y, full, noi, cost, names):
        a.annotate("+%.3f" % c, xy=((f_ + n_) / 2, yi + 0.26), ha="center", va="bottom", fontsize=6.2,
                   color="#333333")
    a.set_yticks(y); a.set_yticklabels(names)
    a.set_xlabel("AUROC at the matched protocol (POPE: authored negatives)")
    a.set_xlim(0.60, 1.02)
    a.legend(loc="lower right", frameon=False, handletextpad=0.4, borderaxespad=0.3)
    a.set_title("Image contribution, largest at the bottom", fontsize=7, loc="left")
    a.text(-0.02, 1.06, "a", transform=a.transAxes, fontsize=9, fontweight="bold", va="top")

    b = ax[1]
    b.scatter(oracle, cost, s=28, color="#2E5E8E", zorder=3)
    for o_, c_, nm in zip(oracle, cost, names):
        dx, dy = (0.012, 0.004)
        if nm in ("SLAKE", "POPE"):
            dx, dy = (0.012, -0.011) if nm == "SLAKE" else (-0.115, 0.006)
        b.annotate(nm, xy=(o_ + dx, c_ + dy), fontsize=6.0, color="#333333")
    b.axhline(0, color="#CCCCCC", lw=0.6, ls=":")
    b.set_xlabel("Text-only oracle AUROC")
    b.set_ylabel("AUROC lost when the image is removed")
    b.set_title("Not a function of the text prior", fontsize=7, loc="left")
    b.text(-0.02, 1.06, "b", transform=b.transAxes, fontsize=9, fontweight="bold", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "edfig4." + ext))
        fig.savefig(os.path.join(HERE, "nmi_analysis_sn", "figures", "edfig4." + ext))
        fig.savefig(os.path.join(HERE, "nmi_analysis", "figures", "edfig4." + ext))
    plt.close(fig)
    print("[edfig4] written")


if __name__ == "__main__" and "--value" in os.sys.argv:
    edfig_value()
