#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-quality figures for the NMI submission.

Design rules followed (Nature Portfolio figure guidance):
  * final width <= 180 mm (we use 160 mm = text block), vector PDF
  * sans-serif labels at 5-7 pt *at final size* -> base 7 pt in a 160 mm canvas
  * bold panel letters, top-left, outside the axes
  * no chartjunk: no top/right spines, no outward ticks, light y-grid only
  * one coherent palette across all panels; direct labels over legends
  * the headline number of each panel is annotated, not left for the reader
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "remote_out")
FIG = os.path.join(HERE, "nmi_analysis", "figures")
os.makedirs(FIG, exist_ok=True)

# ---------------------------------------------------------------- palette
INK = "#12284c"     # primary / full detector
WARM = "#b5442a"    # no image
GREEN = "#2f6b4f"   # text-only
GOLD = "#c08a2e"
PLUM = "#6b3f69"
GREY = "#8a8a8a"
LGREY = "#d8d8d8"
BAND = "#f0ede6"    # highlight band

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 7.0,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "legend.fontsize": 6.2,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

DS = ["vqa_rad", "path_vqa", "omni"]
DSLAB = {"vqa_rad": "VQA-RAD", "path_vqa": "Path-VQA", "omni": "OmniMed-VQA-mini"}
LEVELS = ["L0_leaky", "L1_imgsplit", "L2_len", "L3_type", "L4_fair", "L5_fair_row"]
LVI = {l: i for i, l in enumerate(LEVELS)}
LVI["L6_native"] = 6


def load(n):
    p = os.path.join(RES, n)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def letter(ax, s, dx=-0.115, dy=1.10):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold",
            va="top", ha="left", color="#111111")


def ygrid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, ls=":", lw=0.45, color=LGREY)
    ax.xaxis.grid(False)


# ================================================================ Fig 1
def fig1():
    d = {ds: load("e1_protocol_%s.json" % ds) for ds in DS}
    if any(v is None for v in d.values()):
        print("[fig1] missing"); return
    fig = plt.figure(figsize=(6.30, 4.75))          # 160 mm wide
    gs = fig.add_gridspec(2, 3, height_ratios=[0.80, 1.0], hspace=0.72, wspace=0.26,
                          left=0.075, right=0.985, top=0.96, bottom=0.085)

    # ---------- a: the ladder ----------
    ax = fig.add_subplot(gs[0, :])
    ax.set_axis_off(); ax.set_xlim(0, 10.1); ax.set_ylim(0, 2.62)
    BW, X0, STEP = 1.22, 0.30, 1.62
    xs = [X0 + i * STEP for i in range(6)]
    fills = ["#f7e7e2", "#f6e2d8", "#fbeed3", "#eef1dc", "#dfeade", "#cfe3d4"]
    edges = ["#c9a99f", "#c9a99f", "#cdb98c", "#b6bf9a", "#8fae96", "#6f9a80"]
    for i, (x, f, e) in enumerate(zip(xs, fills, edges)):
        ax.add_patch(FancyBboxPatch((x, 1.16), BW, 0.50,
                                    boxstyle="round,pad=0.015,rounding_size=0.05",
                                    facecolor=f, edgecolor=e, lw=0.8, zorder=2))
        ax.text(x + BW / 2, 1.41, "L%s" % "012345"[i], ha="center", va="center",
                fontsize=8.0, fontweight="bold", color="#222222", zorder=3)
    labs = ["uniform\nnegatives,\npair split", "+ image-\nlevel split", "+ length\nmatch (±2)",
            "+ format\nmatch", "matched\nprotocol", "matched,\npair split"]
    for x, t in zip(xs, labs):
        ax.text(x + BW / 2, 1.06, t, ha="center", va="top", fontsize=5.5,
                color="#333333", linespacing=1.30)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + STEP - 0.03, 1.41), xytext=(x + BW + 0.03, 1.41),
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#b0b0b0",
                                    shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(X0, 1.94), xytext=(xs[-1] + BW, 1.94),
                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="#999999",
                                linestyle=(0, (4, 3))))
    ax.text(X0, 2.04, "leaky", fontsize=6.0, color="#777777", ha="left")
    ax.text(xs[-1] + BW, 2.04, "strictly matched", fontsize=6.0, color="#777777", ha="right")
    ax.text(X0, 0.22, "L6 (OmniMed only) additionally uses the dataset's own wrong multiple-choice options.",
            fontsize=5.4, color="#666666", style="italic", ha="left")
    letter(ax, "a", dx=-0.004, dy=1.12)

    # ---------- b-d ----------
    series = [("mlp", INK, "o", 1.55, 3.0, "full  $[\\mathbf{g},\\mathbf{a},\\mathbf{q}]$", True),
              ("mlp_noimage", WARM, "o", 1.15, 2.6, "no image  $[\\mathbf{a},\\mathbf{q}]$", False),
              ("shortcut_gbm", GREEN, "^", 1.05, 2.6, "text-only oracle", False)]
    for k, ds in enumerate(DS):
        ax = fig.add_subplot(gs[1, k])
        dd, lvs = d[ds], LEVELS + (["L6_native"] if ds == "omni" else [])
        x = np.arange(len(lvs))
        ax.axvspan(LVI["L4_fair"] - 0.42, LVI["L4_fair"] + 0.42, color=BAND, zorder=0)
        for key, col, mk, lw, ms, lab, ci in series:
            ys = [dd[l][key] for l in lvs]
            if ci:
                lo = [dd[l]["mlp_ci"][0] for l in lvs]
                hi = [dd[l]["mlp_ci"][1] for l in lvs]
                ax.fill_between(x, lo, hi, color=col, alpha=0.13, lw=0, zorder=1)
            ax.plot(x, ys, mk + "-", ms=ms, lw=lw, color=col, label=lab,
                    mfc="white" if key == "mlp_noimage" else col, mew=1.0, zorder=3)
        ax.axhline(0.5, ls=(0, (2, 2)), lw=0.7, color=GREY, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(["L%s" % "0123456"[i] for i in range(len(lvs))])
        ax.set_ylim(0.42, 1.03); ax.set_xlim(-0.4, len(lvs) - 0.6)
        ax.set_title(DSLAB[ds], pad=4, color="#222222")
        ax.set_xlabel("protocol rung", labelpad=1.5)
        ygrid(ax)
        if k == 0:
            ax.set_ylabel("AUROC")
            ax.legend(frameon=False, fontsize=5.4, handlelength=1.4, loc="upper center",
                      bbox_to_anchor=(0.5, -0.32), ncol=1, borderpad=0.1, labelspacing=0.22)
        letter(ax, "bcd"[k], dx=-0.20)
    fig.savefig(os.path.join(FIG, "fig1.pdf")); plt.close(fig)
    print("[fig1] ok")


# ================================================================ Fig 2
def fig2():
    d = {ds: load("e1_protocol_%s.json" % ds) for ds in DS}
    if any(v is None for v in d.values()):
        print("[fig2] missing"); return
    fig = plt.figure(figsize=(6.30, 3.95))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.0], hspace=0.62, wspace=0.30,
                          left=0.115, right=0.975, top=0.955, bottom=0.115)

    # ---------- a: horizontal bars, full width ----------
    ax = fig.add_subplot(gs[0, :])
    groups = [("vqa_rad", "L4_fair", "VQA-RAD  (matched)"),
              ("path_vqa", "L4_fair", "Path-VQA  (matched)"),
              ("omni", "L4_fair", "OmniMed  (matched)"),
              ("omni", "L6_native", "OmniMed  (authored negatives)")]
    y = np.arange(len(groups))[::-1]
    for i, (ds, lv, lab) in enumerate(groups):
        e = d[ds][lv]
        yy = y[i]
        ax.barh(yy + 0.19, e["mlp"], 0.34, color=INK, lw=0, zorder=2)
        ax.barh(yy - 0.19, e["mlp"], 0.34, color="none", ec=WARM, lw=0.55,
                ls=(0, (1.6, 1.6)), zorder=3)
        ax.barh(yy - 0.19, e["mlp_noimage"], 0.34, color=WARM, lw=0, zorder=2)
        rec = 100 * e["mlp_noimage"] / e["mlp"]
        ax.text(e["mlp"] + 0.012, yy + 0.19, "%.3f" % e["mlp"], va="center", fontsize=6.3, color=INK)
        ax.text(e["mlp_noimage"] + 0.012, yy - 0.19, "%.3f" % e["mlp_noimage"], va="center",
                fontsize=6.3, color=WARM)
        ax.text(1.055, yy, "%.0f%%" % rec, va="center", ha="right", fontsize=7.0,
                fontweight="bold", color="#7a2f1c" if rec >= 99.5 else INK)
    ax.axvline(0.5, ls=(0, (2, 2)), lw=0.7, color=GREY, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([g[2] for g in groups], fontsize=6.4)
    ax.set_xlim(0.44, 1.075); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("AUROC", labelpad=1.5)
    ax.set_ylim(-0.55, len(groups) - 0.45)
    ax.xaxis.grid(True, ls=":", lw=0.45, color=LGREY); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.legend([Line2D([], [], color=INK, lw=4), Line2D([], [], color=WARM, lw=4)],
              ["full detector", "no image"], frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.26), ncol=2, handlelength=1.2, fontsize=6.0)
    ax.text(0.5, 1.16, "right column: percentage of the full detector recovered without the image",
            transform=ax.transAxes, fontsize=5.5, color="#7a2f1c", style="italic", ha="center")
    letter(ax, "a", dx=-0.075)

    # ---------- b: dumbbell of Delta_no-img ----------
    ax = fig.add_subplot(gs[1, 0])
    for ds, col in (("vqa_rad", INK), ("path_vqa", WARM), ("omni", GREEN)):
        lvs = LEVELS + (["L6_native"] if ds == "omni" else [])
        xs = [LVI[l] for l in lvs]
        ys = [d[ds][l]["mlp"] - d[ds][l]["mlp_noimage"] for l in lvs]
        ax.plot(xs, ys, "-", lw=0.8, color=col, alpha=0.55, zorder=1)
        ax.plot(xs, ys, "o", ms=3.0, color=col, mfc="white", mew=0.9, zorder=2, label=DSLAB[ds])
    ax.axhline(0, lw=0.7, color="#444444")
    ax.set_xticks(range(6)); ax.set_xticklabels(["L%d" % i for i in range(6)])
    ax.set_ylabel("$\\Delta$ AUROC from adding the image", labelpad=1.5)
    ax.set_xlabel("protocol rung", labelpad=1.5)
    ax.set_ylim(-0.02, 0.085)
    ygrid(ax)
    ax.legend(frameon=False, handlelength=1.3, loc="upper left", borderpad=0.1, fontsize=5.8)
    letter(ax, "b", dx=-0.26)

    # ---------- c: frozen alignment ----------
    ax = fig.add_subplot(gs[1, 1])
    for ds, col in (("vqa_rad", INK), ("path_vqa", WARM)):
        ax.plot(range(6), [d[ds][l]["frozen_scalars"]["clipscore_gA"]["auroc"] for l in LEVELS],
                "o-", ms=2.8, lw=1.0, color=col, mfc="white", mew=0.9,
                label="%s CLIPScore" % DSLAB[ds].split("-")[0])
        ax.plot(range(6), [d[ds][l]["frozen_scalars"]["maxpatch_pA"]["auroc"] for l in LEVELS],
                "s:", ms=2.4, lw=0.8, color=col, alpha=0.65,
                label="%s max patch" % DSLAB[ds].split("-")[0])
    ax.axhline(0.5, ls=(0, (2, 2)), lw=0.7, color=GREY)
    ax.set_ylim(0.45, 0.72)
    ax.set_xticks(range(6)); ax.set_xticklabels(["L%d" % i for i in range(6)])
    ax.set_ylabel("AUROC (no training)", labelpad=1.5)
    ax.set_xlabel("protocol rung", labelpad=1.5)
    ygrid(ax)
    ax.legend(frameon=False, fontsize=5.3, handlelength=1.4, loc="upper left",
              borderpad=0.1, labelspacing=0.25)
    letter(ax, "c", dx=-0.24)
    fig.savefig(os.path.join(FIG, "fig2.pdf")); plt.close(fig)
    print("[fig2] ok")


# ================================================================ Fig 3
def fig3():
    e7, e9 = load("e7_failure.json"), load("e9_yesno.json")
    if not e7:
        print("[fig3] missing"); return
    fig = plt.figure(figsize=(6.30, 2.30))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.05, 0.95], wspace=0.42,
                          left=0.085, right=0.985, top=0.90, bottom=0.24)

    # ---------- a: AUROC by question type ----------
    ax = fig.add_subplot(gs[0, 0])
    e5 = load("e5_modality.json") or {}
    byt = e7["vqa_rad|L4_fair"]["by_qtype"]
    order = ["yes_no", "what", "which", "where", "other"]
    qts = [q for q in order if q in byt]

    def _v(q, k):
        if k == "mlp_noimage":
            e = e5.get("vqa_rad|L4_fair|%s" % q, {})
            return e.get("mlp_noimage", np.nan)
        return byt[q].get(k, np.nan)

    M = np.array([[_v(q, k) for q in qts]
                  for k in ("mlp", "mlp_noimage", "clipscore")])
    im = ax.imshow(M, cmap="BuPu", vmin=0.45, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(qts)))
    ax.set_xticklabels(["%s\n%d" % (q.replace("_", "/"), byt[q]["n"]) for q in qts], fontsize=5.8)
    ax.set_yticks(range(3)); ax.set_yticklabels(["full", "no image", "CLIPScore"], fontsize=6.1)
    for i in range(3):
        for j in range(len(qts)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=5.6,
                        color="white" if M[i, j] > 0.76 else "#111111")
    ax.set_xlabel("question type (n pairs)", labelpad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    letter(ax, "a", dx=-0.30)

    # ---------- b: who answers the yes/no block ----------
    ax = fig.add_subplot(gs[0, 1])
    labels, full, noimg, qp = [], [], [], []
    for ds in ("vqa_rad", "path_vqa"):
        if not e9 or ds not in e9 or "yes_no" not in e9[ds]:
            continue
        v = e9[ds]["yes_no"]
        labels.append("%s\nyes/no" % DSLAB[ds].split("-")[0])
        full.append(v["pair_acc_full"]); noimg.append(v["pair_acc_no_image"])
        q = v.get("question_prior")
        qp.append(q["auroc"] if q else np.nan)
        if "open" in e9[ds]:
            w = e9[ds]["open"]
            labels.append("%s\nopen" % DSLAB[ds].split("-")[0])
            full.append(w["pair_acc_full"]); noimg.append(w["pair_acc_no_image"]); qp.append(np.nan)
    x = np.arange(len(labels)); w = 0.26
    hasq = ~np.isnan(np.asarray(qp, float))
    ax.bar(x - w, full, w, color=INK, lw=0, label="full")
    ax.bar(x, noimg, w, color=WARM, lw=0, label="no image")
    if hasq.any():
        ax.bar(x[hasq], np.asarray(qp, float)[hasq], w, color=GREEN, lw=0,
               label="question\nonly")
    for xi in x[~hasq]:
        ax.text(xi + w, 0.435, "n.d.", ha="center", fontsize=5.0, color="#888888",
                rotation=90, va="bottom")
    ax.axhline(0.5, ls=(0, (2, 2)), lw=0.7, color=GREY)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.4, linespacing=1.1)
    ax.set_ylim(0.40, 1.06); ax.set_ylabel("accuracy on matched pairs", labelpad=1.5)
    ax.text(0.5, -0.30, "n.d.: a question-only classifier cannot separate a pair whose question is identical",
            transform=ax.transAxes, fontsize=4.9, color="#777777", ha="center", style="italic")
    ygrid(ax)
    ax.legend(frameon=False, fontsize=5.3, handlelength=1.0, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), columnspacing=1.0, handletextpad=0.4)
    letter(ax, "b", dx=-0.22)

    # ---------- c: error asymmetry ----------
    ax = fig.add_subplot(gs[0, 2])
    names, fpv, fnv = [], [], []
    for ds in DS:
        k = "%s|L4_fair" % ds
        if k not in e7:
            continue
        e = e7[k]["errors"]["mlp"]
        names.append(DSLAB[ds].split("-")[0]); fpv.append(e["n_fp"]); fnv.append(e["n_fn"])
    x = np.arange(len(names))
    ax.bar(x, fnv, 0.52, color=INK, lw=0, label="missed\nhallucination")
    ax.bar(x, fpv, 0.52, bottom=[0] * len(x), color=GOLD, lw=0, label="false\nalarm")
    _top = max(a + b for a, b in zip(fnv, fpv)) if names else 300.0
    for xi, a, b in zip(x, fnv, fpv):
        ax.text(xi, a + b + _top * 0.02, "%.1f:1" % (a / max(1, b)), ha="center", va="bottom",
                fontsize=6.4, color="#333333", clip_on=False)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=5.8)
    ax.set_ylim(0, _top * 1.22); ax.set_ylabel("errors at Youden threshold", labelpad=1.5)
    ygrid(ax)
    ax.legend(frameon=False, fontsize=5.3, handlelength=1.0, loc="upper center",
              bbox_to_anchor=(0.5, -0.28), ncol=1, borderpad=0.1, labelspacing=0.25)
    letter(ax, "c", dx=-0.26)
    fig.savefig(os.path.join(FIG, "fig3.pdf")); plt.close(fig)
    print("[fig3] ok")


# ================================================================ Fig 4 (forest)
def fig4():
    LEGACY = {
        "VASE (LLaVA-Med-1.5)": (0.4941, 0.4950, 0.4911, "sampling"),
        "VASE (MedGemma-4b)": (0.4991, 0.5029, None, "sampling"),
        "Semantic entropy (LLaVA-Med)": (0.4910, 0.5020, 0.4989, "sampling"),
        "RadFlag (LLaVA-Med)": (0.5053, 0.4979, 0.4987, "sampling"),
        "Semantic entropy (Qwen-3B)": (0.500, 0.500, None, "sampling"),
        "RadFlag (Qwen-3B)": (0.500, 0.500, None, "sampling"),
        "Answer consistency (Qwen-3B)": (0.562, 0.688, None, "sampling"),
        "Zero-shot judge (LLaVA-Med)": (0.5547, 0.4961, 0.5717, "internal"),
        "Zero-shot judge (Qwen-3B)": (0.661, None, 0.750, "internal"),
        "Zero-shot judge (Qwen-7B)": (0.712, None, 0.719, "internal"),
        "Zero-shot judge (Qwen-32B-4bit)": (0.561, 0.526, None, "internal"),
        "Zero-shot judge (Lingshu-7B)": (0.687, 0.627, None, "internal"),
        "Answer NLL (Qwen-3B)": (0.602, None, 0.661, "internal"),
        "CLIPScore": (0.5030, 0.5020, None, "alignment"),
        "CHAIR-Medical": (0.5190, 0.5000, None, "alignment"),
        "max patch--answer": (0.530, 0.527, 0.638, "alignment"),
        "MMedPO fine-tune": (0.4950, 0.4990, None, "alignment"),
        "MMed-RAG fine-tune": (0.4760, 0.5190, None, "alignment"),
        "BiomedCLIP fine-tune": (0.4920, 0.4910, None, "alignment"),
        "20-feature text oracle": (0.513, 0.698, 0.769, "language"),
        "answer char $n$-gram": (0.611, 0.496, 0.519, "language"),
        "answer prior": (0.559, 0.540, 0.964, "language"),
    }
    rows = []
    for name, (a, b, c, fam) in LEGACY.items():
        v = [x for x in (a, b, c) if x is not None]
        rows.append((name, fam, float(np.mean(v))))
    rows.append(("trainable head on frozen CLIP", "trained", 0.851))
    rows.append(("+ visual evidence channels", "trained", 0.846))
    rows.append(("medical LMM trained on protocol", "trained", 0.916))

    order = ["sampling", "internal", "alignment", "language", "trained"]
    colmap = {"sampling": WARM, "internal": PLUM, "alignment": "#2c6b9e",
              "language": GREEN, "trained": INK}
    famlabel = {"sampling": "sampling statistics", "internal": "internal signals",
                "alignment": "fixed image--answer alignment",
                "language": "supervised language statistics",
                "trained": "trained on the protocol"}
    rows.sort(key=lambda r: (order.index(r[1]), r[2]))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(4.15, 0.175 * n + 1.35))
    fig.subplots_adjust(left=0.42, right=0.965, top=0.90, bottom=0.11)
    ys, y = [], 0.0
    prev = None
    for name, fam, val in rows:
        if prev is not None and fam != prev:
            y += 0.85
        ax.plot([val], [y], "o" if fam != "trained" else "D",
                ms=3.9 if fam != "trained" else 4.3,
                color=colmap[fam], mfc="white" if fam != "trained" else colmap[fam], mew=1.15)
        ys.append((y, name, fam)); prev = fam; y += 1.0
    ax.axvline(0.5, ls=(0, (2, 2)), lw=0.8, color=GREY, zorder=0)
    ax.axvline(0.8, ls="--", lw=0.8, color="#b5442a", zorder=0)
    ax.text(0.8, -0.60, "clinical bar 0.80", fontsize=5.6, color="#b5442a", ha="center", va="bottom")
    ax.set_yticks([r[0] for r in ys]); ax.set_yticklabels([r[1] for r in ys], fontsize=5.9)
    ax.set_ylim(-0.8, y - 0.1); ax.invert_yaxis()
    ax.set_xlim(0.44, 1.00)
    ax.set_xlabel("AUROC under the matched protocol\n(mean over the benchmarks tested)", labelpad=2)
    ax.xaxis.grid(True, ls=":", lw=0.45, color=LGREY); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", ls="", ms=3.9, color=colmap[f], mfc="white",
                      mew=1.1, label=famlabel[f]) for f in order]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              frameon=False, ncol=2, fontsize=5.4, handletextpad=0.35,
              columnspacing=1.1, borderpad=0.1)
    fig.savefig(os.path.join(FIG, "fig4.pdf")); plt.close(fig)
    print("[fig4] ok (%d rows)" % n)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4()
