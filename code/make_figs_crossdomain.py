"""Two main figures for the cross-domain results.

Fig 5  leakage spectrum: for each of the nine benchmarks, the drop from the
       maximally leaky rung (L1) to the matched rung (L4), ordered, with the
       negative control annotated. Panel b shows the cheap text-only audit.

Fig 6  collapse grid: semantic entropy and RadFlag at the matched rung for
       every backbone and every benchmark we measured, with the leaky-rung
       value shown beside it so the reader can see what was removed.

Colours are chosen to stay distinguishable in greyscale and for the common
forms of colour blindness.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_PATH = next(p for p in (os.path.join(ROOT, "crossdomain_results.json"),
                             os.path.join(ROOT, "results", "crossdomain_results.json"))
                 if os.path.exists(p))
DATA = json.load(open(DATA_PATH, encoding="utf-8"))
OUT = next(p for p in (os.path.join(HERE, "figures"), os.path.join(ROOT, "figures"))
           if os.path.isdir(p))
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "xtick.major.width": 0.7,
                     "ytick.major.width": 0.7, "font.family": "DejaVu Sans"})
MED = {"VQA-RAD", "Path-VQA", "OmniMedVQA"}
C_MED, C_SCI, C_CTRL = "#1f6fb4", "#c2542c", "#7a7a7a"


def fig_leakage():
    rows = DATA["tier1_full_ladder"]["rows"]
    recs = [(r[0], r[10]) for r in rows]          # name, delta L1->L4
    recs.sort(key=lambda x: x[1])
    names = [r[0] for r in recs]
    vals = [r[1] for r in recs]
    cols = [C_CTRL if n == "HallusionBench" else (C_MED if n in MED else C_SCI) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(names))
    ax.barh(y, vals, color=cols, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("AUROC lost when the protocol is matched, L1$\\to$L4", fontsize=7.5)
    ax.axvline(0, color="k", lw=0.7)
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, "%.3f" % v, va="center", fontsize=6.5)
    ax.set_xlim(-0.004, max(vals) * 1.32)
    ax.set_title("a   protocol sensitivity of nine benchmarks", fontsize=8, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    aud = DATA["cheap_audit"]["rows"]
    ax = axes[1]
    d = np.array([r[1] for r in aud])
    o = np.array([r[4] for r in aud])
    ax.scatter(o, d, s=22, color=[C_CTRL if r[0] == "HallusionBench" else
                                  (C_MED if r[0] in MED else C_SCI) for r in aud],
               zorder=3, edgecolor="white", linewidth=0.5)
    lim = [min(o.min(), d.min()) - 0.02, max(o.max(), d.max()) + 0.02]
    ax.plot(lim, lim, ls=":", color="0.55", lw=0.8)
    r = DATA["cheap_audit"]["corr_detector_drop_vs_oracle_drop"]
    ax.text(0.04, 0.93, "$r = %.2f$" % r, transform=ax.transAxes, fontsize=7.5)
    ax.set_xlabel("text-only oracle drop, L0$\\to$L4", fontsize=7.5)
    ax.set_ylabel("detector drop, L1$\\to$L4", fontsize=7.5)
    ax.set_title("b   a text-only audit reproduces it", fontsize=8, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_leakage.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig5_leakage.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_collapse():
    fam = DATA["tier2_six_crossdomain_benchmarks"]["rows"]
    order = ["MMMU", "ScienceQA", "MMBench", "CV-Bench", "ChartQA",
             "HallusionBench", "DocumentVQA", "RSVQA-HR"]
    l4 = {r[0]: r for r in fam if r[7] == "L4_fair"}
    l0 = {r[0]: r for r in fam if r[7] == "L0_leaky"}

    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    n = len(order)
    x = np.arange(n)
    w = 0.36
    for off, (key, lab, col) in zip((-w / 2, w / 2),
                                    ((3, "semantic entropy", "#1f6fb4"),
                                     (4, "RadFlag", "#c2542c"))):
        l0v = [l0[b][key] for b in order]
        l4v = [l4[b][key] for b in order]
        ax.bar(x + off, l4v, w * 0.92, color=col, label=lab + " (L4, matched)")
        ax.scatter(x + off, l0v, marker="_", s=90, color="k", linewidth=1.4,
                   zorder=4, label="L0 (leaky)" if off < 0 else None)
    ax.axhline(0.5, color="k", lw=0.9, ls="--")
    ax.text(-0.42, 0.5015, "chance", fontsize=6.5, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=6.5, rotation=18, ha="right")
    ax.set_ylim(0.47, 0.60)
    ax.set_ylabel("AUROC", fontsize=7.5)
    ax.set_title("Sampling-statistics families at the matched rung, Qwen3.5-4B, eight benchmarks", fontsize=8, loc="left")
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_collapse.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "fig6_collapse.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_leakage()
    fig_collapse()
    print("wrote fig5_leakage.pdf, fig6_collapse.pdf")
