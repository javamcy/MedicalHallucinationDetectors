#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit the one-ladder-many-benchmarks table from the e1_protocol JSONs.

Usage: python make_tab_eleven.py <results_dir> <out.tex>
"""
import json
import os
import sys

ORDER = [("vqa_rad", "VQA-RAD"), ("path_vqa", "Path-VQA"), ("omni", "OmniMedVQA"),
         ("kvasir", "Kvasir-VQA-x1"),
         ("mmmu", "MMMU"), ("scienceqa", "ScienceQA"), ("mmbench", "MMBench"),
         ("cvbench", "CV-Bench"), ("chartqa", "ChartQA"), ("hallusion", "HallusionBench"),
         ("docvqa", "DocumentVQA"), ("rsvqa", "RSVQA-HR")]
LEVELS = ["L0_leaky", "L1_imgsplit", "L2_len", "L3_type", "L4_fair", "L5_fair_row"]


def f(x):
    return "--" if x is None else "%.3f" % x


def main():
    res = sys.argv[1]
    out = sys.argv[2]
    lines = [r"% body only; caption and label are set in the table source",
             r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
             r"\begin{tabular}{@{}lccccccccccc@{}}", r"\toprule",
             r"Benchmark & L0 & L1 & L2 & L3 & L4 & L5 & no img & oracle & "
             r"$\Delta_{L1\to L4}$ & image & $n$ \\", r"\midrule"]
    for name, label in ORDER:
        p = os.path.join(res, "e1_protocol_%s.json" % name)
        if not os.path.exists(p):
            print("missing", p, file=sys.stderr)
            continue
        d = json.load(open(p, encoding="utf-8"))
        cells = [f(d[l]["mlp"]) if l in d else "--" for l in LEVELS]
        e4 = d["L4_fair"]
        noimg = f(e4.get("mlp_noimage"))
        orac = f(e4.get("shortcut_gbm"))
        lam = d["L1_imgsplit"]["mlp"] - e4["mlp"]
        img = e4["mlp"] - e4["mlp_noimage"]
        lines.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %+.3f & %+.3f & %d \\\\" %
                     (label, *cells, noimg, orac, lam, img, e4["n_test"]))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    open(out, "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    main()
