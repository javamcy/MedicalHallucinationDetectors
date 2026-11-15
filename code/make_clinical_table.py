#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit the clinical error-type table and the seed-variance table from the
JSON files written by e11_clinical.py and e10_seeds.py.

Usage: python make_clinical_table.py [--main|--si]
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TAB = REPO / "tables"

TYPES = [("ALL", "All items"),
         ("negation", "Negation (polarity pairs)"),
         ("laterality", "Laterality"),
         ("size_comparison", "Size / comparison"),
         ("lesion_presence", "Lesion presence"),
         ("other", "Other")]

NAME = {"vqa_rad": "VQA-RAD", "path_vqa": "Path-VQA", "omni": "OmniMed-VQA",
        "slake": "SLAKE"}


def f3(x):
    return "%.3f" % x


def cell(a, lo, hi, signed=False):
    if a != a:
        return "--"
    s = ("%+.3f" % a) if signed else f3(a)
    return "%s [%.3f, %.3f]" % (s, lo, hi)


MAIN_GROUPS = ("ALL", "negation", "lesion_presence")


def clinical_tex(d, groups=None):
    types = [(k, v) for k, v in TYPES if groups is None or k in groups]
    lines = [r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Clinical type & Pairs & Full detector & Text only & $\Delta_{\text{img}}$ &"
             r" Missed:false & Opp.\ polarity \\", r"\midrule"]
    for ds in ("vqa_rad", "path_vqa", "omni", "slake"):
        if ds not in d:
            continue
        lines.append(r"\multicolumn{7}{l}{\emph{%s}}\\" % NAME.get(ds, ds))
        for key, lab in types:
            e = d[ds]["types"].get(key)
            if not e:
                continue
            a = e["auroc"]
            full = a.get("full", {})
            txt = a.get("noimage", {})
            dl = e.get("delta_image", {})
            yj = e.get("full_at_youden", {})
            ratio = ""
            if yj:
                fn, fp = yj["n_fn_missed_hallucination"], yj["n_fp_rejected_grounded"]
                ratio = "%.1f:1" % (fn / fp) if fp else "--"
            lines.append("%s & %d & %s & %s & %s & %s & %.0f\\%% \\\\" % (
                lab, e["n_pairs"],
                cell(full.get("auroc", float("nan")), *full.get("ci", (float("nan"),) * 2)),
                f3(txt.get("auroc", float("nan"))) if txt else "--",
                cell(dl.get("diff", float("nan")), *dl.get("ci", (float("nan"),) * 2), signed=True) if dl else "--",
                ratio, 100 * e.get("share_opposite_polarity_yn", 0)))
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines) + "\n"


def seeds_tex(d):
    lines = [r"\small", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Benchmark & $n_{\text{test}}$ & Head & Mean & SD & 95\% seed CI & Range \\",
             r"\midrule"]
    for ds in ("vqa_rad", "path_vqa", "omni", "slake"):
        if ds not in d:
            continue
        r = d[ds]
        for i, (hk, hl) in enumerate((("full", "Full"), ("qce", "Evidence"), ("noimage", "Text only"))):
            h = r["heads"].get(hk)
            if not h:
                continue
            nm = NAME.get(ds, ds) if i == 0 else ""
            lines.append("%s & %d & %s & %.3f & %.3f & [%.3f, %.3f] & %.3f--%.3f \\\\" % (
                nm, r["n_test"], hl, h["mean"], h["sd"], h["ci95_seed"][0], h["ci95_seed"][1],
                h["min"], h["max"]))
        lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    return "\n".join(lines) + "\n"


def controls_tex(d):
    lines = [r"\small", r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Benchmark & $n_{\text{test}}$ & Reference & Constant image & Unrelated image &"
             r" Permuted image & Text only \\", r"\midrule"]
    for ds in ("vqa_rad", "path_vqa", "omni", "slake"):
        if ds not in d:
            continue
        r = d[ds]
        h = r["heads"]

        def g(k):
            return h[k]["test_auroc"] if k in h else float("nan")

        lines.append("%s & %d & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (
            NAME.get(ds, ds), r["n_test"], g("full_reference"), g("full_blank_image"),
            g("full_test_shuffle"), g("full_train_shuffle"), g("noimage")))
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    out = REPO / "results" / "clinical"
    cand = [out, REPO / "results" / "seeds", REPO / "results" / "controls"]
    cp = out / "e11_clinical.json"
    sp = next((c / "e10_seeds.json" for c in cand if (c / "e10_seeds.json").exists()),
              out / "e10_seeds.json")
    if cp.exists():
        cj = json.load(open(cp, encoding="utf-8"))
        s = clinical_tex(cj, MAIN_GROUPS)
        p = TAB / "tab_clinical.tex"
        p.write_text(s, encoding="utf-8")
        print("[TAB] wrote %s" % p)
        s = clinical_tex(cj)
        p = TAB / "tab_clinical_full.tex"
        p.write_text(s, encoding="utf-8")
        print("[TAB] wrote %s" % p)
    else:
        print("[TAB] %s missing" % cp)
    if sp.exists():
        s = seeds_tex(json.load(open(sp, encoding="utf-8")))
        p = TAB / "tab_seeds.tex"
        p.write_text(s, encoding="utf-8")
        print("[TAB] wrote %s" % p)
    else:
        print("[TAB] %s missing" % sp)
    xp = REPO / "results" / "controls" / "e12_controls.json"
    if xp.exists():
        s = controls_tex(json.load(open(xp, encoding="utf-8")))
        p = TAB / "tab_imgcontrols.tex"
        p.write_text(s, encoding="utf-8")
        print("[TAB] wrote %s" % p)
    else:
        print("[TAB] %s missing" % xp)


if __name__ == "__main__":
    main()
