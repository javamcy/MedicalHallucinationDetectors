#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the LaTeX tables of the manuscript from the result JSONs."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "remote_out")
OUT = os.path.join(HERE, "nmi", "tables")
os.makedirs(OUT, exist_ok=True)

DS = ["vqa_rad", "path_vqa", "omni"]
DSLAB = {"vqa_rad": "VQA-RAD", "path_vqa": "Path-VQA", "omni": "OmniMed-VQA-mini"}
LEVELS = ["L0_leaky", "L1_imgsplit", "L2_len", "L3_type", "L4_fair", "L5_fair_row"]
ROWL = {
    "L0_leaky": r"\textbf{L0} uniform negatives, pair split",
    "L1_imgsplit": r"\textbf{L1} $+$ image-level split",
    "L2_len": r"\textbf{L2} $+$ length match ($\pm2$)",
    "L3_type": r"\textbf{L3} $+$ format match",
    "L4_fair": r"\textbf{L4} $+$ length \& format \;(matched)",
    "L5_fair_row": r"\textbf{L5} matched, pair split",
    "L6_native": r"\textbf{L6} authored negatives",
}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _nan(v):
    try:
        return "--" if v != v else "%.3f" % v
    except Exception:
        return "--"


def fmt(x, nd=3):
    return "--" if x is None else ("%.*f" % (nd, x))


def ci(e, key):
    c = e.get("%s_ci" % key)
    if not c:
        return ""
    return " [%.3f, %.3f]" % (c[0], c[1])


def tab_ladder():
    d = {ds: load("e1_protocol_%s.json" % ds) for ds in DS}
    lines = []
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{lcccccc}")
    lines.append(r"\toprule")
    lines.append(r"Protocol rung & Full detector & No-image & $\Delta_{\mathrm{no\,img}}$ & "
                 r"Recovered & Text-only & CLIPScore / \\")
    lines.append(r" & $[\mathbf{g},\mathbf{a},\mathbf{q}]$ & $[\mathbf{a},\mathbf{q}]$ & & (\%) & "
                 r"oracle & max patch \\")
    lines.append(r"\midrule")
    for ds in DS:
        if not d.get(ds):
            continue
        lines.append(r"\multicolumn{7}{l}{\emph{%s} \;($n_{\mathrm{test}}$ = %d pairs)} \\"
                     % (DSLAB[ds], d[ds][LEVELS[0]]["n_test"]))
        lvs = LEVELS + (["L6_native"] if ds == "omni" else [])
        for lv in lvs:
            e = d[ds].get(lv)
            if e is None:
                continue
            full = e["mlp"]
            ni = e["mlp_noimage"]
            lines.append(
                r"%s & %s%s & %.3f & %+.3f & %.0f & %.3f & %.3f / %.3f \\"
                % (ROWL[lv], fmt(full), ci(e, "mlp"), ni, full - ni, 100.0 * ni / full,
                   e["shortcut_gbm"],
                   e["frozen_scalars"]["clipscore_gA"]["auroc"],
                   e["frozen_scalars"]["maxpatch_pA"]["auroc"]))
        lines.append(r"\addlinespace[2pt]")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    open(os.path.join(OUT, "tab_ladder.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[tab] tab_ladder.tex")


def tab_audit():
    d = {ds: load("e1_protocol_%s.json" % ds) for ds in RES_DS}
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Dataset & Rung & $n$ & strict match & widened & fallback \\", r"\midrule"]
    for ds in RES_DS:
        if not d.get(ds):
            continue
        for lv in LEVELS + (["L6_native"] if ds == "omni" else []):
            e = d[ds].get(lv)
            if e is None:
                continue
            a = e["audit"]
            tot = sum(a.values())
            strict = sum(v for k, v in a.items()
                         if k in ("len2", "type+len2", "type_only", "native_len2"))
            wide = sum(v for k, v in a.items() if k in ("len5", "len10", "type+len5",
                                                        "type+len10", "native_lenwide"))
            fb = sum(v for k, v in a.items() if "fallback" in k)
            lines.append(r"%s & %s & %d & %d (%.2f\%%) & %d & %d \\"
                         % (DSLAB[ds], lv.replace("_", r"\_"), tot, strict,
                            100.0 * strict / max(1, tot), wide, fb))
        lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    open(os.path.join(OUT, "tab_audit.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[tab] tab_audit.tex")


def tab_residual():
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Dataset / rung & answer char-$n$-gram & answer words & question--answer & "
             r"20 text feats & answer prior & length gap \\", r"\midrule"]
    for ds in DS:
        d = load("e4_residual_%s.json" % ds)
        if not d:
            continue
        lines.append(r"\multicolumn{7}{l}{\emph{%s}} \\" % DSLAB[ds])
        for lv in LEVELS + (["L6_native"] if ds == "omni" else []):
            e = d.get(lv)
            if e is None:
                continue
            ch = e["tfidf_answer_char"]
            ch = max(ch, 1 - ch)
            wd = e["tfidf_answer_word"]
            wd = max(wd, 1 - wd)
            qa = e["tfidf_qa_word"]
            qa = max(qa, 1 - qa)
            lex = e["lexical_only"]
            lex = max(lex, 1 - lex)
            af = e["answer_frequency_leak"]
            lines.append(r"%s & %.3f & %.3f & %.3f & %.3f & %.3f & %.2f \\"
                         % (lv.replace("_", r"\_"), ch, wd, qa, lex, af, e["len_abs_gap"]))
        lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    open(os.path.join(OUT, "tab_residual.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[tab] tab_residual.tex")


def tab_controls():
    d = {ds: load("e2_random_%s.json" % ds) for ds in DS}
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Dataset & Rung & perm.\ $p$ (full det.) & permuted-label retrain & "
             r"random-score null (95\% interval) \\", r"\midrule"]
    for ds in DS:
        if not d.get(ds):
            continue
        lines.append(r"\multicolumn{5}{l}{\emph{%s}} \\" % DSLAB[ds])
        for lv in LEVELS + (["L6_native"] if ds == "omni" else []):
            e = d[ds].get(lv)
            if e is None:
                continue
            sh = e.get("mlp_shuffled_label_retrain", {}).get("test_auroc")
            null = e.get("random_score_null", {})
            lines.append(r"%s & %s & $<$0.001 & %.3f & [%.3f, %.3f] \\"
                         % (DSLAB[ds].split("-")[0], lv.replace("_", r"\_"),
                            sh if sh is not None else float("nan"),
                            null.get("lo", float("nan")), null.get("hi", float("nan"))))
        lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    open(os.path.join(OUT, "tab_controls.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[tab] tab_controls.tex")


RES_DS = DS

if __name__ == "__main__":
    tab_ladder()
    tab_audit()
    tab_residual()
    tab_controls()


# ---------------------------------------------------------------------------
# Supplementary tables
# ---------------------------------------------------------------------------
FEAT_NAMES = {
    "ans_tokens": r"answer tokens", "ans_chars": r"answer characters",
    "q_tokens": r"question tokens", "ans/q_tok": r"answer/question tokens",
    "ans/q_char": r"answer/question characters", "is_yesno": r"answer is yes/no",
    "qt_is_yesno": r"question is yes/no", "qt_is_wh": r"question is wh-type",
    "qt_other": r"question other type", "type_match": r"answer--question format match",
    "has_digit": r"answer contains a digit", "has_pct": r"answer contains \%",
    "has_comma": r"answer contains a comma", "ends_period": r"answer ends with a period",
    "cap_first": r"answer starts with a capital", "negation_word": r"answer contains a negation",
    "log_ans_freq": r"log answer-string corpus frequency", "log_qt_freq": r"log question corpus frequency",
    "qa_overlap_n": r"question/answer shared-token count", "qa_jaccard": r"question/answer Jaccard",
}


def tab_leakfeat():
    lines = [r"\begin{tabular}{l" + "c" * 6 + "}", r"\toprule",
             r"Single text feature & \multicolumn{2}{c}{VQA-RAD} & \multicolumn{2}{c}{Path-VQA} &"
             r" \multicolumn{2}{c}{OmniMed (authored)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
             r" & L1 & L4 & L1 & L4 & L4 & L6 \\", r"\midrule"]
    d = {ds: load("e1_protocol_%s.json" % ds) for ds in DS}
    if not all(d.values()):
        return
    for k, nice in FEAT_NAMES.items():
        row = [nice]
        for ds, lvs in (("vqa_rad", ["L1_imgsplit", "L4_fair"]),
                        ("path_vqa", ["L1_imgsplit", "L4_fair"]),
                        ("omni", ["L4_fair", "L6_native"])):
            for lv in lvs:
                row.append("%.3f" % d[ds][lv]["leak_feature_auroc"][k])
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    open(os.path.join(OUT, "tab_leakfeat.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[tab] tab_leakfeat.tex")


def tab_bytype():
    d5 = load("e5_modality.json")
    if not d5:
        return
    qts = ["yes_no", "what", "where", "which", "how", "other"]
    lines = [r"\begin{tabular}{llrrcccc}", r"\toprule",
             r"Dataset & Block & $n$ & full & no image & text oracle & CLIPScore \\", r"\midrule"]
    for ds in DS:
        for lv in ("L4_fair", "L6_native"):
            for qt in qts:
                k = "%s|%s|%s" % (ds, lv, qt)
                if k not in d5:
                    continue
                v = d5[k]
                lines.append(r"%s (%s) & %s & %d & %.3f & %.3f & %.3f & %.3f \\"
                             % (DSLAB[ds], lv.replace("_", r"\_"), qt.replace("_", r"\_"),
                                v["n"], v["mlp"], v.get("mlp_noimage", float("nan")),
                                v.get("shortcut_gbm", float("nan")), v["clipscore"]))
        lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    open(os.path.join(OUT, "tab_bytype.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[tab] tab_bytype.tex")


def tab_modality():
    d5 = load("e5_modality.json")
    if not d5:
        return
    lines = [r"\begin{tabular}{lrrcccc}", r"\toprule",
             r"Modality & $n$ & full & no image & text oracle & CLIPScore \\", r"\midrule"]
    for lv in ("L4_fair", "L6_native"):
        lines.append(r"\multicolumn{6}{l}{\emph{OmniMed, %s}} \\" % lv.replace("_", r"\_"))
        items = []
        for k, v in d5.items():
            ds, l, g = k.split("|", 2)
            if ds == "omni" and l == lv and v["n"] >= 40:
                if g.startswith(("RadImageNet", "Retinal", "Yangxi")):
                    continue
                if g in ("yes_no", "what", "where", "which", "how", "other"):
                    continue
                items.append((g, v["n"], v))
        items.sort(key=lambda z: -z[1])
        for g, n, v in items:
            lines.append(r"%s & %d & %.3f & %.3f & %.3f & %.3f \\"
                         % (g.replace("%", r"\%").replace("&", r"\&").replace("_", r"\_"), n, v["mlp"],
                            v.get("mlp_noimage", float("nan")),
                            v.get("shortcut_gbm", float("nan")), v["clipscore"]))
        lines.append(r"\addlinespace[2pt]")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    open(os.path.join(OUT, "tab_modality.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[tab] tab_modality.tex")


def tab_e8():
    lines = []
    for ds in ("vqa_rad", "path_vqa"):
        d = load("e8_sensitivity_%s.json" % ds)
        if not d:
            continue
        lines.append(r"\begin{tabular}{lcccc}" % ())
        lines.append(r"\toprule")
        lines.append(r"Protocol variant & text oracle & CLIPScore & full MLP & QCE-Net \\")
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{5}{l}{\emph{%s}} \\" % DSLAB[ds])
        for k, v in d.items():
            lines.append(r"%s & %.3f & %.3f & %.3f & %.3f \\"
                         % (k.replace("_", r"\_").replace("=", r"$=$").replace(",", r",\;"),
                            v["shortcut_gbm"], v["clipscore"], v["mlp"], v["qce"]))
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\par\vspace{1em}"])
    if lines:
        open(os.path.join(OUT, "tab_e8.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print("[tab] tab_e8.tex")


def tab_e6():
    rows = []
    for tag in ("q3b", "q7b"):
        for ds in DS:
            d = load("e6_family_%s_%s.json" % (ds, tag))
            if not d:
                continue
            for lv, e in d.items():
                rows.append((tag, ds, lv, e))
    if not rows:
        return
    lines = [r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{llcccc}", r"\toprule",
             r"Backbone & Dataset & Rung & Frozen judge & Frozen judge & Answer \\",
             r" & & & (re-oriented) & (natural sign) & NLL \\",
             r"\midrule"]
    have_se = any("semantic_entropy" in e for _, _, _, e in rows)
    for tag, ds, lv, e in rows:
        j = e.get("zero_shot_judge", {})
        n = e.get("answer_nll", {})
        se = e.get("semantic_entropy", {})
        rf = e.get("radflag", {})
        lines.append(r"%s & %s & %s & %s & %s & %s & %s & %s \\"
                     % (tag.upper().replace("Q", "Qwen2.5-VL-").replace("B", "B"),
                        DSLAB[ds].split("-")[0], lv.replace("_", r"\_"),
                        _nan(j.get("auroc", float("nan"))),
                        _nan(j.get("raw_auroc", float("nan"))),
                        _nan(n.get("auroc", float("nan"))),
                        _nan(se.get("auroc", float("nan"))),
                        _nan(rf.get("auroc", float("nan")))))
    lines.extend([r"\bottomrule", r"\end{tabular}}"])
    open(os.path.join(OUT, "tab_e6.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("[tab] tab_e6.tex (semantic families present: %s)" % have_se)


_extra = dict(tab_leakfeat=tab_leakfeat, tab_bytype=tab_bytype, tab_modality=tab_modality,
              tab_e8=tab_e8, tab_e6=tab_e6)
for _f in _extra.values():
    try:
        _f()
    except Exception as _e:
        print("[tab] WARN", _f.__name__, _e)


def _wrap_fit(name):
    """Wrap a generated tabular in \resizebox so it always fits the text width."""
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return
    s = open(p, encoding="utf-8").read()
    if "resizebox" in s:
        return
    lines = s.split("\n")
    out, depth = [], 0
    for ln in lines:
        if ln.strip().startswith("\\begin{tabular}") and depth == 0:
            out.append(r"\resizebox{\textwidth}{!}{%")
            out.append(ln)
            depth = 1
        elif ln.strip().startswith("\\end{tabular}") and depth == 1:
            out.append(ln + "}")
            depth = 0
        else:
            out.append(ln)
    open(p, "w", encoding="utf-8").write("\n".join(out))


for _n in ("tab_audit.tex", "tab_residual.tex", "tab_controls.tex", "tab_leakfeat.tex",
           "tab_bytype.tex", "tab_modality.tex", "tab_e8.tex", "tab_e6.tex"):
    try:
        _wrap_fit(_n)
    except Exception as _e:
        print("[tab] wrap WARN", _n, _e)
