"""Clinical adjudication audit + decision-curve (net benefit) analysis.

Reads the released per-row matched scores at L4 and writes
results/clinical_adjudication.json, used by the reporting tools.

No model is run; everything is computed from released item-level files.
"""
import csv
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(ROOT, "github", "results", "matched_scores")
OUT = os.path.join(ROOT, "github", "results", "clinical", "adjudication.json")

BENCH = [("vqa_rad", "VQA-RAD"), ("path_vqa", "Path-VQA"), ("omni", "OmniMed"),
         ("slake", "SLAKE"), ("kvasir", "Kvasir-VQA-x1")]

ARTICLES = {"a", "an", "the"}
POLARITY_POS = {"yes", "present", "positive", "true", "y"}
POLARITY_NEG = {"no", "none", "negative", "absent", "false", "n", "not"}
# Conservative clinical synonym groups: only terms that denote the same concept in
# a radiology answer slot. Deliberately small; anything not listed is left alone.
SYNONYMS = [
    {"mri", "mr", "magnetic resonance imaging", "magnetic resonance"},
    {"ct", "computed tomography", "cat scan"},
    {"cxr", "chest x-ray", "chest radiograph", "chest xray", "x-ray chest"},
    {"us", "ultrasound", "sonography", "ultrasonography"},
    {"pet", "positron emission tomography"},
    {"spect", "single photon emission computed tomography"},
    {"x-ray", "xray", "radiograph", "plain film"},
    {"mammogram", "mammography"},
    {"dexa", "dxa", "dual energy x-ray absorptiometry"},
    {"fmri", "functional mri", "functional magnetic resonance imaging"},
    {"dwi", "diffusion weighted imaging"},
    {"adc", "apparent diffusion coefficient"},
    {"t1", "t1 weighted", "t1-weighted"},
    {"t2", "t2 weighted", "t2-weighted"},
    {"flair", "fluid attenuated inversion recovery"},
    {"dicom", "dicom format"},
    {"iv", "intravenous"},
    {"gi", "gastrointestinal"},
]
SYN_MAP = {}
for g in SYNONYMS:
    key = min(g, key=len)
    for t in g:
        SYN_MAP[t] = key


def norm(s):
    s = str(s).strip().lower()
    s = s.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"[^a-z0-9\.\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"^(a|an|the)\s+", "", s)
    # singular/plural of the final token only
    tok = s.split()
    if tok and len(tok[-1]) > 3 and tok[-1].endswith("s") and not tok[-1].endswith("ss"):
        tok[-1] = tok[-1][:-1]
    return " ".join(tok)


def canon(s):
    """rule R1: normalisation + polarity + curated synonym + numeric format."""
    n = norm(s)
    if n in POLARITY_POS:
        return "<POS>"
    if n in POLARITY_NEG:
        return "<NEG>"
    if n in SYN_MAP:
        return SYN_MAP[n]
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(mm|cm|m|%|ml|l|kg|g|hu|gy)?", n)
    if m:
        v = float(m.group(1))
        unit = m.group(2) or ""
        scale = {"cm": 10.0, "m": 1000.0, "ml": 1.0, "l": 1000.0, "g": 1.0, "kg": 1000.0}.get(unit, 1.0)
        return "<NUM>%.3f" % (v * scale if unit in ("cm", "m", "l", "kg") else v)
    return n


def tokens(s):
    return set(norm(s).split()) - ARTICLES


def rule_r1(a, b):
    """concept-equivalent under canonicalisation"""
    return canon(a) == canon(b) and canon(a) != ""


def rule_r2(a, b):
    """independent lexical adjudicator: token containment / high overlap"""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    inter = len(ta & tb)
    jac = inter / len(ta | tb)
    contain = inter == min(len(ta), len(tb))
    return jac >= 0.85 or (contain and min(len(ta), len(tb)) >= 2)


def load_pairs(ds):
    """one row per candidate answer; a pair is the two rows sharing a pid"""
    rows = list(csv.DictReader(open(os.path.join(SCORES, ds + "_L4_fair.csv"), encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[r["pid"]].append(r)
    pairs = []
    for k, v in by.items():
        pos = [r for r in v if r["label"] == "0"]
        neg = [r for r in v if r["label"] == "1"]
        if len(pos) == 1 and len(neg) == 1:
            pairs.append((pos[0], neg[0]))
    assert len(pairs) == len(by), (ds, len(pairs), len(by))
    return rows, pairs


def auroc(y, s):
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    o = np.argsort(s)
    r = np.empty(len(s), float)
    r[o] = np.arange(1, len(s) + 1)
    # average ranks for ties
    uniq, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        sums = np.zeros(len(uniq))
        np.add.at(sums, inv, r)
        r = (sums / cnt)[inv]
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_delta(y, s_keep_mask, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    base = auroc(y, np.asarray(s_keep_mask, float))
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() in (0, len(b)):
            continue
        vals.append(auroc(y[b], np.asarray(s_keep_mask, float)[b]))
    return base, [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def net_benefit(y, s, pt, pi):
    """best net benefit attainable from the score's ranking (no calibration assumed).

    Scans every cut of the empirical ROC and keeps the best value of
    TPR*pi - FPR*(1-pi)*pt/(1-pt), i.e. the decision-curve value at weight pt.
    """
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n1 = max((y == 1).sum(), 1)
    n0 = max((y == 0).sum(), 1)
    tpr = np.concatenate([[0.0], tp / n1])
    fpr = np.concatenate([[0.0], fp / n0])
    w = pt / (1 - pt)
    nb = tpr * pi - fpr * (1 - pi) * w
    return float(nb.max())


def main():
    report = {}
    examples = []
    all_flagged = []
    kappa_pool = []
    for ds, label in BENCH:
        rows, pairs = load_pairs(ds)
        n_flag1 = n_flag2 = n_both = 0
        flagged = []
        for pos, neg in pairs:
            a, b = pos["answer"], neg["answer"]
            f1, f2 = rule_r1(a, b), rule_r2(a, b)
            n_flag1 += f1
            n_flag2 += f2
            n_both += (f1 and f2)
            if f1 or f2:
                flagged.append((pos["gid"], pos["question"].strip().lower(), a, b, bool(f1), bool(f2)))
                flagged[-1] = (pos["gid"], pos["question"], a, b, bool(f1), bool(f2))
                all_flagged.append((label, pos["gid"], pos["question"], a, b, bool(f1), bool(f2)))
                if len(examples) < 24 and ds in ("vqa_rad", "slake", "omni"):
                    examples.append({"benchmark": label, "question": pos["question"],
                                     "expert": a, "borrowed_negative": b, "r1": bool(f1), "r2": bool(f2)})
        n = len(pairs)
        kappa_pool.append((n, n_flag1, n_flag2, n_both))

        # sensitivity: drop flagged pairs (both rows) and recompute AUROC
        drop = {(f[0], f[1].strip().lower()) for f in flagged}
        keep = [i for i, (p, q) in enumerate(pairs)
                if (p["gid"], p["question"].strip().lower()) not in drop]
        y_, s_, s_no = [], [], []
        for i in keep:
            p, q = pairs[i]
            y_ += [0, 1]
            s_ += [float(p["score_full"]), float(q["score_full"])]
            s_no += [float(p["score_noimage"]), float(q["score_noimage"])]
        # all rows without dropping
        y_all, s_all = [], []
        for p, q in pairs:
            y_all += [0, 1]
            s_all += [float(p["score_full"]), float(q["score_full"])]
        a_all = auroc(y_all, s_all)
        a_keep = auroc(y_, s_) if len(keep) > 2 else float("nan")

        # decision curve at deployment prevalence
        dc = {}
        for pt in (0.2, 0.5):
            for pi in (0.05, 0.10, 0.20):
                dc["pt%.1f_pi%.2f" % (pt, pi)] = {
                    "full": net_benefit(y_all, s_all, pt, pi),
                    "noimage": net_benefit(y_all, [float(x) for x in
                                              [r["score_noimage"] for p, q in pairs for r in (p, q)]], pt, pi),
                    "qce": net_benefit(y_all, [float(x) for x in
                                               [r["score_qce"] for p, q in pairs for r in (p, q)]], pt, pi),
                    "flag_all": pi - (1 - pi) * pt / (1 - pt),
                    "flag_none": 0.0,
                }
        report[ds] = {
            "label": label,
            "n_pairs": n,
            "n_rows": len(rows),
            "flagged_r1": n_flag1,
            "flagged_r2": n_flag2,
            "flagged_both": n_both,
            "flagged_union": len(flagged),
            "share_union": round(len(flagged) / n, 4),
            "auroc_before": round(a_all, 4),
            "auroc_after_drop": round(a_keep, 4),
            "delta_after_drop": round(a_keep - a_all, 4) if a_keep == a_keep else None,
            "decision_curve": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                   for kk, vv in v.items()} for k, v in dc.items()},
        }

    # the two adjudicators are complementary: report overlap, not kappa
    tot = sum(x[0] for x in kappa_pool)
    f1 = sum(x[1] for x in kappa_pool)
    f2 = sum(x[2] for x in kappa_pool)
    both = sum(x[3] for x in kappa_pool)
    kappa = both / max(f1 + f2 - both, 1)

    csv_path = os.path.join(ROOT, "github", "results", "clinical", "adjudication_sheet.csv")
    import csv as _csv
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["benchmark", "gid", "question", "expert_answer", "borrowed_negative", "rule_canonical", "rule_overlap"])
        for e in all_flagged:
            w.writerow([e[0], e[1], e[2], e[3], e[4], int(e[5]), int(e[6])])
    out = {"overlap_jaccard_r1_r2": round(kappa, 4), "n_pairs_total": tot,
           "flagged_r1_total": f1, "flagged_r2_total": f2,
           "benchmarks": report, "examples": examples}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    print("overlap", round(kappa, 4), "pairs", tot, "flag_r1", f1, "flag_r2", f2)
    for ds, label in BENCH:
        r = report[ds]
        print("%-14s pairs %5d rows %6d flagged %3d (%.2f%%) auroc %.4f -> %.4f (d %.4f)" %
              (label, r["n_pairs"], r["n_rows"], r["flagged_union"], 100 * r["share_union"],
               r["auroc_before"], r["auroc_after_drop"], r["delta_after_drop"] or 0))
        for k in ("pt0.2_pi0.05", "pt0.2_pi0.10", "pt0.5_pi0.10"):
            d = r["decision_curve"][k]
            print("    %s full %.3f noimg %.3f qce %.3f flag_all %.3f" % (k, d["full"], d["noimage"], d["qce"], d["flag_all"]))
    print("sheet:", csv_path, len(all_flagged), "rows")
    print("---- flagged examples ----")
    for e in examples[:14]:
        print(e["benchmark"], "|", e["question"][:60], "|", e["expert"], "VS", e["borrowed_negative"], e["r1"], e["r2"])


if __name__ == "__main__":
    main()
