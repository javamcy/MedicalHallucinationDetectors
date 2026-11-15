#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reference implementation of the six-item reporting standard (Section 3.6).

Input is what a paper already has: a pair file (one line per matched pair) and a
score file (one line per row, one column per detector). The tool prints, and can
write, a Markdown report card that answers each of the six items and marks it
pass / fail / not reported.

  > python reporting_checklist.py --pairs pairs/vqa_rad_L4_fair.jsonl \
        --scores matched_scores/vqa_rad_L4_fair.csv \
        --detector score_full --text-only score_noimage \
        --leakage-control 0.498 --prevalence 0.10 --out report_card.md

Exit code is 0 when all items that can be checked automatically pass.

The six items (Section 3.6):
  1 negative-construction rule and match-failure fraction
  2 image-level split, zero test images seen in training
  3 text-only control
  4 permuted-label leakage control
  5 operating characteristics (sensitivity, specificity, ratio, precision)
  6 confidence interval and full-set results
"""
import argparse
import csv
import json
import sys
from collections import Counter

import numpy as np


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def boot_ci(y, s, n=1000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        b = rng.randint(0, len(y), len(y))
        if len(set(y[b].tolist())) < 2:
            continue
        vals.append(auroc(y[b], s[b]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def youden(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(y, s)
    k = int(np.argmax(tpr - fpr))
    return float(th[k]), float(tpr[k]), float(1 - fpr[k])


def binormal_ppv(A, pi):
    from math import erf, sqrt
    def Phi(x):
        return 0.5 * (1 + erf(x / sqrt(2)))
    from math import sqrt as _s
    # inverse normal via bisection on Phi
    def Phi_inv(p):
        lo, hi = -10.0, 10.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if Phi(mid) < p:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    d = _s(2) * Phi_inv(A)
    tp = pi * Phi(d / 2)
    fp = (1 - pi) * (1 - Phi(d / 2))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def load_scores(path):
    with open(path, encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--detector", default="score_full")
    ap.add_argument("--text-only", default="score_noimage")
    ap.add_argument("--leakage-control", type=float, default=None,
                    help="test AUROC of a model trained on permuted labels, same pipeline")
    ap.add_argument("--prevalence", type=float, default=0.10)
    ap.add_argument("--match-fail-threshold", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pairs = load_pairs(a.pairs)
    rows = load_scores(a.scores)
    L = []
    ok = True

    L.append("# Reporting checklist\n")
    L.append("Benchmark file: `%s`  \nScore file: `%s`  \nDetector: `%s`\n" %
             (a.pairs, a.scores, a.detector))

    # ---- 1 negative construction ----
    L.append("## 1. Negative-construction rule\n")
    mq = Counter(p.get("match", "unknown") for p in pairs)
    n = sum(mq.values())
    fallback = sum(v for k, v in mq.items() if "fallback" in k or k in ("none", "unmatched"))
    frac = fallback / n if n else float("nan")
    L.append("Pairs: %d. Match-quality distribution: %s.\n" % (n, dict(mq)))
    if frac <= a.match_fail_threshold:
        L.append("- **PASS** -- matched pairs %.1f%%, unmatched/fallback %.1f%% (threshold %.0f%%).\n" %
                 (100 * (1 - frac), 100 * frac, 100 * a.match_fail_threshold))
    else:
        ok = False
        L.append("- **FAIL** -- %.1f%% of pairs fell back to unmatched negatives (threshold %.0f%%).\n" %
                 (100 * frac, 100 * a.match_fail_threshold))

    # ---- 2 image-level split ----
    L.append("## 2. Image-level split\n")
    tr_g = {p["gid"] for p in pairs if p.get("split") == "train"}
    te = [p for p in pairs if p.get("split") == "test"]
    leak = sum(1 for p in te if p["gid"] in tr_g)
    if te:
        L.append("Test pairs: %d. Test pairs whose image also appears in training: %d (%.2f%%).\n" %
                 (len(te), leak, 100 * leak / len(te)))
        if leak == 0:
            L.append("- **PASS** -- no test image is in the training split.\n")
        else:
            ok = False
            L.append("- **FAIL** -- the split leaks source images; report AUROC at the image level only.\n")
    else:
        L.append("- **NOT REPORTED** -- the pair file carries no `split` field.\n")
        ok = False

    # ---- 3 text-only control ----
    L.append("## 3. Text-only control\n")
    det = np.array([float(r[a.detector]) for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    if a.text_only in rows[0]:
        txt = np.array([float(r[a.text_only]) for r in rows])
        A_d, A_t = auroc(y, det), auroc(y, txt)
        L.append("Full detector AUROC %.4f, question+answer-only AUROC %.4f (%.0f%% of the full value).\n" %
                 (A_d, A_t, 100 * A_t / A_d if A_d else float("nan")))
        L.append("- **PASS** -- the control is reported.\n")
    else:
        A_d = auroc(y, det)
        L.append("- **NOT REPORTED** -- no text-only score column present.\n")
        ok = False

    # ---- 4 leakage control ----
    L.append("## 4. Permuted-label leakage control\n")
    if a.leakage_control is None:
        L.append("- **NOT REPORTED** -- train the same pipeline on permuted labels and pass its test AUROC "
                 "with `--leakage-control`; it should be 0.50 +- 0.02.\n")
        ok = False
    else:
        good = abs(a.leakage_control - 0.5) <= 0.02
        L.append("Permuted-label AUROC %.4f.\n" % a.leakage_control)
        if good:
            L.append("- **PASS** -- the pipeline generates no signal from shuffled labels.\n")
        else:
            ok = False
            L.append("- **FAIL** -- the pipeline returns signal on permuted labels.\n")

    # ---- 5 operating characteristics ----
    L.append("## 5. Operating characteristics\n")
    t, sens, spec = youden(y, det)
    L.append("Youden threshold %.4f: sensitivity %.3f, specificity %.3f, ratio %.2f:1"
             " (missed hallucinations : false alarms).\n" %
             (t, sens, spec, (1 - sens) / max(1e-9, 1 - spec)))
    pred = det > t
    emp_ppv = float((pred & (y == 1)).sum()) / max(1, pred.sum())
    L.append("Precision at the stated prevalence %.0f%% (binormal model): %.1f%%.\n" %
             (100 * a.prevalence, 100 * binormal_ppv(A_d, a.prevalence)))
    L.append("Observed precision in this test set: %.1f%% (positive class rate %.1f%%).\n" %
             (100 * emp_ppv, 100 * y.mean()))
    L.append("- **PASS** -- operating point reported.\n")

    # ---- 6 CI and full-set ----
    L.append("## 6. Confidence interval and full-set results\n")
    lo, hi = boot_ci(y, det, n=1000, seed=0)
    L.append("Test rows: %d. AUROC %.4f, 95%% bootstrap CI [%.4f, %.4f].\n" % (len(y), A_d, lo, hi))
    L.append("- **PASS** -- interval and full-set size reported.\n")

    L.append("\n---\n**Overall: %s**\n" % ("PASS" if ok else "INCOMPLETE"))
    text = "\n".join(L)
    print(text)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
