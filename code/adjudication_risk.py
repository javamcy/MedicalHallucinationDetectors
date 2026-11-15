# -*- coding: utf-8 -*-
"""Bound the clinical-adjudication risk: Wilson CI plus a 10x sensitivity analysis."""
import csv
import io
import json
import math
import os
import numpy as np
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(ROOT, "github", "results", "matched_scores")
OUT = os.path.join(ROOT, "github", "results", "clinical", "adjudication_risk.json")

FLAGGED = {"vqa_rad": 0, "path_vqa": 1, "omni": 1, "slake": 6, "kvasir": 25}
NAMES = {"vqa_rad": "VQA-RAD", "path_vqa": "Path-VQA", "omni": "OmniMed",
         "slake": "SLAKE", "kvasir": "Kvasir-VQA-x1"}


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    u, inv, c = np.unique(s, return_inverse=True, return_counts=True)
    if (c > 1).any():
        sm = np.zeros(len(u)); np.add.at(sm, inv, r); r = (sm / c)[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(ds):
    rows = list(csv.DictReader(open(os.path.join(SCORES, ds + "_L4_fair.csv"), encoding="utf-8")))
    by = defaultdict(list)
    for r in rows:
        by[r["pid"]].append(r)
    pairs = []
    for _, v in by.items():
        p = [x for x in v if x["label"] == "0"]
        q = [x for x in v if x["label"] == "1"]
        if len(p) == 1 and len(q) == 1:
            pairs.append((p[0], q[0]))
    return pairs


def arrays(pairs):
    y, sf, sn = [], [], []
    for p, q in pairs:
        y += [0, 1]
        sf += [float(p["score_full"]), float(q["score_full"])]
        sn += [float(p["score_noimage"]), float(q["score_noimage"])]
    return np.asarray(y), np.asarray(sf), np.asarray(sn)


tot = sum(FLAGGED.values())
n_tot = 0
report = {}
rng = np.random.default_rng(20261114)
for ds, flag in FLAGGED.items():
    pairs = load(ds)
    n = len(pairs)
    n_tot += n
    y, sf, sn = arrays(pairs)
    base_f = auc(y, sf)
    base_n = auc(y, sn)
    base_d = base_f - base_n
    # 10x the observed flagged rate, applied to random pairs (stress test)
    frac = min(0.20, 10 * flag / n)
    if frac > 0:
        reps = 200
        df, dd = [], []
        m = max(1, int(round(n * frac)))
        for _ in range(reps):
            drop = set(rng.choice(n, m, replace=False).tolist())
            keep = np.asarray([i for i in range(n) if i not in drop])
            ix = np.repeat(keep * 2, 2) + np.tile([0, 1], len(keep))
            f2 = auc(y[ix], sf[ix])
            n2 = auc(y[ix], sn[ix])
            df.append(f2 - base_f)
            dd.append((f2 - n2) - base_d)
        res = dict(frac=round(frac, 4), max_abs_dAUROC=round(max(abs(x) for x in df), 4),
                   p95_abs_dAUROC=round(float(np.percentile(np.abs(df), 95)), 4),
                   max_abs_change_image_delta=round(max(abs(x) for x in dd), 4))
    else:
        res = dict(frac=0.0, max_abs_dAUROC=0.0, p95_abs_dAUROC=0.0,
                   max_abs_change_image_delta=0.0)
    report[ds] = dict(label=NAMES[ds], n_pairs=n, flagged=flag,
                      auroc_full=round(base_f, 4), auroc_noimage=round(base_n, 4),
                      image_delta=round(base_d, 4), stress=res)

lo, hi = wilson(tot, n_tot)
out = dict(flagged_total=tot, pairs_total=n_tot, share=round(tot / n_tot, 5),
           wilson95=[round(lo, 5), round(hi, 5)], benchmarks=report)
json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
print("flagged %d / %d = %.3f%%  Wilson95 = [%.3f%%, %.3f%%]" %
      (tot, n_tot, 100 * tot / n_tot, 100 * lo, 100 * hi))
for ds, r in report.items():
    s = r["stress"]
    print("%-14s flag %2d/%4d  stress-drops %.1f%% pairs -> max|dAUROC| %.4f  max|d(delta)| %.4f" %
          (r["label"], r["flagged"], r["n_pairs"], 100 * s["frac"],
           s["max_abs_dAUROC"], s["max_abs_change_image_delta"]))
