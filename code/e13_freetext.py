#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E13: does the conclusion survive free-text answers?

The four original benchmarks answer in a word or two. Kvasir-VQA-x1 answers are
sentences (median 9 tokens), so the same ladder can be read as a report-sentence
benchmark. This script stratifies the matched-rung scores by answer length and
reports, per stratum: the pair count, the trained head with and without the image,
the lexical oracle, and the single lexical feature with the highest AUROC.
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C

DS = "kvasir"
LEVEL = "L4_fair"
STRATA = [(1, 6), (7, 9), (10, 12), (13, 99)]


def find(name):
    for c in (Path(C.CACHE), Path("/root/hallu/cache")):
        p = Path(c) / name
        if p.exists():
            return p
    return None


def main():
    feat = np.load(str(find("feat_%s_%s.npz" % (DS, LEVEL))), allow_pickle=True)
    sc = np.load(str(find("shortcut_%s_%s.npz" % (DS, LEVEL))), allow_pickle=True)
    tr = np.load(str(find("trained_%s_%s.npz" % (DS, LEVEL))), allow_pickle=True)
    ni = np.load(str(find("trained_noimg_%s_%s.npz" % (DS, LEVEL))), allow_pickle=True)
    rows = json.load(open(str(find("rows_%s_%s.json" % (DS, LEVEL))), encoding="utf-8"))
    te = np.asarray(feat["idx_te"], int)
    pos = np.full(len(rows), -1, int)
    pos[te] = np.arange(len(te))
    y = np.asarray(feat["y"], int)
    lens = np.array([C.token_len(r["answer"]) for r in rows])
    names = [str(n) for n in sc["names"]]
    X = np.asarray(sc["X"], np.float32)
    mlp = np.asarray(tr["mlp_score"], float)
    qce = np.asarray(tr["qce_score"], float)
    noimg = np.asarray(ni["score"], float)
    gb = np.asarray(sc["s_gb"], float)

    out = {"ds": DS, "level": LEVEL, "n_rows": int(len(rows)), "strata": []}
    print("%-9s %6s %6s %6s %6s %6s  %s" % ("tokens", "pairs", "full", "qce", "noimg", "oracle", "top lexical"))
    for lo, hi in STRATA:
        m = np.zeros(len(rows), bool)
        m[te] = True
        m &= (lens >= lo) & (lens <= hi)
        idx = np.where(m)[0]
        if len(idx) < 60 or len(set(y[idx])) < 2:
            print("%-9s %6d  (too few)" % ("%d-%d" % (lo, hi), len(idx) // 2))
            continue
        yi = y[idx]
        loc = pos[idx]
        ent = {"lo": lo, "hi": hi, "n_rows": int(len(idx)), "n_pairs": int(len(idx) // 2),
               "full": float(C.auroc(yi, mlp[loc])),
               "qce": float(C.auroc(yi, qce[loc])),
               "noimage": float(C.auroc(yi, noimg[loc])),
               "oracle_gbm": float(C.auroc(yi, gb[idx]))}
        best, bv = None, -1
        for j, nm in enumerate(names):
            a = float(C.best_orientation(yi, X[idx, j])[1])
            if a > bv:
                best, bv = nm, a
        ent["top_lexical_feature"] = best
        ent["top_lexical_auroc"] = float(bv)
        out["strata"].append(ent)
        print("%-9s %6d %6.3f %6.3f %6.3f %6.3f  %s %.3f" %
              ("%d-%d" % (lo, hi), ent["n_pairs"], ent["full"], ent["qce"], ent["noimage"],
               ent["oracle_gbm"], best, bv))

    dst = Path(C.OUT) / "e13_freetext.json"
    json.dump(out, open(dst, "w"), indent=2, default=C._default)
    print("->", dst)


if __name__ == "__main__":
    main()
