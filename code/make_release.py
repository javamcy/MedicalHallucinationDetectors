#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the released matched-protocol resource from the run caches.

Writes, for every benchmark with an L4 cache:
  results/pairs/<ds>_L4_fair.jsonl     one line per matched pair (pos/neg answer, match quality, split)
  results/matched_scores/<ds>_L4_fair.csv   one line per test row with every detector score
  results/audit/construction_summary.json   per-benchmark negative-construction audit

Test-set scores are what reproduce the reported AUROCs; the pair file covers all
rows so the construction itself can be audited end to end. No images are copied.

Usage: python make_release.py [dataset ...]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
LEVEL = "L4_fair"


def cache_dirs():
    cands = [Path("/root/bench_exp/cache"), Path("/root/hallu_bench/cache"),
             REPO.parent / "paper_crossdomain/results/remote_artefacts/cache",
             REPO.parent / "_localrun" / "cache",
             REPO.parent / "_localrun" / "hallu" / "cache",
             Path("/root/bench_exp/cache"), Path("/root/hallu/cache")]
    out = []
    for c in cands:
        if c.exists() and str(c) not in out:
            out.append(c)
    return out


def find(name, caches):
    for c in caches:
        p = Path(c) / name
        if p.exists():
            return p
    return None


def load_scores(ds, caches):
    tp = find("trained_%s_%s.npz" % (ds, LEVEL), caches)
    npz = find("trained_noimg_%s_%s.npz" % (ds, LEVEL), caches)
    if tp is None:
        return None
    T = np.load(str(tp), allow_pickle=True)
    out = {"idx_te": np.asarray(T["idx_te"], int), "y": np.asarray(T["y"], int)}
    out["full"] = np.asarray(T["mlp_score"], float)
    if "qce_score" in T.files:
        out["qce"] = np.asarray(T["qce_score"], float)
    if npz is not None:
        N = np.load(str(npz), allow_pickle=True)
        for k in ("score", "noimg_score"):
            if k in N.files:
                out["noimage"] = np.asarray(N[k], float)
                break
    elif "noimg_score" in T.files:
        out["noimage"] = np.asarray(T["noimg_score"], float)
    return out


def split_labels(n, idx_tr, idx_va, idx_te):
    lab = np.array(["unused"] * n, dtype=object)
    lab[idx_tr] = "train"
    lab[idx_va] = "val"
    lab[idx_te] = "test"
    return lab


def build(ds, caches, out_root):
    rp = find("rows_%s_%s.json" % (ds, LEVEL), caches)
    tp = find("trained_%s_%s.npz" % (ds, LEVEL), caches)
    fp = find("feat_%s_%s.npz" % (ds, LEVEL), caches)
    if rp is None or (tp is None and fp is None):
        print("[REL] %s: no cache, skip" % ds, flush=True)
        return None
    rows = json.load(open(rp, encoding="utf-8"))
    if tp is not None:
        T = np.load(str(tp), allow_pickle=True)
    if tp is not None and "idx_tr" in T.files:
        tr, va, te = (np.asarray(T[k], int) for k in ("idx_tr", "idx_va", "idx_te"))
    else:
        F = np.load(str(fp), allow_pickle=True)
        tr, va, te = (np.asarray(F[k], int) for k in ("idx_tr", "idx_va", "idx_te"))
    lab = split_labels(len(rows), tr, va, te)

    # ---- pair file: group rows by pid (rows are emitted as +, -, +, - ...) ----
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["pid"], []).append(i)
    pairs = []
    for pid, ks in sorted(by.items()):
        if len(ks) != 2:
            continue
        a, b = ks
        pos = a if rows[a]["label"] == 0 else b
        neg = b if pos == a else a
        if rows[pos]["label"] != 0 or rows[neg]["label"] != 1:
            continue
        pairs.append((pid, dict(pid=pid, gid=rows[pos]["gid"],
                                question=rows[pos]["question"],
                                pos_answer=rows[pos]["answer"],
                                neg_answer=rows[neg]["answer"],
                                question_type=rows[pos]["qt"], match=rows[pos]["match"],
                                split=str(lab[pos]))))
    pdir = out_root / "pairs"
    pdir.mkdir(parents=True, exist_ok=True)
    pp = pdir / ("%s_%s.jsonl" % (ds, LEVEL))
    with open(pp, "w", encoding="utf-8") as f:
        for _, d in pairs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- score file (test rows only) ----
    S = load_scores(ds, caches)
    sdir = out_root / "matched_scores"
    sdir.mkdir(parents=True, exist_ok=True)
    sp = sdir / ("%s_%s.csv" % (ds, LEVEL))
    if S is not None:
        te_s = S["idx_te"]
        cols = ["pid", "gid", "question", "answer", "label", "question_type", "match", "split"]
        dets = [k for k in ("full", "qce", "noimage") if k in S]
        with open(sp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols + ["score_" + k for k in dets])
            for j, i in enumerate(te_s):
                r = rows[i]
                w.writerow([r["pid"], r["gid"], r["question"], r["answer"], r["label"],
                            r["qt"], r["match"], "test"] + ["%.12g" % S[k][j] for k in dets])
    audit = {"benchmark": ds, "level": LEVEL, "n_rows": len(rows), "n_pairs": len(pairs),
             "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
             "match_quality": {q: sum(1 for _, d in pairs if d["match"] == q)
                               for q in sorted({d["match"] for _, d in pairs})}}
    print("[REL] %s: %d pairs, %d test scores -> %s" % (ds, len(pairs), len(te), sp.name), flush=True)
    return audit


def main():
    dss = sys.argv[1:] or ["vqa_rad", "path_vqa", "omni", "slake"]
    caches = cache_dirs()
    out_root = REPO / "results"
    audits = {}
    for ds in dss:
        a = build(ds, caches, out_root)
        if a:
            audits[ds] = a
    ad = out_root / "audit"
    ad.mkdir(parents=True, exist_ok=True)
    p = ad / "construction_summary.json"
    old = {}
    if p.exists():
        old = json.load(open(p, encoding="utf-8"))
    old.update(audits)
    json.dump(old, open(p, "w"), indent=2)
    print("[REL] wrote %s (%d benchmarks)" % (p, len(old)), flush=True)


if __name__ == "__main__":
    main()
