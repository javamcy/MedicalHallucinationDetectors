#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E12: grounding controls -- does the image channel actually carry the decisions?

At the matched rung (L4) we run three controls per benchmark:

  (1) test-time image shuffle : keep the trained head, replace each test row's
      visual evidence (g, e_q, e_a) by that of an unrelated case. If the decisions
      are image-driven, AUROC drops and a large share of decisions flips.
  (2) full image shuffle      : retrain with shuffled visual evidence in train,
      val and test. The image channel is noise; the ceiling is the text-only head.
  (3) blank (constant) image  : a content-free image contributes a constant vector,
      so the blank-image control coincides analytically with the text-only head.
      We verify this numerically instead of asserting it.

Images are shuffled at the case (gid) level, so the two answers of a pair still
share one image, as they do in the benchmark.

Usage: python e12_controls.py [dataset ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C

LEVEL = "L4_fair"


def cache_dirs():
    root = Path(__file__).resolve().parent.parent
    cands = [Path("/root/bench_exp/cache"), Path("/root/hallu_bench/cache"),
             root / "paper_crossdomain/results/remote_artefacts/cache", Path(C.CACHE)]
    cands += [root.parent / "_localrun" / "cache", root.parent / "_localrun" / "hallu" / "cache"]
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


def out_dir():
    root = Path(__file__).resolve().parent.parent
    for c in (root / "results" / "controls", Path(C.OUT)):
        try:
            c.mkdir(parents=True, exist_ok=True)
            (c / ".w").write_text("x")
            (c / ".w").unlink()
            return c
        except Exception:
            continue
    return Path(".")


def youden(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(y, s)
    return float(th[int(np.argmax(tpr - fpr))])


def donor_map(gids, seed=0):
    """gid -> donor gid, a derangement-free permutation (a gid never maps to itself)."""
    uniq = sorted(set(gids))
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(uniq))
    # fix fixed points by a cyclic shift
    for i in range(len(uniq)):
        if perm[i] == i:
            perm[i] = (i + 1) % len(uniq)
    return {uniq[i]: uniq[perm[i]] for i in range(len(uniq))}


def run(ds, caches):
    fp = find("feat_%s_%s.npz" % (ds, LEVEL), caches)
    rp = find("rows_%s_%s.json" % (ds, LEVEL), caches)
    if fp is None or rp is None:
        print("[E12] %s: cache missing, skip" % ds, flush=True)
        return None
    F = np.load(str(fp), allow_pickle=True)
    rows = json.load(open(rp, encoding="utf-8"))
    tr, va, te = (np.asarray(F[k], int) for k in ("idx_tr", "idx_va", "idx_te"))
    y = np.asarray(F["y"], int)
    G, A, Q, EQ, EA = (F[k] for k in ("G", "A", "Q", "EQ", "EA"))

    gids = [r["gid"] for r in rows]
    dm = donor_map(gids, seed=0)
    # index-based donor lookup (avoid O(n^2))
    first = {}
    for i, g in enumerate(gids):
        first.setdefault(g, i)
    donor = np.array([first[dm[g]] for g in gids], int)

    Gs, EQs, EAs = G[donor], EQ[donor], EA[donor]
    Xm = np.concatenate([G, A, Q], 1)
    Xc = np.concatenate([G, A, Q, EQ, EA], 1)
    Xn = np.concatenate([A, Q], 1)
    Xms = np.concatenate([Gs, A, Q], 1)
    Xcs = np.concatenate([Gs, A, Q, EQs, EAs], 1)
    Gm = np.broadcast_to(G.mean(0, keepdims=True), G.shape)
    EQm = np.broadcast_to(EQ.mean(0, keepdims=True), EQ.shape)
    EAm = np.broadcast_to(EA.mean(0, keepdims=True), EA.shape)
    Xcb = np.concatenate([Gm, A, Q, EQm, EAm], 1)

    res = {"ds": ds, "level": LEVEL, "n_test": int(len(te)), "heads": {}}

    def add(name, p, base=None):
        a = float(C.auroc(y[te], p))
        ent = {"test_auroc": a}
        if base is not None:
            t = youden(y[te], base)
            ent["frac_decisions_changed_vs_reference"] = float(np.mean((p > t) != (base > t)))
            ent["mean_abs_score_change"] = float(np.mean(np.abs(p - base)))
        res["heads"][name] = ent
        return ent

    for nm, X, Xs in (("full", Xm, Xms), ("qce", Xc, Xcs)):
        p_ref, _, _ = C.train_mlp_scores(X[tr], X[va], X[te], y[tr], y[va], y[te], seed=0)
        p_ts, _, _ = C.train_mlp_scores(X[tr], X[va], Xs[te], y[tr], y[va], y[te], seed=0)
        p_tr, _, _ = C.train_mlp_scores(Xs[tr], Xs[va], Xs[te], y[tr], y[va], y[te], seed=0)
        add(nm + "_reference", p_ref)
        add(nm + "_test_shuffle", p_ts, p_ref)     # trained matched, tested on wrong images
        add(nm + "_train_shuffle", p_tr, p_ref)    # trained and tested on wrong images
        if nm == "full":
            p_b, _, _ = C.train_mlp_scores(Xcb[tr], Xcb[va], Xcb[te], y[tr], y[va], y[te], seed=0)
            add("full_blank_image", p_b, p_ref)
    p_n, _, _ = C.train_mlp_scores(Xn[tr], Xn[va], Xn[te], y[tr], y[va], y[te], seed=0)
    add("noimage", p_n)

    # frozen CLIPScore under the same shuffle
    cs = (G * A).sum(1)
    css = (Gs * A).sum(1)
    res["frozen_clipscore"] = {"matched": float(C.best_orientation(y[te], cs[te])[1]),
                               "shuffled": float(C.best_orientation(y[te], css[te])[1])}
    print("[E12] %s: %s" % (ds, {k: round(v["test_auroc"], 4)
                                 for k, v in res["heads"].items()}), flush=True)
    return res


def main():
    dss = sys.argv[1:] or ["vqa_rad", "path_vqa", "omni", "slake"]
    caches = cache_dirs()
    dst = out_dir() / "e12_controls.json"
    res = json.load(open(dst, encoding="utf-8")) if dst.exists() else {}
    for ds in dss:
        r = run(ds, caches)
        if r:
            res[ds] = r
            json.dump(res, open(dst, "w"), indent=2, default=C._default)
    print("[E12] done -> %s" % (out_dir() / "e12_controls.json"), flush=True)


if __name__ == "__main__":
    main()
