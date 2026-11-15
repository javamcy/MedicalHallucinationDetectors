#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E10: training-seed variance at the matched rung (L4).

The confidence intervals so far are pair-level bootstrap intervals at a single training
seed. A reviewer can object that the interval ignores the variance of the trained
head itself. This script retrains the same three heads with five seeds and reports
mean +- SD and a 95% interval across seeds (t distribution, df = n_seed - 1).

Heads (identical architecture and schedule to e1_protocol.py):
  full     [g, a, q]                 frozen-CLIP MLP
  qce      [g, a, q, e_q, e_a]       evidence-channel model
  noimage  [a, q]                    text-only control

Usage: python e10_seeds.py [dataset ...] [--seeds 0,1,2,3,4]
"""
import json
import sys
from pathlib import Path

import numpy as np

import common as C

SEEDS = [0, 1, 2, 3, 4]
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
    for c in (root / "results" / "seeds", Path(C.OUT)):
        try:
            c.mkdir(parents=True, exist_ok=True)
            (c / ".w").write_text("x")
            (c / ".w").unlink()
            return c
        except Exception:
            continue
    return Path(".")


def t_ci(vals, alpha=0.05):
    vals = np.asarray(vals, float)
    n = len(vals)
    if n < 2:
        return float("nan"), float("nan")
    m, sd = float(vals.mean()), float(vals.std(ddof=1))
    try:
        from scipy import stats
        q = float(stats.t.ppf(1 - alpha / 2, n - 1))
    except Exception:
        q = 2.776 if n == 5 else 1.96
    return m - q * sd / np.sqrt(n), m + q * sd / np.sqrt(n)


def run(ds, caches, seeds):
    fp = find("feat_%s_%s.npz" % (ds, LEVEL), caches)
    if fp is None:
        print("[E10] %s: no feature cache, skip" % ds, flush=True)
        return None
    F = np.load(str(fp), allow_pickle=True)
    tr, va, te = (np.asarray(F[k], int) for k in ("idx_tr", "idx_va", "idx_te"))
    y = np.asarray(F["y"], int)
    X = {
        "full": np.concatenate([F["G"], F["A"], F["Q"]], 1),
        "qce": np.concatenate([F["G"], F["A"], F["Q"], F["EQ"], F["EA"]], 1),
        "noimage": np.concatenate([F["A"], F["Q"]], 1),
    }
    res = {"ds": ds, "level": LEVEL, "seeds": list(seeds), "n_test": int(len(te)),
           "pos_test": int(y[te].sum()), "heads": {}}
    for name, Xh in X.items():
        rows = []
        for sd in seeds:
            p, v, ep = C.train_mlp_scores(Xh[tr], Xh[va], Xh[te], y[tr], y[va], y[te], seed=sd)
            rows.append({"seed": sd, "test_auroc": float(C.auroc(y[te], p)),
                         "val_auroc": float(v), "best_epoch": int(ep)})
            print("[E10] %s %s seed=%d test=%.4f val=%.4f ep=%d" %
                  (ds, name, sd, rows[-1]["test_auroc"], v, ep), flush=True)
        vals = [r["test_auroc"] for r in rows]
        lo, hi = t_ci(vals)
        res["heads"][name] = {
            "per_seed": rows,
            "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)), "max": float(np.max(vals)),
            "ci95_seed": [lo, hi],
        }
        print("[E10] %s %-8s mean=%.4f sd=%.4f ci95=[%.4f,%.4f] range=[%.4f,%.4f]" %
              (ds, name, res["heads"][name]["mean"], res["heads"][name]["sd"], lo, hi,
               res["heads"][name]["min"], res["heads"][name]["max"]), flush=True)
    # paired seed differences
    a = [r["test_auroc"] for r in res["heads"]["qce"]["per_seed"]]
    b = [r["test_auroc"] for r in res["heads"]["full"]["per_seed"]]
    d = np.asarray(a) - np.asarray(b)
    lo, hi = t_ci(d)
    res["delta_qce_minus_full"] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                   "ci95_seed": [lo, hi],
                                   "n_seeds_qce_better": int((d > 0).sum())}
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = SEEDS
    for a in sys.argv[1:]:
        if a.startswith("--seeds"):
            seeds = [int(x) for x in a.split("=", 1)[1].split(",")]
    dss = args or ["vqa_rad", "path_vqa", "omni", "slake"]
    caches = cache_dirs()
    print("[E10] caches: %s" % caches, flush=True)
    dst = out_dir() / "e10_seeds.json"
    res = json.load(open(dst, encoding="utf-8")) if dst.exists() else {}
    for ds in dss:
        r = run(ds, caches, seeds)
        if r:
            res[ds] = r
            json.dump(res, open(dst, "w"), indent=2, default=C._default)
    print("[E10] done -> %s" % (out_dir() / "e10_seeds.json"), flush=True)


if __name__ == "__main__":
    main()
