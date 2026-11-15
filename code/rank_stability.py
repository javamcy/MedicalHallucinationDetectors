"""Rank stability of hallucination detectors across protocol rungs.

Question: the literature ranks detectors by AUROC on a benchmark. If that
ranking flips when the answer-pool leak is removed, the ranking is reporting
a property of the protocol, not of the detector.

Inputs are the e6_family_<ds>_<tag>.json files written by the ladder runs.
The shipped results/rank_stability.json is the artifact of record and covers two
cross-domain benchmarks in addition to these three; this script recomputes the
medical rows and writes them to results/rank_stability_recomputed.json so the
artifact of record is never overwritten.
Output: Kendall tau and Spearman rho between the L0 ordering and the L4
ordering of the detector families, per dataset and backbone, plus the
identities of the largest rank changes.
"""
import glob
import json
import os
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.environ.get("LADDER_DIR", os.path.join(ROOT, "results", "ladder"))

PRETTY = {
    "semantic_entropy": "semantic entropy",
    "radflag": "RadFlag",
    "answer_consistency_max": "consistency (max)",
    "answer_consistency_mean": "consistency (mean)",
    "greedy_agreement": "greedy agreement",
    "consistency_token_overlap": "token overlap",
    "zero_shot_judge": "zero-shot judge",
    "answer_nll": "answer NLL",
}
DS_PRETTY = {"mmmu": "MMMU", "scienceqa": "ScienceQA", "vqa_rad": "VQA-RAD",
             "path_vqa": "Path-VQA", "omni": "OmniMedVQA"}


def kendall_tau(a, b):
    """Kendall tau-a over paired ranks, ties counted as half concordant."""
    n = len(a)
    if n < 2:
        return float("nan")
    num = den = 0
    for i, j in combinations(range(n), 2):
        da, db = a[i] - a[j], b[i] - b[j]
        if da == 0 or db == 0:
            continue
        den += 1
        num += 1 if (da * db) > 0 else -1
    return float("nan") if den == 0 else num / den


def spearman(a, b):
    n = len(a)
    ra = {v: i for i, v in enumerate(sorted(a))}
    rb = {v: i for i, v in enumerate(sorted(b))}
    xa = [ra[v] for v in a]
    xb = [rb[v] for v in b]
    mx, my = sum(xa) / n, sum(xb) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xa, xb))
    vx = sum((x - mx) ** 2 for x in xa) ** 0.5
    vy = sum((y - my) ** 2 for y in xb) ** 0.5
    return float("nan") if vx * vy == 0 else cov / (vx * vy)


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(SRC, "e6_family_*.json"))):
        base = os.path.basename(path)[:-5]
        parts = base.split("_")
        ds, tag = "_".join(parts[2:-1]), parts[-1]
        d = json.load(open(path))
        if "L0_leaky" not in d or "L4_fair" not in d:
            continue
        fams = [k for k, v in d["L0_leaky"].items()
                if isinstance(v, dict) and "auroc" in v]
        fams = [f for f in fams if f in d["L4_fair"]]
        if len(fams) < 3:
            continue
        a0 = [d["L0_leaky"][f]["auroc"] for f in fams]
        a4 = [d["L4_fair"][f]["auroc"] for f in fams]
        tau = kendall_tau(a0, a4)
        rho = spearman(a0, a4)
        r0 = {f: i for i, f in enumerate(sorted(fams, key=lambda f: -d["L0_leaky"][f]["auroc"]), 1)}
        r4 = {f: i for i, f in enumerate(sorted(fams, key=lambda f: -d["L4_fair"][f]["auroc"]), 1)}
        moves = sorted(((abs(r0[f] - r4[f]), f) for f in fams), reverse=True)
        biggest = moves[0]
        rows.append(dict(ds=ds, tag=tag, n_fams=len(fams), tau=tau, rho=rho,
                         top_L0=min(r0, key=r0.get), top_L4=min(r4, key=r4.get),
                         biggest_move="%s (%d->%d)" % (PRETTY.get(biggest[1], biggest[1]),
                                                       r0[biggest[1]], r4[biggest[1]]),
                         a0=dict(zip(fams, a0)), a4=dict(zip(fams, a4))))
    out = os.path.join(ROOT, "results", "rank_stability_recomputed.json")
    json.dump(rows, open(out, "w"), indent=2)

    print("%-12s %-8s %5s %7s %7s  %-28s %-28s" % (
        "dataset", "backbone", "fams", "tau", "rho", "top at L0", "top at L4"))
    for r in rows:
        print("%-12s %-8s %5d %7.3f %7.3f  %-28s %-28s" % (
            DS_PRETTY.get(r["ds"], r["ds"]), r["tag"], r["n_fams"],
            r["tau"], r["rho"], PRETTY.get(r["top_L0"], r["top_L0"]),
            PRETTY.get(r["top_L4"], r["top_L4"])))
    taus = [r["tau"] for r in rows if r["tau"] == r["tau"]]
    print("\nrows: %d | median tau = %.3f | mean tau = %.3f | min = %.3f" % (
        len(taus), sorted(taus)[len(taus) // 2], sum(taus) / len(taus), min(taus)))
    print("top detector changes identity in %d of %d settings" % (
        sum(1 for r in rows if r["top_L0"] != r["top_L4"]), len(rows)))


if __name__ == "__main__":
    main()
