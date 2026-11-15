#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E11: clinical error-type analysis at the matched rung (L4).

Tags every question with four clinical properties a radiologist would care about
  negation / laterality / size-or-comparison / lesion-presence
and reports, per tag, for the full detector, the evidence-channel detector and the
question+answer-only detector:
  * number of pairs and the share whose two answers are opposite-polarity yes/no
    (the subset on which a text-only model is structurally blind)
  * AUROC with a pair-level bootstrap CI
  * pairwise accuracy: share of pairs where the grounded answer scores lower
  * false negatives (missed hallucinations) and false positives at the Youden point

The tags are rule based; the rule hits are written to an audit sheet so the
tagging can be checked item by item.

Usage: python e11_clinical.py [dataset ...]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

import common as C

# ---------------------------------------------------------------------------
# lexical rules
# ---------------------------------------------------------------------------
# Word-boundary matching; multiword phrases allowed.
RULES = {
    "negation": [
        r"not", r"no", r"without", r"absent", r"absence", r"negative", r"none",
        r"neither", r"nor", r"never", r"lack(?:s|ing)?", r"free of",
        r"rule[sd]? out", r"exclude[sd]?", r"unremarkable", r"no evidence",
        r"isn't", r"aren't", r"doesn't", r"don't", r"cannot", r"can't",
    ],
    "laterality": [
        r"left", r"right", r"bilateral(?:ly)?", r"unilateral(?:ly)?",
        r"contralateral", r"ipsilateral", r"hemisphere", r"hemibody",
        r"lateral(?:ly)?", r"side",
    ],
    "size_comparison": [
        r"large(?:r)?", r"small(?:er)?", r"big(?:ger)?", r"enlarged",
        r"enlargement", r"increas(?:e|ed|ing)", r"decreas(?:e|ed|ing)",
        r"greater", r"lesser", r"worse", r"better", r"severe", r"mild(?:ly)?",
        r"moderate", r"extensive", r"diffuse", r"focal", r"widespread", r"tiny",
        r"huge", r"measurable", r"size", r"diameter", r"\d+\s*(?:mm|cm)",
    ],
    "lesion_presence": [
        r"lesion", r"mass", r"nodule", r"tumou?r", r"neoplasm", r"metasta(?:sis|ses|tic)",
        r"opacity", r"consolidation", r"effusion", r"fracture", r"infiltrat(?:e|ion)",
        r"cyst", r"polyp", r"o?edema", r"h(?:a)?emorrhage", r"bleeding",
        r"calcification", r"thromb(?:us|osis)", r"embolism", r"abscess", r"ulcer",
        r"erosion", r"stenosis", r"aneurysm", r"cardiomegaly", r"pneumonia",
        r"pneumothorax", r"fibrosis", r"atelectasis", r"herniation", r"isch(?:a)?emia",
        r"infarct(?:ion)?", r"glioma", r"meningioma", r"carcinoma", r"sarcoma",
        r"lymphoma", r"lymphadenopathy", r"lymph node", r"abnormal(?:ity)?",
        r"disease", r"disorder", r"pathology", r"defect", r"tear", r"rupture",
        r"dilat(?:ion|ation)", r"hypertrophy", r"atrophy", r"ground[- ]glass",
    ],
}
_ORDER = ["negation", "laterality", "size_comparison", "lesion_presence"]
_COMPILED = {k: re.compile(r"\b(?:" + "|".join(v) + r")\b", re.I) for k, v in RULES.items()}


def tag_question(q):
    """Return (set_of_tags, primary_tag, matched_keywords)."""
    hits = {}
    for k, rx in _COMPILED.items():
        m = rx.findall(q or "")
        if m:
            hits[k] = sorted({x.strip().lower() for x in m})
    tags = set(hits)
    primary = next((k for k in _ORDER if k in tags), "other")
    return tags, primary, hits


YES, NO = {"yes", "true", "y"}, {"no", "false", "n"}


def _pol(a):
    al = str(a).lower().strip().rstrip(".")
    return "y" if al in YES else ("n" if al in NO else "o")


def _auroc(y, s):
    return C.auroc(y, s)


def _fast_auroc(y, s):
    y = np.asarray(y)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pair_boot(y, s, pids, n=1000, seed=0, alpha=0.05):
    """Bootstrap over pairs (all rows of a resampled pair stay together)."""
    rng = np.random.RandomState(seed)
    uniq = np.asarray(sorted(set(pids)))
    if len(uniq) < 5:
        return float("nan"), float("nan")
    pos = {}
    for k, p in enumerate(pids):
        pos.setdefault(p, []).append(k)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([pos[p] for p in pick])
        a = _fast_auroc(y[take], s[take])
        if a == a:
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def _pair_boot_diff(y, s1, s2, pids, n=600, seed=0, alpha=0.05):
    """Bootstrap CI of AUROC(s1) - AUROC(s2), resampling pairs."""
    rng = np.random.RandomState(seed)
    uniq = np.asarray(sorted(set(pids)))
    if len(uniq) < 5:
        return float("nan"), float("nan")
    pos = {}
    for k, p in enumerate(pids):
        pos.setdefault(p, []).append(k)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([pos[p] for p in pick])
        a1 = _fast_auroc(y[take], s1[take])
        a2 = _fast_auroc(y[take], s2[take])
        if a1 == a1 and a2 == a2:
            vals.append(a1 - a2)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def _wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - h) / d), float((c + h) / d)


def youden(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(y, s)
    return float(th[int(np.argmax(tpr - fpr))])


# ---------------------------------------------------------------------------
# cache discovery
# ---------------------------------------------------------------------------
def cache_dirs():
    root = Path(__file__).resolve().parent.parent.parent
    cands = [
        Path("/root/bench_exp/cache"),
        Path("/root/hallu_bench/cache"),
        root / "paper_crossdomain/results/remote_artefacts/cache",
        Path(C.CACHE),
        root / "_localrun" / "cache",
        root / "_localrun" / "hallu" / "cache",
    ]
    out = []
    for c in cands:
        if c.exists() and str(c) not in out:
            out.append(str(c))
    return out


def find(path_name, caches):
    for c in caches:
        p = Path(c) / path_name
        if p.exists():
            return p
    return None


def out_dir():
    root = Path(__file__).resolve().parent.parent
    for c in (root / "results" / "clinical", Path(C.OUT)):
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".w"
            probe.write_text("x")
            probe.unlink()
            return c
        except Exception:
            continue
    return Path(".")


def analyse(ds, level, caches):
    rp = find("rows_%s_%s.json" % (ds, level), caches)
    tp = find("trained_%s_%s.npz" % (ds, level), caches)
    np_ = find("trained_noimg_%s_%s.npz" % (ds, level), caches)
    if not (rp and tp):
        print("[E11] skip %s %s (no cache)" % (ds, level), flush=True)
        return None
    rows = json.load(open(rp, encoding="utf-8"))
    T = np.load(str(tp), allow_pickle=True)
    te = np.asarray(T["idx_te"], int)
    y_all = np.asarray(T["y"], int)

    scores = {"full": np.asarray(T["mlp_score"], float)}
    if "qce_score" in T.files:
        scores["qce"] = np.asarray(T["qce_score"], float)
    if np_ is None and "noimg_score" in T.files:
        scores["noimage"] = np.asarray(T["noimg_score"], float)
    if np_ is not None:
        N = np.load(str(np_), allow_pickle=True)
        key = "score" if "score" in N.files else ("noimg_score" if "noimg_score" in N.files else None)
        if key:
            scores["noimage"] = np.asarray(N[key], float)

    # rows are ordered +, -, +, - ... so pair k is (2k, 2k+1) only when built that way;
    # fall back to grouping on pid.
    pids_te = np.array([rows[i]["pid"] for i in te])
    y = y_all[te]
    tags_te = [tag_question(rows[i]["question"]) for i in te]
    prim_te = np.array([t[1] for t in tags_te])
    yn_pair = np.array([rows[i]["qt"] == "yes_no" for i in te])
    # opposite-polarity pair: within the same pid the two answers are yes vs no
    by_pid = {}
    for k, i in enumerate(te):
        by_pid.setdefault(rows[i]["pid"], []).append(k)
    opp = np.zeros(len(te), bool)
    for p, ks in by_pid.items():
        if len(ks) == 2:
            a0, a1 = _pol(rows[te[ks[0]]]["answer"]), _pol(rows[te[ks[1]]]["answer"])
            if {a0, a1} == {"y", "n"}:
                opp[ks] = True

    # pairwise correctness: for each pid, grounded (label 0) scores below hallucinated (label 1)
    pair_acc = {}
    for name, s in scores.items():
        full = np.full(len(rows), np.nan)
        full[te] = s
        ok = tot = 0
        for p, ks in by_pid.items():
            if len(ks) != 2:
                continue
            lab = [rows[te[k]]["label"] for k in ks]
            if sorted(lab) != [0, 1]:
                continue
            pos_k = ks[lab.index(0)]
            neg_k = ks[lab.index(1)]
            if np.isnan(full[te[pos_k]]) or np.isnan(full[te[neg_k]]):
                continue
            ok += int(full[te[neg_k]] > full[te[pos_k]])
            tot += 1
        pair_acc[name] = (ok, tot)

    out = {"ds": ds, "level": level, "n_rows": int(len(te)), "n_pairs": int(len(by_pid)),
           "share_opposite_polarity_yn": float(opp.mean()), "types": {}, "audit": {}}

    groups = [("ALL", np.ones(len(te), bool))]
    # negation: the canonical test is an answer pair that differs only in polarity
    groups.append(("negation", opp))
    named = np.isin(prim_te, ["laterality", "size_comparison", "lesion_presence"])
    for k in ("laterality", "size_comparison", "lesion_presence"):
        groups.append((k, prim_te == k))
        groups.append((k + "|image_necessary", (prim_te == k) & opp))
    groups.append(("other", ~named))
    for gname, mask in groups:
        n = int(mask.sum())
        if n < 40:
            continue
        yy = y[mask]
        if len(set(yy.tolist())) < 2:
            continue
        ent = {"n_rows": n, "n_pairs": int(len({pids_te[k] for k in np.where(mask)[0]})),
               "share_yn": float(yn_pair[mask].mean()),
               "share_opposite_polarity_yn": float(opp[mask].mean()), "auroc": {}}
        for name, s in scores.items():
            sub = s[mask]
            a = float(_auroc(yy, sub))
            seed = (sum(ord(ch) for ch in gname) * 97 + len(name) * 13) % 10000
            lo, hi = _pair_boot(yy, sub, pids_te[mask], n=600, seed=seed)
            ok, tot = pair_acc[name] if gname == "ALL" else (None, None)
            ent["auroc"][name] = {"auroc": a, "ci": [lo, hi]}
            if ok is not None:
                ent["auroc"][name]["pair_acc"] = float(ok / max(1, tot))
                ent["auroc"][name]["pair_acc_ci"] = list(_wilson(ok, tot))
        if "full" in scores:
            t = youden(yy, scores["full"][mask])
            pred = scores["full"][mask] > t
            ent["full_at_youden"] = {"threshold": float(t),
                                     "n_fn_missed_hallucination": int(((~pred) & (yy == 1)).sum()),
                                     "n_fp_rejected_grounded": int((pred & (yy == 0)).sum())}
        if "full" in scores and "noimage" in scores:
            lo, hi = _pair_boot_diff(yy, scores["full"][mask], scores["noimage"][mask],
                                     pids_te[mask], n=600, seed=seed + 1)
            ent["delta_image"] = {"diff": float(ent["auroc"]["full"]["auroc"] -
                                                ent["auroc"]["noimage"]["auroc"]),
                                  "ci": [lo, hi]}
        out["types"][gname] = ent

    # audit sheet: up to 12 rule hits per tag, with the matched keywords
    for k in _ORDER:
        picked = []
        for j, (tags, _p, hits) in enumerate(tags_te):
            if k in tags and len(picked) < 12:
                i = te[j]
                picked.append(dict(question=rows[i]["question"], answer=rows[i]["answer"],
                                   label=rows[i]["label"], keywords=hits[k]))
        out["audit"][k] = picked
    print("[E11] %s %s: pairs=%d opp_yn=%.2f  %s" %
          (ds, level, out["n_pairs"], out["share_opposite_polarity_yn"],
           " ".join("%s=%.3f" % (g, out["types"][g]["auroc"]["full"]["auroc"])
                    for g in ("ALL",) + tuple(_ORDER) if g in out["types"])), flush=True)
    return out


def main():
    dss = sys.argv[1:] or ["vqa_rad", "path_vqa", "omni", "slake", "pope_random"]
    caches = cache_dirs()
    print("[E11] caches: %s" % caches, flush=True)
    out = out_dir() / "e11_clinical.json"
    res = json.load(open(out, encoding="utf-8")) if out.exists() else {}
    for ds in dss:
        r = analyse(ds, "L4_fair", caches)
        if r:
            res[ds] = r
    json.dump(res, open(out, "w"), indent=2, default=C._default)
    print("[E11] wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
