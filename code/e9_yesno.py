 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E9: the yes/no block -- is the answer determined by the question rather than the image?

Three measurements per dataset, on the matched protocol (L4):
  * matched-pair accuracy of the full detector and of the no-image detector inside the
    yes/no block (the image's marginal value on polarity decisions)
  * AUROC of a TF-IDF classifier over the QUESTION alone for predicting whether the
    grounded answer is affirmative -- if the phrasing fixes the polarity, the subset is
    not an image-grounding task
  * the same for open-ended answer pairs, as a reference

Usage: python e9_yesno.py
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

import common as C


def question_prior(rows, allpos, ps, train_pids):
    """AUROC of question-only prediction of 'the grounded answer is affirmative',
    trained on pairs whose positive row sits in the training split and evaluated on the rest."""
    qtr = [rows[allpos[p]]["question"] for p in ps if p in train_pids]
    ytr = [1 if C.answer_polarity(rows[allpos[p]]["answer"]) == "y" else 0
           for p in ps if p in train_pids]
    qte = [rows[allpos[p]]["question"] for p in ps if p not in train_pids]
    yte = [1 if C.answer_polarity(rows[allpos[p]]["answer"]) == "y" else 0
           for p in ps if p not in train_pids]
    if len(set(ytr)) < 2 or len(qte) < 50 or len(set(yte)) < 2:
        return None
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=2000))
    pipe.fit(qtr, ytr)
    return dict(auroc=float(roc_auc_score(yte, pipe.predict_proba(qte)[:, 1])),
                n_train=len(ytr), n_test=len(yte), base_rate=float(np.mean(yte)))


def main():
    out = {}
    for ds in ("vqa_rad", "path_vqa", "omni"):
        d = C.CACHE / ("trained_%s_L4_fair.npz" % ds)
        ni = C.CACHE / ("trained_noimg_%s_L4_fair.npz" % ds)
        if not (d.exists() and ni.exists()):
            continue
        s = C.load_raw(ds)
        tr, va, te, rows, audit = C.get_level(ds, "L4_fair", s)
        T, N = np.load(str(d)), np.load(str(ni))
        mlp, nimg = T["mlp_score"], N["score"]
        te = T["idx_te"]
        trs = set(int(x) for x in tr)
        pos, neg = {}, {}
        for j, i in enumerate(te):
            (pos if rows[i]["label"] == 0 else neg)[rows[i]["pid"]] = j
        # global (all-split) positive index per pid, and the set of training pids
        allpos = {r["pid"]: i for i, r in enumerate(rows) if r["label"] == 0}
        train_pids = set(rows[i]["pid"] for i in tr)
        common = [p for p in pos if p in neg]
        blocks = {
            "yes_no": [p for p in common if C.answer_is_yesno(rows[te[pos[p]]]["answer"])],
            "open": [p for p in common if not C.answer_is_yesno(rows[te[pos[p]]]["answer"])],
        }
        e = {}
        for name, ps in blocks.items():
            if len(ps) < 50:
                continue
            dm = np.array([mlp[pos[p]] - mlp[neg[p]] for p in ps])
            dn = np.array([nimg[pos[p]] - nimg[neg[p]] for p in ps])
            ps_all = [p for p in allpos
                      if (C.answer_is_yesno(rows[allpos[p]]["answer"]) == (name == "yes_no"))]
            e[name] = dict(n_pairs=len(ps),
                           pair_acc_full=float(1 - (dm > 0).mean()),
                           pair_acc_no_image=float(1 - (dn > 0).mean()),
                           question_prior=question_prior(rows, allpos, ps_all, train_pids),
                           n_all_pairs=len(ps_all))
        out[ds] = e
        qp = e.get("yes_no", {}).get("question_prior")
        print("[E9] %-9s yes/no pairs=%d  full=%.3f noimg=%.3f  q-prior=%s"
              % (ds, e.get("yes_no", {}).get("n_pairs", 0),
                 e.get("yes_no", {}).get("pair_acc_full", float("nan")),
                 e.get("yes_no", {}).get("pair_acc_no_image", float("nan")),
                 ("%.3f" % qp["auroc"]) if qp else "n/a"), flush=True)
    C.save_json(out, C.OUT / "e9_yesno.json")
    print("[E9] done", flush=True)


if __name__ == "__main__":
    main()
