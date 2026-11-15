#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E2: randomisation / negative controls.

Answers the reviewer question "is 0.49 really chance, or is the protocol biased?"

For every (dataset, level) with cached scores we compute
  (a) permutation null for each detector's test AUROC (labels shuffled, 2000x)
  (b) random-score baseline distribution
  (c) a *label-shuffled retrain* of the MLP: training on permuted labels should give
      test AUROC ~ 0.5, which proves the split/protocol itself carries no signal
  (d) a *random-image* control for frozen image-answer scorers (image replaced by a
      random other image) -> AUROC ~ 0.5

Usage: python e2_random.py [dataset ...]
"""
import sys
import numpy as np

import common as C

if not hasattr(C, "load_json"):
    def _load_json(p):
        import json
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    C.load_json = _load_json

NPERM = 2000


def main():
    dss = sys.argv[1:] or ["vqa_rad", "path_vqa", "omni"]
    allres = {}
    for ds in dss:
        res = {}
        for lv in C.LEVEL_ORDER + (["L6_native"] if ds == "omni" else []):
            tr_p = C.CACHE / ("trained_%s_%s.npz" % (ds, lv))
            if not tr_p.exists():
                continue
            d = np.load(str(tr_p))
            te_g = d["idx_te"]
            y_all = d["y"]
            n = len(y_all)
            y = y_all[te_g]

            def to_full(v):
                full = np.full(n, np.nan)
                full[te_g] = np.asarray(v, float)
                return full

            mlp, qce = d["mlp_score"], d["qce_score"]
            ent = {}
            for nm, s in (("mlp", mlp), ("qce", qce)):
                obs, p = C.perm_pvalue(y, s, n=NPERM, seed=0)
                ent[nm] = dict(auroc=float(obs), perm_p=float(p))
            # random scores
            rng = np.random.RandomState(0)
            rs = [C.auroc(y, rng.rand(len(y))) for _ in range(500)]
            ent["random_score_null"] = dict(mean=float(np.mean(rs)), sd=float(np.std(rs)),
                                            lo=float(np.percentile(rs, 2.5)), hi=float(np.percentile(rs, 97.5)))
            # label-shuffled retrain of the MLP (protocol-bias control)
            fp = C.CACHE / ("feat_%s_%s.npz" % (ds, lv))
            if fp.exists():
                F = np.load(str(fp))
                Xm = np.concatenate([F["G"], F["A"], F["Q"]], 1)
                tr, va, te = F["idx_tr"], F["idx_va"], F["idx_te"]
                ytr = F["y"][tr]
                rng2 = np.random.RandomState(7)
                ysh = ytr.copy()
                rng2.shuffle(ysh)
                p, v, ep = C.train_mlp_scores(Xm[tr], Xm[va], Xm[te], ysh, F["y"][va], F["y"][te])
                ent["mlp_shuffled_label_retrain"] = dict(test_auroc=float(C.auroc(F["y"][te], p)),
                                                         val_auroc=float(v), best_epoch=int(ep))
                # random-image control for frozen scorers
                rng3 = np.random.RandomState(11)
                perm = rng3.permutation(len(te))
                Gr = F["G"][te][perm]
                Ate, Qte = F["A"][te], F["Q"][te]
                ent["random_image_control"] = {
                    "clipscore_gA": float(C.auroc(F["y"][te], (Gr * Ate).sum(1))),
                    "qasim_qA": float(C.auroc(F["y"][te], (Qte * Ate).sum(1))),
                }
            # ---- image-memorisation proof: AUROC on test rows whose gid was ALSO
            #      seen in training (row-level split only) vs unseen gids
            rows_p = C.CACHE / ("rows_%s_%s.json" % (ds, lv))
            if rows_p.exists() and fp.exists():
                rows = C.load_json(str(rows_p))
                tr_g = {rows[i]["gid"] for i in tr}
                seen = [i for i in te if rows[i]["gid"] in tr_g]
                unseen = [i for i in te if rows[i]["gid"] not in tr_g]
                ent["gid_overlap_control"] = dict(
                    n_seen_gid_test_rows=len(seen), n_unseen_gid_test_rows=len(unseen),
                    frac_test_rows_with_train_gid=float(len(seen) / max(1, len(te))))
                seen_pos = np.array([k for k, i in enumerate(te) if rows[i]["gid"] in tr_g], int)
                uns_pos = np.array([k for k, i in enumerate(te) if rows[i]["gid"] not in tr_g], int)
                for nm, sc_full in (("mlp", to_full(mlp)), ("qce", to_full(qce))):
                    sc_te = sc_full[te]
                    if len(seen_pos) > 60 and len(uns_pos) > 60:
                        ent["gid_overlap_control"][nm + "_auroc_seen_gid"] = float(
                            C.auroc(y[seen_pos], sc_te[seen_pos]))
                        ent["gid_overlap_control"][nm + "_auroc_unseen_gid"] = float(
                            C.auroc(y[uns_pos], sc_te[uns_pos]))
            res[lv] = ent
            print("[E2] %s %s: %s" % (ds, lv, {k: (round(v["auroc"], 4) if isinstance(v, dict) and "auroc" in v else v)
                                                for k, v in ent.items() if k in ("mlp", "qce")}), flush=True)
        allres[ds] = res
        C.save_json(res, C.OUT / ("e2_random_%s.json" % ds))
    print("[E2] done", flush=True)


if __name__ == "__main__":
    main()
