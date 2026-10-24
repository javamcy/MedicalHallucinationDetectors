 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1: leakage-gradient protocol experiment.

For each dataset x protocol level (L0..L5, plus L6 native distractors on OmniMed):
  * build the paired data and audit the negative construction
  * shortcut oracle (lexical/length/type features only, NO image)   <- key evidence
  * frozen single-scalar detectors (CLIPScore, max-patch, QA sim, ...)
  * trained MLP  [g,a,q]  and QCE-Net [g,a,q,e_q,e_a]

Usage: python e1_protocol.py <dataset> [levels...]
"""
import sys
import time
import numpy as np

import common as C


def single_scalar_scores(F, y):
    """Frozen, training-free single scalars. Return dict name -> (score, auroc_best)."""
    G, A, Q, EQ, EA = F["G"], F["A"], F["Q"], F["EQ"], F["EA"]
    out = {}
    cands = {
        "clipscore_gA": (G * A).sum(1),
        "maxpatch_pA": F["PATCH"].max(1),
        "meanpatch_pA": F["PATCH"].mean(1),
        "qasim_qA": (Q * A).sum(1),
        "eq_g_sim": (EQ * G).sum(1),
        "ea_g_sim": (EA * G).sum(1),
        "eq_ea_sim": (EQ * EA).sum(1),
    }
    # entropy of the question-conditioned evidence map over patches
    for nm, key in (("ent_q", "CONQ"), ("ent_a", "CONA")):
        W = np.clip(F[key], 1e-12, 1.0)
        cands[nm] = -(W * np.log(W)).sum(1)
    for k, v in cands.items():
        s, a = C.best_orientation(y, v)
        out[k] = dict(auroc=float(a), raw_auroc=float(C.auroc(y, v)), score=v)
    return out


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "vqa_rad"
    levels = sys.argv[2:] or C.LEVEL_ORDER
    if ds != "omni" and "L6_native" in levels:
        levels = [l for l in levels if l != "L6_native"]
    if ds == "omni" and "L6_native" not in levels:
        levels = levels + ["L6_native"]

    t0 = time.time()
    print("[E1] loading raw %s ..." % ds, flush=True)
    samples = C.load_raw(ds)
    print("[E1] %s raw=%d" % (ds, len(samples)), flush=True)

    clip = C.ClipFeat()
    clip.load_img_cache(ds)
    res = {}
    for lv in levels:
        t1 = time.time()
        tr, va, te, rows, audit = C.get_level(ds, lv, samples)
        y = np.array([r["label"] for r in rows], np.int32)
        yte = y[te]
        print("[E1] %s %s: rows=%d train=%d val=%d test=%d pos_test=%d audit=%s"
              % (ds, lv, len(rows), len(tr), len(va), len(te), int(yte.sum()), audit), flush=True)

        F = C.features_for_rows(clip, samples, rows, verbose=False)
        np.savez_compressed(C.CACHE / ("feat_%s_%s.npz" % (ds, lv)),
                            **{k: F[k] for k in ("G", "A", "Q", "EQ", "EA", "PATCH")},
                            y=y, idx_tr=np.asarray(tr), idx_va=np.asarray(va), idx_te=np.asarray(te))
        # metadata for later qualitative analysis
        C.save_json([dict(pid=r["pid"], gid=r["gid"], question=r["question"], answer=r["answer"],
                          label=r["label"], qt=r["qt"], match=r["match"],
                          modality=r.get("modality", "unknown"),
                          omni_qtype=r.get("omni_qtype", "unknown"),
                          dataset_name=r.get("dataset_name", "unknown")) for r in rows],
                    C.CACHE / ("rows_%s_%s.json" % (ds, lv)))

        ent = {"n_rows": len(rows), "n_train": len(tr), "n_val": len(va), "n_test": len(te),
               "pos_test": int(yte.sum()), "audit": audit, "level": lv, "ds": ds,
               "level_desc": C.LEVEL_PRETTY.get(lv, lv)}

        # ---- E1b shortcut oracle (no image at all) ----
        ans_freq, q_freq = C.pool_freqs(samples)
        Xl = np.asarray([C.leak_features(r, ans_freq) for r in rows], np.float32)
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import HistGradientBoostingClassifier

        sc = StandardScaler().fit(Xl[tr])
        lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xl[tr]), y[tr])
        s_lr = lr.predict_proba(sc.transform(Xl))[:, 1]
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0).fit(Xl[tr], y[tr])
        s_gb = gb.predict_proba(Xl)[:, 1]
        ent["shortcut_logreg"] = float(C.auroc(yte, s_lr[te]))
        ent["shortcut_gbm"] = float(C.auroc(yte, s_gb[te]))
        lo, hi = C.boot_ci(yte, s_gb[te], n=500, seed=0)
        ent["shortcut_gbm_ci"] = [lo, hi]
        ent["shortcut_logreg_ci"] = list(C.boot_ci(yte, s_lr[te], n=500, seed=1))
        ent["shortcut_perm_p"] = C.perm_pvalue(yte, s_gb[te], n=1000, seed=0)[1]
        # per-feature single-feature AUROC (leak transparency)
        names = ["ans_tokens", "ans_chars", "q_tokens", "ans/q_tok", "ans/q_char", "is_yesno",
                 "qt_is_yesno", "qt_is_wh", "qt_other", "type_match", "has_digit", "has_pct",
                 "has_comma", "ends_period", "cap_first", "negation_word", "log_ans_freq",
                 "log_qt_freq", "qa_overlap_n", "qa_jaccard"]
        ent["leak_feature_auroc"] = {nm: float(C.best_orientation(yte, Xl[te, i])[1])
                                     for i, nm in enumerate(names)}
        np.savez_compressed(C.CACHE / ("shortcut_%s_%s.npz" % (ds, lv)), s_lr=s_lr, s_gb=s_gb, y=y,
                            X=Xl, idx_tr=np.asarray(tr), idx_va=np.asarray(va), idx_te=np.asarray(te),
                            names=np.asarray(names))

        # ---- E1c frozen single scalars ----
        ss = single_scalar_scores(F, y)
        ent["frozen_scalars"] = {k: dict(auroc=v["auroc"], raw_auroc=v["raw_auroc"]) for k, v in ss.items()}
        for k, v in ss.items():
            np.savez_compressed(C.CACHE / ("scalar_%s_%s_%s.npz" % (ds, lv, k)), score=v["score"])

        # ---- E1d trained detectors ----
        Xm = np.concatenate([F["G"], F["A"], F["Q"]], 1)
        Xc = np.concatenate([F["G"], F["A"], F["Q"], F["EQ"], F["EA"]], 1)
        p_mlp, v_mlp, e_mlp = C.train_mlp_scores(Xm[tr], Xm[va], Xm[te], y[tr], y[va], yte)
        p_qce, v_qce, e_qce = C.train_mlp_scores(Xc[tr], Xc[va], Xc[te], y[tr], y[va], yte)
        ent["mlp"] = float(C.auroc(yte, p_mlp))
        ent["qce"] = float(C.auroc(yte, p_qce))
        ent["mlp_ci"] = list(C.boot_ci(yte, p_mlp, n=500, seed=2))
        ent["qce_ci"] = list(C.boot_ci(yte, p_qce, n=500, seed=3))
        ent["mlp_val"] = float(v_mlp)
        ent["qce_val"] = float(v_qce)
        np.savez_compressed(C.CACHE / ("trained_%s_%s.npz" % (ds, lv)),
                            mlp_score=p_mlp, qce_score=p_qce, y=y,
                            idx_tr=np.asarray(tr), idx_va=np.asarray(va), idx_te=np.asarray(te))

        # ---- also: MLP trained WITHOUT image ([a,q]) to test the "no image" claim per level ----
        Xn = np.concatenate([F["A"], F["Q"]], 1)
        p_nimg, v_nimg, _ = C.train_mlp_scores(Xn[tr], Xn[va], Xn[te], y[tr], y[va], yte)
        ent["mlp_noimage"] = float(C.auroc(yte, p_nimg))
        np.savez_compressed(C.CACHE / ("trained_noimg_%s_%s.npz" % (ds, lv)), score=p_nimg, y=y)

        res[lv] = ent
        print("[E1] %s %s done in %.1fs | shortcut=%.4f mlp=%.4f qce=%.4f noimg=%.4f clipscore=%.4f"
              % (ds, lv, time.time() - t1, ent["shortcut_gbm"], ent["mlp"], ent["qce"],
                 ent["mlp_noimage"], ent["frozen_scalars"]["clipscore_gA"]["auroc"]), flush=True)
        C.save_json(res, C.OUT / ("e1_protocol_%s.json" % ds))

    clip.save_img_cache(ds)
    print("[E1] %s ALL DONE in %.1f min" % (ds, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
