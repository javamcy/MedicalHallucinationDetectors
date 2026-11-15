 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E8: protocol-sensitivity analysis ("is the fair protocol artificially strict?").

Varies one protocol hyper-parameter at a time, holding everything else at L4:
  * length window  w in {1, 2, 5, 10, inf}
  * type matching  strict (opposite polarity / same question type) vs off
  * split          image-level ('gid') vs row-level ('row')

Reports, for each variant: the shortcut oracle, CLIPScore, the trained MLP and QCE-Net.
If the collapse survives every variant, the protocol is not the cause.

Usage: python e8_sensitivity.py <dataset>
"""
import sys
import numpy as np

import common as C


def evaluate_rows(ds, samples, clip, tag, rows, tr, va, te, audit, res):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = np.array([r["label"] for r in rows], np.int32)
    if len(te) < 50 or y[te].sum() < 5:
        return
    F = C.features_for_rows(clip, samples, rows, verbose=False)
    ans_freq, _ = C.pool_freqs(samples)
    Xl = np.asarray([C.leak_features(r, ans_freq) for r in rows], np.float32)
    sc = StandardScaler().fit(Xl[tr])
    lr = LogisticRegression(max_iter=3000).fit(sc.transform(Xl[tr]), y[tr])
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0).fit(Xl[tr], y[tr])
    Xm = np.concatenate([F["G"], F["A"], F["Q"]], 1)
    Xc = np.concatenate([F["G"], F["A"], F["Q"], F["EQ"], F["EA"]], 1)
    p_mlp, _, _ = C.train_mlp_scores(Xm[tr], Xm[va], Xm[te], y[tr], y[va], y[te])
    p_qce, _, _ = C.train_mlp_scores(Xc[tr], Xc[va], Xc[te], y[tr], y[va], y[te])
    res[tag] = dict(
        n_test=int(len(te)), pos_test=int(y[te].sum()), audit=audit,
        shortcut_gbm=float(C.auroc(y[te], gb.predict_proba(Xl[te])[:, 1])),
        shortcut_logreg=float(C.auroc(y[te], lr.predict_proba(sc.transform(Xl[te]))[:, 1])),
        clipscore=float(C.best_orientation(y[te], (F["G"][te] * F["A"][te]).sum(1))[1]),
        maxpatch=float(C.best_orientation(y[te], F["PATCH"][te].max(1))[1]),
        mlp=float(C.auroc(y[te], p_mlp)), qce=float(C.auroc(y[te], p_qce)),
        mlp_ci=list(C.boot_ci(y[te], p_mlp, n=400, seed=1)),
        qce_ci=list(C.boot_ci(y[te], p_qce, n=400, seed=2)))
    print("[E8] %s %-18s | sc=%.4f mlp=%.4f qce=%.4f (n=%d)"
          % (ds, tag, res[tag]["shortcut_gbm"], res[tag]["mlp"], res[tag]["qce"], len(te)), flush=True)


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else "vqa_rad"
    samples = C.load_raw(ds)
    clip = C.ClipFeat()
    clip.load_img_cache(ds)
    res = {}
    def run(tag, use_len, use_type, win, split):
        tr, va, te, rows, audit = C.get_level(ds, "L4_fair", samples, use_len=use_len,
                                              use_type=use_type, len_window=win,
                                              split_mode=split)
        recs = None
        evaluate_rows(ds, samples, clip, tag, rows, tr, va, te, audit, res)

    for w in (1, 2, 5, 10):
        run("lenw=%d,type=on,gid" % w, True, True, w, "gid")
    run("lenw=inf,type=on,gid", False, True, 0, "gid")
    run("lenw=2,type=off,gid", True, False, 2, "gid")
    run("lenw=2,type=on,row", True, True, 2, "row")
    run("lenw=inf,type=off,row (leaky)", False, False, 0, "row")
    clip.save_img_cache(ds)
    C.save_json(res, C.OUT / ("e8_sensitivity_%s.json" % ds))
    print("[E8] %s done" % ds, flush=True)


if __name__ == "__main__":
    main()
