 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E7: failure taxonomy + qualitative case sheet.

On the fair protocol (L4) and the native-distractor protocol (L6, OmniMed):
  * AUROC matrix  detector x question type
  * FP/FN counts at the Youden-optimal threshold, and the share contributed by
    the yes/no block
  * a qualitative case sheet: the highest-confidence false negatives (hallucinated
    answers the detector called grounded) and false positives

Usage: python e7_failure.py
"""
import json
import numpy as np
import common as C


def youden(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(y, s)
    return float(th[int(np.argmax(tpr - fpr))])


def analyse(ds, level, out):
    rp = C.CACHE / ("rows_%s_%s.json" % (ds, level))
    tp = C.CACHE / ("trained_%s_%s.npz" % (ds, level))
    fp_ = C.CACHE / ("feat_%s_%s.npz" % (ds, level))
    if not (rp.exists() and tp.exists()):
        return
    rows = json.load(open(rp))
    T = np.load(str(tp))
    F = np.load(str(fp_))
    te, y = T["idx_te"], F["y"]
    n = len(rows)

    def to_full(v):
        full = np.full(n, np.nan)
        full[te] = np.asarray(v, float)
        return full

    dets = {"mlp": to_full(T["mlp_score"]), "qce": to_full(T["qce_score"])}
    dets["clipscore"] = C.best_orientation(y, (F["G"] * F["A"]).sum(1))[0]
    dets["maxpatch"] = C.best_orientation(y, F["PATCH"].max(1))[0]
    qts = sorted({rows[i]["qt"] for i in te})
    ent = {"by_qtype": {}, "errors": {}}
    for qt in qts + ["ALL"]:
        idx = np.array([i for i in te if qt == "ALL" or rows[i]["qt"] == qt])
        if len(idx) < 30 or len(set(y[idx].tolist())) < 2:
            continue
        ent["by_qtype"][qt] = {"n": int(len(idx)), "pos": int(y[idx].sum())}
        for nm, s in dets.items():
            ent["by_qtype"][qt][nm] = float(C.auroc(y[idx], s[idx]))
    # FP / FN concentration
    cases = {}
    for nm, s in dets.items():
        if nm in ("clipscore", "maxpatch"):
            continue
        t = youden(y[te], s[te])
        pred = (s[te] > t).astype(int)
        yt = y[te]
        fp = [int(te[k]) for k in np.where((pred == 1) & (yt == 0))[0]]
        fn = [int(te[k]) for k in np.where((pred == 0) & (yt == 1))[0]]
        yn_fn = sum(1 for i in fn if rows[i]["qt"] == "yes_no")
        ent["errors"][nm] = dict(threshold=float(t), n_fp=len(fp), n_fn=len(fn),
                                 yesno_share_of_fn=float(yn_fn / max(1, len(fn))),
                                 frac_yesno=float(np.mean([rows[i]["qt"] == "yes_no" for i in te])))
        # qualitative: most confident errors
        scored_fn = sorted(fn, key=lambda i: -s[i])[:8]     # called grounded, most confident
        scored_fp = sorted(fp, key=lambda i: s[i])[:8]      # called hallucinated, most confident
        cases[nm] = {
            "false_negatives_missed_hallucinations": [
                dict(qt=rows[i]["qt"], question=rows[i]["question"], answer=rows[i]["answer"],
                     modality=rows[i].get("modality", ""), score=float(s[i]),
                     match=rows[i].get("match", "")) for i in scored_fn],
            "false_positives_rejected_grounded_answers": [
                dict(qt=rows[i]["qt"], question=rows[i]["question"], answer=rows[i]["answer"],
                     modality=rows[i].get("modality", ""), score=float(s[i]),
                     match=rows[i].get("match", "")) for i in scored_fp],
        }
    ent["cases"] = cases
    out["%s|%s" % (ds, level)] = ent
    print("[E7] %s %s qtypes=%s FN(yesno share)=%s" %
          (ds, level, {k: v.get("n") for k, v in ent["by_qtype"].items()},
           {k: round(v["yesno_share_of_fn"], 3) for k, v in ent["errors"].items()}), flush=True)


def main():
    res = {}
    for ds in ("vqa_rad", "path_vqa", "omni"):
        for lv in ("L4_fair", "L6_native", "L0_leaky"):
            analyse(ds, lv, res)
    C.save_json(res, C.OUT / "e7_failure.json")
    print("[E7] done", flush=True)


if __name__ == "__main__":
    main()
