 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E5: cross-modality / cross-subset generalisation of the collapse.

Reported on the *fair* protocol (L4) plus, for OmniMed, on the native-distractor
protocol (L6) where negatives come from the dataset's own wrong MCQ options.

  * OmniMed: AUROC by imaging modality and by source sub-dataset
  * VQA-RAD / Path-VQA / OmniMed: AUROC by question type
Trained detectors are oriented globally (their own decision direction); frozen
scalars are oriented per group so that no group can be trivially penalised.

Usage: python e5_modality.py
"""
import json
import numpy as np
import common as C


def breakdown(ds, level, group_key, out):
    fp = C.CACHE / ("feat_%s_%s.npz" % (ds, level))
    rp = C.CACHE / ("rows_%s_%s.json" % (ds, level))
    tp = C.CACHE / ("trained_%s_%s.npz" % (ds, level))
    sp = C.CACHE / ("shortcut_%s_%s.npz" % (ds, level))
    if not (fp.exists() and rp.exists() and tp.exists()):
        print("[E5] missing cache for %s %s" % (ds, level), flush=True)
        return
    rows = json.load(open(rp))
    F = np.load(str(fp))
    T = np.load(str(tp))
    te, y = T["idx_te"], F["y"]
    n = len(rows)

    def to_full(test_aligned):
        full = np.full(n, np.nan)
        full[te] = np.asarray(test_aligned, float)
        return full

    trained = {"mlp": to_full(T["mlp_score"]), "qce": to_full(T["qce_score"])}
    ni = C.CACHE / ("trained_noimg_%s_%s.npz" % (ds, level))
    if ni.exists():
        trained["mlp_noimage"] = to_full(np.load(str(ni))["score"])
    frozen = {"clipscore": (F["G"] * F["A"]).sum(1), "maxpatch": F["PATCH"].max(1)}
    if sp.exists():
        trained["shortcut_gbm"] = np.load(str(sp))["s_gb"]
    groups = {}
    for i in te:
        groups.setdefault(rows[i].get(group_key, "unknown"), []).append(np.int64(i))
    for g, idxs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        idxs = np.asarray(idxs)
        yy = y[idxs]
        if len(idxs) < 40 or len(set(yy.tolist())) < 2:
            continue
        ent = dict(n=int(len(idxs)), pos=int(yy.sum()))
        for nm, s in trained.items():
            ent[nm] = float(C.auroc(yy, s[idxs]))
        for nm, s in frozen.items():
            ent[nm] = float(C.best_orientation(yy, s[idxs])[1])
        out["%s|%s|%s" % (ds, level, g)] = ent
        print("[E5] %-6s %-9s %-30s n=%4d mlp=%.3f noimg=%.3f qce=%.3f clip=%.3f sc=%.3f"
              % (ds, level, str(g)[:30], len(idxs), ent["mlp"],
                 ent.get("mlp_noimage", float("nan")), ent["qce"],
                 ent["clipscore"], ent.get("shortcut_gbm", float("nan"))), flush=True)


def main():
    res = {}
    for lv in ("L4_fair", "L6_native"):
        breakdown("omni", lv, "modality", res)
        breakdown("omni", lv, "dataset_name", res)
    for ds in ("vqa_rad", "path_vqa", "omni"):
        for lv in ("L4_fair", "L6_native"):
            breakdown(ds, lv, "qt", res)
            breakdown(ds, lv, "omni_qtype", res)
    C.save_json(res, C.OUT / "e5_modality.json")
    print("[E5] done", flush=True)


if __name__ == "__main__":
    main()
