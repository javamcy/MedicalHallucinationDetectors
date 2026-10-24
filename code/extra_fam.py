 
import os, sys, json
import numpy as np
sys.path.insert(0, "/root/nature_exp/exp")
sys.path.insert(0, "/root/hallu_bench/code")
import common as C
import bench2 as B2
import vlm2 as V2

OUT = "/root/hallu_bench/out"

def norm_ans(s):
    return " ".join(str(s).lower().replace(".", " ").replace(",", " ").split())

def main(ds):
    samples = B2.load_slake() if ds == "slake" else B2.load_pope(split="random")
    gp = os.path.join("/root/hallu_bench/cache", "gen_%s_q3b.jsonl" % ds)
    G = {}
    for line in open(gp, encoding="utf-8"):
        r = json.loads(line)
        G[r["image_id"]] = r
    print("[extra] %s generations=%d" % (ds, len(G)), flush=True)
    clip = C.ClipFeat()
    res = {}
    for lv in ("L0_leaky", "L4_fair"):
        tr, va, te, rows, audit = V2.get_level(ds, lv, samples)
        y = np.array([r["label"] for r in rows], np.int32)
        cand = [norm_ans(r["answer"]) for r in rows]
        greed = []
        for r in rows:
            g = G.get(samples[r["pid"]]["image_id"])
            greed.append(norm_ans(g["samples"][0]) if g and g.get("samples") else "")
        uniq = sorted(set(cand) | set(greed))
        embs = np.stack(clip.text_feat_batch(uniq))
        idx = {t: i for i, t in enumerate(uniq)}
        s = np.array([float(embs[idx[a]] @ embs[idx[b]]) if b else np.nan
                      for a, b in zip(cand, greed)], float)
        st = s[te]
        m = ~np.isnan(st)
        if m.sum() > 50 and len(set(y[te][m].tolist())) > 1:
            res[lv] = dict(self_verification=float(C.best_orientation(y[te][m], st[m])[1]),
                           n=int(m.sum()))
            print("[extra] %s %s self-verification=%.3f (n=%d)" % (ds, lv, res[lv]["self_verification"], m.sum()), flush=True)
    C.save_json(res, os.path.join(OUT, "fam_extra_%s.json" % ds))

if __name__ == "__main__":
    main(sys.argv[1])
