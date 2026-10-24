 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VLM-based detectors on the new benchmarks.

  judge   zero-shot groundedness logit difference  (internal-signal family)
  nll     teacher-forced answer negative log-likelihood
  vihd    visual-intervention hallucination detection: occlude the top-k evidence
          patches, re-ask the judge, and score by the drop in groundedness
  vase    sampling-statistics family: K sampled answers -> semantic entropy, RadFlag,
          answer consistency and a candidate-vs-dominant-cluster visual score

Reuses the batched Adapter from e6_family.py (already validated on this host).

Usage:
  python vlm2.py judge  --ds slake --tag q3b --levels L0_leaky,L1_imgsplit,L4_fair
  python vlm2.py vihd   --ds slake --tag q3b --levels L4_fair --topk 16
  python vlm2.py gen    --ds slake --tag q3b --k 4
  python vlm2.py vase   --ds slake --tag q3b --levels L4_fair
"""
import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, "/root/nature_exp/exp")
import common as C
from e6_family import Adapter, DEFAULT_MODELS, _ans_norm, cluster_dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench2 as B2

BASE = "/root/hallu_bench"
OUT = os.path.join(BASE, "out")
CACHE = os.path.join(BASE, "cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)


def get_samples(ds):
    if ds == "slake":
        return B2.load_slake()
    if ds.startswith("pope"):
        return B2.load_pope(split=ds.split("_", 1)[1] if "_" in ds else "random")
    if ds == "medhallmark":
        return B2.load_medhallmark()
    raise SystemExit("unknown dataset " + ds)


def get_level(ds, lv, samples):
    if ds == "slake":
        recs, audit = B2.build_slake_level(samples, lv)
        mode = C.LEVELS[lv]["split"]
    elif ds.startswith("pope"):
        recs, audit = B2.build_pope_level(samples, unmatched=("leaky" in lv))
        mode = "row" if "row" in lv else "gid"
    else:
        recs = []
        for idx, s in enumerate(samples):
            recs.append(dict(pid=idx, gid=s["gid"], image_id=s["image_id"],
                             question=s["question"], pos_answer=s["answer"],
                             neg_answer=s["answer"], qt=C.question_type(s["question"]),
                             yn=C.answer_is_yesno(s["answer"]), alen=C.token_len(s["answer"]),
                             pos_len=C.token_len(s["answer"]), neg_len=C.token_len(s["answer"]),
                             match="native", modality="unknown", omni_qtype="unknown",
                             dataset_name="Med-HallMark"))
        mode, audit = "gid", {}
    rows = C.rows_from_recs(recs)
    tr, va, te = C.split_rows(rows, mode=mode)
    return tr, va, te, rows, audit


# ------------------------------------------------------------------ judge / NLL
def cmd_judge(args):
    samples = get_samples(args.ds)
    ad = Adapter(DEFAULT_MODELS.get(args.tag, args.tag))
    pos_id, neg_id = ad.label_ids()
    for lv in args.levels.split(","):
        tr, va, te, rows, audit = get_level(args.ds, lv, samples)
        outp = os.path.join(CACHE, "llm_%s_%s_%s.npz" % (args.ds, args.tag, lv))
        te_set = set(int(i) for i in te)
        uniq, seen = [], set()
        for i, r in enumerate(rows):
            if i not in te_set:
                continue
            k = "%d|%s" % (r["pid"], r["answer"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append((k, r))
        cache = {}
        if os.path.exists(outp):
            d = np.load(outp, allow_pickle=True)
            for k, j, n in zip(d["keys"], d["judge"], d["nll"]):
                cache[str(k)] = (float(j), float(n))
        todo = [(k, r) for k, r in uniq if k not in cache]
        print("[vlm2] judge %s %s %s uniq=%d todo=%d" % (args.ds, args.tag, lv, len(uniq), len(todo)), flush=True)
        for i in range(0, len(todo), args.bs):
            chunk = todo[i:i + args.bs]
            imgs = [samples[r["pid"]]["image"] for _, r in chunk]
            qs = [r["question"] for _, r in chunk]
            ans = [r["answer"] for _, r in chunk]
            j = ad.judge_batch(imgs, qs, ans, pos_id, neg_id)
            n = ad.nll_batch(imgs, qs, ans)
            for (k, _), jj, nn in zip(chunk, j, n):
                cache[k] = (float(jj), float(nn))
            if (i // args.bs) % 10 == 0:
                print("   %d/%d" % (min(i + args.bs, len(todo)), len(todo)), flush=True)
        J = np.array([cache.get("%d|%s" % (r["pid"], r["answer"]), (np.nan, np.nan))[0] for r in rows])
        N = np.array([cache.get("%d|%s" % (r["pid"], r["answer"]), (np.nan, np.nan))[1] for r in rows])
        y = np.array([r["label"] for r in rows], np.int32)
        np.savez_compressed(outp, judge=J, nll=N, y=y, idx_te=np.asarray(te),
                            keys=np.asarray([r["pid"] for r in rows]))
        e = {"ds": args.ds, "level": lv, "tag": args.tag, "n_test": int(len(te))}
        e["judge"] = float(C.best_orientation(y[te], J[te])[1])
        e["judge_raw"] = float(C.auroc(y[te], J[te]))
        e["answer_nll"] = float(C.best_orientation(y[te], N[te])[1])
        prev = {}
        p = os.path.join(OUT, "vlm2_judge_%s_%s.json" % (args.ds, args.tag))
        if os.path.exists(p):
            prev = json.load(open(p, encoding="utf-8"))
        prev[lv] = e
        C.save_json(prev, p)
        print("[vlm2] %s %s judge=%.3f (raw %.3f) nll=%.3f" %
              (args.ds, lv, e["judge"], e["judge_raw"], e["answer_nll"]), flush=True)


# ------------------------------------------------------------------ VIHD
def _occlude(img, boxes):
    """Zero out (grey-fill) a set of normalised boxes on a copy of the image."""
    from PIL import ImageDraw
    im = img.copy()
    W, H = im.size
    d = ImageDraw.Draw(im)
    for (y0, y1, x0, x1) in boxes:
        d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], fill=(127, 127, 127))
    return im


def cmd_vihd(args):
    """Patch grid: CLIP-L/14@336 -> 24x24 patches. We rank patches by the answer-conditioned
    alignment (cached in feat_*.npz) and occlude the top-k, then re-ask the judge."""
    samples = get_samples(args.ds)
    ad = Adapter(DEFAULT_MODELS.get(args.tag, args.tag))
    pos_id, neg_id = ad.label_ids()
    for lv in args.levels.split(","):
        tr, va, te, rows, audit = get_level(args.ds, lv, samples)
        fp = os.path.join(CACHE, "feat_%s_%s.npz" % (args.ds, lv))
        if not os.path.exists(fp):
            print("[vlm2] vihd needs %s -- run bench2.py first" % fp, flush=True)
            continue
        F = np.load(fp)
        A = F["A"]
        outp = os.path.join(CACHE, "vihd_%s_%s_%s.npz" % (args.ds, args.tag, lv))
        te_set = set(int(i) for i in te)
        uniq, seen = [], set()
        for i, r in enumerate(rows):
            if i not in te_set:
                continue
            k = "%d|%s" % (r["pid"], r["answer"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append((k, r, i))
        done = {}
        if os.path.exists(outp):
            d = np.load(outp, allow_pickle=True)
            for k, v in zip(d["keys"], d["drop"]):
                done[str(k)] = float(v)
        todo = [(k, r, i) for k, r, i in uniq if k not in done]
        print("[vlm2] vihd %s %s %s todo=%d topk=%d" % (args.ds, args.tag, lv, len(todo), args.topk), flush=True)
        GRID = 24  # 336/14
        for i in range(0, len(todo), args.bs):
            chunk = todo[i:i + args.bs]
            base_imgs, occ_imgs, qs, ans = [], [], [], []
            for k, r, gi in chunk:
                s = samples[r["pid"]]
                patches = C_ClipPatches(s, args)  # noqa: F821  (defined below)
                v = patches @ A[gi]
                idx = np.argsort(-v)[:args.topk]
                boxes = []
                for p in idx:
                    y0, x0 = divmod(int(p), GRID)
                    boxes.append((y0 / GRID, (y0 + 0.6) / GRID, x0 / GRID, (x0 + 0.6) / GRID))
                base_imgs.append(s["image"])
                occ_imgs.append(_occlude(s["image"], boxes))
                qs.append(r["question"])
                ans.append(r["answer"])
            j0 = ad.judge_batch(base_imgs, qs, ans, pos_id, neg_id)
            j1 = ad.judge_batch(occ_imgs, qs, ans, pos_id, neg_id)
            n0 = ad.nll_batch(base_imgs, qs, ans)
            n1 = ad.nll_batch(occ_imgs, qs, ans)
            for (k, _, _), a, b, c, d in zip(chunk, j0, j1, n0, n1):
                done[k] = float(a - b)          # groundedness drop (judge)
                done_nll[k] = float(d - c)      # NLL rise after evidence removal

            print("   %d/%d" % (min(i + args.bs, len(todo)), len(todo)), flush=True)
        D = np.array([done.get("%d|%s" % (r["pid"], r["answer"]), np.nan) for r in rows])
        DN = np.array([done_nll.get("%d|%s" % (r["pid"], r["answer"]), np.nan) for r in rows])
        y = np.array([r["label"] for r in rows], np.int32)
        np.savez_compressed(outp, drop=D, dnll=DN, y=y, idx_te=np.asarray(te))
        ok = ~np.isnan(D[te])
        e = {"ds": args.ds, "level": lv, "tag": args.tag, "topk": args.topk,
             "n_test": int(ok.sum())}
        if ok.sum() > 20 and len(set(y[te][ok].tolist())) > 1:
            e["vihd"] = float(C.best_orientation(y[te][ok], D[te][ok])[1])
        ok2 = ~np.isnan(DN[te])
        if ok2.sum() > 20 and len(set(y[te][ok2].tolist())) > 1:
            e["vihd_nll"] = float(C.best_orientation(y[te][ok2], DN[te][ok2])[1])
            e["n_test_nll"] = int(ok2.sum())
        p = os.path.join(OUT, "vlm2_vihd_%s_%s.json" % (args.ds, args.tag))
        prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        prev[lv] = e
        C.save_json(prev, p)
        print("[vlm2] %s %s vihd(judge)=%s vihd(nll)=%s" % (args.ds, lv, e.get("vihd"), e.get("vihd_nll")), flush=True)


# ------------------------------------------------------------------ VASE family
def cmd_gen(args):
    samples = get_samples(args.ds)
    path = os.path.join(CACHE, "gen_%s_%s.jsonl" % (args.ds, args.tag))
    done = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["image_id"])
            except Exception:
                pass
    todo = [s for s in samples if s["image_id"] not in done]
    print("[vlm2] gen %s/%s todo=%d k=%d" % (args.ds, args.tag, len(todo), args.k), flush=True)
    if not todo:
        return
    ad = Adapter(DEFAULT_MODELS.get(args.tag, args.tag))
    fu = open(path, "a", encoding="utf-8")
    store = {}
    for rep in range(args.k + 1):
        for i in range(0, len(todo), args.bs):
            chunk = todo[i:i + args.bs]
            outs = ad.generate_batch([s["image"] for s in chunk], [s["question"] for s in chunk],
                                     do_sample=(rep > 0), max_new_tokens=args.max_new_tokens)
            for s, o in zip(chunk, outs):
                store.setdefault(s["image_id"], dict(image_id=s["image_id"], gid=s["gid"],
                                                     question=s["question"], answer=s.get("answer", ""),
                                                     samples=[]))["samples"].append(o)
        fu.truncate(0); fu.seek(0)
        for s in todo:
            if s["image_id"] in store:
                fu.write(json.dumps(store[s["image_id"]], ensure_ascii=False) + "\n")
        fu.flush()
        print("   rep %d/%d done" % (rep, args.k), flush=True)
    fu.close()


def cmd_vase(args):
    samples = get_samples(args.ds)
    path = os.path.join(CACHE, "gen_%s_%s.jsonl" % (args.ds, args.tag))
    if not os.path.exists(path):
        print("[vlm2] vase needs generations:", path, flush=True)
        return
    G = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        G[r["image_id"]] = r
    clip = C.ClipFeat()
    uniq = sorted({_ans_norm(g) for r in G.values() for g in r["samples"] if str(g).strip()})
    embs = np.stack(clip.text_feat_batch(uniq)) if uniq else np.zeros((0, 768), np.float32)
    tmap = {t: i for i, t in enumerate(uniq)}
    per = {}
    for r in G.values():
        keys = [_ans_norm(g) for g in r["samples"] if _ans_norm(g) in tmap]
        if not keys:
            continue
        E = embs[[tmap[k] for k in keys]]
        sizes = cluster_dist(E)
        p = np.array(sizes, float)
        p = p / p.sum()
        dom = np.argmax(p)
        per[r["image_id"]] = dict(
            se=float(-(p * np.log(p + 1e-12)).sum()),
            radflag=float(p.max()),
            dom_emb=E[list(np.argsort(-p))[:1][0]] if len(E) else None,
            cons_max=float(np.max(E[1:] @ E[0])) if len(E) > 1 else 1.0,
            embs=E)
    for lv in args.levels.split(","):
        tr, va, te, rows, audit = get_level(args.ds, lv, samples)
        y = np.array([r["label"] for r in rows], np.int32)
        clip.text_feat_batch(sorted({_ans_norm(r["answer"]) for r in rows}))
        e = {"ds": args.ds, "level": lv, "tag": args.tag, "n_test": int(len(te))}

        def emit(name, fn):
            s = np.array([fn(r) for r in rows], float)
            st = s[te]
            m = ~np.isnan(st)
            if m.sum() > 20 and len(set(y[te][m].tolist())) > 1:
                e[name] = float(C.best_orientation(y[te][m], st[m])[1])

        def P(r):
            return per.get(samples[r["pid"]]["image_id"])

        emit("semantic_entropy", lambda r: P(r)["se"] if P(r) else np.nan)
        emit("radflag", lambda r: P(r)["radflag"] if P(r) else np.nan)
        emit("answer_consistency", lambda r: P(r)["cons_max"] if P(r) else np.nan)
        emit("vase_visual", lambda r: (float(P(r)["dom_emb"] @ clip.text_feat(_ans_norm(r["answer"])))
                                       if P(r) and P(r)["dom_emb"] is not None else np.nan))
        p = os.path.join(OUT, "vlm2_vase_%s_%s.json" % (args.ds, args.tag))
        prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        prev[lv] = e
        C.save_json(prev, p)
        print("[vlm2] vase %s %s %s" % (args.ds, lv, {k: round(v, 3) for k, v in e.items()
                                                     if isinstance(v, float)}), flush=True)


def C_ClipPatches_init():
    """lazy singleton so cmd_vihd can fetch cached patch embeddings"""
    holder = {}

    def get(s, args):
        if "clip" not in holder:
            holder["clip"] = C.ClipFeat()
            try:
                holder["clip"].load_img_cache(args.ds)
            except Exception:
                pass
        return holder["clip"].image_feat(s["image"], s["gid"])[0].astype(np.float32)
    return get


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["judge", "vihd", "gen", "vase"])
    ap.add_argument("--ds", required=True)
    ap.add_argument("--tag", default="q3b")
    ap.add_argument("--levels", default="L4_fair")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()
    globals()["C_ClipPatches"] = C_ClipPatches_init()
    if args.mode == "judge":
        cmd_judge(args)
    elif args.mode == "vihd":
        globals()["C_ClipPatches"] = C_ClipPatches_init()
        cmd_vihd(args)
    elif args.mode == "gen":
        cmd_gen(args)
    else:
        cmd_vase(args)
