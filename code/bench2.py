 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bench-2: new datasets (SLAKE, POPE, Med-HallMark) x protocol ladder x detector set.

Reuses the protocol machinery and CLIP-feature cache from /root/nature_exp/exp/common.py.

Detectors implemented here (all need only frozen CLIP features):
    shortcut oracle (20 text features, GBM)
    CLIPScore / max patch-answer alignment
    MLP            [g,a,q]
    no-image MLP   [a,q]
    QCE-Net        [g,a,q,e_q,e_a]
    CEBaG          confidence-evidence Bayesian gain (log-odds addition)
    PhraseFC       phrase-grounded fact-checking
    PROVE          proof-guided regime-aware operator verification

VLM-based detectors (judge / NLL / VIHD intervention / VASE) live in vlm2.py.

Usage:  python bench2.py <dataset> [levels]
"""
import os
import sys
import json
import random
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, "/root/nature_exp/exp")
import common as C

BASE = "/root/hallu_bench"
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "out")
CACHE = os.path.join(BASE, "cache")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)

# =============================================================== loaders
def _pil(b):
    from PIL import Image
    import io
    return Image.open(io.BytesIO(b)).convert("RGB")


def load_slake(max_n=None):
    """Prefer the ORIGINAL loader from fair_data.py, so SLAKE is built exactly as in the
    paper (DATA_DIR/SLAKE/samples.json + data/source_xmlab*.jpg). Falls back to the HF
    mirror layout only if the original files are absent."""
    try:
        import fair_data as FD
        s = FD.load_slake_raw(max_n=max_n)
        if s and len(s) > 100:
            print("[bench2] SLAKE via original fair_data loader: %d items" % len(s), flush=True)
            return s
    except Exception as e:
        print("[bench2] original SLAKE loader unavailable (%s); using HF layout" % e, flush=True)
    return load_slake_hf(max_n=max_n)


def load_slake_hf(max_n=None, split="test"):
    """SLAKE from the HF mirror BoKelvin/SLAKE (imgs.zip extracted next to the JSONs).
    Schema: {img_id, img_name: 'xmlab102/source.jpg', question, answer,
             answer_type: OPEN|CLOSED, modality, content_type}"""
    root = os.path.join(DATA, "SLAKE_hf")
    jp = os.path.join(root, "%s.json" % split)
    if not os.path.exists(jp):
        jp = os.path.join(root, "test.json")
    items = json.load(open(jp, encoding="utf-8"))
    from PIL import Image
    out, cache = [], {}
    for it in items:
        name = str(it.get("img_name") or "")
        p = os.path.join(root, name)
        if not os.path.exists(p):
            continue
        if p not in cache:
            try:
                cache[p] = Image.open(p).convert("RGB")
            except Exception:
                cache[p] = None
        img = cache[p]
        q = it.get("question")
        ans = it.get("answer")
        if img is None or not q or not ans:
            continue
        if isinstance(ans, dict):
            ans = ans.get("label", "")
        gid = "slake_%s" % it.get("img_id", os.path.dirname(name))
        out.append(dict(image=img, question=str(q).strip(), answer=str(ans).strip(),
                        source="slake", gid=gid,
                        image_id="%s_%s" % (gid, it.get("qid", len(out))),
                        slake_modality=str(it.get("modality", "unknown")),
                        slake_ctype=str(it.get("content_type", "unknown")),
                        slake_ans_type=str(it.get("answer_type", "unknown"))))
        if max_n and len(out) >= max_n:
            break
    return out


def load_pope(max_n=None, split="random"):
    """POPE: items are (COCO image, 'Is there a X in the image?', yes/no).
    Each item yields a pre-matched yes/no pair: the ground-truth answer vs its flip."""
    from PIL import Image
    import glob
    jsons = sorted(glob.glob(os.path.join(DATA, "POPE_json", "*%s*.json" % split)))
    if not jsons:
        jsons = sorted(glob.glob(os.path.join(DATA, "POPE_json", "*.json")))
    imgdir = os.path.join(DATA, "val2017")
    imgdir_alt = os.path.join(DATA, "val2014")
    recs, cache = [], {}
    for jp in jsons:
        items = []
        raw = open(jp, encoding="utf-8").read().strip()
        try:                                   # array form
            obj = json.loads(raw)
            items = obj if isinstance(obj, list) else obj.get("data", [])
        except Exception:                      # JSONL form (this is what POPE ships)
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                if isinstance(o, dict):
                    items.append(o)
        if not items:
            continue
        for it in items:
            fn = str(it.get("image") or it.get("image_id") or "")
            if not fn.endswith(".jpg"):
                fn = fn + ".jpg"
            # POPE is built on COCO val2014; filenames look like COCO_val2014_000000123456.jpg
            bare = fn.replace("COCO_val2014_", "").replace("COCO_val2017_", "")
            cand = [os.path.join(imgdir, fn), os.path.join(imgdir, bare),
                    os.path.join(imgdir_alt, bare), os.path.join(imgdir_alt, fn)]
            p = next((c for c in cand if os.path.exists(c)), None)
            if p is None:
                continue
            if p not in cache:
                try:
                    cache[p] = Image.open(p).convert("RGB")
                except Exception:
                    cache[p] = None
            img = cache[p]
            if img is None:
                continue
            q = str(it.get("text") or it.get("question") or "").strip()
            lab = str(it.get("label", "")).strip().lower()
            if not q or lab not in ("yes", "no"):
                continue
            flip = "no" if lab == "yes" else "yes"
            gid = "pope_" + os.path.splitext(fn)[0]
            recs.append(dict(image=img, question=q, answer=lab, pos=lab, neg=flip, source="pope",
                             gid=gid, image_id=gid + "_" + str(it.get("question_id", len(recs)))))
            if max_n and len(recs) >= max_n:
                return recs
    return recs


def load_medhallmark(max_n=None):
    """Med-HallMark (750 items, HF parquet). NOTE: the 'response' field is the model's
    hallucination-CATEGORY string (e.g. 'Catastrophic Hallucinations') and
    'classification_label' is the ground-truth category, so this benchmark supports
    hallucination-TYPE classification, not the grounded/ungrounded answer pairs this
    protocol needs. Loaded for inspection / for a type-classification side analysis only."""
    import pyarrow.parquet as pq
    fp = os.path.join(DATA, "MedHallMark", "data-00000-of-00001.parquet")
    if not os.path.exists(fp):
        import glob
        g = glob.glob(os.path.join(DATA, "MedHallMark", "**", "*.parquet"), recursive=True)
        if not g:
            return []
        fp = g[0]
    tb = pq.read_table(fp)
    from PIL import Image
    import io
    out = []
    for r in tb.to_pylist():
        im = r.get("image") or {}
        try:
            img = Image.open(io.BytesIO(im["bytes"])).convert("RGB") if isinstance(im, dict) else None
        except Exception:
            img = None
        if img is None:
            continue
        out.append(dict(image=img, question=str(r.get("question", ""))[:400],
                        answer=str(r.get("response", "")).strip(),
                        source="medhallmark", gid="mhm_" + str(r.get("image_path", "")),
                        image_id="mhm_%d" % len(out),
                        task_type=str(r.get("task_type", "")),
                        label_hint=str(r.get("classification_label", ""))))
        if max_n and len(out) >= max_n:
            break
    return out


# =============================================================== pairs
def build_slake_level(samples, level, seed=42):
    return C.build_pairs_level(samples, level, seed=seed)


def build_pope_level(pope_samples, split="gid", seed=42, unmatched=False):
    """POPE items already define a matched yes/no pair; `unmatched` pairs the answer with
    a random *other item's* answer instead, giving a leaky contrast."""
    rng = random.Random(seed)
    recs = []
    for idx, s in enumerate(pope_samples):
        neg = s["neg"]
        if unmatched:
            j = rng.randrange(len(pope_samples))
            neg = pope_samples[j]["pos"] if rng.random() < 0.5 else pope_samples[j]["neg"]
            if neg == s["pos"]:
                neg = "no" if s["pos"] == "yes" else "yes"
        recs.append(dict(pid=idx, gid=s["gid"], image_id=s["image_id"],
                         question=s["question"], pos_answer=s["pos"], neg_answer=neg,
                         qt="yes_no", yn=True, alen=1, pos_len=1, neg_len=1,
                         match="pope_native" if not unmatched else "pope_unmatched",
                         modality="COCO", omni_qtype="object_presence",
                         dataset_name="POPE"))
    return recs, dict(Counter(r["match"] for r in recs))


# =============================================================== features / detectors
def _phrase(ans):
    """Content phrase of an answer: drop yes/no and stopwords."""
    STOP = set("the a an of in on at is are was were to for with and or not no yes this that it "
               "there image shows show seen present any some".split())
    toks = [w for w in str(ans).lower().replace(",", " ").replace(".", " ").split() if w not in STOP]
    return " ".join(toks) if toks else str(ans)


def detector_features(clip, samples, rows):
    """CLIP-based feature block used by every detector (plus per-row phrase grounding)."""
    F = C.features_for_rows(clip, samples, rows, verbose=False)
    phrases = [_phrase(r["answer"]) for r in rows]
    uniq = sorted(set(phrases))
    emb = {p: e for p, e in zip(uniq, clip.text_feat_batch(uniq))}
    n = len(rows)
    PMAX = np.zeros(n, np.float32)
    PTOP = np.zeros(n, np.float32)
    PMEAN = np.zeros(n, np.float32)
    for i, r in enumerate(rows):
        s = samples[r["pid"]]
        patches = clip.image_feat(s["image"], s["gid"])[0].astype(np.float32)
        v = patches @ emb[phrases[i]]
        PMAX[i] = v.max()
        PTOP[i] = np.sort(v)[-5:].mean()
        PMEAN[i] = v.mean()
    F["PHRASE_MAX"], F["PHRASE_TOP5"], F["PHRASE_MEAN"] = PMAX, PTOP, PMEAN
    return F


def fit_logistic(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(Xtr), ytr)
    return lr.predict_proba(sc.transform(Xte))[:, 1]


def run_dataset(ds, levels, ladder="full"):
    print("[bench2] %s start" % ds, flush=True)
    if ds == "slake":
        samples = load_slake()
        print("[bench2] SLAKE raw=%d" % len(samples), flush=True)
        build = build_slake_level
    elif ds.startswith("pope"):
        split = ds.split("_", 1)[1] if "_" in ds else "random"
        samples = load_pope(split=split)
        print("[bench2] POPE(%s) raw=%d" % (split, len(samples)), flush=True)
    elif ds == "medhallmark":
        samples = load_medhallmark()
        print("[bench2] Med-HallMark raw=%d" % len(samples), flush=True)
    else:
        raise SystemExit("unknown dataset " + ds)

    clip = C.ClipFeat()
    clip.load_img_cache(ds)
    res = {}

    for lv in levels:
        if ds == "slake":
            recs, audit = build(samples, lv)
            mode = C.LEVELS[lv]["split"]
        elif ds.startswith("pope"):
            unmatched = ("leaky" in lv)
            recs, audit = build_pope_level(samples, unmatched=unmatched)
            mode = "row" if "row" in lv else "gid"
        else:  # medhallmark: single matched protocol, items carry their own labels
            recs, audit = [], {}
            for idx, s in enumerate(samples):
                neg = s["answer"]
                recs.append(dict(pid=idx, gid=s["gid"], image_id=s["image_id"],
                                 question=s["question"], pos_answer=s["answer"], neg_answer=neg,
                                 qt=C.question_type(s["question"]), yn=C.answer_is_yesno(s["answer"]),
                                 alen=C.token_len(s["answer"]), pos_len=C.token_len(s["answer"]),
                                 neg_len=C.token_len(neg), match="native",
                                 modality="unknown", omni_qtype="unknown", dataset_name="Med-HallMark"))
            mode = "gid"
        rows = C.rows_from_recs(recs)
        tr, va, te = C.split_rows(rows, mode=mode)
        y = np.array([r["label"] for r in rows], np.int32)
        if len(te) < 40 or len(set(y[te].tolist())) < 2:
            print("[bench2] %s %s: too few test rows (%d)" % (ds, lv, len(te)), flush=True)
            continue

        F = detector_features(clip, samples, rows)
        np.savez_compressed(os.path.join(CACHE, "feat_%s_%s.npz" % (ds, lv)),
                            **{k: F[k] for k in ("G", "A", "Q", "EQ", "EA", "PATCH",
                                                 "PHRASE_MAX", "PHRASE_TOP5", "PHRASE_MEAN")},
                            y=y, idx_tr=np.asarray(tr), idx_va=np.asarray(va), idx_te=np.asarray(te))
        C.save_json([dict(pid=r["pid"], gid=r["gid"], question=r["question"], answer=r["answer"],
                          label=r["label"], qt=r["qt"], match=r["match"]) for r in rows],
                    os.path.join(CACHE, "rows_%s_%s.json" % (ds, lv)))

        e = dict(ds=ds, level=lv, n_rows=len(rows), n_train=len(tr), n_val=len(va),
                 n_test=len(te), pos_test=int(y[te].sum()), audit=audit, split=mode)

        # ---- shortcut oracle ----
        ans_freq, _ = C.pool_freqs(samples)
        Xl = np.asarray([C.leak_features(r, ans_freq) for r in rows], np.float32)
        from sklearn.ensemble import HistGradientBoostingClassifier
        gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
        gb.fit(Xl[tr], y[tr])
        e["shortcut_gbm"] = float(C.auroc(y[te], gb.predict_proba(Xl[te])[:, 1]))

        # ---- frozen alignment ----
        clip_s = (F["G"] * F["A"]).sum(1)
        maxp = F["PATCH"].max(1)
        e["clipscore"] = float(C.best_orientation(y[te], clip_s[te])[1])
        e["maxpatch"] = float(C.best_orientation(y[te], maxp[te])[1])

        # ---- trained heads ----
        Xm = np.concatenate([F["G"], F["A"], F["Q"]], 1)
        Xc = np.concatenate([F["G"], F["A"], F["Q"], F["EQ"], F["EA"]], 1)
        Xn = np.concatenate([F["A"], F["Q"]], 1)
        p_mlp, _, _ = C.train_mlp_scores(Xm[tr], Xm[va], Xm[te], y[tr], y[va], y[te])
        p_qce, _, _ = C.train_mlp_scores(Xc[tr], Xc[va], Xc[te], y[tr], y[va], y[te])
        p_ni, _, _ = C.train_mlp_scores(Xn[tr], Xn[va], Xn[te], y[tr], y[va], y[te])
        e["mlp"], e["qce"], e["mlp_noimage"] = (float(C.auroc(y[te], p_mlp)),
                                                float(C.auroc(y[te], p_qce)),
                                                float(C.auroc(y[te], p_ni)))
        np.savez_compressed(os.path.join(CACHE, "trained_%s_%s.npz" % (ds, lv)),
                            mlp_score=p_mlp, qce_score=p_qce, noimg_score=p_ni, y=y,
                            idx_te=np.asarray(te))

        # ---- PhraseFC: phrase-grounded fact checking ----
        Xp = np.stack([F["PHRASE_MAX"], F["PHRASE_TOP5"], F["PHRASE_MEAN"]], 1)
        s_ph = fit_logistic(Xp[tr], y[tr], Xp[te])
        e["phrase_fc"] = float(C.auroc(y[te], s_ph))

        # ---- CEBaG: Bayesian gain = log-odds of prior (model confidence) + evidence ----
        conf = clip_s  # image-answer confidence proxy
        ev = F["PHRASE_TOP5"]
        ce = np.stack([conf, ev, conf * ev], 1)
        s_ce = fit_logistic(ce[tr], y[tr], ce[te])
        e["cebag"] = float(C.auroc(y[te], s_ce))

        # ---- PROVE: regime-aware operator verification (one head per question regime) ----
        desc = np.stack([clip_s, maxp, F["PHRASE_MAX"], F["PHRASE_TOP5"], F["PHRASE_MEAN"],
                         np.array([C.token_len(r["answer"]) for r in rows], float),
                         np.array([1.0 if (r["qt"] == "yes_no") == (r["qt"] == "yes_no") else 0.0
                                   for r in rows], float)], 1)
        s_pv = np.zeros(len(rows))
        regimes = defaultdict(list)
        for i in tr:
            regimes[rows[i]["qt"]].append(i)
        for qt, idx in regimes.items():
            idx = np.asarray(idx)
            te_q = np.asarray([i for i in te if rows[i]["qt"] == qt])
            if len(idx) < 40 or len(set(y[idx].tolist())) < 2 or len(te_q) < 10:
                continue
            s_pv[te_q] = fit_logistic(desc[idx], y[idx], desc[te_q])
        te_arr = np.asarray(te)
        if (s_pv[te_arr] > 0).mean() > 0.5:
            e["prove"] = float(C.auroc(y[te], s_pv[te]))
        else:
            s_all = fit_logistic(desc[tr], y[tr], desc[te])
            e["prove"] = float(C.auroc(y[te], s_all))

        res[lv] = e
        print("[bench2] %-12s %-14s n=%4d  short=%.3f mlp=%.3f noimg=%.3f qce=%.3f ce=%.3f ph=%.3f pv=%.3f"
              % (ds, lv, len(te), e["shortcut_gbm"], e["mlp"], e["mlp_noimage"], e["qce"],
                 e["cebag"], e["phrase_fc"], e["prove"]), flush=True)
        C.save_json(res, os.path.join(OUT, "bench2_%s.json" % ds))

    clip.save_img_cache(ds)
    print("[bench2] %s done" % ds, flush=True)


if __name__ == "__main__":
    ds = sys.argv[1]
    levels = sys.argv[2:] or ["L4_fair"]
    run_dataset(ds, levels)
