 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Common utilities for the Nature-Perspective experiment batch.

Provides:
  * raw dataset loaders (VQA-RAD, Path-VQA, OmniMed-VQA-mini) with image-level gids
  * protocol-level pair construction (leaky -> fair), with build-audit metadata
  * split helpers (image-level vs row-level)
  * frozen CLIP-L/14 feature cache (image patches + global, text embeddings)
  * leakage / lexical feature extraction ("shortcut oracle" inputs)
  * evaluation helpers (AUROC with bootstrap CI, paired permutation test)
"""
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import io
import json
import math
import hashlib
import random
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

BASE = Path(os.environ.get("NAT_BASE", "/root/nature_exp"))
DATA_ROOT = Path(os.environ.get("DATA_DIR", "/root/scale_study/data"))
CACHE = BASE / "cache"
OUT = BASE / "out"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---------------------------------------------------------------------------
# Raw dataset loaders (mirrors paper_iclr fair_data.py, plus native-option info)
# ---------------------------------------------------------------------------


def _open_image_bytes(b):
    from PIL import Image
    return Image.open(io.BytesIO(b)).convert("RGB")


def _img_hash(img):
    try:
        return hashlib.md5(img.tobytes()).hexdigest()[:16]
    except Exception:
        return str(id(img))


def load_vqa_rad_raw(max_n=None):
    import pyarrow.parquet as pq
    samples, gid_counter = [], {}
    for split, fname in [
        ("train", "train-00000-of-00001-eb8844602202be60.parquet"),
        ("test", "test-00000-of-00001-e5bc3d208bb4deeb.parquet"),
    ]:
        df = pq.read_table(str(DATA_ROOT / "vqa_rad" / "data" / fname)).to_pandas()
        for _, row in df.iterrows():
            try:
                img = _open_image_bytes(row["image"]["bytes"])
            except Exception:
                continue
            q, a = str(row["question"]).strip(), str(row["answer"]).strip()
            if not q or not a:
                continue
            h = _img_hash(img)
            gid_counter.setdefault(h, "vqa_rad_%d" % len(gid_counter))
            samples.append(dict(image=img, question=q, answer=a, source="vqa_rad",
                                gid=gid_counter[h], image_id="%s_%d" % (gid_counter[h], len(samples))))
            if max_n and len(samples) >= max_n:
                break
        if max_n and len(samples) >= max_n:
            break
    return samples[:max_n] if max_n else samples


def load_path_vqa_raw(max_n=None):
    import pyarrow.parquet as pq
    samples, gid_counter = [], {}
    for f in sorted((DATA_ROOT / "Path_VQA" / "data").glob("train-*.parquet")):
        df = pq.read_table(str(f)).to_pandas()
        for _, row in df.iterrows():
            try:
                img = _open_image_bytes(row["image"]["bytes"])
            except Exception:
                continue
            q, a = str(row["question"]).strip(), str(row["answer"]).strip()
            if not q or not a:
                continue
            h = _img_hash(img)
            gid_counter.setdefault(h, "path_vqa_%d" % len(gid_counter))
            samples.append(dict(image=img, question=q, answer=a, source="path_vqa",
                                gid=gid_counter[h], image_id="%s_%d" % (gid_counter[h], len(samples))))
            if max_n and len(samples) >= max_n:
                break
        if max_n and len(samples) >= max_n:
            break
    return samples[:max_n] if max_n else samples


def load_omni_raw(max_n=None):
    """OmniMed-VQA-mini. Prefers the local saved_to_disk copy (no internet needed).

    Keeps the dataset's own MCQ options so we can build *native distractor*
    negatives (no sampling assumption)."""
    from datasets import load_from_disk, load_dataset
    local = DATA_ROOT / "omni_hf"
    ds = load_from_disk(str(local)) if local.exists() else load_dataset("simwit/omni-med-vqa-mini", split="test")
    samples = []
    for i, d in enumerate(ds):
        img = d.get("image")
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        q, a = str(d["question"]).strip(), str(d["gt_answer"]).strip()
        if not q or not a:
            continue
        opts = [str(d.get("option_%s" % L, "")).strip() for L in "ABCD"]
        opts = [o for o in opts if o]
        lab = str(d.get("label", "")).strip()
        gold = opts["ABCD".index(lab)] if lab in "ABCD" and len(opts) == 4 else a
        gid = "omni_" + str(d.get("image_path", i))
        samples.append(dict(image=img, question=q, answer=gold,
                            source="omni", gid=gid, image_id=gid + "_%d" % i,
                            options=opts, label=lab,
                            modality=str(d.get("modality", "unknown")),
                            omni_qtype=str(d.get("question_type", "unknown")),
                            dataset_name=str(d.get("dataset", "unknown"))))
        if max_n and len(samples) >= max_n:
            break
    return samples


LOADERS = {"vqa_rad": load_vqa_rad_raw, "path_vqa": load_path_vqa_raw, "omni": load_omni_raw}

# keep the paper's raw pools identical
MAXN = {"vqa_rad": None, "path_vqa": 6000, "omni": None}


def raw_path(ds):
    return CACHE / ("rawmeta_%s.json" % ds)


def load_raw(ds, cache=True):
    """Load raw samples. NAT_MAXN>0 slices for smoke tests."""
    samples = LOADERS[ds](max_n=MAXN.get(ds))
    lim = int(os.environ.get("NAT_MAXN", "0") or 0)
    if lim > 0:
        samples = samples[:lim]
    return samples


# ---------------------------------------------------------------------------
# question / answer typing
# ---------------------------------------------------------------------------

YESNO_Q_PREFIX = ("is ", "are ", "does ", "do ", "can ", "has ", "have ", "will ",
                  "was ", "were ", "did ", "could ", "should ", "would ")
QWORDS = ("what", "which", "where", "how", "why", "when")


def question_type(q):
    ql = str(q).lower().strip()
    if ql.startswith(YESNO_Q_PREFIX):
        return "yes_no"
    for w in QWORDS:
        if ql.startswith(w):
            return w
    return "other"


def answer_is_yesno(a):
    al = str(a).lower().strip().rstrip(".")
    return al in ("yes", "no", "true", "false", "y", "n")


def token_len(s):
    return len(str(s).split())


# ---------------------------------------------------------------------------
# Protocol levels
# ---------------------------------------------------------------------------

LEVELS = {
    "L0_leaky":     dict(length=False, type=False, split="row"),
    "L1_imgsplit":  dict(length=False, type=False, split="gid"),
    "L2_len":       dict(length=True,  type=False, split="gid"),
    "L3_type":      dict(length=False, type=True,  split="gid"),
    "L4_fair":      dict(length=True,  type=True,  split="gid"),
    "L5_fair_row":  dict(length=True,  type=True,  split="row"),
}
LEVEL_ORDER = ["L0_leaky", "L1_imgsplit", "L2_len", "L3_type", "L4_fair", "L5_fair_row"]
LEVEL_PRETTY = {
    "L0_leaky": "L0 random negatives, row-level split (leaky)",
    "L1_imgsplit": "L1 random negatives + image-level split",
    "L2_len": "L2 + length-matched negatives",
    "L3_type": "L3 + type-matched negatives",
    "L4_fair": "L4 length+type matched + image-level split (FAIR)",
    "L5_fair_row": "L5 length+type matched, row-level split",
    "L6_native": "L6 native distractor negatives (OmniMed only)",
}


YES_SET = {"yes", "true", "y"}
NO_SET = {"no", "false", "n"}


def answer_class(a):
    """Format class of an answer -- the thing a detector must NOT be able to read."""
    al = str(a).lower().strip().rstrip(".")
    if al in YES_SET or al in NO_SET:
        return "yesno"
    if any(c.isdigit() for c in al):
        return "numeric"
    n = token_len(a)
    if n <= 1:
        return "single"
    if n <= 3:
        return "short"
    return "long"


def answer_polarity(a):
    al = str(a).lower().strip().rstrip(".")
    if al in YES_SET:
        return "y"
    if al in NO_SET:
        return "n"
    return "o"


def _pool_index(samples):
    by_cls_len = defaultdict(list)   # (answer_class, token_len)
    by_cls = defaultdict(list)       # answer_class
    by_len = defaultdict(list)       # token_len
    for s in samples:
        s = dict(s)
        s["cls"] = answer_class(s["answer"])
        s["alen"] = token_len(s["answer"])
        s["pol"] = answer_polarity(s["answer"])
        by_cls_len[(s["cls"], s["alen"])].append(s)
        by_cls[s["cls"]].append(s)
        by_len[s["alen"]].append(s)
    return by_cls_len, by_cls, by_len


def _cand_from(pools, key, alen, window, gid, answer, need_class):
    out = []
    if key == "cls_len":
        for l in range(max(1, alen - window), alen + window + 1):
            for c in pools[0].get((need_class, l), ()):
                if c["gid"] != gid and c["answer"] != answer:
                    out.append(c)
    elif key == "cls":
        for c in pools[1].get(need_class, ()):
            if c["gid"] != gid and c["answer"] != answer:
                out.append(c)
    elif key == "len":
        for l in range(max(1, alen - window), alen + window + 1):
            for c in pools[2].get(l, ()):
                if c["gid"] != gid and c["answer"] != answer:
                    out.append(c)
    return out


def build_pairs_level(samples, level, seed=SEED, use_len=None, use_type=None, len_window=2):
    """One positive + one constraint-matched negative per sample.

    Matching is defined on the *answer* (never on the image or the question):
      * type matching  -> same answer FORMAT CLASS (yesno / numeric / single / short / long),
                          and for yes/no answers additionally the OPPOSITE polarity.
      * length matching -> |#tokens(neg) - #tokens(pos)| <= 2
    Because positives and negatives are drawn from the same answer pool, an
    unmatched protocol lets a detector win on answer length / answer format alone.

    Fallbacks are recorded so the construction can be audited.
    """
    cfg = LEVELS[level] if level in LEVELS else dict(length=True, type=True, split="gid")
    if use_len is None:
        use_len = cfg["length"]
    if use_type is None:
        use_type = cfg["type"]
    rng = random.Random(seed + 7919 * (LEVEL_ORDER.index(level) if level in LEVEL_ORDER else 99))
    pools = _pool_index(samples)
    recs, audit = [], Counter()

    for i, s in enumerate(samples):
        qt = question_type(s["question"])
        alen = token_len(s["answer"])
        cls = answer_class(s["answer"])
        pol = answer_polarity(s["answer"])
        gid, ans = s["gid"], s["answer"]

        cand, quality = None, "unmatched"
        if use_len and use_type:
            ws = (len_window,) + tuple(w for w in (5, 10) if w > len_window)
            for w in ws:
                c = _cand_from(pools, "cls_len", alen, w, gid, ans, cls)
                if c:
                    cand, quality = c, "type+len%d" % w
                    break
            if cand is None:
                c = _cand_from(pools, "cls", alen, 0, gid, ans, cls)
                cand, quality = (c, "type_only_fallback") if c else (None, "fallback")
        elif use_len:
            ws = (len_window,) + tuple(w for w in (5, 10) if w > len_window)
            for w in ws:
                c = _cand_from(pools, "len", alen, w, gid, ans, cls)
                if c:
                    cand, quality = c, "len%d" % w
                    break
            if cand is None:
                quality = "fallback"
        elif use_type:
            c = _cand_from(pools, "cls", alen, 0, gid, ans, cls)
            cand, quality = (c, "type_only") if c else (None, "fallback")
        else:
            c = [x for x in samples if x["gid"] != gid and x["answer"] != ans]
            cand, quality = (c, "unmatched") if c else (None, "none")

        if cand and use_type and cls == "yesno":
            # opposite polarity is mandatory when the answer format is yes/no
            opp = [c for c in cand if c["pol"] != pol or c["pol"] == "o"]
            cand = opp or cand

        if not cand:
            c = [x for x in samples if x["gid"] != gid and x["answer"] != ans]
            if not c:
                continue
            cand, quality = c, "fallback"

        neg = rng.choice(cand)
        audit[quality] += 1
        recs.append(dict(pid=i, gid=s["gid"], image_id=s["image_id"],
                         question=s["question"], pos_answer=s["answer"], neg_answer=neg["answer"],
                         qt=qt, yn=(cls == "yesno"), alen=alen, pos_len=alen,
                         neg_len=token_len(neg["answer"]), match=quality,
                         pos_cls=cls, neg_cls=answer_class(neg["answer"]),
                         modality=s.get("modality", "unknown"),
                         omni_qtype=s.get("omni_qtype", "unknown"),
                         dataset_name=s.get("dataset_name", "unknown")))
    return recs, dict(audit)


def build_native_pairs(samples, seed=SEED):
    """OmniMed-only: negatives are the dataset's own wrong MCQ options (no sampling)."""
    rng = random.Random(seed + 33331)
    recs, audit = [], Counter()
    for i, s in enumerate(samples):
        opts = s.get("options") or []
        if len(opts) < 2:
            continue
        wrong = [o for o in opts if o != s["answer"]]
        if not wrong:
            continue
        neg = rng.choice(wrong)
        qt, alen = question_type(s["question"]), token_len(s["answer"])
        audit["native_len2" if abs(token_len(neg) - alen) <= 2 else "native_lenwide"] += 1
        recs.append(dict(pid=i, gid=s["gid"], image_id=s["image_id"],
                         question=s["question"], pos_answer=s["answer"], neg_answer=neg,
                         qt=qt, yn=answer_is_yesno(s["answer"]), alen=alen,
                         pos_len=alen, neg_len=token_len(neg), match="native_distractor",
                         pos_cls=answer_class(s["answer"]), neg_cls=answer_class(neg),
                         modality=s.get("modality", "unknown"),
                         omni_qtype=s.get("omni_qtype", "unknown"),
                         dataset_name=s.get("dataset_name", "unknown")))
    return recs, dict(audit)


def rows_from_recs(recs):
    """Flatten paired records into detector rows (deterministic order: +, -, +, -, ...)."""
    rows = []
    for r in recs:
        rows.append(dict(pid=r["pid"], gid=r["gid"], question=r["question"], answer=r["pos_answer"],
                         label=0, qt=r["qt"], yn=r["yn"], neg_len=r["neg_len"], match=r["match"],
                         pos_cls=r.get("pos_cls", ""), neg_cls=r.get("neg_cls", ""),
                         modality=r.get("modality", "unknown"), omni_qtype=r.get("omni_qtype", "unknown"),
                         dataset_name=r.get("dataset_name", "unknown")))
        rows.append(dict(pid=r["pid"], gid=r["gid"], question=r["question"], answer=r["neg_answer"],
                         label=1, qt=r["qt"], yn=r["yn"], neg_len=r["neg_len"], match=r["match"],
                         modality=r.get("modality", "unknown"), omni_qtype=r.get("omni_qtype", "unknown"),
                         dataset_name=r.get("dataset_name", "unknown")))
    return rows


def split_rows(rows, mode="gid", seed=SEED, ratios=(0.6, 0.2, 0.2)):
    """Image-level ('gid') or row-level ('row') split. Returns index lists."""
    rng = random.Random(seed)
    if mode == "gid":
        keys = sorted(set(r["gid"] for r in rows))
        rng.shuffle(keys)
        n = len(keys)
        a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        tr, va, te = set(keys[:a]), set(keys[a:b]), set(keys[b:])
        idx_tr = [i for i, r in enumerate(rows) if r["gid"] in tr]
        idx_va = [i for i, r in enumerate(rows) if r["gid"] in va]
        idx_te = [i for i, r in enumerate(rows) if r["gid"] in te]
    else:
        idx = list(range(len(rows)))
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
        idx_tr, idx_va, idx_te = idx[:a], idx[a:b], idx[b:]
    return sorted(idx_tr), sorted(idx_va), sorted(idx_te)


def get_level(ds, level, samples=None, seed=SEED, use_len=None, use_type=None,
              len_window=2, split_mode=None):
    """Return (train_idx, val_idx, test_idx, rows, audit) for a dataset/level."""
    if samples is None:
        samples = load_raw(ds)
    if level == "L6_native":
        recs, audit = build_native_pairs(samples, seed=seed)
        mode = "gid"
    else:
        recs, audit = build_pairs_level(samples, level, seed=seed, use_len=use_len,
                                        use_type=use_type, len_window=len_window)
        mode = split_mode or LEVELS[level]["split"]
    rows = rows_from_recs(recs)
    tr, va, te = split_rows(rows, mode=mode, seed=seed)
    return tr, va, te, rows, audit


# ---------------------------------------------------------------------------
# Leakage / lexical features (shortcut oracle inputs)
# ---------------------------------------------------------------------------

LEX_STOP = set("the a an of in on at is are was were to for with and or not no yes this that it".split())


def leak_features(row, pool_freq):
    """Features that need NO image at all."""
    a, q = str(row["answer"]), str(row["question"])
    al, ql = token_len(a), token_len(q)
    at, qt = a.lower().strip(), q.lower().strip()
    yn = answer_is_yesno(a)
    f = [
        float(al), float(len(a)), float(ql),
        float(al) / (ql + 1e-6), float(len(a)) / (len(q) + 1e-6),
        1.0 if yn else 0.0,
        float(qt == "yes_no"),
        float(qt in ("what", "which", "where", "when", "who", "how")),
        float(qt == "other"),
        # type match between answer and question: yes/no answer to yes/no question, etc.
        float((qt == "yes_no") == yn),
        1.0 if any(c.isdigit() for c in a) else 0.0,
        1.0 if "%" in a else 0.0,
        1.0 if "," in a else 0.0,
        1.0 if a.strip().endswith(".") else 0.0,
        1.0 if a[:1].isupper() else 0.0,
        1.0 if at.startswith(("no ", "normal", "unremarkable", "without")) else 0.0,
        math.log1p(pool_freq.get(at, 0)),
        math.log1p(pool_freq.get(qt, 0)),
        float(len(set(at.split()) & set(qt.split()))),
        float(len(set(at.split()) & set(qt.split())) / (len(set(at.split()) | set(qt.split())) + 1e-6)),
    ]
    return f


def pool_freqs(samples):
    ans = Counter(str(s["answer"]).lower().strip() for s in samples)
    qs = Counter(str(s["question"]).lower().strip() for s in samples)
    return ans, qs


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def boot_ci(y, s, n=1000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    y, s = np.asarray(y), np.asarray(s)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        if len(set(y[b].tolist())) < 2:
            continue
        vals.append(auroc(y[b], s[b]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def perm_pvalue(y, s, n=2000, seed=0):
    """Two-sided permutation p-value for AUROC vs 0.5."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    obs = auroc(y, s)
    stat = abs(obs - 0.5)
    cnt = 0
    for _ in range(n):
        yp = rng.permutation(y)
        if abs(auroc(yp, s) - 0.5) >= stat - 1e-12:
            cnt += 1
    return float(obs), float((cnt + 1) / (n + 1))


def best_orientation(y, s):
    a = auroc(y, s)
    return (s, a) if a >= 0.5 else (-np.asarray(s), 1.0 - a)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)
    return str(path)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# Frozen CLIP-L/14 feature extraction with caching
# ---------------------------------------------------------------------------

CLIP_ID = os.environ.get("CLIP_ID", "openai/clip-vit-large-patch14")


class ClipFeat:
    """Frozen CLIP image-patch / global and text features with an on-disk cache."""

    def __init__(self, device="cuda", img_cache_tag="clipL14"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        # CPU float32 / GPU float16
        self.dtype = torch.float32 if self.dev == "cpu" else torch.float16
        self.clip = CLIPModel.from_pretrained(CLIP_ID, torch_dtype=self.dtype).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)
        self.tag = img_cache_tag
        self._img = {}
        self._txt = {}

    # -- disk-backed image feature cache (per dataset, keyed by gid) --
    def load_img_cache(self, ds):
        p = CACHE / ("imgfeat_%s_%s.npz" % (ds, self.tag))
        if p.exists():
            d = np.load(str(p), allow_pickle=True)
            keys = [str(k) for k in d["keys"]]
            P, G = d["P"], d["G"]
            self._img = {k: (P[i], G[i]) for i, k in enumerate(keys)}
            print("  [imgcache] loaded %d images from %s" % (len(keys), p.name), flush=True)
            return True
        return False

    def save_img_cache(self, ds):
        if not self._img:
            return
        keys = sorted(self._img.keys())
        P = np.stack([self._img[k][0] for k in keys])
        G = np.stack([self._img[k][1] for k in keys])
        np.savez(str(CACHE / ("imgfeat_%s_%s.npz" % (ds, self.tag))), keys=np.asarray(keys), P=P, G=G)
        print("  [imgcache] saved %d images (%s)" % (len(keys), ds), flush=True)

    # -- image --
    def image_feat(self, img, key, max_side=336):
        if key in self._img:
            return self._img[key]
        torch = self.torch
        import torch.nn.functional as F
        from PIL import Image
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            sc = max_side / max(w, h)
            img = img.resize((max(1, int(w * sc)), max(1, int(h * sc))))
        inp = self.proc(images=img, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            vo = self.clip.vision_model(pixel_values=inp["pixel_values"].to(self.dtype))
            hh = vo.last_hidden_state
            patches = F.normalize(self.clip.visual_projection(hh[:, 1:, :]).float(), dim=-1).squeeze(0).cpu().numpy()
            g = F.normalize(self.clip.visual_projection(hh[:, 0:1, :]).float(), dim=-1).squeeze(0).squeeze(0).cpu().numpy()
        self._img[key] = (patches.astype(np.float16), g.astype(np.float32))
        return self._img[key]

    # -- text --
    def text_feat(self, s):
        s = str(s)
        if s in self._txt:
            return self._txt[s]
        e = self.text_feat_batch([s])[0]
        return e

    def text_feat_batch(self, strings):
        torch = self.torch
        import torch.nn.functional as F
        todo = [s for s in strings if s not in self._txt]
        for i in range(0, len(todo), 256):
            chunk = todo[i:i + 256]
            inp = self.proc(text=chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=77).to(self.dev)
            with torch.no_grad():
                out = self.clip.text_model(**inp)
                ln = inp["attention_mask"].sum(1) - 1
                eos = torch.gather(out.last_hidden_state, 1,
                                   ln.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, out.last_hidden_state.shape[-1])).squeeze(1)
                feat = F.normalize(self.clip.text_projection(eos).float(), dim=-1).cpu().numpy()
            for s, e in zip(chunk, feat):
                self._txt[s] = e.astype(np.float32)
        return [self._txt[s] for s in strings]

    # -- derived evidence --
    def evidence(self, patches, q, a, kappa=2.0):
        wq = _softmax(patches @ q * kappa)
        wa = _softmax(patches @ a * kappa)
        eq = (wq[:, None] * patches).sum(0)
        ea = (wa[:, None] * patches).sum(0)
        return eq.astype(np.float32), ea.astype(np.float32), wq.astype(np.float32), wa.astype(np.float32)


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def features_for_rows(clip, samples, rows, cache_name=None, verbose=True):
    """Compute [g, a, q, e_q, e_a] for every row, caching image + text features."""
    n = len(rows)
    G = np.zeros((n, 768), np.float32)
    A = np.zeros((n, 768), np.float32)
    Q = np.zeros((n, 768), np.float32)
    EQ = np.zeros((n, 768), np.float32)
    EA = np.zeros((n, 768), np.float32)
    CONQ = np.zeros((n, 256), np.float32)
    CONA = np.zeros((n, 256), np.float32)
    PATCH = np.zeros((n, 256), np.float32)     # max patch-answer alignment
    # pre-warm text cache
    if verbose:
        print("  [feat] pre-encoding unique strings ...", flush=True)
    uniq_q = sorted(set(str(r["question"]) for r in rows))
    uniq_a = sorted(set(str(r["answer"]) for r in rows))
    clip.text_feat_batch(uniq_q)
    clip.text_feat_batch(uniq_a)
    for i, r in enumerate(rows):
        if verbose and i % 500 == 0:
            print("  [feat] %d/%d" % (i, n), flush=True)
        s = samples[r["pid"]]
        patches, g = clip.image_feat(s["image"], s["gid"])
        p32 = patches.astype(np.float32)
        a = clip.text_feat(r["answer"])
        q = clip.text_feat(r["question"])
        eq, ea, wq, wa = clip.evidence(p32, q, a)
        G[i], A[i], Q[i], EQ[i], EA[i] = g, a, q, eq, ea
        CONQ[i] = wq
        CONA[i] = wa
        PATCH[i] = p32 @ a
    return dict(G=G, A=A, Q=Q, EQ=EQ, EA=EA, CONQ=CONQ, CONA=CONA, PATCH=PATCH)


# ---------------------------------------------------------------------------
# MLP / QCE-Net training (identical to the paper's qce_mlp.py recipe)
# ---------------------------------------------------------------------------

def train_mlp_scores(Xtr, Xva, Xte, ytr, yva, yte, seed=0, epochs=120, device=None):
    import torch
    import torch.nn as nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class MLP(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
                nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 1))

        def forward(self, x):
            return self.net(x).squeeze(-1)

    mean, std = Xtr.mean(0), Xtr.std(0).clip(1e-6)
    Xtrn = torch.from_numpy(((Xtr - mean) / std).astype(np.float32)).to(dev)
    Xvan = torch.from_numpy(((Xva - mean) / std).astype(np.float32)).to(dev)
    Xten = torch.from_numpy(((Xte - mean) / std).astype(np.float32)).to(dev)
    ytn = torch.from_numpy(np.asarray(ytr)).float().to(dev)
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.BCEWithLogitsLoss()
    best_val, best_p, best_ep = -1.0, None, -1
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(Xtrn), device=dev)
        for i in range(0, len(Xtrn), 256):
            idx = perm[i:i + 256]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            lf(m(Xtrn[idx]), ytn[idx]).backward()
            opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(Xvan)).cpu().numpy()
            pt = torch.sigmoid(m(Xten)).cpu().numpy()
        av = auroc(yva, pv)
        if av == av and av > best_val:
            best_val, best_p, best_ep = av, pt.copy(), ep
    return best_p if best_p is not None else np.full(len(yte), 0.5), best_val, best_ep


def eval_npz(path, keys):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in keys if k in d.files}
