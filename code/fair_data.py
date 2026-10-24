 
"""Fair-protocol dataset construction for medical VQA hallucination detection.

The protocol removes two benchmark artifacts that leak a shortcut signal into
hallucination detectors:

1. Length bias: negatives are sampled so that |len(neg) - len(pos)| <= 2
   (whitespace tokens), so answer length carries no signal.
2. Type bias: negatives are sampled from the same question type; yes/no
   questions receive the opposite yes/no answer from another image, so the
   detector must look at the image to decide.

Splits are made at image level (by gid): no image straddles train/val/test.

Sample fields:
  gid      - image grouping key (all samples of one image share a gid)
  image_id - unique sample id
  image, question, answer, label, source
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import io
import json
import random
import hashlib
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image
import pyarrow.parquet as pq

DATA_ROOT = Path(os.environ.get("DATA_DIR", "data"))
SLAKE_IMG_DIR = DATA_ROOT / "SLAKE" / "data"

RNG = random.Random(42)


def _open_image_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _img_hash(img: Image.Image) -> str:
    """Content hash of the image, used as the grouping key."""
    try:
        return hashlib.md5(img.tobytes()).hexdigest()[:16]
    except Exception:
        return str(id(img))


# ---------------------------------------------------------------------------
# Raw dataset loading
# ---------------------------------------------------------------------------

def load_vqa_rad_raw(max_n=None):
    """VQA-RAD raw samples (train + test parquet files)."""
    samples = []
    gid_counter = {}
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
            q = str(row["question"]).strip()
            a = str(row["answer"]).strip()
            if not q or not a:
                continue
            h = _img_hash(img)
            if h not in gid_counter:
                gid_counter[h] = f"vqa_rad_{len(gid_counter)}"
            samples.append({
                "image": img, "question": q, "answer": a,
                "source": "vqa_rad", "split": split,
                "gid": gid_counter[h],
                "image_id": f"{gid_counter[h]}_{len(samples)}",
            })
            if max_n and len(samples) >= max_n:
                break
        if max_n and len(samples) >= max_n:
            break
    if max_n:
        samples = samples[:max_n]
    return samples


def _extract_answer_label(a):
    """SLAKE answers may be a string or a Classification dict."""
    if isinstance(a, str):
        return a.strip()
    if isinstance(a, dict):
        return str(a.get("label", "")).strip()
    return str(a).strip()


def load_slake_raw(max_n=None):
    """SLAKE raw samples (one image, multiple questions)."""
    samples = []
    sj = json.load(open(DATA_ROOT / "SLAKE" / "samples.json", "r", encoding="utf-8"))
    entries = sj.get("samples", []) if isinstance(sj, dict) else (sj if isinstance(sj, list) else [])
    for e in entries:
        if not isinstance(e, dict):
            continue
        img_name = e.get("img_name") or e.get("image") or e.get("filepath")
        if img_name is None:
            continue
        import re
        m = re.search(r"xmlab(\d+)", str(img_name))
        img_path = None
        if m:
            img_path = SLAKE_IMG_DIR / f"source_xmlab{m.group(1)}.jpg"
        if img_path is None or not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        gid = f"slake_{img_path.stem}"
        for i in range(8):
            q = e.get(f"question_{i}")
            a = e.get(f"answer_{i}")
            if q and a:
                samples.append({
                    "image": img, "question": str(q).strip(),
                    "answer": _extract_answer_label(a), "source": "slake",
                    "split": "train", "gid": gid,
                    "image_id": f"{gid}_{i}",
                })
    if max_n:
        samples = samples[:max_n]
    return samples


def load_path_vqa_raw(max_n=None):
    """Path-VQA raw parquet samples."""
    samples = []
    files = sorted((DATA_ROOT / "Path_VQA" / "data").glob("train-*.parquet"))
    gid_counter = {}
    for f in files:
        df = pq.read_table(str(f)).to_pandas()
        for _, row in df.iterrows():
            try:
                img = _open_image_bytes(row["image"]["bytes"])
            except Exception:
                continue
            q = str(row["question"]).strip()
            a = str(row["answer"]).strip()
            if not q or not a:
                continue
            h = _img_hash(img)
            if h not in gid_counter:
                gid_counter[h] = f"path_vqa_{len(gid_counter)}"
            samples.append({
                "image": img, "question": q, "answer": a,
                "source": "path_vqa", "split": "train",
                "gid": gid_counter[h],
                "image_id": f"{gid_counter[h]}_{len(samples)}",
            })
            if max_n and len(samples) >= max_n:
                break
        if max_n and len(samples) >= max_n:
            break
    if max_n:
        samples = samples[:max_n]
    return samples


def load_omni_raw(max_n=None):
    """OmniMed-VQA-mini (HF: simwit/omni-med-vqa-mini, test split)."""
    from datasets import load_dataset
    ds = load_dataset("simwit/omni-med-vqa-mini", split="test")
    samples = []
    for d in ds:
        try:
            img = d["image"]
        except Exception:
            continue
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        q = str(d["question"]).strip()
        a = str(d["gt_answer"]).strip()
        if not q or not a:
            continue
        gid = "omni_" + str(d["image_path"])
        samples.append({
            "image": img, "question": q, "answer": a,
            "source": "omni", "split": "train",
            "gid": gid,
            "image_id": gid + "_" + str(len(samples)),
        })
        if max_n and len(samples) >= max_n:
            break
    if max_n:
        samples = samples[:max_n]
    return samples


# ---------------------------------------------------------------------------
# Matched negative construction (length- and type-matched)
# ---------------------------------------------------------------------------

def _token_len(s: str) -> int:
    return len(str(s).split())


def _question_type(q: str) -> str:
    """Question type: yes_no / what / where / how / which / other."""
    ql = q.lower().strip()
    if ql.startswith(("is ", "are ", "does ", "do ", "can ", "has ", "have ",
                      "will ", "was ", "were ")):
        return "yes_no"
    for w in ("what", "which", "where", "how", "why", "when"):
        if ql.startswith(w):
            return w
    return "other"


def _answer_is_yesno(a: str) -> bool:
    al = a.lower().strip().rstrip(".")
    return al in ("yes", "no", "true", "false", "y", "n")


def build_pairs(samples, max_len_diff=2, seed=42, max_pairs=None, type_matched=True):
    """Build one negative pair per positive sample.

    Positive: (image, question, answer), label 0 (grounded).
    Negative: (image, question, wrong_answer), label 1 (hallucinated),
    where the wrong answer is length-matched (|len - len_pos| <= max_len_diff)
    and type-matched (same question type; opposite yes/no for yes/no questions),
    sampled from *other* images.
    """
    rng = random.Random(seed)
    by_type_len = defaultdict(list)
    for s in samples:
        qt = _question_type(s["question"])
        al = _answer_is_yesno(s["answer"])
        by_type_len[(qt, _token_len(s["answer"]), al)].append(s)
    yesno_pool = [s for s in samples if _answer_is_yesno(s["answer"])]

    pairs = []
    for s in samples:
        pairs.append({
            "image": s["image"], "question": s["question"],
            "answer": s["answer"], "label": 0,
            "gid": s["gid"], "image_id": s["image_id"], "source": s["source"],
        })

    n_neg = 0
    for s in samples:
        qt = _question_type(s["question"])
        pos_len = _token_len(s["answer"])
        pos_yesno = _answer_is_yesno(s["answer"])

        if type_matched and qt == "yes_no":
            # yes/no question: use the opposite yes/no answer from another image
            target = not pos_yesno
            cand = [c for c in yesno_pool
                    if c["gid"] != s["gid"] and _answer_is_yesno(c["answer"]) == target
                    and c["answer"] != s["answer"]]
            if cand:
                neg = rng.choice(cand)
                pairs.append({
                    "image": s["image"], "question": s["question"],
                    "answer": neg["answer"], "label": 1,
                    "gid": s["gid"], "image_id": s["image_id"], "source": s["source"],
                })
                n_neg += 1
                continue
            cand = [c for c in yesno_pool if c["gid"] != s["gid"] and c["answer"] != s["answer"]]
            if cand:
                neg = rng.choice(cand)
                pairs.append({
                    "image": s["image"], "question": s["question"],
                    "answer": neg["answer"], "label": 1,
                    "gid": s["gid"], "image_id": s["image_id"], "source": s["source"],
                })
                n_neg += 1
                continue
            cand = [c for c in samples if c["gid"] != s["gid"] and c["answer"] != s["answer"]]
            if cand:
                neg = rng.choice(cand)
                pairs.append({
                    "image": s["image"], "question": s["question"],
                    "answer": neg["answer"], "label": 1,
                    "gid": s["gid"], "image_id": s["image_id"], "source": s["source"],
                })
                n_neg += 1
            continue

        # other question types: same type + length match
        cand_lens = [l for l in range(max(1, pos_len - max_len_diff), pos_len + max_len_diff + 1)]
        candidates = []
        for l in cand_lens:
            candidates.extend(by_type_len.get((qt, l, pos_yesno), []))
        candidates = [c for c in candidates
                      if c["gid"] != s["gid"] and c["answer"] != s["answer"]]
        if not candidates:
            candidates = [c for c in samples
                          if c["gid"] != s["gid"] and c["answer"] != s["answer"]]
            if not candidates:
                continue
        neg = rng.choice(candidates)
        pairs.append({
            "image": s["image"], "question": s["question"],
            "answer": neg["answer"], "label": 1,
            "gid": s["gid"], "image_id": s["image_id"], "source": s["source"],
        })
        n_neg += 1
        if max_pairs and n_neg >= max_pairs:
            break
    return pairs


# ---------------------------------------------------------------------------
# Image-level splits
# ---------------------------------------------------------------------------

def split_by_image(pairs, train_ratio=0.6, val_ratio=0.2, seed=42):
    """Split by gid so that all samples of one image stay in one split."""
    gids = sorted(set(p["gid"] for p in pairs))
    rng = random.Random(seed)
    rng.shuffle(gids)
    n = len(gids)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    tr_set = set(gids[:n_tr])
    va_set = set(gids[n_tr:n_tr + n_va])
    te_set = set(gids[n_tr + n_va:])

    train = [p for p in pairs if p["gid"] in tr_set]
    val = [p for p in pairs if p["gid"] in va_set]
    test = [p for p in pairs if p["gid"] in te_set]
    return train, val, test


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def prepare_dataset(name, max_n=None, max_len_diff=2, seed=42, type_matched=True):
    """Return (train_pairs, val_pairs, test_pairs)."""
    if name == "vqa_rad":
        raw = load_vqa_rad_raw(max_n=max_n)
    elif name == "slake":
        raw = load_slake_raw(max_n=max_n)
    elif name == "path_vqa":
        raw = load_path_vqa_raw(max_n=max_n)
    elif name == "omni":
        raw = load_omni_raw(max_n=max_n)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    pairs = build_pairs(raw, max_len_diff=max_len_diff, seed=seed,
                        type_matched=type_matched)
    train, val, test = split_by_image(pairs, seed=seed)

    lens0 = [len(p["answer"].split()) for p in pairs if p["label"] == 0]
    lens1 = [len(p["answer"].split()) for p in pairs if p["label"] == 1]
    n_gid = len(set(p["gid"] for p in pairs))
    print(f"[{name}] raw={len(raw)} pairs={len(pairs)} n_img={n_gid} "
          f"train={len(train)} val={len(val)} test={len(test)}", flush=True)
    print(f"  len0: mean={np.mean(lens0):.2f}  len1: mean={np.mean(lens1):.2f}  "
          f"(bias removed: {abs(np.mean(lens0)-np.mean(lens1)):.2f})", flush=True)
    return train, val, test


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    for ds, mn in [("vqa_rad", None), ("slake", None), ("path_vqa", 6000), ("omni", None)]:
        train, val, test = prepare_dataset(ds, max_n=mn)
        p0 = [p for p in train if p["label"] == 0][0]
        p1 = [p for p in train if p["label"] == 1][0]
        print(f"  pos: q='{p0['question'][:50]}' a='{p0['answer'][:35]}'", flush=True)
        print(f"  neg: q='{p1['question'][:50]}' a='{p1['answer'][:35]}'", flush=True)
        print(flush=True)
