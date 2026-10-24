 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Register POPE with fair_data so the faithful LLaVA-LoRA script can train on it.

POPE items are (COCO image, "Is there a X in the image?", yes/no ground truth).
Each item yields a pre-matched pair: the ground-truth answer (label 0) and its flipped
polarity (label 1) on the same image and question. All answers are single yes/no tokens,
so length and format are matched by construction.
"""
import json
import os
import random
from collections import defaultdict

BASE = "/root/hallu_bench/data"

_IMG_CACHE = {}


def _img(path):
    from PIL import Image
    if path not in _IMG_CACHE:
        _IMG_CACHE[path] = Image.open(path).convert("RGB")
    return _IMG_CACHE[path]


def load_pope_raw(split="random", max_n=None):
    p = os.path.join(BASE, "POPE_json", "coco_pope_%s.json" % split)
    raw = open(p, encoding="utf-8").read().strip()
    items = []
    try:
        obj = json.loads(raw)
        items = obj if isinstance(obj, list) else obj.get("data", [])
    except Exception:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    out = []
    for it in items:
        fn = str(it.get("image", ""))
        cand = [os.path.join(BASE, "val2014", fn),
                os.path.join(BASE, "val2017", fn),
                os.path.join(BASE, "val2017", fn.replace("COCO_val2014_", ""))]
        path = next((c for c in cand if os.path.exists(c)), None)
        if path is None:
            continue
        lab = str(it.get("label", "")).strip().lower()
        q = str(it.get("text", "")).strip()
        if lab not in ("yes", "no") or not q:
            continue
        flip = "no" if lab == "yes" else "yes"
        out.append(dict(image=_img(path), question=q, answer=lab, neg=flip,
                        gid="pope_" + os.path.splitext(os.path.basename(path))[0],
                        source="pope", image_id="pope_%s" % it.get("question_id", len(out))))
        if max_n and len(out) >= max_n:
            break
    return out


def prepare_pope(max_n=None, seed=42):
    samples = load_pope_raw(max_n=max_n)
    pairs = []
    for s in samples:
        for ans, lab in ((s["answer"], 0), (s["neg"], 1)):
            pairs.append(dict(image=s["image"], question=s["question"], answer=ans,
                              label=lab, gid=s["gid"], image_id=s["image_id"],
                              source="pope"))
    gids = sorted(set(p["gid"] for p in pairs))
    random.Random(seed).shuffle(gids)
    n = len(gids)
    a, b = int(n * 0.6), int(n * 0.8)
    tr, va, te = set(gids[:a]), set(gids[a:b]), set(gids[b:])
    train = [p for p in pairs if p["gid"] in tr]
    val = [p for p in pairs if p["gid"] in va]
    test = [p for p in pairs if p["gid"] in te]
    print("[pope] raw=%d pairs=%d n_img=%d train=%d val=%d test=%d"
          % (len(samples), len(pairs), n, len(train), len(val), len(test)), flush=True)
    return train, val, test


def install():
    """Monkey-patch fair_data.prepare_dataset to accept 'pope_random' / 'pope_popular'."""
    import fair_data
    orig = fair_data.prepare_dataset

    def prepare_dataset(name, max_n=None, **kw):
        if str(name).startswith("pope"):
            split = name.split("_", 1)[1] if "_" in name else "random"
            return prepare_pope(max_n=max_n)
        return orig(name, max_n=max_n, **kw)

    fair_data.prepare_dataset = prepare_dataset
    return fair_data
