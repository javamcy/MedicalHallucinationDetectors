 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Official-implementation scorers (CLIPScore, BiomedCLIP) on the matched protocol.

Motivation: the earlier comparison used our own re-implementations. Two of the published
methods have small, standard reference implementations that can be run exactly, and this
script runs those instead:

  * CLIPScore -- the official repository (github.com/jmhessel/clipscore). The score is
    2.5 * max(0, cos(image, text)) using the CLIP model the authors specify (ViT-B/32).
  * BiomedCLIP -- the official checkpoint (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
    loaded through open_clip, which is the usage the authors document.

Both are training-free, so they are evaluated on the matched-protocol test pairs exactly as
the other training-free detectors were.

Usage: python official_scorers.py <dataset> [level]
"""
import io
import os
import sys
import numpy as np

sys.path.insert(0, "/root/nature_exp/exp")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import bench2 as B2

OUT = "/root/hallu_bench/out"
MODELS = "/cloud/hf_models"
os.makedirs(OUT, exist_ok=True)


def get_rows(ds, level):
    if ds in ("vqa_rad", "path_vqa", "omni"):
        samples = C.load_raw(ds)
        recs, _ = C.build_pairs_level(samples, level)
        rows = C.rows_from_recs(recs)
        tr, va, te = C.split_rows(rows, mode=C.LEVELS[level]["split"])
        return samples, rows, te
    samples = B2.load_slake() if ds == "slake" else B2.load_pope(
        split=ds.split("_", 1)[1] if "_" in ds else "random")
    if ds == "slake":
        recs, _ = B2.build_slake_level(samples, level)
        mode = C.LEVELS[level]["split"]
    else:
        recs, _ = B2.build_pope_level(samples, unmatched=("leaky" in level))
        mode = "row" if "row" in level else "gid"
    rows = C.rows_from_recs(recs)
    tr, va, te = C.split_rows(rows, mode=mode)
    return samples, rows, te


def score_clipscore(samples, rows, device="cuda"):
    """Official CLIPScore: 2.5 * max(0, cos(image, text)) with ViT-B/32."""
    import clip
    import torch
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    cache = {}
    out = np.zeros(len(rows), np.float32)
    with torch.no_grad():
        for i, r in enumerate(rows):
            s = samples[r["pid"]]
            key = s["gid"]
            if key not in cache:
                cache[key] = model.encode_image(
                    preprocess(s["image"].convert("RGB")).unsqueeze(0).to(device)).float()
            it = cache[key]
            tx = model.encode_text(clip.tokenize([str(r["answer"])[:300]], truncate=True).to(device)).float()
            it = it / it.norm(dim=-1, keepdim=True)
            tx = tx / tx.norm(dim=-1, keepdim=True)
            out[i] = 2.5 * max(0.0, float((it * tx).sum().item()))
            if i % 2000 == 0:
                print("   clipscore %d/%d" % (i, len(rows)), flush=True)
    return out


def score_biomedclip(samples, rows, device="cuda"):
    """Official BiomedCLIP checkpoint, built from the authors' open_clip_config.json with the
    text tower taken from a local BiomedBERT copy, then loaded from their official state dict."""
    import json
    import torch
    import open_clip
    from transformers import AutoTokenizer
    d = os.path.join(MODELS, "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    pm = os.path.join(MODELS, "BiomedNLP-BiomedBERT-base-uncased-abstract")
    cfg = json.load(open(os.path.join(d, "open_clip_config.json")))
    mc = cfg["model_cfg"]
    mc["text_cfg"]["hf_model_name"] = pm
    mc["text_cfg"]["hf_tokenizer_name"] = pm
    model = open_clip.create_model("ViT-B-16", embed_dim=mc["embed_dim"],
                                   vision_cfg=mc["vision_cfg"], text_cfg=mc["text_cfg"])
    sd = torch.load(os.path.join(d, "open_clip_pytorch_model.bin"), map_location="cpu")
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(pm)
    preprocess = open_clip.image_transform(224, is_train=False)
    cache, out = {}, np.zeros(len(rows), np.float32)
    with torch.no_grad():
        for i, r in enumerate(rows):
            s = samples[r["pid"]]
            if s["gid"] not in cache:
                cache[s["gid"]] = model.encode_image(
                    preprocess(s["image"].convert("RGB")).unsqueeze(0).to(device)).float()
            it = cache[s["gid"]]
            tb = tok([str(r["answer"])[:300]], padding=True, truncation=True,
                     max_length=256, return_tensors="pt").to(device)
            tx = model.encode_text(tb).float()
            it = it / it.norm(dim=-1, keepdim=True)
            tx = tx / tx.norm(dim=-1, keepdim=True)
            out[i] = float((it * tx).sum().item())
            if i % 2000 == 0:
                print("   biomedclip %d/%d" % (i, len(rows)), flush=True)
    return out


def main():
    ds = sys.argv[1]
    level = sys.argv[2] if len(sys.argv) > 2 else (
        "L4_fair" if ds == "slake" else "native")
    samples, rows, te = get_rows(ds, level)
    y = np.array([r["label"] for r in rows], np.int32)
    print("[official] %s %s rows=%d test=%d" % (ds, level, len(rows), len(te)), flush=True)
    res = {"ds": ds, "level": level, "n_test": int(len(te))}
    for name, fn in (("clipscore_official", score_clipscore),
                     ("biomedclip_official", score_biomedclip)):
        try:
            s = fn(samples, rows)
            a = float(C.best_orientation(y[te], s[te])[1])
            res[name] = a
            print("[official] %s = %.4f" % (name, a), flush=True)
        except Exception as e:
            res[name] = None
            print("[official] %s FAILED: %s" % (name, str(e)[:200]), flush=True)
    p = os.path.join(OUT, "official_%s_%s.json" % (ds, level))
    C.save_json(res, p)
    print("saved", p, flush=True)


if __name__ == "__main__":
    main()
