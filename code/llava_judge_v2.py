 
import os, sys, json, numpy as np, torch
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/root/LLaVA"); sys.path.insert(0, "/root/hallu_bench/code")
import transformers; transformers.logging.set_verbosity_error()
from PIL import Image
import llava.model.language_model.llava_mistral  # noqa
import vase_reproduction as VR, bench2 as B2, vlm2 as V2

PROMPT = "Is this answer grounded in the image?"

def main(ds, limit=0):
    tok, model, ip, _, _ = VR.load_all()
    prm = next(model.parameters()); dev, dt = prm.device, prm.dtype
    _p = tok.encode(" grounded", add_special_tokens=False)
    _n = tok.encode(" ungrounded", add_special_tokens=False)
    _k = next(i for i in range(min(len(_p), len(_n))) if _p[i] != _n[i])
    p_id, n_id = _p[_k], _n[_k]
    print("[ids] pos=%d neg=%d at token index %d" % (p_id, n_id, _k), flush=True)
    samples = B2.load_slake() if ds == "slake" else B2.load_pope(split="random")
    tr, va, te, rows, _ = V2.get_level(ds, "L4_fair", samples)
    te = list(te)
    if limit: te = te[:limit]
    got = {}
    for k, i in enumerate(te):
        r = rows[int(i)]; s = samples[r["pid"]]
        img = s["image"]
        if isinstance(img, str): img = Image.open(img).convert("RGB")
        q = "%s\nAnswer: %s\n%s" % (r["question"], r["answer"], PROMPT)
        ids, img_t, attn = VR.make_inputs(tok, ip, img, q)
        im = (img_t.unsqueeze(0) if img_t.dim() == 3 else img_t).to(dev).to(dt)
        ids = ids.to(dev); attn = attn.to(dev)
        with torch.no_grad():
            o = model(input_ids=ids, images=im, attention_mask=attn, use_cache=False)
        lg = o.logits[0, -1]
        got[int(i)] = float(lg[p_id] - lg[n_id])
        if (k + 1) % 200 == 0: print("%s %d/%d" % (ds, k + 1, len(te)), flush=True)
    y = np.array([rows[int(i)]["label"] for i in te])
    sc = np.array([got[int(i)] for i in te])
    from sklearn.metrics import roc_auc_score
    a1 = roc_auc_score(y, sc); a2 = roc_auc_score(y, -sc)
    print("[judge] %s L4 judge=%.3f raw=%.3f n=%d" % (ds, max(a1, a2), min(a1, a2), len(y)), flush=True)
    np.savez("/root/hallu_bench/out/llava_judge_%s.npz" % ds, score=sc, label=y, idx=np.array(te))
    json.dump({"ds": ds, "auroc": float(max(a1, a2)), "raw_auroc": float(min(a1, a2)), "n": int(len(y))},
              open("/root/hallu_bench/out/llava_judge_%s.json" % ds, "w"))

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
