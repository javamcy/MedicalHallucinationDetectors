"""Reviewer-requested controls for the protocol-trained LLaVA-Med-1.5 LoRA detector.

Three controls, all on the same matched-rung (L4) pairs and the same splits as
the main experiment:

  full_noimage : train ONE LoRA adapter exactly as in llava_lora.py (real images),
                 then score the SAME adapter twice at test time -- once with the real
                 image and once with a zeroed image tensor.  Answers reviewer #1
                 ("the no-image ablation was only done for the MLP head") and the
                 first half of #8.
  textonly     : train a separate LoRA adapter with the image token and image tensor
                 removed in BOTH training and testing.  Second half of #8.
  zeroshot_vqa : no LoRA training at all; score each candidate answer by its mean
                 token log-likelihood under the plain VQA prompt
                 P(answer | image, question).  Third control of #8.

Every run writes per-sample scores (npz), AUROC / paired bootstrap intervals
(json), and prints the numbers.  Model selection uses VAL AUROC only, as in the
paper; the reported number is the TEST AUROC of the best-val epoch.

Usage:
  python lora_controls.py <dataset> <mode> [max_n] [epochs] [seed]

Environment (set by run_controls.sh):
  PYTHONPATH      includes peft, /root/hallu/code, /root/hallu/LLaVA-Med
  DATA_DIR        /root/hallu/data
  MODEL_SNAP      /root/hallu/models/llava-med-v1.5-mistral-7b
  RESULT_DIR      /root/hallu/results/controls
"""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.environ.get("LLAVA_MED_DIR", "/root/hallu/LLaVA-Med"))
import llava.model.language_model.llava_mistral  # noqa: F401 registers architecture
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

DS = sys.argv[1] if len(sys.argv) > 1 else "vqa_rad"
MODE = sys.argv[2] if len(sys.argv) > 2 else "full_noimage"
_ARG_MAX_N = int(sys.argv[3]) if len(sys.argv) > 3 else -1
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
# Verified against the published LoRA result files: VQA-RAD 928, Path-VQA 1676,
# OmniMed 802, SLAKE 2008 test rows.  Path-VQA was run with max_n=4000
# (prepare_dataset('path_vqa', 4000) -> test=1676); the others used the full set.
# A negative CLI value means "use the published setting for this dataset".
_PUBLISHED_MAX_N = {"vqa_rad": None, "path_vqa": 4000, "omni": None, "slake": None}
MAX_N = _PUBLISHED_MAX_N.get(DS, None) if _ARG_MAX_N < 0 else _ARG_MAX_N
if MAX_N is not None and MAX_N < 0:
    MAX_N = None

assert MODE in ("full_noimage", "textonly", "zeroshot_vqa", "pixelshuffle"), MODE

MODEL_SNAP = os.environ.get("MODEL_SNAP", "/root/hallu/models/llava-med-v1.5-mistral-7b")
OUT = os.environ.get("RESULT_DIR", "/root/hallu/results/controls")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"

LR = 3e-5
GRAD_CLIP = 1.0
N_SUB = 6000
BS = 4
POS_TOKEN = "grounded"
NEG_TOKEN = "ungrounded"


def log(*a):
    print("[%s|%s]" % (DS, MODE), *a, time.strftime("%T"), flush=True)


log("loading", MODEL_SNAP)
tokenizer, model, image_processor, ctx = load_pretrained_model(
    MODEL_SNAP, None, "llava_med_mistral", device=DEV)
model.eval()
model.to(torch.bfloat16)

LORA = MODE in ("full_noimage", "textonly", "pixelshuffle")
if LORA:
    from peft import LoraConfig, get_peft_model, TaskType
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                     target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

pos_id = tokenizer.encode(POS_TOKEN, add_special_tokens=False)[0]
neg_id = tokenizer.encode(NEG_TOKEN, add_special_tokens=False)[0]
assert pos_id != neg_id, "tokenizer collapsed the two label tokens"


_SHUF = {"i": 0}


def _shuffle_pixels(t):
    """Permute the spatial positions of one image, keeping every pixel value.

    The image token stays in the prompt and the tensor keeps its shape and its
    marginal pixel distribution, so this control removes spatial structure only,
    not the fact that an image is present.
    """
    _SHUF["i"] += 1
    g = torch.Generator().manual_seed(90001 + _SHUF["i"])
    if t.dim() == 3 and t.shape[0] == 3:              # C, H, W
        c, h, w = t.shape
        perm = torch.randperm(h * w, generator=g)
        return t.reshape(c, h * w)[:, perm].reshape(c, h, w)
    if t.dim() == 3 and t.shape[-1] == 3:             # H, W, C
        h, w, c = t.shape
        perm = torch.randperm(h * w, generator=g)
        return t.reshape(h * w, c)[perm].reshape(h, w, c)
    return t


def make_inputs(img, q, a, blank=False, no_image=False, shuffle=False):
    """Return (input_ids, image_tensor_or_None, attention_mask).

    no_image=True drops the <image> token entirely (text-only LoRA).
    blank=True keeps the token but zeroes the pixels (test-time no-image control).
    shuffle=True keeps the token and the pixels but permutes their positions.
    """
    qq = ("" if no_image else DEFAULT_IMAGE_TOKEN + "\n") + "Question: " + q + "\nAnswer: " + a
    prompt = "USER: {}\nIs this answer grounded in the image? ASSISTANT:".format(qq)
    ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                return_tensors="pt").unsqueeze(0).to(DEV)
    if no_image:
        return ids, None, torch.ones_like(ids)
    img_t = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
    img_t = img_t.to(torch.bfloat16).to(DEV)
    if blank:
        img_t = torch.zeros_like(img_t)
    if shuffle:
        img_t = _shuffle_pixels(img_t)
    return ids, img_t, torch.ones_like(ids)


def fwd(ids, img_t, attn):
    kw = dict(input_ids=ids, attention_mask=attn)
    if img_t is not None:
        kw["images"] = img_t.unsqueeze(0)
    return model(**kw)


def score_grounded(pairs, blank=False, no_image=False, shuffle=False):
    scores, ys = [], []
    for p in pairs:
        ids, img_t, attn = make_inputs(p["image"], p["question"], p["answer"],
                                       blank=blank, no_image=no_image,
                                       shuffle=shuffle)
        with torch.no_grad():
            last = fwd(ids, img_t, attn).logits[0, -1, :]
            d = (last[pos_id] - last[neg_id]).float().item()
            if not (d == d):
                d = 0.0
        scores.append(d)
        ys.append(int(p["label"]))
    return np.asarray(scores), np.asarray(ys, dtype=np.int32)


def score_vqa_likelihood(pairs):
    """Zero-shot VQA-style control: mean log P(candidate answer | image, question).

    No LoRA, no grounded/ungrounded head; the model simply scores how likely the
    candidate answer is as an answer to the question about the image.
    """
    scores, ys = [], []
    logsoft = torch.nn.LogSoftmax(dim=-1)
    for p in pairs:
        q, a = p["question"], p["answer"]
        prefix = "USER: " + DEFAULT_IMAGE_TOKEN + "\nQuestion: " + q + "\nAnswer:"
        full = prefix + " " + a
        pids = tokenizer_image_token(prefix, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors="pt").unsqueeze(0).to(DEV)
        fids = tokenizer_image_token(full, tokenizer, IMAGE_TOKEN_INDEX,
                                     return_tensors="pt").unsqueeze(0).to(DEV)
        start = pids.shape[1]
        if fids.shape[1] <= start:
            scores.append(0.0)
            ys.append(int(p["label"]))
            continue
        img_t = image_processor.preprocess(p["image"], return_tensors="pt")["pixel_values"][0]
        img_t = img_t.to(torch.bfloat16).to(DEV)
        with torch.no_grad():
            out = model(input_ids=fids, attention_mask=torch.ones_like(fids),
                        images=img_t.unsqueeze(0))
            lg = logsoft(out.logits[0, start - 1:-1, :].float())
            tgt = fids[0, start:].unsqueeze(-1)
            lp = lg.gather(1, tgt).squeeze(-1)
            v = lp.mean().item()
        scores.append(v if v == v else 0.0)
        ys.append(int(p["label"]))
    return np.asarray(scores), np.asarray(ys, dtype=np.int32)


def auc(y, s):
    """Hallucination is the positive class; all our scores are grounded-ness."""
    return float(roc_auc_score(y, -s))


def paired_ci(y, s_a, s_b, n_boot=1000, seed=123):
    """Percentile bootstrap CI for auc(y,s_a) - auc(y,s_b), paired over rows."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    diffs = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        ya = y[b]
        if ya.min() == ya.max():
            continue
        diffs.append(roc_auc_score(ya, -s_a[b]) - roc_auc_score(ya, -s_b[b]))
    if not diffs:
        return [float("nan"), float("nan")]
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def save(name, sc, y, extra):
    suff = "" if SEED == 0 else "_s%d" % SEED
    np.savez(os.path.join(OUT, "%s_%s%s_scores.npz" % (DS, name, suff)),
             score=sc, y=y, label="1=hallucination")
    d = dict(ds=DS, mode=MODE, variant=name, seed=SEED, n=len(y),
             test_auroc=auc(y, sc))
    d.update(extra)
    with open(os.path.join(OUT, "%s_%s%s.json" % (DS, name, suff)), "w") as f:
        json.dump(d, f, indent=2)
    log("saved", name, "AUROC=%.4f" % d["test_auroc"], "n=%d" % len(y))


def main():
    sys.path.insert(0, "/root/hallu/code")
    if DS.startswith("pope"):
        import pope_ds
        pope_ds.install()
    from fair_data import prepare_dataset
    train, val, test = prepare_dataset(DS, max_n=MAX_N)
    train, val, test = list(train), list(val), list(test)
    log("data train=%d val=%d test=%d" % (len(train), len(val), len(test)))

    tag = "%s_%s" % (DS, MODE)

    if MODE == "zeroshot_vqa":
        sc, y = score_vqa_likelihood(test)
        save(MODE, sc, y, {"note": "no training; mean log P(answer|image,question)"})
        return

    rng = np.random.RandomState(SEED)
    sub = [train[i] for i in rng.choice(len(train), min(N_SUB, len(train)), replace=False)]
    no_image = MODE == "textonly"

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    lossf = torch.nn.CrossEntropyLoss()

    best_val, best_ep = -1.0, -1
    best = None
    for ep in range(EPOCHS):
        model.train()
        idx = rng.permutation(len(sub))
        total, correct = 0, 0
        for i in range(0, len(sub), BS):
            for j in idx[i:i + BS]:
                p = sub[j]
                ids, img_t, attn = make_inputs(p["image"], p["question"], p["answer"],
                                               no_image=no_image)
                label_id = pos_id if p["label"] == 0 else neg_id
                target = torch.tensor([label_id], device=DEV)
                optimizer.zero_grad()
                out = fwd(ids, img_t, attn)
                lg = out.logits[0, -1, :]
                loss = lossf(lg.unsqueeze(0), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
                optimizer.step()
                pred = pos_id if lg[pos_id] > lg[neg_id] else neg_id
                correct += int(pred == label_id)
                total += 1
            if (i // BS) % 250 == 0:
                log("ep%d batch%d acc=%.3f" % (ep, i // BS, correct / max(total, 1)))
        scheduler.step()

        model.eval()
        val_sc, val_y = score_grounded(val, no_image=no_image)
        val_auc = auc(val_y, val_sc)
        acc = correct / max(total, 1)

        # Only the best-val epoch's test scores are ever saved.  Scoring the
        # test split on every epoch is pure wasted inference and changes no
        # reported number: the retained scores are from the same best epoch.
        if val_auc > best_val:
            best_val, best_ep = val_auc, ep
            test_sc, test_y = score_grounded(test, no_image=no_image)
            best = (test_sc, test_y)
            if MODE == "full_noimage":
                best = (test_sc, test_y,
                        score_grounded(test, blank=True, no_image=False)[0])
            if MODE == "pixelshuffle":
                best = (test_sc, test_y,
                        score_grounded(test, blank=True, no_image=False)[0],
                        score_grounded(test, shuffle=True, no_image=False)[0])
                # dump as soon as a best epoch exists, so an interrupted run still
                # leaves the three-condition comparison on disk
                np.savez(os.path.join(OUT, "%s_pixelshuffle_partial.npz" % DS),
                         full=test_sc, zeroed=best[2], shuffle=best[3], y=test_y)
                _d = dict(ds=DS, mode=MODE, variant="partial", seed=SEED,
                          best_epoch=ep, best_val_auroc=best_val, n=len(test_y),
                          auroc_full=auc(test_y, test_sc),
                          auroc_zeroed=auc(test_y, best[2]),
                          auroc_shuffle=auc(test_y, best[3]))
                with open(os.path.join(OUT, "%s_pixelshuffle_partial.json" % DS), "w") as _f:
                    json.dump(_d, _f, indent=2)
                log("partial full=%.4f zeroed=%.4f shuffle=%.4f" %
                    (_d["auroc_full"], _d["auroc_zeroed"], _d["auroc_shuffle"]))
            log("ep%d train_acc=%.3f val=%.4f test=%.4f  [best]" %
                (ep, acc, val_auc, auc(test_y, test_sc)))
        else:
            log("ep%d train_acc=%.3f val=%.4f  [test scoring skipped]" %
                (ep, acc, val_auc))
        if ep >= 2 and val_auc < best_val - 0.004:
            log("early stop at epoch", ep)
            break

    if MODE == "full_noimage":
        sc_real, y, sc_blank = best
        delta = auc(y, sc_real) - auc(y, sc_blank)
        ci = paired_ci(y, sc_real, sc_blank)
        save("lora_full", sc_real, y,
             {"best_epoch": best_ep, "best_val_auroc": best_val,
              "delta_full_minus_noimage": delta, "delta_ci95": ci})
        save("lora_noimage_test", sc_blank, y,
             {"best_epoch": best_ep, "best_val_auroc": best_val,
              "note": "SAME adapter as lora_full, zeroed image tensor at test time"})
        log("PAIR full=%.4f noimage=%.4f delta=%.4f CI=%s" %
            (auc(y, sc_real), auc(y, sc_blank), delta, ci))
    elif MODE == "pixelshuffle":
        sc_real, y, sc_blank, sc_shuf = best
        d_blank = auc(y, sc_real) - auc(y, sc_blank)
        d_shuf = auc(y, sc_real) - auc(y, sc_shuf)
        save("lora_shuffle_test", sc_shuf, y,
             {"best_epoch": best_ep, "best_val_auroc": best_val,
              "delta_full_minus_shuffle": d_shuf,
              "delta_ci95": paired_ci(y, sc_real, sc_shuf),
              "delta_full_minus_zeroed": d_blank,
              "delta_zeroed_ci95": paired_ci(y, sc_real, sc_blank),
              "note": "SAME adapter as lora_full; pixels permuted at test time"})
        log("SHUF full=%.4f shuffle=%.4f zeroed=%.4f d_shuf=%.4f CI=%s" %
            (auc(y, sc_real), auc(y, sc_shuf), auc(y, sc_blank), d_shuf,
             paired_ci(y, sc_real, sc_shuf)))
    else:
        sc, y = best[0], best[1]
        save("lora_textonly", sc, y,
             {"best_epoch": best_ep, "best_val_auroc": best_val,
              "note": "image token and image tensor absent in training and testing"})

    log("DONE", tag)


if __name__ == "__main__":
    main()
