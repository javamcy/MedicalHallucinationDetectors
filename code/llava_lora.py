 
"""LLaVA-LoRA hallucination detector under the fair protocol.

LoRA adapters (r=8, alpha=16) are trained on the q_proj/v_proj projections of
LLaVA-Med-1.5 to predict whether a candidate answer is grounded in the image
(tokens "grounded"/"ungrounded"). Model selection uses VAL AUROC only (early
stop when val declines by >0.004 after epoch 2); the reported number is the
TEST AUROC of the epoch with max val AUROC.

Usage: python llava_lora.py <dataset> [max_n] [n_sub] [epochs] [seed]
"""
import os, sys, glob, json
os.environ.setdefault("HF_HUB_OFFLINE", "0")
sys.path.insert(0, os.environ.get("LLAVA_MED_DIR", "LLaVA-Med"))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import llava.model.language_model.llava_mistral  # noqa: registers the architecture
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX

MODEL_SNAP = os.environ.get("LLAVA_MED_MODEL", "llava-med-v1.5-mistral-7b")
OUT = os.environ.get("RESULT_DIR", "results")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

DS = sys.argv[1] if len(sys.argv) > 1 else "vqa_rad"
MAX_N = int(sys.argv[2]) if len(sys.argv) > 2 else -1
N_SUB = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
EPOCHS = int(sys.argv[4]) if len(sys.argv) > 4 else 10
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 0
LR = 3e-5
GRAD_CLIP = 1.0
if MAX_N < 0:
    MAX_N = None

tokenizer, model, image_processor, ctx = load_pretrained_model(
    MODEL_SNAP, None, "llava_med_mistral", device=DEV)
model.eval()
# The LLaVA-Med-1.5 builder hardcodes float16, which overflows during LoRA
# training (params become NaN after one AdamW step). bf16 keeps training stable
# and matches the model's native precision.
model.to(torch.bfloat16)

from peft import LoraConfig, get_peft_model, TaskType
lora_cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16,
                      target_modules=["q_proj", "v_proj"], lora_dropout=0.1)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

POS_TOKEN = "grounded"
NEG_TOKEN = "ungrounded"


def make_inputs(img, q, a):
    qq = DEFAULT_IMAGE_TOKEN + "\nQuestion: " + q + "\nAnswer: " + a
    prompt = "USER: {}\nIs this answer grounded in the image? ASSISTANT:".format(qq)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(DEV)
    img_t = image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0].to(torch.bfloat16).to(DEV)
    attn = torch.ones_like(input_ids)
    return input_ids, img_t, attn


# Encode the labels without a leading space so that pos_id != neg_id across
# tokenizer versions (some transformers releases collapse " grounded" and
# " ungrounded" into the same token id).
pos_id = tokenizer.encode(POS_TOKEN, add_special_tokens=False)[0]
neg_id = tokenizer.encode(NEG_TOKEN, add_special_tokens=False)[0]
assert pos_id != neg_id, "pos_id==neg_id=%d: tokenizer split collapsed the two labels" % pos_id


def score_pairs(pairs):
    scores, ys = [], []
    for p in pairs:
        input_ids, img_t, attn = make_inputs(p["image"], p["question"], p["answer"])
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn, images=img_t.unsqueeze(0))
            last = out.logits[0, -1, :]
            d = (last[pos_id] - last[neg_id]).float().item()
            if not (d == d):  # NaN guard: no signal
                d = 0.0
            scores.append(d)
        ys.append(int(p["label"]))
    return np.array(scores), np.array(ys, dtype=np.int32)


def evaluate(pairs, name):
    sc, y = score_pairs(pairs)
    auc = roc_auc_score(y, -sc)
    print("  [%s] AUROC=%.4f (n=%d)" % (name, auc, len(pairs)), flush=True)
    return auc, sc, y


print("MODEL_SNAP=", MODEL_SNAP, flush=True)
from fair_data import prepare_dataset
train, val, test = prepare_dataset(DS, max_n=MAX_N)
train = list(train); val = list(val); test = list(test)
print("DS=%s max_n=%s train=%d val=%d test=%d" % (DS, str(MAX_N), len(train), len(val), len(test)), flush=True)

rng = np.random.RandomState(SEED)
sub = [train[i] for i in rng.choice(len(train), min(N_SUB, len(train)), replace=False)]

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
lossf = torch.nn.CrossEntropyLoss()
bs = 4

best_val = -1.0
best_epoch = -1
best_scores_test = None
best_y_test = None
for ep in range(EPOCHS):
    model.train()
    idx = rng.permutation(len(sub))
    total = 0; correct = 0
    for i in range(0, len(sub), bs):
        batch = idx[i:i + bs]
        for j in batch:
            p = sub[j]
            input_ids, img_t, attn = make_inputs(p["image"], p["question"], p["answer"])
            label_id = pos_id if p["label"] == 0 else neg_id
            target = torch.tensor([label_id], device=DEV)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, images=img_t.unsqueeze(0), labels=None)
            logits = out.logits[0, -1, :]
            loss = lossf(logits.unsqueeze(0), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], GRAD_CLIP)
            optimizer.step()
            pred = pos_id if logits[pos_id] > logits[neg_id] else neg_id
            correct += (pred == label_id); total += 1
        if (i // bs) % 250 == 0:
            print("  ep%d batch %d acc=%.3f" % (ep, i // bs, correct / max(total, 1)), flush=True)
    scheduler.step()
    print("  ep%d train acc=%.3f lr=%.6f" % (ep, correct / max(total, 1), scheduler.get_last_lr()[0]), flush=True)

    val_auc, val_sc, val_y = evaluate(val, "val_ep%d" % ep)
    test_auc, test_sc, test_y = evaluate(test, "test_ep%d" % ep)
    if val_auc > best_val:
        best_val = val_auc
        best_epoch = ep
        best_scores_test = test_sc
        best_y_test = test_y
    if ep >= 2 and val_auc < best_val - 0.004:
        print("  early stop (val declined) at epoch %d" % ep, flush=True)
        break

print("[%s_val] best epoch=%d best_val=%.4f -> test AUROC=%.4f (n=%d)" %
      (DS, best_epoch, best_val, roc_auc_score(best_y_test, -best_scores_test), len(test)), flush=True)

y_long = np.array([int(p["label"]) for p in test], dtype=np.int32)
suffix = "" if SEED == 0 else "_s%d" % SEED
np.savez(os.path.join(OUT, "%s_scores%s.npz" % (DS, suffix)),
         score=best_scores_test, y=y_long, label="1=hallucination",
         best_epoch=np.array([best_epoch]))
json.dump({
    "ds": DS, "seed": SEED, "test_auroc": float(roc_auc_score(best_y_test, -best_scores_test)),
    "n": len(test), "best_epoch": int(best_epoch), "best_val_auroc": float(best_val),
    "note": "val-earlystop lr%.0e-gc%.1f-%d-%dep maxn%s seed%d" % (LR, GRAD_CLIP, N_SUB, EPOCHS, str(MAX_N), SEED)},
    open(os.path.join(OUT, "%s%s.json" % (DS, suffix)), "w"))
print("[%s_val] saved." % DS, flush=True)
