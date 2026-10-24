 
"""Faithful full-scale reproduction of VASE (MICCAI 2025) under the fair protocol.

Reproduces the authors' algorithm and hyperparameters:
  - num_samples = 10, main answer temperature 0.1, sampling temperature 1.0
  - Gaussian input noise std = 0.07
  - DeBERTa transitive-closure entailment clustering (get_semantic_ids, GPU)
  - token-level semantic entropy (SeEnt)
  - ViSeEnt = entropy of softmax(SeDist + alpha * (SeDist - SeDist_noisy)), alpha = 1.0
  - backbone: LLaVA-Med-1.5

Usage: python vase_reproduction.py <dataset> [max_n]
"""
import os, sys, torch, glob, json, random
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.environ.get("LLAVA_MED_DIR", "LLaVA-Med"))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
from PIL import Image
import llava.model.language_model.llava_mistral  # noqa: registers the architecture
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from fair_data import prepare_dataset

MODEL_SNAP = os.environ.get("LLAVA_MED_MODEL", "llava-med-v1.5-mistral-7b")
DEBERTA_SNAP = os.environ.get("DEBERTA_MODEL", "microsoft/deberta-v2-xlarge-mnli")
OUT = os.environ.get("RESULT_DIR", "results")
os.makedirs(OUT, exist_ok=True)

NUM_SAMPLES = 10      # authors' parameter
NOISE_STD = 0.07      # authors' parameter
ALPHA = 1.0           # authors' parameter
MAX_NEW = 60
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_all():
    tokenizer, model, image_processor, ctx = load_pretrained_model(
        MODEL_SNAP, None, "llava_med_mistral", device=DEV)
    model.eval()
    dtok = AutoTokenizer.from_pretrained(DEBERTA_SNAP)
    dmodel = AutoModelForSequenceClassification.from_pretrained(DEBERTA_SNAP)
    dmodel = dmodel.to(DEV).half().eval()
    return tokenizer, model, image_processor, dtok, dmodel


def make_inputs(tokenizer, image_processor, image, question):
    qq = DEFAULT_IMAGE_TOKEN + "\n" + question
    prompt = "USER: {}\nASSISTANT:".format(qq)
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt").unsqueeze(0)
    img_t = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    attn = torch.ones_like(input_ids)
    return input_ids, img_t, attn


def generate_batch(model, tokenizer, input_ids, img_t, attn, num_samples, temp):
    ids = input_ids.repeat(num_samples, 1).to(DEV)
    imgs = img_t.repeat(num_samples, 1, 1, 1).half().to(DEV)
    am = attn.repeat(num_samples, 1).to(DEV)
    with torch.inference_mode():
        out = model.generate(ids, images=imgs, attention_mask=am,
                             max_new_tokens=MAX_NEW, do_sample=True, temperature=temp,
                             top_p=0.9, num_beams=1, use_cache=True,
                             pad_token_id=tokenizer.eos_token_id,
                             return_dict_in_generate=True, output_scores=True,
                             output_attentions=False)
    seqs = out.sequences[:, input_ids.shape[1]:]
    answers = tokenizer.batch_decode(seqs, skip_special_tokens=True)
    return answers, seqs, out.scores


def add_noise(img_t, std=NOISE_STD):
    noise = torch.randn_like(img_t) * std
    return torch.clamp(img_t + noise, 0.0, 1.0)


def entailment_batch(dtok, dmodel, pairs, batch=64):
    """Entailment probability for each (hypothesis, premise) pair."""
    probs = []
    for i in range(0, len(pairs), batch):
        hyps = [p[0] for p in pairs[i:i + batch]]
        prems = [p[1] for p in pairs[i:i + batch]]
        enc = dtok(hyps, prems, padding=True, truncation=True,
                   max_length=256, return_tensors="pt")
        enc = {k: v.to(DEV) for k, v in enc.items()}
        enc["position_ids"] = torch.arange(enc["input_ids"].shape[1], device=DEV).unsqueeze(0).expand(enc["input_ids"].shape[0], -1)
        with torch.no_grad():
            logits = dmodel(**enc).logits
        probs.append(torch.softmax(logits.float(), dim=-1)[:, 2])
    return torch.cat(probs)


def get_semantic_ids(dtok, dmodel, strings):
    """BATCHED_CLUSTER: identical union logic, all pairs scored in one batched pass."""
    n = len(strings)
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    P = Q = None
    if pairs:
        pr = entailment_batch(
            dtok, dmodel,
            [(strings[a], strings[b]) for a, b in pairs] +
            [(strings[b], strings[a]) for a, b in pairs])
        P, Q = pr[:len(pairs)], pr[len(pairs):]
    ids = list(range(n))
    for k, (a, b) in enumerate(pairs):
        if ids[a] == ids[b]:
            continue
        if P[k].item() > 0.5 or Q[k].item() > 0.5:
            old = ids[b]
            for t in range(n):
                if ids[t] == old:
                    ids[t] = ids[a]
    mp, out = {}, []
    for x in ids:
        if x not in mp:
            mp[x] = len(mp)
        out.append(mp[x])
    return out


def get_sequence_log_logits(token_ids, scores, special_ids):
    """Per-token log probabilities."""
    logits = torch.stack(scores, dim=1)          # [B, seq_len, vocab]
    log_probs = torch.log_softmax(torch.nan_to_num(logits.float(), nan=0.0), dim=-1)
    tok = token_ids.unsqueeze(-1)
    sel = torch.gather(log_probs, 2, tok).squeeze(-1)   # [B, seq_len]
    mask = ~torch.isin(token_ids.cpu(), special_ids.cpu())
    return [sel[b][mask[b]].cpu() for b in range(token_ids.shape[0])]


def sentence_se(log_logits, semantic_ids):
    """Sentence-level semantic entropy over semantic clusters."""
    n_clusters = max(semantic_ids) + 1
    logits_by_cluster = [[] for _ in range(n_clusters)]
    for ll, sid in zip(log_logits, semantic_ids):
        logits_by_cluster[sid].append(ll)
    cluster_log_prob = []
    for ll_list in logits_by_cluster:
        if not ll_list:
            cluster_log_prob.append(0.0)
            continue
        min_len = min(l.shape[0] for l in ll_list)
        if min_len == 0:
            cluster_log_prob.append(0.0)
            continue
        avg = torch.stack([l[:min_len] for l in ll_list]).mean(0).sum().item()
        cluster_log_prob.append(avg)
    log_p = torch.tensor(cluster_log_prob)
    log_p = torch.nan_to_num(log_p, nan=0.0, posinf=0.0, neginf=0.0)
    log_p = log_p - log_p.max()
    p = torch.softmax(log_p, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)
    ent = -torch.sum(p * torch.log(p + 1e-10)).item()
    return ent, p


def build_rows(ds_name):
    """Rows for SLAKE / POPE (bench2) or OmniMed (common), matched protocol, test rows only."""
    if ds_name == "omni":
        import sys as _s; _s.path.insert(0, "/root/nature_exp/exp")
        import common as C
        samples = C.load_raw("omni")
        tr, va, te, rows, audit = C.get_level("omni", "L4_fair", samples)
    else:
        import bench2 as B2
        import vlm2 as V2
        samples = B2.load_slake() if ds_name == "slake" else B2.load_pope(split="random")
        tr, va, te, rows, audit = V2.get_level(ds_name, "L4_fair", samples)
    out = []
    for i in te:
        r = rows[int(i)]
        s = samples[r["pid"]]
        out.append({"image": s["image"], "question": r["question"], "label": int(r["label"])})
    print("[build_rows] %s rows=%d (pos=%d)" % (ds_name, len(out), sum(o["label"] for o in out)), flush=True)
    return out


def main(ds_name, max_n=20):
    tokenizer, model, image_processor, dtok, dmodel = load_all()
    print("ALL MODELS LOADED (VASE-full)", flush=True)
    if ds_name in ("slake", "pope_random", "omni"):
        test = build_rows(ds_name)
    else:
        train, val, test = prepare_dataset(ds_name, max_n=(4000 if ds_name == "path_vqa" else None))
        random.seed(42)
        test = random.sample(list(test), min(max_n, len(test)))

    cache_path = os.path.join(OUT, "vasefull_cache_%s.npz" % ds_name)
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        scores_v = list(d["v"]); scores_se = list(d["se"])
        scores_rad = list(d["rad"]); labels = list(d["lab"])
        done_n = len(scores_v)
        print("RESUME from {}: {} done".format(cache_path, done_n), flush=True)
    else:
        scores_v, scores_se, scores_rad, labels = [], [], [], []
        done_n = 0
    special_ids = torch.tensor(tokenizer.all_special_ids, device="cpu")
    for i, p in enumerate(test):
        if i < done_n:
            continue
        try:
            img = p["image"]
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            q = p["question"]
            input_ids, img_t, attn = make_inputs(tokenizer, image_processor, img, q)

            # main answer (temp 0.1) + sampled answers (temp 1.0)
            ans_main, seqs_main, scores_main = generate_batch(model, tokenizer, input_ids, img_t, attn, 1, 0.1)
            ans_sam, seqs_sam, scores_sam = generate_batch(model, tokenizer, input_ids, img_t, attn, NUM_SAMPLES, 1.0)
            # noisy-image sampling
            img_noisy = add_noise(img_t.clone())
            ans_noisy, seqs_noisy, scores_noisy = generate_batch(model, tokenizer, input_ids, img_noisy, attn, NUM_SAMPLES, 1.0)

            # semantic clustering over (main + clean samples + noisy samples)
            all_ans = [ans_main[0]] + list(ans_sam) + list(ans_noisy)
            cids = get_semantic_ids(dtok, dmodel, all_ans)
            cids_sam = cids[1:1 + NUM_SAMPLES]
            cids_noisy = cids[1 + NUM_SAMPLES:]

            # RadFlag
            rad = cids_sam.count(cids_sam[0]) / NUM_SAMPLES if NUM_SAMPLES > 0 else 0.0

            # SeEnt (token level)
            ll = get_sequence_log_logits(seqs_sam, scores_sam, special_ids)
            se_ent, se_dist = sentence_se(ll, cids_sam)

            # ViSeEnt
            ll_noisy = get_sequence_log_logits(seqs_noisy, scores_noisy, special_ids)
            _, se_dist_noisy = sentence_se(ll_noisy, cids_noisy)
            n_c = max(cids) + 1
            align = torch.zeros(n_c)
            align_noisy = torch.zeros(n_c)
            for sid, dval in zip(cids_sam, se_dist):
                align[sid] = dval
            for sid, dval in zip(cids_noisy, se_dist_noisy):
                align_noisy[sid] = dval
            visedist = align + ALPHA * (align - align_noisy)
            visedist = torch.softmax(visedist, dim=-1)
            v_ent = -torch.sum(visedist * torch.log(visedist + 1e-10)).item()
        except Exception as e:
            print("  [sample {}] ERROR: {} (skipped)".format(i, str(e)[:120]), flush=True)
            torch.cuda.empty_cache()
            scores_v.append(float("nan")); scores_se.append(float("nan"))
            scores_rad.append(float("nan")); labels.append(int(p["label"]))
            if (i + 1) % 25 == 0:
                np.savez(cache_path, v=np.array(scores_v), se=np.array(scores_se),
                         rad=np.array(scores_rad), lab=np.array(labels))
            continue

        scores_v.append(v_ent); scores_se.append(se_ent)
        scores_rad.append(rad); labels.append(int(p["label"]))
        if (i + 1) % 5 == 0:
            print("  {}/{} done, ViSE={:.3f} SE={:.3f} Rad={:.2f}".format(
                i + 1, len(test), v_ent, se_ent, rad), flush=True)
        if (i + 1) % 25 == 0:
            np.savez(cache_path, v=np.array(scores_v), se=np.array(scores_se),
                     rad=np.array(scores_rad), lab=np.array(labels))
            print("  [checkpoint {}]".format(i + 1), flush=True)
    np.savez(cache_path, v=np.array(scores_v), se=np.array(scores_se),
             rad=np.array(scores_rad), lab=np.array(labels))

    from sklearn.metrics import roc_auc_score
    sv = np.array(scores_v); ss = np.array(scores_se); sr = np.array(scores_rad)
    lab = np.array(labels)
    m = ~np.isnan(sv)
    sv, ss, sr, lab = sv[m], ss[m], sr[m], lab[m]
    for name, s in [("VASE", sv), ("SE", ss), ("RadFlag", sr)]:
        a1 = roc_auc_score(lab, s)
        a2 = roc_auc_score(lab, [-x for x in s])
        print("=== {} {} AUROC(high=halluc)={:.4f} AUROC(low=halluc)={:.4f} (n={}) ===".format(
            name, ds_name, a1, a2, len(lab)), flush=True)
    out = {"method": "VASE-FULL-LLaVA-Med", "dataset": ds_name,
           "vase": float(roc_auc_score(labels, scores_v)),
           "se": float(roc_auc_score(labels, scores_se)),
           "radflag": float(roc_auc_score(labels, scores_rad)),
           "num_samples": NUM_SAMPLES, "n": len(test)}
    with open(os.path.join(OUT, "results_vasefull_%s.json" % ds_name), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "vqa_rad"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    main(ds, n)
