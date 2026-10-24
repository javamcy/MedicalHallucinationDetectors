#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E6: cross-family detector sweep across the protocol gradient, using frozen
instruction-tuned VLMs that were never trained on the protocol.

Sub-commands (run in order):

  gen     For every raw (image, question) sample: 1 greedy + K sampled answers.
  score   For every unique (image, question, answer) triple needed by the chosen
          levels: the frozen zero-shot "is it grounded?" judge score and the
          teacher-forced answer NLL (internal-confidence family).
  eval    Build sampling-statistics detectors (semantic entropy, RadFlag, answer
          consistency) from the generations and produce a per-level AUROC table.

All forwards are batched (left padding) so the sweep is tractable on one GPU.

Usage:
  python e6_family.py gen   --ds vqa_rad --tag q3b [--k 5] [--bs 24]
  python e6_family.py score --ds vqa_rad --tag q3b --levels L0_leaky,L1_imgsplit,L4_fair
  python e6_family.py eval  --ds vqa_rad --tag q3b --levels L0_leaky,L1_imgsplit,L4_fair
"""
import argparse
import json
import numpy as np

import common as C

DEFAULT_MODELS = {
    "q3b": "/root/scale_study/models/Qwen2.5-VL-3B-Instruct",
    "q7b": "/cloud/scale_models/Qwen2.5-VL-7B-Instruct",
    "q32b": "/cloud/scale_models/Qwen2.5-VL-32B-Instruct",
    "l7b": "/cloud/scale_models/Lingshu-7B",
}

PROMPT_JUDGE = "Is this answer grounded in the image?"
POS_TOKEN, NEG_TOKEN = "grounded", "ungrounded"


class Adapter:
    def __init__(self, model_path, device="cuda"):
        import torch
        from transformers import AutoProcessor
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        except Exception:
            from transformers import Qwen2VLForConditionalGeneration as Cls
        import os as _os
        if _os.environ.get("QUANT4BIT") == "1":
            from transformers import BitsAndBytesConfig
            _bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_use_double_quant=True)
            self.model = Cls.from_pretrained(model_path, quantization_config=_bnb,
                                             device_map={"": 0}).eval()
            print("[Adapter] 4-bit nf4", flush=True)
        else:
            self.model = Cls.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device).eval()

    # ------------------------------------------------------------------
    def label_ids(self):
        for a, b in ((" " + POS_TOKEN, " " + NEG_TOKEN), (POS_TOKEN, NEG_TOKEN)):
            p = self.tokenizer.encode(a, add_special_tokens=False)
            n = self.tokenizer.encode(b, add_special_tokens=False)
            if len(p) == 1 and len(n) == 1 and p[0] != n[0]:
                return p[0], n[0]
        p = self.tokenizer.encode(POS_TOKEN, add_special_tokens=False)
        n = self.tokenizer.encode(NEG_TOKEN, add_special_tokens=False)
        return p[0], n[0]

    _imghash = {}

    @staticmethod
    def _fit(img, max_side=336):
        """Cap the visual token budget: medical VQA images need no more than this,
        and it keeps a 24 GB device comfortable at batch size 8."""
        try:
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_side:
                s = max_side / float(max(w, h))
                img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
        except Exception:
            pass
        return img

    def batch_inputs(self, imgs, texts):
        """Left-padded batch of chat inputs."""
        from qwen_vl_utils import process_vision_info
        imgs = [self._fit(im) for im in imgs]
        # (resizing is idempotent and cheap; the expensive part is the vision tower)
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": tx}]}]
                for im, tx in zip(imgs, texts)]
        texts_t = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                   for m in msgs]
        ii, vi = process_vision_info(msgs)
        inp = self.processor(text=texts_t, images=ii, videos=vi, return_tensors="pt", padding=True)
        return {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}

    # ------------------------------------------------------------------
    def generate_batch(self, imgs, texts, do_sample, max_new_tokens=20,
                       temperature=1.0, top_p=0.9):
        torch = self.torch
        inp = self.batch_inputs(imgs, texts)
        L = inp["input_ids"].shape[1]
        gkw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, num_beams=1, use_cache=True)
        if do_sample:
            gkw.update(temperature=temperature, top_p=top_p)
        try:
            with torch.no_grad():
                o = self.model.generate(**inp, **gkw)
        except torch.OutOfMemoryError:
            del inp
            torch.cuda.empty_cache()
            outs = []
            for k in range(len(texts)):
                sub = self.batch_inputs([imgs[k]], [texts[k]])
                Ls = sub["input_ids"].shape[1]
                with torch.no_grad():
                    oo = self.model.generate(**sub, **gkw)
                outs.append(self.tokenizer.decode(oo[0][Ls:], skip_special_tokens=True).strip())
                del sub, oo
                torch.cuda.empty_cache()
            return outs
        outs = []
        for r in range(o.shape[0]):
            outs.append(self.tokenizer.decode(o[r][L:], skip_special_tokens=True).strip())
        del inp, o
        torch.cuda.empty_cache()
        return outs

    # ------------------------------------------------------------------
    def judge_batch(self, imgs, qs, ans, pos_id, neg_id):
        torch = self.torch
        texts = ["Question: %s\nAnswer: %s\n%s" % (q, a, PROMPT_JUDGE) for q, a in zip(qs, ans)]
        inp = self.batch_inputs(imgs, texts)
        try:
            with torch.no_grad():
                out = self.model(**inp)
        except torch.OutOfMemoryError:
            del inp
            torch.cuda.empty_cache()
            d = []
            for k in range(len(texts)):
                si = self.batch_inputs([imgs[k]], [texts[k]])
                with torch.no_grad():
                    oo = self.model(**si)
                lt = oo.logits[:, -1, :].float()
                d.append(float(lt[0, pos_id] - lt[0, neg_id]))
                del si, oo, lt
                torch.cuda.empty_cache()
            return np.nan_to_num(np.asarray(d), nan=0.0)
        last = out.logits[:, -1, :].float()
        d = (last[:, pos_id] - last[:, neg_id]).cpu().numpy()
        d = np.nan_to_num(d, nan=0.0)
        del out, inp
        torch.cuda.empty_cache()
        return d

    def nll_batch(self, imgs, qs, ans):
        """Teacher-forced mean NLL of the answer tokens, appended after
        'Question: q\\nAnswer:'.  Left padding keeps every row's real tokens
        right-aligned, so the appended answer span is contiguous."""
        torch = self.torch
        texts = ["Question: %s\nAnswer:" % q for q in qs]
        inp = self.batch_inputs(imgs, texts)
        ids, att = inp["input_ids"], inp["attention_mask"]
        B, L = ids.shape
        toks = [self.tokenizer.encode(" " + str(a), add_special_tokens=False) for a in ans]
        M = max(1, max(len(t) for t in toks))
        apad = torch.zeros((B, M), dtype=ids.dtype, device=self.device)
        amask = torch.zeros((B, M), dtype=torch.long, device=self.device)
        for i, t in enumerate(toks):
            if t:
                apad[i, :len(t)] = torch.tensor(t, dtype=ids.dtype, device=self.device)
                amask[i, :len(t)] = 1
        full = torch.cat([ids, apad], 1)
        fatt = torch.cat([att, amask], 1)
        kw = {k: v for k, v in inp.items() if k not in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = self.model(input_ids=full, attention_mask=fatt, **kw)
        lg = out.logits[:, L - 1:L - 1 + M, :].float()
        lp = torch.log_softmax(lg, -1)
        nll = -(lp.gather(2, apad.unsqueeze(-1)).squeeze(-1) * amask.float()).sum(1) / amask.sum(1).clamp(min=1)
        res = nll.cpu().numpy()
        del out, lg, lp, full, fatt
        torch.cuda.empty_cache()
        return res


# ---------------------------------------------------------------------------
def _batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def cmd_gen(args):
    samples = C.load_raw(args.ds)
    if args.max_n > 0:
        samples = samples[:args.max_n]
    path = C.CACHE / ("gen_%s_%s.jsonl" % (args.ds, args.tag))
    done = set()
    if path.exists():
        for line in open(path):
            try:
                done.add(json.loads(line)["image_id"])
            except Exception:
                pass
    todo = [s for s in samples if s["image_id"] not in done]
    print("[E6-gen] %s/%s n=%d done=%d todo=%d k=%d bs=%d"
          % (args.ds, args.tag, len(samples), len(done), len(todo), args.k, args.bs), flush=True)
    if not todo:
        print("[E6-gen] nothing to do", flush=True)
        return
    ad = Adapter(DEFAULT_MODELS.get(args.tag, args.tag))
    fout = open(path, "a")
    store = {}
    for rep in range(args.k + 1):
        nb = 0
        for chunk in _batches(todo, args.bs):
            outs = ad.generate_batch([s["image"] for s in chunk], [s["question"] for s in chunk],
                                     do_sample=(rep > 0), max_new_tokens=args.max_new_tokens)
            for s, o in zip(chunk, outs):
                store.setdefault(s["image_id"], dict(
                    image_id=s["image_id"], gid=s["gid"], question=s["question"],
                    answer=s["answer"], samples=[]))["samples"].append(o)
            nb += 1
            if nb % 10 == 0:
                print("  [E6-gen] rep %d/%d  %d/%d" % (rep, args.k, nb * args.bs, len(todo)), flush=True)
        print("  [E6-gen] rep %d/%d complete" % (rep, args.k), flush=True)
        # flush after every repetition so partial work survives
        fout.truncate(0)
        fout.seek(0)
        for s in todo:
            if s["image_id"] in store:
                fout.write(json.dumps(store[s["image_id"]], ensure_ascii=False) + "\n")
        fout.flush()
    fout.close()
    print("[E6-gen] %s/%s DONE" % (args.ds, args.tag), flush=True)


def cmd_score(args):
    samples = C.load_raw(args.ds)
    ad = Adapter(DEFAULT_MODELS.get(args.tag, args.tag))
    pos_id, neg_id = ad.label_ids()
    for lv in args.levels.split(","):
        tr, va, te, rows, audit = C.get_level(args.ds, lv, samples)
        outp = C.CACHE / ("llm_%s_%s_%s.npz" % (args.ds, args.tag, lv))
        keys, uniq = set(), []
        te_set = set(int(i) for i in te)
        for i, r in enumerate(rows):
            if args.test_only and i not in te_set:
                continue
            k = "%d|%s" % (r["pid"], r["answer"])
            if k in keys:
                continue
            keys.add(k)
            uniq.append((k, r))
        cache = {}
        if outp.exists():
            d = np.load(str(outp), allow_pickle=True)
            for k, j, n in zip(d["keys"], d["judge"], d["nll"]):
                cache[str(k)] = (float(j), float(n))
        todo = [(k, r) for k, r in uniq if k not in cache]
        print("[E6-score] %s %s %s: uniq=%d todo=%d" % (args.ds, args.tag, lv, len(uniq), len(todo)), flush=True)
        for chunk in _batches(todo, args.bs):
            imgs = [samples[r["pid"]]["image"] for _, r in chunk]
            qs = [r["question"] for _, r in chunk]
            ans = [r["answer"] for _, r in chunk]
            j = ad.judge_batch(imgs, qs, ans, pos_id, neg_id)
            n = ad.nll_batch(imgs, qs, ans)
            for (k, _), jj, nn in zip(chunk, j, n):
                cache[k] = (float(jj), float(nn))
            if len(cache) % 400 < args.bs:
                print("  [E6-score] %d/%d" % (len(cache), len(uniq)), flush=True)
        J = np.array([cache.get("%d|%s" % (r["pid"], r["answer"]), (np.nan, np.nan))[0] for r in rows], float)
        N = np.array([cache.get("%d|%s" % (r["pid"], r["answer"]), (np.nan, np.nan))[1] for r in rows], float)
        print("[E6-score] %s %s test-covered rows: %d/%d" % (args.ds, lv,
              int(np.isfinite(J).sum()), len(rows)), flush=True)
        y = np.array([r["label"] for r in rows], np.int32)
        np.savez_compressed(str(outp), judge=J, nll=N, y=y, idx_te=np.asarray(te),
                            keys=np.asarray([r["pid"] for r in rows]))
        print("[E6-score] %s %s saved (%d rows, nan=%d)" % (args.ds, lv, len(rows), int(np.isnan(J).sum())), flush=True)


# ---------------------------------------------------------------------------
def _ans_norm(s):
    return " ".join(str(s).lower().replace(".", " ").replace(",", " ").split())


def cluster_dist(embs, thr=0.90):
    cs, sizes = [], []
    for i, e in enumerate(embs):
        best, bsim = -1, -1.0
        for j, c in enumerate(cs):
            s = float(np.dot(e, embs[c[0]]))
            if s > bsim:
                bsim, best = s, j
        if best >= 0 and bsim >= thr:
            cs[best].append(i)
        else:
            cs.append([i])
    return [len(c) for c in cs]


def cmd_eval(args):
    samples = C.load_raw(args.ds)
    gpath = C.CACHE / ("gen_%s_%s.jsonl" % (args.ds, args.tag))
    have_gen = gpath.exists()
    if not have_gen:
        print("[E6-eval] no generations for %s/%s -- evaluating the internal-signal family only"
              % (args.ds, args.tag), flush=True)
    G = {}
    if have_gen:
        for line in open(gpath):
            r = json.loads(line)
            G[r["image_id"]] = r
    clip = C.ClipFeat() if have_gen else None
    items = [r for r in G.values() if r.get("samples")] if have_gen else []
    uniq = sorted({_ans_norm(g) for r in items for g in r["samples"] if str(g).strip()}) if items else []
    print("[E6-eval] %s/%s: %d samples, %d unique generations" % (args.ds, args.tag, len(items), len(uniq)), flush=True)
    embs = np.stack(clip.text_feat_batch(uniq)) if uniq else np.zeros((0, 768), np.float32)
    tmap = {t: i for i, t in enumerate(uniq)}
    per_sample = {}
    for r in items:
        keys = [_ans_norm(g) for g in r["samples"] if _ans_norm(g) in tmap]
        if not keys:
            continue
        E = embs[[tmap[k] for k in keys]]
        sizes = cluster_dist(E)
        p = np.array(sizes, float)
        p = p / p.sum()
        se = float(-(p * np.log(p + 1e-12)).sum())
        radflag = float(p.max())
        g0 = E[0]
        cons_max = float(np.max(E[1:] @ g0)) if len(E) > 1 else 1.0
        cons_mean = float(np.mean(E[1:] @ g0)) if len(E) > 1 else 1.0
        per_sample[r["image_id"]] = dict(se=se, radflag=radflag, cons_max=cons_max,
                                         cons_mean=cons_mean, embs=E, samples=r["samples"])

    res = {}
    for lv in args.levels.split(","):
        tr, va, te, rows, audit = C.get_level(args.ds, lv, samples)
        y = np.array([r["label"] for r in rows], np.int32)
        yte = y[te]
        e = dict(level=lv, n_test=int(len(te)), pos_test=int(yte.sum()))

        def emit(name, s):
            st = np.asarray(s, float)[te]
            if np.isnan(st).any():
                st = np.nan_to_num(st, nan=float(np.nanmedian(st)))
            e[name] = dict(auroc=float(C.best_orientation(yte, st)[1]),
                           raw_auroc=float(C.auroc(yte, st)))

        def ps(r):
            return per_sample.get(samples[r["pid"]]["image_id"])

        if have_gen:
            emit("semantic_entropy", [ps(r)["se"] if ps(r) else np.nan for r in rows])
            emit("radflag", [ps(r)["radflag"] if ps(r) else np.nan for r in rows])
            clip.text_feat_batch(sorted({_ans_norm(r["answer"]) for r in rows}))

        def cons(r, mode):
            x = ps(r)
            if not x:
                return np.nan
            ca = clip.text_feat(_ans_norm(r["answer"]))
            v = (x["embs"][1:] @ ca) if len(x["embs"]) > 1 else np.array([1.0])
            return float(v.max() if mode == "max" else v.mean())

        if have_gen:
            emit("answer_consistency_max", [cons(r, "max") for r in rows])
            emit("answer_consistency_mean", [cons(r, "mean") for r in rows])

        def greedy_agree(r):
            """Verification family: agreement between the candidate and the model's own
            greedily elicited answer to the same question. High agreement -> likely grounded."""
            x = ps(r)
            if not x:
                return np.nan
            ca = clip.text_feat(_ans_norm(r["answer"]))
            return float(x["embs"][0] @ ca)

        if have_gen:
            emit("greedy_agreement", [greedy_agree(r) for r in rows])

        def tok_overlap(r):
            x = ps(r)
            if not x:
                return np.nan
            ct = set(_ans_norm(r["answer"]).split())
            st = set(" ".join(_ans_norm(z) for z in x["samples"]).split())
            return -len(ct & st) / (len(ct) + 1e-6)

        if have_gen:
            emit("consistency_token_overlap", [tok_overlap(r) for r in rows])

        lp = C.CACHE / ("llm_%s_%s_%s.npz" % (args.ds, args.tag, lv))
        if lp.exists():
            d = np.load(str(lp))
            J, N = d["judge"], d["nll"]
            if len(J) == len(rows):
                e["zero_shot_judge"] = dict(auroc=float(C.best_orientation(yte, J[te])[1]),
                                            raw_auroc=float(C.auroc(yte, J[te])))
                e["answer_nll"] = dict(auroc=float(C.best_orientation(yte, N[te])[1]),
                                       raw_auroc=float(C.auroc(yte, N[te])))
        res[lv] = e
        print("[E6-eval] %s %s %s: judge=%s nll=%s SE=%s RadFlag=%s verify=%s consMax=%s"
              % (args.ds, args.tag, lv,
                 round(e.get("zero_shot_judge", {}).get("auroc", float("nan")), 3),
                 round(e.get("answer_nll", {}).get("auroc", float("nan")), 3),
                 round(e.get("semantic_entropy", {}).get("auroc", float("nan")), 3),
                 round(e.get("radflag", {}).get("auroc", float("nan")), 3),
                 round(e.get("greedy_agreement", {}).get("auroc", float("nan")), 3),
                 round(e.get("answer_consistency_max", {}).get("auroc", float("nan")), 3)),
              flush=True)
        C.save_json(res, C.OUT / ("e6_family_%s_%s.json" % (args.ds, args.tag)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "score", "eval"])
    ap.add_argument("--ds", default="vqa_rad")
    ap.add_argument("--tag", default="q3b")
    ap.add_argument("--levels", default="L0_leaky,L1_imgsplit,L4_fair")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    ap.add_argument("--max-n", type=int, default=-1)
    ap.add_argument("--test-only", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "gen":
        cmd_gen(args)
    elif args.mode == "score":
        cmd_score(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
