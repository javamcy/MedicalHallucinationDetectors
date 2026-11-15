#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E4: residual-leakage audit of the fair protocol ("is the fair protocol really fair?").

For every (dataset, level) we fit classifiers that see ONLY text, never the image:
  * answer length / type / frequency features (same as the shortcut oracle)
  * TF-IDF of the answer
  * TF-IDF of the question
  * TF-IDF of question+answer
  * character n-gram TF-IDF of the answer (format / template capture)
and report test AUROC.  If the matched-negative construction really removes the
shortcut, all of these must drop to ~0.5 at L4.

We also audit the construction itself: how many negatives were actually
length/type matched, and what the residual length gap is.

Usage: python e4_residual.py [dataset ...]
"""
import sys
import numpy as np

import common as C


def tfidf_auroc(train_texts, train_y, test_texts, test_y, analyzer="word", ngram=(1, 2), mindf=2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    try:
        pipe = make_pipeline(
            TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, min_df=mindf, sublinear_tf=True,
                            lowercase=True, max_features=200000),
            LogisticRegression(max_iter=2000, C=1.0))
        pipe.fit(train_texts, train_y)
        s = pipe.predict_proba(test_texts)[:, 1]
        return float(C.auroc(test_y, s))
    except Exception as e:
        print("   [tfidf %s/%s] failed: %s" % (analyzer, ngram, e), flush=True)
        return float("nan")


def main():
    dss = sys.argv[1:] or ["vqa_rad", "path_vqa", "omni"]
    for ds in dss:
        samples = C.load_raw(ds)
        ans_freq, q_freq = C.pool_freqs(samples)
        res = {}
        levels = C.LEVEL_ORDER + (["L6_native"] if ds == "omni" else [])
        for lv in levels:
            tr, va, te, rows, audit = C.get_level(ds, lv, samples)
            y = np.array([r["label"] for r in rows], np.int32)
            ans = [str(r["answer"]) for r in rows]
            qs = [str(r["question"]) for r in rows]
            qa = [q + " || " + a for q, a in zip(qs, ans)]
            e = dict(level=lv, audit=audit,
                     n_test=int(len(te)), pos_test=int(y[te].sum()))
            # residual length gap between positives and negatives
            pl = np.array([len(str(r["answer"]).split()) for r in rows if r["label"] == 0], float)
            nl = np.array([len(str(r["answer"]).split()) for r in rows if r["label"] == 1], float)
            e["len_mean_pos"] = float(pl.mean())
            e["len_mean_neg"] = float(nl.mean())
            e["len_abs_gap"] = float(abs(pl.mean() - nl.mean()))
            e["frac_neg_within2"] = float(np.mean([abs(int(r["neg_len"]) - len(str(r["answer"]).split())) <= 2
                                                   for r in rows if r["label"] == 0]))
            # text-only classifiers
            e["tfidf_answer_word"] = tfidf_auroc([ans[i] for i in tr], y[tr], [ans[i] for i in te], y[te])
            e["tfidf_answer_char"] = tfidf_auroc([ans[i] for i in tr], y[tr], [ans[i] for i in te], y[te],
                                                 analyzer="char_wb", ngram=(2, 5))
            e["tfidf_question_word"] = tfidf_auroc([qs[i] for i in tr], y[tr], [qs[i] for i in te], y[te])
            e["tfidf_qa_word"] = tfidf_auroc([qa[i] for i in tr], y[tr], [qa[i] for i in te], y[te])
            # lexical feature classifier (identical to the shortcut oracle)
            Xl = np.asarray([C.leak_features(r, ans_freq) for r in rows], np.float32)
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xl[tr])
            lr = LogisticRegression(max_iter=3000).fit(sc.transform(Xl[tr]), y[tr])
            e["lexical_only"] = float(C.auroc(y[te], lr.predict_proba(sc.transform(Xl[te]))[:, 1]))
            # answer-identity frequency: a memorised answer->label map
            from collections import Counter
            cnt = Counter()
            for i in tr:
                cnt[ans[i]] += 1 if y[i] == 1 else -1
            s = np.array([cnt.get(ans[i], 0) for i in range(len(rows))], float)
            e["answer_frequency_leak"] = float(C.best_orientation(y[te], s[te])[1])
            res[lv] = e
            print("[E4] %s %s | leb=%.4f tfidfA=%.4f tfidfQA=%.4f lex=%.4f gap=%.2f"
                  % (ds, lv, e["tfidf_answer_char"], e["tfidf_answer_word"], e["tfidf_qa_word"],
                     e["lexical_only"], e["len_abs_gap"]), flush=True)
        C.save_json(res, C.OUT / ("e4_residual_%s.json" % ds))
    print("[E4] done", flush=True)


if __name__ == "__main__":
    main()
