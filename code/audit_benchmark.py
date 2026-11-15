"""Stage 1 of the prospective audit: predict a benchmark's protocol sensitivity
without training any vision--language model.

This script deliberately does not touch CLIP or any VLM. It builds the paired rows
at L0 and L4, fits the twenty-feature lexical oracle at each rung, and applies the
regression frozen on the eleven benchmarks in the released results:

    predicted_detector_drop = 0.039 + 0.365 * (oracle_L0 - oracle_L4)

The output JSON is written *before* anyone runs the ladder. That ordering is the
entire point: a prediction recorded after the fact proves nothing.

Usage
-----
    python audit_benchmark.py --ds vizwiz --out prospective/vizwiz_prediction.json

The benchmark must already be registered in `common.LOADERS`, either natively or
through `xdomain.py`.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

# regression frozen on the eleven released benchmark results
REG_A = 0.039
REG_B = 0.365
REG_R = 0.595
REG_LOO_MAE = 0.029
REG_LOO_MAX = 0.064
REG_N = 11

LEX_NAMES = ["ans_tokens", "ans_chars", "q_tokens", "ans/q_tok", "ans/q_char",
             "is_yesno", "qt_is_yesno", "qt_is_wh", "qt_other", "type_match",
             "has_digit", "has_pct", "has_comma", "ends_period", "cap_first",
             "negation_word", "log_ans_freq", "log_qt_freq", "qa_overlap_n",
             "qa_jaccard"]


def oracle_auroc(samples, level, seed=0):
    """AUROC of the text-only oracle at one rung, on the held-out split."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import common as C

    tr, va, te, rows, audit = C.get_level(CURRENT_DS, level, samples)
    y = np.array([r["label"] for r in rows], np.int32)
    ans_freq, q_freq = C.pool_freqs(samples)
    X = np.asarray([C.leak_features(r, ans_freq) for r in rows], np.float32)

    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(X[tr]), y[tr])
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                        random_state=seed).fit(X[tr], y[tr])
    s_lr = lr.predict_proba(sc.transform(X))[:, 1]
    s_gb = gb.predict_proba(X)[:, 1]
    yte = y[te]

    def best(s):
        a = float(C.auroc(yte, s[te]))
        return max(a, 1.0 - a)

    return dict(logreg=best(s_lr), gbm=best(s_gb), n_rows=len(rows),
                n_test=len(te), pos_test=int(yte.sum()), audit=audit,
                auroc=best(s_gb))


CURRENT_DS = None


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global CURRENT_DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--levels", default="L0_leaky,L4_fair")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.environ.get("BENCH_BASE", "/root/bench_exp") + "/exp")
    import common as C
    try:
        import xdomain  # noqa: F401
    except Exception:
        pass

    CURRENT_DS = a.ds
    samples = C.load_raw(a.ds)
    l0, l4 = a.levels.split(",")

    r0 = oracle_auroc(samples, l0)
    r4 = oracle_auroc(samples, l4)
    oracle_drop = r0["auroc"] - r4["auroc"]
    predicted = REG_A + REG_B * oracle_drop

    rec = dict(
        benchmark=a.ds,
        stage="1-prediction-only",
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_items=len(samples),
        oracle_L0=r0, oracle_L4=r4,
        oracle_drop=oracle_drop,
        regression=dict(intercept=REG_A, slope=REG_B, r=REG_R,
                        loo_mae=REG_LOO_MAE, loo_max=REG_LOO_MAX, n_fit=REG_N),
        predicted_detector_drop=predicted,
        prediction_interval=dict(lo=predicted - REG_LOO_MAX,
                                 hi=predicted + REG_LOO_MAX),
        caveat=("This file must be committed before stage 2 is run. The audit reads "
                "only lexical features of the question and the candidate answer; no "
                "CLIP feature and no vision-language model is used here."),
        script_sha256=sha256_of(os.path.abspath(__file__)),
    )

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2)

    print("[audit] %s: oracle L0=%.3f L4=%.3f drop=%.3f -> predicted detector drop=%.3f "
          "(interval %.3f..%.3f)" % (a.ds, r0["auroc"], r4["auroc"], oracle_drop,
                                     predicted, predicted - REG_LOO_MAX,
                                     predicted + REG_LOO_MAX))
    print("[audit] written to %s -- commit this before running the ladder" % a.out)


if __name__ == "__main__":
    main()
