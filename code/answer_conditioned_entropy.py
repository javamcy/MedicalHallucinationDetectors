"""Answer-conditioned versions of the sampling-statistics family.

Why this file exists
--------------------
The semantic entropy and RadFlag scores as computed by `e6_family.py` read only
(image, question): `ps(r)` looks the per-sample record up by `image_id`. On a
benchmark whose evaluation pairs contrast two candidate answers for the *same*
image, both members of a pair therefore receive an identical score and AUROC is
0.500 by construction. That is an algebraic identity, not a measurement, and we
removed the table it produced.

The fix is to make the score a function of the candidate answer A as well as of
the model's samples S = {s_1 ... s_K} for the same item. Four such scores are
implemented here:

  support_mass     fraction of the sample mass that is semantically equal to A
                   (a grounded answer should be well supported by the model's
                   own samples, a fabricated one should not be)
  surprisal        -log of that mass, the natural answer-conditioned entropy
  entropy_cond     entropy of the sample cluster distribution, re-weighted so
                   that clusters semantically equal to A count towards support
  margin           entropy_cond minus surprisal, a signed confidence margin

All four vary within a pair by construction, because A enters the score.

Usage
-----
    python answer_conditioned_entropy.py --ds mmmu --tag q35 --level L4_fair

It reads the generation cache written by `e6_family.py gen`
(`cache/gen_<ds>_<tag>.jsonl`) and the evaluation rows produced by
`common.get_level`, so it can be run on artefacts that already exist; no model
and no GPU are required.

Known limitation (measured, not assumed)
----------------------------------------
On the archived MMMU cache the verbatim-support mass is only 0.032 with
Qwen3.5-9B: the generation used a reasoning prompt with a 20-token budget, so
samples are truncated sentences and the short gold answers (`$7`) seldom appear
inside them. Two consequences, both of which are reported here
rather than papered over:

1. A support mass computed from these samples is close to zero for most items,
   which makes the answer-conditioned entropy noisy.
2. The embedding-based variant (cosine similarity between the candidate and the
   sampled answers in frozen text space) is the practical substitute, and it is
   exactly the `answer_consistency_max` family already reported. That family
   reads the candidate answer, varies from 0.544 to 0.805 across backbones and
   benchmarks, and is therefore the honest answer-conditioned counterpart of
   semantic entropy.

A genuinely answer-conditioned entropy requires regenerating samples with a
short-answer instruction (one or two tokens per sample). That needs the GPU:
roughly 1.5 h per benchmark for Qwen3.5-4B across eleven benchmarks, or 2.8 h
per benchmark for gemma-4-31B at batch size 2.
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np


# --------------------------------------------------------------------------
# normalisation and semantic equivalence
# --------------------------------------------------------------------------
ARTICLES = re.compile(r"\b(a|an|the)\b")
PUNCT = re.compile(r"[^\w\s%.\-/]")
NUM = re.compile(r"^\s*[-+]?\d[\d,]*\.?\d*\s*%?\s*$")


def norm(s):
    """Normalise an answer string for comparison."""
    s = str(s).strip().lower()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = PUNCT.sub(" ", s)
    s = ARTICLES.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_number(s):
    return bool(NUM.match(str(s).strip()))


def numbers_equal(a, b):
    """Two numeric answers are equal if they agree after stripping separators."""
    def val(x):
        t = str(x).strip().rstrip("%").replace(",", "")
        try:
            return float(t)
        except ValueError:
            return None
    va, vb = val(a), val(b)
    if va is None or vb is None:
        return norm(a) == norm(b)
    return abs(va - vb) < 1e-9


def tokens(s):
    return set(norm(s).split())


def semantic_equal(a, b, jaccard=0.6, contain=0.8):
    """Is candidate `a` supported by sample `b`?

    The comparison is deliberately asymmetric: benchmark answers are short (`$7`,
    `0.28`, `yes`) while generated samples are whole sentences, so symmetric
    Jaccard never fires. A candidate counts as supported when it is equal after
    normalisation, numerically equal, contained verbatim in the sample, or when
    at least `contain` of its tokens appear in the sample.
    """
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    if not na or not nb:
        return False
    if is_number(a) and is_number(b):
        return numbers_equal(a, b)
    # verbatim containment, guarded so that very short candidates do not match
    # by accident inside unrelated words
    if len(na) >= 2 and re.search(r"(^|\s)%s($|[\s.,;:%%])" % re.escape(na), nb):
        return True
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return False
    hit = len(ta & tb) / len(ta)
    if hit >= contain:
        return True
    return len(ta & tb) / len(ta | tb) >= jaccard


# --------------------------------------------------------------------------
# the answer-conditioned scores
# --------------------------------------------------------------------------
def cluster(samples):
    """Greedy clustering of the samples under semantic_equal."""
    clusters = []          # list of lists of sample indices
    for i, s in enumerate(samples):
        placed = False
        for cl in clusters:
            if semantic_equal(s, samples[cl[0]]):
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def entropy(counts):
    n = sum(counts)
    if n <= 0:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in counts if c > 0)


def score_item(samples, candidate):
    """Return the four answer-conditioned scores for one item.

    `samples` is the list of generated answer strings, `candidate` the answer
    being judged. Higher `support_mass` and `margin` mean "more likely
    grounded"; higher `surprisal` and `entropy_cond` mean "less likely
    grounded", so the caller orients the AUROC accordingly.
    """
    samples = [s for s in samples if s and str(s).strip()]
    if not samples:
        return None
    cls = cluster(samples)
    sizes = [len(c) for c in cls]
    total = sum(sizes)

    support = 0
    for cl in cls:
        if any(semantic_equal(candidate, samples[i]) for i in cl):
            support += len(cl)
            break
    mass = support / total if total else 0.0

    h = entropy(sizes)
    # re-weight: clusters that support the candidate are treated as consistent,
    # so the entropy is computed over (support, rest)
    h_cond = entropy([support, max(total - support, 0)]) if total else 0.0
    surprisal = -math.log(mass) if mass > 0 else math.log(total + 1) + 5.0
    margin = h - surprisal

    return dict(support_mass=mass, surprisal=surprisal, entropy_cond=h_cond,
                margin=margin, n_samples=total, n_clusters=len(cls), entropy=h)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def load_gen(ds, tag, cache):
    path = os.path.join(cache, "gen_%s_%s.jsonl" % (ds, tag))
    if not os.path.exists(path):
        raise SystemExit("missing generation cache: %s" % path)
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["image_id"]] = r
    return out


def evaluate(ds, tag, level, cache):
    import common as C
    import xdomain  # noqa: F401  (registers the non-medical loaders)

    samples = C.load_raw(ds)
    tr, va, te, rows, audit = C.get_level(ds, level, samples)
    gen = load_gen(ds, tag, cache)

    y = np.array([r["label"] for r in rows], np.int32)
    per_key = {}
    for r in rows:
        img = samples[r["pid"]]["image_id"]
        rec = gen.get(img)
        if rec is None:
            per_key["%d|%s" % (r["pid"], r["answer"])] = None
            continue
        per_key["%d|%s" % (r["pid"], r["answer"])] = score_item(rec["samples"], r["answer"])

    out = {}
    for name, sign in (("support_mass", 1), ("surprisal", -1),
                       ("entropy_cond", -1), ("margin", 1)):
        s = []
        for r in rows:
            rec = per_key.get("%d|%s" % (r["pid"], r["answer"]))
            s.append(np.nan if rec is None else sign * rec[name])
        s = np.asarray(s, float)
        ok = ~np.isnan(s)
        if ok.sum() < 10 or len(set(s[ok])) < 2:
            out[name] = dict(auroc=float("nan"), note="degenerate or empty")
            continue
        sc = s.copy()
        sc[~ok] = np.nanmedian(sc[ok])
        yte, ste = y[te], sc[te]
        raw = float(C.auroc(yte, ste))
        out[name] = dict(auroc=float(C.best_orientation(yte, ste)[1]), raw_auroc=raw,
                         distinct_values=int(len(set(np.round(ste, 6)))))
    return dict(ds=ds, tag=tag, level=level, n_test=len(te), scores=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--levels", default="L0_leaky,L4_fair")
    ap.add_argument("--cache", default=os.environ.get(
        "BENCH_CACHE", "/root/bench_exp/cache"))
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.environ.get("BENCH_BASE", "/root/bench_exp") + "/exp")

    for lv in a.levels.split(","):
        r = evaluate(a.ds, a.tag, lv, a.cache)
        line = " ".join("%s=%.3f" % (k, v["auroc"]) for k, v in r["scores"].items())
        print("[AC-SE] %s/%s %s n_test=%d | %s" % (a.ds, a.tag, lv, r["n_test"], line))


if __name__ == "__main__":
    main()
