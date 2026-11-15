"""Twenty lexical features of a question and a candidate answer.

These are the features used by the shortcut oracle. They are
reimplemented here so that the package has no dependency on the research
codebase: only numpy is required.

The point of the feature set is to be *cheap*. It reads nothing but the two
strings, so any benchmark can be screened on a laptop before a vision--language
model is ever loaded.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

STOP = set("the a an of in on at is are was were to for with and or not no yes "
           "this that it what which who how many much".split())
NEGATIONS = {"no", "not", "none", "neither", "nor", "without", "absent"}

YESNO = {"yes", "no", "true", "false"}
WH = {"what", "which", "who", "where", "when", "why", "how"}

NUM = re.compile(r"\d")
PCT = re.compile(r"%")
COMMA = re.compile(r",")


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def question_type(q: str) -> str:
    t = set(_tokens(q))
    if t & {"is", "are", "was", "were", "do", "does", "did", "can", "could"}:
        return "yesno"
    if t & WH:
        return "wh"
    return "other"


def answer_class(a: str) -> str:
    al = a.strip().lower()
    if al in YESNO:
        return "yesno"
    if NUM.search(al) and len(_tokens(al)) <= 2:
        return "number"
    if len(_tokens(al)) == 1:
        return "single"
    return "phrase"


def features(question: str, answer: str, ans_freq: Counter, qt_freq: Counter) -> list[float]:
    """The twenty numbers, in a fixed order (see FEATURE_NAMES)."""
    qt = _tokens(question)
    at = _tokens(answer)
    ac = answer_class(answer)
    qcls = question_type(question)
    overlap = len(set(qt) & set(at))
    union = len(set(qt) | set(at)) or 1
    return [
        float(len(at)),
        float(len(answer)),
        float(len(qt)),
        len(at) / max(len(qt), 1),
        len(answer) / max(len(question), 1),
        float(ac == "yesno"),
        float(qcls == "yesno"),
        float(qcls == "wh"),
        float(qcls == "other"),
        float(ac == qcls),
        float(bool(NUM.search(answer))),
        float(bool(PCT.search(answer))),
        float(bool(COMMA.search(answer))),
        float(answer.strip().endswith(".")),
        float(answer[:1].isupper()),
        float(bool(set(at) & NEGATIONS)),
        math.log1p(ans_freq.get(answer.strip().lower(), 0)),
        math.log1p(qt_freq.get(" ".join(qt[:4]), 0)),
        float(overlap),
        overlap / union,
    ]


FEATURE_NAMES = [
    "ans_tokens", "ans_chars", "q_tokens", "ans_per_q_tok", "ans_per_q_char",
    "is_yesno", "qt_is_yesno", "qt_is_wh", "qt_other", "type_match",
    "has_digit", "has_pct", "has_comma", "ends_period", "cap_first",
    "negation_word", "log_ans_freq", "log_qt_freq", "qa_overlap_n", "qa_jaccard",
]


def build_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Turn rows of {question, answer, label} into (X, y)."""
    ans_freq = Counter(r["answer"].strip().lower() for r in rows)
    qt_freq = Counter(" ".join(_tokens(r["question"])[:4]) for r in rows)
    X = np.asarray([features(r["question"], r["answer"], ans_freq, qt_freq) for r in rows],
                   dtype=np.float32)
    y = np.asarray([int(r["label"]) for r in rows], dtype=np.int32)
    return X, y
