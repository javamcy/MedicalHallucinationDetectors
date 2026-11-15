"""The audit itself: two-rung lexical probing and the frozen regression."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from .features import build_matrix

# Regression fitted on eleven benchmarks; see results/ for the fitting set.
REG_INTERCEPT = 0.039
REG_SLOPE = 0.365
REG_R = 0.595
REG_LOO_MAE = 0.029
REG_LOO_MAX = 0.064
REG_N = 11

# The oracle drop observed across the eleven fitted benchmarks ran from 0.000 to
# 0.135. Outside that window the regression is extrapolating, and the prospective
# test showed that its absolute error roughly doubles there.
FIT_X_RANGE = (0.000, 0.135)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def oriented(y: np.ndarray, s: np.ndarray) -> float:
    a = auroc(y, s)
    return float(max(a, 1.0 - a)) if a == a else a


@dataclass
class RungResult:
    name: str
    n_rows: int
    n_test: int
    auroc: float


@dataclass
class AuditResult:
    leaky: RungResult
    matched: RungResult
    oracle_drop: float
    predicted_detector_drop: float
    interval: tuple[float, float]
    extrapolating: bool = False

    def to_json(self) -> str:
        d = asdict(self)
        d["regression"] = dict(intercept=REG_INTERCEPT, slope=REG_SLOPE, r=REG_R,
                               loo_mae=REG_LOO_MAE, loo_max=REG_LOO_MAX, n_fit=REG_N,
                               fitted_oracle_drop_range=list(FIT_X_RANGE))
        return json.dumps(d, indent=2)


def _split(n: int, seed: int, ratios=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    a = int(n * ratios[0])
    b = a + int(n * ratios[1])
    return idx[:a], idx[a:b], idx[b:]


def score_rung(name: str, rows: list[dict], seed: int = 0) -> RungResult:
    """Fit the lexical oracle on the training split and score the test split."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = build_matrix(rows)
    tr, va, te = _split(len(rows), seed)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         random_state=seed).fit(X[tr], y[tr])
    s = clf.predict_proba(X)[:, 1]
    return RungResult(name=name, n_rows=len(rows), n_test=len(te),
                      auroc=oriented(y[te], s[te]))


def audit(leaky_rows: list[dict], matched_rows: list[dict], seed: int = 0) -> AuditResult:
    """Compute both rungs and apply the frozen regression."""
    a = score_rung("leaky", leaky_rows, seed)
    b = score_rung("matched", matched_rows, seed)
    drop = a.auroc - b.auroc
    pred = REG_INTERCEPT + REG_SLOPE * drop
    return AuditResult(leaky=a, matched=b, oracle_drop=drop,
                       predicted_detector_drop=pred,
                       interval=(pred - REG_LOO_MAX, pred + REG_LOO_MAX),
                       extrapolating=not (FIT_X_RANGE[0] <= drop <= FIT_X_RANGE[1]))
