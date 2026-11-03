"""Metric functions. Every reported number comes from a function here, and each
one has a known-answer test in tests/test_metrics_*.py.

All ranking metrics accept optional non-negative integer-like weights, which act
exactly like duplicating rows. The bootstrap uses this to resample without
copying arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import yaml
from scipy import stats

ArrayLike = Sequence[float] | npt.NDArray[np.floating] | npt.NDArray[np.integer]


def _prep(
    y: ArrayLike, s: ArrayLike, w: ArrayLike | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    ya = np.asarray(y, dtype=float)
    sa = np.asarray(s, dtype=float)
    wa = np.ones_like(ya) if w is None else np.asarray(w, dtype=float)
    if not (ya.shape == sa.shape == wa.shape):
        raise ValueError("y, scores and weights must have the same shape")
    return ya, sa, wa


def _threshold_curve(
    y: ArrayLike, s: ArrayLike, w: ArrayLike | None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cumulative (weighted) true and false positives at each distinct score, high to low."""
    ya, sa, wa = _prep(y, s, w)
    order = np.argsort(-sa, kind="mergesort")
    ys, ss, ws = ya[order], sa[order], wa[order]
    tp = np.cumsum(ys * ws)
    fp = np.cumsum((1 - ys) * ws)
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]  # last index of each tie block
    return tp[last], fp[last]


def average_precision(y: ArrayLike, s: ArrayLike, w: ArrayLike | None = None) -> float:
    """Area under the precision-recall curve, step-wise (same definition as sklearn)."""
    tp, fp = _threshold_curve(y, s, w)
    if tp[-1] <= 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / tp[-1]
    d_recall = np.diff(np.r_[0.0, recall])
    return float(np.sum(d_recall * precision))


def roc_auc(y: ArrayLike, s: ArrayLike, w: ArrayLike | None = None) -> float:
    """Probability a random positive outscores a random negative (ties count 1/2)."""
    tp, fp = _threshold_curve(y, s, w)
    if tp[-1] <= 0 or fp[-1] <= 0:
        return float("nan")
    tpr = np.r_[0.0, tp / tp[-1]]
    fpr = np.r_[0.0, fp / fp[-1]]
    return float(np.trapezoid(tpr, fpr))


def base_rate(y: ArrayLike) -> float:
    return float(np.mean(np.asarray(y, dtype=float)))


def precision_recall_at_k(y: ArrayLike, s: ArrayLike, k: int) -> tuple[float, float]:
    """Precision and recall among the top-k scores (a fixed review budget).
    Ties are broken by input order (stable sort), so results are deterministic."""
    ya, sa = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    k = min(k, len(ya))
    top = np.argsort(-sa, kind="mergesort")[:k]
    hits = float(ya[top].sum())
    n_pos = float(ya.sum())
    return hits / k, (hits / n_pos if n_pos else float("nan"))


def percentile_interval(samples: ArrayLike, alpha: float) -> tuple[float, float]:
    """Two-sided percentile interval with total tail mass alpha."""
    a = np.asarray(samples, dtype=float)
    a = a[~np.isnan(a)]
    if not len(a):
        return float("nan"), float("nan")
    lo, hi = np.quantile(a, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def expected_random_ap(n_pos: int, n: int) -> float:
    """Exact expected average precision of a uniformly random ranking of n items with
    n_pos positives: H_n / n + (n_pos - 1) / (n (n - 1)) * (n - H_n).

    With few positives this is noticeably above the base rate n_pos / n, so it is the
    fairer "no signal" reference for PR-AUC."""
    if n_pos <= 0 or n <= 0:
        return float("nan")
    if n == 1:
        return 1.0
    h = float(np.sum(1.0 / np.arange(1, n + 1)))
    return h / n + (n_pos - 1) / (n * (n - 1)) * (n - h)


def null_expected_ap(scores: ArrayLike, n_pos: int) -> float:
    """Exact expected AP when n_pos labels are assigned at random to these scores,
    keeping the scores' tie structure (sklearn tie handling: one threshold per tie block).

    For tie block b of size m_b, with c_b items in earlier blocks, p = P/N and
    q = P(P-1)/(N(N-1)):  E[AP] = (1/P) sum_b [m_b c_b q + m_b p + m_b (m_b - 1) q] / (c_b + m_b).
    No ties gives expected_random_ap(P, N); all tied gives the base rate P/N."""
    sa = np.sort(np.asarray(scores, dtype=float))[::-1]
    n = len(sa)
    if n_pos <= 0 or n == 0:
        return float("nan")
    p = n_pos / n
    q = n_pos * (n_pos - 1) / (n * (n - 1)) if n > 1 else 0.0
    _, sizes = np.unique(-sa, return_counts=True)  # block sizes, high score to low
    m = sizes.astype(float)
    c = np.r_[0.0, np.cumsum(m)[:-1]]
    return float(np.sum((m * c * q + m * p + m * (m - 1) * q) / (c + m)) / n_pos)


def expansion_factor(n_units: int, alpha: float) -> float:
    """Few-cluster widening for a percentile interval: sqrt(n/(n-1)) * t(1-a/2, n-1) / z(1-a/2).

    The bootstrap spread of a statistic over n resampled units underestimates its
    sampling spread by about (n-1)/n, and normal-like tails understate the
    uncertainty with few units; this follows the motivation of Hesterberg's
    "expanded percentile interval"."""
    n = max(int(n_units), 2)
    t = float(stats.t.ppf(1 - alpha / 2, n - 1))
    z = float(stats.norm.ppf(1 - alpha / 2))
    return float(np.sqrt(n / (n - 1)) * t / z)


def expanded_interval(
    est: float, interval: tuple[float, float], n_units: int, alpha: float
) -> tuple[float, float]:
    """Widen each side of a percentile interval around the point estimate by
    expansion_factor(n_units, alpha). Never narrows; clipped to [0, 1]."""
    lo, hi = interval
    if np.isnan(lo) or np.isnan(hi) or np.isnan(est):
        return lo, hi
    f = expansion_factor(n_units, alpha)
    new_lo = min(lo, est - f * max(est - lo, 0.0))
    new_hi = max(hi, est + f * max(hi - est, 0.0))
    return float(max(0.0, new_lo)), float(min(1.0, new_hi))


def bootstrap_metrics(
    y: ArrayLike,
    s: ArrayLike,
    units: Sequence[str] | npt.NDArray[Any],
    n_boot: int,
    seed: int,
    continuity: bool = True,
) -> dict[str, npt.NDArray[np.float64]]:
    """Stratified cluster bootstrap of AP and ROC-AUC, with a continuity correction.

    Positive units and negative units are resampled separately, with replacement.
    A unit is a group for coordinated families and an account otherwise; all
    members of a drawn unit enter together.

    Continuity correction (in the spirit of Agresti-Coull's "add two successes and
    two failures"): each class gets two pseudo-units, one scored above every real
    score and one below, and each replicate draws (real units + 2) units per class.
    Without it, a class in which no member fired (e.g. 0 of 21 abuse accounts hit a
    rule that fires on 3% of benign accounts) can only produce zero-width intervals
    on that side, which makes a null scorer look significant. Point estimates are
    computed elsewhere, on real data only. The evaluation applies the correction to
    ROC-AUC only: a pseudo-positive above every score adds ~1/(P+2) to AP, which
    shifts AP intervals upward (null calibration on dev seeds, PREREGISTRATION.md
    AMENDMENTS).

    Returns per-replicate AP and ROC-AUC, plus the augmented y, scores and weight
    matrix actually used (rows = replicates)."""
    ya = np.asarray(y, dtype=float)
    sa = np.asarray(s, dtype=float)
    ua = np.asarray(units).astype(str)
    if ya.sum() == 0 or ya.sum() == len(ya):  # undefined without both classes
        nan = np.full(n_boot, np.nan)
        return {
            "ap": nan,
            "roc": nan.copy(),
            "weights": np.zeros((n_boot, len(ya))),
            "y_aug": ya,
            "s_aug": sa,
        }
    if continuity:
        hi, lo = sa.max() + 1.0, sa.min() - 1.0
        ya = np.r_[ya, 1, 1, 0, 0]
        sa = np.r_[sa, hi, lo, hi, lo]
        ua = np.r_[ua, ["\0pos_hi", "\0pos_lo", "\0neg_hi", "\0neg_lo"]]
    uniq, inv = np.unique(ua, return_inverse=True)
    unit_pos = np.zeros(len(uniq), dtype=bool)
    unit_pos[inv[ya == 1]] = True
    pos_u, neg_u = np.nonzero(unit_pos)[0], np.nonzero(~unit_pos)[0]
    rng = np.random.default_rng(seed)
    unit_w = np.zeros((n_boot, len(uniq)))
    for b in range(n_boot):
        unit_w[b] += np.bincount(rng.choice(pos_u, size=len(pos_u)), minlength=len(uniq))
        unit_w[b] += np.bincount(rng.choice(neg_u, size=len(neg_u)), minlength=len(uniq))
    w = unit_w[:, inv]

    order = np.argsort(-sa, kind="mergesort")
    ys, ss, ws = ya[order], sa[order], w[:, order]
    tp = np.cumsum(ws * ys, axis=1)
    fp = np.cumsum(ws * (1 - ys), axis=1)
    last = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    tp, fp = tp[:, last], fp[:, last]
    P, N = tp[:, -1:], fp[:, -1:]
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(P, 1e-12)
    d_recall = np.diff(np.c_[np.zeros(n_boot), recall], axis=1)
    ap = np.sum(d_recall * precision, axis=1)
    tpr = np.c_[np.zeros(n_boot), tp / np.maximum(P, 1e-12)]
    fpr = np.c_[np.zeros(n_boot), fp / np.maximum(N, 1e-12)]
    roc = np.trapezoid(tpr, fpr, axis=1)
    return {"ap": ap, "roc": roc, "weights": w, "y_aug": ya, "s_aug": sa}


def cost_per_1000(
    tokens_in: float, tokens_out: float, n_accounts: int, in_per_mtok: float, out_per_mtok: float
) -> float:
    """API cost of scoring 1,000 accounts, from token totals and per-million-token prices."""
    total = (tokens_in * in_per_mtok + tokens_out * out_per_mtok) / 1e6
    return total / n_accounts * 1000


def load_prices(path: Path) -> dict[str, Any]:
    doc: dict[str, Any] = yaml.safe_load(path.read_text())
    if "models" not in doc or "placeholder" not in doc:
        raise ValueError(f"{path}: needs 'placeholder' and 'models'")
    return doc
