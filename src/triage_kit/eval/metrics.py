"""Metric functions. Every reported number comes from a function here, and each
one has a known-answer test in tests/test_metrics_*.py.

All ranking metrics accept optional non-negative integer-like weights, which act
exactly like duplicating rows. The bootstrap uses this to resample without
copying arrays.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

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
