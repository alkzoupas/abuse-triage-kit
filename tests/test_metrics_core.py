"""Known-answer tests: tiny hand-computed inputs with known outputs."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from triage_kit.eval.metrics import average_precision, roc_auc


def test_average_precision_known_answer() -> None:
    # ranks: P N P N -> precision at hits 1/1 and 2/3 -> AP = (1 + 2/3) / 2
    assert average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]) == pytest.approx(5 / 6)


def test_average_precision_ties_form_one_threshold() -> None:
    assert average_precision([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_average_precision_weights_equal_duplication() -> None:
    y, s = [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]
    w = [2, 1, 0, 3]
    dup_y = [1, 1, 0, 0, 0, 0]
    dup_s = [0.9, 0.9, 0.8, 0.6, 0.6, 0.6]
    assert average_precision(y, s, w) == pytest.approx(average_precision(dup_y, dup_s))


def test_roc_auc_known_answer() -> None:
    # pos {0.9, 0.7} vs neg {0.8, 0.6}: 3 of 4 pairs ordered correctly
    assert roc_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]) == pytest.approx(0.75)
    assert roc_auc([1, 0], [0.5, 0.5]) == pytest.approx(0.5)


def test_match_sklearn_on_random_input() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = np.round(rng.random(300), 2)  # rounding creates ties
    assert average_precision(y, s) == pytest.approx(average_precision_score(y, s))
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s))
