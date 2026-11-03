"""Known-answer tests for every remaining reported number."""

from pathlib import Path

import numpy as np
import pytest

from triage_kit.eval.metrics import (
    average_precision,
    base_rate,
    bootstrap_metrics,
    cost_per_1000,
    load_prices,
    percentile_interval,
    precision_recall_at_k,
    roc_auc,
)

ROOT = Path(__file__).resolve().parents[1]


def test_precision_recall_at_k_known_answer() -> None:
    y = [1, 0, 1, 0, 1]
    s = [0.9, 0.8, 0.7, 0.6, 0.1]
    # top-2 = {0.9 P, 0.8 N}: precision 1/2, recall 1/3
    assert precision_recall_at_k(y, s, 2) == pytest.approx((0.5, 1 / 3))
    # k larger than n: everything is reviewed
    assert precision_recall_at_k(y, s, 10) == pytest.approx((3 / 5, 1.0))


def test_precision_at_k_breaks_ties_deterministically_by_input_order() -> None:
    assert precision_recall_at_k([0, 1], [0.5, 0.5], 1) == pytest.approx((0.0, 0.0))


def test_base_rate_known_answer() -> None:
    assert base_rate([1, 0, 0, 0]) == pytest.approx(0.25)


def test_percentile_interval_known_answer() -> None:
    samples = np.arange(101.0)  # 0..100
    assert percentile_interval(samples, 0.10) == pytest.approx((5.0, 95.0))


def test_bootstrap_adds_two_pseudo_units_per_class_at_the_score_extremes() -> None:
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    units = np.array(["g1", "g1", "b1", "b2", "b3"])
    out = bootstrap_metrics(y, s, units, n_boot=200, seed=0)
    assert list(out["y_aug"][-4:]) == [1, 1, 0, 0]
    assert list(out["s_aug"][-4:]) == [1.9, -0.9, 1.9, -0.9]
    # a perfect ranking no longer yields a zero-width interval on 1 real positive group
    assert out["roc"].max() == pytest.approx(1.0) and out["roc"].min() < 1.0


def test_bootstrap_widens_when_no_positive_fires() -> None:
    # 0 of 20 positives fire; 3 of 100 negatives fire. Without the correction every
    # replicate has ROC < 0.5; with it, 0.5 must be inside the 95% interval.
    y = np.r_[np.ones(20), np.zeros(100)]
    s = np.r_[np.zeros(20), np.ones(3), np.zeros(97)]
    units = np.array([f"u{i}" for i in range(120)])
    out = bootstrap_metrics(y, s, units, n_boot=1000, seed=0)
    lo, hi = percentile_interval(out["roc"], 0.05)
    assert lo < 0.5 < hi


def test_bootstrap_rows_equal_weighted_scalar_metrics_and_resample_whole_units() -> None:
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(10), np.zeros(30)].astype(int)
    s = rng.random(40)
    units = np.array([f"g{i // 5}" for i in range(10)] + [f"b{i}" for i in range(30)])
    out = bootstrap_metrics(y, s, units, n_boot=20, seed=3)
    w, ya, sa = out["weights"], out["y_aug"], out["s_aug"]
    for b in range(20):
        assert out["ap"][b] == pytest.approx(average_precision(ya, sa, w[b]))
        assert out["roc"][b] == pytest.approx(roc_auc(ya, sa, w[b]))
        # members of a group always share a weight (whole groups are resampled)
        for g in range(2):
            assert len(set(w[b, g * 5 : g * 5 + 5])) == 1
        # stratified sizes: 2 real positive groups + 2 pseudo units, 30 + 2 negatives
        pos_w = w[b, :10].sum() / 5 + w[b, 40] + w[b, 41]
        assert pos_w == 4 and w[b, 10:40].sum() + w[b, 42] + w[b, 43] == 32
    again = bootstrap_metrics(y, s, units, n_boot=20, seed=3)
    assert np.array_equal(again["ap"], out["ap"])


def test_cost_per_1000_known_answer() -> None:
    # 2,000 accounts, 3M input + 1M output tokens at $2 / $10 per Mtok = $16 -> $8 per 1,000
    assert cost_per_1000(3_000_000, 1_000_000, 2000, 2.0, 10.0) == pytest.approx(8.0)


def test_prices_file_is_marked_placeholder() -> None:
    prices = load_prices(ROOT / "config" / "prices.yaml")
    assert prices["placeholder"] is True
    assert "PLACEHOLDER" in (ROOT / "config" / "prices.yaml").read_text()
    for m in prices["models"].values():
        assert {"input_per_mtok", "output_per_mtok"} <= set(m)


def test_bootstrap_without_positives_is_undefined_not_zero() -> None:
    out = bootstrap_metrics(np.zeros(5), np.arange(5.0), np.array(list("abcde")), 10, 0)
    assert np.isnan(out["ap"]).all() and np.isnan(out["roc"]).all()
    lo, hi = percentile_interval(out["ap"], 0.05)
    assert np.isnan(lo) and np.isnan(hi)


def test_expanded_interval_known_answer() -> None:
    from triage_kit.eval.metrics import expanded_interval, expansion_factor

    # n = 11 units, 95%: sqrt(11/10) * t(0.975, 10) / z(0.975) = 1.0488 * 2.2281 / 1.9600
    assert expansion_factor(11, 0.05) == pytest.approx(1.19228, abs=1e-4)
    # symmetric case: est 0.5, percentile interval (0.4, 0.6) widens by the factor
    lo, hi = expanded_interval(0.5, (0.4, 0.6), n_units=11, alpha=0.05)
    assert (lo, hi) == pytest.approx((0.5 - 0.119228, 0.5 + 0.119228), abs=1e-5)
    # never narrows, and clips to [0, 1]
    assert expanded_interval(0.98, (0.9, 1.0), n_units=3, alpha=0.05)[1] == 1.0
    assert expanded_interval(0.5, (0.6, 0.7), n_units=11, alpha=0.05)[0] <= 0.6


def test_expansion_vanishes_with_many_units() -> None:
    from triage_kit.eval.metrics import expansion_factor

    assert expansion_factor(100_000, 0.05) == pytest.approx(1.0, abs=1e-3)
    assert expansion_factor(12, 0.05 / 84) > expansion_factor(12, 0.05) > 1.0


def test_expected_random_ap_known_answers() -> None:
    from itertools import permutations

    from triage_kit.eval.metrics import expected_random_ap

    # N=2, P=1: positive at rank 1 (AP 1) or rank 2 (AP 1/2) -> 3/4
    assert expected_random_ap(1, 2) == pytest.approx(0.75)
    # all positives -> 1
    assert expected_random_ap(4, 4) == pytest.approx(1.0)
    # brute force over every ranking of 2 positives among 5
    aps = [average_precision(list(p), [5, 4, 3, 2, 1]) for p in set(permutations([1, 1, 0, 0, 0]))]
    assert expected_random_ap(2, 5) == pytest.approx(float(np.mean(aps)))
    # above the base rate when positives are few
    assert expected_random_ap(20, 1000) > 20 / 1000


def test_null_expected_ap_respects_ties() -> None:
    from itertools import permutations

    from triage_kit.eval.metrics import expected_random_ap, null_expected_ap

    # no ties -> the random-ranking formula
    assert null_expected_ap(np.arange(50.0), 5) == pytest.approx(expected_random_ap(5, 50))
    # everything tied -> exactly the base rate
    assert null_expected_ap(np.zeros(40), 4) == pytest.approx(0.1)
    # brute force with ties: scores [3, 2, 2, 1, 1, 1], 2 positives
    s = [3.0, 2.0, 2.0, 1.0, 1.0, 1.0]
    aps = [average_precision(list(p), s) for p in set(permutations([1, 1, 0, 0, 0, 0]))]
    assert null_expected_ap(np.array(s), 2) == pytest.approx(float(np.mean(aps)))
