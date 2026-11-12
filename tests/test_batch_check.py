"""Batch-independence check (PREREGISTRATION.md, amendment 2026-11-12). Known-answer tests."""

from pathlib import Path

import numpy as np
import pytest

from triage_kit.eval.batch_check import (
    pool_recall,
    read_replies,
    score_changes,
    select_pool,
    verdict,
)
from triage_kit.eval.second_stage import two_stage


def test_pool_holds_every_surviving_family_account_and_fills_with_top_benign() -> None:
    family = ["f1", "f2", "f3"]
    benign = {"b1": 0.9, "b2": 0.05, "b3": 0.4, "b4": 0.2, "b5": 0.03}
    pool = select_pool(family, benign, size=6, seed=1)
    assert set(family) <= set(pool)
    assert set(pool) - set(family) == {"b1", "b3", "b4"}  # the three highest benign scores
    assert pool == sorted(pool) and len(pool) == 6
    with pytest.raises(ValueError):
        select_pool(family, benign, size=2, seed=1)  # the family accounts alone do not fit


def test_pool_is_deterministic_and_breaks_score_ties_by_seed() -> None:
    benign = {f"b{i:02d}": 0.2 for i in range(40)}  # one tie block, 5 places to fill
    a = select_pool(["f1"], benign, size=6, seed=7)
    assert a == select_pool(["f1"], benign, size=6, seed=7)
    picks = {tuple(select_pool(["f1"], benign, size=6, seed=s)) for s in range(8)}
    assert len(picks) > 1  # the seed decides which tied accounts get in, not their names


def test_pool_recall_with_original_scores_reproduces_full_ranking_recall() -> None:
    # 12 accounts, each stage flags 2. The pool holds stage 2's whole top block.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s1 = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    s2 = np.array([0.9, 0.9, 0.7, 0.2, 0.9, 0.6, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05])
    in_pool = s2 >= 0.2
    full = two_stage(y, s1, s2, np.ones(12), 2)["added_recall"]
    assert full == pytest.approx(0.5)  # accounts 1 and 2 of four family accounts
    assert pool_recall(y, s1, s2, in_pool, 2) == pytest.approx(full)


def test_pool_recall_ranks_accounts_outside_the_pool_last() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    s1 = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.8])  # stage 1 flags the last two accounts
    s2 = np.array([0.5, 0.9, 0.3, 0.95, 0.0, 0.0])
    in_pool = np.array([True, True, True, False, False, False])
    # account 3 has the best stage-2 score: the full ranking flags it and one family account
    assert two_stage(y, s1, s2, np.ones(6), 2)["added_recall"] == pytest.approx(0.5)
    # it is outside the pool, so the pool ranking flags both family accounts instead
    assert pool_recall(y, s1, s2, in_pool, 2) == pytest.approx(1.0)


def test_score_changes_known_answer() -> None:
    old = [0.2, 0.9, 0.05, 0.5]
    new = [0.3, 0.6, 0.05, 0.75]
    c = score_changes(old, new)
    assert c["n"] == 4
    assert c["share_moved"] == pytest.approx(0.5)  # 0.2 -> 0.3 is exactly 0.1: not "more than"
    assert c["mean_abs"] == pytest.approx((0.1 + 0.3 + 0.0 + 0.25) / 4)
    assert c["mean_signed"] == pytest.approx((0.1 - 0.3 + 0.0 + 0.25) / 4)
    assert c["spearman"] == pytest.approx(0.8)
    assert score_changes(old, old)["spearman"] == pytest.approx(1.0)
    assert score_changes([], [])["n"] == 0


def test_verdict_three_outcomes() -> None:
    small = {"ato": 0.02, "card_testing": 0.10, "scraping": 0.0, "promo_abuse": 0.05}
    assert verdict(small) == "supported"  # 0.10 is within the limit
    assert verdict({**small, "review_ring": 0.11}) == "headline_only"
    assert verdict({**small, "ato": 0.12}) == "not_supported"
    with pytest.raises(ValueError):
        verdict({"scraping": 0.0})  # a headline family is missing


def test_collect_refuses_when_a_reply_is_missing(tmp_path: Path) -> None:
    batches = [["a1", "a2"], ["a3"]]
    (tmp_path / "00.txt").write_text(
        '{"id": "a1", "verdict": "flag", "score": 0.9, "rationale": "x"}\n'
        '{"id": "a2", "verdict": "pass", "score": 0.1, "rationale": "y"}\n'
    )
    with pytest.raises(FileNotFoundError):
        read_replies(tmp_path, batches, [0, 1])
    assert read_replies(tmp_path, batches, [0]) == {"a1": 0.9, "a2": 0.1}
    (tmp_path / "01.txt").write_text("no json here\n")
    with pytest.raises(ValueError):
        read_replies(tmp_path, batches, [0, 1])  # an unparsed account is not scored 0.5


def _moves(share: float) -> dict[str, dict[str, object]]:
    one = {"n": 10, "share_moved": share, "mean_abs": 0.04, "mean_signed": -0.01, "spearman": 0.9}
    return {"all": one, "family": one, "benign": one}


def test_report_shows_batch_check_beside_second_stage() -> None:
    from triage_kit.report import _batch_check

    assert _batch_check(None) == ""
    d = {
        "pool": 188,
        "pool_family": 102,
        "batches": 5,
        "rerun_batches": [0, 1],
        "max_recall_change": 0.1,
        "moved_threshold": 0.1,
        "k": 50,
        "verdict": "headline_only",
        "families": [
            {"family": "ato", "n_pos": 42, "added_recall_original": 0.879121,
             "added_recall_rebatched": 0.857143, "change": -0.021978},
            {"family": "review_ring", "n_pos": 19, "added_recall_original": 0.263158,
             "added_recall_rebatched": 0.105263, "change": -0.157895},
        ],
        "original_vs_rebatched": _moves(0.25),
        "rebatched_vs_rerun": _moves(0.05),
        "original_vs_rebatched_on_rerun_accounts": _moves(0.2),
    }  # fmt: skip
    html = _batch_check(d)
    assert "Exploratory" in html and "2026-11-12" in html
    assert "0.879" in html and "0.857" in html and "−0.022" in html
    assert "−0.158" in html
    assert "supported for the headline families only" in html.lower()
    assert "stress test" in html  # says what the dense batches are and are not
    assert "25%" in html and "5%" in html  # movement beside the rerun noise floor
    assert "not rescored" in html  # the limit: accounts outside the pool
    for v, text in (("supported", "Supported"), ("not_supported", "Not supported")):
        assert text in _batch_check({**d, "verdict": v})
