"""Acceptance for the second-stage recall analysis (PREREGISTRATION.md, AMENDMENTS, 2026-11-09).

Known-answer tests: every expected number below is worked out by hand in the comments.
"""

import numpy as np
import pytest

from triage_kit.eval.second_stage import (
    cell_second_stage,
    take_top_k,
    two_stage,
)
from triage_kit.report import _second_stage


def test_take_top_k_takes_the_highest_scores() -> None:
    s = np.array([0.1, 0.9, 0.5, 0.7])
    got = take_top_k(s, np.ones(4), 2)
    assert got.tolist() == [0.0, 1.0, 0.0, 1.0]


def test_take_top_k_splits_a_tie_block_that_straddles_the_budget() -> None:
    # 0.9 is taken whole (1 of budget 2); the three 0.5s share the last slot: 1/3 each
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.1])
    got = take_top_k(s, np.ones(5), 2)
    assert got == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3, 0.0])
    assert got.sum() == pytest.approx(2.0)


def test_take_top_k_respects_weights_and_small_pools() -> None:
    # weights are bootstrap multiplicities: the top account counts twice and fills the budget
    s = np.array([0.9, 0.5, 0.1])
    assert take_top_k(s, np.array([2.0, 1.0, 1.0]), 2).tolist() == [1.0, 0.0, 0.0]
    # zero-weight accounts never use budget; budget larger than the pool takes everything
    assert take_top_k(s, np.array([0.0, 1.0, 1.0]), 5).tolist() == [0.0, 1.0, 1.0]


def test_two_stage_known_answer() -> None:
    # 10 accounts, 4 abusive (ids 0-3). Budget 2 per stage.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s1 = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2])
    # stage 1 flags account 0 (abuse) and account 4 (benign): recall 1/4
    # stage 2 ranks the 8 allowed accounts; it puts abuse 1 and 2 on top: adds 2/4
    s2 = np.array([0.0, 0.9, 0.8, 0.1, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2])
    r = two_stage(y, s1, s2, np.ones(10), 2)
    assert r["stage1_recall"] == pytest.approx(0.25)
    assert r["added_recall"] == pytest.approx(0.5)
    assert r["stage2_precision"] == pytest.approx(1.0)


def test_two_stage_never_reflags_what_stage_one_caught() -> None:
    y = np.array([1, 1, 0, 0, 0, 0])
    s1 = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1])  # stage 1 catches both
    s2 = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1])  # stage 2 loves the same two
    r = two_stage(y, s1, s2, np.ones(6), 2)
    assert r["stage1_recall"] == pytest.approx(1.0)
    assert r["added_recall"] == pytest.approx(0.0)


def test_random_second_stage_is_the_exact_expectation() -> None:
    # stage 1 flags accounts 0 and 4, leaving 8 allowed of which 3 are abusive.
    # 2 random flags of 8 catch 3 * 2/8 = 0.75 abuse accounts on average: 0.75/4 added recall
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    s1 = np.array([0.9, 0.1, 0.1, 0.1, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2])
    r = two_stage(y, s1, np.zeros(10), np.ones(10), 2)
    assert r["added_recall"] == pytest.approx(0.75 / 4)
    assert r["stage2_precision"] == pytest.approx(3 / 8)


def _toy() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(0)
    n, n_pos = 400, 40
    y = np.r_[np.ones(n_pos), np.zeros(n - n_pos)].astype(int)
    s1 = rng.random(n)  # an uninformative first stage
    good = y + rng.normal(0, 0.3, n)  # an informative second stage
    units = np.array([f"u{i}" for i in range(n)])
    return y, s1, {"good": good, "more-stage-1": s1}, units


def test_cell_reports_every_second_stage_and_a_difference_from_random() -> None:
    y, s1, second, units = _toy()
    c = cell_second_stage(y, s1, second, units, k=20, n_boot=200, seed=1)
    assert set(c["stage2"]) == {"good", "more-stage-1", "random"}
    g = c["stage2"]["good"]
    assert g["added_recall"] > c["stage2"]["random"]["added_recall"] + 0.2
    lo, hi = g["diff_vs_random_ci95"]
    assert lo < g["diff_vs_random"] < hi
    assert lo > 0  # an informative second stage clearly beats random here
    # more of an uninformative first stage is no better than random: interval covers 0
    lo_m, hi_m = c["stage2"]["more-stage-1"]["diff_vs_random_ci95"]
    assert lo_m < 0 < hi_m
    assert c["n_pos"] == 40 and c["k"] == 20
    assert 0 <= c["stage1_recall"] <= 1


def test_cell_is_deterministic_for_a_seed() -> None:
    y, s1, second, units = _toy()
    a = cell_second_stage(y, s1, second, units, k=20, n_boot=100, seed=7)
    b = cell_second_stage(y, s1, second, units, k=20, n_boot=100, seed=7)
    assert a == b


def test_bootstrap_resamples_whole_groups() -> None:
    # all positives belong to one group: resampling units can never change which positives
    # are present, so every replicate gives the same added recall and the interval has no width
    y = np.r_[np.ones(5), np.zeros(45)].astype(int)
    s1 = np.r_[np.zeros(5), np.arange(1.0, 46.0)]  # stage 1 only ever flags benign accounts
    s2 = np.r_[np.ones(5), np.zeros(45)]
    units = np.array(["g"] * 5 + [f"b{i}" for i in range(45)])
    c = cell_second_stage(y, s1, {"x": s2, "more-stage-1": s1}, units, k=5, n_boot=50, seed=3)
    assert c["stage2"]["x"]["added_recall_ci95_plain"] == pytest.approx([1.0, 1.0])


def test_report_section_is_empty_without_results_and_scores_the_criterion() -> None:
    assert _second_stage(None) == ""
    fams = ["ato", "review_ring", "promo_abuse", "scraping", "card_testing", "template_operator"]

    def cell(fam: str, cond: str, v2: float, lo: float) -> dict[str, object]:
        def st(x: float, ci_lo: float) -> dict[str, object]:
            return {
                "added_recall": x,
                "added_recall_ci95": [max(0.0, x - 0.1), min(1.0, x + 0.1)],
                "stage2_precision": 0.5,
                "diff_vs_random": x - 0.01,
                "diff_vs_random_ci95": [ci_lo, x + 0.1],
            }

        return {
            "family": fam,
            "condition": cond,
            "n_pos": 30,
            "k": 50,
            "stage1_recall": 0.2,
            "stage2": {
                "llm:v2": st(v2, lo),
                "llm:v1": st(v2 + 0.05, lo),
                "rules:all": st(0.3, 0.1),
                "more-stage-1": st(0.05, -0.02),
                "random": st(0.01, 0.0),
            },
        }

    # held out: 4 families clear random, 2 do not -> the criterion (at least 4 of 6) is a hit
    cells = [cell(f, "heldout", 0.4, 0.2 if i < 4 else -0.05) for i, f in enumerate(fams)]
    cells += [cell(f, "in_family", 0.02, -0.01) for f in fams]
    page = _second_stage({"model": "m", "k": 50, "stage1": "gb+sim", "cells": cells})
    assert "<h2>Second stage: what the LLM adds after the feature model</h2>" in page
    assert "4 of 6" in page and "hit" in page
    assert "Exploratory" in page
    # with only 3 families clearing random it is a MISS
    cells[3]["stage2"]["llm:v2"]["diff_vs_random_ci95"] = [-0.05, 0.5]  # type: ignore[index]
    page = _second_stage({"model": "m", "k": 50, "stage1": "gb+sim", "cells": cells})
    assert "3 of 6" in page and "MISS" in page


# --- Story 22: fixes from the round-1 design review (reviews/round-1) ---


def test_fractional_stage_one_tie_is_a_surrogate_not_the_sequential_expectation() -> None:
    # Reviewer's counterexample. Accounts A, B, C; A is the only abuse; budget 1 per stage.
    # Stage 1 ties A and B. Really breaking that tie at random:
    #   removes A (prob 1/2): stage 2 can catch no abuse                       -> 0
    #   removes B (prob 1/2): stage 2 ties A and C, catches A with prob 1/2   -> 1/2
    # so the sequential expectation is 1/4. The fractional rule instead leaves weights
    # (1/2, 1/2, 1), takes 1 of the 3/2 weight tied at stage-2 score 1, and catches
    # (1/2) / (3/2) = 1/3. The rule is a defined surrogate; it is exact only when stage 1
    # has no tie at its budget. This test documents that limit.
    r = two_stage([1, 0, 0], [1, 1, 0], [1, 0, 1], np.ones(3), 1)
    assert r["added_recall"] == pytest.approx(1 / 3)
    assert abs(r["added_recall"] - 1 / 4) > 0.08


def test_cell_records_whether_stage_one_ties_at_the_budget() -> None:
    y = np.array([1, 0, 0])
    s2 = {"x": np.array([1.0, 0.0, 1.0]), "more-stage-1": np.array([1.0, 1.0, 0.0])}
    units = np.array(["a", "b", "c"])
    tied = cell_second_stage(y, np.array([1.0, 1.0, 0.0]), s2, units, k=1, n_boot=5, seed=1)
    clean = cell_second_stage(y, np.array([1.0, 0.5, 0.0]), s2, units, k=1, n_boot=5, seed=1)
    assert tied["stage1_boundary_tie"] is True
    assert clean["stage1_boundary_tie"] is False


_FAMS = ["ato", "review_ring", "promo_abuse", "scraping", "card_testing", "template_operator"]


def _report_page(tie: bool | None = False, tied_family: str | None = None) -> str:
    def st(x: float, precision: float) -> dict[str, object]:
        return {
            "added_recall": x,
            "added_recall_ci95": [max(0.0, x - 0.1), min(1.0, x + 0.1)],
            "stage2_precision": precision,
            "diff_vs_random": x - 0.01,
            "diff_vs_random_ci95": [0.02, x + 0.1],
        }

    cells = []
    for cond in ("heldout", "in_family"):
        for fam in _FAMS:
            c: dict[str, object] = {
                "family": fam,
                "condition": cond,
                "n_pos": 30,
                "k": 50,
                "stage1_recall": 0.2,
                "stage2": {
                    "llm:v2": st(0.2, 0.12),
                    "llm:v1": st(0.25, 0.15),
                    "rules:all": st(0.3, 0.18),
                    "more-stage-1": st(0.05, 0.03),
                    "random": st(0.01, 0.006),
                },
            }
            if tie is not None:
                c["stage1_boundary_tie"] = tie or (cond == "heldout" and fam == tied_family)
            cells.append(c)
    return _second_stage({"model": "m", "k": 50, "stage1": "gb+sim", "cells": cells})


def test_report_describes_a_retrospective_calculation_on_full_test_scores() -> None:
    # finding 1: the reviewer scored every test account in mixed batches; stage 1's filter is
    # applied afterwards in the ranking. The report must not claim survivor-only inputs.
    page = _report_page()
    assert "sees only what stage 1 allowed" not in page
    assert "retrospective" in page
    assert "scored every test account" in page
    assert "not tested" in page  # independence of a score from its batch
    assert "does not measure" in page  # reduced reviewer work / a routed deployment


def test_report_says_what_the_intervals_are_conditional_on() -> None:
    # finding 2: the bootstrap resamples accounts and holds the reviewer's scores fixed
    page = _report_page()
    assert "conditional on" in page
    assert "do not cover" in page


def test_report_shows_stage_two_precision_beside_added_recall() -> None:
    # finding 4: precision 0.12 at k = 50 is 6.0 of the family's accounts in 50 flags
    page = _report_page()
    assert "llm:v2 precision" in page
    assert "0.120 (6.0 of 50)" in page


def test_report_states_the_tie_rule_as_a_surrogate_and_whether_stage_one_tied() -> None:
    # finding 3
    assert "fractional" in _report_page()
    assert "No cell in this run has a stage-1 tie at the budget" in _report_page(tie=False)
    tied = _report_page(tie=False, tied_family="scraping")
    assert "Stage 1 ties at the budget in: scraping (held out)" in tied
    assert "does not record" in _report_page(tie=None)
