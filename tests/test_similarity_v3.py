import numpy as np
import pytest

from tests.conftest import Dataset
from triage_kit.detectors.similarity_v3 import SimilarityIndexV3, v3_tokens
from triage_kit.schema import Account, AccountEvents, Event


def _ae(aid: str, steps: list[tuple[str, str | None, str | None]]) -> AccountEvents:
    evs = [
        Event(aid, 1000 + 60 * k, t, "10.0.0.1", "d", target_id=tgt, text=txt)
        for k, (t, tgt, txt) in enumerate(steps)
    ]
    return AccountEvents(Account(aid, 0, "DE", "x.example"), tuple(evs))


def test_tokens_are_ordered_view_pairs_within_three_views_and_queries() -> None:
    ae = _ae(
        "a",
        [
            ("view", "A", None),
            ("search", None, "red lamp"),
            ("view", "B", None),
            ("view", "C", None),
            ("view", "A", None),
            ("view", "D", None),
        ],
    )
    toks = v3_tokens(ae)
    assert {"A>B", "A>C", "B>C", "B>A", "B>D", "C>A", "C>D", "A>D", "q:red lamp"} == toks
    assert "B>A" in toks and "A>A" not in toks  # x != y; order matters


def test_shared_count_not_diluted_by_long_history() -> None:
    # four crew accounts share a 4-view script (3 adjacent + 2 skip + 1 skip-2 pairs = 6 tokens);
    # one of them also has a long unique history, which must not lower its value.
    script = [("view", f"S{k}", None) for k in range(4)]
    crew = [_ae(f"c{i}", script) for i in range(3)]
    long = _ae("c3", [("view", f"U{k}", None) for k in range(200)] + script)
    pop = crew + [long] + [_ae(f"z{i}", [("view", f"Z{i}", None)]) for i in range(1000)]
    idx = SimilarityIndexV3(pop)  # max df = max(2, 0.2% of 1004) = 2 ... crew tokens have df 4
    assert idx.value(pop[0]) == pytest.approx(0.0)  # df 4 > 2: filtered out at this size
    pop += [_ae(f"y{i}", [("view", f"Y{i}", None)]) for i in range(1000)]
    idx = SimilarityIndexV3(pop)  # max df = 4 now: script pairs kept, 6 shared with each mate
    assert idx.value(pop[0]) == pytest.approx(np.log1p(6.0))
    assert idx.value(long) == pytest.approx(np.log1p(6.0))
    assert idx.value(pop[10]) == pytest.approx(0.0)


def test_values_nonnegative_and_deterministic(dev_s1: Dataset) -> None:
    sub = dev_s1.data[:800]
    a = SimilarityIndexV3(sub).all_values()
    b = SimilarityIndexV3(sub).all_values()
    assert np.array_equal(a, b) and (a >= 0).all()
