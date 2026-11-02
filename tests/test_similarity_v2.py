import numpy as np
import pytest

from tests.conftest import Dataset
from triage_kit.detectors.similarity_v2 import SimilarityIndexV2, v2_tokens
from triage_kit.schema import Account, AccountEvents, Event


def _ae(aid: str, steps: list[tuple[str, str | None, int]]) -> AccountEvents:
    evs, t = [], 1000
    for type_, tgt, gap in steps:
        t += gap
        evs.append(Event(aid, t, type_, "10.0.0.1", "d", target_id=tgt))
    return AccountEvents(Account(aid, 0, "DE", "x.example"), tuple(evs))


def test_exact_jaccard_known_answer() -> None:
    # five accounts replay one script (5 filtered tokens each: view:L1, view:L2, two
    # 3-grams, one 4-gram); "c" is different; 600 one-off accounts add population mass
    # so max df = 1% of 606 = 6 keeps the script tokens (df 5).
    script = [("login", None, 5), ("view", "L1", 40), ("view", "L2", 40), ("search", None, 90)]
    other = [("login", None, 7), ("purchase", "L9", 3000), ("review", "L9", 9000)]
    pop = [_ae(f"s{i}", script) for i in range(5)] + [_ae("c", other)]
    pop += [_ae(f"z{i}", [("login", None, 5), ("view", f"Q{i}", 600 + i)]) for i in range(600)]
    idx = SimilarityIndexV2(pop)
    # top-3 Jaccard to copies = 1, 1, 1 -> mean 1, times n/(n+5) = 5/10
    assert idx.value(pop[0]) == pytest.approx(0.5)
    assert idx.value(pop[5]) == pytest.approx(0.0)


def test_tokens_use_only_events() -> None:
    ae = _ae("a", [("login", None, 5), ("view", "L1", 40)])
    toks = v2_tokens(ae)
    assert "view:L1" in toks and all("a" != t for t in toks)


def test_values_bounded_and_deterministic(dev_s1: Dataset) -> None:
    sub = dev_s1.data[:800]
    a = SimilarityIndexV2(sub).all_values()
    b = SimilarityIndexV2(sub).all_values()
    assert np.array_equal(a, b) and (a >= 0).all() and (a <= 1).all()
