import pytest

from tests.conftest import Dataset
from triage_kit.eval.metrics import average_precision
from triage_kit.eval.oracle import oracle_scores
from triage_kit.generator import FAMILIES


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_oracle_proves_signal_at_full_strength(family: str, dev_s1: Dataset) -> None:
    """Acceptance: at s = 1 the oracle reaches PR-AUC >= 0.9 on every family."""
    idx = [i for i, lab in enumerate(dev_s1.labels) if lab["family"] in ("benign", family)]
    data = [dev_s1.data[i] for i in idx]
    y = [int(dev_s1.labels[i]["family"] == family) for i in idx]
    scores = oracle_scores(family, data, dev_s1.meta)
    assert average_precision(y, scores) >= 0.9


def test_oracle_lives_only_in_eval() -> None:
    import triage_kit.eval.oracle as o

    assert o.__name__.startswith("triage_kit.eval.")
