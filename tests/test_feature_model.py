import numpy as np
import pytest

from tests.conftest import Dataset
from triage_kit.detectors.feature_model import VARIANTS, FeatureModel
from triage_kit.detectors.features import feature_table


def test_four_fixed_variants() -> None:
    assert sorted(VARIANTS) == ["gb", "gb+sim", "lr", "lr+sim"]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_fit_score_interface_and_consistency(variant: str, dev_s1: Dataset) -> None:
    data = dev_s1.data[:1500]
    y = np.array([int(lab["is_abuse"]) for lab in dev_s1.labels[:1500]])
    _, X, names = feature_table(data, reference=data)
    m = FeatureModel(variant).fit_table(X[:1000], y[:1000], names, reference=data)
    p = m.predict_table(X[1000:])
    assert ((0 <= p) & (p <= 1)).all()
    s, why = m.score(data[1100])
    assert s == pytest.approx(p[100])
    assert why.startswith(variant)
    assert ("template_sim" in m.columns) == variant.endswith("+sim")
    m2 = FeatureModel(variant).fit_table(X[:1000], y[:1000], names, reference=data)
    assert np.array_equal(m2.predict_table(X[1000:]), p)  # deterministic


def test_model_learns_an_obvious_signal() -> None:
    rng = np.random.default_rng(0)
    names = ["a", "b", "template_sim"]
    X = rng.normal(size=(400, 3))
    y = (X[:, 0] > 1).astype(int)
    for v in VARIANTS:
        m = FeatureModel(v).fit_table(X, y, names)
        p = m.predict_table(X)
        assert p[y == 1].mean() > p[y == 0].mean() + 0.3
