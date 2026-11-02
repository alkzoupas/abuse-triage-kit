"""Feature model: logistic regression or gradient boosting on engineered features,
with or without the template-similarity feature. Settings are fixed in
PREREGISTRATION.md; there is no tuning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from triage_kit.detectors.features import FEATURE_NAMES, account_features
from triage_kit.detectors.similarity import SimilarityIndex
from triage_kit.schema import AccountEvents

VARIANTS = ("lr", "lr+sim", "gb", "gb+sim")


def _make(kind: str) -> Any:
    if kind == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
        )
    return HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_leaf_nodes=15, early_stopping=False, random_state=0
    )


class FeatureModel:
    def __init__(self, variant: str) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
        self.name = variant
        self.kind = variant.split("+")[0]
        self.use_sim = variant.endswith("+sim")
        self.columns: list[str] = []
        self.model: Any = None
        self.index: SimilarityIndex | None = None
        self._mu = np.zeros(0)
        self._sd = np.ones(0)

    def fit_table(
        self,
        X: np.ndarray,
        y: Sequence[int] | np.ndarray,
        names: Sequence[str],
        reference: Sequence[AccountEvents] | None = None,
    ) -> FeatureModel:
        """Fit on a precomputed feature table. `reference` is the unlabeled population
        used to compute template_sim for accounts scored later via `score`."""
        self.columns = [n for n in names if self.use_sim or n != "template_sim"]
        self._cols = [list(names).index(c) for c in self.columns]
        Xs = X[:, self._cols]
        self.model = _make(self.kind).fit(Xs, np.asarray(y))
        self._mu, self._sd = Xs.mean(axis=0), Xs.std(axis=0) + 1e-9
        if self.use_sim and reference is not None:
            self.index = SimilarityIndex(reference)
        return self

    def predict_table(self, X: np.ndarray) -> np.ndarray:
        p: np.ndarray = self.model.predict_proba(X[:, self._cols])[:, 1]
        return p

    def _row(self, ae: AccountEvents) -> np.ndarray:
        f = account_features(ae)
        vals = [f[n] for n in FEATURE_NAMES]
        if "template_sim" in self.columns:
            if self.index is None:
                raise RuntimeError("+sim model needs a reference population to score accounts")
            vals.append(self.index.value(ae))
        full = list(FEATURE_NAMES) + (["template_sim"] if "template_sim" in self.columns else [])
        return np.array([vals[full.index(c)] for c in self.columns], dtype=float)

    def score(self, account_events: AccountEvents) -> tuple[float, str]:
        row = self._row(account_events)
        p = float(self.model.predict_proba(row[None, :])[0, 1])
        z = (row - self._mu) / self._sd
        if isinstance(self.model, Pipeline):  # LR: contribution = coef * standardized value
            coef = self.model[-1].coef_[0]
            contrib = coef * z
        else:
            contrib = np.abs(z)
        top = np.argsort(-contrib)[:3]
        parts = ", ".join(f"{self.columns[i]}={row[i]:.3g} (z={z[i]:+.1f})" for i in top)
        return p, f"{self.name} p={p:.3f}; most influential: {parts}"
