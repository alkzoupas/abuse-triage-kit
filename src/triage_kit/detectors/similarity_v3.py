"""Template-similarity feature, version 3 (POST-HOC; see PREREGISTRATION.md AMENDMENTS).

Designed after v1 and v2 found nothing when template scripts use only ordinary actions:
Jaccard over whole accounts is diluted by ordinary history. v3 counts shared rare
evidence instead of a ratio. Designed knowing the template-operator mechanism. Uses
events only, never labels.

tokens       = ordered view-target pairs "x>y" (x != y, y one to three views after x,
               view events only, in time order) plus "q:<text>" for search queries
vocab        = tokens found in at least 2 and at most max(2, 0.2% of N) reference accounts
shared(i, j) = number of vocabulary tokens both accounts have
value        = log(1 + mean of the 3 largest shared(i, j) over other accounts j)
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from triage_kit.schema import AccountEvents

TOP_K = 3
WINDOW = 3
MAX_DF = 0.002


def v3_tokens(ae: AccountEvents) -> set[str]:
    views = [e.target_id for e in ae.events if e.type == "view" and e.target_id]
    out = {f"q:{e.text}" for e in ae.events if e.type == "search" and e.text}
    for i, x in enumerate(views):
        for y in views[i + 1 : i + 1 + WINDOW]:
            if y != x:
                out.add(f"{x}>{y}")
    return out


class SimilarityIndexV3:
    def __init__(self, reference: Sequence[AccountEvents]) -> None:
        self.ids = [ae.account_id for ae in reference]
        self.pos = {a: i for i, a in enumerate(self.ids)}
        toks = [v3_tokens(ae) for ae in reference]
        df = Counter(t for ts in toks for t in ts)
        max_df = max(2, int(MAX_DF * len(reference)))
        self.vocab = {
            t: i for i, t in enumerate(sorted(t for t, c in df.items() if 2 <= c <= max_df))
        }
        self.M = self._matrix(toks)
        self._cache: np.ndarray | None = None

    def _matrix(self, toks: Sequence[set[str]]) -> sparse.csr_matrix:
        rows, cols = [], []
        for i, ts in enumerate(toks):
            for t in ts:
                j = self.vocab.get(t)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        data = np.ones(len(rows), dtype=np.float64)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(toks), len(self.vocab)))

    def _value_rows(self, Q: sparse.csr_matrix, self_idx: np.ndarray) -> np.ndarray:
        shared = (Q @ self.M.T).toarray()
        ok = self_idx >= 0
        shared[np.nonzero(ok)[0], self_idx[ok]] = 0.0
        k = min(TOP_K, shared.shape[1])
        top = -np.sort(-shared, axis=1)[:, :k] if k else np.zeros((Q.shape[0], 1))
        return np.log1p(top.mean(axis=1))

    def all_values(self) -> np.ndarray:
        if self._cache is None:
            out = np.zeros(len(self.ids))
            for start in range(0, len(self.ids), 500):
                stop = min(start + 500, len(self.ids))
                out[start:stop] = self._value_rows(self.M[start:stop], np.arange(start, stop))
            self._cache = out
        return self._cache

    def value(self, ae: AccountEvents) -> float:
        i = self.pos.get(ae.account_id)
        if i is not None:
            return float(self.all_values()[i])
        return float(self._value_rows(self._matrix([v3_tokens(ae)]), np.array([-1]))[0])
