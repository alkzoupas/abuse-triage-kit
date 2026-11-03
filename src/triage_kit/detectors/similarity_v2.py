"""Template-similarity feature, version 2 (POST-HOC; see PREREGISTRATION.md AMENDMENTS).

Designed after v1 turned out too coarse, from a dev-seed diagnosis, knowing the
template-operator mechanism. Uses events only, never labels.

tokens   = 3-grams and 4-grams of "type|b" (b = min(20, floor(log2(1 + gap s))))
           plus "type:target" for view / review / purchase events
vocab    = tokens found in at least 2 and at most 1% of reference accounts
           (unique tokens can never match; common ones are generic behaviour)
sim(i,j) = exact Jaccard of the filtered token sets (sparse matrix product)
value    = mean of the 3 largest sim(i, j) over other accounts j, times n / (n + 5),
           n = the account's number of filtered tokens
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy import sparse

from triage_kit.schema import AccountEvents

TOP_K = 3
MAX_DF = 0.01
SHRINK = 5
TARGET_TYPES = ("view", "review", "purchase")


def v2_tokens(ae: AccountEvents) -> set[str]:
    seq = []
    prev: int | None = None
    out: set[str] = set()
    for e in ae.events:
        b = 0 if prev is None else min(20, int(math.floor(math.log2(1 + max(0, e.ts - prev)))))
        seq.append(f"{e.type}|{b}")
        prev = e.ts
        if e.type in TARGET_TYPES and e.target_id:
            out.add(f"{e.type}:{e.target_id}")
    for k in (3, 4):
        out.update(" ".join(seq[i : i + k]) for i in range(len(seq) - k + 1))
    return out


class SimilarityIndexV2:
    def __init__(self, reference: Sequence[AccountEvents]) -> None:
        self.ids = [ae.account_id for ae in reference]
        self.pos = {a: i for i, a in enumerate(self.ids)}
        toks = [v2_tokens(ae) for ae in reference]
        df = Counter(t for ts in toks for t in ts)
        max_df = max(2, int(MAX_DF * len(reference)))
        self.vocab = {
            t: i for i, t in enumerate(sorted(t for t, c in df.items() if 2 <= c <= max_df))
        }
        self.M = self._matrix(toks)
        self.n = np.asarray(self.M.sum(axis=1)).ravel()
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

    def _value_rows(self, Q: sparse.csr_matrix, nq: np.ndarray, self_idx: np.ndarray) -> np.ndarray:
        inter = (Q @ self.M.T).toarray()
        union = nq[:, None] + self.n[None, :] - inter
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        ok = self_idx >= 0
        jac[np.nonzero(ok)[0], self_idx[ok]] = 0.0
        k = min(TOP_K, jac.shape[1])
        top = -np.sort(-jac, axis=1)[:, :k] if k else np.zeros((len(nq), 1))
        return top.mean(axis=1) * nq / (nq + SHRINK)

    def all_values(self) -> np.ndarray:
        if self._cache is None:
            out = np.zeros(len(self.ids))
            for start in range(0, len(self.ids), 500):
                stop = min(start + 500, len(self.ids))
                out[start:stop] = self._value_rows(
                    self.M[start:stop], self.n[start:stop], np.arange(start, stop)
                )
            self._cache = out
        return self._cache

    def value(self, ae: AccountEvents) -> float:
        i = self.pos.get(ae.account_id)
        if i is not None:
            return float(self.all_values()[i])
        Q = self._matrix([v2_tokens(ae)])
        nq = np.asarray(Q.sum(axis=1)).ravel()
        return float(self._value_rows(Q, nq, np.array([-1]))[0])
