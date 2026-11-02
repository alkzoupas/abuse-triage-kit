"""Template-similarity feature (declared in PREREGISTRATION.md).

tokens   = "type|b" per event, b = min(20, floor(log2(1 + gap seconds))), 0 for the first
shingles = set of consecutive 4-token windows
MinHash  = 64 seeded hash functions (fixed salt)
value    = max over other accounts in the reference population of estimated Jaccard,
           times n / (n + 20) with n = number of distinct shingles

Designed knowing the template-operator mechanism. Uses events only, never labels.
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Sequence

import numpy as np

from triage_kit.schema import AccountEvents

N_HASH = 64
SHINGLE = 4
SHRINK = 20
_P = np.uint64(2**31 - 1)
_rng = np.random.default_rng(20261102)  # fixed salt
_A = _rng.integers(1, 2**31 - 1, size=N_HASH, dtype=np.uint64)
_B = _rng.integers(0, 2**31 - 1, size=N_HASH, dtype=np.uint64)
_EMPTY = np.uint64(2**32)


def tokens(ae: AccountEvents) -> list[str]:
    out = []
    prev: int | None = None
    for e in ae.events:
        b = 0 if prev is None else min(20, int(math.floor(math.log2(1 + max(0, e.ts - prev)))))
        out.append(f"{e.type}|{b}")
        prev = e.ts
    return out


def shingles(ae: AccountEvents) -> set[int]:
    t = tokens(ae)
    return {zlib.crc32(" ".join(t[i : i + SHINGLE]).encode()) for i in range(len(t) - SHINGLE + 1)}


def signature(sh: set[int]) -> np.ndarray:
    if not sh:
        return np.full(N_HASH, _EMPTY, dtype=np.uint64)
    x = np.fromiter(sh, dtype=np.uint64, count=len(sh)) % _P
    return ((_A[:, None] * x[None, :] + _B[:, None]) % _P).min(axis=1)


class SimilarityIndex:
    """MinHash signatures for a reference population of accounts."""

    def __init__(self, reference: Sequence[AccountEvents]) -> None:
        self.ids = [ae.account_id for ae in reference]
        self.pos = {a: i for i, a in enumerate(self.ids)}
        sh = [shingles(ae) for ae in reference]
        self.n = np.array([len(s) for s in sh], dtype=float)
        self.sig = np.stack([signature(s) for s in sh]) if sh else np.zeros((0, N_HASH), np.uint64)
        self.valid = self.n > 0
        self._cache: np.ndarray | None = None

    def _max_sim(self, sig: np.ndarray, exclude: np.ndarray) -> np.ndarray:
        """sig: (b, N_HASH); exclude: (b,) row index into the reference to ignore (or -1)."""
        eq = (sig[:, None, :] == self.sig[None, :, :]).mean(axis=2)
        eq[:, ~self.valid] = 0.0
        rows = np.arange(len(sig))
        ok = exclude >= 0
        eq[rows[ok], exclude[ok]] = 0.0
        return eq.max(axis=1) if eq.shape[1] else np.zeros(len(sig))

    def all_values(self) -> np.ndarray:
        """Feature value for every reference account (cached)."""
        if self._cache is None:
            out = np.zeros(len(self.ids))
            for start in range(0, len(self.ids), 200):
                stop = min(start + 200, len(self.ids))
                idx = np.arange(start, stop)
                out[start:stop] = self._max_sim(self.sig[start:stop], idx)
            out[~self.valid] = 0.0
            self._cache = out * self.n / (self.n + SHRINK)
        return self._cache

    def value(self, ae: AccountEvents) -> float:
        i = self.pos.get(ae.account_id)
        if i is not None:
            return float(self.all_values()[i])
        sh = shingles(ae)
        if not sh:
            return 0.0
        m = self._max_sim(signature(sh)[None, :], np.array([-1]))[0]
        return float(m * len(sh) / (len(sh) + SHRINK))
