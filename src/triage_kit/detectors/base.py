"""The shared detector interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from triage_kit.schema import AccountEvents


@runtime_checkable
class Detector(Protocol):
    name: str

    def score(self, account_events: AccountEvents) -> tuple[float, str]:
        """Return (score in [0, 1], human-readable rationale)."""
        ...


def score_all(det: Detector, data: Sequence[AccountEvents]) -> np.ndarray:
    """Score many accounts. Detectors may provide a faster `score_many`."""
    fast = getattr(det, "score_many", None)
    if fast is not None:
        return np.asarray(fast(data), dtype=float)
    return np.array([det.score(ae)[0] for ae in data], dtype=float)
