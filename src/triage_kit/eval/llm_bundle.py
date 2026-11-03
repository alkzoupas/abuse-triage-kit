"""Metrics for a real-reviewer results bundle (API pass or Claude Code subagent pass).

A bundle holds only scores keyed by account id. The report needs per-family PR-AUC on
the in-family test cells, computed exactly as `run_eval` computes every other cell, so
it is computed here, when the bundle is written, with the run's own labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from triage_kit.eval.run_eval import RunConfig, _cell_metrics
from triage_kit.eval.splits import Split, cell_ids, unit_of


def bundle_metrics(
    cfg: RunConfig,
    split: Split,
    labels: Mapping[str, Mapping[str, Any]],
    scores: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Per-family metrics on `split`'s test cells, over the accounts present in `scores`.

    A family cell keeps only scored accounts, so a partial pass (``--limit``) is measured
    on the subset it scored; the report says so.
    """
    out: dict[str, dict[str, Any]] = {}
    fams = list(split.test_families)
    for ci, fam in enumerate(fams):
        ids = [a for a in cell_ids(split, fam, labels) if a in scores]
        y = np.array([int(labels[a]["family"] == fam) for a in ids])
        if len(ids) == 0 or y.sum() == 0:
            out[fam] = {"n": len(ids), "n_pos": int(y.sum()), "ap": None}
            continue
        s = np.array([float(scores[a]) for a in ids])
        units = np.array([unit_of(labels[a]) for a in ids])
        out[fam] = _cell_metrics(cfg, 50_000 + ci, split.name, fam, "llm", y, s, units, 0.05)
    return out
