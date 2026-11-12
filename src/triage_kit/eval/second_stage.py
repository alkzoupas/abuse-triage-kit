"""Second-stage recall: what a reviewer adds after the feature model has flagged its budget.

See PREREGISTRATION.md, AMENDMENTS (2026-11-09). Exploratory.

    python -m triage_kit.eval.second_stage --seed 9001 --s 0.5 --model opus

Stage 1 (`gb+sim`) flags its top k accounts in a cell. Stage 2 ranks only the accounts stage 1
did not flag and flags its own top k. Added recall = family accounts stage 2 catches / all
family accounts in the cell. Reads the subagent bundles written by `subagent_pass collect`
(prompt v1 and v2) and writes results/<run>/second_stage.json.

This is a retrospective calculation on fixed scores. The reviewer scored every test account
once, in mixed batches, before any stage-1 filtering; the same score per account is reused in
every cell and stage 1's flags are removed afterwards, in the ranking. That equals a reviewer
that is only shown stage 1's survivors only if a score does not depend on the other accounts
in its batch, which was not tested. It does not measure reduced reviewer work or a deployed
routed review.

Intervals resample accounts and hold every score fixed: they are conditional on this split,
these fitted models, this batch assignment and these replies, and do not cover a rerun of the
reviewer or a rebatching.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from triage_kit.detectors.feature_model import FeatureModel
from triage_kit.detectors.features import FEATURE_NAMES, feature_table
from triage_kit.detectors.rules import RulesDetector
from triage_kit.eval.metrics import expansion_factor, percentile_interval
from triage_kit.eval.run_eval import RunConfig
from triage_kit.eval.splits import HELDOUT_FOLDS, Split, cell_ids, make_splits, unit_of
from triage_kit.eval.truth import load_truth
from triage_kit.generator import FAMILIES, GeneratorConfig, generate, write_generated
from triage_kit.schema import AccountEvents, read_dataset

STAGE1 = "gb+sim"
Arr = npt.NDArray[np.float64]


def take_top_k(s: npt.ArrayLike, w: npt.ArrayLike, k: float) -> Arr:
    """Share of each account taken when flagging the top `k` (by weight) of scores `s`.

    A block of tied scores that straddles the budget is taken fractionally: for one stage,
    the expected value under random tie-breaking. `w` are multiplicities (1 for the point
    estimate, bootstrap counts otherwise; stage 2 passes what stage 1 left)."""
    sa, wa = np.asarray(s, dtype=float), np.asarray(w, dtype=float)
    order = np.argsort(-sa, kind="mergesort")
    ss, ws = sa[order], wa[order]
    starts = np.r_[0, np.nonzero(np.diff(ss))[0] + 1]
    bw = np.add.reduceat(ws, starts)
    before = np.cumsum(bw) - bw
    frac = np.zeros(len(bw))
    ok = bw > 0
    frac[ok] = np.clip((k - before[ok]) / bw[ok], 0.0, 1.0)
    block = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(ss)]))
    out = np.empty(len(sa))
    out[order] = frac[block]
    return out


def two_stage(
    y: npt.ArrayLike, s1: npt.ArrayLike, s2: npt.ArrayLike, w: npt.ArrayLike, k: float
) -> dict[str, float]:
    """Stage-1 recall at k, and the recall and precision of stage 2's k flags among the
    accounts stage 1 allowed.

    Stage 2 is allocated against the fractional weights stage 1 leaves. That is a defined
    surrogate: it equals the expectation under sequential random tie-breaking when stage 1
    has no tie at its budget, and can differ when it has one (1/3 against 1/4 in
    tests/test_second_stage.py)."""
    ya, wa = np.asarray(y, dtype=float), np.asarray(w, dtype=float)
    t1 = take_top_k(s1, wa, k)
    w2 = wa * (1.0 - t1)
    flagged2 = w2 * take_top_k(s2, w2, k)
    n_pos = float(np.sum(wa * ya))
    hits2 = float(np.sum(flagged2 * ya))
    n2 = float(np.sum(flagged2))
    nan = float("nan")
    return {
        "stage1_recall": float(np.sum(wa * t1 * ya)) / n_pos if n_pos else nan,
        "added_recall": hits2 / n_pos if n_pos else nan,
        "stage2_precision": hits2 / n2 if n2 else nan,
    }


def _r(x: float) -> float | None:
    return None if np.isnan(x) else round(float(x), 6)


def _widened(est: float, samples: Arr, n_units: int) -> tuple[list[float | None], ...]:
    """Percentile 95% interval, and the same widened for few clusters (not clipped)."""
    lo, hi = percentile_interval(samples, 0.05)
    f = expansion_factor(n_units, 0.05)
    wide = (min(lo, est - f * max(est - lo, 0.0)), max(hi, est + f * max(hi - est, 0.0)))
    return [_r(lo), _r(hi)], [_r(wide[0]), _r(wide[1])]


def cell_second_stage(
    y: npt.ArrayLike,
    s1: npt.ArrayLike,
    second: Mapping[str, npt.ArrayLike],
    units: Sequence[str] | npt.NDArray[Any],
    k: int,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """One cell: every second stage in `second` plus `random`, with bootstrap intervals.

    Stratified cluster bootstrap: abuse units and benign units are resampled separately and
    the whole two-stage procedure is recomputed with the resampling weights."""
    ya = np.asarray(y, dtype=int)
    s1a = np.asarray(s1, dtype=float)
    scorers = {name: np.asarray(v, dtype=float) for name, v in second.items()}
    scorers["random"] = np.zeros(len(ya))  # one tie block: the exact expectation
    ones = np.ones(len(ya))
    point = {name: two_stage(ya, s1a, s2, ones, k) for name, s2 in scorers.items()}
    t1 = take_top_k(s1a, ones, k)
    boundary_tie = bool(np.any((t1 > 0.0) & (t1 < 1.0)))  # point estimate only

    uniq, inv = np.unique(np.asarray(units).astype(str), return_inverse=True)
    unit_pos = np.zeros(len(uniq), dtype=bool)
    unit_pos[inv[ya == 1]] = True
    pos_u, neg_u = np.nonzero(unit_pos)[0], np.nonzero(~unit_pos)[0]
    rng = np.random.default_rng(seed)
    added = {name: np.empty(n_boot) for name in scorers}
    stage1 = np.empty(n_boot)
    for b in range(n_boot):
        uw = np.bincount(rng.choice(pos_u, size=len(pos_u)), minlength=len(uniq)).astype(float)
        uw += np.bincount(rng.choice(neg_u, size=len(neg_u)), minlength=len(uniq))
        w = uw[inv]
        for name, s2 in scorers.items():
            r = two_stage(ya, s1a, s2, w, k)
            added[name][b] = r["added_recall"]
        stage1[b] = r["stage1_recall"]

    n_units = min(len(pos_u), len(neg_u))
    s1_point = point["random"]["stage1_recall"]
    out: dict[str, Any] = {
        "n": len(ya),
        "n_pos": int(ya.sum()),
        "n_pos_units": len(pos_u),
        "k": k,
        "stage1_recall": _r(s1_point),
        "stage1_recall_ci95": _widened(s1_point, stage1, n_units)[1],
        "stage1_boundary_tie": boundary_tie,
        "stage2": {},
    }
    for name in scorers:
        est = point[name]["added_recall"]
        diff = est - point["random"]["added_recall"]
        plain, wide = _widened(est, added[name], n_units)
        out["stage2"][name] = {
            "added_recall": _r(est),
            "added_recall_ci95_plain": plain,
            "added_recall_ci95": wide,
            "stage2_precision": _r(point[name]["stage2_precision"]),
            "diff_vs_random": _r(diff),
            "diff_vs_random_ci95": _widened(diff, added[name] - added["random"], n_units)[1],
        }
    return out


def bundle_path(results_dir: Path, model: str, version: str) -> Path:
    suffix = "" if version == "v1" else f"_{version}"
    return results_dir / f"llm_bundle_subagent_{model.replace('/', '_')}{suffix}.json"


class Stage1(NamedTuple):
    """The run's data and `gb+sim` scores for every test account, per split."""

    cfg: RunConfig
    data: list[AccountEvents]
    labels: Mapping[str, Mapping[str, Any]]
    splits: dict[str, Split]
    test_ids: list[str]
    scores: dict[str, dict[str, float]]
    X: npt.NDArray[Any]
    te_idx: npt.NDArray[Any]


def fit_stage1(seed: int, s: float, n: int, n_boot: int = 1000) -> Stage1:
    cfg = RunConfig(seed=seed, s=s, n_accounts=n, n_boot=n_boot)
    data_dir = cfg.data_root / cfg.name
    if not (data_dir / "events.jsonl").exists():
        write_generated(generate(GeneratorConfig(seed=seed, s=s, n_accounts=n)), data_dir)
    data = read_dataset(data_dir)
    labels = load_truth(data_dir).labels
    splits = {sp.name: sp for sp in make_splits([ae.account for ae in data], labels)}
    pos = {ae.account_id: i for i, ae in enumerate(data)}
    test_ids = sorted(splits["in_family"].test_ids)
    te_idx = np.array([pos[a] for a in test_ids])
    _, X, names = feature_table(data, reference=data)

    stage1: dict[str, dict[str, float]] = {}
    for name, sp in splits.items():
        tr_idx = np.array([pos[a] for a in sorted(sp.train_ids)])
        y_tr = np.array([int(labels[a]["is_abuse"]) for a in sorted(sp.train_ids)])
        m = FeatureModel(STAGE1).fit_table(X[tr_idx], y_tr, names)
        stage1[name] = dict(zip(test_ids, m.predict_table(X[te_idx]).tolist(), strict=True))
    return Stage1(cfg, data, labels, splits, test_ids, stage1, X, te_idx)


def heldout_split_of() -> dict[str, str]:
    """The held-out split that hides each family."""
    return {fam: f"heldout_{i}" for i, fold in enumerate(HELDOUT_FOLDS, 1) for fam in fold}


def run(seed: int, s: float, n: int, model: str, n_boot: int = 1000) -> Path:
    cfg, data, labels, splits, test_ids, stage1, X, te_idx = fit_stage1(seed, s, n, n_boot)
    pos = {ae.account_id: i for i, ae in enumerate(data)}
    rules = RulesDetector(seen_families=FAMILIES)  # every rule on, whatever the fold
    test_data = [data[pos[a]] for a in test_ids]
    rules_all = dict(
        zip(test_ids, rules.score_many(test_data, X[te_idx, : len(FEATURE_NAMES)]), strict=True)
    )

    out_dir = cfg.results_root / cfg.name
    llm: dict[str, dict[str, float]] = {}
    for version in ("v2", "v1"):
        p = bundle_path(out_dir, model, version)
        if p.exists():
            llm[f"llm:{version}"] = json.loads(p.read_text())["scores"]

    heldout_of = heldout_split_of()
    cells = []
    for cond in ("heldout", "in_family"):
        for fi, fam in enumerate(FAMILIES):
            sp = splits["in_family" if cond == "in_family" else heldout_of[fam]]
            ids = cell_ids(sp, fam, labels)
            y = np.array([int(labels[a]["family"] == fam) for a in ids])
            s1 = np.array([stage1[sp.name][a] for a in ids])
            second: dict[str, npt.ArrayLike] = {
                name: np.array([float(sc.get(a, 0.5)) for a in ids]) for name, sc in llm.items()
            }
            second["rules:all"] = np.array([float(rules_all[a]) for a in ids])
            second["more-stage-1"] = s1
            units = np.array([unit_of(labels[a]) for a in ids])
            bseed = seed * 100_000 + 70_000 + (0 if cond == "heldout" else 100) + fi
            c = cell_second_stage(y, s1, second, units, cfg.k, n_boot, bseed)
            cells.append({"family": fam, "condition": cond, "split": sp.name, **c})
    result = {
        "run": cfg.name,
        "model": model,
        "stage1": STAGE1,
        "k": cfg.k,
        "n_boot": n_boot,
        "llm_present": sorted(llm),
        "cells": cells,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "second_stage.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--model", default="unknown")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(argv)
    print(f"wrote {run(a.seed, a.s, a.n, a.model, a.boot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
