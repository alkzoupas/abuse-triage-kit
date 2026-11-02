"""Full evaluation entry point.

    python -m triage_kit.eval.run_eval --seed 101 --s 1.0

Generates a dataset, writes it to data/, reads it back through the public schema
(the detectors' only view), trains and scores every detector on every split, and
writes results/<run>/metrics.json (deterministic for a given seed).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triage_kit.detectors.feature_model import (
    ALL_VARIANTS,
    POSTHOC_VARIANTS,
    VARIANTS,
    FeatureModel,
)
from triage_kit.detectors.features import FEATURE_NAMES, feature_table
from triage_kit.detectors.llm.mock import MockLLM
from triage_kit.detectors.llm.reviewer import LLMReviewer
from triage_kit.detectors.rules import RulesDetector
from triage_kit.eval.metrics import (
    average_precision,
    base_rate,
    bootstrap_metrics,
    cost_per_1000,
    expanded_interval,
    load_prices,
    null_expected_ap,
    percentile_interval,
    precision_recall_at_k,
    roc_auc,
)
from triage_kit.eval.oracle import oracle_scores
from triage_kit.eval.splits import LeakageError, Split, cell_ids, make_splits, unit_of
from triage_kit.eval.truth import load_truth
from triage_kit.generator import GeneratorConfig, generate, health_check, write_generated
from triage_kit.schema import AccountEvents, read_dataset

DETECTORS: tuple[str, ...] = ("rules", *VARIANTS, *POSTHOC_VARIANTS, "llm:mock", "oracle")
ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class RunConfig:
    seed: int
    s: float
    n_accounts: int = 5000
    n_boot: int = 1000
    k: int = 50
    data_root: Path = field(default=Path("data"))
    results_root: Path = field(default=Path("results"))
    prices: Path = field(default=ROOT / "config" / "prices.yaml")

    @property
    def name(self) -> str:
        return f"seed{self.seed}_s{self.s:.2f}_n{self.n_accounts}"


def _r(x: float) -> float | None:
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 6)


def _assert_train_ok(split: Split, train_ids: Sequence[str], labels: dict[str, Any]) -> None:
    fams = {labels[a]["family"] for a in train_ids}
    if split.name != "in_family" and fams & set(split.test_families):
        raise LeakageError(
            f"{split.name}: training on held-out family {fams & set(split.test_families)}"
        )
    if set(train_ids) & split.test_ids:
        raise LeakageError(f"{split.name}: training accounts overlap test accounts")


def score_mock_llm(
    test_data: Sequence[AccountEvents],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """MockLLM verdicts for the given accounts. Takes events only: no labels exist here."""
    reviewer = LLMReviewer(MockLLM())
    scores = {ae.account_id: reviewer.score(ae)[0] for ae in test_data}
    calls = [asdict(c) for c in reviewer.calls]
    return scores, calls


def run_eval(cfg: RunConfig, write: bool = True) -> dict[str, Any]:
    data_dir = cfg.data_root / cfg.name
    g = generate(GeneratorConfig(seed=cfg.seed, s=cfg.s, n_accounts=cfg.n_accounts))
    problems = health_check(g)
    if problems:
        raise RuntimeError("generator health check failed: " + "; ".join(problems))
    write_generated(g, data_dir)
    del g
    data = read_dataset(data_dir)  # the detectors' view: schema only
    truth = load_truth(data_dir)  # eval-only view
    return evaluate(cfg, data, truth.labels, truth.meta, write=write)


def evaluate(
    cfg: RunConfig,
    data: list[AccountEvents],
    labels: dict[str, dict[str, Any]],
    meta: dict[str, Any],
    write: bool = True,
) -> dict[str, Any]:
    by_id = {ae.account_id: ae for ae in data}
    pos = {ae.account_id: i for i, ae in enumerate(data)}
    splits = make_splits([ae.account for ae in data], labels)
    test_ids = sorted(splits[0].test_ids)
    assert all(sp.test_ids == splits[0].test_ids for sp in splits)
    test_data = [by_id[a] for a in test_ids]
    te_idx = np.array([pos[a] for a in test_ids])

    _, X, names = feature_table(data, reference=data)  # template_sim uses events only
    n_base = len(FEATURE_NAMES)

    # untrained detectors: identical across splits
    llm_scores, llm_calls = score_mock_llm(test_data)
    oracle: dict[str, dict[str, float]] = {}
    for fam in meta["families"]:
        o = oracle_scores(fam, test_data, meta)
        oracle[fam] = dict(zip(test_ids, o.tolist(), strict=True))

    scores: dict[tuple[str, str], dict[str, float]] = {}
    for sp in splits:
        tr_ids = sorted(sp.train_ids)
        _assert_train_ok(sp, tr_ids, labels)
        tr_idx = np.array([pos[a] for a in tr_ids])
        y_tr = np.array([int(labels[a]["is_abuse"]) for a in tr_ids])
        for v in ALL_VARIANTS:
            m = FeatureModel(v).fit_table(X[tr_idx], y_tr, names)
            scores[(sp.name, v)] = dict(
                zip(test_ids, m.predict_table(X[te_idx]).tolist(), strict=True)
            )
        rules = RulesDetector(seen_families=sp.train_families)  # only seen-family rules
        rs = rules.score_many(test_data, X[te_idx, :n_base])
        scores[(sp.name, "rules")] = dict(zip(test_ids, rs.tolist(), strict=True))
        scores[(sp.name, "llm:mock")] = llm_scores

    cells_spec = [(sp, fam, det) for sp in splits for fam in sp.test_families for det in DETECTORS]
    n_cells = len(cells_spec)
    alpha_bonf = 0.05 / n_cells
    cells = []
    for ci, (sp, fam, det) in enumerate(cells_spec):
        ids = cell_ids(sp, fam, labels)
        y = np.array([int(labels[a]["family"] == fam) for a in ids])
        src = oracle[fam] if det == "oracle" else scores[(sp.name, det)]
        s = np.array([src[a] for a in ids])
        units = np.array([unit_of(labels[a]) for a in ids])
        cells.append(_cell_metrics(cfg, ci, sp.name, fam, det, y, s, units, alpha_bonf))

    pooled = []
    ids = test_ids
    y = np.array([int(labels[a]["is_abuse"]) for a in ids])
    units = np.array([unit_of(labels[a]) for a in ids])
    for di, det in enumerate(d for d in DETECTORS if d != "oracle"):
        s = np.array([scores[("in_family", det)][a] for a in ids])
        pooled.append(_cell_metrics(cfg, 10_000 + di, "in_family", "all", det, y, s, units, 0.05))

    prices = load_prices(cfg.prices)
    pm = prices["models"][prices["report_model"]]
    t_in = sum(c["tokens_in"] for c in llm_calls)
    t_out = sum(c["tokens_out"] for c in llm_calls)
    cost = {
        "n_accounts_scored": len(llm_calls),
        "llm_tokens_in_total": t_in,
        "llm_tokens_out_total": t_out,
        "llm_tokens_in_per_call": _r(t_in / max(1, len(llm_calls))),
        "llm_tokens_out_per_call": _r(t_out / max(1, len(llm_calls))),
        "llm_price_model": prices["report_model"],
        "llm_prices_placeholder": bool(prices["placeholder"]),
        "llm_cost_per_1000": _r(
            cost_per_1000(
                t_in, t_out, max(1, len(llm_calls)), pm["input_per_mtok"], pm["output_per_mtok"]
            )
        ),
        "rules_cost_per_1000": 0.0,
        "feature_model_cost_per_1000": 0.0,
    }
    test_groups = {
        fam: len({labels[a]["group_id"] for a in test_ids if labels[a]["family"] == fam})
        for fam in meta["families"]
        if meta["families"][fam]["coordinated"]
    }
    result = {
        "run": cfg.name,
        "config": {
            "seed": cfg.seed,
            "s": cfg.s,
            "n_accounts": cfg.n_accounts,
            "n_boot": cfg.n_boot,
            "k": cfg.k,
        },
        "detectors": list(DETECTORS),
        "n_cells": n_cells,
        "alpha_bonferroni": alpha_bonf,
        "n_train": {sp.name: len(sp.train_ids) for sp in splits},
        "n_test": len(test_ids),
        "test_groups": test_groups,
        "cells": cells,
        "pooled": pooled,
        "cost": cost,
    }
    if write:
        out = cfg.results_root / cfg.name
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(result, sort_keys=True, indent=1) + "\n")
        with (out / "llm_calls.jsonl").open("w") as f:
            for c in llm_calls:
                f.write(json.dumps(c, sort_keys=True) + "\n")
        with (out / "scores_in_family.jsonl").open("w") as f:
            for a in test_ids:
                row = {
                    "account_id": a,
                    **{d: _r(scores[("in_family", d)][a]) for d in DETECTORS if d != "oracle"},
                }
                f.write(json.dumps(row, sort_keys=True) + "\n")
    result["_llm_scores"] = llm_scores  # in-memory only (used by tests)
    return result


def _cell_metrics(
    cfg: RunConfig,
    ci: int,
    split: str,
    fam: str,
    det: str,
    y: np.ndarray,
    s: np.ndarray,
    units: np.ndarray,
    alpha_bonf: float,
) -> dict[str, Any]:
    p_k, r_k = precision_recall_at_k(y, s, cfg.k)
    rec: dict[str, Any] = {
        "split": split,
        "family": fam,
        "detector": det,
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "n_pos_units": int(len(set(units[y == 1]))),
        "base_rate": _r(base_rate(y)),
        "ap_random": _r(null_expected_ap(s, int(y.sum()))),  # null AP given this scorer's ties
        # degenerate: no positive scored above the lowest score, so AP == base rate and
        # the bootstrap cannot vary it (zero-width interval); a no-signal cell by construction
        "degenerate": bool(y.sum() > 0 and s[y == 1].max() <= s.min()),
        "ap": _r(average_precision(y, s)),
        "roc": _r(roc_auc(y, s)),
        "p_at_k": _r(p_k),
        "r_at_k": _r(r_k),
    }
    if cfg.n_boot:
        # AP: plain stratified cluster bootstrap. ROC-AUC: same, plus the continuity
        # correction (two pseudo-units per class). See PREREGISTRATION.md AMENDMENTS.
        bseed = cfg.seed * 100_000 + ci
        samples = {
            "ap": bootstrap_metrics(y, s, units, cfg.n_boot, seed=bseed, continuity=False)["ap"],
            "roc": bootstrap_metrics(y, s, units, cfg.n_boot, seed=bseed, continuity=True)["roc"],
        }
        # few-cluster correction: widen by the unit count of the scarcer class
        n_units = min(len(set(units[y == 1])), len(set(units[y == 0])))
        rec["n_units_scarce"] = n_units
        for key, arr in samples.items():
            for tag, a in (("ci95", 0.05), ("ci_bonf", alpha_bonf)):
                plain = percentile_interval(arr, a)
                rec[f"{key}_{tag}_plain"] = [_r(v) for v in plain]  # pre-registered method
                rec[f"{key}_{tag}"] = [
                    _r(v) for v in expanded_interval(rec[key], plain, n_units, a)
                ]
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--s", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    res = run_eval(RunConfig(seed=a.seed, s=a.s, n_accounts=a.n, n_boot=a.boot))
    print(f"wrote results/{res['run']}/metrics.json ({res['n_cells']} cells)")


if __name__ == "__main__":
    main()
