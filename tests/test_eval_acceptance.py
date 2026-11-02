"""Acceptance tests that run the full evaluation pipeline."""

import json
import random
from pathlib import Path

import pytest

from triage_kit.eval.run_eval import DETECTORS, RunConfig, evaluate, run_eval
from triage_kit.eval.splits import LeakageError
from triage_kit.eval.truth import load_truth
from triage_kit.schema import read_dataset

DEV_SEED = 101


@pytest.fixture(scope="module")
def null_run(tmp_path_factory: pytest.TempPathFactory) -> dict:  # type: ignore[type-arg]
    root = tmp_path_factory.mktemp("null")
    return run_eval(
        RunConfig(seed=DEV_SEED, s=0.0, data_root=root / "data", results_root=root / "results")
    )


def test_null_control_pooled_bonferroni(null_run: dict) -> None:  # type: ignore[type-arg]
    """At s = 0 no detector x family x split cell (oracle included) may exclude the base
    rate (PR-AUC) or 0.5 (ROC-AUC) with a Bonferroni-corrected interval."""
    cells = null_run["cells"]
    assert len(cells) == null_run["n_cells"] == len(DETECTORS) * 6 * 2
    assert null_run["alpha_bonferroni"] == pytest.approx(0.05 / len(cells))
    assert {c["detector"] for c in cells} >= {"oracle", "rules", "gb+sim", "llm:mock"}
    bad = []
    for c in cells:
        assert c["n_pos"] > 0 and c["n_pos"] < c["n"], c  # every cell must be testable
        lo, hi = c["ap_ci_bonf"]
        if not lo <= c["base_rate"] <= hi:
            bad.append(("ap", c["split"], c["family"], c["detector"], lo, c["base_rate"], hi))
        lo, hi = c["roc_ci_bonf"]
        if not lo <= 0.5 <= hi:
            bad.append(("roc", c["split"], c["family"], c["detector"], lo, hi))
    assert not bad, bad


def test_same_seed_gives_byte_identical_metrics(tmp_path: Path) -> None:
    outs = []
    for name in ("a", "b"):
        cfg = RunConfig(
            seed=7,
            s=1.0,
            n_accounts=1500,
            n_boot=100,
            data_root=tmp_path / name / "data",
            results_root=tmp_path / name / "res",
        )
        r = run_eval(cfg)
        outs.append((tmp_path / name / "res" / r["run"] / "metrics.json").read_bytes())
    assert outs[0] == outs[1]


def test_mock_llm_verdicts_identical_when_labels_shuffled(tmp_path: Path) -> None:
    cfg = RunConfig(
        seed=5,
        s=1.0,
        n_accounts=1500,
        n_boot=0,
        data_root=tmp_path / "data",
        results_root=tmp_path / "res",
    )
    run_eval(cfg, write=False)
    d = tmp_path / "data" / cfg.name
    data = read_dataset(d)
    truth = load_truth(d)
    a = evaluate(cfg, data, truth.labels, truth.meta, write=False)
    # shuffle every label field across accounts (families, groups, abuse flags)
    ids = list(truth.labels)
    vals = [dict(v) for v in truth.labels.values()]
    random.Random(0).shuffle(vals)
    shuffled = {i: {**v, "account_id": i} for i, v in zip(ids, vals, strict=True)}
    b = evaluate(cfg, data, shuffled, truth.meta, write=False)
    common = set(a["_llm_scores"]) & set(b["_llm_scores"])
    assert len(common) > 100
    assert all(a["_llm_scores"][x] == b["_llm_scores"][x] for x in common)


def test_heldout_training_is_asserted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import triage_kit.eval.run_eval as re_mod
    from triage_kit.eval.splits import Split

    real = re_mod.make_splits

    def leaky(accounts, labels):  # type: ignore[no-untyped-def]
        sp = real(accounts, labels)
        h = sp[1]
        victim = next(a for a in h.test_ids if labels[a]["family"] in h.test_families)
        other = next(a for a in sp[0].train_ids if labels[a]["family"] in h.test_families)
        sp[1] = Split(h.name, h.train_ids | {other}, h.test_ids, h.train_families, h.test_families)
        del victim
        return sp

    monkeypatch.setattr(re_mod, "make_splits", leaky)
    cfg = RunConfig(
        seed=5,
        s=1.0,
        n_accounts=1500,
        n_boot=0,
        data_root=tmp_path / "data",
        results_root=tmp_path / "res",
    )
    with pytest.raises(LeakageError, match="held-out"):
        run_eval(cfg, write=False)


def test_results_files_are_scores_and_metrics_only(tmp_path: Path) -> None:
    cfg = RunConfig(
        seed=3,
        s=1.0,
        n_accounts=1500,
        n_boot=0,
        data_root=tmp_path / "data",
        results_root=tmp_path / "res",
    )
    r = run_eval(cfg)
    m = json.loads((tmp_path / "res" / r["run"] / "metrics.json").read_text())
    assert m["cost"]["llm_prices_placeholder"] is True
    assert m["cost"]["llm_tokens_in_total"] > 0
