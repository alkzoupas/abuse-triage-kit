import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from triage_kit.generator import FAMILIES, GeneratorConfig, generate, health_check, write_generated
from triage_kit.schema import read_dataset, validate_account, validate_event

SMALL = dict(n_accounts=1500)


def _digest(directory: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(directory).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_six_families_including_template_operator_with_documented_mechanisms() -> None:
    assert len(FAMILIES) == 6
    assert "template_operator" in FAMILIES
    allowed = {"text", "counts/velocity", "timing/sequence", "cross-account", "account attributes"}
    for fam in FAMILIES.values():
        assert len(fam.mechanism) > 80, fam.name
        assert fam.channels and set(fam.channels) <= allowed
    assert FAMILIES["template_operator"].coordinated
    assert FAMILIES["review_ring"].coordinated


def test_same_seed_gives_byte_identical_files(tmp_path: Path) -> None:
    for name in ("a", "b"):
        write_generated(generate(GeneratorConfig(seed=7, s=1.0, **SMALL)), tmp_path / name)
    assert _digest(tmp_path / "a") == _digest(tmp_path / "b")
    write_generated(generate(GeneratorConfig(seed=8, s=1.0, **SMALL)), tmp_path / "c")
    assert _digest(tmp_path / "a") != _digest(tmp_path / "c")


def test_truth_is_written_separately_from_events(tmp_path: Path) -> None:
    write_generated(generate(GeneratorConfig(seed=3, s=1.0, **SMALL)), tmp_path)
    assert (tmp_path / "truth" / "labels.jsonl").exists()
    assert (tmp_path / "truth" / "meta.json").exists()
    for fname in ("accounts.jsonl", "events.jsonl"):
        text = (tmp_path / fname).read_text()
        for word in ("family", "is_abuse", "group_id", "template_id", "tpl-", "grp-"):
            assert word not in text, (fname, word)
    data = read_dataset(tmp_path, validate=True)
    assert len(data) == 1500


def test_records_validate_against_json_schema() -> None:
    g = generate(GeneratorConfig(seed=4, s=1.0, **SMALL))
    for a in g.accounts[:200]:
        validate_account(a.to_dict())
    for e in g.events[:5000]:
        validate_event(e.to_dict())


def test_health_check_passes_at_both_extremes() -> None:
    for s in (0.0, 0.5, 1.0):
        g = generate(GeneratorConfig(seed=5, s=s, **SMALL))
        assert health_check(g) == []


def test_every_family_present_and_coordinated_families_have_groups() -> None:
    g = generate(GeneratorConfig(seed=5, s=1.0))
    fams = [lab["family"] for lab in g.labels]
    for name, fam in FAMILIES.items():
        assert fams.count(name) >= 50, name
        if fam.coordinated:
            groups = {lab["group_id"] for lab in g.labels if lab["family"] == name}
            assert len(groups) >= GeneratorConfig().min_groups, name


@pytest.mark.parametrize("s", [0.0, 1.0])
def test_signal_knob_changes_abuse_accounts_only_when_positive(s: float) -> None:
    g = generate(GeneratorConfig(seed=11, s=s))
    is_abuse = {lab["account_id"]: lab["is_abuse"] for lab in g.labels}
    n_events: dict[str, int] = {}
    for e in g.events:
        n_events[e.account_id] = n_events.get(e.account_id, 0) + 1
    ab = [n_events.get(a.account_id, 0) for a in g.accounts if is_abuse[a.account_id]]
    bn = [n_events.get(a.account_id, 0) for a in g.accounts if not is_abuse[a.account_id]]
    ratio = np.mean(ab) / np.mean(bn)
    if s == 0.0:
        assert 0.8 < ratio < 1.25  # drawn from the benign distribution
    else:
        assert ratio > 1.25  # bots / scrapers / card testers add activity


def test_account_ids_carry_no_order_information() -> None:
    g = generate(GeneratorConfig(seed=2, s=1.0, **SMALL))
    ids = [a.account_id for a in g.accounts]
    assert ids == sorted(ids)
    flags = [lab["is_abuse"] for lab in sorted(g.labels, key=lambda r: r["account_id"])]
    first, second = flags[: len(flags) // 2], flags[len(flags) // 2 :]
    assert abs(sum(first) - sum(second)) < 0.1 * len(flags)


def test_meta_contains_no_per_account_labels(tmp_path: Path) -> None:
    g = generate(GeneratorConfig(seed=2, s=1.0, **SMALL))
    write_generated(g, tmp_path)
    meta = json.loads((tmp_path / "truth" / "meta.json").read_text())
    assert meta["config"]["seed"] == 2
    assert set(meta["families"]) == set(FAMILIES)


def test_cross_only_template_mode_keeps_replays_and_benign_profile() -> None:
    from triage_kit.eval.oracle import oracle_scores
    from triage_kit.schema import group_events

    std = generate(GeneratorConfig(seed=13, s=0.5, **SMALL))
    xo = generate(GeneratorConfig(seed=13, s=0.5, template_mode="cross_only", **SMALL))
    assert health_check(xo) == []
    assert "template_mode" not in std.meta["config"]  # default output unchanged
    assert xo.meta["config"]["template_mode"] == "cross_only"
    # other families are untouched: identical events outside template_operator
    fam = {r["account_id"]: r["family"] for r in std.labels}
    key = lambda g: [e for e in g.events if fam[e.account_id] != "template_operator"]  # noqa: E731
    assert key(std) == key(xo)
    # every template account replays its script (oracle sees template targets in all of them)
    data = group_events(xo.accounts, xo.events)
    idx = [i for i, r in enumerate(xo.labels) if r["family"] == "template_operator"]
    o = oracle_scores("template_operator", [data[i] for i in idx], xo.meta)
    assert (o > 0).mean() > 0.95
    with pytest.raises(ValueError, match="template mode"):
        generate(GeneratorConfig(seed=1, template_mode="nope", **SMALL))


def test_ordinary_template_mode_uses_only_benign_session_actions() -> None:
    from triage_kit.eval.oracle import oracle_scores
    from triage_kit.schema import group_events

    std = generate(GeneratorConfig(seed=13, s=0.5, **SMALL))
    od = generate(GeneratorConfig(seed=13, s=0.5, template_mode="ordinary", **SMALL))
    assert health_check(od) == []
    assert od.meta["config"]["template_mode"] == "ordinary"
    fam = {r["account_id"]: r["family"] for r in std.labels}
    key = lambda g: [e for e in g.events if fam[e.account_id] != "template_operator"]  # noqa: E731
    assert key(std) == key(od)  # other families untouched
    # scripts contain only ordinary session actions
    for tpl in od.meta["families"]["template_operator"]["templates"].values():
        assert {st["type"] for st in tpl["steps"]} <= {"login", "search", "view"}
    # template accounts never take the unusual script actions more than benign accounts do
    tpl_ids = {a for a, f in fam.items() if f == "template_operator"}
    odd = {"review", "listing_create", "profile_update", "promo_redeem"}
    n_odd = sum(1 for e in od.events if e.account_id in tpl_ids and e.type in odd)
    n_odd_std = sum(1 for e in std.events if e.account_id in tpl_ids and e.type in odd)
    assert n_odd < n_odd_std
    # every account replays its crew's script (oracle sees script targets in nearly all)
    data = group_events(od.accounts, od.events)
    idx = [i for i, r in enumerate(od.labels) if r["family"] == "template_operator"]
    o = oracle_scores("template_operator", [data[i] for i in idx], od.meta)
    assert (o > 0).mean() > 0.9
