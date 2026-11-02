from pathlib import Path

import pytest

from triage_kit.detectors.rules import RuleError, RulesDetector, load_rules
from triage_kit.schema import Account, AccountEvents, Event

FAMS = ("ato", "review_ring", "promo_abuse", "scraping", "card_testing", "template_operator")


def _ae(events: list[Event], domain: str = "mail.example") -> AccountEvents:
    return AccountEvents(Account("a1", 1000, "DE", domain), tuple(events))


def _e(ts: int, type_: str, **kw: object) -> Event:
    return Event("a1", ts, type_, "10.0.0.1", "d1", **kw)  # type: ignore[arg-type]


def test_default_rules_are_tagged_and_readable() -> None:
    rules = load_rules()
    assert {r.family for r in rules} == set(FAMS)
    for r in rules:
        assert r.says and 0 < r.weight <= 1


def test_rules_fire_and_explain() -> None:
    det = RulesDetector(seen_families=FAMS)
    fails = [_e(2000 + 10 * i, "login_fail", success=False) for i in range(12)]
    score, why = det.score(_ae(fails))
    assert score > 0.5 and "failed_login_burst" in why
    score0, why0 = det.score(_ae([_e(2000, "login", success=True)]))
    assert score0 == 0.0 and "no rule fired" in why0


def test_only_seen_family_rules_are_active() -> None:
    det = RulesDetector(seen_families=("scraping",))
    fails = [_e(2000 + 10 * i, "login_fail", success=False) for i in range(12)]
    assert det.score(_ae(fails))[0] == 0.0
    assert {r.family for r in det.rules} == {"scraping"}


def test_malformed_rule_file_is_rejected(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text(
        "rules:\n  - id: x\n    family: ato\n    says: hi\n"
        "    if: {feature: nope, at_least: 1}\n    weight: 0.5\n"
    )
    with pytest.raises(RuleError, match="nope"):
        load_rules(p)
