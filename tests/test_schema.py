import json

import pytest

from triage_kit.schema import (
    EVENT_TYPES,
    Account,
    AccountEvents,
    Event,
    SchemaError,
    account_schema,
    event_schema,
    group_events,
    read_dataset,
    validate_account,
    validate_event,
    write_dataset,
)


def _acct() -> Account:
    return Account(account_id="a1", created_at=100, country="DE", email_domain="mail.example")


def _ev(ts: int, type_: str = "login", **kw: object) -> Event:
    return Event(account_id="a1", ts=ts, type=type_, ip="10.0.0.1", device_id="d1", **kw)  # type: ignore[arg-type]


def test_schemas_are_valid_json_schema_documents() -> None:
    for sch in (account_schema(), event_schema()):
        assert sch["type"] == "object"
        assert "required" in sch
    assert set(event_schema()["properties"]["type"]["enum"]) == set(EVENT_TYPES)


def test_round_trip_dict() -> None:
    e = _ev(5, "payment_attempt", amount=3.5, success=False, target_id="L1")
    assert Event.from_dict(e.to_dict()) == e
    a = _acct()
    assert Account.from_dict(a.to_dict()) == a


def test_validate_rejects_unknown_type_and_missing_field() -> None:
    bad = _ev(5).to_dict()
    bad["type"] = "teleport"
    with pytest.raises(SchemaError, match="type"):
        validate_event(bad)
    bad2 = _acct().to_dict()
    del bad2["created_at"]
    with pytest.raises(SchemaError, match="created_at"):
        validate_account(bad2)


def test_rating_bounds() -> None:
    bad = _ev(5, "review", rating=9, target_id="L1").to_dict()
    with pytest.raises(SchemaError):
        validate_event(bad)


def test_group_events_sorts_by_time_and_keeps_accounts_without_events() -> None:
    a2 = Account(account_id="a2", created_at=1, country="FR", email_domain="x.example")
    evs = [_ev(30), _ev(10), _ev(20)]
    grouped = group_events([_acct(), a2], evs)
    assert [e.ts for e in grouped[0].events] == [10, 20, 30]
    assert grouped[1].events == ()
    assert isinstance(grouped[0], AccountEvents)


def test_write_read_dataset_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    accts = [_acct()]
    evs = [_ev(10), _ev(20, "view", target_id="L2")]
    write_dataset(tmp_path, accts, evs)
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["ts"] == 10
    got = read_dataset(tmp_path)
    assert got[0].account == accts[0]
    assert list(got[0].events) == evs
