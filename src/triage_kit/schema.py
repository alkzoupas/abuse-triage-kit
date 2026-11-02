"""Event schema: accounts plus timestamped events.

This is the ONLY data contract detectors see. Users plug their own data into the
same schema (see docs/byo-events.md). Labels and generator metadata are never
part of it; they live in a separate truth file read only by `triage_kit.eval`.

Times are integer Unix seconds (UTC).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from functools import cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema

EVENT_TYPES: tuple[str, ...] = (
    "login",  # successful login
    "login_fail",  # failed login attempt
    "view",  # viewed a listing (target_id = listing)
    "search",  # ran a search (text = query, optional)
    "payment_attempt",  # tried a payment (amount, success)
    "purchase",  # completed purchase (target_id, amount)
    "review",  # left a review (target_id, rating 1-5, text)
    "listing_create",  # created a listing (target_id, text, amount = price)
    "promo_redeem",  # redeemed a promo/referral code (target_id = code, success)
    "profile_update",  # changed profile details (e.g. email, password)
    "payout_change",  # changed payout destination
)


class SchemaError(ValueError):
    """Raised when a record does not match the schema. Message names the field."""


@dataclass(frozen=True, slots=True)
class Account:
    account_id: str
    created_at: int
    country: str
    email_domain: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Account:
        return Account(
            account_id=str(d["account_id"]),
            created_at=int(d["created_at"]),
            country=str(d["country"]),
            email_domain=str(d["email_domain"]),
        )


@dataclass(frozen=True, slots=True)
class Event:
    account_id: str
    ts: int
    type: str
    ip: str
    device_id: str
    target_id: str | None = None
    amount: float | None = None
    success: bool | None = None
    rating: int | None = None
    text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Event:
        amount = d.get("amount")
        rating = d.get("rating")
        success = d.get("success")
        return Event(
            account_id=str(d["account_id"]),
            ts=int(d["ts"]),
            type=str(d["type"]),
            ip=str(d["ip"]),
            device_id=str(d["device_id"]),
            target_id=None if d.get("target_id") is None else str(d["target_id"]),
            amount=None if amount is None else float(amount),
            success=None if success is None else bool(success),
            rating=None if rating is None else int(rating),
            text=None if d.get("text") is None else str(d["text"]),
        )


@dataclass(frozen=True, slots=True)
class AccountEvents:
    """One account and its events, sorted by time. The unit every detector scores."""

    account: Account
    events: tuple[Event, ...] = field(default_factory=tuple)

    @property
    def account_id(self) -> str:
        return self.account.account_id


def _load_schema(name: str) -> dict[str, Any]:
    text = resources.files("triage_kit.schemas").joinpath(name).read_text()
    out: dict[str, Any] = json.loads(text)
    return out


@cache
def account_schema() -> dict[str, Any]:
    return _load_schema("account.schema.json")


@cache
def event_schema() -> dict[str, Any]:
    return _load_schema("event.schema.json")


@cache
def _validator(kind: str) -> jsonschema.Draft202012Validator:
    sch = account_schema() if kind == "account" else event_schema()
    return jsonschema.Draft202012Validator(sch)


def _validate(kind: str, d: Any, where: str) -> None:
    errors = sorted(_validator(kind).iter_errors(d), key=lambda e: list(e.path))
    if errors:
        e = errors[0]
        path = ".".join(str(p) for p in e.path) or "(record)"
        raise SchemaError(f"{where}{kind} field {path}: {e.message}")


def validate_account(d: Any, where: str = "") -> None:
    _validate("account", d, where)


def validate_event(d: Any, where: str = "") -> None:
    _validate("event", d, where)


def group_events(accounts: Sequence[Account], events: Iterable[Event]) -> list[AccountEvents]:
    """Attach events to accounts; result follows the order of `accounts`."""
    by_id: dict[str, list[Event]] = {a.account_id: [] for a in accounts}
    for e in events:
        if e.account_id not in by_id:
            raise SchemaError(f"event references unknown account_id {e.account_id!r}")
        by_id[e.account_id].append(e)
    out = []
    for a in accounts:
        evs = sorted(by_id[a.account_id], key=lambda e: (e.ts, e.type))
        out.append(AccountEvents(account=a, events=tuple(evs)))
    return out


def _dump(d: dict[str, Any]) -> str:
    return json.dumps(d, sort_keys=True, separators=(",", ":"))


def write_dataset(directory: Path, accounts: Sequence[Account], events: Sequence[Event]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "accounts.jsonl").open("w") as f:
        for a in accounts:
            f.write(_dump(a.to_dict()) + "\n")
    with (directory / "events.jsonl").open("w") as f:
        for e in events:
            f.write(_dump(e.to_dict()) + "\n")


def read_dataset(directory: Path, validate: bool = False) -> list[AccountEvents]:
    accounts = []
    for i, line in enumerate((directory / "accounts.jsonl").read_text().splitlines(), 1):
        d = json.loads(line)
        if validate:
            validate_account(d, where=f"accounts.jsonl line {i}: ")
        accounts.append(Account.from_dict(d))
    events = []
    for i, line in enumerate((directory / "events.jsonl").read_text().splitlines(), 1):
        d = json.loads(line)
        if validate:
            validate_event(d, where=f"events.jsonl line {i}: ")
        events.append(Event.from_dict(d))
    return group_events(accounts, events)
