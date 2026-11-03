"""Oracle scorers: one per family, written with full knowledge of the planted
mechanism and allowed to read generator internals (meta.json: ring targets and
burst times, operator templates, mechanism constants).

The oracle is NOT a detector. It exists to prove that the signal is present in
the events at s = 1 (acceptance: PR-AUC >= 0.9 per family) and to sit in the
null-control grid at s = 0. It never reads an account's own label, family or
group: it scores each account from its events using family-level internals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from triage_kit.generator.benign import PUBLIC_PROMO_CODES
from triage_kit.generator.families import DISPOSABLE
from triage_kit.schema import AccountEvents

HOUR = 3600


def _max_in_window(ts: Sequence[int], width: int) -> int:
    if not ts:
        return 0
    a = np.asarray(ts)
    j = np.searchsorted(a, a + width, side="left")
    return int((j - np.arange(len(a))).max())


def _ato(ae: AccountEvents, meta: dict[str, Any]) -> float:
    fails = [e.ts for e in ae.events if e.type == "login_fail"]
    seen: set[str] = set()
    takeover = 0
    for i, e in enumerate(ae.events):
        if e.type == "login" and e.device_id not in seen and seen:
            later = ae.events[i + 1 : i + 12]
            if any(
                x.type in ("payout_change", "profile_update") and x.ts - e.ts < HOUR for x in later
            ):
                takeover = 1
        seen.add(e.device_id)
    return _max_in_window(fails, HOUR) + 10 * takeover


def _ring(ae: AccountEvents, meta: dict[str, Any]) -> float:
    groups = meta["families"]["review_ring"]["groups"].values()
    hits = 0
    for e in ae.events:
        if e.type != "review":
            continue
        for g in groups:
            if e.target_id in g["targets"] and any(
                b <= e.ts <= b + 6 * HOUR + 3 * 86400 for b in g["bursts"]
            ):
                hits += 1
                break
    return float(hits)


def _promo(ae: AccountEvents, meta: dict[str, Any]) -> float:
    c = ae.account.created_at
    early = sum(1 for e in ae.events if e.type == "promo_redeem" and e.ts - c < HOUR)
    referral = sum(
        1 for e in ae.events if e.type == "promo_redeem" and e.target_id not in PUBLIC_PROMO_CODES
    )
    return early + referral + 2.0 * (ae.account.email_domain in DISPOSABLE)


def _scraping(ae: AccountEvents, meta: dict[str, Any]) -> float:
    views = [e.ts for e in ae.events if e.type == "view"]
    return float(_max_in_window(views, 600))


def _card(ae: AccountEvents, meta: dict[str, Any]) -> float:
    return float(
        sum(
            1
            for e in ae.events
            if e.type == "payment_attempt" and e.success is False and (e.amount or 0) <= 5
        )
    )


def _template(ae: AccountEvents, meta: dict[str, Any]) -> float:
    toks = {(e.type, e.target_id) for e in ae.events if e.type in ("view", "review")}
    best = 0.0
    for tpl in meta["families"]["template_operator"]["templates"].values():
        want = {(st["type"], st["target"]) for st in tpl["steps"] if "target" in st}
        if want:
            best = max(best, len(toks & want) / len(want))
    return best


ORACLES: dict[str, Callable[[AccountEvents, dict[str, Any]], float]] = {
    "ato": _ato,
    "review_ring": _ring,
    "promo_abuse": _promo,
    "scraping": _scraping,
    "card_testing": _card,
    "template_operator": _template,
}


def oracle_scores(family: str, data: Sequence[AccountEvents], meta: dict[str, Any]) -> np.ndarray:
    """Raw oracle statistic per account (higher = more like the planted mechanism)."""
    fn = ORACLES[family]
    return np.array([fn(ae, meta) for ae in data], dtype=float)
