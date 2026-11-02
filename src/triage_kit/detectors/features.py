"""Engineered per-account features, shared by the rules and the feature model.

Each feature is computed from one account's events and profile only. The one
population-level feature, `template_sim`, lives in similarity.py and is added
by `feature_table` when a reference population is given.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median

import numpy as np

from triage_kit.detectors.similarity import SimilarityIndex
from triage_kit.schema import AccountEvents

HOUR = 3600
SESSION_GAP = 1800
# Generic human heuristic for throwaway email providers (substring match on the domain).
DISPOSABLE_HINTS = ("temp", "throwaway", "trash", "burner", "10min", "mailinator")

FEATURE_NAMES: tuple[str, ...] = (
    "n_events",
    "n_sessions",
    "active_days",
    "events_per_session",
    "n_login_fail",
    "max_login_fail_1h",
    "n_devices",
    "n_ips",
    "new_device_then_account_change",
    "n_profile_update",
    "n_payout_change",
    "n_view",
    "n_search",
    "max_views_10min",
    "median_gap_s",
    "gap_cv",
    "n_payment_attempt",
    "n_failed_payments",
    "max_failed_payments_1h",
    "payment_fail_rate",
    "median_payment_amount",
    "n_purchase",
    "max_purchase_amount",
    "n_review",
    "mean_rating",
    "share_5star",
    "reviews_without_purchase",
    "mean_review_words",
    "n_listing_create",
    "n_promo_redeem",
    "promo_first_hour",
    "n_promo_failed",
    "email_disposable_hint",
    "first_event_delay_h",
)


def _max_in_window(ts: list[int], width: int) -> int:
    if not ts:
        return 0
    a = np.asarray(ts)
    j = np.searchsorted(a, a + width, side="left")
    return int((j - np.arange(len(a))).max())


def account_features(ae: AccountEvents) -> dict[str, float]:
    ev = ae.events
    c = ae.account.created_at
    ts = [e.ts for e in ev]
    gaps = np.diff(ts) if len(ts) > 1 else np.array([])
    within = gaps[gaps <= SESSION_GAP]
    n_sessions = (1 + int((gaps > SESSION_GAP).sum())) if ev else 0

    fails = [e.ts for e in ev if e.type == "login_fail"]
    views = [e.ts for e in ev if e.type == "view"]
    pays = [e for e in ev if e.type == "payment_attempt"]
    failed_pay = [e.ts for e in pays if e.success is False]
    purchases = [e for e in ev if e.type == "purchase"]
    reviews = [e for e in ev if e.type == "review"]
    promos = [e for e in ev if e.type == "promo_redeem"]

    seen_dev: set[str] = set()
    takeover = 0.0
    for i, e in enumerate(ev):
        if e.type == "login" and seen_dev and e.device_id not in seen_dev:
            for x in ev[i + 1 : i + 12]:
                if x.type in ("profile_update", "payout_change") and x.ts - e.ts <= HOUR:
                    takeover = 1.0
        seen_dev.add(e.device_id)

    bought: set[str] = set()
    no_purchase_reviews = 0
    for e in ev:
        if e.type == "purchase" and e.target_id:
            bought.add(e.target_id)
        if e.type == "review" and e.target_id not in bought:
            no_purchase_reviews += 1

    ratings = [e.rating for e in reviews if e.rating is not None]
    amounts = [e.amount for e in pays if e.amount is not None]
    domain = ae.account.email_domain.lower()
    return {
        "n_events": float(len(ev)),
        "n_sessions": float(n_sessions),
        "active_days": float(len({t // 86400 for t in ts})),
        "events_per_session": len(ev) / n_sessions if n_sessions else 0.0,
        "n_login_fail": float(len(fails)),
        "max_login_fail_1h": float(_max_in_window(fails, HOUR)),
        "n_devices": float(len({e.device_id for e in ev})),
        "n_ips": float(len({e.ip for e in ev})),
        "new_device_then_account_change": takeover,
        "n_profile_update": float(sum(e.type == "profile_update" for e in ev)),
        "n_payout_change": float(sum(e.type == "payout_change" for e in ev)),
        "n_view": float(len(views)),
        "n_search": float(sum(e.type == "search" for e in ev)),
        "max_views_10min": float(_max_in_window(views, 600)),
        "median_gap_s": float(np.median(within)) if len(within) else 0.0,
        "gap_cv": float(within.std() / within.mean())
        if len(within) >= 5 and within.mean() > 0
        else 1.0,
        "n_payment_attempt": float(len(pays)),
        "n_failed_payments": float(len(failed_pay)),
        "max_failed_payments_1h": float(_max_in_window(failed_pay, HOUR)),
        "payment_fail_rate": len(failed_pay) / len(pays) if pays else 0.0,
        "median_payment_amount": float(median(amounts)) if amounts else 0.0,
        "n_purchase": float(len(purchases)),
        "max_purchase_amount": max((e.amount or 0.0 for e in purchases), default=0.0),
        "n_review": float(len(reviews)),
        "mean_rating": float(np.mean(ratings)) if ratings else 0.0,
        "share_5star": sum(r == 5 for r in ratings) / len(ratings) if ratings else 0.0,
        "reviews_without_purchase": float(no_purchase_reviews),
        "mean_review_words": (
            float(np.mean([len((e.text or "").split()) for e in reviews])) if reviews else 0.0
        ),
        "n_listing_create": float(sum(e.type == "listing_create" for e in ev)),
        "n_promo_redeem": float(len(promos)),
        "promo_first_hour": float(sum(e.ts - c <= HOUR for e in promos)),
        "n_promo_failed": float(sum(e.success is False for e in promos)),
        "email_disposable_hint": float(any(h in domain for h in DISPOSABLE_HINTS)),
        "first_event_delay_h": (ts[0] - c) / HOUR if ts else 0.0,
    }


def feature_matrix(data: Sequence[AccountEvents]) -> np.ndarray:
    rows = [account_features(ae) for ae in data]
    return np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=float).reshape(
        len(rows), len(FEATURE_NAMES)
    )


def feature_table(
    data: Sequence[AccountEvents], reference: Sequence[AccountEvents] | None = None
) -> tuple[list[str], np.ndarray, list[str]]:
    """(account ids, feature matrix, column names). Adds `template_sim` when a
    reference population is given."""
    ids = [ae.account_id for ae in data]
    X = feature_matrix(data)
    names = list(FEATURE_NAMES)
    if reference is not None:
        idx = SimilarityIndex(reference)
        sim = np.array([idx.value(ae) for ae in data])
        X = np.column_stack([X, sim])
        names.append("template_sim")
    return ids, X, names
