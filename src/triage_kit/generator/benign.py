"""Benign population: ordinary buyers and sellers.

Every abuse account starts from a timeline drawn by this same code, so at
signal strength s = 0 abuse accounts are drawn from the benign distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from triage_kit.generator import text
from triage_kit.schema import Event

DAY = 86_400
COUNTRIES = ["DE", "FR", "GB", "ES", "IT", "NL", "PL", "US", "SE", "PT"]
COUNTRY_W = np.array([0.18, 0.14, 0.14, 0.1, 0.1, 0.08, 0.08, 0.1, 0.04, 0.04])
EMAIL_DOMAINS = [
    "mail.example",
    "post.example",
    "inbox.example",
    "webmail.example",
    "fastmail.example",
    "uni.example",
    "isp-one.example",
    "isp-two.example",
    "tempbox.example",
    "throwaway.example",
]
EMAIL_W = np.array([0.3, 0.2, 0.15, 0.1, 0.07, 0.06, 0.05, 0.05, 0.01, 0.01])
PUBLIC_PROMO_CODES = ["WELCOME10", "SPRING5", "FREESHIP", "NEWUSER", "FRIEND15"]


@dataclass(frozen=True)
class World:
    start_ts: int
    end_ts: int
    signup_end_ts: int
    catalog: list[str]
    popularity: np.ndarray  # probabilities over catalog
    cdf: np.ndarray

    def draw_listing(self, rng: np.random.Generator) -> str:
        i = int(np.searchsorted(self.cdf, rng.random(), side="right"))
        return self.catalog[min(i, len(self.catalog) - 1)]


def make_world(rng: np.random.Generator, start_ts: int, days: int, n_listings: int) -> World:
    catalog = [f"L{i:05d}" for i in range(n_listings)]
    ranks = rng.permutation(n_listings) + 1
    pop = 1.0 / ranks.astype(float) ** 0.8
    pop = pop / pop.sum()
    return World(
        start_ts=start_ts,
        end_ts=start_ts + days * DAY,
        signup_end_ts=start_ts + int(days * 0.75) * DAY,
        catalog=catalog,
        popularity=pop,
        cdf=np.cumsum(pop),
    )


def random_ip(rng: np.random.Generator) -> str:
    a, b, c = (int(x) for x in rng.integers(0, 256, size=3))
    return f"10.{a}.{b}.{c}"


def random_device(rng: np.random.Generator) -> str:
    return "dev-" + "".join(f"{int(x):02x}" for x in rng.integers(0, 256, size=4))


def draw_created_at(rng: np.random.Generator, world: World) -> int:
    return int(rng.integers(world.start_ts, world.signup_end_ts))


def draw_profile(rng: np.random.Generator) -> tuple[str, str]:
    country = COUNTRIES[int(rng.choice(len(COUNTRIES), p=COUNTRY_W))]
    domain = EMAIL_DOMAINS[int(rng.choice(len(EMAIL_DOMAINS), p=EMAIL_W))]
    return country, domain


@dataclass
class Persona:
    account_id: str
    created_at: int
    devices: list[str]
    home_ip: str
    is_seller: bool


def draw_persona(rng: np.random.Generator, account_id: str, created_at: int) -> Persona:
    n_dev = 1 + int(rng.random() < 0.3)
    return Persona(
        account_id=account_id,
        created_at=created_at,
        devices=[random_device(rng) for _ in range(n_dev)],
        home_ip=random_ip(rng),
        is_seller=bool(rng.random() < 0.12),
    )


def _gap(rng: np.random.Generator) -> int:
    return max(1, int(rng.lognormal(np.log(40), 0.9)))


class _Session:
    """Appends events with benign think-time gaps between them."""

    def __init__(self, rng: np.random.Generator, p: Persona, t: int, ip: str, dev: str) -> None:
        self.rng, self.p, self.t, self.ip, self.dev = rng, p, t, ip, dev
        self.events: list[Event] = []

    def add(self, type_: str, **kw: Any) -> None:
        self.events.append(
            Event(
                account_id=self.p.account_id,
                ts=self.t,
                type=type_,
                ip=self.ip,
                device_id=self.dev,
                **kw,
            )
        )
        self.t += _gap(self.rng)


def benign_sessions(rng: np.random.Generator, world: World, p: Persona) -> list[list[Event]]:
    """Draw a benign timeline as a list of sessions (each a list of events)."""
    lam = rng.lognormal(np.log(5.0), 0.8)
    n_sessions = min(40, 1 + int(rng.poisson(lam)))
    starts = np.sort(rng.integers(p.created_at, world.end_ts, size=n_sessions - 1))
    starts = np.concatenate([[p.created_at + int(rng.integers(0, 600))], starts])
    purchased: list[str] = []
    listing_no = 0
    sessions: list[list[Event]] = []
    for si, start in enumerate(starts):
        dev = p.devices[int(rng.integers(len(p.devices)))]
        ip = p.home_ip if rng.random() < 0.8 else random_ip(rng)
        ss = _Session(rng, p, int(start), ip, dev)
        if rng.random() < 0.04:
            for _ in range(1 + int(rng.random() < 0.25)):
                ss.add("login_fail", success=False)
        ss.add("login", success=True)
        if si == 0 and rng.random() < 0.15:
            ss.add("promo_redeem", target_id=text.pick(rng, PUBLIC_PROMO_CODES), success=True)
        if purchased and rng.random() < 0.25:
            ss.add(
                "review",
                target_id=purchased.pop(0),
                rating=int(rng.choice(5, p=[0.05, 0.05, 0.1, 0.3, 0.5])) + 1,
                text=text.benign_review(rng),
            )
        n_views = int(rng.geometric(0.25))
        last_view = None
        for _ in range(n_views):
            if rng.random() < 0.3:
                ss.add("search", text=text.benign_search(rng))
            last_view = world.draw_listing(rng)
            ss.add("view", target_id=last_view)
        if last_view is not None and rng.random() < 0.12:
            amount = round(float(rng.lognormal(np.log(30), 0.8)), 2)
            ok = bool(rng.random() < 0.96)
            ss.add("payment_attempt", amount=amount, success=ok)
            if not ok:
                ok = bool(rng.random() < 0.8)
                ss.add("payment_attempt", amount=amount, success=ok)
            if ok:
                ss.add("purchase", target_id=last_view, amount=amount)
                purchased.append(last_view)
        if p.is_seller and rng.random() < 0.15:
            listing_no += 1
            ss.add(
                "listing_create",
                target_id=f"N{p.account_id[:6]}{listing_no:02d}",
                text=text.benign_listing(rng),
                amount=round(float(rng.lognormal(np.log(40), 0.9)), 2),
            )
        if rng.random() < 0.02:
            ss.add("profile_update")
        if p.is_seller and rng.random() < 0.01:
            ss.add("payout_change")
        sessions.append([e for e in ss.events if e.ts < world.end_ts])
    return sessions


def thin_sessions(
    rng: np.random.Generator, sessions: list[list[Event]], keep_p: float
) -> list[list[Event]]:
    """Keep the signup session; keep each later session with probability keep_p."""
    return [sess for i, sess in enumerate(sessions) if i == 0 or rng.random() < keep_p]
