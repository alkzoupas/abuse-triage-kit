"""Six abuse families, each written from its mechanism description.

Every abuse account starts from a benign persona and benign timeline drawn by
`benign.py`. The family mechanism is then applied with intensity s in [0, 1]:
counts are drawn as Binomial(n, s) or Poisson(rate * s), and every optional
behaviour happens with probability proportional to s. At s = 0 no mechanism
event is added, no benign session is removed and every attribute keeps its
benign draw, so abuse accounts are drawn from the benign distribution (the
null control).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from triage_kit.generator import benign, text
from triage_kit.generator.benign import DAY, Persona, World
from triage_kit.schema import Account, Event

Built = tuple[Account, list[Event], dict[str, Any]]


@dataclass(frozen=True)
class Family:
    name: str
    coordinated: bool
    channels: tuple[str, ...]
    mechanism: str
    build: Callable[..., tuple[list[Built], dict[str, Any]]] = field(repr=False)


def _lerp(a: float, b: float, s: float) -> float:
    return (1.0 - s) * a + s * b


def _base(
    rng: np.random.Generator, world: World, account_id: str, created_at: int | None = None
) -> tuple[Account, Persona, list[list[Event]]]:
    """A benign account: profile, persona and timeline (identical code path to benign)."""
    c = benign.draw_created_at(rng, world) if created_at is None else created_at
    country, domain = benign.draw_profile(rng)
    persona = benign.draw_persona(rng, account_id, c)
    sessions = benign.benign_sessions(rng, world, persona)
    return Account(account_id, c, country, domain), persona, sessions


def _ev(p: Persona, ts: int, type_: str, ip: str, dev: str, **kw: Any) -> Event:
    return Event(account_id=p.account_id, ts=int(ts), type=type_, ip=ip, device_id=dev, **kw)


def _flat(sessions: list[list[Event]]) -> list[Event]:
    return [e for sess in sessions for e in sess]


def _label(account_id: str, family: str, **kw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "account_id": account_id,
        "is_abuse": True,
        "family": family,
        "group_id": None,
        "template_id": None,
    }
    out.update(kw)
    return out


def _clip(world: World, evs: list[Event]) -> list[Event]:
    return [e for e in evs if e.ts < world.end_ts]


# --------------------------------------------------------------------------- ATO
ATO_MECHANISM = (
    "Credential stuffing leading to account takeover. An attacker replays leaked "
    "username/password pairs against a real, previously benign account. Many failed logins "
    "arrive in a short burst from rotating IPs and an unfamiliar device. If one succeeds, the "
    "attacker logs in from that new device and, within minutes, changes profile details and "
    "the payout destination and makes high-value purchases. The genuine owner's later activity "
    "mostly stops."
)


def build_ato(
    rng: np.random.Generator, world: World, s: float, ids: list[str]
) -> tuple[list[Built], dict[str, Any]]:
    out: list[Built] = []
    for aid in ids:
        acct, p, sessions = _base(rng, world, aid)
        lo = min(acct.created_at + 7 * DAY, world.end_ts - 2 * DAY)
        t_a = int(rng.integers(lo, world.end_ts - DAY))
        kept_before = [x for x in sessions if not x or x[0].ts < t_a]
        after = [x for x in sessions if x and x[0].ts >= t_a]
        after = [x for x in after if rng.random() >= 0.7 * s]
        evs = _flat(kept_before + after)
        attacker_dev = benign.random_device(rng)
        t = t_a
        attacker_ip = benign.random_ip(rng)
        for _ in range(int(rng.poisson(25 * s))):
            ip = benign.random_ip(rng) if rng.random() < s else attacker_ip
            evs.append(_ev(p, t, "login_fail", ip, attacker_dev, success=False))
            t += 1 + int(rng.exponential(20))
        if rng.random() < s:
            evs.append(_ev(p, t, "login", attacker_ip, attacker_dev, success=True))
            t += int(rng.integers(20, 300))
            if rng.random() < s:
                evs.append(_ev(p, t, "profile_update", attacker_ip, attacker_dev))
                t += int(rng.integers(20, 300))
            if rng.random() < s:
                evs.append(_ev(p, t, "payout_change", attacker_ip, attacker_dev))
                t += int(rng.integers(20, 300))
            for _ in range(int(rng.binomial(3, s))):
                tgt = world.draw_listing(rng)
                amt = round(float(rng.lognormal(np.log(150), 0.5)), 2)
                evs.append(_ev(p, t, "view", attacker_ip, attacker_dev, target_id=tgt))
                t += int(rng.integers(10, 120))
                evs.append(
                    _ev(
                        p, t, "payment_attempt", attacker_ip, attacker_dev, amount=amt, success=True
                    )
                )
                t += int(rng.integers(5, 30))
                evs.append(
                    _ev(p, t, "purchase", attacker_ip, attacker_dev, target_id=tgt, amount=amt)
                )
                t += int(rng.integers(30, 600))
        out.append((acct, _clip(world, evs), _label(aid, "ato", attack_ts=t_a)))
    return out, {"mean_failed_logins_at_s1": 25}


# --------------------------------------------------------------------------- review ring
RING_PHRASEBOOK = [
    "Amazing product!! Highly recommend",
    "Best seller ever, 5 stars",
    "Perfect, super fast shipping",
    "Great great great, love it",
    "Must buy, top quality",
    "Excellent seller, A+++",
    "Exactly perfect, will buy again!!",
]
RING_MECHANISM = (
    "Fake-review ring. A seller controls a group of accounts that post positive reviews on the "
    "seller's own listings to inflate their rating. Ring members review the same small set of "
    "target listings during a few coordinated bursts, usually without having bought the item, "
    "give five stars and copy short promotional lines from a shared phrasebook. Members are "
    "often created in a batch around the same time and otherwise show thin, normal-looking "
    "browsing."
)


def build_ring(
    rng: np.random.Generator, world: World, s: float, groups: list[list[str]]
) -> tuple[list[Built], dict[str, Any]]:
    out: list[Built] = []
    meta: dict[str, Any] = {"groups": {}}
    for gi, ids in enumerate(groups):
        gid = f"grp-ring-{gi:03d}"
        targets = [
            str(x) for x in rng.choice(world.catalog, size=int(rng.integers(3, 6)), replace=False)
        ]
        base_c = int(rng.integers(world.start_ts, world.signup_end_ts - 3 * DAY))
        bursts = sorted(
            int(rng.integers(base_c + 3 * DAY, world.end_ts - DAY))
            for _ in range(int(rng.integers(2, 4)))
        )
        meta["groups"][gid] = {"targets": targets, "bursts": bursts}
        for aid in ids:
            c_benign = benign.draw_created_at(rng, world)
            c = int(_lerp(c_benign, base_c + int(rng.integers(0, 2 * DAY)), s))
            acct, p, sessions = _base(rng, world, aid, created_at=c)
            sessions = benign.thin_sessions(rng, sessions, 1.0 - 0.6 * s)
            evs = _flat(sessions)
            dev = p.devices[0]
            for tgt in targets:
                if rng.random() >= s:
                    continue
                t = bursts[int(rng.integers(len(bursts)))] + int(rng.integers(0, 6 * 3600))
                t = max(t, acct.created_at + 3600)  # cannot review before the account exists
                ip = p.home_ip
                evs.append(_ev(p, t, "login", ip, dev, success=True))
                t += int(rng.integers(5, 60))
                evs.append(_ev(p, t, "view", ip, dev, target_id=tgt))
                t += int(rng.integers(10, 120))
                if rng.random() >= s:  # at full strength the ring skips the purchase
                    amt = round(float(rng.lognormal(np.log(30), 0.8)), 2)
                    evs.append(_ev(p, t, "payment_attempt", ip, dev, amount=amt, success=True))
                    evs.append(_ev(p, t + 5, "purchase", ip, dev, target_id=tgt, amount=amt))
                    t += int(rng.integers(60, 600))
                rating = (
                    5 if rng.random() < s else int(rng.choice(5, p=[0.05, 0.05, 0.1, 0.3, 0.5])) + 1
                )
                body = (
                    text.pick(rng, RING_PHRASEBOOK) if rng.random() < s else text.benign_review(rng)
                )
                evs.append(_ev(p, t, "review", ip, dev, target_id=tgt, rating=rating, text=body))
            out.append((acct, _clip(world, evs), _label(aid, "review_ring", group_id=gid)))
    return out, meta


# --------------------------------------------------------------------------- promo abuse
DISPOSABLE = ["tempbox.example", "throwaway.example"]
PROMO_MECHANISM = (
    "Promo and referral abuse. One person creates many throwaway accounts to collect new-user "
    "promotions and referral credit repeatedly. Each account signs up with a disposable email "
    "address, redeems several promo or referral codes within minutes of signing up (some are "
    "rejected), spends the credit on one small purchase and then goes dormant."
)


def build_promo(
    rng: np.random.Generator, world: World, s: float, ids: list[str]
) -> tuple[list[Built], dict[str, Any]]:
    out: list[Built] = []
    for aid in ids:
        acct, p, sessions = _base(rng, world, aid)
        if rng.random() < 0.8 * s:
            acct = Account(aid, acct.created_at, acct.country, text.pick(rng, DISPOSABLE))
        sessions = benign.thin_sessions(rng, sessions, 1.0 - 0.9 * s)
        evs = _flat(sessions)
        dev, ip = p.devices[0], p.home_ip
        t = acct.created_at + int(rng.integers(30, 300))
        for _ in range(int(rng.binomial(4, s))):
            code = (
                text.pick(rng, benign.PUBLIC_PROMO_CODES)
                if rng.random() < 0.5
                else "REF-" + "".join(f"{int(x):02X}" for x in rng.integers(0, 256, size=3))
            )
            evs.append(
                _ev(p, t, "promo_redeem", ip, dev, target_id=code, success=bool(rng.random() < 0.6))
            )
            t += int(rng.integers(10, 120))
        if rng.random() < s:
            tgt = world.draw_listing(rng)
            amt = round(float(rng.uniform(1, 12)), 2)
            evs.append(_ev(p, t, "view", ip, dev, target_id=tgt))
            evs.append(_ev(p, t + 30, "payment_attempt", ip, dev, amount=amt, success=True))
            evs.append(_ev(p, t + 35, "purchase", ip, dev, target_id=tgt, amount=amt))
        out.append((acct, _clip(world, evs), _label(aid, "promo_abuse")))
    return out, {"codes_per_account_at_s1": 4}


# --------------------------------------------------------------------------- scraping
SCRAPE_MECHANISM = (
    "Scraping. A bot logs into an account and crawls listing pages and search results to "
    "harvest prices and content. It requests pages at a very high, machine-regular rate, walks "
    "listing ids in order, runs searches at fixed intervals, comes from datacenter IP space and "
    "never buys anything."
)


def build_scraping(
    rng: np.random.Generator, world: World, s: float, ids: list[str]
) -> tuple[list[Built], dict[str, Any]]:
    out: list[Built] = []
    n_cat = len(world.catalog)
    for aid in ids:
        acct, p, sessions = _base(rng, world, aid)
        sessions = benign.thin_sessions(rng, sessions, 1.0 - 0.8 * s)
        evs = _flat(sessions)
        dev = p.devices[0]
        dc_ip = f"198.18.{int(rng.integers(0, 256))}.{int(rng.integers(0, 256))}"
        period = float(rng.uniform(1.5, 4.0))
        for _ in range(int(rng.binomial(4, s))):
            t = int(rng.integers(acct.created_at, world.end_ts - DAY))
            ip = dc_ip if rng.random() < s else p.home_ip
            evs.append(_ev(p, t, "login", ip, dev, success=True))
            pos = int(rng.integers(n_cat))
            for k in range(int(rng.integers(100, 200))):
                t += max(1, int(round(period * (1 + 0.1 * rng.standard_normal()))))
                if k % 20 == 0:
                    evs.append(_ev(p, t, "search", ip, dev, text=text.pick(rng, text.ITEMS)))
                else:
                    evs.append(_ev(p, t, "view", ip, dev, target_id=world.catalog[pos % n_cat]))
                    pos += 1
        out.append((acct, _clip(world, evs), _label(aid, "scraping")))
    return out, {"datacenter_prefix": "198.18."}


# --------------------------------------------------------------------------- card testing
CARD_MECHANISM = (
    "Card testing. A fraudster holding a list of stolen card numbers uses an account to find "
    "out which cards are live. The account makes rapid bursts of small payment attempts, most "
    "of which are declined; the few successful ones are followed by a cheap purchase. Bursts "
    "are short and dense."
)


def build_card(
    rng: np.random.Generator, world: World, s: float, ids: list[str]
) -> tuple[list[Built], dict[str, Any]]:
    out: list[Built] = []
    for aid in ids:
        acct, p, sessions = _base(rng, world, aid)
        sessions = benign.thin_sessions(rng, sessions, 1.0 - 0.7 * s)
        evs = _flat(sessions)
        dev, ip = p.devices[0], p.home_ip
        for _ in range(int(rng.binomial(3, s))):
            t = int(rng.integers(acct.created_at, world.end_ts - DAY))
            evs.append(_ev(p, t, "login", ip, dev, success=True))
            tgt = world.draw_listing(rng)
            for _ in range(5 + int(rng.poisson(15))):
                t += 1 + int(rng.exponential(30))
                amt = round(float(rng.uniform(0.5, 5)), 2)
                ok = bool(rng.random() < 0.15)
                evs.append(_ev(p, t, "payment_attempt", ip, dev, amount=amt, success=ok))
                if ok and rng.random() < 0.5:
                    evs.append(_ev(p, t + 3, "purchase", ip, dev, target_id=tgt, amount=amt))
        out.append((acct, _clip(world, evs), _label(aid, "card_testing")))
    return out, {"decline_rate": 0.85}


# --------------------------------------------------------------------------- template operator
TEMPLATE_MECHANISM = (
    "Template operator. One operator runs many accounts from a script. The script is an action "
    "template: an ordered list of steps (log in, search, view specific listings, create a "
    "listing from a text skeleton, update the profile, and so on) with typical delays. Each "
    "account replays the template a few times with small perturbations: timing jitter, an "
    "occasional skipped or repeated step, a different view target now and then, and synonym "
    "swaps in the text slots. Individually each account looks like an ordinary user; the "
    "evidence is that many accounts share near-identical action sequences. Accounts in a crew "
    "are created in a batch."
)
_STEP_TYPES = [
    "search",
    "view",
    "view",
    "view",
    "listing_create",
    "profile_update",
    "review",
    "promo_redeem",
]
_SYNONYMS = [
    ["great", "nice", "good", "lovely"],
    ["deal", "offer", "bargain", "price"],
    ["fast", "quick", "express", "speedy"],
    ["new", "unused", "mint", "pristine"],
]


def _make_template(rng: np.random.Generator, world: World) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [{"type": "login", "delay": int(rng.integers(1, 10))}]
    for _ in range(int(rng.integers(10, 18))):
        st = _STEP_TYPES[int(rng.integers(len(_STEP_TYPES)))]
        step: dict[str, Any] = {"type": st, "delay": max(2, int(rng.lognormal(np.log(45), 0.8)))}
        if st in ("view", "review"):
            step["target"] = str(world.catalog[int(rng.integers(len(world.catalog)))])
        if st == "search":
            step["query"] = text.benign_search(rng)
        if st == "listing_create":
            slots = [int(i) for i in rng.choice(len(_SYNONYMS), size=2, replace=False)]
            step["skeleton"] = (
                f"{{{slots[0]}}} {text.pick(rng, text.ITEMS)} - {{{slots[1]}}} "
                f"{text.pick(rng, text.CONDITIONS)}"
            )
            step["price"] = round(float(rng.lognormal(np.log(40), 0.6)), 2)
        if st == "review":
            step["rating"] = int(rng.integers(3, 6))
            slot = int(rng.integers(len(_SYNONYMS)))
            step["skeleton"] = f"{{{slot}}} item, {text.pick(rng, text.REVIEW_CLAUSES)}"
        if st == "promo_redeem":
            step["code"] = text.pick(rng, benign.PUBLIC_PROMO_CODES)
        steps.append(step)
    return {"steps": steps}


def _fill(rng: np.random.Generator, skeleton: str) -> str:
    out = skeleton
    for i, syn in enumerate(_SYNONYMS):
        out = out.replace("{" + str(i) + "}", text.pick(rng, syn))
    return out


TEMPLATE_MODES = ("standard", "cross_only", "ordinary")


def _make_ordinary_template(rng: np.random.Generator, world: World) -> dict[str, Any]:
    """A script shaped like benign sessions: login, searches and popular-listing views only.

    Owner-approved post-hoc addition (PREREGISTRATION.md AMENDMENTS). Stored delays are
    LogNormal(log 40, 0.6); replay jitter LogNormal(0, 0.67) gives the benign gap marginal
    LogNormal(log 40, 0.9).
    """

    def delay() -> int:
        return max(1, int(rng.lognormal(np.log(40), 0.6)))

    sessions: list[list[dict[str, Any]]] = []
    for _ in range(int(rng.integers(2, 4))):
        steps: list[dict[str, Any]] = [{"type": "login", "delay": delay()}]
        for _ in range(int(rng.geometric(0.25))):
            if rng.random() < 0.3:
                steps.append({"type": "search", "delay": delay(), "query": text.benign_search(rng)})
            tgt = str(world.catalog[int(rng.integers(len(world.catalog)))])  # uniform (rev. 1)
            steps.append({"type": "view", "delay": delay(), "target": tgt})
        sessions.append(steps)
    return {"sessions": sessions, "steps": [st for sess in sessions for st in sess]}


def _replay_ordinary(
    rng: np.random.Generator, world: World, p: Persona, acct: Account, tpl: dict[str, Any]
) -> list[Event]:
    """One replay of an ordinary script; timing, device, IP and purchases drawn per account."""
    out: list[Event] = []
    starts = np.sort(rng.integers(acct.created_at, world.end_ts - DAY, size=len(tpl["sessions"])))
    for start, steps in zip(starts, tpl["sessions"], strict=True):
        dev = p.devices[int(rng.integers(len(p.devices)))]
        ip = p.home_ip if rng.random() < 0.8 else benign.random_ip(rng)
        t = int(start)
        last_view: str | None = None
        for step in steps:
            if step["type"] != "login" and rng.random() < 0.1:
                continue  # skipped step
            if step["type"] == "login":
                out.append(_ev(p, t, "login", ip, dev, success=True))
            elif step["type"] == "search":
                out.append(_ev(p, t, "search", ip, dev, text=step["query"]))
            else:
                tgt = step["target"] if rng.random() < 0.7 else world.draw_listing(rng)
                out.append(_ev(p, t, "view", ip, dev, target_id=tgt))
                last_view = tgt
            t += max(1, int(step["delay"] * rng.lognormal(0, 0.67)))
        if last_view is not None and rng.random() < 0.12:  # benign purchase probabilities
            amount = round(float(rng.lognormal(np.log(30), 0.8)), 2)
            ok = bool(rng.random() < 0.96)
            out.append(_ev(p, t, "payment_attempt", ip, dev, amount=amount, success=ok))
            t += max(1, int(rng.lognormal(np.log(40), 0.9)))
            if not ok:
                ok = bool(rng.random() < 0.8)
                out.append(_ev(p, t, "payment_attempt", ip, dev, amount=amount, success=ok))
                t += max(1, int(rng.lognormal(np.log(40), 0.9)))
            if ok:
                out.append(_ev(p, t, "purchase", ip, dev, target_id=last_view, amount=amount))
    return out


def build_template(
    rng: np.random.Generator,
    world: World,
    s: float,
    groups: list[list[str]],
    mode: str = "standard",
) -> tuple[list[Built], dict[str, Any]]:
    """mode="standard": the mechanism above, scaled by s.

    mode="cross_only" (owner-approved post-hoc addition, see PREREGISTRATION.md AMENDMENTS):
    every account replays its crew's script exactly once, keeps all its benign sessions,
    gets benign-like jitter between steps and a benign signup time. Only the shared step
    order, targets and text skeletons remain, i.e. the cross-account evidence. Ignores s.

    mode="ordinary" (second owner-approved post-hoc addition): as cross_only, but the script
    is built only from what benign sessions do (login, search, popular-listing views, the
    occasional purchase), so the only evidence is the shared order of targets and queries.
    """
    if mode not in TEMPLATE_MODES:
        raise ValueError(f"unknown template mode {mode!r}")
    cross = mode in ("cross_only", "ordinary")
    ordinary = mode == "ordinary"
    out: list[Built] = []
    meta: dict[str, Any] = {"templates": {}, "groups": {}}
    for gi, ids in enumerate(groups):
        gid = f"grp-tpl-{gi:03d}"
        tid = f"tpl-{gi:03d}"
        tpl = _make_ordinary_template(rng, world) if ordinary else _make_template(rng, world)
        meta["templates"][tid] = tpl
        base_c = int(rng.integers(world.start_ts, world.signup_end_ts - 3 * DAY))
        meta["groups"][gid] = {"template_id": tid}
        for aid in ids:
            c_benign = benign.draw_created_at(rng, world)
            c = int(_lerp(c_benign, base_c + int(rng.integers(0, 2 * DAY)), s))
            if cross:
                c = c_benign  # no batch creation
            acct, p, sessions = _base(rng, world, aid, created_at=c)
            sessions = benign.thin_sessions(rng, sessions, 1.0 if cross else 1.0 - 0.7 * s)
            evs = _flat(sessions)
            dev = p.devices[0]
            n_runs = 0 if ordinary else 1 if cross else int(rng.binomial(3, s))
            if ordinary:
                evs.extend(_replay_ordinary(rng, world, p, acct, tpl))
            jitter = 0.9 if cross else 0.25
            for _ in range(n_runs):
                t = int(rng.integers(acct.created_at, world.end_ts - DAY))
                ip = p.home_ip if rng.random() < 0.8 else benign.random_ip(rng)
                for step in tpl["steps"]:
                    if step["type"] != "login" and rng.random() < 0.1:
                        continue  # skipped step
                    reps = 2 if rng.random() < 0.05 else 1
                    for _ in range(reps):
                        t += max(1, int(step["delay"] * rng.lognormal(0, jitter)))
                        evs.append(_template_event(rng, world, p, t, ip, dev, step))
            out.append(
                (
                    acct,
                    _clip(world, evs),
                    _label(aid, "template_operator", group_id=gid, template_id=tid),
                )
            )
    return out, meta


def _template_event(
    rng: np.random.Generator,
    world: World,
    p: Persona,
    t: int,
    ip: str,
    dev: str,
    step: dict[str, Any],
) -> Event:
    st = step["type"]
    if st == "login":
        return _ev(p, t, "login", ip, dev, success=True)
    if st == "search":
        return _ev(p, t, "search", ip, dev, text=step["query"])
    if st == "view":
        tgt = (
            step["target"]
            if rng.random() < 0.7
            else world.catalog[int(rng.integers(len(world.catalog)))]
        )
        return _ev(p, t, "view", ip, dev, target_id=tgt)
    if st == "listing_create":
        lid = "N" + "".join(f"{int(x):02x}" for x in rng.integers(0, 256, size=4))
        return _ev(
            p,
            t,
            "listing_create",
            ip,
            dev,
            target_id=lid,
            text=_fill(rng, step["skeleton"]),
            amount=step["price"],
        )
    if st == "review":
        return _ev(
            p,
            t,
            "review",
            ip,
            dev,
            target_id=step["target"],
            rating=step["rating"],
            text=_fill(rng, step["skeleton"]),
        )
    if st == "promo_redeem":
        return _ev(p, t, "promo_redeem", ip, dev, target_id=step["code"], success=True)
    return _ev(p, t, st, ip, dev)


FAMILIES: dict[str, Family] = {
    f.name: f
    for f in [
        Family("ato", False, ("counts/velocity", "timing/sequence"), ATO_MECHANISM, build_ato),
        Family(
            "review_ring",
            True,
            ("cross-account", "text", "counts/velocity"),
            RING_MECHANISM,
            build_ring,
        ),
        Family(
            "promo_abuse",
            False,
            ("account attributes", "counts/velocity", "timing/sequence"),
            PROMO_MECHANISM,
            build_promo,
        ),
        Family(
            "scraping",
            False,
            ("counts/velocity", "timing/sequence"),
            SCRAPE_MECHANISM,
            build_scraping,
        ),
        Family("card_testing", False, ("counts/velocity",), CARD_MECHANISM, build_card),
        Family(
            "template_operator",
            True,
            ("cross-account", "timing/sequence", "text"),
            TEMPLATE_MECHANISM,
            build_template,
        ),
    ]
}
