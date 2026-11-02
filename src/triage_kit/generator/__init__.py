"""Seeded, deterministic synthetic marketplace generator.

Outputs, under one directory:
  accounts.jsonl, events.jsonl   -- the schema detectors see
  truth/labels.jsonl             -- per-account labels (read only by triage_kit.eval)
  truth/meta.json                -- generator internals (templates, ring targets, config)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triage_kit.generator import benign
from triage_kit.generator.families import FAMILIES, Family
from triage_kit.schema import Account, Event, write_dataset

__all__ = [
    "FAMILIES",
    "Family",
    "GeneratedData",
    "GeneratorConfig",
    "generate",
    "health_check",
    "write_generated",
]

BASE_N = 5000
DEFAULT_SIZES = {  # accounts (or groups, for coordinated families) per 5,000 accounts
    "ato": 100,
    "review_ring": 30,
    "promo_abuse": 100,
    "scraping": 60,
    "card_testing": 80,
    "template_operator": 30,
}


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 0
    s: float = 1.0
    n_accounts: int = BASE_N
    days: int = 60
    start_ts: int = 1_767_225_600  # 2026-01-01T00:00:00Z
    n_listings: int = 2000
    min_groups: int = 12
    group_size: tuple[int, int] = (4, 8)
    sizes: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SIZES))


@dataclass
class GeneratedData:
    config: GeneratorConfig
    accounts: list[Account]
    events: list[Event]
    labels: list[dict[str, Any]]
    meta: dict[str, Any]


def _ids(rng: np.random.Generator, n: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    while len(out) < n:
        x = "".join(f"{int(b):02x}" for b in rng.integers(0, 256, size=6))
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def generate(cfg: GeneratorConfig) -> GeneratedData:
    if not 0.0 <= cfg.s <= 1.0:
        raise ValueError("signal strength s must be in [0, 1]")
    ss = np.random.SeedSequence(cfg.seed)
    child = ss.spawn(2 + len(FAMILIES))
    rng_world = np.random.default_rng(child[0])
    rng_benign = np.random.default_rng(child[1])
    world = benign.make_world(rng_world, cfg.start_ts, cfg.days, cfg.n_listings)
    ids = _ids(rng_world, cfg.n_accounts)
    scale = cfg.n_accounts / BASE_N

    # carve out family membership
    plan: dict[str, Any] = {}
    cursor = 0
    for name, fam in FAMILIES.items():
        if fam.coordinated:
            n_groups = max(cfg.min_groups, round(cfg.sizes[name] * scale))
            groups = []
            for _ in range(n_groups):
                k = int(rng_world.integers(cfg.group_size[0], cfg.group_size[1] + 1))
                groups.append(ids[cursor : cursor + k])
                cursor += k
            plan[name] = groups
        else:
            k = max(1, round(cfg.sizes[name] * scale))
            plan[name] = ids[cursor : cursor + k]
            cursor += k
    if cursor >= cfg.n_accounts:
        raise ValueError("n_accounts too small for the configured family sizes")

    accounts: list[Account] = []
    events: list[Event] = []
    labels: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"config": _cfg_dict(cfg), "families": {}}
    for aid in ids[cursor:]:
        c = benign.draw_created_at(rng_benign, world)
        country, domain = benign.draw_profile(rng_benign)
        persona = benign.draw_persona(rng_benign, aid, c)
        sess = benign.benign_sessions(rng_benign, world, persona)
        accounts.append(Account(aid, c, country, domain))
        events.extend(e for x in sess for e in x)
        labels.append(
            {
                "account_id": aid,
                "is_abuse": False,
                "family": "benign",
                "group_id": None,
                "template_id": None,
            }
        )
    for i, (name, fam) in enumerate(FAMILIES.items()):
        rng_f = np.random.default_rng(child[2 + i])
        built, fmeta = fam.build(rng_f, world, cfg.s, plan[name])
        meta["families"][name] = {
            "coordinated": fam.coordinated,
            "channels": list(fam.channels),
            "mechanism": fam.mechanism,
            **fmeta,
        }
        for acct, evs, lab in built:
            accounts.append(acct)
            events.extend(evs)
            labels.append(lab)

    accounts.sort(key=lambda a: a.account_id)
    events.sort(key=lambda e: (e.account_id, e.ts, e.type, e.target_id or ""))
    labels.sort(key=lambda r: r["account_id"])
    meta["world"] = {
        "start_ts": world.start_ts,
        "end_ts": world.end_ts,
        "n_listings": cfg.n_listings,
    }
    return GeneratedData(cfg, accounts, events, labels, meta)


def _cfg_dict(cfg: GeneratorConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["group_size"] = list(cfg.group_size)
    return d


def health_check(g: GeneratedData) -> list[str]:
    """Structural sanity checks. Returns a list of problems (empty = healthy)."""
    problems: list[str] = []
    start, end = g.meta["world"]["start_ts"], g.meta["world"]["end_ts"]
    created = {a.account_id: a.created_at for a in g.accounts}
    if len(created) != len(g.accounts):
        problems.append("duplicate account ids")
    n_bad_time = sum(1 for e in g.events if not (start <= e.ts < end))
    if n_bad_time:
        problems.append(f"{n_bad_time} events outside the simulation window")
    n_before = sum(1 for e in g.events if e.ts < created[e.account_id])
    if n_before:
        problems.append(f"{n_before} events before account creation")
    fams = [r["family"] for r in g.labels]
    for name, fam in FAMILIES.items():
        if fams.count(name) == 0:
            problems.append(f"family {name} has no accounts")
        if fam.coordinated:
            groups = {r["group_id"] for r in g.labels if r["family"] == name}
            if len(groups) < g.config.min_groups:
                problems.append(f"family {name} has {len(groups)} groups < {g.config.min_groups}")
    share = sum(r["is_abuse"] for r in g.labels) / len(g.labels)
    if not 0.03 <= share <= 0.25:
        problems.append(f"abuse share {share:.3f} outside [0.03, 0.25]")
    n_with_events = len({e.account_id for e in g.events})
    if n_with_events < 0.99 * len(g.accounts):
        problems.append("more than 1% of accounts have no events")
    return problems


def write_generated(g: GeneratedData, directory: Path) -> None:
    write_dataset(directory, g.accounts, g.events)
    truth = directory / "truth"
    truth.mkdir(parents=True, exist_ok=True)
    with (truth / "labels.jsonl").open("w") as f:
        for r in g.labels:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")
    (truth / "meta.json").write_text(json.dumps(g.meta, sort_keys=True, indent=1) + "\n")
