"""Train/test splits.

- Split by time: units created before the cutoff train, later units test.
- A unit is an account, except in coordinated families (review_ring,
  template_operator) where the unit is the whole group, timed by the signup time
  of its lowest-account-id member (ids are random, so at s = 0 this time is
  distributed like any single account's; timing by the earliest member would push
  every group into train).
- Group members whose own signup falls on the other side of the cutoff are
  excluded from both train and test. So the split is strictly time-ordered per
  account, and test group members are the same age as benign test accounts (at
  s = 0, keeping whole independently-aged groups made test members older, which
  leaked group membership through account age). No account and no group ever
  straddles train and test.
- in_family: train and test on all six families.
- heldout_k: train on four families, test on the two unseen ones (three
  rotations; every family is held out exactly once). Test accounts are the same
  time-partition test accounts as in_family, so the two are directly comparable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from triage_kit.generator import FAMILIES
from triage_kit.schema import Account

HELDOUT_FOLDS: tuple[tuple[str, str], ...] = (
    ("template_operator", "card_testing"),
    ("review_ring", "scraping"),
    ("ato", "promo_abuse"),
)
TRAIN_FRACTION = 0.6


class LeakageError(AssertionError):
    pass


@dataclass(frozen=True)
class Split:
    name: str
    train_ids: frozenset[str]
    test_ids: frozenset[str]
    train_families: tuple[str, ...]
    test_families: tuple[str, ...]
    excluded_ids: frozenset[str] = frozenset()


def unit_of(label: Mapping[str, Any]) -> str:
    """Resampling / splitting unit: group id for coordinated families, else account id."""
    return str(label["group_id"]) if label.get("group_id") else str(label["account_id"])


def unit_times(
    accounts: Sequence[Account], labels: Mapping[str, Mapping[str, Any]]
) -> dict[str, int]:
    """Time of each unit: created_at of its lowest-account-id member."""
    best: dict[str, tuple[str, int]] = {}
    for a in accounts:
        u = unit_of(labels[a.account_id])
        if u not in best or a.account_id < best[u][0]:
            best[u] = (a.account_id, a.created_at)
    return {u: t for u, (_, t) in best.items()}


def _partition(
    accounts: Sequence[Account], labels: Mapping[str, Mapping[str, Any]]
) -> tuple[set[str], set[str], set[str]]:
    unit_time = unit_times(accounts, labels)
    cutoff = float(np.quantile(sorted(unit_time.values()), TRAIN_FRACTION))
    train: set[str] = set()
    test: set[str] = set()
    excluded: set[str] = set()
    for a in accounts:
        unit_side_train = unit_time[unit_of(labels[a.account_id])] <= cutoff
        own_side_train = a.created_at <= cutoff
        if unit_side_train != own_side_train:
            excluded.add(a.account_id)  # group member on the other side of the cutoff
        else:
            (train if unit_side_train else test).add(a.account_id)
    return train, test, excluded


def make_splits(
    accounts: Sequence[Account], labels: Mapping[str, Mapping[str, Any]]
) -> list[Split]:
    train, test, excluded = _partition(accounts, labels)
    ex = frozenset(excluded)
    fams = tuple(FAMILIES)
    out = [Split("in_family", frozenset(train), frozenset(test), fams, fams, ex)]
    for k, held in enumerate(HELDOUT_FOLDS, 1):
        seen = tuple(f for f in fams if f not in held)
        tr = frozenset(a for a in train if labels[a]["family"] not in held)
        out.append(Split(f"heldout_{k}", tr, frozenset(test), seen, held, ex))
    for s in out:
        assert_split_ok(s, labels)
    return out


def assert_split_ok(split: Split, labels: Mapping[str, Mapping[str, Any]]) -> None:
    """Raise LeakageError if any account or group is on both sides, or a held-out
    family appears in training."""
    both = split.train_ids & split.test_ids
    if both:
        raise LeakageError(f"{split.name}: {len(both)} accounts in both train and test")
    tr_groups = {labels[a]["group_id"] for a in split.train_ids} - {None}
    te_groups = {labels[a]["group_id"] for a in split.test_ids} - {None}
    if tr_groups & te_groups:
        raise LeakageError(
            f"{split.name}: groups straddle train/test: {sorted(tr_groups & te_groups)[:3]}"
        )
    if split.name != "in_family":
        trained = {labels[a]["family"] for a in split.train_ids}
        bad = trained & set(split.test_families)
        if bad:
            raise LeakageError(f"{split.name}: trains on held-out families {sorted(bad)}")


def cell_ids(split: Split, family: str, labels: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """Test accounts for one (split, family) cell: benign test accounts plus the
    family's test accounts. Sorted for determinism."""
    if family not in split.test_families:
        raise ValueError(f"{family} is not evaluated in {split.name}")
    return sorted(a for a in split.test_ids if labels[a]["family"] in ("benign", family))
