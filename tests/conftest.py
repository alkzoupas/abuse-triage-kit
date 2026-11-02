"""Shared fixtures. Generated datasets are cached per session to keep tests fast."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import pytest

from triage_kit.generator import GeneratorConfig, generate
from triage_kit.schema import AccountEvents, group_events

DEV_SEED = 101  # first dev seed (see PREREGISTRATION.md)


@dataclass
class Dataset:
    data: list[AccountEvents]
    labels: list[dict[str, Any]]  # aligned with data
    meta: dict[str, Any]


@cache
def make_dataset(seed: int, s: float, n: int = 5000) -> Dataset:
    g = generate(GeneratorConfig(seed=seed, s=s, n_accounts=n))
    return Dataset(group_events(g.accounts, g.events), g.labels, g.meta)


@pytest.fixture(scope="session")
def dev_s1() -> Dataset:
    return make_dataset(DEV_SEED, 1.0)


@pytest.fixture(scope="session")
def dev_s0() -> Dataset:
    return make_dataset(DEV_SEED, 0.0)
