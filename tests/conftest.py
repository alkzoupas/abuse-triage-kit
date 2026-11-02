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


# ---------------------------------------------------------------------------
# Guard: no test may create a real Anthropic client. Every real client goes through
# triage_kit.detectors.llm.anthropic_provider._make_real_client, which is replaced
# for the whole session. Any attempt is recorded, and pytest_sessionfinish fails the
# run if an unacknowledged attempt remains.
REAL_CLIENT_ATTEMPTS: list[str] = []


class RealClientForbidden(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _forbid_real_llm_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import triage_kit.detectors.llm.anthropic_provider as ap

    def _blocked() -> Any:
        REAL_CLIENT_ATTEMPTS.append("attempt")
        raise RealClientForbidden("tests must pass a stub client")

    monkeypatch.setattr(ap, "_make_real_client", _blocked)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if REAL_CLIENT_ATTEMPTS:
        print(f"\nFAIL: {len(REAL_CLIENT_ATTEMPTS)} test(s) tried to create a real LLM client")
        session.exitstatus = 1
